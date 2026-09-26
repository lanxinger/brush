//! Guards the fusion behaviour the kernel-boundary work bought.
//!
//! The backward writes one gradient row per visible splat and the render
//! backward expands it with a gather. What that should cost is exactly one
//! fused kernel per parameter: no zero-filled dense buffer, and no separate
//! mask multiply of the kind the visibility masks used to need. burn's
//! `FusionInspector` reports what actually fused, so pin it down here.
//!
//! Note burn keeps each gather in its own block rather than folding it into
//! the optimizer's kernel, so a dense gradient per parameter is still
//! written once. The portable SH second moment is also gathered after reducing
//! compact rows; native fused SH Adam calculates that statistic internally.

#![cfg(not(target_family = "wasm"))]

use brush_dataset::scene::SceneBatch;
use brush_render::{
    AlphaMode,
    bounding_box::BoundingBox,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats},
    kernels::camera_model::CameraModel::Pinhole,
};
use brush_train::{config::TrainConfig, train::SplatTrainer};
use burn::backend::ir::{BaseOperationIr, OperationIr};
use burn::tensor::{Device, Tensor, TensorData};
use burn_fusion::inspect::{FusionInspector, FusionReport};
use burn_fusion::stream::StreamId;
use glam::{Quat, Vec3};
use rand::{RngExt, SeedableRng};

const TEST_SEED: u64 = 12345;

fn test_splats(device: &Device, count: usize) -> Splats {
    let mut rng = rand::rngs::StdRng::seed_from_u64(TEST_SEED);
    let means: Vec<f32> = (0..count)
        .flat_map(|_| {
            [
                rng.random_range(-2.0..2.0),
                rng.random_range(-2.0..2.0),
                rng.random_range(1.0..5.0),
            ]
        })
        .collect();
    let rots: Vec<f32> = (0..count).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect();
    let log_scales: Vec<f32> = (0..count).flat_map(|_| [-2.0, -2.0, -2.0]).collect();
    let coeffs: Vec<f32> = (0..count).flat_map(|_| [0.5, 0.5, 0.5]).collect();
    let opacities: Vec<f32> = (0..count).map(|_| 0.5).collect();
    Splats::from_raw(
        means,
        rots,
        log_scales,
        coeffs,
        opacities,
        SplatRenderMode::Default,
        device,
    )
}

fn test_batch(width: u32, height: u32) -> SceneBatch {
    let pixels = (width * height) as usize;
    let img: Vec<i32> = (0..pixels)
        .map(|i| {
            let v = (i % 200) as u32;
            (v | v << 8 | v << 16 | 255 << 24) as i32
        })
        .collect();
    SceneBatch {
        img_packed: TensorData::new(img, [height as usize, width as usize]),
        has_alpha: false,
        view_index: 0,
        alpha_mode: AlphaMode::Transparent,
        camera: Camera::new(
            Vec3::new(0.0, 0.0, -3.0),
            Quat::IDENTITY,
            0.6,
            0.6,
            glam::vec2(0.5, 0.5),
            Pinhole,
        ),
    }
}

/// The compact-to-dense gradient expansion.
fn is_select(op: &OperationIr) -> bool {
    matches!(op, OperationIr::BaseFloat(BaseOperationIr::Select(_)))
}

/// `transforms`, `sh_coeffs`, `raw_opacities`, and the refine-weight holder.
const GRADIENT_PARAMS: usize = 4;
/// Portable Adam additionally consumes the compactly reduced SH second moment.
const PORTABLE_GRADIENT_EXPANSIONS: usize = GRADIENT_PARAMS + 1;

#[tokio::test]
async fn gradient_expansions_are_fused() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let config = TrainConfig::default();
    let mut trainer = SplatTrainer::new(
        &config,
        &device,
        BoundingBox::from_min_max(Vec3::ZERO, Vec3::ONE),
    );

    // Warm up first: the optimizer builds its state on step one, which is not
    // the steady state we care about.
    let mut splats = test_splats(&device, 256);
    for _ in 0..2 {
        (splats, _) = trainer.step(test_batch(32, 32), splats).await;
    }
    // Native SH updates are separate lazy custom ops. Read all parameters so
    // the measured forward cannot pull the last warmup update into its trace.
    execute(splats.transforms.val()).await;
    execute(splats.sh_coeffs.val()).await;
    execute(splats.raw_opacities.val()).await;

    let inspector = FusionInspector::install(StreamId::current());
    let (splats, stats) = trainer.step(test_batch(32, 32), splats).await;
    assert!(
        stats.num_visible > 0 && stats.num_visible < splats.num_splats(),
        "the fusion fixture must contain both visible and culled splats"
    );
    // Reading means alone does not force an independent native SH update.
    execute(splats.transforms.val()).await;
    execute(splats.sh_coeffs.val()).await;
    execute(splats.raw_opacities.val()).await;

    let reports = inspector.drain();
    assert!(
        !reports.is_empty(),
        "inspector saw no execution plans; is the step running on this stream?"
    );

    let native_sh_updates = reports
        .iter()
        .flat_map(|report| &report.blocks)
        .flat_map(|block| &block.operations)
        .filter_map(|op| match op {
            OperationIr::Custom(custom)
                if custom.id == "fused_sh_adam" || custom.id == "sparse_sh_adam" =>
            {
                Some(custom.id.as_str())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(
        native_sh_updates.len() <= 1,
        "one SH update per training step, saw {native_sh_updates:?}:\n{}",
        reports
            .iter()
            .map(FusionReport::format_table)
            .collect::<Vec<_>>()
            .join("\n")
    );
    // Require the exact native kernel in the trace before allowing an omitted
    // gather. Compiling native-MSL or requesting it does not prove eligibility.
    let expected = match native_sh_updates.first().copied() {
        Some("sparse_sh_adam") => GRADIENT_PARAMS - 1,
        Some("fused_sh_adam") => GRADIENT_PARAMS,
        _ => PORTABLE_GRADIENT_EXPANSIONS,
    };
    assert_gradient_expansions(&reports, expected);
}

fn assert_gradient_expansions(reports: &[FusionReport], expected: usize) {
    let mut gathers = 0;
    let mut unfused = Vec::new();
    for report in reports {
        for block in &report.blocks {
            let selects = block.operations.iter().filter(|op| is_select(op)).count();
            gathers += selects;
            if selects > 0 && block.fuser_name().is_none() {
                unfused.push(report.format_table());
            }
        }
    }

    assert!(
        unfused.is_empty(),
        "a gradient gather ran unfused:\n{}",
        unfused.join("\n")
    );
    assert_eq!(
        gathers,
        expected,
        "expected one gather per consumed parameter/statistic; more means the expansion grew \
         extra ops, fewer means a gradient stopped reaching its parameter:\n{}",
        reports
            .iter()
            .map(FusionReport::format_table)
            .collect::<Vec<_>>()
            .join("\n")
    );
}

async fn execute<const D: usize>(tensor: Tensor<D>) {
    let _ = tensor.into_data_async().await.expect("gradient readback");
}

#[tokio::test]
async fn explicitly_requested_compact_moment_expansion_is_fused() {
    let device = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let splats = test_splats(&device, 256);
    let batch = test_batch(32, 32);
    let output = brush_render::bwd::render_splats_for_training(
        splats.clone(),
        &batch.camera,
        glam::uvec2(32, 32),
        Vec3::ZERO,
        true,
        false,
        true,
    )
    .await;
    assert!(
        output.num_visible > 0 && output.num_visible < splats.num_splats(),
        "the fusion fixture must contain both visible and culled splats"
    );
    execute(output.img.clone()).await;
    let inspector = FusionInspector::install(StreamId::current());
    let mut grads = output.img.mean().backward();
    execute(
        splats
            .transforms
            .grad_remove(&mut grads)
            .expect("transforms gradient"),
    )
    .await;
    execute(
        splats
            .sh_coeffs
            .grad_remove(&mut grads)
            .expect("SH gradient"),
    )
    .await;
    execute(
        splats
            .raw_opacities
            .grad_remove(&mut grads)
            .expect("opacity gradient"),
    )
    .await;
    execute(
        output
            .refine_weight_holder
            .grad_remove(&mut grads)
            .expect("refine gradient"),
    )
    .await;
    execute(
        output
            .coeffs_grad_sq_holder
            .grad_remove(&mut grads)
            .expect("SH second moment"),
    )
    .await;
    assert_gradient_expansions(&inspector.drain(), PORTABLE_GRADIENT_EXPANSIONS);
}
