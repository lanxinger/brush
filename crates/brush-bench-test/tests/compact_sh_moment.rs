//! Compare the training-only compact SH second moment with the dense gradient.

use brush_render::{
    bwd::render_splats_for_training,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats},
    kernels::camera_model::CameraModel::Pinhole,
};
use burn::tensor::Device;
use glam::{Quat, Vec3};
use wasm_bindgen_test::wasm_bindgen_test;

#[cfg(target_family = "wasm")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn compact_sh_second_moment_matches_dense_for_every_degree_and_culling() {
    let device = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, -3.0),
        Quat::IDENTITY,
        0.6,
        0.6,
        glam::vec2(0.5, 0.5),
        Pinhole,
    );
    // Alternate visible and culled rows so compact order differs from dense
    // order. All-culled scenes also exercise the zero sentinel row.
    let count = 7;
    for degree in 0..=4 {
        let coefficients = (degree + 1) * (degree + 1);
        for fully_culled in [false, true] {
            for compute_second_moment in [false, true] {
                let means = (0..count)
                    .flat_map(|i| {
                        let culled = fully_culled || i % 2 == 1;
                        [
                            (i as f32 - 3.0) * 0.08,
                            (i as f32 - 2.0) * 0.04,
                            if culled { -5.0 } else { i as f32 * 0.1 },
                        ]
                    })
                    .collect();
                let splats = Splats::from_raw(
                    means,
                    (0..count).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect(),
                    vec![-1.8; count * 3],
                    (0..count * coefficients * 3)
                        .map(|i| 0.1 + (i % 13) as f32 * 0.02)
                        .collect(),
                    vec![1.5; count],
                    SplatRenderMode::Default,
                    &device,
                );
                let output = render_splats_for_training(
                    splats.clone(),
                    &camera,
                    glam::uvec2(32, 32),
                    Vec3::new(0.1, 0.2, 0.3),
                    false,
                    false,
                    compute_second_moment,
                )
                .await;
                assert_eq!(
                    output.num_visible,
                    if fully_culled { 0 } else { 4 },
                    "scene must exercise the intended compact row mapping"
                );
                let mut grads = output.img.mean().backward();
                let compact = output.coeffs_grad_sq_holder.grad_remove(&mut grads);
                if !compute_second_moment {
                    assert!(
                        compact.is_none(),
                        "disabled statistic must not create a gradient"
                    );
                    assert!(
                        splats.sh_coeffs.grad(&grads).is_some(),
                        "disabling the statistic must preserve coefficient gradients"
                    );
                    continue;
                }

                let compact = compact.expect("requested SH second moment");
                assert_eq!(
                    compact.dims(),
                    [count, 1, 1],
                    "one moment per dense splat row"
                );
                let compact = compact
                    .into_data_async()
                    .await
                    .expect("compact moment readback")
                    .try_into_vec::<f32>()
                    .expect("f32 second moment");
                let dense = splats
                    .sh_coeffs
                    .grad(&grads)
                    .expect("dense SH gradient remains available")
                    .into_data_async()
                    .await
                    .expect("dense gradient readback")
                    .try_into_vec::<f32>()
                    .expect("f32 gradient");
                let row_width = coefficients * 3;
                for (row, gradient) in dense.chunks_exact(row_width).enumerate() {
                    let expected = gradient.iter().map(|g| g * g).sum::<f32>() / row_width as f32;
                    let actual = compact[row];
                    assert!(
                        actual.is_finite(),
                        "degree {degree}, row {row}: nonfinite moment"
                    );
                    assert!(
                        (actual - expected).abs() <= expected.abs() * 2e-5 + 1e-12,
                        "degree {degree}, row {row}: compact {actual}, dense {expected}"
                    );
                    if fully_culled || row % 2 == 1 {
                        assert_eq!(actual, 0.0, "culled row {row} must have a zero moment");
                    } else {
                        assert!(
                            actual > 0.0,
                            "visible row {row} must exercise the reduction"
                        );
                    }
                }
            }
        }
    }
}
