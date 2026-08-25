//! Host-Metal regression coverage for large appearance-kernel dispatches.
//!
//! PPISP uses 128-thread workgroups, so a 3840-square image exceeds WebGPU's
//! 65,535-workgroup per-dimension limit. Bilateral-grid and PPISP-grid use
//! 256-thread workgroups; 4096 square is the first size over the limit, while
//! 4097 square also produces a padded 2D workgroup rectangle and exercises the
//! PPISP-grid backward partial-buffer tail guard.

#![cfg(not(target_family = "wasm"))]

use brush_appearance::GradSubsample;
use brush_appearance::bilagrid::bilagrid_apply;
use brush_appearance::ppisp::{PpispStages, ppisp_apply};
use brush_appearance::ppisp_grid::{GridPayload, ppisp_grid_apply};
use burn::tensor::{Device, Tensor};

/// 61,952 PPISP workgroups: the high-resolution path still fits in 1D.
const SUB_LIMIT: usize = 2816;
/// 115,200 PPISP workgroups: PPISP must tile this dispatch into 2D.
const PPISP_OVER_LIMIT: usize = 3840;
/// 65,569 256-thread workgroups, padded to a 256x257 dispatch rectangle.
const WIDE_BLOCK_OVER_LIMIT: usize = 4097;

const FRAME_ONLY: PpispStages = PpispStages {
    frame: true,
    vignetting: false,
    crf: false,
};

const EXPOSURE_ONLY_GRID: GridPayload = GridPayload {
    color: false,
    crf: false,
    vignetting: true,
};

async fn ad_device() -> Device {
    Device::from(brush_cube::test_helpers::test_device().await).autodiff()
}

fn pattern(n: usize, seed: u32, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
            let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
            let hash = (word >> 22) ^ word;
            lo + (hi - lo) * (hash as f32 / u32::MAX as f32)
        })
        .collect()
}

fn checksum(values: &[f32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

async fn read<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor
        .into_data_async()
        .await
        .expect("readback")
        .try_to_vec()
        .expect("f32 data")
}

fn assert_scaled(actual: &[f32], source: &[f32], scale: f32, tolerance: f32, label: &str) -> f32 {
    assert_eq!(actual.len(), source.len(), "{label}: length mismatch");
    let mut worst = 0.0f32;
    for (index, (got, input)) in actual.iter().zip(source).enumerate() {
        let expected = input * scale;
        let error = (got - expected).abs();
        assert!(
            error <= tolerance,
            "{label}: mismatch at {index}: got {got}, expected {expected}"
        );
        assert!(got.is_finite(), "{label}: non-finite value at {index}");
        worst = worst.max(error);
    }
    worst
}

fn assert_relative(actual: f64, expected: f64, tolerance: f64, label: &str) -> f64 {
    let relative = (actual - expected).abs() / expected.abs().max(1.0);
    assert!(
        relative <= tolerance,
        "{label}: got {actual}, expected {expected} (relative error {relative})"
    );
    relative
}

async fn run_ppisp_case(device: &Device, label: &str, size: usize) {
    let count = size * size * 3;
    let rgb_data = pattern(count, 41, 0.05, 0.8);
    let weight_data = pattern(count, 47, -1.0, 1.0);
    let exposure_value = 0.75f32;

    let exposure = Tensor::from_floats([exposure_value], device).require_grad();
    let rgb = Tensor::<1>::from_floats(rgb_data.as_slice(), device)
        .reshape([size, size, 3])
        .require_grad();
    let weights = Tensor::<1>::from_floats(weight_data.as_slice(), device).reshape([size, size, 3]);
    let output = ppisp_apply(
        exposure.clone(),
        Tensor::zeros([1, 3, 5], device),
        Tensor::zeros([1, 8], device),
        Tensor::zeros([1, 3, 4], device),
        rgb.clone(),
        0,
        0,
        FRAME_ONLY,
    );
    let gradients = (output.clone() * weights).sum().backward();
    let output_values = read(output).await;
    let rgb_grad = read(rgb.grad(&gradients).expect("PPISP RGB gradient")).await;
    let exposure_grad = read(exposure.grad(&gradients).expect("PPISP exposure gradient")).await[0];

    let gain = (exposure_value * std::f32::consts::LN_2).exp();
    let max_output_error = assert_scaled(
        &output_values,
        &rgb_data,
        gain,
        2e-4,
        &format!("PPISP {label} output"),
    );
    let max_rgb_grad_error = assert_scaled(
        &rgb_grad,
        &weight_data,
        gain,
        2e-4,
        &format!("PPISP {label} RGB gradient"),
    );
    let exposure_reference = rgb_data
        .iter()
        .zip(&weight_data)
        .map(|(source, weight)| f64::from(*source) * f64::from(gain) * f64::from(*weight))
        .sum::<f64>()
        * f64::from(std::f32::consts::LN_2);
    let exposure_relative = assert_relative(
        f64::from(exposure_grad),
        exposure_reference,
        1e-3,
        &format!("PPISP {label} exposure gradient"),
    );

    println!(
        "DISPATCH PPISP {label} size={size} fwd_chk={:016x} rgbgrad_chk={:016x} \
         exp_grad_bits={:08x} max_fwd_err={max_output_error:e} \
         max_rgbgrad_err={max_rgb_grad_error:e} exp_grad_rel={exposure_relative:e}",
        checksum(&output_values),
        checksum(&rgb_grad),
        exposure_grad.to_bits(),
    );
}

fn identity_bilagrid(grid_l: usize, grid_h: usize, grid_w: usize) -> Vec<f32> {
    let cells = grid_l * grid_h * grid_w;
    let mut grid = vec![0.0; 12 * cells];
    for coefficient in [0, 5, 10] {
        grid[coefficient * cells..(coefficient + 1) * cells].fill(1.0);
    }
    grid
}

fn bilagrid_gradient_reference(rgb: &[f32], weights: &[f32]) -> [f64; 12] {
    let mut reference = [0.0f64; 12];
    for (pixel, upstream) in rgb
        .as_chunks::<3>()
        .0
        .iter()
        .zip(weights.as_chunks::<3>().0)
    {
        let input = [
            f64::from(pixel[0]),
            f64::from(pixel[1]),
            f64::from(pixel[2]),
            1.0,
        ];
        for row in 0..3 {
            for column in 0..4 {
                reference[row * 4 + column] += f64::from(upstream[row]) * input[column];
            }
        }
    }
    reference
}

async fn run_bilagrid_case(device: &Device, label: &str, size: usize) {
    let count = size * size * 3;
    let rgb_data = pattern(count, 83, 0.05, 0.95);
    let weight_data = pattern(count, 89, 0.1, 1.0);
    let (grid_l, grid_h, grid_w) = (8, 8, 8);
    let cells = grid_l * grid_h * grid_w;
    let grid_data = identity_bilagrid(grid_l, grid_h, grid_w);

    let grids = Tensor::<1>::from_floats(grid_data.as_slice(), device)
        .reshape([1, 12, grid_l, grid_h, grid_w])
        .require_grad();
    let rgb = Tensor::<1>::from_floats(rgb_data.as_slice(), device)
        .reshape([size, size, 3])
        .require_grad();
    let weights = Tensor::<1>::from_floats(weight_data.as_slice(), device).reshape([size, size, 3]);
    let output = bilagrid_apply(grids.clone(), rgb.clone(), 0);
    let gradients = (output.clone() * weights).sum().backward();
    let output_values = read(output).await;
    let rgb_grad = read(rgb.grad(&gradients).expect("bilateral-grid RGB gradient")).await;
    let grid_grad = read(
        grids
            .grad(&gradients)
            .expect("bilateral-grid parameter gradient"),
    )
    .await;

    let max_output_error = assert_scaled(
        &output_values,
        &rgb_data,
        1.0,
        2e-4,
        &format!("bilateral-grid {label} output"),
    );
    let max_rgb_grad_error = assert_scaled(
        &rgb_grad,
        &weight_data,
        1.0,
        2e-4,
        &format!("bilateral-grid {label} RGB gradient"),
    );

    let expected_grid_grad = bilagrid_gradient_reference(&rgb_data, &weight_data);
    let mut max_grid_grad_relative = 0.0f64;
    // Grid scatters use atomics, so their bitwise checksum is not stable.
    // Validate coefficient aggregates against the CPU reference instead.
    for coefficient in 0..12 {
        let actual = grid_grad[coefficient * cells..(coefficient + 1) * cells]
            .iter()
            .map(|value| f64::from(*value))
            .sum();
        let relative = assert_relative(
            actual,
            expected_grid_grad[coefficient],
            2e-3,
            &format!("bilateral-grid {label} coefficient {coefficient} gradient"),
        );
        max_grid_grad_relative = max_grid_grad_relative.max(relative);
    }

    println!(
        "DISPATCH BILAGRID {label} size={size} fwd_chk={:016x} rgbgrad_chk={:016x} \
         max_fwd_err={max_output_error:e} max_rgbgrad_err={max_rgb_grad_error:e} \
         grid_grad_rel={max_grid_grad_relative:e}",
        checksum(&output_values),
        checksum(&rgb_grad),
    );
}

fn ppisp_grid_vignetting_reference(
    rgb: &[f32],
    weights: &[f32],
    height: usize,
    width: usize,
) -> [f64; 15] {
    let mut reference = [0.0f64; 15];
    let max_resolution = height.max(width) as f32;
    for y in 0..height {
        let uv_y = (y as f32 + 0.5 - height as f32 * 0.5) / max_resolution;
        for x in 0..width {
            let uv_x = (x as f32 + 0.5 - width as f32 * 0.5) / max_resolution;
            let radius2 = uv_x * uv_x + uv_y * uv_y;
            let radius4 = radius2 * radius2;
            let radius6 = radius4 * radius2;
            let base = (y * width + x) * 3;
            for channel in 0..3 {
                let gain = rgb[base + channel] * weights[base + channel];
                let offset = channel * 5;
                reference[offset + 2] += f64::from(gain * radius2);
                reference[offset + 3] += f64::from(gain * radius4);
                reference[offset + 4] += f64::from(gain * radius6);
            }
        }
    }
    reference
}

async fn run_ppisp_grid_case(device: &Device, label: &str, size: usize) {
    let count = size * size * 3;
    let rgb_data = pattern(count, 101, 0.05, 0.95);
    let weight_data = pattern(count, 103, 0.1, 1.0);
    let (grid_l, grid_h, grid_w) = (8, 8, 8);

    let grids = Tensor::zeros([1, 1, grid_l, grid_h, grid_w], device).require_grad();
    let vignetting = Tensor::zeros([1, 3, 5], device).require_grad();
    let rgb = Tensor::<1>::from_floats(rgb_data.as_slice(), device)
        .reshape([size, size, 3])
        .require_grad();
    let weights = Tensor::<1>::from_floats(weight_data.as_slice(), device).reshape([size, size, 3]);
    let output = ppisp_grid_apply(
        grids.clone(),
        vignetting.clone(),
        rgb.clone(),
        0,
        0,
        EXPOSURE_ONLY_GRID,
        GradSubsample::default(),
    );
    let gradients = (output.clone() * weights).sum().backward();
    let output_values = read(output).await;
    let rgb_grad = read(rgb.grad(&gradients).expect("PPISP-grid RGB gradient")).await;
    let grid_grad = read(grids.grad(&gradients).expect("PPISP-grid payload gradient")).await;
    let vignetting_grad = read(
        vignetting
            .grad(&gradients)
            .expect("PPISP-grid vignetting gradient"),
    )
    .await;

    let max_output_error = assert_scaled(
        &output_values,
        &rgb_data,
        1.0,
        2e-4,
        &format!("PPISP-grid {label} output"),
    );
    let max_rgb_grad_error = assert_scaled(
        &rgb_grad,
        &weight_data,
        1.0,
        2e-4,
        &format!("PPISP-grid {label} RGB gradient"),
    );

    let grid_grad_actual: f64 = grid_grad.iter().map(|value| f64::from(*value)).sum();
    let grid_grad_reference = rgb_data
        .iter()
        .zip(&weight_data)
        .map(|(source, weight)| f64::from(*source) * f64::from(*weight))
        .sum::<f64>()
        * f64::from(std::f32::consts::LN_2);
    let grid_grad_relative = assert_relative(
        grid_grad_actual,
        grid_grad_reference,
        2e-3,
        &format!("PPISP-grid {label} payload gradient"),
    );
    // Grid scatters use atomics, so their bitwise checksum is not stable.
    // The aggregate CPU comparison above is the portable regression gate.

    let vignetting_reference = ppisp_grid_vignetting_reference(&rgb_data, &weight_data, size, size);
    let mut max_vignetting_relative = 0.0f64;
    for index in 0..15 {
        let relative = assert_relative(
            f64::from(vignetting_grad[index]),
            vignetting_reference[index],
            2e-3,
            &format!("PPISP-grid {label} vignetting gradient {index}"),
        );
        max_vignetting_relative = max_vignetting_relative.max(relative);
    }

    println!(
        "DISPATCH PPISP_GRID {label} size={size} fwd_chk={:016x} rgbgrad_chk={:016x} \
         viggrad_chk={:016x} max_fwd_err={max_output_error:e} \
         max_rgbgrad_err={max_rgb_grad_error:e} grid_grad_rel={grid_grad_relative:e} \
         vig_grad_rel={max_vignetting_relative:e}",
        checksum(&output_values),
        checksum(&rgb_grad),
        checksum(&vignetting_grad),
    );
}

/// Run explicitly with:
///
/// `cargo test -p brush-appearance --test dispatch_tiling -- --ignored --nocapture`
///
/// The cases are intentionally serialized: each large forward/backward case
/// needs roughly 1.5 GiB at peak, and concurrent cases can exhaust unified
/// memory before they reach the dispatch being tested.
#[tokio::test]
#[ignore = "large host-Metal dispatch regression (~1.5 GiB peak)"]
async fn appearance_dispatch_tiling_spans_workgroup_limits() {
    let device = ad_device().await;

    run_ppisp_case(&device, "sub_limit", SUB_LIMIT).await;
    run_bilagrid_case(&device, "sub_limit", SUB_LIMIT).await;
    run_ppisp_grid_case(&device, "sub_limit", SUB_LIMIT).await;

    run_ppisp_case(&device, "over_limit", PPISP_OVER_LIMIT).await;
    run_bilagrid_case(&device, "requested_3840", PPISP_OVER_LIMIT).await;
    run_ppisp_grid_case(&device, "requested_3840", PPISP_OVER_LIMIT).await;

    run_bilagrid_case(&device, "over_limit", WIDE_BLOCK_OVER_LIMIT).await;
    run_ppisp_grid_case(&device, "over_limit_with_tail", WIDE_BLOCK_OVER_LIMIT).await;
}
