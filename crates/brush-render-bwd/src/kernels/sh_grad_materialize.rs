//! Coalesced dense spherical-harmonic gradient materialization.
//!
//! One 32-lane SIMD plane owns one global splat row. The project backward
//! pass is followed by a compact-gradient lookup for contributing splats;
//! every row is written exactly once, including exact zeros for
//! non-contributors.

use brush_render::kernels::sh::{SH_C0, num_sh_coeffs};
use brush_render::kernels::types::ProjectUniforms;
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

pub const PLANE_SIZE: u32 = 32;
pub const WG_SIZE: u32 = 256;
pub const SPLATS_PER_WG: u32 = WG_SIZE / PLANE_SIZE;

/// Invert the compact-to-global projection map for rows with a non-zero SH
/// coefficient gradient. Keeping this as a candidate-only pass preserves the
/// original ProjectBackwards storage-buffer signature on every fallback.
#[cube(launch)]
pub fn build_compact_sh_map_kernel(
    global_from_compact_gid: &Tensor<u32>,
    v_combined: &Tensor<f32>,
    compact_plus_one_from_global: &mut Tensor<u32>,
    u: ProjectUniforms,
) {
    let compact_gid = ABSOLUTE_POS as u32;
    if compact_gid >= u.num_visible {
        terminate!();
    }

    let grad_base = (compact_gid * 10u32) as usize;
    let v_color_r = v_combined[grad_base + 5];
    let v_color_g = v_combined[grad_base + 6];
    let v_color_b = v_combined[grad_base + 7];
    if v_color_r != 0.0f32 || v_color_g != 0.0f32 || v_color_b != 0.0f32 {
        let global_gid = global_from_compact_gid[compact_gid as usize];
        compact_plus_one_from_global[global_gid as usize] = compact_gid + 1u32;
    }
}

#[cube]
fn sh_basis(index: u32, #[comptime] degree: u32, x: f32, y: f32, z: f32) -> f32 {
    let mut basis = 0.0f32;
    if index == 0u32 {
        basis = SH_C0;
    }
    if comptime![degree >= 1u32] {
        let f0a = 0.488_602_5f32;
        if index == 1u32 {
            basis = -f0a * y;
        } else if index == 2u32 {
            basis = f0a * z;
        } else if index == 3u32 {
            basis = -f0a * x;
        }
    }
    if comptime![degree >= 2u32] {
        let z2 = z * z;
        let f0b = -1.092_548_5f32 * z;
        let f1a = 0.546_274_24f32;
        let fc1 = x * x - y * y;
        let fs1 = 2.0f32 * x * y;
        if index == 4u32 {
            basis = f1a * fs1;
        } else if index == 5u32 {
            basis = f0b * y;
        } else if index == 6u32 {
            basis = 0.946_174_7f32 * z2 - 0.315_391_57f32;
        } else if index == 7u32 {
            basis = f0b * x;
        } else if index == 8u32 {
            basis = f1a * fc1;
        }
    }
    if comptime![degree >= 3u32] {
        let z2 = z * z;
        let f0c = -2.285_229f32 * z2 + 0.457_045_8f32;
        let f1b = 1.445_305_7f32 * z;
        let f2a = -0.590_043_6f32;
        let fc1 = x * x - y * y;
        let fs1 = 2.0f32 * x * y;
        let fc2 = x * fc1 - y * fs1;
        let fs2 = x * fs1 + y * fc1;
        if index == 9u32 {
            basis = f2a * fs2;
        } else if index == 10u32 {
            basis = f1b * fs1;
        } else if index == 11u32 {
            basis = f0c * y;
        } else if index == 12u32 {
            basis = z * (1.865_881_7f32 * z2 - 1.119_529f32);
        } else if index == 13u32 {
            basis = f0c * x;
        } else if index == 14u32 {
            basis = f1b * fc1;
        } else if index == 15u32 {
            basis = f2a * fc2;
        }
    }
    if comptime![degree >= 4u32] {
        let z2 = z * z;
        let f0d = z * (-4.683_326f32 * z2 + 2.007_139_6f32);
        let f1c = 3.311_611_4f32 * z2 - 0.473_087_35f32;
        let f2b = -1.770_130_8f32 * z;
        let f3a = 0.625_835_8f32;
        let fc1 = x * x - y * y;
        let fs1 = 2.0f32 * x * y;
        let fc2 = x * fc1 - y * fs1;
        let fs2 = x * fs1 + y * fc1;
        let fc3 = x * fc2 - y * fs2;
        let fs3 = x * fs2 + y * fc2;
        let p_sh6 = 0.946_174_7f32 * z2 - 0.315_391_57f32;
        let p_sh12 = z * (1.865_881_7f32 * z2 - 1.119_529f32);
        if index == 16u32 {
            basis = f3a * fs3;
        } else if index == 17u32 {
            basis = f2b * fs2;
        } else if index == 18u32 {
            basis = f1c * fs1;
        } else if index == 19u32 {
            basis = f0d * y;
        } else if index == 20u32 {
            basis = 1.984_313_5f32 * z * p_sh12 + -1.006_230_6f32 * p_sh6;
        } else if index == 21u32 {
            basis = f0d * x;
        } else if index == 22u32 {
            basis = f1c * fc1;
        } else if index == 23u32 {
            basis = f2b * fc2;
        } else if index == 24u32 {
            basis = f3a * fc3;
        }
    }
    basis
}

#[cube]
fn color_component(index: u32, r: f32, g: f32, b: f32) -> f32 {
    let channel = index % 3u32;
    let mut value = b;
    if channel == 0u32 {
        value = r;
    } else if channel == 1u32 {
        value = g;
    }
    value
}

#[cube(launch, launch_unchecked)]
pub fn materialize_sh_grad_kernel(
    transforms: &Tensor<f32>,
    compact_plus_one_from_global: &Tensor<u32>,
    v_combined: &Tensor<f32>,
    v_coeffs: &mut Tensor<f32>,
    u: ProjectUniforms,
    #[comptime] sh_degree: u32,
) {
    let global_gid = CUBE_POS as u32 * SPLATS_PER_WG + PLANE_POS;
    let lane = UNIT_POS_PLANE;
    let active = global_gid < u.total_splats;

    let mut compact_plus_one = 0u32;
    if active && lane == 0u32 {
        compact_plus_one = compact_plus_one_from_global[global_gid as usize];
    }
    compact_plus_one = plane_broadcast(compact_plus_one, 0u32);
    let has_grad = compact_plus_one > 0u32;
    let compact_gid = max(compact_plus_one, 1u32) - 1u32;
    let row_len = comptime![num_sh_coeffs(sh_degree) * 3u32];
    let row_base = global_gid * row_len;
    let index_0 = lane;
    let index_1 = lane + PLANE_SIZE;
    let index_2 = lane + 2u32 * PLANE_SIZE;

    // Most global splats do not contribute to the sampled view. Their dense
    // rows still need exact zeros for Adam's momentum decay, but they do not
    // need any SH polynomial or SIMD shuffle work.
    if !has_grad {
        if active && index_0 < row_len {
            v_coeffs[(row_base + index_0) as usize] = 0.0f32;
        }
        if active && index_1 < row_len {
            v_coeffs[(row_base + index_1) as usize] = 0.0f32;
        }
        if active && index_2 < row_len {
            v_coeffs[(row_base + index_2) as usize] = 0.0f32;
        }
        terminate!();
    }

    let mut field = 0.0f32;
    let transform_base = (global_gid * 10u32) as usize;
    let grad_base = (compact_gid * 10u32) as usize;
    if lane == 0u32 {
        field = transforms[transform_base];
    } else if lane == 1u32 {
        field = transforms[transform_base + 1];
    } else if lane == 2u32 {
        field = transforms[transform_base + 2];
    } else if lane == 3u32 {
        field = v_combined[grad_base + 5];
    } else if lane == 4u32 {
        field = v_combined[grad_base + 6];
    } else if lane == 5u32 {
        field = v_combined[grad_base + 7];
    }
    let mean_x = plane_broadcast(field, 0u32);
    let mean_y = plane_broadcast(field, 1u32);
    let mean_z = plane_broadcast(field, 2u32);
    let v_color_r = plane_broadcast(field, 3u32);
    let v_color_g = plane_broadcast(field, 4u32);
    let v_color_b = plane_broadcast(field, 5u32);

    let camera = u.camera_pos();
    let dx = mean_x - camera.x();
    let dy = mean_y - camera.y();
    let dz = mean_z - camera.z();
    let inv_len = 1.0f32 / f32::sqrt(dx * dx + dy * dy + dz * dz);
    let view_x = dx * inv_len;
    let view_y = dy * inv_len;
    let view_z = dz * inv_len;

    let basis = sh_basis(lane, sh_degree, view_x, view_y, view_z);
    let basis_0 = plane_shuffle(basis, index_0 / 3u32);
    let basis_1 = plane_shuffle(basis, index_1 / 3u32);
    let basis_2 = plane_shuffle(basis, index_2 / 3u32);
    let grad_0 = basis_0 * color_component(index_0, v_color_r, v_color_g, v_color_b);
    let grad_1 = basis_1 * color_component(index_1, v_color_r, v_color_g, v_color_b);
    let grad_2 = basis_2 * color_component(index_2, v_color_r, v_color_g, v_color_b);

    if active && index_0 < row_len {
        v_coeffs[(row_base + index_0) as usize] = grad_0;
    }
    if active && index_1 < row_len {
        v_coeffs[(row_base + index_1) as usize] = grad_1;
    }
    if active && index_2 < row_len {
        v_coeffs[(row_base + index_2) as usize] = grad_2;
    }
}
