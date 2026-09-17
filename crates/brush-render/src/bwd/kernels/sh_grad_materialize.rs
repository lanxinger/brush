//! Coalesced compact spherical-harmonic gradients.
//! One SIMD plane writes each visible row plus the zero sentinel.

use crate::kernels::sh::{num_sh_coeffs, sh_basis, sh_color_component};
use crate::kernels::types::ProjectUniforms;
use burn::cubecl;
use burn::cubecl::cube;
use burn::cubecl::prelude::*;

pub const PLANE_SIZE: u32 = 32;
pub const WG_SIZE: u32 = 256;
pub const SPLATS_PER_WG: u32 = WG_SIZE / PLANE_SIZE;

#[cube(launch)]
pub fn materialize_sh_grad_kernel(
    transforms: &Tensor<f32>,
    global_from_compact_gid: &Tensor<u32>,
    v_combined: &Tensor<f32>,
    v_coeffs: &mut Tensor<f32>,
    u: ProjectUniforms,
    #[comptime] sh_degree: u32,
) {
    let row = CUBE_POS as u32 * SPLATS_PER_WG + PLANE_POS;
    let lane = UNIT_POS_PLANE;
    let active = row <= u.num_visible;
    let has_grad = active && row > 0u32;
    let compact_gid = max(row, 1u32) - 1u32;
    let row_len = comptime![num_sh_coeffs(sh_degree) * 3u32];
    let row_base = row * row_len;
    let index_0 = lane;
    let index_1 = lane + PLANE_SIZE;
    let index_2 = lane + 2u32 * PLANE_SIZE;

    // Row zero supplies exact zeros for culled splats. Inactive planes
    // skip all SH polynomial and shuffle work.
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

    let global_gid = global_from_compact_gid[compact_gid as usize];
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
    let grad_0 = basis_0 * sh_color_component(index_0, v_color_r, v_color_g, v_color_b);
    let grad_1 = basis_1 * sh_color_component(index_1, v_color_r, v_color_g, v_color_b);
    let grad_2 = basis_2 * sh_color_component(index_2, v_color_r, v_color_g, v_color_b);

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
