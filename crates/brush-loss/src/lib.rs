//! Image-loss kernels for Brush.
//!
//! GT lives on the GPU as a `Tensor<u32>` of shape `[H, W]`, where each u32
//! packs `[r8, g8, b8, a8]` (LSB → MSB). Conversion to f32 happens inside
//! the kernels via shift-and-divide-by-255. No f32 GT image is ever
//! materialised on the autograd tape.
//!
//! Public surface:
//! - [`image_loss`]: mean of `l1_w * |pred - gt_eff| + ssim_w * ssim(pred,
//!   gt_eff)` over the RGB pixels, plus an optional weighted alpha-match
//!   term, with optional background-compositing of GT (`gt_eff = gt + (1 -
//!   gt.a) * bg`) and optional mask multiplication (`out = out * gt.a`)
//!   folded into the kernel. The differentiable entry; the kernel reduces
//!   its own 16×16 tiles so the per-pixel map never touches memory.
//! - [`image_loss_eval`]: forward-only per-pixel loss map, for eval.
//! - [`psnr_from_mse`] / [`psnr`]: PSNR in dB, with MSE floored so identical
//!   images report 100 dB rather than infinity.
//!
//! Backward recomputes SSIM partials inline so no per-pixel state survives
//! across the autograd tape.

use brush_cube::create_tensor;
use brush_cube::fusion::register_custom;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::{Autodiff, AutodiffBackend};
use burn::{
    backend::{
        Backend, TensorMetadata,
        autodiff::{
            checkpoint::{base::Checkpointer, strategy::NoCheckpointing},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
        },
        tensor::{FloatTensor, IntTensor},
    },
    tensor::{DType, Int, Shape, Tensor},
};
use burn_cubecl::{CubeBackend, kernel::into_contiguous, tensor::CubeTensor};
use burn_fusion::Fusion;
use glam::Vec3;

#[cfg(all(
    feature = "native-msl",
    target_os = "macos",
    target_arch = "aarch64",
    not(target_family = "wasm")
))]
fn use_saved_loss_partials() -> bool {
    use std::sync::OnceLock;

    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let enabled = brush_render::native_msl::option_requested(
            brush_render::native_msl::SAVED_LOSS_PARTIALS_ENV,
        );
        if enabled {
            tracing::warn!("experimental native-MSL saved loss partials enabled");
        }
        enabled
    })
}

#[cfg(not(all(
    feature = "native-msl",
    target_os = "macos",
    target_arch = "aarch64",
    not(target_family = "wasm")
)))]
fn use_saved_loss_partials() -> bool {
    false
}

mod kernels {
    use burn::cubecl;
    use burn::cubecl::cube;
    use burn::cubecl::frontend::CompilationArg;
    use burn::cubecl::frontend::IndexMutExpand;
    use burn::cubecl::prelude::*;

    /// 11-tap Gaussian weights at sigma = 1.5, normalised to sum to 1.
    /// Called from `comptime!` so it runs once per kernel build, baking each
    /// weight as an f32 literal into the generated kernel.
    fn gauss_taps() -> [f32; 11] {
        let sigma = 1.5_f32;
        let mut w = [0.0_f32; 11];
        let mut sum = 0.0;
        for (i, w) in w.iter_mut().enumerate() {
            let x = i as f32 - 5.0;
            *w = (-x * x / (2.0 * sigma * sigma)).exp();
            sum += *w;
        }
        for w in &mut w {
            *w /= sum;
        }
        w
    }

    pub const BLOCK_X: u32 = 16;
    pub const BLOCK_Y: u32 = 16;
    const HALO: u32 = 5;
    const SHARED_X: u32 = BLOCK_X + 2 * HALO; // 26
    const SHARED_Y: u32 = BLOCK_Y + 2 * HALO; // 26
    pub const BWD_TILE_SMALL: u32 = 8;
    pub const BWD_TILE_LARGE: u32 = 16;

    const fn backward_shared_elements(tile: u32) -> usize {
        let shared = tile + 2 * HALO;
        let extended = tile + 4 * HALO;
        (extended * extended * 2 + extended * shared * 5) as usize
    }

    /// Shared-memory footprint of the fast 16x16 f32 specialization.
    pub const BWD_LARGE_SHARED_BYTES: usize =
        backward_shared_elements(BWD_TILE_LARGE) * size_of::<f32>();

    const C1: f32 = 0.01 * 0.01;
    const C2: f32 = 0.03 * 0.03;
    const INV_255: f32 = 1.0 / 255.0;

    /// Read `pred[y, x, c]` (`[H, W, C]` layout, `cn` channels) returning
    /// zero for out-of-bounds. The
    /// `if/else` form generated a non-uniform branch that Naga's MSL
    /// backend tracked into the post-load `workgroupBarrier()`; we use
    /// `select` to keep control flow uniform. The read always executes —
    /// for OOB threads `(y, x) = (0, 0)` (see `coords`), so the index
    /// `c` is always in-bounds.
    #[cube]
    fn read_pred<F: Float>(
        pred: &Tensor<F>,
        c: u32,
        y: u32,
        x: u32,
        oob: bool,
        cn: u32,
        w: u32,
    ) -> F {
        let v = pred[((y * w + x) * cn + c) as usize];
        select(oob, F::cast_from(0.0_f32), v)
    }

    /// Sum of `value` over the 16x16 forward tile. A shared-memory tree
    /// rather than plane ops: WGSL only allows subgroup builtins in
    /// one-dimensional workgroups, and this kernel's is 2D.
    #[cube]
    fn tile_sum<F: Float>(value: F) -> F {
        let mut buf = Shared::new_slice((BLOCK_X * BLOCK_Y) as usize);
        let tid = UNIT_POS_Y * BLOCK_X + UNIT_POS_X;
        buf[tid as usize] = value;
        sync_cube();
        #[unroll]
        for s in 0u32..8u32 {
            let stride = 128u32 >> s;
            if tid < stride {
                let other = buf[(tid + stride) as usize];
                buf[tid as usize] += other;
            }
            sync_cube();
        }
        buf[0]
    }

    /// Number of forward tiles across the image; partial sums are laid out
    /// `[4, tiles_y * tiles_x]` in forward-tile order.
    #[cube]
    #[allow(clippy::manual_div_ceil)]
    fn tiles_x(w: u32) -> u32 {
        (w + BLOCK_X - 1u32) / BLOCK_X
    }

    #[cube]
    #[allow(clippy::manual_div_ceil)]
    fn num_tiles(h: u32, w: u32) -> u32 {
        tiles_x(w) * ((h + BLOCK_Y - 1u32) / BLOCK_Y)
    }

    /// Upstream gradient for channel `c` at pixel `(y, x)`: the gradient of
    /// the loss w.r.t. that pixel's forward-tile partial sum, times the
    /// channel weight the forward folded into that sum.
    #[cube]
    #[allow(clippy::too_many_arguments)]
    fn chain_at<F: Float>(
        dl_dpartials: &Tensor<F>,
        c: u32,
        y: u32,
        x: u32,
        h: u32,
        w: u32,
        weight: F,
    ) -> F {
        let tile = (y / BLOCK_Y) * tiles_x(w) + x / BLOCK_X;
        dl_dpartials[(c * num_tiles(h, w) + tile) as usize] * weight
    }

    #[cube]
    fn ssim_partials<F: Float>(mu1: F, mu2: F, a: F, b: F, c_top: F, d_top: F) -> (F, F, F) {
        let zero = F::cast_from(0.0_f32);
        let one = F::cast_from(1.0_f32);
        let two = F::cast_from(2.0_f32);
        let inv_ab = one / (a * b);
        let cd = c_top * d_top * inv_ab;
        let clamped = cd < F::cast_from(-1.0_f32) || cd > one;
        let dmu1 = if clamped {
            zero
        } else {
            two * mu2 * inv_ab * (d_top - c_top) - two * mu1 * cd * (one / a - one / b)
        };
        let dsigma1 = if clamped { zero } else { -cd / b };
        let dsigma12 = if clamped { zero } else { two * c_top * inv_ab };
        (dmu1, dsigma1, dsigma12)
    }

    /// Read one saved SSIM partial, returning zero for image padding. The
    /// cache is `SoA` `[partial=3, rgb=3, H, W]`, flattened as `[9, H, W]`.
    #[cube]
    fn read_saved_partial<F: Float>(
        partials: &Tensor<F>,
        partial: u32,
        c: u32,
        y: u32,
        x: u32,
        oob: bool,
        h: u32,
        w: u32,
    ) -> F {
        let idx = (((partial * 3u32 + c) * h + y) * w + x) as usize;
        select(oob, F::cast_from(0.0_f32), partials[idx])
    }

    /// Read one `[r8 g8 b8 a8]`-packed pixel from `gt_packed`. Returns the
    /// requested colour byte and the alpha byte, both in `[0, 1]`. The alpha
    /// is always returned so it's available for compositing or masking when
    /// those flags are on. As with `read_pred`, the body runs unconditionally
    /// and `oob` is folded in via `select` so we don't emit a non-uniform
    /// branch before a workgroup barrier.
    #[cube]
    fn read_gt<F: Float>(
        gt_packed: &Tensor<u32>,
        c: u32,
        y: u32,
        x: u32,
        oob: bool,
        w: u32,
    ) -> (F, F) {
        let val = gt_packed[(y * w + x) as usize];
        let byte_c = f32::cast_from((val >> (c * 8u32)) & 0xffu32);
        let byte_a = f32::cast_from((val >> 24u32) & 0xffu32);
        let zero = F::cast_from(0.0_f32);
        let gt_c = F::cast_from(byte_c * INV_255);
        let gt_a = F::cast_from(byte_a * INV_255);
        (select(oob, zero, gt_c), select(oob, zero, gt_a))
    }

    /// Map a tile-local position offset by `halo` to global image coords.
    #[cube]
    fn coords(
        tile_y0: u32,
        tile_x0: u32,
        local_y: u32,
        local_x: u32,
        #[comptime] halo: u32,
        h: u32,
        w: u32,
    ) -> (u32, u32, bool) {
        let total_y = tile_y0 + local_y;
        let total_x = tile_x0 + local_x;
        let oob_under = total_y < halo || total_x < halo;
        let zero = u32::cast_from(0u32);
        let gy = select(oob_under, zero, total_y - halo);
        let gx = select(oob_under, zero, total_x - halo);
        (gy, gx, oob_under || gy >= h || gx >= w)
    }

    #[cube]
    fn gw<F: Float>(#[comptime] i: u32) -> F {
        F::new(comptime![gauss_taps()[i as usize]])
    }

    /// Forward: produce the L1 + SSIM loss map. When dispatched with `C = 4`,
    /// the workgroup at `c == 3` produces `|pred.a - gt.a|` into the alpha
    /// channel of the loss map — folding the previously-separate alpha-match
    /// kernel into the same launch.
    ///
    /// Comptime flags:
    /// - `composite`: apply `gt + (1 - gt.a) * bg` to the gt sample. Set when
    ///   the source has real alpha and `bg != 0`; opaque/synthesised alpha or
    ///   zero bg make the math a no-op so callers gate it off to skip the work.
    /// - `mask`: multiply the loss-map output by `gt.a` per pixel.
    #[allow(clippy::assign_op_pattern, clippy::fn_params_excessive_bools)]
    #[cube(launch)]
    pub fn image_loss_forward_kernel<F: Float>(
        pred: &Tensor<F>,
        gt_packed: &Tensor<u32>,
        loss_map: &mut Tensor<F>,
        partials: &mut Tensor<F>,
        saved_partials: ComptimeOption<&mut Tensor<F>>,
        h: u32,
        w: u32,
        cn: u32,
        l1_weight: f32,
        ssim_weight: f32,
        rgb_weight: f32,
        alpha_weight: f32,
        bg_r: f32,
        bg_g: f32,
        bg_b: f32,
        #[comptime] composite: bool,
        #[comptime] mask: bool,
        #[comptime] reduce: bool,
        #[comptime] l1_only: bool,
    ) {
        let c = CUBE_POS_Z;
        let tile_y0 = CUBE_POS_Y * BLOCK_Y;
        let tile_x0 = CUBE_POS_X * BLOCK_X;
        let pix_y = tile_y0 + UNIT_POS_Y;
        let pix_x = tile_x0 + UNIT_POS_X;
        let in_bounds = pix_x < w && pix_y < h;
        let tile_id = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;

        // Alpha-match channel, only launched when matching: per-pixel
        // `|pred - gt.a|`, no blur.
        if c == 3u32 {
            let mut v = F::cast_from(0.0_f32);
            if in_bounds {
                let idx = ((pix_y * w + pix_x) * cn + 3u32) as usize;
                let (_, gt_a) = read_gt::<F>(gt_packed, 0u32, pix_y, pix_x, false, w);
                v = F::abs(pred[idx] - gt_a);
                if mask {
                    v = v * gt_a;
                }
            }
            if reduce {
                let total = tile_sum::<F>(v);
                if UNIT_POS == 0u32 {
                    partials[(3u32 * num_tiles(h, w) + tile_id) as usize] =
                        total * F::cast_from(alpha_weight);
                }
            } else if in_bounds {
                loss_map[((pix_y * w + pix_x) * cn + 3u32) as usize] = v;
            }
            terminate!();
        }

        if l1_only {
            let mut value = F::cast_from(0.0_f32);
            if in_bounds {
                let idx = ((pix_y * w + pix_x) * cn + c) as usize;
                let (gt_c, gt_a) = read_gt::<F>(gt_packed, c, pix_y, pix_x, false, w);
                let bg = if c == 0u32 {
                    bg_r
                } else if c == 1u32 {
                    bg_g
                } else {
                    bg_b
                };
                let gt_eff = if composite {
                    gt_c + (F::cast_from(1.0_f32) - gt_a) * F::cast_from(bg)
                } else {
                    gt_c
                };
                value = F::cast_from(l1_weight) * F::abs(pred[idx] - gt_eff);
                if mask {
                    value = value * gt_a;
                }
            }
            if reduce {
                let total = tile_sum::<F>(value);
                if UNIT_POS == 0u32 {
                    partials[(c * num_tiles(h, w) + tile_id) as usize] =
                        total * F::cast_from(rgb_weight);
                }
            } else if in_bounds {
                loss_map[((pix_y * w + pix_x) * cn + c) as usize] = value;
            }
            terminate!();
        }

        // Tile + halo of (pred, gt_eff_c) interleaved as 2 floats. cubecl's
        // WGSL backend over-counts shared memory by 2x (it reports double the
        // bytes actually declared in WGSL), so this kernel has to stay under
        // ~half the real Apple Metal threadgroup budget. gt_a was previously
        // carried here too; the mask=true path now re-reads it at the centre.
        let mut s_tile = Shared::new_slice((SHARED_Y * SHARED_X * 2) as usize);
        let mut x_conv = Shared::new_slice((SHARED_Y * BLOCK_X * 5) as usize);

        let bg_c = if composite {
            if c == 0u32 {
                F::cast_from(bg_r)
            } else if c == 1u32 {
                F::cast_from(bg_g)
            } else {
                F::cast_from(bg_b)
            }
        } else {
            F::cast_from(0.0_f32)
        };

        let thread_rank = UNIT_POS_Y * BLOCK_X + UNIT_POS_X;
        let threads = BLOCK_X * BLOCK_Y;
        let tile_size = SHARED_Y * SHARED_X;
        #[unroll]
        for s in 0u32..3u32 {
            let tid = s * threads + thread_rank;
            if tid < tile_size {
                let local_y = tid / SHARED_X;
                let local_x = tid % SHARED_X;
                let (gy, gx, oob) = coords(tile_y0, tile_x0, local_y, local_x, HALO, h, w);
                let pv = read_pred::<F>(pred, c, gy, gx, oob, cn, w);
                let (gt_c, gt_a) = read_gt::<F>(gt_packed, c, gy, gx, oob, w);
                let gt_eff = if composite {
                    gt_c + (F::cast_from(1.0_f32) - gt_a) * bg_c
                } else {
                    gt_c
                };
                let base = ((local_y * SHARED_X + local_x) * 2u32) as usize;
                s_tile[base] = pv;
                s_tile[base + 1] = gt_eff;
            }
        }
        sync_cube();

        // Horizontal 11-tap blur over (pred, gt_eff_c) -> 5 sums per pixel.
        let lx = UNIT_POS_X + HALO;
        #[unroll]
        for pass in 0u32..2u32 {
            let ly = UNIT_POS_Y + pass * BLOCK_Y;
            if ly < SHARED_Y {
                let mut sum_x = F::cast_from(0.0_f32);
                let mut sum_x2 = F::cast_from(0.0_f32);
                let mut sum_y = F::cast_from(0.0_f32);
                let mut sum_y2 = F::cast_from(0.0_f32);
                let mut sum_xy = F::cast_from(0.0_f32);
                #[unroll]
                for d in 1u32..6u32 {
                    let w_d = gw::<F>(comptime![5u32 - d]);
                    let il = (ly * SHARED_X + (lx - d)) as usize;
                    let ir = (ly * SHARED_X + (lx + d)) as usize;
                    let xl = s_tile[il * 2];
                    let yl = s_tile[il * 2 + 1];
                    let xr = s_tile[ir * 2];
                    let yr = s_tile[ir * 2 + 1];
                    sum_x += (xl + xr) * w_d;
                    sum_x2 += (xl * xl + xr * xr) * w_d;
                    sum_y += (yl + yr) * w_d;
                    sum_y2 += (yl * yl + yr * yr) * w_d;
                    sum_xy += (xl * yl + xr * yr) * w_d;
                }
                let ic = (ly * SHARED_X + lx) as usize;
                let xc = s_tile[ic * 2];
                let yc = s_tile[ic * 2 + 1];
                let wc = gw::<F>(5u32);
                sum_x += xc * wc;
                sum_x2 += xc * xc * wc;
                sum_y += yc * wc;
                sum_y2 += yc * yc * wc;
                sum_xy += xc * yc * wc;
                let base = ((ly * BLOCK_X + UNIT_POS_X) * 5) as usize;
                x_conv[base] = sum_x;
                x_conv[base + 1] = sum_x2;
                x_conv[base + 2] = sum_y;
                x_conv[base + 3] = sum_y2;
                x_conv[base + 4] = sum_xy;
            }
        }
        sync_cube();

        // Vertical 11-tap blur, then derive SSIM and emit L1 + SSIM loss.
        let ly = UNIT_POS_Y + HALO;
        let lx = UNIT_POS_X;
        let mut out0 = F::cast_from(0.0_f32);
        let mut out1 = F::cast_from(0.0_f32);
        let mut out2 = F::cast_from(0.0_f32);
        let mut out3 = F::cast_from(0.0_f32);
        let mut out4 = F::cast_from(0.0_f32);
        #[unroll]
        for d in 1u32..6u32 {
            let w_d = gw::<F>(comptime![5u32 - d]);
            let bt = (((ly - d) * BLOCK_X + lx) * 5) as usize;
            let bb = (((ly + d) * BLOCK_X + lx) * 5) as usize;
            out0 += (x_conv[bt] + x_conv[bb]) * w_d;
            out1 += (x_conv[bt + 1] + x_conv[bb + 1]) * w_d;
            out2 += (x_conv[bt + 2] + x_conv[bb + 2]) * w_d;
            out3 += (x_conv[bt + 3] + x_conv[bb + 3]) * w_d;
            out4 += (x_conv[bt + 4] + x_conv[bb + 4]) * w_d;
        }
        let bc = ((ly * BLOCK_X + lx) * 5) as usize;
        let wc = gw::<F>(5u32);
        out0 += x_conv[bc] * wc;
        out1 += x_conv[bc + 1] * wc;
        out2 += x_conv[bc + 2] * wc;
        out3 += x_conv[bc + 3] * wc;
        out4 += x_conv[bc + 4] * wc;

        let mut loss_out = F::cast_from(0.0_f32);
        if in_bounds {
            let zero = F::cast_from(0.0_f32);
            let two = F::cast_from(2.0_f32);
            let mu1 = out0;
            let mu2 = out2;
            let mu1_sq = mu1 * mu1;
            let mu2_sq = mu2 * mu2;
            let sigma1_sq = F::max(zero, out1 - mu1_sq);
            let sigma2_sq = F::max(zero, out3 - mu2_sq);
            let sigma12 = out4 - mu1 * mu2;
            let a = mu1_sq + mu2_sq + F::new(C1);
            let b = sigma1_sq + sigma2_sq + F::new(C2);
            let c_top = two * mu1 * mu2 + F::new(C1);
            let d_top = two * sigma12 + F::new(C2);
            let raw = (c_top * d_top) / (a * b);
            let val = clamp(raw, F::cast_from(-1.0_f32), F::cast_from(1.0_f32));

            let centre = ((UNIT_POS_Y + HALO) * SHARED_X + (UNIT_POS_X + HALO)) as usize;
            let p1 = s_tile[centre * 2];
            let p2 = s_tile[centre * 2 + 1];
            let l1 = F::abs(p1 - p2);
            let mut loss_v = F::cast_from(l1_weight) * l1 + F::cast_from(ssim_weight) * val;
            if mask {
                let (_, gt_a) = read_gt::<F>(gt_packed, c, pix_y, pix_x, false, w);
                loss_v = loss_v * gt_a;
            }
            loss_out = loss_v;
            #[comptime]
            if let ComptimeOption::Some(saved_partials) = saved_partials {
                let (dmu1, dsigma1, dsigma12) = ssim_partials::<F>(mu1, mu2, a, b, c_top, d_top);
                let pixel_idx = (pix_y * w + pix_x) as usize;
                saved_partials[(c * h * w) as usize + pixel_idx] = dmu1;
                saved_partials[((3u32 + c) * h * w) as usize + pixel_idx] = dsigma1;
                saved_partials[((6u32 + c) * h * w) as usize + pixel_idx] = dsigma12;
            }
        }
        if reduce {
            // Weighted per-tile sum; the host adds up the tiles. Keeps the
            // per-pixel map off memory entirely.
            let total = tile_sum::<F>(loss_out);
            if UNIT_POS == 0u32 {
                partials[(c * num_tiles(h, w) + tile_id) as usize] =
                    total * F::cast_from(rgb_weight);
            }
        } else if in_bounds {
            loss_map[((pix_y * w + pix_x) * cn + c) as usize] = loss_out;
        }
    }

    /// Backward with adaptive tiles and optional saved SSIM partials.
    #[allow(clippy::assign_op_pattern, clippy::fn_params_excessive_bools)]
    #[cube(launch)]
    pub fn image_loss_backward_kernel<F: Float>(
        pred: &Tensor<F>,
        gt_packed: &Tensor<u32>,
        dl_dpartials: &Tensor<F>,
        saved_partials: ComptimeOption<&Tensor<F>>,
        dl_dpred: &mut Tensor<F>,
        h: u32,
        w: u32,
        cn: u32,
        l1_weight: f32,
        ssim_weight: f32,
        rgb_weight: f32,
        alpha_weight: f32,
        bg_r: f32,
        bg_g: f32,
        bg_b: f32,
        #[comptime] composite: bool,
        #[comptime] mask: bool,
        #[comptime] tile: u32,
        #[comptime] alpha_match: bool,
        #[comptime] l1_only: bool,
    ) {
        let shared = comptime![tile + 2u32 * HALO];
        let extended = comptime![tile + 4u32 * HALO];
        let threads = comptime![tile * tile];
        let load_iters = comptime![(extended * extended).div_ceil(threads)];
        let hblur_iters = comptime![(extended * shared).div_ceil(threads)];
        let partial_iters = comptime![(shared * shared).div_ceil(threads)];
        let inner_h_passes = comptime![shared.div_ceil(tile)];
        let saved = comptime![saved_partials.is_some()];

        let c = CUBE_POS_Z;
        let tile_y0 = CUBE_POS_Y * tile;
        let tile_x0 = CUBE_POS_X * tile;
        let pix_y = tile_y0 + UNIT_POS_Y;
        let pix_x = tile_x0 + UNIT_POS_X;

        // Alpha channel: sign-of-diff when matching alpha, else a zero
        // gradient (the output is uninitialised, so it must be written).
        if c == 3u32 {
            if pix_x < w && pix_y < h {
                let idx = ((pix_y * w + pix_x) * cn + 3u32) as usize;
                let mut g = F::cast_from(0.0_f32);
                if alpha_match {
                    let (_, gt_a) = read_gt::<F>(gt_packed, 0u32, pix_y, pix_x, false, w);
                    let diff = pred[idx] - gt_a;
                    let zero = F::cast_from(0.0_f32);
                    let sign = if diff > zero {
                        F::cast_from(1.0_f32)
                    } else if diff < zero {
                        F::cast_from(-1.0_f32)
                    } else {
                        zero
                    };
                    let mut chain = chain_at::<F>(
                        dl_dpartials,
                        3u32,
                        pix_y,
                        pix_x,
                        h,
                        w,
                        F::cast_from(alpha_weight),
                    );
                    if mask {
                        chain = chain * gt_a;
                    }
                    g = sign * chain;
                }
                dl_dpred[idx] = g;
            }
            terminate!();
        }

        if l1_only {
            if pix_x < w && pix_y < h {
                let idx = ((pix_y * w + pix_x) * cn + c) as usize;
                let (gt_c, gt_a) = read_gt::<F>(gt_packed, c, pix_y, pix_x, false, w);
                let bg = if c == 0u32 {
                    bg_r
                } else if c == 1u32 {
                    bg_g
                } else {
                    bg_b
                };
                let gt_eff = if composite {
                    gt_c + (F::cast_from(1.0_f32) - gt_a) * F::cast_from(bg)
                } else {
                    gt_c
                };
                let diff = pred[idx] - gt_eff;
                let zero = F::cast_from(0.0_f32);
                let sign = if diff > zero {
                    F::cast_from(1.0_f32)
                } else if diff < zero {
                    F::cast_from(-1.0_f32)
                } else {
                    zero
                };
                let mut chain = chain_at::<F>(
                    dl_dpartials,
                    c,
                    pix_y,
                    pix_x,
                    h,
                    w,
                    F::cast_from(rgb_weight),
                );
                if mask {
                    chain = chain * gt_a;
                }
                dl_dpred[idx] = F::cast_from(l1_weight) * sign * chain;
            }
            terminate!();
        }

        // In recompute mode buf_a/b hold the image tile and first blur before
        // being reused for chain*partials and the second blur. The saved mode
        // compiles to the smaller allocations only (13,104 bytes at 16x16).
        let mut buf_a = Shared::new_slice(comptime![if saved {
            (shared * shared * 3u32) as usize
        } else {
            (extended * extended * 2u32) as usize
        }]);
        let mut buf_b = Shared::new_slice(comptime![if saved {
            (shared * tile * 3u32) as usize
        } else {
            (extended * shared * 5u32) as usize
        }]);

        let bg_c = if composite {
            if c == 0u32 {
                F::cast_from(bg_r)
            } else if c == 1u32 {
                F::cast_from(bg_g)
            } else {
                F::cast_from(bg_b)
            }
        } else {
            F::cast_from(0.0_f32)
        };

        let thread_rank = UNIT_POS_Y * tile + UNIT_POS_X;

        #[comptime]
        match saved_partials {
            ComptimeOption::None => {
                // Load pred and effective-gt with halo of 2*HALO into buf_a.
                let ext_size = extended * extended;
                #[unroll]
                for s in 0u32..load_iters {
                    let tid = s * threads + thread_rank;
                    if tid < ext_size {
                        let local_y = tid / extended;
                        let local_x = tid % extended;
                        let (gy, gx, oob) =
                            coords(tile_y0, tile_x0, local_y, local_x, 2u32 * HALO, h, w);
                        let pv = read_pred::<F>(pred, c, gy, gx, oob, cn, w);
                        let (gt_c, gt_a) = read_gt::<F>(gt_packed, c, gy, gx, oob, w);
                        let gt_eff = if composite {
                            gt_c + (F::cast_from(1.0_f32) - gt_a) * bg_c
                        } else {
                            gt_c
                        };
                        let base = ((local_y * extended + local_x) * 2u32) as usize;
                        buf_a[base] = pv;
                        buf_a[base + 1] = gt_eff;
                    }
                }
                sync_cube();

                // Horizontal blur over the extended tile.
                let horiz_size = extended * shared;
                #[unroll]
                for s in 0u32..hblur_iters {
                    let tid = s * threads + thread_rank;
                    if tid < horiz_size {
                        let row_y = tid / shared;
                        let col_x = tid % shared;
                        let center = col_x + HALO;
                        let mut sum_x = F::cast_from(0.0_f32);
                        let mut sum_x2 = F::cast_from(0.0_f32);
                        let mut sum_y = F::cast_from(0.0_f32);
                        let mut sum_y2 = F::cast_from(0.0_f32);
                        let mut sum_xy = F::cast_from(0.0_f32);
                        #[unroll]
                        for d in 1u32..6u32 {
                            let w_d = gw::<F>(comptime![5u32 - d]);
                            let il = ((row_y * extended + (center - d)) * 2u32) as usize;
                            let ir = ((row_y * extended + (center + d)) * 2u32) as usize;
                            let xl = buf_a[il];
                            let yl = buf_a[il + 1];
                            let xr = buf_a[ir];
                            let yr = buf_a[ir + 1];
                            sum_x += (xl + xr) * w_d;
                            sum_x2 += (xl * xl + xr * xr) * w_d;
                            sum_y += (yl + yr) * w_d;
                            sum_y2 += (yl * yl + yr * yr) * w_d;
                            sum_xy += (xl * yl + xr * yr) * w_d;
                        }
                        let ic = ((row_y * extended + center) * 2u32) as usize;
                        let xc = buf_a[ic];
                        let yc = buf_a[ic + 1];
                        let wc = gw::<F>(5u32);
                        sum_x += xc * wc;
                        sum_x2 += xc * xc * wc;
                        sum_y += yc * wc;
                        sum_y2 += yc * yc * wc;
                        sum_xy += xc * yc * wc;
                        let base = ((row_y * shared + col_x) * 5u32) as usize;
                        buf_b[base] = sum_x;
                        buf_b[base + 1] = sum_x2;
                        buf_b[base + 2] = sum_y;
                        buf_b[base + 3] = sum_y2;
                        buf_b[base + 4] = sum_xy;
                    }
                }
                sync_cube();

                // Vertical blur, derive SSIM partials, multiply by chain * (mask if any).
                // Reuses buf_a (image tile is dead) for chain*partials.
                let partial_size = shared * shared;
                #[unroll]
                for s in 0u32..partial_iters {
                    let tid = s * threads + thread_rank;
                    if tid < partial_size {
                        let part_y = tid / shared;
                        let part_x = tid % shared;
                        let center = part_y + HALO;

                        let mut out0 = F::cast_from(0.0_f32);
                        let mut out1 = F::cast_from(0.0_f32);
                        let mut out2 = F::cast_from(0.0_f32);
                        let mut out3 = F::cast_from(0.0_f32);
                        let mut out4 = F::cast_from(0.0_f32);
                        #[unroll]
                        for d in 1u32..6u32 {
                            let w_d = gw::<F>(comptime![5u32 - d]);
                            let bt = (((center - d) * shared + part_x) * 5u32) as usize;
                            let bb = (((center + d) * shared + part_x) * 5u32) as usize;
                            out0 += (buf_b[bt] + buf_b[bb]) * w_d;
                            out1 += (buf_b[bt + 1] + buf_b[bb + 1]) * w_d;
                            out2 += (buf_b[bt + 2] + buf_b[bb + 2]) * w_d;
                            out3 += (buf_b[bt + 3] + buf_b[bb + 3]) * w_d;
                            out4 += (buf_b[bt + 4] + buf_b[bb + 4]) * w_d;
                        }
                        let bc = ((center * shared + part_x) * 5u32) as usize;
                        let wc = gw::<F>(5u32);
                        out0 += buf_b[bc] * wc;
                        out1 += buf_b[bc + 1] * wc;
                        out2 += buf_b[bc + 2] * wc;
                        out3 += buf_b[bc + 3] * wc;
                        out4 += buf_b[bc + 4] * wc;

                        let zero = F::cast_from(0.0_f32);
                        let two = F::cast_from(2.0_f32);
                        let mu1 = out0;
                        let mu2 = out2;
                        let mu1_sq = mu1 * mu1;
                        let mu2_sq = mu2 * mu2;
                        let sigma1_sq = F::max(zero, out1 - mu1_sq);
                        let sigma2_sq = F::max(zero, out3 - mu2_sq);
                        let sigma12 = out4 - mu1 * mu2;
                        let a = mu1_sq + mu2_sq + F::new(C1);
                        let b = sigma1_sq + sigma2_sq + F::new(C2);
                        let c_top = two * mu1 * mu2 + F::new(C1);
                        let d_top = two * sigma12 + F::new(C2);
                        let (dmu1, dsigma1, dsigma12) =
                            ssim_partials::<F>(mu1, mu2, a, b, c_top, d_top);

                        let (gy, gx, oob) = coords(tile_y0, tile_x0, part_y, part_x, HALO, h, w);
                        let mut chain = select(
                            oob,
                            F::cast_from(0.0_f32),
                            chain_at::<F>(dl_dpartials, c, gy, gx, h, w, F::cast_from(rgb_weight)),
                        );
                        if mask {
                            let (_unused, gt_a) = read_gt::<F>(gt_packed, c, gy, gx, oob, w);
                            chain = chain * gt_a;
                        }

                        let base = ((part_y * shared + part_x) * 3u32) as usize;
                        buf_a[base] = dmu1 * chain;
                        buf_a[base + 1] = dsigma1 * chain;
                        buf_a[base + 2] = dsigma12 * chain;
                    }
                }
                sync_cube();
            }
            ComptimeOption::Some(saved_partials) => {
                // Load saved partials with one halo, fold in the arbitrary
                // upstream chain and optional alpha mask, then join the common
                // second-blur/finalization path below.
                let partial_size = shared * shared;
                #[unroll]
                for s in 0u32..partial_iters {
                    let tid = s * threads + thread_rank;
                    if tid < partial_size {
                        let part_y = tid / shared;
                        let part_x = tid % shared;
                        let (gy, gx, oob) = coords(tile_y0, tile_x0, part_y, part_x, HALO, h, w);
                        let mut chain = select(
                            oob,
                            F::cast_from(0.0_f32),
                            chain_at::<F>(dl_dpartials, c, gy, gx, h, w, F::cast_from(rgb_weight)),
                        );
                        if mask {
                            let (_unused, gt_a) = read_gt::<F>(gt_packed, c, gy, gx, oob, w);
                            chain = chain * gt_a;
                        }
                        let base = ((part_y * shared + part_x) * 3u32) as usize;
                        buf_a[base] =
                            read_saved_partial::<F>(saved_partials, 0u32, c, gy, gx, oob, h, w)
                                * chain;
                        buf_a[base + 1] =
                            read_saved_partial::<F>(saved_partials, 1u32, c, gy, gx, oob, h, w)
                                * chain;
                        buf_a[base + 2] =
                            read_saved_partial::<F>(saved_partials, 2u32, c, gy, gx, oob, h, w)
                                * chain;
                    }
                }
                sync_cube();
            }
        }

        // Second horizontal blur over chain * partials.
        // Reuses buf_b (1st-blur sums are dead) for the inner-blur output.
        let lx_b = UNIT_POS_X + HALO;
        #[unroll]
        for pass in 0u32..inner_h_passes {
            let ly_b = UNIT_POS_Y + pass * tile;
            if ly_b < shared {
                let mut a0 = F::cast_from(0.0_f32);
                let mut a1 = F::cast_from(0.0_f32);
                let mut a2 = F::cast_from(0.0_f32);
                #[unroll]
                for d in 1u32..6u32 {
                    let w_d = gw::<F>(comptime![5u32 - d]);
                    let il = ((ly_b * shared + (lx_b - d)) * 3u32) as usize;
                    let ir = ((ly_b * shared + (lx_b + d)) * 3u32) as usize;
                    a0 += (buf_a[il] + buf_a[ir]) * w_d;
                    a1 += (buf_a[il + 1] + buf_a[ir + 1]) * w_d;
                    a2 += (buf_a[il + 2] + buf_a[ir + 2]) * w_d;
                }
                let ic = ((ly_b * shared + lx_b) * 3u32) as usize;
                let wc = gw::<F>(5u32);
                a0 += buf_a[ic] * wc;
                a1 += buf_a[ic + 1] * wc;
                a2 += buf_a[ic + 2] * wc;
                let base = ((ly_b * tile + UNIT_POS_X) * 3u32) as usize;
                buf_b[base] = a0;
                buf_b[base + 1] = a1;
                buf_b[base + 2] = a2;
            }
        }
        sync_cube();

        // Second vertical blur + L1 sign + write.
        if pix_x < w && pix_y < h {
            let ly = UNIT_POS_Y + HALO;
            let lx = UNIT_POS_X;
            let mut s0 = F::cast_from(0.0_f32);
            let mut s1 = F::cast_from(0.0_f32);
            let mut s2 = F::cast_from(0.0_f32);
            #[unroll]
            for d in 1u32..6u32 {
                let w_d = gw::<F>(comptime![5u32 - d]);
                let bt = (((ly - d) * tile + lx) * 3u32) as usize;
                let bb = (((ly + d) * tile + lx) * 3u32) as usize;
                s0 += (buf_b[bt] + buf_b[bb]) * w_d;
                s1 += (buf_b[bt + 1] + buf_b[bb + 1]) * w_d;
                s2 += (buf_b[bt + 2] + buf_b[bb + 2]) * w_d;
            }
            let bc = ((ly * tile + lx) * 3u32) as usize;
            let wc = gw::<F>(5u32);
            s0 += buf_b[bc] * wc;
            s1 += buf_b[bc + 1] * wc;
            s2 += buf_b[bc + 2] * wc;

            let pix_idx = ((pix_y * w + pix_x) * cn + c) as usize;
            let p1 = pred[pix_idx];
            let (gt_c, gt_a) = read_gt::<F>(gt_packed, c, pix_y, pix_x, false, w);
            let gt_eff = if composite {
                gt_c + (F::cast_from(1.0_f32) - gt_a) * bg_c
            } else {
                gt_c
            };
            let ssim_grad = s0 + (F::cast_from(2.0_f32) * p1) * s1 + gt_eff * s2;
            let diff = p1 - gt_eff;
            let zero = F::cast_from(0.0_f32);
            let l1_sign = if diff > zero {
                F::cast_from(1.0_f32)
            } else if diff < zero {
                F::cast_from(-1.0_f32)
            } else {
                zero
            };
            let mut chain_centre = chain_at::<F>(
                dl_dpartials,
                c,
                pix_y,
                pix_x,
                h,
                w,
                F::cast_from(rgb_weight),
            );
            if mask {
                chain_centre = chain_centre * gt_a;
            }
            dl_dpred[pix_idx] = F::cast_from(ssim_weight) * ssim_grad
                + F::cast_from(l1_weight) * l1_sign * chain_centre;
        }
    }

    /// Decode `gt_packed` to `[H, W, 3]` f32 RGB. Comptime `composite` gates
    /// the `gt + (1 - gt.a) * bg` math; callers pass false when the source
    /// has no real alpha or when `bg == 0`. Used by the LPIPS path.
    #[cube(launch)]
    pub fn unpack_gt_rgb_kernel<F: Float>(
        gt_packed: &Tensor<u32>,
        out: &mut Tensor<F>,
        h: u32,
        w: u32,
        bg_r: f32,
        bg_g: f32,
        bg_b: f32,
        #[comptime] composite: bool,
    ) {
        let pix_y = CUBE_POS_Y * BLOCK_Y + UNIT_POS_Y;
        let pix_x = CUBE_POS_X * BLOCK_X + UNIT_POS_X;
        if pix_x >= w || pix_y >= h {
            terminate!();
        }
        let val = gt_packed[(pix_y * w + pix_x) as usize];
        let mut r = f32::cast_from(val & 0xffu32) * INV_255;
        let mut g = f32::cast_from((val >> 8u32) & 0xffu32) * INV_255;
        let mut b = f32::cast_from((val >> 16u32) & 0xffu32) * INV_255;
        if composite {
            let inv_a = 1.0_f32 - f32::cast_from(val >> 24u32) * INV_255;
            r += inv_a * bg_r;
            g += inv_a * bg_g;
            b += inv_a * bg_b;
        }
        let base = ((pix_y * w + pix_x) * 3u32) as usize;
        out[base] = F::cast_from(r);
        out[base + 1] = F::cast_from(g);
        out[base + 2] = F::cast_from(b);
    }
}

/// Image-loss configuration.
///
/// `composite_bg = Some(bg)` folds `gt + (1 - gt.a) * bg` into the kernel
/// before comparing against `pred`. `None` skips the math entirely — set it
/// when GT has no real alpha (synthesised `a = 1` makes the term zero) or
/// when `bg == 0`, since the kernel pays for the always-on math otherwise.
#[derive(Debug, Clone, Copy)]
pub struct ImageLossConfig {
    pub l1_weight: f32,
    pub ssim_weight: f32,
    pub composite_bg: Option<Vec3>,
    /// If true, multiply each loss-map pixel by `gt.a`.
    pub mask: bool,
    /// Weight of the alpha-match term, `mean |pred.a - gt.a|`, added to the
    /// RGB mean when `pred` has 4 channels. Zero disables it: the alpha
    /// channel is then ignored and gets a zero gradient. The eval map's alpha
    /// channel holds the unweighted per-pixel term.
    pub alpha_weight: f32,
}

/// Side of the square tiles the loss kernel reduces; partial sums come back
/// one per tile.
pub const TILE_SIZE: usize = kernels::BLOCK_X as usize;

/// Number of tiles the loss kernel splits an `h × w` image into.
fn loss_tiles(h: usize, w: usize) -> usize {
    w.div_ceil(TILE_SIZE) * h.div_ceil(TILE_SIZE)
}

/// Whether the alpha-match term runs: it needs a weight and a 4-channel pred.
fn alpha_match(cfg: &ImageLossConfig, cn: u32) -> bool {
    cfg.alpha_weight > 0.0 && cn == 4
}

/// Channel planes the kernels launch: rgb, plus alpha when matching.
fn planes(cfg: &ImageLossConfig, cn: u32) -> u32 {
    if alpha_match(cfg, cn) { 4 } else { 3 }
}

/// Weights that turn the per-tile channel sums into the loss: rgb averaged
/// over `3·H·W`, alpha over `H·W` times its weight.
fn channel_weights(cfg: &ImageLossConfig, h: u32, w: u32) -> (f32, f32) {
    let pixels = (h * w) as f32;
    (1.0 / (3.0 * pixels), cfg.alpha_weight / pixels)
}

#[derive(Debug, Clone)]
struct ImageLossForwardSaved<B: Backend> {
    map: FloatTensor<B>,
    partials: FloatTensor<B>,
}

trait SavedLossOps: Backend {
    fn image_loss_forward_saved(
        pred: FloatTensor<Self>,
        gt_packed: IntTensor<Self>,
        cfg: ImageLossConfig,
    ) -> ImageLossForwardSaved<Self>;

    fn image_loss_backward_saved(
        pred: FloatTensor<Self>,
        gt_packed: IntTensor<Self>,
        dl_dmap: FloatTensor<Self>,
        partials: FloatTensor<Self>,
        cfg: ImageLossConfig,
    ) -> FloatTensor<Self>;
}

/// Backend hooks for the loss kernels. `pred` is `[H, W, C]` with 3 or 4
/// channels, the rasterizer's native layout. When alpha matching, the
/// `c == 3` workgroup runs the alpha-match path (`|pred.a - gt.a|`) instead
/// of SSIM + L1, folded into the same launch.
#[burn::backend::backend_extension(Cube, Autodiff, Fusion)]
pub trait LossOps: Backend {
    /// Forward loss. `reduce` picks the output: `false` writes the per-pixel
    /// map `[H, W, C]` (eval only), `true` the weighted per-tile partial sums
    /// `[planes, tiles]` (rows r, g, b, and alpha when matching) that add up
    /// to the loss, so the per-pixel map never touches memory. Only the
    /// reduced form carries a gradient.
    #[fusion(dtype = pred, shape = {
        if *reduce {
            Shape::new([
                planes(cfg, pred[2] as u32) as usize,
                loss_tiles(pred[0], pred[1]),
            ])
        } else {
            pred.clone()
        }
    })]
    fn image_loss_forward(
        pred: FloatTensor<Self>,
        gt_packed: IntTensor<Self>,
        cfg: ImageLossConfig,
        reduce: bool,
    ) -> FloatTensor<Self>;

    /// Gradient of the loss w.r.t. `pred` given the gradient w.r.t. the
    /// partial sums `[planes, tiles]`.
    #[fusion(dtype = pred, shape = pred)]
    fn image_loss_backward(
        pred: FloatTensor<Self>,
        gt_packed: IntTensor<Self>,
        dl_dpartials: FloatTensor<Self>,
        cfg: ImageLossConfig,
    ) -> FloatTensor<Self>;

    #[fusion(dtype = DType::F32, shape = Shape::new([gt_packed[0], gt_packed[1], 3]))]
    fn unpack_gt_rgb(gt_packed: IntTensor<Self>, composite_bg: Option<Vec3>) -> FloatTensor<Self>;
}

/// `[H, W, C]` dims of a pred tensor, checked against `gt_packed`.
fn image_dims(pred: &CubeTensor, gt_packed: &CubeTensor) -> (u32, u32, u32) {
    let dims = pred.shape().as_slice().to_vec();
    assert_eq!(dims.len(), 3, "image loss expects [H, W, C] pred");
    let (h, w, c) = (dims[0] as u32, dims[1] as u32, dims[2] as u32);
    assert!(
        c == 3 || c == 4,
        "image loss expects 3 or 4 channels, got {c}"
    );
    let gt_dims = gt_packed.shape().as_slice().to_vec();
    assert_eq!(gt_dims.len(), 2, "image loss expects [H, W] gt_packed");
    assert_eq!(gt_dims[0] as u32, h, "gt_packed height must match pred");
    assert_eq!(gt_dims[1] as u32, w, "gt_packed width must match pred");
    (h, w, c)
}

fn cube_count_3d(c: u32, h: u32, w: u32) -> burn::cubecl::prelude::CubeCount {
    use burn::cubecl::prelude::CubeCount;
    CubeCount::Static(
        w.div_ceil(kernels::BLOCK_X),
        h.div_ceil(kernels::BLOCK_Y),
        c,
    )
}

fn cube_count_3d_bwd(c: u32, h: u32, w: u32, tile: u32) -> burn::cubecl::prelude::CubeCount {
    use burn::cubecl::prelude::CubeCount;
    CubeCount::Static(w.div_ceil(tile), h.div_ceil(tile), c)
}

fn select_backward_tile(
    max_shared_memory_size: usize,
    max_units_per_cube: u32,
    max_cube_dim: (u32, u32, u32),
) -> u32 {
    if max_shared_memory_size >= kernels::BWD_LARGE_SHARED_BYTES
        && max_units_per_cube >= kernels::BWD_TILE_LARGE * kernels::BWD_TILE_LARGE
        && max_cube_dim.0 >= kernels::BWD_TILE_LARGE
        && max_cube_dim.1 >= kernels::BWD_TILE_LARGE
    {
        kernels::BWD_TILE_LARGE
    } else {
        kernels::BWD_TILE_SMALL
    }
}

/// Runs the forward kernel. `reduce = false` writes the `[H, W, C]` loss map
/// (alpha channel only when matching); `reduce = true` writes weighted
/// per-tile partial sums `[planes, tiles]` instead and leaves the map
/// untouched.
fn launch_image_forward(
    pred: CubeTensor,
    gt_packed: CubeTensor,
    cfg: ImageLossConfig,
    reduce: bool,
) -> CubeTensor {
    launch_image_forward_impl(pred, gt_packed, cfg, reduce, false).0
}

fn launch_image_forward_saved(
    pred: CubeTensor,
    gt_packed: CubeTensor,
    cfg: ImageLossConfig,
) -> (CubeTensor, CubeTensor) {
    let (out, saved) = launch_image_forward_impl(pred, gt_packed, cfg, true, true);
    (out, saved.expect("saved SSIM partials"))
}

fn launch_image_forward_impl(
    pred: CubeTensor,
    gt_packed: CubeTensor,
    cfg: ImageLossConfig,
    reduce: bool,
    save_partials: bool,
) -> (CubeTensor, Option<CubeTensor>) {
    use burn::cubecl::prelude::CubeDim;

    let pred = into_contiguous(pred);
    let gt_packed = into_contiguous(gt_packed);
    let (h, w, cn) = image_dims(&pred, &gt_packed);
    let planes = planes(&cfg, cn);
    let (rgb_weight, alpha_weight) = channel_weights(&cfg, h, w);

    let composite = cfg.composite_bg.is_some();
    let bg = cfg.composite_bg.unwrap_or(Vec3::ZERO);
    let client = pred.client.clone();
    let device = pred.device.clone();
    // The kernel only touches the output its mode selects; the other is a
    // placeholder. Every partial is written by its own cube, so no fill.
    let (map, partials) = if reduce {
        let tiles = loss_tiles(h as usize, w as usize);
        (
            create_tensor([1], &device, DType::F32),
            create_tensor([planes as usize, tiles], &device, DType::F32),
        )
    } else {
        (
            burn_cubecl::ops::numeric::zeros_client(
                client.clone(),
                device.clone(),
                Shape::new([h as usize, w as usize, cn as usize]),
                DType::F32,
            ),
            create_tensor([1], &device, DType::F32),
        )
    };
    let saved =
        save_partials.then(|| create_tensor([9, h as usize, w as usize], &device, DType::F32));
    kernels::image_loss_forward_kernel::launch::<f32>(
        &client,
        cube_count_3d(planes, h, w),
        CubeDim::new_2d(kernels::BLOCK_X, kernels::BLOCK_Y),
        pred.into_tensor_arg(),
        gt_packed.into_tensor_arg(),
        map.clone().into_tensor_arg(),
        partials.clone().into_tensor_arg(),
        saved.clone().map(|t| t.into_tensor_arg()).into(),
        h,
        w,
        cn,
        cfg.l1_weight,
        cfg.ssim_weight,
        rgb_weight,
        alpha_weight,
        bg.x,
        bg.y,
        bg.z,
        composite,
        cfg.mask,
        reduce,
        cfg.ssim_weight == 0.0 && !save_partials,
    );
    (if reduce { partials } else { map }, saved)
}

fn launch_image_backward(
    pred: CubeTensor,
    gt_packed: CubeTensor,
    chain: CubeTensor,
    cfg: ImageLossConfig,
) -> CubeTensor {
    launch_image_backward_with_tile(pred, gt_packed, chain, cfg, None)
}

fn launch_image_backward_with_tile(
    pred: CubeTensor,
    gt_packed: CubeTensor,
    chain: CubeTensor,
    cfg: ImageLossConfig,
    tile: Option<u32>,
) -> CubeTensor {
    launch_image_backward_impl(pred, gt_packed, chain, cfg, tile, None)
}

fn launch_image_backward_saved(
    pred: CubeTensor,
    gt_packed: CubeTensor,
    chain: CubeTensor,
    saved: CubeTensor,
    cfg: ImageLossConfig,
) -> CubeTensor {
    launch_image_backward_impl(pred, gt_packed, chain, cfg, None, Some(saved))
}

fn launch_image_backward_impl(
    pred: CubeTensor,
    gt_packed: CubeTensor,
    dl_dpartials: CubeTensor,
    cfg: ImageLossConfig,
    tile_override: Option<u32>,
    saved: Option<CubeTensor>,
) -> CubeTensor {
    use burn::cubecl::prelude::CubeDim;

    let pred = into_contiguous(pred);
    let gt_packed = into_contiguous(gt_packed);
    let dl_dpartials = into_contiguous(dl_dpartials);
    let (h, w, cn) = image_dims(&pred, &gt_packed);
    let alpha_match = alpha_match(&cfg, cn);
    let (rgb_weight, alpha_weight) = channel_weights(&cfg, h, w);
    assert_eq!(
        dl_dpartials.shape().as_slice(),
        &[
            planes(&cfg, cn) as usize,
            loss_tiles(h as usize, w as usize)
        ],
        "dl_dpartials must be [planes, tiles]"
    );

    let composite = cfg.composite_bg.is_some();
    let bg = cfg.composite_bg.unwrap_or(Vec3::ZERO);
    // Every pixel of every channel is written (the alpha channel with zeros
    // when not matching), so the output needs no fill.
    let dl_dpred = create_tensor(
        [h as usize, w as usize, cn as usize],
        &pred.device,
        DType::F32,
    );
    let client = pred.client.clone();

    let hardware = &client.properties().hardware;
    let tile = tile_override.unwrap_or_else(|| {
        select_backward_tile(
            hardware.max_shared_memory_size,
            hardware.max_units_per_cube,
            hardware.max_cube_dim,
        )
    });
    kernels::image_loss_backward_kernel::launch::<f32>(
        &client,
        cube_count_3d_bwd(cn, h, w, tile),
        CubeDim::new_2d(tile, tile),
        pred.into_tensor_arg(),
        gt_packed.into_tensor_arg(),
        dl_dpartials.into_tensor_arg(),
        saved.map(|t| into_contiguous(t).into_tensor_arg()).into(),
        dl_dpred.clone().into_tensor_arg(),
        h,
        w,
        cn,
        cfg.l1_weight,
        cfg.ssim_weight,
        rgb_weight,
        alpha_weight,
        bg.x,
        bg.y,
        bg.z,
        composite,
        cfg.mask,
        tile,
        alpha_match,
        cfg.ssim_weight == 0.0,
    );
    dl_dpred
}

fn launch_unpack_gt_rgb(gt_packed: CubeTensor, composite_bg: Option<Vec3>) -> CubeTensor {
    use burn::cubecl::prelude::{CubeCount, CubeDim};

    let gt_packed = into_contiguous(gt_packed);
    let dims = gt_packed.shape().as_slice().to_vec();
    assert_eq!(dims.len(), 2, "unpack_gt_rgb expects [H, W] gt_packed");
    let (h, w) = (dims[0] as u32, dims[1] as u32);
    let composite = composite_bg.is_some();
    let bg = composite_bg.unwrap_or(Vec3::ZERO);

    let client = gt_packed.client.clone();
    let out = burn_cubecl::ops::numeric::zeros_client(
        client.clone(),
        gt_packed.device.clone(),
        Shape::new([h as usize, w as usize, 3]),
        DType::F32,
    );
    let cube_count = CubeCount::Static(
        w.div_ceil(kernels::BLOCK_X),
        h.div_ceil(kernels::BLOCK_Y),
        1,
    );
    kernels::unpack_gt_rgb_kernel::launch::<f32>(
        &client,
        cube_count,
        CubeDim::new_2d(kernels::BLOCK_X, kernels::BLOCK_Y),
        gt_packed.into_tensor_arg(),
        out.clone().into_tensor_arg(),
        h,
        w,
        bg.x,
        bg.y,
        bg.z,
        composite,
    );
    out
}

impl LossOps for CubeBackend {
    fn image_loss_forward(
        pred: FloatTensor<Self>,
        gt_packed: IntTensor<Self>,
        cfg: ImageLossConfig,
        reduce: bool,
    ) -> FloatTensor<Self> {
        launch_image_forward(pred, gt_packed, cfg, reduce)
    }

    fn image_loss_backward(
        pred: FloatTensor<Self>,
        gt_packed: IntTensor<Self>,
        dl_dpartials: FloatTensor<Self>,
        cfg: ImageLossConfig,
    ) -> FloatTensor<Self> {
        launch_image_backward(pred, gt_packed, dl_dpartials, cfg)
    }

    fn unpack_gt_rgb(gt_packed: IntTensor<Self>, composite_bg: Option<Vec3>) -> FloatTensor<Self> {
        launch_unpack_gt_rgb(gt_packed, composite_bg)
    }
}

impl SavedLossOps for CubeBackend {
    fn image_loss_forward_saved(
        pred: FloatTensor<Self>,
        gt_packed: IntTensor<Self>,
        cfg: ImageLossConfig,
    ) -> ImageLossForwardSaved<Self> {
        let (map, partials) = launch_image_forward_saved(pred, gt_packed, cfg);
        ImageLossForwardSaved { map, partials }
    }

    fn image_loss_backward_saved(
        pred: FloatTensor<Self>,
        gt_packed: IntTensor<Self>,
        dl_dmap: FloatTensor<Self>,
        partials: FloatTensor<Self>,
        cfg: ImageLossConfig,
    ) -> FloatTensor<Self> {
        launch_image_backward_saved(pred, gt_packed, dl_dmap, partials, cfg)
    }
}

impl SavedLossOps for Fusion<CubeBackend> {
    fn image_loss_forward_saved(
        pred: FloatTensor<Self>,
        gt_packed: IntTensor<Self>,
        cfg: ImageLossConfig,
    ) -> ImageLossForwardSaved<Self> {
        let [h, w, c] = pred.shape().dims();
        let client = pred.client.clone();
        let [map, partials] = register_custom(
            &client,
            "image_loss_forward_saved",
            [pred, gt_packed],
            [
                (
                    Shape::new([planes(&cfg, c as u32) as usize, loss_tiles(h, w)]),
                    DType::F32,
                ),
                (Shape::new([9, h, w]), DType::F32),
            ],
            move |desc, h| {
                let ([pred, gt], [map, partials]) = desc.as_fixed();
                let out = <CubeBackend as SavedLossOps>::image_loss_forward_saved(
                    h.get_float_tensor::<CubeBackend>(pred),
                    h.get_int_tensor::<CubeBackend>(gt),
                    cfg,
                );
                h.register_float_tensor::<CubeBackend>(&map.id, out.map);
                h.register_float_tensor::<CubeBackend>(&partials.id, out.partials);
            },
        );
        ImageLossForwardSaved { map, partials }
    }
    fn image_loss_backward_saved(
        pred: FloatTensor<Self>,
        gt_packed: IntTensor<Self>,
        chain: FloatTensor<Self>,
        partials: FloatTensor<Self>,
        cfg: ImageLossConfig,
    ) -> FloatTensor<Self> {
        let shape = pred.shape();
        let client = pred.client.clone();
        let [out] = register_custom(
            &client,
            "image_loss_backward_saved",
            [pred, gt_packed, chain, partials],
            [(shape, DType::F32)],
            move |desc, h| {
                let ([pred, gt, chain, partials], [out]) = desc.as_fixed();
                let result = <CubeBackend as SavedLossOps>::image_loss_backward_saved(
                    h.get_float_tensor::<CubeBackend>(pred),
                    h.get_int_tensor::<CubeBackend>(gt),
                    h.get_float_tensor::<CubeBackend>(chain),
                    h.get_float_tensor::<CubeBackend>(partials),
                    cfg,
                );
                h.register_float_tensor::<CubeBackend>(&out.id, result);
            },
        );
        out
    }
}

#[derive(Debug)]
struct ImageLossBackward;

#[derive(Debug, Clone)]
struct ImageLossState<B: Backend> {
    saved_partials: Option<FloatTensor<B>>,
    pred: FloatTensor<B>,
    gt_packed: IntTensor<B>,
    cfg: ImageLossConfig,
}

impl<B: Backend + LossOps + SavedLossOps> Backward<B, 1> for ImageLossBackward {
    type State = ImageLossState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 1>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let state = ops.state;
        let dl_dpartials = grads.consume::<B>(&ops.node);
        let [pred_parent] = ops.parents;
        let dl_dpred = if let Some(saved) = state.saved_partials {
            B::image_loss_backward_saved(
                state.pred,
                state.gt_packed,
                dl_dpartials,
                saved,
                state.cfg,
            )
        } else {
            B::image_loss_backward(state.pred, state.gt_packed, dl_dpartials, state.cfg)
        };
        if let Some(node) = pred_parent {
            grads.register::<B>(node.id, dl_dpred);
        }
    }
}

/// L1 + SSIM image loss with optional bg-compositing, masking and alpha
/// matching, folded into a single kernel that also reduces each tile. `pred`
/// is `[H, W, C]` with 3 or 4 channels, on an autodiff-enabled device.
pub fn image_loss(pred: Tensor<3>, gt_packed: Tensor<2, Int>, cfg: ImageLossConfig) -> Tensor<1> {
    image_loss_partials(pred, gt_packed, cfg).sum()
}

/// The pieces of [`image_loss`] before the final sum: weighted per-tile
/// sums `[planes, tiles]` (rows r, g, b, and alpha when matching), tiles in
/// row-major order with [`TILE_SIZE`] pixels a side. For tests that want to
/// probe part of the image.
pub fn image_loss_partials(
    pred: Tensor<3>,
    gt_packed: Tensor<2, Int>,
    cfg: ImageLossConfig,
) -> Tensor<2> {
    let partials = <burn::backend::Dispatch as LossOps>::image_loss_forward(
        pred.into_dispatch(),
        gt_packed.into_dispatch(),
        cfg,
        true,
    );
    Tensor::<2>::from_dispatch(partials)
}

impl<B: Backend + LossOps + SavedLossOps, C: CheckpointStrategy> LossOps for Autodiff<B, C> {
    fn image_loss_forward(
        pred: FloatTensor<Self>,
        gt_packed: IntTensor<Self>,
        cfg: ImageLossConfig,
        reduce: bool,
    ) -> FloatTensor<Self> {
        if !reduce {
            // The per-pixel map is eval-only; gradients go through the
            // reduced form.
            return <Self as AutodiffBackend>::from_inner(<B as LossOps>::image_loss_forward(
                pred.into_primitive(),
                gt_packed,
                cfg,
                false,
            ));
        }

        let prep = ImageLossBackward
            .prepare::<NoCheckpointing>([pred.node()])
            .compute_bound()
            .stateful();

        let pred_p = pred.into_primitive();
        let (partials, saved_partials) = if use_saved_loss_partials() && cfg.ssim_weight != 0.0 {
            let out = B::image_loss_forward_saved(pred_p.clone(), gt_packed.clone(), cfg);
            (out.map, Some(out.partials))
        } else {
            (
                B::image_loss_forward(pred_p.clone(), gt_packed.clone(), cfg, true),
                None,
            )
        };

        match prep {
            OpsKind::Tracked(prep) => prep.finish(
                ImageLossState {
                    saved_partials,
                    pred: pred_p,
                    gt_packed,
                    cfg,
                },
                partials,
            ),
            OpsKind::UnTracked(prep) => prep.finish(partials),
        }
    }

    fn image_loss_backward(
        pred: FloatTensor<Self>,
        gt_packed: IntTensor<Self>,
        dl_dpartials: FloatTensor<Self>,
        cfg: ImageLossConfig,
    ) -> FloatTensor<Self> {
        <Self as AutodiffBackend>::from_inner(<B as LossOps>::image_loss_backward(
            pred.into_primitive(),
            gt_packed,
            dl_dpartials.into_primitive(),
            cfg,
        ))
    }

    fn unpack_gt_rgb(gt_packed: IntTensor<Self>, composite_bg: Option<Vec3>) -> FloatTensor<Self> {
        <Self as AutodiffBackend>::from_inner(<B as LossOps>::unpack_gt_rgb(
            gt_packed,
            composite_bg,
        ))
    }
}

/// Forward-only loss map for non-differentiable backends. Same kernel as
/// the training forward; eval picks `cfg` to compute SSIM, L1, or whatever
/// combination it needs (e.g. MSE = `l1_eval(...).powi(2).mean()`).
pub fn image_loss_eval(
    pred: Tensor<3>,
    gt_packed: Tensor<2, Int>,
    cfg: ImageLossConfig,
) -> Tensor<3> {
    let map = <burn::backend::Dispatch as LossOps>::image_loss_forward(
        pred.into_dispatch(),
        gt_packed.into_dispatch(),
        cfg,
        false,
    );
    Tensor::<3>::from_dispatch(map)
}

/// Smallest MSE PSNR distinguishes: identical images report 100 dB instead
/// of infinity, which would poison any average or plot it feeds.
const PSNR_MIN_MSE: f32 = 1e-10;

/// PSNR in dB from a mean squared error between images in `[0, 1]`.
pub fn psnr_from_mse(mse: Tensor<1>) -> Tensor<1> {
    mse.clamp_min(PSNR_MIN_MSE).recip().log() * (10.0 / std::f32::consts::LN_10)
}

/// PSNR in dB between two `[H, W, 3]` images in `[0, 1]`.
pub fn psnr(a: Tensor<3>, b: Tensor<3>) -> Tensor<1> {
    psnr_from_mse((a - b).powi_scalar(2).mean())
}

/// Decode `gt_packed` back to a `[H, W, 3]` f32 RGB tensor. `composite_bg =
/// Some(bg)` folds in `gt + (1 - gt.a) * bg`; `None` skips that math.
/// Materialising f32 GT defeats the whole point of the packed format, so
/// this is reserved for the LPIPS path which feeds f32 RGB into a VGG
/// forward and has no kernel-fused alternative today.
pub fn unpack_gt_rgb(gt_packed: Tensor<2, Int>, composite_bg: Option<Vec3>) -> Tensor<3> {
    let out = <burn::backend::Dispatch as LossOps>::unpack_gt_rgb(
        gt_packed.into_dispatch(),
        composite_bg,
    );
    Tensor::from_dispatch(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backward_tile_selection_respects_device_limits() {
        let large = kernels::BWD_TILE_LARGE;
        let generous_dims = (large, large, 1);
        let generous_units = large * large;

        assert_eq!(kernels::BWD_LARGE_SHARED_BYTES, 29_088);
        assert_eq!(
            select_backward_tile(29_087, generous_units, generous_dims),
            kernels::BWD_TILE_SMALL
        );
        assert_eq!(
            select_backward_tile(29_088, generous_units, generous_dims),
            large
        );
        assert_eq!(
            select_backward_tile(29_088, generous_units - 1, generous_dims),
            kernels::BWD_TILE_SMALL
        );
        assert_eq!(
            select_backward_tile(29_088, generous_units, (large - 1, large, 1)),
            kernels::BWD_TILE_SMALL
        );
        assert_eq!(
            select_backward_tile(29_088, generous_units, (large, large - 1, 1)),
            kernels::BWD_TILE_SMALL
        );
    }

    #[cfg(not(target_family = "wasm"))]
    #[tokio::test]
    async fn backward_tile_specializations_match() {
        use brush_cube::{CubeDevice, CubeTensor, create_tensor_from_slice};
        use burn::tensor::Shape;

        fn shaped_f32(data: &[f32], shape: Shape, device: &CubeDevice) -> CubeTensor {
            let flat = create_tensor_from_slice(data, device, DType::F32);
            CubeTensor::new_contiguous(flat.client, flat.device, shape, flat.handle, flat.dtype)
        }

        fn shaped_i32(data: &[i32], shape: Shape, device: &CubeDevice) -> CubeTensor {
            let flat = create_tensor_from_slice(data, device, DType::I32);
            CubeTensor::new_contiguous(flat.client, flat.device, shape, flat.handle, flat.dtype)
        }

        let device = CubeDevice::Wgpu(brush_cube::test_helpers::test_device().await);
        let (c, h, w) = (4usize, 17usize, 19usize);
        let pred: Vec<f32> = (0..c * h * w)
            .map(|i| 0.1 + ((i * 17 + 3) % 71) as f32 / 100.0)
            .collect();
        let chain: Vec<f32> = (0..c * loss_tiles(h, w))
            .map(|i| {
                let value = 0.2 + ((i * 13 + 5) % 37) as f32 / 50.0;
                if i % 2 == 0 { value } else { -value }
            })
            .collect();
        let gt: Vec<i32> = (0..h * w)
            .map(|i| {
                let r = (30 + (i * 7) % 101) as u32;
                let g = (70 + (i * 11) % 101) as u32;
                let b = (110 + (i * 13) % 101) as u32;
                let a = (100 + (i * 17) % 131) as u32;
                (r | g << 8 | b << 16 | a << 24) as i32
            })
            .collect();
        let cfg = ImageLossConfig {
            l1_weight: 0.8,
            ssim_weight: -0.2,
            composite_bg: Some(Vec3::new(0.05, 0.1, 0.15)),
            mask: true,
            alpha_weight: 0.3,
        };

        let make_pred = || shaped_f32(&pred, Shape::new([h, w, c]), &device);
        let make_gt = || shaped_i32(&gt, Shape::new([h, w]), &device);
        let make_chain = || shaped_f32(&chain, Shape::new([c, loss_tiles(h, w)]), &device);
        let small_pred = make_pred();
        let selected_tile = {
            let hardware = &small_pred.client.properties().hardware;
            select_backward_tile(
                hardware.max_shared_memory_size,
                hardware.max_units_per_cube,
                hardware.max_cube_dim,
            )
        };
        let small = launch_image_backward_with_tile(
            small_pred,
            make_gt(),
            make_chain(),
            cfg,
            Some(kernels::BWD_TILE_SMALL),
        );
        let small: Vec<f32> = burn_cubecl::ops::into_data_sync(small)
            .try_to_vec()
            .expect("small-tile gradient data");
        assert!(
            small.iter().all(|value| value.is_finite()),
            "small-tile gradients must be finite"
        );
        if selected_tile != kernels::BWD_TILE_LARGE {
            return;
        }

        let large = launch_image_backward_with_tile(
            make_pred(),
            make_gt(),
            make_chain(),
            cfg,
            Some(kernels::BWD_TILE_LARGE),
        );
        let large: Vec<f32> = burn_cubecl::ops::into_data_sync(large)
            .try_to_vec()
            .expect("large-tile gradient data");

        for (index, (&small, &large)) in small.iter().zip(&large).enumerate() {
            let tolerance = 5e-5 + 5e-5 * small.abs().max(large.abs());
            assert!(
                (small - large).abs() <= tolerance,
                "tile gradients differ at {index}: 8x8={small}, 16x16={large}, tolerance={tolerance}"
            );
        }
    }

    #[cfg(not(target_family = "wasm"))]
    #[tokio::test]
    async fn saved_partials_match_recomputed_forward_and_vjp() {
        use brush_cube::{CubeDevice, CubeTensor, create_tensor_from_slice};
        use burn::tensor::Shape;

        fn shaped_f32(data: &[f32], shape: Shape, device: &CubeDevice) -> CubeTensor {
            let flat = create_tensor_from_slice(data, device, DType::F32);
            CubeTensor::new_contiguous(flat.client, flat.device, shape, flat.handle, flat.dtype)
        }

        fn shaped_i32(data: &[i32], shape: Shape, device: &CubeDevice) -> CubeTensor {
            let flat = create_tensor_from_slice(data, device, DType::I32);
            CubeTensor::new_contiguous(flat.client, flat.device, shape, flat.handle, flat.dtype)
        }

        fn assert_close(label: &str, expected: &[f32], actual: &[f32]) {
            assert_eq!(expected.len(), actual.len(), "{label} length");
            for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
                let tolerance = 5e-5 + 5e-5 * expected.abs().max(actual.abs());
                assert!(
                    (expected - actual).abs() <= tolerance,
                    "{label} differs at {index}: expected={expected}, actual={actual}, tolerance={tolerance}"
                );
            }
        }

        let device = CubeDevice::Wgpu(brush_cube::test_helpers::test_device().await);
        let (c, h, w) = (4usize, 17usize, 19usize);
        let pred_data: Vec<f32> = (0..c * h * w)
            .map(|i| 0.05 + ((i * 17 + 3) % 83) as f32 / 100.0)
            .collect();
        let chain_data: Vec<f32> = (0..c * loss_tiles(h, w))
            .map(|i| {
                let value = 0.15 + ((i * 13 + 5) % 41) as f32 / 50.0;
                if i % 2 == 0 { value } else { -value }
            })
            .collect();
        let gt_data: Vec<i32> = (0..h * w)
            .map(|i| {
                let r = (20 + (i * 7) % 151) as u32;
                let g = (50 + (i * 11) % 151) as u32;
                let b = (90 + (i * 13) % 151) as u32;
                let alpha_values = [0u32, 1, 127, 254, 255];
                let a = alpha_values[i % alpha_values.len()];
                (r | g << 8 | b << 16 | a << 24) as i32
            })
            .collect();
        let cfg = ImageLossConfig {
            l1_weight: 0.8,
            ssim_weight: -0.2,
            composite_bg: Some(Vec3::new(0.05, 0.1, 0.15)),
            mask: true,
            alpha_weight: 0.3,
        };
        let make_pred = || shaped_f32(&pred_data, Shape::new([h, w, c]), &device);
        let make_gt = || shaped_i32(&gt_data, Shape::new([h, w]), &device);
        let make_chain = || shaped_f32(&chain_data, Shape::new([c, loss_tiles(h, w)]), &device);

        let control_map = launch_image_forward(make_pred(), make_gt(), cfg, true);
        let (saved_map, partials) = launch_image_forward_saved(make_pred(), make_gt(), cfg);
        let control_grad = launch_image_backward(make_pred(), make_gt(), make_chain(), cfg);
        let saved_grad =
            launch_image_backward_saved(make_pred(), make_gt(), make_chain(), partials, cfg);

        let control_map: Vec<f32> = burn_cubecl::ops::into_data_sync(control_map)
            .try_to_vec()
            .expect("control map data");
        let saved_map: Vec<f32> = burn_cubecl::ops::into_data_sync(saved_map)
            .try_to_vec()
            .expect("saved map data");
        let control_grad: Vec<f32> = burn_cubecl::ops::into_data_sync(control_grad)
            .try_to_vec()
            .expect("control gradient data");
        let saved_grad: Vec<f32> = burn_cubecl::ops::into_data_sync(saved_grad)
            .try_to_vec()
            .expect("saved gradient data");

        assert_close("forward map", &control_map, &saved_map);
        assert_close("prediction VJP", &control_grad, &saved_grad);
        let control_alpha: Vec<_> = control_grad.iter().skip(3).step_by(4).collect();
        let saved_alpha: Vec<_> = saved_grad.iter().skip(3).step_by(4).collect();
        assert_eq!(
            control_alpha, saved_alpha,
            "alpha VJP must remain bit-identical"
        );
    }
}
