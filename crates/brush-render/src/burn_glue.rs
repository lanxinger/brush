#![allow(clippy::match_wildcard_for_single_variants)]

use brush_cube::fusion::register_custom;
use brush_cube::{MainBackend, MainBackendBase};
use burn::backend::{
    Autodiff, BackendTensor, DispatchAutodiffContext, DispatchTensor, DispatchTensorKind,
    GradientCheckpointingStrategy, TensorMetadata,
    tensor::{FloatTensor, IntTensor},
};
use burn::tensor::{DType, Int, Tensor};
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
use burn_fusion::Fusion;
use glam::Vec3;

use crate::{
    RenderAuxInner, SplatOps, SplatRasterizerOps, backend_kind,
    camera::Camera,
    gaussian_splats::{Rasterizer, SplatRenderMode},
    render_aux::RenderOutput,
};

/// Inner Cube autodiff backend.
/// Used as the primitive backend for autodiff `Tensor<D>` operations.
pub type AutodiffMain = Autodiff<MainBackend>;

// ---------------------------------------------------------------------------
// `Tensor<D>` ↔ backend-level primitive bridges.
//
// `Tensor<D>` is pinned to burn's `Dispatch` backend; brush only ever runs on
// a Cube device, so every helper here assumes a `DispatchTensorKind::Cube`
// (optionally wrapped in `Autodiff`) and panics otherwise. The forward render
// now goes through the `#[backend_extension]`-generated `Dispatch` impl
// instead; these stay for the hand-rolled backward path in `crate::bwd` and
// the LPIPS custom ops (brush-loss).
// ---------------------------------------------------------------------------

/// Extract the inner fusion-Wgpu float tensor from a non-autodiff
/// `Tensor<D>`.
pub fn unwrap_wgpu_float<const D: usize>(t: Tensor<D>) -> FloatTensor<MainBackend> {
    let dispatch: DispatchTensor = t.into_dispatch();
    match dispatch.kind {
        backend_kind!(bt) => bt.float(),
        other => panic!(
            "expected Wgpu tensor, got: {:?}",
            std::mem::discriminant(&other)
        ),
    }
}

/// Extract the inner fusion-Wgpu int tensor from a non-autodiff
/// `Tensor<D, Int>`.
pub fn unwrap_wgpu_int<const D: usize>(t: Tensor<D, Int>) -> IntTensor<MainBackend> {
    let dispatch: DispatchTensor = t.into_dispatch();
    match dispatch.kind {
        backend_kind!(bt) => bt.int(),
        other => panic!(
            "expected Wgpu int tensor, got: {:?}",
            std::mem::discriminant(&other)
        ),
    }
}

/// Inverse of [`unwrap_wgpu_float`]: wraps a fusion-Wgpu float tensor as a
/// user-facing `Tensor<D>`.
pub fn wrap_wgpu_float<const D: usize>(t: FloatTensor<MainBackend>) -> Tensor<D> {
    Tensor::from_dispatch(DispatchTensor {
        kind: backend_kind!(BackendTensor::Float(t)),
        autodiff: DispatchAutodiffContext::Disabled,
    })
}

/// Like [`wrap_wgpu_float`] for an int tensor.
pub fn wrap_wgpu_int<const D: usize>(t: IntTensor<MainBackend>) -> Tensor<D, Int> {
    Tensor::from_dispatch(DispatchTensor {
        kind: backend_kind!(BackendTensor::Int(t)),
        autodiff: DispatchAutodiffContext::Disabled,
    })
}

/// Extract the inner `AutodiffTensor<MainBackend>` from a `Tensor<D>` on an
/// autodiff-enabled Wgpu device. Panics on any other shape.
pub fn unwrap_ad_wgpu_float<const D: usize>(t: Tensor<D>) -> FloatTensor<AutodiffMain> {
    let prim: DispatchTensor = t.into_dispatch();
    match prim.kind {
        DispatchTensorKind::Autodiff(inner) => match *inner {
            backend_kind!(BackendTensor::Autodiff(t)) => t,
            other => panic!(
                "autodiff inner kind is not Wgpu: {:?}",
                std::mem::discriminant(&other)
            ),
        },
        other => panic!(
            "expected autodiff-enabled tensor; got: {:?}",
            std::mem::discriminant(&other)
        ),
    }
}

/// Extract the inner Wgpu `IntTensor` regardless of whether the tensor is
/// wrapped in an autodiff device — ints are never autodiff-tracked.
pub fn unwrap_ad_wgpu_int<const D: usize>(t: Tensor<D, Int>) -> IntTensor<MainBackend> {
    let dispatch: DispatchTensor = t.into_dispatch();
    let kind = match dispatch.kind {
        DispatchTensorKind::Autodiff(inner) => *inner,
        other => other,
    };
    match kind {
        backend_kind!(bt) => bt.int(),
        other => panic!(
            "expected Wgpu int tensor; got: {:?}",
            std::mem::discriminant(&other)
        ),
    }
}

/// Inverse of [`unwrap_ad_wgpu_float`]: wraps an autodiff tensor as a
/// user-facing `Tensor<D>` on the autodiff device.
pub fn wrap_ad_wgpu_float<const D: usize>(t: FloatTensor<AutodiffMain>) -> Tensor<D> {
    Tensor::from_dispatch(DispatchTensor {
        kind: DispatchTensorKind::Autodiff(Box::new(backend_kind!(BackendTensor::Autodiff(t)))),
        autodiff: DispatchAutodiffContext::Enabled(GradientCheckpointingStrategy::Disabled),
    })
}

/// Remove autodiff association while preserving the tensor's device.
/// Already-inner tensors are returned unchanged.
pub fn detach_autodiff<const D: usize>(t: Tensor<D>) -> Tensor<D> {
    t.without_autodiff()
}

/// Associate a tensor with autodiff without requiring its gradient.
#[doc(hidden)]
pub fn lift_to_autodiff<const D: usize>(t: Tensor<D>) -> Tensor<D> {
    t.autodiff()
}

/// Compatibility alias for [`detach_autodiff`].
pub fn strip_autodiff_float<const D: usize>(t: Tensor<D>) -> Tensor<D> {
    t.without_autodiff()
}

fn is_autodiff<const D: usize>(t: &Tensor<D>) -> bool {
    matches!(
        t.clone().into_dispatch().kind,
        DispatchTensorKind::Autodiff(_)
    )
}

/// Put `t` on the same autodiff/inner backend variant as `reference`. Brush
/// keeps some frozen tensors (e.g. the 3D-filter floor) on the inner backend
/// but folds them against params that may be lifted to autodiff; this aligns
/// both operands so dispatch ops don't trip a cross-backend assertion.
pub(crate) fn match_backend<const D: usize, const DR: usize>(
    t: Tensor<D>,
    reference: &Tensor<DR>,
) -> Tensor<D> {
    if is_autodiff(reference) {
        lift_to_autodiff(t)
    } else {
        detach_autodiff(t)
    }
}

/// Like [`detach_autodiff`] for `Tensor<D, Int>`.
pub fn detach_autodiff_int<const D: usize>(t: Tensor<D, Int>) -> Tensor<D, Int> {
    t.without_autodiff()
}

/// Resolve a `Tensor<D>` down to the underlying `CubeTensor`, draining any
/// pending fusion ops. Used for direct GPU resource access, e.g. binding the
/// buffer into a wgpu pipeline, so it stays tied to the main backend rather
/// than being generic over the runtime.
pub fn resolve_to_cube_float<const D: usize>(tensor: Tensor<D>) -> CubeTensor {
    let fusion = unwrap_wgpu_float(tensor);
    let client = fusion.client.clone();
    client.resolve_tensor_float::<MainBackendBase>(fusion)
}

impl SplatOps for Fusion<CubeBackend> {
    async fn render(
        camera: &Camera,
        img_size: glam::UVec2,
        transforms: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        min_scale: FloatTensor<Self>,
        has_min_scale: bool,
        _refine_weight: FloatTensor<Self>,
        render_mode: SplatRenderMode,
        background: Vec3,
        pass: crate::gaussian_splats::RasterPass,
    ) -> RenderOutput<Self> {
        <Self as SplatRasterizerOps>::render_with_rasterizer(
            camera,
            img_size,
            transforms,
            sh_coeffs,
            raw_opacities,
            min_scale,
            has_min_scale,
            render_mode,
            background,
            pass,
            Rasterizer::Legacy,
        )
        .await
    }
}

impl SplatRasterizerOps for Fusion<CubeBackend> {
    async fn render_with_rasterizer(
        camera: &Camera,
        img_size: glam::UVec2,
        transforms: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        min_scale: FloatTensor<Self>,
        has_min_scale: bool,
        render_mode: SplatRenderMode,
        background: Vec3,
        pass: crate::gaussian_splats::RasterPass,
        rasterizer: Rasterizer,
    ) -> RenderOutput<Self> {
        let client = transforms.client.clone();

        // Resolve fusion inputs to MainBackendBase tensors. This
        // drains any pending fusion operations into a concrete buffer.
        let base_transforms = client
            .clone()
            .resolve_tensor_float::<CubeBackend>(transforms);
        let base_sh_coeffs = client
            .clone()
            .resolve_tensor_float::<CubeBackend>(sh_coeffs);
        let base_raw_opac = client
            .clone()
            .resolve_tensor_float::<CubeBackend>(raw_opacities);

        let base_min_scale = client
            .clone()
            .resolve_tensor_float::<CubeBackend>(min_scale);

        // Run the full pipeline on the concrete cube backend.
        let out = <CubeBackend as SplatRasterizerOps>::render_with_rasterizer(
            camera,
            img_size,
            base_transforms,
            base_sh_coeffs,
            base_raw_opac,
            base_min_scale,
            has_min_scale,
            render_mode,
            background,
            pass,
            rasterizer,
        )
        .await;

        // The render is sized by a mid-pipeline readback, so it can't run as a
        // stream op itself; hand its finished outputs back to the stream as a
        // zero-input custom op that just binds them.
        let RenderOutput {
            out_img,
            aux,
            projected_splats,
            compact_gid_from_isect,
            project_uniforms,
            global_from_compact_gid,
            compact_from_global,
        } = out;
        let RenderAuxInner {
            num_visible,
            num_intersections,
            visible,
            max_radius,
            opacities,
            tile_offsets,
            img_size,
        } = aux;

        let [
            out_img,
            visible,
            max_radius,
            opacities,
            projected_splats,
            tile_offsets,
            compact_gid_from_isect,
            global_from_compact_gid,
            compact_from_global,
        ] = {
            // The float outputs first, then the int ones; `register_custom`
            // hands back one stream tensor per entry, in order.
            let floats = [out_img, visible, max_radius, opacities, projected_splats];
            let ints = [
                tile_offsets,
                compact_gid_from_isect,
                global_from_compact_gid,
                compact_from_global,
            ];
            let shapes = std::array::from_fn(|i| match floats.get(i) {
                Some(t) => (t.shape(), DType::F32),
                None => (ints[i - floats.len()].shape(), DType::U32),
            });
            register_custom(&client, "render_bind", [], shapes, move |desc, h| {
                let (_, outs) = desc.as_fixed::<0, 9>();
                for (out, t) in outs.iter().zip(&floats) {
                    h.register_float_tensor::<CubeBackend>(&out.id, t.clone());
                }
                for (out, t) in outs[floats.len()..].iter().zip(&ints) {
                    h.register_int_tensor::<CubeBackend>(&out.id, t.clone());
                }
            })
        };

        RenderOutput {
            out_img,
            aux: RenderAuxInner {
                num_visible,
                num_intersections,
                visible,
                max_radius,
                opacities,
                tile_offsets,
                img_size,
            },
            projected_splats,
            compact_gid_from_isect,
            project_uniforms,
            global_from_compact_gid,
            compact_from_global,
        }
    }
}
