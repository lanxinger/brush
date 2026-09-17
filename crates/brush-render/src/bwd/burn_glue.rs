#![allow(clippy::match_wildcard_for_single_variants)]

use crate::burn_glue::{
    AutodiffMain, lift_to_autodiff, unwrap_ad_wgpu_float, wrap_ad_wgpu_float, wrap_wgpu_float,
    wrap_wgpu_int,
};
use crate::{
    SplatOps, SplatRasterizerOps,
    camera::Camera,
    gaussian_splats::{Rasterizer, SplatRenderMode, Splats},
    sh::sh_coeffs_for_degree,
    shaders::helpers::ProjectUniforms,
};
use brush_cube::{MainBackend, fusion::register_custom};
use burn::backend::Autodiff;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::{
    backend::{
        AutodiffBackend, Backend, TensorMetadata,
        autodiff::{
            checkpoint::{base::Checkpointer, strategy::NoCheckpointing},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
        },
        tensor::{FloatTensor, IntTensor},
    },
    module::Param,
    tensor::{DType, Gradients as TensorGradients, Int, Shape, Tensor},
};
use burn_cubecl::CubeBackend;
use burn_fusion::Fusion;
use glam::Vec3;

fn training_rasterizer() -> Rasterizer {
    use std::sync::OnceLock;

    static SELECTED: OnceLock<Rasterizer> = OnceLock::new();
    *SELECTED.get_or_init(|| {
        if crate::native_msl::fine_raster_tiles_requested() {
            tracing::info!("16x8 training raster tiles enabled");
            Rasterizer::Candidate
        } else {
            Rasterizer::Legacy
        }
    })
}

/// Intermediate gradients from the rasterize backward pass.
///
/// Sparse buffer of shape `[num_visible, 10]`, indexed by `compact_gid`.
/// Slots 0..8 are projected splat gradients, slot 8 is the raw opacity
/// gradient, slot 9 is the refinement weight.
#[derive(Debug, Clone)]
pub(crate) struct RasterizeGrads<B: Backend> {
    pub v_combined: FloatTensor<B>,
}

/// Final gradients w.r.t. splat inputs from the project backward pass.
#[derive(Debug, Clone)]
pub(crate) struct SplatGrads<B: Backend> {
    pub v_transforms: FloatTensor<B>,
    pub v_coeffs: FloatTensor<B>,
    pub v_raw_opac: FloatTensor<B>,
    pub v_refine_weight: FloatTensor<B>,
}

/// Compact projection gradients when SH coefficient materialization is
/// deferred to the optimizer. Other gradients use the same lazy gather.
#[derive(Debug, Clone)]
pub struct DeferredSplatGrads<B: Backend> {
    pub v_transforms: FloatTensor<B>,
    pub v_raw_opac: FloatTensor<B>,
    pub v_refine_weight: FloatTensor<B>,
}

/// Concrete backward kernels behind [`SplatOps::render`].
///
/// Deliberately not `: SplatOps`. This is the set of backends that have
/// backward kernels, which is smaller than the set you can render on:
/// `AutodiffMain` and the generated `Dispatch` impl `SplatOps` but have no
/// meaningful `rasterize_bwd`, since these run on concrete tensors and are
/// called from the `Backward` impl on the inner backend.
pub(crate) trait SplatBwdOps: Backend {
    /// Backward pass for rasterization. Returns the sparse `v_combined`
    /// buffer, `[num_visible, 10]` indexed by `compact_gid`: slots 0..8 are
    /// projected splat gradients, slot 8 the raw opacity gradient, slot 9 the
    /// refinement weight.
    #[allow(clippy::too_many_arguments)]
    fn rasterize_bwd(
        out_img: FloatTensor<Self>,
        projected_splats: FloatTensor<Self>,
        compact_gid_from_isect: IntTensor<Self>,
        tile_offsets: IntTensor<Self>,
        background: Vec3,
        img_size: glam::UVec2,
        v_output: FloatTensor<Self>,
        smooth_cutoff: bool,
    ) -> RasterizeGrads<Self>;

    /// Specialized raster backward which may omit the refinement-only
    /// statistic. Backends that do not override this retain the compatible
    /// full-gradient behavior.
    #[allow(clippy::too_many_arguments)]
    fn rasterize_bwd_with_refine_weight(
        out_img: FloatTensor<Self>,
        projected_splats: FloatTensor<Self>,
        compact_gid_from_isect: IntTensor<Self>,
        tile_offsets: IntTensor<Self>,
        background: Vec3,
        img_size: glam::UVec2,
        v_output: FloatTensor<Self>,
        smooth_cutoff: bool,
        _compute_refine_weight: bool,
    ) -> RasterizeGrads<Self> {
        Self::rasterize_bwd(
            out_img,
            projected_splats,
            compact_gid_from_isect,
            tile_offsets,
            background,
            img_size,
            v_output,
            smooth_cutoff,
        )
    }

    /// Backward pass for projection.
    /// Reads sparse `v_combined` [`num_visible`, 10] and writes compact
    /// outputs: a zero row, then one row per visible splat in `compact_gid`
    /// order. The caller gathers them per global splat through
    /// `compact_from_global`.
    /// `sh_coeffs` is the original (input) SH coefficient tensor — needed
    /// so the kernel can backprop `v_color` through the SH basis to the
    /// view direction and then to the mean.
    #[allow(clippy::too_many_arguments)]
    fn project_bwd(
        transforms: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opac: FloatTensor<Self>,
        min_scale: FloatTensor<Self>,
        has_min_scale: bool,
        global_from_compact_gid: IntTensor<Self>,
        project_uniforms: ProjectUniforms,
        render_mode: SplatRenderMode,
        v_combined: FloatTensor<Self>,
    ) -> SplatGrads<Self>;

    /// Projection backward without allocating or writing a dense SH
    /// coefficient gradient. Used only by the private training bridge after
    /// optimizer compatibility has been checked.
    #[allow(clippy::too_many_arguments)]
    fn project_bwd_deferred_sh(
        transforms: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opac: FloatTensor<Self>,
        min_scale: FloatTensor<Self>,
        has_min_scale: bool,
        global_from_compact_gid: IntTensor<Self>,
        project_uniforms: ProjectUniforms,
        render_mode: SplatRenderMode,
        v_combined: FloatTensor<Self>,
    ) -> DeferredSplatGrads<Self> {
        // Preserve compatibility for external backend implementations. The
        // built-in Wgpu paths override this to avoid the dense coefficient
        // allocation; other backends may compute and discard it safely.
        let grads = Self::project_bwd(
            transforms,
            sh_coeffs,
            raw_opac,
            min_scale,
            has_min_scale,
            global_from_compact_gid,
            project_uniforms,
            render_mode,
            v_combined,
        );
        DeferredSplatGrads {
            v_transforms: grads.v_transforms,
            v_raw_opac: grads.v_raw_opac,
            v_refine_weight: grads.v_refine_weight,
        }
    }
}

/// Raster inputs saved by this crate's forward pass.
///
/// The fields and constructor stay private to this module so downstream safe
/// callers cannot claim the invariants required by the unchecked native-MSL
/// launch. Backend implementations can only consume a value produced by the
/// private autodiff bridge below.
pub(crate) struct ForwardRasterBackward<B: Backend> {
    out_img: FloatTensor<B>,
    projected_splats: FloatTensor<B>,
    compact_gid_from_isect: IntTensor<B>,
    tile_offsets: IntTensor<B>,
    background: Vec3,
    img_size: glam::UVec2,
    v_output: FloatTensor<B>,
    rasterizer: Rasterizer,
    smooth_cutoff: bool,
    compute_refine_weight: bool,
}

impl<B: Backend> ForwardRasterBackward<B> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        out_img: FloatTensor<B>,
        projected_splats: FloatTensor<B>,
        compact_gid_from_isect: IntTensor<B>,
        tile_offsets: IntTensor<B>,
        background: Vec3,
        img_size: glam::UVec2,
        v_output: FloatTensor<B>,
        rasterizer: Rasterizer,
        smooth_cutoff: bool,
        compute_refine_weight: bool,
    ) -> Self {
        Self {
            out_img,
            projected_splats,
            compact_gid_from_isect,
            tile_offsets,
            background,
            img_size,
            v_output,
            rasterizer,
            smooth_cutoff,
            compute_refine_weight,
        }
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn into_parts(
        self,
    ) -> (
        FloatTensor<B>,
        FloatTensor<B>,
        IntTensor<B>,
        IntTensor<B>,
        Vec3,
        glam::UVec2,
        FloatTensor<B>,
        Rasterizer,
        bool,
        bool,
    ) {
        (
            self.out_img,
            self.projected_splats,
            self.compact_gid_from_isect,
            self.tile_offsets,
            self.background,
            self.img_size,
            self.v_output,
            self.rasterizer,
            self.smooth_cutoff,
            self.compute_refine_weight,
        )
    }
}

/// Crate-private bridge for backward work that relies on forward-produced
/// tensor relationships. This is deliberately separate from [`SplatBwdOps`],
/// whose safe public methods always retain kernel bounds checks.
pub(crate) trait InternalSplatBwdOps: SplatBwdOps {
    fn rasterize_bwd_from_forward(input: ForwardRasterBackward<Self>) -> RasterizeGrads<Self>;
}

/// State saved during forward pass for backward computation.
#[derive(Debug, Clone)]
struct GaussianBackwardState<B: Backend> {
    transforms: FloatTensor<B>,
    sh_coeffs: FloatTensor<B>,
    raw_opacity: FloatTensor<B>,
    min_scale: FloatTensor<B>,
    has_min_scale: bool,

    projected_splats: FloatTensor<B>,
    project_uniforms: ProjectUniforms,
    global_from_compact_gid: IntTensor<B>,
    compact_from_global: IntTensor<B>,

    out_img: FloatTensor<B>,
    compact_gid_from_isect: IntTensor<B>,
    tile_offsets: IntTensor<B>,

    render_mode: SplatRenderMode,
    pass: crate::gaussian_splats::RasterPass,
    rasterizer: Rasterizer,
    background: Vec3,
    img_size: glam::UVec2,
}

#[derive(Debug)]
struct RenderBackwards;

const NUM_BWD_ARGS: usize = 5;

// Implement gradient registration when rendering backwards.
impl<B: Backend + InternalSplatBwdOps> Backward<B, NUM_BWD_ARGS> for RenderBackwards {
    type State = GaussianBackwardState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, NUM_BWD_ARGS>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let _span = tracing::trace_span!("render_gaussians backwards").entered();

        let state = ops.state;
        let v_output = grads.consume::<B>(&ops.node);

        // Register gradients for parent nodes (This code is already skipped entirely
        // if no parent nodes require gradients).
        let [
            transforms_parent,
            refine_weight,
            coeffs_parent,
            raw_opacity_parent,
            deferred_sh_parent,
        ] = ops.parents;
        let compute_refine_weight = refine_weight.is_some();

        let rasterize_grads = B::rasterize_bwd_from_forward(ForwardRasterBackward::new(
            state.out_img,
            state.projected_splats,
            state.compact_gid_from_isect,
            state.tile_offsets,
            state.background,
            state.img_size,
            v_output,
            state.rasterizer,
            state.pass.smooth_cutoff(),
            compute_refine_weight,
        ));

        // Row zero represents culled splats; leave the expansion lazy so
        // fusion can absorb it into the optimizer.
        let inv = state.compact_from_global;
        let dense = |compact: FloatTensor<B>| B::float_select(compact, 0, inv.clone());
        if let Some(deferred_sh_parent) = deferred_sh_parent {
            let compact_sh_grads = rasterize_grads.v_combined.clone();
            let splat_grads = B::project_bwd_deferred_sh(
                state.transforms,
                state.sh_coeffs,
                state.raw_opacity,
                state.min_scale,
                state.has_min_scale,
                state.global_from_compact_gid,
                state.project_uniforms,
                state.render_mode,
                rasterize_grads.v_combined,
            );

            if let Some(node) = transforms_parent {
                grads.register::<B>(node.id, dense(splat_grads.v_transforms));
            }
            if let Some(node) = refine_weight {
                grads.register::<B>(node.id, dense(splat_grads.v_refine_weight));
            }
            if let Some(node) = raw_opacity_parent {
                grads.register::<B>(node.id, dense(splat_grads.v_raw_opac));
            }
            grads.register::<B>(deferred_sh_parent.id, compact_sh_grads);
        } else {
            let splat_grads = B::project_bwd(
                state.transforms,
                state.sh_coeffs,
                state.raw_opacity,
                state.min_scale,
                state.has_min_scale,
                state.global_from_compact_gid,
                state.project_uniforms,
                state.render_mode,
                rasterize_grads.v_combined,
            );

            if let Some(node) = transforms_parent {
                grads.register::<B>(node.id, dense(splat_grads.v_transforms));
            }
            if let Some(node) = refine_weight {
                grads.register::<B>(node.id, dense(splat_grads.v_refine_weight));
            }
            if let Some(node) = coeffs_parent {
                grads.register::<B>(node.id, dense(splat_grads.v_coeffs));
            }
            if let Some(node) = raw_opacity_parent {
                grads.register::<B>(node.id, dense(splat_grads.v_raw_opac));
            }
        }
    }
}

/// Sparse SH-gradient payload extracted after the render backward pass.
/// Tensor clones in this payload are handle-only and preserve the exact
/// forward-time projection inputs.
#[doc(hidden)]
pub struct DeferredShGrad {
    pub compact_grads: Tensor<2>,
    pub render_transforms: Tensor<2>,
    pub global_from_compact_gid: Tensor<1, Int>,
    pub project_uniforms: ProjectUniforms,
}

/// Autodiff holder used to route compact raster gradients across Burn's
/// backward boundary without assigning them to the dense SH parameter.
#[doc(hidden)]
pub struct DeferredShGradHandle {
    holder: Tensor<2>,
    render_transforms: Tensor<2>,
    global_from_compact_gid: Tensor<1, Int>,
    project_uniforms: ProjectUniforms,
}

impl DeferredShGradHandle {
    pub fn take(self, grads: &mut TensorGradients) -> Option<DeferredShGrad> {
        Some(DeferredShGrad {
            compact_grads: self.holder.grad_remove(grads)?,
            render_transforms: self.render_transforms,
            global_from_compact_gid: self.global_from_compact_gid,
            project_uniforms: self.project_uniforms,
        })
    }
}

pub struct SplatOutputDiff {
    /// Rendered image, on the autodiff graph (this is what the loss backprops through).
    pub img: Tensor<3>,
    pub num_visible: u32,
    /// Per-splat visibility aux — on the **inner** backend (no gradients).
    pub visible: Tensor<1>,
    /// Per-splat max screen radius aux — on the **inner** backend (no gradients).
    pub max_radius: Tensor<1>,
    /// Per-splat opacity with the scale floor folded in — on the **inner**
    /// backend (no gradients). Zero for culled splats.
    pub opacities: Tensor<1>,
    pub refine_weight_holder: Tensor<1>,
}

/// Private cross-crate training protocol. Keeping this separate preserves the
/// public [`SplatOutputDiff`] layout for downstream struct construction and
/// exhaustive destructuring.
#[doc(hidden)]
pub struct TrainingSplatOutputDiff {
    pub opacities: Tensor<1>,
    pub img: Tensor<3>,
    pub num_visible: u32,
    pub visible: Tensor<1>,
    pub max_radius: Tensor<1>,
    pub refine_weight_holder: Tensor<1>,
    pub deferred_sh_grad: Option<DeferredShGradHandle>,
}

impl TrainingSplatOutputDiff {
    fn into_public(self) -> SplatOutputDiff {
        SplatOutputDiff {
            img: self.img,
            num_visible: self.num_visible,
            visible: self.visible,
            max_radius: self.max_radius,
            opacities: self.opacities,
            refine_weight_holder: self.refine_weight_holder,
        }
    }
}

/// Enable autodiff on trainable splat parameters while preserving parameter
/// identities and the frozen scale floor.
pub fn lift_splats_to_autodiff(splats: Splats) -> Splats {
    let mip = splats.render_mip;
    let min_scale = splats.min_scale.clone();
    let (transforms_id, transforms, _) = splats.transforms.consume();
    let (sh_coeffs_id, sh_coeffs, _) = splats.sh_coeffs.consume();
    let (raw_opacity_id, raw_opacity, _) = splats.raw_opacities.consume();
    Splats {
        transforms: Param::initialized(transforms_id, lift_to_autodiff(transforms).require_grad()),
        sh_coeffs: Param::initialized(sh_coeffs_id, lift_to_autodiff(sh_coeffs).require_grad()),
        raw_opacities: Param::initialized(
            raw_opacity_id,
            lift_to_autodiff(raw_opacity).require_grad(),
        ),
        render_mip: mip,
        // Keep the frozen floor on the inner backend. `#[module(skip)]` fields
        // aren't converted by `.valid()`, so lifting it here would leave an
        // autodiff `f` on an inner module after eval-strip and mix backends in
        // `scales()`/`opacities()`. The bwd render lifts a temporary copy.
        min_scale,
    }
}

/// Render splats on a differentiable device.
///
/// Panics if the device is not autodiff-enabled.
pub async fn render_splats(
    splats: Splats,
    camera: &Camera,
    img_size: glam::UVec2,
    background: Vec3,
) -> SplatOutputDiff {
    render_splats_with_refine_weight(splats, camera, img_size, background, true).await
}

/// Render splats on a differentiable device, optionally tracking the
/// refinement-only screen-space gradient statistic.
///
/// Model gradients and render auxiliaries are unchanged when
/// `compute_refine_weight` is false.
pub async fn render_splats_with_refine_weight(
    splats: Splats,
    camera: &Camera,
    img_size: glam::UVec2,
    background: Vec3,
    compute_refine_weight: bool,
) -> SplatOutputDiff {
    render_splats_with_pass_and_refine_weight(
        splats,
        camera,
        img_size,
        background,
        crate::gaussian_splats::RasterPass::Backward,
        Rasterizer::Legacy,
        compute_refine_weight,
        false,
    )
    .await
    .into_public()
}

/// Training-only render entry point that can route compact SH gradients to a
/// compatible optimizer. When `defer_sh_grad` is honored, backward does not
/// populate `splats.sh_coeffs.grad()`; the caller must extract and consume
/// `deferred_sh_grad` with that optimizer. Unsupported builds ignore the
/// request and preserve the dense coefficient gradient.
#[doc(hidden)]
pub async fn render_splats_for_training(
    splats: Splats,
    camera: &Camera,
    img_size: glam::UVec2,
    background: Vec3,
    compute_refine_weight: bool,
    defer_sh_grad: bool,
) -> TrainingSplatOutputDiff {
    let defer_sh_grad = defer_sh_grad
        && cfg!(all(
            feature = "native-msl",
            target_os = "macos",
            target_arch = "aarch64",
            not(target_family = "wasm")
        ));
    render_splats_with_pass_and_refine_weight(
        splats,
        camera,
        img_size,
        background,
        crate::gaussian_splats::RasterPass::Backward,
        training_rasterizer(),
        compute_refine_weight,
        defer_sh_grad,
    )
    .await
}

/// Like [`render_splats`] but lets the caller pick the
/// [`crate::gaussian_splats::RasterPass`]. Used by the finite-diff
/// test suite to enable the C^1 smooth-cutoff surrogate; production code
/// should use [`render_splats`].
pub async fn render_splats_with_pass(
    splats: Splats,
    camera: &Camera,
    img_size: glam::UVec2,
    background: Vec3,
    pass: crate::gaussian_splats::RasterPass,
) -> SplatOutputDiff {
    render_splats_with_pass_and_rasterizer(
        splats,
        camera,
        img_size,
        background,
        pass,
        Rasterizer::Legacy,
    )
    .await
}

/// Selector-aware differentiable render entry point for internal rasterizer
/// parity tests. Product code should use [`render_splats_with_pass`].
#[doc(hidden)]
pub async fn render_splats_with_pass_and_rasterizer(
    splats: Splats,
    camera: &Camera,
    img_size: glam::UVec2,
    background: Vec3,
    pass: crate::gaussian_splats::RasterPass,
    rasterizer: Rasterizer,
) -> SplatOutputDiff {
    render_splats_with_pass_and_refine_weight(
        splats, camera, img_size, background, pass, rasterizer, true, false,
    )
    .await
    .into_public()
}

async fn render_splats_with_pass_and_refine_weight(
    splats: Splats,
    camera: &Camera,
    img_size: glam::UVec2,
    background: Vec3,
    pass: crate::gaussian_splats::RasterPass,
    rasterizer: Rasterizer,
    compute_refine_weight: bool,
    defer_sh_grad: bool,
) -> TrainingSplatOutputDiff {
    splats.clone().validate_values().await;

    let device = splats.device();
    assert!(
        device.is_autodiff(),
        "brush_render::bwd::render_splats requires an autodiff-enabled device"
    );

    let refine_weight_holder = Tensor::<1>::zeros([1], &device);
    let refine_weight_holder = if compute_refine_weight {
        refine_weight_holder.require_grad()
    } else {
        refine_weight_holder
    };
    let deferred_sh_holder = Tensor::<2>::zeros([1, 1], &device);
    let deferred_sh_holder = if defer_sh_grad {
        deferred_sh_holder.require_grad()
    } else {
        deferred_sh_holder
    };

    // The 3D-filter floor is applied inside the projection kernels. It lives
    // on the inner backend and carries no gradient, so lifting it onto the
    // autodiff device is just a wrap.
    let (min_scale, has_min_scale) = splats.min_scale_arg();

    let transforms_ad = unwrap_ad_wgpu_float(splats.transforms.val());
    let sh_coeffs_ad = unwrap_ad_wgpu_float(splats.sh_coeffs.val());
    let raw_opac_ad = unwrap_ad_wgpu_float(splats.raw_opacities.val());
    let refine_weight_ad = unwrap_ad_wgpu_float(refine_weight_holder.clone());
    let deferred_sh_ad = unwrap_ad_wgpu_float(deferred_sh_holder.clone());

    let prep_nodes = RenderBackwards
        .prepare::<NoCheckpointing>([
            transforms_ad.node(),
            refine_weight_ad.node(),
            sh_coeffs_ad.node(),
            raw_opac_ad.node(),
            deferred_sh_ad.node(),
        ])
        .compute_bound()
        .stateful();

    let render_mode = if splats.render_mip {
        SplatRenderMode::Mip
    } else {
        SplatRenderMode::Default
    };

    let min_scale_inner = crate::burn_glue::unwrap_wgpu_float(min_scale);
    let transforms_inner: FloatTensor<MainBackend> = transforms_ad.primitive().clone();
    let sh_inner: FloatTensor<MainBackend> = sh_coeffs_ad.into_primitive();
    let raw_opac_inner: FloatTensor<MainBackend> = raw_opac_ad.primitive().clone();

    assert!(
        pass.bwd_info(),
        "render_splats_with_pass requires a Backward variant"
    );
    let output = <MainBackend as SplatRasterizerOps>::render_with_rasterizer(
        camera,
        img_size,
        transforms_inner.clone(),
        sh_inner.clone(),
        raw_opac_inner.clone(),
        min_scale_inner.clone(),
        has_min_scale,
        render_mode,
        background,
        pass,
        rasterizer,
    )
    .await;

    output.clone().validate().await;

    let num_visible = output.aux.num_visible;
    let visible_inner = output.aux.visible.clone();
    let max_radius_inner = output.aux.max_radius.clone();
    let opacities_inner = output.aux.opacities.clone();
    let deferred_sh_grad = defer_sh_grad.then(|| DeferredShGradHandle {
        holder: deferred_sh_holder,
        render_transforms: wrap_wgpu_float(transforms_inner.clone()),
        global_from_compact_gid: wrap_wgpu_int(output.global_from_compact_gid.clone()),
        project_uniforms: output.project_uniforms,
    });

    let img_ad: FloatTensor<AutodiffMain> = match prep_nodes {
        OpsKind::Tracked(prep) => {
            let state = GaussianBackwardState {
                transforms: transforms_inner,
                sh_coeffs: sh_inner,
                raw_opacity: raw_opac_inner,
                min_scale: min_scale_inner,
                has_min_scale,
                out_img: output.out_img.clone(),
                projected_splats: output.projected_splats,
                project_uniforms: output.project_uniforms,
                tile_offsets: output.aux.tile_offsets.clone(),
                compact_gid_from_isect: output.compact_gid_from_isect,
                render_mode,
                pass,
                rasterizer,
                global_from_compact_gid: output.global_from_compact_gid,
                compact_from_global: output.compact_from_global,
                background,
                img_size,
            };
            prep.finish(state, output.out_img)
        }
        OpsKind::UnTracked(prep) => prep.finish(output.out_img),
    };

    TrainingSplatOutputDiff {
        img: wrap_ad_wgpu_float(img_ad),
        num_visible,
        // `visible` / `max_radius` are render aux — they only feed refine
        // bookkeeping and never get a backward. Hand them back on the inner
        // backend directly so callers don't have to strip autodiff off them.
        visible: wrap_wgpu_float(visible_inner),
        max_radius: wrap_wgpu_float(max_radius_inner),
        opacities: wrap_wgpu_float(opacities_inner),
        refine_weight_holder,
        deferred_sh_grad,
    }
}

#[allow(clippy::too_many_arguments)]
fn rasterize_bwd_fusion(
    out_img: FloatTensor<Fusion<CubeBackend>>,
    projected_splats: FloatTensor<Fusion<CubeBackend>>,
    compact_gid_from_isect: IntTensor<Fusion<CubeBackend>>,
    tile_offsets: IntTensor<Fusion<CubeBackend>>,
    background: Vec3,
    img_size: glam::UVec2,
    v_output: FloatTensor<Fusion<CubeBackend>>,
    rasterizer: Rasterizer,
    smooth_cutoff: bool,
    compute_refine_weight: bool,
    trusted_forward: bool,
) -> RasterizeGrads<Fusion<CubeBackend>> {
    let num_visible = projected_splats.shape()[0].max(1);
    let client = v_output.client.clone();
    let [v_combined] = register_custom(
        &client,
        "rasterize_bwd",
        [
            v_output,
            out_img,
            projected_splats,
            compact_gid_from_isect,
            tile_offsets,
        ],
        [(Shape::new([num_visible, 10]), DType::F32)],
        move |desc, h| {
            let (
                [
                    v_output,
                    out_img,
                    projected_splats,
                    compact_gid_from_isect,
                    tile_offsets,
                ],
                [v_combined],
            ) = desc.as_fixed();
            let grads = if trusted_forward {
                <CubeBackend as InternalSplatBwdOps>::rasterize_bwd_from_forward(
                    ForwardRasterBackward::new(
                        h.get_float_tensor::<CubeBackend>(out_img),
                        h.get_float_tensor::<CubeBackend>(projected_splats),
                        h.get_int_tensor::<CubeBackend>(compact_gid_from_isect),
                        h.get_int_tensor::<CubeBackend>(tile_offsets),
                        background,
                        img_size,
                        h.get_float_tensor::<CubeBackend>(v_output),
                        rasterizer,
                        smooth_cutoff,
                        compute_refine_weight,
                    ),
                )
            } else {
                <CubeBackend as SplatBwdOps>::rasterize_bwd_with_refine_weight(
                    h.get_float_tensor::<CubeBackend>(out_img),
                    h.get_float_tensor::<CubeBackend>(projected_splats),
                    h.get_int_tensor::<CubeBackend>(compact_gid_from_isect),
                    h.get_int_tensor::<CubeBackend>(tile_offsets),
                    background,
                    img_size,
                    h.get_float_tensor::<CubeBackend>(v_output),
                    smooth_cutoff,
                    compute_refine_weight,
                )
            };

            h.register_float_tensor::<CubeBackend>(&v_combined.id, grads.v_combined);
        },
    );

    RasterizeGrads { v_combined }
}

impl<B: Backend + SplatOps + InternalSplatBwdOps, C: CheckpointStrategy> SplatOps
    for Autodiff<B, C>
{
    #[allow(clippy::too_many_arguments)]
    async fn render(
        camera: &Camera,
        img_size: glam::UVec2,
        transforms: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        min_scale: FloatTensor<Self>,
        has_min_scale: bool,
        refine_weight: FloatTensor<Self>,
        render_mode: SplatRenderMode,
        background: Vec3,
        pass: crate::gaussian_splats::RasterPass,
    ) -> crate::RenderOutput<Self> {
        // Keep the fork's five-parent backward operation uniform. This
        // untracked tensor occupies the deferred-SH slot for the canonical
        // backend-extension path, so normal renders still materialize the
        // dense coefficient gradient.
        let deferred_sh = <Self as AutodiffBackend>::from_inner(sh_coeffs.primitive().clone());
        let prep_nodes = RenderBackwards
            .prepare::<NoCheckpointing>([
                transforms.node(),
                refine_weight.node(),
                sh_coeffs.node(),
                raw_opacities.node(),
                deferred_sh.node(),
            ])
            .compute_bound()
            .stateful();

        let transforms_inner: FloatTensor<B> = transforms.primitive().clone();
        let sh_inner: FloatTensor<B> = sh_coeffs.into_primitive();
        let raw_opac_inner: FloatTensor<B> = raw_opacities.primitive().clone();
        let min_scale_inner: FloatTensor<B> = min_scale.into_primitive();

        let output = <B as SplatOps>::render(
            camera,
            img_size,
            transforms_inner.clone(),
            sh_inner.clone(),
            raw_opac_inner.clone(),
            min_scale_inner.clone(),
            has_min_scale,
            refine_weight.into_primitive(),
            render_mode,
            background,
            pass,
        )
        .await;

        output.clone().validate().await;

        let img_ad: FloatTensor<Self> = match prep_nodes {
            OpsKind::Tracked(prep) => {
                let state = GaussianBackwardState {
                    transforms: transforms_inner,
                    sh_coeffs: sh_inner,
                    raw_opacity: raw_opac_inner,
                    min_scale: min_scale_inner,
                    has_min_scale,
                    out_img: output.out_img.clone(),
                    projected_splats: output.projected_splats.clone(),
                    project_uniforms: output.project_uniforms,
                    tile_offsets: output.aux.tile_offsets.clone(),
                    compact_gid_from_isect: output.compact_gid_from_isect.clone(),
                    render_mode,
                    pass,
                    rasterizer: Rasterizer::Legacy,
                    global_from_compact_gid: output.global_from_compact_gid.clone(),
                    compact_from_global: output.compact_from_global.clone(),
                    background,
                    img_size,
                };
                prep.finish(state, output.out_img)
            }
            OpsKind::UnTracked(prep) => prep.finish(output.out_img),
        };

        // Lift the remaining float aux onto the autodiff graph. None of these
        // carry a backward — they only feed refine bookkeeping — but the
        // extension trait's output is uniformly `RenderOutput<Self>`, so they
        // ride along as untracked autodiff tensors. Int tensors share the
        // inner backend's primitive and pass through unchanged.
        let lift = <Self as AutodiffBackend>::from_inner;

        crate::RenderOutput {
            out_img: img_ad,
            aux: crate::RenderAuxInner {
                num_visible: output.aux.num_visible,
                num_intersections: output.aux.num_intersections,
                visible: lift(output.aux.visible),
                max_radius: lift(output.aux.max_radius),
                opacities: lift(output.aux.opacities),
                tile_offsets: output.aux.tile_offsets,
                img_size: output.aux.img_size,
            },
            projected_splats: lift(output.projected_splats),
            compact_gid_from_isect: output.compact_gid_from_isect,
            project_uniforms: output.project_uniforms,
            global_from_compact_gid: output.global_from_compact_gid,
            compact_from_global: output.compact_from_global,
        }
    }
}

impl SplatBwdOps for Fusion<CubeBackend> {
    #[allow(clippy::too_many_arguments)]
    fn rasterize_bwd(
        out_img: FloatTensor<Self>,
        projected_splats: FloatTensor<Self>,
        compact_gid_from_isect: IntTensor<Self>,
        tile_offsets: IntTensor<Self>,
        background: Vec3,
        img_size: glam::UVec2,
        v_output: FloatTensor<Self>,
        smooth_cutoff: bool,
    ) -> RasterizeGrads<Self> {
        rasterize_bwd_fusion(
            out_img,
            projected_splats,
            compact_gid_from_isect,
            tile_offsets,
            background,
            img_size,
            v_output,
            Rasterizer::Legacy,
            smooth_cutoff,
            true,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn rasterize_bwd_with_refine_weight(
        out_img: FloatTensor<Self>,
        projected_splats: FloatTensor<Self>,
        compact_gid_from_isect: IntTensor<Self>,
        tile_offsets: IntTensor<Self>,
        background: Vec3,
        img_size: glam::UVec2,
        v_output: FloatTensor<Self>,
        smooth_cutoff: bool,
        compute_refine_weight: bool,
    ) -> RasterizeGrads<Self> {
        rasterize_bwd_fusion(
            out_img,
            projected_splats,
            compact_gid_from_isect,
            tile_offsets,
            background,
            img_size,
            v_output,
            Rasterizer::Legacy,
            smooth_cutoff,
            compute_refine_weight,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn project_bwd(
        transforms: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opac: FloatTensor<Self>,
        min_scale: FloatTensor<Self>,
        has_min_scale: bool,
        global_from_compact_gid: IntTensor<Self>,
        project_uniforms: ProjectUniforms,
        render_mode: SplatRenderMode,
        v_combined: FloatTensor<Self>,
    ) -> SplatGrads<Self> {
        let client = transforms.client.clone();
        let rows = project_uniforms.num_visible as usize + 1;
        let coeffs = sh_coeffs_for_degree(project_uniforms.sh_degree) as usize;
        let [v_transforms, v_coeffs, v_raw_opac, v_refine_weight] = register_custom(
            &client,
            "project_bwd",
            [
                transforms,
                sh_coeffs,
                raw_opac,
                min_scale,
                global_from_compact_gid,
                v_combined,
            ],
            [
                (Shape::new([rows, 10]), DType::F32),
                (Shape::new([rows, coeffs, 3]), DType::F32),
                (Shape::new([rows]), DType::F32),
                (Shape::new([rows]), DType::F32),
            ],
            move |desc, h| {
                let (
                    [
                        transforms,
                        sh_coeffs,
                        raw_opac,
                        min_scale,
                        global_from_compact_gid,
                        v_combined,
                    ],
                    [v_transforms, v_coeffs, v_raw_opac, v_refine_weight],
                ) = desc.as_fixed();
                let grads = <CubeBackend as SplatBwdOps>::project_bwd(
                    h.get_float_tensor::<CubeBackend>(transforms),
                    h.get_float_tensor::<CubeBackend>(sh_coeffs),
                    h.get_float_tensor::<CubeBackend>(raw_opac),
                    h.get_float_tensor::<CubeBackend>(min_scale),
                    has_min_scale,
                    h.get_int_tensor::<CubeBackend>(global_from_compact_gid),
                    project_uniforms,
                    render_mode,
                    h.get_float_tensor::<CubeBackend>(v_combined),
                );
                h.register_float_tensor::<CubeBackend>(&v_transforms.id, grads.v_transforms);
                h.register_float_tensor::<CubeBackend>(&v_coeffs.id, grads.v_coeffs);
                h.register_float_tensor::<CubeBackend>(&v_raw_opac.id, grads.v_raw_opac);
                h.register_float_tensor::<CubeBackend>(&v_refine_weight.id, grads.v_refine_weight);
            },
        );
        SplatGrads {
            v_transforms,
            v_coeffs,
            v_raw_opac,
            v_refine_weight,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn project_bwd_deferred_sh(
        transforms: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opac: FloatTensor<Self>,
        min_scale: FloatTensor<Self>,
        has_min_scale: bool,
        global_from_compact_gid: IntTensor<Self>,
        project_uniforms: ProjectUniforms,
        render_mode: SplatRenderMode,
        v_combined: FloatTensor<Self>,
    ) -> DeferredSplatGrads<Self> {
        let client = transforms.client.clone();
        let rows = project_uniforms.num_visible as usize + 1;
        let [v_transforms, v_raw_opac, v_refine_weight] = register_custom(
            &client,
            "project_bwd_deferred_sh",
            [
                transforms,
                sh_coeffs,
                raw_opac,
                min_scale,
                global_from_compact_gid,
                v_combined,
            ],
            [
                (Shape::new([rows, 10]), DType::F32),
                (Shape::new([rows]), DType::F32),
                (Shape::new([rows]), DType::F32),
            ],
            move |desc, h| {
                let (
                    [
                        transforms,
                        sh_coeffs,
                        raw_opac,
                        min_scale,
                        global_from_compact_gid,
                        v_combined,
                    ],
                    [v_transforms, v_raw_opac, v_refine_weight],
                ) = desc.as_fixed();
                let grads = <CubeBackend as SplatBwdOps>::project_bwd_deferred_sh(
                    h.get_float_tensor::<CubeBackend>(transforms),
                    h.get_float_tensor::<CubeBackend>(sh_coeffs),
                    h.get_float_tensor::<CubeBackend>(raw_opac),
                    h.get_float_tensor::<CubeBackend>(min_scale),
                    has_min_scale,
                    h.get_int_tensor::<CubeBackend>(global_from_compact_gid),
                    project_uniforms,
                    render_mode,
                    h.get_float_tensor::<CubeBackend>(v_combined),
                );
                h.register_float_tensor::<CubeBackend>(&v_transforms.id, grads.v_transforms);
                h.register_float_tensor::<CubeBackend>(&v_raw_opac.id, grads.v_raw_opac);
                h.register_float_tensor::<CubeBackend>(&v_refine_weight.id, grads.v_refine_weight);
            },
        );

        DeferredSplatGrads {
            v_transforms,
            v_raw_opac,
            v_refine_weight,
        }
    }
}

impl InternalSplatBwdOps for Fusion<CubeBackend> {
    fn rasterize_bwd_from_forward(input: ForwardRasterBackward<Self>) -> RasterizeGrads<Self> {
        let (
            out_img,
            projected_splats,
            compact_gid_from_isect,
            tile_offsets,
            background,
            img_size,
            v_output,
            rasterizer,
            smooth_cutoff,
            compute_refine_weight,
        ) = input.into_parts();
        rasterize_bwd_fusion(
            out_img,
            projected_splats,
            compact_gid_from_isect,
            tile_offsets,
            background,
            img_size,
            v_output,
            rasterizer,
            smooth_cutoff,
            compute_refine_weight,
            true,
        )
    }
}
