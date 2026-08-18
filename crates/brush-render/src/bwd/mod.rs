//! Backward (differentiable) render path.
//!
//! Lives in `brush-render` rather than a separate crate because the
//! `#[backend_extension]`-generated `Dispatch` impl calls the `Autodiff` arm,
//! so `impl SplatOps for Autodiff<..>` must be visible from the crate that
//! defines `SplatOps`.
pub mod burn_glue;
mod kernels;
mod render_bwd;

pub use burn_glue::{
    DeferredShGrad, DeferredShGradHandle, SplatOutputDiff, TrainingSplatOutputDiff,
    lift_splats_to_autodiff, render_splats, render_splats_for_training, render_splats_with_pass,
    render_splats_with_pass_and_rasterizer, render_splats_with_refine_weight,
};
