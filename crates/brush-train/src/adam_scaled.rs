use burn::tensor::{ElementConversion, Tensor};

#[cfg(all(
    feature = "native-msl",
    target_os = "macos",
    target_arch = "aarch64",
    not(target_family = "wasm")
))]
use brush_render::shaders::helpers::ProjectUniforms;
#[cfg(all(
    feature = "native-msl",
    target_os = "macos",
    target_arch = "aarch64",
    not(target_family = "wasm")
))]
use burn::tensor::Int;

#[cfg(all(
    feature = "native-msl",
    target_os = "macos",
    target_arch = "aarch64",
    not(target_family = "wasm")
))]
fn use_fused_sh_adam() -> bool {
    use std::sync::OnceLock;

    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let enabled =
            brush_render::native_msl::option_requested(brush_render::native_msl::FUSED_SH_ADAM_ENV);
        if enabled {
            tracing::warn!("experimental native-MSL fused SH Adam enabled");
        }
        enabled
    })
}

/// Adam with per-parameter second-moment reduction (via [`AdamState::reduce_moment_2`])
/// and per-component learning-rate scaling (via [`AdamState::scaling`]).
///
/// Hand-rolled rather than built on burn's `Adam`, which has neither the
/// per-component LR scaling nor the second-moment reduction below. The
/// trainer also edits momentum tensors in place during refine (split/prune)
/// and swaps the transforms LR scaling every step.
#[derive(Clone)]
pub(crate) struct AdamScaled {
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
}

/// Per-parameter momentum state. When `reduce_moment_2` is set on the owning
/// [`AdamState`], `moment_2` has size 1 in trailing dims; [`AdamState::map_momentum`]
/// callers must stay shape-agnostic along those.
#[derive(Clone)]
pub(crate) struct MomentumState<const D: usize> {
    pub moment_1: Tensor<D>,
    pub moment_2: Tensor<D>,
    pub time: usize,
}

/// Per-parameter optimizer state.
#[derive(Clone)]
pub(crate) struct AdamState<const D: usize> {
    pub momentum: Option<MomentumState<D>>,
    /// Per-component learning rate scaling (e.g. different LR for means vs
    /// rotations vs scales within the transforms tensor).
    pub scaling: Option<Tensor<D>>,
    /// When true, the second moment is reduced to a scalar per row. Set by the
    /// caller when initializing state for parameters where per-element variance
    /// is not needed.
    pub reduce_moment_2: bool,
}

impl<const D: usize> AdamState<D> {
    pub fn new(scaling: Option<Tensor<D>>, reduce_moment_2: bool) -> Self {
        Self {
            momentum: None,
            scaling,
            reduce_moment_2,
        }
    }

    /// Apply `map_fn` to both momentum tensors (no-op before the first step).
    /// `map_fn` must be shape-agnostic along trailing dims since `moment_2`
    /// may have size-1 trailing dims under `reduce_moment_2`.
    pub fn map_momentum(&mut self, map_fn: impl Fn(Tensor<D>) -> Tensor<D>) {
        self.momentum = self.momentum.take().map(|mut moment| {
            moment.moment_1 = map_fn(moment.moment_1);
            moment.moment_2 = map_fn(moment.moment_2);
            moment
        });
    }
}

impl AdamScaled {
    pub fn new(epsilon: f32) -> Self {
        Self {
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon,
        }
    }

    #[cfg(all(
        feature = "native-msl",
        target_os = "macos",
        target_arch = "aarch64",
        not(target_family = "wasm")
    ))]
    fn sh_adam_config(&self, lr: f64, next_time: usize) -> crate::sh_adam::ShAdamConfig {
        let time = next_time as i32;
        crate::sh_adam::ShAdamConfig {
            beta_1: self.beta_1,
            beta_2: self.beta_2,
            bias_correction_1: 1.0 - self.beta_1.powi(time),
            bias_correction_2: 1.0 - self.beta_2.powi(time),
            epsilon: self.epsilon,
            learning_rate: lr as f32,
        }
    }

    #[cfg(all(
        feature = "native-msl",
        target_os = "macos",
        target_arch = "aarch64",
        not(target_family = "wasm")
    ))]
    pub(crate) fn fused_sh_compatible<const D: usize>(
        param: &Tensor<D>,
        state: &AdamState<D>,
    ) -> bool {
        if !use_fused_sh_adam() || !state.reduce_moment_2 || D != 3 {
            return false;
        }
        let shape = param.dims();
        let mut reduced_shape = [1usize; D];
        reduced_shape[0] = shape[0];
        let mut scaling_shape = [1usize; D];
        scaling_shape[1] = shape[1];
        let (Some(momentum), Some(scaling)) = (state.momentum.as_ref(), state.scaling.as_ref())
        else {
            return false;
        };
        shape[0] > 0
            && shape[2] == 3
            && matches!(shape[1], 1 | 4 | 9 | 16 | 25)
            && momentum.moment_1.dims() == shape
            && momentum.moment_2.dims() == reduced_shape
            && scaling.dims() == scaling_shape
            && crate::sh_adam::fused_sh_adam_supported(param)
    }

    #[cfg(all(
        feature = "native-msl",
        target_os = "macos",
        target_arch = "aarch64",
        not(target_family = "wasm")
    ))]
    pub(crate) fn sparse_sh_compatible(param: &Tensor<3>, state: &AdamState<3>) -> bool {
        if !state.reduce_moment_2 {
            return false;
        }
        let [num_splats, coeffs, channels] = param.dims();
        let Some(momentum) = state.momentum.as_ref() else {
            return false;
        };
        let Some(scaling) = state.scaling.as_ref() else {
            return false;
        };
        channels == 3
            && num_splats > 0
            && matches!(coeffs, 1 | 4 | 9 | 16 | 25)
            && momentum.moment_1.dims() == [num_splats, coeffs, 3]
            && momentum.moment_2.dims() == [num_splats, 1, 1]
            && scaling.dims() == [1, coeffs, 1]
    }

    #[cfg(all(
        feature = "native-msl",
        target_os = "macos",
        target_arch = "aarch64",
        not(target_family = "wasm")
    ))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step_sparse_sh(
        &self,
        lr: f64,
        param: Tensor<3>,
        render_transforms: Tensor<2>,
        global_from_compact_gid: Tensor<1, Int>,
        compact_grads: Tensor<2>,
        project_uniforms: ProjectUniforms,
        state: &mut AdamState<3>,
    ) -> Tensor<3> {
        assert!(
            Self::sparse_sh_compatible(&param, state),
            "sparse SH Adam requires preflighted parameter and optimizer state"
        );
        let momentum = state
            .momentum
            .take()
            .expect("sparse SH Adam momentum was preflighted");
        let scaling = state
            .scaling
            .as_ref()
            .expect("sparse SH Adam scaling was preflighted")
            .clone();
        let next_time = momentum.time + 1;
        let config = self.sh_adam_config(lr, next_time);
        let (param, moment_1, moment_2) = crate::sh_adam::sparse_sh_adam(
            param,
            render_transforms,
            global_from_compact_gid,
            compact_grads,
            momentum.moment_1,
            momentum.moment_2,
            scaling,
            project_uniforms,
            config,
        );
        state.momentum = Some(MomentumState {
            moment_1,
            moment_2,
            time: next_time,
        });
        param
    }

    /// One Adam step for a single parameter. `tensor` and `grad` live on the
    /// inner (non-autodiff) backend; `state` is updated in place.
    ///
    /// `grad_sq_mean` supplies the reduced second moment when the caller can
    /// produce it more cheaply than squaring `grad`. It must equal what
    /// `mean_trailing_dims(grad * grad)` would give, and is only read when
    /// [`AdamState::reduce_moment_2`] is set. Native fused SH Adam computes
    /// this statistic within its update kernel instead.
    pub fn step<const D: usize>(
        &self,
        lr: f64,
        tensor: Tensor<D>,
        grad: &Tensor<D>,
        grad_sq_mean: Option<Tensor<D>>,
        state: &mut AdamState<D>,
    ) -> Tensor<D> {
        #[cfg(all(
            feature = "native-msl",
            target_os = "macos",
            target_arch = "aarch64",
            not(target_family = "wasm")
        ))]
        if Self::fused_sh_compatible(&tensor, state) {
            let shape = tensor.dims();
            let coeffs = shape[1];
            let mut reduced_shape = [1usize; D];
            reduced_shape[0] = shape[0];
            let momentum = state
                .momentum
                .as_ref()
                .expect("fused SH momentum was preflighted");
            let scaling = state
                .scaling
                .as_ref()
                .expect("fused SH scaling was preflighted");
            let next_time = momentum.time + 1;
            let config = self.sh_adam_config(lr, next_time);
            let (tensor, moment_1, moment_2) = crate::sh_adam::sh_adam(
                tensor.reshape([shape[0], coeffs, 3]),
                grad.clone().reshape([shape[0], coeffs, 3]),
                momentum.moment_1.clone().reshape([shape[0], coeffs, 3]),
                momentum.moment_2.clone().reshape([shape[0], 1, 1]),
                scaling.clone().reshape([1, coeffs, 1]),
                config,
            );
            state.momentum = Some(MomentumState {
                moment_1: moment_1.reshape(shape),
                moment_2: moment_2.reshape(reduced_shape),
                time: next_time,
            });
            return tensor.reshape(shape);
        }

        let (grad, momentum) = self.transform(
            grad,
            grad_sq_mean,
            state.momentum.take(),
            state.reduce_moment_2,
        );
        state.momentum = Some(momentum);

        let delta = if let Some(scale) = &state.scaling {
            grad * (scale.clone() * lr).unsqueeze()
        } else {
            grad * lr
        };
        tensor - delta
    }

    fn transform<const D: usize>(
        &self,
        grad: &Tensor<D>,
        grad_sq_mean: Option<Tensor<D>>,
        momentum_state: Option<MomentumState<D>>,
        reduce_moment_2: bool,
    ) -> (Tensor<D>, MomentumState<D>) {
        let grad_sq_for_moment = if reduce_moment_2 && D > 1 {
            grad_sq_mean.unwrap_or_else(|| mean_trailing_dims(grad.clone().powi_scalar(2)))
        } else {
            grad.clone().powi_scalar(2)
        };

        let state = if let Some(mut state) = momentum_state {
            let factor = 1.0 - self.beta_1;
            state.moment_1 = state
                .moment_1
                .mul_scalar(self.beta_1)
                .add(grad.clone().mul_scalar(factor));

            let factor = 1.0 - self.beta_2;
            state.moment_2 = state
                .moment_2
                .mul_scalar(self.beta_2)
                .add(grad_sq_for_moment.mul_scalar(factor));

            state.time += 1;
            state
        } else {
            let factor = 1.0 - self.beta_1;
            let moment_1 = grad.clone().mul_scalar(factor);

            let factor = 1.0 - self.beta_2;
            let moment_2 = grad_sq_for_moment.mul_scalar(factor);

            MomentumState {
                moment_1,
                moment_2,
                time: 1,
            }
        };

        let time = (state.time as i32).elem();
        let moment_1_corrected = state
            .moment_1
            .clone()
            .div_scalar(1f32 - self.beta_1.powi(time));
        let moment_2_corrected = state
            .moment_2
            .clone()
            .div_scalar(1f32 - self.beta_2.powi(time));
        // moment_2_corrected broadcasts when it has reduced trailing dims.
        let grad = moment_1_corrected.div(moment_2_corrected.sqrt().add_scalar(self.epsilon));
        (grad, state)
    }
}

/// Reduce to a single mean per row by averaging across all trailing dims (1..D).
/// Result has size 1 in each trailing dim so it broadcasts back to the full shape.
fn mean_trailing_dims<const D: usize>(t: Tensor<D>) -> Tensor<D> {
    debug_assert!(D > 1, "mean_trailing_dims requires D > 1");
    let shape = t.dims();
    let trailing_count: usize = shape[1..].iter().product();

    // Reduce over the trailing dims directly; a flatten + reshape would end
    // burn's fusion block.
    let dims: Vec<usize> = (1..D).collect();
    t.sum_dims(&dims) / trailing_count as f32
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Device, TensorData};

    use super::*;

    async fn assert_close<const D: usize>(actual: Tensor<D>, expected: Tensor<D>) {
        assert_eq!(actual.dims(), expected.dims());
        let actual: Vec<f32> = actual
            .into_data_async()
            .await
            .expect("actual readback")
            .try_to_vec()
            .expect("actual f32 data");
        let expected: Vec<f32> = expected
            .into_data_async()
            .await
            .expect("expected readback")
            .try_to_vec()
            .expect("expected f32 data");
        for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
            let tolerance = 1e-7 + expected.abs() * 2e-5;
            assert!(
                (actual - expected).abs() <= tolerance,
                "element {index}: {actual} != {expected} (tolerance {tolerance})"
            );
        }
    }

    #[tokio::test]
    async fn compact_second_moment_matches_dense_for_fresh_and_resumed_adam() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let adam = AdamScaled::new(1e-15);
        for coeffs in [1, 4, 9, 16, 25] {
            let shape = [3, coeffs, 3];
            let row_len = coeffs * 3;
            for initial_time in [0, 73] {
                let mut expected_state = (initial_time > 0).then(|| MomentumState {
                    moment_1: Tensor::full(shape, 0.02, &device),
                    moment_2: Tensor::full([3, 1, 1], 0.003, &device),
                    time: initial_time,
                });
                let mut actual_state = expected_state.clone();
                for step in 0..2 {
                    // Change which row is invisible between updates. Its
                    // gradient is zero but resumed momentum must still decay.
                    let gradients: Vec<f32> = (0..3 * row_len)
                        .map(|index| {
                            if index / row_len == step {
                                0.0
                            } else {
                                ((index * 7 + step * 3) % 29) as f32 * 0.03 - 0.42
                            }
                        })
                        .collect();
                    let reduced: Vec<f32> = gradients
                        .chunks_exact(row_len)
                        .map(|row| {
                            row.iter().map(|value| value * value).sum::<f32>() / row_len as f32
                        })
                        .collect();
                    let grad = Tensor::from_data(TensorData::new(gradients, shape), &device);
                    let reduced = Tensor::from_data(TensorData::new(reduced, [3, 1, 1]), &device);

                    // Exercise the generic update directly even when the
                    // machine enables the fork's fused native Adam by default.
                    let (expected_update, expected) =
                        adam.transform(&grad, None, expected_state.take(), true);
                    let (actual_update, actual) =
                        adam.transform(&grad, Some(reduced), actual_state.take(), true);
                    assert_eq!(actual.time, initial_time + step + 1);
                    assert_eq!(actual.moment_2.dims(), [3, 1, 1]);
                    assert_close(actual_update, expected_update).await;
                    assert_close(actual.moment_1.clone(), expected.moment_1.clone()).await;
                    assert_close(actual.moment_2.clone(), expected.moment_2.clone()).await;
                    expected_state = Some(expected);
                    actual_state = Some(actual);
                }
            }
        }
    }

    #[tokio::test]
    async fn unreduced_adam_ignores_precomputed_second_moment() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let adam = AdamScaled::new(1e-15);
        let grad = Tensor::<2>::from_floats([[0.2, -0.3, 0.5], [0.0, 0.6, -0.7]], &device);
        let param = Tensor::full([2, 3], 1.0, &device);
        let mut expected = AdamState::new(None, false);
        let mut actual = expected.clone();
        for _ in 0..2 {
            let expected_param = adam.step(0.01, param.clone(), &grad, None, &mut expected);
            let actual_param = adam.step(
                0.01,
                param.clone(),
                &grad,
                Some(Tensor::full([2, 1], f32::NAN, &device)),
                &mut actual,
            );
            assert_close(actual_param, expected_param).await;
            assert_close(
                actual.momentum.as_ref().unwrap().moment_2.clone(),
                expected.momentum.as_ref().unwrap().moment_2.clone(),
            )
            .await;
        }
    }
}
