//! Native-MSL fused Adam update for spherical-harmonic coefficients.
//!
//! Each Apple 32-lane SIMD group owns one splat. It reduces the full SH
//! gradient row to the scalar second moment, then updates the full first
//! moment and parameter row without materialising intermediate tensors.

use brush_cube::{MainBackend as Wgpu, MainBackendBase, calc_cube_count_1d};
use burn::{
    Tensor,
    backend::{
        Backend, Dispatch, ExtensionType, TensorMetadata, tensor::FloatTensor, wgpu::WgpuRuntime,
    },
    tensor::{DType, Shape},
};
use burn_cubecl::{
    CubeRuntime, cubecl::features::Plane, fusion::FusionCubeRuntime, kernel::into_contiguous,
    tensor::CubeTensor,
};
use burn_fusion::{
    Fusion, FusionHandle,
    stream::{Operation, StreamId},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};

const PLANE_SIZE: u32 = 32;
const WORKGROUP_SIZE: u32 = 256;
const SPLATS_PER_WORKGROUP: u32 = WORKGROUP_SIZE / PLANE_SIZE;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ShAdamConfig {
    pub beta_1: f32,
    pub beta_2: f32,
    pub bias_correction_1: f32,
    pub bias_correction_2: f32,
    pub epsilon: f32,
    pub learning_rate: f32,
}

#[derive(Debug, Clone, ExtensionType)]
pub(crate) struct ShAdamOutput<B: Backend> {
    pub param: FloatTensor<B>,
    pub moment_1: FloatTensor<B>,
    pub moment_2: FloatTensor<B>,
}

#[burn::backend::backend_extension(Wgpu)]
pub(crate) trait ShAdamOps: Backend {
    fn sh_adam(
        param: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        moment_1: FloatTensor<Self>,
        moment_2: FloatTensor<Self>,
        scaling: FloatTensor<Self>,
        config: ShAdamConfig,
    ) -> ShAdamOutput<Self>;
}

pub(crate) fn sh_adam(
    param: Tensor<3>,
    grad: Tensor<3>,
    moment_1: Tensor<3>,
    moment_2: Tensor<3>,
    scaling: Tensor<3>,
    config: ShAdamConfig,
) -> (Tensor<3>, Tensor<3>, Tensor<3>) {
    let output = <Dispatch as ShAdamOps>::sh_adam(
        param.into_dispatch(),
        grad.into_dispatch(),
        moment_1.into_dispatch(),
        moment_2.into_dispatch(),
        scaling.into_dispatch(),
        config,
    );
    (
        Tensor::from_dispatch(output.param),
        Tensor::from_dispatch(output.moment_1),
        Tensor::from_dispatch(output.moment_2),
    )
}

mod kernel {
    use burn_cubecl::cubecl;
    use burn_cubecl::cubecl::cube;
    use burn_cubecl::cubecl::prelude::*;

    use super::{PLANE_SIZE, SPLATS_PER_WORKGROUP};

    #[allow(clippy::too_many_arguments)]
    #[cube]
    fn update_element(
        param: &Tensor<f32>,
        moment_1: &Tensor<f32>,
        scaling: &Tensor<f32>,
        out_param: &mut Tensor<f32>,
        out_moment_1: &mut Tensor<f32>,
        row_base: u32,
        element: u32,
        grad: f32,
        update_factor: f32,
        beta_1: f32,
    ) {
        let index = (row_base + element) as usize;
        let new_moment_1 = moment_1[index] * beta_1 + grad * (1.0f32 - beta_1);
        let coeff_scale = scaling[(element / 3u32) as usize];
        out_moment_1[index] = new_moment_1;
        out_param[index] = param[index] - new_moment_1 * update_factor * coeff_scale;
    }

    #[allow(clippy::too_many_arguments)]
    #[cube(launch, launch_unchecked)]
    pub fn sh_adam_kernel(
        param: &Tensor<f32>,
        grad: &Tensor<f32>,
        moment_1: &Tensor<f32>,
        moment_2: &Tensor<f32>,
        scaling: &Tensor<f32>,
        out_param: &mut Tensor<f32>,
        out_moment_1: &mut Tensor<f32>,
        out_moment_2: &mut Tensor<f32>,
        num_splats: u32,
        beta_1: f32,
        beta_2: f32,
        bias_correction_1: f32,
        bias_correction_2: f32,
        epsilon: f32,
        learning_rate: f32,
        #[comptime] row_len: u32,
    ) {
        let splat_id = CUBE_POS as u32 * SPLATS_PER_WORKGROUP + PLANE_POS;
        let active = splat_id < num_splats;
        let lane = UNIT_POS_PLANE;
        let row_base = splat_id * row_len;

        // Degree 4 has 75 values per splat, so each lane owns at most three.
        // Keep them in registers across the plane reduction to avoid rereading
        // the 222 MB gradient tensor during the update phase.
        let index_0 = lane;
        let index_1 = lane + PLANE_SIZE;
        let index_2 = lane + 2u32 * PLANE_SIZE;
        let mut grad_0 = 0.0f32;
        let mut grad_1 = 0.0f32;
        let mut grad_2 = 0.0f32;
        if active && index_0 < row_len {
            grad_0 = grad[(row_base + index_0) as usize];
        }
        if active && index_1 < row_len {
            grad_1 = grad[(row_base + index_1) as usize];
        }
        if active && index_2 < row_len {
            grad_2 = grad[(row_base + index_2) as usize];
        }

        let sum_sq = plane_sum(grad_0 * grad_0 + grad_1 * grad_1 + grad_2 * grad_2);
        let mean_sq = sum_sq / row_len as f32;

        let mut update_factor = 0.0f32;
        if active && lane == 0u32 {
            let new_moment_2 = moment_2[splat_id as usize] * beta_2 + mean_sq * (1.0f32 - beta_2);
            out_moment_2[splat_id as usize] = new_moment_2;
            update_factor = learning_rate
                / bias_correction_1
                / (f32::sqrt(new_moment_2 / bias_correction_2) + epsilon);
        }
        update_factor = plane_broadcast(update_factor, 0u32);

        if active && index_0 < row_len {
            update_element(
                param,
                moment_1,
                scaling,
                out_param,
                out_moment_1,
                row_base,
                index_0,
                grad_0,
                update_factor,
                beta_1,
            );
        }
        if active && index_1 < row_len {
            update_element(
                param,
                moment_1,
                scaling,
                out_param,
                out_moment_1,
                row_base,
                index_1,
                grad_1,
                update_factor,
                beta_1,
            );
        }
        if active && index_2 < row_len {
            update_element(
                param,
                moment_1,
                scaling,
                out_param,
                out_moment_1,
                row_base,
                index_2,
                grad_2,
                update_factor,
                beta_1,
            );
        }
    }
}

fn empty_like<R: CubeRuntime>(template: &CubeTensor<R>) -> CubeTensor<R> {
    let shape = Shape::from(template.shape().as_slice().to_vec());
    let buffer = template
        .client
        .empty(shape.num_elements() * template.dtype.size());
    CubeTensor::new_contiguous(
        template.client.clone(),
        template.device.clone(),
        shape,
        buffer,
        template.dtype,
    )
}

impl ShAdamOps for MainBackendBase {
    fn sh_adam(
        param: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        moment_1: FloatTensor<Self>,
        moment_2: FloatTensor<Self>,
        scaling: FloatTensor<Self>,
        config: ShAdamConfig,
    ) -> ShAdamOutput<Self> {
        param.assert_is_on_same_device(&grad);
        param.assert_is_on_same_device(&moment_1);
        param.assert_is_on_same_device(&moment_2);
        param.assert_is_on_same_device(&scaling);

        let param = into_contiguous(param);
        let grad = into_contiguous(grad);
        let moment_1 = into_contiguous(moment_1);
        let moment_2 = into_contiguous(moment_2);
        let scaling = into_contiguous(scaling);

        for (name, tensor) in [
            ("parameter", &param),
            ("gradient", &grad),
            ("moment_1", &moment_1),
            ("moment_2", &moment_2),
            ("scaling", &scaling),
        ] {
            assert_eq!(tensor.dtype, DType::F32, "fused SH Adam {name} must be f32");
        }

        let shape = param.shape();
        let dims = shape.as_slice();
        assert_eq!(dims.len(), 3, "fused SH Adam expects [N, C, 3]");
        assert_eq!(dims[2], 3, "fused SH Adam expects RGB coefficients");
        assert_eq!(grad.shape(), shape, "gradient shape must match parameter");
        assert_eq!(
            moment_1.shape(),
            shape,
            "moment_1 shape must match parameter"
        );
        assert_eq!(
            moment_2.shape().as_slice(),
            &[dims[0], 1, 1],
            "moment_2 must be scalar per splat"
        );
        assert_eq!(
            scaling.shape().as_slice(),
            &[1, dims[1], 1],
            "scaling must be one value per SH coefficient"
        );
        assert!(dims[0] > 0, "fused SH Adam requires at least one splat");
        assert!(dims[1] <= 25, "fused SH Adam supports degrees 0 through 4");
        assert!(
            u32::try_from(shape.num_elements()).is_ok(),
            "fused SH Adam flattened indices must fit in u32"
        );

        let properties = param.client.properties();
        assert!(
            param.client.features().plane.contains(Plane::Ops),
            "fused SH Adam requires plane operations"
        );
        assert_eq!(
            properties.hardware.plane_size_min, PLANE_SIZE,
            "fused SH Adam requires 32-lane planes"
        );
        assert_eq!(
            properties.hardware.plane_size_max, PLANE_SIZE,
            "fused SH Adam requires a fixed plane size"
        );
        assert!(
            properties.hardware.max_units_per_cube >= WORKGROUP_SIZE,
            "fused SH Adam requires 256 units per workgroup"
        );
        assert!(
            properties.hardware.max_cube_dim.0 >= WORKGROUP_SIZE,
            "fused SH Adam requires a 256-wide workgroup"
        );

        let num_splats = u32::try_from(dims[0]).expect("splat count exceeds u32");
        let row_len = u32::try_from(dims[1] * dims[2]).expect("SH row exceeds u32");
        let workgroups = calc_cube_count_1d(num_splats, SPLATS_PER_WORKGROUP);
        let out_param = empty_like(&param);
        let out_moment_1 = empty_like(&moment_1);
        let out_moment_2 = empty_like(&moment_2);
        let client = param.client.clone();

        // SAFETY: all inputs are contiguous and shape-checked above. Every
        // active plane owns one row, its row indices are below N*C*3, its
        // scaling index is below C, and lane zero alone writes moment_2[N].
        unsafe {
            kernel::sh_adam_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                workgroups,
                burn_cubecl::cubecl::CubeDim::new_1d(WORKGROUP_SIZE),
                param.into_tensor_arg(),
                grad.into_tensor_arg(),
                moment_1.into_tensor_arg(),
                moment_2.into_tensor_arg(),
                scaling.into_tensor_arg(),
                out_param.clone().into_tensor_arg(),
                out_moment_1.clone().into_tensor_arg(),
                out_moment_2.clone().into_tensor_arg(),
                num_splats,
                config.beta_1,
                config.beta_2,
                config.bias_correction_1,
                config.bias_correction_2,
                config.epsilon,
                config.learning_rate,
                row_len,
            );
        }

        ShAdamOutput {
            param: out_param,
            moment_1: out_moment_1,
            moment_2: out_moment_2,
        }
    }
}

#[derive(Debug)]
struct ShAdamFusionOp {
    desc: CustomOpIr,
    config: ShAdamConfig,
}

impl Operation<FusionCubeRuntime<WgpuRuntime>> for ShAdamFusionOp {
    fn execute(&self, handles: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>) {
        let ([param, grad, moment_1, moment_2, scaling], [out_param, out_moment_1, out_moment_2]) =
            self.desc.as_fixed();
        let output = <MainBackendBase as ShAdamOps>::sh_adam(
            handles.get_float_tensor::<MainBackendBase>(param),
            handles.get_float_tensor::<MainBackendBase>(grad),
            handles.get_float_tensor::<MainBackendBase>(moment_1),
            handles.get_float_tensor::<MainBackendBase>(moment_2),
            handles.get_float_tensor::<MainBackendBase>(scaling),
            self.config,
        );
        handles.register_float_tensor::<MainBackendBase>(&out_param.id, output.param);
        handles.register_float_tensor::<MainBackendBase>(&out_moment_1.id, output.moment_1);
        handles.register_float_tensor::<MainBackendBase>(&out_moment_2.id, output.moment_2);
    }
}

impl ShAdamOps for Fusion<MainBackendBase> {
    fn sh_adam(
        param: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        moment_1: FloatTensor<Self>,
        moment_2: FloatTensor<Self>,
        scaling: FloatTensor<Self>,
        config: ShAdamConfig,
    ) -> ShAdamOutput<Self> {
        let client = param.client.clone();
        let out_param = TensorIr::uninit(client.create_empty_handle(), param.shape(), DType::F32);
        let out_moment_1 =
            TensorIr::uninit(client.create_empty_handle(), moment_1.shape(), DType::F32);
        let out_moment_2 =
            TensorIr::uninit(client.create_empty_handle(), moment_2.shape(), DType::F32);
        let desc = CustomOpIr::new(
            "fused_sh_adam",
            &[
                param.into_ir(),
                grad.into_ir(),
                moment_1.into_ir(),
                moment_2.into_ir(),
                scaling.into_ir(),
            ],
            &[out_param, out_moment_1, out_moment_2],
        );
        let operation = ShAdamFusionOp {
            desc: desc.clone(),
            config,
        };
        let [param, moment_1, moment_2] = client
            .register(StreamId::current(), OperationIr::Custom(desc), operation)
            .outputs();
        ShAdamOutput {
            param,
            moment_1,
            moment_2,
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Device, TensorData};

    use super::*;

    fn reference_update(
        param: Tensor<3>,
        grad: Tensor<3>,
        moment_1: Tensor<3>,
        moment_2: Tensor<3>,
        scaling: Tensor<3>,
        config: ShAdamConfig,
    ) -> (Tensor<3>, Tensor<3>, Tensor<3>) {
        let [num_splats, coeffs, channels] = param.dims();
        let row_len = coeffs * channels;
        let new_moment_1 =
            moment_1.mul_scalar(config.beta_1) + grad.clone().mul_scalar(1.0 - config.beta_1);
        let grad_sq_flat: Tensor<2> = grad.powi_scalar(2).flatten(1, 2);
        let mean_grad_sq = grad_sq_flat
            .sum_dim(1)
            .div_scalar(row_len as f32)
            .reshape([num_splats, 1, 1]);
        let new_moment_2 =
            moment_2.mul_scalar(config.beta_2) + mean_grad_sq.mul_scalar(1.0 - config.beta_2);
        let normalized = new_moment_1
            .clone()
            .div_scalar(config.bias_correction_1)
            .div(
                new_moment_2
                    .clone()
                    .div_scalar(config.bias_correction_2)
                    .sqrt()
                    .add_scalar(config.epsilon),
            );
        let out_param = param - normalized * scaling.mul_scalar(config.learning_rate);
        (out_param, new_moment_1, new_moment_2)
    }

    fn patterned_values(len: usize, multiplier: usize, modulus: usize, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let centered = (index * multiplier + 3) % modulus;
                (centered as f32 - modulus as f32 * 0.5) * scale
            })
            .collect()
    }

    fn assert_close(label: &str, actual: &[f32], expected: &[f32], relative: f32, absolute: f32) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = absolute + relative * expected.abs();
            assert!(
                (actual - expected).abs() <= tolerance,
                "{label}[{index}]: actual={actual:e}, expected={expected:e}, diff={:e}, tolerance={tolerance:e}",
                (actual - expected).abs(),
            );
        }
    }

    async fn compare_case(num_splats: usize, coeffs: usize, time: i32) {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let num_values = num_splats * coeffs * 3;
        let param_values = patterned_values(num_values, 17, 101, 0.002);
        let mut grad_values = patterned_values(num_values, 13, 67, 0.0007);
        let moment_1_values = patterned_values(num_values, 19, 79, 0.0002);
        let moment_2_values: Vec<f32> = (0..num_splats)
            .map(|row| 0.0001 + (row % 11) as f32 * 0.00003)
            .collect();
        let scaling_values: Vec<f32> = (0..coeffs)
            .map(|coeff| {
                if coeff == 0 {
                    1.0
                } else {
                    0.035 + (coeff % 7) as f32 * 0.017
                }
            })
            .collect();

        // Exercise zero-gradient rows with non-zero momentum. They must still
        // receive the decaying first-moment update.
        for row in (0..num_splats).step_by(4) {
            grad_values[row * coeffs * 3..(row + 1) * coeffs * 3].fill(0.0);
        }

        let config = ShAdamConfig {
            beta_1: 0.87,
            beta_2: 0.996,
            bias_correction_1: 1.0 - 0.87f32.powi(time),
            bias_correction_2: 1.0 - 0.996f32.powi(time),
            epsilon: 1e-15,
            learning_rate: 0.0023,
        };
        let param = Tensor::from_data(
            TensorData::new(param_values.clone(), [num_splats, coeffs, 3]),
            &device,
        );
        let grad = Tensor::from_data(
            TensorData::new(grad_values, [num_splats, coeffs, 3]),
            &device,
        );
        let moment_1 = Tensor::from_data(
            TensorData::new(moment_1_values, [num_splats, coeffs, 3]),
            &device,
        );
        let moment_2 = Tensor::from_data(
            TensorData::new(moment_2_values, [num_splats, 1, 1]),
            &device,
        );
        let scaling = Tensor::from_data(TensorData::new(scaling_values, [1, coeffs, 1]), &device);

        let expected = reference_update(
            param.clone(),
            grad.clone(),
            moment_1.clone(),
            moment_2.clone(),
            scaling.clone(),
            config,
        );
        let actual = sh_adam(param, grad, moment_1, moment_2, scaling, config);
        let expected_param: Vec<f32> = expected
            .0
            .into_data_async()
            .await
            .expect("reference parameter readback")
            .to_vec()
            .expect("reference parameter type");
        let expected_moment_1: Vec<f32> = expected
            .1
            .into_data_async()
            .await
            .expect("reference moment_1 readback")
            .to_vec()
            .expect("reference moment_1 type");
        let expected_moment_2: Vec<f32> = expected
            .2
            .into_data_async()
            .await
            .expect("reference moment_2 readback")
            .to_vec()
            .expect("reference moment_2 type");
        let actual_param: Vec<f32> = actual
            .0
            .into_data_async()
            .await
            .expect("fused parameter readback")
            .to_vec()
            .expect("fused parameter type");
        let actual_moment_1: Vec<f32> = actual
            .1
            .into_data_async()
            .await
            .expect("fused moment_1 readback")
            .to_vec()
            .expect("fused moment_1 type");
        let actual_moment_2: Vec<f32> = actual
            .2
            .into_data_async()
            .await
            .expect("fused moment_2 readback")
            .to_vec()
            .expect("fused moment_2 type");

        let case = format!("N={num_splats}, C={coeffs}, time={time}");
        assert_close(
            &format!("{case} parameter"),
            &actual_param,
            &expected_param,
            2e-5,
            2e-7,
        );
        assert_close(
            &format!("{case} moment_1"),
            &actual_moment_1,
            &expected_moment_1,
            2e-5,
            2e-7,
        );
        assert_close(
            &format!("{case} moment_2"),
            &actual_moment_2,
            &expected_moment_2,
            5e-5,
            1e-12,
        );
        let actual_delta: Vec<f32> = param_values
            .iter()
            .zip(&actual_param)
            .map(|(before, after)| before - after)
            .collect();
        let expected_delta: Vec<f32> = param_values
            .iter()
            .zip(&expected_param)
            .map(|(before, after)| before - after)
            .collect();
        assert_close(
            &format!("{case} update delta"),
            &actual_delta,
            &expected_delta,
            2e-5,
            1e-8,
        );
    }

    #[tokio::test]
    async fn fused_matches_generic_for_all_sh_degrees_and_times() {
        for coeffs in [1, 4, 9, 16, 25] {
            for time in [2, 200, 15_000] {
                compare_case(9, coeffs, time).await;
            }
        }
    }

    #[tokio::test]
    async fn fused_updates_partial_workgroups() {
        for num_splats in [1, 7, 8, 9, 15, 16, 17] {
            compare_case(num_splats, 16, 37).await;
        }
    }

    #[tokio::test]
    #[ignore = "large native-Metal launch geometry soak"]
    async fn fused_crosses_2d_dispatch_boundary() {
        compare_case(524_281, 1, 15_000).await;
    }
}
