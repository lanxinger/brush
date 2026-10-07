//! Projection regressions for packed passes and uniform viewer scaling.

use super::{Scene, scene_to_splats};
use crate::{
    RenderOutput, SplatRasterizerOps,
    camera::Camera,
    gaussian_splats::{
        RasterPass, Rasterizer, SplatRenderMode, TextureMode, render_splats_with_rasterizer,
    },
    kernels::camera_model::CameraModel,
};
use burn::{
    backend::Dispatch,
    module::Param,
    tensor::{Int, Tensor},
};
use glam::Vec3;
use wasm_bindgen_test::wasm_bindgen_test;

fn camera() -> Camera {
    Camera::new(
        Vec3::ZERO,
        glam::Quat::IDENTITY,
        0.5,
        0.5,
        glam::vec2(0.5, 0.5),
        CameraModel::Pinhole,
    )
}

async fn render_primitives(
    scene: &Scene,
    device: &burn::tensor::Device,
    pass: RasterPass,
    rasterizer: Rasterizer,
) -> RenderOutput<Dispatch> {
    let splats = scene_to_splats(scene, device);
    let (min_scale, has_min_scale) = splats.min_scale_arg();
    <Dispatch as SplatRasterizerOps>::render_with_rasterizer(
        &camera(),
        glam::uvec2(33, 25),
        splats.transforms.val().into_dispatch(),
        splats.sh_coeffs.val().into_dispatch(),
        splats.raw_opacities.val().into_dispatch(),
        min_scale.into_dispatch(),
        has_min_scale,
        1.0,
        SplatRenderMode::Default,
        Vec3::new(0.13, 0.07, 0.19),
        pass,
        rasterizer,
    )
    .await
}

/// IDs cross multiple projection workgroups and greatly exceed the packed
/// inverse-map placeholder. Culled rows must still be zero in the full map.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn packed_projection_omits_inverse_map_and_backward_keeps_global_rows() {
    let device = burn::tensor::Device::from(brush_cube::test_helpers::test_device().await);
    let n = 4097;
    let mut scene = Scene {
        means: vec![[0.0, 0.0, -1.0]; n],
        quats: vec![glam::Quat::IDENTITY.to_array(); n],
        log_scales: vec![[-2.5, -2.3, -2.0]; n],
        sh_dc: vec![[0.5, 0.2, 0.1]; n],
        raw_opacity: vec![2.0; n],
    };
    for rasterizer in [Rasterizer::Legacy, Rasterizer::Candidate] {
        for visible in [true, false] {
            scene.means[17] = [0.0, 0.0, if visible { 3.0 } else { -1.0 }];
            scene.means[n - 1] = [0.1, 0.0, if visible { 4.0 } else { -1.0 }];
            let packed = render_primitives(&scene, &device, RasterPass::Forward, rasterizer).await;
            let full = render_primitives(&scene, &device, RasterPass::Backward, rasterizer).await;
            let packed_map = Tensor::<1, Int>::from_dispatch(packed.compact_from_global);
            let full_map = Tensor::<1, Int>::from_dispatch(full.compact_from_global);
            assert_eq!(packed_map.dims(), [1]);
            assert_eq!(full_map.dims(), [n]);
            let expected_visible = if visible { 2 } else { 0 };
            assert_eq!(packed.aux.num_visible, expected_visible);
            assert_eq!(full.aux.num_visible, expected_visible);
            assert_eq!(packed.aux.num_intersections, full.aux.num_intersections);
            let mut expected_map = vec![0u32; n];
            if visible {
                expected_map[17] = 1;
                expected_map[n - 1] = 2;
            }
            assert_eq!(
                full_map
                    .into_data_async()
                    .await
                    .unwrap()
                    .try_to_vec::<u32>()
                    .unwrap(),
                expected_map,
            );
            if visible {
                let ids = Tensor::<1, Int>::from_dispatch(full.global_from_compact_gid)
                    .into_data_async()
                    .await
                    .unwrap()
                    .try_to_vec::<u32>()
                    .unwrap();
                assert_eq!(ids, [17, (n - 1) as u32]);
            }
            let actual = Tensor::<3>::from_dispatch(packed.out_img)
                .into_data_async()
                .await
                .unwrap()
                .try_to_vec::<f32>()
                .unwrap();
            let expected = Tensor::<3>::from_dispatch(full.out_img)
                .into_data_async()
                .await
                .unwrap()
                .try_to_vec::<f32>()
                .unwrap();
            for (pixel, rgba) in actual.iter().zip(expected.as_chunks::<4>().0) {
                let mut bits = 0u32;
                for (channel, value) in rgba.iter().enumerate() {
                    bits |= ((*value * 255.0).clamp(0.0, 255.0) as u32) << (channel * 8);
                }
                assert_eq!(pixel.to_bits(), bits, "packed projection changed the image");
            }
        }
    }
}

async fn read_floats<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor
        .into_data_async()
        .await
        .unwrap()
        .try_to_vec::<f32>()
        .unwrap()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() < 1e-5,
            "uniform scaling differs from materialized scaling: {a} vs {b}"
        );
    }
}

/// Reproduce the previous host tensor transformations independently, including
/// the floor, and compare images and public projection auxiliaries.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn viewer_scale_uniform_matches_materialized_tensors() {
    let device = burn::tensor::Device::from(brush_cube::test_helpers::test_device().await);
    let scene = Scene {
        means: vec![[0.0, 0.0, -1.0], [-0.1, 0.0, 3.0], [0.1, 0.0, 4.0]],
        quats: vec![glam::Quat::IDENTITY.to_array(); 3],
        log_scales: vec![[-3.0, -2.5, -2.0]; 3],
        sh_dc: vec![[0.5, 0.2, 0.1]; 3],
        raw_opacity: vec![2.0; 3],
    };
    for rasterizer in [Rasterizer::Legacy, Rasterizer::Candidate] {
        for mip in [false, true] {
            for floor in [false, true] {
                let mut splats = scene_to_splats(&scene, &device);
                splats.render_mip = mip;
                if floor {
                    splats = splats.with_min_scale(Tensor::from_floats([0.1, 0.12, 0.14], &device));
                }
                for scale in [None, Some(1.0f32), Some(0.37), Some(2.3)] {
                    let mut materialized = splats.clone();
                    if let Some(scale) = scale {
                        let transforms = materialized.transforms.val();
                        let adjusted = transforms.clone().slice([0..3, 7..10]) + scale.ln();
                        materialized.transforms =
                            Param::from_tensor(transforms.slice_assign([0..3, 7..10], adjusted));
                        materialized.min_scale = materialized.min_scale.map(|floor| floor * scale);
                    }
                    for texture_mode in [TextureMode::Float, TextureMode::Packed] {
                        let (actual, aux) = render_splats_with_rasterizer(
                            splats.clone(),
                            &camera(),
                            glam::uvec2(33, 25),
                            Vec3::ZERO,
                            scale,
                            texture_mode,
                            rasterizer,
                        )
                        .await;
                        let (expected, expected_aux) = render_splats_with_rasterizer(
                            materialized.clone(),
                            &camera(),
                            glam::uvec2(33, 25),
                            Vec3::ZERO,
                            None,
                            texture_mode,
                            rasterizer,
                        )
                        .await;
                        let actual = read_floats(actual).await;
                        let expected = read_floats(expected).await;
                        if matches!(texture_mode, TextureMode::Packed) {
                            assert_eq!(actual.len(), expected.len());
                            for (a, b) in actual.iter().zip(&expected) {
                                assert_eq!(
                                    a.to_bits(),
                                    b.to_bits(),
                                    "packed viewer scale mismatch"
                                );
                            }
                        } else {
                            assert_close(&actual, &expected);
                        }
                        assert_eq!(aux.num_visible, 2);
                        assert_eq!(aux.num_visible, expected_aux.num_visible);
                        assert_eq!(aux.num_intersections, expected_aux.num_intersections);
                        assert_close(
                            &read_floats(aux.max_radius).await,
                            &read_floats(expected_aux.max_radius).await,
                        );
                        assert_close(
                            &read_floats(aux.opacities).await,
                            &read_floats(expected_aux.opacities).await,
                        );
                        assert_eq!(
                            aux.tile_offsets.into_data_async().await.unwrap(),
                            expected_aux.tile_offsets.into_data_async().await.unwrap(),
                        );
                    }
                }
            }
        }
    }
}
