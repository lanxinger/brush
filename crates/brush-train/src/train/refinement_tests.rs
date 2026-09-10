use super::*;
use crate::adam_scaled::MomentumState;
use brush_render::gaussian_splats::SplatRenderMode;

async fn values<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor
        .into_data_async()
        .await
        .expect("readback")
        .try_into_vec()
        .expect("f32 tensor")
}

fn seeded_state<const D: usize>(
    shape: [usize; D],
    second_shape: [usize; D],
    device: &Device,
) -> AdamState<D> {
    let moment = |shape: [usize; D], split_value: f32| {
        let len = shape.iter().product::<usize>();
        let row_len = len / 3;
        let data = (0..len)
            .map(|index| [3.0, split_value, 7.0][index / row_len])
            .collect::<Vec<_>>();
        Tensor::from_data(TensorData::new(data, shape), device)
    };
    AdamState {
        momentum: Some(MomentumState {
            // Assignment must clear these; adding the negated value leaves NaN.
            moment_1: moment(shape, f32::INFINITY),
            moment_2: moment(second_shape, f32::NAN),
            time: 9,
        }),
        scaling: None,
        reduce_moment_2: shape != second_shape,
    }
}

async fn assert_reset<const D: usize>(
    state: AdamState<D>,
    shape: [usize; D],
    second_shape: [usize; D],
) {
    let momentum = state.momentum.expect("initialized momentum");
    assert_eq!(momentum.time, 9);
    assert_eq!(momentum.moment_1.dims(), shape);
    assert_eq!(momentum.moment_2.dims(), second_shape);
    for tensor in [momentum.moment_1, momentum.moment_2] {
        let data = values(tensor).await;
        for (row, actual) in data.chunks_exact(data.len() / 4).enumerate() {
            let expected = [3.0, 0.0, 7.0, 0.0][row];
            assert!(
                actual.iter().all(|&value| value == expected),
                "momentum row {row}: {actual:?}, expected {expected}"
            );
        }
    }
}

#[tokio::test]
async fn split_preserves_geometry_and_resets_nonfinite_momentum() {
    let device = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let opt_device = device.clone().inner();

    // Cover ordinary covariance splitting and the fork's screen-size cap.
    for screen_cap in [0.0, 0.25] {
        let config = TrainConfig {
            split_at_screen_size: screen_cap,
            opac_decay: 0.0,
            ..Default::default()
        };
        let mut trainer = SplatTrainer::new(
            &config,
            &device,
            BoundingBox::from_min_max(glam::Vec3::ZERO, glam::Vec3::splat(10.0)),
        );
        let splats = Splats::from_raw(
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 2.0f32.ln(), 0.0, 0.5f32.ln(), 0.0, 0.0, 0.0],
            (0..36).map(|index| index as f32 / 10.0).collect(),
            vec![0.25, 0.8, -0.25],
            SplatRenderMode::Default,
            &device,
        );
        let original = values(splats.transforms.val()).await;
        let original_sh = values(splats.sh_coeffs.val()).await;
        let optim = SplatOptim {
            adam: AdamScaled::new(1e-15),
            transforms: seeded_state([3, 10], [3, 10], &opt_device),
            sh_coeffs: seeded_state([3, 4, 3], [3, 1, 1], &opt_device),
            opacities: seeded_state([3], [3], &opt_device),
        };
        let split = trainer.refine_splats(
            &device,
            optim,
            splats,
            HashSet::from_iter([1]),
            Tensor::from_floats([0.1, 1.0, 0.1], &device),
            0,
            100,
        );

        assert_eq!(split.num_splats(), 4);
        let transforms = values(split.transforms.val()).await;
        assert_eq!(&transforms[..10], &original[..10]);
        assert_eq!(&transforms[20..30], &original[20..30]);
        assert_eq!(&transforms[13..17], &[2.0, 0.0, 0.0, 0.0]);
        assert_eq!(&transforms[33..37], &[1.0, 0.0, 0.0, 0.0]);

        let max_shrink = if screen_cap == 0.0 {
            FRAC_1_SQRT_2
        } else {
            screen_cap
        };
        for (axis, scale) in [2.0f32, 1.0, 0.5].into_iter().enumerate() {
            let shrink = 1.0 - scale.powi(2) / 4.0 * (1.0 - max_shrink);
            let offset = scale * (1.0 - shrink.powi(2)).sqrt();
            let parent = transforms[10 + axis];
            let child = transforms[30 + axis];
            assert!(((parent + child) * 0.5 - original[10 + axis]).abs() < 2e-6);
            assert!(((child - parent) * 0.5 - offset).abs() < 2e-6);
            for row in [1, 3] {
                assert!((transforms[row * 10 + 7 + axis].exp() - scale * shrink).abs() < 2e-6);
            }
        }

        let sh = values(split.sh_coeffs.val()).await;
        assert_eq!(&sh[..36], &original_sh);
        assert_eq!(&sh[36..], &original_sh[12..24]);
        let opacity = values(split.opacities()).await;
        let sigmoid = |value: f32| 1.0 / (1.0 + (-value).exp());
        let child_opacity = 1.0 - (1.0 - sigmoid(0.8)).powf(FRAC_1_SQRT_2);
        for (actual, expected) in
            opacity
                .into_iter()
                .zip([sigmoid(0.25), child_opacity, sigmoid(-0.25), child_opacity])
        {
            assert!((actual - expected).abs() < 2e-6);
        }

        let optim = trainer.optim.take().expect("refined optimizer");
        assert_reset(optim.transforms, [4, 10], [4, 10]).await;
        assert_reset(optim.sh_coeffs, [4, 4, 3], [4, 1, 1]).await;
        assert_reset(optim.opacities, [4], [4]).await;
    }
}
