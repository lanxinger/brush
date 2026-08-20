#[cfg(not(target_family = "wasm"))]
use std::path::Path;

use anyhow::Result;
use brush_dataset::scene::view_to_packed_data;
use brush_loss::{ImageLossConfig, image_loss_eval};
use brush_render::camera::Camera;
use brush_render::gaussian_splats::Splats;
use brush_render::{AlphaMode, RenderAux, TextureMode, render_splats};
use burn::tensor::{Device, Int, Tensor, s};
use glam::Vec3;
use image::DynamicImage;

pub struct EvalSample {
    pub gt_img: DynamicImage,
    pub rendered: Tensor<3>,
    /// Legacy full-frame metric retained for backward compatibility.
    pub psnr: Tensor<1>,
    /// Legacy full-frame metric retained for backward compatibility.
    pub ssim: Tensor<1>,
    /// Alpha-mask coverage when masked evaluation was requested and the image
    /// has an alpha channel. A zero value means masked metrics were skipped.
    pub mask_coverage: Option<f32>,
    pub masked: Option<MaskedEvalMetrics>,
    pub render_aux: RenderAux,
}

pub struct MaskedEvalMetrics {
    pub psnr: Tensor<1>,
    /// SSIM windows still observe neighbouring RGB values outside the mask;
    /// alpha weights the window centres included in the reported mean.
    pub ssim: Tensor<1>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct MaskWeights {
    coverage: f32,
    mse_weight: f32,
}

pub async fn eval_stats(
    splats: Splats,
    gt_cam: &Camera,
    gt_img: DynamicImage,
    alpha_mode: AlphaMode,
    device: &Device,
    correction: Option<&(dyn Fn(Tensor<3>) -> Tensor<3> + Sync)>,
) -> Result<EvalSample> {
    let res = glam::uvec2(gt_img.width(), gt_img.height());

    let mask_weights = mask_weights(&gt_img, alpha_mode);
    let mask_coverage = mask_weights.map(|weights| weights.coverage);
    let (gt_packed_data, _has_alpha) = view_to_packed_data(gt_img.clone(), alpha_mode);
    let gt_packed: Tensor<2, Int> = Tensor::from_data(gt_packed_data, device);

    // Render on reference black background.
    let (img, render_aux) =
        render_splats(splats, gt_cam, res, Vec3::ZERO, None, TextureMode::Float).await;
    let render_rgb = img.slice(s![.., .., 0..3]);

    // Apply the learned per-view appearance correction when scoring a
    // training view (`--train-on-eval`): without it, scores on
    // appearance-varying datasets mostly measure the splat <-> average
    // appearance offset rather than reconstruction quality.
    let render_rgb = match correction {
        Some(f) => f(render_rgb),
        None => render_rgb,
    };

    // Simulate an 8-bit roundtrip for fair comparison.
    let render_rgb = (render_rgb * 255.0).round() / 255.0;

    let cfg = |l1, ssim, mask| ImageLossConfig {
        l1_weight: l1,
        ssim_weight: ssim,
        composite_bg: None,
        mask,
    };
    // MSE = mean(L1^2) since |a - b|^2 == (a - b)^2.
    let mse = image_loss_eval(render_rgb.clone(), gt_packed.clone(), cfg(1.0, 0.0, false))
        .powi_scalar(2)
        .mean();
    let psnr = mse.recip().log() * 10.0 / std::f32::consts::LN_10;
    let ssim = image_loss_eval(render_rgb.clone(), gt_packed.clone(), cfg(0.0, 1.0, false)).mean();

    let masked = mask_weights
        .filter(|weights| weights.mse_weight > 0.0)
        .map(|weights| {
            // The mask is applied to L1 before squaring, so normalizing MSE
            // requires mean(alpha^2), not mean(alpha).
            let mse = image_loss_eval(render_rgb.clone(), gt_packed.clone(), cfg(1.0, 0.0, true))
                .powi_scalar(2)
                .mean()
                / weights.mse_weight;
            let psnr = mse.recip().log() * 10.0 / std::f32::consts::LN_10;
            let ssim = image_loss_eval(render_rgb.clone(), gt_packed, cfg(0.0, 1.0, true)).mean()
                / weights.coverage;
            MaskedEvalMetrics { psnr, ssim }
        });

    Ok(EvalSample {
        gt_img,
        psnr,
        ssim,
        mask_coverage,
        masked,
        rendered: render_rgb,
        render_aux,
    })
}

fn mask_weights(image: &DynamicImage, alpha_mode: AlphaMode) -> Option<MaskWeights> {
    if alpha_mode != AlphaMode::Masked || !image.color().has_alpha() {
        return None;
    }

    let rgba = image.to_rgba8();
    let count = rgba.width() as f64 * rgba.height() as f64;
    if count == 0.0 {
        return Some(MaskWeights {
            coverage: 0.0,
            mse_weight: 0.0,
        });
    }
    let (sum, sum_squared) = rgba.pixels().fold((0.0, 0.0), |(sum, sum_squared), pixel| {
        let alpha = f64::from(pixel[3]) / 255.0;
        (sum + alpha, sum_squared + alpha * alpha)
    });

    Some(MaskWeights {
        coverage: (sum / count) as f32,
        mse_weight: (sum_squared / count) as f32,
    })
}

impl EvalSample {
    #[cfg(not(target_family = "wasm"))]
    pub async fn save_to_disk(&self, path: &Path) -> anyhow::Result<()> {
        use image::Rgb32FImage;
        log::info!("Saving eval image to disk.");
        let img = self.rendered.clone();
        let [h, w, _] = [img.dims()[0], img.dims()[1], img.dims()[2]];
        let data = img.clone().into_data_async().await?.try_into_vec::<f32>()?;
        let img: image::DynamicImage = Rgb32FImage::from_raw(w as u32, h as u32, data)
            .expect("Failed to create image from tensor")
            .into();
        let img: image::DynamicImage = img.into_rgb8().into();
        let parent = path.parent().expect("Eval must have a filename");
        tokio::fs::create_dir_all(parent).await?;
        log::info!("Saving eval view to {path:?}");
        img.save(path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgba, RgbaImage};

    fn image_with_alpha(alpha: &[u8]) -> DynamicImage {
        let mut image = RgbaImage::new(alpha.len() as u32, 1);
        for (pixel, alpha) in image.pixels_mut().zip(alpha.iter().copied()) {
            *pixel = Rgba([10, 20, 30, alpha]);
        }
        DynamicImage::ImageRgba8(image)
    }

    #[test]
    fn binary_mask_uses_the_same_mean_for_mse_and_coverage() {
        let weights = mask_weights(&image_with_alpha(&[255, 0]), AlphaMode::Masked).unwrap();

        assert_eq!(weights.coverage, 0.5);
        assert_eq!(weights.mse_weight, 0.5);
    }

    #[test]
    fn soft_mask_normalizes_squared_error_by_alpha_squared() {
        let weights = mask_weights(&image_with_alpha(&[128, 64]), AlphaMode::Masked).unwrap();
        let first = 128.0_f32 / 255.0;
        let second = 64.0_f32 / 255.0;

        assert!((weights.coverage - (first + second) / 2.0).abs() < 1e-6);
        assert!((weights.mse_weight - (first * first + second * second) / 2.0).abs() < 1e-6);
    }

    #[test]
    fn empty_mask_has_zero_weights_and_transparent_mode_is_not_a_mask() {
        let image = image_with_alpha(&[0, 0]);

        assert_eq!(
            mask_weights(&image, AlphaMode::Masked),
            Some(MaskWeights {
                coverage: 0.0,
                mse_weight: 0.0,
            })
        );
        assert_eq!(mask_weights(&image, AlphaMode::Transparent), None);
    }
}
