use std::io::Write;

use brush_render::gaussian_splats::Splats;
use brush_render::sh::sh_coeffs_for_degree;
use burn::tensor::{Transaction, s};
use glam::Vec3;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ExportError {
    #[error("Failed to fetch splat data from GPU")]
    FetchFailed,
    #[error("Failed to convert tensor data to f32 - data may be corrupted")]
    DataConversion,
    #[error("PLY serialization failed: {0}")]
    Serialize(#[from] serde_ply::SerializeError),
    #[error("Failed to write PLY: {0}")]
    Io(#[from] std::io::Error),
}

const SH_NAMES: [&str; 72] = brush_serde_macros::sh_field_names!();

/// Maximum splats per GPU readback. CPU staging stays bounded by one chunk;
/// a writer or the compatibility `Vec<u8>` API may still retain the full file.
pub const EXPORT_CHUNK_SPLATS: usize = 1 << 18;

/// Stream the splats as a binary little-endian PLY into `writer`, reading them
/// back from the GPU in chunks of [`EXPORT_CHUNK_SPLATS`]. The caller must
/// flush the writer after this returns successfully.
pub async fn splat_to_ply_writer(
    splats: Splats,
    up_axis: Option<Vec3>,
    mut writer: impl Write,
) -> Result<(), ExportError> {
    write_ply_chunked(splats, up_axis, &mut writer, EXPORT_CHUNK_SPLATS).await
}

pub async fn splat_to_ply(splats: Splats, up_axis: Option<Vec3>) -> Result<Vec<u8>, ExportError> {
    let mut buf = Vec::new();
    splat_to_ply_writer(splats, up_axis, &mut buf).await?;
    Ok(buf)
}

async fn write_ply_chunked(
    splats: Splats,
    up_axis: Option<Vec3>,
    writer: &mut impl Write,
    chunk_splats: usize,
) -> Result<(), ExportError> {
    let chunk_splats = chunk_splats.max(1);
    // Fold any 3D-filter floor into the stored scales/opacity so the ply holds
    // ordinary derived values — the floor is never written as a separate field.
    let splats = splats.bake_min_scale();
    let sh_degree = splats.sh_degree();
    let num_splats = splats.num_splats() as usize;
    let n_coeffs = sh_coeffs_for_degree(sh_degree) as usize;
    let render_mode_str = if splats.render_mip { "mip" } else { "default" };

    let mut comments = vec!["Exported from Brush".to_owned()];
    if let Some(up) = up_axis {
        comments.push(format!("Vertical axis: {} {} {}", up.x, up.y, up.z));
    } else {
        comments.push("Vertical axis: y".to_owned());
    }
    comments.push(format!("SH degree: {sh_degree}"));
    comments.push(format!("SplatRenderMode: {render_mode_str}"));

    write_header(writer, &comments, num_splats, n_coeffs)?;

    let row_floats = 11 + 3 * n_coeffs;
    let mut body = Vec::with_capacity(chunk_splats.min(num_splats) * row_floats * size_of::<f32>());

    for start in (0..num_splats).step_by(chunk_splats) {
        let end = (start + chunk_splats).min(num_splats);
        let data = Transaction::default()
            .register(splats.transforms.val().slice(s![start..end]))
            .register(splats.raw_opacities.val().slice(s![start..end]))
            // Permute to inria format ([n, channel, coeffs]).
            .register(
                splats
                    .sh_coeffs
                    .val()
                    .slice(s![start..end])
                    .permute([0, 2, 1]),
            )
            .execute_async()
            .await
            .map_err(|_fetch| ExportError::FetchFailed)?;

        let [transforms, raw_opacities, sh_coeffs]: [Vec<f32>; 3] = data
            .into_iter()
            .map(|x| {
                x.try_into_vec()
                    .map_err(|_convert| ExportError::DataConversion)
            })
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_convert| ExportError::DataConversion)?;

        let n = end - start;
        if transforms.len() != n * 10
            || raw_opacities.len() != n
            || sh_coeffs.len() != n * n_coeffs * 3
        {
            return Err(ExportError::DataConversion);
        }

        body.clear();
        for i in 0..n {
            let t = &transforms[i * 10..i * 10 + 10];
            let sh = &sh_coeffs[i * n_coeffs * 3..(i + 1) * n_coeffs * 3];
            let (r0, r1, r2, r3) = (t[3], t[4], t[5], t[6]);
            let rn = (r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3).sqrt().max(1e-12);

            let row = [
                t[0],
                t[1],
                t[2],
                t[7],
                t[8],
                t[9],
                raw_opacities[i],
                r0 / rn,
                r1 / rn,
                r2 / rn,
                r3 / rn,
                sh[0],
                sh[n_coeffs],
                sh[n_coeffs * 2],
            ];
            // f_rest_* are all red rest coeffs, then green, then blue.
            let rest = (0..3).flat_map(|c| &sh[c * n_coeffs + 1..(c + 1) * n_coeffs]);
            for v in row.iter().chain(rest) {
                body.extend_from_slice(&v.to_le_bytes());
            }
        }
        writer.write_all(&body)?;
    }
    Ok(())
}

/// Header layout matches what `serde_ply` produced for the previous exporter.
fn write_header(
    writer: &mut impl Write,
    comments: &[String],
    num_splats: usize,
    n_coeffs: usize,
) -> std::io::Result<()> {
    let mut header = String::from("ply\nformat binary_little_endian 1.0\n");
    for c in comments {
        header.push_str(&format!("comment {c}\n"));
    }
    header.push_str(&format!("element vertex {num_splats}\n"));
    let base = [
        "x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity", "rot_0", "rot_1", "rot_2",
        "rot_3", "f_dc_0", "f_dc_1", "f_dc_2",
    ];
    for name in base.iter().chain(&SH_NAMES[..3 * (n_coeffs - 1)]) {
        header.push_str(&format!("property float {name}\n"));
    }
    header.push_str("end_header\n");
    writer.write_all(header.as_bytes())
}

#[cfg(test)]
mod legacy;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::import::load_splat_from_ply;
    use crate::test_utils::create_test_splats;

    use brush_render::gaussian_splats::SplatRenderMode;
    use std::io::Cursor;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn streaming_matches_original_bytes_across_chunks_and_floors() {
        use crate::test_utils::create_test_splats_with_count;
        use burn::{module::Param, tensor::Tensor};

        let device = brush_cube::test_helpers::test_device().await.into();
        for degree in 0..=4 {
            let mut splats = create_test_splats_with_count(degree, 17);
            // Exercise quaternion normalization, unequal channel coefficients,
            // the baked floor, and dataset-unit conversion in the real writer.
            let rotation = Tensor::from_floats([[2.0, 1.0, -0.5, 0.0]], &device).repeat_dim(0, 17);
            splats.transforms =
                Param::from_tensor(splats.transforms.val().slice_assign(s![.., 3..7], rotation));
            for floor in [false, true] {
                let mut splats = splats.clone();
                if floor {
                    splats = splats.with_min_scale(Tensor::full([17], 0.4, &device));
                    splats.render_mip = true;
                }
                let splats = splats.scaled(1000.0);
                for up in [None, Some(Vec3::Z)] {
                    let expected = legacy::splat_to_ply(splats.clone(), up).await.unwrap();
                    assert_eq!(splat_to_ply(splats.clone(), up).await.unwrap(), expected);
                    for chunk in [1, 4, 17, 32] {
                        let mut actual = Vec::new();
                        write_ply_chunked(splats.clone(), up, &mut actual, chunk)
                            .await
                            .unwrap();
                        assert_eq!(actual, expected, "SH{degree}, floor={floor}, chunk={chunk}");
                    }
                }
            }
        }
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn streaming_propagates_partial_writer_failure() {
        struct FailingWriter {
            remaining: usize,
        }
        impl Write for FailingWriter {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if self.remaining == 0 {
                    return Err(std::io::ErrorKind::BrokenPipe.into());
                }
                let written = self.remaining.min(bytes.len()).min(7);
                self.remaining -= written;
                Ok(written)
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        brush_cube::test_helpers::test_device().await;
        let splats = crate::test_utils::create_test_splats_with_count(3, 17);
        let mut header = Vec::new();
        write_header(
            &mut header,
            &[
                "Exported from Brush".to_owned(),
                "Vertical axis: y".to_owned(),
                "SH degree: 3".to_owned(),
                "SplatRenderMode: default".to_owned(),
            ],
            17,
            16,
        )
        .unwrap();
        // Fail partway through the second four-splat chunk.
        let mut writer = FailingWriter {
            remaining: header.len() + 5 * 59 * 4,
        };
        let error = write_ply_chunked(splats, None, &mut writer, 4)
            .await
            .unwrap_err();
        assert!(
            matches!(error, ExportError::Io(error) if error.kind() == std::io::ErrorKind::BrokenPipe)
        );
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn streaming_bounds_writes_at_the_production_chunk_boundary() {
        struct CountWrites(Vec<usize>);
        impl Write for CountWrites {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.push(bytes.len());
                Ok(bytes.len())
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        brush_cube::test_helpers::test_device().await;
        let splats = crate::test_utils::create_test_splats_with_count(0, EXPORT_CHUNK_SPLATS + 1);
        let mut writer = CountWrites(Vec::new());
        splat_to_ply_writer(splats, None, &mut writer)
            .await
            .unwrap();
        // One header, one full chunk, and one trailing row; the sink never
        // retains an output buffer for the whole model.
        assert_eq!(writer.0.len(), 3);
        assert_eq!(writer.0[1], EXPORT_CHUNK_SPLATS * 14 * size_of::<f32>());
        assert_eq!(writer.0[2], 14 * size_of::<f32>());
    }

    #[cfg(target_family = "wasm")]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    async fn assert_coeffs_match(orig: &Splats, imported: &Splats) {
        let orig_sh: Vec<f32> = orig
            .sh_coeffs
            .val()
            .into_data_async()
            .await
            .unwrap()
            .try_into_vec()
            .expect("Failed to convert SH coefficients to vector");
        let import_sh: Vec<f32> = imported
            .sh_coeffs
            .val()
            .into_data_async()
            .await
            .unwrap()
            .try_into_vec()
            .expect("Failed to convert SH coefficients to vector");

        assert_eq!(orig_sh.len(), import_sh.len());
        for (i, (&orig, &imported)) in orig_sh.iter().zip(import_sh.iter()).enumerate() {
            assert!(
                (orig - imported).abs() < 1e-6_f32,
                "SH coeffs mismatch at index {i}: orig={orig}, imported={imported}",
            );
        }
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_sh_degree_exports() {
        let _device = brush_cube::test_helpers::test_device().await;
        for degree in 0..=4 {
            let splats = create_test_splats(degree);
            assert_eq!(splats.sh_degree(), degree);

            let bytes = splat_to_ply(splats, None).await.unwrap();
            let header_end = bytes
                .windows(b"end_header\n".len())
                .position(|w| w == b"end_header\n")
                .unwrap()
                + b"end_header\n".len();
            let expected_rest_coeffs = if degree == 0 {
                0
            } else {
                (sh_coeffs_for_degree(degree) - 1) * 3
            };

            assert_eq!(
                bytes.len() - header_end,
                (14 + expected_rest_coeffs as usize) * size_of::<f32>()
            );
        }
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_ply_field_count_matches_sh_degree() {
        let _device = brush_cube::test_helpers::test_device().await;
        let test_cases = [(0, 0), (1, 9), (2, 24)];

        for (degree, expected_rest_fields) in test_cases {
            let splats = create_test_splats(degree);
            let ply_bytes = splat_to_ply(splats, None).await.unwrap();
            let ply_string = String::from_utf8_lossy(&ply_bytes);

            let actual_rest_fields = ply_string.matches("property float f_rest_").count();
            assert_eq!(
                actual_rest_fields, expected_rest_fields,
                "Degree {degree} should have {expected_rest_fields} f_rest_ fields",
            );

            assert!(ply_string.contains("f_dc_0"));
            if expected_rest_fields > 0 {
                assert!(ply_string.contains("f_rest_0"));
                assert!(!ply_string.contains(&format!("f_rest_{expected_rest_fields}")));
            } else {
                assert!(!ply_string.contains("f_rest_0"));
            }
        }
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_roundtrip_sh_coefficient_ordering() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();

        for degree in [0, 1, 2] {
            let original_splats = create_test_splats(degree);
            let ply_bytes = splat_to_ply(original_splats.clone(), None)
                .await
                .expect("Failed to serialize splats");

            let cursor = Cursor::new(ply_bytes);
            let imported_message = load_splat_from_ply(cursor, None)
                .await
                .expect("Failed to deserialize splats");
            let imported_splats = imported_message
                .data
                .into_splats(&device, SplatRenderMode::Default);

            assert_eq!(imported_splats.sh_degree(), degree);
            assert_coeffs_match(&original_splats, &imported_splats).await;
        }
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_export_roundtrip_multiple_splats() {
        use crate::test_utils::create_test_splats_with_count;

        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let num_splats = 100;

        for degree in [0, 1, 2, 3] {
            let original = create_test_splats_with_count(degree, num_splats);
            assert_eq!(original.num_splats(), num_splats as u32);

            let ply_bytes = splat_to_ply(original.clone(), None)
                .await
                .expect("Failed to export splats");

            assert!(!ply_bytes.is_empty(), "Exported PLY should not be empty");

            let cursor = Cursor::new(ply_bytes);
            let imported_message = load_splat_from_ply(cursor, None)
                .await
                .expect("Failed to reimport exported splats");
            let imported = imported_message
                .data
                .into_splats(&device, SplatRenderMode::Default);

            assert_eq!(
                imported.num_splats(),
                num_splats as u32,
                "Splat count mismatch after roundtrip"
            );
            assert_eq!(imported.sh_degree(), degree);
            assert_coeffs_match(&original, &imported).await;
        }
    }
}
