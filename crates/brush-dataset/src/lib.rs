#![recursion_limit = "256"]

pub mod config;
pub mod load_image;
pub mod scene;
pub mod scene_loader;

mod formats;

pub use formats::{DatasetLoadResult, load_dataset};

use glam::{Mat3, Mat4, Vec3};
use scene::Scene;
use scene::SceneView;

/// Orthonormal eigenvectors of a symmetric matrix, ordered by decreasing
/// eigenvalue. Jacobi rotations also handle zero and repeated eigenvalues.
pub fn compute_sorted_eigenvectors(matrix: Mat3) -> (Vec3, Vec3, Vec3) {
    if !matrix.is_finite() {
        return (Vec3::X, Vec3::Y, Vec3::Z);
    }
    let mut a: [[f64; 3]; 3] =
        std::array::from_fn(|row| std::array::from_fn(|col| f64::from(matrix.col(col)[row])));
    let mut vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let scale = a.iter().flatten().map(|x| x.abs()).fold(0.0, f64::max);

    for _ in 0..32 {
        let (p, q) = [(0, 1), (0, 2), (1, 2)]
            .into_iter()
            .max_by(|&(p, q), &(r, s)| a[p][q].abs().total_cmp(&a[r][s].abs()))
            .unwrap();
        if a[p][q].abs() <= scale * 1e-12 {
            break;
        }
        let tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        a[p][p] -= t * a[p][q];
        a[q][q] += t * a[p][q];
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for k in 0..3 {
            if k != p && k != q {
                let kp = c * a[k][p] - s * a[k][q];
                let kq = s * a[k][p] + c * a[k][q];
                a[k][p] = kp;
                a[p][k] = kp;
                a[k][q] = kq;
                a[q][k] = kq;
            }
            let kp = c * vectors[k][p] - s * vectors[k][q];
            let kq = s * vectors[k][p] + c * vectors[k][q];
            vectors[k][p] = kp;
            vectors[k][q] = kq;
        }
    }
    let mut order = [0, 1, 2];
    order.sort_by(|&i, &j| a[j][j].total_cmp(&a[i][i]));
    let axis = |i| {
        Vec3::new(
            vectors[0][i] as f32,
            vectors[1][i] as f32,
            vectors[2][i] as f32,
        )
    };
    (axis(order[0]), axis(order[1]), axis(order[2]))
}

#[derive(Clone)]
pub struct Dataset {
    pub train: Scene,
    pub eval: Option<Scene>,
}

impl Dataset {
    pub fn empty() -> Self {
        Self {
            train: Scene::new(vec![]),
            eval: None,
        }
    }

    pub fn from_views(train_views: Vec<SceneView>, eval_views: Vec<SceneView>) -> Self {
        Self {
            train: Scene::new(train_views),
            eval: if eval_views.is_empty() {
                None
            } else {
                Some(Scene::new(eval_views))
            },
        }
    }

    pub fn estimate_up(&self) -> Vec3 {
        // based on https://github.com/jonbarron/camp_zipnerf/blob/8e6d57e3aee34235faf3ef99decca0994efe66c9/camp_zipnerf/internal/camera_utils.py#L233
        let (c2ws, ts): (Vec<_>, Vec<_>) = self
            .train
            .views
            .iter()
            .chain(self.eval.iter().flat_map(|e| e.views.as_slice()))
            .map(|v| (v.camera.local_to_world(), v.camera.position))
            .collect();

        // A line or a single camera does not define a capture plane. Use the
        // cameras' down axes in that case, with the same output convention.
        let camera_up = || {
            let down = c2ws
                .iter()
                .map(|c2w| Vec3::from(c2w.matrix3.y_axis))
                .sum::<Vec3>();
            let down = down.try_normalize().unwrap_or(Vec3::Y);
            Vec3::new(-down.x, -down.y, down.z)
        };
        if ts.len() < 3 {
            return camera_up();
        }
        let mean_t = ts.iter().sum::<Vec3>() / ts.len() as f32;

        // Compute 3x3 covariance by t^T * t ((3, N) * (N, 3) -> (3, 3))
        let cov = ts.iter().map(|&p| p - mean_t).fold(Mat3::ZERO, |acc, p| {
            acc + Mat3::from_cols(p * p.x, p * p.y, p * p.z).transpose()
        });
        let (e0, e1, e2) = compute_sorted_eigenvectors(cov);
        let largest_variance = e0.dot(cov * e0);
        let second_variance = e1.dot(cov * e1);
        if !largest_variance.is_finite()
            || largest_variance <= 0.0
            || second_variance <= largest_variance * 1e-6
        {
            return camera_up();
        }
        let mut rot = Mat3::from_cols(e0, e1, e2).transpose();

        if rot.determinant() < 0.0 {
            let diag = Mat3::from_diagonal(Vec3::new(1.0, 1.0, -1.0));
            rot = diag.mul_mat3(&rot);
        }

        let mut transform = Mat4::from_cols(
            rot.col(0).extend(0.0),
            rot.col(1).extend(0.0),
            rot.col(2).extend(0.0),
            rot.mul_vec3(-mean_t).extend(1.0),
        );

        let mut y_axis_z = 0.0;
        for c2w in c2ws {
            y_axis_z += transform.mul_mat4(&Mat4::from(c2w)).col(1).z;
        }

        // Flip coordinate system if z component of y-axis is negative
        if y_axis_z < 0.0 {
            let scale = Mat4::from_scale(Vec3::new(1.0, -1.0, -1.0));
            transform = scale.mul_mat4(&transform);
        }

        Vec3::new(-transform.col(0).z, -transform.col(1).z, transform.col(2).z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brush_render::camera::Camera;
    use brush_vfs::BrushVfs;
    use glam::Quat;
    use std::sync::Arc;
    use wasm_bindgen_test::wasm_bindgen_test;

    fn check_eigenbasis(matrix: Mat3) {
        let (a, b, c) = compute_sorted_eigenvectors(matrix);
        let axes = [a, b, c];
        let values = axes.map(|axis| axis.dot(matrix * axis));
        assert!(values[0] >= values[1] - 1e-5 && values[1] >= values[2] - 1e-5);
        for (axis, value) in axes.into_iter().zip(values) {
            assert!(axis.is_finite());
            assert!((axis.length() - 1.0).abs() < 1e-5);
            assert!((matrix * axis - value * axis).length() < 1e-5);
        }
        assert!(a.dot(b).abs() < 1e-5);
        assert!(a.dot(c).abs() < 1e-5);
        assert!(b.dot(c).abs() < 1e-5);
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn planar_covariance_has_axis_aligned_normal() {
        let covariance = Mat3::from_diagonal(Vec3::new(4.0, 0.0, 1.0));
        check_eigenbasis(covariance);
        let (_, _, normal) = compute_sorted_eigenvectors(covariance);
        assert!(normal.dot(Vec3::Y).abs() > 0.99999);
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn eigenbasis_handles_rotated_and_repeated_eigenvalues() {
        let rotation = Mat3::from_quat(Quat::from_euler(glam::EulerRot::XYZ, 0.3, 0.7, -0.2));
        for diagonal in [
            Vec3::new(4.0, 1.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::ONE,
            Vec3::ZERO,
        ] {
            check_eigenbasis(rotation * Mat3::from_diagonal(diagonal) * rotation.transpose());
        }
    }

    fn dataset(positions: &[Vec3], rotation: Quat) -> Dataset {
        let image = scene::LoadImage::new(
            Arc::new(BrushVfs::empty()),
            "unused.png".into(),
            None,
            1920,
            None,
            false,
        );
        Dataset::from_views(
            positions
                .iter()
                .map(|&position| SceneView {
                    image: image.clone(),
                    camera: Camera {
                        position,
                        rotation,
                        ..Default::default()
                    },
                })
                .collect(),
            vec![],
        )
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn planar_capture_estimates_camera_up() {
        let capture = dataset(
            &[
                Vec3::new(-2.0, 0.0, -1.0),
                Vec3::new(-2.0, 0.0, 1.0),
                Vec3::new(2.0, 0.0, -1.0),
                Vec3::new(2.0, 0.0, 1.0),
            ],
            Quat::IDENTITY,
        );
        assert!((capture.estimate_up() - Vec3::NEG_Y).length() < 1e-5);
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn degenerate_capture_uses_camera_orientation() {
        let rotation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        for positions in [vec![Vec3::ZERO], vec![Vec3::NEG_Y, Vec3::ZERO, Vec3::Y]] {
            assert!((dataset(&positions, rotation).estimate_up() - Vec3::X).length() < 1e-5);
        }
        assert_eq!(Dataset::empty().estimate_up(), Vec3::NEG_Y);
    }
}
