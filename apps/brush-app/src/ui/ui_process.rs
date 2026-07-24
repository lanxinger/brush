use anyhow::Result;
use brush_async::Actor;
use brush_process::{RunningProcess, message::ProcessMessage, slot::Slot};
use brush_render::{
    bounding_box::BoundingBox, camera::Camera, gaussian_splats::Splats,
    kernels::camera_model::CameraModel,
};
use burn_wgpu::WgpuDevice;
use egui::{Response, TextureHandle};
use glam::{Affine3A, Quat, Vec3};
use std::sync::RwLock;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;

use crate::ui::{UiMode, app::CameraSettings, camera_controls::CameraController};

const FRAME_MARGIN: f32 = 1.1;
const RENDER_NEAR_PLANE: f32 = 0.01;

#[derive(Debug, Clone)]
enum ControlMessage {
    Paused(bool),
}

struct ProcessHandle {
    messages: mpsc::UnboundedReceiver<anyhow::Result<ProcessMessage>>,
    control: mpsc::UnboundedSender<ControlMessage>,
    splat_view: Slot<Splats>,
}

/// A thread-safe wrapper around the UI process.
/// This allows the UI process to be accessed from multiple threads.
///
/// Mixing a sync lock and async code is asking for trouble, but there's no other good way in egui currently.
/// The "precondition" to avoid deadlocks, is to only holds locks _within the trait functions_. As long as you don't ever hold them
/// over an await point, things shouldn't be able to deadlock.
pub struct UiProcess(RwLock<UiProcessInner>);

#[derive(Debug, Clone, Copy)]
pub enum BackgroundStyle {
    Black,
    Checkerboard,
}

#[derive(Clone)]
pub struct TexHandle {
    pub handle: TextureHandle,
    pub has_alpha: bool,
    /// resolution of the source view.
    pub train_size: (u32, u32),
}

impl UiProcess {
    pub fn new(dev: WgpuDevice, ui_ctx: egui::Context) -> Self {
        let actor = Actor::new("ui-process");
        Self(RwLock::new(UiProcessInner::new(dev, ui_ctx, actor)))
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, UiProcessInner> {
        self.0.read().expect("RwLock poisoned")
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, UiProcessInner> {
        self.0.write().expect("RwLock poisoned")
    }

    pub(crate) fn background_style(&self) -> BackgroundStyle {
        self.read().background_style
    }

    #[allow(unused)]
    pub(crate) fn set_background_style(&self, style: BackgroundStyle) {
        self.write().background_style = style;
    }

    pub(crate) fn current_splats(&self) -> Slot<Splats> {
        self.read()
            .process_handle
            .as_ref()
            .map_or(Slot::default(), |s| s.splat_view.clone())
    }

    pub fn is_loading(&self) -> bool {
        self.read().is_loading
    }

    pub fn is_training(&self) -> bool {
        self.read().is_training
    }

    pub fn tick_controls(&self, response: &Response, ui: &egui::Ui) {
        self.write().controls.tick(response, ui);
    }

    pub fn model_local_to_world(&self) -> glam::Affine3A {
        self.read().controls.model_local_to_world
    }

    pub fn current_camera(&self) -> Camera {
        let inner = self.read();
        // Keep controls & camera position in sync.
        let mut cam = inner.camera;
        cam.position = inner.controls.position;
        cam.rotation = inner.controls.rotation;
        cam
    }

    pub fn set_train_paused(&self, paused: bool) {
        self.write().train_paused = paused;
        if let Some(process) = self.read().process_handle.as_ref() {
            let _ = process.control.send(ControlMessage::Paused(paused));
        }
    }

    pub fn is_train_paused(&self) -> bool {
        self.read().train_paused
    }

    pub(crate) fn train_iter(&self) -> u32 {
        self.read().train_iter
    }

    pub fn get_cam_settings(&self) -> CameraSettings {
        self.read().controls.settings.clone()
    }

    pub fn get_grid_opacity(&self) -> f32 {
        let inner = self.read();
        if inner.controls.settings.grid_enabled.is_some_and(|g| g) {
            1.0 // Grid fully visible when enabled
        } else {
            inner.controls.get_grid_opacity() // Use fade timer when disabled
        }
    }

    pub fn set_cam_settings(&self, settings: &CameraSettings) {
        let mut inner = self.write();
        inner.controls.settings = settings.clone();
        inner.splat_scale = settings.splat_scale;
    }

    #[allow(dead_code)] // Used from wasm.rs / android.rs.
    pub fn set_cam_transform(&self, position: Vec3, rotation: Quat) {
        let mut inner = self.write();
        inner.camera_override = true;
        inner.set_camera_transform(position, rotation);
        drop(inner);
        self.read().repaint();
    }

    #[allow(dead_code)] // Used from wasm.rs / android.rs.
    pub fn set_focal_point(&self, focal_point: Vec3, focus_distance: f32, rotation: Quat) {
        let mut inner = self.write();
        inner.camera_override = true;
        inner.set_focal_point(focal_point, focus_distance, rotation);
        drop(inner);
        self.read().repaint();
    }

    pub fn set_cam_fov(&self, fov_y: f64) {
        let mut inner = self.write();
        // Scale fov_x proportionally to maintain the camera's aspect ratio.
        // This allows setting FOV smaller than the dataset FOV.
        let old_fov_y = inner.camera.fov_y;
        let aspect = (inner.camera.fov_x / 2.0).tan() / (old_fov_y / 2.0).tan();
        inner.camera.fov_y = fov_y;
        inner.camera.fov_x = 2.0 * (aspect * (fov_y / 2.0).tan()).atan();
        drop(inner);
        self.read().repaint();
    }

    pub fn focus_view(&self, cam: &Camera) {
        // Also focus this view.
        let mut inner = self.write();
        inner.camera = *cam;
        inner.controls.stop_movement();

        // We want to set the view matrix such that MV == view view matrix.
        // new_view_mat * model_mat == view_view_mat
        // new_view_mat = view_view_mat * model_mat.inverse()
        let new_view_mat = cam.world_to_local() * inner.controls.model_local_to_world.inverse();

        let view_local_to_world = new_view_mat.inverse();
        let (_, rot, translate) = view_local_to_world.to_scale_rotation_translation();
        inner.controls.position = translate;
        inner.controls.rotation = rot;
        if let Some(target) = inner.navigation_target {
            inner.controls.focus_distance =
                projected_focus_distance(target, translate, rot, RENDER_NEAR_PLANE);
        }
        inner.repaint();
    }

    pub fn can_auto_frame(&self) -> bool {
        let inner = self.read();
        !inner.camera_override && !inner.did_auto_frame
    }

    /// Frames robust model-space bounds once, preserving the current viewing
    /// direction while replacing the arbitrary COLMAP pivot and distance.
    pub fn frame_scene(&self, bounds: BoundingBox, viewport_aspect: f32) {
        let mut inner = self.write();
        if inner.camera_override || inner.did_auto_frame {
            return;
        }

        let Some((target, distance)) = fit_bounds(
            bounds,
            inner.controls.model_local_to_world,
            inner.controls.rotation,
            inner.camera.fov_x,
            inner.camera.fov_y,
            viewport_aspect,
            FRAME_MARGIN,
        ) else {
            return;
        };

        // Dataset intrinsics are restored by `focus_view` when a reference
        // image is explicitly selected. Free navigation uses a centered
        // pinhole camera so the orbit target is also the visual center.
        inner.camera.center_uv = glam::vec2(0.5, 0.5);
        inner.camera.camera_model = CameraModel::Pinhole;

        let rotation = inner.controls.rotation;
        inner.controls.stop_movement();
        inner.set_focal_point(target, distance, rotation);
        inner.did_auto_frame = true;
        inner.repaint();
    }

    pub fn set_model_up(&self, up_axis: Vec3) {
        let mut inner = self.write();
        inner.up_axis = Some(up_axis);
        inner.controls.model_local_to_world = Affine3A::from_rotation_translation(
            Quat::from_rotation_arc(Vec3::NEG_Y, up_axis.normalize()),
            Vec3::ZERO,
        );
        inner.repaint();
    }

    pub fn up_axis(&self) -> Option<Vec3> {
        self.read().up_axis
    }

    /// Connect to an existing running process.
    pub fn connect_to_process(&self, process: RunningProcess) {
        {
            let mut inner = self.write();
            let reset = UiProcessInner::new(
                inner.burn_device.clone(),
                inner.ui_ctx.clone(),
                inner.actor.clone(),
            );
            *inner = reset;
        }

        let (sender, receiver) = mpsc::unbounded_channel();
        let (train_sender, mut train_receiver) = mpsc::unbounded_channel();

        let mut process = process;

        let egui_ctx = self.read().ui_ctx.clone();

        self.read()
            .actor
            .run(move || async move {
                while let Some(msg) = process.stream.next().await {
                    // Stop the process if no one is listening anymore.
                    if sender.send(msg).is_err() {
                        break;
                    }

                    // Check if training is paused. Don't care about other messages as pausing loading
                    // doesn't make much sense.
                    if matches!(train_receiver.try_recv(), Ok(ControlMessage::Paused(true))) {
                        // Pause until we're explicitly unpaused. If the control channel
                        // closed (the process was reset / replaced, dropping the sender),
                        // `recv()` returns `None` immediately and forever.
                        loop {
                            match train_receiver.recv().await {
                                Some(ControlMessage::Paused(false)) | None => break,
                                _ => {}
                            }
                            // Yield back to the runtime can't starve the browser event loop.
                            brush_async::yield_now().await;
                        }
                    }

                    // Mark egui as needing a repaint.
                    egui_ctx.request_repaint();

                    // Give back control to the runtime.
                    // This only really matters in the browser:
                    // on native, receiving also yields. In the browser that doesn't yield
                    // back control fully though whereas yield_now() does.
                    brush_async::yield_now().await;
                }
            })
            .detach();

        self.write().process_handle = Some(ProcessHandle {
            messages: receiver,
            control: train_sender,
            splat_view: process.splat_view,
        });
    }

    pub fn message_queue(&self) -> Vec<Result<ProcessMessage>> {
        let mut ret = vec![];
        let mut inner = self.write();
        if let Some(process) = inner.process_handle.as_mut() {
            while let Ok(msg) = process.messages.try_recv() {
                ret.push(msg);
            }
        }

        for msg in &ret {
            // Keep track of things the ui process needs.
            match msg {
                Ok(ProcessMessage::StartLoading { training, .. }) => {
                    inner.is_training = *training;
                    inner.is_loading = true;
                    inner.train_iter = 0;
                }
                Ok(ProcessMessage::DoneLoading) => {
                    inner.is_loading = false;
                }
                Ok(ProcessMessage::TrainMessage(
                    brush_process::message::TrainMessage::TrainStep { iter, .. },
                )) => {
                    inner.train_iter = *iter;
                }
                Err(_) => {
                    inner.is_loading = false;
                    inner.is_training = false;
                }
                _ => (),
            }
        }
        drop(inner);
        ret
    }

    pub fn ui_mode(&self) -> UiMode {
        self.read().ui_mode
    }

    pub fn set_ui_mode(&self, mode: UiMode) {
        self.write().ui_mode = mode;
    }

    pub fn request_reset_layout(&self) {
        self.write().reset_layout_requested = true;
    }

    pub fn take_reset_layout_request(&self) -> bool {
        let mut inner = self.write();
        let requested = inner.reset_layout_requested;
        inner.reset_layout_requested = false;
        requested
    }

    pub fn reset_session(&self) {
        let mut inner = self.write();
        *inner = UiProcessInner::new(
            inner.burn_device.clone(),
            inner.ui_ctx.clone(),
            inner.actor.clone(),
        );
        inner.session_reset_requested = true;
    }

    pub fn take_session_reset_request(&self) -> bool {
        let mut inner = self.write();
        let requested = inner.session_reset_requested;
        inner.session_reset_requested = false;
        requested
    }

    pub fn burn_device(&self) -> WgpuDevice {
        self.read().burn_device.clone()
    }

    pub(crate) fn actor(&self) -> Actor {
        self.read().actor.clone()
    }
}

struct UiProcessInner {
    is_loading: bool,
    is_training: bool,
    camera: Camera,
    splat_scale: Option<f32>,
    controls: CameraController,
    process_handle: Option<ProcessHandle>,
    ui_mode: UiMode,
    background_style: BackgroundStyle,
    train_paused: bool,
    train_iter: u32,
    reset_layout_requested: bool,
    session_reset_requested: bool,
    ui_ctx: egui::Context,
    burn_device: WgpuDevice,
    actor: Actor,
    up_axis: Option<Vec3>,
    /// An explicit embedding-host camera wins over automatic scene framing.
    camera_override: bool,
    /// Prevent later bounds-bearing updates from snapping an interactive camera.
    did_auto_frame: bool,
    /// Last explicit or automatic orbit target, retained across reference views.
    navigation_target: Option<Vec3>,
}

impl UiProcessInner {
    pub fn new(burn_device: WgpuDevice, ui_ctx: egui::Context, actor: Actor) -> Self {
        let position = -Vec3::Z * 2.5;
        let rotation = Quat::IDENTITY;

        let controls = CameraController::new(position, rotation, CameraSettings::default());
        let camera = Camera::new(
            Vec3::ZERO,
            Quat::IDENTITY,
            0.8,
            0.8,
            glam::vec2(0.5, 0.5),
            CameraModel::Pinhole,
        );

        Self {
            camera,
            controls,
            splat_scale: None,
            is_loading: false,
            is_training: false,
            train_iter: 0,
            process_handle: None,
            ui_mode: UiMode::Default,
            background_style: BackgroundStyle::Black,
            train_paused: false,
            reset_layout_requested: false,
            session_reset_requested: false,
            burn_device,
            ui_ctx,
            actor,
            up_axis: None,
            camera_override: false,
            did_auto_frame: false,
            navigation_target: None,
        }
    }

    fn repaint(&self) {
        self.ui_ctx.request_repaint();
    }

    #[allow(dead_code)] // Used from wasm.rs / android.rs.
    fn set_camera_transform(&mut self, position: Vec3, rotation: Quat) {
        self.controls.position = position;
        self.controls.rotation = rotation;
        self.camera.position = position;
        self.camera.rotation = rotation;
    }

    #[allow(dead_code)] // Used from wasm.rs / android.rs.
    fn set_focal_point(&mut self, focal_point: Vec3, focus_distance: f32, rotation: Quat) {
        let position = focal_point - rotation * Vec3::Z * focus_distance;
        self.set_camera_transform(position, rotation);
        self.controls.focus_distance = focus_distance;
        self.navigation_target = Some(focal_point);
    }
}

fn projected_focus_distance(target: Vec3, position: Vec3, rotation: Quat, minimum: f32) -> f32 {
    let projection = (target - position).dot(rotation * Vec3::Z);
    if projection.is_finite() {
        projection.max(minimum)
    } else {
        minimum
    }
}

fn fit_bounds(
    bounds: BoundingBox,
    model_local_to_world: Affine3A,
    camera_rotation: Quat,
    fov_x: f64,
    fov_y: f64,
    viewport_aspect: f32,
    margin: f32,
) -> Option<(Vec3, f32)> {
    if !bounds.center.is_finite()
        || !bounds.extent.is_finite()
        || !camera_rotation.is_finite()
        || !viewport_aspect.is_finite()
        || viewport_aspect <= 0.0
        || !margin.is_finite()
        || margin < 1.0
    {
        return None;
    }

    let mut tan_x = (fov_x * 0.5).tan() as f32;
    let mut tan_y = (fov_y * 0.5).tan() as f32;
    if !tan_x.is_finite() || !tan_y.is_finite() || tan_x <= 0.0 || tan_y <= 0.0 {
        return None;
    }

    // Match ScenePanel's "show at least the dataset view" pinhole FOV expansion
    // for the actual viewport aspect.
    let camera_aspect = tan_x / tan_y;
    if viewport_aspect > camera_aspect {
        tan_x = tan_y * viewport_aspect;
    } else {
        tan_y = tan_x / viewport_aspect;
    }

    let target = model_local_to_world.transform_point3(bounds.center);
    let camera_from_world = camera_rotation.inverse();
    let extent = bounds.extent.abs();
    let mut distance = 0.0_f32;

    for x in [-1.0, 1.0] {
        for y in [-1.0, 1.0] {
            for z in [-1.0, 1.0] {
                let local_corner = bounds.center + extent * Vec3::new(x, y, z);
                let world_corner = model_local_to_world.transform_point3(local_corner);
                let corner = camera_from_world * (world_corner - target);

                distance = distance
                    .max(corner.x.abs() * margin / tan_x - corner.z)
                    .max(corner.y.abs() * margin / tan_y - corner.z)
                    .max(RENDER_NEAR_PLANE - corner.z);
            }
        }
    }

    if !target.is_finite() || !distance.is_finite() {
        return None;
    }

    Some((target, distance.max(RENDER_NEAR_PLANE)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_bounds_fit(
        bounds: BoundingBox,
        model_local_to_world: Affine3A,
        rotation: Quat,
        viewport_aspect: f32,
    ) {
        let fov_x = 0.8;
        let fov_y = 0.8;
        let (target, distance) = fit_bounds(
            bounds,
            model_local_to_world,
            rotation,
            fov_x,
            fov_y,
            viewport_aspect,
            FRAME_MARGIN,
        )
        .expect("valid bounds should fit");

        assert!(
            target.abs_diff_eq(model_local_to_world.transform_point3(bounds.center), 1e-5),
            "orbit target must be the transformed robust-bounds center"
        );

        let mut tan_x = (fov_x * 0.5).tan() as f32;
        let mut tan_y = (fov_y * 0.5).tan() as f32;
        let camera_aspect = tan_x / tan_y;
        if viewport_aspect > camera_aspect {
            tan_x = tan_y * viewport_aspect;
        } else {
            tan_y = tan_x / viewport_aspect;
        }

        let camera_from_world = rotation.inverse();
        for x in [-1.0, 1.0] {
            for y in [-1.0, 1.0] {
                for z in [-1.0, 1.0] {
                    let local_corner = bounds.center + bounds.extent.abs() * Vec3::new(x, y, z);
                    let world_corner = model_local_to_world.transform_point3(local_corner);
                    let corner = camera_from_world * (world_corner - target);
                    let depth = distance + corner.z;

                    assert!(depth >= RENDER_NEAR_PLANE - 1e-5);
                    assert!(corner.x.abs() / depth <= tan_x / FRAME_MARGIN + 1e-5);
                    assert!(corner.y.abs() / depth <= tan_y / FRAME_MARGIN + 1e-5);
                }
            }
        }
    }

    #[test]
    fn fit_bounds_centers_and_contains_corners_across_scales_and_aspects() {
        let rotation = Quat::from_euler(glam::EulerRot::YXZ, 0.7, -0.35, 0.0).normalize();
        let model_local_to_world = Affine3A::from_rotation_translation(
            Quat::from_rotation_x(0.4),
            Vec3::new(3.0, -2.0, 5.0),
        );

        for scale in [0.01, 1.0, 100.0] {
            let bounds = BoundingBox {
                center: Vec3::new(12.0, -7.0, 4.0) * scale,
                extent: Vec3::new(2.0, 1.0, 3.0) * scale,
            };
            for aspect in [0.5, 1.0, 2.0] {
                assert_bounds_fit(bounds, model_local_to_world, rotation, aspect);
            }
        }
    }

    #[test]
    fn fit_bounds_rejects_invalid_input() {
        let bounds = BoundingBox {
            center: Vec3::ZERO,
            extent: Vec3::ONE,
        };

        assert!(
            fit_bounds(
                bounds,
                Affine3A::IDENTITY,
                Quat::IDENTITY,
                0.8,
                0.8,
                0.0,
                FRAME_MARGIN,
            )
            .is_none()
        );
    }

    #[test]
    fn reference_view_uses_closest_on_axis_scene_target() {
        let target = Vec3::new(4.0, -3.0, 7.0);
        let distance =
            projected_focus_distance(target, Vec3::new(1.0, 2.0, 1.0), Quat::IDENTITY, 0.01);
        assert!((distance - 6.0).abs() < 1e-6);

        let behind =
            projected_focus_distance(Vec3::NEG_Z, Vec3::ZERO, Quat::IDENTITY, RENDER_NEAR_PLANE);
        assert_eq!(behind, RENDER_NEAR_PLANE);
    }
}
