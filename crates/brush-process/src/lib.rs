pub mod args_file;
pub mod config;
pub mod message;
pub mod slot;
pub mod train_stream;

#[cfg(not(target_family = "wasm"))]
mod training_metrics;

pub use brush_vfs::DataSource;

/// Git-derived identifier for the source used to build Brush.
pub const BUILD_ID: &str = env!("BRUSH_BUILD_ID");

/// Package version with the Git-derived build identifier.
pub const VERSION: &str = env!("BRUSH_VERSION");

#[cfg(not(target_os = "ios"))]
use burn_wgpu::graphics::AutoGraphicsApi;
use burn_wgpu::{RuntimeOptions, WgpuDevice, graphics::GraphicsApi};
use wgpu::{Adapter, Device, Queue};

use std::future::Future;
use std::pin::{Pin, pin};

use anyhow::Error;
use async_fn_stream::{TryStreamEmitter, try_fn_stream};
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_vfs::SendNotWasm;
use tokio_stream::{Stream, StreamExt};

// AutoGraphicsApi has no iOS selection arm and falls back to Vulkan. Apple
// mobile devices only expose Metal, so choose it explicitly on iOS.
#[cfg(target_os = "ios")]
type DefaultGraphicsApi = burn_wgpu::graphics::Metal;
#[cfg(not(target_os = "ios"))]
type DefaultGraphicsApi = AutoGraphicsApi;

fn burn_options() -> RuntimeOptions {
    RuntimeOptions {
        tasks_max: 64,
        memory_config: burn_wgpu::MemoryConfiguration::ExclusivePages,
    }
}

/// Open the compute device.
///
/// Its own device, separate from anything the viewer owns, so training never
/// contends with GUI work on the same queue. wgpu needs an explicit setup so
/// the adapter and limits are chosen before any kernel runs.
async fn open_device() -> RegisteredDevice {
    burn_wgpu::init_setup_async::<DefaultGraphicsApi>(&WgpuDevice::DefaultDevice, burn_options())
        .await;
    RegisteredDevice {
        burn: WgpuDevice::DefaultDevice.into(),
        host: None,
    }
}

/// Open the compute device now, rather than waiting for the first thing that
/// needs it. Only worth calling to front-load the cost.
pub async fn burn_init_setup() -> WgpuDevice {
    registered_wgpu(DEVICE.get_or_init(open_device).await).clone()
}

/// Why a host-provided device could not become Brush's compute device.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum BurnInitDeviceError {
    /// Another, different device was registered first.
    AlreadyInitialized,
    /// Another task is currently opening the compute device.
    InitializationInProgress,
}

impl std::fmt::Display for BurnInitDeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyInitialized => {
                f.write_str("Brush is already initialized with a different GPU device")
            }
            Self::InitializationInProgress => {
                f.write_str("Brush GPU initialization is already in progress")
            }
        }
    }
}

impl std::error::Error for BurnInitDeviceError {}

/// Register a wgpu setup the host already owns.
///
/// Repeating the call with the same [`Device`] is idempotent. A different
/// device, or a call racing asynchronous initialization, returns an error so a
/// host never mistakes buffers from another device for its own.
pub fn try_burn_init_device(
    adapter: Adapter,
    device: Device,
    queue: Queue,
) -> Result<WgpuDevice, BurnInitDeviceError> {
    if let Some(existing) = DEVICE.get() {
        return registered_host(existing, &device);
    }

    let host_device = device.clone();

    let setup = burn_wgpu::WgpuSetup {
        instance: wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle()), // unused... need to fix this in Burn.
        adapter,
        device,
        queue,
        backend: DefaultGraphicsApi::backend(),
    };
    let burn = burn_wgpu::init_device(setup, burn_options());
    let registered = RegisteredDevice {
        burn: burn.clone().into(),
        host: Some(host_device.clone()),
    };

    match DEVICE.set(registered) {
        Ok(()) => Ok(burn),
        Err(tokio::sync::SetError::AlreadyInitializedError(_)) => registered_host(
            DEVICE
                .get()
                .expect("an initialized OnceCell must contain its value"),
            &host_device,
        ),
        Err(tokio::sync::SetError::InitializingError(_)) => {
            Err(BurnInitDeviceError::InitializationInProgress)
        }
    }
}

/// Register a host device with first-initializer-wins compatibility.
///
/// Hosts that must guarantee device identity should use
/// [`try_burn_init_device`] and handle a conflicting prior initialization.
pub fn burn_init_device(adapter: Adapter, device: Device, queue: Queue) -> WgpuDevice {
    match try_burn_init_device(adapter, device, queue) {
        Ok(device) => device,
        Err(BurnInitDeviceError::AlreadyInitialized) => registered_wgpu(
            DEVICE
                .get()
                .expect("already-initialized device must be registered"),
        )
        .clone(),
        Err(BurnInitDeviceError::InitializationInProgress) => {
            panic!("Brush GPU initialization is already in progress")
        }
    }
}

use crate::{
    message::ProcessMessage,
    slot::{Slot, SlotSender},
    train_stream::train_stream,
};

pub trait ProcessStream: Stream<Item = Result<ProcessMessage, Error>> + SendNotWasm {}
impl<T> ProcessStream for T where T: Stream<Item = Result<ProcessMessage, Error>> + SendNotWasm {}

pub struct RunningProcess {
    pub stream: Pin<Box<dyn ProcessStream>>,
    pub splat_view: Slot<Splats>,
}

/// Convenience alias for the emitter `try_fn_stream` hands us inside
/// the producer body — `try_fn_stream` itself drives the state
/// machine, so this is just the channel for `emit(msg).await`.
pub(crate) type Emitter = TryStreamEmitter<ProcessMessage, Error>;

use tokio::sync::OnceCell;

struct RegisteredDevice {
    burn: burn::tensor::Device,
    host: Option<Device>,
}

fn registered_wgpu(device: &RegisteredDevice) -> &WgpuDevice {
    let burn::backend::DispatchDevice::Wgpu(device) = device.burn.as_dispatch() else {
        unreachable!("the registered compute device is created by the wgpu backend")
    };
    device
}

fn host_matches<T: PartialEq>(registered: Option<&T>, requested: &T) -> bool {
    registered.is_some_and(|registered| registered == requested)
}

fn registered_host(
    registered: &RegisteredDevice,
    requested: &Device,
) -> Result<WgpuDevice, BurnInitDeviceError> {
    if host_matches(registered.host.as_ref(), requested) {
        Ok(registered_wgpu(registered).clone())
    } else {
        Err(BurnInitDeviceError::AlreadyInitialized)
    }
}

/// Free cached GPU memory on whichever runtime the device belongs to.
///
/// `memory_cleanup` / `memory_usage` live on the cubecl client, which is
/// runtime-specific, so this has to branch on the dispatch variant. Only wgpu
/// is wired up today; another cubecl runtime would add an arm here.
pub fn device_memory_cleanup(device: &burn::tensor::Device) {
    use burn::backend::DispatchDevice;
    use burn::cubecl::Runtime;
    if let DispatchDevice::Wgpu(d) = device.as_dispatch() {
        burn_wgpu::WgpuRuntime::<burn_wgpu::AutoCompiler>::client(d).memory_cleanup();
    }
}

/// Bytes currently reserved by the runtime's memory pool, if it reports them.
pub fn device_memory_usage(device: &burn::tensor::Device) -> Option<burn::cubecl::MemoryUsage> {
    use burn::backend::DispatchDevice;
    use burn::cubecl::Runtime;
    match device.as_dispatch() {
        DispatchDevice::Wgpu(d) => {
            Some(burn_wgpu::WgpuRuntime::<burn_wgpu::AutoCompiler>::client(d).memory_usage())
        }
        // Autodiff wraps a device rather than being one; nothing to report.
        DispatchDevice::Autodiff(_) => None,
    }
}

static DEVICE: OnceCell<RegisteredDevice> = OnceCell::const_new();

/// The compute device, opening it if this is the first call.
pub async fn device() -> &'static burn::tensor::Device {
    &DEVICE.get_or_init(open_device).await.burn
}

/// The compute device, but only if it is already open. For UI that wants to
/// report on the device without causing it to be created.
pub fn try_device() -> Option<&'static burn::tensor::Device> {
    DEVICE.get().map(|registered| &registered.burn)
}

/// Wait for the compute device, opening the default device if necessary.
///
/// This preserves the former registration entry point while following the
/// current lazy-initialization behavior.
pub async fn wait_for_device() -> WgpuDevice {
    burn_init_setup().await
}

#[cfg(test)]
mod device_registration_tests {
    use super::host_matches;

    #[test]
    fn host_identity_distinguishes_idempotence_from_conflicts() {
        assert!(host_matches(Some(&7_u8), &7));
        assert!(!host_matches(Some(&7_u8), &8));
        assert!(!host_matches(None, &7));
    }
}

fn is_training_source(vfs: &brush_vfs::BrushVfs, ply_count: usize) -> bool {
    if ply_count == 0 {
        return true;
    }

    // A PLY-only archive remains a viewer source even when it contains
    // ancillary files such as a README. Supported training datasets have both
    // source images and a recognizable camera-metadata file.
    const IMAGE_EXTENSIONS: &[&str] = &[
        "avif", "bmp", "exr", "gif", "jpeg", "jpg", "png", "pnm", "qoi", "tga", "tif", "tiff",
        "webp",
    ];
    let has_images = vfs.iter_files().any(|path| {
        path.extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| {
                IMAGE_EXTENSIONS
                    .iter()
                    .any(|candidate| extension.eq_ignore_ascii_case(candidate))
            })
    });
    let has_metadata = vfs.iter_files().any(|path| {
        let extension = path.extension().and_then(|extension| extension.to_str());
        extension.is_some_and(|extension| {
            extension.eq_ignore_ascii_case("json") || extension.eq_ignore_ascii_case("csv")
        }) || path.file_name().is_some_and(|name| {
            name.eq_ignore_ascii_case("cameras.bin") || name.eq_ignore_ascii_case("cameras.txt")
        })
    });

    has_images && has_metadata
}

/// Create a running process from a datasource and args.
///
/// The `config_fn` callback receives the initial config (loaded from
/// args.txt if present, otherwise defaults) and returns the final
/// config to use. This allows the caller to modify or override
/// settings as needed.
pub fn create_process<
    Fun: FnOnce(crate::config::TrainStreamConfig) -> Fut + SendNotWasm + 'static,
    Fut: Future<Output = Option<crate::config::TrainStreamConfig>> + SendNotWasm,
>(
    source: DataSource,
    config_fn: Fun,
) -> RunningProcess {
    let (splat_tx, splat_view) = crate::slot::channel();

    let stream =
        try_fn_stream(
            |emitter| async move { run_process(source, config_fn, &emitter, splat_tx).await },
        );

    RunningProcess {
        stream: Box::pin(stream),
        splat_view,
    }
}

async fn run_process<
    Fun: FnOnce(crate::config::TrainStreamConfig) -> Fut + SendNotWasm + 'static,
    Fut: Future<Output = Option<crate::config::TrainStreamConfig>>,
>(
    source: DataSource,
    config_fn: Fun,
    emitter: &Emitter,
    splat_view: SlotSender<Splats>,
) -> Result<(), Error> {
    log::info!("Starting process with source {source:?}");
    emitter.emit(ProcessMessage::NewProcess).await;

    let vfs = source.clone().into_vfs().await?;
    let vfs_counts = vfs.file_count();

    if vfs_counts == 0 {
        return Err(anyhow::anyhow!("No files found."));
    }

    let ply_count = vfs.files_with_extension("ply").count();

    log::info!(
        "Mounted VFS with {} files. (plys: {})",
        vfs.file_count(),
        ply_count
    );

    let is_training = is_training_source(&vfs, ply_count);

    // Emit source info - just the display name
    let paths: Vec<_> = vfs.file_paths().collect();
    let source_name = if let Some(base_path) = vfs.base_path() {
        base_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(if is_training { "dataset" } else { "file" })
            .to_owned()
    } else if paths.len() == 1 {
        paths[0]
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("input.ply")
            .to_owned()
    } else {
        format!("{} files", paths.len())
    };

    let base_path = vfs.base_path();

    // Load initial config from args.txt via VFS if present
    let initial_config = args_file::load_config_from_vfs(&vfs).await;

    emitter
        .emit(ProcessMessage::StartLoading {
            name: source_name,
            source,
            training: is_training,
            base_path,
        })
        .await;

    if !is_training {
        let device = device().await;
        let mut paths: Vec<_> = vfs.files_with_extension("ply").collect();
        alphanumeric_sort::sort_path_slice(&mut paths);
        let total_frames = paths.len() as u32;

        for (frame, path) in paths.iter().enumerate() {
            log::info!("Loading single ply file");

            let mut splat_stream = pin!(brush_serde::stream_splat_from_ply(
                vfs.reader_at_path(path).await?,
                None,
                true,
            ));

            while let Some(message) = splat_stream.next().await {
                let message = message?;

                let mode = message.meta.render_mode.unwrap_or(SplatRenderMode::Default);
                let splats = message.data.into_splats(device, mode);

                // As loading concatenates splats each time, memory usage tends to accumulate a lot
                // over time. Clear out memory after each step to prevent this buildup.
                device_memory_cleanup(device);

                // For the first frame of a new file, clear existing frames
                if frame == 0 {
                    splat_view.clear();
                }

                // Capture stats before moving splats
                let num_splats = splats.num_splats();
                let sh_degree = splats.sh_degree();
                splat_view.set(frame, splats);

                emitter
                    .emit(ProcessMessage::SplatsUpdated {
                        up_axis: message.meta.up_axis,
                        frame: frame as u32,
                        total_frames,
                        num_splats,
                        sh_degree,
                    })
                    .await;
            }
        }

        emitter.emit(ProcessMessage::DoneLoading).await;
    } else {
        // Pass initial config (from args.txt or defaults) to the callback.
        // Returning `None` from `config_fn` aborts cleanly without
        // surfacing as an error.
        let base_config = initial_config.unwrap_or_default();
        let Some(config) = config_fn(base_config).await else {
            log::info!("config_fn returned None — aborting before training");
            return Ok(());
        };
        train_stream(vfs, config, emitter, splat_view).await?;
    };

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::is_training_source;
    use brush_vfs::BrushVfs;
    use std::path::PathBuf;

    #[test]
    fn ancillary_files_do_not_turn_ply_animations_into_datasets() {
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("000.ply"),
            PathBuf::from("001.ply"),
            PathBuf::from("README.txt"),
        ]);
        assert!(!is_training_source(&vfs, 2));
    }

    #[test]
    fn ply_initialization_with_dataset_metadata_still_trains() {
        for paths in [
            vec!["init.ply", "transforms.json", "images/0001.png"],
            vec!["init.ply", "cameras.bin", "images/0001.jpg"],
            vec!["init.ply", "cameras.csv", "images/0001.tif"],
        ] {
            let vfs = BrushVfs::create_test_vfs(paths.into_iter().map(PathBuf::from).collect());
            assert!(is_training_source(&vfs, 1));
        }
    }
}
