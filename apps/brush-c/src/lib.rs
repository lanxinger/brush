// brush-c is a native-only FFI shim. The crate compiles to an empty stub on wasm.
#![cfg(not(target_family = "wasm"))]

use brush_process::DataSource;
use brush_process::burn_init_setup;
use brush_process::config::TrainStreamConfig;
use brush_process::message::TrainMessage;
use brush_process::{create_process, message::ProcessMessage};
use brush_render::AlphaMode;
use std::convert::TryFrom;
use std::ffi::{CStr, c_char, c_void};
use tokio::sync::OnceCell;
use tokio_stream::StreamExt;

#[repr(C)]
pub enum TrainExitCode {
    Success = 0,
    Error = 1,
}

#[repr(C)]
pub enum ProgressMessage {
    NewProcess,
    Training { iter: u32 },
    DoneTraining,
}

impl TryFrom<ProcessMessage> for ProgressMessage {
    type Error = ();

    fn try_from(value: ProcessMessage) -> Result<Self, Self::Error> {
        match value {
            ProcessMessage::NewProcess => Ok(Self::NewProcess),
            ProcessMessage::TrainMessage(TrainMessage::TrainStep { iter, .. }) => {
                Ok(Self::Training { iter })
            }
            ProcessMessage::TrainMessage(TrainMessage::DoneTraining) => Ok(Self::DoneTraining),
            _ => Err(()),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TrainOptions {
    pub total_train_steps: u32,
    pub refine_every: u32,
    pub max_resolution: u32,
    pub export_every: u32,
    pub output_path: *const c_char,
}

impl TrainOptions {
    /// # Safety
    ///
    /// If `output_path` is not null, it must be a valid pointer to a null-terminated C string.
    unsafe fn into_train_stream_config(self) -> TrainStreamConfig {
        // SAFETY: The caller upholds the same output-path invariant.
        unsafe {
            base_train_stream_config(
                self.total_train_steps,
                self.refine_every,
                self.max_resolution,
                self.export_every,
                self.output_path,
            )
        }
    }
}

/// Extended options for memory-constrained hosts.
///
/// This is a separate ABI from [`TrainOptions`] so existing callers keep their
/// original layout. Use it with [`train_and_save_v2`].
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TrainOptionsV2 {
    pub total_train_steps: u32,
    pub refine_every: u32,
    pub max_resolution: u32,
    pub export_every: u32,
    /// 0 = automatic, 1 = masked, 2 = transparent.
    pub alpha_mode: u32,
    /// Upper bound on splat count. Zero keeps Brush's default.
    pub max_splats: u32,
    pub output_path: *const c_char,
}

impl TrainOptionsV2 {
    /// # Safety
    ///
    /// If `output_path` is not null, it must be a valid pointer to a null-terminated C string.
    unsafe fn into_train_stream_config(self) -> Result<TrainStreamConfig, &'static str> {
        // SAFETY: The caller upholds the same output-path invariant.
        let mut process_args = unsafe {
            base_train_stream_config(
                self.total_train_steps,
                self.refine_every,
                self.max_resolution,
                self.export_every,
                self.output_path,
            )
        };
        process_args.load_config.alpha_mode = match self.alpha_mode {
            0 => None,
            1 => Some(AlphaMode::Masked),
            2 => Some(AlphaMode::Transparent),
            _ => return Err("alpha_mode must be 0 (auto), 1 (masked), or 2 (transparent)"),
        };
        if self.max_splats != 0 {
            process_args.train_config.max_splats = self.max_splats;
        }
        Ok(process_args)
    }
}

/// # Safety
///
/// If `output_path` is not null, it must point to a valid null-terminated C string.
unsafe fn base_train_stream_config(
    total_train_steps: u32,
    refine_every: u32,
    max_resolution: u32,
    export_every: u32,
    output_path: *const c_char,
) -> TrainStreamConfig {
    let mut process_args = TrainStreamConfig::default();
    if !output_path.is_null() {
        // SAFETY: Path is not null, caller guarantees the string is a valid C-string.
        process_args.process_config.export_path =
            unsafe { CStr::from_ptr(output_path).to_string_lossy().into_owned() };
    }
    process_args.train_config.total_train_iters = total_train_steps;
    process_args.train_config.refine_every = refine_every;
    process_args.load_config.max_resolution = max_resolution;
    process_args.process_config.export_every = export_every;
    process_args.process_config.eval_save_to_disk = true;
    process_args
}

pub type ProgressCallback =
    extern "C" fn(progress_message: ProgressMessage, user_data: *mut c_void);

static SETUP: OnceCell<()> = OnceCell::const_new();

fn run_training(
    dataset_path: String,
    process_args: TrainStreamConfig,
    progress_callback: Option<ProgressCallback>,
    user_data: *mut c_void,
) -> TrainExitCode {
    let source = DataSource::Path(dataset_path);
    let mut process = create_process(source, async move |_| Some(process_args));

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("Failed to create tokio runtime")
        .block_on(async {
            SETUP
                .get_or_init(async move || {
                    burn_init_setup().await;
                })
                .await;

            while let Some(message_result) = process.stream.next().await {
                match message_result {
                    Ok(message) => {
                        if let (Some(progress_callback), Ok(progress_message)) =
                            (progress_callback, message.try_into())
                        {
                            progress_callback(progress_message, user_data);
                        }
                    }
                    Err(error) => {
                        eprintln!("brush-c: process stream error: {error:#}");
                        return TrainExitCode::Error;
                    }
                }
            }

            TrainExitCode::Success
        })
}

/// Trains a model from a dataset and saves the result.
///
/// This function is designed to be called from other languages via FFI. It will
/// block the current thread until training is complete.
///
/// # Arguments
///
/// * `dataset_path` - A pointer to a null-terminated C string representing the path to the dataset.
/// * `options` - A pointer to a `TrainOptions` struct.
/// * `progress_callback` - An optional callback invoked with progress updates.
/// * `user_data` - An opaque pointer passed to the `progress_callback`.
///
/// # Safety
///
/// The caller must uphold several invariants. Passing `null` for `dataset_path` or `options`
/// is safe and will result in an error code, but if they are non-null, they must be valid.
///
/// - If `dataset_path` is not null, it must point to a valid, null-terminated C string. The
///   memory it points to must be valid for reading for the duration of this call.
///
/// - If `options` is not null, it must point to a valid `TrainOptions` struct. The memory it
///   points to must be valid for reading for the duration of this call. It's `output_path` must
///   be a valid, null-terminated C string if not null.
///
/// - When `progress_callback` is present, the `user_data` pointer is passed to it but is not
///   dereferenced by this function. If it is not null, the caller must ensure it points to memory
///   that remains valid for the entire duration of this function call, as the callback may
///   dereference it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn train_and_save(
    dataset_path: *const c_char,
    options: *const TrainOptions,
    progress_callback: Option<ProgressCallback>,
    user_data: *mut c_void,
) -> TrainExitCode {
    if dataset_path.is_null() || options.is_null() {
        return TrainExitCode::Error;
    }

    // A Rust panic must not unwind across this `extern "C"` boundary (that
    // aborts the whole process). Catch it and surface it as an error code.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let dataset_path_str =
            // SAFETY: Checked if dataset_path is not null, caller guarantees the string is a valid C-string.
            unsafe { CStr::from_ptr(dataset_path).to_string_lossy().into_owned() };

        // SAFETY: Option is checked to not be null before the future.
        let train_options = unsafe { *options };
        // SAFETY: Caller guarantees the output_path is a valid C-string if not null.
        let process_args = unsafe { train_options.into_train_stream_config() };
        run_training(dataset_path_str, process_args, progress_callback, user_data)
    }));

    result.unwrap_or(TrainExitCode::Error)
}

/// Trains and saves using [`TrainOptionsV2`].
///
/// Like [`train_and_save`], this call blocks its calling thread. The v2
/// options add explicit alpha handling and a splat-count cap without changing
/// the original C ABI.
///
/// # Safety
///
/// `dataset_path` and `options` may be null, which returns an error. Otherwise
/// they must point to valid values for the duration of this call. A non-null
/// `options.output_path` must be a valid null-terminated C string. If a
/// callback is supplied, `user_data` must remain valid for every callback.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn train_and_save_v2(
    dataset_path: *const c_char,
    options: *const TrainOptionsV2,
    progress_callback: Option<ProgressCallback>,
    user_data: *mut c_void,
) -> TrainExitCode {
    if dataset_path.is_null() || options.is_null() {
        return TrainExitCode::Error;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: The pointer was checked for null and the caller guarantees a valid C string.
        let dataset_path = unsafe { CStr::from_ptr(dataset_path).to_string_lossy().into_owned() };
        // SAFETY: The pointer was checked for null and the caller guarantees a valid struct.
        let train_options = unsafe { *options };
        // SAFETY: Caller guarantees output_path is a valid C string when non-null.
        let process_args = match unsafe { train_options.into_train_stream_config() } {
            Ok(process_args) => process_args,
            Err(error) => {
                eprintln!("brush-c: invalid v2 options: {error}");
                return TrainExitCode::Error;
            }
        };
        run_training(dataset_path, process_args, progress_callback, user_data)
    }));

    result.unwrap_or(TrainExitCode::Error)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(alpha_mode: u32, max_splats: u32) -> TrainOptionsV2 {
        TrainOptionsV2 {
            total_train_steps: 1_500,
            refine_every: 200,
            max_resolution: 1_024,
            export_every: 250,
            alpha_mode,
            max_splats,
            output_path: std::ptr::null(),
        }
    }

    #[test]
    fn v2_options_apply_mobile_memory_controls() {
        // SAFETY: The output path is null.
        let config = unsafe { options(1, 220_000).into_train_stream_config() }.unwrap();

        assert_eq!(config.load_config.alpha_mode, Some(AlphaMode::Masked));
        assert_eq!(config.train_config.max_splats, 220_000);
    }

    #[test]
    fn v2_zero_values_keep_automatic_alpha_and_default_splat_cap() {
        let default_max_splats = TrainStreamConfig::default().train_config.max_splats;
        // SAFETY: The output path is null.
        let config = unsafe { options(0, 0).into_train_stream_config() }.unwrap();

        assert_eq!(config.load_config.alpha_mode, None);
        assert_eq!(config.train_config.max_splats, default_max_splats);
    }

    #[test]
    fn v2_rejects_unknown_alpha_mode() {
        // SAFETY: The output path is null.
        let result = unsafe { options(3, 220_000).into_train_stream_config() };

        assert!(result.is_err());
    }
}
