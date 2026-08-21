use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_EVERY: u32 = 50;
const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Serialize)]
struct RunRecord<'a> {
    schema_version: u32,
    record_type: &'static str,
    run_id: &'a str,
    package_version: &'static str,
    build_id: &'static str,
    dataset: &'a str,
    start_iter: u32,
    total_iters: u32,
    metrics_every: u32,
    started_unix_ms: u128,
}

#[derive(Debug, Serialize)]
struct StepRecord<'a> {
    schema_version: u32,
    record_type: &'static str,
    run_id: &'a str,
    iter: u32,
    total_iters: u32,
    loss: Option<f32>,
    loss_status: &'static str,
    splats: u32,
    step_ms: f64,
    elapsed_ms: f64,
    final_step: bool,
}

pub(crate) struct TrainingMetrics {
    writer: BufWriter<File>,
    run_id: String,
    start_iter: u32,
    total_iters: u32,
    every: u32,
}

impl TrainingMetrics {
    pub(crate) fn from_env(dataset: &str, start_iter: u32, total_iters: u32) -> Option<Self> {
        let path = std::env::var_os("BRUSH_METRICS_LOG").map(PathBuf::from)?;
        let every = metrics_every_from_env();
        let started_unix_ms = unix_time_ms();
        let run_id = format!("{started_unix_ms}-{}", std::process::id());
        let file = match OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&path)
        {
            Ok(file) => file,
            Err(error) => {
                log::warn!(
                    "Could not open BRUSH_METRICS_LOG at {}: {error}",
                    path.display()
                );
                return None;
            }
        };

        let mut metrics = Self {
            writer: BufWriter::new(file),
            run_id,
            start_iter,
            total_iters,
            every,
        };
        let start = RunRecord {
            schema_version: SCHEMA_VERSION,
            record_type: "run_start",
            run_id: &metrics.run_id,
            package_version: env!("CARGO_PKG_VERSION"),
            build_id: crate::BUILD_ID,
            dataset,
            start_iter,
            total_iters,
            metrics_every: every,
            started_unix_ms,
        };
        if let Err(error) = write_json_line(&mut metrics.writer, &start) {
            log::warn!(
                "Could not initialize BRUSH_METRICS_LOG at {}: {error}",
                path.display()
            );
            return None;
        }

        log::info!("Writing training metrics to {}", path.display());
        Some(metrics)
    }

    pub(crate) fn should_record(&self, iter: u32, final_step: bool) -> bool {
        should_record(iter, self.start_iter, self.every, final_step)
    }

    pub(crate) fn record_step(
        &mut self,
        iter: u32,
        loss: f32,
        splats: u32,
        step_ms: f64,
        elapsed_ms: f64,
        final_step: bool,
    ) -> std::io::Result<()> {
        let (loss, loss_status) = finite_value(loss);
        let record = StepRecord {
            schema_version: SCHEMA_VERSION,
            record_type: "train_step",
            run_id: &self.run_id,
            iter,
            total_iters: self.total_iters,
            loss,
            loss_status,
            splats,
            step_ms,
            elapsed_ms,
            final_step,
        };
        write_json_line(&mut self.writer, &record)
    }
}

fn metrics_every_from_env() -> u32 {
    let Some(raw) = std::env::var_os("BRUSH_METRICS_EVERY") else {
        return DEFAULT_EVERY;
    };
    let raw = raw.to_string_lossy();
    match raw.parse::<u32>() {
        Ok(value) if value > 0 => value,
        _ => {
            log::warn!("Ignoring invalid BRUSH_METRICS_EVERY={raw:?}; using {DEFAULT_EVERY}");
            DEFAULT_EVERY
        }
    }
}

fn unix_time_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn should_record(iter: u32, start_iter: u32, every: u32, final_step: bool) -> bool {
    iter == start_iter.saturating_add(1) || iter.is_multiple_of(every) || final_step
}

fn finite_value(value: f32) -> (Option<f32>, &'static str) {
    if value.is_finite() {
        (Some(value), "finite")
    } else if value.is_nan() {
        (None, "nan")
    } else if value.is_sign_positive() {
        (None, "positive_infinity")
    } else {
        (None, "negative_infinity")
    }
}

fn write_json_line(writer: &mut impl Write, value: &impl Serialize) -> std::io::Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_first_cadence_and_final_steps() {
        assert!(should_record(8, 7, 50, false));
        assert!(!should_record(49, 7, 50, false));
        assert!(should_record(50, 7, 50, false));
        assert!(should_record(73, 7, 50, true));
    }

    #[test]
    fn serializes_non_finite_loss_as_null_with_status() {
        let (loss, loss_status) = finite_value(f32::NAN);
        let record = StepRecord {
            schema_version: SCHEMA_VERSION,
            record_type: "train_step",
            run_id: "test-run",
            iter: 1,
            total_iters: 1,
            loss,
            loss_status,
            splats: 10,
            step_ms: 1.0,
            elapsed_ms: 1.0,
            final_step: true,
        };
        let value = serde_json::to_value(record).expect("record should serialize");

        assert!(value["loss"].is_null());
        assert_eq!(value["loss_status"], "nan");
    }
}
