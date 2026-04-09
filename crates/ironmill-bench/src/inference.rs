use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Result;
use ironmill_inference::coreml_runtime::{ComputeUnits, Model, build_dummy_input};

/// Results from a single inference benchmark run.
pub struct InferenceResult {
    pub latencies: Vec<Duration>,
    pub load_time: Duration,
    /// Per-iteration timing breakdown.
    pub utilization: Option<UtilizationMetrics>,
    /// Memory footprint measurements.
    pub memory: Option<MemoryMetrics>,
}

/// Timing breakdown for ANE utilization analysis.
#[derive(Debug, Clone)]
pub struct UtilizationMetrics {
    /// Time spent in model.predict() calls (the compute portion).
    pub predict_times: Vec<Duration>,
    /// Total iteration time (marshal + predict + output_read).
    pub total_iteration_times: Vec<Duration>,
}

impl UtilizationMetrics {
    /// ANE utilization as percentage: predict_time / total_time × 100.
    pub fn utilization_pct(&self) -> f64 {
        let total_predict: f64 = self.predict_times.iter().map(|d| d.as_secs_f64()).sum();
        let total_iter: f64 = self
            .total_iteration_times
            .iter()
            .map(|d| d.as_secs_f64())
            .sum();
        if total_iter > 0.0 {
            (total_predict / total_iter) * 100.0
        } else {
            0.0
        }
    }

    /// Average dispatch overhead in milliseconds per iteration.
    pub fn dispatch_overhead_ms(&self) -> f64 {
        let n = self.predict_times.len() as f64;
        if n == 0.0 {
            return 0.0;
        }
        let total_overhead: f64 = self
            .total_iteration_times
            .iter()
            .zip(self.predict_times.iter())
            .map(|(total, predict)| total.as_secs_f64() - predict.as_secs_f64())
            .sum();
        (total_overhead / n) * 1000.0
    }
}

/// Memory footprint measurements across the inference lifecycle.
#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    /// RSS before model load (bytes).
    pub rss_before_load: u64,
    /// RSS after model load (bytes).
    pub rss_after_load: u64,
    /// Peak RSS during inference (bytes).
    pub peak_rss: u64,
    /// Model file size on disk (bytes).
    pub model_file_size: u64,
}

impl MemoryMetrics {
    /// RSS growth during inference (peak - post-load), in MB.
    pub fn rss_growth_mb(&self) -> f64 {
        self.peak_rss.saturating_sub(self.rss_after_load) as f64 / (1024.0 * 1024.0)
    }

    /// Memory efficiency ratio: model_file_size / runtime_rss_delta.
    pub fn efficiency_ratio(&self) -> f64 {
        let delta = self.rss_after_load.saturating_sub(self.rss_before_load);
        if delta > 0 {
            self.model_file_size as f64 / delta as f64
        } else {
            0.0
        }
    }
}

/// Get current process RSS in bytes.
fn current_rss() -> u64 {
    #[cfg(target_os = "macos")]
    {
        ironmill_ane_sys::process::current_rss()
    }
    #[cfg(not(target_os = "macos"))]
    {
        0
    }
}

/// Run inference on a compiled CoreML model and return per-iteration latencies.
///
/// Loads the model once and runs all iterations on it. If you need to run
/// multiple independent "runs", call this once with the full iteration count
/// or use [`run_inference_on_model`] to reuse an already-loaded model.
#[allow(dead_code)]
pub fn run_inference(
    mlmodelc_path: &Path,
    compute_units: ComputeUnits,
    iterations: usize,
    warmup: usize,
) -> Result<InferenceResult> {
    let model_file_size = std::fs::metadata(mlmodelc_path)
        .map(|m| m.len())
        .unwrap_or(0);

    let rss_before_load = current_rss();

    let load_start = Instant::now();
    let model = Model::load(mlmodelc_path, compute_units)?;
    let load_time = load_start.elapsed();

    let rss_after_load = current_rss();

    let desc = model.input_description()?;
    let input = build_dummy_input(&desc)?;

    let run_result = run_inference_on_model(&model, &input, iterations, warmup)?;

    let mut result = run_result;
    result.load_time = load_time;
    result.memory = Some(MemoryMetrics {
        rss_before_load,
        rss_after_load,
        peak_rss: result
            .memory
            .as_ref()
            .map_or(rss_after_load, |m| m.peak_rss),
        model_file_size,
    });

    Ok(result)
}

/// Run multiple independent benchmark runs on the same compiled CoreML model,
/// loading it only once.
///
/// Returns one [`InferenceResult`] per run. The `load_time` in each result
/// reflects the single model load (not per-run).
pub fn run_inference_multi_run(
    mlmodelc_path: &Path,
    compute_units: ComputeUnits,
    iterations: usize,
    warmup: usize,
    runs: usize,
) -> Result<(Vec<InferenceResult>, std::time::Duration)> {
    let model_file_size = std::fs::metadata(mlmodelc_path)
        .map(|m| m.len())
        .unwrap_or(0);

    let rss_before_load = current_rss();

    let load_start = Instant::now();
    let model = Model::load(mlmodelc_path, compute_units)?;
    let load_time = load_start.elapsed();

    let rss_after_load = current_rss();

    let desc = model.input_description()?;
    let input = build_dummy_input(&desc)?;

    let mut results = Vec::with_capacity(runs);
    for _ in 0..runs {
        let mut run_result = run_inference_on_model(&model, &input, iterations, warmup)?;
        run_result.load_time = load_time;
        run_result.memory = Some(MemoryMetrics {
            rss_before_load,
            rss_after_load,
            peak_rss: run_result
                .memory
                .as_ref()
                .map_or(rss_after_load, |m| m.peak_rss),
            model_file_size,
        });
        results.push(run_result);
    }

    Ok((results, load_time))
}

/// Run inference iterations on an already-loaded model.
fn run_inference_on_model(
    model: &Model,
    input: &ironmill_inference::coreml_runtime::PredictionInput,
    iterations: usize,
    warmup: usize,
) -> Result<InferenceResult> {
    for _ in 0..warmup {
        model.predict(input)?;
    }

    let rss_baseline = current_rss();
    let mut latencies = Vec::with_capacity(iterations);
    let mut total_iteration_times = Vec::with_capacity(iterations);
    let mut peak_rss = rss_baseline;

    for i in 0..iterations {
        let iter_start = Instant::now();

        let predict_start = Instant::now();
        model.predict(input)?;
        let predict_time = predict_start.elapsed();

        let iter_time = iter_start.elapsed();

        latencies.push(predict_time);
        total_iteration_times.push(iter_time);

        if i % 100 == 0 {
            let rss = current_rss();
            if rss > peak_rss {
                peak_rss = rss;
            }
        }
    }

    let rss_final = current_rss();
    if rss_final > peak_rss {
        peak_rss = rss_final;
    }

    let utilization = UtilizationMetrics {
        predict_times: latencies.clone(),
        total_iteration_times,
    };

    Ok(InferenceResult {
        latencies,
        load_time: Duration::ZERO,
        utilization: Some(utilization),
        memory: Some(MemoryMetrics {
            rss_before_load: 0,
            rss_after_load: 0,
            peak_rss,
            model_file_size: 0,
        }),
    })
}

/// Run inference benchmark using the ANE direct runtime.
///
/// This is gated behind `#[cfg(feature = "ane-direct")]` since it requires
/// the ironmill-ane crate.
#[cfg(feature = "ane-direct")]
pub fn run_ane_direct_inference(
    program: &mil_rs::ir::Program,
    config: ironmill_inference::AneConfig,
    warmup: usize,
    iterations: usize,
) -> Result<InferenceResult> {
    let compile_start = Instant::now();
    let device = std::sync::Arc::new(
        ironmill_inference::ane::HardwareAneDevice::new()
            .map_err(|e| anyhow::anyhow!("ANE device init failed: {e}"))?,
    );
    let bundle = ironmill_compile::ane::bundle::compile_model_bundle(
        program,
        &ironmill_compile::ane::bundle::AneCompileConfig::default(),
    )
    .map_err(|e| anyhow::anyhow!("ANE compile failed: {e}"))?;
    let tmp = tempfile::tempdir()?;
    let bundle_path = tmp.path().join("model.ironml");
    bundle
        .save(&bundle_path)
        .map_err(|e| anyhow::anyhow!("failed to save bundle: {e}"))?;
    let mut model = ironmill_inference::AneModel::from_bundle(device, &bundle_path, config)
        .map_err(|e| anyhow::anyhow!("ANE load failed: {e}"))?;
    let load_time = compile_start.elapsed();

    let desc = model.input_description();
    let dummy_inputs: Vec<ironmill_iosurface::AneTensor> = desc
        .iter()
        .map(|td| {
            ironmill_iosurface::AneTensor::new(td.shape[1], td.shape[3], td.dtype)
                .map_err(|e| anyhow::anyhow!("tensor creation failed: {e}"))
        })
        .collect::<Result<Vec<_>>>()?;

    for _ in 0..warmup {
        let _ = model.predict(&dummy_inputs);
    }

    let mut latencies = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = model.predict(&dummy_inputs)?;
        latencies.push(start.elapsed());
    }

    Ok(InferenceResult {
        latencies,
        load_time,
        utilization: None,
        memory: None,
    })
}
