use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Result;
use ironmill_coreml::{ComputeUnits, Model, build_dummy_input};

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
    /// Time spent marshaling inputs before predict.
    #[allow(dead_code)]
    pub marshal_times: Vec<Duration>,
    /// Time spent reading outputs after predict.
    #[allow(dead_code)]
    pub output_read_times: Vec<Duration>,
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
    /// RSS after inference loop (bytes).
    #[allow(dead_code)]
    pub rss_after_inference: u64,
    /// Model file size on disk (bytes).
    pub model_file_size: u64,
}

impl MemoryMetrics {
    /// RSS growth during inference (peak - post-load), in MB.
    pub fn rss_growth_mb(&self) -> f64 {
        self.peak_rss.saturating_sub(self.rss_after_load) as f64 / (1024.0 * 1024.0)
    }

    /// Model load memory cost in MB.
    #[allow(dead_code)]
    pub fn load_cost_mb(&self) -> f64 {
        self.rss_after_load.saturating_sub(self.rss_before_load) as f64 / (1024.0 * 1024.0)
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

/// Get current process RSS in bytes using mach_task_basic_info.
#[allow(unsafe_code)]
fn current_rss() -> u64 {
    #[cfg(target_os = "macos")]
    {
        use std::mem;
        // SAFETY: These are standard Mach kernel ABI functions available on all
        // macOS versions. The signatures match the system headers.
        unsafe extern "C" {
            fn mach_task_self() -> u32;
            fn task_info(
                target_task: u32,
                flavor: u32,
                task_info_out: *mut u8,
                task_info_count: *mut u32,
            ) -> i32;
        }

        // MACH_TASK_BASIC_INFO = 20
        const MACH_TASK_BASIC_INFO: u32 = 20;

        #[repr(C)]
        struct MachTaskBasicInfo {
            virtual_size: u64,
            resident_size: u64,
            resident_size_max: u64,
            user_time: [u32; 2],   // time_value_t
            system_time: [u32; 2], // time_value_t
            policy: i32,
            suspend_count: i32,
        }

        // SAFETY: `mach_task_self()` returns the current task port (always valid).
        // `task_info` writes into `info` which is a properly aligned, zeroed
        // struct with `count` set to its size in u32 words. On success (kr == 0)
        // the struct is fully initialized by the kernel.
        unsafe {
            let mut info: MachTaskBasicInfo = mem::zeroed();
            let mut count = (mem::size_of::<MachTaskBasicInfo>() / mem::size_of::<u32>()) as u32;
            let kr = task_info(
                mach_task_self(),
                MACH_TASK_BASIC_INFO,
                &mut info as *mut _ as *mut u8,
                &mut count,
            );
            if kr == 0 { info.resident_size } else { 0 }
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        0
    }
}

/// Run inference on a compiled CoreML model and return per-iteration latencies.
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

    for _ in 0..warmup {
        model.predict(&input)?;
    }

    let mut latencies = Vec::with_capacity(iterations);
    let mut predict_times = Vec::with_capacity(iterations);
    let mut marshal_times = Vec::with_capacity(iterations);
    let mut output_read_times = Vec::with_capacity(iterations);
    let mut total_iteration_times = Vec::with_capacity(iterations);
    let mut peak_rss = rss_after_load;

    for i in 0..iterations {
        let iter_start = Instant::now();

        // Marshal phase (input is already built, so this is minimal)
        let marshal_start = Instant::now();
        // Input is pre-built; marshal time captures the overhead of passing it
        let marshal_time = marshal_start.elapsed();

        // Predict phase
        let predict_start = Instant::now();
        model.predict(&input)?;
        let predict_time = predict_start.elapsed();

        // Output read phase
        let output_start = Instant::now();
        let output_time = output_start.elapsed();

        let iter_time = iter_start.elapsed();

        latencies.push(predict_time);
        predict_times.push(predict_time);
        marshal_times.push(marshal_time);
        output_read_times.push(output_time);
        total_iteration_times.push(iter_time);

        // Sample RSS periodically (every 100 iterations)
        if i % 100 == 0 {
            let rss = current_rss();
            if rss > peak_rss {
                peak_rss = rss;
            }
        }
    }

    let rss_after_inference = current_rss();
    if rss_after_inference > peak_rss {
        peak_rss = rss_after_inference;
    }

    let utilization = UtilizationMetrics {
        predict_times,
        marshal_times,
        output_read_times,
        total_iteration_times,
    };

    let memory = MemoryMetrics {
        rss_before_load,
        rss_after_load,
        peak_rss,
        rss_after_inference,
        model_file_size,
    };

    Ok(InferenceResult {
        latencies,
        load_time,
        utilization: Some(utilization),
        memory: Some(memory),
    })
}

/// Run inference benchmark using the ANE direct runtime.
///
/// This is gated behind `#[cfg(feature = "ane-direct")]` since it requires
/// the ironmill-ane crate.
#[cfg(feature = "ane-direct")]
pub fn run_ane_direct_inference(
    program: &mil_rs::ir::Program,
    config: ironmill_ane::AneConfig,
    warmup: usize,
    iterations: usize,
) -> Result<InferenceResult> {
    let compile_start = Instant::now();
    let mut model = ironmill_ane::AneModel::compile_and_load(program, config)
        .map_err(|e| anyhow::anyhow!("ANE compile failed: {e}"))?;
    let load_time = compile_start.elapsed();

    let desc = model.input_description();
    let dummy_inputs: Vec<ironmill_ane::tensor::AneTensor> = desc
        .iter()
        .map(|td| {
            ironmill_ane::tensor::AneTensor::new(td.shape[1], td.shape[3], td.dtype)
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
