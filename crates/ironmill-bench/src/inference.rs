use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Result;
use ironmill_coreml::{ComputeUnits, Model, build_dummy_input};

/// Results from a single inference benchmark run.
#[allow(dead_code)]
pub struct InferenceResult {
    pub latencies: Vec<Duration>,
    pub load_time: Duration,
}

/// Run inference on a compiled CoreML model and return per-iteration latencies.
pub fn run_inference(
    mlmodelc_path: &Path,
    compute_units: ComputeUnits,
    iterations: usize,
    warmup: usize,
) -> Result<InferenceResult> {
    let load_start = Instant::now();
    let model = Model::load(mlmodelc_path, compute_units)?;
    let load_time = load_start.elapsed();

    let desc = model.input_description()?;
    let input = build_dummy_input(&desc)?;

    for _ in 0..warmup {
        model.predict(&input)?;
    }

    let mut latencies = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        model.predict(&input)?;
        latencies.push(start.elapsed());
    }

    Ok(InferenceResult {
        latencies,
        load_time,
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
    })
}
