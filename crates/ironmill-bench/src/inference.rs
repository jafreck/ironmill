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
