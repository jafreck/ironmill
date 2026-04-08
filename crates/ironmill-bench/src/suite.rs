//! Benchmark suite trait and registry.
//!
//! Every benchmark type (decode latency, prefill throughput, quality, perplexity,
//! etc.) implements [`BenchmarkSuite`]. The runner discovers registered suites,
//! filters by config, and executes them uniformly.

use std::collections::HashMap;

use anyhow::Result;
use serde::Serialize;

use crate::config::BenchMatrix;
use crate::stats::{AggregatedResult, aggregate_runs, compute_stats};

/// Identifies which backend a suite targets.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "kebab-case")]
#[allow(dead_code)]
pub enum BackendKind {
    CoremlCpu,
    CoremlGpu,
    CoremlAne,
    CoremlAll,
    Metal,
    AneDirect,
}

impl std::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendKind::CoremlCpu => write!(f, "coreml-cpu"),
            BackendKind::CoremlGpu => write!(f, "coreml-gpu"),
            BackendKind::CoremlAne => write!(f, "coreml-ane"),
            BackendKind::CoremlAll => write!(f, "coreml-all"),
            BackendKind::Metal => write!(f, "metal"),
            BackendKind::AneDirect => write!(f, "ane-direct"),
        }
    }
}

/// A single benchmark result from any suite.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkResult {
    /// Which suite produced this result.
    pub suite: String,
    /// Model name.
    pub model: String,
    /// Optimization config name.
    pub optimization: String,
    /// Backend identifier (e.g. "metal-baseline", "coreml-ane").
    pub backend: String,
    /// Sub-label for variant benchmarks (e.g. "prefill-1024", "ctx-4096").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant: Option<String>,
    /// Aggregated latency/throughput statistics.
    pub result: AggregatedResult,
    /// GPU memory in MB (if measurable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_memory_mb: Option<f64>,
    /// Model load time in ms.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_time_ms: Option<f64>,
    /// Perplexity (for quality/perplexity suites).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub perplexity: Option<f64>,
    /// Extra key-value metadata specific to the suite.
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
}

/// Context passed to every suite's `run()` method.
pub struct BenchmarkContext<'a> {
    pub matrix: &'a BenchMatrix,
    pub cache_dir: &'a std::path::Path,
    pub no_cache: bool,
    pub power_enabled: bool,
    pub idle_power: &'a Option<crate::power::PowerMetrics>,
    /// Extra CLI-level options suites may need.
    pub extra: HashMap<String, String>,
}

/// The core trait every benchmark suite implements.
///
/// # Adding a new benchmark
///
/// 1. Create a new file in `suites/` (e.g. `suites/my_bench.rs`)
/// 2. Implement `BenchmarkSuite` for your struct
/// 3. Register it in `suites/mod.rs` via `register_suites()`
///
/// That's it — the runner handles execution, reporting, and output formatting.
pub trait BenchmarkSuite: Send {
    /// Human-readable name (e.g. "Metal Decode Latency").
    fn name(&self) -> &str;

    /// Short identifier used in config and CLI (e.g. "decode", "prefill").
    fn id(&self) -> &str;

    /// Which backends this suite supports.
    #[allow(dead_code)]
    fn supported_backends(&self) -> &[BackendKind];

    /// Whether this suite should run given the current config and CLI flags.
    fn should_run(&self, ctx: &BenchmarkContext) -> bool;

    /// Execute the benchmark and return results.
    fn run(&self, ctx: &BenchmarkContext) -> Result<Vec<BenchmarkResult>>;
}

/// Registry of available benchmark suites.
pub struct SuiteRegistry {
    suites: Vec<Box<dyn BenchmarkSuite>>,
}

impl SuiteRegistry {
    pub fn new() -> Self {
        Self { suites: Vec::new() }
    }

    pub fn register(&mut self, suite: Box<dyn BenchmarkSuite>) {
        self.suites.push(suite);
    }

    /// Run all suites that should execute given the context.
    pub fn run_all(&self, ctx: &BenchmarkContext) -> Result<Vec<BenchmarkResult>> {
        let mut all_results = Vec::new();
        for suite in &self.suites {
            if suite.should_run(ctx) {
                eprintln!("\n  {}", suite.name());
                eprintln!("  {}", "─".repeat(40));
                match suite.run(ctx) {
                    Ok(results) => {
                        eprintln!("  ✓ {} result(s)", results.len());
                        all_results.extend(results);
                    }
                    Err(e) => {
                        eprintln!("  ✗ {} failed: {e}", suite.name());
                    }
                }
            }
        }
        Ok(all_results)
    }

    /// List all registered suite IDs.
    pub fn suite_ids(&self) -> Vec<&str> {
        self.suites.iter().map(|s| s.id()).collect()
    }

    /// Run only the suites matching the given IDs.
    ///
    /// Explicit selection bypasses `should_run()` — if the user asks for a
    /// suite by name, we run it unconditionally.
    pub fn run_selected(
        &self,
        ids: &[&str],
        ctx: &BenchmarkContext,
    ) -> Result<Vec<BenchmarkResult>> {
        let mut all_results = Vec::new();
        for suite in &self.suites {
            if ids.contains(&suite.id()) {
                eprintln!("\n  {}", suite.name());
                eprintln!("  {}", "─".repeat(40));
                match suite.run(ctx) {
                    Ok(results) => {
                        eprintln!("  ✓ {} result(s)", results.len());
                        all_results.extend(results);
                    }
                    Err(e) => {
                        eprintln!("  ✗ {} failed: {e}", suite.name());
                    }
                }
            }
        }
        Ok(all_results)
    }
}

/// Helper: run a timed closure `iterations` times after `warmup` warmups,
/// across `runs` independent runs. Returns an `AggregatedResult`.
#[allow(dead_code)]
pub fn run_timed_benchmark<F>(
    label: &str,
    warmup: usize,
    iterations: usize,
    runs: usize,
    mut setup: impl FnMut(),
    mut f: F,
) -> Result<AggregatedResult>
where
    F: FnMut(usize) -> Result<f64>,
{
    let mut run_results = Vec::with_capacity(runs);

    for run_idx in 0..runs {
        setup();

        for _ in 0..warmup {
            let _ = f(0)?;
        }

        let mut latencies_ms = Vec::with_capacity(iterations);
        for iter in 0..iterations {
            latencies_ms.push(f(iter)?);
        }

        let run_label = format!("{label}/run{run_idx}");
        run_results.push(compute_stats(&run_label, &latencies_ms));
    }

    Ok(aggregate_runs(label, &run_results))
}
