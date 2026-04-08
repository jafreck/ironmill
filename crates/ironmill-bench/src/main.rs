#![deny(unsafe_code)]

mod baseline;
mod compiler;
mod config;
mod hardware;
mod inference;
#[cfg_attr(not(feature = "ane-direct"), allow(dead_code))]
mod perplexity;
mod power;
mod quality;
mod report;
mod stats;
mod suite;
mod suites;

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use config::{ModelConfig, Settings};
use report::OutputFormat;
use suite::{BenchmarkContext, SuiteRegistry};

/// Backends to benchmark.
/// Values: coreml-cpu, coreml-gpu, coreml-ane, coreml-all, metal
#[derive(Clone, Debug, PartialEq, clap::ValueEnum)]
enum Backend {
    /// CoreML with CPU-only compute units
    CoremlCpu,
    /// CoreML with CPU + GPU compute units
    CoremlGpu,
    /// CoreML with CPU + ANE compute units
    CoremlAne,
    /// CoreML with all compute units
    CoremlAll,
    /// Direct Metal GPU backend (custom kernels + MPS)
    Metal,
}

impl Backend {
    fn to_config_string(&self) -> &str {
        match self {
            Backend::CoremlCpu => "cpu",
            Backend::CoremlGpu => "gpu",
            Backend::CoremlAne => "ane",
            Backend::CoremlAll => "all",
            Backend::Metal => "metal",
        }
    }
}

#[derive(Parser)]
#[command(name = "ironmill-bench", about = "Inference benchmark harness")]
struct Cli {
    /// TOML config file (default: built-in matrix)
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Add model (repeatable, overrides config)
    #[arg(short, long)]
    model: Vec<PathBuf>,

    /// Iterations per measurement
    #[arg(short, long, default_value = "1")]
    iterations: usize,

    /// Warmup iterations
    #[arg(short, long, default_value = "0")]
    warmup: usize,

    /// Number of full runs
    #[arg(short, long, default_value = "1")]
    runs: usize,

    /// Backends to benchmark. May be specified multiple times.
    /// Values: coreml-cpu, coreml-gpu, coreml-ane, coreml-all, metal
    #[arg(short, long, value_delimiter = ',')]
    backend: Vec<Backend>,

    /// Output format
    #[arg(short, long, default_value = "table", value_enum)]
    output: OutputFormat,

    /// Mark a config as baseline for significance tests
    #[arg(long)]
    baseline: Option<String>,

    /// Significance level
    #[arg(long, default_value = "0.05")]
    alpha: f64,

    /// Force recompilation (skip cache)
    #[arg(long)]
    no_cache: bool,

    /// Remove all cached compilation artifacts
    #[arg(long)]
    clean_cache: bool,

    /// Enable energy sampling (requires sudo for powermetrics)
    #[arg(long)]
    power: bool,

    /// Save results as a named baseline for regression tracking
    #[arg(long)]
    save_baseline: Option<String>,

    /// Compare against a saved baseline
    #[arg(long)]
    compare_baseline: Option<String>,

    /// Run weight fidelity quality benchmarks for quantized optimizations
    #[arg(long)]
    quality: bool,

    /// Run perplexity evaluation (requires model weights + dataset)
    #[arg(long)]
    perplexity: bool,

    /// Number of sequences for perplexity evaluation (default: 50)
    #[arg(long, default_value = "50")]
    perplexity_sequences: usize,

    /// Path to pre-tokenized dataset for perplexity evaluation
    #[arg(long, default_value = "tests/fixtures/quality/wikitext2-qwen3.json")]
    perplexity_dataset: PathBuf,

    /// Stride for sliding-window PPL evaluation.
    #[arg(long, default_value = "512")]
    perplexity_stride: usize,

    /// Run prefill throughput benchmark at various input lengths
    #[arg(long)]
    prefill_bench: bool,

    /// Comma-separated context lengths for decode-at-context benchmark.
    #[arg(long, value_delimiter = ',')]
    context_lengths: Vec<usize>,

    /// Run only specific benchmark suites (comma-separated).
    /// Available: coreml, decode, prefill, context-decode, quality, perplexity
    #[arg(long, value_delimiter = ',')]
    suite: Vec<String>,

    /// List available benchmark suites and exit
    #[arg(long)]
    list_suites: bool,

    /// Enable per-pipeline GPU timing instrumentation.
    #[arg(long)]
    kernel_timing: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let cache_dir = PathBuf::from("target/bench-cache");

    if cli.clean_cache {
        compiler::clean_cache(&cache_dir)?;
        println!("Cache cleaned.");
        return Ok(());
    }

    // Build the suite registry
    let mut registry = SuiteRegistry::new();
    suites::register_suites(&mut registry);

    if cli.list_suites {
        println!("Available benchmark suites:");
        for id in registry.suite_ids() {
            println!("  {id}");
        }
        return Ok(());
    }

    // Load config
    let mut matrix = if let Some(config_path) = &cli.config {
        config::load_config(config_path)?
    } else {
        config::default_matrix()
    };

    // CLI overrides
    if !cli.model.is_empty() {
        matrix.models = cli
            .model
            .iter()
            .map(|p| ModelConfig {
                name: p
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string(),
                path: p.clone(),
                input_shapes: vec![],
                model_dir: None,
            })
            .collect();
    }

    matrix.settings = Settings {
        iterations: cli.iterations,
        warmup: cli.warmup,
        runs: cli.runs,
        backends: if cli.backend.is_empty() {
            if matrix.settings.backends.is_empty() {
                vec!["cpu".to_string(), "gpu".to_string(), "ane".to_string()]
            } else {
                matrix.settings.backends.clone()
            }
        } else {
            cli.backend
                .iter()
                .map(|b| b.to_config_string().to_string())
                .collect()
        },
    };

    // Power sampling setup
    let power_enabled = cli.power && power::is_power_available();
    if cli.power && !power_enabled {
        eprintln!("Energy metrics unavailable (requires sudo)");
    }

    let idle_power = if power_enabled {
        eprintln!("Sampling idle power (2s)...");
        power::sample_idle_power(std::time::Duration::from_secs(2))
    } else {
        None
    };

    // Build extra context from CLI flags and config bools.
    // Config bools (decode, prefill, perplexity, quality) are resolved into
    // benchmarks.suites by resolve_suites(), but suites' run() methods also
    // read ctx.extra for parameters — so we populate both paths.
    let mut extra = HashMap::new();
    if cli.quality || matrix.benchmarks.quality {
        extra.insert("quality".to_string(), "true".to_string());
    }
    if cli.perplexity || matrix.benchmarks.perplexity {
        extra.insert("perplexity".to_string(), "true".to_string());
        let seq = if cli.perplexity {
            cli.perplexity_sequences
        } else {
            matrix.benchmarks.perplexity_sequences
        };
        let stride = if cli.perplexity {
            cli.perplexity_stride
        } else {
            matrix.benchmarks.perplexity_stride
        };
        let dataset = if cli.perplexity {
            cli.perplexity_dataset.to_string_lossy().to_string()
        } else {
            matrix.benchmarks.perplexity_dataset.clone()
        };
        extra.insert("perplexity_sequences".to_string(), seq.to_string());
        extra.insert("perplexity_stride".to_string(), stride.to_string());
        extra.insert("perplexity_dataset".to_string(), dataset);
    }
    if cli.prefill_bench
        || matrix.benchmarks.prefill
        || !matrix.benchmarks.prefill_lengths.is_empty()
    {
        extra.insert("prefill_bench".to_string(), "true".to_string());
        if !matrix.benchmarks.prefill_lengths.is_empty() {
            extra.insert(
                "prefill_lengths".to_string(),
                matrix
                    .benchmarks
                    .prefill_lengths
                    .iter()
                    .map(|l| l.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
    }
    if !cli.context_lengths.is_empty() || !matrix.benchmarks.context_lengths.is_empty() {
        let lengths = if !cli.context_lengths.is_empty() {
            &cli.context_lengths
        } else {
            &matrix.benchmarks.context_lengths
        };
        extra.insert(
            "context_lengths".to_string(),
            lengths
                .iter()
                .map(|l| l.to_string())
                .collect::<Vec<_>>()
                .join(","),
        );
    }
    if cli.kernel_timing {
        extra.insert("kernel_timing".to_string(), "true".to_string());
    }

    let ctx = BenchmarkContext {
        matrix: &matrix,
        cache_dir: &cache_dir,
        no_cache: cli.no_cache,
        power_enabled,
        idle_power: &idle_power,
        extra,
    };

    // Run suites
    let suite_results = if !cli.suite.is_empty() {
        let ids: Vec<&str> = cli.suite.iter().map(|s| s.as_str()).collect();
        registry.run_selected(&ids, &ctx)?
    } else if !matrix.benchmarks.suites.is_empty() {
        let ids: Vec<&str> = matrix
            .benchmarks
            .suites
            .iter()
            .map(|s| s.as_str())
            .collect();
        registry.run_selected(&ids, &ctx)?
    } else {
        registry.run_all(&ctx)?
    };

    // Convert suite results to legacy ReportRows for output formatting
    let report_rows: Vec<report::ReportRow> = suite_results
        .iter()
        .map(|r| report::ReportRow {
            model: r.model.clone(),
            optimization: r.optimization.clone(),
            backend: if let Some(ref v) = r.variant {
                format!("{}[{}]", r.backend, v)
            } else {
                r.backend.clone()
            },
            kv_quant: r
                .metadata
                .get("kv_quant")
                .cloned()
                .unwrap_or_else(|| "none".to_string()),
            result: r.result.clone(),
            significance: None,
            energy: None,
            utilization: None,
            memory: r.gpu_memory_mb.map(|mb| report::MemorySummary {
                rss_after_load_mb: mb,
                peak_rss_mb: mb,
                rss_growth_mb: 0.0,
                model_file_size_mb: 0.0,
                efficiency_ratio: 0.0,
            }),
            load_time_ms: r.load_time_ms,
        })
        .collect();

    // Apply significance tests if baseline specified
    let mut report_rows = report_rows;
    if let Some(baseline_name) = &cli.baseline {
        apply_baseline_significance(&mut report_rows, baseline_name, cli.alpha);
    }

    let bench_report = report::BenchReport {
        rows: report_rows,
        settings: matrix.settings.clone(),
    };

    if !bench_report.rows.is_empty() {
        print!("{}", report::format_report(&bench_report, cli.output));
    }

    // Save baseline if requested
    if let Some(baseline_name) = &cli.save_baseline {
        let entries = build_baseline_entries(&bench_report.rows);
        let path = baseline::save_baseline(baseline_name, &entries)?;
        eprintln!("Baseline saved: {}", path.display());
    }

    // Compare against baseline if requested
    if let Some(baseline_name) = &cli.compare_baseline {
        let baseline_data = baseline::load_baseline(baseline_name)?;
        let current_entries = build_baseline_entries(&bench_report.rows);
        let regression_report =
            baseline::compare_against_baseline(&baseline_data, &current_entries);
        eprintln!(
            "\n{}",
            baseline::format_regression_report(&regression_report)
        );
    }

    Ok(())
}

/// Apply Welch t-test significance against a named baseline optimization.
fn apply_baseline_significance(
    report_rows: &mut [report::ReportRow],
    baseline_name: &str,
    alpha: f64,
) {
    let baseline_rows: Vec<_> = report_rows
        .iter()
        .filter(|r| r.optimization == *baseline_name)
        .cloned()
        .collect();

    for row in report_rows.iter_mut() {
        if row.optimization == *baseline_name {
            continue;
        }
        if let Some(bl) = baseline_rows
            .iter()
            .find(|b| b.model == row.model && b.backend == row.backend)
        {
            let sig = stats::welch_t_test(&bl.result.pooled, &row.result.pooled, alpha);
            row.significance = Some(sig);
        }
    }
}

/// Convert report rows to baseline entries for save/compare.
fn build_baseline_entries(rows: &[report::ReportRow]) -> Vec<baseline::BaselineEntry> {
    rows.iter()
        .map(|row| baseline::BaselineEntry {
            model: row.model.clone(),
            optimization: row.optimization.clone(),
            backend: row.backend.clone(),
            median_ms: row.result.pooled.median,
            mean_ms: row.result.pooled.mean,
            p95_ms: row.result.pooled.p95,
            inferences_per_sec: row.result.pooled.inferences_per_sec,
            tflops: row.result.pooled.tflops,
        })
        .collect()
}
