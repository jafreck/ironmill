mod compiler;
mod config;
mod inference;
mod quality;
mod report;
mod stats;

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use config::{ModelConfig, Settings};
use report::{BenchReport, OutputFormat, ReportRow};
use stats::{aggregate_runs, compute_stats, welch_t_test};

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
    #[arg(short, long, default_value = "200")]
    iterations: usize,

    /// Warmup iterations
    #[arg(short, long, default_value = "20")]
    warmup: usize,

    /// Number of full runs
    #[arg(short, long, default_value = "3")]
    runs: usize,

    /// Backend filter: cpu, gpu, ane, all (default: cpu,gpu,ane)
    #[arg(short, long)]
    backend: Vec<String>,

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

    /// Also benchmark the ANE direct runtime (experimental, requires --features ane-direct).
    #[arg(long)]
    ane_direct: bool,

    /// Remove all cached compilation artifacts
    #[arg(long)]
    clean_cache: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let cache_dir = PathBuf::from("target/bench-cache");

    if cli.clean_cache {
        compiler::clean_cache(&cache_dir)?;
        println!("Cache cleaned.");
        return Ok(());
    }

    let mut matrix = if let Some(config_path) = &cli.config {
        config::load_config(config_path)?
    } else {
        config::default_matrix()
    };

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
            })
            .collect();
    }

    matrix.settings = Settings {
        iterations: cli.iterations,
        warmup: cli.warmup,
        runs: cli.runs,
        backends: if cli.backend.is_empty() {
            // Default: benchmark on each compute unit individually to show the
            // performance matrix across hardware targets.
            vec!["cpu".to_string(), "gpu".to_string(), "ane".to_string()]
        } else {
            cli.backend.clone()
        },
    };

    let compute_units: Vec<ironmill_coreml::ComputeUnits> = matrix
        .settings
        .backends
        .iter()
        .map(|b| b.parse::<ironmill_coreml::ComputeUnits>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| anyhow::anyhow!("invalid backend: {e}"))?;

    let mut report_rows = Vec::new();

    for model_cfg in &matrix.models {
        for opt_cfg in &matrix.optimizations {
            eprintln!("Compiling {} with {}...", model_cfg.name, opt_cfg.name);
            let mlmodelc = compiler::compile_model(model_cfg, opt_cfg, &cache_dir, cli.no_cache)?;

            for &cu in &compute_units {
                eprintln!("  Running inference ({cu})...");

                let mut run_results = Vec::new();
                for run_idx in 0..matrix.settings.runs {
                    let result = inference::run_inference(
                        &mlmodelc,
                        cu,
                        matrix.settings.iterations,
                        matrix.settings.warmup,
                    )?;

                    let latencies_ms: Vec<f64> = result
                        .latencies
                        .iter()
                        .map(|d| d.as_secs_f64() * 1000.0)
                        .collect();

                    let label =
                        format!("{}/{}/{}/run{}", model_cfg.name, opt_cfg.name, cu, run_idx);
                    run_results.push(compute_stats(&label, &latencies_ms));
                }

                let label = format!("{}/{}/{}", model_cfg.name, opt_cfg.name, cu);
                let aggregated = aggregate_runs(&label, &run_results);

                report_rows.push(ReportRow {
                    model: model_cfg.name.clone(),
                    optimization: opt_cfg.name.clone(),
                    backend: cu.to_string(),
                    result: aggregated,
                    significance: None,
                });
            }
        }
    }

    if cli.ane_direct {
        #[cfg(feature = "ane-direct")]
        {
            eprintln!("\n  ANE Direct Runtime Benchmark");
            eprintln!("  {}", "─".repeat(40));
            eprintln!("  ⚠ ANE direct benchmarking requires runtime verification");
            eprintln!("    (private API selectors must be validated on this macOS version)");

            for model_cfg in &matrix.models {
                for opt_cfg in &matrix.optimizations {
                    eprintln!(
                        "  Compiling {} with {} (ANE direct)...",
                        model_cfg.name, opt_cfg.name
                    );

                    let program = match compiler::build_optimized_program(model_cfg, opt_cfg) {
                        Ok(p) => p,
                        Err(e) => {
                            eprintln!("  ✗ failed to build program: {e}");
                            continue;
                        }
                    };

                    let config = ironmill_ane::AneConfig::default();

                    let mut run_results = Vec::new();
                    for run_idx in 0..matrix.settings.runs {
                        let result = match inference::run_ane_direct_inference(
                            &program,
                            config.clone(),
                            matrix.settings.warmup,
                            matrix.settings.iterations,
                        ) {
                            Ok(r) => r,
                            Err(e) => {
                                eprintln!("  ✗ ANE direct inference failed: {e}");
                                continue;
                            }
                        };

                        let latencies_ms: Vec<f64> = result
                            .latencies
                            .iter()
                            .map(|d| d.as_secs_f64() * 1000.0)
                            .collect();

                        let label = format!(
                            "{}/{}/ane-direct/run{}",
                            model_cfg.name, opt_cfg.name, run_idx
                        );
                        run_results.push(compute_stats(&label, &latencies_ms));
                    }

                    if !run_results.is_empty() {
                        let label = format!("{}/{}/ane-direct", model_cfg.name, opt_cfg.name);
                        let aggregated = aggregate_runs(&label, &run_results);

                        report_rows.push(ReportRow {
                            model: model_cfg.name.clone(),
                            optimization: opt_cfg.name.clone(),
                            backend: "ane-direct".to_string(),
                            result: aggregated,
                            significance: None,
                        });
                    }
                }
            }
        }
        #[cfg(not(feature = "ane-direct"))]
        {
            eprintln!("warning: --ane-direct requires --features ane-direct, skipping");
        }
    }

    if let Some(baseline_name) = &cli.baseline {
        let baseline_rows: Vec<_> = report_rows
            .iter()
            .filter(|r| r.optimization == *baseline_name)
            .cloned()
            .collect();

        for row in &mut report_rows {
            if row.optimization == *baseline_name {
                continue;
            }
            if let Some(baseline) = baseline_rows
                .iter()
                .find(|b| b.model == row.model && b.backend == row.backend)
            {
                let sig = welch_t_test(&baseline.result.pooled, &row.result.pooled, cli.alpha);
                row.significance = Some(sig);
            }
        }
    }

    let bench_report = BenchReport {
        rows: report_rows,
        settings: matrix.settings,
    };

    print!("{}", report::format_report(&bench_report, cli.output));

    Ok(())
}
