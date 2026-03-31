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

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use config::{ModelConfig, Settings};
use report::{BenchReport, MemorySummary, OutputFormat, ReportRow, UtilizationSummary};
#[allow(unused_imports)]
use stats::{aggregate_runs, compute_stats, compute_stats_with_flops, welch_t_test};

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

    /// Benchmark the Metal GPU backend (experimental, requires --features metal).
    #[arg(long)]
    metal: bool,

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

    /// Run perplexity evaluation (requires ane-direct feature + model weights + dataset)
    #[arg(long)]
    perplexity: bool,

    /// Number of sequences for perplexity evaluation (default: 50)
    #[arg(long, default_value = "50")]
    perplexity_sequences: usize,

    /// Path to pre-tokenized dataset for perplexity evaluation
    #[arg(long, default_value = "tests/fixtures/quality/wikitext2-qwen3.json")]
    perplexity_dataset: PathBuf,
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
            vec!["cpu".to_string(), "gpu".to_string(), "ane".to_string()]
        } else {
            cli.backend.clone()
        },
    };

    let compute_units: Vec<ironmill_inference::coreml_runtime::ComputeUnits> = matrix
        .settings
        .backends
        .iter()
        .map(|b| b.parse::<ironmill_inference::coreml_runtime::ComputeUnits>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| anyhow::anyhow!("invalid backend: {e}"))?;

    // Power sampling setup
    let power_enabled = cli.power && power::is_power_available();
    if cli.power && !power_enabled {
        eprintln!("Energy metrics unavailable (requires sudo)");
    }

    // Sample idle power if power measurement is enabled
    let idle_power = if power_enabled {
        eprintln!("Sampling idle power (2s)...");
        power::sample_idle_power(std::time::Duration::from_secs(2))
    } else {
        None
    };

    let mut report_rows = Vec::new();

    // Skip the main benchmark loop if only perplexity is requested
    // or only metal/ane-direct are requested (they have their own paths)
    let run_latency_bench = !cli.perplexity || cli.ane_direct || cli.metal || cli.quality;
    let run_coreml_bench = run_latency_bench
        && (!cli.metal || cli.ane_direct || !cli.backend.is_empty() || cli.quality);

    if run_coreml_bench {
        for model_cfg in &matrix.models {
            // Compute model FLOPs if we can parse the model
            let model_flops = compute_model_flops(model_cfg);
            if let Some(flops) = model_flops {
                eprintln!(
                    "  Model FLOPs: {:.2}G ({:.2}M MACs)",
                    flops as f64 / 1e9,
                    flops as f64 / 2e6
                );
            }

            for opt_cfg in &matrix.optimizations {
                eprintln!("Compiling {} with {}...", model_cfg.name, opt_cfg.name);
                let mlmodelc =
                    compiler::compile_model(model_cfg, opt_cfg, &cache_dir, cli.no_cache)?;

                for &cu in &compute_units {
                    eprintln!("  Running inference ({cu})...");

                    // Start power sampling if enabled
                    let power_sampler = if power_enabled {
                        let expected_duration_sec =
                            matrix.settings.iterations as f64 * 0.001 * matrix.settings.runs as f64;
                        let samples = (expected_duration_sec * 10.0).max(20.0) as usize;
                        power::PowerSampler::start(100, samples)
                    } else {
                        None
                    };

                    let mut run_results = Vec::new();
                    let mut last_utilization = None;
                    let mut last_memory = None;
                    let mut last_load_time = None;

                    for run_idx in 0..matrix.settings.runs {
                        let result = inference::run_inference(
                            &mlmodelc,
                            cu,
                            matrix.settings.iterations,
                            matrix.settings.warmup,
                        )?;

                        last_load_time = Some(result.load_time.as_secs_f64() * 1000.0);
                        last_utilization = result.utilization;
                        last_memory = result.memory;

                        let latencies_ms: Vec<f64> = result
                            .latencies
                            .iter()
                            .map(|d| d.as_secs_f64() * 1000.0)
                            .collect();

                        let label =
                            format!("{}/{}/{}/run{}", model_cfg.name, opt_cfg.name, cu, run_idx);
                        run_results.push(compute_stats_with_flops(
                            &label,
                            &latencies_ms,
                            model_flops,
                        ));
                    }

                    let label = format!("{}/{}/{}", model_cfg.name, opt_cfg.name, cu);
                    let aggregated = aggregate_runs(&label, &run_results);

                    // Collect power metrics
                    let energy = power_sampler.and_then(|sampler| {
                        let power_metrics = sampler.finish()?;
                        Some(power::compute_energy_metrics(
                            power_metrics,
                            idle_power.clone(),
                            aggregated.pooled.inferences_per_sec,
                            aggregated.pooled.median / 1000.0,
                            aggregated.pooled.tflops,
                            aggregated.pooled.tokens_per_sec,
                        ))
                    });

                    let utilization_summary = last_utilization
                        .as_ref()
                        .map(UtilizationSummary::from_metrics);
                    let memory_summary = last_memory.as_ref().map(MemorySummary::from_metrics);

                    report_rows.push(ReportRow {
                        model: model_cfg.name.clone(),
                        optimization: opt_cfg.name.clone(),
                        backend: cu.to_string(),
                        kv_quant: opt_cfg.kv_quant.to_string(),
                        result: aggregated,
                        significance: None,
                        energy,
                        utilization: utilization_summary,
                        memory: memory_summary,
                        load_time_ms: last_load_time,
                    });
                }
            }
        }
    } // end run_coreml_bench

    // ── ANE-direct and Metal benchmarks (independent of CoreML) ──

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

                    let config = ironmill_inference::AneConfig::default();

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
                            kv_quant: opt_cfg.kv_quant.to_string(),
                            result: aggregated,
                            significance: None,
                            energy: None,
                            utilization: None,
                            memory: None,
                            load_time_ms: None,
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

    if cli.metal {
        #[cfg(feature = "metal")]
        {
            use ironmill_compile::weights::SafeTensorsProvider;
            use ironmill_inference::engine::InferenceEngine;
            use ironmill_inference::gpu::{GpuConfig, GpuInference};

            eprintln!("\n  Metal GPU Backend Benchmark");
            eprintln!("  {}", "─".repeat(40));

            // Run two configurations: FP16 baseline and TurboQuant INT4
            let configs: Vec<(&str, GpuConfig)> = vec![
                (
                    "fp16",
                    GpuConfig {
                        enable_turboquant: false,
                        ..GpuConfig::default()
                    },
                ),
                (
                    "tq-int4",
                    GpuConfig {
                        enable_turboquant: true,
                        n_bits: 4,
                        ..GpuConfig::default()
                    },
                ),
            ];

            for model_cfg in &matrix.models {
                // Load weights once, reuse across configs
                let model_dir = if model_cfg.path.is_dir() {
                    model_cfg.path.clone()
                } else if let Some(parent) = model_cfg.path.parent() {
                    parent.to_path_buf()
                } else {
                    eprintln!(
                        "  ✗ cannot determine model directory for {}",
                        model_cfg.name
                    );
                    continue;
                };

                let provider = match SafeTensorsProvider::load(&model_dir) {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!(
                            "  ✗ failed to load weights from {}: {e}",
                            model_dir.display()
                        );
                        continue;
                    }
                };

                for (config_name, gpu_config) in &configs {
                    eprintln!("  Metal/{config_name}: {}...", model_cfg.name);

                    let mut engine = match GpuInference::new(gpu_config.clone()) {
                        Ok(e) => e,
                        Err(e) => {
                            eprintln!("  ✗ Metal GPU init failed: {e}");
                            continue;
                        }
                    };

                    let load_start = std::time::Instant::now();
                    if let Err(e) = engine.load_weights(&provider, gpu_config.clone()) {
                        eprintln!("  ✗ Metal model load failed: {e}");
                        continue;
                    }
                    let load_time_ms = load_start.elapsed().as_secs_f64() * 1000.0;
                    eprintln!("  ✓ loaded in {load_time_ms:.1}ms");

                    // Prefill with a short prompt (token IDs for "Hello world")
                    let prompt_tokens: Vec<u32> = vec![9707, 1879];

                    let mut run_results = Vec::new();
                    for run_idx in 0..matrix.settings.runs {
                        engine.reset();

                        // Prefill
                        if let Err(e) = engine.prefill(&prompt_tokens) {
                            eprintln!("  ✗ prefill failed: {e}");
                            continue;
                        }

                        // Warmup decode steps
                        let mut last_token = prompt_tokens.last().copied().unwrap_or(0);
                        for _ in 0..matrix.settings.warmup {
                            match engine.decode_step(last_token) {
                                Ok(logits) => {
                                    last_token = logits
                                        .iter()
                                        .enumerate()
                                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                        .map(|(i, _)| i as u32)
                                        .unwrap_or(0);
                                }
                                Err(e) => {
                                    eprintln!("  ✗ warmup decode failed: {e}");
                                    break;
                                }
                            }
                        }

                        // Timed decode steps
                        let mut latencies = Vec::with_capacity(matrix.settings.iterations);
                        for _ in 0..matrix.settings.iterations {
                            let t0 = std::time::Instant::now();
                            match engine.decode_step(last_token) {
                                Ok(logits) => {
                                    let elapsed = t0.elapsed();
                                    latencies.push(elapsed);
                                    last_token = logits
                                        .iter()
                                        .enumerate()
                                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                        .map(|(i, _)| i as u32)
                                        .unwrap_or(0);
                                }
                                Err(e) => {
                                    eprintln!("  ✗ decode failed at iteration: {e}");
                                    break;
                                }
                            }
                        }

                        if latencies.is_empty() {
                            continue;
                        }

                        let latencies_ms: Vec<f64> =
                            latencies.iter().map(|d| d.as_secs_f64() * 1000.0).collect();

                        let label = format!(
                            "{}/{}/metal-{config_name}/run{run_idx}",
                            model_cfg.name, "default"
                        );
                        run_results.push(compute_stats(&label, &latencies_ms));
                    }

                    if !run_results.is_empty() {
                        let label = format!("{}/metal-{config_name}", model_cfg.name);
                        let aggregated = aggregate_runs(&label, &run_results);

                        let tok_per_sec = 1000.0 / aggregated.pooled.mean;
                        eprintln!(
                            "  ✓ {config_name}: {:.2}ms/tok ({:.1} tok/s) ± {:.2}ms",
                            aggregated.pooled.mean, tok_per_sec, aggregated.pooled.stddev
                        );

                        report_rows.push(ReportRow {
                            model: model_cfg.name.clone(),
                            optimization: "default".to_string(),
                            backend: format!("metal-{config_name}"),
                            kv_quant: config_name.to_string(),
                            result: aggregated,
                            significance: None,
                            energy: None,
                            utilization: None,
                            memory: None,
                            load_time_ms: Some(load_time_ms),
                        });
                    }
                }
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            eprintln!("warning: --metal requires --features metal, skipping");
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
            if let Some(bl) = baseline_rows
                .iter()
                .find(|b| b.model == row.model && b.backend == row.backend)
            {
                let sig = welch_t_test(&bl.result.pooled, &row.result.pooled, cli.alpha);
                row.significance = Some(sig);
            }
        }
    }

    let bench_report = BenchReport {
        rows: report_rows,
        settings: matrix.settings.clone(),
    };

    if run_latency_bench && !bench_report.rows.is_empty() {
        print!("{}", report::format_report(&bench_report, cli.output));
    }

    // Quality benchmarks — measure weight fidelity impact of quantization
    if cli.quality {
        eprintln!("\nRunning weight fidelity quality benchmarks...");
        let mut summaries = Vec::new();

        for model_cfg in &matrix.models {
            for opt_cfg in &matrix.optimizations {
                // Only run quality for quantized optimizations
                let (method, bits) = match (&opt_cfg.polar_quantize, &opt_cfg.palettize) {
                    (Some(b), _) => ("polar", *b),
                    (_, Some(b)) => ("palettize", *b),
                    _ => continue,
                };

                eprintln!(
                    "  Quality: {} with {} ({}-bit)...",
                    model_cfg.name, opt_cfg.name, bits
                );

                match compiler::build_optimized_program(model_cfg, opt_cfg) {
                    Ok(program) => {
                        let results = quality::measure_program_quality(&program, method, bits);
                        if let Some(summary) = quality::summarize_quality(&model_cfg.name, &results)
                        {
                            summaries.push(summary);
                        }
                    }
                    Err(e) => {
                        eprintln!("    ✗ failed: {e}");
                    }
                }
            }
        }

        if !summaries.is_empty() {
            eprintln!();
            eprint!("{}", quality::format_quality_summary(&summaries));
        } else {
            eprintln!("  No quantized optimizations to measure.");
        }
    }

    // Perplexity evaluation — measure model quality via cross-entropy on text corpus
    #[cfg(feature = "ane-direct")]
    if cli.perplexity {
        use ironmill_inference::ane::{AneInference, HardwareAneDevice};
        use std::sync::Arc;

        eprintln!("\nRunning perplexity evaluation...");

        let dataset = match perplexity::PerplexityDataset::load(&cli.perplexity_dataset) {
            Ok(ds) => {
                eprintln!(
                    "  Dataset: {} ({} sequences, {} tokens each)",
                    ds.name, ds.num_sequences, ds.seq_len
                );
                ds
            }
            Err(e) => {
                eprintln!(
                    "  ✗ Failed to load dataset '{}': {e}",
                    cli.perplexity_dataset.display()
                );
                eprintln!("  Run: python scripts/prepare-quality-dataset.py");
                return Ok(());
            }
        };

        // Build an optimized program from the first model config
        let model_cfg = match matrix.models.first() {
            Some(m) => m,
            None => {
                eprintln!("  ✗ No model configured. Use --model <path.onnx>");
                return Ok(());
            }
        };
        let opt_cfg = &matrix.optimizations[0]; // baseline

        eprintln!("  Model: {} ({})", model_cfg.name, model_cfg.path.display());
        eprintln!(
            "  Evaluating {} sequences...",
            cli.perplexity_sequences.min(dataset.num_sequences)
        );

        let program = match compiler::build_optimized_program(model_cfg, opt_cfg) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("  ✗ Failed to build program: {e}");
                return Ok(());
            }
        };

        let device = Arc::new(HardwareAneDevice::new().map_err(|e| anyhow::anyhow!("{e}"))?);
        match AneInference::compile(device, &program, None) {
            Ok(mut inference) => {
                match perplexity::evaluate_perplexity(
                    &mut inference,
                    &dataset,
                    Some(cli.perplexity_sequences),
                ) {
                    Ok(mut result) => {
                        result.config_name = format!("{} ({})", model_cfg.name, opt_cfg.name);
                        eprintln!();
                        eprint!("{}", perplexity::format_perplexity_table(&[result]));
                    }
                    Err(e) => {
                        eprintln!("  ✗ Perplexity evaluation failed: {e}");
                    }
                }
            }
            Err(e) => {
                eprintln!("  ✗ Failed to compile ANE inference: {e}");
            }
        }
    }

    #[cfg(not(feature = "ane-direct"))]
    if cli.perplexity {
        eprintln!("Perplexity evaluation requires --features ane-direct");
        eprintln!("Run: cargo run -p ironmill-bench --features ane-direct -- --perplexity ...");
    }

    // Save baseline if requested
    if let Some(baseline_name) = &cli.save_baseline {
        let entries: Vec<baseline::BaselineEntry> = bench_report
            .rows
            .iter()
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
            .collect();

        let path = baseline::save_baseline(baseline_name, &entries)?;
        eprintln!("Baseline saved: {}", path.display());
    }

    // Compare against baseline if requested
    if let Some(baseline_name) = &cli.compare_baseline {
        let baseline_data = baseline::load_baseline(baseline_name)?;
        let current_entries: Vec<baseline::BaselineEntry> = bench_report
            .rows
            .iter()
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
            .collect();

        let regression_report =
            baseline::compare_against_baseline(&baseline_data, &current_entries);
        eprintln!(
            "\n{}",
            baseline::format_regression_report(&regression_report)
        );
    }

    Ok(())
}

/// Try to compute model FLOPs by parsing the ONNX model into MIL IR.
fn compute_model_flops(model_cfg: &ModelConfig) -> Option<u64> {
    if !model_cfg.path.exists() {
        return None;
    }

    let ext = model_cfg.path.extension()?.to_str()?;
    match ext {
        "onnx" => {
            let (onnx, model_dir) =
                ironmill_compile::mil::read_onnx_with_dir(model_cfg.path.to_str()?).ok()?;
            let config = ironmill_compile::mil::ConversionConfig {
                model_dir: Some(model_dir),
                ..Default::default()
            };
            let result = ironmill_compile::mil::onnx_to_program_with_config(&onnx, &config).ok()?;
            let flops = result.program.total_flops();
            if flops > 0 { Some(flops) } else { None }
        }
        _ => None,
    }
}
