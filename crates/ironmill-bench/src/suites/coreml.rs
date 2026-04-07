//! CoreML inference benchmark suite.

use std::collections::HashMap;

use anyhow::Result;

use crate::compiler;
use crate::inference;
use crate::power;
use crate::report::{MemorySummary, UtilizationSummary};
use crate::stats::{aggregate_runs, compute_stats_with_flops};
use crate::suite::{BackendKind, BenchmarkContext, BenchmarkResult, BenchmarkSuite};

pub struct CoremlSuite;

impl BenchmarkSuite for CoremlSuite {
    fn name(&self) -> &str {
        "CoreML Inference"
    }

    fn id(&self) -> &str {
        "coreml"
    }

    fn supported_backends(&self) -> &[BackendKind] {
        &[
            BackendKind::CoremlCpu,
            BackendKind::CoremlGpu,
            BackendKind::CoremlAne,
            BackendKind::CoremlAll,
        ]
    }

    fn should_run(&self, ctx: &BenchmarkContext) -> bool {
        let backends = &ctx.matrix.settings.backends;
        backends.iter().any(|b| {
            matches!(
                b.as_str(),
                "cpu"
                    | "gpu"
                    | "ane"
                    | "all"
                    | "coreml-cpu"
                    | "coreml-gpu"
                    | "coreml-ane"
                    | "coreml-all"
            )
        })
    }

    fn run(&self, ctx: &BenchmarkContext) -> Result<Vec<BenchmarkResult>> {
        use ironmill_inference::coreml_runtime::ComputeUnits;

        let compute_units: Vec<ComputeUnits> = ctx
            .matrix
            .settings
            .backends
            .iter()
            .filter_map(|b| match b.as_str() {
                "cpu" | "coreml-cpu" => Some(ComputeUnits::CpuOnly),
                "gpu" | "coreml-gpu" => Some(ComputeUnits::CpuAndGpu),
                "ane" | "coreml-ane" => Some(ComputeUnits::CpuAndNeuralEngine),
                "all" | "coreml-all" => Some(ComputeUnits::All),
                _ => None,
            })
            .collect();

        if compute_units.is_empty() {
            return Ok(Vec::new());
        }

        // Pre-parse ONNX models once
        let mut parsed_programs: HashMap<String, ironmill_compile::mil::Program> = HashMap::new();
        for model_cfg in &ctx.matrix.models {
            if !model_cfg.path.exists() {
                continue;
            }
            let ext = model_cfg.path.extension().and_then(|e| e.to_str());
            if ext != Some("onnx") {
                continue;
            }
            eprintln!("  Parsing {} (ONNX → MIL IR)...", model_cfg.name);
            match compiler::parse_model(model_cfg) {
                Ok(program) => {
                    parsed_programs.insert(model_cfg.name.clone(), program);
                }
                Err(e) => {
                    eprintln!("  ✗ failed to parse {}: {e}", model_cfg.name);
                }
            }
        }

        let mut results = Vec::new();

        for model_cfg in &ctx.matrix.models {
            let model_flops = parsed_programs.get(&model_cfg.name).and_then(|p| {
                let f = p.total_flops();
                if f > 0 { Some(f) } else { None }
            });

            if let Some(flops) = model_flops {
                eprintln!(
                    "  Model FLOPs: {:.2}G ({:.2}M MACs)",
                    flops as f64 / 1e9,
                    flops as f64 / 2e6
                );
            }

            for opt_cfg in &ctx.matrix.optimizations {
                eprintln!("  Compiling {} with {}...", model_cfg.name, opt_cfg.name);
                let mlmodelc = if let Some(prog) = parsed_programs.get(&model_cfg.name) {
                    compiler::compile_model_from_program(
                        prog,
                        model_cfg,
                        opt_cfg,
                        ctx.cache_dir,
                        ctx.no_cache,
                    )?
                } else {
                    compiler::compile_model(model_cfg, opt_cfg, ctx.cache_dir, ctx.no_cache)?
                };

                for &cu in &compute_units {
                    eprintln!("  Running inference ({cu})...");

                    let power_sampler = if ctx.power_enabled {
                        let expected_duration_sec = ctx.matrix.settings.iterations as f64
                            * 0.001
                            * ctx.matrix.settings.runs as f64;
                        let samples = (expected_duration_sec * 10.0).max(20.0) as usize;
                        power::PowerSampler::start(100, samples)
                    } else {
                        None
                    };

                    let (run_inference_results, _load_time) = inference::run_inference_multi_run(
                        &mlmodelc,
                        cu,
                        ctx.matrix.settings.iterations,
                        ctx.matrix.settings.warmup,
                        ctx.matrix.settings.runs,
                    )?;

                    let mut run_results = Vec::new();
                    let mut last_utilization = None;
                    let mut last_memory = None;
                    let mut last_load_time = None;

                    for (run_idx, result) in run_inference_results.into_iter().enumerate() {
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

                    let _energy = power_sampler.and_then(|sampler| {
                        let power_metrics = sampler.finish()?;
                        Some(power::compute_energy_metrics(
                            power_metrics,
                            ctx.idle_power.clone(),
                            aggregated.pooled.inferences_per_sec,
                            aggregated.pooled.median / 1000.0,
                            aggregated.pooled.tflops,
                            aggregated.pooled.tokens_per_sec,
                        ))
                    });

                    let _utilization_summary = last_utilization
                        .as_ref()
                        .map(UtilizationSummary::from_metrics);
                    let memory_summary = last_memory.as_ref().map(MemorySummary::from_metrics);

                    results.push(BenchmarkResult {
                        suite: self.id().to_string(),
                        model: model_cfg.name.clone(),
                        optimization: opt_cfg.name.clone(),
                        backend: cu.to_string(),
                        variant: None,
                        gpu_memory_mb: memory_summary.as_ref().map(|m| m.rss_after_load_mb),
                        load_time_ms: last_load_time,
                        perplexity: None,
                        metadata: HashMap::new(),
                        result: aggregated,
                    });
                }
            }
        }

        Ok(results)
    }
}
