//! Weight fidelity quality benchmark suite.

use std::collections::HashMap;

use anyhow::Result;

use crate::compiler;
use crate::quality;
use crate::suite::{BackendKind, BenchmarkContext, BenchmarkResult, BenchmarkSuite};

pub struct QualitySuite;

impl BenchmarkSuite for QualitySuite {
    fn name(&self) -> &str {
        "Weight Fidelity Quality"
    }

    fn id(&self) -> &str {
        "quality"
    }

    fn supported_backends(&self) -> &[BackendKind] {
        &[BackendKind::CoremlAll, BackendKind::Metal]
    }

    fn should_run(&self, ctx: &BenchmarkContext) -> bool {
        ctx.extra.contains_key("quality")
    }

    fn run(&self, ctx: &BenchmarkContext) -> Result<Vec<BenchmarkResult>> {
        eprintln!("  Running weight fidelity quality benchmarks...");
        let mut results = Vec::new();
        let mut summaries = Vec::new();

        // Pre-parse models
        let mut parsed_programs: HashMap<String, ironmill_compile::mil::Program> = HashMap::new();
        for model_cfg in &ctx.matrix.models {
            if !model_cfg.path.exists() {
                continue;
            }
            if model_cfg.path.extension().and_then(|e| e.to_str()) == Some("onnx") {
                if let Ok(prog) = compiler::parse_model(model_cfg) {
                    parsed_programs.insert(model_cfg.name.clone(), prog);
                }
            }
        }

        for model_cfg in &ctx.matrix.models {
            for opt_cfg in &ctx.matrix.optimizations {
                let (method, bits) = match (&opt_cfg.polar_quantize, &opt_cfg.palettize) {
                    (Some(b), _) => ("polar", *b),
                    (_, Some(b)) => ("palettize", *b),
                    _ => continue,
                };

                eprintln!(
                    "  Quality: {} with {} ({}-bit)...",
                    model_cfg.name, opt_cfg.name, bits
                );

                let program_result = if let Some(prog) = parsed_programs.get(&model_cfg.name) {
                    compiler::build_optimized_program_from(prog, opt_cfg)
                } else {
                    compiler::build_optimized_program(model_cfg, opt_cfg)
                };

                match program_result {
                    Ok(program) => {
                        let qresults = quality::measure_program_quality(&program, method, bits);
                        if let Some(summary) =
                            quality::summarize_quality(&model_cfg.name, &qresults)
                        {
                            let mut metadata = HashMap::new();
                            metadata.insert("method".to_string(), method.to_string());
                            metadata.insert("bits".to_string(), bits.to_string());
                            metadata.insert(
                                "avg_psnr_db".to_string(),
                                format!("{:.1}", summary.avg_psnr_db),
                            );
                            metadata
                                .insert("avg_mse".to_string(), format!("{:.6}", summary.avg_mse));
                            metadata.insert(
                                "compression_ratio".to_string(),
                                format!("{:.1}", summary.avg_compression_ratio),
                            );

                            // Quality benchmarks don't have latency data, create a minimal result
                            let dummy_result = crate::stats::AggregatedResult {
                                config_label: format!("{}/{}", model_cfg.name, opt_cfg.name),
                                pooled: crate::stats::compute_stats("quality", &[0.0]),
                                per_run_means: vec![],
                                between_run_stddev: 0.0,
                                runs: 1,
                            };

                            results.push(BenchmarkResult {
                                suite: self.id().to_string(),
                                model: model_cfg.name.clone(),
                                optimization: opt_cfg.name.clone(),
                                backend: "quality".to_string(),
                                variant: None,
                                result: dummy_result,
                                gpu_memory_mb: None,
                                load_time_ms: None,
                                perplexity: None,
                                metadata,
                            });

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

        Ok(results)
    }
}
