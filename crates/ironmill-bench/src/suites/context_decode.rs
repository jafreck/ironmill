//! Metal decode-at-context benchmark suite.
//!
//! Measures decode latency after prefilling a long context.

use std::collections::HashMap;

use anyhow::Result;

use crate::stats::{aggregate_runs, compute_stats};
use crate::suite::{BackendKind, BenchmarkContext, BenchmarkResult, BenchmarkSuite};

pub struct MetalContextDecodeSuite;

impl BenchmarkSuite for MetalContextDecodeSuite {
    fn name(&self) -> &str {
        "Metal Decode at Long Context"
    }

    fn id(&self) -> &str {
        "context-decode"
    }

    fn supported_backends(&self) -> &[BackendKind] {
        &[BackendKind::Metal]
    }

    fn should_run(&self, ctx: &BenchmarkContext) -> bool {
        ctx.matrix.settings.backends.iter().any(|b| b == "metal")
            && ctx.extra.contains_key("context_lengths")
    }

    #[cfg(feature = "metal")]
    fn run(&self, ctx: &BenchmarkContext) -> Result<Vec<BenchmarkResult>> {
        use ironmill_inference::engine::InferenceEngine;

        let context_lengths: Vec<usize> = ctx
            .extra
            .get("context_lengths")
            .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
            .unwrap_or_default();

        if context_lengths.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = Vec::new();

        for model_cfg in &ctx.matrix.models {
            for opt_cfg in &ctx.matrix.optimizations {
                let mut gpu_config = super::decode::build_metal_config(opt_cfg);
                gpu_config.kernel_timing = ctx.extra.contains_key("kernel_timing");
                let config_name = &opt_cfg.name;

                let mut handle = match super::decode::load_metal_engine(
                    model_cfg,
                    opt_cfg,
                    &gpu_config,
                    config_name,
                ) {
                    Ok(h) => h,
                    Err(e) => {
                        eprintln!("  ✗ {e}");
                        continue;
                    }
                };

                for &ctx_len in &context_lengths {
                    if ctx_len > gpu_config.max_seq_len {
                        eprintln!(
                            "  ⚠ skipping ctx={ctx_len}: exceeds max_seq_len={}",
                            gpu_config.max_seq_len
                        );
                        continue;
                    }

                    let prefill_tokens: Vec<u32> = vec![9707u32; ctx_len];
                    let mut run_results = Vec::new();

                    for run_idx in 0..ctx.matrix.settings.runs {
                        handle.engine.reset();

                        if let Err(e) = handle.engine.prefill(&prefill_tokens) {
                            eprintln!("  ✗ prefill failed at ctx={ctx_len}: {e}");
                            continue;
                        }

                        let mut last_token = 9707u32;
                        let mut latencies = Vec::with_capacity(ctx.matrix.settings.iterations);

                        for _ in 0..ctx.matrix.settings.iterations {
                            let t0 = std::time::Instant::now();
                            match handle.engine.decode_step(last_token) {
                                Ok(logits) => {
                                    latencies.push(t0.elapsed());
                                    last_token = logits
                                        .iter()
                                        .enumerate()
                                        .max_by(|(_, a), (_, b)| {
                                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                                        })
                                        .map(|(i, _)| i as u32)
                                        .unwrap_or(0);
                                }
                                Err(e) => {
                                    eprintln!("  ✗ decode failed at ctx={ctx_len}: {e}");
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
                            "{}/metal-{config_name}[ctx={ctx_len}]/run{run_idx}",
                            model_cfg.name
                        );
                        run_results.push(compute_stats(&label, &latencies_ms));
                    }

                    if run_results.is_empty() {
                        continue;
                    }

                    let label = format!("{}/metal-{config_name}[ctx={ctx_len}]", model_cfg.name);
                    let mut aggregated = aggregate_runs(&label, &run_results);

                    let tok_per_sec = if aggregated.pooled.median > 0.0 {
                        1000.0 / aggregated.pooled.median
                    } else {
                        0.0
                    };
                    aggregated.pooled.tokens_per_sec = Some(tok_per_sec);
                    aggregated.pooled.decode_tok_per_sec = Some(tok_per_sec);

                    eprintln!(
                        "  ✓ {config_name}[ctx={ctx_len}]: {:.2}ms/tok ({:.1} tok/s) ± {:.2}ms",
                        aggregated.pooled.mean, tok_per_sec, aggregated.pooled.stddev
                    );

                    results.push(BenchmarkResult {
                        suite: self.id().to_string(),
                        model: model_cfg.name.clone(),
                        optimization: opt_cfg.name.clone(),
                        backend: format!("metal-{config_name}"),
                        variant: Some(format!("ctx-{ctx_len}")),
                        result: aggregated,
                        gpu_memory_mb: Some(handle.gpu_mb),
                        load_time_ms: Some(handle.load_time_ms),
                        perplexity: None,
                        metadata: {
                            let mut m = HashMap::new();
                            m.insert("context_length".to_string(), ctx_len.to_string());
                            m
                        },
                    });
                }
            }
        }

        Ok(results)
    }

    #[cfg(not(feature = "metal"))]
    fn run(&self, _ctx: &BenchmarkContext) -> Result<Vec<BenchmarkResult>> {
        eprintln!("  Metal backend not available (compile with --features metal)");
        Ok(Vec::new())
    }
}
