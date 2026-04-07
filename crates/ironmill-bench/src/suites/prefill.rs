//! Metal prefill throughput benchmark suite.

use std::collections::HashMap;

use anyhow::Result;

use crate::stats::compute_stats;
use crate::suite::{BackendKind, BenchmarkContext, BenchmarkResult, BenchmarkSuite};

pub struct MetalPrefillSuite;

/// Default prefill lengths to benchmark.
const DEFAULT_PREFILL_LENGTHS: &[usize] = &[128, 512, 1024, 2048, 4096];

impl BenchmarkSuite for MetalPrefillSuite {
    fn name(&self) -> &str {
        "Metal Prefill Throughput"
    }

    fn id(&self) -> &str {
        "prefill"
    }

    fn supported_backends(&self) -> &[BackendKind] {
        &[BackendKind::Metal]
    }

    fn should_run(&self, ctx: &BenchmarkContext) -> bool {
        ctx.matrix.settings.backends.iter().any(|b| b == "metal")
            && ctx.extra.contains_key("prefill_bench")
    }

    #[cfg(feature = "metal")]
    fn run(&self, ctx: &BenchmarkContext) -> Result<Vec<BenchmarkResult>> {
        use ironmill_inference::engine::InferenceEngine;

        let mut results = Vec::new();

        let prefill_lengths: Vec<usize> = ctx
            .extra
            .get("prefill_lengths")
            .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
            .unwrap_or_else(|| DEFAULT_PREFILL_LENGTHS.to_vec());

        for model_cfg in &ctx.matrix.models {
            for opt_cfg in &ctx.matrix.optimizations {
                let gpu_config = super::decode::build_metal_config(opt_cfg);
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

                for &prefill_len in &prefill_lengths {
                    if prefill_len > gpu_config.max_seq_len {
                        eprintln!(
                            "  ⚠ skipping prefill_len={prefill_len}: exceeds max_seq_len={}",
                            gpu_config.max_seq_len
                        );
                        continue;
                    }

                    let tokens: Vec<u32> = vec![9707u32; prefill_len];
                    let mut prefill_times_ms = Vec::new();

                    for _ in 0..ctx.matrix.settings.runs.max(1) {
                        handle.engine.reset();
                        let t0 = std::time::Instant::now();
                        match handle.engine.prefill(&tokens) {
                            Ok(_) => {
                                prefill_times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
                            }
                            Err(e) => {
                                eprintln!("  ✗ prefill failed at len={prefill_len}: {e}");
                                break;
                            }
                        }
                    }

                    if prefill_times_ms.is_empty() {
                        continue;
                    }

                    let mean_ms =
                        prefill_times_ms.iter().sum::<f64>() / prefill_times_ms.len() as f64;
                    let tok_per_sec = prefill_len as f64 / (mean_ms / 1000.0);

                    let label = format!(
                        "{}/metal-{config_name}/prefill-{prefill_len}",
                        model_cfg.name
                    );
                    let mut stats = compute_stats(&label, &prefill_times_ms);
                    stats.tokens_per_sec = Some(tok_per_sec);

                    let stddev = if prefill_times_ms.len() > 1 {
                        let var = prefill_times_ms
                            .iter()
                            .map(|v| (v - mean_ms).powi(2))
                            .sum::<f64>()
                            / (prefill_times_ms.len() - 1) as f64;
                        var.sqrt()
                    } else {
                        0.0
                    };

                    eprintln!(
                        "  ✓ {config_name}[prefill={prefill_len}]: {mean_ms:.2}ms ({tok_per_sec:.0} tok/s) ± {stddev:.2}ms"
                    );

                    let aggregated = crate::stats::AggregatedResult {
                        config_label: label,
                        pooled: stats,
                        per_run_means: prefill_times_ms,
                        between_run_stddev: stddev,
                        runs: ctx.matrix.settings.runs.max(1),
                    };

                    results.push(BenchmarkResult {
                        suite: self.id().to_string(),
                        model: model_cfg.name.clone(),
                        optimization: opt_cfg.name.clone(),
                        backend: format!("metal-{config_name}"),
                        variant: Some(format!("prefill-{prefill_len}")),
                        result: aggregated,
                        gpu_memory_mb: Some(handle.gpu_mb),
                        load_time_ms: Some(handle.load_time_ms),
                        perplexity: None,
                        metadata: {
                            let mut m = HashMap::new();
                            m.insert("prefill_length".to_string(), prefill_len.to_string());
                            m.insert("tok_per_sec".to_string(), format!("{tok_per_sec:.0}"));
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
