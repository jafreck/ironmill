//! Metal decode latency benchmark suite.

use std::collections::HashMap;

use anyhow::Result;

use crate::config;
use crate::stats::{aggregate_runs, compute_stats};
use crate::suite::{BackendKind, BenchmarkContext, BenchmarkResult, BenchmarkSuite};

pub struct MetalDecodeSuite;

/// Shared context for loading a Metal engine with the right weight quantization.
#[cfg(feature = "metal")]
pub(crate) struct MetalEngineHandle {
    pub engine: ironmill_inference::metal::MetalInference,
    pub gpu_mb: f64,
    #[allow(dead_code)]
    pub gpu_growth_mb: f64,
    pub load_time_ms: f64,
    #[allow(dead_code)]
    pub config_name: String,
}

#[cfg(feature = "metal")]
pub(crate) fn load_metal_engine(
    model_cfg: &crate::config::ModelConfig,
    opt: &crate::config::OptConfig,
    gpu_config: &ironmill_inference::metal::MetalConfig,
    config_name: &str,
) -> Result<MetalEngineHandle> {
    use ironmill_compile::weights::SafeTensorsProvider;
    use ironmill_compile::weights::quantized::{D2QuantConfig, QuantizedWeightProvider};
    use ironmill_inference::metal::MetalInference;

    let model_dir = if let Some(ref md) = model_cfg.model_dir {
        md.clone()
    } else if model_cfg.path.is_dir() {
        model_cfg.path.clone()
    } else if let Some(parent) = model_cfg.path.parent() {
        parent.to_path_buf()
    } else {
        anyhow::bail!("cannot determine model directory for {}", model_cfg.name);
    };

    let provider = SafeTensorsProvider::load(&model_dir)?;

    let mut engine = MetalInference::new(gpu_config.clone()).map_err(|e| anyhow::anyhow!("{e}"))?;

    let load_start = std::time::Instant::now();
    let gpu_before = engine.gpu_allocated_bytes();

    let d2quant_bits = opt.d2quant.unwrap_or(0);
    if d2quant_bits > 0 {
        eprintln!("    JIT D2Quant-{d2quant_bits} quantization...");
        let d2q_config = D2QuantConfig {
            bits: d2quant_bits,
            group_size: 128,
            outlier_threshold: 0.99,
        };
        let q_provider = QuantizedWeightProvider::new(&provider, d2q_config);
        engine
            .load_weights(&q_provider, gpu_config.clone())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        eprintln!("    calibrating DAC...");
        let dac_tokens: Vec<u32> = (100..2148).collect();
        if let Err(e) = engine.calibrate_dac(&provider, &dac_tokens) {
            eprintln!("    ⚠ DAC calibration failed: {e}");
        }
    } else if opt.int4 {
        eprintln!("    JIT INT4 quantization...");
        let int4_config = ironmill_compile::weights::quantized::AffineQuantConfig::default();
        let q_provider = QuantizedWeightProvider::new_int4(&provider, int4_config);
        engine
            .load_weights(&q_provider, gpu_config.clone())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    } else {
        engine
            .load_weights(&provider, gpu_config.clone())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }

    let load_time_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    let gpu_after = engine.gpu_allocated_bytes();
    let gpu_mb = gpu_after as f64 / (1024.0 * 1024.0);
    let gpu_growth_mb = (gpu_after - gpu_before) as f64 / (1024.0 * 1024.0);

    eprintln!(
        "  ✓ loaded in {load_time_ms:.1}ms (GPU: {gpu_mb:.1} MB, model: {gpu_growth_mb:.1} MB)"
    );

    Ok(MetalEngineHandle {
        engine,
        gpu_mb,
        gpu_growth_mb,
        load_time_ms,
        config_name: config_name.to_string(),
    })
}

#[cfg(feature = "metal")]
pub(crate) fn build_metal_config(
    opt: &crate::config::OptConfig,
) -> ironmill_inference::metal::MetalConfig {
    let mut c = ironmill_inference::metal::MetalConfig::default();
    c.max_seq_len = opt.max_seq_len;
    match opt.kv_quant {
        config::KvQuantMode::None => {
            c.enable_turboquant = false;
        }
        config::KvQuantMode::TurboInt4 => {
            c.enable_turboquant = true;
            c.n_bits = 4;
        }
        config::KvQuantMode::TurboInt8 => {
            c.enable_turboquant = true;
            c.n_bits = 8;
        }
        config::KvQuantMode::TurboInt8Qjl => {
            c.enable_turboquant = true;
            c.n_bits = 8;
        }
    }
    c
}

#[cfg(feature = "metal")]
pub(crate) fn run_decode_loop(
    engine: &mut ironmill_inference::metal::MetalInference,
    warmup: usize,
    iterations: usize,
    runs: usize,
    label_prefix: &str,
) -> Result<(crate::stats::AggregatedResult, f64)> {
    use ironmill_inference::engine::InferenceEngine;

    let prompt_tokens: Vec<u32> = vec![9707, 1879];
    let mut run_results = Vec::new();

    for run_idx in 0..runs {
        engine.reset();

        let warmup_tokens: Vec<u32> = if warmup > 0 {
            let mut tokens = prompt_tokens.clone();
            tokens.resize(prompt_tokens.len() + warmup, 9707);
            tokens
        } else {
            prompt_tokens.clone()
        };

        engine
            .prefill(&warmup_tokens)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let mut last_token = warmup_tokens.last().copied().unwrap_or(0);
        let mut latencies = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            let t0 = std::time::Instant::now();
            match engine.decode_step(last_token) {
                Ok(logits) => {
                    let elapsed = t0.elapsed();
                    latencies.push(elapsed);
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
                    eprintln!("  ✗ decode failed: {e}");
                    break;
                }
            }
        }

        if latencies.is_empty() {
            continue;
        }

        let latencies_ms: Vec<f64> = latencies.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
        let label = format!("{label_prefix}/run{run_idx}");
        run_results.push(compute_stats(&label, &latencies_ms));
    }

    if run_results.is_empty() {
        anyhow::bail!("all runs failed");
    }

    let aggregated = aggregate_runs(label_prefix, &run_results);
    let tok_per_sec = if aggregated.pooled.median > 0.0 {
        1000.0 / aggregated.pooled.median
    } else {
        0.0
    };

    Ok((aggregated, tok_per_sec))
}

impl BenchmarkSuite for MetalDecodeSuite {
    fn name(&self) -> &str {
        "Metal Decode Latency"
    }

    fn id(&self) -> &str {
        "decode"
    }

    fn supported_backends(&self) -> &[BackendKind] {
        &[BackendKind::Metal]
    }

    fn should_run(&self, ctx: &BenchmarkContext) -> bool {
        ctx.matrix.settings.backends.iter().any(|b| b == "metal")
    }

    #[cfg(feature = "metal")]
    fn run(&self, ctx: &BenchmarkContext) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();

        for model_cfg in &ctx.matrix.models {
            for opt_cfg in &ctx.matrix.optimizations {
                let gpu_config = build_metal_config(opt_cfg);
                let config_name = &opt_cfg.name;
                eprintln!("  Metal/{config_name}: {}...", model_cfg.name);

                let mut handle =
                    match load_metal_engine(model_cfg, opt_cfg, &gpu_config, config_name) {
                        Ok(h) => h,
                        Err(e) => {
                            eprintln!("  ✗ {e}");
                            continue;
                        }
                    };

                let label = format!("{}/metal-{config_name}", model_cfg.name);
                let (mut aggregated, tok_per_sec) = match run_decode_loop(
                    &mut handle.engine,
                    ctx.matrix.settings.warmup,
                    ctx.matrix.settings.iterations,
                    ctx.matrix.settings.runs,
                    &label,
                ) {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("  ✗ decode loop failed: {e}");
                        continue;
                    }
                };

                aggregated.pooled.tokens_per_sec = Some(tok_per_sec);
                aggregated.pooled.decode_tok_per_sec = Some(tok_per_sec);

                eprintln!(
                    "  ✓ {config_name}: {:.2}ms/tok ({:.1} tok/s) ± {:.2}ms | GPU: {:.1} MB",
                    aggregated.pooled.mean, tok_per_sec, aggregated.pooled.stddev, handle.gpu_mb
                );

                results.push(BenchmarkResult {
                    suite: self.id().to_string(),
                    model: model_cfg.name.clone(),
                    optimization: opt_cfg.name.clone(),
                    backend: format!("metal-{config_name}"),
                    variant: None,
                    result: aggregated,
                    gpu_memory_mb: Some(handle.gpu_mb),
                    load_time_ms: Some(handle.load_time_ms),
                    perplexity: None,
                    metadata: HashMap::new(),
                });
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
