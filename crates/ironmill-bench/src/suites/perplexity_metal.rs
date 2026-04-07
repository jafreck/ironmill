//! Metal perplexity evaluation suite.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;

use crate::perplexity;
use crate::suite::{BackendKind, BenchmarkContext, BenchmarkResult, BenchmarkSuite};

pub struct MetalPerplexitySuite;

impl BenchmarkSuite for MetalPerplexitySuite {
    fn name(&self) -> &str {
        "Metal Perplexity Evaluation"
    }

    fn id(&self) -> &str {
        "perplexity"
    }

    fn supported_backends(&self) -> &[BackendKind] {
        &[BackendKind::Metal]
    }

    fn should_run(&self, ctx: &BenchmarkContext) -> bool {
        ctx.matrix.settings.backends.iter().any(|b| b == "metal")
            && ctx.extra.contains_key("perplexity")
    }

    #[cfg(feature = "metal")]
    fn run(&self, ctx: &BenchmarkContext) -> Result<Vec<BenchmarkResult>> {
        use ironmill_compile::weights::SafeTensorsProvider;
        use ironmill_compile::weights::quantized::{D2QuantConfig, QuantizedWeightProvider};
        use ironmill_inference::engine::InferenceEngine;
        use ironmill_inference::metal::MetalInference;

        let dataset_path = ctx
            .extra
            .get("perplexity_dataset")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("tests/fixtures/quality/wikitext2-qwen3.json"));

        let dataset = perplexity::PerplexityDataset::load(&dataset_path)?;
        eprintln!(
            "  Dataset: {} ({} seqs × {} tokens)",
            dataset.name, dataset.num_sequences, dataset.seq_len
        );

        let num_seqs: usize = ctx
            .extra
            .get("perplexity_sequences")
            .and_then(|s| s.parse().ok())
            .unwrap_or(50)
            .min(dataset.num_sequences);

        let stride: usize = ctx
            .extra
            .get("perplexity_stride")
            .and_then(|s| s.parse().ok())
            .unwrap_or(512);

        let mut results = Vec::new();

        for model_cfg in &ctx.matrix.models {
            let model_dir = if let Some(ref md) = model_cfg.model_dir {
                md.clone()
            } else if model_cfg.path.is_dir() {
                model_cfg.path.clone()
            } else {
                model_cfg.path.parent().unwrap().to_path_buf()
            };

            let provider = SafeTensorsProvider::load(&model_dir)?;

            for opt_cfg in &ctx.matrix.optimizations {
                let gpu_config = super::decode::build_metal_config(opt_cfg);
                let config_name = &opt_cfg.name;
                eprintln!("  Evaluating {config_name}...");

                let mut engine =
                    MetalInference::new(gpu_config.clone()).map_err(|e| anyhow::anyhow!("{e}"))?;
                let gpu_before = engine.gpu_allocated_bytes();

                let d2quant_bits = opt_cfg.d2quant.unwrap_or(0);
                if d2quant_bits > 0 {
                    let d2q_config = D2QuantConfig {
                        bits: d2quant_bits,
                        group_size: 128,
                        outlier_threshold: 0.99,
                    };
                    let q_provider = QuantizedWeightProvider::new(&provider, d2q_config);
                    engine
                        .load_weights(&q_provider, gpu_config.clone())
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    let dac_tokens: Vec<u32> = dataset
                        .sequences
                        .iter()
                        .flatten()
                        .copied()
                        .take(2048)
                        .collect();
                    let _ = engine.calibrate_dac(&provider, &dac_tokens);
                } else if opt_cfg.int4 {
                    let int4_config = if let Some(ref awq_dir) = opt_cfg.awq_calib_dir {
                        let mag_path = std::path::Path::new(awq_dir).join("awq_magnitudes.json");
                        let magnitudes: std::collections::HashMap<String, Vec<f32>> =
                            serde_json::from_str(
                                &std::fs::read_to_string(&mag_path)
                                    .map_err(|e| anyhow::anyhow!("{e}"))?,
                            )
                            .map_err(|e| anyhow::anyhow!("{e}"))?;
                        ironmill_compile::weights::quantized::AffineQuantConfig::int4_awq(
                            128, magnitudes,
                        )
                    } else {
                        ironmill_compile::weights::quantized::AffineQuantConfig::default()
                    };
                    let q_provider = QuantizedWeightProvider::new_int4(&provider, int4_config);
                    engine
                        .load_weights(&q_provider, gpu_config.clone())
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                } else {
                    engine
                        .load_weights(&provider, gpu_config.clone())
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                }

                let gpu_after = engine.gpu_allocated_bytes();
                let gpu_mb = gpu_after as f64 / (1024.0 * 1024.0);
                let model_mb = (gpu_after - gpu_before) as f64 / (1024.0 * 1024.0);
                eprintln!("  GPU memory: {gpu_mb:.1} MB (model: {model_mb:.1} MB)");

                let full_tokens: Vec<u32> = dataset
                    .sequences
                    .iter()
                    .take(num_seqs)
                    .flatten()
                    .copied()
                    .collect();
                let max_length = dataset.seq_len;
                let windows =
                    perplexity::sliding_window_schedule(full_tokens.len(), max_length, stride);
                let num_windows = windows.len();

                let mut all_losses = Vec::new();
                let start = std::time::Instant::now();

                for (win_idx, step) in windows.iter().enumerate() {
                    engine.reset();
                    let window = &full_tokens[step.begin..step.end];
                    if window.len() < 2 {
                        continue;
                    }
                    let all_logits = engine
                        .prefill_all_logits(window)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    for pos in step.loss_start..window.len() - 1 {
                        let target = window[pos + 1];
                        let ce = perplexity::cross_entropy(&all_logits[pos], target);
                        all_losses.push(ce);
                    }
                    let running_ppl = perplexity::perplexity_from_losses(&all_losses);
                    let elapsed = start.elapsed().as_secs_f64();
                    let tok_per_sec = all_losses.len() as f64 / elapsed;
                    eprintln!(
                        "  [{}/{}] PPL: {:.2} ({} tokens, {:.1} tok/s)",
                        win_idx + 1,
                        num_windows,
                        running_ppl,
                        all_losses.len(),
                        tok_per_sec,
                    );
                }

                if all_losses.is_empty() {
                    eprintln!("  ⚠ {config_name}: no valid losses computed");
                    continue;
                }

                let ppl = perplexity::perplexity_from_losses(&all_losses);
                let avg_ce = all_losses.iter().sum::<f64>() / all_losses.len() as f64;
                eprintln!(
                    "  ✓ {config_name}: PPL={:.2}, Avg CE={:.4}, GPU={gpu_mb:.1} MB",
                    ppl, avg_ce
                );

                let dummy_result = crate::stats::AggregatedResult {
                    config_label: format!("{}/{config_name}/ppl", model_cfg.name),
                    pooled: crate::stats::compute_stats("ppl", &[0.0]),
                    per_run_means: vec![],
                    between_run_stddev: 0.0,
                    runs: 1,
                };

                let mut metadata = HashMap::new();
                metadata.insert("avg_cross_entropy".to_string(), format!("{avg_ce:.4}"));
                metadata.insert("num_tokens".to_string(), all_losses.len().to_string());
                metadata.insert("num_sequences".to_string(), num_seqs.to_string());

                results.push(BenchmarkResult {
                    suite: self.id().to_string(),
                    model: model_cfg.name.clone(),
                    optimization: opt_cfg.name.clone(),
                    backend: format!("metal-{config_name}"),
                    variant: None,
                    result: dummy_result,
                    gpu_memory_mb: Some(gpu_mb),
                    load_time_ms: None,
                    perplexity: Some(ppl),
                    metadata,
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
