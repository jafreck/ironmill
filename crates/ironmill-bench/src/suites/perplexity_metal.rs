//! Metal perplexity evaluation suite.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;

use crate::perplexity;
use crate::suite::{BackendKind, BenchmarkContext, BenchmarkResult, BenchmarkSuite};

/// Result of a perplexity evaluation on a loaded engine.
#[cfg(feature = "metal")]
pub(crate) struct PerplexityEvalResult {
    pub ppl: f64,
    pub avg_ce: f64,
    pub num_tokens: usize,
    pub num_seqs: usize,
}

/// Run perplexity evaluation on an already-loaded engine.
///
/// Reusable by any suite that has a `MetalInference` engine ready.
#[cfg(feature = "metal")]
pub(crate) fn run_perplexity_eval(
    engine: &mut ironmill_inference::metal::MetalInference,
    dataset: &perplexity::PerplexityDataset,
    num_seqs: usize,
    stride: usize,
    config_name: &str,
) -> Result<PerplexityEvalResult> {
    use ironmill_inference::engine::InferenceEngine;

    let full_tokens: Vec<u32> = dataset
        .sequences
        .iter()
        .take(num_seqs)
        .flatten()
        .copied()
        .collect();
    let max_length = dataset.seq_len;
    let windows = perplexity::sliding_window_schedule(full_tokens.len(), max_length, stride);
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
        anyhow::bail!("{config_name}: no valid losses computed");
    }

    let ppl = perplexity::perplexity_from_losses(&all_losses);
    let avg_ce = all_losses.iter().sum::<f64>() / all_losses.len() as f64;
    eprintln!("  ✓ {config_name}: PPL={ppl:.2}, Avg CE={avg_ce:.4}");

    Ok(PerplexityEvalResult {
        ppl,
        avg_ce,
        num_tokens: all_losses.len(),
        num_seqs,
    })
}

/// Parse perplexity-related settings from the benchmark context.
#[cfg(feature = "metal")]
pub(crate) struct PerplexitySettings {
    pub dataset: perplexity::PerplexityDataset,
    pub num_seqs: usize,
    pub stride: usize,
}

#[cfg(feature = "metal")]
impl PerplexitySettings {
    pub fn from_context(ctx: &BenchmarkContext) -> Result<Self> {
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

        Ok(Self {
            dataset,
            num_seqs,
            stride,
        })
    }
}

#[cfg(feature = "metal")]
fn build_ppl_result(
    suite_id: &str,
    model_name: &str,
    config_name: &str,
    gpu_mb: f64,
    load_time_ms: Option<f64>,
    eval: &PerplexityEvalResult,
) -> BenchmarkResult {
    let dummy_result = crate::stats::AggregatedResult {
        config_label: format!("{model_name}/{config_name}/ppl"),
        pooled: crate::stats::compute_stats("ppl", &[0.0]),
        per_run_means: vec![],
        between_run_stddev: 0.0,
        runs: 1,
    };

    let mut metadata = HashMap::new();
    metadata.insert(
        "avg_cross_entropy".to_string(),
        format!("{:.4}", eval.avg_ce),
    );
    metadata.insert("num_tokens".to_string(), eval.num_tokens.to_string());
    metadata.insert("num_sequences".to_string(), eval.num_seqs.to_string());

    BenchmarkResult {
        suite: suite_id.to_string(),
        model: model_name.to_string(),
        optimization: config_name.to_string(),
        backend: format!("metal-{config_name}"),
        variant: None,
        result: dummy_result,
        gpu_memory_mb: Some(gpu_mb),
        load_time_ms,
        perplexity: Some(eval.ppl),
        metadata,
    }
}

// ---------------------------------------------------------------------------
// Standalone perplexity suite (loads its own engine)
// ---------------------------------------------------------------------------

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

    /// Only runs when explicitly selected via `--suite perplexity`.
    /// When `run_all` is used with `--perplexity`, the combined `decode-ppl`
    /// suite handles perplexity evaluation to avoid a redundant model load.
    fn should_run(&self, _ctx: &BenchmarkContext) -> bool {
        false
    }

    #[cfg(feature = "metal")]
    fn run(&self, ctx: &BenchmarkContext) -> Result<Vec<BenchmarkResult>> {
        let ppl_settings = PerplexitySettings::from_context(ctx)?;
        let mut results = Vec::new();

        for model_cfg in &ctx.matrix.models {
            for opt_cfg in &ctx.matrix.optimizations {
                let mut gpu_config = super::decode::build_metal_config(opt_cfg);
                gpu_config.kernel_timing = ctx.extra.contains_key("kernel_timing");
                let config_name = &opt_cfg.name;
                eprintln!("  Evaluating {config_name}...");

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

                match run_perplexity_eval(
                    &mut handle.engine,
                    &ppl_settings.dataset,
                    ppl_settings.num_seqs,
                    ppl_settings.stride,
                    config_name,
                ) {
                    Ok(eval) => {
                        results.push(build_ppl_result(
                            self.id(),
                            &model_cfg.name,
                            config_name,
                            handle.gpu_mb,
                            Some(handle.load_time_ms),
                            &eval,
                        ));
                    }
                    Err(e) => eprintln!("  ⚠ {e}"),
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

// ---------------------------------------------------------------------------
// Combined decode + perplexity suite (single model load)
// ---------------------------------------------------------------------------

pub struct MetalDecodePerplexitySuite;

impl BenchmarkSuite for MetalDecodePerplexitySuite {
    fn name(&self) -> &str {
        "Metal Decode + Perplexity"
    }

    fn id(&self) -> &str {
        "decode-ppl"
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
        let ppl_settings = PerplexitySettings::from_context(ctx)?;
        let mut results = Vec::new();

        for model_cfg in &ctx.matrix.models {
            for opt_cfg in &ctx.matrix.optimizations {
                let mut gpu_config = super::decode::build_metal_config(opt_cfg);
                gpu_config.kernel_timing = ctx.extra.contains_key("kernel_timing");
                let config_name = &opt_cfg.name;
                eprintln!("  Metal/{config_name}: {}...", model_cfg.name);

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

                // Decode latency benchmark
                let label = format!("{}/metal-{config_name}", model_cfg.name);
                match super::decode::run_decode_loop(
                    &mut handle.engine,
                    ctx.matrix.settings.warmup,
                    ctx.matrix.settings.iterations,
                    ctx.matrix.settings.runs,
                    &label,
                ) {
                    Ok((mut aggregated, tok_per_sec)) => {
                        aggregated.pooled.tokens_per_sec = Some(tok_per_sec);
                        aggregated.pooled.decode_tok_per_sec = Some(tok_per_sec);
                        eprintln!(
                            "  ✓ {config_name}: {:.2}ms/tok ({:.1} tok/s) ± {:.2}ms",
                            aggregated.pooled.mean, tok_per_sec, aggregated.pooled.stddev,
                        );
                        results.push(BenchmarkResult {
                            suite: "decode".to_string(),
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
                    Err(e) => {
                        eprintln!("  ✗ decode failed: {e}");
                    }
                }

                // Perplexity evaluation (reuses the same loaded engine)
                eprintln!("  Evaluating perplexity for {config_name}...");
                match run_perplexity_eval(
                    &mut handle.engine,
                    &ppl_settings.dataset,
                    ppl_settings.num_seqs,
                    ppl_settings.stride,
                    config_name,
                ) {
                    Ok(eval) => {
                        results.push(build_ppl_result(
                            "perplexity",
                            &model_cfg.name,
                            config_name,
                            handle.gpu_mb,
                            Some(handle.load_time_ms),
                            &eval,
                        ));
                    }
                    Err(e) => eprintln!("  ⚠ perplexity: {e}"),
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
