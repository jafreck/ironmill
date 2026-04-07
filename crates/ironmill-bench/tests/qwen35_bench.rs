//! Qwen 3.5-0.8B Metal GPU benchmark tests.
//!
//! Validates:
//! - FP16 baseline perplexity matches HuggingFace reference (PPL < 20)
//! - TurboQuant INT4 KV cache maintains acceptable PPL (< 5% degradation)
//! - D2Quant 3-bit simulation produces bounded PPL (not NaN/Inf)
//! - Decode throughput is functional for all configs
//! - GPU memory is reported correctly
//!
//! Requires: Metal GPU, Qwen3.5-0.8B SafeTensors, wikitext2-qwen35 dataset.
//! Run: `cargo test -p ironmill-bench --features metal -- qwen35 --ignored --nocapture`

#[cfg(feature = "metal")]
mod qwen35_bench {
    use std::path::PathBuf;
    use std::time::Instant;

    use ironmill_compile::weights::{SafeTensorsProvider, WeightProvider};
    use ironmill_inference::InferenceEngine;
    use ironmill_inference::metal::{MetalConfig, MetalInference};

    fn fixture_path(name: &str) -> PathBuf {
        let manifest = env!("CARGO_MANIFEST_DIR");
        PathBuf::from(manifest)
            .join("../../tests/fixtures")
            .join(name)
    }

    fn qwen35_model_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/Qwen3.5-4B")
    }

    fn skip_if_missing(path: &PathBuf, label: &str) {
        if !path.exists() {
            panic!(
                "Fixture {label} not found at {}. Download model weights first.",
                path.display()
            );
        }
    }

    /// Build a MetalInference engine from SafeTensors and return (engine, gpu_mb, load_ms).
    fn load_engine(config: MetalConfig) -> (MetalInference, f64, f64) {
        let model_dir = qwen35_model_dir();
        skip_if_missing(&model_dir, "Qwen3.5-4B");

        let provider =
            SafeTensorsProvider::load(&model_dir).expect("SafeTensorsProvider::load failed");

        let mut engine = MetalInference::new(config.clone()).expect("MetalInference::new failed");
        let gpu_before = engine.gpu_allocated_bytes();
        let t0 = Instant::now();
        engine
            .load_weights(&provider, config)
            .expect("load_weights failed");
        let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let gpu_mb = engine.gpu_allocated_bytes() as f64 / (1024.0 * 1024.0);
        let _ = gpu_before;
        (engine, gpu_mb, load_ms)
    }

    fn fp16_config() -> MetalConfig {
        let mut c = MetalConfig::default();
        c.enable_turboquant = false;
        c.prefill_chunk_size = Some(256);
        c
    }

    fn tq_int4_config() -> MetalConfig {
        let mut c = MetalConfig::default();
        c.enable_turboquant = true;
        c.n_bits = 4;
        c.prefill_chunk_size = Some(256);
        c
    }

    /// Benchmark decode throughput: prefill a short prompt, then decode N tokens.
    fn bench_decode(engine: &mut MetalInference, decode_tokens: usize) -> (f64, f64) {
        engine.reset();
        let prompt: Vec<u32> = vec![9419, 1814]; // "Hello world"
        engine.prefill(&prompt).expect("prefill failed");

        let mut last_token = *prompt.last().unwrap();
        let mut latencies = Vec::with_capacity(decode_tokens);

        for _ in 0..decode_tokens {
            let t0 = Instant::now();
            let logits = engine.decode_step(last_token).expect("decode_step failed");
            latencies.push(t0.elapsed().as_secs_f64() * 1000.0);
            last_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
        }

        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_ms = latencies[latencies.len() / 2];
        let tok_s = 1000.0 / median_ms;
        (median_ms, tok_s)
    }

    /// Compute perplexity using sliding-window evaluation.
    fn eval_perplexity(
        engine: &mut MetalInference,
        sequences: &[Vec<u32>],
        max_seqs: usize,
        stride: usize,
    ) -> f64 {
        let full_tokens: Vec<u32> = sequences.iter().take(max_seqs).flatten().copied().collect();
        // Cap window size at 512 to avoid Metal buffer limits on large models
        // (logits buffer = window × vocab × 2 bytes). 512 × 248K × 2 = 254 MB.
        let seq_len = sequences.first().map(|s| s.len()).unwrap_or(2048);
        let max_length = seq_len.min(512);
        let windows = sliding_window_schedule(full_tokens.len(), max_length, stride);

        let mut all_losses = Vec::new();
        for step in &windows {
            engine.reset();
            let window = &full_tokens[step.begin..step.end];
            if window.len() < 2 {
                continue;
            }
            let all_logits = engine
                .prefill_all_logits(window)
                .expect("prefill_all_logits failed");
            for pos in step.loss_start..window.len() - 1 {
                let target = window[pos + 1];
                let ce = cross_entropy(&all_logits[pos], target);
                all_losses.push(ce);
            }
        }

        perplexity_from_losses(&all_losses)
    }

    fn cross_entropy(logits: &[f32], target: u32) -> f64 {
        let target = target as usize;
        if target >= logits.len() {
            return (logits.len() as f64).ln();
        }
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp: f64 = logits
            .iter()
            .map(|&x| ((x - max_logit) as f64).exp())
            .sum::<f64>()
            .ln()
            + max_logit as f64;
        -(logits[target] as f64 - log_sum_exp)
    }

    fn perplexity_from_losses(losses: &[f64]) -> f64 {
        if losses.is_empty() {
            return f64::INFINITY;
        }
        (losses.iter().sum::<f64>() / losses.len() as f64).exp()
    }

    struct WindowStep {
        begin: usize,
        end: usize,
        loss_start: usize,
    }

    fn sliding_window_schedule(
        total_tokens: usize,
        max_length: usize,
        stride: usize,
    ) -> Vec<WindowStep> {
        let mut steps = Vec::new();
        let mut prev_end: usize = 0;
        let mut begin: usize = 0;
        while begin < total_tokens {
            let end = (begin + max_length).min(total_tokens);
            if end <= 1 {
                break;
            }
            let trg_len = end - prev_end;
            let window_len = end - begin;
            let loss_start = window_len.saturating_sub(trg_len);
            steps.push(WindowStep {
                begin,
                end,
                loss_start,
            });
            prev_end = end;
            if end == total_tokens {
                break;
            }
            begin += stride;
        }
        steps
    }

    fn load_dataset() -> Vec<Vec<u32>> {
        let path = fixture_path("quality/wikitext2-qwen35.json");
        skip_if_missing(&path, "wikitext2-qwen35.json");
        let data: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        data["sequences"]
            .as_array()
            .unwrap()
            .iter()
            .map(|seq| {
                seq.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect()
            })
            .collect()
    }

    // ── Tests ────────────────────────────────────────────────────────

    /// PPL regression guard for INT4+TQ-INT8 — the primary deployment config.
    ///
    /// Baseline captured 2026-04-07 on M2 Max 64 GB (512-token windows):
    ///   FP16:          PPL ~12.3
    ///   INT4+TQ-INT8:  PPL ~13.3 (+8.6%)
    ///   INT4+TQ-INT8 GPU: 4675 MB
    ///
    /// Full-context (2048-token) baselines from bench harness:
    ///   FP16: PPL 7.37 | INT4+TQ-INT8: PPL 7.53 (+2.2%)
    ///
    /// Thresholds are calibrated for the 512-token test window. The INT4
    /// delta is larger here (~9%) than full-context (~2%) because shorter
    /// windows amplify quantization error. Real deployment uses full context.
    #[test]
    #[ignore] // requires Metal GPU + Qwen3.5-4B + wikitext2-qwen35 dataset
    fn qwen35_int4_tq_int8_ppl_regression() {
        let sequences = load_dataset();
        let ppl_seqs = 1;
        let stride = 512;

        // FP16 baseline PPL
        let (mut engine_fp16, fp16_gpu, _) = load_engine(fp16_config());
        let fp16_ppl = eval_perplexity(&mut engine_fp16, &sequences, ppl_seqs, stride);
        drop(engine_fp16);

        // INT4 + TQ-INT8
        let mut int4_tq_config = MetalConfig::default();
        int4_tq_config.enable_turboquant = true;
        int4_tq_config.n_bits = 8;
        int4_tq_config.prefill_chunk_size = Some(256);

        let model_dir = qwen35_model_dir();
        let provider =
            SafeTensorsProvider::load(&model_dir).expect("SafeTensorsProvider::load failed");
        let int4_config = ironmill_compile::weights::quantized::AffineQuantConfig::default();
        let q_provider = ironmill_compile::weights::quantized::QuantizedWeightProvider::new_int4(
            &provider,
            int4_config,
        );

        let mut engine_int4 =
            MetalInference::new(int4_tq_config.clone()).expect("MetalInference::new failed");
        let gpu_before = engine_int4.gpu_allocated_bytes();
        engine_int4
            .load_weights(&q_provider, int4_tq_config)
            .expect("load_weights failed");
        let int4_gpu = engine_int4.gpu_allocated_bytes() as f64 / (1024.0 * 1024.0);
        let _ = gpu_before;

        let int4_ppl = eval_perplexity(&mut engine_int4, &sequences, ppl_seqs, stride);
        drop(engine_int4);

        let ppl_delta_pct = (int4_ppl - fp16_ppl) / fp16_ppl * 100.0;

        println!("\n  PPL Regression Check (INT4+TQ-INT8)");
        println!("  ────────────────────────────────────");
        println!("  FP16 baseline PPL: {fp16_ppl:.2}");
        println!("  INT4+TQ-INT8 PPL:  {int4_ppl:.2} ({ppl_delta_pct:+.1}%)");
        println!("  INT4+TQ-INT8 GPU:  {int4_gpu:.0} MB\n");

        assert!(
            fp16_ppl < 15.0,
            "FP16 PPL regression: expected <15.0 (512-token window), got {fp16_ppl:.2}"
        );
        assert!(
            ppl_delta_pct < 15.0,
            "INT4+TQ-INT8 PPL regression: expected <15% vs FP16, got {ppl_delta_pct:.1}%"
        );
        assert!(
            int4_gpu < 7000.0,
            "INT4+TQ-INT8 GPU memory regression: expected <7000 MB, got {int4_gpu:.0} MB"
        );
    }

    /// Full comparison: FP16 vs TQ-INT4 vs D2Q-3 vs D2Q-3+TQ-INT4.
    ///
    /// Measures PPL, tok/s, and GPU memory for all four configs in a single
    /// test, producing one summary table. This avoids redundant model loads
    /// and gives a complete picture in one run.
    #[test]
    #[ignore] // requires Metal GPU + Qwen3.5-4B + wikitext2-qwen35 dataset
    fn qwen35_full_comparison() {
        let sequences = load_dataset();
        let ppl_seqs = 5;
        let decode_tokens = 20;

        // ── Config 1: FP16 baseline ──
        let (mut engine_fp16, fp16_gpu, _) = load_engine(fp16_config());
        let (fp16_ms, fp16_tps) = bench_decode(&mut engine_fp16, decode_tokens);
        let fp16_ppl = eval_perplexity(&mut engine_fp16, &sequences, ppl_seqs, 512);
        drop(engine_fp16);

        // ── Config 2: FP16 weights + TurboQuant INT4 KV cache ──
        let (mut engine_tq, tq_gpu, _) = load_engine(tq_int4_config());
        let (tq_ms, tq_tps) = bench_decode(&mut engine_tq, decode_tokens);
        let tq_ppl = eval_perplexity(&mut engine_tq, &sequences, ppl_seqs, 512);
        drop(engine_tq);

        // ── Config 3: D2Quant 3-bit weights, FP16 KV cache ──
        let (mut engine_d2q_fp16, d2q_fp16_gpu, _) = load_engine(fp16_config());
        engine_d2q_fp16.weights_mut().apply_d2quant_simulation(3);
        let (d2q_fp16_ms, d2q_fp16_tps) = bench_decode(&mut engine_d2q_fp16, decode_tokens);
        let d2q_fp16_ppl = eval_perplexity(&mut engine_d2q_fp16, &sequences, ppl_seqs, 512);
        drop(engine_d2q_fp16);

        // ── Config 4: D2Quant 3-bit weights + TurboQuant INT4 KV cache ──
        let (mut engine_d2q_tq, d2q_tq_gpu, _) = load_engine(tq_int4_config());
        engine_d2q_tq.weights_mut().apply_d2quant_simulation(3);
        let (d2q_tq_ms, d2q_tq_tps) = bench_decode(&mut engine_d2q_tq, decode_tokens);
        let d2q_tq_ppl = eval_perplexity(&mut engine_d2q_tq, &sequences, ppl_seqs, 512);
        drop(engine_d2q_tq);

        // ── Summary table ──
        let tq_ppl_delta = (tq_ppl - fp16_ppl) / fp16_ppl * 100.0;
        let d2q_fp16_ppl_delta = (d2q_fp16_ppl - fp16_ppl) / fp16_ppl * 100.0;
        let d2q_tq_ppl_delta = (d2q_tq_ppl - fp16_ppl) / fp16_ppl * 100.0;

        println!("\n╔══════════════════════════════════════════════════════════════════╗");
        println!("║  Qwen3.5-4B Metal GPU Benchmark Comparison                     ║");
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║  Config              PPL      ΔPPL    tok/s    ms/tok   GPU MB  ║");
        println!("║  ─────────────────  ──────  ───────  ───────  ───────  ───────  ║");
        println!(
            "║  FP16 baseline      {:>6.2}       —   {:>6.1}   {:>6.2}   {:>6.0}  ║",
            fp16_ppl, fp16_tps, fp16_ms, fp16_gpu
        );
        println!(
            "║  FP16 + TQ-INT4     {:>6.2}  {:>+6.1}%  {:>6.1}   {:>6.2}   {:>6.0}  ║",
            tq_ppl, tq_ppl_delta, tq_tps, tq_ms, tq_gpu
        );
        println!(
            "║  D2Q-3              {:>6.2}  {:>+6.1}%  {:>6.1}   {:>6.2}   {:>6.0}  ║",
            d2q_fp16_ppl, d2q_fp16_ppl_delta, d2q_fp16_tps, d2q_fp16_ms, d2q_fp16_gpu
        );
        println!(
            "║  D2Q-3 + TQ-INT4    {:>6.2}  {:>+6.1}%  {:>6.1}   {:>6.2}   {:>6.0}  ║",
            d2q_tq_ppl, d2q_tq_ppl_delta, d2q_tq_tps, d2q_tq_ms, d2q_tq_gpu
        );
        println!("╚══════════════════════════════════════════════════════════════════╝\n");

        // ── Assertions ──
        // FP16 PPL should be reasonable for 4B model
        assert!(
            fp16_ppl > 8.0 && fp16_ppl < 20.0,
            "FP16 PPL should be 8–20 for 4B model, got {fp16_ppl:.2}"
        );
        assert!(fp16_tps > 5.0, "FP16 should be >5 tok/s, got {fp16_tps:.1}");

        // TQ-INT4 should degrade PPL <5%
        assert!(
            tq_ppl_delta < 5.0,
            "TQ-INT4 ΔPPL should be <5%, got {tq_ppl_delta:.1}%"
        );
        assert!(tq_tps > 5.0, "TQ-INT4 should be >5 tok/s, got {tq_tps:.1}");
        assert!(tq_gpu < fp16_gpu, "TQ-INT4 GPU should be < FP16");

        // D2Q-3 PPL will be significantly worse but finite
        assert!(
            d2q_fp16_ppl.is_finite() && d2q_fp16_ppl < 200.0,
            "D2Q-3 PPL should be finite and <200, got {d2q_fp16_ppl:.2}"
        );
        assert!(
            d2q_fp16_tps > 5.0,
            "D2Q-3 should be >5 tok/s, got {d2q_fp16_tps:.1}"
        );

        // D2Q-3+TQ PPL should be close to D2Q-3 alone (TQ adds minimal noise)
        let d2q_delta = ((d2q_tq_ppl - d2q_fp16_ppl) / d2q_fp16_ppl * 100.0).abs();
        assert!(
            d2q_delta < 10.0,
            "D2Q-3+TQ PPL should be within 10% of D2Q-3 alone, got {d2q_delta:.1}%"
        );
        assert!(
            d2q_tq_tps > 5.0,
            "D2Q-3+TQ should be >5 tok/s, got {d2q_tq_tps:.1}"
        );
    }

    #[test]
    #[ignore] // requires Metal GPU + Qwen3.5-0.8B
    fn qwen35_hidden_state_readback() {
        let (mut engine, _, _) = load_engine(fp16_config());
        let prompt: Vec<u32> = vec![9419, 1814];
        engine.prefill(&prompt).expect("prefill failed");

        let hidden = engine
            .last_hidden_state()
            .expect("last_hidden_state failed");
        let model_info = engine.model_info();
        let expected_len = model_info.hidden_size;

        assert_eq!(
            hidden.len(),
            expected_len,
            "hidden state length should be {expected_len}, got {}",
            hidden.len()
        );
        assert!(
            hidden.iter().any(|&v| v != 0.0),
            "hidden state should not be all zeros"
        );
        // Hidden state values should be in a reasonable range (not exploding).
        let max_abs = hidden.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs < 1000.0,
            "hidden state max abs should be <1000, got {max_abs:.1}"
        );
        assert!(
            max_abs > 0.01,
            "hidden state max abs suspiciously small: {max_abs:.6}"
        );

        let (logits, hidden2) = engine
            .decode_step_with_hidden(9419)
            .expect("decode_step_with_hidden failed");
        assert_eq!(hidden2.len(), expected_len);
        assert!(!logits.is_empty(), "logits should not be empty");
    }
}
