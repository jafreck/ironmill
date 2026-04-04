//! GPTQ INT4 GPU benchmark tests for Qwen3-0.6B.
//!
//! Validates that GPTQ-4 weight quantization:
//! - Reduces GPU memory footprint vs FP16 baseline
//! - Maintains acceptable perplexity (within tolerance of FP16)
//! - Reports tok/s performance
//!
//! The test runs end-to-end: calibration → GPTQ quantization → inference.
//!
//! Requires: Metal GPU, Qwen3-0.6B SafeTensors fixture, wikitext2 dataset,
//!           `gptq` and `metal` features enabled.
//! Run: `cargo test -p ironmill-bench --features metal,gptq -- gptq --ignored --nocapture`

#[cfg(all(feature = "metal", feature = "gptq"))]
mod gptq_bench {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::time::Instant;

    use ironmill_compile::gpu::GpuCompileBuilder;
    use ironmill_compile::weights::{SafeTensorsProvider, WeightProvider};
    use ironmill_inference::calibration::{CalibrationDataset, CalibrationRunner};
    use ironmill_inference::engine::InferenceEngine;
    use ironmill_inference::metal::{MetalConfig, MetalInference};

    fn fixture_path(name: &str) -> PathBuf {
        let manifest = env!("CARGO_MANIFEST_DIR");
        PathBuf::from(manifest)
            .join("../../tests/fixtures")
            .join(name)
    }

    fn qwen_model_dir() -> PathBuf {
        fixture_path("Qwen3-0.6B")
    }

    fn skip_if_missing(path: &PathBuf, label: &str) {
        if !path.exists() {
            panic!(
                "Fixture {label} not found at {}. Run scripts/download-fixtures.sh first.",
                path.display()
            );
        }
    }

    /// Build a MetalInference engine from a WeightProvider and return (engine, gpu_mb, load_ms).
    fn load_gpu_engine(
        provider: &dyn WeightProvider,
        config: MetalConfig,
    ) -> (MetalInference, f64, f64) {
        let mut engine = MetalInference::new(config.clone()).expect("MetalInference::new failed");

        let t0 = Instant::now();
        let gpu_before = engine.gpu_allocated_bytes();
        engine
            .load_weights(provider, config)
            .expect("load_weights failed");
        let gpu_after = engine.gpu_allocated_bytes();
        let load_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let gpu_mb = (gpu_after - gpu_before) as f64 / (1024.0 * 1024.0);
        (engine, gpu_mb, load_ms)
    }

    /// Run decode benchmark: prefill with prompt, then decode N tokens.
    /// Returns (median_ms_per_tok, tok_per_sec).
    fn bench_decode(engine: &mut MetalInference, decode_tokens: usize) -> (f64, f64) {
        let prompt: Vec<u32> = vec![9707, 1879]; // "Hello world" in Qwen tokenizer
        engine.prefill(&prompt).expect("prefill failed");

        // Warmup.
        for _ in 0..5 {
            let _ = engine.decode_step(1879);
        }
        engine.reset();
        engine.prefill(&prompt).expect("prefill failed");

        // Timed decode.
        let mut latencies = Vec::with_capacity(decode_tokens);
        let mut last_token = 1879u32;
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

    /// Compute perplexity over a dataset of token sequences.
    fn eval_perplexity(
        engine: &mut MetalInference,
        sequences: &[Vec<u32>],
        max_seqs: usize,
    ) -> f64 {
        let mut total_ce = 0.0f64;
        let mut total_tokens = 0usize;

        for sequence in sequences.iter().take(max_seqs) {
            if sequence.len() < 2 {
                continue;
            }
            engine.reset();
            let all_logits = engine
                .prefill_all_logits(sequence)
                .expect("prefill_all_logits failed");

            for pos in 0..sequence.len() - 1 {
                let target = sequence[pos + 1] as usize;
                let logits = &all_logits[pos];

                let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum_exp: f64 = logits.iter().map(|&x| ((x - max_logit) as f64).exp()).sum();
                let log_softmax = (logits[target] - max_logit) as f64 - sum_exp.ln();
                total_ce -= log_softmax;
                total_tokens += 1;
            }
        }

        if total_tokens == 0 {
            return f64::INFINITY;
        }
        (total_ce / total_tokens as f64).exp()
    }

    /// Load the wikitext2 perplexity dataset (128-token sequences).
    fn load_dataset() -> Vec<Vec<u32>> {
        let path = fixture_path("quality/wikitext2-qwen3.json");
        skip_if_missing(&path, "wikitext2-qwen3.json");
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

    /// Build the key map from MIL const-op weight names to calibration store keys.
    ///
    /// The Metal calibration dispatch captures activations at two hook points per
    /// layer: `"attn_norm"` (RMSNorm output feeding Q/K/V/O projections) and
    /// `"ffn_norm"` (RMSNorm output feeding gate/up/down projections). The store
    /// keys follow the pattern `"layer_{i}_{name}"`.
    ///
    /// The MIL const-op names from SafeTensors import follow HuggingFace naming:
    /// `"model.layers.{i}.self_attn.{proj}.weight"` and
    /// `"model.layers.{i}.mlp.{proj}.weight"`.
    fn build_gptq_key_map(n_layers: usize) -> HashMap<String, String> {
        let mut map = HashMap::new();
        for i in 0..n_layers {
            for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
                map.insert(
                    format!("model.layers.{i}.self_attn.{proj}.weight"),
                    format!("layer_{i}_attn_norm"),
                );
            }
            for proj in &["gate_proj", "up_proj", "down_proj"] {
                map.insert(
                    format!("model.layers.{i}.mlp.{proj}.weight"),
                    format!("layer_{i}_ffn_norm"),
                );
            }
        }
        map
    }

    /// Calibrate the model and collect GPTQ Hessian data.
    ///
    /// Returns the Hessian data in pass format (keyed by MIL const-op names).
    fn calibrate_gptq(
        engine: &mut MetalInference,
        dataset: &[Vec<u32>],
        n_layers: usize,
        max_sequences: usize,
    ) -> HashMap<String, (Vec<f32>, usize, usize)> {
        let cal_dataset = CalibrationDataset {
            name: "wikitext2-calibration".into(),
            model: "Qwen/Qwen3-0.6B".into(),
            vocab_size: 151936,
            seq_len: dataset[0].len(),
            num_sequences: dataset.len(),
            eos_token_id: Some(151643),
            sequences: dataset.to_vec(),
        };

        let mut runner = CalibrationRunner::new();
        runner.max_sequences = Some(max_sequences);
        let store = runner
            .collect_gptq_stats(engine, &cal_dataset)
            .expect("GPTQ calibration failed");

        let key_map = build_gptq_key_map(n_layers);
        store.to_pass_format(&key_map)
    }

    /// Compile a GPTQ-INT4 model from SafeTensors with pre-computed Hessian data.
    fn compile_gptq(
        model_dir: &std::path::Path,
        hessian_data: HashMap<String, (Vec<f32>, usize, usize)>,
    ) -> ironmill_compile::weights::MilWeightProvider {
        use ironmill_compile::mil::PassPipeline;

        let pipeline = PassPipeline::new()
            .with_gptq(hessian_data, 4, 128, 128, 0.01)
            .expect("GPTQ pipeline config failed");

        GpuCompileBuilder::new(model_dir)
            .with_pass_pipeline(pipeline)
            .build()
            .expect("GPTQ compile failed")
    }

    // ── Tests ────────────────────────────────────────────────────────

    #[test]
    #[ignore] // requires Metal GPU + Qwen3-0.6B fixture + wikitext2 dataset
    fn gptq4_reduces_gpu_memory() {
        let model_dir = qwen_model_dir();
        skip_if_missing(&model_dir, "Qwen3-0.6B");

        let config = MetalConfig {
            enable_turboquant: false,
            ..MetalConfig::default()
        };
        let sequences = load_dataset();

        // FP16 baseline.
        let fp16_provider =
            SafeTensorsProvider::load(&model_dir).expect("SafeTensorsProvider::load failed");
        let n_layers = fp16_provider.config().num_hidden_layers;
        let (mut engine_fp16, fp16_gpu_mb, fp16_load_ms) =
            load_gpu_engine(&fp16_provider, config.clone());

        // Calibrate on the FP16 model.
        eprintln!("Calibrating GPTQ Hessians...");
        let t0 = Instant::now();
        let hessian_data = calibrate_gptq(&mut engine_fp16, &sequences, n_layers, 3);
        let cal_elapsed = t0.elapsed();
        eprintln!(
            "Calibration done: {} projections, {:.1}s",
            hessian_data.len(),
            cal_elapsed.as_secs_f64()
        );

        // Compile GPTQ-INT4.
        eprintln!("Compiling GPTQ-INT4...");
        let gptq_provider = compile_gptq(&model_dir, hessian_data);
        let (_, gptq_gpu_mb, gptq_load_ms) = load_gpu_engine(&gptq_provider, config);

        let reduction_pct = (1.0 - gptq_gpu_mb / fp16_gpu_mb) * 100.0;

        println!("\n╔══════════════════════════════════════════════════╗");
        println!("║       GPU Memory: FP16 vs GPTQ-INT4               ║");
        println!("╠══════════════════════════════════════════════════╣");
        println!("║  FP16:      {fp16_gpu_mb:>8.1} MB  (load: {fp16_load_ms:.0}ms)     ║");
        println!("║  GPTQ-INT4: {gptq_gpu_mb:>8.1} MB  (load: {gptq_load_ms:.0}ms)     ║");
        println!("║  Reduction: {reduction_pct:>8.1}%                          ║");
        println!("╚══════════════════════════════════════════════════╝\n");

        assert!(
            reduction_pct > 30.0,
            "GPTQ-INT4 should reduce GPU memory by >30%, got {reduction_pct:.1}%"
        );
    }

    #[test]
    #[ignore] // requires Metal GPU + Qwen3-0.6B fixture + wikitext2 dataset
    fn gptq4_maintains_perplexity() {
        let model_dir = qwen_model_dir();
        skip_if_missing(&model_dir, "Qwen3-0.6B");

        let config = MetalConfig {
            enable_turboquant: false,
            ..MetalConfig::default()
        };
        let sequences = load_dataset();
        let max_seqs = 5;

        // FP16 baseline perplexity.
        let fp16_provider =
            SafeTensorsProvider::load(&model_dir).expect("SafeTensorsProvider::load failed");
        let n_layers = fp16_provider.config().num_hidden_layers;
        let (mut engine_fp16, _, _) = load_gpu_engine(&fp16_provider, config.clone());
        let fp16_ppl = eval_perplexity(&mut engine_fp16, &sequences, max_seqs);

        // Calibrate on the FP16 model.
        eprintln!("Calibrating GPTQ Hessians...");
        let hessian_data = calibrate_gptq(&mut engine_fp16, &sequences, n_layers, 3);

        // GPTQ-INT4 perplexity.
        eprintln!("Compiling GPTQ-INT4...");
        let gptq_provider = compile_gptq(&model_dir, hessian_data);
        let (mut engine_gptq, _, _) = load_gpu_engine(&gptq_provider, config);
        let gptq_ppl = eval_perplexity(&mut engine_gptq, &sequences, max_seqs);

        let ppl_delta_pct = (gptq_ppl - fp16_ppl) / fp16_ppl * 100.0;

        println!("\n╔══════════════════════════════════════════════════╗");
        println!("║       Perplexity: FP16 vs GPTQ-INT4               ║");
        println!("╠══════════════════════════════════════════════════╣");
        println!("║  FP16 PPL:      {fp16_ppl:>10.2}                       ║");
        println!("║  GPTQ-INT4 PPL: {gptq_ppl:>10.2}                       ║");
        println!("║  ΔPPL:          {ppl_delta_pct:>+10.1}%                      ║");
        println!("╚══════════════════════════════════════════════════╝\n");

        // GPTQ should do better than naive RTN INT4. Allow up to 50% PPL increase
        // (should be much lower in practice — GPTQ typically <5% degradation).
        assert!(
            ppl_delta_pct < 50.0,
            "GPTQ-INT4 PPL degradation should be <50%, got {ppl_delta_pct:.1}%"
        );
    }

    #[test]
    #[ignore] // requires Metal GPU + Qwen3-0.6B fixture + wikitext2 dataset
    fn gptq4_full_comparison() {
        let model_dir = qwen_model_dir();
        skip_if_missing(&model_dir, "Qwen3-0.6B");

        let config = MetalConfig {
            enable_turboquant: false,
            ..MetalConfig::default()
        };
        let decode_tokens = 50;
        let sequences = load_dataset();
        let max_seqs = 3;

        // ── FP16 baseline ────────────────────────────────────────
        let fp16_provider =
            SafeTensorsProvider::load(&model_dir).expect("SafeTensorsProvider::load failed");
        let n_layers = fp16_provider.config().num_hidden_layers;
        let (mut engine_fp16, fp16_gpu_mb, _) = load_gpu_engine(&fp16_provider, config.clone());
        let (fp16_ms, fp16_tps) = bench_decode(&mut engine_fp16, decode_tokens);
        let fp16_ppl = eval_perplexity(&mut engine_fp16, &sequences, max_seqs);

        // ── Calibrate GPTQ Hessians ─────────────────────────────
        eprintln!("Calibrating GPTQ Hessians ({} sequences)...", max_seqs);
        let t0 = Instant::now();
        let hessian_data = calibrate_gptq(&mut engine_fp16, &sequences, n_layers, max_seqs);
        let cal_elapsed = t0.elapsed();
        eprintln!(
            "Calibration: {} projections, {:.1}s",
            hessian_data.len(),
            cal_elapsed.as_secs_f64()
        );

        // ── GPTQ-INT4 ──────────────────────────────────────────
        eprintln!("Compiling GPTQ-INT4...");
        let t0 = Instant::now();
        let gptq_provider = compile_gptq(&model_dir, hessian_data);
        let compile_elapsed = t0.elapsed();
        eprintln!("GPTQ compile: {:.1}s", compile_elapsed.as_secs_f64());

        let (mut engine_gptq, gptq_gpu_mb, _) = load_gpu_engine(&gptq_provider, config);
        let (gptq_ms, gptq_tps) = bench_decode(&mut engine_gptq, decode_tokens);
        let gptq_ppl = eval_perplexity(&mut engine_gptq, &sequences, max_seqs);

        let mem_delta = (1.0 - gptq_gpu_mb / fp16_gpu_mb) * 100.0;
        let ppl_delta = (gptq_ppl - fp16_ppl) / fp16_ppl * 100.0;

        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║       Qwen3-0.6B: FP16 vs GPTQ-INT4 Comparison               ║");
        println!("╠════════════╤═══════════╤═══════════╤═══════════╤══════════════╣");
        println!("║ Config     │  PPL      │  tok/s    │  ms/tok   │  GPU MB      ║");
        println!("╠════════════╪═══════════╪═══════════╪═══════════╪══════════════╣");
        println!(
            "║ FP16       │ {:>8.2} │ {:>8.1} │ {:>8.2} │ {:>11.0} ║",
            fp16_ppl, fp16_tps, fp16_ms, fp16_gpu_mb
        );
        println!(
            "║ GPTQ-INT4  │ {:>8.2} │ {:>8.1} │ {:>8.2} │ {:>11.0} ║",
            gptq_ppl, gptq_tps, gptq_ms, gptq_gpu_mb
        );
        println!("╠════════════╪═══════════╪═══════════╪═══════════╪══════════════╣");
        println!(
            "║ Δ          │ {:>+8.1}% │           │           │ {:>+10.1}% ║",
            ppl_delta, mem_delta
        );
        println!("╚════════════╧═══════════╧═══════════╧═══════════╧══════════════╝\n");
        println!(
            "  Calibration: {:.1}s | GPTQ compile: {:.1}s",
            cal_elapsed.as_secs_f64(),
            compile_elapsed.as_secs_f64()
        );

        // Assertions.
        assert!(
            ppl_delta < 50.0,
            "GPTQ-INT4 PPL degradation should be <50%, got {ppl_delta:.1}%"
        );
        assert!(gptq_tps > 0.0, "GPTQ-INT4 should decode >0 tok/s");
    }
}
