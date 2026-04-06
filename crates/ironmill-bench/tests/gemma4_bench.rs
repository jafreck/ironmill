//! Gemma 4 E2B Metal GPU benchmark tests.
//!
//! Individual tests for each Metal configuration so you can run exactly
//! the validation you need without loading weights multiple times:
//!
//! ```bash
//! # Quick smoke test — FP16 decode (use --release for ~10× faster load)
//! cargo test --release -p ironmill-bench --features metal -- gemma4::fp16_decode --ignored --nocapture
//!
//! # D2Quant 3-bit decode
//! cargo test --release -p ironmill-bench --features metal -- gemma4::d2quant3_decode --ignored --nocapture
//!
//! # All Gemma 4 benchmarks
//! cargo test --release -p ironmill-bench --features metal -- gemma4 --ignored --nocapture
//! ```
//!
//! Requires: Metal GPU, Gemma 4 E2B-IT weights in HuggingFace cache.

#[cfg(feature = "metal")]
mod gemma4 {
    use std::path::PathBuf;
    use std::time::Instant;

    use ironmill_compile::gpu::GpuCompileBuilder;
    use ironmill_compile::weights::{SafeTensorsProvider, WeightProvider};
    use ironmill_inference::engine::InferenceEngine;
    use ironmill_inference::metal::{MetalConfig, MetalInference};

    /// Locate the Gemma 4 E2B-IT model directory from the HuggingFace cache.
    fn gemma4_model_dir() -> PathBuf {
        let home = std::env::var("HOME").expect("HOME env var not set");
        let hf_cache =
            PathBuf::from(home).join(".cache/huggingface/hub/models--google--gemma-4-e2b-it");
        let snapshots = hf_cache.join("snapshots");
        if !snapshots.exists() {
            panic!(
                "Gemma 4 E2B-IT not found. Download with:\n  \
                 huggingface-cli download google/gemma-4-e2b-it"
            );
        }
        // Use the first (usually only) snapshot directory.
        let entry = std::fs::read_dir(&snapshots)
            .expect("read snapshots dir")
            .filter_map(|e| e.ok())
            .find(|e| e.file_type().map_or(false, |ft| ft.is_dir()))
            .unwrap_or_else(|| {
                panic!("no snapshot directory in {}", snapshots.display());
            });
        entry.path()
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
        let prompt: Vec<u32> = vec![2, 651, 3488]; // "Hello world" approx for Gemma tokenizer
        engine.prefill(&prompt).expect("prefill failed");

        // Warmup
        for _ in 0..3 {
            let _ = engine.decode_step(3488);
        }
        engine.reset();
        engine.prefill(&prompt).expect("prefill failed");

        // Timed decode
        let mut latencies = Vec::with_capacity(decode_tokens);
        let mut last_token = 3488u32;
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

    // ── Tests ────────────────────────────────────────────────────────

    #[test]
    #[ignore] // requires Metal GPU + Gemma 4 E2B-IT weights
    fn fp16_decode_throughput() {
        let model_dir = gemma4_model_dir();
        let config = MetalConfig::default().without_turboquant();
        let decode_tokens = 20;

        let provider =
            SafeTensorsProvider::load(&model_dir).expect("SafeTensorsProvider::load failed");
        let (mut engine, gpu_mb, load_ms) = load_gpu_engine(&provider, config);
        let (ms_tok, tok_s) = bench_decode(&mut engine, decode_tokens);

        println!("\n╔══════════════════════════════════════════════════╗");
        println!("║   Gemma 4 E2B — FP16 Decode                      ║");
        println!("╠══════════════════════════════════════════════════╣");
        println!("║  Load:     {load_ms:>8.0} ms                          ║");
        println!("║  GPU:      {gpu_mb:>8.1} MB                          ║");
        println!("║  Decode:   {ms_tok:>8.2} ms/tok                      ║");
        println!("║  Speed:    {tok_s:>8.1} tok/s                        ║");
        println!("╚══════════════════════════════════════════════════╝\n");

        assert!(tok_s > 0.0, "FP16 decode should produce >0 tok/s");
    }

    #[test]
    #[ignore] // requires Metal GPU + Gemma 4 E2B-IT weights
    fn fp16_tq_int4_decode_throughput() {
        let model_dir = gemma4_model_dir();
        let config = MetalConfig::default().with_turboquant(4);
        let decode_tokens = 20;

        let provider =
            SafeTensorsProvider::load(&model_dir).expect("SafeTensorsProvider::load failed");
        let (mut engine, gpu_mb, load_ms) = load_gpu_engine(&provider, config);
        let (ms_tok, tok_s) = bench_decode(&mut engine, decode_tokens);

        println!("\n╔══════════════════════════════════════════════════╗");
        println!("║   Gemma 4 E2B — FP16 + TurboQuant-INT4 KV        ║");
        println!("╠══════════════════════════════════════════════════╣");
        println!("║  Load:     {load_ms:>8.0} ms                          ║");
        println!("║  GPU:      {gpu_mb:>8.1} MB                          ║");
        println!("║  Decode:   {ms_tok:>8.2} ms/tok                      ║");
        println!("║  Speed:    {tok_s:>8.1} tok/s                        ║");
        println!("╚══════════════════════════════════════════════════╝\n");

        assert!(tok_s > 0.0, "FP16+TQ-INT4 decode should produce >0 tok/s");
    }

    #[test]
    #[ignore] // requires Metal GPU + Gemma 4 E2B-IT weights
    fn d2quant3_decode_throughput() {
        let model_dir = gemma4_model_dir();
        let config = MetalConfig::default().with_turboquant(4);
        let decode_tokens = 20;

        let provider = GpuCompileBuilder::new(model_dir.clone())
            .with_pass_pipeline(
                ironmill_compile::mil::PassPipeline::new()
                    .with_d2quant(3, 128, 0.99, None)
                    .expect("D2Quant config"),
            )
            .build()
            .expect("D2Quant compile failed");

        let (mut engine, gpu_mb, load_ms) = load_gpu_engine(&provider, config);
        let (ms_tok, tok_s) = bench_decode(&mut engine, decode_tokens);

        println!("\n╔══════════════════════════════════════════════════╗");
        println!("║   Gemma 4 E2B — D2Quant-3 + TQ-INT4 KV           ║");
        println!("╠══════════════════════════════════════════════════╣");
        println!("║  Load:     {load_ms:>8.0} ms                          ║");
        println!("║  GPU:      {gpu_mb:>8.1} MB                          ║");
        println!("║  Decode:   {ms_tok:>8.2} ms/tok                      ║");
        println!("║  Speed:    {tok_s:>8.1} tok/s                        ║");
        println!("╚══════════════════════════════════════════════════╝\n");

        assert!(tok_s > 0.0, "D2Quant-3 decode should produce >0 tok/s");
    }

    #[test]
    #[ignore] // requires Metal GPU + Gemma 4 E2B-IT weights
    fn d2quant3_prefill_functional() {
        let model_dir = gemma4_model_dir();
        let config = MetalConfig::default().with_turboquant(4);

        let provider = GpuCompileBuilder::new(model_dir.clone())
            .with_pass_pipeline(
                ironmill_compile::mil::PassPipeline::new()
                    .with_d2quant(3, 128, 0.99, None)
                    .expect("D2Quant config"),
            )
            .build()
            .expect("D2Quant compile failed");

        let (mut engine, gpu_mb, load_ms) = load_gpu_engine(&provider, config);

        // Longer prefill to exercise the batched matmul path.
        let prompt: Vec<u32> = vec![2, 651, 3488, 573, 1069, 3488, 573, 1069, 651, 3488];
        let t0 = Instant::now();
        engine.prefill(&prompt).expect("prefill failed");
        let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // One decode step to verify end-to-end.
        let logits = engine.decode_step(3488).expect("decode_step failed");
        assert!(
            !logits.is_empty(),
            "decode_step should return non-empty logits"
        );

        println!("\n╔══════════════════════════════════════════════════╗");
        println!("║   Gemma 4 E2B — D2Quant-3 Prefill Functional     ║");
        println!("╠══════════════════════════════════════════════════╣");
        println!("║  Load:     {load_ms:>8.0} ms                          ║");
        println!("║  GPU:      {gpu_mb:>8.1} MB                          ║");
        println!(
            "║  Prefill:  {prefill_ms:>8.1} ms ({} tokens)                ║",
            prompt.len()
        );
        println!(
            "║  Logits:   {}>0 ✓                                 ║",
            logits.len()
        );
        println!("╚══════════════════════════════════════════════════╝\n");
    }

    #[test]
    #[ignore] // requires Metal GPU + Gemma 4 E2B-IT weights
    fn fp16_vs_d2quant3_memory() {
        let model_dir = gemma4_model_dir();
        let config = MetalConfig::default().with_turboquant(4);

        // FP16 baseline
        let fp16_provider =
            SafeTensorsProvider::load(&model_dir).expect("SafeTensorsProvider::load failed");
        let fp16_config = config.clone().without_turboquant();
        let (_engine_fp16, fp16_gpu_mb, fp16_load_ms) =
            load_gpu_engine(&fp16_provider, fp16_config);

        // D2Quant 3-bit
        let d2q_provider = GpuCompileBuilder::new(model_dir.clone())
            .with_pass_pipeline(
                ironmill_compile::mil::PassPipeline::new()
                    .with_d2quant(3, 128, 0.99, None)
                    .expect("D2Quant config"),
            )
            .build()
            .expect("D2Quant compile failed");
        let (_engine_d2q, d2q_gpu_mb, d2q_load_ms) = load_gpu_engine(&d2q_provider, config);

        let reduction_pct = (1.0 - d2q_gpu_mb / fp16_gpu_mb) * 100.0;

        println!("\n╔══════════════════════════════════════════════════╗");
        println!("║   GPU Memory: FP16 vs D2Quant-3 (Gemma 4 E2B)    ║");
        println!("╠══════════════════════════════════════════════════╣");
        println!("║  FP16:       {fp16_gpu_mb:>8.1} MB  (load: {fp16_load_ms:.0}ms)      ║");
        println!("║  D2Quant-3:  {d2q_gpu_mb:>8.1} MB  (load: {d2q_load_ms:.0}ms)      ║");
        println!("║  Reduction:  {reduction_pct:>8.1}%                          ║");
        println!("╚══════════════════════════════════════════════════╝\n");

        assert!(
            reduction_pct > 20.0,
            "D2Quant-3 should reduce GPU memory by >20%, got {reduction_pct:.1}%"
        );
    }

    #[test]
    #[ignore] // requires Metal GPU + Gemma 4 E2B-IT weights
    fn profile_load_phases() {
        let model_dir = gemma4_model_dir();

        // Phase 1: SafeTensorsProvider::load (mmap + index)
        let t0 = Instant::now();
        let provider =
            SafeTensorsProvider::load(&model_dir).expect("SafeTensorsProvider::load failed");
        let t_provider = t0.elapsed();
        println!(
            "\nPhase 1 — SafeTensorsProvider::load: {:.1}ms",
            t_provider.as_secs_f64() * 1000.0
        );
        println!("  Tensor count: {}", provider.tensor_names().len());

        // Phase 2: MetalInference::new (device init)
        let config = MetalConfig::default().without_turboquant();
        let t1 = Instant::now();
        let mut engine = MetalInference::new(config.clone()).expect("MetalInference::new failed");
        let t_new = t1.elapsed();
        println!(
            "Phase 2 — MetalInference::new: {:.1}ms",
            t_new.as_secs_f64() * 1000.0
        );

        // Phase 3: load_weights (BF16 conv + block pack + GPU upload + shader compile)
        let t2 = Instant::now();
        engine
            .load_weights(&provider, config)
            .expect("load_weights failed");
        let t_weights = t2.elapsed();
        println!(
            "Phase 3 — load_weights: {:.1}ms",
            t_weights.as_secs_f64() * 1000.0
        );
        println!(
            "  GPU allocated: {:.1} MB",
            engine.gpu_allocated_bytes() as f64 / (1024.0 * 1024.0)
        );

        // Phase 4: first prefill (includes any lazy shader compilation)
        let prompt: Vec<u32> = vec![2, 651, 3488];
        let t3 = Instant::now();
        engine.prefill(&prompt).expect("prefill failed");
        let t_prefill = t3.elapsed();
        println!(
            "Phase 4 — first prefill ({} tokens): {:.1}ms",
            prompt.len(),
            t_prefill.as_secs_f64() * 1000.0
        );

        // Phase 5: individual decode steps
        let mut last_token = 3488u32;
        println!("Phase 5 — decode steps:");
        for i in 0..10 {
            let t = Instant::now();
            let logits = engine.decode_step(last_token).expect("decode failed");
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            println!("  step {i}: {ms:.2}ms ({:.1} tok/s)", 1000.0 / ms);
            last_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
        }

        println!(
            "\nTotal: {:.1}ms",
            (t_provider + t_new + t_weights).as_secs_f64() * 1000.0
        );
    }
}
