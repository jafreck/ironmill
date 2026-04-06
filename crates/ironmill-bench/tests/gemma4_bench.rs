//! Gemma 4 E2B Metal GPU benchmark.
//!
//! Single test that loads each config once and reports all metrics in one table.
//! **Always use `--release`** — debug mode is 27× slower.
//!
//! ```bash
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
        std::fs::read_dir(&snapshots)
            .expect("read snapshots dir")
            .filter_map(|e| e.ok())
            .find(|e| e.file_type().map_or(false, |ft| ft.is_dir()))
            .unwrap_or_else(|| panic!("no snapshot directory in {}", snapshots.display()))
            .path()
    }

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

    /// Prefill + decode N tokens. Returns (median_ms, tok_per_sec).
    fn bench_decode(engine: &mut MetalInference, decode_tokens: usize) -> (f64, f64) {
        let prompt: Vec<u32> = vec![2, 651, 3488];
        engine.prefill(&prompt).expect("prefill failed");

        // Warmup
        for _ in 0..3 {
            let _ = engine.decode_step(3488);
        }
        engine.reset();
        engine.prefill(&prompt).expect("prefill failed");

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

    struct BenchResult {
        label: &'static str,
        load_ms: f64,
        gpu_mb: f64,
        ms_tok: f64,
        tok_s: f64,
    }

    #[test]
    #[ignore] // requires Metal GPU + Gemma 4 E2B-IT weights
    fn benchmark() {
        let model_dir = gemma4_model_dir();
        let decode_tokens = 20;
        let mut results: Vec<BenchResult> = Vec::new();

        // ── FP16 baseline ────────────────────────────────────────
        let fp16_provider =
            SafeTensorsProvider::load(&model_dir).expect("SafeTensorsProvider::load failed");

        {
            let config = MetalConfig::default().without_turboquant();
            let (mut engine, gpu_mb, load_ms) = load_gpu_engine(&fp16_provider, config);
            let (ms_tok, tok_s) = bench_decode(&mut engine, decode_tokens);
            results.push(BenchResult {
                label: "FP16",
                load_ms,
                gpu_mb,
                ms_tok,
                tok_s,
            });
        }

        // ── FP16 + TurboQuant-INT4 KV ────────────────────────────
        {
            let config = MetalConfig::default().with_turboquant(4);
            let (mut engine, gpu_mb, load_ms) = load_gpu_engine(&fp16_provider, config);
            let (ms_tok, tok_s) = bench_decode(&mut engine, decode_tokens);
            results.push(BenchResult {
                label: "FP16 + TQ-INT4 KV",
                load_ms,
                gpu_mb,
                ms_tok,
                tok_s,
            });
        }

        drop(fp16_provider);

        // ── D2Quant 3-bit + TurboQuant-INT4 KV ──────────────────
        {
            let t_compile = Instant::now();
            let d2q_provider = GpuCompileBuilder::new(model_dir.clone())
                .with_pass_pipeline(
                    ironmill_compile::mil::PassPipeline::new()
                        .with_d2quant(3, 128, 0.99, None)
                        .expect("D2Quant config"),
                )
                .build()
                .expect("D2Quant compile failed");
            let compile_ms = t_compile.elapsed().as_secs_f64() * 1000.0;

            let config = MetalConfig::default().with_turboquant(4);
            let (mut engine, gpu_mb, load_ms) = load_gpu_engine(&d2q_provider, config);

            // Verify prefill works
            let prompt: Vec<u32> = vec![2, 651, 3488, 573, 1069, 3488, 573, 1069, 651, 3488];
            engine.prefill(&prompt).expect("D2Quant prefill failed");
            let logits = engine.decode_step(3488).expect("D2Quant decode failed");
            assert!(!logits.is_empty(), "D2Quant should return logits");
            engine.reset();

            let (ms_tok, tok_s) = bench_decode(&mut engine, decode_tokens);
            eprintln!("  D2Quant compile: {compile_ms:.0}ms");
            results.push(BenchResult {
                label: "D2Q-3 + TQ-INT4 KV",
                load_ms,
                gpu_mb,
                ms_tok,
                tok_s,
            });
        }

        // ── Print results table ──────────────────────────────────
        let fp16_gpu = results[0].gpu_mb;
        println!();
        println!("  Gemma 4 E2B-IT — Metal Benchmark Results");
        println!("  ─────────────────────────────────────────────────────────────");
        println!(
            "  {:<20} {:>8} {:>8} {:>8} {:>10}",
            "Config", "GPU MB", "ms/tok", "tok/s", "Load ms"
        );
        println!("  ─────────────────────────────────────────────────────────────");
        for r in &results {
            let mem_note = if r.gpu_mb < fp16_gpu * 0.95 {
                format!(" (−{:.0}%)", (1.0 - r.gpu_mb / fp16_gpu) * 100.0)
            } else {
                String::new()
            };
            println!(
                "  {:<20} {:>7.0}{:<7} {:>7.2} {:>7.1} {:>10.0}",
                r.label, r.gpu_mb, mem_note, r.ms_tok, r.tok_s, r.load_ms
            );
        }
        println!("  ─────────────────────────────────────────────────────────────");
        println!();

        // ── Assertions ───────────────────────────────────────────
        for r in &results {
            assert!(
                r.tok_s > 0.0,
                "{} should produce >0 tok/s, got {:.1}",
                r.label,
                r.tok_s
            );
        }
        let d2q = &results[2];
        let mem_reduction = (1.0 - d2q.gpu_mb / fp16_gpu) * 100.0;
        assert!(
            mem_reduction > 20.0,
            "D2Quant-3 should reduce GPU memory by >20%, got {mem_reduction:.1}%"
        );
    }
}
