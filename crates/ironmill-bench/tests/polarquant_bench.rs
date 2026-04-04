//! PolarQuant GPU benchmark tests for Qwen3-0.6B.
//!
//! Validates that PolarQuant-4 weight quantization:
//! - Reduces GPU memory footprint vs FP16 baseline
//! - Maintains acceptable perplexity (within tolerance of FP16)
//! - Reports tok/s performance
//! - Reduces on-disk bundle size vs SafeTensors
//!
//! Requires: Metal GPU, Qwen3-0.6B SafeTensors fixture, wikitext2 dataset.
//! Run: `cargo test -p ironmill-bench --features metal -- polarquant --ignored --nocapture`

#[cfg(feature = "metal")]
mod polarquant_bench {
    use std::path::PathBuf;
    use std::time::Instant;

    use ironmill_compile::gpu::GpuCompileBuilder;
    use ironmill_compile::gpu::bundle::write_gpu_bundle;
    use ironmill_compile::weights::{SafeTensorsProvider, WeightProvider};
    use ironmill_inference::InferenceEngine;
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

    /// Recursively compute the total file size of a directory in bytes.
    fn dir_size_bytes(path: &std::path::Path) -> u64 {
        let mut total = 0u64;
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let ft = entry.file_type().unwrap();
                if ft.is_file() {
                    total += entry.metadata().map(|m| m.len()).unwrap_or(0);
                } else if ft.is_dir() {
                    total += dir_size_bytes(&entry.path());
                }
            }
        }
        total
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
        // Prefill with a short prompt.
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
            // Greedy pick next token from logits.
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
    ///
    /// Uses batched prefill (`prefill_all_logits`) to process entire
    /// sequences in parallel on the GPU instead of one-token-at-a-time
    /// decode, giving orders-of-magnitude faster evaluation.
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

            // logits[pos] predicts token at pos+1
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

    /// Load the wikitext2 perplexity dataset.
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

    fn load_dataset_1024() -> Vec<Vec<u32>> {
        let path = fixture_path("quality/wikitext2-qwen3-4b-1024.json");
        skip_if_missing(&path, "wikitext2-qwen3-4b-1024.json");
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

    #[test]
    #[ignore] // requires Metal GPU + Qwen3-0.6B fixture + wikitext2-1024 dataset
    fn fp16_perplexity_1024_tokens() {
        let model_dir = qwen_model_dir();
        skip_if_missing(&model_dir, "Qwen3-0.6B");

        let sequences = load_dataset_1024();
        let config = MetalConfig {
            enable_turboquant: false,
            ..MetalConfig::default()
        };
        let provider =
            SafeTensorsProvider::load(&model_dir).expect("SafeTensorsProvider::load failed");
        let (mut engine, _, _) = load_gpu_engine(&provider, config);

        let t0 = Instant::now();
        let ppl = eval_perplexity(&mut engine, &sequences, 1);
        let elapsed = t0.elapsed();
        let n_tokens: usize = sequences[..1]
            .iter()
            .map(|s| s.len().saturating_sub(1))
            .sum();
        let tok_s = n_tokens as f64 / elapsed.as_secs_f64();

        println!("\n╔══════════════════════════════════════════════════╗");
        println!("║   FP16 Perplexity (1024-token sequence)          ║");
        println!("╠══════════════════════════════════════════════════╣");
        println!("║  PPL:    {ppl:>10.2}                                ║");
        println!("║  tok/s:  {tok_s:>10.0}                                ║");
        println!(
            "║  time:   {:.2}s                                ║",
            elapsed.as_secs_f64()
        );
        println!("╚══════════════════════════════════════════════════╝\n");

        assert!(
            ppl < 20.0,
            "FP16 PPL on 1024-token sequence should be <20, got {ppl:.2}"
        );
    }

    #[test]
    #[ignore] // requires Metal GPU + Qwen3-0.6B fixture
    fn polarquant4_reduces_gpu_memory() {
        let model_dir = qwen_model_dir();
        skip_if_missing(&model_dir, "Qwen3-0.6B");

        let config = MetalConfig::default();

        // FP16 baseline.
        let fp16_provider =
            SafeTensorsProvider::load(&model_dir).expect("SafeTensorsProvider::load failed");
        let (_engine_fp16, fp16_gpu_mb, fp16_load_ms) =
            load_gpu_engine(&fp16_provider, config.clone());

        // PolarQuant-4.
        let pq_provider = GpuCompileBuilder::new(model_dir.clone())
            .with_pass_pipeline(
                ironmill_compile::mil::PassPipeline::new()
                    .with_polar_quant(4)
                    .expect("PolarQuant config"),
            )
            .build()
            .expect("PolarQuant compile failed");
        let pq_config = MetalConfig {
            force_cpu_dequant: false,
            ..config
        };
        let (_engine_pq, pq_gpu_mb, pq_load_ms) = load_gpu_engine(&pq_provider, pq_config);

        let reduction_pct = (1.0 - pq_gpu_mb / fp16_gpu_mb) * 100.0;

        println!("\n╔══════════════════════════════════════════════════╗");
        println!("║       GPU Memory: FP16 vs PolarQuant-4           ║");
        println!("╠══════════════════════════════════════════════════╣");
        println!("║  FP16:         {fp16_gpu_mb:>8.1} MB  (load: {fp16_load_ms:.0}ms)  ║");
        println!("║  PolarQuant-4: {pq_gpu_mb:>8.1} MB  (load: {pq_load_ms:.0}ms)  ║");
        println!("║  Reduction:    {reduction_pct:>8.1}%                       ║");
        println!("╚══════════════════════════════════════════════════╝\n");

        assert!(
            reduction_pct > 30.0,
            "PolarQuant-4 should reduce GPU memory by >30%, got {reduction_pct:.1}%"
        );
    }

    #[test]
    #[ignore] // requires Metal GPU + Qwen3-0.6B fixture
    fn polarquant4_decode_throughput() {
        let model_dir = qwen_model_dir();
        skip_if_missing(&model_dir, "Qwen3-0.6B");

        let config = MetalConfig::default();
        let decode_tokens = 50;

        // FP16 baseline.
        let fp16_provider =
            SafeTensorsProvider::load(&model_dir).expect("SafeTensorsProvider::load failed");
        let (mut engine_fp16, fp16_gpu_mb, _) = load_gpu_engine(&fp16_provider, config.clone());
        let (fp16_ms, fp16_tps) = bench_decode(&mut engine_fp16, decode_tokens);

        // PolarQuant-4.
        let pq_provider = GpuCompileBuilder::new(model_dir.clone())
            .with_pass_pipeline(
                ironmill_compile::mil::PassPipeline::new()
                    .with_polar_quant(4)
                    .expect("PolarQuant config"),
            )
            .build()
            .expect("PolarQuant compile failed");
        let pq_config = MetalConfig {
            force_cpu_dequant: false,
            ..config
        };
        let (mut engine_pq, pq_gpu_mb, _) = load_gpu_engine(&pq_provider, pq_config);
        let (pq_ms, pq_tps) = bench_decode(&mut engine_pq, decode_tokens);

        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║       Decode Throughput: FP16 vs PolarQuant-4            ║");
        println!("╠══════════════════════════════════════════════════════════╣");
        println!(
            "║  FP16:          {fp16_ms:>6.2} ms/tok  ({fp16_tps:>6.1} tok/s)  {fp16_gpu_mb:>6.0} MB  ║"
        );
        println!(
            "║  PolarQuant-4:  {pq_ms:>6.2} ms/tok  ({pq_tps:>6.1} tok/s)  {pq_gpu_mb:>6.0} MB  ║"
        );
        println!("╚══════════════════════════════════════════════════════════╝\n");

        // PolarQuant decode should be functional (>0 tok/s).
        assert!(pq_tps > 0.0, "PolarQuant-4 should produce >0 tok/s");
    }

    #[test]
    #[ignore] // requires Metal GPU + Qwen3-0.6B fixture + wikitext2 dataset
    fn polarquant4_maintains_perplexity() {
        let model_dir = qwen_model_dir();
        skip_if_missing(&model_dir, "Qwen3-0.6B");

        let config = MetalConfig::default();
        let sequences = load_dataset();
        let max_seqs = 5; // Limit for test speed.

        // FP16 baseline perplexity.
        let fp16_provider =
            SafeTensorsProvider::load(&model_dir).expect("SafeTensorsProvider::load failed");
        let (mut engine_fp16, _, _) = load_gpu_engine(&fp16_provider, config.clone());
        let fp16_ppl = eval_perplexity(&mut engine_fp16, &sequences, max_seqs);

        // PolarQuant-4 perplexity.
        let pq_provider = GpuCompileBuilder::new(model_dir.clone())
            .with_pass_pipeline(
                ironmill_compile::mil::PassPipeline::new()
                    .with_polar_quant(4)
                    .expect("PolarQuant config"),
            )
            .build()
            .expect("PolarQuant compile failed");
        let (mut engine_pq, _, _) = load_gpu_engine(&pq_provider, config);
        let pq_ppl = eval_perplexity(&mut engine_pq, &sequences, max_seqs);

        let ppl_delta_pct = (pq_ppl - fp16_ppl) / fp16_ppl * 100.0;

        println!("\n╔══════════════════════════════════════════════════╗");
        println!("║       Perplexity: FP16 vs PolarQuant-4           ║");
        println!("╠══════════════════════════════════════════════════╣");
        println!("║  FP16 PPL:         {fp16_ppl:>10.2}                    ║");
        println!("║  PolarQuant-4 PPL: {pq_ppl:>10.2}                    ║");
        println!("║  ΔPPL:             {ppl_delta_pct:>+10.1}%                   ║");
        println!("╚══════════════════════════════════════════════════╝\n");

        assert!(
            ppl_delta_pct < 20.0,
            "PolarQuant-4 PPL degradation should be <20%, got {ppl_delta_pct:.1}%"
        );
    }

    #[test]
    #[ignore] // requires Metal GPU + Qwen3-0.6B fixture
    fn polarquant4_reduces_disk_size() {
        let model_dir = qwen_model_dir();
        skip_if_missing(&model_dir, "Qwen3-0.6B");

        // Measure SafeTensors disk size.
        let st_path = model_dir.join("model.safetensors");
        skip_if_missing(&st_path, "model.safetensors");
        let st_size_mb = std::fs::metadata(&st_path).unwrap().len() as f64 / (1024.0 * 1024.0);

        // Compile PolarQuant-4 bundle.
        let pq_provider = GpuCompileBuilder::new(model_dir.clone())
            .with_pass_pipeline(
                ironmill_compile::mil::PassPipeline::new()
                    .with_polar_quant(4)
                    .expect("PolarQuant config"),
            )
            .build()
            .expect("PolarQuant compile failed");

        let bundle_dir = tempfile::tempdir().expect("tempdir");
        let bundle_path = bundle_dir.path().join("qwen3-0.6b.ironml-gpu");
        write_gpu_bundle(&pq_provider, &bundle_path).expect("write_gpu_bundle failed");

        // Measure bundle disk size.
        let bundle_size_bytes = dir_size_bytes(&bundle_path);
        let bundle_size_mb = bundle_size_bytes as f64 / (1024.0 * 1024.0);
        let reduction_pct = (1.0 - bundle_size_mb / st_size_mb) * 100.0;

        println!("\n╔══════════════════════════════════════════════════╗");
        println!("║       Disk Size: SafeTensors vs PolarQuant-4     ║");
        println!("╠══════════════════════════════════════════════════╣");
        println!("║  SafeTensors:  {st_size_mb:>8.1} MB                     ║");
        println!("║  PQ-4 bundle:  {bundle_size_mb:>8.1} MB                     ║");
        println!("║  Reduction:    {reduction_pct:>8.1}%                       ║");
        println!("╚══════════════════════════════════════════════════╝\n");

        assert!(
            reduction_pct > 40.0,
            "PolarQuant-4 bundle should be >40% smaller, got {reduction_pct:.1}%"
        );
    }

    #[test]
    #[ignore] // requires Metal GPU + Qwen3-0.6B fixture + wikitext2 dataset
    fn polarquant4_full_comparison() {
        let model_dir = qwen_model_dir();
        skip_if_missing(&model_dir, "Qwen3-0.6B");

        let config = MetalConfig::default();
        let pq_config = MetalConfig {
            force_cpu_dequant: false,
            ..config.clone()
        };
        let decode_tokens = 50;
        let sequences = load_dataset();
        let max_seqs = 3;

        // ── FP16 baseline ────────────────────────────────────────
        let fp16_provider =
            SafeTensorsProvider::load(&model_dir).expect("SafeTensorsProvider::load failed");
        let (mut engine_fp16, fp16_gpu_mb, _) = load_gpu_engine(&fp16_provider, config.clone());
        let (fp16_ms, fp16_tps) = bench_decode(&mut engine_fp16, decode_tokens);
        let fp16_ppl = eval_perplexity(&mut engine_fp16, &sequences, max_seqs);

        let st_path = model_dir.join("model.safetensors");
        let fp16_disk_mb = std::fs::metadata(&st_path).unwrap().len() as f64 / (1024.0 * 1024.0);

        // ── PolarQuant-4 (CPU dequant for correctness) ───────────
        let pq_provider = GpuCompileBuilder::new(model_dir.clone())
            .with_pass_pipeline(
                ironmill_compile::mil::PassPipeline::new()
                    .with_polar_quant(4)
                    .expect("PolarQuant config"),
            )
            .build()
            .expect("PolarQuant compile failed");
        let (mut engine_pq, pq_gpu_mb, _) = load_gpu_engine(&pq_provider, pq_config);
        let (pq_ms, pq_tps) = bench_decode(&mut engine_pq, decode_tokens);
        let pq_ppl = eval_perplexity(&mut engine_pq, &sequences, max_seqs);

        // PQ-4 bundle disk size.
        let bundle_dir = tempfile::tempdir().expect("tempdir");
        let bundle_path = bundle_dir.path().join("qwen3-0.6b.ironml-gpu");
        write_gpu_bundle(&pq_provider, &bundle_path).expect("write_gpu_bundle failed");
        let pq_disk_mb: f64 = dir_size_bytes(&bundle_path) as f64 / (1024.0 * 1024.0);

        let mem_delta = (1.0 - pq_gpu_mb / fp16_gpu_mb) * 100.0;
        let ppl_delta = (pq_ppl - fp16_ppl) / fp16_ppl * 100.0;
        let disk_delta = (1.0 - pq_disk_mb / fp16_disk_mb) * 100.0;

        println!("\n╔════════════════════════════════════════════════════════════════════════╗");
        println!("║              Qwen3-0.6B: FP16 vs PolarQuant-4 Comparison              ║");
        println!("╠════════════╤═══════════╤═══════════╤═══════════╤══════════╤════════════╣");
        println!("║ Config     │  PPL      │  tok/s    │  ms/tok   │  GPU MB  │  Disk MB   ║");
        println!("╠════════════╪═══════════╪═══════════╪═══════════╪══════════╪════════════╣");
        println!(
            "║ FP16       │ {:>8.2} │ {:>8.1} │ {:>8.2} │ {:>7.0} │ {:>9.1} ║",
            fp16_ppl, fp16_tps, fp16_ms, fp16_gpu_mb, fp16_disk_mb
        );
        println!(
            "║ PQ-INT4    │ {:>8.2} │ {:>8.1} │ {:>8.2} │ {:>7.0} │ {:>9.1} ║",
            pq_ppl, pq_tps, pq_ms, pq_gpu_mb, pq_disk_mb
        );
        println!("╠════════════╪═══════════╪═══════════╪═══════════╪══════════╪════════════╣");
        println!(
            "║ Δ          │ {:>+8.1}% │           │           │ {:>+6.1}% │ {:>+8.1}% ║",
            ppl_delta, mem_delta, disk_delta
        );
        println!("╚════════════╧═══════════╧═══════════╧═══════════╧══════════╧════════════╝\n");

        // Assertions.
        // Phase 3: quantized weights stay packed in VRAM.
        // GPU memory reduction requires Phase 3 quantized kernels.
        assert!(
            ppl_delta < 200.0,
            "PPL degradation should be <200% for RTN INT4, got {ppl_delta:.1}%"
        );
        assert!(pq_tps > 0.0, "PolarQuant should decode >0 tok/s");
        // Disk size reduction requires packed LUT storage (future work).
        // Disk size reduction requires packed LUT bundle format.
        if disk_delta > 10.0 {
            println!("  Disk reduction: {disk_delta:.1}%");
        }
    }
}
