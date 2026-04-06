//! Gemma 4 E2B / E4B Metal GPU benchmark.
//!
//! Single test per model size that loads each config once, measures decode
//! throughput, GPU memory, and perplexity, reporting all metrics in one table.
//! **Always use `--release`** — debug mode is 27× slower.
//!
//! ```bash
//! cargo test --release -p ironmill-bench --features metal -- gemma4 --ignored --nocapture
//! ```
//!
//! Requires: Metal GPU, Gemma 4 weights in HuggingFace cache,
//! OASST1 instruct dataset at tests/fixtures/quality/oasst1-gemma4-instruct.json.

#[cfg(feature = "metal")]
mod gemma4 {
    use std::path::PathBuf;
    use std::time::Instant;

    use ironmill_compile::gpu::GpuCompileBuilder;
    use ironmill_compile::weights::quantized::{D2QuantConfig, QuantizedWeightProvider};
    use ironmill_compile::weights::{SafeTensorsProvider, WeightProvider};
    use ironmill_inference::engine::InferenceEngine;
    use ironmill_inference::metal::{MetalConfig, MetalInference};

    fn hf_model_dir(model_id: &str) -> PathBuf {
        let home = std::env::var("HOME").expect("HOME env var not set");
        let hf_name = model_id.replace('/', "--");
        let hf_cache = PathBuf::from(home)
            .join(".cache/huggingface/hub")
            .join(format!("models--{hf_name}"));
        let snapshots = hf_cache.join("snapshots");
        if !snapshots.exists() {
            panic!(
                "{model_id} not found. Download with:\n  \
                 huggingface-cli download {model_id}"
            );
        }
        std::fs::read_dir(&snapshots)
            .expect("read snapshots dir")
            .filter_map(|e| e.ok())
            .find(|e| e.file_type().map_or(false, |ft| ft.is_dir()))
            .unwrap_or_else(|| panic!("no snapshot directory in {}", snapshots.display()))
            .path()
    }

    fn gemma4_model_dir() -> PathBuf {
        hf_model_dir("google/gemma-4-e2b-it")
    }

    fn fixture_path(name: &str) -> PathBuf {
        let manifest = env!("CARGO_MANIFEST_DIR");
        PathBuf::from(manifest)
            .join("../../tests/fixtures")
            .join(name)
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

    // ── Quality metrics helpers ──────────────────────────────────

    fn softmax_f64(logits: &[f32]) -> Vec<f64> {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let exps: Vec<f64> = logits.iter().map(|&x| (x as f64 - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
        p.iter()
            .zip(q.iter())
            .map(|(&pi, &qi)| {
                if pi > 1e-30 {
                    pi * (pi / (qi + 1e-30)).ln()
                } else {
                    0.0
                }
            })
            .sum()
    }

    fn top_k_agreement(a: &[f32], b: &[f32], k: usize) -> f64 {
        let mut a_idx: Vec<usize> = (0..a.len()).collect();
        let mut b_idx: Vec<usize> = (0..b.len()).collect();
        a_idx.sort_by(|&i, &j| a[j].partial_cmp(&a[i]).unwrap_or(std::cmp::Ordering::Equal));
        b_idx.sort_by(|&i, &j| b[j].partial_cmp(&b[i]).unwrap_or(std::cmp::Ordering::Equal));
        let a_set: std::collections::HashSet<usize> = a_idx[..k].iter().copied().collect();
        let b_set: std::collections::HashSet<usize> = b_idx[..k].iter().copied().collect();
        a_set.intersection(&b_set).count() as f64 / k as f64
    }

    #[allow(dead_code)]
    struct QualityMetrics {
        label: &'static str,
        mean_kl: f64,
        median_kl: f64,
        p95_kl: f64,
        max_kl: f64,
        top1_agree: f64,
        top5_agree: f64,
        top10_agree: f64,
        ppl_ref: f64,
        ppl_test: f64,
        n_positions: usize,
    }

    fn compute_metrics(
        label: &'static str,
        ref_logits: &[Vec<Vec<f32>>],
        test_logits: &[Vec<Vec<f32>>],
        sequences: &[Vec<u32>],
        eval_seqs: usize,
    ) -> QualityMetrics {
        let mut kl_values = Vec::new();
        let mut top1_matches = 0usize;
        let mut top5_total = 0.0f64;
        let mut top10_total = 0.0f64;
        let mut ref_ce = 0.0f64;
        let mut test_ce = 0.0f64;
        let mut n = 0usize;

        for (si, seq) in sequences.iter().take(eval_seqs).enumerate() {
            let ref_lg = &ref_logits[si];
            let test_lg = &test_logits[si];
            for pos in 0..seq.len() - 1 {
                let target = seq[pos + 1] as usize;
                let p = softmax_f64(&ref_lg[pos]);
                let q = softmax_f64(&test_lg[pos]);
                kl_values.push(kl_divergence(&p, &q));
                ref_ce -= p[target].max(1e-30).ln();
                test_ce -= q[target].max(1e-30).ln();
                let r = &ref_lg[pos];
                let t = &test_lg[pos];
                top1_matches += top_k_agreement(r, t, 1) as usize;
                top5_total += top_k_agreement(r, t, 5);
                top10_total += top_k_agreement(r, t, 10);
                n += 1;
            }
        }

        kl_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let nan_count = kl_values.iter().filter(|v| v.is_nan()).count();
        let finite_kl: Vec<f64> = kl_values
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        let mean_kl = if finite_kl.is_empty() {
            f64::NAN
        } else {
            finite_kl.iter().sum::<f64>() / finite_kl.len() as f64
        };
        let median_kl = if finite_kl.is_empty() {
            f64::NAN
        } else {
            finite_kl[finite_kl.len() / 2]
        };
        let p95_kl = if finite_kl.is_empty() {
            f64::NAN
        } else {
            finite_kl[(finite_kl.len() as f64 * 0.95) as usize]
        };
        let max_kl = finite_kl.last().copied().unwrap_or(f64::NAN);
        if nan_count > 0 {
            eprintln!(
                "    ⚠ {label}: {nan_count}/{} positions have NaN KL",
                kl_values.len()
            );
        }

        QualityMetrics {
            label,
            mean_kl,
            median_kl,
            p95_kl,
            max_kl,
            top1_agree: top1_matches as f64 / n as f64 * 100.0,
            top5_agree: top5_total / n as f64 * 100.0,
            top10_agree: top10_total / n as f64 * 100.0,
            ppl_ref: (ref_ce / n as f64).exp(),
            ppl_test: (test_ce / n as f64).exp(),
            n_positions: n,
        }
    }

    /// Evaluate perplexity using batched prefill (prefill_all_logits).
    /// Returns (ppl, num_tokens, eval_tok_per_sec).
    fn eval_perplexity(
        engine: &mut MetalInference,
        sequences: &[Vec<u32>],
        max_seqs: usize,
    ) -> (f64, usize, f64) {
        let mut total_ce = 0.0f64;
        let mut total_tokens = 0usize;
        let start = Instant::now();

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

        let elapsed = start.elapsed().as_secs_f64();
        let ppl = if total_tokens > 0 {
            (total_ce / total_tokens as f64).exp()
        } else {
            f64::INFINITY
        };
        let tok_s = total_tokens as f64 / elapsed;
        (ppl, total_tokens, tok_s)
    }

    /// Load the OASST1 instruct-formatted dataset for Gemma 4.
    /// Contiguous conversation threads formatted with Gemma's chat template,
    /// split into 512-token sequences (~20K tokens total).
    fn load_eval_dataset() -> Vec<Vec<u32>> {
        let path = fixture_path("quality/oasst1-gemma4-instruct.json");
        if !path.exists() {
            panic!(
                "OASST1 dataset not found at {}.\n\
                 Run: python /tmp/prepare_oasst_dataset.py",
                path.display()
            );
        }
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

    #[allow(dead_code)]
    struct BenchResult {
        label: &'static str,
        load_ms: f64,
        gpu_mb: f64,
        ms_tok: f64,
        tok_s: f64,
        ppl: f64,
        ppl_tokens: usize,
    }

    #[test]
    #[ignore] // requires Metal GPU + Gemma 4 E2B-IT weights + Alpaca dataset
    fn benchmark() {
        let model_dir = gemma4_model_dir();
        let decode_tokens = 20;
        let sequences = load_eval_dataset();
        let ppl_seqs = sequences.len(); // use all sequences (~39)
        let mut results: Vec<BenchResult> = Vec::new();

        eprintln!("  Alpaca dataset: {} sequences loaded", sequences.len());

        // ── FP16 baseline (FP16 KV cache) ────────────────────────
        let fp16_provider =
            SafeTensorsProvider::load(&model_dir).expect("SafeTensorsProvider::load failed");

        {
            let config = MetalConfig::default().without_turboquant();
            let (mut engine, gpu_mb, load_ms) = load_gpu_engine(&fp16_provider, config);
            let (ms_tok, tok_s) = bench_decode(&mut engine, decode_tokens);
            let (ppl, ppl_tokens, ppl_tps) = eval_perplexity(&mut engine, &sequences, ppl_seqs);
            eprintln!("  FP16 PPL: {ppl:.2} ({ppl_tokens} tokens, {ppl_tps:.0} tok/s)");
            results.push(BenchResult {
                label: "FP16",
                load_ms,
                gpu_mb,
                ms_tok,
                tok_s,
                ppl,
                ppl_tokens,
            });
        }

        // ── FP16 + TurboQuant-INT4 KV ────────────────────────────
        {
            let config = MetalConfig::default().with_turboquant(4);
            let (mut engine, gpu_mb, load_ms) = load_gpu_engine(&fp16_provider, config);
            let (ms_tok, tok_s) = bench_decode(&mut engine, decode_tokens);
            let (ppl, ppl_tokens, ppl_tps) = eval_perplexity(&mut engine, &sequences, ppl_seqs);
            eprintln!("  FP16+TQ-INT4 PPL: {ppl:.2} ({ppl_tokens} tokens, {ppl_tps:.0} tok/s)");
            results.push(BenchResult {
                label: "FP16 + TQ-INT4 KV",
                load_ms,
                gpu_mb,
                ms_tok,
                tok_s,
                ppl,
                ppl_tokens,
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
            let (ppl, ppl_tokens, ppl_tps) = eval_perplexity(&mut engine, &sequences, ppl_seqs);
            eprintln!("  D2Q-3 PPL: {ppl:.2} ({ppl_tokens} tokens, {ppl_tps:.0} tok/s)");
            eprintln!("  D2Quant compile: {compile_ms:.0}ms");
            results.push(BenchResult {
                label: "D2Q-3 + TQ-INT4 KV",
                load_ms,
                gpu_mb,
                ms_tok,
                tok_s,
                ppl,
                ppl_tokens,
            });
        }

        // ── Print results table ──────────────────────────────────
        let fp16_gpu = results[0].gpu_mb;
        let fp16_ppl = results[0].ppl;
        println!();
        println!(
            "  Gemma 4 E2B-IT — Metal Benchmark (OASST1 instruct, {ppl_seqs} seqs, {} tokens)",
            ppl_seqs * 512
        );
        println!("  ──────────────────────────────────────────────────────────────────────────");
        println!(
            "  {:<20} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
            "Config", "GPU MB", "ms/tok", "tok/s", "PPL", "ΔPPL", "Load ms"
        );
        println!("  ──────────────────────────────────────────────────────────────────────────");
        for r in &results {
            let mem_note = if r.gpu_mb < fp16_gpu * 0.95 {
                format!(" (−{:.0}%)", (1.0 - r.gpu_mb / fp16_gpu) * 100.0)
            } else {
                String::new()
            };
            let ppl_delta = if (r.ppl - fp16_ppl).abs() > 0.01 {
                format!("{:>+.1}%", (r.ppl - fp16_ppl) / fp16_ppl * 100.0)
            } else {
                "—".to_string()
            };
            println!(
                "  {:<20} {:>7.0}{:<7} {:>7.2} {:>7.1} {:>7.2} {:>9} {:>10.0}",
                r.label, r.gpu_mb, mem_note, r.ms_tok, r.tok_s, r.ppl, ppl_delta, r.load_ms
            );
        }
        println!("  ──────────────────────────────────────────────────────────────────────────");
        println!();

        // ── Assertions ───────────────────────────────────────────
        for r in &results {
            assert!(
                r.tok_s > 0.0,
                "{} should produce >0 tok/s, got {:.1}",
                r.label,
                r.tok_s
            );
            assert!(
                r.ppl.is_finite() && r.ppl > 0.0,
                "{} PPL should be finite and positive, got {:.2}",
                r.label,
                r.ppl
            );
        }
        let d2q = &results[2];
        let mem_reduction = (1.0 - d2q.gpu_mb / fp16_gpu) * 100.0;
        assert!(
            mem_reduction > 20.0,
            "D2Quant-3 should reduce GPU memory by >20%, got {mem_reduction:.1}%"
        );
    }

    /// Compare per-token logit distributions between FP16 and quantized
    /// configs using KL divergence, top-k agreement, and CE analysis.
    #[test]
    #[ignore]
    fn investigate_tq_ppl() {
        let model_dir = gemma4_model_dir();
        let sequences = load_eval_dataset();
        // Use multiple sequences for more stable statistics
        let eval_seqs = 5;

        let provider =
            SafeTensorsProvider::load(&model_dir).expect("SafeTensorsProvider::load failed");

        // FP16 baseline logits (all sequences)
        let fp16_config = MetalConfig::default().without_turboquant();
        let (mut fp16_engine, _, _) = load_gpu_engine(&provider, fp16_config);
        let mut fp16_all: Vec<Vec<Vec<f32>>> = Vec::new();
        for seq in sequences.iter().take(eval_seqs) {
            fp16_engine.reset();
            fp16_all.push(fp16_engine.prefill_all_logits(seq).expect("FP16 prefill"));
        }

        // FP16 + TQ-INT8 logits
        let tq8_config = MetalConfig::default().with_turboquant(8);
        let (mut tq8_engine, _, _) = load_gpu_engine(&provider, tq8_config);
        let mut tq8_all: Vec<Vec<Vec<f32>>> = Vec::new();
        for seq in sequences.iter().take(eval_seqs) {
            tq8_engine.reset();
            tq8_all.push(tq8_engine.prefill_all_logits(seq).expect("TQ8 prefill"));
        }

        // FP16 + TQ-INT4 logits
        let tq_config = MetalConfig::default().with_turboquant(4);
        let (mut tq_engine, _, _) = load_gpu_engine(&provider, tq_config);
        let mut tq_all: Vec<Vec<Vec<f32>>> = Vec::new();
        for seq in sequences.iter().take(eval_seqs) {
            tq_engine.reset();
            tq_all.push(tq_engine.prefill_all_logits(seq).expect("TQ prefill"));
        }

        drop(provider);

        // D2Quant-3 + TQ-INT4 logits
        let d2q_provider = GpuCompileBuilder::new(model_dir.clone())
            .with_pass_pipeline(
                ironmill_compile::mil::PassPipeline::new()
                    .with_d2quant(3, 128, 0.99, None)
                    .expect("D2Quant config"),
            )
            .build()
            .expect("D2Quant compile failed");

        let d2q4_config = MetalConfig::default().with_turboquant(4);
        let (mut d2q4_engine, _, _) = load_gpu_engine(&d2q_provider, d2q4_config);
        let mut d2q4_all: Vec<Vec<Vec<f32>>> = Vec::new();
        for seq in sequences.iter().take(eval_seqs) {
            d2q4_engine.reset();
            d2q4_all.push(
                d2q4_engine
                    .prefill_all_logits(seq)
                    .expect("D2Q+TQ4 prefill"),
            );
        }
        drop(d2q4_engine);

        // D2Quant-3 + TQ-INT8 logits (reuse same provider)
        let d2q8_config = MetalConfig::default().with_turboquant(8);
        let (mut d2q8_engine, _, _) = load_gpu_engine(&d2q_provider, d2q8_config);
        let mut d2q8_all: Vec<Vec<Vec<f32>>> = Vec::new();
        for seq in sequences.iter().take(eval_seqs) {
            d2q8_engine.reset();
            d2q8_all.push(
                d2q8_engine
                    .prefill_all_logits(seq)
                    .expect("D2Q+TQ8 prefill"),
            );
        }

        let tq8_metrics =
            compute_metrics("FP16+TQ-INT8", &fp16_all, &tq8_all, &sequences, eval_seqs);
        let tq_metrics = compute_metrics("FP16+TQ-INT4", &fp16_all, &tq_all, &sequences, eval_seqs);
        let d2q8_metrics =
            compute_metrics("D2Q-3+TQ-INT8", &fp16_all, &d2q8_all, &sequences, eval_seqs);
        let d2q_metrics =
            compute_metrics("D2Q-3+TQ-INT4", &fp16_all, &d2q4_all, &sequences, eval_seqs);

        println!(
            "\n  Quantization Quality vs FP16 Baseline ({} positions)",
            tq_metrics.n_positions
        );
        println!("  ═══════════════════════════════════════════════════════════════════════════");
        println!(
            "  {:<18} {:>8} {:>8} {:>8} {:>8} {:>7} {:>7} {:>7} {:>8} {:>8}",
            "Config",
            "mean KL",
            "med KL",
            "p95 KL",
            "max KL",
            "top1%",
            "top5%",
            "top10%",
            "FP16 PPL",
            "test PPL"
        );
        println!("  ───────────────────────────────────────────────────────────────────────────");
        for m in [&tq8_metrics, &tq_metrics, &d2q8_metrics, &d2q_metrics] {
            println!(
                "  {:<18} {:>8.4} {:>8.4} {:>8.3} {:>8.2} {:>6.1}% {:>6.1}% {:>6.1}% {:>8.2} {:>8.2}",
                m.label,
                m.mean_kl,
                m.median_kl,
                m.p95_kl,
                m.max_kl,
                m.top1_agree,
                m.top5_agree,
                m.top10_agree,
                m.ppl_ref,
                m.ppl_test
            );
        }
        println!("  ═══════════════════════════════════════════════════════════════════════════");
        println!();
        println!("  KL divergence guide: ~0.01 = negligible, ~0.1 = noticeable, ~1.0 = severe");
    }

    /// KL divergence test for Gemma 4 E4B — larger model, same head_dim=256.
    /// Tests whether D2Quant-3 quality improves with a bigger model.
    #[test]
    #[ignore]
    fn investigate_e4b_quality() {
        let model_dir = hf_model_dir("google/gemma-4-E4B-it");
        let sequences = load_eval_dataset();
        let eval_seqs = 3; // fewer seqs — E4B is much larger

        let provider =
            SafeTensorsProvider::load(&model_dir).expect("SafeTensorsProvider::load failed");

        // FP16 baseline
        let fp16_config = MetalConfig::default().without_turboquant();
        let (mut fp16_engine, _, _) = load_gpu_engine(&provider, fp16_config);
        let mut fp16_all: Vec<Vec<Vec<f32>>> = Vec::new();
        for seq in sequences.iter().take(eval_seqs) {
            fp16_engine.reset();
            fp16_all.push(fp16_engine.prefill_all_logits(seq).expect("FP16 prefill"));
        }

        // FP16 + TQ-INT8
        let tq8_config = MetalConfig::default().with_turboquant(8);
        let (mut tq8_engine, _, _) = load_gpu_engine(&provider, tq8_config);
        let mut tq8_all: Vec<Vec<Vec<f32>>> = Vec::new();
        for seq in sequences.iter().take(eval_seqs) {
            tq8_engine.reset();
            tq8_all.push(tq8_engine.prefill_all_logits(seq).expect("TQ8 prefill"));
        }

        drop(provider);
        drop(tq8_engine);

        // D2Quant-3 + TQ-INT8
        eprintln!("  Compiling D2Quant-3 for E4B...");
        let d2q_provider = GpuCompileBuilder::new(model_dir.clone())
            .with_pass_pipeline(
                ironmill_compile::mil::PassPipeline::new()
                    .with_d2quant(3, 128, 0.99, None)
                    .expect("D2Quant config"),
            )
            .build()
            .expect("D2Quant compile failed");
        let d2q8_config = MetalConfig::default().with_turboquant(8);
        let (mut d2q8_engine, _, _) = load_gpu_engine(&d2q_provider, d2q8_config);
        let mut d2q8_all: Vec<Vec<Vec<f32>>> = Vec::new();
        for seq in sequences.iter().take(eval_seqs) {
            d2q8_engine.reset();
            d2q8_all.push(
                d2q8_engine
                    .prefill_all_logits(seq)
                    .expect("D2Q+TQ8 prefill"),
            );
        }

        let tq8_metrics =
            compute_metrics("FP16+TQ-INT8", &fp16_all, &tq8_all, &sequences, eval_seqs);
        let d2q8_metrics =
            compute_metrics("D2Q-3+TQ-INT8", &fp16_all, &d2q8_all, &sequences, eval_seqs);

        println!(
            "\n  Gemma 4 E4B-IT — Quantization Quality ({} positions)",
            tq8_metrics.n_positions
        );
        println!("  ═══════════════════════════════════════════════════════════════════════════");
        println!(
            "  {:<18} {:>8} {:>8} {:>8} {:>8} {:>7} {:>7} {:>7} {:>8} {:>8}",
            "Config",
            "mean KL",
            "med KL",
            "p95 KL",
            "max KL",
            "top1%",
            "top5%",
            "top10%",
            "FP16 PPL",
            "test PPL"
        );
        println!("  ───────────────────────────────────────────────────────────────────────────");
        for m in [&tq8_metrics, &d2q8_metrics] {
            println!(
                "  {:<18} {:>8.4} {:>8.4} {:>8.3} {:>8.2} {:>6.1}% {:>6.1}% {:>6.1}% {:>8.2} {:>8.2}",
                m.label,
                m.mean_kl,
                m.median_kl,
                m.p95_kl,
                m.max_kl,
                m.top1_agree,
                m.top5_agree,
                m.top10_agree,
                m.ppl_ref,
                m.ppl_test
            );
        }
        println!("  ═══════════════════════════════════════════════════════════════════════════");

        // Compare with E2B reference numbers
        println!();
        println!("  E2B reference: D2Q-3+TQ-INT8 mean KL=2.21, top-1=55.2%");
        println!(
            "  E4B result:    D2Q-3+TQ-INT8 mean KL={:.4}, top-1={:.1}%",
            d2q8_metrics.mean_kl, d2q8_metrics.top1_agree
        );
    }

    /// Gemma 4 E4B-IT benchmark: FP16 baseline, TQ-INT8, and D2Q-3+TQ-INT8.
    /// Measures load time, GPU memory, decode throughput, and perplexity.
    #[test]
    #[ignore] // requires Metal GPU + Gemma 4 E4B-IT weights + OASST1 dataset
    fn benchmark_e4b() {
        let model_dir = hf_model_dir("google/gemma-4-E4B-it");
        let decode_tokens = 20;
        let sequences = load_eval_dataset();
        let ppl_seqs = 3; // fewer seqs — E4B is larger
        let mut results: Vec<BenchResult> = Vec::new();

        eprintln!(
            "  OASST1 dataset: {} sequences loaded (eval {})",
            sequences.len(),
            ppl_seqs
        );

        // ── FP16 baseline (FP16 KV cache) ────────────────────────
        let fp16_provider =
            SafeTensorsProvider::load(&model_dir).expect("SafeTensorsProvider::load failed");

        {
            let config = MetalConfig::default().without_turboquant();
            let (mut engine, gpu_mb, load_ms) = load_gpu_engine(&fp16_provider, config);
            let (ms_tok, tok_s) = bench_decode(&mut engine, decode_tokens);
            let (ppl, ppl_tokens, ppl_tps) = eval_perplexity(&mut engine, &sequences, ppl_seqs);
            eprintln!("  FP16 PPL: {ppl:.2} ({ppl_tokens} tokens, {ppl_tps:.0} tok/s)");
            results.push(BenchResult {
                label: "FP16",
                load_ms,
                gpu_mb,
                ms_tok,
                tok_s,
                ppl,
                ppl_tokens,
            });
        }

        // ── FP16 + TurboQuant-INT8 KV ────────────────────────────
        {
            let config = MetalConfig::default().with_turboquant(8);
            let (mut engine, gpu_mb, load_ms) = load_gpu_engine(&fp16_provider, config);
            let (ms_tok, tok_s) = bench_decode(&mut engine, decode_tokens);
            let (ppl, ppl_tokens, ppl_tps) = eval_perplexity(&mut engine, &sequences, ppl_seqs);
            eprintln!("  FP16+TQ-INT8 PPL: {ppl:.2} ({ppl_tokens} tokens, {ppl_tps:.0} tok/s)");
            results.push(BenchResult {
                label: "FP16 + TQ-INT8 KV",
                load_ms,
                gpu_mb,
                ms_tok,
                tok_s,
                ppl,
                ppl_tokens,
            });
        }

        drop(fp16_provider);

        // ── D2Quant 3-bit (JIT) + TurboQuant-INT8 KV ─────────────
        {
            let d2q_provider = QuantizedWeightProvider::new(
                SafeTensorsProvider::load(&model_dir).expect("SafeTensorsProvider::load failed"),
                D2QuantConfig::three_bit(),
            );

            let config = MetalConfig::default().with_turboquant(8);
            let (mut engine, gpu_mb, load_ms) = load_gpu_engine(&d2q_provider, config);

            let (ms_tok, tok_s) = bench_decode(&mut engine, decode_tokens);
            let (ppl, ppl_tokens, ppl_tps) = eval_perplexity(&mut engine, &sequences, ppl_seqs);
            eprintln!("  D2Q-3 (JIT) PPL: {ppl:.2} ({ppl_tokens} tokens, {ppl_tps:.0} tok/s)");
            results.push(BenchResult {
                label: "D2Q-3 JIT + TQ-INT8",
                load_ms,
                gpu_mb,
                ms_tok,
                tok_s,
                ppl,
                ppl_tokens,
            });
        }

        // ── Print results table ──────────────────────────────────
        let fp16_gpu = results[0].gpu_mb;
        let fp16_ppl = results[0].ppl;
        println!();
        println!(
            "  Gemma 4 E4B-IT — Metal Benchmark (OASST1 instruct, {ppl_seqs} seqs, {} tokens)",
            results[0].ppl_tokens
        );
        println!("  ──────────────────────────────────────────────────────────────────────────");
        println!(
            "  {:<20} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
            "Config", "GPU MB", "ms/tok", "tok/s", "PPL", "ΔPPL", "Load ms"
        );
        println!("  ──────────────────────────────────────────────────────────────────────────");
        for r in &results {
            let mem_note = if r.gpu_mb < fp16_gpu * 0.95 {
                format!(" (−{:.0}%)", (1.0 - r.gpu_mb / fp16_gpu) * 100.0)
            } else {
                String::new()
            };
            let ppl_delta = if (r.ppl - fp16_ppl).abs() > 0.01 {
                format!("{:>+.1}%", (r.ppl - fp16_ppl) / fp16_ppl * 100.0)
            } else {
                "—".to_string()
            };
            println!(
                "  {:<20} {:>7.0}{:<7} {:>7.2} {:>7.1} {:>7.2} {:>9} {:>10.0}",
                r.label, r.gpu_mb, mem_note, r.ms_tok, r.tok_s, r.ppl, ppl_delta, r.load_ms
            );
        }
        println!("  ──────────────────────────────────────────────────────────────────────────");
        println!();

        // ── Assertions ───────────────────────────────────────────
        for r in &results {
            assert!(
                r.tok_s > 0.0,
                "{} should produce >0 tok/s, got {:.1}",
                r.label,
                r.tok_s
            );
            assert!(
                r.ppl.is_finite() && r.ppl > 0.0,
                "{} PPL should be finite and positive, got {:.2}",
                r.label,
                r.ppl
            );
        }
    }
}
