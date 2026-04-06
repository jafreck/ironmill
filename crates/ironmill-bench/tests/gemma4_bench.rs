//! Gemma 4 E2B Metal GPU benchmark.
//!
//! Single test that loads each config once, measures decode throughput and
//! perplexity, and reports all metrics in one table.
//! **Always use `--release`** — debug mode is 27× slower.
//!
//! ```bash
//! cargo test --release -p ironmill-bench --features metal -- gemma4 --ignored --nocapture
//! ```
//!
//! Requires: Metal GPU, Gemma 4 E2B-IT weights in HuggingFace cache,
//! OASST1 instruct dataset at tests/fixtures/quality/oasst1-gemma4-instruct.json.

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

    /// Compare per-token logits between FP16 and FP16+TQ-INT4 on a single
    /// sequence to diagnose why TQ-INT4 shows lower PPL than FP16.
    #[test]
    #[ignore]
    fn investigate_tq_ppl() {
        let model_dir = gemma4_model_dir();
        let sequences = load_eval_dataset();
        let seq = &sequences[0]; // first 512-token sequence

        let provider =
            SafeTensorsProvider::load(&model_dir).expect("SafeTensorsProvider::load failed");

        // FP16 baseline logits
        let fp16_config = MetalConfig::default().without_turboquant();
        let (mut fp16_engine, _, _) = load_gpu_engine(&provider, fp16_config);
        let fp16_logits = fp16_engine
            .prefill_all_logits(seq)
            .expect("FP16 prefill failed");

        // FP16 + TQ-INT4 logits
        let tq_config = MetalConfig::default().with_turboquant(4);
        let (mut tq_engine, _, _) = load_gpu_engine(&provider, tq_config);
        let tq_logits = tq_engine
            .prefill_all_logits(seq)
            .expect("TQ prefill failed");

        assert_eq!(fp16_logits.len(), tq_logits.len());

        // Per-position analysis
        println!(
            "\n  FP16 vs TQ-INT4 logit comparison ({} positions)",
            seq.len()
        );
        println!("  ─────────────────────────────────────────────────────────────────");
        println!(
            "  {:>4} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10}",
            "pos", "FP16 CE", "TQ CE", "ΔCE", "max|Δlog|", "cosine", "top1match"
        );
        println!("  ─────────────────────────────────────────────────────────────────");

        let mut fp16_total_ce = 0.0f64;
        let mut tq_total_ce = 0.0f64;
        let mut ce_better = 0usize; // positions where TQ has lower CE
        let mut ce_worse = 0usize;
        let mut top1_mismatch = 0usize;
        let mut max_logit_diff_positions: Vec<(usize, f64)> = Vec::new();

        for pos in 0..seq.len() - 1 {
            let target = seq[pos + 1] as usize;
            let fp = &fp16_logits[pos];
            let tq = &tq_logits[pos];

            // Cross-entropy
            let fp_max = fp.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let fp_lse: f64 = fp
                .iter()
                .map(|&x| ((x - fp_max) as f64).exp())
                .sum::<f64>()
                .ln()
                + fp_max as f64;
            let fp_ce = -(fp[target] as f64 - fp_lse);

            let tq_max = tq.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let tq_lse: f64 = tq
                .iter()
                .map(|&x| ((x - tq_max) as f64).exp())
                .sum::<f64>()
                .ln()
                + tq_max as f64;
            let tq_ce = -(tq[target] as f64 - tq_lse);

            fp16_total_ce += fp_ce;
            tq_total_ce += tq_ce;

            if tq_ce < fp_ce - 0.01 {
                ce_better += 1;
            } else if tq_ce > fp_ce + 0.01 {
                ce_worse += 1;
            }

            // Max absolute logit difference
            let max_diff: f64 = fp
                .iter()
                .zip(tq.iter())
                .map(|(&a, &b)| (a as f64 - b as f64).abs())
                .fold(0.0, f64::max);

            // Cosine similarity
            let dot: f64 = fp
                .iter()
                .zip(tq.iter())
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum();
            let fp_norm: f64 = fp.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
            let tq_norm: f64 = tq.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
            let cosine = dot / (fp_norm * tq_norm + 1e-10);

            // Top-1 match
            let fp_top1 = fp
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
            let tq_top1 = tq
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
            let top1_match = fp_top1 == tq_top1;
            if !top1_match {
                top1_mismatch += 1;
            }

            max_logit_diff_positions.push((pos, max_diff));

            // Print every 50th position + first/last + any with large CE delta
            let delta_ce = tq_ce - fp_ce;
            if pos < 5
                || pos % 100 == 0
                || pos == seq.len() - 2
                || delta_ce.abs() > 2.0
                || max_diff > 5.0
            {
                println!(
                    "  {:>4} {:>8.3} {:>8.3} {:>+10.3} {:>10.3} {:>10.6} {:>10}",
                    pos,
                    fp_ce,
                    tq_ce,
                    delta_ce,
                    max_diff,
                    cosine,
                    if top1_match { "✓" } else { "✗" }
                );
            }
        }

        // Sort by max logit difference
        max_logit_diff_positions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let fp_ppl = (fp16_total_ce / (seq.len() - 1) as f64).exp();
        let tq_ppl = (tq_total_ce / (seq.len() - 1) as f64).exp();

        println!("  ─────────────────────────────────────────────────────────────────");
        println!("\n  Summary:");
        println!("    FP16 PPL: {fp_ppl:.2}");
        println!(
            "    TQ-INT4 PPL: {tq_ppl:.2} ({:+.1}%)",
            (tq_ppl - fp_ppl) / fp_ppl * 100.0
        );
        println!(
            "    Positions where TQ is better: {ce_better} / worse: {ce_worse} / same: {}",
            seq.len() - 1 - ce_better - ce_worse
        );
        println!("    Top-1 prediction mismatches: {top1_mismatch}");
        println!(
            "    Top 5 max |Δlogit| positions: {:?}",
            max_logit_diff_positions[..5]
                .iter()
                .map(|(p, d)| format!("pos {p}: {d:.2}"))
                .collect::<Vec<_>>()
        );
        println!(
            "    Avg max |Δlogit|: {:.3}",
            max_logit_diff_positions.iter().map(|(_, d)| d).sum::<f64>()
                / max_logit_diff_positions.len() as f64
        );
    }
}
