//! TurboQuant E2E inference benchmark — FP16 baseline vs INT8 TurboQuant.
//!
//! Run with: cargo run -p ironmill-ane --example turboquant_e2e_bench --release
//!
//! Loads Qwen3-0.6B from ONNX, compiles with `AneInference` in both modes,
//! and measures throughput, memory, and output quality (token agreement).

use std::time::Instant;

use ironmill_ane::inference::AneInference;
use ironmill_ane::turboquant::TurboQuantConfig;

/// Maximum sequence length for KV cache allocation.
const MAX_SEQ_LEN: usize = 512;

/// Number of tokens to generate for benchmarking.
const GEN_TOKENS: usize = 128;

/// Example prompt tokens (small prompt for benchmarking).
const PROMPT: &[u32] = &[1, 1820, 338, 263, 1243, 2462]; // "This is a test prompt"

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║       TurboQuant E2E Inference Benchmark                  ║");
    println!("║  Model: Qwen3-0.6B (ONNX → MIL IR → ANE)                ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    // ── Load ONNX model ──────────────────────────────────────────────
    let onnx_path = "tests/fixtures/qwen3-0.6b.onnx";
    println!("  Loading ONNX model from {onnx_path}...");

    let onnx = match mil_rs::reader::onnx::read_onnx(onnx_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("  ✗ Failed to read ONNX model: {e}");
            eprintln!("    Place qwen3-0.6b.onnx in tests/fixtures/ to run this benchmark.");
            return;
        }
    };

    println!("  Converting ONNX → MIL IR...");
    // Use model_dir so external weight data files are resolved.
    let model_dir = std::path::Path::new(onnx_path)
        .parent()
        .map(|p| p.to_path_buf());
    let config = mil_rs::convert::onnx_graph::ConversionConfig {
        model_dir,
        ..Default::default()
    };
    let conversion = match mil_rs::convert::onnx_graph::onnx_to_program_with_config(&onnx, &config)
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  ✗ ONNX conversion failed: {e}");
            return;
        }
    };
    let program = conversion.program;

    // ── Detect architecture ──────────────────────────────────────────
    let arch = match mil_rs::analysis::arch::detect_model_arch(&program) {
        Some(a) => a,
        None => {
            eprintln!("  ✗ Failed to detect model architecture");
            return;
        }
    };

    println!(
        "  Architecture: {} layers, {} heads ({} KV), head_dim={}, hidden={}",
        arch.num_layers, arch.num_heads, arch.num_kv_heads, arch.head_dim, arch.hidden_size
    );
    println!();

    // ── Baseline: FP16 KV cache ──────────────────────────────────────
    println!("  ┌─ Baseline (FP16 KV cache) ──────────────────────────────");
    println!("  │  Compiling...");

    let baseline_compile_start = Instant::now();
    let baseline_result = AneInference::compile(&program, None);
    let baseline_compile_time = baseline_compile_start.elapsed();

    let (baseline_output, baseline_tps) = match baseline_result {
        Ok(mut baseline) => {
            println!("  │  Compile time: {:.2?}", baseline_compile_time);
            println!("  │  Generating {GEN_TOKENS} tokens...");
            let baseline_gen_start = Instant::now();
            match baseline.generate(PROMPT, GEN_TOKENS, 0.0) {
                Ok(tokens) => {
                    let gen_time = baseline_gen_start.elapsed();
                    let tps = tokens.len() as f64 / gen_time.as_secs_f64();
                    let fp16_kv_bytes =
                        arch.num_layers * 2 * arch.num_kv_heads * arch.head_dim * MAX_SEQ_LEN * 2;
                    println!("  │  Tokens generated: {}", tokens.len());
                    println!("  │  Throughput: {tps:.1} tok/s");
                    println!(
                        "  │  KV cache size: {:.1} MB",
                        fp16_kv_bytes as f64 / 1_048_576.0
                    );
                    (Some(tokens), Some(tps))
                }
                Err(e) => {
                    eprintln!("  │  ✗ Baseline generation failed: {e}");
                    (None, None)
                }
            }
        }
        Err(e) => {
            eprintln!("  │  ✗ Baseline compilation failed: {e}");
            (None, None)
        }
    };
    println!("  └────────────────────────────────────────────────────────");
    println!();

    // ── TurboQuant: INT8 KV cache ────────────────────────────────────
    println!("  ┌─ TurboQuant (INT8 KV cache) ─────────────────────────────");
    println!("  │  Compiling...");

    let tq_config = match TurboQuantConfig::from_arch(&arch, MAX_SEQ_LEN) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  │  ✗ TurboQuant config failed: {e}");
            return;
        }
    };

    let turbo_compile_start = Instant::now();
    let mut turbo = match AneInference::compile(&program, Some(tq_config)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("  │  ✗ TurboQuant compilation failed: {e}");
            return;
        }
    };
    let turbo_compile_time = turbo_compile_start.elapsed();
    println!("  │  Compile time: {:.2?}", turbo_compile_time);

    println!("  │  Generating {GEN_TOKENS} tokens...");
    let turbo_gen_start = Instant::now();
    let turbo_output = match turbo.generate(PROMPT, GEN_TOKENS, 0.0) {
        Ok(tokens) => tokens,
        Err(e) => {
            eprintln!("  │  ✗ TurboQuant generation failed: {e}");
            return;
        }
    };
    let turbo_gen_time = turbo_gen_start.elapsed();
    let turbo_tps = turbo_output.len() as f64 / turbo_gen_time.as_secs_f64();

    let int8_kv_bytes = arch.num_layers * 2 * arch.num_kv_heads * arch.head_dim * MAX_SEQ_LEN * 1; // 1 byte/elem for INT8

    println!("  │  Tokens generated: {}", turbo_output.len());
    println!("  │  Throughput: {turbo_tps:.1} tok/s");
    println!(
        "  │  KV cache size: {:.1} MB",
        int8_kv_bytes as f64 / 1_048_576.0
    );
    println!("  └────────────────────────────────────────────────────────");
    println!();

    // ── Comparison ───────────────────────────────────────────────────
    println!("  ┌─ Comparison ─────────────────────────────────────────────");

    let fp16_kv_bytes = arch.num_layers * 2 * arch.num_kv_heads * arch.head_dim * MAX_SEQ_LEN * 2;
    let memory_ratio = int8_kv_bytes as f64 / fp16_kv_bytes as f64;
    println!("  │  Memory ratio:      {memory_ratio:.2}x (TQ vs baseline)");

    if let (Some(bl_out), Some(bl_tps)) = (&baseline_output, baseline_tps) {
        let speedup = turbo_tps / bl_tps;
        let min_len = bl_out.len().min(turbo_output.len());
        let agreement = if min_len > 0 {
            let matching = bl_out[..min_len]
                .iter()
                .zip(turbo_output[..min_len].iter())
                .filter(|(a, b)| a == b)
                .count();
            matching as f64 / min_len as f64
        } else {
            0.0
        };
        println!("  │  Throughput ratio:  {speedup:.2}x (TQ vs baseline)");
        println!(
            "  │  Token agreement:   {:.1}% ({}/{})",
            agreement * 100.0,
            (agreement * min_len as f64) as usize,
            min_len,
        );
    } else {
        println!("  │  Throughput ratio:  N/A (baseline unavailable)");
        println!("  │  Token agreement:   N/A (baseline unavailable)");
    }
    println!("  └────────────────────────────────────────────────────────");
}
