//! TurboQuant E2E inference benchmark — FP16 baseline vs INT8 TurboQuant.
//!
//! Usage:
//!   cargo run -p ironmill-inference --example turboquant_e2e_bench --release [-- MODE]
//!
//! MODE:
//!   fp16       Run FP16 baseline only
//!   tq         Run TurboQuant INT8 only
//!   both       Run both and compare (default)

use std::sync::Arc;
use std::time::Instant;

use ironmill_inference::ane::HardwareAneDevice;
use ironmill_inference::ane::decode::AneInference;
use ironmill_inference::ane::turboquant::TurboQuantConfig;

const MAX_SEQ_LEN: usize = 512;
const GEN_TOKENS: usize = 128;
const PROMPT: &[u32] = &[1, 1820, 338, 263, 1243, 2462];

fn run_model(
    label: &str,
    program: &mil_rs::ir::Program,
    arch: &mil_rs::analysis::arch::ModelArch,
    turbo_config: Option<TurboQuantConfig>,
) -> Option<(Vec<u32>, f64)> {
    use ironmill_compile::ane::bundle::{AneDecodeConfig, compile_decode_bundle};

    let is_tq = turbo_config.is_some();
    let cache_label = if is_tq { "INT8" } else { "FP16" };
    println!("  ┌─ {label} ({cache_label} KV cache) ──────────────────────────");
    println!("  │  Compiling...");

    let compile_start = Instant::now();
    let device = Arc::new(HardwareAneDevice::new().expect("Failed to init ANE device"));

    let config = AneDecodeConfig {
        max_seq_len: MAX_SEQ_LEN,
        num_heads: arch.num_heads,
        num_kv_heads: arch.num_kv_heads,
        head_dim: arch.head_dim,
        rope_theta: 1_000_000.0,
        eos_tokens: Vec::new(),
        fuse_cache_write: turbo_config.is_some(),
        enable_qjl: false,
    };

    let bundle = match compile_decode_bundle(program, &config) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("  │  ✗ Compilation failed: {e}");
            println!("  └────────────────────────────────────────────────────────");
            return None;
        }
    };
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let bundle_path = tmp.path().join("model.ironml");
    bundle.save(&bundle_path).expect("failed to save bundle");

    let mut model = match AneInference::from_bundle(device, &bundle_path, turbo_config) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("  │  ✗ Bundle load failed: {e}");
            println!("  └────────────────────────────────────────────────────────");
            return None;
        }
    };
    println!("  │  Compile time: {:.2?}", compile_start.elapsed());

    println!("  │  Generating {GEN_TOKENS} tokens...");
    let gen_start = Instant::now();
    let tokens = match model.generate(PROMPT, GEN_TOKENS, 0.0) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("  │  ✗ Generation failed: {e}");
            println!("  └────────────────────────────────────────────────────────");
            return None;
        }
    };
    let tps = tokens.len() as f64 / gen_start.elapsed().as_secs_f64();

    let elem_bytes = if is_tq { 1 } else { 2 };
    let kv_bytes =
        arch.num_layers * 2 * arch.num_kv_heads * arch.head_dim * MAX_SEQ_LEN * elem_bytes;
    println!("  │  Tokens generated: {}", tokens.len());
    println!("  │  Throughput: {tps:.1} tok/s");
    println!(
        "  │  KV cache size: {:.1} MB",
        kv_bytes as f64 / 1_048_576.0
    );
    println!("  └────────────────────────────────────────────────────────");
    println!();

    Some((tokens, tps))
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "both".into());
    let run_fp16 = mode == "both" || mode == "fp16";
    let run_tq = mode == "both" || mode == "tq";

    if !run_fp16 && !run_tq {
        eprintln!("Usage: turboquant_e2e_bench [fp16|tq|both]");
        std::process::exit(1);
    }

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║       TurboQuant E2E Inference Benchmark                  ║");
    println!("║  Model: Qwen3-0.6B (ONNX → MIL IR → ANE)                ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!("  Mode: {mode}");
    println!();

    let onnx_path = "tests/fixtures/qwen3-0.6b.onnx";
    println!("  Loading ONNX model from {onnx_path}...");
    let mut onnx = match mil_rs::reader::onnx::read_onnx(onnx_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("  ✗ Failed to read ONNX model: {e}");
            eprintln!("    Place qwen3-0.6b.onnx in tests/fixtures/");
            return;
        }
    };

    println!("  Converting ONNX → MIL IR...");
    let model_dir = std::path::Path::new(onnx_path)
        .parent()
        .map(|p| p.to_path_buf());
    let config = mil_rs::convert::onnx_graph::ConversionConfig {
        model_dir,
        ..Default::default()
    };
    let program = match mil_rs::convert::onnx_graph::onnx_to_program_with_config(&mut onnx, &config)
    {
        Ok(c) => c.program,
        Err(e) => {
            eprintln!("  ✗ ONNX conversion failed: {e}");
            return;
        }
    };

    let arch = match mil_rs::analysis::arch::detect_model_arch(&program) {
        Some(a) => a,
        None => {
            eprintln!("  ✗ Could not detect model architecture");
            return;
        }
    };
    println!(
        "  Architecture: {} layers, {} heads ({} KV), head_dim={}, hidden={}",
        arch.num_layers, arch.num_heads, arch.num_kv_heads, arch.head_dim, arch.hidden_size,
    );
    println!();

    let fp16_result = if run_fp16 {
        run_model("Baseline", &program, &arch, None)
    } else {
        None
    };

    let tq_result = if run_tq {
        match TurboQuantConfig::from_arch(&arch, MAX_SEQ_LEN) {
            Ok(c) => run_model("TurboQuant", &program, &arch, Some(c)),
            Err(e) => {
                eprintln!("  ✗ TurboQuant config failed: {e}");
                None
            }
        }
    } else {
        None
    };

    // Comparison (only when both ran successfully).
    if let (Some((fp16_tok, fp16_tps)), Some((tq_tok, tq_tps))) = (&fp16_result, &tq_result) {
        println!("  ┌─ Comparison ─────────────────────────────────────────────");
        let fp16_kv = arch.num_layers * 2 * arch.num_kv_heads * arch.head_dim * MAX_SEQ_LEN * 2;
        let int8_kv = arch.num_layers * 2 * arch.num_kv_heads * arch.head_dim * MAX_SEQ_LEN;
        println!(
            "  │  Memory ratio:      {:.2}x",
            int8_kv as f64 / fp16_kv as f64
        );
        println!(
            "  │  Throughput ratio:  {:.2}x (TQ vs baseline)",
            tq_tps / fp16_tps
        );
        let min_len = fp16_tok.len().min(tq_tok.len());
        if min_len > 0 {
            let matching = fp16_tok[..min_len]
                .iter()
                .zip(tq_tok[..min_len].iter())
                .filter(|(a, b)| a == b)
                .count();
            println!(
                "  │  Token agreement:   {:.1}% ({}/{})",
                matching as f64 / min_len as f64 * 100.0,
                matching,
                min_len
            );
        }
        println!("  └────────────────────────────────────────────────────────");
    }
}
