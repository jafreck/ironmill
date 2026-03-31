//! INT8 vs FP16 KV cache bandwidth benchmark on ANE.
//!
//! Tests whether TurboQuant's INT8 cache provides real throughput gains
//! over FP16 at larger model dimensions and longer sequence lengths.
//!
//! Run with: cargo run -p ironmill-inference --example cache_bandwidth_bench --release
//!
//! This is a synthetic benchmark — no model weights needed. It compiles
//! and evaluates only the attention sub-program with configurable
//! dimensions, directly measuring cache read bandwidth impact.

use std::time::Instant;

use ironmill_ane_sys::AneCompiler;
use ironmill_inference::ane::AneRuntime;
use ironmill_inference::ane::turboquant::mil_emitter;
use ironmill_inference::ane::turboquant::{AttentionMilConfig, compute_deq_scale};
use ironmill_iosurface::{AneTensor, uniform_alloc_size};
use mil_rs::ir::ScalarType;

const WARMUP: usize = 10;
const ITERATIONS: usize = 100;

struct BenchConfig {
    label: &'static str,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║    INT8 vs FP16 KV Cache Bandwidth Benchmark (ANE)       ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    let configs = vec![
        // Qwen3-0.6B dimensions at various seq lengths
        BenchConfig {
            label: "Qwen3-0.6B @ 512",
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 64,
            max_seq_len: 512,
        },
        BenchConfig {
            label: "Qwen3-0.6B @ 2048",
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 64,
            max_seq_len: 2048,
        },
        // Larger KV dimensions (Qwen3-8B style: 32 heads, 8 KV heads, 128 head_dim)
        BenchConfig {
            label: "8B-style (8 KV) @ 512",
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 512,
        },
        BenchConfig {
            label: "8B-style (8 KV) @ 2048",
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 2048,
        },
        BenchConfig {
            label: "8B-style (8 KV) @ 4096",
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 4096,
        },
    ];

    println!(
        "  {:30} {:>10} {:>10} {:>8} {:>8} {:>7}",
        "Config", "INT8 (μs)", "FP16 (μs)", "Speedup", "INT8 MB", "FP16 MB"
    );
    println!("  {}", "─".repeat(79));

    for cfg in &configs {
        let result = bench_config(cfg);
        match result {
            Ok((int8_us, fp16_us)) => {
                let kv_ch = cfg.num_kv_heads * cfg.head_dim;
                let int8_mb = 2.0 * kv_ch as f64 * cfg.max_seq_len as f64 / 1024.0 / 1024.0;
                let fp16_mb = int8_mb * 2.0;
                let speedup = fp16_us / int8_us;
                println!(
                    "  {:30} {:>10.0} {:>10.0} {:>7.2}x {:>7.1} {:>7.1}",
                    cfg.label, int8_us, fp16_us, speedup, int8_mb, fp16_mb
                );
            }
            Err(e) => {
                println!("  {:30} FAILED: {e}", cfg.label);
            }
        }
    }
    println!();
}

fn bench_config(cfg: &BenchConfig) -> Result<(f64, f64), String> {
    let kv_ch = cfg.num_kv_heads * cfg.head_dim;
    let q_ch = cfg.num_heads * cfg.head_dim;

    // --- INT8 attention (TurboQuant path) ---
    let deq_scale = mil_emitter::compute_deq_scale(cfg.head_dim, 8);
    let attn_config_int8 = mil_emitter::AttentionMilConfig {
        num_heads: cfg.num_heads,
        num_kv_heads: cfg.num_kv_heads,
        head_dim: cfg.head_dim,
        max_seq_len: cfg.max_seq_len,
        seq_len: cfg.max_seq_len,
        dequant_scale: Some(deq_scale),
        unrotation_seed: Some(42),
        cache_int8: true,
    };
    let (int8_mil, _) = mil_emitter::emit_attention_mil(&attn_config_int8);
    let int8_compiled = AneCompiler::compile_mil_text(&int8_mil, &[])
        .map_err(|e| format!("INT8 attention compile failed: {e}"))?;

    // --- FP16 attention (baseline) ---
    let fp16_mil = mil_emitter::emit_fp16_attention_mil(
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
        cfg.max_seq_len,
        cfg.max_seq_len,
    );
    let fp16_compiled = AneCompiler::compile_mil_text(&fp16_mil, &[])
        .map_err(|e| format!("FP16 attention compile failed: {e}"))?;

    // --- Allocate tensors ---
    let int8_alloc = uniform_alloc_size(&[
        ([1, q_ch, 1, 32], ScalarType::Float16),
        ([1, kv_ch, 1, cfg.max_seq_len], ScalarType::Int8),
        ([1, kv_ch, 1, cfg.max_seq_len], ScalarType::Int8),
        ([1, 1, cfg.head_dim, cfg.head_dim], ScalarType::Float16),
    ]);
    let fp16_alloc = uniform_alloc_size(&[
        ([1, q_ch, 1, 32], ScalarType::Float16),
        ([1, kv_ch, 1, cfg.max_seq_len], ScalarType::Float16),
        ([1, kv_ch, 1, cfg.max_seq_len], ScalarType::Float16),
    ]);

    let runtime = AneRuntime::new().map_err(|e| format!("Runtime init failed: {e}"))?;
    let int8_loaded = runtime
        .load_program(&int8_compiled)
        .map_err(|e| format!("INT8 load failed: {e}"))?;
    let fp16_loaded = runtime
        .load_program(&fp16_compiled)
        .map_err(|e| format!("FP16 load failed: {e}"))?;

    // INT8 inputs: Q, K_cache(int8), V_cache(int8), rotation_matrix
    let q_int8 = AneTensor::new_with_min_alloc(q_ch, 32, ScalarType::Float16, int8_alloc)
        .map_err(|e| format!("{e}"))?;
    let k_cache_int8 =
        AneTensor::new_with_min_alloc(kv_ch, cfg.max_seq_len, ScalarType::Int8, int8_alloc)
            .map_err(|e| format!("{e}"))?;
    let v_cache_int8 =
        AneTensor::new_with_min_alloc(kv_ch, cfg.max_seq_len, ScalarType::Int8, int8_alloc)
            .map_err(|e| format!("{e}"))?;
    let rot =
        AneTensor::new_with_min_alloc(cfg.head_dim, cfg.head_dim, ScalarType::Float16, int8_alloc)
            .map_err(|e| format!("{e}"))?;
    let mut out_int8 = AneTensor::new_with_min_alloc(q_ch, 32, ScalarType::Float16, int8_alloc)
        .map_err(|e| format!("{e}"))?;

    // FP16 inputs: Q, K_cache(fp16), V_cache(fp16)
    let q_fp16 = AneTensor::new_with_min_alloc(q_ch, 32, ScalarType::Float16, fp16_alloc)
        .map_err(|e| format!("{e}"))?;
    let k_cache_fp16 =
        AneTensor::new_with_min_alloc(kv_ch, cfg.max_seq_len, ScalarType::Float16, fp16_alloc)
            .map_err(|e| format!("{e}"))?;
    let v_cache_fp16 =
        AneTensor::new_with_min_alloc(kv_ch, cfg.max_seq_len, ScalarType::Float16, fp16_alloc)
            .map_err(|e| format!("{e}"))?;
    let mut out_fp16 = AneTensor::new_with_min_alloc(q_ch, 32, ScalarType::Float16, fp16_alloc)
        .map_err(|e| format!("{e}"))?;

    // --- Benchmark INT8 ---
    for _ in 0..WARMUP {
        let _ = runtime.eval(
            &int8_loaded,
            &[&q_int8, &k_cache_int8, &v_cache_int8, &rot],
            &mut [&mut out_int8],
        );
    }
    let mut int8_latencies = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let t = Instant::now();
        runtime
            .eval(
                &int8_loaded,
                &[&q_int8, &k_cache_int8, &v_cache_int8, &rot],
                &mut [&mut out_int8],
            )
            .map_err(|e| format!("INT8 eval failed: {e}"))?;
        int8_latencies.push(t.elapsed().as_micros() as f64);
    }
    int8_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let int8_p50 = int8_latencies[int8_latencies.len() / 2];

    // --- Benchmark FP16 ---
    for _ in 0..WARMUP {
        let _ = runtime.eval(
            &fp16_loaded,
            &[&q_fp16, &k_cache_fp16, &v_cache_fp16],
            &mut [&mut out_fp16],
        );
    }
    let mut fp16_latencies = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let t = Instant::now();
        runtime
            .eval(
                &fp16_loaded,
                &[&q_fp16, &k_cache_fp16, &v_cache_fp16],
                &mut [&mut out_fp16],
            )
            .map_err(|e| format!("FP16 eval failed: {e}"))?;
        fp16_latencies.push(t.elapsed().as_micros() as f64);
    }
    fp16_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let fp16_p50 = fp16_latencies[fp16_latencies.len() / 2];

    Ok((int8_p50, fp16_p50))
}
