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

use std::sync::Arc;
use std::time::Instant;

use ironmill_compile::ane::mil_text::{MilTextConfig, program_to_mil_text};
use ironmill_inference::ane::turboquant::mil_emitter;
use ironmill_inference::ane::turboquant::{AttentionMilConfig, compute_deq_scale};
use ironmill_inference::ane::{AneDevice, HardwareAneDevice};
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
        // MHA models (32 KV heads) — cache hits SRAM limit (~32MB) faster
        // K+V cache at 4096: 32×128×4096×2=32MB (FP16) vs 16MB (INT8)
        BenchConfig {
            label: "8B MHA (32 KV) @ 2048",
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            max_seq_len: 2048,
        },
        BenchConfig {
            label: "8B MHA (32 KV) @ 4096",
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            max_seq_len: 4096,
        },
        // Beyond SRAM — cache spills to DRAM
        BenchConfig {
            label: "8B MHA (32 KV) @ 8192",
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            max_seq_len: 8192,
        },
        BenchConfig {
            label: "8B GQA (8 KV) @ 8192",
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 8192,
        },
    ];

    println!(
        "  {:28} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Config", "INT8+TQ", "INT8raw", "FP16", "raw/fp16", "Cache"
    );
    println!(
        "  {:28} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "", "(μs)", "(μs)", "(μs)", "", "(MB)"
    );
    println!("  {}", "─".repeat(72));

    for cfg in &configs {
        let result = bench_config(cfg);
        match result {
            Ok((int8_tq_us, int8_raw_us, fp16_us)) => {
                let kv_ch = cfg.num_kv_heads * cfg.head_dim;
                let fp16_mb = 2.0 * kv_ch as f64 * cfg.max_seq_len as f64 * 2.0 / 1024.0 / 1024.0;
                let speedup = fp16_us / int8_raw_us;
                println!(
                    "  {:28} {:>8.0} {:>8.0} {:>8.0} {:>7.2}x {:>7.1}",
                    cfg.label, int8_tq_us, int8_raw_us, fp16_us, speedup, fp16_mb
                );
            }
            Err(e) => {
                println!("  {:28} FAILED: {e}", cfg.label);
            }
        }
    }
    println!();
}

fn bench_config(cfg: &BenchConfig) -> Result<(f64, f64, f64), String> {
    let kv_ch = cfg.num_kv_heads * cfg.head_dim;
    let q_ch = cfg.num_heads * cfg.head_dim;

    // --- INT8 attention (TurboQuant path: cast + dequant + rotation) ---
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
    let (int8_program, _) = mil_emitter::build_attention_program(&attn_config_int8);
    let mil_config = MilTextConfig::default();
    let (int8_mil, _) = program_to_mil_text(&int8_program, &mil_config)
        .map_err(|e| format!("INT8+TQ attention MIL text failed: {e}"))?;

    let device =
        Arc::new(HardwareAneDevice::new().map_err(|e| format!("Device init failed: {e}"))?);

    let int8_program_loaded = device
        .compile(&int8_mil, &[])
        .map_err(|e| format!("INT8+TQ attention compile failed: {e}"))?;

    // --- INT8 raw attention (cast only, no dequant, no rotation) ---
    // Isolates the pure INT8→FP16 bandwidth cost
    let attn_config_int8_raw = mil_emitter::AttentionMilConfig {
        num_heads: cfg.num_heads,
        num_kv_heads: cfg.num_kv_heads,
        head_dim: cfg.head_dim,
        max_seq_len: cfg.max_seq_len,
        seq_len: cfg.max_seq_len,
        dequant_scale: None,   // no dequant mul
        unrotation_seed: None, // no rotation
        cache_int8: true,      // still INT8 input → cast to fp16
    };
    let (int8_raw_program, _) = mil_emitter::build_attention_program(&attn_config_int8_raw);
    let (int8_raw_mil, _) = program_to_mil_text(&int8_raw_program, &mil_config)
        .map_err(|e| format!("INT8 raw attention MIL text failed: {e}"))?;
    let int8_raw_loaded = device
        .compile(&int8_raw_mil, &[])
        .map_err(|e| format!("INT8 raw attention compile failed: {e}"))?;

    // --- FP16 attention (baseline) ---
    let fp16_program = mil_emitter::build_fp16_attention_program(
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
        cfg.max_seq_len,
        cfg.max_seq_len,
    );
    let (fp16_mil, _) = program_to_mil_text(&fp16_program, &mil_config)
        .map_err(|e| format!("FP16 attention MIL text failed: {e}"))?;
    let fp16_loaded = device
        .compile(&fp16_mil, &[])
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

    // INT8+TQ inputs: Q, K_cache(int8), V_cache(int8), rotation_matrix
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

    // INT8 raw inputs: Q, K_cache(int8), V_cache(int8) — no rotation tensor
    let int8_raw_alloc = uniform_alloc_size(&[
        ([1, q_ch, 1, 32], ScalarType::Float16),
        ([1, kv_ch, 1, cfg.max_seq_len], ScalarType::Int8),
        ([1, kv_ch, 1, cfg.max_seq_len], ScalarType::Int8),
    ]);
    let q_int8_raw = AneTensor::new_with_min_alloc(q_ch, 32, ScalarType::Float16, int8_raw_alloc)
        .map_err(|e| format!("{e}"))?;
    let k_cache_int8_raw =
        AneTensor::new_with_min_alloc(kv_ch, cfg.max_seq_len, ScalarType::Int8, int8_raw_alloc)
            .map_err(|e| format!("{e}"))?;
    let v_cache_int8_raw =
        AneTensor::new_with_min_alloc(kv_ch, cfg.max_seq_len, ScalarType::Int8, int8_raw_alloc)
            .map_err(|e| format!("{e}"))?;
    let mut out_int8_raw =
        AneTensor::new_with_min_alloc(q_ch, 32, ScalarType::Float16, int8_raw_alloc)
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
        let _ = device.eval(
            &int8_program_loaded,
            &[&q_int8, &k_cache_int8, &v_cache_int8, &rot],
            &mut [&mut out_int8],
        );
    }
    let mut int8_latencies = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let t = Instant::now();
        device
            .eval(
                &int8_program_loaded,
                &[&q_int8, &k_cache_int8, &v_cache_int8, &rot],
                &mut [&mut out_int8],
            )
            .map_err(|e| format!("INT8 eval failed: {e}"))?;
        int8_latencies.push(t.elapsed().as_micros() as f64);
    }
    int8_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let int8_p50 = int8_latencies[int8_latencies.len() / 2];

    // --- Benchmark INT8 raw (cast-only, no dequant/rotation) ---
    for _ in 0..WARMUP {
        let _ = device.eval(
            &int8_raw_loaded,
            &[&q_int8_raw, &k_cache_int8_raw, &v_cache_int8_raw],
            &mut [&mut out_int8_raw],
        );
    }
    let mut int8_raw_latencies = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let t = Instant::now();
        device
            .eval(
                &int8_raw_loaded,
                &[&q_int8_raw, &k_cache_int8_raw, &v_cache_int8_raw],
                &mut [&mut out_int8_raw],
            )
            .map_err(|e| format!("INT8 raw eval failed: {e}"))?;
        int8_raw_latencies.push(t.elapsed().as_micros() as f64);
    }
    int8_raw_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let int8_raw_p50 = int8_raw_latencies[int8_raw_latencies.len() / 2];

    // --- Benchmark FP16 ---
    for _ in 0..WARMUP {
        let _ = device.eval(
            &fp16_loaded,
            &[&q_fp16, &k_cache_fp16, &v_cache_fp16],
            &mut [&mut out_fp16],
        );
    }
    let mut fp16_latencies = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let t = Instant::now();
        device
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

    Ok((int8_p50, int8_raw_p50, fp16_p50))
}
