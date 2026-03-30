//! TurboQuant ANE benchmark — measures compile + per-token attention latency.
//!
//! Run with: cargo run -p ironmill-ane --example turboquant_bench --release
//!
//! Tests TurboQuantModel with real model dimensions (Qwen3-0.6B architecture).

use std::time::Instant;

use half::f16;
use ironmill_ane::tensor::AneTensor;
use ironmill_ane::turboquant::{TurboQuantConfig, TurboQuantModel};
use mil_rs::ir::ScalarType;

/// Qwen3-0.6B architecture parameters.
const NUM_HEADS: usize = 14;
const NUM_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const NUM_LAYERS: usize = 28;
const MAX_SEQ_LEN: usize = 512; // keep short for benchmark speed

const WARMUP: usize = 5;
const ITERATIONS: usize = 50;

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║         TurboQuant ANE Benchmark                          ║");
    println!("║  Model: Qwen3-0.6B architecture (dummy weights)           ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    let kv_ch = NUM_KV_HEADS * HEAD_DIM;
    let q_ch = NUM_HEADS * HEAD_DIM;

    // --- Compile ---
    println!("  Compiling TurboQuant sub-programs...");
    let compile_start = Instant::now();
    let config = TurboQuantConfig::new(
        8,
        MAX_SEQ_LEN,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        NUM_LAYERS,
    )
    .expect("invalid config");

    let mut model = match TurboQuantModel::compile(config) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("  ✗ Compilation failed: {e}");
            eprintln!("    (This requires macOS with ANE hardware)");
            std::process::exit(1);
        }
    };
    let compile_time = compile_start.elapsed();
    println!(
        "  ✓ Compiled in {:.1}ms",
        compile_time.as_secs_f64() * 1000.0
    );
    println!();

    // --- Create dummy tensors with uniform alloc (ANE requirement) ---
    let (cw_alloc, attn_alloc) = model.alloc_sizes();

    let mut q = AneTensor::new_with_min_alloc(q_ch, 1, ScalarType::Float16, attn_alloc).unwrap();
    let mut k_proj =
        AneTensor::new_with_min_alloc(kv_ch, 1, ScalarType::Float16, cw_alloc).unwrap();
    let mut v_proj =
        AneTensor::new_with_min_alloc(kv_ch, 1, ScalarType::Float16, cw_alloc).unwrap();

    // Fill with small random-ish values
    let q_data: Vec<f16> = (0..q_ch)
        .map(|i| f16::from_f32((i as f32 - 448.0) * 0.01))
        .collect();
    let kv_data: Vec<f16> = (0..kv_ch)
        .map(|i| f16::from_f32((i as f32 - 64.0) * 0.02))
        .collect();
    q.write_f16(&q_data).unwrap();
    k_proj.write_f16(&kv_data).unwrap();
    v_proj.write_f16(&kv_data).unwrap();

    // --- Benchmark single-layer step_attention ---
    println!("  Benchmarking single-layer step_attention...");
    println!("    Q: [{q_ch}] fp16, K/V proj: [{kv_ch}] fp16");
    println!("    Warmup: {WARMUP}, Iterations: {ITERATIONS}");
    println!();

    // Warmup
    for _ in 0..WARMUP {
        model.reset();
        let _ = model.step_attention(0, &q, &k_proj, &v_proj);
        // Must advance seq_pos after step_attention
        // (we're testing single-layer so manually advance)
    }

    // Timed iterations (single layer, fresh cache each time)
    let mut latencies_us = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        model.reset();
        let start = Instant::now();
        let _out = model.step_attention(0, &q, &k_proj, &v_proj).unwrap();
        latencies_us.push(start.elapsed().as_micros() as f64);
    }

    let mean = latencies_us.iter().sum::<f64>() / latencies_us.len() as f64;
    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies_us[latencies_us.len() / 2];
    let p99 = latencies_us[(latencies_us.len() as f64 * 0.99) as usize];
    let min = latencies_us[0];
    let max = latencies_us[latencies_us.len() - 1];

    println!("  Single-layer step_attention (cache-write + attention):");
    println!("    mean:  {mean:.0} μs");
    println!("    p50:   {p50:.0} μs");
    println!("    p99:   {p99:.0} μs");
    println!("    min:   {min:.0} μs");
    println!("    max:   {max:.0} μs");
    println!();

    // --- Benchmark full model step (all layers) ---
    println!("  Benchmarking full-model step ({NUM_LAYERS} layers)...");

    // Create per-layer projections
    let projections: Vec<(AneTensor, AneTensor, AneTensor)> = (0..NUM_LAYERS)
        .map(|_| {
            let mut q =
                AneTensor::new_with_min_alloc(q_ch, 1, ScalarType::Float16, attn_alloc).unwrap();
            let mut k =
                AneTensor::new_with_min_alloc(kv_ch, 1, ScalarType::Float16, cw_alloc).unwrap();
            let mut v =
                AneTensor::new_with_min_alloc(kv_ch, 1, ScalarType::Float16, cw_alloc).unwrap();
            q.write_f16(&q_data).unwrap();
            k.write_f16(&kv_data).unwrap();
            v.write_f16(&kv_data).unwrap();
            (q, k, v)
        })
        .collect();

    // Warmup
    for _ in 0..WARMUP {
        model.reset();
        let _ = model.step(&projections);
    }

    // Timed iterations
    let mut full_latencies_us = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        model.reset();
        let start = Instant::now();
        let _outputs = model.step(&projections).unwrap();
        full_latencies_us.push(start.elapsed().as_micros() as f64);
    }

    let full_mean = full_latencies_us.iter().sum::<f64>() / full_latencies_us.len() as f64;
    full_latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let full_p50 = full_latencies_us[full_latencies_us.len() / 2];
    let full_p99 = full_latencies_us[(full_latencies_us.len() as f64 * 0.99) as usize];
    let full_min = full_latencies_us[0];
    let full_max = full_latencies_us[full_latencies_us.len() - 1];

    println!("  Full step ({NUM_LAYERS} layers, INT8 KV cache):");
    println!(
        "    mean:  {:.2} ms ({:.0} μs/layer)",
        full_mean / 1000.0,
        full_mean / NUM_LAYERS as f64
    );
    println!("    p50:   {:.2} ms", full_p50 / 1000.0);
    println!("    p99:   {:.2} ms", full_p99 / 1000.0);
    println!("    min:   {:.2} ms", full_min / 1000.0);
    println!("    max:   {:.2} ms", full_max / 1000.0);
    println!();

    // --- Summary ---
    let cache_bytes = 2 * NUM_LAYERS * kv_ch * MAX_SEQ_LEN; // INT8: 1 byte/elem, K+V
    let cache_fp16_bytes = cache_bytes * 2;
    println!("  ═══════════════════════════════════════════════════════");
    println!("  Summary (Qwen3-0.6B, seq_len={MAX_SEQ_LEN}):");
    println!(
        "    KV cache (INT8):  {:.1} MB",
        cache_bytes as f64 / 1024.0 / 1024.0
    );
    println!(
        "    KV cache (FP16):  {:.1} MB (baseline)",
        cache_fp16_bytes as f64 / 1024.0 / 1024.0
    );
    println!(
        "    Bandwidth saved:  {:.1} MB ({:.0}%)",
        (cache_fp16_bytes - cache_bytes) as f64 / 1024.0 / 1024.0,
        (1.0 - cache_bytes as f64 / cache_fp16_bytes as f64) * 100.0
    );
    println!(
        "    Compile time:     {:.1} ms",
        compile_time.as_secs_f64() * 1000.0
    );
    println!("    Per-token (all layers): {:.2} ms", full_mean / 1000.0);
    println!(
        "    Per-layer:        {:.0} μs",
        full_mean / NUM_LAYERS as f64
    );
}
