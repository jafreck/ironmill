//! FA2 vs fused SDPA prefill attention kernel benchmark.
//!
//! Measures raw kernel throughput at various sequence lengths and model
//! dimensions to quantify the speedup from FlashAttention-2 style
//! multi-query tiling vs the register-tiled fused SDPA kernel.
//!
//! No model weights needed — uses synthetic random data.
//!
//! Run with: cargo run -p ironmill-bench --features metal --example fa2_prefill_bench --release

#[cfg(not(feature = "metal"))]
fn main() {
    eprintln!("This benchmark requires the 'metal' feature.");
    std::process::exit(1);
}

#[cfg(feature = "metal")]
fn main() {
    use half::f16;
    use ironmill_inference::metal::ops::{
        self, FusedSdpaParams, MetalPipelines, PrefillAttentionParams,
    };
    use ironmill_metal_sys::{MetalDevice, StorageMode};
    use std::time::Instant;

    const WARMUP: usize = 20;
    const ITERATIONS: usize = 100;

    struct BenchConfig {
        label: &'static str,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    }

    struct SeqConfig {
        token_count: usize,
        max_seq_len: usize,
    }

    let model_configs = vec![
        BenchConfig {
            label: "Qwen3-0.6B",
            num_q_heads: 16,
            num_kv_heads: 8,
            head_dim: 64,
        },
        BenchConfig {
            label: "Qwen3.5-4B",
            num_q_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
        },
    ];

    let seq_configs = vec![
        SeqConfig {
            token_count: 32,
            max_seq_len: 64,
        },
        SeqConfig {
            token_count: 128,
            max_seq_len: 256,
        },
        SeqConfig {
            token_count: 512,
            max_seq_len: 1024,
        },
        SeqConfig {
            token_count: 1024,
            max_seq_len: 2048,
        },
        SeqConfig {
            token_count: 2048,
            max_seq_len: 4096,
        },
    ];

    let device = MetalDevice::system_default().expect("no Metal device");
    let queue = device.create_command_queue().expect("command queue");

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║         FA2 vs Fused SDPA Prefill Attention Benchmark            ║");
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║  Warmup: {WARMUP}  ·  Iterations: {ITERATIONS}  ·  Median latency         ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // Pre-generate random data (largest needed size)
    let max_q_heads = model_configs.iter().map(|c| c.num_q_heads).max().unwrap();
    let max_kv_heads = model_configs.iter().map(|c| c.num_kv_heads).max().unwrap();
    let max_head_dim = model_configs.iter().map(|c| c.head_dim).max().unwrap();
    let max_tokens = seq_configs.iter().map(|c| c.token_count).max().unwrap();
    let max_seq = seq_configs.iter().map(|c| c.max_seq_len).max().unwrap();

    // Generate random FP16 data
    let mut rng_state = 42u64;
    let mut rand_f16 = |count: usize| -> Vec<u8> {
        (0..count)
            .flat_map(|_| {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                let v = (rng_state as f32 / u64::MAX as f32) * 2.0 - 1.0;
                f16::from_f32(v * 0.5).to_le_bytes()
            })
            .collect()
    };

    let q_data = rand_f16(max_tokens * max_q_heads * max_head_dim);
    let k_data = rand_f16(max_kv_heads * max_seq * max_head_dim);
    let v_data = rand_f16(max_kv_heads * max_seq * max_head_dim);

    let q_buf = device
        .create_buffer_with_data(&q_data, StorageMode::Shared)
        .expect("q buf");
    let k_buf = device
        .create_buffer_with_data(&k_data, StorageMode::Shared)
        .expect("k buf");
    let v_buf = device
        .create_buffer_with_data(&v_data, StorageMode::Shared)
        .expect("v buf");

    let out_size = max_tokens * max_q_heads * max_head_dim * 2;
    let out_fa2 = device
        .create_buffer(out_size, StorageMode::Shared)
        .expect("out fa2");
    let out_sdpa = device
        .create_buffer(out_size, StorageMode::Shared)
        .expect("out sdpa");

    // Cache compiled pipelines per head_dim
    let mut pipeline_cache: std::collections::HashMap<usize, MetalPipelines> =
        std::collections::HashMap::new();

    for mc in &model_configs {
        println!(
            "── {} ({} Q heads, {} KV heads, hd={}) ──",
            mc.label, mc.num_q_heads, mc.num_kv_heads, mc.head_dim
        );
        println!(
            "  {:>8}  {:>10}  {:>10}  {:>10}  {:>8}  {:>8}",
            "tokens", "V2 (µs)", "FA2 (µs)", "SDPA (µs)", "V2/SDPA", "V2/FA2"
        );

        let pipelines = pipeline_cache.entry(mc.head_dim).or_insert_with(|| {
            MetalPipelines::compile(&device, mc.head_dim, mc.head_dim).expect("compile pipelines")
        });

        let scale = 1.0 / (mc.head_dim as f32).sqrt();

        for sc in &seq_configs {
            let nh = mc.num_q_heads as u32;
            let nkv = mc.num_kv_heads as u32;
            let hd = mc.head_dim as u32;
            let tc = sc.token_count as u32;
            let ms = sc.max_seq_len as u32;

            // --- Benchmark FA2 ---
            let mut fa2_times = Vec::with_capacity(WARMUP + ITERATIONS);
            for i in 0..(WARMUP + ITERATIONS) {
                let cmd = queue.command_buffer().expect("cmd");
                let enc = cmd.compute_encoder().expect("enc");
                ops::encode_fa2_prefill_attention(
                    &enc,
                    &pipelines.prefill_attention_fa2,
                    &PrefillAttentionParams {
                        q: &q_buf,
                        k_cache: &k_buf,
                        v_cache: &v_buf,
                        output: &out_fa2,
                        num_heads: nh,
                        num_kv_heads: nkv,
                        head_dim: hd,
                        max_seq_len: ms,
                        seq_offset: 0,
                        token_count: tc,
                        window_size: 0,
                        attn_scale: scale,
                    },
                );
                enc.end_encoding();
                let t0 = Instant::now();
                cmd.commit();
                cmd.wait_until_completed();
                let elapsed = t0.elapsed();
                if i >= WARMUP {
                    fa2_times.push(elapsed.as_micros() as f64);
                }
            }

            // --- Benchmark fused SDPA ---
            let mut sdpa_times = Vec::with_capacity(WARMUP + ITERATIONS);
            for i in 0..(WARMUP + ITERATIONS) {
                let cmd = queue.command_buffer().expect("cmd");
                let enc = cmd.compute_encoder().expect("enc");
                ops::encode_fused_sdpa(
                    &enc,
                    &pipelines.fused_sdpa,
                    &FusedSdpaParams {
                        q: &q_buf,
                        k: &k_buf,
                        v: &v_buf,
                        output: &out_sdpa,
                        seq_len: tc, // total seq = token_count (fresh prefill)
                        token_count: tc,
                        head_dim: hd,
                        num_q_heads: nh,
                        num_kv_heads: nkv,
                        scale,
                        max_seq_len: ms,
                    },
                    None,
                );
                enc.end_encoding();
                let t0 = Instant::now();
                cmd.commit();
                cmd.wait_until_completed();
                let elapsed = t0.elapsed();
                if i >= WARMUP {
                    sdpa_times.push(elapsed.as_micros() as f64);
                }
            }

            // --- Benchmark V2 (register-tiled FA2) ---
            let mut v2_times = Vec::with_capacity(WARMUP + ITERATIONS);
            for i in 0..(WARMUP + ITERATIONS) {
                let cmd = queue.command_buffer().expect("cmd");
                let enc = cmd.compute_encoder().expect("enc");
                ops::encode_v2_prefill_attention(
                    &enc,
                    &pipelines.prefill_attention_v2,
                    &PrefillAttentionParams {
                        q: &q_buf,
                        k_cache: &k_buf,
                        v_cache: &v_buf,
                        output: &out_fa2,
                        num_heads: nh,
                        num_kv_heads: nkv,
                        head_dim: hd,
                        max_seq_len: ms,
                        seq_offset: 0,
                        token_count: tc,
                        window_size: 0,
                        attn_scale: scale,
                    },
                );
                enc.end_encoding();
                let t0 = Instant::now();
                cmd.commit();
                cmd.wait_until_completed();
                let elapsed = t0.elapsed();
                if i >= WARMUP {
                    v2_times.push(elapsed.as_micros() as f64);
                }
            }

            fa2_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sdpa_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v2_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let fa2_median = fa2_times[fa2_times.len() / 2];
            let sdpa_median = sdpa_times[sdpa_times.len() / 2];
            let v2_median = v2_times[v2_times.len() / 2];

            let fmt_speedup = |ratio: f64| -> String {
                if ratio > 1.05 {
                    format!("\x1b[32m{ratio:.2}×\x1b[0m")
                } else if ratio < 0.95 {
                    format!("\x1b[31m{ratio:.2}×\x1b[0m")
                } else {
                    format!("{ratio:.2}×")
                }
            };

            println!(
                "  {:>8}  {:>10.0}  {:>10.0}  {:>10.0}  {:>8}  {:>8}",
                sc.token_count,
                v2_median,
                fa2_median,
                sdpa_median,
                fmt_speedup(sdpa_median / v2_median),
                fmt_speedup(fa2_median / v2_median),
            );
        }
        println!();
    }
}
