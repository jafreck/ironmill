//! Metal kernel bandwidth micro-benchmark.
//!
//! Measures per-kernel GPU bandwidth for INT4 quantized compute kernels
//! with synthetic data. No model weights needed.
//!
//! Answers the critical questions from the profiling infrastructure design:
//! - Q1: Does projection bandwidth scale with N?
//! - Q2: Does projection bandwidth scale with K?
//! - Q3: What is FFN gate+up bandwidth vs FFN down bandwidth?
//!
//! Run with:
//!   cargo run -p ironmill-bench --features metal --example kernel_bandwidth_bench --release
//!
//! Sweep modes:
//!   --sweep-n 256,512,1024,2048,4096,8192,16384 --k 2560
//!   --sweep-k 256,512,1024,2560,4096,9216 --n 4096
//!   --compare-kernels --n 4096 --k 2560
//!   --json  (machine-parseable output)

#[cfg(not(feature = "metal"))]
fn main() {
    eprintln!("This benchmark requires the 'metal' feature.");
    std::process::exit(1);
}

#[cfg(feature = "metal")]
fn main() {
    use half::f16;
    use ironmill_inference::metal::ops::MetalPipelines;
    use ironmill_inference::metal::weights::AffineQuantizedWeight;
    use ironmill_metal_sys::{MetalDevice, StorageMode};

    // ── CLI argument parsing ─────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();

    let mut sweep_n: Option<Vec<usize>> = None;
    let mut sweep_k: Option<Vec<usize>> = None;
    let mut fixed_n: usize = 4096;
    let mut fixed_k: usize = 2560;
    let mut group_size: u32 = 128;
    let mut compare_kernels = false;
    let mut json_output = false;
    let mut warmup: usize = 20;
    let mut iterations: usize = 100;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--sweep-n" => {
                i += 1;
                sweep_n = Some(args[i].split(',').map(|s| s.parse().unwrap()).collect());
            }
            "--sweep-k" => {
                i += 1;
                sweep_k = Some(args[i].split(',').map(|s| s.parse().unwrap()).collect());
            }
            "--n" => {
                i += 1;
                fixed_n = args[i].parse().unwrap();
            }
            "--k" => {
                i += 1;
                fixed_k = args[i].parse().unwrap();
            }
            "--gs" | "--group-size" => {
                i += 1;
                group_size = args[i].parse().unwrap();
            }
            "--compare-kernels" => compare_kernels = true,
            "--json" => json_output = true,
            "--warmup" => {
                i += 1;
                warmup = args[i].parse().unwrap();
            }
            "--iterations" => {
                i += 1;
                iterations = args[i].parse().unwrap();
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: kernel_bandwidth_bench [OPTIONS]\n\n\
                     Options:\n\
                       --sweep-n N1,N2,...   Sweep output dimension N at fixed K\n\
                       --sweep-k K1,K2,...   Sweep input dimension K at fixed N\n\
                       --n N                 Fixed N (default: 4096)\n\
                       --k K                 Fixed K (default: 2560)\n\
                       --gs GS               Group size (default: 128)\n\
                       --compare-kernels     Compare all kernel variants at fixed N,K\n\
                       --json                JSON output\n\
                       --warmup W            Warmup iterations (default: 20)\n\
                       --iterations I        Measured iterations (default: 100)"
                );
                return;
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // Default: if nothing specified, run a standard comparison
    if sweep_n.is_none() && sweep_k.is_none() && !compare_kernels {
        compare_kernels = true;
    }

    // ── Metal setup ──────────────────────────────────────────────
    let device = MetalDevice::system_default().expect("no Metal device");
    let queue = device.create_command_queue().expect("command queue");
    let device_name = device.name();

    // Compile pipelines (head_dim doesn't matter for projection kernels)
    let pipelines = MetalPipelines::compile(&device, 128, 128).expect("compile pipelines");

    if !json_output {
        eprintln!("╔═══════════════════════════════════════════════════════════╗");
        eprintln!("║      Metal Kernel Bandwidth Micro-Benchmark              ║");
        eprintln!("╠═══════════════════════════════════════════════════════════╣");
        eprintln!(
            "║  Device: {:<48}║",
            &device_name[..device_name.len().min(48)]
        );
        eprintln!("║  GS: {group_size:<4}  Warmup: {warmup:<4}  Iterations: {iterations:<14}║");
        eprintln!("╚═══════════════════════════════════════════════════════════╝");
        eprintln!();
    }

    // ── Measurement helper ───────────────────────────────────────
    struct BenchResult {
        kernel: String,
        n: usize,
        k: usize,
        gpu_us: f64,
        total_bytes: usize,
        bw_gbs: f64,
        peak_pct: f64,
        gflops: f64,
    }

    // Assume ~400 GB/s peak for M-series (M2 Max: 400, M1 Max: 400, M3 Max: 400,
    // M4 Max: 546). User can override with env var.
    let peak_bw_gbs: f64 = std::env::var("PEAK_BW_GBS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(400.0);

    let mut all_results: Vec<BenchResult> = Vec::new();

    // ── Synthetic weight + buffer creation ───────────────────────
    use std::cell::RefCell;
    let rng_state = RefCell::new(42u64);
    let next_rand = || -> u8 {
        let mut s = rng_state.borrow_mut();
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        (*s & 0xFF) as u8
    };

    let create_weight =
        |device: &MetalDevice, n: usize, k: usize, gs: u32| -> AffineQuantizedWeight {
            let num_groups = k.div_ceil(gs as usize);
            // INT4: 2 elements per byte
            let data_bytes = n * k / 2;
            let scale_zero_count = n * num_groups;

            let data: Vec<u8> = (0..data_bytes).map(|_| next_rand()).collect();
            let scales: Vec<u8> = (0..scale_zero_count)
                .flat_map(|_| f16::from_f32(0.01).to_le_bytes())
                .collect();
            let zeros: Vec<u8> = (0..scale_zero_count)
                .flat_map(|_| f16::from_f32(8.0).to_le_bytes())
                .collect();

            let data_buf = device
                .create_buffer_with_data(&data, StorageMode::Shared)
                .expect("weight data");
            let scales_buf = device
                .create_buffer_with_data(&scales, StorageMode::Shared)
                .expect("scales");
            let zeros_buf = device
                .create_buffer_with_data(&zeros, StorageMode::Shared)
                .expect("zeros");

            AffineQuantizedWeight {
                data: data_buf,
                scales: Some(scales_buf),
                zeros: Some(zeros_buf),
                group_size: gs,
                bit_width: 4,
                shape: (n, k),
                awq_scales: None,
            }
        };

    // Benchmark a single matvec kernel dispatch
    let bench_matvec = |device: &MetalDevice, n: usize, k: usize, gs: u32| -> f64 {
        let weight = create_weight(device, n, k, gs);
        let input_data: Vec<u8> = (0..k * 2).map(|i| (i % 256) as u8).collect();
        let input = device
            .create_buffer_with_data(&input_data, StorageMode::Shared)
            .expect("input");
        let output = device
            .create_buffer(n * 2, StorageMode::Shared)
            .expect("output");

        let pipeline = pipelines
            .affine
            .matvec_int4
            .get(gs)
            .expect("no pipeline for group_size");

        let mut times = Vec::with_capacity(warmup + iterations);
        for _ in 0..(warmup + iterations) {
            let cmd = queue.command_buffer().expect("cmd");
            let enc = cmd.compute_encoder().expect("enc");

            enc.set_pipeline(pipeline);
            enc.set_buffer(&input, 0, 0);
            enc.set_buffer(&weight.data, 0, 1);
            enc.set_buffer(&output, 0, 2);
            enc.set_bytes(&(n as u32).to_le_bytes(), 3);
            enc.set_bytes(&(k as u32).to_le_bytes(), 4);
            // No AWQ
            enc.set_buffer(&weight.data, 0, 5); // dummy
            enc.set_bytes(&0u32.to_le_bytes(), 6);
            enc.set_buffer(weight.scales.as_ref().unwrap(), 0, 7);
            enc.set_buffer(weight.zeros.as_ref().unwrap(), 0, 8);
            enc.dispatch_threadgroups((n.div_ceil(8), 1, 1), (64, 1, 1));

            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            let gpu_us = (cmd.gpu_end_time() - cmd.gpu_start_time()) * 1_000_000.0;
            times.push(gpu_us);
        }
        times.drain(..warmup);
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times[times.len() / 2]
    };

    // Benchmark fused FFN gate+up+activation kernel
    let bench_fused_ffn = |device: &MetalDevice, n: usize, k: usize, gs: u32| -> f64 {
        let gate_weight = create_weight(device, n, k, gs);
        let up_weight = create_weight(device, n, k, gs);
        let input_data: Vec<u8> = (0..k * 2).map(|i| (i % 256) as u8).collect();
        let input = device
            .create_buffer_with_data(&input_data, StorageMode::Shared)
            .expect("input");
        let output = device
            .create_buffer(n * 2, StorageMode::Shared)
            .expect("output");

        let pipeline = pipelines
            .affine
            .fused_ffn_gate_up_act_int4
            .get(gs)
            .expect("no fused_ffn pipeline");

        let mut times = Vec::with_capacity(warmup + iterations);
        for _ in 0..(warmup + iterations) {
            let cmd = queue.command_buffer().expect("cmd");
            let enc = cmd.compute_encoder().expect("enc");

            enc.set_pipeline(pipeline);
            enc.set_buffer(&input, 0, 0);
            enc.set_buffer(&gate_weight.data, 0, 1);
            enc.set_buffer(&up_weight.data, 0, 2);
            enc.set_buffer(&output, 0, 3);
            enc.set_bytes(&(n as u32).to_le_bytes(), 4);
            enc.set_bytes(&(k as u32).to_le_bytes(), 5);
            // No AWQ
            enc.set_buffer(&gate_weight.data, 0, 6); // dummy
            enc.set_bytes(&0u32.to_le_bytes(), 7);
            enc.set_bytes(&0u32.to_le_bytes(), 8); // use_gelu=false (SiLU)
            enc.set_buffer(gate_weight.scales.as_ref().unwrap(), 0, 9);
            enc.set_buffer(gate_weight.zeros.as_ref().unwrap(), 0, 10);
            enc.set_buffer(up_weight.scales.as_ref().unwrap(), 0, 11);
            enc.set_buffer(up_weight.zeros.as_ref().unwrap(), 0, 12);
            enc.dispatch_threadgroups((n, 1, 1), (32, 1, 1));

            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            let gpu_us = (cmd.gpu_end_time() - cmd.gpu_start_time()) * 1_000_000.0;
            times.push(gpu_us);
        }
        times.drain(..warmup);
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times[times.len() / 2]
    };

    // Benchmark batched FFN gate+up kernel (separate gate+up, no activation)
    let bench_batched_matvec = |device: &MetalDevice, n: usize, k: usize, gs: u32| -> f64 {
        let gate_weight = create_weight(device, n, k, gs);
        let up_weight = create_weight(device, n, k, gs);
        let input_data: Vec<u8> = (0..k * 2).map(|i| (i % 256) as u8).collect();
        let input = device
            .create_buffer_with_data(&input_data, StorageMode::Shared)
            .expect("input");
        let gate_out = device
            .create_buffer(n * 2, StorageMode::Shared)
            .expect("gate_out");
        let up_out = device
            .create_buffer(n * 2, StorageMode::Shared)
            .expect("up_out");

        let pipeline = pipelines
            .affine
            .batched_matvec_int4
            .get(gs)
            .expect("no batched_matvec pipeline");

        let mut times = Vec::with_capacity(warmup + iterations);
        for _ in 0..(warmup + iterations) {
            let cmd = queue.command_buffer().expect("cmd");
            let enc = cmd.compute_encoder().expect("enc");

            enc.set_pipeline(pipeline);
            enc.set_buffer(&input, 0, 0);
            enc.set_buffer(&gate_weight.data, 0, 1);
            enc.set_buffer(&gate_out, 0, 2);
            enc.set_buffer(&up_weight.data, 0, 3);
            enc.set_buffer(&up_out, 0, 4);
            enc.set_bytes(&(n as u32).to_le_bytes(), 5);
            enc.set_bytes(&(k as u32).to_le_bytes(), 6);
            // No AWQ
            enc.set_buffer(&gate_weight.data, 0, 7); // dummy
            enc.set_bytes(&0u32.to_le_bytes(), 8);
            enc.set_buffer(gate_weight.scales.as_ref().unwrap(), 0, 9);
            enc.set_buffer(gate_weight.zeros.as_ref().unwrap(), 0, 10);
            enc.set_buffer(up_weight.scales.as_ref().unwrap(), 0, 11);
            enc.set_buffer(up_weight.zeros.as_ref().unwrap(), 0, 12);
            let tg_count = 2 * n;
            enc.dispatch_threadgroups((tg_count, 1, 1), (32, 1, 1));

            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            let gpu_us = (cmd.gpu_end_time() - cmd.gpu_start_time()) * 1_000_000.0;
            times.push(gpu_us);
        }
        times.drain(..warmup);
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times[times.len() / 2]
    };

    // ── Bandwidth calculation ────────────────────────────────────
    /// Calculate effective bandwidth for a single matvec (INT4 weight read + FP16 input + FP16 output).
    fn matvec_bytes(n: usize, k: usize, gs: u32) -> usize {
        let num_groups = k.div_ceil(gs as usize);
        let weight_bytes = n * k / 2; // INT4 packed
        let scale_bytes = n * num_groups * 2; // FP16 scales
        let zero_bytes = n * num_groups * 2; // FP16 zeros
        let input_bytes = k * 2; // FP16 input
        let output_bytes = n * 2; // FP16 output
        weight_bytes + scale_bytes + zero_bytes + input_bytes + output_bytes
    }

    /// For fused/batched: 2 weight matrices read, 1 input, outputs vary.
    fn dual_matvec_bytes(n: usize, k: usize, gs: u32, num_outputs: usize) -> usize {
        let num_groups = k.div_ceil(gs as usize);
        let weight_bytes = 2 * n * k / 2;
        let scale_bytes = 2 * n * num_groups * 2;
        let zero_bytes = 2 * n * num_groups * 2;
        let input_bytes = k * 2;
        let output_bytes = num_outputs * n * 2;
        weight_bytes + scale_bytes + zero_bytes + input_bytes + output_bytes
    }

    let compute_result =
        |kernel: &str, n: usize, k: usize, total_bytes: usize, gpu_us: f64| -> BenchResult {
            let gpu_s = gpu_us / 1_000_000.0;
            let bw_gbs = total_bytes as f64 / gpu_s / 1e9;
            let peak_pct = bw_gbs / peak_bw_gbs * 100.0;
            // 2 FLOPs per multiply-add
            let flops = 2.0 * n as f64 * k as f64;
            let gflops = flops / gpu_s / 1e9;
            BenchResult {
                kernel: kernel.to_string(),
                n,
                k,
                gpu_us,
                total_bytes,
                bw_gbs,
                peak_pct,
                gflops,
            }
        };

    // ── Run sweeps ───────────────────────────────────────────────

    // Sweep N at fixed K
    if let Some(ref n_values) = sweep_n {
        if !json_output {
            eprintln!("── Sweep N (K={fixed_k}, GS={group_size}) ─────────────────────────");
            eprintln!(
                "  {:>8} {:>8} {:>12} {:>10} {:>10} {:>8} {:>8}",
                "N", "K", "kernel", "GPU (µs)", "BW (GB/s)", "% peak", "GFLOP/s"
            );
        }
        for &n in n_values {
            let gpu_us = bench_matvec(&device, n, fixed_k, group_size);
            let total_bytes = matvec_bytes(n, fixed_k, group_size);
            let r = compute_result("matvec_int4", n, fixed_k, total_bytes, gpu_us);
            if !json_output {
                eprintln!(
                    "  {:>8} {:>8} {:>12} {:>10.1} {:>10.1} {:>7.1}% {:>8.1}",
                    r.n, r.k, r.kernel, r.gpu_us, r.bw_gbs, r.peak_pct, r.gflops
                );
            }
            all_results.push(r);
        }
        if !json_output {
            eprintln!();
        }
    }

    // Sweep K at fixed N
    if let Some(ref k_values) = sweep_k {
        if !json_output {
            eprintln!("── Sweep K (N={fixed_n}, GS={group_size}) ─────────────────────────");
            eprintln!(
                "  {:>8} {:>8} {:>12} {:>10} {:>10} {:>8} {:>8}",
                "N", "K", "kernel", "GPU (µs)", "BW (GB/s)", "% peak", "GFLOP/s"
            );
        }
        for &k in k_values {
            let gpu_us = bench_matvec(&device, fixed_n, k, group_size);
            let total_bytes = matvec_bytes(fixed_n, k, group_size);
            let r = compute_result("matvec_int4", fixed_n, k, total_bytes, gpu_us);
            if !json_output {
                eprintln!(
                    "  {:>8} {:>8} {:>12} {:>10.1} {:>10.1} {:>7.1}% {:>8.1}",
                    r.n, r.k, r.kernel, r.gpu_us, r.bw_gbs, r.peak_pct, r.gflops
                );
            }
            all_results.push(r);
        }
        if !json_output {
            eprintln!();
        }
    }

    // Compare all kernel variants at fixed N, K
    if compare_kernels {
        if !json_output {
            eprintln!("── Compare kernels (N={fixed_n}, K={fixed_k}, GS={group_size}) ──────────");
            eprintln!(
                "  {:>24} {:>10} {:>10} {:>8} {:>8}",
                "kernel", "GPU (µs)", "BW (GB/s)", "% peak", "GFLOP/s"
            );
        }

        // 1. superblock_matvec_int4
        {
            let gpu_us = bench_matvec(&device, fixed_n, fixed_k, group_size);
            let total = matvec_bytes(fixed_n, fixed_k, group_size);
            let r = compute_result("matvec_int4", fixed_n, fixed_k, total, gpu_us);
            if !json_output {
                eprintln!(
                    "  {:>24} {:>10.1} {:>10.1} {:>7.1}% {:>8.1}",
                    r.kernel, r.gpu_us, r.bw_gbs, r.peak_pct, r.gflops
                );
            }
            all_results.push(r);
        }

        // 2. fused_ffn_gate_up_act_int4
        {
            let gpu_us = bench_fused_ffn(&device, fixed_n, fixed_k, group_size);
            let total = dual_matvec_bytes(fixed_n, fixed_k, group_size, 1);
            let r = compute_result("fused_ffn_gate_up_act", fixed_n, fixed_k, total, gpu_us);
            if !json_output {
                eprintln!(
                    "  {:>24} {:>10.1} {:>10.1} {:>7.1}% {:>8.1}",
                    r.kernel, r.gpu_us, r.bw_gbs, r.peak_pct, r.gflops
                );
            }
            all_results.push(r);
        }

        // 3. batched_matvec_int4 (gate+up)
        {
            let gpu_us = bench_batched_matvec(&device, fixed_n, fixed_k, group_size);
            let total = dual_matvec_bytes(fixed_n, fixed_k, group_size, 2);
            let r = compute_result("batched_matvec_int4", fixed_n, fixed_k, total, gpu_us);
            if !json_output {
                eprintln!(
                    "  {:>24} {:>10.1} {:>10.1} {:>7.1}% {:>8.1}",
                    r.kernel, r.gpu_us, r.bw_gbs, r.peak_pct, r.gflops
                );
            }
            all_results.push(r);
        }

        // 4. matvec_int4 at typical FFN dimensions (N=inter, K=hidden) vs (N=hidden, K=inter)
        // This answers Q3: FFN gate+up BW vs FFN down BW
        let inter = fixed_n * 4; // typical intermediate = 4× hidden
        {
            let gpu_us = bench_matvec(&device, inter, fixed_k, group_size);
            let total = matvec_bytes(inter, fixed_k, group_size);
            let r = compute_result("matvec (gate_up N)", inter, fixed_k, total, gpu_us);
            if !json_output {
                eprintln!(
                    "  {:>24} {:>10.1} {:>10.1} {:>7.1}% {:>8.1}  [N={}, K={}]",
                    r.kernel, r.gpu_us, r.bw_gbs, r.peak_pct, r.gflops, inter, fixed_k
                );
            }
            all_results.push(r);
        }
        {
            let gpu_us = bench_matvec(&device, fixed_n, inter, group_size);
            let total = matvec_bytes(fixed_n, inter, group_size);
            let r = compute_result("matvec (down N)", fixed_n, inter, total, gpu_us);
            if !json_output {
                eprintln!(
                    "  {:>24} {:>10.1} {:>10.1} {:>7.1}% {:>8.1}  [N={}, K={}]",
                    r.kernel, r.gpu_us, r.bw_gbs, r.peak_pct, r.gflops, fixed_n, inter
                );
            }
            all_results.push(r);
        }

        // 5. Batched QKV: 3 projections in one dispatch vs 3 separate matvecs
        // Simulates GQA: Q has full N, K and V have N/4
        {
            let n_q = fixed_n;
            let n_kv = fixed_n / 4; // GQA: num_kv_heads = num_heads/4
            // 3 separate dispatches (baseline)
            let gpu_q = bench_matvec(&device, n_q, fixed_k, group_size);
            let gpu_k = bench_matvec(&device, n_kv, fixed_k, group_size);
            let gpu_v = bench_matvec(&device, n_kv, fixed_k, group_size);
            let sep_total_us = gpu_q + gpu_k + gpu_v;
            let sep_bytes = matvec_bytes(n_q, fixed_k, group_size)
                + matvec_bytes(n_kv, fixed_k, group_size) * 2;
            let r_sep = compute_result("3x separate Q/K/V", n_q, fixed_k, sep_bytes, sep_total_us);
            if !json_output {
                eprintln!(
                    "  {:>24} {:>10.1} {:>10.1} {:>7.1}% {:>8.1}  [Q:N={}, KV:N={}]",
                    r_sep.kernel,
                    r_sep.gpu_us,
                    r_sep.bw_gbs,
                    r_sep.peak_pct,
                    r_sep.gflops,
                    n_q,
                    n_kv,
                );
            }
            let sep_gpu_us = r_sep.gpu_us;
            all_results.push(r_sep);

            // Batched QKV dispatch
            let q_weight = create_weight(&device, n_q, fixed_k, group_size);
            let k_weight = create_weight(&device, n_kv, fixed_k, group_size);
            let v_weight = create_weight(&device, n_kv, fixed_k, group_size);
            let input_data: Vec<u8> = (0..fixed_k * 2).map(|i| (i % 256) as u8).collect();
            let input = device
                .create_buffer_with_data(&input_data, StorageMode::Shared)
                .expect("input");
            let q_out = device
                .create_buffer(n_q * 2, StorageMode::Shared)
                .expect("q_out");
            let k_out = device
                .create_buffer(n_kv * 2, StorageMode::Shared)
                .expect("k_out");
            let v_out = device
                .create_buffer(n_kv * 2, StorageMode::Shared)
                .expect("v_out");

            let qkv_pipeline = pipelines
                .affine
                .batched_qkv_matvec_int4
                .get(group_size)
                .expect("no batched_qkv pipeline");

            let mut qkv_times = Vec::with_capacity(warmup + iterations);
            for _ in 0..(warmup + iterations) {
                let cmd = queue.command_buffer().expect("cmd");
                let enc = cmd.compute_encoder().expect("enc");

                enc.set_pipeline(qkv_pipeline);
                enc.set_buffer(&input, 0, 0);
                enc.set_buffer(&q_weight.data, 0, 1);
                enc.set_buffer(&q_out, 0, 2);
                enc.set_buffer(&k_weight.data, 0, 3);
                enc.set_buffer(&k_out, 0, 4);
                enc.set_buffer(&v_weight.data, 0, 5);
                enc.set_buffer(&v_out, 0, 6);
                let has_awq = 0u32;
                let params: [u32; 5] = [
                    n_q as u32,
                    n_kv as u32,
                    n_kv as u32,
                    fixed_k as u32,
                    has_awq,
                ];
                let params_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
                enc.set_bytes(&params_bytes, 7);
                enc.set_buffer(&q_weight.data, 0, 8); // dummy AWQ
                enc.set_buffer(q_weight.scales.as_ref().unwrap(), 0, 9);
                enc.set_buffer(q_weight.zeros.as_ref().unwrap(), 0, 10);
                enc.set_buffer(k_weight.scales.as_ref().unwrap(), 0, 11);
                enc.set_buffer(k_weight.zeros.as_ref().unwrap(), 0, 12);
                enc.set_buffer(v_weight.scales.as_ref().unwrap(), 0, 13);
                enc.set_buffer(v_weight.zeros.as_ref().unwrap(), 0, 14);
                let tg_count = n_q.div_ceil(8) + n_kv.div_ceil(8) + n_kv.div_ceil(8);
                enc.dispatch_threadgroups((tg_count, 1, 1), (64, 1, 1));

                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                let gpu_us = (cmd.gpu_end_time() - cmd.gpu_start_time()) * 1_000_000.0;
                qkv_times.push(gpu_us);
            }
            qkv_times.drain(..warmup);
            qkv_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let qkv_median = qkv_times[qkv_times.len() / 2];

            let r_batch = compute_result("batched QKV", n_q, fixed_k, sep_bytes, qkv_median);
            if !json_output {
                let speedup = sep_gpu_us / r_batch.gpu_us;
                eprintln!(
                    "  {:>24} {:>10.1} {:>10.1} {:>7.1}% {:>8.1}  [{:.2}× vs separate]",
                    r_batch.kernel,
                    r_batch.gpu_us,
                    r_batch.bw_gbs,
                    r_batch.peak_pct,
                    r_batch.gflops,
                    speedup,
                );
            }
            all_results.push(r_batch);
        }

        if !json_output {
            eprintln!();
        }
    }

    // ── JSON output ──────────────────────────────────────────────
    if json_output {
        let results: Vec<serde_json::Value> = all_results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "kernel": r.kernel,
                    "n": r.n,
                    "k": r.k,
                    "group_size": group_size,
                    "gpu_us": (r.gpu_us * 10.0).round() / 10.0,
                    "total_bytes": r.total_bytes,
                    "bw_gbs": (r.bw_gbs * 10.0).round() / 10.0,
                    "peak_pct": (r.peak_pct * 10.0).round() / 10.0,
                    "gflops": (r.gflops * 10.0).round() / 10.0,
                })
            })
            .collect();

        let output = serde_json::json!({
            "benchmark": "kernel_bandwidth",
            "hardware": device_name,
            "peak_bw_gbs": peak_bw_gbs,
            "warmup": warmup,
            "iterations": iterations,
            "results": results,
        });
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
    } else {
        // Summary analysis
        if all_results.len() >= 2 {
            eprintln!("── Analysis ──────────────────────────────────────────────");
            let max_bw = all_results.iter().map(|r| r.bw_gbs).fold(0.0f64, f64::max);
            let min_bw = all_results
                .iter()
                .map(|r| r.bw_gbs)
                .fold(f64::MAX, f64::min);
            eprintln!("  BW range: {min_bw:.1} – {max_bw:.1} GB/s");
            eprintln!(
                "  Best utilization: {:.1}% of {peak_bw_gbs:.0} GB/s peak",
                max_bw / peak_bw_gbs * 100.0
            );

            // N-scaling analysis
            if sweep_n.is_some() {
                let n_results: Vec<&BenchResult> = all_results
                    .iter()
                    .filter(|r| r.kernel == "matvec_int4")
                    .collect();
                if n_results.len() >= 2 {
                    let first = n_results.first().unwrap();
                    let last = n_results.last().unwrap();
                    let ratio = last.bw_gbs / first.bw_gbs;
                    eprintln!(
                        "  N scaling: {:.1}× BW increase from N={} to N={} (BW {} → {} GB/s)",
                        ratio, first.n, last.n, first.bw_gbs as u64, last.bw_gbs as u64,
                    );
                    if ratio > 1.5 {
                        eprintln!(
                            "  → BW scales strongly with N. Small-N projections are memory-latency bound."
                        );
                        eprintln!("  → Concatenating projections (larger effective N) would help.");
                    } else {
                        eprintln!(
                            "  → BW is relatively flat across N. Bottleneck is per-TG, not total TG count."
                        );
                    }
                }
            }
        }
    }
}
