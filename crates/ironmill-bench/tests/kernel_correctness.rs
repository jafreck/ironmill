//! Phase 3 Metal kernel correctness test.
//!
//! Verifies the polarquant_matvec_int4 kernel produces the same output as
//! a CPU reference implementation for a single matmul operation.

#[cfg(feature = "metal")]
#[test]
#[ignore]
fn polarquant_matvec_int4_correctness() {
    use half::f16;
    use ironmill_inference::gpu::ops::GpuPipelines;
    use ironmill_metal_sys::{MetalDevice, StorageMode};

    let device = MetalDevice::system_default().expect("no Metal device");
    let queue = device.create_command_queue().expect("command queue");
    let pipelines = GpuPipelines::compile(&device).expect("compile pipelines");

    // Small test case: A=[1, 8], B=[4, 8] → C=[1, 4]
    let m = 1usize;
    let n = 4usize;
    let k = 8usize;

    // Input activations A: [1, 8] as FP16
    let a_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let a_fp16: Vec<u8> = a_f32
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_le_bytes())
        .collect();

    // Weight matrix B: [4, 8] — we'll create known values
    // Row 0: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    // Row 1: [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    // Row 2: [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    // Row 3: [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5]
    let weights: Vec<Vec<f32>> = vec![
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        vec![-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5],
    ];

    // CPU reference: C[j] = sum_k(A[k] * B[j][k])
    let mut expected = vec![0.0f32; n];
    for j in 0..n {
        for ki in 0..k {
            expected[j] += a_f32[ki] * weights[j][ki];
        }
    }
    println!("Expected output: {:?}", expected);

    // Now quantize the weights to INT4 with per-row absmax + LUT
    // LUT: 16 symmetric levels in [-1, +1]
    let n_levels = 16usize;
    let half_n = n_levels as f32 / 2.0;
    let levels: Vec<f32> = (0..n_levels)
        .map(|i| (i as f32 + 0.5 - half_n) / half_n)
        .collect();
    let boundaries: Vec<f32> = levels.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();

    let lut_fp16: Vec<u8> = levels
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_le_bytes())
        .collect();

    // Per-row quantize
    let mut norms_f32 = vec![0.0f32; n];
    let mut packed_indices = Vec::new(); // [N, K/2] packed bytes

    for j in 0..n {
        let row = &weights[j];
        let absmax = row.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-10);
        norms_f32[j] = absmax;

        // Quantize this row to indices
        let mut row_indices = Vec::new();
        for &v in row {
            let normalized = (v / absmax).clamp(-1.0, 1.0);
            let idx = boundaries
                .iter()
                .position(|&b| normalized < b)
                .unwrap_or(n_levels - 1);
            row_indices.push(idx as u8);
        }

        // Pack 4-bit pairs into bytes (K/2 bytes per row)
        for pair in row_indices.chunks(2) {
            let lo = pair[0] & 0xF;
            let hi = if pair.len() > 1 { pair[1] & 0xF } else { 0 };
            packed_indices.push(lo | (hi << 4));
        }
    }

    let norms_fp16: Vec<u8> = norms_f32
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_le_bytes())
        .collect();

    // CPU dequant reference (to verify the quantization round-trip)
    let mut cpu_output = vec![0.0f32; n];
    for j in 0..n {
        let norm = norms_f32[j];
        for ki in 0..k {
            let byte_idx = j * (k / 2) + ki / 2;
            let packed = packed_indices[byte_idx];
            let idx = if ki % 2 == 0 {
                packed & 0xF
            } else {
                (packed >> 4) & 0xF
            };
            let w = levels[idx as usize] * norm;
            cpu_output[j] += a_f32[ki] * w;
        }
    }
    println!("CPU dequant output: {:?}", cpu_output);

    // Create Metal buffers
    let a_buf = device
        .create_buffer_with_data(&a_fp16, StorageMode::Shared)
        .expect("A buffer");
    let b_buf = device
        .create_buffer_with_data(&packed_indices, StorageMode::Shared)
        .expect("B buffer");
    let lut_buf = device
        .create_buffer_with_data(&lut_fp16, StorageMode::Shared)
        .expect("LUT buffer");
    let norms_buf = device
        .create_buffer_with_data(&norms_fp16, StorageMode::Shared)
        .expect("norms buffer");
    let c_buf = device
        .create_buffer(n * 2, StorageMode::Shared)
        .expect("C buffer");

    // Dispatch the kernel
    let cmd_buf = queue.command_buffer().expect("command buffer");
    let encoder = cmd_buf.compute_encoder().expect("encoder");

    encoder.set_pipeline(&pipelines.polarquant_matvec_int4);
    encoder.set_buffer(&a_buf, 0, 0);
    encoder.set_buffer(&b_buf, 0, 1);
    encoder.set_buffer(&lut_buf, 0, 2);
    encoder.set_buffer(&norms_buf, 0, 3);
    encoder.set_buffer(&c_buf, 0, 4);
    encoder.set_bytes(&(n as u32).to_le_bytes(), 5);
    encoder.set_bytes(&(k as u32).to_le_bytes(), 6);
    encoder.dispatch_threadgroups((n, 1, 1), (32, 1, 1));

    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    // Read back results
    let mut c_data = vec![0u8; n * 2];
    c_buf.read_bytes(&mut c_data, 0).expect("read C buffer");
    let mut gpu_output = vec![0.0f32; n];
    for i in 0..n {
        gpu_output[i] = f16::from_le_bytes([c_data[i * 2], c_data[i * 2 + 1]]).to_f32();
    }
    println!("GPU kernel output: {:?}", gpu_output);

    // Compare GPU vs CPU dequant reference
    for i in 0..n {
        let diff = (gpu_output[i] - cpu_output[i]).abs();
        assert!(
            diff < 0.5,
            "element {i}: GPU={:.4} CPU={:.4} diff={:.4}",
            gpu_output[i],
            cpu_output[i],
            diff
        );
    }

    println!("✅ Kernel produces correct output within tolerance");
}
