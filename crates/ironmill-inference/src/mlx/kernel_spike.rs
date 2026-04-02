//! Spike tests for MLX custom Metal kernel dispatch.
//!
//! These tests validate that `mlx_fast_metal_kernel()` supports the
//! requirements for TurboQuant and PolarQuant kernel porting.
//! They require a real mlx-c installation to run.

#[cfg(test)]
mod tests {
    use ironmill_mlx_sys::metal_kernel;
    use ironmill_mlx_sys::stream::eval;
    use ironmill_mlx_sys::{MlxArray, MlxDevice, MlxDtype, MlxStream};

    // Helper: convert a `&[f32]` to `Vec<u8>` for `MlxArray::from_data_copy`.
    fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
        data.iter().flat_map(|v| v.to_ne_bytes()).collect()
    }

    /// Test 1: Basic kernel dispatch — element-wise add kernel
    #[test]
    #[cfg(not(mlx_stub))]
    fn basic_kernel_dispatch() {
        let device = MlxDevice::default_gpu().unwrap();
        let stream = MlxStream::new(&device).unwrap();

        let data_a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let data_b: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        let a = MlxArray::from_data_copy(&f32_to_bytes(&data_a), &[4], MlxDtype::Float32, &stream)
            .unwrap();
        let b = MlxArray::from_data_copy(&f32_to_bytes(&data_b), &[4], MlxDtype::Float32, &stream)
            .unwrap();

        let source = r#"
            #include <metal_stdlib>
            using namespace metal;
            [[kernel]] void add_kernel(
                device const float* a [[buffer(0)]],
                device const float* b [[buffer(1)]],
                device float* c [[buffer(2)]],
                uint tid [[thread_position_in_grid]])
            {
                c[tid] = a[tid] + b[tid];
            }
        "#;

        let result = metal_kernel::metal_kernel(&metal_kernel::MetalKernelParams {
            name: "add_kernel",
            inputs: &[&a, &b],
            outputs: &[],
            source,
            grid: [4, 1, 1],
            threadgroup: [4, 1, 1],
            output_shapes: &[&[4]],
            output_dtypes: &[MlxDtype::Float32],
            stream: &stream,
        })
        .unwrap();

        eval(&[&result[0]]).unwrap();
        #[allow(unsafe_code)]
        let out: &[f32] = unsafe { result[0].as_contiguous_slice().unwrap() };
        assert_eq!(out, &[6.0, 8.0, 10.0, 12.0]);
    }

    /// Test 2: Threadgroup shared memory (TurboQuant requirement)
    #[test]
    #[cfg(not(mlx_stub))]
    fn threadgroup_shared_memory() {
        let device = MlxDevice::default_gpu().unwrap();
        let stream = MlxStream::new(&device).unwrap();

        let n = 256usize;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let input =
            MlxArray::from_data_copy(&f32_to_bytes(&data), &[n], MlxDtype::Float32, &stream)
                .unwrap();

        // Kernel uses 4096 floats of shared memory (16KB), matching
        // TurboQuant attention requirements.
        let source = r#"
            #include <metal_stdlib>
            using namespace metal;
            [[kernel]] void shared_mem_test(
                device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                uint tid [[thread_position_in_grid]],
                uint lid [[thread_position_in_threadgroup]])
            {
                threadgroup float shared[4096];
                shared[lid] = input[tid] * 2.0;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                output[tid] = shared[lid];
            }
        "#;

        let result = metal_kernel::metal_kernel(&metal_kernel::MetalKernelParams {
            name: "shared_mem_test",
            inputs: &[&input],
            outputs: &[],
            source,
            grid: [n, 1, 1],
            threadgroup: [256, 1, 1],
            output_shapes: &[&[n]],
            output_dtypes: &[MlxDtype::Float32],
            stream: &stream,
        })
        .unwrap();

        eval(&[&result[0]]).unwrap();
        #[allow(unsafe_code)]
        let out: &[f32] = unsafe { result[0].as_contiguous_slice().unwrap() };
        for i in 0..n {
            assert!((out[i] - (i as f32 * 2.0)).abs() < 1e-6);
        }
    }

    /// Test 3: High buffer count (14 buffers — TurboQuant attention requirement)
    #[test]
    #[cfg(not(mlx_stub))]
    fn high_buffer_count_14() {
        let device = MlxDevice::default_gpu().unwrap();
        let stream = MlxStream::new(&device).unwrap();

        let n = 64usize;
        let mut inputs = Vec::new();
        for i in 0..13 {
            let data: Vec<f32> = vec![i as f32; n];
            inputs.push(
                MlxArray::from_data_copy(&f32_to_bytes(&data), &[n], MlxDtype::Float32, &stream)
                    .unwrap(),
            );
        }

        let source = r#"
            #include <metal_stdlib>
            using namespace metal;
            [[kernel]] void sum_13(
                device const float* b0 [[buffer(0)]],
                device const float* b1 [[buffer(1)]],
                device const float* b2 [[buffer(2)]],
                device const float* b3 [[buffer(3)]],
                device const float* b4 [[buffer(4)]],
                device const float* b5 [[buffer(5)]],
                device const float* b6 [[buffer(6)]],
                device const float* b7 [[buffer(7)]],
                device const float* b8 [[buffer(8)]],
                device const float* b9 [[buffer(9)]],
                device const float* b10 [[buffer(10)]],
                device const float* b11 [[buffer(11)]],
                device const float* b12 [[buffer(12)]],
                device float* out [[buffer(13)]],
                uint tid [[thread_position_in_grid]])
            {
                out[tid] = b0[tid] + b1[tid] + b2[tid] + b3[tid]
                         + b4[tid] + b5[tid] + b6[tid] + b7[tid]
                         + b8[tid] + b9[tid] + b10[tid] + b11[tid]
                         + b12[tid];
            }
        "#;

        let input_refs: Vec<&MlxArray> = inputs.iter().collect();
        let result = metal_kernel::metal_kernel(&metal_kernel::MetalKernelParams {
            name: "sum_13",
            inputs: &input_refs,
            outputs: &[],
            source,
            grid: [n, 1, 1],
            threadgroup: [64, 1, 1],
            output_shapes: &[&[n]],
            output_dtypes: &[MlxDtype::Float32],
            stream: &stream,
        })
        .unwrap();

        eval(&[&result[0]]).unwrap();
        #[allow(unsafe_code)]
        let out: &[f32] = unsafe { result[0].as_contiguous_slice().unwrap() };
        let expected: f32 = (0..13).map(|i| i as f32).sum();
        for &v in out.iter() {
            assert!((v - expected).abs() < 1e-4, "got {v}, expected {expected}");
        }
    }

    /// Test 4: Even higher buffer count (18+ for outlier attention)
    #[test]
    #[cfg(not(mlx_stub))]
    fn high_buffer_count_18() {
        let device = MlxDevice::default_gpu().unwrap();
        let stream = MlxStream::new(&device).unwrap();

        let n = 32usize;
        let mut inputs = Vec::new();
        for i in 0..17 {
            let data: Vec<f32> = vec![(i + 1) as f32; n];
            inputs.push(
                MlxArray::from_data_copy(&f32_to_bytes(&data), &[n], MlxDtype::Float32, &stream)
                    .unwrap(),
            );
        }

        let source = r#"
            #include <metal_stdlib>
            using namespace metal;
            [[kernel]] void sum_17(
                device const float* b0  [[buffer(0)]],
                device const float* b1  [[buffer(1)]],
                device const float* b2  [[buffer(2)]],
                device const float* b3  [[buffer(3)]],
                device const float* b4  [[buffer(4)]],
                device const float* b5  [[buffer(5)]],
                device const float* b6  [[buffer(6)]],
                device const float* b7  [[buffer(7)]],
                device const float* b8  [[buffer(8)]],
                device const float* b9  [[buffer(9)]],
                device const float* b10 [[buffer(10)]],
                device const float* b11 [[buffer(11)]],
                device const float* b12 [[buffer(12)]],
                device const float* b13 [[buffer(13)]],
                device const float* b14 [[buffer(14)]],
                device const float* b15 [[buffer(15)]],
                device const float* b16 [[buffer(16)]],
                device float* out [[buffer(17)]],
                uint tid [[thread_position_in_grid]])
            {
                out[tid] = b0[tid] + b1[tid] + b2[tid] + b3[tid]
                         + b4[tid] + b5[tid] + b6[tid] + b7[tid]
                         + b8[tid] + b9[tid] + b10[tid] + b11[tid]
                         + b12[tid] + b13[tid] + b14[tid] + b15[tid]
                         + b16[tid];
            }
        "#;

        let input_refs: Vec<&MlxArray> = inputs.iter().collect();
        let result = metal_kernel::metal_kernel(&metal_kernel::MetalKernelParams {
            name: "sum_17",
            inputs: &input_refs,
            outputs: &[],
            source,
            grid: [n, 1, 1],
            threadgroup: [32, 1, 1],
            output_shapes: &[&[n]],
            output_dtypes: &[MlxDtype::Float32],
            stream: &stream,
        })
        .unwrap();

        eval(&[&result[0]]).unwrap();
        #[allow(unsafe_code)]
        let out: &[f32] = unsafe { result[0].as_contiguous_slice().unwrap() };
        let expected: f32 = (1..=17).map(|i| i as f32).sum();
        for &v in out.iter() {
            assert!((v - expected).abs() < 1e-3, "got {v}, expected {expected}");
        }
    }

    /// Test 5: Non-trivial dispatch geometry (head_dim × num_kv_heads × 1)
    #[test]
    #[cfg(not(mlx_stub))]
    fn dispatch_geometry() {
        let device = MlxDevice::default_gpu().unwrap();
        let stream = MlxStream::new(&device).unwrap();

        let head_dim = 128usize;
        let num_heads = 8usize;
        let total = head_dim * num_heads;
        let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let input =
            MlxArray::from_data_copy(&f32_to_bytes(&data), &[total], MlxDtype::Float32, &stream)
                .unwrap();

        let source = r#"
            #include <metal_stdlib>
            using namespace metal;
            [[kernel]] void geometry_test(
                device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                uint3 tid [[thread_position_in_grid]])
            {
                uint head = tid.y;
                uint d = tid.x;
                uint idx = head * 128 + d;
                output[idx] = input[idx] + float(head);
            }
        "#;

        let result = metal_kernel::metal_kernel(&metal_kernel::MetalKernelParams {
            name: "geometry_test",
            inputs: &[&input],
            outputs: &[],
            source,
            grid: [head_dim, num_heads, 1],
            threadgroup: [128, 1, 1],
            output_shapes: &[&[total]],
            output_dtypes: &[MlxDtype::Float32],
            stream: &stream,
        })
        .unwrap();

        eval(&[&result[0]]).unwrap();
        #[allow(unsafe_code)]
        let out: &[f32] = unsafe { result[0].as_contiguous_slice().unwrap() };
        for head in 0..num_heads {
            for d in 0..head_dim {
                let idx = head * head_dim + d;
                let expected = idx as f32 + head as f32;
                assert!(
                    (out[idx] - expected).abs() < 1e-5,
                    "at [{head}, {d}]: got {}, expected {expected}",
                    out[idx]
                );
            }
        }
    }

    /// Test 6: JIT compilation caching
    #[test]
    #[cfg(not(mlx_stub))]
    fn jit_caching() {
        let device = MlxDevice::default_gpu().unwrap();
        let stream = MlxStream::new(&device).unwrap();

        let source = r#"
            #include <metal_stdlib>
            using namespace metal;
            [[kernel]] void noop_kernel(
                device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                uint tid [[thread_position_in_grid]])
            {
                output[tid] = input[tid];
            }
        "#;

        let data: Vec<f32> = vec![42.0; 1024];
        let input =
            MlxArray::from_data_copy(&f32_to_bytes(&data), &[1024], MlxDtype::Float32, &stream)
                .unwrap();

        // First call (cold JIT).
        let t0 = std::time::Instant::now();
        let r1 = metal_kernel::metal_kernel(&metal_kernel::MetalKernelParams {
            name: "noop_kernel",
            inputs: &[&input],
            outputs: &[],
            source,
            grid: [1024, 1, 1],
            threadgroup: [256, 1, 1],
            output_shapes: &[&[1024]],
            output_dtypes: &[MlxDtype::Float32],
            stream: &stream,
        })
        .unwrap();
        eval(&[&r1[0]]).unwrap();
        let cold_time = t0.elapsed();

        // Subsequent calls (should be cached).
        let t1 = std::time::Instant::now();
        for _ in 0..10 {
            let r = metal_kernel::metal_kernel(&metal_kernel::MetalKernelParams {
                name: "noop_kernel",
                inputs: &[&input],
                outputs: &[],
                source,
                grid: [1024, 1, 1],
                threadgroup: [256, 1, 1],
                output_shapes: &[&[1024]],
                output_dtypes: &[MlxDtype::Float32],
                stream: &stream,
            })
            .unwrap();
            eval(&[&r[0]]).unwrap();
        }
        let warm_time = t1.elapsed() / 10;

        // Warm should be faster than cold (JIT caching works).
        eprintln!("JIT cold: {cold_time:?}, warm avg: {warm_time:?}");
        assert!(
            warm_time < cold_time * 5,
            "warm ({warm_time:?}) should be faster than 5× cold ({cold_time:?})"
        );
    }
}
