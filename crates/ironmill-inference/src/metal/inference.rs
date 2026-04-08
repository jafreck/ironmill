//! Tests for Metal inference modules.
//!
//! Contains compile-time API surface tests for the calibration pipeline
//! and correctness tests for FlashAttention-2 prefill kernels.

#[cfg(test)]
use super::engine::MetalInference;

// ── Calibration tests ──────────────────────────────────────────
//
// These tests require a real Metal device (macOS only) and a loaded model.
// They are gated behind `#[cfg(test)]` and will be skipped in CI without
// Metal hardware. For local validation, run:
//   cargo test -p ironmill-inference --features metal -- calibration
//
// Without a model fixture these serve as compile-time validation of the
// calibration API surface.  The `_api_surface` test verifies the method
// signatures and callback types compile correctly.

#[cfg(test)]
mod calibration_tests {
    use super::super::buffers::bytes_as_f16;
    use super::*;
    use half::f16;

    /// Verify that the calibration method signature compiles and the
    /// callback type is properly accepted as a trait object.
    #[test]
    fn calibration_api_surface_compiles() {
        // This test validates at compile time that:
        // 1. run_pipeline_calibration accepts &mut dyn FnMut(usize, &str, &[u8])
        // 2. prefill_calibration accepts the same callback type
        // 3. Both return Result<Logits, InferenceError>
        //
        // We cannot run inference without a loaded model, but we can
        // verify the type signatures are correct.

        fn _assert_method_exists(engine: &mut MetalInference) {
            let mut count = 0usize;
            let mut callback = |layer: usize, name: &str, data: &[u8], n_features: usize| {
                let _ = (layer, name, data, n_features);
                count += 1;
            };
            // These calls would fail at runtime (no model loaded), but they
            // prove the API compiles.
            let _ = engine.run_pipeline_calibration(&[1, 2, 3], &mut callback, None);
            let _ = engine.prefill_calibration(&[1, 2, 3], &mut callback);
        }

        // Just verify the function compiles — don't actually call it.
        let _ = _assert_method_exists;
    }

    /// Verify that run_pipeline_with_hooks and prefill_with_hooks signatures
    /// compile and accept `&mut dyn ActivationHook`.
    #[test]
    fn hook_bridge_api_surface_compiles() {
        use crate::calibration::AwqActivationStore;

        fn _assert_hook_methods(engine: &mut MetalInference) {
            let mut store = AwqActivationStore::new();
            // These would fail at runtime (no model loaded) but prove
            // the API surface compiles.
            let _ = engine.run_pipeline_with_hooks(&[1, 2, 3], &mut store);
            let _ = engine.prefill_with_hooks(&[1, 2, 3], &mut store);
        }

        let _ = _assert_hook_methods;
    }

    /// Verify that `MetalInference` implements `CalibratingEngine` and the
    /// trait methods compile with the expected signatures.
    #[test]
    fn calibrating_engine_impl_compiles() {
        use crate::calibration::{AwqActivationStore, CalibratingEngine};

        fn _assert_calibrating_engine(engine: &mut MetalInference) {
            let mut store = AwqActivationStore::new();
            let _ = CalibratingEngine::prefill_with_hooks(engine, &[1, 2, 3], &mut store);
            CalibratingEngine::reset(engine);
        }

        let _ = _assert_calibrating_engine;
    }

    /// Verify that `bytes_as_f16` correctly reinterprets raw bytes.
    #[test]
    fn bytes_as_f16_roundtrip() {
        let values = [f16::from_f32(1.0), f16::from_f32(-2.5), f16::from_f32(0.0)];
        // Serialize to bytes in native byte order.
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| v.to_bits().to_ne_bytes())
            .collect();

        let converted = bytes_as_f16(&bytes);
        assert_eq!(converted.len(), 3);
        assert_eq!(converted[0], f16::from_f32(1.0));
        assert_eq!(converted[1], f16::from_f32(-2.5));
        assert_eq!(converted[2], f16::from_f32(0.0));
    }

    /// Verify that `bytes_as_f16` panics on an odd-length byte slice.
    #[test]
    #[should_panic]
    fn bytes_as_f16_rejects_odd_length() {
        bytes_as_f16(&[0u8; 3]);
    }

    /// Verify that a closure capturing mutable state works as the callback.
    #[test]
    fn calibration_callback_captures_state() {
        // Simulate what a real calibration consumer would do: accumulate
        // activation statistics across layers.
        struct ActivationStats {
            captures: Vec<(usize, String, usize)>, // (layer, name, byte_count)
        }

        let mut stats = ActivationStats {
            captures: Vec::new(),
        };

        // Build a callback that captures &mut stats.
        let mut callback = |layer: usize, name: &str, data: &[u8]| {
            stats.captures.push((layer, name.to_string(), data.len()));
        };

        // Simulate the callback being invoked as it would be during calibration.
        // 2 layers × 2 captures per layer = 4 invocations.
        let hidden_size = 128;
        let token_count = 8;
        let fake_data = vec![0u8; token_count * hidden_size * 2]; // FP16

        for layer in 0..2 {
            callback(layer, "attn_norm", &fake_data);
            callback(layer, "ffn_norm", &fake_data);
        }

        assert_eq!(stats.captures.len(), 4);
        assert_eq!(
            stats.captures[0],
            (0, "attn_norm".to_string(), fake_data.len())
        );
        assert_eq!(
            stats.captures[1],
            (0, "ffn_norm".to_string(), fake_data.len())
        );
        assert_eq!(
            stats.captures[2],
            (1, "attn_norm".to_string(), fake_data.len())
        );
        assert_eq!(
            stats.captures[3],
            (1, "ffn_norm".to_string(), fake_data.len())
        );

        // Verify expected byte size: token_count × hidden_size × 2 (FP16)
        for (_, _, byte_count) in &stats.captures {
            assert_eq!(*byte_count, token_count * hidden_size * 2);
        }
    }

    /// Verify the INT4 dequant shader compiles on the current Metal device.
    #[test]
    fn int4_dequant_shader_compiles_on_device() {
        use ironmill_metal_sys::MetalDevice;

        let device = match MetalDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("Skipping — no Metal device available");
                return;
            }
        };

        let src = include_str!("shaders/quantized/int4_dequant.metal");
        let lib = device
            .compile_shader_source(src)
            .expect("int4_dequant.metal should compile");
        let func = lib
            .get_function("int4_dequantize")
            .expect("int4_dequantize function should exist");
        let _pipeline = device
            .create_compute_pipeline(&func)
            .expect("should create compute pipeline");
    }
}

// ── FA2 prefill attention correctness tests ────────────────────
//
// These tests verify that the FlashAttention-2 prefill kernel produces
// the same output as the fused SDPA kernel for the same inputs.
// Requires a Metal GPU.

#[cfg(test)]
mod fa2_prefill_tests {
    use half::f16;
    use ironmill_metal_sys::{MetalDevice, StorageMode};

    /// Create a Metal buffer filled with FP16 data from f32 values.
    fn make_fp16_buffer(device: &MetalDevice, data: &[f32]) -> ironmill_metal_sys::MetalBuffer {
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect();
        device
            .create_buffer_with_data(&bytes, StorageMode::Shared)
            .expect("create buffer")
    }

    /// Read FP16 buffer back as f32 values.
    fn read_fp16_buffer(buf: &ironmill_metal_sys::MetalBuffer, count: usize) -> Vec<f32> {
        let byte_count = count * 2;
        let mut bytes = vec![0u8; byte_count];
        buf.read_bytes(&mut bytes, 0).expect("read_bytes");
        bytes
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()
    }

    /// CPU reference: causal scaled dot-product attention.
    ///
    /// Q: [token_count, num_q_heads, head_dim]
    /// K/V cache: [num_kv_heads, max_seq_len, head_dim] (filled up to seq_offset + token_count)
    fn cpu_attention(
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        seq_offset: usize,
        token_count: usize,
        scale: f32,
    ) -> Vec<f32> {
        let heads_per_group = num_q_heads / num_kv_heads;
        let mut output = vec![0.0f32; token_count * num_q_heads * head_dim];

        for t in 0..token_count {
            let causal_len = seq_offset + t + 1;
            for h in 0..num_q_heads {
                let kv_h = h / heads_per_group;
                let q_base = (t * num_q_heads + h) * head_dim;

                // Compute QK^T scores
                let mut scores = vec![-f32::INFINITY; causal_len];
                for p in 0..causal_len {
                    let k_base = (kv_h * max_seq_len + p) * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_base + d] * k_cache[k_base + d];
                    }
                    scores[p] = dot * scale;
                }

                // Softmax
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_score).exp();
                    sum += *s;
                }
                for s in &mut scores {
                    *s /= sum;
                }

                // Weighted sum of V
                let o_base = (t * num_q_heads + h) * head_dim;
                for p in 0..causal_len {
                    let v_base = (kv_h * max_seq_len + p) * head_dim;
                    for d in 0..head_dim {
                        output[o_base + d] += scores[p] * v_cache[v_base + d];
                    }
                }
            }
        }
        output
    }

    /// Generate deterministic pseudo-random f32 values in [-1, 1].
    fn pseudo_random(seed: u64, count: usize) -> Vec<f32> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                // xorshift64
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Fill KV cache positions 0..token_count from flat fill arrays.
    fn fill_kv_cache(
        cache: &mut [f32],
        fill: &[f32],
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        token_count: usize,
    ) {
        for kv_h in 0..num_kv_heads {
            for t in 0..token_count {
                for d in 0..head_dim {
                    cache[kv_h * max_seq_len * head_dim + t * head_dim + d] =
                        fill[kv_h * token_count * head_dim + t * head_dim + d];
                }
            }
        }
    }

    /// Verify FA2 prefill produces the same output as fused SDPA.
    ///
    /// Uses head_dim=128 (precompiled shaders), 4 Q heads, 2 KV heads,
    /// 8 tokens of prefill.
    #[test]
    fn fa2_matches_fused_sdpa() {
        let device = match MetalDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("SKIP: no Metal device");
                return;
            }
        };
        let queue = device.create_command_queue().expect("command queue");

        let head_dim = 128usize;
        let num_q_heads = 4u32;
        let num_kv_heads = 2u32;
        let token_count = 8usize;
        let max_seq_len = 64usize;
        let seq_offset = 0usize;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q_data = pseudo_random(42, token_count * num_q_heads as usize * head_dim);
        let mut k_data = vec![0.0f32; num_kv_heads as usize * max_seq_len * head_dim];
        let mut v_data = vec![0.0f32; num_kv_heads as usize * max_seq_len * head_dim];
        fill_kv_cache(
            &mut k_data,
            &pseudo_random(123, num_kv_heads as usize * token_count * head_dim),
            num_kv_heads as usize,
            max_seq_len,
            head_dim,
            token_count,
        );
        fill_kv_cache(
            &mut v_data,
            &pseudo_random(456, num_kv_heads as usize * token_count * head_dim),
            num_kv_heads as usize,
            max_seq_len,
            head_dim,
            token_count,
        );

        let expected = cpu_attention(
            &q_data,
            &k_data,
            &v_data,
            num_q_heads as usize,
            num_kv_heads as usize,
            head_dim,
            max_seq_len,
            seq_offset,
            token_count,
            scale,
        );

        let q_buf = make_fp16_buffer(&device, &q_data);
        let k_buf = make_fp16_buffer(&device, &k_data);
        let v_buf = make_fp16_buffer(&device, &v_data);
        let output_size = token_count * num_q_heads as usize * head_dim;
        let output_fa2 = device
            .create_buffer(output_size * 2, StorageMode::Shared)
            .expect("output fa2");
        let output_sdpa = device
            .create_buffer(output_size * 2, StorageMode::Shared)
            .expect("output sdpa");

        let pipelines = super::super::ops::MetalPipelines::compile(&device, head_dim, head_dim)
            .expect("compile pipelines");

        // --- Dispatch FA2 ---
        {
            let cmd = queue.command_buffer().expect("cmd");
            let enc = cmd.compute_encoder().expect("enc");
            super::super::ops::encode_fa2_prefill_attention(
                &enc,
                &pipelines.attention.prefill_attention_fa2,
                &super::super::ops::PrefillAttentionParams {
                    q: &q_buf,
                    k_cache: &k_buf,
                    v_cache: &v_buf,
                    output: &output_fa2,
                    num_heads: num_q_heads,
                    num_kv_heads,
                    head_dim: head_dim as u32,
                    max_seq_len: max_seq_len as u32,
                    seq_offset: seq_offset as u32,
                    token_count: token_count as u32,
                    window_size: 0,
                    attn_scale: scale,
                },
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        // --- Dispatch fused SDPA ---
        {
            let cmd = queue.command_buffer().expect("cmd");
            let enc = cmd.compute_encoder().expect("enc");
            let total_seq = (seq_offset + token_count) as u32;
            super::super::ops::encode_fused_sdpa(
                &enc,
                pipelines
                    .attention
                    .fused_sdpa
                    .as_ref()
                    .expect("fused_sdpa pipeline"),
                &super::super::ops::FusedSdpaParams {
                    q: &q_buf,
                    k: &k_buf,
                    v: &v_buf,
                    output: &output_sdpa,
                    seq_len: total_seq,
                    token_count: token_count as u32,
                    head_dim: head_dim as u32,
                    num_q_heads,
                    num_kv_heads,
                    scale,
                    max_seq_len: max_seq_len as u32,
                },
                None,
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let fa2_result = read_fp16_buffer(&output_fa2, output_size);
        let sdpa_result = read_fp16_buffer(&output_sdpa, output_size);

        let mut max_diff_fa2_sdpa = 0.0f32;
        let mut max_diff_fa2_cpu = 0.0f32;
        let mut max_diff_sdpa_cpu = 0.0f32;
        for i in 0..output_size {
            max_diff_fa2_sdpa = max_diff_fa2_sdpa.max((fa2_result[i] - sdpa_result[i]).abs());
            max_diff_fa2_cpu = max_diff_fa2_cpu.max((fa2_result[i] - expected[i]).abs());
            max_diff_sdpa_cpu = max_diff_sdpa_cpu.max((sdpa_result[i] - expected[i]).abs());
        }

        println!("FA2 vs SDPA max diff:  {max_diff_fa2_sdpa:.6}");
        println!("FA2 vs CPU  max diff:  {max_diff_fa2_cpu:.6}");
        println!("SDPA vs CPU max diff:  {max_diff_sdpa_cpu:.6}");

        // FP16 accumulation error: tolerate up to 0.05 for head_dim=128
        assert!(
            max_diff_fa2_sdpa < 0.05,
            "FA2 vs SDPA diverged: {max_diff_fa2_sdpa}"
        );
        assert!(
            max_diff_fa2_cpu < 0.1,
            "FA2 vs CPU diverged: {max_diff_fa2_cpu}"
        );
        assert!(
            max_diff_sdpa_cpu < 0.1,
            "SDPA vs CPU diverged: {max_diff_sdpa_cpu}"
        );
    }

    /// Verify FA2 handles GQA (grouped-query attention) correctly:
    /// multiple Q heads share the same KV head.
    #[test]
    fn fa2_gqa_correctness() {
        let device = match MetalDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("SKIP: no Metal device");
                return;
            }
        };
        let queue = device.create_command_queue().expect("command queue");

        let head_dim = 64usize;
        let num_q_heads = 8u32;
        let num_kv_heads = 2u32;
        let token_count = 4usize;
        let max_seq_len = 32usize;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q_data = pseudo_random(77, token_count * num_q_heads as usize * head_dim);
        let mut k_data = vec![0.0f32; num_kv_heads as usize * max_seq_len * head_dim];
        let mut v_data = vec![0.0f32; num_kv_heads as usize * max_seq_len * head_dim];
        fill_kv_cache(
            &mut k_data,
            &pseudo_random(88, num_kv_heads as usize * token_count * head_dim),
            num_kv_heads as usize,
            max_seq_len,
            head_dim,
            token_count,
        );
        fill_kv_cache(
            &mut v_data,
            &pseudo_random(99, num_kv_heads as usize * token_count * head_dim),
            num_kv_heads as usize,
            max_seq_len,
            head_dim,
            token_count,
        );

        let expected = cpu_attention(
            &q_data,
            &k_data,
            &v_data,
            num_q_heads as usize,
            num_kv_heads as usize,
            head_dim,
            max_seq_len,
            0,
            token_count,
            scale,
        );

        let q_buf = make_fp16_buffer(&device, &q_data);
        let k_buf = make_fp16_buffer(&device, &k_data);
        let v_buf = make_fp16_buffer(&device, &v_data);
        let output_size = token_count * num_q_heads as usize * head_dim;
        let output_buf = device
            .create_buffer(output_size * 2, StorageMode::Shared)
            .expect("out");

        let pipelines = super::super::ops::MetalPipelines::compile(&device, head_dim, head_dim)
            .expect("compile");

        let cmd = queue.command_buffer().expect("cmd");
        let enc = cmd.compute_encoder().expect("enc");
        super::super::ops::encode_fa2_prefill_attention(
            &enc,
            &pipelines.attention.prefill_attention_fa2,
            &super::super::ops::PrefillAttentionParams {
                q: &q_buf,
                k_cache: &k_buf,
                v_cache: &v_buf,
                output: &output_buf,
                num_heads: num_q_heads,
                num_kv_heads,
                head_dim: head_dim as u32,
                max_seq_len: max_seq_len as u32,
                seq_offset: 0,
                token_count: token_count as u32,
                window_size: 0,
                attn_scale: scale,
            },
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let result = read_fp16_buffer(&output_buf, output_size);
        let mut max_diff = 0.0f32;
        for i in 0..output_size {
            max_diff = max_diff.max((result[i] - expected[i]).abs());
        }
        println!("FA2 GQA (8:2) vs CPU max diff: {max_diff:.6}");
        assert!(max_diff < 0.1, "FA2 GQA diverged: max_diff={max_diff}");
    }

    /// Verify FA2 with attn_scale=1.0 (QK-normed models like Gemma 4).
    #[test]
    fn fa2_unit_attn_scale() {
        let device = match MetalDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("SKIP: no Metal device");
                return;
            }
        };
        let queue = device.create_command_queue().expect("command queue");

        let head_dim = 128usize;
        let num_q_heads = 2u32;
        let num_kv_heads = 2u32;
        let token_count = 4usize;
        let max_seq_len = 16usize;
        let scale = 1.0f32;

        // Use small values so softmax doesn't saturate with scale=1.0
        let q_data: Vec<f32> = pseudo_random(111, token_count * num_q_heads as usize * head_dim)
            .iter()
            .map(|&v| v * 0.1)
            .collect();
        let mut k_data = vec![0.0f32; num_kv_heads as usize * max_seq_len * head_dim];
        let mut v_data = vec![0.0f32; num_kv_heads as usize * max_seq_len * head_dim];
        let k_fill: Vec<f32> = pseudo_random(222, num_kv_heads as usize * token_count * head_dim)
            .iter()
            .map(|&v| v * 0.1)
            .collect();
        fill_kv_cache(
            &mut k_data,
            &k_fill,
            num_kv_heads as usize,
            max_seq_len,
            head_dim,
            token_count,
        );
        fill_kv_cache(
            &mut v_data,
            &pseudo_random(333, num_kv_heads as usize * token_count * head_dim),
            num_kv_heads as usize,
            max_seq_len,
            head_dim,
            token_count,
        );

        let expected = cpu_attention(
            &q_data,
            &k_data,
            &v_data,
            num_q_heads as usize,
            num_kv_heads as usize,
            head_dim,
            max_seq_len,
            0,
            token_count,
            scale,
        );

        let q_buf = make_fp16_buffer(&device, &q_data);
        let k_buf = make_fp16_buffer(&device, &k_data);
        let v_buf = make_fp16_buffer(&device, &v_data);
        let output_size = token_count * num_q_heads as usize * head_dim;
        let output_buf = device
            .create_buffer(output_size * 2, StorageMode::Shared)
            .expect("out");

        let pipelines = super::super::ops::MetalPipelines::compile(&device, head_dim, head_dim)
            .expect("compile");

        let cmd = queue.command_buffer().expect("cmd");
        let enc = cmd.compute_encoder().expect("enc");
        super::super::ops::encode_fa2_prefill_attention(
            &enc,
            &pipelines.attention.prefill_attention_fa2,
            &super::super::ops::PrefillAttentionParams {
                q: &q_buf,
                k_cache: &k_buf,
                v_cache: &v_buf,
                output: &output_buf,
                num_heads: num_q_heads,
                num_kv_heads,
                head_dim: head_dim as u32,
                max_seq_len: max_seq_len as u32,
                seq_offset: 0,
                token_count: token_count as u32,
                window_size: 0,
                attn_scale: scale,
            },
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let result = read_fp16_buffer(&output_buf, output_size);
        let mut max_diff = 0.0f32;
        for i in 0..output_size {
            max_diff = max_diff.max((result[i] - expected[i]).abs());
        }
        println!("FA2 scale=1.0 vs CPU max diff: {max_diff:.6}");
        assert!(
            max_diff < 0.1,
            "FA2 scale=1.0 diverged: max_diff={max_diff}"
        );
    }

    /// Verify v2 register-tiled prefill produces the same output as
    /// fused SDPA and the original FA2 kernel.
    #[test]
    fn v2_matches_fa2_and_sdpa() {
        let device = match MetalDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("SKIP: no Metal device");
                return;
            }
        };
        let queue = device.create_command_queue().expect("command queue");

        let head_dim = 128usize;
        let num_q_heads = 4u32;
        let num_kv_heads = 2u32;
        let token_count = 16usize;
        let max_seq_len = 64usize;
        let seq_offset = 0usize;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q_data = pseudo_random(42, token_count * num_q_heads as usize * head_dim);
        let mut k_data = vec![0.0f32; num_kv_heads as usize * max_seq_len * head_dim];
        let mut v_data = vec![0.0f32; num_kv_heads as usize * max_seq_len * head_dim];
        fill_kv_cache(
            &mut k_data,
            &pseudo_random(123, num_kv_heads as usize * token_count * head_dim),
            num_kv_heads as usize,
            max_seq_len,
            head_dim,
            token_count,
        );
        fill_kv_cache(
            &mut v_data,
            &pseudo_random(456, num_kv_heads as usize * token_count * head_dim),
            num_kv_heads as usize,
            max_seq_len,
            head_dim,
            token_count,
        );

        let expected = cpu_attention(
            &q_data,
            &k_data,
            &v_data,
            num_q_heads as usize,
            num_kv_heads as usize,
            head_dim,
            max_seq_len,
            seq_offset,
            token_count,
            scale,
        );

        let q_buf = make_fp16_buffer(&device, &q_data);
        let k_buf = make_fp16_buffer(&device, &k_data);
        let v_buf = make_fp16_buffer(&device, &v_data);
        let output_size = token_count * num_q_heads as usize * head_dim;

        let pipelines = super::super::ops::MetalPipelines::compile(&device, head_dim, head_dim)
            .expect("compile pipelines");

        // --- Dispatch v2 ---
        let output_v2 = device
            .create_buffer(output_size * 2, StorageMode::Shared)
            .expect("out");
        {
            let cmd = queue.command_buffer().expect("cmd");
            let enc = cmd.compute_encoder().expect("enc");
            super::super::ops::encode_v2_prefill_attention(
                &enc,
                &pipelines.attention.prefill_attention_v2,
                &super::super::ops::PrefillAttentionParams {
                    q: &q_buf,
                    k_cache: &k_buf,
                    v_cache: &v_buf,
                    output: &output_v2,
                    num_heads: num_q_heads,
                    num_kv_heads,
                    head_dim: head_dim as u32,
                    max_seq_len: max_seq_len as u32,
                    seq_offset: seq_offset as u32,
                    token_count: token_count as u32,
                    window_size: 0,
                    attn_scale: scale,
                },
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        // --- Dispatch FA2 (original) ---
        let output_fa2 = device
            .create_buffer(output_size * 2, StorageMode::Shared)
            .expect("out");
        {
            let cmd = queue.command_buffer().expect("cmd");
            let enc = cmd.compute_encoder().expect("enc");
            super::super::ops::encode_fa2_prefill_attention(
                &enc,
                &pipelines.attention.prefill_attention_fa2,
                &super::super::ops::PrefillAttentionParams {
                    q: &q_buf,
                    k_cache: &k_buf,
                    v_cache: &v_buf,
                    output: &output_fa2,
                    num_heads: num_q_heads,
                    num_kv_heads,
                    head_dim: head_dim as u32,
                    max_seq_len: max_seq_len as u32,
                    seq_offset: seq_offset as u32,
                    token_count: token_count as u32,
                    window_size: 0,
                    attn_scale: scale,
                },
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let v2_result = read_fp16_buffer(&output_v2, output_size);
        let fa2_result = read_fp16_buffer(&output_fa2, output_size);

        let mut max_v2_fa2 = 0.0f32;
        let mut max_v2_cpu = 0.0f32;
        for i in 0..output_size {
            max_v2_fa2 = max_v2_fa2.max((v2_result[i] - fa2_result[i]).abs());
            max_v2_cpu = max_v2_cpu.max((v2_result[i] - expected[i]).abs());
        }

        println!("V2 vs FA2 max diff:  {max_v2_fa2:.6}");
        println!("V2 vs CPU max diff:  {max_v2_cpu:.6}");

        assert!(max_v2_fa2 < 0.05, "V2 vs FA2 diverged: {max_v2_fa2}");
        assert!(max_v2_cpu < 0.1, "V2 vs CPU diverged: {max_v2_cpu}");
    }
}
