//! End-to-end tests for the PolarQuant GPU inference pipeline.
//!
//! Validates: ONNX model → GpuCompileBuilder → MilWeightProvider → GpuWeights::load
//! → GPU inference → output comparison with FP16 baseline.
//!
//! These tests require Metal GPU hardware and a model fixture, so they are
//! marked `#[ignore]`. Run explicitly with:
//!
//! ```sh
//! cargo test -p ironmill-inference --features metal --tests -- --ignored
//! ```
//!
//! Set `IRONMILL_TEST_MODEL` to a path to an ONNX/SafeTensors/GGUF model or
//! place a fixture at `tests/fixtures/test_model.onnx`.

#[cfg(feature = "metal")]
mod gpu_polarquant_e2e {
    use std::path::PathBuf;

    use ironmill_compile::gpu::GpuCompileBuilder;
    use ironmill_compile::weights::{QuantizationInfo, WeightProvider};
    use ironmill_inference::InferenceEngine;
    use ironmill_inference::gpu::{GpuConfig, GpuInference};

    /// Resolve the path to a test model.
    ///
    /// Checks the `IRONMILL_TEST_MODEL` env var first, then falls back to
    /// a fixture file next to this test file.
    fn test_model_path() -> Option<PathBuf> {
        let path_str = std::env::var("IRONMILL_TEST_MODEL").ok().or_else(|| {
            let p =
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test_model.onnx");
            p.exists().then(|| p.to_string_lossy().into_owned())
        })?;
        let p = PathBuf::from(path_str);
        p.exists().then_some(p)
    }

    /// Build a default GpuConfig suitable for testing (small sequence length,
    /// TurboQuant disabled to keep things simple).
    fn test_gpu_config() -> GpuConfig {
        GpuConfig {
            max_seq_len: 128,
            enable_turboquant: false,
            ..GpuConfig::default()
        }
    }

    /// Compute the root-mean-square error between two logit vectors.
    fn rmse(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "logit vectors must have the same length");
        let mse: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            / a.len() as f32;
        mse.sqrt()
    }

    // ── Full E2E pipeline test ──────────────────────────────────────────

    #[test]
    #[ignore] // requires Metal GPU + model fixture
    fn polarquant_gpu_e2e() {
        let model_path = test_model_path().expect(
            "test model not found — set IRONMILL_TEST_MODEL or place a fixture at \
             tests/fixtures/test_model.onnx",
        );
        let config = test_gpu_config();

        // ── 1. FP16 baseline ────────────────────────────────────────────
        // Build an unquantized provider by setting min_elements extremely
        // high so no tensor qualifies for quantization.
        let fp16_provider = GpuCompileBuilder::new(&model_path)
            .min_elements(usize::MAX)
            .build()
            .expect("FP16 compile failed");

        let mut fp16_engine =
            GpuInference::new(config.clone()).expect("failed to create FP16 GPU engine");
        fp16_engine
            .load_weights(&fp16_provider, config.clone())
            .expect("FP16 weight load failed");

        // Use a deterministic prompt token (token id 1 is typically <s>/BOS).
        let fp16_logits = fp16_engine.prefill(&[1]).expect("FP16 prefill failed");

        // ── 2. PolarQuant-4 pipeline ────────────────────────────────────
        let pq_provider = GpuCompileBuilder::new(&model_path)
            .polar_quantize(4)
            .min_elements(1024)
            .build()
            .expect("PolarQuant compile failed");

        let mut pq_engine =
            GpuInference::new(config.clone()).expect("failed to create PQ GPU engine");
        pq_engine
            .load_weights(&pq_provider, config)
            .expect("PolarQuant weight load failed");

        let pq_logits = pq_engine.prefill(&[1]).expect("PolarQuant prefill failed");

        // ── 3. Compare logits ───────────────────────────────────────────
        assert_eq!(
            fp16_logits.len(),
            pq_logits.len(),
            "logit dimension mismatch between FP16 and PolarQuant"
        );

        let error = rmse(&fp16_logits, &pq_logits);
        // Quantization noise tolerance — PolarQuant-4 should be close but
        // not bitwise identical. An RMSE below 2.0 is reasonable for 4-bit
        // weight quantization on typical LLM logits.
        assert!(
            error < 2.0,
            "FP16 vs PolarQuant-4 RMSE too large: {error:.4} (threshold: 2.0)"
        );

        // Sanity: the outputs should not be identical (that would mean
        // quantization had no effect at all).
        let max_diff: f32 = fp16_logits
            .iter()
            .zip(pq_logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 0.0,
            "FP16 and PolarQuant logits are bitwise identical — quantization may not have run"
        );
    }

    // ── Provider quant-info validation ──────────────────────────────────

    #[test]
    #[ignore] // requires model fixture
    fn polarquant_provider_has_correct_quant_info() {
        let model_path = test_model_path().expect(
            "test model not found — set IRONMILL_TEST_MODEL or place a fixture at \
             tests/fixtures/test_model.onnx",
        );

        let provider = GpuCompileBuilder::new(&model_path)
            .polar_quantize(4)
            .min_elements(1024)
            .build()
            .expect("PolarQuant compile failed");

        let names = provider.tensor_names();
        assert!(
            !names.is_empty(),
            "provider should expose at least one tensor"
        );

        let mut found_quantized = false;
        let mut found_unquantized = false;

        for name in &names {
            let tensor = provider
                .tensor(name)
                .unwrap_or_else(|e| panic!("failed to read tensor '{name}': {e}"));

            let num_elements = tensor.num_elements();

            match &tensor.quant_info {
                QuantizationInfo::LutToDense { n_bits, .. } => {
                    // Large weight tensors should be quantized to 4-bit.
                    assert_eq!(
                        *n_bits, 4,
                        "expected 4-bit quantization for '{name}', got {n_bits}"
                    );
                    assert!(
                        num_elements >= 1024,
                        "quantized tensor '{name}' has only {num_elements} elements \
                         (below min_elements threshold)"
                    );
                    found_quantized = true;
                }
                QuantizationInfo::None => {
                    // Small tensors (norms, biases) should remain unquantized.
                    found_unquantized = true;
                }
                other => {
                    // AffineDequantize is not expected from PolarQuant — flag it
                    // but don't hard-fail since the pass may evolve.
                    eprintln!("warning: unexpected quant_info variant for '{name}': {other:?}");
                }
            }
        }

        assert!(
            found_quantized,
            "expected at least one tensor with LutToDense quantization"
        );
        assert!(
            found_unquantized,
            "expected at least one tensor without quantization (small norm/bias)"
        );
    }

    // ── Compile-to-load path (lighter-weight, no full inference) ────────

    #[test]
    #[ignore] // requires Metal GPU + model fixture
    fn polarquant_compile_and_load_weights() {
        let model_path = test_model_path().expect(
            "test model not found — set IRONMILL_TEST_MODEL or place a fixture at \
             tests/fixtures/test_model.onnx",
        );
        let config = test_gpu_config();

        let provider = GpuCompileBuilder::new(&model_path)
            .polar_quantize(4)
            .min_elements(1024)
            .build()
            .expect("PolarQuant compile failed");

        // Verify the provider round-trips through GpuInference::load_weights
        // without panicking.
        let mut engine = GpuInference::new(config.clone()).expect("failed to create GPU engine");
        engine
            .load_weights(&provider, config)
            .expect("PolarQuant weight load into GPU failed");

        // The engine should report non-zero allocated memory.
        let allocated = engine.gpu_allocated_bytes();
        assert!(
            allocated > 0,
            "expected non-zero GPU memory after loading weights, got {allocated}"
        );
    }
}
