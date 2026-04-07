//! Qwen 3.5 Metal correctness tests.
//!
//! These are genuine correctness tests (not benchmarks) that verify
//! Metal inference produces valid outputs. For performance benchmarks,
//! use: `cargo run --release -p ironmill-bench --features metal -- --config configs/qwen35-bench.toml -b metal`
//!
//! Requires: Metal GPU, Qwen3.5-4B SafeTensors.
//! Run: `cargo test -p ironmill-bench --features metal -- qwen35 --ignored --nocapture`

#[cfg(feature = "metal")]
mod qwen35_correctness {
    use std::path::PathBuf;

    use ironmill_compile::weights::{SafeTensorsProvider, WeightProvider};
    use ironmill_inference::InferenceEngine;
    use ironmill_inference::metal::{MetalConfig, MetalInference};

    fn qwen35_model_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/Qwen3.5-4B")
    }

    fn skip_if_missing(path: &PathBuf, label: &str) {
        if !path.exists() {
            panic!(
                "Fixture {label} not found at {}. Download model weights first.",
                path.display()
            );
        }
    }

    fn load_engine(config: MetalConfig) -> MetalInference {
        let model_dir = qwen35_model_dir();
        skip_if_missing(&model_dir, "Qwen3.5-4B");

        let provider =
            SafeTensorsProvider::load(&model_dir).expect("SafeTensorsProvider::load failed");
        let mut engine = MetalInference::new(config.clone()).expect("MetalInference::new failed");
        engine
            .load_weights(&provider, config)
            .expect("load_weights failed");
        engine
    }

    fn fp16_config() -> MetalConfig {
        let mut c = MetalConfig::default();
        c.enable_turboquant = false;
        c.prefill_chunk_size = Some(256);
        c
    }

    #[test]
    #[ignore] // requires Metal GPU + Qwen3.5-4B
    fn qwen35_hidden_state_readback() {
        let mut engine = load_engine(fp16_config());
        let prompt: Vec<u32> = vec![9419, 1814];
        engine.prefill(&prompt).expect("prefill failed");

        let hidden = engine
            .last_hidden_state()
            .expect("last_hidden_state failed");
        let model_info = engine.model_info();
        let expected_len = model_info.hidden_size;

        assert_eq!(
            hidden.len(),
            expected_len,
            "hidden state length should be {expected_len}, got {}",
            hidden.len()
        );
        assert!(
            hidden.iter().any(|&v| v != 0.0),
            "hidden state should not be all zeros"
        );
        let max_abs = hidden.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs < 1000.0,
            "hidden state max abs should be <1000, got {max_abs:.1}"
        );
        assert!(
            max_abs > 0.01,
            "hidden state max abs suspiciously small: {max_abs:.6}"
        );

        let (logits, hidden2) = engine
            .decode_step_with_hidden(9419)
            .expect("decode_step_with_hidden failed");
        assert_eq!(hidden2.len(), expected_len);
        assert!(!logits.is_empty(), "logits should not be empty");
    }
}
