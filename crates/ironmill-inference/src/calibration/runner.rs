//! End-to-end calibration runner.
//!
//! [`CalibrationRunner`] orchestrates model calibration: iterates over a
//! [`CalibrationDataset`], runs forward passes through a [`CalibratingEngine`],
//! and returns populated activation stores for AWQ or GPTQ quantisation.

use super::awq_store::AwqActivationStore;
use super::dataset::CalibrationDataset;
use super::gptq_store::GptqActivationStore;
use super::hook::ActivationHook;

/// Hessian accumulator for QuIP# LDLQ rounding at a single capture point.
///
/// Computes H ≈ X^T X / n where X is the [n_tokens × n_features] matrix
/// of input activations collected during calibration.
pub struct QuipHessianAccumulator {
    /// Accumulated X^T X matrix, stored as a flat [n_features × n_features] array.
    pub xtx: Vec<f64>,
    /// Number of features (columns per token).
    pub n_features: usize,
    /// Total number of token samples accumulated.
    pub sample_count: usize,
}

impl QuipHessianAccumulator {
    pub fn new(n_features: usize) -> Self {
        Self {
            xtx: vec![0.0; n_features * n_features],
            n_features,
            sample_count: 0,
        }
    }

    /// Finalize the Hessian: returns H = X^T X / n as a flat f32 vector.
    pub fn finalize(&self) -> Vec<f32> {
        let n = self.sample_count.max(1) as f64;
        self.xtx.iter().map(|&v| (v / n) as f32).collect()
    }
}

/// Activation hook that collects Hessian statistics (X^T X) for QuIP#
/// LDLQ rounding.
pub struct HessianHook {
    /// Per-capture-point Hessian accumulators, keyed by `"layer_{i}_{name}"`.
    pub accumulators: std::collections::HashMap<String, QuipHessianAccumulator>,
}

impl HessianHook {
    pub fn new() -> Self {
        Self {
            accumulators: std::collections::HashMap::new(),
        }
    }
}

impl Default for HessianHook {
    fn default() -> Self {
        Self::new()
    }
}

impl ActivationHook for HessianHook {
    fn on_linear_input(
        &mut self,
        layer: usize,
        name: &str,
        activation: &[half::f16],
        n_features: usize,
    ) {
        let key = super::hook::store_key(layer, name);
        let acc = self
            .accumulators
            .entry(key)
            .or_insert_with(|| QuipHessianAccumulator::new(n_features));

        let n_tokens = activation.len() / n_features;
        acc.sample_count += n_tokens;

        // Accumulate X^T X: for each token row, add outer product.
        for t in 0..n_tokens {
            let row_start = t * n_features;
            for i in 0..n_features {
                let xi = activation[row_start + i].to_f64();
                for j in i..n_features {
                    let xj = activation[row_start + j].to_f64();
                    let product = xi * xj;
                    acc.xtx[i * n_features + j] += product;
                    if i != j {
                        acc.xtx[j * n_features + i] += product;
                    }
                }
            }
        }
    }
}

/// Trait for inference engines that support calibration-mode forward passes.
///
/// Implemented by backends such as `MetalInference` and mock engines for
/// testing. Kept separate from [`InferenceEngine`] because not every backend
/// supports activation hooks.
pub trait CalibratingEngine {
    /// Run a forward pass over `tokens`, invoking `hooks` at each linear
    /// projection capture point.
    fn prefill_with_hooks(
        &mut self,
        tokens: &[u32],
        hooks: &mut dyn ActivationHook,
    ) -> Result<(), Box<dyn std::error::Error>>;

    /// Reset engine state (KV cache, position) for a new sequence.
    fn reset(&mut self);
}

/// Orchestrates calibration: iterates over dataset sequences, runs forward
/// passes with hooks, and returns populated activation stores.
pub struct CalibrationRunner {
    /// Number of sequences per progress-reporting batch.
    pub batch_size: usize,
    /// Maximum number of sequences to process (`None` = use all).
    pub max_sequences: Option<usize>,
}

impl CalibrationRunner {
    pub fn new() -> Self {
        Self {
            batch_size: 4,
            max_sequences: None,
        }
    }

    /// Run calibration and collect AWQ activation statistics.
    ///
    /// Generic over the engine so tests can supply a mock.
    pub fn collect_awq_stats<E: CalibratingEngine>(
        &self,
        engine: &mut E,
        dataset: &CalibrationDataset,
    ) -> Result<AwqActivationStore, Box<dyn std::error::Error>> {
        let mut store = AwqActivationStore::new();
        self.run_calibration(engine, dataset, &mut store)?;
        Ok(store)
    }

    /// Run calibration and collect GPTQ Hessian statistics.
    ///
    /// Generic over the engine so tests can supply a mock.
    pub fn collect_gptq_stats<E: CalibratingEngine>(
        &self,
        engine: &mut E,
        dataset: &CalibrationDataset,
    ) -> Result<GptqActivationStore, Box<dyn std::error::Error>> {
        let mut store = GptqActivationStore::new();
        self.run_calibration(engine, dataset, &mut store)?;
        Ok(store)
    }

    /// Run calibration and collect Hessian statistics for QuIP# LDLQ rounding.
    ///
    /// Returns a [`HessianHook`] containing per-layer Hessian accumulators.
    /// Generic over the engine so tests can supply a mock.
    pub fn collect_hessian_stats<E: CalibratingEngine>(
        &self,
        engine: &mut E,
        dataset: &CalibrationDataset,
    ) -> Result<HessianHook, Box<dyn std::error::Error>> {
        let mut hook = HessianHook::new();
        self.run_calibration(engine, dataset, &mut hook)?;
        Ok(hook)
    }

    /// Shared calibration loop: iterate batches, run each sequence, report
    /// progress.
    fn run_calibration<E: CalibratingEngine>(
        &self,
        engine: &mut E,
        dataset: &CalibrationDataset,
        hook: &mut dyn ActivationHook,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let n_total = match self.max_sequences {
            Some(max) => max.min(dataset.sequences.len()),
            None => dataset.sequences.len(),
        };

        let sequences = &dataset.sequences[..n_total];
        let mut processed = 0usize;

        let batch_size = self.batch_size.max(1);
        for batch in sequences.chunks(batch_size) {
            for seq in batch {
                engine.reset();
                engine.prefill_with_hooks(seq, hook)?;
                processed += 1;
            }
            eprintln!("[calibration] {processed}/{n_total} sequences processed");
        }

        Ok(())
    }
}

impl Default for CalibrationRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    /// Mock engine that calls hooks with synthetic (all-ones) activations.
    struct MockEngine {
        n_layers: usize,
        n_features: usize,
        call_count: usize,
    }

    impl MockEngine {
        fn new(n_layers: usize, n_features: usize) -> Self {
            Self {
                n_layers,
                n_features,
                call_count: 0,
            }
        }
    }

    impl CalibratingEngine for MockEngine {
        fn prefill_with_hooks(
            &mut self,
            tokens: &[u32],
            hooks: &mut dyn ActivationHook,
        ) -> Result<(), Box<dyn std::error::Error>> {
            self.call_count += 1;
            let n_tokens = tokens.len();
            let activation: Vec<f16> = vec![f16::from_f32(1.0); n_tokens * self.n_features];
            for layer in 0..self.n_layers {
                hooks.on_linear_input(layer, "attn_norm", &activation, self.n_features);
                hooks.on_linear_input(layer, "ffn_norm", &activation, self.n_features);
            }
            Ok(())
        }

        fn reset(&mut self) {}
    }

    #[test]
    fn runner_default_values() {
        let runner = CalibrationRunner::new();
        assert_eq!(runner.batch_size, 4);
        assert!(runner.max_sequences.is_none());
    }

    #[test]
    fn default_trait_matches_new() {
        let a = CalibrationRunner::new();
        let b = CalibrationRunner::default();
        assert_eq!(a.batch_size, b.batch_size);
        assert_eq!(a.max_sequences, b.max_sequences);
    }

    #[test]
    fn collect_awq_stats_populates_store() {
        let mut engine = MockEngine::new(2, 4);
        let dataset = CalibrationDataset::random(100, 8, 6, 42);
        let runner = CalibrationRunner::new();

        let store = runner.collect_awq_stats(&mut engine, &dataset).unwrap();

        // 2 layers × 2 capture points = 4 keys
        assert_eq!(store.magnitudes.len(), 4);
        assert!(store.magnitudes.contains_key("layer_0_attn_norm"));
        assert!(store.magnitudes.contains_key("layer_0_ffn_norm"));
        assert!(store.magnitudes.contains_key("layer_1_attn_norm"));
        assert!(store.magnitudes.contains_key("layer_1_ffn_norm"));

        // 6 sequences × 8 tokens = 48 total tokens per capture point
        for mag in store.magnitudes.values() {
            assert_eq!(mag.sample_count, 48);
            assert_eq!(mag.mean_abs.len(), 4);
        }

        assert_eq!(engine.call_count, 6);
    }

    #[test]
    fn collect_gptq_stats_populates_store() {
        let mut engine = MockEngine::new(2, 3);
        let dataset = CalibrationDataset::random(100, 4, 4, 42);
        let runner = CalibrationRunner::new();

        let store = runner.collect_gptq_stats(&mut engine, &dataset).unwrap();

        assert_eq!(store.hessians.len(), 4);
        for acc in store.hessians.values() {
            assert_eq!(acc.n_features, 3);
            assert_eq!(acc.sample_count, 16); // 4 seq × 4 tokens
            assert_eq!(acc.xtx.len(), 9); // 3 × 3
        }
    }

    #[test]
    fn max_sequences_limits_processing() {
        let mut engine = MockEngine::new(1, 2);
        let dataset = CalibrationDataset::random(100, 4, 10, 42);
        let mut runner = CalibrationRunner::new();
        runner.max_sequences = Some(3);

        let store = runner.collect_awq_stats(&mut engine, &dataset).unwrap();

        let mag = &store.magnitudes["layer_0_attn_norm"];
        assert_eq!(mag.sample_count, 12); // 3 seq × 4 tokens
        assert_eq!(engine.call_count, 3);
    }

    #[test]
    fn max_sequences_exceeding_dataset_uses_all() {
        let mut engine = MockEngine::new(1, 2);
        let dataset = CalibrationDataset::random(100, 4, 3, 42);
        let mut runner = CalibrationRunner::new();
        runner.max_sequences = Some(100);

        let store = runner.collect_awq_stats(&mut engine, &dataset).unwrap();
        let mag = &store.magnitudes["layer_0_attn_norm"];
        assert_eq!(mag.sample_count, 12); // 3 × 4
        assert_eq!(engine.call_count, 3);
    }

    #[test]
    fn batch_size_affects_grouping_not_results() {
        let mut engine = MockEngine::new(1, 2);
        let dataset = CalibrationDataset::random(100, 4, 7, 42);
        let mut runner = CalibrationRunner::new();
        runner.batch_size = 3;

        let store = runner.collect_awq_stats(&mut engine, &dataset).unwrap();
        let mag = &store.magnitudes["layer_0_attn_norm"];
        assert_eq!(mag.sample_count, 28); // 7 × 4
        assert_eq!(engine.call_count, 7);
    }

    #[test]
    fn empty_dataset_produces_empty_store() {
        let mut engine = MockEngine::new(2, 4);
        let dataset = CalibrationDataset {
            name: "empty".into(),
            model: "test".into(),
            vocab_size: 100,
            seq_len: 0,
            num_sequences: 0,
            eos_token_id: None,
            sequences: vec![],
        };
        let runner = CalibrationRunner::new();

        let store = runner.collect_awq_stats(&mut engine, &dataset).unwrap();
        assert!(store.magnitudes.is_empty());
        assert_eq!(engine.call_count, 0);
    }

    #[test]
    fn calibrating_engine_is_object_safe() {
        fn _accepts_dyn(_e: &mut dyn CalibratingEngine) {}
    }

    #[test]
    fn engine_error_propagates() {
        struct FailEngine;
        impl CalibratingEngine for FailEngine {
            fn prefill_with_hooks(
                &mut self,
                _tokens: &[u32],
                _hooks: &mut dyn ActivationHook,
            ) -> Result<(), Box<dyn std::error::Error>> {
                Err("simulated failure".into())
            }
            fn reset(&mut self) {}
        }

        let mut engine = FailEngine;
        let dataset = CalibrationDataset::random(100, 4, 2, 0);
        let runner = CalibrationRunner::new();
        let err = runner.collect_awq_stats(&mut engine, &dataset).unwrap_err();
        assert!(err.to_string().contains("simulated failure"));
    }
}
