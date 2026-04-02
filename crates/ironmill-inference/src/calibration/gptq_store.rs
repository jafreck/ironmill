//! GPTQ-style activation store: Hessian (X^T X) accumulation.

use std::collections::HashMap;

use half::f16;

use super::hook::{ActivationHook, store_key};

/// Incremental Hessian accumulator for a single linear projection.
///
/// Stores the symmetric matrix X^T X in row-major order, where X is
/// the `(n_tokens, n_features)` activation matrix observed so far.
/// Memory cost is O(n_features²) per projection.
#[derive(Debug, Clone)]
pub struct HessianAccumulator {
    /// Flattened `n_features × n_features` symmetric matrix, row-major.
    pub xtx: Vec<f32>,
    /// Number of input features (columns of X).
    pub n_features: usize,
    /// Total number of tokens (rows) accumulated.
    pub sample_count: usize,
}

impl HessianAccumulator {
    /// Create a zero-initialised accumulator for `n_features` columns.
    pub fn new(n_features: usize) -> Self {
        Self {
            xtx: vec![0.0; n_features * n_features],
            n_features,
            sample_count: 0,
        }
    }

    /// Add the X^T X contribution from a flattened activation batch.
    ///
    /// `activation` has length `n_tokens × n_features` in row-major order.
    /// We exploit the symmetry of X^T X to halve the inner-loop work:
    /// compute only the upper triangle and mirror to the lower.
    pub fn accumulate(&mut self, activation: &[f16], n_features: usize) {
        assert_eq!(
            n_features, self.n_features,
            "feature count mismatch: expected {}, got {n_features}",
            self.n_features,
        );
        assert!(
            !activation.is_empty() && activation.len() % n_features == 0,
            "activation length {} is not a multiple of n_features {}",
            activation.len(),
            n_features,
        );

        let n_tokens = activation.len() / n_features;
        let n = self.n_features;

        for t in 0..n_tokens {
            let row = &activation[t * n..(t + 1) * n];
            // Upper triangle (including diagonal).
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                let xi = row[i].to_f32();
                for j in i..n {
                    let xj = row[j].to_f32();
                    self.xtx[i * n + j] += xi * xj;
                }
            }
        }

        // Mirror upper triangle to lower triangle.
        for i in 0..n {
            for j in (i + 1)..n {
                self.xtx[j * n + i] = self.xtx[i * n + j];
            }
        }

        self.sample_count += n_tokens;
    }
}

/// Activation store for GPTQ-style quantisation.
///
/// Incrementally accumulates X^T X Hessian matrices for each linear
/// projection in each transformer layer. Memory cost is O(n_features²) per
/// projection.
#[derive(Debug, Clone)]
pub struct GptqActivationStore {
    /// Maps `"layer_{i}_{name}"` → accumulated Hessian.
    pub hessians: HashMap<String, HessianAccumulator>,
}

impl GptqActivationStore {
    pub fn new() -> Self {
        Self {
            hessians: HashMap::new(),
        }
    }
}

impl Default for GptqActivationStore {
    fn default() -> Self {
        Self::new()
    }
}

impl GptqActivationStore {
    /// Convert to the format expected by GPTQ quantization passes.
    ///
    /// The Metal hook captures activations with keys like `"layer_0_q_proj"`.
    /// The MIL pass needs Hessian data keyed by const-op output names.
    /// Caller provides the mapping from weight op names to store keys.
    pub fn to_pass_format(
        &self,
        key_map: &HashMap<String, String>,
    ) -> HashMap<String, (Vec<f32>, usize, usize)> {
        let mut result = HashMap::new();
        for (weight_name, store_key) in key_map {
            if let Some(acc) = self.hessians.get(store_key) {
                result.insert(
                    weight_name.clone(),
                    (acc.xtx.clone(), acc.n_features, acc.sample_count),
                );
            }
        }
        result
    }
}

impl ActivationHook for GptqActivationStore {
    fn on_linear_input(&mut self, layer: usize, name: &str, activation: &[f16], n_features: usize) {
        if activation.is_empty() || n_features == 0 {
            return;
        }
        let key = store_key(layer, name);
        let acc = self
            .hessians
            .entry(key)
            .or_insert_with(|| HessianAccumulator::new(n_features));
        debug_assert_eq!(
            acc.n_features, n_features,
            "n_features changed between calls"
        );
        acc.accumulate(activation, n_features);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16s(vals: &[f32]) -> Vec<f16> {
        vals.iter().copied().map(f16::from_f32).collect()
    }

    #[test]
    fn hessian_single_token_identity() {
        // X = [[1, 0], [0, 1]]  →  X^T X = [[1, 0], [0, 1]]
        let mut acc = HessianAccumulator::new(2);
        acc.accumulate(&f16s(&[1.0, 0.0, 0.0, 1.0]), 2);

        assert_eq!(acc.sample_count, 2);
        assert!((acc.xtx[0] - 1.0).abs() < 1e-2); // (0,0)
        assert!((acc.xtx[1]).abs() < 1e-2); // (0,1)
        assert!((acc.xtx[2]).abs() < 1e-2); // (1,0)
        assert!((acc.xtx[3] - 1.0).abs() < 1e-2); // (1,1)
    }

    #[test]
    fn hessian_single_row() {
        // X = [[2, 3]]  →  X^T X = [[4, 6], [6, 9]]
        let mut acc = HessianAccumulator::new(2);
        acc.accumulate(&f16s(&[2.0, 3.0]), 2);

        assert_eq!(acc.sample_count, 1);
        assert!((acc.xtx[0] - 4.0).abs() < 1e-2);
        assert!((acc.xtx[1] - 6.0).abs() < 1e-2);
        assert!((acc.xtx[2] - 6.0).abs() < 1e-2); // symmetric
        assert!((acc.xtx[3] - 9.0).abs() < 1e-2);
    }

    #[test]
    fn hessian_streaming_equals_batch() {
        let batch = f16s(&[1.0, 2.0, 3.0, 4.0]); // 2 tokens × 2 features

        // All at once.
        let mut batch_acc = HessianAccumulator::new(2);
        batch_acc.accumulate(&batch, 2);

        // One token at a time.
        let mut stream_acc = HessianAccumulator::new(2);
        stream_acc.accumulate(&f16s(&[1.0, 2.0]), 2);
        stream_acc.accumulate(&f16s(&[3.0, 4.0]), 2);

        assert_eq!(batch_acc.sample_count, stream_acc.sample_count);
        for (a, b) in batch_acc.xtx.iter().zip(stream_acc.xtx.iter()) {
            assert!((a - b).abs() < 1e-2, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn hessian_symmetry() {
        let mut acc = HessianAccumulator::new(3);
        acc.accumulate(&f16s(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 3);

        let n = 3;
        for i in 0..n {
            for j in 0..n {
                let diff = (acc.xtx[i * n + j] - acc.xtx[j * n + i]).abs();
                assert!(diff < 1e-6, "not symmetric at ({i},{j})");
            }
        }
    }

    #[test]
    fn gptq_store_via_hook_trait() {
        let mut store = GptqActivationStore::new();
        // First call: 1 token, 3 features
        store.on_linear_input(0, "q_proj", &f16s(&[1.0, 2.0, 3.0]), 3);

        assert!(store.hessians.contains_key("layer_0_q_proj"));
        let acc = &store.hessians["layer_0_q_proj"];
        assert_eq!(acc.n_features, 3);
        assert_eq!(acc.sample_count, 1);

        // Second call: 2 tokens, 3 features
        store.on_linear_input(0, "q_proj", &f16s(&[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]), 3);
        let acc = &store.hessians["layer_0_q_proj"];
        assert_eq!(acc.sample_count, 3);
    }

    #[test]
    fn gptq_store_ignores_empty_activation() {
        let mut store = GptqActivationStore::new();
        store.on_linear_input(0, "q_proj", &[], 0);
        assert!(store.hessians.is_empty());
    }

    #[test]
    #[should_panic(expected = "not a multiple")]
    fn hessian_bad_shape() {
        let mut acc = HessianAccumulator::new(3);
        acc.accumulate(&f16s(&[1.0, 2.0]), 3);
    }

    #[test]
    fn to_pass_format_maps_hessians_via_key_map() {
        let mut store = GptqActivationStore::new();
        // 1 token × 2 features: X = [[2, 3]]  →  X^T X = [[4, 6], [6, 9]]
        store.on_linear_input(0, "q_proj", &f16s(&[2.0, 3.0]), 2);

        let mut key_map = HashMap::new();
        key_map.insert(
            "layers.0.q_proj.weight".to_string(),
            "layer_0_q_proj".to_string(),
        );
        key_map.insert(
            "layers.0.missing.weight".to_string(),
            "layer_99_q_proj".to_string(),
        );

        let result = store.to_pass_format(&key_map);
        assert_eq!(result.len(), 1);
        assert!(result.contains_key("layers.0.q_proj.weight"));
        assert!(!result.contains_key("layers.0.missing.weight"));

        let (xtx, n_features, sample_count) = &result["layers.0.q_proj.weight"];
        assert_eq!(*n_features, 2);
        assert_eq!(*sample_count, 1);
        assert_eq!(xtx.len(), 4);
        assert!((xtx[0] - 4.0).abs() < 1e-2);
        assert!((xtx[1] - 6.0).abs() < 1e-2);
        assert!((xtx[2] - 6.0).abs() < 1e-2);
        assert!((xtx[3] - 9.0).abs() < 1e-2);
    }
}
