//! AWQ-style activation store: lightweight per-channel magnitude statistics.

use std::collections::HashMap;

use half::f16;

use super::hook::{ActivationHook, store_key};

/// Per-channel magnitude statistics accumulated over calibration tokens.
#[derive(Debug, Clone)]
pub struct ChannelMagnitudes {
    /// Running mean of |x| per channel.
    pub mean_abs: Vec<f32>,
    /// Running max of |x| per channel.
    pub max_abs: Vec<f32>,
    /// Total number of tokens (rows) observed so far.
    pub sample_count: usize,
}

impl ChannelMagnitudes {
    /// Create a zero-initialised accumulator for `n_features` channels.
    pub fn new(n_features: usize) -> Self {
        Self {
            mean_abs: vec![0.0; n_features],
            max_abs: vec![0.0; n_features],
            sample_count: 0,
        }
    }

    /// Incrementally update statistics from a flattened activation slice.
    ///
    /// `activation` has length `n_tokens × n_features` in row-major order.
    /// The update uses a numerically-stable running mean.
    pub fn update(&mut self, activation: &[f16], n_features: usize) {
        assert!(
            !activation.is_empty() && activation.len() % n_features == 0,
            "activation length {} is not a multiple of n_features {}",
            activation.len(),
            n_features,
        );
        let n_tokens = activation.len() / n_features;

        // Accumulate sum-of-abs and max-of-abs for each channel across all
        // tokens in this batch.
        let mut batch_sum = vec![0.0f32; n_features];
        let mut batch_max = vec![0.0f32; n_features];

        for t in 0..n_tokens {
            let row = &activation[t * n_features..(t + 1) * n_features];
            for (c, &val) in row.iter().enumerate() {
                let abs_val = val.to_f32().abs();
                batch_sum[c] += abs_val;
                if abs_val > batch_max[c] {
                    batch_max[c] = abs_val;
                }
            }
        }

        let old_count = self.sample_count as f32;
        let new_count = (self.sample_count + n_tokens) as f32;

        for c in 0..n_features {
            // Weighted combination of old running mean and new batch mean.
            self.mean_abs[c] = (self.mean_abs[c] * old_count + batch_sum[c]) / new_count;
            if batch_max[c] > self.max_abs[c] {
                self.max_abs[c] = batch_max[c];
            }
        }

        self.sample_count += n_tokens;
    }
}

/// Activation store for AWQ-style quantisation.
///
/// Accumulates per-channel magnitude statistics for each linear projection
/// in each transformer layer. Memory cost is O(n_features) per projection
/// plus one raw activation snapshot per capture point.
#[derive(Debug, Clone)]
pub struct AwqActivationStore {
    /// Maps `"layer_{i}_{name}"` → per-channel magnitude statistics.
    pub magnitudes: HashMap<String, ChannelMagnitudes>,
    /// Raw activation snapshots: key → list of flattened `[n_tokens × n_features]` tensors.
    /// Only the first snapshot per key is kept to bound memory usage.
    pub activations: HashMap<String, Vec<Vec<f32>>>,
    /// Maximum number of activation snapshots to keep per key (default 1).
    pub max_snapshots_per_key: usize,
}

impl AwqActivationStore {
    pub fn new() -> Self {
        Self {
            magnitudes: HashMap::new(),
            activations: HashMap::new(),
            max_snapshots_per_key: 1,
        }
    }
}

impl Default for AwqActivationStore {
    fn default() -> Self {
        Self::new()
    }
}

impl AwqActivationStore {
    /// Convert to the format expected by `AwqQuantizePass`.
    ///
    /// The Metal hook captures activations at norm outputs (`"layer_{i}_attn_norm"`,
    /// `"layer_{i}_ffn_norm"`). The MIL pass needs per-weight-op magnitudes.
    /// Since each norm output feeds multiple projections (attn_norm → Q/K/V/O,
    /// ffn_norm → gate/up/down), we duplicate the magnitudes for each projection.
    ///
    /// `weight_names` maps const-op output names to (layer_index, projection_group).
    /// projection_group is `"attn"` or `"ffn"`.
    pub fn to_channel_magnitudes(
        &self,
        weight_names: &[(String, usize, &str)],
    ) -> HashMap<String, Vec<f32>> {
        let mut result = HashMap::new();
        for (name, layer_idx, group) in weight_names {
            let store_key = format!("layer_{}_{}_norm", layer_idx, group);
            if let Some(mag) = self.magnitudes.get(&store_key) {
                result.insert(name.clone(), mag.mean_abs.clone());
            }
        }
        result
    }

    /// Simpler conversion: map all stored magnitudes using a key transform.
    /// Each stored key `"layer_{i}_{name}"` gets mapped to all weight op names
    /// that match the layer. Caller provides the mapping.
    pub fn to_pass_format(&self, key_map: &HashMap<String, String>) -> HashMap<String, Vec<f32>> {
        let mut result = HashMap::new();
        for (weight_name, store_key) in key_map {
            if let Some(mag) = self.magnitudes.get(store_key) {
                result.insert(weight_name.clone(), mag.mean_abs.clone());
            }
        }
        result
    }

    /// Convert raw activations to the format expected by `AwqQuantizePass`.
    ///
    /// Uses the same norm→projection mapping as `to_channel_magnitudes`:
    /// each norm key is duplicated across all projections it feeds.
    /// Returns only the first captured activation snapshot per weight op.
    pub fn to_activations(
        &self,
        weight_names: &[(String, usize, &str)],
    ) -> HashMap<String, Vec<f32>> {
        let mut result = HashMap::new();
        for (name, layer_idx, group) in weight_names {
            let store_key = format!("layer_{}_{}_norm", layer_idx, group);
            if let Some(snapshots) = self.activations.get(&store_key) {
                if let Some(first) = snapshots.first() {
                    result.insert(name.clone(), first.clone());
                }
            }
        }
        result
    }
}

impl ActivationHook for AwqActivationStore {
    fn on_linear_input(&mut self, layer: usize, name: &str, activation: &[f16], n_features: usize) {
        if activation.is_empty() || n_features == 0 {
            return;
        }
        let key = store_key(layer, name);
        let mag = self
            .magnitudes
            .entry(key.clone())
            .or_insert_with(|| ChannelMagnitudes::new(n_features));
        debug_assert_eq!(
            mag.mean_abs.len(),
            n_features,
            "n_features changed between calls"
        );
        mag.update(activation, n_features);

        // Capture raw activation snapshot (bounded by max_snapshots_per_key).
        let snapshots = self.activations.entry(key).or_default();
        if snapshots.len() < self.max_snapshots_per_key {
            let f32_data: Vec<f32> = activation.iter().map(|v| v.to_f32()).collect();
            snapshots.push(f32_data);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16s(vals: &[f32]) -> Vec<f16> {
        vals.iter().copied().map(f16::from_f32).collect()
    }

    #[test]
    fn channel_magnitudes_single_token() {
        let mut mag = ChannelMagnitudes::new(3);
        let act = f16s(&[1.0, -2.0, 3.0]);
        mag.update(&act, 3);

        assert_eq!(mag.sample_count, 1);
        assert!((mag.mean_abs[0] - 1.0).abs() < 1e-2);
        assert!((mag.mean_abs[1] - 2.0).abs() < 1e-2);
        assert!((mag.mean_abs[2] - 3.0).abs() < 1e-2);
        assert!((mag.max_abs[0] - 1.0).abs() < 1e-2);
        assert!((mag.max_abs[1] - 2.0).abs() < 1e-2);
        assert!((mag.max_abs[2] - 3.0).abs() < 1e-2);
    }

    #[test]
    fn channel_magnitudes_multi_token() {
        let mut mag = ChannelMagnitudes::new(2);
        // Two tokens: [1, -4], [3, 2]
        let act = f16s(&[1.0, -4.0, 3.0, 2.0]);
        mag.update(&act, 2);

        assert_eq!(mag.sample_count, 2);
        // mean_abs[0] = (1+3)/2 = 2.0, mean_abs[1] = (4+2)/2 = 3.0
        assert!((mag.mean_abs[0] - 2.0).abs() < 1e-2);
        assert!((mag.mean_abs[1] - 3.0).abs() < 1e-2);
        // max_abs[0] = 3, max_abs[1] = 4
        assert!((mag.max_abs[0] - 3.0).abs() < 1e-2);
        assert!((mag.max_abs[1] - 4.0).abs() < 1e-2);
    }

    #[test]
    fn channel_magnitudes_streaming() {
        let mut mag = ChannelMagnitudes::new(2);

        // First batch: single token [2, 6]
        mag.update(&f16s(&[2.0, 6.0]), 2);
        assert_eq!(mag.sample_count, 1);
        assert!((mag.mean_abs[0] - 2.0).abs() < 1e-2);
        assert!((mag.mean_abs[1] - 6.0).abs() < 1e-2);

        // Second batch: single token [4, 2]
        mag.update(&f16s(&[4.0, 2.0]), 2);
        assert_eq!(mag.sample_count, 2);
        // mean_abs[0] = (2+4)/2 = 3.0, mean_abs[1] = (6+2)/2 = 4.0
        assert!((mag.mean_abs[0] - 3.0).abs() < 1e-2);
        assert!((mag.mean_abs[1] - 4.0).abs() < 1e-2);
        // max_abs[0] = 4, max_abs[1] = 6
        assert!((mag.max_abs[0] - 4.0).abs() < 1e-2);
        assert!((mag.max_abs[1] - 6.0).abs() < 1e-2);
    }

    #[test]
    fn awq_store_via_hook_trait() {
        let mut store = AwqActivationStore::new();

        // 1 token × 2 features
        let act1 = f16s(&[1.0, 2.0]);
        store.on_linear_input(0, "q_proj", &act1, 2);

        assert!(store.magnitudes.contains_key("layer_0_q_proj"));
        let mag = &store.magnitudes["layer_0_q_proj"];
        assert_eq!(mag.sample_count, 1);
        assert_eq!(mag.mean_abs.len(), 2);

        // 2 tokens × 2 features
        let act = f16s(&[1.0, 2.0, 3.0, 4.0]);
        store.on_linear_input(0, "q_proj", &act, 2);
        let mag = &store.magnitudes["layer_0_q_proj"];
        assert_eq!(mag.sample_count, 3); // 1 + 2
    }

    #[test]
    fn awq_store_multiple_projections() {
        let mut store = AwqActivationStore::new();
        store.on_linear_input(0, "q_proj", &f16s(&[1.0, 2.0]), 2);
        store.on_linear_input(0, "k_proj", &f16s(&[3.0, 4.0, 5.0]), 3);
        store.on_linear_input(1, "q_proj", &f16s(&[6.0]), 1);

        assert_eq!(store.magnitudes.len(), 3);
        assert!(store.magnitudes.contains_key("layer_0_q_proj"));
        assert!(store.magnitudes.contains_key("layer_0_k_proj"));
        assert!(store.magnitudes.contains_key("layer_1_q_proj"));
    }

    #[test]
    #[should_panic(expected = "not a multiple")]
    fn channel_magnitudes_bad_shape() {
        let mut mag = ChannelMagnitudes::new(3);
        mag.update(&f16s(&[1.0, 2.0]), 3); // 2 is not a multiple of 3
    }

    #[test]
    fn awq_store_ignores_empty_activation() {
        let mut store = AwqActivationStore::new();
        store.on_linear_input(0, "q_proj", &[], 0);
        assert!(store.magnitudes.is_empty());
    }

    #[test]
    fn to_channel_magnitudes_maps_norm_to_projections() {
        let mut store = AwqActivationStore::new();
        store.on_linear_input(0, "attn_norm", &f16s(&[1.0, 2.0]), 2);
        store.on_linear_input(0, "ffn_norm", &f16s(&[3.0, 4.0]), 2);

        let weight_names = vec![
            ("layers.0.self_attn.q_proj.weight".into(), 0, "attn"),
            ("layers.0.self_attn.k_proj.weight".into(), 0, "attn"),
            ("layers.0.mlp.gate_proj.weight".into(), 0, "ffn"),
            ("layers.1.self_attn.q_proj.weight".into(), 1, "attn"), // no data
        ];

        let result = store.to_channel_magnitudes(&weight_names);
        assert_eq!(result.len(), 3);
        assert!(result.contains_key("layers.0.self_attn.q_proj.weight"));
        assert!(result.contains_key("layers.0.self_attn.k_proj.weight"));
        assert!(result.contains_key("layers.0.mlp.gate_proj.weight"));
        assert!(!result.contains_key("layers.1.self_attn.q_proj.weight"));

        // attn_norm and q_proj should share the same magnitudes
        let q = &result["layers.0.self_attn.q_proj.weight"];
        let k = &result["layers.0.self_attn.k_proj.weight"];
        assert_eq!(q, k);
        assert!((q[0] - 1.0).abs() < 1e-2);
        assert!((q[1] - 2.0).abs() < 1e-2);

        let gate = &result["layers.0.mlp.gate_proj.weight"];
        assert!((gate[0] - 3.0).abs() < 1e-2);
        assert!((gate[1] - 4.0).abs() < 1e-2);
    }

    #[test]
    fn to_pass_format_maps_via_key_map() {
        let mut store = AwqActivationStore::new();
        store.on_linear_input(0, "attn_norm", &f16s(&[5.0, 6.0]), 2);

        let mut key_map = HashMap::new();
        key_map.insert("weight_op_q".to_string(), "layer_0_attn_norm".to_string());
        key_map.insert(
            "weight_op_missing".to_string(),
            "layer_99_attn_norm".to_string(),
        );

        let result = store.to_pass_format(&key_map);
        assert_eq!(result.len(), 1);
        assert!(result.contains_key("weight_op_q"));
        assert!(!result.contains_key("weight_op_missing"));

        let mags = &result["weight_op_q"];
        assert!((mags[0] - 5.0).abs() < 1e-2);
        assert!((mags[1] - 6.0).abs() < 1e-2);
    }
}
