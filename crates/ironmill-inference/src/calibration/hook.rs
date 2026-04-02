//! Activation hook trait for calibration data collection.

use half::f16;

/// Hook invoked during calibration forward passes to collect activation
/// statistics without storing the raw tensors.
pub trait ActivationHook {
    /// Called with the input activations at each capture point during calibration.
    ///
    /// * `layer` — transformer layer index (0-based).
    /// * `name` — capture point name. The Metal calibration dispatch provides
    ///   `"attn_norm"` (RMSNorm output feeding Q/K/V projections) and
    ///   `"ffn_norm"` (RMSNorm output feeding gate/up/down projections).
    ///   Other backends may provide per-projection names (e.g. `"q_proj"`).
    /// * `activation` — FP16 input tensor, flattened as `n_tokens × n_features`.
    /// * `n_features` — number of features (columns) per token. Required to
    ///   correctly reshape the flat activation slice.
    fn on_linear_input(&mut self, layer: usize, name: &str, activation: &[f16], n_features: usize);
}

/// Builds the canonical key used by activation stores: `"layer_{i}_{name}"`.
pub(crate) fn store_key(layer: usize, name: &str) -> String {
    format!("layer_{layer}_{name}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_key_format() {
        assert_eq!(store_key(0, "q_proj"), "layer_0_q_proj");
        assert_eq!(store_key(31, "down_proj"), "layer_31_down_proj");
    }

    /// Ensure the trait is object-safe so callers can use `dyn ActivationHook`.
    #[test]
    fn trait_is_object_safe() {
        fn _accepts_dyn(_hook: &mut dyn ActivationHook) {}
    }
}
