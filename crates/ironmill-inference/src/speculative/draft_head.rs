//! EAGLE-3 draft MLP with multi-layer feature fusion.
//!
//! The [`DraftHead`] is a lightweight MLP that projects fused hidden states
//! from the target model into draft logits. Weights are stored in `f16` and
//! promoted to `f32` for the forward pass (no GPU required — the draft head
//! is intentionally cheap to evaluate on the CPU).

use half::f16;

/// EAGLE-3 draft MLP that produces speculative token probabilities by
/// fusing features from multiple target-model layers.
#[derive(Debug)]
pub struct DraftHead {
    /// Projection weight matrices, one per fused layer.
    /// Each matrix is `[hidden_dim × vocab_size]` stored row-major in `f16`.
    pub(crate) layer_weights: Vec<Vec<f16>>,
    /// Hidden dimension of the target model.
    pub(crate) hidden_dim: usize,
    /// Vocabulary size (number of output logits).
    pub(crate) vocab_size: usize,
}

impl DraftHead {
    /// Create a new draft head with the given projection weights.
    ///
    /// Each entry in `layer_weights` is a flattened `[hidden_dim × vocab_size]`
    /// weight matrix for one fused layer.
    ///
    /// # Panics
    ///
    /// Panics if any weight matrix length does not equal `hidden_dim * vocab_size`.
    pub fn new(layer_weights: Vec<Vec<f16>>, hidden_dim: usize, vocab_size: usize) -> Self {
        let expected = hidden_dim * vocab_size;
        for (i, w) in layer_weights.iter().enumerate() {
            assert_eq!(
                w.len(),
                expected,
                "layer_weights[{i}] length {} != hidden_dim({hidden_dim}) * vocab_size({vocab_size}) = {expected}",
                w.len(),
            );
        }
        Self {
            layer_weights,
            hidden_dim,
            vocab_size,
        }
    }

    /// Run the draft head forward pass on fused hidden states.
    ///
    /// `fused_hidden` is the sum (or mean) of hidden-state vectors extracted
    /// from the target model's fused layers, with length `hidden_dim`.
    ///
    /// Returns logits of length `vocab_size`.
    pub fn forward(&self, fused_hidden: &[f32]) -> Vec<f32> {
        assert_eq!(
            fused_hidden.len(),
            self.hidden_dim,
            "fused_hidden length {} != hidden_dim {}",
            fused_hidden.len(),
            self.hidden_dim,
        );

        // Average the projections across all fused layers.
        let mut logits = vec![0.0f32; self.vocab_size];
        let num_layers = self.layer_weights.len();
        if num_layers == 0 {
            return logits;
        }

        for layer_w in &self.layer_weights {
            for (v, logit) in logits.iter_mut().enumerate().take(self.vocab_size) {
                let mut dot = 0.0f32;
                let row_offset = v * self.hidden_dim;
                for h in 0..self.hidden_dim {
                    dot += fused_hidden[h] * f16::to_f32(layer_w[row_offset + h]);
                }
                *logit += dot;
            }
        }

        // Average across fused layers.
        let scale = 1.0 / num_layers as f32;
        for l in &mut logits {
            *l *= scale;
        }

        logits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_head(hidden_dim: usize, vocab_size: usize) -> DraftHead {
        // Identity-ish weights: each layer weight is a simple pattern.
        let weights: Vec<f16> = (0..hidden_dim * vocab_size)
            .map(|i| f16::from_f32((i % 7) as f32 * 0.1 - 0.3))
            .collect();
        DraftHead::new(vec![weights], hidden_dim, vocab_size)
    }

    #[test]
    fn speculative_draft_head_forward_shape() {
        let head = make_test_head(4, 8);
        let hidden = vec![1.0f32; 4];
        let logits = head.forward(&hidden);
        assert_eq!(logits.len(), 8);
    }

    #[test]
    fn speculative_draft_head_zero_hidden() {
        let head = make_test_head(4, 8);
        let hidden = vec![0.0f32; 4];
        let logits = head.forward(&hidden);
        assert!(logits.iter().all(|&l| l == 0.0));
    }

    #[test]
    fn speculative_draft_head_multi_layer_averages() {
        let hidden_dim = 2;
        let vocab_size = 3;
        // Layer 0: all 1.0
        let w0 = vec![f16::from_f32(1.0); hidden_dim * vocab_size];
        // Layer 1: all 3.0
        let w1 = vec![f16::from_f32(3.0); hidden_dim * vocab_size];
        let head = DraftHead::new(vec![w0, w1], hidden_dim, vocab_size);

        let hidden = vec![1.0f32; hidden_dim];
        let logits = head.forward(&hidden);
        // Each vocab position: layer0 dot = 1*1 + 1*1 = 2, layer1 dot = 1*3 + 1*3 = 6
        // Average: (2 + 6) / 2 = 4.0
        for &l in &logits {
            assert!((l - 4.0).abs() < 1e-2, "expected ~4.0, got {l}");
        }
    }

    #[test]
    #[should_panic(expected = "layer_weights[0] length")]
    fn speculative_draft_head_wrong_weight_size() {
        DraftHead::new(vec![vec![f16::from_f32(0.0); 10]], 4, 8);
    }

    #[test]
    #[should_panic(expected = "fused_hidden length")]
    fn speculative_draft_head_wrong_hidden_size() {
        let head = make_test_head(4, 8);
        head.forward(&[1.0, 2.0]); // wrong length
    }
}
