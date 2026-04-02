//! Outlier channel detection for TurboQuant mixed-precision quantization.
//!
//! Implements Section 4.3 of the TurboQuant paper: channels with high
//! post-rotation variance are quantized at higher bit-width while the
//! remaining channels use fewer bits.

/// Outlier channel configuration for mixed-precision quantization.
#[derive(Debug, Clone)]
pub struct OutlierConfig {
    /// Indices of outlier channels in the original KV dimension space.
    pub outlier_channels: Vec<usize>,
    /// Quantization bits for outlier channels (default: 4).
    pub outlier_bits: u8,
    /// Quantization bits for non-outlier channels (default: 3).
    pub non_outlier_bits: u8,
}

impl OutlierConfig {
    /// Detect outlier channels from K/V projection weight column norms.
    ///
    /// Paper §4.3: channels with high variance after rotation are outliers.
    /// Since the Hadamard rotation is orthogonal, the post-rotation variance
    /// of channel `i` is proportional to the squared column norm of the
    /// projection weight matrix at that channel index. We select the top
    /// `n_outlier` channels by column norm across all layers.
    ///
    /// `weight_data` is a slice of (K weight, V weight) per layer, each as
    /// row-major f16 bytes with shape [out_features × in_features].
    /// `head_dim` is the per-head dimension (= out_features / num_kv_heads).
    /// `n_outlier` is how many channels to flag (paper uses d/4 = 32 for d=128).
    pub fn from_weight_norms(
        weight_data: &[(&[u8], &[u8])],
        out_features: usize,
        head_dim: usize,
        n_outlier: usize,
        outlier_bits: u8,
        non_outlier_bits: u8,
    ) -> Self {
        // Accumulate per-channel (per-head-dim-index) squared norms across
        // all KV heads and layers. Channel index is dim % head_dim.
        let mut channel_energy = vec![0.0f64; head_dim];

        for (k_bytes, v_bytes) in weight_data {
            for weight_bytes in [k_bytes, v_bytes] {
                // Weight is [out_features × in_features] stored as f16, row-major.
                // Column `j` contributes to channel `j % head_dim`.
                // The norm of column `j` indicates the energy of that output dimension.
                let n_rows = weight_bytes.len() / (out_features * 2);
                if n_rows == 0 {
                    continue;
                }
                for out_idx in 0..out_features {
                    let ch = out_idx % head_dim;
                    // Compute squared norm of row `out_idx` as a proxy
                    // (row norm = how much this output channel depends on inputs).
                    let row_start = out_idx * n_rows * 2;
                    let row_end = row_start + n_rows * 2;
                    if row_end > weight_bytes.len() {
                        continue;
                    }
                    let mut sq_sum = 0.0f64;
                    for i in (row_start..row_end).step_by(2) {
                        let val = half::f16::from_le_bytes([weight_bytes[i], weight_bytes[i + 1]])
                            .to_f64();
                        sq_sum += val * val;
                    }
                    channel_energy[ch] += sq_sum;
                }
            }
        }

        // Select top n_outlier channels by energy
        let mut indexed: Vec<(usize, f64)> = channel_energy.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut outlier_channels: Vec<usize> =
            indexed[..n_outlier].iter().map(|(i, _)| *i).collect();
        outlier_channels.sort(); // keep in ascending order for index buffer

        Self {
            outlier_channels,
            outlier_bits,
            non_outlier_bits,
        }
    }

    /// Create an outlier config with the default paper settings:
    /// top d/4 channels by weight energy, 4-bit outlier / 3-bit non-outlier.
    pub fn auto_from_weights(
        weight_data: &[(&[u8], &[u8])],
        out_features: usize,
        head_dim: usize,
    ) -> Self {
        Self::from_weight_norms(
            weight_data,
            out_features,
            head_dim,
            head_dim / 4, // d/4 outlier channels (paper default)
            4,            // 4-bit outlier
            3,            // 3-bit non-outlier
        )
    }
}
