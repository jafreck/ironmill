//! Shared buffer-size calculations for TurboQuant KV cache allocation.
//!
//! Both the Metal and MLX backends need the same set of buffers with the
//! same dimensions.  [`TurboQuantCacheLayout`] centralises the arithmetic
//! so each backend only has to map sizes → its own buffer type.

use super::outlier::OutlierConfig;

/// Precomputed buffer sizes for one layer of a TurboQuant KV cache.
///
/// Construct via [`TurboQuantCacheLayout::new`] from model parameters;
/// then use the public fields when allocating Metal buffers or MLX arrays.
#[derive(Debug, Clone)]
pub struct TurboQuantCacheLayout {
    /// Number of KV heads.
    pub num_kv_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Maximum sequence length the cache can hold.
    pub max_seq_len: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Quantization bit-width (e.g. 4 or 8).
    pub n_bits: u8,

    // ── Per-layer sizes ────────────────────────────────────────────
    /// Byte size of one K or V cache buffer (packed uint8).
    /// `num_kv_heads × max_seq_len × elements_per_pos`
    pub cache_bytes: usize,

    /// Element count for a scale / norm array (one f32 per head per position).
    /// `num_kv_heads × max_seq_len`
    pub scale_count: usize,

    /// Byte size of the QJL sign-bit buffer (K cache only).
    /// `num_kv_heads × max_seq_len × (head_dim / 8)`
    pub qjl_signs_bytes: usize,

    /// Outlier-split sizes, present only when outlier mode is enabled.
    pub outlier: Option<OutlierCacheLayout>,
}

/// Per-layer sizes for the outlier / non-outlier split buffers.
#[derive(Debug, Clone)]
pub struct OutlierCacheLayout {
    /// Byte size of the outlier-group cache buffer.
    pub outlier_cache_bytes: usize,
    /// Byte size of the non-outlier-group cache buffer.
    pub non_outlier_cache_bytes: usize,
}

impl TurboQuantCacheLayout {
    /// Compute the cache layout from model parameters.
    pub fn new(
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        num_layers: usize,
        n_bits: u8,
        outlier: Option<&OutlierConfig>,
    ) -> Self {
        let bytes_per_pos = head_dim * (n_bits as usize) / 8;
        let cache_bytes = num_kv_heads * max_seq_len * bytes_per_pos;
        let scale_count = num_kv_heads * max_seq_len;
        let qjl_signs_bytes = num_kv_heads * max_seq_len * (head_dim / 8);

        let outlier_layout = outlier.map(|cfg| {
            let d_o_padded = cfg.outlier_channels.len().next_power_of_two();
            let d_n = head_dim - cfg.outlier_channels.len();
            let d_n_padded = d_n.next_power_of_two();
            OutlierCacheLayout {
                outlier_cache_bytes: num_kv_heads * max_seq_len * (d_o_padded / 2),
                non_outlier_cache_bytes: num_kv_heads * max_seq_len * (d_n_padded / 2),
            }
        });

        Self {
            num_kv_heads,
            head_dim,
            max_seq_len,
            num_layers,
            n_bits,
            cache_bytes,
            scale_count,
            qjl_signs_bytes,
            outlier: outlier_layout,
        }
    }

    /// Byte size of a scale / norm buffer (for backends that allocate raw bytes).
    pub fn scale_bytes(&self) -> usize {
        self.scale_count * std::mem::size_of::<f32>()
    }
}
