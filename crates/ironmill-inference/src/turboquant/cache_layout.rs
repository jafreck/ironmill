//! Shared buffer-size calculations for TurboQuant KV cache allocation.
//!
//! The Metal backend needs a set of buffers with specific dimensions.
//! [`TurboQuantCacheLayout`] centralises the arithmetic
//! so the backend only has to map sizes → its own buffer type.

use std::fmt;

use super::outlier::OutlierConfig;

/// Error returned when buffer-size arithmetic overflows `usize`.
#[derive(Debug, Clone)]
pub struct CacheLayoutOverflow {
    detail: &'static str,
}

impl fmt::Display for CacheLayoutOverflow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TurboQuant cache layout overflow: {}", self.detail)
    }
}

impl std::error::Error for CacheLayoutOverflow {}

/// Multiply two `usize` values via `u64`, returning an error on overflow.
fn checked_mul(a: usize, b: usize, ctx: &'static str) -> Result<usize, CacheLayoutOverflow> {
    let result = (a as u64)
        .checked_mul(b as u64)
        .ok_or(CacheLayoutOverflow { detail: ctx })?;
    usize::try_from(result).map_err(|_| CacheLayoutOverflow { detail: ctx })
}

/// Multiply three `usize` values via `u64`, returning an error on overflow.
fn checked_mul3(
    a: usize,
    b: usize,
    c: usize,
    ctx: &'static str,
) -> Result<usize, CacheLayoutOverflow> {
    let result = (a as u64)
        .checked_mul(b as u64)
        .and_then(|v| v.checked_mul(c as u64))
        .ok_or(CacheLayoutOverflow { detail: ctx })?;
    usize::try_from(result).map_err(|_| CacheLayoutOverflow { detail: ctx })
}

/// Precomputed buffer sizes for one layer of a TurboQuant KV cache.
///
/// Construct via [`TurboQuantCacheLayout::new`] from model parameters;
/// then use the public fields when allocating Metal buffers.
#[derive(Debug, Clone)]
pub struct TurboQuantCacheLayout {
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
    ///
    /// Returns an error if the resulting buffer sizes would overflow `usize`.
    pub fn new(
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        _num_layers: usize,
        n_bits: u8,
        outlier: Option<&OutlierConfig>,
    ) -> Result<Self, CacheLayoutOverflow> {
        let bytes_per_pos = checked_mul(
            head_dim,
            n_bits as usize,
            "bytes_per_pos (head_dim * n_bits)",
        )? / 8;
        let cache_bytes = checked_mul3(num_kv_heads, max_seq_len, bytes_per_pos, "cache_bytes")?;
        let scale_count = checked_mul(num_kv_heads, max_seq_len, "scale_count")?;
        let qjl_signs_bytes =
            checked_mul3(num_kv_heads, max_seq_len, head_dim / 8, "qjl_signs_bytes")?;
        let outlier_layout = match outlier {
            Some(cfg) => {
                let n_outlier = cfg.outlier_channels.len();
                let d_o_padded = if n_outlier == 0 {
                    0
                } else {
                    n_outlier.next_power_of_two()
                };
                let d_n = head_dim.saturating_sub(n_outlier);
                let d_n_padded = if d_n == 0 { 0 } else { d_n.next_power_of_two() };
                Some(OutlierCacheLayout {
                    outlier_cache_bytes: if d_o_padded == 0 {
                        0
                    } else {
                        checked_mul3(
                            num_kv_heads,
                            max_seq_len,
                            d_o_padded / 2,
                            "outlier_cache_bytes",
                        )?
                    },
                    non_outlier_cache_bytes: if d_n_padded == 0 {
                        0
                    } else {
                        checked_mul3(
                            num_kv_heads,
                            max_seq_len,
                            d_n_padded / 2,
                            "non_outlier_cache_bytes",
                        )?
                    },
                })
            }
            None => None,
        };

        Ok(Self {
            cache_bytes,
            scale_count,
            qjl_signs_bytes,
            outlier: outlier_layout,
        })
    }

    /// Byte size of a scale / norm buffer (for backends that allocate raw bytes).
    pub fn scale_bytes(&self) -> usize {
        self.scale_count * std::mem::size_of::<f32>()
    }
}
