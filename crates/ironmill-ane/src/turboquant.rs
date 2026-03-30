//! TurboQuant INT8 KV cache compression configuration.

use crate::{AneError, Result};

/// Configuration for TurboQuant INT8 KV cache compression.
///
/// Controls runtime KV cache quantization using rotation + Beta-optimal
/// scalar quantization. Storage format is always INT8 (1 byte/element);
/// `n_bits` controls the number of distinct quantization levels within
/// the INT8 range.
pub struct TurboQuantConfig {
    /// Number of quantization bits (1, 2, 4, 6, or 8).
    /// Controls quality via 2^n_bits distinct Beta-optimal levels
    /// mapped into the [-128, 127] INT8 range.
    pub n_bits: u8,
    /// Maximum sequence length for cache allocation.
    pub max_seq_len: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of KV heads (may differ from num_heads for GQA).
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Hadamard rotation seed (deterministic, shared with dequant).
    pub rotation_seed: u64,
    /// Enable QJL 1-bit bias correction.
    pub enable_qjl: bool,
}

const VALID_N_BITS: &[u8] = &[1, 2, 4, 6, 8];

impl TurboQuantConfig {
    /// Create a new TurboQuantConfig, validating parameters.
    pub fn new(
        n_bits: u8,
        max_seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        num_layers: usize,
    ) -> Result<Self> {
        if !VALID_N_BITS.contains(&n_bits) {
            return Err(AneError::Other(anyhow::anyhow!(
                "invalid n_bits {n_bits}: must be one of {VALID_N_BITS:?}"
            )));
        }
        if max_seq_len == 0 {
            return Err(AneError::Other(anyhow::anyhow!("max_seq_len must be > 0")));
        }
        if num_heads == 0 {
            return Err(AneError::Other(anyhow::anyhow!("num_heads must be > 0")));
        }
        if num_kv_heads == 0 {
            return Err(AneError::Other(anyhow::anyhow!("num_kv_heads must be > 0")));
        }
        if head_dim == 0 {
            return Err(AneError::Other(anyhow::anyhow!("head_dim must be > 0")));
        }
        if num_layers == 0 {
            return Err(AneError::Other(anyhow::anyhow!("num_layers must be > 0")));
        }
        if num_heads % num_kv_heads != 0 {
            return Err(AneError::Other(anyhow::anyhow!(
                "num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )));
        }

        Ok(Self {
            n_bits,
            max_seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            num_layers,
            rotation_seed: 42,
            enable_qjl: false,
        })
    }

    /// Enable or disable QJL 1-bit bias correction.
    pub fn with_qjl(mut self, enable: bool) -> Self {
        self.enable_qjl = enable;
        self
    }

    /// Set the Hadamard rotation seed.
    pub fn with_rotation_seed(mut self, seed: u64) -> Self {
        self.rotation_seed = seed;
        self
    }
}
