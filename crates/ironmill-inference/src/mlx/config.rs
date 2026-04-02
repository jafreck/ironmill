//! Configuration for the MLX inference backend.

/// Configuration for the MLX inference backend.
///
/// Controls sequence length limits, prefill chunking, and TurboQuant
/// quantized KV cache settings (disabled by default — activated by the
/// MLX-TURBOQUANT task later).
#[derive(Debug, Clone)]
pub struct MlxConfig {
    /// Maximum sequence length for KV cache allocation.
    pub max_seq_len: usize,
    /// Prefill chunk size. `None` = process entire prompt at once.
    pub prefill_chunk_size: Option<usize>,
    /// Enable TurboQuant quantized KV cache compression.
    pub enable_turboquant: bool,
    /// TurboQuant rotation seed (must match between cache write and attention).
    pub rotation_seed: u64,
    /// Number of quantization bits for TurboQuant KV cache (4 or 8).
    pub n_bits: u8,
}

impl Default for MlxConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 2048,
            prefill_chunk_size: None,
            enable_turboquant: false,
            rotation_seed: 42,
            n_bits: 8,
        }
    }
}

impl MlxConfig {
    /// Validate configuration. Returns an error message if invalid.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_seq_len == 0 {
            return Err("max_seq_len must be > 0".into());
        }
        if self.n_bits != 4 && self.n_bits != 8 {
            return Err(format!("n_bits must be 4 or 8, got {}", self.n_bits));
        }
        if let Some(chunk) = self.prefill_chunk_size {
            if chunk == 0 {
                return Err("prefill_chunk_size must be > 0".into());
            }
        }
        Ok(())
    }
}
