//! Configuration for the Metal GPU inference backend.

/// Configuration for the Metal GPU inference backend.
#[derive(Debug, Clone)]
pub struct MetalConfig {
    /// Maximum sequence length for KV cache allocation.
    pub max_seq_len: usize,
    /// Attention tile size (sequence positions per tile in threadgroup memory).
    /// Tunable per chip generation. Default: computed from head_dim.
    pub attention_tile_size: Option<usize>,
    /// Prefill chunk size. None = process entire prompt at once.
    pub prefill_chunk_size: Option<usize>,
    /// Enable TurboQuant quantized KV cache compression.
    pub enable_turboquant: bool,
    /// TurboQuant rotation seed (must match between cache write and attention).
    pub rotation_seed: u64,
    /// Number of quantization bits for TurboQuant KV cache (4 or 8).
    pub n_bits: u8,
    /// When true, always dequantize weights to FP16 on the CPU before
    /// creating Metal buffers (Phase 1 load-time dequant). When false,
    /// keep quantized weights packed in VRAM and use custom kernels.
    pub force_cpu_dequant: bool,
    /// Use FlashAttention-2 style multi-query prefill attention kernel.
    /// Groups multiple query tokens per threadgroup for better KV tile
    /// reuse. Beneficial for large models (7B+) and long sequences; may
    /// be slower for small models where KV tiles fit in L1 cache.
    pub use_fa2_prefill: bool,
}

impl Default for MetalConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 2048,
            attention_tile_size: None,
            prefill_chunk_size: None,
            enable_turboquant: true,
            rotation_seed: 42,
            n_bits: 8,
            force_cpu_dequant: false,
            use_fa2_prefill: false,
        }
    }
}

impl MetalConfig {
    /// Validate configuration. Returns an error message if invalid.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_seq_len == 0 {
            return Err("max_seq_len must be > 0".to_string());
        }
        if self.n_bits != 4 && self.n_bits != 8 {
            return Err(format!("n_bits must be 4 or 8, got {}", self.n_bits));
        }
        Ok(())
    }
}
