//! Configuration for the Metal GPU inference backend.

/// Cross-Layer Attention configuration.
///
/// When configured, only anchor layers maintain their own KV cache buffers.
/// Non-anchor layers reuse the KV cache of the nearest preceding anchor,
/// reducing KV memory by up to 2×.
#[derive(Debug, Clone)]
pub struct ClaConfig {
    /// Which layers are "anchor" layers that store their own KV cache.
    /// Non-anchor layers reuse the KV cache of the nearest preceding anchor.
    /// Must be sorted in ascending order and non-empty.
    pub anchor_layers: Vec<usize>,
}

impl ClaConfig {
    /// Returns true if `layer` is an anchor layer.
    pub fn is_anchor(&self, layer: usize) -> bool {
        self.anchor_layers.binary_search(&layer).is_ok()
    }

    /// Validate the CLA configuration against the model's layer count.
    pub fn validate(&self, num_layers: usize) -> Result<(), String> {
        if self.anchor_layers.is_empty() {
            return Err("anchor_layers must not be empty".to_string());
        }
        // Must be sorted and deduplicated.
        for w in self.anchor_layers.windows(2) {
            if w[0] >= w[1] {
                return Err(format!(
                    "anchor_layers must be sorted and unique, found {} before {}",
                    w[0], w[1]
                ));
            }
        }
        // All indices must be valid layer indices.
        if let Some(&last) = self.anchor_layers.last() {
            if last >= num_layers {
                return Err(format!(
                    "anchor layer index {} >= num_layers {}",
                    last, num_layers
                ));
            }
        }
        // Layer 0 must be an anchor (first layer has no preceding anchor).
        if self.anchor_layers[0] != 0 {
            return Err("layer 0 must be an anchor layer".to_string());
        }
        Ok(())
    }
}

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
    /// Cross-Layer Attention config. None = all layers are anchors (standard behavior).
    pub cla_config: Option<ClaConfig>,
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
            cla_config: None,
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
