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

/// Sliding window attention configuration.
///
/// When configured, layers below `max_window_layers` use sliding window
/// attention with a bounded KV cache (ring buffer of `window_size` entries),
/// while layers at or above `max_window_layers` use full attention.
#[derive(Debug, Clone)]
pub struct SlidingWindowConfig {
    /// Number of recent tokens each SWA layer attends to (e.g. 4096).
    pub window_size: usize,
    /// Layers `0..max_window_layers` use sliding window attention;
    /// layers `max_window_layers..num_layers` use full attention.
    pub max_window_layers: usize,
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
    /// Tile size for the fused SDPA kernel (Br: Q block size).
    /// None = auto-select based on head_dim.
    pub fused_sdpa_tile_br: Option<usize>,
    /// Tile size for KV blocks (Bc). None = auto.
    pub fused_sdpa_tile_bc: Option<usize>,
    /// Cross-Layer Attention config. None = all layers are anchors (standard behavior).
    pub cla_config: Option<ClaConfig>,
    /// Sliding window attention config. None = full attention for all layers.
    pub sliding_window: Option<SlidingWindowConfig>,
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
            fused_sdpa_tile_br: None,
            fused_sdpa_tile_bc: None,
            cla_config: None,
            sliding_window: None,
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
        if let Some(ref sw) = self.sliding_window {
            if sw.window_size == 0 {
                return Err("sliding_window.window_size must be > 0".to_string());
            }
        }
        Ok(())
    }

    /// Compute the effective max_seq_len for a given layer, accounting for
    /// sliding window attention. SWA layers use `window_size` as their
    /// buffer stride; full-attention layers use the global `max_seq_len`.
    pub fn effective_max_seq_len(&self, layer: usize) -> usize {
        if let Some(ref sw) = self.sliding_window {
            if layer < sw.max_window_layers {
                return sw.window_size;
            }
        }
        self.max_seq_len
    }

    /// Returns the sliding window size for a layer, or 0 for full attention.
    pub fn layer_window_size(&self, layer: usize) -> usize {
        if let Some(ref sw) = self.sliding_window {
            if layer < sw.max_window_layers {
                return sw.window_size;
            }
        }
        0
    }
}
