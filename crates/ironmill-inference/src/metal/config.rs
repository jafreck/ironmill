//! Configuration for the Metal GPU inference backend.

use mil_rs;

/// Per-layer attention configuration for Gemma 4 models with heterogeneous layers.
#[derive(Debug, Clone)]
pub struct Gemma4LayerConfig {
    /// Head dimension for this layer's attention (may differ for global layers).
    pub head_dim: usize,
    /// Number of KV heads for this layer (may differ for global layers).
    pub num_kv_heads: usize,
    /// Sliding window size for this layer. 0 = full attention.
    pub window_size: usize,
    /// Whether this is a global (full_attention) layer.
    pub is_global: bool,
    /// Index of RoPE frequency table to use. 0 = default/sliding, 1 = global.
    pub rope_table_index: usize,
    /// For KV shared layers: the anchor layer index whose KV cache to reuse.
    /// None for non-shared layers and anchor layers.
    pub kv_anchor: Option<usize>,
    /// Effective intermediate size for MLP (may be doubled for KV-shared layers).
    pub intermediate_size: usize,
}

/// Gemma 4 model-specific configuration, computed at model load time.
#[derive(Debug, Clone)]
pub struct Gemma4Config {
    /// Per-layer attention configuration.
    pub layer_configs: Vec<Gemma4LayerConfig>,
    /// Global head dim (may differ from default head_dim for global layers).
    pub global_head_dim: usize,
    /// Per-Layer Embedding hidden size. 0 = no PLE.
    pub ple_hidden_size: usize,
    /// Final logit softcapping value. None = no softcapping.
    pub final_logit_softcapping: Option<f32>,
}

impl Gemma4Config {
    /// Build Gemma4Config from a ModelConfig, if it's a Gemma 4 model.
    ///
    /// Returns None for non-Gemma-4 models (no layer_types in config).
    pub fn from_model_config(mc: &mil_rs::weights::ModelConfig) -> Option<Self> {
        let layer_types = mc.layer_types()?;
        let global_head_dim = mc.global_head_dim();
        let num_global_kv_heads = mc.num_global_key_value_heads();
        let sliding_window = mc.sliding_window().unwrap_or(0);

        let num_kv_shared = mc
            .extra
            .get("num_kv_shared_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let first_shared_idx = mc.num_hidden_layers.saturating_sub(num_kv_shared);

        let use_double_wide_mlp = mc
            .extra
            .get("use_double_wide_mlp")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let ple_hidden_size = mc
            .extra
            .get("hidden_size_per_layer_input")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let final_logit_softcapping = mc
            .extra
            .get("final_logit_softcapping")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32);

        // Build per-layer configs
        let prev_types = &layer_types[..first_shared_idx];
        let mut layer_configs = Vec::with_capacity(mc.num_hidden_layers);
        for layer_idx in 0..mc.num_hidden_layers {
            let is_global = layer_types[layer_idx] == "full_attention";
            let is_kv_shared = num_kv_shared > 0 && layer_idx >= first_shared_idx;

            let kv_anchor = if is_kv_shared {
                let lt = &layer_types[layer_idx];
                prev_types.iter().rposition(|t| t == lt)
            } else {
                None
            };

            let intermediate_size = if use_double_wide_mlp && is_kv_shared {
                mc.intermediate_size * 2
            } else {
                mc.intermediate_size
            };

            layer_configs.push(Gemma4LayerConfig {
                head_dim: if is_global {
                    global_head_dim
                } else {
                    mc.head_dim
                },
                num_kv_heads: if is_global {
                    num_global_kv_heads
                } else {
                    mc.num_key_value_heads
                },
                window_size: if is_global { 0 } else { sliding_window },
                is_global,
                rope_table_index: if is_global { 1 } else { 0 },
                kv_anchor,
                intermediate_size,
            });
        }

        Some(Self {
            layer_configs,
            global_head_dim,
            ple_hidden_size,
            final_logit_softcapping,
        })
    }
}

/// Cross-Layer Attention configuration.
///
/// When configured, only anchor layers maintain their own KV cache buffers.
/// Non-anchor layers reuse the KV cache of the nearest preceding anchor,
/// reducing KV memory by up to 2×.
#[non_exhaustive]
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
    pub fn validate(&self, num_layers: usize) -> Result<(), crate::engine::InferenceError> {
        if self.anchor_layers.is_empty() {
            return Err(crate::engine::InferenceError::runtime(
                "anchor_layers must not be empty",
            ));
        }
        // Must be sorted and deduplicated.
        for w in self.anchor_layers.windows(2) {
            if w[0] >= w[1] {
                return Err(crate::engine::InferenceError::runtime(format!(
                    "anchor_layers must be sorted and unique, found {} before {}",
                    w[0], w[1]
                )));
            }
        }
        // All indices must be valid layer indices.
        if let Some(&last) = self.anchor_layers.last() {
            if last >= num_layers {
                return Err(crate::engine::InferenceError::runtime(format!(
                    "anchor layer index {} >= num_layers {}",
                    last, num_layers
                )));
            }
        }
        // Layer 0 must be an anchor (first layer has no preceding anchor).
        if self.anchor_layers[0] != 0 {
            return Err(crate::engine::InferenceError::runtime(
                "layer 0 must be an anchor layer",
            ));
        }
        Ok(())
    }
}

/// Sliding window attention configuration.
///
/// When configured, layers below `max_window_layers` use sliding window
/// attention with a bounded KV cache (ring buffer of `window_size` entries),
/// while layers at or above `max_window_layers` use full attention.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct SlidingWindowConfig {
    /// Number of recent tokens each SWA layer attends to (e.g. 4096).
    pub window_size: usize,
    /// Layers `0..max_window_layers` use sliding window attention;
    /// layers `max_window_layers..num_layers` use full attention.
    pub max_window_layers: usize,
}

/// Configuration for the Metal GPU inference backend.
#[non_exhaustive]
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
    /// Create a new `MetalConfig` with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum sequence length.
    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    /// Enable TurboQuant KV cache compression with the given bit-width.
    pub fn with_turboquant(mut self, bits: u8) -> Self {
        self.enable_turboquant = true;
        self.n_bits = bits;
        self
    }

    /// Enable or disable FlashAttention-2 style prefill.
    pub fn with_fa2_prefill(mut self, enable: bool) -> Self {
        self.use_fa2_prefill = enable;
        self
    }

    /// Set the Cross-Layer Attention configuration.
    pub fn with_cla(mut self, cla_config: ClaConfig) -> Self {
        self.cla_config = Some(cla_config);
        self
    }

    /// Set the sliding window attention configuration.
    pub fn with_sliding_window(mut self, sliding_window: SlidingWindowConfig) -> Self {
        self.sliding_window = Some(sliding_window);
        self
    }

    /// Set the prefill chunk size.
    pub fn with_prefill_chunks(mut self, chunk_size: usize) -> Self {
        self.prefill_chunk_size = Some(chunk_size);
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<(), crate::engine::InferenceError> {
        if self.max_seq_len == 0 {
            return Err(crate::engine::InferenceError::runtime(
                "max_seq_len must be > 0",
            ));
        }
        if self.n_bits != 4 && self.n_bits != 8 {
            return Err(crate::engine::InferenceError::runtime(format!(
                "n_bits must be 4 or 8, got {}",
                self.n_bits
            )));
        }
        if let Some(ref sw) = self.sliding_window {
            if sw.window_size == 0 {
                return Err(crate::engine::InferenceError::runtime(
                    "sliding_window.window_size must be > 0",
                ));
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
