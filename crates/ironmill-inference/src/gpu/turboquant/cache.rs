//! MTLBuffer-backed quantized KV cache for GPU TurboQuant inference.

use ironmill_metal_sys::{MetalBuffer, MetalDevice, StorageMode};

use super::TurboQuantGpuConfig;
use crate::gpu::error::GpuError;

/// GPU-resident quantized KV cache for TurboQuant inference.
///
/// Each layer has separate K and V cache buffers stored as INT8 or INT4,
/// plus per-head per-position dequantization scale buffers (f32).
/// Layout per cache buffer:
///   INT8: `[num_kv_heads × max_seq_len × head_dim]`     (1 byte/element)
///   INT4: `[num_kv_heads × max_seq_len × head_dim / 2]` (2 elements/byte)
/// Layout per scale buffer:
///   `[num_kv_heads × max_seq_len]` (1 f32 per head per position)
#[allow(dead_code)]
pub struct GpuKvCache {
    /// K cache per layer.
    k_caches: Vec<MetalBuffer>,
    /// V cache per layer.
    v_caches: Vec<MetalBuffer>,
    /// Per-head per-position K dequantization scales per layer.
    k_scales: Vec<MetalBuffer>,
    /// Per-head per-position V dequantization scales per layer.
    v_scales: Vec<MetalBuffer>,
    /// Current sequence position.
    seq_pos: usize,
    /// Number of KV heads.
    num_kv_heads: usize,
    /// Maximum sequence length.
    max_seq_len: usize,
    /// Dimension per head.
    head_dim: usize,
    /// Quantization bits (4 or 8).
    n_bits: u8,
}

impl GpuKvCache {
    /// Allocate KV cache buffers for all layers.
    ///
    /// Uses shared storage mode for CPU-side inspection during development.
    pub fn new(device: &MetalDevice, config: &TurboQuantGpuConfig) -> Result<Self, GpuError> {
        let elements_per_pos = if config.n_bits == 4 {
            config.head_dim / 2
        } else {
            config.head_dim
        };
        let cache_size = config.num_kv_heads * config.max_seq_len * elements_per_pos;
        let scale_size = config.num_kv_heads * config.max_seq_len * std::mem::size_of::<f32>();
        let mut k_caches = Vec::with_capacity(config.num_layers);
        let mut v_caches = Vec::with_capacity(config.num_layers);
        let mut k_scales = Vec::with_capacity(config.num_layers);
        let mut v_scales = Vec::with_capacity(config.num_layers);

        for _ in 0..config.num_layers {
            k_caches.push(
                device
                    .create_buffer(cache_size, StorageMode::Shared)
                    .map_err(GpuError::Metal)?,
            );
            v_caches.push(
                device
                    .create_buffer(cache_size, StorageMode::Shared)
                    .map_err(GpuError::Metal)?,
            );
            k_scales.push(
                device
                    .create_buffer(scale_size, StorageMode::Shared)
                    .map_err(GpuError::Metal)?,
            );
            v_scales.push(
                device
                    .create_buffer(scale_size, StorageMode::Shared)
                    .map_err(GpuError::Metal)?,
            );
        }

        Ok(Self {
            k_caches,
            v_caches,
            k_scales,
            v_scales,
            seq_pos: 0,
            num_kv_heads: config.num_kv_heads,
            max_seq_len: config.max_seq_len,
            head_dim: config.head_dim,
            n_bits: config.n_bits,
        })
    }

    /// Get K and V cache buffers for a specific layer.
    pub fn layer_caches(&self, layer: usize) -> (&MetalBuffer, &MetalBuffer) {
        (&self.k_caches[layer], &self.v_caches[layer])
    }

    /// Get K and V scale buffers for a specific layer.
    pub fn layer_scales(&self, layer: usize) -> (&MetalBuffer, &MetalBuffer) {
        (&self.k_scales[layer], &self.v_scales[layer])
    }

    /// Current sequence position.
    pub fn seq_pos(&self) -> usize {
        self.seq_pos
    }

    /// Advance sequence position by one token.
    pub fn advance(&mut self) {
        self.seq_pos += 1;
    }

    /// Advance by multiple positions (for prefill).
    pub fn advance_by(&mut self, count: usize) {
        self.seq_pos += count;
    }

    /// Current sequence length (number of cached tokens).
    pub fn seq_len(&self) -> usize {
        self.seq_pos
    }

    /// Reset cache for a new conversation.
    pub fn reset(&mut self) {
        self.seq_pos = 0;
        // Note: we don't zero the buffers — the seq_pos prevents
        // reading uninitialized positions.
    }
}
