//! MTLBuffer-backed INT8 KV cache for GPU TurboQuant inference.

use ironmill_metal_sys::{MetalBuffer, MetalDevice, StorageMode};

use super::TurboQuantGpuConfig;
use crate::gpu::error::GpuError;

/// GPU-resident INT8 KV cache for TurboQuant inference.
///
/// Each layer has separate K and V cache buffers stored as INT8.
/// Layout: `[num_kv_heads × max_seq_len × head_dim]` per buffer.
#[allow(dead_code)]
pub struct GpuKvCache {
    /// K cache per layer.
    k_caches: Vec<MetalBuffer>,
    /// V cache per layer.
    v_caches: Vec<MetalBuffer>,
    /// Current sequence position.
    seq_pos: usize,
    /// Number of KV heads.
    num_kv_heads: usize,
    /// Maximum sequence length.
    max_seq_len: usize,
    /// Dimension per head.
    head_dim: usize,
}

impl GpuKvCache {
    /// Allocate KV cache buffers for all layers.
    ///
    /// Uses shared storage mode for CPU-side inspection during development.
    pub fn new(device: &MetalDevice, config: &TurboQuantGpuConfig) -> Result<Self, GpuError> {
        let cache_size = config.num_kv_heads * config.max_seq_len * config.head_dim;
        let mut k_caches = Vec::with_capacity(config.num_layers);
        let mut v_caches = Vec::with_capacity(config.num_layers);

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
        }

        Ok(Self {
            k_caches,
            v_caches,
            seq_pos: 0,
            num_kv_heads: config.num_kv_heads,
            max_seq_len: config.max_seq_len,
            head_dim: config.head_dim,
        })
    }

    /// Get K and V cache buffers for a specific layer.
    pub fn layer_caches(&self, layer: usize) -> (&MetalBuffer, &MetalBuffer) {
        (&self.k_caches[layer], &self.v_caches[layer])
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
