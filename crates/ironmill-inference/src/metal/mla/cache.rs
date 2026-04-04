//! Compressed KV cache for Multi-Head Latent Attention.
//!
//! Instead of storing full-dimensional K and V tensors per layer, MLA
//! stores the compressed latent vector (dimension `kv_latent_dim`) and
//! the RoPE-applied key portion separately. This reduces KV cache memory
//! by `(num_heads * head_dim) / kv_latent_dim` compared to standard MHA.

use ironmill_metal_sys::{MetalBuffer, MetalDevice, StorageMode};
use mil_rs::weights::MlaConfig;

use crate::metal::error::MetalError;

/// Compressed KV cache storing latents instead of full K/V.
///
/// Each layer stores:
/// - A joint KV latent buffer of shape `[max_seq_len, kv_latent_dim]`
/// - A RoPE key buffer of shape `[max_seq_len, qk_rope_head_dim]`
///
/// During inference the latent is used directly in attention (after weight
/// absorption eliminates the need for up-projection), and the RoPE key is
/// concatenated into the attention score computation.
pub struct MlaKvCache {
    /// Per-layer joint KV latent buffers: `[max_seq_len × kv_latent_dim]` FP16.
    latent_caches: Vec<MetalBuffer>,
    /// Per-layer RoPE key buffers: `[max_seq_len × qk_rope_head_dim]` FP16.
    rope_k_caches: Vec<MetalBuffer>,
    /// Current sequence position (number of tokens cached).
    seq_pos: usize,
    /// Maximum sequence length this cache was allocated for.
    max_seq_len: usize,
}

impl MlaKvCache {
    /// Allocate latent and RoPE key caches for all layers.
    pub fn new(
        device: &MetalDevice,
        config: &MlaConfig,
        num_layers: usize,
        max_seq_len: usize,
    ) -> Result<Self, MetalError> {
        let latent_bytes = max_seq_len * config.kv_latent_dim * 2; // FP16
        let rope_bytes = max_seq_len * config.qk_rope_head_dim * 2; // FP16

        let mut latent_caches = Vec::with_capacity(num_layers);
        let mut rope_k_caches = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            latent_caches.push(
                device
                    .create_buffer(latent_bytes, StorageMode::Shared)
                    .map_err(MetalError::Metal)?,
            );
            rope_k_caches.push(
                device
                    .create_buffer(rope_bytes, StorageMode::Shared)
                    .map_err(MetalError::Metal)?,
            );
        }

        Ok(Self {
            latent_caches,
            rope_k_caches,
            seq_pos: 0,
            max_seq_len,
        })
    }

    /// Get the latent cache buffer for a specific layer.
    pub fn latent_cache(&self, layer: usize) -> &MetalBuffer {
        &self.latent_caches[layer]
    }

    /// Get the RoPE key cache buffer for a specific layer.
    pub fn rope_k_cache(&self, layer: usize) -> &MetalBuffer {
        &self.rope_k_caches[layer]
    }

    /// Current number of cached tokens.
    pub fn seq_pos(&self) -> usize {
        self.seq_pos
    }

    /// Advance the sequence position by one token.
    pub fn advance(&mut self) -> Result<(), MetalError> {
        self.advance_by(1)
    }

    /// Advance the sequence position by `count` tokens.
    pub fn advance_by(&mut self, count: usize) -> Result<(), MetalError> {
        let new_pos = self.seq_pos + count;
        if new_pos > self.max_seq_len {
            return Err(MetalError::Config(format!(
                "MLA KV cache overflow: position {} + {} exceeds max_seq_len {}",
                self.seq_pos, count, self.max_seq_len,
            )));
        }
        self.seq_pos = new_pos;
        Ok(())
    }

    /// Reset the cache to empty (position 0).
    pub fn reset(&mut self) {
        self.seq_pos = 0;
    }

    /// Truncate the cache to a specific position.
    ///
    /// # Panics
    ///
    /// Panics if `pos` exceeds the current sequence position.
    pub fn truncate_to(&mut self, pos: usize) {
        assert!(
            pos <= self.seq_pos,
            "truncate_to({pos}) exceeds current seq_pos ({})",
            self.seq_pos,
        );
        self.seq_pos = pos;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that MlaKvCache methods compile and have the expected
    /// signatures. Cannot run without a Metal device.
    #[test]
    fn mla_kv_cache_api_surface_compiles() {
        fn _assert_api(cache: &mut MlaKvCache) {
            let _ = cache.latent_cache(0);
            let _ = cache.rope_k_cache(0);
            let _ = cache.seq_pos();
            let _ = cache.advance();
            let _ = cache.advance_by(4);
            cache.reset();
            cache.truncate_to(0);
        }
        let _ = _assert_api;
    }
}
