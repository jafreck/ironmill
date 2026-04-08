//! FP16 KV cache for non-TurboQuant inference.

use ironmill_metal_sys::{MetalBuffer, MetalDevice, StorageMode};

use super::error::MetalError;

/// FP16 KV cache (when TurboQuant is disabled).
pub(crate) struct Fp16KvCache {
    /// K caches per buffer slot: `[num_kv_heads × effective_max_seq × head_dim]` FP16.
    pub(crate) k_caches: Vec<MetalBuffer>,
    /// V caches per buffer slot.
    pub(crate) v_caches: Vec<MetalBuffer>,
    pub(crate) seq_pos: usize,
    /// Global max sequence length (for full-attention layers).
    pub(crate) _max_seq_len: usize,
    /// CLA anchor layers. When set, only anchor layers have physical buffers.
    pub(crate) anchor_layers: Option<Vec<usize>>,
    /// Per-buffer sliding window sizes. `0` = full attention for that buffer.
    pub(crate) _window_sizes: Vec<usize>,
}

impl Fp16KvCache {
    pub(crate) fn new(
        device: &MetalDevice,
        num_layers: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        anchor_layers: Option<Vec<usize>>,
        layer_window_sizes: &[usize],
        per_layer_kv_dims: Option<&[(usize, usize)]>,
    ) -> Result<Self, MetalError> {
        let num_buffers = if let Some(ref anchors) = anchor_layers {
            anchors.len()
        } else {
            num_layers
        };

        let per_buffer_window: Vec<usize> = (0..num_buffers)
            .map(|buf_idx| {
                let layer = if let Some(ref anchors) = anchor_layers {
                    anchors[buf_idx]
                } else {
                    buf_idx
                };
                layer_window_sizes.get(layer).copied().unwrap_or(0)
            })
            .collect();

        let mut k_caches = Vec::with_capacity(num_buffers);
        let mut v_caches = Vec::with_capacity(num_buffers);
        for buf_idx in 0..num_buffers {
            let ws = per_buffer_window[buf_idx];
            let effective_seq = if ws > 0 { ws } else { max_seq_len };
            // Use per-layer KV dimensions when available (Gemma 4).
            let layer = if let Some(ref anchors) = anchor_layers {
                anchors[buf_idx]
            } else {
                buf_idx
            };
            let (layer_nkv, layer_hd) = per_layer_kv_dims
                .and_then(|dims| dims.get(layer).copied())
                .unwrap_or((num_kv_heads, head_dim));
            let size_bytes = layer_nkv * effective_seq * layer_hd * 2; // FP16
            k_caches.push(
                device
                    .create_buffer(size_bytes, StorageMode::Shared)
                    .map_err(MetalError::Metal)?,
            );
            v_caches.push(
                device
                    .create_buffer(size_bytes, StorageMode::Shared)
                    .map_err(MetalError::Metal)?,
            );
        }
        Ok(Self {
            k_caches,
            v_caches,
            seq_pos: 0,
            _max_seq_len: max_seq_len,
            anchor_layers,
            _window_sizes: per_buffer_window,
        })
    }

    /// Map a layer index to its physical buffer index.
    pub(crate) fn kv_buffer_for_layer(&self, layer: usize) -> usize {
        match self.anchor_layers {
            Some(ref anchors) => match anchors.binary_search(&layer) {
                Ok(idx) => idx,
                Err(idx) => idx.saturating_sub(1),
            },
            None => layer,
        }
    }

    /// Returns true if `layer` is an anchor layer (or CLA is not configured).
    pub(crate) fn _is_anchor_layer(&self, layer: usize) -> bool {
        match self.anchor_layers {
            Some(ref anchors) => anchors.binary_search(&layer).is_ok(),
            None => true,
        }
    }

    pub(crate) fn layer_caches(&self, layer: usize) -> (&MetalBuffer, &MetalBuffer) {
        let idx = self.kv_buffer_for_layer(layer);
        (&self.k_caches[idx], &self.v_caches[idx])
    }

    /// Write position for ring buffer. For SWA layers, wraps at window_size.
    pub(crate) fn _ring_pos(&self, layer: usize) -> usize {
        let idx = self.kv_buffer_for_layer(layer);
        let ws = self._window_sizes.get(idx).copied().unwrap_or(0);
        if ws > 0 {
            self.seq_pos % ws
        } else {
            self.seq_pos
        }
    }

    /// Effective max_seq_len for a layer's buffer.
    pub(crate) fn _effective_max_seq_len(&self, layer: usize) -> usize {
        let idx = self.kv_buffer_for_layer(layer);
        let ws = self._window_sizes.get(idx).copied().unwrap_or(0);
        if ws > 0 { ws } else { self._max_seq_len }
    }

    /// Effective seq_len for attention (capped at window_size for SWA layers).
    pub(crate) fn _effective_seq_len(&self, layer: usize, total_tokens: usize) -> usize {
        let idx = self.kv_buffer_for_layer(layer);
        let ws = self._window_sizes.get(idx).copied().unwrap_or(0);
        if ws > 0 {
            total_tokens.min(ws)
        } else {
            total_tokens
        }
    }

    /// Window size for a layer. 0 = full attention.
    pub(crate) fn _window_size_for_layer(&self, layer: usize) -> usize {
        let idx = self.kv_buffer_for_layer(layer);
        self._window_sizes.get(idx).copied().unwrap_or(0)
    }

    pub(crate) fn reset(&mut self) {
        self.seq_pos = 0;
    }

    pub(crate) fn _seq_pos(&self) -> usize {
        self.seq_pos
    }

    pub(crate) fn truncate_to(&mut self, pos: usize) {
        assert!(pos <= self.seq_pos);
        self.seq_pos = pos;
    }
}
