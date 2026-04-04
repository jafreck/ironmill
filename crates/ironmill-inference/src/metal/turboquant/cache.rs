//! MTLBuffer-backed quantized KV cache for GPU TurboQuant inference.

use ironmill_metal_sys::{MetalBuffer, MetalDevice, StorageMode};

use super::TurboQuantMetalConfig;
use crate::metal::error::MetalError;
use crate::turboquant::cache_layout::TurboQuantCacheLayout;

/// GPU-resident quantized KV cache for TurboQuant inference.
///
/// Each layer has separate K and V cache buffers stored as INT8 or INT4,
/// plus per-head per-position dequantization scale buffers (f32).
/// Layout per cache buffer:
///   INT8: `[num_kv_heads × max_seq_len × head_dim]`     (1 byte/element)
///   INT4: `[num_kv_heads × max_seq_len × head_dim / 2]` (2 elements/byte)
/// Layout per scale buffer:
///   `[num_kv_heads × max_seq_len]` (1 f32 per head per position)
///
/// When CLA (cross-layer attention) is configured, only anchor layers
/// have physical buffers. Non-anchor layers share their nearest preceding
/// anchor's buffers via [`kv_buffer_for_layer`].
pub struct MetalKvCache {
    /// K cache per layer.
    k_caches: Vec<MetalBuffer>,
    /// V cache per layer.
    v_caches: Vec<MetalBuffer>,
    /// Per-head per-position K dequantization scales per layer.
    k_scales: Vec<MetalBuffer>,
    /// Per-head per-position V dequantization scales per layer.
    v_scales: Vec<MetalBuffer>,

    // ── QJL residual correction (K cache only) ──
    /// Packed sign bits of QJL-projected residuals per K position.
    /// Layout: `[num_kv_heads × max_seq_len × head_dim/8]` bytes per layer.
    k_qjl_signs: Vec<MetalBuffer>,
    /// L2 norm of quantization residual per K position.
    /// Layout: `[num_kv_heads × max_seq_len]` f32 per layer.
    k_r_norms: Vec<MetalBuffer>,

    // ── Outlier channel strategy (optional) ──
    /// Outlier group K cache per layer.
    k_outlier_caches: Vec<MetalBuffer>,
    /// Outlier group V cache per layer.
    v_outlier_caches: Vec<MetalBuffer>,
    /// Non-outlier group K cache per layer.
    k_non_outlier_caches: Vec<MetalBuffer>,
    /// Non-outlier group V cache per layer.
    v_non_outlier_caches: Vec<MetalBuffer>,
    /// Outlier group K/V scales per layer.
    k_outlier_scales: Vec<MetalBuffer>,
    v_outlier_scales: Vec<MetalBuffer>,
    /// Non-outlier group K/V scales per layer.
    k_non_outlier_scales: Vec<MetalBuffer>,
    v_non_outlier_scales: Vec<MetalBuffer>,

    // ── QJL residual norms for outlier groups (K cache only) ──
    /// Outlier group residual norms per layer: `[num_kv_heads × max_seq_len]` f32.
    k_outlier_r_norms: Vec<MetalBuffer>,
    /// Non-outlier group residual norms per layer: `[num_kv_heads × max_seq_len]` f32.
    k_non_outlier_r_norms: Vec<MetalBuffer>,

    /// Current sequence position.
    seq_pos: usize,
    /// Maximum sequence length.
    max_seq_len: usize,
    /// CLA anchor layers. When set, only anchor layers have physical buffers.
    /// The Vec index maps buffer index → layer index.
    anchor_layers: Option<Vec<usize>>,
    /// Per-buffer sliding window sizes. `0` = full attention for that buffer.
    /// When a buffer has `window_size > 0`, it is allocated with `window_size`
    /// slots instead of `max_seq_len` and uses ring-buffer write semantics.
    window_sizes: Vec<usize>,
}

impl MetalKvCache {
    /// Allocate KV cache buffers for all layers (or only anchor layers when CLA is configured).
    ///
    /// Uses shared storage mode for CPU-side inspection during development.
    /// When `config.window_sizes` is non-empty, SWA layers allocate smaller
    /// ring buffers bounded by their window size.
    pub fn new(device: &MetalDevice, config: &TurboQuantMetalConfig) -> Result<Self, MetalError> {
        let num_buffers = if let Some(ref anchors) = config.anchor_layers {
            anchors.len()
        } else {
            config.num_layers
        };

        // Compute per-buffer window sizes. Buffer index maps to layer via
        // anchor_layers (CLA) or identity.
        let per_buffer_window: Vec<usize> = (0..num_buffers)
            .map(|buf_idx| {
                let layer = if let Some(ref anchors) = config.anchor_layers {
                    anchors[buf_idx]
                } else {
                    buf_idx
                };
                config.window_sizes.get(layer).copied().unwrap_or(0)
            })
            .collect();

        let mut k_caches = Vec::with_capacity(num_buffers);
        let mut v_caches = Vec::with_capacity(num_buffers);
        let mut k_scales = Vec::with_capacity(num_buffers);
        let mut v_scales = Vec::with_capacity(num_buffers);
        let mut k_qjl_signs = Vec::with_capacity(num_buffers);
        let mut k_r_norms = Vec::with_capacity(num_buffers);

        for buf_idx in 0..num_buffers {
            let ws = per_buffer_window[buf_idx];
            let effective_seq = if ws > 0 { ws } else { config.max_seq_len };
            let layer = if let Some(ref anchors) = config.anchor_layers {
                anchors[buf_idx]
            } else {
                buf_idx
            };
            let layer_head_dim = config
                .layer_configs
                .get(layer)
                .map(|lc| lc.head_dim)
                .unwrap_or(config.head_dim);
            let layer_num_kv = config
                .layer_configs
                .get(layer)
                .map(|lc| lc.num_kv_heads)
                .unwrap_or(config.num_kv_heads);
            let layout = TurboQuantCacheLayout::new(
                layer_num_kv,
                layer_head_dim,
                effective_seq,
                config.num_layers,
                config.n_bits,
                config.outlier.as_ref(),
            );

            let cache_size = layout.cache_bytes;
            let scale_size = layout.scale_bytes();
            let qjl_signs_size = layout.qjl_signs_bytes;
            let qjl_r_norm_size = scale_size;

            k_caches.push(
                device
                    .create_buffer(cache_size, StorageMode::Shared)
                    .map_err(MetalError::Metal)?,
            );
            v_caches.push(
                device
                    .create_buffer(cache_size, StorageMode::Shared)
                    .map_err(MetalError::Metal)?,
            );
            k_scales.push(
                device
                    .create_buffer(scale_size, StorageMode::Shared)
                    .map_err(MetalError::Metal)?,
            );
            v_scales.push(
                device
                    .create_buffer(scale_size, StorageMode::Shared)
                    .map_err(MetalError::Metal)?,
            );
            k_qjl_signs.push(
                device
                    .create_buffer(qjl_signs_size, StorageMode::Shared)
                    .map_err(MetalError::Metal)?,
            );
            k_r_norms.push(
                device
                    .create_buffer(qjl_r_norm_size, StorageMode::Shared)
                    .map_err(MetalError::Metal)?,
            );
        }

        // Outlier cache buffers (allocated even when not used — zero-sized
        // buffers are fine and simplify dispatch).
        let mut k_outlier_caches = Vec::with_capacity(num_buffers);
        let mut v_outlier_caches = Vec::with_capacity(num_buffers);
        let mut k_non_outlier_caches = Vec::with_capacity(num_buffers);
        let mut v_non_outlier_caches = Vec::with_capacity(num_buffers);
        let mut k_outlier_scales = Vec::with_capacity(num_buffers);
        let mut v_outlier_scales = Vec::with_capacity(num_buffers);
        let mut k_non_outlier_scales = Vec::with_capacity(num_buffers);
        let mut v_non_outlier_scales = Vec::with_capacity(num_buffers);
        let mut k_outlier_r_norms = Vec::with_capacity(num_buffers);
        let mut k_non_outlier_r_norms = Vec::with_capacity(num_buffers);

        for buf_idx in 0..num_buffers {
            let ws = per_buffer_window[buf_idx];
            let effective_seq = if ws > 0 { ws } else { config.max_seq_len };
            let layer = if let Some(ref anchors) = config.anchor_layers {
                anchors[buf_idx]
            } else {
                buf_idx
            };
            let layer_head_dim = config
                .layer_configs
                .get(layer)
                .map(|lc| lc.head_dim)
                .unwrap_or(config.head_dim);
            let layer_num_kv = config
                .layer_configs
                .get(layer)
                .map(|lc| lc.num_kv_heads)
                .unwrap_or(config.num_kv_heads);
            let layout = TurboQuantCacheLayout::new(
                layer_num_kv,
                layer_head_dim,
                effective_seq,
                config.num_layers,
                config.n_bits,
                config.outlier.as_ref(),
            );
            let scale_size = layout.scale_bytes();
            let (outlier_cache_size, non_outlier_cache_size) = if let Some(ref ol) = layout.outlier
            {
                (ol.outlier_cache_bytes, ol.non_outlier_cache_bytes)
            } else {
                (1, 1)
            };

            k_outlier_caches.push(
                device
                    .create_buffer(outlier_cache_size, StorageMode::Shared)
                    .map_err(MetalError::Metal)?,
            );
            v_outlier_caches.push(
                device
                    .create_buffer(outlier_cache_size, StorageMode::Shared)
                    .map_err(MetalError::Metal)?,
            );
            k_non_outlier_caches.push(
                device
                    .create_buffer(non_outlier_cache_size, StorageMode::Shared)
                    .map_err(MetalError::Metal)?,
            );
            v_non_outlier_caches.push(
                device
                    .create_buffer(non_outlier_cache_size, StorageMode::Shared)
                    .map_err(MetalError::Metal)?,
            );
            k_outlier_scales.push(
                device
                    .create_buffer(scale_size, StorageMode::Shared)
                    .map_err(MetalError::Metal)?,
            );
            v_outlier_scales.push(
                device
                    .create_buffer(scale_size, StorageMode::Shared)
                    .map_err(MetalError::Metal)?,
            );
            k_non_outlier_scales.push(
                device
                    .create_buffer(scale_size, StorageMode::Shared)
                    .map_err(MetalError::Metal)?,
            );
            v_non_outlier_scales.push(
                device
                    .create_buffer(scale_size, StorageMode::Shared)
                    .map_err(MetalError::Metal)?,
            );
            k_outlier_r_norms.push(
                device
                    .create_buffer(scale_size, StorageMode::Shared)
                    .map_err(MetalError::Metal)?,
            );
            k_non_outlier_r_norms.push(
                device
                    .create_buffer(scale_size, StorageMode::Shared)
                    .map_err(MetalError::Metal)?,
            );
        }

        Ok(Self {
            k_caches,
            v_caches,
            k_scales,
            v_scales,
            k_qjl_signs,
            k_r_norms,
            k_outlier_caches,
            v_outlier_caches,
            k_non_outlier_caches,
            v_non_outlier_caches,
            k_outlier_scales,
            v_outlier_scales,
            k_non_outlier_scales,
            v_non_outlier_scales,
            k_outlier_r_norms,
            k_non_outlier_r_norms,
            seq_pos: 0,
            max_seq_len: config.max_seq_len,
            anchor_layers: config.anchor_layers.clone(),
            window_sizes: per_buffer_window,
        })
    }

    /// Map a layer index to its physical buffer index.
    ///
    /// When CLA is configured, non-anchor layers map to their nearest
    /// preceding anchor's buffer. When CLA is not configured, returns
    /// the layer index unchanged (identity mapping).
    pub fn kv_buffer_for_layer(&self, layer: usize) -> usize {
        match self.anchor_layers {
            Some(ref anchors) => {
                // Binary search for the largest anchor <= layer.
                match anchors.binary_search(&layer) {
                    Ok(idx) => idx,
                    Err(idx) => idx.saturating_sub(1),
                }
            }
            None => layer,
        }
    }

    /// Returns true if `layer` is an anchor layer (or CLA is not configured).
    pub fn is_anchor_layer(&self, layer: usize) -> bool {
        match self.anchor_layers {
            Some(ref anchors) => anchors.binary_search(&layer).is_ok(),
            None => true,
        }
    }

    /// Write position for ring buffer. For SWA layers, wraps at window_size.
    /// For full-attention layers, returns seq_pos unchanged.
    pub fn ring_pos(&self, layer: usize) -> usize {
        let idx = self.kv_buffer_for_layer(layer);
        let ws = self.window_sizes.get(idx).copied().unwrap_or(0);
        if ws > 0 {
            self.seq_pos % ws
        } else {
            self.seq_pos
        }
    }

    /// Effective max_seq_len for a layer's buffer. SWA layers use window_size
    /// as the buffer stride; full-attention layers use the global max_seq_len.
    pub fn effective_max_seq_len(&self, layer: usize) -> usize {
        let idx = self.kv_buffer_for_layer(layer);
        let ws = self.window_sizes.get(idx).copied().unwrap_or(0);
        if ws > 0 { ws } else { self.max_seq_len }
    }

    /// Effective seq_len for attention. For SWA layers, capped at window_size.
    /// For full-attention layers, returns the actual cached token count.
    pub fn effective_seq_len(&self, layer: usize, total_tokens: usize) -> usize {
        let idx = self.kv_buffer_for_layer(layer);
        let ws = self.window_sizes.get(idx).copied().unwrap_or(0);
        if ws > 0 {
            total_tokens.min(ws)
        } else {
            total_tokens
        }
    }

    /// Window size for a layer. 0 = full attention.
    pub fn window_size_for_layer(&self, layer: usize) -> usize {
        let idx = self.kv_buffer_for_layer(layer);
        self.window_sizes.get(idx).copied().unwrap_or(0)
    }

    /// Number of physical buffer slots allocated.
    pub fn num_buffers(&self) -> usize {
        self.k_caches.len()
    }

    /// Get K and V cache buffers for a specific layer.
    /// When CLA is configured, non-anchor layers return their anchor's buffers.
    pub fn layer_caches(&self, layer: usize) -> (&MetalBuffer, &MetalBuffer) {
        let idx = self.kv_buffer_for_layer(layer);
        (&self.k_caches[idx], &self.v_caches[idx])
    }

    /// Get K and V scale buffers for a specific layer.
    pub fn layer_scales(&self, layer: usize) -> (&MetalBuffer, &MetalBuffer) {
        let idx = self.kv_buffer_for_layer(layer);
        (&self.k_scales[idx], &self.v_scales[idx])
    }

    /// Get QJL sign and residual norm buffers for the K cache of a specific layer.
    pub fn layer_k_qjl(&self, layer: usize) -> (&MetalBuffer, &MetalBuffer) {
        let idx = self.kv_buffer_for_layer(layer);
        (&self.k_qjl_signs[idx], &self.k_r_norms[idx])
    }

    /// Get outlier and non-outlier cache buffers for a specific layer.
    pub fn layer_outlier_caches(
        &self,
        layer: usize,
    ) -> ((&MetalBuffer, &MetalBuffer), (&MetalBuffer, &MetalBuffer)) {
        let idx = self.kv_buffer_for_layer(layer);
        (
            (&self.k_outlier_caches[idx], &self.v_outlier_caches[idx]),
            (
                &self.k_non_outlier_caches[idx],
                &self.v_non_outlier_caches[idx],
            ),
        )
    }

    /// Get outlier and non-outlier scale buffers for a specific layer.
    pub fn layer_outlier_scales(
        &self,
        layer: usize,
    ) -> ((&MetalBuffer, &MetalBuffer), (&MetalBuffer, &MetalBuffer)) {
        let idx = self.kv_buffer_for_layer(layer);
        (
            (&self.k_outlier_scales[idx], &self.v_outlier_scales[idx]),
            (
                &self.k_non_outlier_scales[idx],
                &self.v_non_outlier_scales[idx],
            ),
        )
    }

    /// Get outlier and non-outlier QJL residual norm buffers for K cache.
    pub fn layer_outlier_r_norms(&self, layer: usize) -> (&MetalBuffer, &MetalBuffer) {
        let idx = self.kv_buffer_for_layer(layer);
        (
            &self.k_outlier_r_norms[idx],
            &self.k_non_outlier_r_norms[idx],
        )
    }

    /// Current sequence position.
    pub fn seq_pos(&self) -> usize {
        self.seq_pos
    }

    /// Advance sequence position by one token.
    pub fn advance(&mut self) -> Result<(), MetalError> {
        self.advance_by(1)
    }

    /// Advance by multiple positions (for prefill).
    pub fn advance_by(&mut self, count: usize) -> Result<(), MetalError> {
        let new_pos = self.seq_pos + count;
        if new_pos > self.max_seq_len {
            return Err(MetalError::Config(format!(
                "sequence position overflow: {} + {} = {} exceeds max_seq_len {}",
                self.seq_pos, count, new_pos, self.max_seq_len,
            )));
        }
        self.seq_pos = new_pos;
        Ok(())
    }

    /// Current sequence length (number of cached tokens).
    pub fn seq_len(&self) -> usize {
        self.seq_pos
    }

    /// Reset cache for a new conversation.
    pub fn reset(&mut self) {
        self.seq_pos = 0;
    }

    /// Truncate cache to `pos` tokens. Does not deallocate — just moves
    /// the write pointer back. Subsequent writes overwrite old data.
    pub fn truncate_to(&mut self, pos: usize) {
        assert!(pos <= self.seq_pos);
        self.seq_pos = pos;
    }
}
