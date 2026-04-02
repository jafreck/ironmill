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
        // 1 scale (L2 norm or absmax) per head per position
        let scale_size = config.num_kv_heads * config.max_seq_len * std::mem::size_of::<f32>();

        // QJL sign bits: head_dim/8 bytes per head per position
        let qjl_signs_size = config.num_kv_heads * config.max_seq_len * (config.head_dim / 8);
        // QJL residual norms: 1 f32 per head per position (same as scale_size)
        let qjl_r_norm_size = scale_size;

        let mut k_caches = Vec::with_capacity(config.num_layers);
        let mut v_caches = Vec::with_capacity(config.num_layers);
        let mut k_scales = Vec::with_capacity(config.num_layers);
        let mut v_scales = Vec::with_capacity(config.num_layers);
        let mut k_qjl_signs = Vec::with_capacity(config.num_layers);
        let mut k_r_norms = Vec::with_capacity(config.num_layers);

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
            k_qjl_signs.push(
                device
                    .create_buffer(qjl_signs_size, StorageMode::Shared)
                    .map_err(GpuError::Metal)?,
            );
            k_r_norms.push(
                device
                    .create_buffer(qjl_r_norm_size, StorageMode::Shared)
                    .map_err(GpuError::Metal)?,
            );
        }

        // Outlier cache buffers (allocated even when not used — zero-sized
        // buffers are fine and simplify dispatch).
        let (outlier_cache_size, non_outlier_cache_size) =
            if let Some(ref outlier_cfg) = config.outlier {
                let d_o_padded = outlier_cfg.outlier_channels.len().next_power_of_two();
                let d_n = config.head_dim - outlier_cfg.outlier_channels.len();
                let d_n_padded = d_n.next_power_of_two();
                // Both groups use 4-bit packing (outlier at full levels, non-outlier at fewer)
                (
                    config.num_kv_heads * config.max_seq_len * (d_o_padded / 2),
                    config.num_kv_heads * config.max_seq_len * (d_n_padded / 2),
                )
            } else {
                (1, 1) // minimal allocation when not used
            };

        let mut k_outlier_caches = Vec::with_capacity(config.num_layers);
        let mut v_outlier_caches = Vec::with_capacity(config.num_layers);
        let mut k_non_outlier_caches = Vec::with_capacity(config.num_layers);
        let mut v_non_outlier_caches = Vec::with_capacity(config.num_layers);
        let mut k_outlier_scales = Vec::with_capacity(config.num_layers);
        let mut v_outlier_scales = Vec::with_capacity(config.num_layers);
        let mut k_non_outlier_scales = Vec::with_capacity(config.num_layers);
        let mut v_non_outlier_scales = Vec::with_capacity(config.num_layers);
        let mut k_outlier_r_norms = Vec::with_capacity(config.num_layers);
        let mut k_non_outlier_r_norms = Vec::with_capacity(config.num_layers);

        for _ in 0..config.num_layers {
            k_outlier_caches.push(
                device
                    .create_buffer(outlier_cache_size, StorageMode::Shared)
                    .map_err(GpuError::Metal)?,
            );
            v_outlier_caches.push(
                device
                    .create_buffer(outlier_cache_size, StorageMode::Shared)
                    .map_err(GpuError::Metal)?,
            );
            k_non_outlier_caches.push(
                device
                    .create_buffer(non_outlier_cache_size, StorageMode::Shared)
                    .map_err(GpuError::Metal)?,
            );
            v_non_outlier_caches.push(
                device
                    .create_buffer(non_outlier_cache_size, StorageMode::Shared)
                    .map_err(GpuError::Metal)?,
            );
            k_outlier_scales.push(
                device
                    .create_buffer(scale_size, StorageMode::Shared)
                    .map_err(GpuError::Metal)?,
            );
            v_outlier_scales.push(
                device
                    .create_buffer(scale_size, StorageMode::Shared)
                    .map_err(GpuError::Metal)?,
            );
            k_non_outlier_scales.push(
                device
                    .create_buffer(scale_size, StorageMode::Shared)
                    .map_err(GpuError::Metal)?,
            );
            v_non_outlier_scales.push(
                device
                    .create_buffer(scale_size, StorageMode::Shared)
                    .map_err(GpuError::Metal)?,
            );
            k_outlier_r_norms.push(
                device
                    .create_buffer(scale_size, StorageMode::Shared)
                    .map_err(GpuError::Metal)?,
            );
            k_non_outlier_r_norms.push(
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

    /// Get QJL sign and residual norm buffers for the K cache of a specific layer.
    pub fn layer_k_qjl(&self, layer: usize) -> (&MetalBuffer, &MetalBuffer) {
        (&self.k_qjl_signs[layer], &self.k_r_norms[layer])
    }

    /// Get outlier and non-outlier cache buffers for a specific layer.
    pub fn layer_outlier_caches(
        &self,
        layer: usize,
    ) -> ((&MetalBuffer, &MetalBuffer), (&MetalBuffer, &MetalBuffer)) {
        (
            (&self.k_outlier_caches[layer], &self.v_outlier_caches[layer]),
            (
                &self.k_non_outlier_caches[layer],
                &self.v_non_outlier_caches[layer],
            ),
        )
    }

    /// Get outlier and non-outlier scale buffers for a specific layer.
    pub fn layer_outlier_scales(
        &self,
        layer: usize,
    ) -> ((&MetalBuffer, &MetalBuffer), (&MetalBuffer, &MetalBuffer)) {
        (
            (&self.k_outlier_scales[layer], &self.v_outlier_scales[layer]),
            (
                &self.k_non_outlier_scales[layer],
                &self.v_non_outlier_scales[layer],
            ),
        )
    }

    /// Get outlier and non-outlier QJL residual norm buffers for K cache.
    pub fn layer_outlier_r_norms(&self, layer: usize) -> (&MetalBuffer, &MetalBuffer) {
        (
            &self.k_outlier_r_norms[layer],
            &self.k_non_outlier_r_norms[layer],
        )
    }

    /// Current sequence position.
    pub fn seq_pos(&self) -> usize {
        self.seq_pos
    }

    /// Advance sequence position by one token.
    pub fn advance(&mut self) -> Result<(), GpuError> {
        self.advance_by(1)
    }

    /// Advance by multiple positions (for prefill).
    pub fn advance_by(&mut self, count: usize) -> Result<(), GpuError> {
        let new_pos = self.seq_pos + count;
        if new_pos > self.max_seq_len {
            return Err(GpuError::Config(format!(
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
}
