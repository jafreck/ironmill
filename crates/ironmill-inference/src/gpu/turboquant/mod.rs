//! TurboQuant quantized KV cache compression for GPU inference.
//!
//! Supports INT8 (n_bits=8) and INT4 (n_bits=4) quantization.

pub mod cache;

pub use cache::GpuKvCache;

use ironmill_metal_sys::MetalDevice;

use super::error::GpuError;

/// TurboQuant model state for GPU inference.
///
/// Manages the Hadamard rotation matrix and dequantization scale
/// used by the fused cache write and attention kernels.
pub struct GpuTurboQuantModel {
    /// Hadamard rotation matrix [head_dim × head_dim] FP16.
    pub rotation_matrix: ironmill_metal_sys::MetalBuffer,
    /// Inverse quantization scale for cache write.
    pub inv_scale: f32,
    /// Dequantization scale for attention.
    pub deq_scale: f32,
    /// Config.
    pub config: TurboQuantGpuConfig,
}

/// TurboQuant-specific configuration extracted from GpuConfig + model params.
#[derive(Debug, Clone)]
pub struct TurboQuantGpuConfig {
    pub n_bits: u8,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub num_layers: usize,
    pub rotation_seed: u64,
}

impl GpuTurboQuantModel {
    /// Initialize TurboQuant with rotation matrix and quantization scales.
    pub fn new(device: &MetalDevice, config: TurboQuantGpuConfig) -> Result<Self, GpuError> {
        // Reuse the ANE TurboQuant rotation matrix generation, which is
        // pub(crate) and produces [head_dim × head_dim] FP16 LE bytes.
        let rotation_bytes = crate::ane::turboquant::mil_emitter::generate_rotation_weights(
            config.head_dim,
            config.rotation_seed,
        );

        let rotation_matrix = device
            .create_buffer_with_data(&rotation_bytes, ironmill_metal_sys::StorageMode::Shared)
            .map_err(GpuError::Metal)?;

        // Reuse the ANE TurboQuant scale computation.
        let inv_scale =
            crate::ane::turboquant::mil_emitter::compute_inv_scale(config.head_dim, config.n_bits);
        let deq_scale = crate::ane::turboquant::compute_deq_scale(config.head_dim, config.n_bits);

        Ok(Self {
            rotation_matrix,
            inv_scale,
            deq_scale,
            config,
        })
    }
}
