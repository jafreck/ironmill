//! TurboQuant INT8 KV cache compression for GPU inference.

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
        // Hadamard rotation matrix from the ANE TurboQuant path.
        let rotation_bytes = crate::ane::turboquant::mil_emitter::generate_rotation_weights(
            config.head_dim,
            config.rotation_seed,
        );

        let rotation_matrix = device
            .create_buffer_with_data(&rotation_bytes, ironmill_metal_sys::StorageMode::Shared)
            .map_err(GpuError::Metal)?;

        // Uniform scale for Hadamard-rotated values.
        //
        // After Hadamard rotation (norm-preserving), per-element values are
        // concentrated but the vector norm is preserved. For a Qwen3-0.6B
        // K/V projection with hidden_size=1024, head_dim=128, typical
        // per-element values are ~O(1). After rotation, they spread to
        // ~O(||x||/√dim) with occasional outliers.
        //
        // max_abs must be large enough to avoid clipping outliers across
        // all layers and positions. Empirically calibrated.
        let max_abs = 32.0_f32;
        let inv_scale = 127.0 / max_abs;
        let deq_scale = max_abs / 127.0;

        Ok(Self {
            rotation_matrix,
            inv_scale,
            deq_scale,
            config,
        })
    }
}
