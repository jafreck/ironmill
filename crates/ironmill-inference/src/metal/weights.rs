//! Weight loading from SafeTensors/GGUF into Metal buffers.

use ironmill_metal_sys::{MetalBuffer, MetalDevice, StorageMode};
use mil_rs::weights::{ModelConfig, QuantizationInfo, WeightProvider};

use super::dequant::{dequant_affine, dequant_lut_to_dense};
use super::error::MetalError;
use crate::weight_loading::{self, LoadedLayer, WeightVisitor};

/// A weight buffer that is either dense FP16 or packed quantized data.
pub enum WeightBuffer {
    /// Dense FP16 buffer for MPS matmul, with an optional pre-packed blocked
    /// buffer for the custom matvec kernel (decode path, M=1).
    Dense {
        /// Original row-major [N, K] FP16 buffer (used by MPS for prefill).
        buf: MetalBuffer,
        /// Blocked [N/8, K/8, 8, 8] FP16 buffer for the custom matvec kernel.
        /// `None` if N or K is not a multiple of 8.
        packed: Option<MetalBuffer>,
    },
    /// Packed quantized buffer for custom kernel.
    Quantized(QuantizedWeight),
}

impl WeightBuffer {
    /// Get the underlying buffer for MPS matmul (Dense only).
    /// Returns an error if called on a Quantized variant.
    pub fn as_dense(&self) -> anyhow::Result<&MetalBuffer> {
        match self {
            WeightBuffer::Dense { buf, .. } => Ok(buf),
            WeightBuffer::Quantized(_) => Err(anyhow::anyhow!(
                "expected dense buffer, got quantized weight"
            )),
        }
    }

    /// Get the pre-packed blocked buffer for the custom matvec kernel.
    /// Returns `None` for Quantized weights or Dense weights where packing
    /// was not possible (dimensions not multiples of 8).
    pub fn packed_buf(&self) -> Option<&MetalBuffer> {
        match self {
            WeightBuffer::Dense { packed, .. } => packed.as_ref(),
            WeightBuffer::Quantized(_) => None,
        }
    }
}

/// Packed quantized weight stored as separate Metal buffers for the custom
/// matmul kernel.
pub struct QuantizedWeight {
    /// Packed n-bit indices.
    pub indices: MetalBuffer,
    /// Reconstruction look-up table `[2^n_bits]`.
    pub lut: MetalBuffer,
    /// Per-row norms `[rows]`.
    pub norms: MetalBuffer,
    /// Bit-width of the palette indices (e.g. 2, 4).
    pub n_bits: u8,
    /// `(out_features, in_features)`.
    pub shape: (usize, usize),
}

/// Weights for a single transformer layer (type alias over shared layout).
pub type LayerWeights = LoadedLayer<MetalBuffer, WeightBuffer>;

/// All model weights loaded into Metal buffers, organized by layer.
pub struct MetalWeights {
    /// Embedding table [vocab_size × hidden_size] FP16.
    pub embedding: MetalBuffer,
    /// Per-layer weights.
    pub layers: Vec<LayerWeights>,
    /// Final RMSNorm weight [hidden_size] FP16.
    pub final_norm: MetalBuffer,
    /// LM head weight [vocab_size × hidden_size] FP16.
    pub lm_head: MetalBuffer,
    /// Pre-packed LM head in blocked [N/8, K/8, 8, 8] FP16 (for custom matvec).
    pub lm_head_packed: Option<MetalBuffer>,
    /// Model configuration extracted from weight metadata.
    pub config: ModelConfig,
}

/// Backend-specific visitor that loads tensors into Metal buffers.
struct MetalVisitor<'a> {
    device: &'a MetalDevice,
    force_cpu_dequant: bool,
}

impl WeightVisitor for MetalVisitor<'_> {
    type Dense = MetalBuffer;
    type Weight = WeightBuffer;
    type Error = MetalError;

    fn load_dense(
        &self,
        provider: &dyn WeightProvider,
        name: &str,
    ) -> Result<MetalBuffer, MetalError> {
        load_dense_buffer(self.device, provider, name)
    }

    fn load_weight(
        &self,
        provider: &dyn WeightProvider,
        name: &str,
    ) -> Result<WeightBuffer, MetalError> {
        load_weight_buffer(self.device, provider, name, self.force_cpu_dequant)
    }
}

impl MetalWeights {
    /// Load model weights from a [`WeightProvider`] into Metal buffers.
    ///
    /// Weights are loaded into shared-mode buffers for CPU→GPU transfer.
    pub fn load(
        device: &MetalDevice,
        provider: &dyn WeightProvider,
        force_cpu_dequant: bool,
    ) -> Result<Self, MetalError> {
        let visitor = MetalVisitor {
            device,
            force_cpu_dequant,
        };
        let core = weight_loading::load_model_weights(&visitor, provider)?;
        let config = core.config;

        let lm_head = if config.tie_word_embeddings {
            load_dense_buffer(device, provider, "model.embed_tokens.weight")?
        } else {
            load_dense_buffer(device, provider, "lm_head.weight")?
        };

        let lm_head_packed =
            pack_weight_blocked(device, &lm_head, config.vocab_size, config.hidden_size)?;

        Ok(Self {
            embedding: core.embedding,
            layers: core.layers,
            final_norm: core.final_norm,
            lm_head,
            lm_head_packed,
            config,
        })
    }
}

/// Extract (N, K) from a weight tensor shape [N, K]. Returns (1, N) for 1-D.
fn dense_shape(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        1 => (1, shape[0]),
        _ => (shape[0], shape[1]),
    }
}

/// Pack a row-major [N, K] FP16 weight buffer into blocked [N/8, K/8, 8, 8] format.
///
/// Returns `None` if N or K is not a multiple of 8.
fn pack_weight_blocked(
    device: &MetalDevice,
    original: &MetalBuffer,
    n: usize,
    k: usize,
) -> Result<Option<MetalBuffer>, MetalError> {
    if n % 8 != 0 || k % 8 != 0 {
        return Ok(None);
    }

    let total_bytes = n * k * 2; // FP16
    let mut src = vec![0u8; total_bytes];
    original
        .read_bytes(&mut src, 0)
        .map_err(MetalError::Metal)?;

    let mut dst = vec![0u8; total_bytes];
    let n_blocks = n / 8;
    let k_blocks = k / 8;

    for nb in 0..n_blocks {
        for kb in 0..k_blocks {
            for ri in 0..8u32 {
                for ki in 0..8u32 {
                    let src_row = nb * 8 + ri as usize;
                    let src_col = kb * 8 + ki as usize;
                    let src_offset = (src_row * k + src_col) * 2;

                    let dst_idx = ((nb * k_blocks + kb) * 8 + ri as usize) * 8 + ki as usize;
                    let dst_offset = dst_idx * 2;

                    dst[dst_offset] = src[src_offset];
                    dst[dst_offset + 1] = src[src_offset + 1];
                }
            }
        }
    }

    let buf = device
        .create_buffer_with_data(&dst, StorageMode::Shared)
        .map_err(MetalError::Metal)?;
    Ok(Some(buf))
}

/// Load a single weight tensor into a [`WeightBuffer`].
///
/// Returns [`WeightBuffer::Quantized`] for LUT-quantized tensors, keeping
/// packed indices, LUT, and norms as separate Metal buffers. Falls back to
/// dense FP16 for unquantized or affine-quantized tensors.
/// Dense weights are also pre-packed into blocked format for the custom
/// matvec kernel when dimensions are multiples of 8.
fn load_weight_buffer(
    device: &MetalDevice,
    provider: &dyn WeightProvider,
    name: &str,
    force_cpu_dequant: bool,
) -> Result<WeightBuffer, MetalError> {
    let tensor = provider
        .tensor(name)
        .map_err(|e| MetalError::WeightLoading(format!("{name}: {e}")))?;
    match &tensor.quant_info {
        QuantizationInfo::None => {
            let buf = device
                .create_buffer_with_data(&tensor.data, StorageMode::Shared)
                .map_err(MetalError::Metal)?;
            let (n, k) = dense_shape(&tensor.shape);
            let packed = pack_weight_blocked(device, &buf, n, k)?;
            Ok(WeightBuffer::Dense { buf, packed })
        }
        QuantizationInfo::LutToDense {
            lut,
            lut_dtype,
            indices,
            original_shape,
            n_bits,
            row_norms,
            norms_dtype,
            polar_quant_seed,
        } => {
            if force_cpu_dequant {
                let data = dequant_lut_to_dense(
                    indices,
                    lut,
                    *lut_dtype,
                    original_shape,
                    *n_bits,
                    row_norms,
                    *norms_dtype,
                    *polar_quant_seed,
                )?;
                let buf = device
                    .create_buffer_with_data(&data, StorageMode::Shared)
                    .map_err(MetalError::Metal)?;
                let (n, k) = dense_shape(original_shape);
                let packed = pack_weight_blocked(device, &buf, n, k)?;
                Ok(WeightBuffer::Dense { buf, packed })
            } else {
                let indices_buf = device
                    .create_buffer_with_data(indices, StorageMode::Shared)
                    .map_err(MetalError::Metal)?;
                let lut_buf = device
                    .create_buffer_with_data(lut, StorageMode::Shared)
                    .map_err(MetalError::Metal)?;
                let norms_buf = device
                    .create_buffer_with_data(row_norms, StorageMode::Shared)
                    .map_err(MetalError::Metal)?;
                let rows = original_shape[0];
                let cols = if original_shape.len() > 1 {
                    original_shape[1]
                } else {
                    1
                };
                Ok(WeightBuffer::Quantized(QuantizedWeight {
                    indices: indices_buf,
                    lut: lut_buf,
                    norms: norms_buf,
                    n_bits: *n_bits,
                    shape: (rows, cols),
                }))
            }
        }
        QuantizationInfo::AffineDequantize {
            scale,
            zero_point,
            scale_dtype,
            zero_point_dtype,
            axis,
        } => {
            let data = dequant_affine(
                &tensor.data,
                scale,
                zero_point,
                *scale_dtype,
                *zero_point_dtype,
                *axis,
                &tensor.shape,
            )?;
            let buf = device
                .create_buffer_with_data(&data, StorageMode::Shared)
                .map_err(MetalError::Metal)?;
            let (n, k) = dense_shape(&tensor.shape);
            let packed = pack_weight_blocked(device, &buf, n, k)?;
            Ok(WeightBuffer::Dense { buf, packed })
        }
    }
}

/// Load a single weight tensor into a dense FP16 [`MetalBuffer`].
///
/// Always dequantizes to FP16 — used for embeddings, norms, and lm_head
/// which are always consumed by MPS or element-wise kernels.
fn load_dense_buffer(
    device: &MetalDevice,
    provider: &dyn WeightProvider,
    name: &str,
) -> Result<MetalBuffer, MetalError> {
    let tensor = provider
        .tensor(name)
        .map_err(|e| MetalError::WeightLoading(format!("{name}: {e}")))?;
    let data = match &tensor.quant_info {
        QuantizationInfo::None => {
            // Zero-copy: pass borrowed data directly to Metal without
            // allocating an intermediate Vec.
            return device
                .create_buffer_with_data(&tensor.data, StorageMode::Shared)
                .map_err(MetalError::Metal);
        }
        QuantizationInfo::LutToDense {
            lut,
            lut_dtype,
            indices,
            original_shape,
            n_bits,
            row_norms,
            norms_dtype,
            polar_quant_seed,
        } => dequant_lut_to_dense(
            indices,
            lut,
            *lut_dtype,
            original_shape,
            *n_bits,
            row_norms,
            *norms_dtype,
            *polar_quant_seed,
        )?,
        QuantizationInfo::AffineDequantize {
            scale,
            zero_point,
            scale_dtype,
            zero_point_dtype,
            axis,
        } => dequant_affine(
            &tensor.data,
            scale,
            zero_point,
            *scale_dtype,
            *zero_point_dtype,
            *axis,
            &tensor.shape,
        )?,
    };
    device
        .create_buffer_with_data(&data, StorageMode::Shared)
        .map_err(MetalError::Metal)
}
