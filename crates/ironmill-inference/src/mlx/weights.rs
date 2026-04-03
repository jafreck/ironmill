//! Weight loading from [`WeightProvider`] into MLX arrays.
//!
//! Mirrors the structure of [`crate::metal::weights`] but loads into
//! [`MlxArray`]s instead of Metal buffers. Quantized weights are stored
//! as separate index/LUT/norms arrays for dispatch via custom kernels;
//! affine-quantized weights are dequantized to dense FP16 on the CPU.

use ironmill_mlx_sys::{MlxArray, MlxDtype, MlxStream};
use mil_rs::ir::ScalarType;
use mil_rs::weights::{ModelConfig, QuantizationInfo, WeightProvider};

use super::error::MlxError;
use crate::dequant::dequant_affine_to_fp16;
use crate::weight_loading::{
    self, CpuDequant, LoadedLayer, WeightVisitor, dequant_tensor_to_dense,
};

// ── Weight buffer types ─────────────────────────────────────────

/// A weight buffer that is either dense FP16 or packed quantized data
/// (PolarQuant LUT indices + norms).
pub enum MlxWeightBuffer {
    /// Dense FP16 weight array.
    Dense(MlxArray),
    /// Packed quantized weight for custom kernel dispatch.
    Quantized(MlxQuantizedWeight),
}

/// Packed quantized weight stored as separate MLX arrays for the custom
/// Metal kernel (PolarQuant / palettization).
pub struct MlxQuantizedWeight {
    /// Packed n-bit palette indices.
    pub indices: MlxArray,
    /// Reconstruction look-up table.
    pub lut: MlxArray,
    /// Per-row normalization factors.
    pub norms: MlxArray,
    /// Bit-width of the palette indices (e.g. 2, 4).
    pub bit_width: u8,
    /// Output features (rows).
    pub n: usize,
    /// Input features (columns).
    pub k: usize,
}

// ── Per-layer weights (type alias over shared layout) ───────────

/// Weights for a single transformer layer stored as MLX arrays.
pub type MlxLayerWeights = LoadedLayer<MlxArray, MlxWeightBuffer>;

// ── Full model weights ──────────────────────────────────────────

/// All model weights loaded into MLX arrays, organized by layer.
pub struct MlxWeights {
    /// Embedding table `[vocab_size, hidden_size]` FP16.
    pub embedding: MlxArray,
    /// Per-layer weights.
    pub layers: Vec<MlxLayerWeights>,
    /// Final RMSNorm weight `[hidden_size]` FP16.
    pub final_norm: MlxArray,
    /// LM head weight `[vocab_size, hidden_size]`.
    pub lm_head: MlxWeightBuffer,
    /// Model configuration extracted from weight metadata.
    pub config: ModelConfig,
}

/// Backend-specific visitor that loads tensors into MLX arrays.
struct MlxVisitor<'a> {
    stream: &'a MlxStream,
}

impl WeightVisitor for MlxVisitor<'_> {
    type Dense = MlxArray;
    type Weight = MlxWeightBuffer;
    type Error = MlxError;

    fn load_dense(&self, provider: &dyn WeightProvider, name: &str) -> Result<MlxArray, MlxError> {
        load_dense_array(provider, name, self.stream)
    }

    fn load_weight(
        &self,
        provider: &dyn WeightProvider,
        name: &str,
    ) -> Result<MlxWeightBuffer, MlxError> {
        load_weight_buffer(provider, name, self.stream)
    }
}

impl MlxWeights {
    /// Load model weights from a [`WeightProvider`] into MLX arrays.
    pub fn load(provider: &dyn WeightProvider, stream: &MlxStream) -> Result<Self, MlxError> {
        let visitor = MlxVisitor { stream };
        let core = weight_loading::load_model_weights(&visitor, provider)?;
        let config = core.config;

        let lm_head = if config.tie_word_embeddings {
            MlxWeightBuffer::Dense(load_dense_array(
                provider,
                "model.embed_tokens.weight",
                stream,
            )?)
        } else {
            load_weight_buffer(provider, "lm_head.weight", stream)?
        };

        Ok(Self {
            embedding: core.embedding,
            layers: core.layers,
            final_norm: core.final_norm,
            lm_head,
            config,
        })
    }
}

// ── Helper: ScalarType → MlxDtype ───────────────────────────────

fn scalar_to_mlx_dtype(st: ScalarType) -> MlxDtype {
    match st {
        ScalarType::Float16 => MlxDtype::Float16,
        ScalarType::Float32 => MlxDtype::Float32,
        ScalarType::Float64 => MlxDtype::Float32, // downcast
        ScalarType::Int8 => MlxDtype::Int8,
        ScalarType::Int16 => MlxDtype::Int16,
        ScalarType::Int32 => MlxDtype::Int32,
        ScalarType::Int64 => MlxDtype::Int64,
        ScalarType::UInt8 => MlxDtype::Uint8,
        ScalarType::UInt16 => MlxDtype::Uint16,
        ScalarType::UInt32 => MlxDtype::Uint32,
        ScalarType::UInt64 => MlxDtype::Uint64,
        ScalarType::Bool => MlxDtype::Bool,
    }
}

// ── CPU dequant operations for MLX ───────────────────────────────

/// CPU dequant operations for the MLX backend.
struct MlxDequantOps;

impl CpuDequant for MlxDequantOps {
    fn dequant_lut(
        indices: &[u8],
        lut: &[u8],
        lut_dtype: ScalarType,
        original_shape: &[usize],
        n_bits: u8,
        row_norms: &[u8],
        norms_dtype: ScalarType,
        polar_quant_seed: Option<u64>,
    ) -> anyhow::Result<Vec<u8>> {
        crate::dequant::dequant_lut_to_fp16(
            indices,
            lut,
            lut_dtype,
            original_shape,
            n_bits,
            row_norms,
            norms_dtype,
            polar_quant_seed,
        )
    }

    fn dequant_affine(
        data: &[u8],
        scale: &[u8],
        zero_point: &[u8],
        scale_dtype: ScalarType,
        zero_point_dtype: ScalarType,
        axis: Option<usize>,
        shape: &[usize],
        _bit_width: u8,
        _group_size: Option<usize>,
    ) -> anyhow::Result<Vec<u8>> {
        dequant_affine_to_fp16(
            data,
            scale,
            zero_point,
            scale_dtype,
            zero_point_dtype,
            axis,
            shape,
        )
    }
}

// ── Load a single weight into an MlxWeightBuffer ────────────────

fn load_weight_buffer(
    provider: &dyn WeightProvider,
    name: &str,
    stream: &MlxStream,
) -> Result<MlxWeightBuffer, MlxError> {
    let tensor = provider
        .tensor(name)
        .map_err(|e| MlxError::WeightLoading(format!("{name}: {e}")))?;

    match &tensor.quant_info {
        QuantizationInfo::None => {
            let dtype = scalar_to_mlx_dtype(tensor.dtype);
            let arr = MlxArray::from_data_copy(&tensor.data, &tensor.shape, dtype, stream)?;
            Ok(MlxWeightBuffer::Dense(arr))
        }
        QuantizationInfo::LutToDense {
            lut,
            lut_dtype,
            indices,
            original_shape,
            n_bits,
            row_norms,
            norms_dtype,
            polar_quant_seed: _,
        } => {
            // Store as separate arrays for custom kernel dispatch.
            let idx_arr =
                MlxArray::from_data_copy(indices, &[indices.len()], MlxDtype::Uint8, stream)?;
            let lut_arr = MlxArray::from_data_copy(
                lut,
                &[lut.len() / lut_dtype.byte_size()],
                scalar_to_mlx_dtype(*lut_dtype),
                stream,
            )?;
            let norms_arr = MlxArray::from_data_copy(
                row_norms,
                &[row_norms.len() / norms_dtype.byte_size()],
                scalar_to_mlx_dtype(*norms_dtype),
                stream,
            )?;

            let n = original_shape[0];
            let k = if original_shape.len() > 1 {
                original_shape[1]
            } else {
                1
            };

            Ok(MlxWeightBuffer::Quantized(MlxQuantizedWeight {
                indices: idx_arr,
                lut: lut_arr,
                norms: norms_arr,
                bit_width: *n_bits,
                n,
                k,
            }))
        }
        QuantizationInfo::AffineDequantize {
            scale,
            zero_point,
            scale_dtype,
            zero_point_dtype,
            axis,
            ..
        } => {
            let fp16_data = dequant_affine_to_fp16(
                &tensor.data,
                scale,
                zero_point,
                *scale_dtype,
                *zero_point_dtype,
                *axis,
                &tensor.shape,
            )
            .map_err(|e| MlxError::WeightLoading(e.to_string()))?;
            let arr =
                MlxArray::from_data_copy(&fp16_data, &tensor.shape, MlxDtype::Float16, stream)?;
            Ok(MlxWeightBuffer::Dense(arr))
        }
    }
}

// ── Load a single weight as a dense FP16 MlxArray ───────────────

fn load_dense_array(
    provider: &dyn WeightProvider,
    name: &str,
    stream: &MlxStream,
) -> Result<MlxArray, MlxError> {
    let tensor = provider
        .tensor(name)
        .map_err(|e| MlxError::WeightLoading(format!("{name}: {e}")))?;
    let dense = dequant_tensor_to_dense::<MlxDequantOps>(&tensor)
        .map_err(|e| MlxError::WeightLoading(e.to_string()))?;
    let dtype = scalar_to_mlx_dtype(dense.dtype);
    let arr = MlxArray::from_data_copy(&dense.bytes, dense.shape, dtype, stream)?;
    Ok(arr)
}
