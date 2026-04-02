//! Weight loading from [`WeightProvider`] into MLX arrays.
//!
//! Mirrors the structure of [`crate::metal::weights`] but loads into
//! [`MlxArray`]s instead of Metal buffers. Quantized weights are stored
//! as separate index/LUT/norms arrays for dispatch via custom kernels;
//! affine-quantized weights are dequantized to dense FP16 on the CPU.

use half::f16;
use ironmill_mlx_sys::{MlxArray, MlxDtype, MlxStream};
use mil_rs::ir::ScalarType;
use mil_rs::weights::{ModelConfig, QuantizationInfo, WeightProvider};

use super::error::MlxError;
use crate::dequant::{read_typed_f32, unpack_indices};

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

// ── Per-layer weights ───────────────────────────────────────────

/// Weights for a single transformer layer stored as MLX arrays.
pub struct MlxLayerWeights {
    /// Input layernorm weight `[hidden_size]` FP16.
    pub input_norm: MlxArray,
    /// Q projection `[num_heads × head_dim, hidden_size]`.
    pub q_proj: MlxWeightBuffer,
    /// K projection `[num_kv_heads × head_dim, hidden_size]`.
    pub k_proj: MlxWeightBuffer,
    /// V projection `[num_kv_heads × head_dim, hidden_size]`.
    pub v_proj: MlxWeightBuffer,
    /// Output projection `[hidden_size, num_heads × head_dim]`.
    pub o_proj: MlxWeightBuffer,
    /// Post-attention layernorm weight `[hidden_size]` FP16.
    pub post_attn_norm: MlxArray,
    /// Gate projection `[intermediate_size, hidden_size]`.
    pub gate_proj: MlxWeightBuffer,
    /// Up projection `[intermediate_size, hidden_size]`.
    pub up_proj: MlxWeightBuffer,
    /// Down projection `[hidden_size, intermediate_size]`.
    pub down_proj: MlxWeightBuffer,
    /// Optional Q normalization weight `[head_dim]` FP16 (Qwen3 QK norm).
    pub q_norm: Option<MlxArray>,
    /// Optional K normalization weight `[head_dim]` FP16 (Qwen3 QK norm).
    pub k_norm: Option<MlxArray>,
}

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

impl MlxWeights {
    /// Load model weights from a [`WeightProvider`] into MLX arrays.
    pub fn load(provider: &dyn WeightProvider, stream: &MlxStream) -> Result<Self, MlxError> {
        let config = provider.config().clone();
        let num_layers = config.num_hidden_layers;

        let embedding = load_dense_array(provider, "model.embed_tokens.weight", stream)?;

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let prefix = format!("model.layers.{i}");
            layers.push(MlxLayerWeights {
                input_norm: load_dense_array(
                    provider,
                    &format!("{prefix}.input_layernorm.weight"),
                    stream,
                )?,
                q_proj: load_weight_buffer(
                    provider,
                    &format!("{prefix}.self_attn.q_proj.weight"),
                    stream,
                )?,
                k_proj: load_weight_buffer(
                    provider,
                    &format!("{prefix}.self_attn.k_proj.weight"),
                    stream,
                )?,
                v_proj: load_weight_buffer(
                    provider,
                    &format!("{prefix}.self_attn.v_proj.weight"),
                    stream,
                )?,
                o_proj: load_weight_buffer(
                    provider,
                    &format!("{prefix}.self_attn.o_proj.weight"),
                    stream,
                )?,
                post_attn_norm: load_dense_array(
                    provider,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    stream,
                )?,
                gate_proj: load_weight_buffer(
                    provider,
                    &format!("{prefix}.mlp.gate_proj.weight"),
                    stream,
                )?,
                up_proj: load_weight_buffer(
                    provider,
                    &format!("{prefix}.mlp.up_proj.weight"),
                    stream,
                )?,
                down_proj: load_weight_buffer(
                    provider,
                    &format!("{prefix}.mlp.down_proj.weight"),
                    stream,
                )?,
                q_norm: if provider.has_tensor(&format!("{prefix}.self_attn.q_norm.weight")) {
                    Some(load_dense_array(
                        provider,
                        &format!("{prefix}.self_attn.q_norm.weight"),
                        stream,
                    )?)
                } else {
                    None
                },
                k_norm: if provider.has_tensor(&format!("{prefix}.self_attn.k_norm.weight")) {
                    Some(load_dense_array(
                        provider,
                        &format!("{prefix}.self_attn.k_norm.weight"),
                        stream,
                    )?)
                } else {
                    None
                },
            });
        }

        let final_norm = load_dense_array(provider, "model.norm.weight", stream)?;

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
            embedding,
            layers,
            final_norm,
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

// ── CPU dequant: affine → FP16 bytes ────────────────────────────

fn dequant_affine_to_fp16(
    data: &[u8],
    scale: &[u8],
    zero_point: &[u8],
    scale_dtype: ScalarType,
    zero_point_dtype: ScalarType,
    _axis: Option<usize>,
    shape: &[usize],
) -> anyhow::Result<Vec<u8>> {
    let num_elements: usize = shape.iter().product();
    let mut output = Vec::with_capacity(num_elements * 2);

    let scale_elem_size = scale_dtype.byte_size();
    let zp_elem_size = zero_point_dtype.byte_size();

    for i in 0..num_elements {
        let q_val = data[i] as f32;
        let s_idx = if scale.len() / scale_elem_size > 1 {
            // Per-channel scale: index by row.
            let cols = if shape.len() > 1 { shape[1] } else { 1 };
            i / cols
        } else {
            0
        };
        let s = read_typed_f32(scale, s_idx * scale_elem_size, scale_dtype)?;
        let zp = if zero_point.is_empty() {
            0.0
        } else {
            let zp_idx = s_idx.min(zero_point.len() / zp_elem_size - 1);
            read_typed_f32(zero_point, zp_idx * zp_elem_size, zero_point_dtype)?
        };
        let val = (q_val - zp) * s;
        let h = f16::from_f32(val);
        output.extend_from_slice(&h.to_le_bytes());
    }

    Ok(output)
}

// ── CPU dequant: LUT → FP16 bytes ───────────────────────────────

fn dequant_lut_to_fp16(
    indices: &[u8],
    lut: &[u8],
    lut_dtype: ScalarType,
    original_shape: &[usize],
    n_bits: u8,
    row_norms: &[u8],
    norms_dtype: ScalarType,
    _polar_quant_seed: Option<u64>,
) -> anyhow::Result<Vec<u8>> {
    let num_elements: usize = original_shape.iter().product();
    let cols = if original_shape.len() > 1 {
        original_shape[1]
    } else {
        num_elements
    };
    let rows = num_elements / cols;

    let palette_size = 1usize << n_bits;
    let lut_elem_size = lut_dtype.byte_size();

    let unpacked = unpack_indices(indices, n_bits, num_elements);

    let mut output = Vec::with_capacity(num_elements * 2);

    for row in 0..rows {
        let norm = read_typed_f32(row_norms, row * norms_dtype.byte_size(), norms_dtype)?;
        let lut_row_offset = row * palette_size * lut_elem_size;

        for col in 0..cols {
            let idx = unpacked[row * cols + col];
            let lut_val = read_typed_f32(lut, lut_row_offset + idx * lut_elem_size, lut_dtype)?;
            let val = lut_val * norm;
            let h = f16::from_f32(val);
            output.extend_from_slice(&h.to_le_bytes());
        }
    }

    Ok(output)
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
        } => {
            // CPU dequant to dense FP16, then load as MlxArray.
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

    match &tensor.quant_info {
        QuantizationInfo::None => {
            let dtype = scalar_to_mlx_dtype(tensor.dtype);
            let arr = MlxArray::from_data_copy(&tensor.data, &tensor.shape, dtype, stream)?;
            Ok(arr)
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
            let fp16_data = dequant_lut_to_fp16(
                indices,
                lut,
                *lut_dtype,
                original_shape,
                *n_bits,
                row_norms,
                *norms_dtype,
                *polar_quant_seed,
            )
            .map_err(|e| MlxError::WeightLoading(e.to_string()))?;
            let arr =
                MlxArray::from_data_copy(&fp16_data, original_shape, MlxDtype::Float16, stream)?;
            Ok(arr)
        }
        QuantizationInfo::AffineDequantize {
            scale,
            zero_point,
            scale_dtype,
            zero_point_dtype,
            axis,
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
            Ok(arr)
        }
    }
}
