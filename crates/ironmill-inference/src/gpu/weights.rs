//! Weight loading from SafeTensors/GGUF into Metal buffers.

use ironmill_metal_sys::{MetalBuffer, MetalDevice, StorageMode};
use mil_rs::weights::{ModelConfig, QuantizationInfo, WeightProvider};

use super::dequant::{dequant_affine, dequant_lut_to_dense};
use super::error::GpuError;

/// A weight buffer that is either dense FP16 or packed quantized data.
pub enum WeightBuffer {
    /// Dense FP16 buffer for MPS matmul.
    Dense(MetalBuffer),
    /// Packed quantized buffer for custom kernel.
    Quantized(QuantizedWeight),
}

impl WeightBuffer {
    /// Get the underlying buffer for MPS matmul (Dense only).
    /// Panics if called on Quantized — callers must check type first.
    pub fn as_dense(&self) -> &MetalBuffer {
        match self {
            WeightBuffer::Dense(b) => b,
            WeightBuffer::Quantized(_) => panic!("expected dense buffer"),
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

/// All model weights loaded into Metal buffers, organized by layer.
pub struct GpuWeights {
    /// Embedding table [vocab_size × hidden_size] FP16.
    pub embedding: MetalBuffer,
    /// Per-layer weights.
    pub layers: Vec<LayerWeights>,
    /// Final RMSNorm weight [hidden_size] FP16.
    pub final_norm: MetalBuffer,
    /// LM head weight [vocab_size × hidden_size] FP16.
    pub lm_head: MetalBuffer,
    /// Model configuration extracted from weight metadata.
    pub config: ModelConfig,
}

/// Weights for a single transformer layer.
pub struct LayerWeights {
    /// Input layernorm [hidden_size] FP16.
    pub input_norm: MetalBuffer,
    /// Q projection [num_heads × head_dim, hidden_size].
    pub q_proj: WeightBuffer,
    /// K projection [num_kv_heads × head_dim, hidden_size].
    pub k_proj: WeightBuffer,
    /// V projection [num_kv_heads × head_dim, hidden_size].
    pub v_proj: WeightBuffer,
    /// Output projection [hidden_size, num_heads × head_dim].
    pub o_proj: WeightBuffer,
    /// Post-attention layernorm [hidden_size] FP16.
    pub post_attn_norm: MetalBuffer,
    /// Gate projection [intermediate_size, hidden_size].
    pub gate_proj: WeightBuffer,
    /// Up projection [intermediate_size, hidden_size].
    pub up_proj: WeightBuffer,
    /// Down projection [hidden_size, intermediate_size].
    pub down_proj: WeightBuffer,
    /// Optional Q normalization weight [head_dim] FP16 (Qwen3 QK norm).
    pub q_norm: Option<MetalBuffer>,
    /// Optional K normalization weight [head_dim] FP16 (Qwen3 QK norm).
    pub k_norm: Option<MetalBuffer>,
}

impl GpuWeights {
    /// Load model weights from a [`WeightProvider`] into Metal buffers.
    ///
    /// Weights are loaded into shared-mode buffers for CPU→GPU transfer.
    pub fn load(
        device: &MetalDevice,
        provider: &dyn WeightProvider,
        force_cpu_dequant: bool,
    ) -> Result<Self, GpuError> {
        let config = provider.config().clone();
        let num_layers = config.num_hidden_layers;

        let embedding = load_dense_buffer(device, provider, "model.embed_tokens.weight")?;

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let prefix = format!("model.layers.{i}");
            layers.push(LayerWeights {
                input_norm: load_dense_buffer(
                    device,
                    provider,
                    &format!("{prefix}.input_layernorm.weight"),
                )?,
                q_proj: load_weight_buffer(
                    device,
                    provider,
                    &format!("{prefix}.self_attn.q_proj.weight"),
                    force_cpu_dequant,
                )?,
                k_proj: load_weight_buffer(
                    device,
                    provider,
                    &format!("{prefix}.self_attn.k_proj.weight"),
                    force_cpu_dequant,
                )?,
                v_proj: load_weight_buffer(
                    device,
                    provider,
                    &format!("{prefix}.self_attn.v_proj.weight"),
                    force_cpu_dequant,
                )?,
                o_proj: load_weight_buffer(
                    device,
                    provider,
                    &format!("{prefix}.self_attn.o_proj.weight"),
                    force_cpu_dequant,
                )?,
                post_attn_norm: load_dense_buffer(
                    device,
                    provider,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                )?,
                gate_proj: load_weight_buffer(
                    device,
                    provider,
                    &format!("{prefix}.mlp.gate_proj.weight"),
                    force_cpu_dequant,
                )?,
                up_proj: load_weight_buffer(
                    device,
                    provider,
                    &format!("{prefix}.mlp.up_proj.weight"),
                    force_cpu_dequant,
                )?,
                down_proj: load_weight_buffer(
                    device,
                    provider,
                    &format!("{prefix}.mlp.down_proj.weight"),
                    force_cpu_dequant,
                )?,
                q_norm: if provider.has_tensor(&format!("{prefix}.self_attn.q_norm.weight")) {
                    Some(load_dense_buffer(
                        device,
                        provider,
                        &format!("{prefix}.self_attn.q_norm.weight"),
                    )?)
                } else {
                    None
                },
                k_norm: if provider.has_tensor(&format!("{prefix}.self_attn.k_norm.weight")) {
                    Some(load_dense_buffer(
                        device,
                        provider,
                        &format!("{prefix}.self_attn.k_norm.weight"),
                    )?)
                } else {
                    None
                },
            });
        }

        let final_norm = load_dense_buffer(device, provider, "model.norm.weight")?;

        let lm_head = if config.tie_word_embeddings {
            load_dense_buffer(device, provider, "model.embed_tokens.weight")?
        } else {
            load_dense_buffer(device, provider, "lm_head.weight")?
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

/// Load a single weight tensor into a [`WeightBuffer`].
///
/// Returns [`WeightBuffer::Quantized`] for LUT-quantized tensors, keeping
/// packed indices, LUT, and norms as separate Metal buffers. Falls back to
/// dense FP16 for unquantized or affine-quantized tensors.
fn load_weight_buffer(
    device: &MetalDevice,
    provider: &dyn WeightProvider,
    name: &str,
    force_cpu_dequant: bool,
) -> Result<WeightBuffer, GpuError> {
    let tensor = provider
        .tensor(name)
        .map_err(|e| GpuError::WeightLoading(format!("{name}: {e}")))?;
    match &tensor.quant_info {
        QuantizationInfo::None => {
            let buf = device
                .create_buffer_with_data(&tensor.data, StorageMode::Shared)
                .map_err(GpuError::Metal)?;
            Ok(WeightBuffer::Dense(buf))
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
                );
                let buf = device
                    .create_buffer_with_data(&data, StorageMode::Shared)
                    .map_err(GpuError::Metal)?;
                Ok(WeightBuffer::Dense(buf))
            } else {
                let indices_buf = device
                    .create_buffer_with_data(indices, StorageMode::Shared)
                    .map_err(GpuError::Metal)?;
                let lut_buf = device
                    .create_buffer_with_data(lut, StorageMode::Shared)
                    .map_err(GpuError::Metal)?;
                let norms_buf = device
                    .create_buffer_with_data(row_norms, StorageMode::Shared)
                    .map_err(GpuError::Metal)?;
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
            );
            let buf = device
                .create_buffer_with_data(&data, StorageMode::Shared)
                .map_err(GpuError::Metal)?;
            Ok(WeightBuffer::Dense(buf))
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
) -> Result<MetalBuffer, GpuError> {
    let tensor = provider
        .tensor(name)
        .map_err(|e| GpuError::WeightLoading(format!("{name}: {e}")))?;
    let data = match &tensor.quant_info {
        QuantizationInfo::None => {
            // Zero-copy: pass borrowed data directly to Metal without
            // allocating an intermediate Vec.
            return device
                .create_buffer_with_data(&tensor.data, StorageMode::Shared)
                .map_err(GpuError::Metal);
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
        ),
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
        ),
    };
    device
        .create_buffer_with_data(&data, StorageMode::Shared)
        .map_err(GpuError::Metal)
}
