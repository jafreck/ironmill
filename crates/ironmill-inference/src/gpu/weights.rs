//! Weight loading from SafeTensors/GGUF into Metal buffers.

use ironmill_metal_sys::{MetalBuffer, MetalDevice, StorageMode};
use mil_rs::weights::{ModelConfig, QuantizationInfo, WeightProvider};

use super::dequant::{dequant_affine, dequant_lut_to_dense};
use super::error::GpuError;

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
    /// Q projection [num_heads × head_dim, hidden_size] FP16.
    pub q_proj: MetalBuffer,
    /// K projection [num_kv_heads × head_dim, hidden_size] FP16.
    pub k_proj: MetalBuffer,
    /// V projection [num_kv_heads × head_dim, hidden_size] FP16.
    pub v_proj: MetalBuffer,
    /// Output projection [hidden_size, num_heads × head_dim] FP16.
    pub o_proj: MetalBuffer,
    /// Post-attention layernorm [hidden_size] FP16.
    pub post_attn_norm: MetalBuffer,
    /// Gate projection [intermediate_size, hidden_size] FP16.
    pub gate_proj: MetalBuffer,
    /// Up projection [intermediate_size, hidden_size] FP16.
    pub up_proj: MetalBuffer,
    /// Down projection [hidden_size, intermediate_size] FP16.
    pub down_proj: MetalBuffer,
    /// Optional Q normalization weight [head_dim] FP16 (Qwen3 QK norm).
    pub q_norm: Option<MetalBuffer>,
    /// Optional K normalization weight [head_dim] FP16 (Qwen3 QK norm).
    pub k_norm: Option<MetalBuffer>,
}

impl GpuWeights {
    /// Load model weights from a [`WeightProvider`] into Metal buffers.
    ///
    /// Weights are loaded into shared-mode buffers for CPU→GPU transfer.
    pub fn load(device: &MetalDevice, provider: &dyn WeightProvider) -> Result<Self, GpuError> {
        let config = provider.config().clone();
        let num_layers = config.num_hidden_layers;

        let embedding = load_weight_buffer(device, provider, "model.embed_tokens.weight")?;

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let prefix = format!("model.layers.{i}");
            layers.push(LayerWeights {
                input_norm: load_weight_buffer(
                    device,
                    provider,
                    &format!("{prefix}.input_layernorm.weight"),
                )?,
                q_proj: load_weight_buffer(
                    device,
                    provider,
                    &format!("{prefix}.self_attn.q_proj.weight"),
                )?,
                k_proj: load_weight_buffer(
                    device,
                    provider,
                    &format!("{prefix}.self_attn.k_proj.weight"),
                )?,
                v_proj: load_weight_buffer(
                    device,
                    provider,
                    &format!("{prefix}.self_attn.v_proj.weight"),
                )?,
                o_proj: load_weight_buffer(
                    device,
                    provider,
                    &format!("{prefix}.self_attn.o_proj.weight"),
                )?,
                post_attn_norm: load_weight_buffer(
                    device,
                    provider,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                )?,
                gate_proj: load_weight_buffer(
                    device,
                    provider,
                    &format!("{prefix}.mlp.gate_proj.weight"),
                )?,
                up_proj: load_weight_buffer(
                    device,
                    provider,
                    &format!("{prefix}.mlp.up_proj.weight"),
                )?,
                down_proj: load_weight_buffer(
                    device,
                    provider,
                    &format!("{prefix}.mlp.down_proj.weight"),
                )?,
                q_norm: if provider.has_tensor(&format!("{prefix}.self_attn.q_norm.weight")) {
                    Some(load_weight_buffer(
                        device,
                        provider,
                        &format!("{prefix}.self_attn.q_norm.weight"),
                    )?)
                } else {
                    None
                },
                k_norm: if provider.has_tensor(&format!("{prefix}.self_attn.k_norm.weight")) {
                    Some(load_weight_buffer(
                        device,
                        provider,
                        &format!("{prefix}.self_attn.k_norm.weight"),
                    )?)
                } else {
                    None
                },
            });
        }

        let final_norm = load_weight_buffer(device, provider, "model.norm.weight")?;

        let lm_head = if config.tie_word_embeddings {
            load_weight_buffer(device, provider, "model.embed_tokens.weight")?
        } else {
            load_weight_buffer(device, provider, "lm_head.weight")?
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

/// Load a single weight tensor into a [`MetalBuffer`].
fn load_weight_buffer(
    device: &MetalDevice,
    provider: &dyn WeightProvider,
    name: &str,
) -> Result<MetalBuffer, GpuError> {
    let tensor = provider
        .tensor(name)
        .map_err(|e| GpuError::WeightLoading(format!("{name}: {e}")))?;
    let data = match &tensor.quant_info {
        QuantizationInfo::None => tensor.data.into_owned(),
        QuantizationInfo::LutToDense {
            lut,
            lut_dtype,
            indices,
            original_shape,
            n_bits,
            row_norms,
            norms_dtype,
        } => dequant_lut_to_dense(
            indices,
            lut,
            *lut_dtype,
            original_shape,
            *n_bits,
            row_norms,
            *norms_dtype,
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
