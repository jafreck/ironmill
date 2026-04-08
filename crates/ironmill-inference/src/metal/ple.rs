//! PLE (Per-Layer Embedding) computation for Gemma 4.

use half::f16;
use ironmill_metal_sys::{ComputeEncoder, MetalBuffer, MetalDevice, StorageMode};
use mil_rs::weights::ModelConfig;

use super::buffers::IntermediateBuffers;
use super::config::Gemma4Config;
use super::ops;
use super::ops::MetalPipelines;
use super::projection::encode_projection;
use super::weights::{LayerWeights, MetalWeights, WeightBuffer};
use crate::engine::InferenceError;

/// Encode PLE model-level computation (before layer loop).
///
/// When `apply_scaling` is true (normal inference), the embedding output is
/// scaled by `sqrt(ple_hidden_size)`, the projection output is scaled by
/// `1/sqrt(h)`, and RMSNorm is applied per-layer (hidden_size = ple_h,
/// token_count = token_count × num_layers).
///
/// When `apply_scaling` is false (calibration), those scaling steps are
/// skipped and RMSNorm treats the full ple_total width as hidden_size.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_ple_model_level(
    enc: &ComputeEncoder,
    pipelines: &MetalPipelines,
    device: &MetalDevice,
    weights: &MetalWeights,
    bufs: &IntermediateBuffers,
    token_ids_buf: &MetalBuffer,
    mc: &ModelConfig,
    gemma4_config: Option<&Gemma4Config>,
    token_count: usize,
    h: usize,
    vocab: usize,
    eps: f32,
    apply_scaling: bool,
) -> Result<(), InferenceError> {
    let g4 = match gemma4_config {
        Some(g4) => g4,
        None => return Ok(()),
    };
    let ple_h = g4.ple_hidden_size;
    if ple_h == 0 {
        return Ok(());
    }
    let (ple_embed, ple_proj, ple_norm, ple_buf) = match (
        &weights.ple_embed_tokens,
        &weights.ple_model_projection,
        &weights.ple_projection_norm,
        &bufs.ple_per_layer_input,
    ) {
        (Some(embed), Some(proj), Some(norm), Some(buf)) => (embed, proj, norm, buf),
        _ => return Ok(()),
    };
    // Ensure ple_scratch is present (needed for model-level bookkeeping).
    if bufs.ple_scratch.is_none() {
        return Ok(());
    }

    let ple_total = mc.num_hidden_layers * ple_h;

    enc.memory_barrier_with_resources(&[&bufs.hidden_state]);

    // 1. Gather from ple_embed_tokens using token_ids → ple_per_layer_input
    //    Shape: [tokens, num_layers * ple_hidden]
    match ple_embed {
        WeightBuffer::Dense { buf: Some(buf), .. } => {
            ops::encode_embedding_lookup(
                enc,
                &pipelines.embedding.embedding_lookup,
                &ops::EmbeddingLookupParams {
                    token_ids: token_ids_buf,
                    embedding_table: buf,
                    output: ple_buf,
                    hidden_size: ple_total as u32,
                    token_count: token_count as u32,
                    vocab_size: vocab as u32,
                },
            );
        }
        WeightBuffer::DualScaleQuantized(dq) => {
            ops::encode_d2quant_embedding_lookup(
                enc,
                &pipelines.d2quant.embedding_lookup_3bit,
                &ops::D2QuantEmbeddingLookupParams {
                    token_ids: token_ids_buf,
                    weight: dq,
                    output: ple_buf,
                    hidden_size: ple_total as u32,
                    token_count: token_count as u32,
                    vocab_size: vocab as u32,
                },
            );
        }
        other => {
            return Err(InferenceError::runtime(format!(
                "unsupported PLE embedding weight type: {:?}",
                std::mem::discriminant(other)
            )));
        }
    }
    enc.memory_barrier_with_resources(&[ple_buf]);

    // Scale PLE embeddings by sqrt(ple_hidden_size) to match HF
    // (only in normal inference; calibration skips this).
    if apply_scaling {
        let ple_embed_scale = (ple_h as f32).sqrt();
        let scale_half = f16::from_f32(ple_embed_scale);
        let scale_buf = device
            .create_buffer_with_data(&scale_half.to_le_bytes(), StorageMode::Shared)
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
        ops::encode_scale_buffer(
            enc,
            &pipelines.elementwise.scale_buffer,
            ple_buf,
            &scale_buf,
            (token_count * ple_total) as u32,
        );
        enc.memory_barrier_with_resources(&[ple_buf]);
    }

    // 2. Project hidden_state via ple_model_projection → ffn_gate (temp)
    encode_projection(
        enc,
        &bufs.hidden_state,
        ple_proj,
        &bufs.ffn_gate,
        pipelines,
        token_count,
        ple_total,
        h,
    )?;
    enc.memory_barrier_with_resources(&[&bufs.ffn_gate]);

    // Scale projection output by 1/sqrt(hidden_size) to match HF's
    // per_layer_model_projection_scale (only in normal inference).
    if apply_scaling {
        let proj_scale = 1.0 / (h as f32).sqrt();
        let scale_half = f16::from_f32(proj_scale);
        let scale_buf = device
            .create_buffer_with_data(&scale_half.to_le_bytes(), StorageMode::Shared)
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
        ops::encode_scale_buffer(
            enc,
            &pipelines.elementwise.scale_buffer,
            &bufs.ffn_gate,
            &scale_buf,
            (token_count * ple_total) as u32,
        );
        enc.memory_barrier_with_resources(&[&bufs.ffn_gate]);
    }

    // 3. RMSNorm the projection output.
    //    Inference: per-layer norm — hidden_size=ple_h, token_count=tokens*num_layers.
    //    Calibration: bulk norm — hidden_size=ple_total, token_count=tokens.
    let (norm_hidden, norm_tokens) = if apply_scaling {
        (ple_h as u32, (token_count * mc.num_hidden_layers) as u32)
    } else {
        (ple_total as u32, token_count as u32)
    };
    ops::encode_rms_norm(
        enc,
        &pipelines.norm.rms_norm,
        &ops::RmsNormParams {
            input: &bufs.ffn_gate,
            weight: ple_norm,
            output: &bufs.ffn_up,
            hidden_size: norm_hidden,
            token_count: norm_tokens,
            eps,
        },
    );
    enc.memory_barrier_with_resources(&[&bufs.ffn_up]);

    // 4+5. Add embed (ple_buf) + normed projection (ffn_up), scale by 2^(-0.5)
    //       → store result back in ple_per_layer_input
    let ple_scale: f32 = std::f32::consts::FRAC_1_SQRT_2;
    ops::encode_add_scale(
        enc,
        &pipelines.activation.ple_add_scale,
        ple_buf,
        &bufs.ffn_up,
        ple_buf,
        (token_count * ple_total) as u32,
        ple_scale,
    );
    enc.memory_barrier_with_resources(&[ple_buf]);

    Ok(())
}

/// Encode PLE per-layer gate and addition (inside layer loop).
///
/// Returns `true` if PLE was applied (caller should skip the normal
/// fused residual+norm path), or `false` if PLE is not active for this
/// layer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_ple_per_layer(
    enc: &ComputeEncoder,
    pipelines: &MetalPipelines,
    bufs: &IntermediateBuffers,
    lw: &LayerWeights,
    next_input_norm: Option<&MetalBuffer>,
    gemma4_config: Option<&Gemma4Config>,
    num_hidden_layers: usize,
    layer_idx: usize,
    token_count: usize,
    h: usize,
    eps: f32,
) -> Result<bool, InferenceError> {
    let has_ple = gemma4_config
        .as_ref()
        .is_some_and(|g4| g4.ple_hidden_size > 0)
        && lw.ple_gate.is_some()
        && bufs.ple_per_layer_input.is_some();

    if !has_ple {
        return Ok(false);
    }

    let g4 = gemma4_config.unwrap();
    let ple_h = g4.ple_hidden_size;
    let ple_total = num_hidden_layers * ple_h;
    let ple_gate = lw.ple_gate.as_ref().unwrap();
    let ple_proj = lw.ple_projection.as_ref().unwrap();
    let ple_post_norm = lw.ple_post_norm.as_ref().unwrap();
    let ple_buf = bufs.ple_per_layer_input.as_ref().unwrap();
    let ple_scratch = bufs.ple_scratch.as_ref().unwrap();

    // 1. Standalone FFN residual add: hidden_state = residual + ffn_down
    ops::encode_residual_add(
        enc,
        &pipelines.elementwise.residual_add,
        &bufs.residual,
        &bufs.ffn_down,
        &bufs.hidden_state,
        (token_count * h) as u32,
    );
    enc.memory_barrier_with_resources(&[&bufs.hidden_state]);

    // 2. PLE gate: linear(hidden_state → ple_scratch) [hidden → ple_hidden]
    encode_projection(
        enc,
        &bufs.hidden_state,
        ple_gate,
        ple_scratch,
        pipelines,
        token_count,
        ple_h,
        h,
    )?;
    enc.memory_barrier_with_resources(&[ple_scratch]);

    // 3. GELU activation + multiply with per-layer input slice
    ops::encode_gelu_gate(
        enc,
        &pipelines.activation.ple_gelu_gate,
        ple_scratch,
        ple_buf,
        ple_scratch, // in-place
        ple_h as u32,
        token_count as u32,
        ple_total as u32,           // stride: full row width
        (layer_idx * ple_h) as u32, // offset: this layer's slice
    );
    enc.memory_barrier_with_resources(&[ple_scratch]);

    // 4. Project back: linear(ple_scratch → ffn_down) [ple_hidden → hidden]
    encode_projection(
        enc,
        ple_scratch,
        ple_proj,
        &bufs.ffn_down, // reuse ffn_down as temp
        pipelines,
        token_count,
        h,
        ple_h,
    )?;
    enc.memory_barrier_with_resources(&[&bufs.ffn_down]);

    // 5. RMSNorm the projected output
    ops::encode_rms_norm(
        enc,
        &pipelines.norm.rms_norm,
        &ops::RmsNormParams {
            input: &bufs.ffn_down,
            weight: ple_post_norm,
            output: &bufs.ffn_up, // temp
            hidden_size: h as u32,
            token_count: token_count as u32,
            eps,
        },
    );
    enc.memory_barrier_with_resources(&[&bufs.ffn_up]);

    // 6. PLE residual add: hidden_state += normed PLE output
    ops::encode_residual_add(
        enc,
        &pipelines.elementwise.residual_add,
        &bufs.hidden_state,
        &bufs.ffn_up,
        &bufs.hidden_state, // in-place
        (token_count * h) as u32,
    );
    enc.memory_barrier_with_resources(&[&bufs.hidden_state]);

    // 7. Layer scalar: HF applies hidden_states *= layer_scalar
    //    AFTER all residual adds (including PLE).
    if let Some(ref scalar) = lw.layer_scalar {
        ops::encode_scale_buffer(
            enc,
            &pipelines.elementwise.scale_buffer,
            &bufs.hidden_state,
            scalar,
            (token_count * h) as u32,
        );
        enc.memory_barrier_with_resources(&[&bufs.hidden_state]);
    }

    // 8. Next layer's input norm (or skip for last layer)
    if let Some(next_norm) = next_input_norm {
        ops::encode_rms_norm(
            enc,
            &pipelines.norm.rms_norm,
            &ops::RmsNormParams {
                input: &bufs.hidden_state,
                weight: next_norm,
                output: &bufs.norm_out,
                hidden_size: h as u32,
                token_count: token_count as u32,
                eps,
            },
        );
        enc.memory_barrier_with_resources(&[&bufs.norm_out]);
    }

    Ok(true)
}
