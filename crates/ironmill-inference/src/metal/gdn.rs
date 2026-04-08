//! GDN (Gated Delta Network) state, GPU/CPU encode functions, and CPU fallback.

use ironmill_metal_sys::{ComputeEncoder, MetalBuffer, MetalDevice, StorageMode};

use super::buffers::IntermediateBuffers;
use super::error::MetalError;
use super::ops;
use super::projection::encode_projection;
use super::weights::WeightBuffer;
use crate::engine::InferenceError;

/// Per-layer GDN recurrent state buffers.
pub(crate) struct GdnLayerState {
    /// Conv1d sliding window state `[qkv_dim, kernel_size - 1]` FP16.
    pub(crate) conv_state: MetalBuffer,
    /// Recurrent state per head `[num_v_heads, v_head_dim, k_head_dim]` FP16.
    pub(crate) recurrent_state: MetalBuffer,
}

/// GDN state for all linear-attention layers.
pub(crate) struct GdnState {
    /// Per-GDN-layer state (indexed by position in `gdn_layer_indices`).
    pub(crate) layers: Vec<GdnLayerState>,
    /// Model config for GDN layers.
    pub(crate) config: super::config::GdnModelConfig,
    /// Shared-mode scratch buffer for CPU→GPU GDN output transfer.
    /// Size: `[max_tokens * hidden_size]` FP16.
    pub(crate) scratch: MetalBuffer,
    // ── GPU intermediate buffers (shared across all GDN layers) ──
    /// QKV projection output `[qkv_dim]` FP16 (Private).
    pub(crate) gpu_temp_qkv: MetalBuffer,
    /// Z projection output `[value_dim]` FP16 (Private).
    pub(crate) gpu_temp_z: MetalBuffer,
    /// Alpha projection output `[num_v_heads]` FP16 (Private).
    pub(crate) gpu_temp_a: MetalBuffer,
    /// Beta projection output `[num_v_heads]` FP16 (Private).
    pub(crate) gpu_temp_b: MetalBuffer,
    /// Conv1d + SiLU output `[qkv_dim]` FP16 (Private).
    pub(crate) gpu_conv_out: MetalBuffer,
    /// Gated output `[value_dim]` FP16 (Private) — input to out_proj.
    pub(crate) gpu_gated_output: MetalBuffer,
}

impl GdnState {
    pub(crate) fn new(
        device: &MetalDevice,
        gdn_cfg: &super::config::GdnModelConfig,
        hidden_size: usize,
    ) -> Result<Self, MetalError> {
        let conv_state_size = gdn_cfg.qkv_dim * (gdn_cfg.conv_kernel_size - 1) * 4; // FP32
        let recurrent_state_size =
            gdn_cfg.num_v_heads * gdn_cfg.v_head_dim * gdn_cfg.k_head_dim * 4; // FP16

        let mut layers = Vec::with_capacity(gdn_cfg.gdn_layer_indices.len());
        for _ in &gdn_cfg.gdn_layer_indices {
            let conv_state = device
                .create_buffer(conv_state_size.max(16), StorageMode::Shared)
                .map_err(MetalError::Metal)?;
            // Zero-initialize
            conv_state
                .write_bytes(&vec![0u8; conv_state_size], 0)
                .map_err(MetalError::Metal)?;

            let recurrent_state = device
                .create_buffer(recurrent_state_size.max(16), StorageMode::Shared)
                .map_err(MetalError::Metal)?;
            recurrent_state
                .write_bytes(&vec![0u8; recurrent_state_size], 0)
                .map_err(MetalError::Metal)?;

            layers.push(GdnLayerState {
                conv_state,
                recurrent_state,
            });
        }

        // Scratch buffer for GPU→post-GDN output. Now Private since GDN
        // runs entirely on GPU; only used as output_proj result buffer.
        let scratch_size = (hidden_size * 2).max(16); // 1 token × hidden_size × FP16
        let scratch = device
            .create_buffer(scratch_size, StorageMode::Private)
            .map_err(MetalError::Metal)?;

        // GPU intermediate buffers (shared across all GDN layers).
        let alloc_private = |size_elems: usize| -> Result<MetalBuffer, MetalError> {
            let bytes = (size_elems * 2).max(16); // FP16
            device
                .create_buffer(bytes, StorageMode::Private)
                .map_err(MetalError::Metal)
        };

        let gpu_temp_qkv = alloc_private(gdn_cfg.qkv_dim)?;
        let gpu_temp_z = alloc_private(gdn_cfg.value_dim)?;
        let gpu_temp_a = alloc_private(gdn_cfg.num_v_heads)?;
        let gpu_temp_b = alloc_private(gdn_cfg.num_v_heads)?;
        let gpu_conv_out = alloc_private(gdn_cfg.qkv_dim)?;
        let gpu_gated_output = alloc_private(gdn_cfg.value_dim)?;

        Ok(Self {
            layers,
            config: gdn_cfg.clone(),
            scratch,
            gpu_temp_qkv,
            gpu_temp_z,
            gpu_temp_a,
            gpu_temp_b,
            gpu_conv_out,
            gpu_gated_output,
        })
    }

    /// Ensure scratch and temp buffers are large enough for `token_count` tokens.
    pub(crate) fn ensure_scratch_capacity(
        &mut self,
        device: &MetalDevice,
        token_count: usize,
        hidden_size: usize,
    ) -> Result<(), MetalError> {
        let gdn_cfg = &self.config;

        // Scratch: token_count * hidden_size * 2
        let scratch_needed = token_count * hidden_size * 2;
        if self.scratch.length() < scratch_needed {
            self.scratch = device
                .create_buffer(scratch_needed, StorageMode::Private)
                .map_err(MetalError::Metal)?;
        }

        // Temp QKV: token_count * qkv_dim * 2
        let qkv_needed = token_count * gdn_cfg.qkv_dim * 2;
        if self.gpu_temp_qkv.length() < qkv_needed {
            self.gpu_temp_qkv = device
                .create_buffer(qkv_needed.max(16), StorageMode::Private)
                .map_err(MetalError::Metal)?;
        }

        // Temp Z: token_count * value_dim * 2
        let z_needed = token_count * gdn_cfg.value_dim * 2;
        if self.gpu_temp_z.length() < z_needed {
            self.gpu_temp_z = device
                .create_buffer(z_needed.max(16), StorageMode::Private)
                .map_err(MetalError::Metal)?;
        }

        // Temp A: token_count * num_v_heads * 2
        let a_needed = token_count * gdn_cfg.num_v_heads * 2;
        if self.gpu_temp_a.length() < a_needed {
            self.gpu_temp_a = device
                .create_buffer(a_needed.max(16), StorageMode::Private)
                .map_err(MetalError::Metal)?;
        }

        // Temp B: token_count * num_v_heads * 2
        if self.gpu_temp_b.length() < a_needed {
            self.gpu_temp_b = device
                .create_buffer(a_needed.max(16), StorageMode::Private)
                .map_err(MetalError::Metal)?;
        }

        // Conv out: token_count * qkv_dim * 2
        if self.gpu_conv_out.length() < qkv_needed {
            self.gpu_conv_out = device
                .create_buffer(qkv_needed.max(16), StorageMode::Private)
                .map_err(MetalError::Metal)?;
        }

        // Gated output: token_count * value_dim * 2
        if self.gpu_gated_output.length() < z_needed {
            self.gpu_gated_output = device
                .create_buffer(z_needed.max(16), StorageMode::Private)
                .map_err(MetalError::Metal)?;
        }

        Ok(())
    }

    /// Reset all state buffers to zero.
    pub(crate) fn reset(&self) -> Result<(), MetalError> {
        let gdn_cfg = &self.config;
        let conv_state_size = gdn_cfg.qkv_dim * (gdn_cfg.conv_kernel_size - 1) * 4; // FP32
        let recurrent_state_size =
            gdn_cfg.num_v_heads * gdn_cfg.v_head_dim * gdn_cfg.k_head_dim * 4;
        for layer in &self.layers {
            layer
                .conv_state
                .write_bytes(&vec![0u8; conv_state_size], 0)
                .map_err(MetalError::Metal)?;
            layer
                .recurrent_state
                .write_bytes(&vec![0u8; recurrent_state_size], 0)
                .map_err(MetalError::Metal)?;
        }
        Ok(())
    }

    /// Find the GDN state index for a given global layer index.
    pub(crate) fn gdn_index_for_layer(&self, layer_idx: usize) -> Option<usize> {
        self.config
            .gdn_layer_indices
            .iter()
            .position(|&l| l == layer_idx)
    }
}

/// Encode a full GDN (linear-attention) layer for prefill (token_count > 1).
///
/// Dispatches: QKV/Z/A/B projections → conv1d+SiLU → prefill recurrent →
/// output projection → residual + post-attention RMSNorm.
///
/// Shared between `run_pipeline_inner` and `run_pipeline_calibration`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gdn_prefill(
    enc: &ComputeEncoder,
    bufs: &IntermediateBuffers,
    gdn: &GdnState,
    lw: &super::weights::LayerWeights,
    pipelines: &super::ops::MetalPipelines,
    gdn_idx: usize,
    token_count: usize,
    h: usize,
    eps: f32,
) -> Result<(), InferenceError> {
    let gdn_cfg = &gdn.config;

    let qkv_dim = gdn_cfg.qkv_dim;
    let value_dim = gdn_cfg.value_dim;
    let num_v_heads = gdn_cfg.num_v_heads;
    let k_head_dim = gdn_cfg.k_head_dim;
    let v_head_dim = gdn_cfg.v_head_dim;
    let key_dim = gdn_cfg.key_dim;

    encode_projection(
        enc,
        &bufs.norm_out,
        lw.gdn_in_proj_qkv.as_ref().unwrap(),
        &gdn.gpu_temp_qkv,
        pipelines,
        token_count,
        qkv_dim,
        h,
    )?;
    encode_projection(
        enc,
        &bufs.norm_out,
        lw.gdn_in_proj_z.as_ref().unwrap(),
        &gdn.gpu_temp_z,
        pipelines,
        token_count,
        value_dim,
        h,
    )?;
    encode_projection(
        enc,
        &bufs.norm_out,
        lw.gdn_in_proj_a.as_ref().unwrap(),
        &gdn.gpu_temp_a,
        pipelines,
        token_count,
        num_v_heads,
        h,
    )?;
    encode_projection(
        enc,
        &bufs.norm_out,
        lw.gdn_in_proj_b.as_ref().unwrap(),
        &gdn.gpu_temp_b,
        pipelines,
        token_count,
        num_v_heads,
        h,
    )?;
    enc.memory_barrier_with_resources(&[
        &gdn.gpu_temp_qkv,
        &gdn.gpu_temp_z,
        &gdn.gpu_temp_a,
        &gdn.gpu_temp_b,
    ]);

    let layer_state = &gdn.layers[gdn_idx];
    ops::encode_gdn_prefill_conv1d_silu(
        enc,
        &pipelines.gdn.prefill_conv1d_silu,
        &gdn.gpu_temp_qkv,
        lw.gdn_conv1d_weight.as_ref().unwrap(),
        &layer_state.conv_state,
        &gdn.gpu_conv_out,
        qkv_dim as u32,
        gdn_cfg.conv_kernel_size as u32,
        token_count as u32,
    );
    enc.memory_barrier_with_resources(&[&gdn.gpu_conv_out]);

    ops::encode_gdn_prefill_recurrent(
        enc,
        &pipelines.gdn.prefill_recurrent,
        &ops::GdnPrefillRecurrentParams {
            all_conv_out: &gdn.gpu_conv_out,
            all_a: &gdn.gpu_temp_a,
            all_b: &gdn.gpu_temp_b,
            a_log: lw.gdn_a_log.as_ref().unwrap(),
            dt_bias: lw.gdn_dt_bias.as_ref().unwrap(),
            norm_weight: lw.gdn_norm.as_ref().unwrap(),
            all_z: &gdn.gpu_temp_z,
            recurrent_state: &layer_state.recurrent_state,
            all_output: &gdn.gpu_gated_output,
            token_count: token_count as u32,
            qkv_dim: qkv_dim as u32,
            key_dim: key_dim as u32,
            value_dim: value_dim as u32,
            num_v_heads: num_v_heads as u32,
            k_head_dim: k_head_dim as u32,
            v_head_dim: v_head_dim as u32,
            eps,
            num_k_heads: gdn_cfg.num_k_heads as u32,
        },
    );
    enc.memory_barrier_with_resources(&[&gdn.gpu_gated_output]);

    encode_projection(
        enc,
        &gdn.gpu_gated_output,
        lw.gdn_out_proj.as_ref().unwrap(),
        &gdn.scratch,
        pipelines,
        token_count,
        h,
        value_dim,
    )?;
    enc.memory_barrier_with_resources(&[&gdn.scratch]);

    ops::encode_fused_residual_rms_norm(
        enc,
        &pipelines.norm.fused_residual_rms_norm,
        &ops::FusedResidualRmsNormParams {
            a: &bufs.hidden_state,
            b: &gdn.scratch,
            weight: &lw.post_attn_norm,
            normed_output: &bufs.norm_out,
            residual_output: &bufs.residual,
            eps,
            hidden_size: h as u32,
            token_count: token_count as u32,
        },
    );
    Ok(())
}

/// Encode a full GDN layer for single-token decode.
///
/// Dispatches: QKV/Z/A/B projections → conv1d+SiLU → recurrent update →
/// output gate → output projection → residual + post-attention RMSNorm.
///
/// Shared between `run_pipeline_inner` and `run_pipeline_calibration`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gdn_decode(
    enc: &ComputeEncoder,
    bufs: &IntermediateBuffers,
    gdn: &GdnState,
    lw: &super::weights::LayerWeights,
    pipelines: &super::ops::MetalPipelines,
    gdn_idx: usize,
    h: usize,
    eps: f32,
    skip_qkv: bool,
) -> Result<(), InferenceError> {
    let gdn_cfg = &gdn.config;

    let qkv_dim = gdn_cfg.qkv_dim;
    let value_dim = gdn_cfg.value_dim;
    let num_v_heads = gdn_cfg.num_v_heads;
    let k_head_dim = gdn_cfg.k_head_dim;
    let v_head_dim = gdn_cfg.v_head_dim;
    let key_dim = gdn_cfg.key_dim;

    // Try batched matvec for all 4 GDN projections in one dispatch.
    let w_qkv = lw.gdn_in_proj_qkv.as_ref().unwrap();
    let w_z = lw.gdn_in_proj_z.as_ref().unwrap();
    let w_a = lw.gdn_in_proj_a.as_ref().unwrap();
    let w_b = lw.gdn_in_proj_b.as_ref().unwrap();

    if skip_qkv {
        // QKV was pre-computed by P1 fusion — only dispatch Z/A/B projections.
        for (weight, output_buf, out_features) in [
            (w_z, &gdn.gpu_temp_z, value_dim),
            (w_a, &gdn.gpu_temp_a, num_v_heads),
            (w_b, &gdn.gpu_temp_b, num_v_heads),
        ] {
            encode_projection(
                enc,
                &bufs.norm_out,
                weight,
                output_buf,
                pipelines,
                1,
                out_features,
                h,
            )?;
        }
    } else if let (Some(p_qkv), Some(p_z), Some(p_a), Some(p_b)) = (
        w_qkv.packed_buf(),
        w_z.packed_buf(),
        w_a.packed_buf(),
        w_b.packed_buf(),
    ) {
        // All 4 weights are dense with packed buffers: use batched FP16 matvec.
        ops::encode_gdn_batched_matvec(
            enc,
            &pipelines.gdn.batched_matvec,
            &ops::GdnBatchedMatvecParams {
                input: &bufs.norm_out,
                w_qkv: p_qkv,
                w_z: p_z,
                w_a: p_a,
                w_b: p_b,
                y_qkv: &gdn.gpu_temp_qkv,
                y_z: &gdn.gpu_temp_z,
                y_a: &gdn.gpu_temp_a,
                y_b: &gdn.gpu_temp_b,
                k: h as u32,
                n_qkv: qkv_dim as u32,
                n_z: value_dim as u32,
                n_a: num_v_heads as u32,
                n_b: num_v_heads as u32,
            },
        );
    } else if let (
        WeightBuffer::AffineQuantized(aq_qkv),
        WeightBuffer::AffineQuantized(aq_z),
        WeightBuffer::AffineQuantized(aq_a),
        WeightBuffer::AffineQuantized(aq_b),
    ) = (w_qkv, w_z, w_a, w_b)
    {
        if aq_qkv.bit_width == 4
            && aq_z.bit_width == 4
            && aq_a.bit_width == 4
            && aq_b.bit_width == 4
        {
            // All 4 weights are INT4 affine: use batched INT4 matvec.
            ops::encode_gdn_batched_affine_matvec_int4(
                enc,
                &pipelines.affine.gdn_batched_matvec_int4,
                &ops::GdnBatchedAffineInt4Params {
                    input: &bufs.norm_out,
                    w0: aq_qkv,
                    out0: &gdn.gpu_temp_qkv,
                    n0: qkv_dim as u32,
                    w1: aq_z,
                    out1: &gdn.gpu_temp_z,
                    n1: value_dim as u32,
                    w2: aq_a,
                    out2: &gdn.gpu_temp_a,
                    n2: num_v_heads as u32,
                    w3: aq_b,
                    out3: &gdn.gpu_temp_b,
                    n3: num_v_heads as u32,
                    k: h as u32,
                },
            );
        } else {
            // Mixed bit widths: fallback to individual projections.
            for (weight, output_buf, out_features) in [
                (w_qkv as &WeightBuffer, &gdn.gpu_temp_qkv, qkv_dim),
                (w_z as &WeightBuffer, &gdn.gpu_temp_z, value_dim),
                (w_a as &WeightBuffer, &gdn.gpu_temp_a, num_v_heads),
                (w_b as &WeightBuffer, &gdn.gpu_temp_b, num_v_heads),
            ] {
                encode_projection(
                    enc,
                    &bufs.norm_out,
                    weight,
                    output_buf,
                    pipelines,
                    1,
                    out_features,
                    h,
                )?;
            }
        }
    } else {
        // Fallback: individual projections for other weight types.
        for (weight, output_buf, out_features) in [
            (w_qkv, &gdn.gpu_temp_qkv, qkv_dim),
            (w_z, &gdn.gpu_temp_z, value_dim),
            (w_a, &gdn.gpu_temp_a, num_v_heads),
            (w_b, &gdn.gpu_temp_b, num_v_heads),
        ] {
            encode_projection(
                enc,
                &bufs.norm_out,
                weight,
                output_buf,
                pipelines,
                1,
                out_features,
                h,
            )?;
        }
    }
    enc.memory_barrier_with_resources(&[
        &gdn.gpu_temp_qkv,
        &gdn.gpu_temp_z,
        &gdn.gpu_temp_a,
        &gdn.gpu_temp_b,
    ]);

    // Fused conv1d+SiLU + recurrent update + output gate in a single dispatch.
    // Replaces 3 separate dispatches (conv1d, recurrent, output_gate) and
    // 2 intermediate barriers.
    let layer_state = &gdn.layers[gdn_idx];
    ops::encode_gdn_fused_decode(
        enc,
        &pipelines.gdn.fused_decode,
        &ops::GdnFusedDecodeParams {
            input_qkv: &gdn.gpu_temp_qkv,
            conv_weight: lw.gdn_conv1d_weight.as_ref().unwrap(),
            conv_state: &layer_state.conv_state,
            a_proj: &gdn.gpu_temp_a,
            b_proj: &gdn.gpu_temp_b,
            a_log: lw.gdn_a_log.as_ref().unwrap(),
            dt_bias: lw.gdn_dt_bias.as_ref().unwrap(),
            recurrent_state: &layer_state.recurrent_state,
            z_proj: &gdn.gpu_temp_z,
            norm_weight: lw.gdn_norm.as_ref().unwrap(),
            output: &gdn.gpu_gated_output,
            conv_out_scratch: &gdn.gpu_conv_out, // reuse as conv_out_scratch
            qkv_dim: qkv_dim as u32,
            kernel_size: gdn_cfg.conv_kernel_size as u32,
            key_dim: key_dim as u32,
            value_dim: value_dim as u32,
            num_v_heads: num_v_heads as u32,
            k_head_dim: k_head_dim as u32,
            v_head_dim: v_head_dim as u32,
            num_k_heads: gdn_cfg.num_k_heads as u32,
            eps,
        },
    );
    enc.memory_barrier_with_resources(&[&gdn.gpu_gated_output]);

    encode_projection(
        enc,
        &gdn.gpu_gated_output,
        lw.gdn_out_proj.as_ref().unwrap(),
        &gdn.scratch,
        pipelines,
        1,
        h,
        value_dim,
    )?;
    enc.memory_barrier_with_resources(&[&gdn.scratch]);

    ops::encode_fused_residual_rms_norm(
        enc,
        &pipelines.norm.fused_residual_rms_norm,
        &ops::FusedResidualRmsNormParams {
            a: &bufs.hidden_state,
            b: &gdn.scratch,
            weight: &lw.post_attn_norm,
            normed_output: &bufs.norm_out,
            residual_output: &bufs.residual,
            eps,
            hidden_size: h as u32,
            token_count: 1,
        },
    );
    Ok(())
}
