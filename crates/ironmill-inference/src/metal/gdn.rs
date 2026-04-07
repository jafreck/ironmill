//! GDN (Gated Delta Network) state, GPU/CPU encode functions, and CPU fallback.

use ironmill_metal_sys::{ComputeEncoder, MetalBuffer, MetalDevice, StorageMode};

use super::buffers::{IntermediateBuffers, read_buffer_f32, read_weight_f32, write_buffer_f32};
use super::error::MetalError;
use super::ops;
use super::projection::encode_projection;
use super::weights::{LayerWeights, WeightBuffer};
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
    /// Recurrent update raw output `[value_dim]` FP16 (Private).
    pub(crate) gpu_raw_output: MetalBuffer,
    /// Gated output `[value_dim]` FP16 (Private) — input to out_proj.
    pub(crate) gpu_gated_output: MetalBuffer,
    /// Single-token input buffer for prefill `[hidden_size]` FP16 (Private).
    pub(crate) gpu_gdn_input: MetalBuffer,
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
        let gpu_raw_output = alloc_private(gdn_cfg.value_dim)?;
        let gpu_gated_output = alloc_private(gdn_cfg.value_dim)?;
        let gpu_gdn_input = alloc_private(hidden_size)?;

        Ok(Self {
            layers,
            config: gdn_cfg.clone(),
            scratch,
            gpu_temp_qkv,
            gpu_temp_z,
            gpu_temp_a,
            gpu_temp_b,
            gpu_conv_out,
            gpu_raw_output,
            gpu_gated_output,
            gpu_gdn_input,
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
        &pipelines.gdn_prefill_conv1d_silu,
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
        &pipelines.gdn_prefill_recurrent,
        &gdn.gpu_conv_out,
        &gdn.gpu_temp_a,
        &gdn.gpu_temp_b,
        lw.gdn_a_log.as_ref().unwrap(),
        lw.gdn_dt_bias.as_ref().unwrap(),
        lw.gdn_norm.as_ref().unwrap(),
        &gdn.gpu_temp_z,
        &layer_state.recurrent_state,
        &gdn.gpu_gated_output,
        token_count as u32,
        qkv_dim as u32,
        key_dim as u32,
        value_dim as u32,
        num_v_heads as u32,
        k_head_dim as u32,
        v_head_dim as u32,
        eps,
        gdn_cfg.num_k_heads as u32,
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
        &pipelines.fused_residual_rms_norm,
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
            &pipelines.gdn_batched_matvec,
            &bufs.norm_out,
            p_qkv,
            p_z,
            p_a,
            p_b,
            &gdn.gpu_temp_qkv,
            &gdn.gpu_temp_z,
            &gdn.gpu_temp_a,
            &gdn.gpu_temp_b,
            h as u32,
            qkv_dim as u32,
            value_dim as u32,
            num_v_heads as u32,
            num_v_heads as u32,
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
                &pipelines.gdn_batched_affine_matvec_int4,
                &bufs.norm_out,
                aq_qkv,
                &gdn.gpu_temp_qkv,
                qkv_dim as u32,
                aq_z,
                &gdn.gpu_temp_z,
                value_dim as u32,
                aq_a,
                &gdn.gpu_temp_a,
                num_v_heads as u32,
                aq_b,
                &gdn.gpu_temp_b,
                num_v_heads as u32,
                h as u32,
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
        &pipelines.gdn_fused_decode,
        &gdn.gpu_temp_qkv,
        lw.gdn_conv1d_weight.as_ref().unwrap(),
        &layer_state.conv_state,
        &gdn.gpu_temp_a,
        &gdn.gpu_temp_b,
        lw.gdn_a_log.as_ref().unwrap(),
        lw.gdn_dt_bias.as_ref().unwrap(),
        &layer_state.recurrent_state,
        &gdn.gpu_temp_z,
        lw.gdn_norm.as_ref().unwrap(),
        &gdn.gpu_gated_output,
        &gdn.gpu_conv_out, // reuse as conv_out_scratch
        qkv_dim as u32,
        gdn_cfg.conv_kernel_size as u32,
        key_dim as u32,
        value_dim as u32,
        num_v_heads as u32,
        k_head_dim as u32,
        v_head_dim as u32,
        gdn_cfg.num_k_heads as u32,
        eps,
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
        &pipelines.fused_residual_rms_norm,
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

// ── GDN (Gated Delta Network) CPU inference ─────────────────────

/// Run one GDN layer on CPU for `token_count` tokens.
///
/// Reads `norm_out` (the layer input) and writes the result to the GDN
/// scratch buffer. Updates GDN conv and recurrent state.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_gdn_layer_cpu(
    layer_idx: usize,
    token_count: usize,
    hidden_size: usize,
    eps: f32,
    bufs: &IntermediateBuffers,
    lw: &LayerWeights,
    gdn_state: &mut GdnState,
) -> Result<(), InferenceError> {
    let gdn_cfg = gdn_state.config.clone();
    let gdn_idx = gdn_state
        .gdn_index_for_layer(layer_idx)
        .ok_or_else(|| InferenceError::runtime(format!("layer {layer_idx} is not a GDN layer")))?;

    let qkv_dim = gdn_cfg.qkv_dim;
    let key_dim = gdn_cfg.key_dim;
    let value_dim = gdn_cfg.value_dim;
    let _num_k_heads = gdn_cfg.num_k_heads;
    let num_v_heads = gdn_cfg.num_v_heads;
    let k_head_dim = gdn_cfg.k_head_dim;
    let v_head_dim = gdn_cfg.v_head_dim;
    let kernel_size = gdn_cfg.conv_kernel_size;
    let conv_state_width = kernel_size - 1; // columns of history kept

    // Load weights to CPU (f32).
    let w_qkv = read_weight_f32(lw.gdn_in_proj_qkv.as_ref().unwrap(), qkv_dim * hidden_size)?;
    let w_z = read_weight_f32(lw.gdn_in_proj_z.as_ref().unwrap(), value_dim * hidden_size)?;
    let w_a = read_weight_f32(
        lw.gdn_in_proj_a.as_ref().unwrap(),
        num_v_heads * hidden_size,
    )?;
    let w_b = read_weight_f32(
        lw.gdn_in_proj_b.as_ref().unwrap(),
        num_v_heads * hidden_size,
    )?;
    // Conv1d weight: [qkv_dim, 1, kernel_size] → we flatten to [qkv_dim, kernel_size]
    let conv_w = read_buffer_f32(
        lw.gdn_conv1d_weight.as_ref().unwrap(),
        qkv_dim * kernel_size,
    )?;
    let a_log = read_buffer_f32(lw.gdn_a_log.as_ref().unwrap(), num_v_heads)?;
    let dt_bias = read_buffer_f32(lw.gdn_dt_bias.as_ref().unwrap(), num_v_heads)?;
    let w_out = read_weight_f32(lw.gdn_out_proj.as_ref().unwrap(), hidden_size * value_dim)?;
    let norm_w = read_buffer_f32(lw.gdn_norm.as_ref().unwrap(), v_head_dim)?;

    // Load conv state: [qkv_dim, conv_state_width]
    let mut conv_state_buf = read_buffer_f32(
        &gdn_state.layers[gdn_idx].conv_state,
        qkv_dim * conv_state_width,
    )?;
    // Load recurrent state: [num_v_heads, v_head_dim, k_head_dim]
    let mut rec_state = read_buffer_f32(
        &gdn_state.layers[gdn_idx].recurrent_state,
        num_v_heads * v_head_dim * k_head_dim,
    )?;

    // Read input: norm_out [token_count, hidden_size]
    let input = read_buffer_f32(&bufs.norm_out, token_count * hidden_size)?;

    // Output buffer [token_count, hidden_size]
    let mut output = vec![0.0f32; token_count * hidden_size];

    // Process each token sequentially.
    for t in 0..token_count {
        let x = &input[t * hidden_size..(t + 1) * hidden_size];

        // 1. Projections: matvec on CPU
        let mut qkv_raw = vec![0.0f32; qkv_dim];
        matvec(&w_qkv, x, &mut qkv_raw, qkv_dim, hidden_size);

        let mut z = vec![0.0f32; value_dim];
        matvec(&w_z, x, &mut z, value_dim, hidden_size);

        let mut a_proj = vec![0.0f32; num_v_heads];
        matvec(&w_a, x, &mut a_proj, num_v_heads, hidden_size);

        let mut b_proj = vec![0.0f32; num_v_heads];
        matvec(&w_b, x, &mut b_proj, num_v_heads, hidden_size);

        // 2. Causal conv1d: shift state left, append new qkv
        for ch in 0..qkv_dim {
            // Shift left by 1
            for j in 0..conv_state_width.saturating_sub(1) {
                conv_state_buf[ch * conv_state_width + j] =
                    conv_state_buf[ch * conv_state_width + j + 1];
            }
            // Append new value
            conv_state_buf[ch * conv_state_width + conv_state_width - 1] = qkv_raw[ch];
        }

        // Conv1d: for each channel, dot product of conv_state with kernel
        let mut qkv_conv = vec![0.0f32; qkv_dim];
        for ch in 0..qkv_dim {
            let mut sum = 0.0f32;
            for k in 0..kernel_size {
                // conv_state has conv_state_width = kernel_size - 1 columns of history,
                // plus we treat the current value as the last. The kernel covers all
                // kernel_size positions. Index into conv_state: position offset.
                let state_idx = k as isize - (kernel_size as isize - conv_state_width as isize);
                if state_idx >= 0 && (state_idx as usize) < conv_state_width {
                    sum += conv_state_buf[ch * conv_state_width + state_idx as usize]
                        * conv_w[ch * kernel_size + k];
                }
            }
            qkv_conv[ch] = sum;
        }

        // Apply SiLU
        for v in &mut qkv_conv {
            *v = *v * (1.0 / (1.0 + (-*v).exp())); // silu(x) = x * sigmoid(x)
        }

        // 3. Split: q [key_dim], k [key_dim], v [value_dim]
        let q_flat = &qkv_conv[..key_dim];
        let k_flat = &qkv_conv[key_dim..2 * key_dim];
        let v_flat = &qkv_conv[2 * key_dim..2 * key_dim + value_dim];

        // 5. Compute gates
        let mut beta = vec![0.0f32; num_v_heads];
        let mut decay = vec![0.0f32; num_v_heads];
        for h_idx in 0..num_v_heads {
            beta[h_idx] = 1.0 / (1.0 + (-b_proj[h_idx]).exp()); // sigmoid
            let dt = softplus(a_proj[h_idx] + dt_bias[h_idx]);
            let a_val = (-a_log[h_idx].exp()) * dt; // -exp(A_log) * dt
            decay[h_idx] = a_val.exp();
        }

        // 6. Recurrent state update + output
        let mut o_flat = vec![0.0f32; value_dim];
        for h_idx in 0..num_v_heads {
            let q_head = &q_flat[h_idx * k_head_dim..(h_idx + 1) * k_head_dim];
            let k_head = &k_flat[h_idx * k_head_dim..(h_idx + 1) * k_head_dim];
            let v_head = &v_flat[h_idx * v_head_dim..(h_idx + 1) * v_head_dim];

            let s_offset = h_idx * v_head_dim * k_head_dim;
            let d = decay[h_idx];
            let b = beta[h_idx];

            // S[h] = decay[h] * S[h] + beta[h] * outer(v[h], k[h])
            for vi in 0..v_head_dim {
                for ki in 0..k_head_dim {
                    let idx = s_offset + vi * k_head_dim + ki;
                    rec_state[idx] = d * rec_state[idx] + b * v_head[vi] * k_head[ki];
                }
            }

            // o[h] = S[h] @ q[h]
            for vi in 0..v_head_dim {
                let mut sum = 0.0f32;
                for ki in 0..k_head_dim {
                    sum += rec_state[s_offset + vi * k_head_dim + ki] * q_head[ki];
                }
                o_flat[h_idx * v_head_dim + vi] = sum;
            }
        }

        // 7. Output gating: per-head RMSNorm, then multiply by silu(z)
        let rms_eps = eps;
        for h_idx in 0..num_v_heads {
            let head_start = h_idx * v_head_dim;
            let head_slice = &mut o_flat[head_start..head_start + v_head_dim];

            // RMSNorm per head
            let mut sq_sum = 0.0f32;
            for &v in head_slice.iter() {
                sq_sum += v * v;
            }
            let rms = (sq_sum / v_head_dim as f32 + rms_eps).sqrt();
            for (i, v) in head_slice.iter_mut().enumerate() {
                *v = *v / rms * norm_w[i];
            }
        }

        // Multiply by silu(z)
        for i in 0..value_dim {
            let z_silu = z[i] * (1.0 / (1.0 + (-z[i]).exp()));
            o_flat[i] *= z_silu;
        }

        // 8. Output projection: output = out_proj @ o_gated
        let out_token = &mut output[t * hidden_size..(t + 1) * hidden_size];
        matvec(&w_out, &o_flat, out_token, hidden_size, value_dim);
    }

    // Write back conv state and recurrent state (borrows layers).
    {
        let state = &gdn_state.layers[gdn_idx];
        write_buffer_f32(&state.conv_state, &conv_state_buf)?;
        write_buffer_f32(&state.recurrent_state, &rec_state)?;
    }

    // Write output to GDN scratch buffer (Shared storage, CPU-writable).
    write_buffer_f32(&gdn_state.scratch, &output)?;

    Ok(())
}

/// Simple CPU matrix-vector multiply: y = W @ x, where W is [rows, cols].
fn matvec(w: &[f32], x: &[f32], y: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let mut sum = 0.0f32;
        let row_start = r * cols;
        for c in 0..cols {
            sum += w[row_start + c] * x[c];
        }
        y[r] = sum;
    }
}

/// softplus(x) = ln(1 + exp(x))
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x // Avoid overflow
    } else {
        (1.0 + x.exp()).ln()
    }
}
