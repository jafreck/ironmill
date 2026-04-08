//! Normalization pipeline states and dispatch helpers.

use ironmill_metal_sys::{ComputeEncoder, ComputePipeline, MetalBuffer};

use super::METAL_MAX_THREADS_PER_THREADGROUP;

/// Normalization pipeline states.
pub struct NormPipelines {
    /// RMSNorm kernel.
    pub rms_norm: ComputePipeline,
    /// Fused residual-add + RMSNorm kernel.
    pub fused_residual_rms_norm: ComputePipeline,
    /// Fused embedding lookup + first-layer norm kernel.
    pub fused_embedding_norm: ComputePipeline,
    /// Fused QK normalization + RoPE kernel.
    pub fused_qk_norm_rope: ComputePipeline,
    /// Fused softcapping kernel.
    pub fused_softcap: ComputePipeline,
}

// ── Parameter structs ────────────────────────────────────────────

/// Parameters for [`encode_rms_norm`].
pub struct RmsNormParams<'a> {
    /// Input buffer.
    pub input: &'a MetalBuffer,
    /// Norm weight buffer.
    pub weight: &'a MetalBuffer,
    /// Output buffer.
    pub output: &'a MetalBuffer,
    /// Hidden dimension size.
    pub hidden_size: u32,
    /// Number of tokens in the batch.
    pub token_count: u32,
    /// Epsilon for numerical stability.
    pub eps: f32,
}

/// Parameters for [`encode_fused_residual_rms_norm`].
pub struct FusedResidualRmsNormParams<'a> {
    /// First input buffer for residual addition.
    pub a: &'a MetalBuffer,
    /// Second input buffer for residual addition.
    pub b: &'a MetalBuffer,
    /// Norm weight buffer.
    pub weight: &'a MetalBuffer,
    /// Normalized output buffer.
    pub normed_output: &'a MetalBuffer,
    /// Residual output buffer.
    pub residual_output: &'a MetalBuffer,
    /// Epsilon for numerical stability.
    pub eps: f32,
    /// Hidden dimension size.
    pub hidden_size: u32,
    /// Number of tokens in the batch.
    pub token_count: u32,
}

// ── Dispatch helpers ─────────────────────────────────────────────

/// Encode an RMSNorm operation.
pub fn encode_rms_norm(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &RmsNormParams<'_>,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.input, 0, 0);
    encoder.set_buffer(params.weight, 0, 1);
    encoder.set_buffer(params.output, 0, 2);
    encoder.set_bytes(&params.hidden_size.to_le_bytes(), 3);
    encoder.set_bytes(&params.token_count.to_le_bytes(), 4);
    encoder.set_bytes(&params.eps.to_le_bytes(), 5);
    // Cap threadgroup size to Metal's 1024-thread limit. The shader uses a
    // strided loop so it handles hidden_size > tg_size correctly.
    let tg_size = METAL_MAX_THREADS_PER_THREADGROUP.min(params.hidden_size as usize);
    encoder.dispatch_threadgroups((params.token_count as usize, 1, 1), (tg_size, 1, 1));
}

/// Encode a fused residual-add + RMSNorm operation.
///
/// Computes `residual = a + b` and `normed = rms_norm(residual, weight)` in a
/// single kernel dispatch, avoiding the intermediate global-memory round-trip
/// that two separate dispatches would require.
pub fn encode_fused_residual_rms_norm(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &FusedResidualRmsNormParams<'_>,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.a, 0, 0);
    encoder.set_buffer(params.b, 0, 1);
    encoder.set_buffer(params.weight, 0, 2);
    encoder.set_buffer(params.normed_output, 0, 3);
    encoder.set_buffer(params.residual_output, 0, 4);
    encoder.set_bytes(&params.eps.to_le_bytes(), 5);
    encoder.set_bytes(&params.hidden_size.to_le_bytes(), 6);
    encoder.set_bytes(&params.token_count.to_le_bytes(), 7);
    let tg_size = METAL_MAX_THREADS_PER_THREADGROUP.min(params.hidden_size as usize);
    encoder.dispatch_threadgroups((params.token_count as usize, 1, 1), (tg_size, 1, 1));
}

/// Encode fused logit softcapping: `data[i] = softcap * tanh(data[i] / softcap)`.
pub fn encode_fused_softcap(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    data: &MetalBuffer,
    softcap: f32,
    count: u32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(data, 0, 0);
    encoder.set_bytes(&softcap.to_le_bytes(), 1);
    encoder.set_bytes(&count.to_le_bytes(), 2);
    let threads = count as usize;
    let tg_size = super::DEFAULT_THREADGROUP_WIDTH.min(threads);
    let tg_count = threads.div_ceil(tg_size);
    encoder.dispatch_threadgroups((tg_count, 1, 1), (tg_size, 1, 1));
}
