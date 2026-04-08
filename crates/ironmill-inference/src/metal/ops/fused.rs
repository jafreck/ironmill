//! Fused operation pipeline states and dispatch helpers.

use ironmill_metal_sys::{ComputeEncoder, ComputePipeline, MetalBuffer};

use super::DEFAULT_THREADGROUP_WIDTH;

/// Fused operation pipeline states.
pub struct FusedPipelines {
    /// Fused residual+RMSNorm+dense matvec in one dispatch.
    pub residual_norm_matvec: ComputePipeline,
    /// Fused residual+RMSNorm+affine INT4 matvec in one dispatch (blocked layout — legacy).
    pub residual_norm_affine_matvec_int4: ComputePipeline,
    /// INT4 dequantization kernel.
    pub int4_dequantize: ComputePipeline,
    /// Superblock fused residual+RMSNorm+affine INT4 matvec.
    pub sb_residual_norm_affine_matvec_int4: ComputePipeline,
}

// ── Parameter structs ────────────────────────────────────────────

/// Parameters for [`encode_fused_residual_norm_matvec`].
pub struct FusedResidualNormMatvecParams<'a> {
    /// First residual addend.
    pub a: &'a MetalBuffer,
    /// Second residual addend.
    pub b: &'a MetalBuffer,
    /// RMSNorm weight buffer.
    pub norm_weight: &'a MetalBuffer,
    /// Output buffer for residual sum (a + b).
    pub residual_output: &'a MetalBuffer,
    /// Packed FP16 weight matrix.
    pub w_packed: &'a MetalBuffer,
    /// Matvec output buffer.
    pub y: &'a MetalBuffer,
    /// Output buffer for the normed intermediate.
    pub normed_output: &'a MetalBuffer,
    /// Input dimension (hidden size).
    pub k: u32,
    /// Output dimension (projection size).
    pub n: u32,
    /// RMSNorm epsilon.
    pub eps: f32,
}

/// Parameters for [`encode_fused_residual_norm_affine_matvec_int4`].
pub struct FusedResidualNormAffineInt4Params<'a> {
    /// First residual addend.
    pub a: &'a MetalBuffer,
    /// Second residual addend.
    pub b: &'a MetalBuffer,
    /// RMSNorm weight buffer.
    pub norm_weight: &'a MetalBuffer,
    /// Output buffer for residual sum (a + b).
    pub residual_output: &'a MetalBuffer,
    /// Affine INT4 quantized weight.
    pub weight: &'a crate::metal::weights::AffineQuantizedWeight,
    /// Matvec output buffer.
    pub output: &'a MetalBuffer,
    /// Output buffer for the normed intermediate.
    pub normed_output: &'a MetalBuffer,
    /// Output dimension (projection size).
    pub n: u32,
    /// Input dimension (hidden size).
    pub k: u32,
    /// RMSNorm epsilon.
    pub eps: f32,
}

// ── Dispatch helpers ─────────────────────────────────────────────

/// Encode fused residual + RMSNorm + dense FP16 matvec.
///
/// Replaces `encode_fused_residual_rms_norm` + barrier + `encode_matvec`.
/// Each threadgroup computes (a+b), derives rms_inv, and performs the
/// matvec using the normed input — all in one dispatch.
/// Threadgroup 0 writes `residual_output = a + b` and `normed_output = RMSNorm(a+b)`.
pub(crate) fn encode_fused_residual_norm_matvec(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &FusedResidualNormMatvecParams<'_>,
) {
    const ROWS_PER_TG: u32 = 64;
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.a, 0, 0);
    encoder.set_buffer(params.b, 0, 1);
    encoder.set_buffer(params.norm_weight, 0, 2);
    encoder.set_buffer(params.residual_output, 0, 3);
    encoder.set_buffer(params.w_packed, 0, 4);
    encoder.set_buffer(params.y, 0, 5);
    let gpu_params: [u32; 4] = [params.k, params.n, params.eps.to_bits(), 0];
    let params_bytes: Vec<u8> = gpu_params.iter().flat_map(|v| v.to_le_bytes()).collect();
    encoder.set_bytes(&params_bytes, 6);
    encoder.set_buffer(params.normed_output, 0, 7);
    let tg_count = params.n.div_ceil(ROWS_PER_TG);
    encoder.dispatch_threadgroups((tg_count as usize, 1, 1), (DEFAULT_THREADGROUP_WIDTH, 1, 1));
}

/// Encode fused residual + RMSNorm + affine INT4 matvec.
///
/// Uses superblock layout: weight contains inline scale/zero.
/// One threadgroup per output row, 32 threads per group.
/// Also writes `normed_output` for subsequent projections.
pub(crate) fn encode_fused_residual_norm_affine_matvec_int4(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &FusedResidualNormAffineInt4Params<'_>,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.a, 0, 0);
    encoder.set_buffer(params.b, 0, 1);
    encoder.set_buffer(params.norm_weight, 0, 2);
    encoder.set_buffer(params.residual_output, 0, 3);
    encoder.set_buffer(&params.weight.data, 0, 4); // superblock
    encoder.set_buffer(params.output, 0, 5);
    let gpu_params: [u32; 4] = [
        params.n,
        params.k,
        params.weight.group_size,
        params.eps.to_bits(),
    ];
    let params_bytes: Vec<u8> = gpu_params.iter().flat_map(|v| v.to_le_bytes()).collect();
    encoder.set_bytes(&params_bytes, 6);
    if let Some(ref awq) = params.weight.awq_scales {
        encoder.set_buffer(awq, 0, 7);
        encoder.set_bytes(&1u32.to_le_bytes(), 8);
    } else {
        encoder.set_buffer(&params.weight.data, 0, 7); // dummy
        encoder.set_bytes(&0u32.to_le_bytes(), 8);
    }
    encoder.set_buffer(params.normed_output, 0, 9);
    encoder.dispatch_threadgroups((params.n as usize, 1, 1), (32, 1, 1));
}
