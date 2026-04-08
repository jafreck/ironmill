//! Fused operation pipeline states and dispatch helpers.

use ironmill_metal_sys::{ComputeEncoder, ComputePipeline, MetalBuffer};

use super::DEFAULT_THREADGROUP_WIDTH;

/// Fused operation pipeline states.
pub struct FusedPipelines {
    /// Fused residual+RMSNorm+dense matvec in one dispatch.
    pub residual_norm_matvec: ComputePipeline,
    /// Fused residual+RMSNorm+affine INT4 matvec in one dispatch.
    pub residual_norm_affine_matvec_int4: ComputePipeline,
    /// INT4 dequantization kernel.
    pub int4_dequantize: ComputePipeline,
}

// ── Dispatch helpers ─────────────────────────────────────────────

/// Encode fused residual + RMSNorm + dense FP16 matvec.
///
/// Replaces `encode_fused_residual_rms_norm` + barrier + `encode_matvec`.
/// Each threadgroup computes (a+b), derives rms_inv, and performs the
/// matvec using the normed input — all in one dispatch.
/// Threadgroup 0 writes `residual_output = a + b` and `normed_output = RMSNorm(a+b)`.
#[allow(clippy::too_many_arguments)]
pub fn encode_fused_residual_norm_matvec(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    a: &MetalBuffer,
    b: &MetalBuffer,
    norm_weight: &MetalBuffer,
    residual_output: &MetalBuffer,
    w_packed: &MetalBuffer,
    y: &MetalBuffer,
    normed_output: &MetalBuffer,
    k: u32,
    n: u32,
    eps: f32,
) {
    const ROWS_PER_TG: u32 = 64;
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(a, 0, 0);
    encoder.set_buffer(b, 0, 1);
    encoder.set_buffer(norm_weight, 0, 2);
    encoder.set_buffer(residual_output, 0, 3);
    encoder.set_buffer(w_packed, 0, 4);
    encoder.set_buffer(y, 0, 5);
    let params: [u32; 4] = [k, n, eps.to_bits(), 0];
    let params_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
    encoder.set_bytes(&params_bytes, 6);
    encoder.set_buffer(normed_output, 0, 7);
    let tg_count = n.div_ceil(ROWS_PER_TG);
    encoder.dispatch_threadgroups((tg_count as usize, 1, 1), (DEFAULT_THREADGROUP_WIDTH, 1, 1));
}

/// Encode fused residual + RMSNorm + affine INT4 matvec.
///
/// Same fusion as `encode_fused_residual_norm_matvec` but for INT4 weights.
/// One threadgroup per output row, 32 threads per group.
/// Also writes `normed_output` for subsequent projections.
#[allow(clippy::too_many_arguments)]
pub fn encode_fused_residual_norm_affine_matvec_int4(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    a: &MetalBuffer,
    b: &MetalBuffer,
    norm_weight: &MetalBuffer,
    residual_output: &MetalBuffer,
    weight: &crate::metal::weights::AffineQuantizedWeight,
    output: &MetalBuffer,
    normed_output: &MetalBuffer,
    n: u32,
    k: u32,
    eps: f32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(a, 0, 0);
    encoder.set_buffer(b, 0, 1);
    encoder.set_buffer(norm_weight, 0, 2);
    encoder.set_buffer(residual_output, 0, 3);
    encoder.set_buffer(&weight.data, 0, 4);
    encoder.set_buffer(&weight.scales, 0, 5);
    encoder.set_buffer(&weight.zeros, 0, 6);
    encoder.set_buffer(output, 0, 7);
    let params: [u32; 4] = [n, k, weight.group_size, eps.to_bits()];
    let params_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
    encoder.set_bytes(&params_bytes, 8);
    if let Some(ref awq) = weight.awq_scales {
        encoder.set_buffer(awq, 0, 9);
        encoder.set_bytes(&1u32.to_le_bytes(), 10);
    } else {
        encoder.set_buffer(&weight.data, 0, 9); // dummy
        encoder.set_bytes(&0u32.to_le_bytes(), 10);
    }
    encoder.set_buffer(normed_output, 0, 11);
    encoder.dispatch_threadgroups((n as usize, 1, 1), (32, 1, 1));
}
