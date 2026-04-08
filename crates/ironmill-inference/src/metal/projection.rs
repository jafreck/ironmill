//! Linear projection dispatch by weight format.

use ironmill_metal_sys::{ComputeEncoder, MetalBuffer};

use super::buffers::Q8_GROUP_SIZE;
use super::ops;
use super::ops::LinearKernelKind;
use super::weights::{
    AffineQuantizedWeight, DualScaleQuantizedWeight, QuantizedWeight, WeightBuffer,
};
use crate::engine::InferenceError;

// ── Matmul tile dimensions — must match Metal shader constants ──
const MATMUL_TM_TILE: usize = 64;
const MATMUL_TN_TILE: usize = 64;
const MATMUL_THREADS_PER_TG: usize = 256;

/// Q8-quantized input references for INT4×Q8 decode path.
pub(crate) struct Q8Input<'a> {
    pub(crate) data: &'a MetalBuffer,
    pub(crate) scales: &'a MetalBuffer,
}

/// Encode a single Q/K/V-style linear projection.
///
/// Automatically selects the optimal kernel variant based on `token_count`:
/// - **Decode (token_count == 1):** memory-bandwidth-optimized matvec kernels.
/// - **Prefill (token_count > 1):** compute-optimized batched matmul kernels.
///
/// When `q8` is `Some(...)`, uses the INT4×Q8 integer dot product kernel for
/// affine INT4 decode, which is ~2× faster than the float dequant path.
///
/// This applies uniformly across all weight representations (Dense, PolarQuant,
/// AffineQuantized).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_projection(
    enc: &ComputeEncoder,
    input_buf: &MetalBuffer,
    weight: &WeightBuffer,
    output_buf: &MetalBuffer,
    pipelines: &super::ops::MetalPipelines,
    token_count: usize,
    out_features: usize,
    in_features: usize,
) -> Result<(), InferenceError> {
    encode_projection_q8(
        enc,
        input_buf,
        weight,
        output_buf,
        pipelines,
        token_count,
        out_features,
        in_features,
        None,
    )
}

/// Encode a linear projection with optional Q8 input for INT4 decode.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_projection_q8(
    enc: &ComputeEncoder,
    input_buf: &MetalBuffer,
    weight: &WeightBuffer,
    output_buf: &MetalBuffer,
    pipelines: &super::ops::MetalPipelines,
    token_count: usize,
    out_features: usize,
    in_features: usize,
    q8: Option<&Q8Input<'_>>,
) -> Result<(), InferenceError> {
    let kernel_kind = LinearKernelKind::for_token_count(token_count);

    match weight {
        WeightBuffer::Dense {
            buf: None,
            packed: None,
        } => {
            // Empty placeholder (e.g., GDN layer Q/K/V/O) — skip dispatch.
            return Ok(());
        }
        WeightBuffer::Dense { buf: _, packed } => {
            if let Some(packed_buf) = packed {
                let pipeline = pipelines.dense_linear_pipeline(kernel_kind);
                if kernel_kind.is_decode() {
                    ops::encode_matvec(
                        enc,
                        pipeline,
                        input_buf,
                        packed_buf,
                        output_buf,
                        out_features as u32,
                        in_features as u32,
                    );
                } else {
                    ops::encode_matmul(
                        enc,
                        pipeline,
                        input_buf,
                        packed_buf,
                        output_buf,
                        token_count as u32,
                        out_features as u32,
                        in_features as u32,
                    );
                }
                return Ok(());
            }
            return Err(InferenceError::runtime(
                "MPS dense fallback not supported in single-encoder mode",
            ));
        }
        WeightBuffer::Quantized(q) => {
            encode_polarquant_projection(
                enc,
                input_buf,
                q,
                output_buf,
                pipelines,
                kernel_kind,
                token_count,
            )?;
        }
        WeightBuffer::AffineQuantized(aq) => {
            // For decode + INT4 + no AWQ: use Q8 integer dot product if available.
            if let Some(q8_input) = q8 {
                if kernel_kind.is_decode() && aq.bit_width == 4 && aq.awq_scales.is_none() {
                    ops::encode_affine_matvec_int4xq8(
                        enc,
                        &pipelines.affine_matvec_int4xq8,
                        q8_input.data,
                        q8_input.scales,
                        aq,
                        output_buf,
                        out_features as u32,
                        in_features as u32,
                        Q8_GROUP_SIZE as u32,
                    );
                    return Ok(());
                }
            }
            encode_affine_projection(
                enc,
                input_buf,
                aq,
                output_buf,
                pipelines,
                kernel_kind,
                token_count,
            )?;
        }
        WeightBuffer::DualScaleQuantized(dq) => {
            encode_d2quant_projection(
                enc,
                input_buf,
                dq,
                output_buf,
                pipelines,
                kernel_kind,
                token_count,
            )?;
        }
    }
    Ok(())
}

// ── PolarQuant kernel dispatch ──────────────────────────────────

/// Encode a PolarQuant quantized projection via compute kernel.
///
/// Dispatches to the correct Metal kernel based on `n_bits` and kernel kind:
/// - Matvec (`LinearKernelKind::Matvec`): one threadgroup per output row
/// - Matmul (`LinearKernelKind::Matmul`): tiled matmul with `token_count` rows
fn encode_polarquant_projection(
    encoder: &ironmill_metal_sys::ComputeEncoder,
    input: &MetalBuffer,
    weight: &QuantizedWeight,
    output: &MetalBuffer,
    pipelines: &super::ops::MetalPipelines,
    kind: LinearKernelKind,
    token_count: usize,
) -> Result<(), InferenceError> {
    let (n, k) = weight.shape; // (out_features, in_features)

    let pipeline = pipelines
        .polarquant_pipeline(weight.n_bits.into(), kind)
        .ok_or_else(|| InferenceError::runtime(format!("unsupported n_bits: {}", weight.n_bits)))?;

    encoder.set_pipeline(pipeline);
    encoder.set_buffer(input, 0, 0);
    encoder.set_buffer(&weight.indices, 0, 1);
    encoder.set_buffer(&weight.lut, 0, 2);
    encoder.set_buffer(&weight.norms, 0, 3);
    encoder.set_buffer(output, 0, 4);

    if kind.is_decode() {
        // matvec: one threadgroup per output row
        encoder.set_bytes(&(n as u32).to_le_bytes(), 5);
        encoder.set_bytes(&(k as u32).to_le_bytes(), 6);
        let threads_per_group = 32; // SIMD width
        encoder.dispatch_threadgroups((n, 1, 1), (threads_per_group, 1, 1));
    } else {
        // matmul: simdgroup tiled (column-major dispatch)
        encoder.set_bytes(&(token_count as u32).to_le_bytes(), 5);
        encoder.set_bytes(&(n as u32).to_le_bytes(), 6);
        encoder.set_bytes(&(k as u32).to_le_bytes(), 7);
        encoder.dispatch_threadgroups(
            (
                token_count.div_ceil(MATMUL_TM_TILE),
                n.div_ceil(MATMUL_TN_TILE),
                1,
            ),
            (MATMUL_THREADS_PER_TG, 1, 1),
        );
    }
    Ok(())
}

/// Encode a fused affine quantized projection via compute kernel.
///
/// Dispatches to the correct Metal kernel based on `bit_width` and kernel kind:
/// - Matvec (`LinearKernelKind::Matvec`): one threadgroup per output row
/// - Matmul (`LinearKernelKind::Matmul`): tiled matmul with `token_count` rows
fn encode_affine_projection(
    encoder: &ironmill_metal_sys::ComputeEncoder,
    input: &MetalBuffer,
    weight: &AffineQuantizedWeight,
    output: &MetalBuffer,
    pipelines: &super::ops::MetalPipelines,
    kind: LinearKernelKind,
    token_count: usize,
) -> Result<(), InferenceError> {
    let (n, k) = weight.shape;

    let pipeline = pipelines
        .affine_pipeline(weight.bit_width.into(), kind)
        .ok_or_else(|| {
            InferenceError::runtime(format!(
                "unsupported affine bit_width: {}",
                weight.bit_width
            ))
        })?;

    encoder.set_pipeline(pipeline);
    encoder.set_buffer(input, 0, 0);
    encoder.set_buffer(&weight.data, 0, 1);
    encoder.set_buffer(&weight.scales, 0, 2);
    encoder.set_buffer(&weight.zeros, 0, 3);
    encoder.set_buffer(output, 0, 4);

    if kind.is_decode() {
        encoder.set_bytes(&(n as u32).to_le_bytes(), 5);
        encoder.set_bytes(&(k as u32).to_le_bytes(), 6);
        encoder.set_bytes(&weight.group_size.to_le_bytes(), 7);
        // AWQ scales: buffer 8 = scales data, buffer 9 = has_awq flag
        let has_awq: u32 = if weight.awq_scales.is_some() { 1 } else { 0 };
        if let Some(ref awq_buf) = weight.awq_scales {
            encoder.set_buffer(awq_buf, 0, 8);
        } else {
            // Bind the data buffer as a dummy (won't be read when has_awq=0).
            encoder.set_buffer(&weight.data, 0, 8);
        }
        encoder.set_bytes(&has_awq.to_le_bytes(), 9);
        let amx_rows_per_tg = 64usize;
        let threads_per_group = 256;
        let num_tgs = n.div_ceil(amx_rows_per_tg);
        encoder.dispatch_threadgroups((num_tgs, 1, 1), (threads_per_group, 1, 1));
    } else {
        encoder.set_bytes(&(token_count as u32).to_le_bytes(), 5);
        encoder.set_bytes(&(n as u32).to_le_bytes(), 6);
        encoder.set_bytes(&(k as u32).to_le_bytes(), 7);
        encoder.set_bytes(&weight.group_size.to_le_bytes(), 8);
        let has_awq: u32 = if weight.awq_scales.is_some() { 1 } else { 0 };
        if let Some(ref awq_buf) = weight.awq_scales {
            encoder.set_buffer(awq_buf, 0, 9);
        } else {
            encoder.set_buffer(&weight.data, 0, 9);
        }
        encoder.set_bytes(&has_awq.to_le_bytes(), 10);
        encoder.dispatch_threadgroups(
            (
                token_count.div_ceil(MATMUL_TM_TILE),
                n.div_ceil(MATMUL_TN_TILE),
                1,
            ),
            (MATMUL_THREADS_PER_TG, 1, 1),
        );
    }
    Ok(())
}

// ── D2Quant kernel dispatch ────────────────────────────────────

/// Encode a fused D2Quant dual-scale quantized projection via compute kernel.
///
/// Dispatches to the correct Metal kernel based on `bit_width` and kernel kind:
/// - Matvec (`LinearKernelKind::Matvec`): one threadgroup per output row
/// - Matmul (`LinearKernelKind::Matmul`): tiled matmul with `token_count` rows
#[allow(clippy::too_many_arguments)]
fn encode_d2quant_projection(
    encoder: &ironmill_metal_sys::ComputeEncoder,
    input: &MetalBuffer,
    weight: &DualScaleQuantizedWeight,
    output: &MetalBuffer,
    pipelines: &super::ops::MetalPipelines,
    kind: LinearKernelKind,
    token_count: usize,
) -> Result<(), InferenceError> {
    let (n, k) = weight.shape;

    let pipeline = pipelines
        .d2quant_pipeline(weight.bit_width.into(), kind)
        .ok_or_else(|| {
            InferenceError::runtime(format!(
                "unsupported d2quant bit_width: {}",
                weight.bit_width
            ))
        })?;

    encoder.set_pipeline(pipeline);
    encoder.set_buffer(input, 0, 0);
    encoder.set_buffer(&weight.data, 0, 1);
    encoder.set_buffer(&weight.normal_scale, 0, 2);
    encoder.set_buffer(&weight.normal_zero, 0, 3);
    encoder.set_buffer(&weight.outlier_scale, 0, 4);
    encoder.set_buffer(&weight.outlier_zero, 0, 5);
    encoder.set_buffer(&weight.outlier_mask, 0, 6);
    encoder.set_buffer(output, 0, 7);

    if kind.is_decode() {
        encoder.set_bytes(&(n as u32).to_le_bytes(), 8);
        encoder.set_bytes(&(k as u32).to_le_bytes(), 9);
        encoder.set_bytes(&weight.group_size.to_le_bytes(), 10);
        let amx_rows_per_tg = 64usize;
        let threads_per_group = 256;
        let num_tgs = n.div_ceil(amx_rows_per_tg);
        encoder.dispatch_threadgroups((num_tgs, 1, 1), (threads_per_group, 1, 1));
    } else {
        encoder.set_bytes(&(token_count as u32).to_le_bytes(), 8);
        encoder.set_bytes(&(n as u32).to_le_bytes(), 9);
        encoder.set_bytes(&(k as u32).to_le_bytes(), 10);
        encoder.set_bytes(&weight.group_size.to_le_bytes(), 11);
        encoder.dispatch_threadgroups(
            (
                token_count.div_ceil(MATMUL_TM_TILE),
                n.div_ceil(MATMUL_TN_TILE),
                1,
            ),
            (MATMUL_THREADS_PER_TG, 1, 1),
        );
    }
    Ok(())
}
