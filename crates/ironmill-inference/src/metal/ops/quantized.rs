//! Quantized matmul pipeline states and dispatch helpers.

use ironmill_metal_sys::{ComputeEncoder, ComputePipeline, MetalBuffer};

use super::{LinearKernelKind, METAL_MAX_THREADS_PER_THREADGROUP};

/// Affine quantized pipeline states.
pub struct AffinePipelines {
    /// Affine INT4 matvec kernel.
    pub matvec_int4: ComputePipeline,
    /// Affine INT4 matmul kernel.
    pub matmul_int4: ComputePipeline,
    /// Affine INT8 matvec kernel.
    pub matvec_int8: ComputePipeline,
    /// Affine INT8 matmul kernel.
    pub matmul_int8: ComputePipeline,
    /// AMX-accelerated INT4 matvec: dequant to threadgroup memory + simdgroup matrix multiply.
    pub matvec_int4_amx: ComputePipeline,
    /// AMX-accelerated INT8 matvec: dequant to threadgroup memory + simdgroup matrix multiply.
    pub matvec_int8_amx: ComputePipeline,
    /// Batched affine INT4 matvec for FFN gate+up in one dispatch.
    pub batched_matvec_int4: ComputePipeline,
    /// Batched affine INT4 matvec for 4 GDN projections in one dispatch.
    pub gdn_batched_matvec_int4: ComputePipeline,
    /// Fused FFN gate+up+activation for INT4 decode: gate+up dot products + SiLU/GELU inline.
    pub fused_ffn_gate_up_act_int4: ComputePipeline,
    /// INT4×Q8 integer dot product matvec (decode path).
    pub matvec_int4xq8: ComputePipeline,
    /// INT4 2-row matvec: 2 output rows per TG with 64 threads (2 simdgroups).
    pub matvec_int4_2row: ComputePipeline,
    /// INT4 affine embedding lookup with on-the-fly dequantization.
    pub embedding_lookup_int4: ComputePipeline,
}

impl AffinePipelines {
    /// Select the affine-quantized pipeline for a given bit-width and phase.
    #[inline]
    pub fn for_bits_and_kind(
        &self,
        bit_width: u32,
        kind: LinearKernelKind,
    ) -> Option<&ComputePipeline> {
        match (bit_width, kind) {
            (4, LinearKernelKind::Matvec) => Some(&self.matvec_int4),
            (4, LinearKernelKind::Matmul) => Some(&self.matmul_int4),
            (8, LinearKernelKind::Matvec) => Some(&self.matvec_int8),
            (8, LinearKernelKind::Matmul) => Some(&self.matmul_int8),
            _ => None,
        }
    }
}

/// PolarQuant pipeline states.
pub struct PolarQuantPipelines {
    /// PolarQuant INT4 matvec kernel.
    pub matvec_int4: ComputePipeline,
    /// PolarQuant INT4 matmul kernel.
    pub matmul_int4: ComputePipeline,
    /// PolarQuant INT8 matvec kernel.
    pub matvec_int8: ComputePipeline,
    /// PolarQuant INT8 matmul kernel.
    pub matmul_int8: ComputePipeline,
}

impl PolarQuantPipelines {
    /// Select the PolarQuant pipeline for a given bit-width and phase.
    #[inline]
    pub fn for_bits_and_kind(
        &self,
        n_bits: u32,
        kind: LinearKernelKind,
    ) -> Option<&ComputePipeline> {
        match (n_bits, kind) {
            (4, LinearKernelKind::Matvec) => Some(&self.matvec_int4),
            (4, LinearKernelKind::Matmul) => Some(&self.matmul_int4),
            (8, LinearKernelKind::Matvec) => Some(&self.matvec_int8),
            (8, LinearKernelKind::Matmul) => Some(&self.matmul_int8),
            _ => None,
        }
    }
}

/// D2Quant pipeline states.
pub struct D2QuantPipelines {
    /// D2Quant 3-bit matvec kernel.
    pub matvec_3bit: ComputePipeline,
    /// D2Quant 3-bit matmul kernel.
    pub matmul_3bit: ComputePipeline,
    /// D2Quant 3-bit embedding lookup kernel.
    pub embedding_lookup_3bit: ComputePipeline,
    /// D2Quant AMX-accelerated 3-bit matvec: dual-scale dequant + simdgroup matrix multiply.
    pub matvec_3bit_amx: ComputePipeline,
}

impl D2QuantPipelines {
    /// Select the D2Quant dual-scale pipeline for a given bit-width and phase.
    #[inline]
    pub fn for_bits_and_kind(
        &self,
        bit_width: u32,
        kind: LinearKernelKind,
    ) -> Option<&ComputePipeline> {
        match (bit_width, kind) {
            (3, LinearKernelKind::Matvec) => Some(&self.matvec_3bit),
            (3, LinearKernelKind::Matmul) => Some(&self.matmul_3bit),
            _ => None,
        }
    }
}

/// QuIP# pipeline states.
pub struct QuipPipelines {
    /// QuIP# matvec kernel.
    pub matvec: ComputePipeline,
    /// QuIP# matmul kernel.
    pub matmul: ComputePipeline,
}

impl QuipPipelines {
    /// Select the QuIP# pipeline for the given phase.
    #[inline]
    pub fn for_kind(&self, kind: LinearKernelKind) -> &ComputePipeline {
        match kind {
            LinearKernelKind::Matvec => &self.matvec,
            LinearKernelKind::Matmul => &self.matmul,
        }
    }
}

// ── Parameter structs ────────────────────────────────────────────

/// Parameters for [`encode_gdn_batched_affine_matvec_int4`].
pub struct GdnBatchedAffineInt4Params<'a> {
    /// Shared input buffer.
    pub input: &'a MetalBuffer,
    /// Projection 0 (QKV) weight.
    pub w0: &'a crate::metal::weights::AffineQuantizedWeight,
    /// Projection 0 output buffer.
    pub out0: &'a MetalBuffer,
    /// Projection 0 output dimension.
    pub n0: u32,
    /// Projection 1 (Z) weight.
    pub w1: &'a crate::metal::weights::AffineQuantizedWeight,
    /// Projection 1 output buffer.
    pub out1: &'a MetalBuffer,
    /// Projection 1 output dimension.
    pub n1: u32,
    /// Projection 2 (A) weight.
    pub w2: &'a crate::metal::weights::AffineQuantizedWeight,
    /// Projection 2 output buffer.
    pub out2: &'a MetalBuffer,
    /// Projection 2 output dimension.
    pub n2: u32,
    /// Projection 3 (B) weight.
    pub w3: &'a crate::metal::weights::AffineQuantizedWeight,
    /// Projection 3 output buffer.
    pub out3: &'a MetalBuffer,
    /// Projection 3 output dimension.
    pub n3: u32,
    /// Input dimension (hidden size).
    pub k: u32,
}

// ── Dispatch helpers ─────────────────────────────────────────────

/// Encode batched affine INT4 matvec for FFN gate+up in a single dispatch.
///
/// Computes gate = x · W_gate^T and up = x · W_up^T concurrently.
/// Saves 1 dispatch per layer compared to 2 separate `affine_matvec_int4` calls.
pub fn encode_batched_affine_matvec_int4(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    input: &MetalBuffer,
    gate_weight: &crate::metal::weights::AffineQuantizedWeight,
    gate_output: &MetalBuffer,
    up_weight: &crate::metal::weights::AffineQuantizedWeight,
    up_output: &MetalBuffer,
    n_gate: u32,
    k: u32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(input, 0, 0);
    encoder.set_buffer(&gate_weight.data, 0, 1);
    encoder.set_buffer(&gate_weight.scales, 0, 2);
    encoder.set_buffer(&gate_weight.zeros, 0, 3);
    encoder.set_buffer(gate_output, 0, 4);
    encoder.set_buffer(&up_weight.data, 0, 5);
    encoder.set_buffer(&up_weight.scales, 0, 6);
    encoder.set_buffer(&up_weight.zeros, 0, 7);
    encoder.set_buffer(up_output, 0, 8);
    encoder.set_bytes(&n_gate.to_le_bytes(), 9);
    encoder.set_bytes(&k.to_le_bytes(), 10);
    encoder.set_bytes(&gate_weight.group_size.to_le_bytes(), 11);
    // AWQ scales: use gate_weight's (shared between gate/up)
    if let Some(ref awq) = gate_weight.awq_scales {
        encoder.set_buffer(awq, 0, 12);
        encoder.set_bytes(&1u32.to_le_bytes(), 13);
    } else {
        encoder.set_buffer(&gate_weight.data, 0, 12); // dummy
        encoder.set_bytes(&0u32.to_le_bytes(), 13);
    }
    let tg_count = (2 * n_gate) as usize;
    encoder.dispatch_threadgroups((tg_count, 1, 1), (32, 1, 1));
}

/// Encode fused FFN gate+up+activation for INT4 decode.
///
/// Computes output[i] = activation(x · W_gate^T[i]) * (x · W_up^T[i]) in one dispatch.
/// Eliminates gate/up intermediate writes + the separate activation dispatch.
/// Saves 1 dispatch + 1 barrier per layer compared to batched_affine_matvec_int4 + silu_gate.
pub fn encode_fused_ffn_gate_up_act_int4(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    input: &MetalBuffer,
    gate_weight: &crate::metal::weights::AffineQuantizedWeight,
    up_weight: &crate::metal::weights::AffineQuantizedWeight,
    output: &MetalBuffer,
    n: u32,
    k: u32,
    use_gelu: bool,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(input, 0, 0);
    encoder.set_buffer(&gate_weight.data, 0, 1);
    encoder.set_buffer(&gate_weight.scales, 0, 2);
    encoder.set_buffer(&gate_weight.zeros, 0, 3);
    encoder.set_buffer(&up_weight.data, 0, 4);
    encoder.set_buffer(&up_weight.scales, 0, 5);
    encoder.set_buffer(&up_weight.zeros, 0, 6);
    encoder.set_buffer(output, 0, 7);
    encoder.set_bytes(&n.to_le_bytes(), 8);
    encoder.set_bytes(&k.to_le_bytes(), 9);
    encoder.set_bytes(&gate_weight.group_size.to_le_bytes(), 10);
    if let Some(ref awq) = gate_weight.awq_scales {
        encoder.set_buffer(awq, 0, 11);
        encoder.set_bytes(&1u32.to_le_bytes(), 12);
    } else {
        encoder.set_buffer(&gate_weight.data, 0, 11); // dummy
        encoder.set_bytes(&0u32.to_le_bytes(), 12);
    }
    encoder.set_bytes(&(use_gelu as u32).to_le_bytes(), 13);
    encoder.dispatch_threadgroups((n as usize, 1, 1), (32, 1, 1));
}

///
/// Computes qkv = x·W0^T, z = x·W1^T, a = x·W2^T, b = x·W3^T concurrently.
/// Saves 3 dispatches per GDN layer compared to 4 separate `affine_matvec_int4` calls.
pub(crate) fn encode_gdn_batched_affine_matvec_int4(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &GdnBatchedAffineInt4Params<'_>,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.input, 0, 0);
    encoder.set_buffer(&params.w0.data, 0, 1);
    encoder.set_buffer(&params.w0.scales, 0, 2);
    encoder.set_buffer(&params.w0.zeros, 0, 3);
    encoder.set_buffer(params.out0, 0, 4);
    encoder.set_buffer(&params.w1.data, 0, 5);
    encoder.set_buffer(&params.w1.scales, 0, 6);
    encoder.set_buffer(&params.w1.zeros, 0, 7);
    encoder.set_buffer(params.out1, 0, 8);
    encoder.set_buffer(&params.w2.data, 0, 9);
    encoder.set_buffer(&params.w2.scales, 0, 10);
    encoder.set_buffer(&params.w2.zeros, 0, 11);
    encoder.set_buffer(params.out2, 0, 12);
    encoder.set_buffer(&params.w3.data, 0, 13);
    encoder.set_buffer(&params.w3.scales, 0, 14);
    encoder.set_buffer(&params.w3.zeros, 0, 15);
    encoder.set_buffer(params.out3, 0, 16);
    let has_awq = params.w0.awq_scales.is_some() as u32;
    let params_words: [u32; 7] = [
        params.n0,
        params.n1,
        params.n2,
        params.n3,
        params.k,
        params.w0.group_size,
        has_awq,
    ];
    let params_bytes: Vec<u8> = params_words.iter().flat_map(|v| v.to_le_bytes()).collect();
    encoder.set_bytes(&params_bytes, 17);
    if let Some(ref awq) = params.w0.awq_scales {
        encoder.set_buffer(awq, 0, 18);
    } else {
        encoder.set_buffer(&params.w0.data, 0, 18); // dummy
    }
    let tg_count = (params.n0 + params.n1 + params.n2 + params.n3) as usize;
    encoder.dispatch_threadgroups((tg_count, 1, 1), (32, 1, 1));
}

/// Encode Q8 input quantization: FP16 → INT8 with per-group scale factors.
///
/// One dispatch quantizes the full input vector. The result is reused by
/// all subsequent INT4×Q8 projections reading the same input.
pub fn encode_quantize_input_q8(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    input: &MetalBuffer,
    q8_data: &MetalBuffer,
    q8_scales: &MetalBuffer,
    k: u32,
    group_size: u32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(input, 0, 0);
    encoder.set_buffer(q8_data, 0, 1);
    encoder.set_buffer(q8_scales, 0, 2);
    encoder.set_bytes(&k.to_le_bytes(), 3);
    encoder.set_bytes(&group_size.to_le_bytes(), 4);
    let num_groups = k.div_ceil(group_size) as usize;
    let tg_size = (group_size as usize).min(METAL_MAX_THREADS_PER_THREADGROUP);
    encoder.dispatch_threadgroups((num_groups, 1, 1), (tg_size, 1, 1));
}

/// Encode INT4×Q8 integer dot product matvec for decode.
///
/// Uses pre-quantized INT8 input and per-group scales instead of FP16 input.
/// The integer multiply-add inner loop is ~2× faster than float dequant.
pub fn encode_affine_matvec_int4xq8(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    q8_data: &MetalBuffer,
    q8_scales: &MetalBuffer,
    weight: &crate::metal::weights::AffineQuantizedWeight,
    output: &MetalBuffer,
    n: u32,
    k: u32,
    q8_group_size: u32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(q8_data, 0, 0);
    encoder.set_buffer(q8_scales, 0, 1);
    encoder.set_buffer(&weight.data, 0, 2);
    encoder.set_buffer(&weight.scales, 0, 3);
    encoder.set_buffer(&weight.zeros, 0, 4);
    encoder.set_buffer(output, 0, 5);
    encoder.set_bytes(&n.to_le_bytes(), 6);
    encoder.set_bytes(&k.to_le_bytes(), 7);
    encoder.set_bytes(&weight.group_size.to_le_bytes(), 8);
    encoder.set_bytes(&q8_group_size.to_le_bytes(), 9);
    encoder.dispatch_threadgroups((n as usize, 1, 1), (32, 1, 1));
}
