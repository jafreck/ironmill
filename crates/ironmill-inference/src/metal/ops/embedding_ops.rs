//! Embedding pipeline states and dispatch helpers.

use ironmill_metal_sys::{ComputeEncoder, ComputePipeline, MetalBuffer};

use super::DEFAULT_THREADGROUP_WIDTH;

/// Embedding pipeline states.
pub struct EmbeddingPipelines {
    /// Token embedding lookup kernel.
    pub embedding_lookup: ComputePipeline,
}

// ── Parameter structs ────────────────────────────────────────────

/// Parameters for [`encode_embedding_lookup`].
pub struct EmbeddingLookupParams<'a> {
    /// Token ID buffer.
    pub token_ids: &'a MetalBuffer,
    /// Embedding weight table buffer.
    pub embedding_table: &'a MetalBuffer,
    /// Output buffer.
    pub output: &'a MetalBuffer,
    /// Hidden dimension size.
    pub hidden_size: u32,
    /// Number of tokens in the batch.
    pub token_count: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
}

/// Parameters for [`encode_d2quant_embedding_lookup`].
pub struct D2QuantEmbeddingLookupParams<'a> {
    /// Token ID buffer.
    pub token_ids: &'a MetalBuffer,
    /// D2Quant dual-scale quantized weight table.
    pub weight: &'a crate::metal::weights::DualScaleQuantizedWeight,
    /// Output buffer (FP16).
    pub output: &'a MetalBuffer,
    /// Hidden dimension size (K = num_layers * ple_hidden_size).
    pub hidden_size: u32,
    /// Number of tokens in the batch.
    pub token_count: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
}

// ── Dispatch helpers ─────────────────────────────────────────────

/// Encode embedding lookup.
pub fn encode_embedding_lookup(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &EmbeddingLookupParams<'_>,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.token_ids, 0, 0);
    encoder.set_buffer(params.embedding_table, 0, 1);
    encoder.set_buffer(params.output, 0, 2);
    encoder.set_bytes(&params.hidden_size.to_le_bytes(), 3);
    encoder.set_bytes(&params.token_count.to_le_bytes(), 4);
    encoder.set_bytes(&params.vocab_size.to_le_bytes(), 5);
    let tg_size = DEFAULT_THREADGROUP_WIDTH.min(params.hidden_size as usize);
    encoder.dispatch_threadgroups(
        (
            (params.hidden_size as usize).div_ceil(tg_size),
            params.token_count as usize,
            1,
        ),
        (tg_size, 1, 1),
    );
}

/// Encode D2Quant 3-bit embedding lookup: gather rows from a packed
/// dual-scale quantized table and dequantize to FP16.
pub fn encode_d2quant_embedding_lookup(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &D2QuantEmbeddingLookupParams<'_>,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.token_ids, 0, 0);
    encoder.set_buffer(&params.weight.data, 0, 1);
    encoder.set_buffer(&params.weight.normal_scale, 0, 2);
    encoder.set_buffer(&params.weight.normal_zero, 0, 3);
    encoder.set_buffer(&params.weight.outlier_scale, 0, 4);
    encoder.set_buffer(&params.weight.outlier_zero, 0, 5);
    encoder.set_buffer(&params.weight.outlier_mask, 0, 6);
    encoder.set_buffer(params.output, 0, 7);
    encoder.set_bytes(&params.hidden_size.to_le_bytes(), 8);
    encoder.set_bytes(&params.token_count.to_le_bytes(), 9);
    encoder.set_bytes(&params.vocab_size.to_le_bytes(), 10);
    encoder.set_bytes(&params.weight.group_size.to_le_bytes(), 11);
    let tg_size = DEFAULT_THREADGROUP_WIDTH.min(params.hidden_size as usize);
    encoder.dispatch_threadgroups(
        (
            (params.hidden_size as usize).div_ceil(tg_size),
            params.token_count as usize,
            1,
        ),
        (tg_size, 1, 1),
    );
}
