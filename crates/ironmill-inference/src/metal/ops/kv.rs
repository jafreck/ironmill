//! KV cache pipeline states and dispatch helpers.

use ironmill_metal_sys::{ComputeEncoder, ComputePipeline, MetalBuffer};

use super::DEFAULT_THREADGROUP_WIDTH;

/// KV cache pipeline states.
pub struct KvPipelines {
    /// KV scatter (cache write) kernel.
    pub kv_scatter: ComputePipeline,
}

// ── Parameter structs ────────────────────────────────────────────

/// Parameters for [`encode_kv_scatter`].
pub struct KvScatterParams<'a> {
    /// Projection buffer to scatter into cache.
    pub proj: &'a MetalBuffer,
    /// KV cache buffer.
    pub cache: &'a MetalBuffer,
    /// Current sequence position.
    pub seq_pos: u32,
    /// Number of tokens in the batch.
    pub token_count: u32,
    /// Number of key-value heads.
    pub num_kv_heads: u32,
    /// Per-head dimension.
    pub head_dim: u32,
    /// Maximum sequence length (cache stride).
    pub max_seq_len: u32,
}

// ── Dispatch helpers ─────────────────────────────────────────────

/// Encode KV scatter — copy projections into FP16 KV cache on GPU.
pub fn encode_kv_scatter(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &KvScatterParams<'_>,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.proj, 0, 0);
    encoder.set_buffer(params.cache, 0, 1);
    encoder.set_bytes(&params.seq_pos.to_le_bytes(), 2);
    encoder.set_bytes(&params.token_count.to_le_bytes(), 3);
    encoder.set_bytes(&params.num_kv_heads.to_le_bytes(), 4);
    encoder.set_bytes(&params.head_dim.to_le_bytes(), 5);
    encoder.set_bytes(&params.max_seq_len.to_le_bytes(), 6);
    let tg_x = (params.head_dim as usize).min(DEFAULT_THREADGROUP_WIDTH);
    encoder.dispatch_threads(
        (
            params.head_dim as usize,
            params.num_kv_heads as usize,
            params.token_count as usize,
        ),
        (tg_x, 1, 1),
    );
}
