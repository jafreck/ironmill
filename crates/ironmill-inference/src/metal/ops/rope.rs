//! Rope pipeline states and dispatch helpers.

use ironmill_metal_sys::{ComputeEncoder, ComputePipeline, MetalBuffer};

use super::DEFAULT_THREADGROUP_WIDTH;

/// Rope pipeline states.
pub struct RopePipelines {
    /// Rotary positional embedding kernel.
    pub rope: ComputePipeline,
}

// ── Parameter structs ────────────────────────────────────────────

/// Parameters for [`encode_rope`].
pub struct RopeParams<'a> {
    /// Q or K projection buffer (modified in-place).
    pub qk: &'a MetalBuffer,
    /// Cosine cache buffer.
    pub cos_cache: &'a MetalBuffer,
    /// Sine cache buffer.
    pub sin_cache: &'a MetalBuffer,
    /// Number of attention heads.
    pub num_heads: u32,
    /// Per-head dimension.
    pub head_dim: u32,
    /// Sequence offset for position calculation.
    pub seq_offset: u32,
    /// Number of tokens in the batch.
    pub token_count: u32,
}

// ── Dispatch helpers ─────────────────────────────────────────────

/// Encode RoPE application.
pub fn encode_rope(encoder: &ComputeEncoder, pipeline: &ComputePipeline, params: &RopeParams<'_>) {
    debug_assert!(
        params.token_count == 0 || params.seq_offset.checked_add(params.token_count).is_some(),
        "rope: seq_offset + token_count overflows u32"
    );
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.qk, 0, 0);
    encoder.set_buffer(params.cos_cache, 0, 1);
    encoder.set_buffer(params.sin_cache, 0, 2);
    encoder.set_bytes(&params.num_heads.to_le_bytes(), 3);
    encoder.set_bytes(&params.head_dim.to_le_bytes(), 4);
    encoder.set_bytes(&params.seq_offset.to_le_bytes(), 5);
    encoder.set_bytes(&params.token_count.to_le_bytes(), 6);
    let half_dim = params.head_dim / 2;
    encoder.dispatch_threads(
        (
            half_dim as usize,
            params.num_heads as usize,
            params.token_count as usize,
        ),
        ((half_dim as usize).min(DEFAULT_THREADGROUP_WIDTH), 1, 1),
    );
}
