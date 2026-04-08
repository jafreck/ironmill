//! Attention pipeline states and dispatch helpers.

use ironmill_metal_sys::{ComputeEncoder, ComputePipeline, MetalBuffer};

use super::{ATTENTION_MIN_THREADGROUP_WIDTH, METAL_MAX_THREADS_PER_THREADGROUP};

/// Attention pipeline states.
pub struct AttentionPipelines {
    /// Standard FP16 attention (decode) kernel.
    pub standard_attention: ComputePipeline,
    /// Prefill attention kernel.
    pub prefill_attention: ComputePipeline,
    /// FlashAttention-2 prefill kernel.
    pub prefill_attention_fa2: ComputePipeline,
    /// Register-tiled FA2 prefill kernel (v2).
    pub prefill_attention_v2: ComputePipeline,
    /// Fused scaled dot-product attention kernel.
    /// `None` when the kernel exceeds threadgroup memory limits (head_dim ≥ 256).
    /// Decode uses FlashDecoding split+reduce instead.
    pub fused_sdpa: Option<ComputePipeline>,
    /// FlashDecoding split kernel: each threadgroup processes a KV slice.
    pub fused_sdpa_split: ComputePipeline,
    /// FlashDecoding reduce kernel: combines partial results across splits.
    pub fused_sdpa_reduce: ComputePipeline,
    /// TurboQuant KV cache write kernel.
    pub turboquant_cache_write: ComputePipeline,
    /// TurboQuant attention kernel.
    pub turboquant_attention: ComputePipeline,
    /// TurboQuant outlier KV cache write kernel.
    pub turboquant_outlier_cache_write: ComputePipeline,
    /// TurboQuant outlier attention kernel.
    pub turboquant_outlier_attention: ComputePipeline,
}

// ── Parameter structs ────────────────────────────────────────────

/// Parameters for [`encode_standard_attention`].
pub struct StandardAttentionParams<'a> {
    /// Query projection buffer.
    pub q: &'a MetalBuffer,
    /// K cache buffer.
    pub k_cache: &'a MetalBuffer,
    /// V cache buffer.
    pub v_cache: &'a MetalBuffer,
    /// Output buffer.
    pub output: &'a MetalBuffer,
    /// Number of query heads.
    pub num_heads: u32,
    /// Number of key-value heads.
    pub num_kv_heads: u32,
    /// Per-head dimension.
    pub head_dim: u32,
    /// Maximum sequence length (cache stride).
    pub max_seq_len: u32,
    /// Current sequence length.
    pub seq_len: u32,
}

/// Parameters for [`encode_prefill_attention`].
pub struct PrefillAttentionParams<'a> {
    /// Query projection buffer.
    pub q: &'a MetalBuffer,
    /// K cache buffer.
    pub k_cache: &'a MetalBuffer,
    /// V cache buffer.
    pub v_cache: &'a MetalBuffer,
    /// Output buffer.
    pub output: &'a MetalBuffer,
    /// Number of query heads.
    pub num_heads: u32,
    /// Number of key-value heads.
    pub num_kv_heads: u32,
    /// Per-head dimension.
    pub head_dim: u32,
    /// Maximum sequence length (cache stride).
    pub max_seq_len: u32,
    /// Sequence offset (position of first token in batch).
    pub seq_offset: u32,
    /// Number of tokens in the batch.
    pub token_count: u32,
    /// Sliding window size (0 = full attention).
    pub window_size: u32,
    /// Attention scale factor (1/sqrt(head_dim) or 1.0 for QK-normed models).
    pub attn_scale: f32,
}

/// Parameters for [`encode_fused_sdpa`].
pub struct FusedSdpaParams<'a> {
    /// Query buffer.
    pub q: &'a MetalBuffer,
    /// Key buffer.
    pub k: &'a MetalBuffer,
    /// Value buffer.
    pub v: &'a MetalBuffer,
    /// Output buffer.
    pub output: &'a MetalBuffer,
    /// Current sequence length.
    pub seq_len: u32,
    /// Number of tokens in the batch.
    pub token_count: u32,
    /// Per-head dimension.
    pub head_dim: u32,
    /// Number of query heads.
    pub num_q_heads: u32,
    /// Number of key-value heads.
    pub num_kv_heads: u32,
    /// Attention scale factor.
    pub scale: f32,
    /// Maximum sequence length (cache stride).
    pub max_seq_len: u32,
}

// ── Dispatch helpers ─────────────────────────────────────────────

/// Encode standard FP16 attention.
pub fn encode_standard_attention(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &StandardAttentionParams<'_>,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.q, 0, 0);
    encoder.set_buffer(params.k_cache, 0, 1);
    encoder.set_buffer(params.v_cache, 0, 2);
    encoder.set_buffer(params.output, 0, 3);
    encoder.set_bytes(&params.num_heads.to_le_bytes(), 4);
    encoder.set_bytes(&params.num_kv_heads.to_le_bytes(), 5);
    encoder.set_bytes(&params.head_dim.to_le_bytes(), 6);
    encoder.set_bytes(&params.max_seq_len.to_le_bytes(), 7);
    encoder.set_bytes(&params.seq_len.to_le_bytes(), 8);
    // HEAD_DIM is now exact — use at least ATTENTION_MIN_THREADGROUP_WIDTH
    // threads for better occupancy and parallel QK^T position processing.
    encoder.dispatch_threadgroups(
        (params.num_heads as usize, 1, 1),
        (
            ATTENTION_MIN_THREADGROUP_WIDTH
                .max(params.head_dim as usize)
                .min(METAL_MAX_THREADS_PER_THREADGROUP),
            1,
            1,
        ),
    );
}

/// Encode batched prefill attention — all query tokens in one dispatch.
pub fn encode_prefill_attention(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &PrefillAttentionParams<'_>,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.q, 0, 0);
    encoder.set_buffer(params.k_cache, 0, 1);
    encoder.set_buffer(params.v_cache, 0, 2);
    encoder.set_buffer(params.output, 0, 3);
    encoder.set_bytes(&params.num_heads.to_le_bytes(), 4);
    encoder.set_bytes(&params.num_kv_heads.to_le_bytes(), 5);
    encoder.set_bytes(&params.head_dim.to_le_bytes(), 6);
    encoder.set_bytes(&params.max_seq_len.to_le_bytes(), 7);
    encoder.set_bytes(&params.seq_offset.to_le_bytes(), 8);
    encoder.set_bytes(&params.token_count.to_le_bytes(), 9);
    encoder.set_bytes(&params.window_size.to_le_bytes(), 10);
    encoder.set_bytes(&params.attn_scale.to_le_bytes(), 11);
    encoder.dispatch_threadgroups(
        (params.num_heads as usize, params.token_count as usize, 1),
        (
            ATTENTION_MIN_THREADGROUP_WIDTH
                .max(params.head_dim as usize)
                .min(METAL_MAX_THREADS_PER_THREADGROUP),
            1,
            1,
        ),
    );
}

/// FA2 Q-chunk sizes matching the compile-time constants in attention.metal.
fn fa2_q_chunk(head_dim: u32) -> u32 {
    if head_dim >= 512 {
        4
    } else if head_dim >= 256 {
        8
    } else {
        32
    }
}

/// Encode FlashAttention-2 style prefill attention.
///
/// Groups `FA2_Q_CHUNK` queries per threadgroup, loading each KV tile once
/// and reusing it across all queries. Better bandwidth utilization for
/// large models where KV tiles don't fit in L1 cache.
pub fn encode_fa2_prefill_attention(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &PrefillAttentionParams<'_>,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.q, 0, 0);
    encoder.set_buffer(params.k_cache, 0, 1);
    encoder.set_buffer(params.v_cache, 0, 2);
    encoder.set_buffer(params.output, 0, 3);
    encoder.set_bytes(&params.num_heads.to_le_bytes(), 4);
    encoder.set_bytes(&params.num_kv_heads.to_le_bytes(), 5);
    encoder.set_bytes(&params.head_dim.to_le_bytes(), 6);
    encoder.set_bytes(&params.max_seq_len.to_le_bytes(), 7);
    encoder.set_bytes(&params.seq_offset.to_le_bytes(), 8);
    encoder.set_bytes(&params.token_count.to_le_bytes(), 9);
    encoder.set_bytes(&params.window_size.to_le_bytes(), 10);
    encoder.set_bytes(&params.attn_scale.to_le_bytes(), 11);
    let q_chunk = fa2_q_chunk(params.head_dim) as usize;
    let q_blocks = (params.token_count as usize).div_ceil(q_chunk);
    encoder.dispatch_threadgroups(
        (params.num_heads as usize, q_blocks, 1),
        (
            ATTENTION_MIN_THREADGROUP_WIDTH
                .max(params.head_dim as usize)
                .min(METAL_MAX_THREADS_PER_THREADGROUP),
            1,
            1,
        ),
    );
}

/// V2 query block size matching the compile-time constant in attention.metal.
fn v2_br(head_dim: u32) -> u32 {
    if head_dim >= 256 { 16 } else { 32 }
}

/// Encode register-tiled FA2 prefill attention (v2).
///
/// Combines cooperative KV tile loading (amortized across GQA group) with
/// register-based accumulation. Dispatches one threadgroup per
/// (kv_group, q_block), with one simdgroup per Q head in the group.
pub fn encode_v2_prefill_attention(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &PrefillAttentionParams<'_>,
) {
    assert!(params.num_kv_heads > 0, "num_kv_heads must be > 0");
    assert!(
        params.num_heads % params.num_kv_heads == 0,
        "num_heads must be divisible by num_kv_heads"
    );
    let heads_per_group = (params.num_heads / params.num_kv_heads) as usize;
    let num_kv_groups = params.num_kv_heads as usize;
    let br = v2_br(params.head_dim) as usize;
    let q_blocks = (params.token_count as usize).div_ceil(br);

    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.q, 0, 0);
    encoder.set_buffer(params.k_cache, 0, 1);
    encoder.set_buffer(params.v_cache, 0, 2);
    encoder.set_buffer(params.output, 0, 3);
    encoder.set_bytes(&params.num_heads.to_le_bytes(), 4);
    encoder.set_bytes(&params.num_kv_heads.to_le_bytes(), 5);
    encoder.set_bytes(&params.head_dim.to_le_bytes(), 6);
    encoder.set_bytes(&params.max_seq_len.to_le_bytes(), 7);
    encoder.set_bytes(&params.seq_offset.to_le_bytes(), 8);
    encoder.set_bytes(&params.token_count.to_le_bytes(), 9);
    encoder.set_bytes(&params.window_size.to_le_bytes(), 10);
    encoder.set_bytes(&params.attn_scale.to_le_bytes(), 11);
    let threads_per_tg = (heads_per_group * 32).min(METAL_MAX_THREADS_PER_THREADGROUP);
    encoder.dispatch_threadgroups((num_kv_groups, q_blocks, 1), (threads_per_tg, 1, 1));
}

/// Default Q-block tile size (Br) for fused SDPA.
const FUSED_SDPA_DEFAULT_BR: usize = 32;

/// Encode fused scaled dot-product attention.
///
/// Handles both prefill (token_count > 1) and decode (token_count == 1).
/// Dispatches one threadgroup per (kv_head_group, q_block), with one
/// simdgroup per Q head within each group.
pub fn encode_fused_sdpa(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &FusedSdpaParams<'_>,
    tile_br: Option<usize>,
) {
    let br = tile_br.unwrap_or(FUSED_SDPA_DEFAULT_BR);
    let heads_per_group = (params.num_q_heads / params.num_kv_heads) as usize;

    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.q, 0, 0);
    encoder.set_buffer(params.k, 0, 1);
    encoder.set_buffer(params.v, 0, 2);
    encoder.set_buffer(params.output, 0, 3);
    encoder.set_bytes(&params.seq_len.to_le_bytes(), 4);
    encoder.set_bytes(&params.token_count.to_le_bytes(), 5);
    encoder.set_bytes(&params.head_dim.to_le_bytes(), 6);
    encoder.set_bytes(&params.num_q_heads.to_le_bytes(), 7);
    encoder.set_bytes(&params.num_kv_heads.to_le_bytes(), 8);
    encoder.set_bytes(&params.scale.to_le_bytes(), 9);
    encoder.set_bytes(&params.max_seq_len.to_le_bytes(), 10);

    let num_kv_groups = params.num_kv_heads as usize;
    let q_blocks = (params.token_count as usize).div_ceil(br);
    // One simdgroup (32 threads) per Q head in the group.
    assert!(
        heads_per_group <= 32,
        "fused SDPA: GQA ratio {heads_per_group}:1 exceeds max 32 simdgroups per threadgroup"
    );
    let threads_per_tg = (heads_per_group * 32).min(METAL_MAX_THREADS_PER_THREADGROUP);

    encoder.dispatch_threadgroups((num_kv_groups, q_blocks, 1), (threads_per_tg, 1, 1));
}

/// Minimum sequence length to trigger FlashDecoding split.
/// Below this, the original single-pass kernel is used.
const FLASH_DECODE_MIN_SEQ: usize = 256;

/// KV positions per split — each split gets a chunk of the sequence.
/// 256 gives good granularity: at 16K context with 256 per split = 64 splits.
const FLASH_DECODE_KV_PER_SPLIT: usize = 256;

/// Encode FlashDecoding++: persistent split kernel with max hint, then reduce.
///
/// Improvements over basic FlashDecoding:
/// - Persistent threadgroups: launch fewer TGs that each loop over multiple
///   KV chunks, improving cache locality and reducing dispatch overhead.
/// - Unified max hint: passes the previous step's softmax max to the split
///   kernel so the O accumulator avoids rescaling when max hasn't changed.
/// - Reduce kernel writes back the global max for the next step's hint.
#[allow(clippy::too_many_arguments)]
pub fn encode_flash_decode(
    encoder: &ComputeEncoder,
    split_pipeline: &ComputePipeline,
    reduce_pipeline: &ComputePipeline,
    sdpa_fallback: Option<&ComputePipeline>,
    params: &FusedSdpaParams<'_>,
    partial_o: &MetalBuffer,
    partial_max: &MetalBuffer,
    partial_sum: &MetalBuffer,
    max_hint: &MetalBuffer,
    max_splits: usize,
    gpu_max_threadgroups: usize,
) {
    let seq_len = params.seq_len as usize;

    // Short sequences: use single-pass if available, otherwise still split.
    if seq_len < FLASH_DECODE_MIN_SEQ {
        if let Some(sdpa) = sdpa_fallback {
            encode_fused_sdpa(encoder, sdpa, params, None);
            return;
        }
        // No fallback — use split+reduce even for short sequences.
    }

    let heads_per_group = (params.num_q_heads as usize) / (params.num_kv_heads as usize).max(1);
    let num_kv_groups = params.num_kv_heads as usize;
    let target_splits = (gpu_max_threadgroups / num_kv_groups.max(1)).max(2);
    let splits_from_seq = seq_len.div_ceil(FLASH_DECODE_KV_PER_SPLIT);
    let num_splits = target_splits.min(splits_from_seq).min(max_splits).max(2);

    if num_splits <= 1 {
        if let Some(sdpa) = sdpa_fallback {
            encode_fused_sdpa(encoder, sdpa, params, None);
            return;
        }
    }

    // Persistent split kernel: grid Y = min(num_splits, target_persistent_tgs).
    let persistent_tgs = num_splits.min(target_splits);
    let threads_per_tg = (heads_per_group * 32).min(METAL_MAX_THREADS_PER_THREADGROUP);

    encoder.set_pipeline(split_pipeline);
    encoder.set_buffer(params.q, 0, 0);
    encoder.set_buffer(params.k, 0, 1);
    encoder.set_buffer(params.v, 0, 2);
    encoder.set_buffer(partial_o, 0, 3);
    encoder.set_buffer(partial_max, 0, 4);
    encoder.set_buffer(partial_sum, 0, 5);
    encoder.set_bytes(&params.seq_len.to_le_bytes(), 6);
    encoder.set_bytes(&params.head_dim.to_le_bytes(), 7);
    encoder.set_bytes(&params.num_q_heads.to_le_bytes(), 8);
    encoder.set_bytes(&params.num_kv_heads.to_le_bytes(), 9);
    encoder.set_bytes(&params.scale.to_le_bytes(), 10);
    encoder.set_bytes(&params.max_seq_len.to_le_bytes(), 11);
    encoder.set_bytes(&(num_splits as u32).to_le_bytes(), 12);
    // Pass -INFINITY as initial max_hint (safe default; per-head hint via
    // buffer would be more accurate but requires an extra buffer binding).
    let neg_inf = f32::NEG_INFINITY;
    encoder.set_bytes(&neg_inf.to_le_bytes(), 13);
    encoder.dispatch_threadgroups((num_kv_groups, persistent_tgs, 1), (threads_per_tg, 1, 1));

    // Barrier between split and reduce.
    encoder.memory_barrier_with_resources(&[partial_o, partial_max, partial_sum]);

    // Reduce kernel: grid (num_q_heads, 1, 1)
    let num_q = params.num_q_heads as usize;
    encoder.set_pipeline(reduce_pipeline);
    encoder.set_buffer(partial_o, 0, 0);
    encoder.set_buffer(partial_max, 0, 1);
    encoder.set_buffer(partial_sum, 0, 2);
    encoder.set_buffer(params.output, 0, 3);
    encoder.set_bytes(&params.num_q_heads.to_le_bytes(), 4);
    encoder.set_bytes(&params.head_dim.to_le_bytes(), 5);
    encoder.set_bytes(&(num_splits as u32).to_le_bytes(), 6);
    encoder.set_buffer(max_hint, 0, 7);
    encoder.dispatch_threadgroups((num_q, 1, 1), (32, 1, 1));
}

/// Dispatch the FlashDecoding reduce kernel — shared by FP16 and TQ paths.
///
/// Combines partial (unnormalized) attention outputs from split kernels,
/// applies softmax normalization, and writes the final half output.
#[allow(clippy::too_many_arguments)]
pub fn encode_flash_decode_reduce(
    encoder: &ComputeEncoder,
    reduce_pipeline: &ComputePipeline,
    partial_o: &MetalBuffer,
    partial_max: &MetalBuffer,
    partial_sum: &MetalBuffer,
    output: &MetalBuffer,
    max_hint: &MetalBuffer,
    num_q_heads: u32,
    head_dim: u32,
    num_splits: u32,
) {
    encoder.set_pipeline(reduce_pipeline);
    encoder.set_buffer(partial_o, 0, 0);
    encoder.set_buffer(partial_max, 0, 1);
    encoder.set_buffer(partial_sum, 0, 2);
    encoder.set_buffer(output, 0, 3);
    encoder.set_bytes(&num_q_heads.to_le_bytes(), 4);
    encoder.set_bytes(&head_dim.to_le_bytes(), 5);
    encoder.set_bytes(&num_splits.to_le_bytes(), 6);
    encoder.set_buffer(max_hint, 0, 7);
    encoder.dispatch_threadgroups((num_q_heads as usize, 1, 1), (32, 1, 1));
}
