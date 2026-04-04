#include <metal_stdlib>
using namespace metal;

#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif

// Tiled flash attention — standard FP16 (no quantization, no rotation).
// Loads KV cache in tiles into threadgroup SRAM to reduce global memory
// bandwidth. Uses online softmax (per-tile) for numerical stability.
//
// Uses simd_sum for QK^T dot-product reduction (zero intra-simdgroup
// barriers) and processes multiple positions in parallel across
// simdgroups, drastically reducing threadgroup barrier count.
//
// Buffers:
//   buffer(0) q:         [num_heads × head_dim]                     half
//   buffer(1) k_cache:   [num_kv_heads × max_seq_len × head_dim]   half
//   buffer(2) v_cache:   [num_kv_heads × max_seq_len × head_dim]   half
//   buffer(3) output:    [num_heads × head_dim]                     half
//   buffer(4) num_heads:    uint
//   buffer(5) num_kv_heads: uint
//   buffer(6) head_dim:     uint
//   buffer(7) max_seq_len:  uint
//   buffer(8) seq_len:      uint  (number of valid positions in cache)
//
// GQA: kv_head = head_idx / (num_heads / num_kv_heads)
//
// Dispatch: num_heads threadgroups, max(256, head_dim) threads per group.

kernel void standard_attention(
    device const half* q         [[buffer(0)]],
    device const half* k_cache   [[buffer(1)]],
    device const half* v_cache   [[buffer(2)]],
    device half* output          [[buffer(3)]],
    constant uint& num_heads     [[buffer(4)]],
    constant uint& num_kv_heads  [[buffer(5)]],
    constant uint& head_dim      [[buffer(6)]],
    constant uint& max_seq_len   [[buffer(7)]],
    constant uint& seq_len       [[buffer(8)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint TILE = 32;
    constexpr uint SIMD_SIZE = 32;

    threadgroup float shared_q[HEAD_DIM];
    threadgroup half  kv_tile[TILE * HEAD_DIM];
    threadgroup float tile_scores[TILE];
    threadgroup float shared_output[HEAD_DIM];
    threadgroup float softmax_max[1];
    threadgroup float softmax_sum[1];
    threadgroup float tile_correction[1];

    uint num_simdgroups = tg_size / SIMD_SIZE;

    uint head_idx = tgid;
    if (head_idx >= num_heads) return;
    if (head_dim != HEAD_DIM) return;

    uint heads_per_group = num_heads / num_kv_heads;
    uint kv_head = (num_kv_heads < num_heads) ? head_idx / heads_per_group : head_idx;

    float scale = 1.0f / sqrt(float(head_dim));

    // Load Q vector into shared memory and zero accumulator
    uint q_base = head_idx * head_dim;
    for (uint d = tid; d < head_dim; d += tg_size) {
        shared_q[d] = float(q[q_base + d]);
        shared_output[d] = 0.0f;
    }
    if (tid == 0) {
        softmax_max[0] = -INFINITY;
        softmax_sum[0] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kv_base = kv_head * max_seq_len * head_dim;

    // ---- Tiled flash attention ----
    for (uint tile_start = 0; tile_start < seq_len; tile_start += TILE) {
        uint tile_end = min(tile_start + TILE, seq_len);
        uint actual_tile = tile_end - tile_start;

        // Cooperative load of K tile into threadgroup SRAM
        uint k_elems = actual_tile * head_dim;
        for (uint i = tid; i < k_elems; i += tg_size) {
            uint p = i / head_dim;
            uint d = i % head_dim;
            kv_tile[p * head_dim + d] =
                k_cache[kv_base + (tile_start + p) * head_dim + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // QK^T: each simdgroup computes one position's dot product.
        // simd_sum reduces within a simdgroup with zero barriers.
        for (uint pbase = 0; pbase < actual_tile; pbase += num_simdgroups) {
            uint p = pbase + sgid;
            if (p < actual_tile) {
                float dot = 0.0f;
                for (uint d = lane; d < head_dim; d += SIMD_SIZE) {
                    dot += shared_q[d] * float(kv_tile[p * head_dim + d]);
                }
                dot = simd_sum(dot);
                if (lane == 0) tile_scores[p] = dot * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Per-tile online softmax (parallel via simd ops) ----
        // TILE=32 matches SIMD_SIZE=32, so the first simdgroup can process
        // all tile positions in parallel — one lane per position.
        if (sgid == 0) {
            float my_score = (lane < actual_tile) ? tile_scores[lane] : -INFINITY;
            float tm = simd_max(my_score);

            float old_max = softmax_max[0];
            float new_max = max(old_max, tm);
            float corr = exp(old_max - new_max);

            if (lane == 0) {
                tile_correction[0] = corr;
                softmax_max[0] = new_max;
                softmax_sum[0] = softmax_sum[0] * corr;
            }

            float w = (lane < actual_tile) ? exp(my_score - new_max) : 0.0f;
            if (lane < actual_tile) tile_scores[lane] = w;
            float tile_sum = simd_sum(w);
            if (lane == 0) softmax_sum[0] += tile_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rescale accumulator by correction factor
        float corr = tile_correction[0];
        for (uint d = tid; d < head_dim; d += tg_size)
            shared_output[d] *= corr;

        // Load V tile (aliasing K tile memory)
        uint v_elems = actual_tile * head_dim;
        for (uint i = tid; i < v_elems; i += tg_size) {
            uint p = i / head_dim;
            uint d = i % head_dim;
            kv_tile[p * head_dim + d] =
                v_cache[kv_base + (tile_start + p) * head_dim + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate weighted V from tile
        for (uint p = 0; p < actual_tile; p++) {
            float w = tile_scores[p];
            for (uint d = tid; d < head_dim; d += tg_size)
                shared_output[d] += w * float(kv_tile[p * head_dim + d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize and write output
    float denom = max(softmax_sum[0], 1e-10f);
    uint out_base = head_idx * head_dim;
    for (uint d = tid; d < head_dim; d += tg_size) {
        output[out_base + d] = half(shared_output[d] / denom);
    }
}

// ============================================================================
// Prefill attention — batched across all query tokens in a single dispatch.
//
// Same tiled flash attention algorithm as standard_attention, but processes
// multiple query tokens via a 2D threadgroup grid. Each threadgroup handles
// one (head, query_token) pair with causal masking: query at position t
// attends only to KV cache positions 0..seq_offset+t+1.
//
// This replaces the per-token dispatch loop in the FP16 prefill path,
// eliminating ~token_count × num_layers kernel launch overhead.
//
// Buffers:
//   buffer(0) q:           [token_count × num_heads × head_dim]     half
//   buffer(1) k_cache:     [num_kv_heads × max_seq_len × head_dim]  half
//   buffer(2) v_cache:     [num_kv_heads × max_seq_len × head_dim]  half
//   buffer(3) output:      [token_count × num_heads × head_dim]     half
//   buffer(4) num_heads:    uint
//   buffer(5) num_kv_heads: uint
//   buffer(6) head_dim:     uint
//   buffer(7) max_seq_len:  uint
//   buffer(8) seq_offset:   uint  (KV positions before this prefill batch)
//   buffer(9) token_count:  uint
//
// Dispatch: (num_heads, token_count, 1) threadgroups,
//           (max(256, head_dim), 1, 1) threads per group.

kernel void prefill_attention(
    device const half* q         [[buffer(0)]],
    device const half* k_cache   [[buffer(1)]],
    device const half* v_cache   [[buffer(2)]],
    device half* output          [[buffer(3)]],
    constant uint& num_heads     [[buffer(4)]],
    constant uint& num_kv_heads  [[buffer(5)]],
    constant uint& head_dim      [[buffer(6)]],
    constant uint& max_seq_len   [[buffer(7)]],
    constant uint& seq_offset    [[buffer(8)]],
    constant uint& token_count   [[buffer(9)]],
    uint3 tid_3d  [[thread_position_in_threadgroup]],
    uint3 tgid_3d [[threadgroup_position_in_grid]],
    uint3 tg_size_3d [[threads_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint TILE = 32;
    constexpr uint SIMD_SIZE = 32;

    uint tid = tid_3d.x;
    uint tg_size = tg_size_3d.x;

    threadgroup float shared_q[HEAD_DIM];
    threadgroup half  kv_tile[TILE * HEAD_DIM];
    threadgroup float tile_scores[TILE];
    threadgroup float shared_output[HEAD_DIM];
    threadgroup float softmax_max[1];
    threadgroup float softmax_sum[1];
    threadgroup float tile_correction[1];

    uint num_simdgroups = tg_size / SIMD_SIZE;

    uint head_idx = tgid_3d.x;
    uint token_idx = tgid_3d.y;
    if (head_idx >= num_heads || token_idx >= token_count) return;
    if (head_dim != HEAD_DIM) return;

    // Causal mask: this query attends to positions 0..causal_len.
    uint causal_len = seq_offset + token_idx + 1;

    uint heads_per_group = num_heads / num_kv_heads;
    uint kv_head = (num_kv_heads < num_heads) ? head_idx / heads_per_group : head_idx;

    float scale = 1.0f / sqrt(float(head_dim));

    // Load Q vector for this (token, head) pair
    uint q_base = (token_idx * num_heads + head_idx) * head_dim;
    for (uint d = tid; d < head_dim; d += tg_size) {
        shared_q[d] = float(q[q_base + d]);
        shared_output[d] = 0.0f;
    }
    if (tid == 0) {
        softmax_max[0] = -INFINITY;
        softmax_sum[0] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kv_base = kv_head * max_seq_len * head_dim;

    // ---- Tiled flash attention with causal masking ----
    for (uint tile_start = 0; tile_start < causal_len; tile_start += TILE) {
        uint tile_end = min(tile_start + TILE, causal_len);
        uint actual_tile = tile_end - tile_start;

        uint k_elems = actual_tile * head_dim;
        for (uint i = tid; i < k_elems; i += tg_size) {
            uint p = i / head_dim;
            uint d = i % head_dim;
            kv_tile[p * head_dim + d] =
                k_cache[kv_base + (tile_start + p) * head_dim + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint pbase = 0; pbase < actual_tile; pbase += num_simdgroups) {
            uint p = pbase + sgid;
            if (p < actual_tile) {
                float dot = 0.0f;
                for (uint d = lane; d < head_dim; d += SIMD_SIZE) {
                    dot += shared_q[d] * float(kv_tile[p * head_dim + d]);
                }
                dot = simd_sum(dot);
                if (lane == 0) tile_scores[p] = dot * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgid == 0) {
            float my_score = (lane < actual_tile) ? tile_scores[lane] : -INFINITY;
            float tm = simd_max(my_score);

            float old_max = softmax_max[0];
            float new_max = max(old_max, tm);
            float corr = exp(old_max - new_max);

            if (lane == 0) {
                tile_correction[0] = corr;
                softmax_max[0] = new_max;
                softmax_sum[0] = softmax_sum[0] * corr;
            }

            float w = (lane < actual_tile) ? exp(my_score - new_max) : 0.0f;
            if (lane < actual_tile) tile_scores[lane] = w;
            float tile_sum = simd_sum(w);
            if (lane == 0) softmax_sum[0] += tile_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float corr = tile_correction[0];
        for (uint d = tid; d < head_dim; d += tg_size)
            shared_output[d] *= corr;

        uint v_elems = actual_tile * head_dim;
        for (uint i = tid; i < v_elems; i += tg_size) {
            uint p = i / head_dim;
            uint d = i % head_dim;
            kv_tile[p * head_dim + d] =
                v_cache[kv_base + (tile_start + p) * head_dim + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint p = 0; p < actual_tile; p++) {
            float w = tile_scores[p];
            for (uint d = tid; d < head_dim; d += tg_size)
                shared_output[d] += w * float(kv_tile[p * head_dim + d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize and write output
    float denom = max(softmax_sum[0], 1e-10f);
    uint out_base = (token_idx * num_heads + head_idx) * head_dim;
    for (uint d = tid; d < head_dim; d += tg_size) {
        output[out_base + d] = half(shared_output[d] / denom);
    }
}

// ============================================================================
// FlashAttention-2 style prefill — multi-query per threadgroup.
//
// Groups Q_CHUNK consecutive queries per threadgroup, loading each KV tile
// once and reusing it across all queries. Better KV bandwidth utilization
// for large models where tiles don't fit in L1 cache.
//
// For small models (0.6B), the per-query prefill_attention kernel above is
// faster because KV tiles fit in L1 and the multi-query bookkeeping adds
// overhead.
//
// Grid: (num_heads, ceil(token_count / Q_CHUNK), 1) threadgroups.
// Threadgroup: (max(256, HEAD_DIM), 1, 1) threads.

constant constexpr uint FA2_Q_CHUNK = 32;
constant constexpr uint FA2_KV_TILE = 32;

kernel void prefill_attention_fa2(
    device const half* q         [[buffer(0)]],
    device const half* k_cache   [[buffer(1)]],
    device const half* v_cache   [[buffer(2)]],
    device half* output          [[buffer(3)]],
    constant uint& num_heads     [[buffer(4)]],
    constant uint& num_kv_heads  [[buffer(5)]],
    constant uint& head_dim      [[buffer(6)]],
    constant uint& max_seq_len   [[buffer(7)]],
    constant uint& seq_offset    [[buffer(8)]],
    constant uint& token_count   [[buffer(9)]],
    uint3 tid_3d  [[thread_position_in_threadgroup]],
    uint3 tgid_3d [[threadgroup_position_in_grid]],
    uint3 tg_size_3d [[threads_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint SIMD_SIZE = 32;

    uint tid = tid_3d.x;
    uint tg_size = tg_size_3d.x;
    uint num_simdgroups = tg_size / SIMD_SIZE;

    uint head_idx = tgid_3d.x;
    uint q_chunk_idx = tgid_3d.y;
    if (head_idx >= num_heads) return;
    if (head_dim != HEAD_DIM) return;

    uint q_start = q_chunk_idx * FA2_Q_CHUNK;
    uint q_end = min(q_start + FA2_Q_CHUNK, token_count);
    uint actual_q = q_end - q_start;
    if (actual_q == 0) return;

    uint heads_per_group = num_heads / num_kv_heads;
    uint kv_head = (num_kv_heads < num_heads) ? head_idx / heads_per_group : head_idx;
    float scale = 1.0f / sqrt(float(HEAD_DIM));

    threadgroup half  kv_tile[FA2_KV_TILE * HEAD_DIM];
    threadgroup float out_acc[FA2_Q_CHUNK * HEAD_DIM];
    threadgroup float all_scores[FA2_Q_CHUNK * FA2_KV_TILE];
    threadgroup float q_max[FA2_Q_CHUNK];
    threadgroup float q_sum[FA2_Q_CHUNK];
    threadgroup float q_corr[FA2_Q_CHUNK];

    for (uint i = tid; i < actual_q * HEAD_DIM; i += tg_size)
        out_acc[i] = 0.0f;
    for (uint i = tid; i < actual_q; i += tg_size) {
        q_max[i] = -INFINITY;
        q_sum[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint max_causal = seq_offset + q_end;
    uint kv_base = kv_head * max_seq_len * HEAD_DIM;

    for (uint kv_start = 0; kv_start < max_causal; kv_start += FA2_KV_TILE) {
        uint kv_end_pos = min(kv_start + FA2_KV_TILE, max_causal);
        uint actual_kv = kv_end_pos - kv_start;

        // Load K tile
        uint k_elems = actual_kv * HEAD_DIM;
        for (uint i = tid; i < k_elems; i += tg_size) {
            uint p = i / HEAD_DIM;
            uint d = i % HEAD_DIM;
            kv_tile[p * HEAD_DIM + d] =
                k_cache[kv_base + (kv_start + p) * HEAD_DIM + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute QK^T for all queries × KV positions
        for (uint qb = 0; qb < actual_q; qb += num_simdgroups) {
            uint qi = qb + sgid;
            if (qi < actual_q) {
                uint q_global = q_start + qi;
                uint causal_len = seq_offset + q_global + 1;
                uint q_base_addr = (q_global * num_heads + head_idx) * HEAD_DIM;

                for (uint p = 0; p < actual_kv; p++) {
                    uint kv_pos = kv_start + p;
                    if (kv_pos >= causal_len) {
                        if (lane == 0) all_scores[qi * FA2_KV_TILE + p] = -INFINITY;
                    } else {
                        float dot = 0.0f;
                        for (uint d = lane; d < HEAD_DIM; d += SIMD_SIZE)
                            dot += float(q[q_base_addr + d]) * float(kv_tile[p * HEAD_DIM + d]);
                        dot = simd_sum(dot);
                        if (lane == 0) all_scores[qi * FA2_KV_TILE + p] = dot * scale;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax — one thread per query
        if (tid < actual_q) {
            uint qi = tid;
            float old_max = q_max[qi];
            float tile_max = -INFINITY;
            for (uint p = 0; p < actual_kv; p++)
                tile_max = max(tile_max, all_scores[qi * FA2_KV_TILE + p]);
            float new_max = max(old_max, tile_max);
            float corr = exp(old_max - new_max);
            q_max[qi] = new_max;
            q_sum[qi] *= corr;
            q_corr[qi] = corr;
            for (uint p = 0; p < actual_kv; p++) {
                float s = all_scores[qi * FA2_KV_TILE + p];
                float w = (s > -INFINITY + 1.0f) ? exp(s - new_max) : 0.0f;
                all_scores[qi * FA2_KV_TILE + p] = w;
                q_sum[qi] += w;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rescale output accumulators
        for (uint i = tid; i < actual_q * HEAD_DIM; i += tg_size) {
            uint qi = i / HEAD_DIM;
            out_acc[i] *= q_corr[qi];
        }

        // Load V tile
        uint v_elems = actual_kv * HEAD_DIM;
        for (uint i = tid; i < v_elems; i += tg_size) {
            uint p = i / HEAD_DIM;
            uint d = i % HEAD_DIM;
            kv_tile[p * HEAD_DIM + d] =
                v_cache[kv_base + (kv_start + p) * HEAD_DIM + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate weighted V
        for (uint qi = 0; qi < actual_q; qi++) {
            for (uint p = 0; p < actual_kv; p++) {
                float w = all_scores[qi * FA2_KV_TILE + p];
                if (w > 0.0f) {
                    for (uint d = tid; d < HEAD_DIM; d += tg_size)
                        out_acc[qi * HEAD_DIM + d] += w * float(kv_tile[p * HEAD_DIM + d]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint qi = 0; qi < actual_q; qi++) {
        float denom = max(q_sum[qi], 1e-10f);
        uint out_base = ((q_start + qi) * num_heads + head_idx) * HEAD_DIM;
        for (uint d = tid; d < HEAD_DIM; d += tg_size)
            output[out_base + d] = half(out_acc[qi * HEAD_DIM + d] / denom);
    }
}
