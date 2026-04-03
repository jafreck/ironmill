#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif

#include <metal_stdlib>
using namespace metal;

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

        // ---- Per-tile online softmax ----
        if (tid == 0) {
            float tm = -INFINITY;
            for (uint p = 0; p < actual_tile; p++)
                tm = max(tm, tile_scores[p]);

            float old_max = softmax_max[0];
            float new_max = max(old_max, tm);
            float corr = exp(old_max - new_max);

            tile_correction[0] = corr;
            softmax_max[0] = new_max;
            softmax_sum[0] = softmax_sum[0] * corr;

            for (uint p = 0; p < actual_tile; p++) {
                float w = exp(tile_scores[p] - new_max);
                tile_scores[p] = w;
                softmax_sum[0] += w;
            }
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
