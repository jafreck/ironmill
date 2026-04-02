#include <metal_stdlib>
using namespace metal;

// Tiled flash attention — standard FP16 (no quantization, no rotation).
// Loads KV cache in tiles into threadgroup SRAM to reduce global memory
// bandwidth. Uses online softmax (per-tile) for numerical stability.
//
// Buffers (unchanged):
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
// Dispatch: num_heads threadgroups, min(head_dim, 1024) threads per group.

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
    uint tg_size [[threads_per_threadgroup]])
{
    // Shared memory budget (head_dim=512 worst case):
    //   shared_q:      512 × 4       =  2,048 B
    //   kv_tile:       32 × 512 × 2  = 32,768 B  (half; aliased K then V)
    //   shared_reduce: 512 × 4       =  2,048 B
    //   tile_scores:   32 × 4        =    128 B
    //   shared_output: 512 × 4       =  2,048 B
    //   softmax/corr:  3 × 4         =     12 B
    //   Total: ~39.1 KB  (NOTE: exceeds 32 KB for head_dim=512;
    //    half kv_tile dominates. Actual usage depends on runtime head_dim.)
    constexpr uint TILE = 32;
    constexpr uint MAX_DIM = 512;

    threadgroup float shared_q[MAX_DIM];
    threadgroup half  kv_tile[TILE * MAX_DIM];
    threadgroup float shared_reduce[MAX_DIM];
    threadgroup float tile_scores[TILE];
    threadgroup float shared_output[MAX_DIM];
    threadgroup float softmax_max[1];
    threadgroup float softmax_sum[1];
    threadgroup float tile_correction[1];

    uint head_idx = tgid;
    if (head_idx >= num_heads) return;

    uint heads_per_group = num_heads / num_kv_heads;
    uint kv_head = head_idx / heads_per_group;

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

        // Compute QK^T for every position in the tile
        for (uint p = 0; p < actual_tile; p++) {
            float partial_dot = 0.0f;
            for (uint d = tid; d < head_dim; d += tg_size) {
                partial_dot += shared_q[d] * float(kv_tile[p * head_dim + d]);
            }

            // Tree reduction
            shared_reduce[tid] = partial_dot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint rs = 1;
            while (rs < tg_size) rs <<= 1;
            for (uint s = rs / 2; s > 0; s >>= 1) {
                if (tid < s && (tid + s) < tg_size)
                    shared_reduce[tid] += shared_reduce[tid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) tile_scores[p] = shared_reduce[0] * scale;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

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
