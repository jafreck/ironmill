#include <metal_stdlib>
using namespace metal;

// Standard FP16 attention (no quantization, no rotation).
// For use when TurboQuant is disabled.
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
// Standard scaled dot-product attention with GQA support.
// GQA: kv_head = head_idx / (num_heads / num_kv_heads)
//
// Dispatch: num_heads threadgroups, head_dim threads per group.
// Uses online softmax for numerical stability.

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
    // Shared memory (within 32 KB budget):
    //   q_local:    [head_dim] float    — query vector for this head
    //   reduction:  [head_dim] float    — scratch for parallel reductions
    //   output_acc: [head_dim] float    — weighted V accumulator
    // For head_dim=256: 256 * 3 * 4 = 3072 bytes
    threadgroup float shared_q[256];
    threadgroup float shared_reduce[256];
    threadgroup float shared_output[256];

    uint head_idx = tgid;
    if (head_idx >= num_heads) return;

    uint heads_per_group = num_heads / num_kv_heads;
    uint kv_head = head_idx / heads_per_group;

    float scale = 1.0f / sqrt(float(head_dim));

    // Load Q vector for this head into shared memory
    uint q_base = head_idx * head_dim;
    for (uint d = tid; d < head_dim; d += tg_size) {
        shared_q[d] = float(q[q_base + d]);
    }

    // Zero output accumulator
    for (uint d = tid; d < head_dim; d += tg_size) {
        shared_output[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kv_base = kv_head * max_seq_len * head_dim;

    // ---- Online softmax + weighted V accumulation ----
    // We maintain running max and denominator to avoid a separate softmax pass.
    // When a new max is found, we rescale the existing accumulator.

    // Thread 0 tracks softmax state; all threads participate in dot products
    // and V accumulation.
    threadgroup float softmax_max[1];
    threadgroup float softmax_sum[1];

    if (tid == 0) {
        softmax_max[0] = -INFINITY;
        softmax_sum[0] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint p = 0; p < seq_len; p++) {
        uint k_offset = kv_base + p * head_dim;

        // Compute dot(Q, K[p]) — parallel reduction
        float partial_dot = 0.0f;
        for (uint d = tid; d < head_dim; d += tg_size) {
            partial_dot += shared_q[d] * float(k_cache[k_offset + d]);
        }
        shared_reduce[tid] = partial_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Round tg_size up to next power of 2 for reduction
        uint reduce_size = 1;
        while (reduce_size < tg_size) reduce_size <<= 1;

        for (uint stride = reduce_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride && (tid + stride) < tg_size) {
                shared_reduce[tid] += shared_reduce[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float score = shared_reduce[0] * scale;

        // Online softmax update (thread 0 computes new max/sum)
        float old_max = softmax_max[0];
        float new_max = max(old_max, score);
        float rescale = exp(old_max - new_max);
        float weight = exp(score - new_max);

        if (tid == 0) {
            softmax_sum[0] = softmax_sum[0] * rescale + weight;
            softmax_max[0] = new_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rescale existing accumulator and add weighted V
        uint v_offset = kv_base + p * head_dim;
        for (uint d = tid; d < head_dim; d += tg_size) {
            shared_output[d] = shared_output[d] * rescale + weight * float(v_cache[v_offset + d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize by softmax denominator and write output
    float denom = softmax_sum[0];
    uint out_base = head_idx * head_dim;
    for (uint d = tid; d < head_dim; d += tg_size) {
        output[out_base + d] = half(shared_output[d] / denom);
    }
}
