#include <metal_stdlib>
using namespace metal;

// ============================================================================
// TurboQuant: Hadamard rotation + beta-optimal INT8 quantization for KV cache
// ============================================================================

// === Fused Cache Write ===
//
// Rotates K/V via Hadamard rotation matrix, quantizes to INT8, writes to cache.
//
// Buffers:
//   buffer(0) kv_proj:          [num_kv_heads × head_dim]                  half
//   buffer(1) rotation_matrix:  [head_dim × head_dim]                     half
//   buffer(2) cache:            [num_kv_heads × max_seq_len × head_dim]   char (INT8)
//   buffer(3) num_kv_heads:     uint
//   buffer(4) head_dim:         uint
//   buffer(5) max_seq_len:      uint
//   buffer(6) seq_pos:          uint  (current write position)
//   buffer(7) inv_scale:        float (1/scale for quantization)
//
// Dispatch: num_kv_heads threadgroups, head_dim threads per group.
// Each threadgroup processes one KV head:
//   1. Load input vector into shared memory
//   2. Matrix-vector multiply with rotation_matrix (each thread computes one output)
//   3. Quantize: round(clamp(rotated * inv_scale, -128, 127))
//   4. Write INT8 to cache[head][seq_pos][dim]

kernel void turboquant_cache_write(
    device const half* kv_proj          [[buffer(0)]],
    device const half* rotation_matrix  [[buffer(1)]],
    device char* cache                  [[buffer(2)]],
    constant uint& num_kv_heads         [[buffer(3)]],
    constant uint& head_dim             [[buffer(4)]],
    constant uint& max_seq_len          [[buffer(5)]],
    constant uint& seq_pos              [[buffer(6)]],
    constant float& inv_scale           [[buffer(7)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    // Shared memory for input vector (one head_dim row, max 4096 halfs = 8 KB)
    threadgroup half shared_input[4096];

    uint head_idx = tgid;
    if (head_idx >= num_kv_heads) return;

    // Step 1: Cooperatively load input vector for this head into shared memory
    uint input_base = head_idx * head_dim;
    for (uint i = tid; i < head_dim; i += tg_size) {
        shared_input[i] = kv_proj[input_base + i];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Each thread computes one element of rotated = rotation_matrix @ input
    // Thread tid computes output[tid] = dot(rotation_matrix[tid, :], input[:])
    for (uint out_dim = tid; out_dim < head_dim; out_dim += tg_size) {
        float acc = 0.0f;
        uint row_base = out_dim * head_dim;
        for (uint k = 0; k < head_dim; k++) {
            acc += float(rotation_matrix[row_base + k]) * float(shared_input[k]);
        }

        // Step 3: Quantize to INT8
        float scaled = acc * inv_scale;
        scaled = clamp(scaled, -128.0f, 127.0f);
        char quantized = char(rint(scaled));

        // Step 4: Write to cache at [head][seq_pos][dim]
        uint cache_idx = (head_idx * max_seq_len + seq_pos) * head_dim + out_dim;
        cache[cache_idx] = quantized;
    }
}


// === Fused TurboQuant Attention ===
//
// Performs full attention with INT8 KV cache using online softmax:
//   1. Rotate Q to match K's rotated space
//   2. For each position: dequantize K, compute score, update online softmax,
//      rescale running V accumulator, accumulate weighted dequantized V
//   3. Normalize V accumulator by softmax denominator
//   4. Un-rotate output back to original space
//
// Buffers:
//   buffer(0)  q:               [num_heads × head_dim]                     half
//   buffer(1)  k_cache:         [num_kv_heads × max_seq_len × head_dim]   char (INT8)
//   buffer(2)  v_cache:         [num_kv_heads × max_seq_len × head_dim]   char (INT8)
//   buffer(3)  rotation_matrix: [head_dim × head_dim]                     half
//   buffer(4)  output:          [num_heads × head_dim]                     half
//   buffer(5)  num_heads:       uint
//   buffer(6)  num_kv_heads:    uint
//   buffer(7)  head_dim:        uint
//   buffer(8)  max_seq_len:     uint
//   buffer(9)  seq_len:         uint  (number of valid positions in cache)
//   buffer(10) deq_scale:       float (dequantization scale factor)
//
// GQA: heads_per_group = num_heads / num_kv_heads
// Each attention head maps to kv_head = head_idx / heads_per_group
//
// Dispatch: num_heads threadgroups, head_dim threads per group.
// Single-pass online softmax: no need to store or recompute scores.

kernel void turboquant_attention(
    device const half* q                [[buffer(0)]],
    device const char* k_cache          [[buffer(1)]],
    device const char* v_cache          [[buffer(2)]],
    device const half* rotation_matrix  [[buffer(3)]],
    device half* output                 [[buffer(4)]],
    constant uint& num_heads            [[buffer(5)]],
    constant uint& num_kv_heads         [[buffer(6)]],
    constant uint& head_dim             [[buffer(7)]],
    constant uint& max_seq_len          [[buffer(8)]],
    constant uint& seq_len              [[buffer(9)]],
    constant float& deq_scale           [[buffer(10)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    // Shared memory layout (within 32 KB):
    //   shared_q_rot:  [256] float — rotated query vector
    //   shared_reduce: [256] float — scratch for dot-product reductions
    //   shared_output: [256] float — weighted V accumulator (in rotated space)
    //   shared_softmax: [2] float  — [0]=running_max, [1]=running_sum
    // For head_dim=256: (256*3 + 2) * 4 = 3080 bytes
    threadgroup float shared_q_rot[256];
    threadgroup float shared_reduce[256];
    threadgroup float shared_output[256];
    threadgroup float shared_softmax[2]; // [0]=max, [1]=sum

    uint head_idx = tgid;
    if (head_idx >= num_heads) return;

    uint heads_per_group = num_heads / num_kv_heads;
    uint kv_head = head_idx / heads_per_group;
    float scale = 1.0f / sqrt(float(head_dim));
    uint q_base = head_idx * head_dim;
    uint kv_base = kv_head * max_seq_len * head_dim;

    // ---- Step 1: Rotate Q into shared memory ----
    for (uint d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        uint row = d * head_dim;
        for (uint k = 0; k < head_dim; k++) {
            acc += float(rotation_matrix[row + k]) * float(q[q_base + k]);
        }
        shared_q_rot[d] = acc;
    }

    // Zero output accumulator
    for (uint d = tid; d < head_dim; d += tg_size) {
        shared_output[d] = 0.0f;
    }
    if (tid == 0) {
        shared_softmax[0] = -INFINITY; // running max
        shared_softmax[1] = 0.0f;      // running sum
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Step 2: Online softmax attention over sequence positions ----
    for (uint p = 0; p < seq_len; p++) {
        uint k_offset = kv_base + p * head_dim;

        // Compute dot(Q_rot, dequant(K[p])) — parallel reduction
        float partial_dot = 0.0f;
        for (uint d = tid; d < head_dim; d += tg_size) {
            float k_val = float(k_cache[k_offset + d]) * deq_scale;
            partial_dot += shared_q_rot[d] * k_val;
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

        // Online softmax: update max, compute rescale factor and new weight
        float old_max = shared_softmax[0];
        float new_max = max(old_max, score);
        float rescale = exp(old_max - new_max);
        float weight = exp(score - new_max);

        if (tid == 0) {
            shared_softmax[0] = new_max;
            shared_softmax[1] = shared_softmax[1] * rescale + weight;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rescale existing accumulator and add weighted dequantized V
        uint v_offset = kv_base + p * head_dim;
        for (uint d = tid; d < head_dim; d += tg_size) {
            float v_val = float(v_cache[v_offset + d]) * deq_scale;
            shared_output[d] = shared_output[d] * rescale + weight * v_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Step 3: Normalize by softmax denominator ----
    float denom = shared_softmax[1];
    for (uint d = tid; d < head_dim; d += tg_size) {
        shared_output[d] /= denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Step 4: Un-rotate output ----
    // For orthogonal rotation matrix R, R^{-1} = R^T.
    // final[d] = sum_k( R^T[d][k] * output_rot[k] ) = sum_k( R[k][d] * output_rot[k] )
    uint out_base = head_idx * head_dim;
    for (uint d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (uint k = 0; k < head_dim; k++) {
            acc += float(rotation_matrix[k * head_dim + d]) * shared_output[k];
        }
        output[out_base + d] = half(acc);
    }
}
