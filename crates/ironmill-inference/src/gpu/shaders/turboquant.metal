#include <metal_stdlib>
using namespace metal;

// ============================================================================
// TurboQuant: Hadamard rotation + quantized KV cache (INT8 or INT4)
// ============================================================================

// ── Helpers ──────────────────────────────────────────────────────

/// Compute the byte offset for a KV head's cache region.
inline uint kv_cache_base(uint kv_head, uint max_seq_len, uint head_dim, uint n_bits) {
    uint bytes_per_pos = (n_bits == 4) ? (head_dim / 2) : head_dim;
    return kv_head * max_seq_len * bytes_per_pos;
}

/// Read one dequantized cache value.
inline float read_quantized(device const char* cache, uint base,
                            uint pos, uint dim, uint head_dim,
                            uint n_bits, float deq_scale) {
    if (n_bits == 4) {
        uint packed_stride = head_dim / 2;
        uint byte_idx = base + pos * packed_stride + dim / 2;
        uchar packed = ((device const uchar*)cache)[byte_idx];
        uchar nibble = (dim % 2 == 0) ? (packed & 0xF) : (packed >> 4);
        // Sign-extend from 4 bits
        int val = int(nibble);
        if (val >= 8) val -= 16;
        return float(val) * deq_scale;
    } else {
        return float(cache[base + pos * head_dim + dim]) * deq_scale;
    }
}

// === Fused Cache Write ===
//
// Rotates K/V via Hadamard rotation matrix, quantizes, writes to cache.
// Supports both INT8 (n_bits=8) and INT4 (n_bits=4) quantization.
//
// Buffers:
//   buffer(0) kv_proj:          [num_kv_heads × head_dim]           half
//   buffer(1) rotation_matrix:  [head_dim × head_dim]               half
//   buffer(2) cache:            see below                           char/packed
//   buffer(3) num_kv_heads:     uint
//   buffer(4) head_dim:         uint
//   buffer(5) max_seq_len:      uint
//   buffer(6) seq_pos:          uint  (current write position)
//   buffer(7) inv_scale:        float (1/scale for quantization)
//   buffer(8) n_bits:           uint  (4 or 8)
//
// Cache layout:
//   INT8: [num_kv_heads × max_seq_len × head_dim]       1 byte per element
//   INT4: [num_kv_heads × max_seq_len × head_dim/2]     2 elements packed per byte
//
// Dispatch: num_kv_heads threadgroups, head_dim threads per group.

kernel void turboquant_cache_write(
    device const half* kv_proj          [[buffer(0)]],
    device const half* rotation_matrix  [[buffer(1)]],
    device char* cache                  [[buffer(2)]],
    constant uint& num_kv_heads         [[buffer(3)]],
    constant uint& head_dim             [[buffer(4)]],
    constant uint& max_seq_len          [[buffer(5)]],
    constant uint& seq_pos              [[buffer(6)]],
    constant float& inv_scale           [[buffer(7)]],
    constant uint& n_bits               [[buffer(8)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup half shared_input[4096];
    threadgroup char shared_quant[4096];

    uint head_idx = tgid;
    if (head_idx >= num_kv_heads) return;

    // Step 1: Load input vector into shared memory
    uint input_base = head_idx * head_dim;
    for (uint i = tid; i < head_dim; i += tg_size) {
        shared_input[i] = kv_proj[input_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Rotate and quantize
    float clamp_lo = (n_bits == 4) ? -8.0f : -128.0f;
    float clamp_hi = (n_bits == 4) ?  7.0f :  127.0f;

    for (uint out_dim = tid; out_dim < head_dim; out_dim += tg_size) {
        float acc = 0.0f;
        uint row_base = out_dim * head_dim;
        for (uint k = 0; k < head_dim; k++) {
            acc += float(rotation_matrix[row_base + k]) * float(shared_input[k]);
        }

        float scaled = clamp(acc * inv_scale, clamp_lo, clamp_hi);
        char quantized = char(rint(scaled));

        if (n_bits == 4) {
            shared_quant[out_dim] = quantized;
        } else {
            uint cache_idx = kv_cache_base(head_idx, max_seq_len, head_dim, 8)
                           + seq_pos * head_dim + out_dim;
            cache[cache_idx] = quantized;
        }
    }

    // Step 3 (INT4 only): Pack pairs of 4-bit values into bytes
    if (n_bits == 4) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint packed_stride = head_dim / 2;
        uint base = kv_cache_base(head_idx, max_seq_len, head_dim, 4)
                  + seq_pos * packed_stride;

        for (uint d = tid * 2; d < head_dim; d += tg_size * 2) {
            uchar lo = uchar(shared_quant[d]     & 0xF);
            uchar hi = uchar(shared_quant[d + 1] & 0xF);
            ((device uchar*)cache)[base + d / 2] = lo | (hi << 4);
        }
    }
}


// === Fused TurboQuant Attention ===
//
// Full attention with quantized KV cache using online softmax.
// Supports both INT8 (n_bits=8) and INT4 (n_bits=4).
//
// Buffers:
//   buffer(0)  q:               [num_heads × head_dim]              half
//   buffer(1)  k_cache:         quantized KV cache                  char/packed
//   buffer(2)  v_cache:         quantized KV cache                  char/packed
//   buffer(3)  rotation_matrix: [head_dim × head_dim]               half
//   buffer(4)  output:          [num_heads × head_dim]              half
//   buffer(5)  num_heads:       uint
//   buffer(6)  num_kv_heads:    uint
//   buffer(7)  head_dim:        uint
//   buffer(8)  max_seq_len:     uint
//   buffer(9)  seq_len:         uint  (number of valid positions)
//   buffer(10) deq_scale:       float
//   buffer(11) n_bits:          uint  (4 or 8)
//
// Dispatch: num_heads threadgroups, head_dim threads per group.

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
    constant uint& n_bits               [[buffer(11)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
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
    uint kv_base = kv_cache_base(kv_head, max_seq_len, head_dim, n_bits);

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
        // Compute dot(Q_rot, dequant(K[p])) — parallel reduction
        float partial_dot = 0.0f;
        for (uint d = tid; d < head_dim; d += tg_size) {
            float k_val = read_quantized(k_cache, kv_base, p, d, head_dim, n_bits, deq_scale);
            partial_dot += shared_q_rot[d] * k_val;
        }
        shared_reduce[tid] = partial_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Tree reduction
        uint reduce_size = 1;
        while (reduce_size < tg_size) reduce_size <<= 1;

        for (uint stride = reduce_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride && (tid + stride) < tg_size) {
                shared_reduce[tid] += shared_reduce[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float score = shared_reduce[0] * scale;

        // Online softmax update
        float old_max = shared_softmax[0];
        float new_max = max(old_max, score);
        float rescale = exp(old_max - new_max);
        float weight = exp(score - new_max);

        if (tid == 0) {
            shared_softmax[0] = new_max;
            shared_softmax[1] = shared_softmax[1] * rescale + weight;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rescale accumulator and add weighted dequantized V
        for (uint d = tid; d < head_dim; d += tg_size) {
            float v_val = read_quantized(v_cache, kv_base, p, d, head_dim, n_bits, deq_scale);
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
    uint out_base = head_idx * head_dim;
    for (uint d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (uint k = 0; k < head_dim; k++) {
            acc += float(rotation_matrix[k * head_dim + d]) * shared_output[k];
        }
        output[out_base + d] = half(acc);
    }
}
