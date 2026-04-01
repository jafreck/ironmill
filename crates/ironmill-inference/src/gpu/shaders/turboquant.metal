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

/// Read one dequantized cache value (INT4 with codebook from buffer).
inline float read_quantized(device const char* cache, uint base,
                            uint pos, uint dim, uint head_dim,
                            uint n_bits, float deq_scale,
                            device const float* codebook) {
    if (n_bits == 4) {
        uint packed_stride = head_dim / 2;
        uint byte_idx = base + pos * packed_stride + dim / 2;
        uchar packed = ((device const uchar*)cache)[byte_idx];
        uchar nibble = (dim % 2 == 0) ? (packed & 0xF) : (packed >> 4);
        return codebook[nibble] * deq_scale;
    } else {
        return float(cache[base + pos * head_dim + dim]) * deq_scale;
    }
}

// ── In-place Walsh-Hadamard butterfly transform ──────────────────
//
// Applies the randomized Hadamard rotation R = (1/√d)·D·H·D in-place
// on a threadgroup-shared buffer. O(d log d) compute, O(d) storage for
// the sign vector (vs O(d²) for the dense matrix approach).
//
// `shared_data` must have at least `head_dim` elements.
// `rotation_signs` holds d floats (±1.0).

inline void hadamard_rotate_inplace(
    threadgroup float* shared_data,
    device const float* rotation_signs,
    uint head_dim,
    uint tid,
    uint tg_size)
{
    // Step 1: Apply diagonal sign matrix D
    for (uint d = tid; d < head_dim; d += tg_size) {
        shared_data[d] *= rotation_signs[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: In-place butterfly (unnormalized Walsh-Hadamard)
    for (uint half = 1; half < head_dim; half *= 2) {
        for (uint i = tid; i < head_dim / 2; i += tg_size) {
            uint block = i / half;
            uint offset = i % half;
            uint j = block * (half * 2) + offset;
            float a = shared_data[j];
            float b = shared_data[j + half];
            shared_data[j] = a + b;
            shared_data[j + half] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Step 3: Apply D again (R = D·H·D is symmetric and self-inverse)
    float scale = 1.0f / sqrt(float(head_dim));
    for (uint d = tid; d < head_dim; d += tg_size) {
        shared_data[d] *= rotation_signs[d] * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// === Fused Cache Write ===
//
// Rotates K/V via in-place Walsh-Hadamard butterfly, quantizes, writes to cache.
// Supports both INT8 (n_bits=8) and INT4 (n_bits=4) quantization.
// For K cache (is_k_cache=1), also computes QJL residual correction data.
//
// Buffers:
//   buffer(0)  kv_proj:          [num_kv_heads × head_dim]           half
//   buffer(1)  rotation_signs:   [head_dim]                          float (±1.0)
//   buffer(2)  cache:            see below                           char/packed
//   buffer(3)  num_kv_heads:     uint
//   buffer(4)  head_dim:         uint
//   buffer(5)  max_seq_len:      uint
//   buffer(6)  seq_pos:          uint  (current write position)
//   buffer(7)  inv_scale:        float (UNUSED — kept for API compat)
//   buffer(8)  n_bits:           uint  (4 or 8)
//   buffer(9)  scale_buf:        [num_kv_heads × max_seq_len] float
//   buffer(10) codebook:         [n_levels] float (INT4 codebook centroids)
//   buffer(11) boundaries:       [n_levels-1] float (INT4 boundary values)
//   buffer(12) n_levels:         uint (number of codebook levels, e.g. 16 for 4-bit)
//   buffer(13) qjl_matrix:       [head_dim × head_dim] float (QJL projection)
//   buffer(14) qjl_signs_buf:    [num_kv_heads × max_seq_len × head_dim/8] uchar
//   buffer(15) r_norms_buf:      [num_kv_heads × max_seq_len] float
//   buffer(16) is_k_cache:       uint (1 = K cache, 0 = V cache)
//
// Cache layout:
//   INT8: [num_kv_heads × max_seq_len × head_dim]       1 byte per element
//   INT4: [num_kv_heads × max_seq_len × head_dim/2]     2 elements packed per byte
//
// Dispatch: num_kv_heads threadgroups, head_dim threads per group.

kernel void turboquant_cache_write(
    device const half* kv_proj          [[buffer(0)]],
    device const float* rotation_signs  [[buffer(1)]],
    device char* cache                  [[buffer(2)]],
    constant uint& num_kv_heads         [[buffer(3)]],
    constant uint& head_dim             [[buffer(4)]],
    constant uint& max_seq_len          [[buffer(5)]],
    constant uint& seq_pos              [[buffer(6)]],
    constant float& inv_scale           [[buffer(7)]],  // UNUSED — kept for API compat
    constant uint& n_bits               [[buffer(8)]],
    device float* scale_buf             [[buffer(9)]],
    device const float* codebook        [[buffer(10)]],
    device const float* boundaries      [[buffer(11)]],
    constant uint& n_levels             [[buffer(12)]],
    device const float* qjl_matrix      [[buffer(13)]],
    device uchar* qjl_signs_buf         [[buffer(14)]],
    device float* r_norms_buf           [[buffer(15)]],
    constant uint& is_k_cache           [[buffer(16)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float shared_rotated[4096];
    threadgroup float shared_reduce[256];
    threadgroup char shared_quant[4096];

    uint head_idx = tgid;
    if (head_idx >= num_kv_heads) return;

    // Step 1: Load input into shared memory as float
    uint input_base = head_idx * head_dim;
    for (uint i = tid; i < head_dim; i += tg_size) {
        shared_rotated[i] = float(kv_proj[input_base + i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: In-place Hadamard rotation
    hadamard_rotate_inplace(shared_rotated, rotation_signs, head_dim, tid, tg_size);

    // Step 3: Compute per-thread absmax (needed for INT8)
    float local_max = 0.0f;
    for (uint out_dim = tid; out_dim < head_dim; out_dim += tg_size) {
        local_max = max(local_max, fabs(shared_rotated[out_dim]));
    }

    shared_reduce[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint reduce_size = 1;
    while (reduce_size < tg_size) reduce_size <<= 1;
    for (uint stride = reduce_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (tid + stride) < tg_size)
            shared_reduce[tid] = max(shared_reduce[tid], shared_reduce[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float max_val = max(shared_reduce[0], 1e-10f);

    if (n_bits == 4) {
        // TurboQuant MSE (arXiv:2504.19874 Algorithm 1):
        float local_sq = 0.0f;
        for (uint d = tid; d < head_dim; d += tg_size)
            local_sq += shared_rotated[d] * shared_rotated[d];
        shared_reduce[tid] = local_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) shared_reduce[tid] += shared_reduce[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float l2_norm = sqrt(max(shared_reduce[0], 1e-20f));

        if (tid == 0)
            scale_buf[head_idx * max_seq_len + seq_pos] = l2_norm;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Normalize and find nearest codebook centroid per coordinate
        float inv_norm = 1.0f / max(l2_norm, 1e-10f);
        uint n_boundaries = n_levels - 1;
        for (uint out_dim = tid; out_dim < head_dim; out_dim += tg_size) {
            float normalized = shared_rotated[out_dim] * inv_norm;
            uint idx = 0;
            for (uint b = 0; b < n_boundaries; b++) {
                if (normalized >= boundaries[b]) idx = b + 1;
            }
            shared_quant[out_dim] = char(idx);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Pack INT4 pairs into cache (must happen before QJL overwrites shared_quant)
        uint packed_stride = head_dim / 2;
        uint cache_base = kv_cache_base(head_idx, max_seq_len, head_dim, 4)
                        + seq_pos * packed_stride;
        for (uint d = tid * 2; d < head_dim; d += tg_size * 2) {
            uchar lo = uchar(shared_quant[d]     & 0xF);
            uchar hi = uchar(shared_quant[d + 1] & 0xF);
            ((device uchar*)cache)[cache_base + d / 2] = lo | (hi << 4);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── QJL residual correction (K cache only, INT4 only) ──
        // Paper §3.2: project quantization residual, store sign bits + norm.
        if (is_k_cache == 1) {
            // Compute residual = rotated - l2_norm * codebook[idx]
            // shared_rotated still has original rotated values (never overwritten)
            for (uint d = tid; d < head_dim; d += tg_size) {
                uchar q_idx = uchar(shared_quant[d]) & 0xF;
                shared_rotated[d] -= codebook[q_idx] * l2_norm;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Compute ||residual||_2
            float res_sq = 0.0f;
            for (uint d = tid; d < head_dim; d += tg_size)
                res_sq += shared_rotated[d] * shared_rotated[d];
            shared_reduce[tid] = res_sq;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = tg_size / 2; s > 0; s >>= 1) {
                if (tid < s) shared_reduce[tid] += shared_reduce[tid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float r_norm = sqrt(max(shared_reduce[0], 1e-30f));

            if (tid == 0)
                r_norms_buf[head_idx * max_seq_len + seq_pos] = r_norm;

            // QJL projection: sign(S · residual), packed into bits
            for (uint out_d = tid; out_d < head_dim; out_d += tg_size) {
                float proj = 0.0f;
                uint row_base = out_d * head_dim;
                for (uint k = 0; k < head_dim; k++) {
                    proj += qjl_matrix[row_base + k] * shared_rotated[k];
                }
                shared_quant[out_d] = (proj >= 0.0f) ? char(1) : char(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Pack 8 sign bits per byte
            uint signs_per_pos = head_dim / 8;
            uint signs_base = (head_idx * max_seq_len + seq_pos) * signs_per_pos;
            for (uint byte_idx = tid; byte_idx < signs_per_pos; byte_idx += tg_size) {
                uchar packed = 0;
                uint base_dim = byte_idx * 8;
                for (uint b = 0; b < 8; b++) {
                    if (shared_quant[base_dim + b] == 1)
                        packed |= (1u << b);
                }
                qjl_signs_buf[signs_base + byte_idx] = packed;
            }
        }
    } else {
        // INT8: per-head absmax (unchanged)
        float local_max = 0.0f;
        for (uint d = tid; d < head_dim; d += tg_size)
            local_max = max(local_max, fabs(shared_rotated[d]));
        shared_reduce[tid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint reduce_size = 1;
        while (reduce_size < tg_size) reduce_size <<= 1;
        for (uint stride = reduce_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride && (tid + stride) < tg_size)
                shared_reduce[tid] = max(shared_reduce[tid], shared_reduce[tid + stride]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float max_val = max(shared_reduce[0], 1e-10f);
        float dyn_inv_scale = 127.0f / max_val;
        float dyn_deq_scale = max_val / 127.0f;
        if (tid == 0)
            scale_buf[head_idx * max_seq_len + seq_pos] = dyn_deq_scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d = tid; d < head_dim; d += tg_size) {
            float scaled = clamp(shared_rotated[d] * dyn_inv_scale, -128.0f, 127.0f);
            cache[kv_cache_base(head_idx, max_seq_len, head_dim, 8)
                + seq_pos * head_dim + d] = char(rint(scaled));
        }
    }
}


// === Fused TurboQuant Attention ===
//
// Full attention with quantized KV cache using online softmax.
// For INT4, includes QJL residual correction on K cache inner products.
//
// Buffers:
//   buffer(0)  q:               [num_heads × head_dim]              half
//   buffer(1)  k_cache:         quantized KV cache                  char/packed
//   buffer(2)  v_cache:         quantized KV cache                  char/packed
//   buffer(3)  rotation_signs:  [head_dim]                          float (±1.0)
//   buffer(4)  output:          [num_heads × head_dim]              half
//   buffer(5)  num_heads:       uint
//   buffer(6)  num_kv_heads:    uint
//   buffer(7)  head_dim:        uint
//   buffer(8)  max_seq_len:     uint
//   buffer(9)  seq_len:         uint
//   buffer(10) deq_scale:       float (UNUSED — API compat)
//   buffer(11) n_bits:          uint (4 or 8)
//   buffer(12) k_scale_buf:     [num_kv_heads × max_seq_len] float
//   buffer(13) v_scale_buf:     [num_kv_heads × max_seq_len] float
//   buffer(14) codebook:        [n_levels] float
//   buffer(15) qjl_matrix:      [head_dim × head_dim] float
//   buffer(16) k_qjl_signs:     [num_kv_heads × max_seq_len × head_dim/8] uchar
//   buffer(17) k_r_norms:       [num_kv_heads × max_seq_len] float
//
// Dispatch: num_heads threadgroups, head_dim threads per group.

kernel void turboquant_attention(
    device const half* q                [[buffer(0)]],
    device const char* k_cache          [[buffer(1)]],
    device const char* v_cache          [[buffer(2)]],
    device const float* rotation_signs  [[buffer(3)]],
    device half* output                 [[buffer(4)]],
    constant uint& num_heads            [[buffer(5)]],
    constant uint& num_kv_heads         [[buffer(6)]],
    constant uint& head_dim             [[buffer(7)]],
    constant uint& max_seq_len          [[buffer(8)]],
    constant uint& seq_len              [[buffer(9)]],
    constant float& deq_scale           [[buffer(10)]],
    constant uint& n_bits               [[buffer(11)]],
    device const float* k_scale_buf     [[buffer(12)]],
    device const float* v_scale_buf     [[buffer(13)]],
    device const float* codebook        [[buffer(14)]],
    device const float* qjl_matrix      [[buffer(15)]],
    device const uchar* k_qjl_signs     [[buffer(16)]],
    device const float* k_r_norms       [[buffer(17)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float shared_q_rot[256];
    threadgroup float shared_s_q[256];     // S · q_rot for QJL correction
    threadgroup float shared_reduce[256];
    threadgroup float shared_output[256];
    threadgroup float shared_softmax[2];   // [0]=max, [1]=sum

    uint head_idx = tgid;
    if (head_idx >= num_heads) return;

    uint heads_per_group = num_heads / num_kv_heads;
    uint kv_head = head_idx / heads_per_group;
    float scale = 1.0f / sqrt(float(head_dim));
    uint q_base = head_idx * head_dim;
    uint kv_base = kv_cache_base(kv_head, max_seq_len, head_dim, n_bits);

    // ---- Step 1: Load Q and rotate via butterfly ----
    for (uint d = tid; d < head_dim; d += tg_size) {
        shared_q_rot[d] = float(q[q_base + d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hadamard_rotate_inplace(shared_q_rot, rotation_signs, head_dim, tid, tg_size);

    // ---- Step 1b (INT4 only): Precompute S · q_rot for QJL correction ----
    if (n_bits == 4) {
        for (uint out_d = tid; out_d < head_dim; out_d += tg_size) {
            float proj = 0.0f;
            uint row_base = out_d * head_dim;
            for (uint k = 0; k < head_dim; k++) {
                proj += qjl_matrix[row_base + k] * shared_q_rot[k];
            }
            shared_s_q[out_d] = proj;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Zero output accumulator
    for (uint d = tid; d < head_dim; d += tg_size) {
        shared_output[d] = 0.0f;
    }
    if (tid == 0) {
        shared_softmax[0] = -INFINITY;
        shared_softmax[1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // QJL correction constants
    // √(π/2) / d — from the QJL unbiased estimator (arXiv:2504.19874 §3.2)
    float qjl_coeff = (n_bits == 4) ? (sqrt(M_PI_F / 2.0f) / float(head_dim)) : 0.0f;
    uint signs_per_pos = head_dim / 8;

    // ---- Step 2: Online softmax attention ----
    for (uint p = 0; p < seq_len; p++) {
        float k_deq = k_scale_buf[kv_head * max_seq_len + p];

        // Stage 1: <q_rot, codebook[indices]> * l2_norm
        float partial_dot = 0.0f;
        for (uint d = tid; d < head_dim; d += tg_size) {
            float k_val = read_quantized(k_cache, kv_base, p, d, head_dim, n_bits, k_deq, codebook);
            partial_dot += shared_q_rot[d] * k_val;
        }

        // Stage 2 (INT4 only): QJL correction = qjl_coeff * r_norm * l2_norm * <S·q, signs>
        if (n_bits == 4) {
            float r_norm = k_r_norms[kv_head * max_seq_len + p];
            uint sign_base = (kv_head * max_seq_len + p) * signs_per_pos;

            float sign_dot = 0.0f;
            for (uint d = tid; d < head_dim; d += tg_size) {
                uint byte_idx = d / 8;
                uint bit_idx = d % 8;
                uchar packed = k_qjl_signs[sign_base + byte_idx];
                float sign_val = ((packed >> bit_idx) & 1) ? 1.0f : -1.0f;
                sign_dot += shared_s_q[d] * sign_val;
            }
            partial_dot += qjl_coeff * r_norm * k_deq * sign_dot;
        }

        shared_reduce[tid] = partial_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Tree reduction
        uint reduce_size = 1;
        while (reduce_size < tg_size) reduce_size <<= 1;
        for (uint stride = reduce_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride && (tid + stride) < tg_size)
                shared_reduce[tid] += shared_reduce[tid + stride];
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
        float v_deq = v_scale_buf[kv_head * max_seq_len + p];
        for (uint d = tid; d < head_dim; d += tg_size) {
            float v_val = read_quantized(v_cache, kv_base, p, d, head_dim, n_bits, v_deq, codebook);
            shared_output[d] = shared_output[d] * rescale + weight * v_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Step 3: Normalize ----
    float denom = shared_softmax[1];
    for (uint d = tid; d < head_dim; d += tg_size) {
        shared_output[d] /= denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Step 4: Un-rotate via butterfly (self-inverse) ----
    hadamard_rotate_inplace(shared_output, rotation_signs, head_dim, tid, tg_size);

    uint out_base = head_idx * head_dim;
    for (uint d = tid; d < head_dim; d += tg_size) {
        output[out_base + d] = half(shared_output[d]);
    }
}


// ============================================================================
// Outlier Channel Strategy (Section 4.3)
//
// Splits KV dimensions into outlier and non-outlier channel groups.
// Each group is independently rotated (Hadamard), quantized (Lloyd-Max),
// and stored in separate cache regions.
// ============================================================================

// Helper: quantize a sub-vector in shared memory using TurboQuant MSE.
// Writes L2 norm to scale_buf, packed INT4 indices to cache.
// `shared_vec` holds the rotated sub-vector of length `dim`.
// `shared_scratch` is scratch space of length >= dim.
// Returns via `shared_quant` starting at `quant_offset`.
inline void turboquant_quantize_group(
    threadgroup float* shared_vec,
    threadgroup float* shared_reduce,
    threadgroup char* shared_quant,
    device const float* codebook,
    device const float* boundaries,
    uint n_levels,
    device char* cache,
    device float* scale_buf,
    uint dim,
    uint head_idx,
    uint max_seq_len,
    uint seq_pos,
    uint bytes_per_pos,
    uint tid,
    uint tg_size)
{
    // L2 norm
    float local_sq = 0.0f;
    for (uint d = tid; d < dim; d += tg_size)
        local_sq += shared_vec[d] * shared_vec[d];
    shared_reduce[tid] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared_reduce[tid] += shared_reduce[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float l2_norm = sqrt(max(shared_reduce[0], 1e-20f));

    if (tid == 0)
        scale_buf[head_idx * max_seq_len + seq_pos] = l2_norm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Quantize
    float inv_norm = 1.0f / max(l2_norm, 1e-10f);
    uint n_boundaries = n_levels - 1;
    for (uint d = tid; d < dim; d += tg_size) {
        float normalized = shared_vec[d] * inv_norm;
        uint idx = 0;
        for (uint b = 0; b < n_boundaries; b++) {
            if (normalized >= boundaries[b]) idx = b + 1;
        }
        shared_quant[d] = char(idx);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pack INT4
    uint cache_base = head_idx * max_seq_len * bytes_per_pos + seq_pos * bytes_per_pos;
    for (uint d = tid * 2; d < dim; d += tg_size * 2) {
        uchar lo = uchar(shared_quant[d]     & 0xF);
        uchar hi = (d + 1 < dim) ? uchar(shared_quant[d + 1] & 0xF) : 0;
        ((device uchar*)cache)[cache_base + d / 2] = lo | (hi << 4);
    }
}


// === Outlier Cache Write ===
//
// Splits KV vector into outlier and non-outlier groups, applies independent
// TurboQuant to each, stores results in separate cache buffers.
//
// Buffers:
//   buffer(0)  kv_proj:                  [num_kv_heads × head_dim] half
//   buffer(1)  channel_indices:          [head_dim] uint (outlier first, then non-outlier)
//   buffer(2)  outlier_cache:            packed 4-bit
//   buffer(3)  non_outlier_cache:        packed 4-bit
//   buffer(4)  outlier_rotation_signs:   [d_outlier_padded] float
//   buffer(5)  non_outlier_rotation_signs: [d_non_padded] float
//   buffer(6)  outlier_codebook:         [outlier_n_levels] float
//   buffer(7)  outlier_boundaries:       [outlier_n_levels-1] float
//   buffer(8)  non_outlier_codebook:     [non_outlier_n_levels] float
//   buffer(9)  non_outlier_boundaries:   [non_outlier_n_levels-1] float
//   buffer(10) outlier_scale_buf:        [num_kv_heads × max_seq_len] float
//   buffer(11) non_outlier_scale_buf:    [num_kv_heads × max_seq_len] float
//   buffer(12) num_kv_heads:             uint
//   buffer(13) head_dim:                 uint
//   buffer(14) max_seq_len:              uint
//   buffer(15) seq_pos:                  uint
//   buffer(16) n_outlier:                uint
//   buffer(17) d_outlier_padded:         uint
//   buffer(18) d_non_padded:             uint
//   buffer(19) outlier_n_levels:         uint
//   buffer(20) non_outlier_n_levels:     uint
//
// Dispatch: num_kv_heads threadgroups, max(d_outlier_padded, d_non_padded) threads.

kernel void turboquant_outlier_cache_write(
    device const half* kv_proj                      [[buffer(0)]],
    device const uint* channel_indices              [[buffer(1)]],
    device char* outlier_cache                      [[buffer(2)]],
    device char* non_outlier_cache                  [[buffer(3)]],
    device const float* outlier_rotation_signs      [[buffer(4)]],
    device const float* non_outlier_rotation_signs  [[buffer(5)]],
    device const float* outlier_codebook            [[buffer(6)]],
    device const float* outlier_boundaries          [[buffer(7)]],
    device const float* non_outlier_codebook        [[buffer(8)]],
    device const float* non_outlier_boundaries      [[buffer(9)]],
    device float* outlier_scale_buf                 [[buffer(10)]],
    device float* non_outlier_scale_buf             [[buffer(11)]],
    constant uint& num_kv_heads                     [[buffer(12)]],
    constant uint& head_dim                         [[buffer(13)]],
    constant uint& max_seq_len                      [[buffer(14)]],
    constant uint& seq_pos                          [[buffer(15)]],
    constant uint& n_outlier                        [[buffer(16)]],
    constant uint& d_outlier_padded                 [[buffer(17)]],
    constant uint& d_non_padded                     [[buffer(18)]],
    constant uint& outlier_n_levels                 [[buffer(19)]],
    constant uint& non_outlier_n_levels             [[buffer(20)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float shared_outlier[2048];
    threadgroup float shared_non_outlier[2048];
    threadgroup float shared_reduce[256];
    threadgroup char shared_quant[2048];

    uint head_idx = tgid;
    if (head_idx >= num_kv_heads) return;

    uint n_non = head_dim - n_outlier;
    uint input_base = head_idx * head_dim;

    // Extract outlier channels (zero-pad to d_outlier_padded)
    for (uint i = tid; i < d_outlier_padded; i += tg_size) {
        if (i < n_outlier) {
            uint src_idx = channel_indices[i];
            shared_outlier[i] = float(kv_proj[input_base + src_idx]);
        } else {
            shared_outlier[i] = 0.0f;
        }
    }

    // Extract non-outlier channels (zero-pad to d_non_padded)
    for (uint i = tid; i < d_non_padded; i += tg_size) {
        if (i < n_non) {
            uint src_idx = channel_indices[n_outlier + i];
            shared_non_outlier[i] = float(kv_proj[input_base + src_idx]);
        } else {
            shared_non_outlier[i] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Rotate outlier group
    hadamard_rotate_inplace(shared_outlier, outlier_rotation_signs, d_outlier_padded, tid, tg_size);

    // Quantize outlier group
    turboquant_quantize_group(
        shared_outlier, shared_reduce, shared_quant,
        outlier_codebook, outlier_boundaries, outlier_n_levels,
        outlier_cache, outlier_scale_buf,
        d_outlier_padded, head_idx, max_seq_len, seq_pos,
        d_outlier_padded / 2,
        tid, tg_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Rotate non-outlier group
    hadamard_rotate_inplace(shared_non_outlier, non_outlier_rotation_signs, d_non_padded, tid, tg_size);

    // Quantize non-outlier group
    turboquant_quantize_group(
        shared_non_outlier, shared_reduce, shared_quant,
        non_outlier_codebook, non_outlier_boundaries, non_outlier_n_levels,
        non_outlier_cache, non_outlier_scale_buf,
        d_non_padded, head_idx, max_seq_len, seq_pos,
        d_non_padded / 2,
        tid, tg_size);
}


// === Outlier Attention ===
//
// Attention with dual-group quantized KV cache. Each group is independently
// dequantized and un-rotated, then combined for attention scoring.
//
// Buffers mirror the cache write kernel, plus Q and output buffers.
//
// Dispatch: num_heads threadgroups, max(d_outlier_padded, d_non_padded) threads.

kernel void turboquant_outlier_attention(
    device const half* q                            [[buffer(0)]],
    device const char* k_outlier_cache              [[buffer(1)]],
    device const char* v_outlier_cache              [[buffer(2)]],
    device const char* k_non_outlier_cache          [[buffer(3)]],
    device const char* v_non_outlier_cache          [[buffer(4)]],
    device const uint* channel_indices              [[buffer(5)]],
    device const float* outlier_rotation_signs      [[buffer(6)]],
    device const float* non_outlier_rotation_signs  [[buffer(7)]],
    device const float* outlier_codebook            [[buffer(8)]],
    device const float* non_outlier_codebook        [[buffer(9)]],
    device half* output                             [[buffer(10)]],
    device const float* k_outlier_scales            [[buffer(11)]],
    device const float* v_outlier_scales            [[buffer(12)]],
    device const float* k_non_outlier_scales        [[buffer(13)]],
    device const float* v_non_outlier_scales        [[buffer(14)]],
    constant uint& num_heads                        [[buffer(15)]],
    constant uint& num_kv_heads                     [[buffer(16)]],
    constant uint& head_dim                         [[buffer(17)]],
    constant uint& max_seq_len                      [[buffer(18)]],
    constant uint& seq_len                          [[buffer(19)]],
    constant uint& n_outlier                        [[buffer(20)]],
    constant uint& d_outlier_padded                 [[buffer(21)]],
    constant uint& d_non_padded                     [[buffer(22)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float shared_q_outlier[2048];
    threadgroup float shared_q_non_outlier[2048];
    threadgroup float shared_reduce[256];
    threadgroup float shared_output_outlier[2048];
    threadgroup float shared_output_non_outlier[2048];
    threadgroup float shared_softmax[2];

    uint head_idx = tgid;
    if (head_idx >= num_heads) return;

    uint heads_per_group = num_heads / num_kv_heads;
    uint kv_head = head_idx / heads_per_group;
    float scale = 1.0f / sqrt(float(head_dim));
    uint q_base = head_idx * head_dim;
    uint n_non = head_dim - n_outlier;

    // ---- Load and rotate Q for both groups ----
    // Outlier Q channels
    for (uint i = tid; i < d_outlier_padded; i += tg_size) {
        if (i < n_outlier) {
            shared_q_outlier[i] = float(q[q_base + channel_indices[i]]);
        } else {
            shared_q_outlier[i] = 0.0f;
        }
    }
    for (uint i = tid; i < d_non_padded; i += tg_size) {
        if (i < n_non) {
            shared_q_non_outlier[i] = float(q[q_base + channel_indices[n_outlier + i]]);
        } else {
            shared_q_non_outlier[i] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    hadamard_rotate_inplace(shared_q_outlier, outlier_rotation_signs, d_outlier_padded, tid, tg_size);
    hadamard_rotate_inplace(shared_q_non_outlier, non_outlier_rotation_signs, d_non_padded, tid, tg_size);

    // Zero output accumulators (in rotated space, un-rotated at the end)
    for (uint d = tid; d < d_outlier_padded; d += tg_size)
        shared_output_outlier[d] = 0.0f;
    for (uint d = tid; d < d_non_padded; d += tg_size)
        shared_output_non_outlier[d] = 0.0f;
    if (tid == 0) {
        shared_softmax[0] = -INFINITY;
        shared_softmax[1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint o_bytes_per_pos = d_outlier_padded / 2;
    uint n_bytes_per_pos = d_non_padded / 2;

    // ---- Online softmax attention ----
    for (uint p = 0; p < seq_len; p++) {
        // Compute Q · K dot product across both groups
        float k_o_deq = k_outlier_scales[kv_head * max_seq_len + p];
        float k_n_deq = k_non_outlier_scales[kv_head * max_seq_len + p];

        float partial_dot = 0.0f;
        // Outlier K contribution
        uint o_base = kv_head * max_seq_len * o_bytes_per_pos;
        for (uint d = tid; d < d_outlier_padded; d += tg_size) {
            float k_val = read_quantized(k_outlier_cache, o_base, p, d, d_outlier_padded, 4, k_o_deq, outlier_codebook);
            partial_dot += shared_q_outlier[d] * k_val;
        }
        // Non-outlier K contribution
        uint n_base = kv_head * max_seq_len * n_bytes_per_pos;
        for (uint d = tid; d < d_non_padded; d += tg_size) {
            float k_val = read_quantized(k_non_outlier_cache, n_base, p, d, d_non_padded, 4, k_n_deq, non_outlier_codebook);
            partial_dot += shared_q_non_outlier[d] * k_val;
        }

        shared_reduce[tid] = partial_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint reduce_size = 1;
        while (reduce_size < tg_size) reduce_size <<= 1;
        for (uint stride = reduce_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride && (tid + stride) < tg_size)
                shared_reduce[tid] += shared_reduce[tid + stride];
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

        // Accumulate weighted V from both groups
        float v_o_deq = v_outlier_scales[kv_head * max_seq_len + p];
        for (uint d = tid; d < d_outlier_padded; d += tg_size) {
            float v_val = read_quantized(v_outlier_cache, o_base, p, d, d_outlier_padded, 4, v_o_deq, outlier_codebook);
            shared_output_outlier[d] = shared_output_outlier[d] * rescale + weight * v_val;
        }
        float v_n_deq = v_non_outlier_scales[kv_head * max_seq_len + p];
        for (uint d = tid; d < d_non_padded; d += tg_size) {
            float v_val = read_quantized(v_non_outlier_cache, n_base, p, d, d_non_padded, 4, v_n_deq, non_outlier_codebook);
            shared_output_non_outlier[d] = shared_output_non_outlier[d] * rescale + weight * v_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Normalize ----
    float denom = shared_softmax[1];
    for (uint d = tid; d < d_outlier_padded; d += tg_size)
        shared_output_outlier[d] /= denom;
    for (uint d = tid; d < d_non_padded; d += tg_size)
        shared_output_non_outlier[d] /= denom;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Un-rotate both groups ----
    hadamard_rotate_inplace(shared_output_outlier, outlier_rotation_signs, d_outlier_padded, tid, tg_size);
    hadamard_rotate_inplace(shared_output_non_outlier, non_outlier_rotation_signs, d_non_padded, tid, tg_size);

    // ---- Scatter back to original channel positions ----
    uint out_base = head_idx * head_dim;
    for (uint i = tid; i < n_outlier; i += tg_size) {
        uint dst_idx = channel_indices[i];
        output[out_base + dst_idx] = half(shared_output_outlier[i]);
    }
    for (uint i = tid; i < n_non; i += tg_size) {
        uint dst_idx = channel_indices[n_outlier + i];
        output[out_base + dst_idx] = half(shared_output_non_outlier[i]);
    }
}
