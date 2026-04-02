//! Metal kernel source strings for TurboQuant and PolarQuant on MLX.
//!
//! Each kernel is ported from the corresponding `.metal` file in
//! `gpu/shaders/`, with function signatures adapted from explicit
//! `[[buffer(N)]]` bindings to MLX's ordered input/output array convention
//! (inputs first, outputs after).

// ── TurboQuant helpers (shared across kernels) ──────────────────

/// Common helper functions included by all TurboQuant kernels.
#[allow(dead_code)]
const TURBOQUANT_HELPERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline uint kv_cache_base(uint kv_head, uint max_seq_len, uint head_dim, uint n_bits) {
    uint bytes_per_pos = (n_bits == 4) ? (head_dim / 2) : head_dim;
    return kv_head * max_seq_len * bytes_per_pos;
}

inline float read_quantized_tile(threadgroup const char* tile,
                                 uint pos, uint dim, uint head_dim,
                                 uint n_bits, float deq_scale,
                                 device const float* codebook) {
    if (n_bits == 4) {
        uint packed_stride = head_dim / 2;
        uint byte_idx = pos * packed_stride + dim / 2;
        uchar packed = ((threadgroup const uchar*)tile)[byte_idx];
        uchar nibble = (dim % 2 == 0) ? (packed & 0xF) : (packed >> 4);
        return codebook[nibble] * deq_scale;
    } else {
        return float(tile[pos * head_dim + dim]) * deq_scale;
    }
}

inline void hadamard_rotate_inplace(
    threadgroup float* shared_data,
    device const float* rotation_signs,
    uint head_dim,
    uint tid,
    uint tg_size)
{
    for (uint d = tid; d < head_dim; d += tg_size) {
        shared_data[d] *= rotation_signs[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = 1; step < head_dim; step *= 2) {
        for (uint i = tid; i < head_dim / 2; i += tg_size) {
            uint block = i / step;
            uint offset = i % step;
            uint j = block * (step * 2) + offset;
            float a = shared_data[j];
            float b = shared_data[j + step];
            shared_data[j] = a + b;
            shared_data[j + step] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float scale = 1.0f / sqrt(float(head_dim));
    for (uint d = tid; d < head_dim; d += tg_size) {
        shared_data[d] *= rotation_signs[d] * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

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

    uint cache_base = head_idx * max_seq_len * bytes_per_pos + seq_pos * bytes_per_pos;
    for (uint d = tid * 2; d < dim; d += tg_size * 2) {
        uchar lo = uchar(shared_quant[d]     & 0xF);
        uchar hi = (d + 1 < dim) ? uchar(shared_quant[d + 1] & 0xF) : 0;
        ((device uchar*)cache)[cache_base + d / 2] = lo | (hi << 4);
    }
}
"#;

// ── 1. turboquant_cache_write ───────────────────────────────────

/// Cache write kernel: rotates K/V via Hadamard butterfly, quantizes,
/// writes to cache. INT4 or INT8.
///
/// MLX buffer order (inputs 0..16, output at index 17):
///   0: kv_proj [num_kv_heads × head_dim] half
///   1: rotation_signs [head_dim] float
///   2: cache (read/write — passed as input)
///   3: scale_buf (read/write — passed as input)
///   4: codebook [n_levels] float
///   5: boundaries [n_levels-1] float
///   6: qjl_matrix [head_dim × head_dim] float
///   7: qjl_signs_buf (read/write)
///   8: r_norms_buf (read/write)
///   9: params [7] uint32 — packed {num_kv_heads, head_dim, max_seq_len, seq_pos, n_bits, n_levels, is_k_cache}
///  Output 0: dummy [1] float (kernel writes to cache in-place)
pub const TURBOQUANT_CACHE_WRITE: &str = concat!(
    r#"
#include <metal_stdlib>
using namespace metal;

inline uint kv_cache_base_cw(uint kv_head, uint max_seq_len, uint head_dim, uint n_bits) {
    uint bytes_per_pos = (n_bits == 4) ? (head_dim / 2) : head_dim;
    return kv_head * max_seq_len * bytes_per_pos;
}

inline void hadamard_rotate_inplace_cw(
    threadgroup float* shared_data,
    device const float* rotation_signs,
    uint head_dim, uint tid, uint tg_size)
{
    for (uint d = tid; d < head_dim; d += tg_size)
        shared_data[d] *= rotation_signs[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 1; step < head_dim; step *= 2) {
        for (uint i = tid; i < head_dim / 2; i += tg_size) {
            uint block = i / step;
            uint offset = i % step;
            uint j = block * (step * 2) + offset;
            float a = shared_data[j];
            float b = shared_data[j + step];
            shared_data[j] = a + b;
            shared_data[j + step] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = 1.0f / sqrt(float(head_dim));
    for (uint d = tid; d < head_dim; d += tg_size)
        shared_data[d] *= rotation_signs[d] * scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

[[kernel]] void turboquant_cache_write(
    device const half* kv_proj          [[buffer(0)]],
    device const float* rotation_signs  [[buffer(1)]],
    device char* cache                  [[buffer(2)]],
    device float* scale_buf             [[buffer(3)]],
    device const float* codebook        [[buffer(4)]],
    device const float* boundaries      [[buffer(5)]],
    device const float* qjl_matrix      [[buffer(6)]],
    device uchar* qjl_signs_buf         [[buffer(7)]],
    device float* r_norms_buf           [[buffer(8)]],
    device const uint* params           [[buffer(9)]],
    device float* dummy_out             [[buffer(10)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint num_kv_heads = params[0];
    uint head_dim     = params[1];
    uint max_seq_len  = params[2];
    uint seq_pos      = params[3];
    uint n_bits       = params[4];
    uint n_levels     = params[5];
    uint is_k_cache   = params[6];

    threadgroup float shared_rotated[4096];
    threadgroup float shared_reduce[256];
    threadgroup char shared_quant[4096];

    uint head_idx = tgid;
    if (head_idx >= num_kv_heads) return;

    // Load input into shared memory
    uint input_base = head_idx * head_dim;
    for (uint i = tid; i < head_dim; i += tg_size)
        shared_rotated[i] = float(kv_proj[input_base + i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hadamard rotation
    hadamard_rotate_inplace_cw(shared_rotated, rotation_signs, head_dim, tid, tg_size);

    if (n_bits == 4) {
        // L2 norm for TurboQuant
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

        // Quantize via nearest codebook centroid
        float inv_norm = 1.0f / max(l2_norm, 1e-10f);
        uint n_boundaries = n_levels - 1;
        for (uint d = tid; d < head_dim; d += tg_size) {
            float normalized = shared_rotated[d] * inv_norm;
            uint idx = 0;
            for (uint b = 0; b < n_boundaries; b++) {
                if (normalized >= boundaries[b]) idx = b + 1;
            }
            shared_quant[d] = char(idx);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Pack INT4 nibbles
        uint packed_stride = head_dim / 2;
        uint cache_base = kv_cache_base_cw(head_idx, max_seq_len, head_dim, 4)
                        + seq_pos * packed_stride;
        for (uint d = tid * 2; d < head_dim; d += tg_size * 2) {
            uchar lo = uchar(shared_quant[d]     & 0xF);
            uchar hi = uchar(shared_quant[d + 1] & 0xF);
            ((device uchar*)cache)[cache_base + d / 2] = lo | (hi << 4);
        }
    } else {
        // INT8: per-head absmax quantization
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
            cache[kv_cache_base_cw(head_idx, max_seq_len, head_dim, 8)
                + seq_pos * head_dim + d] = char(rint(scaled));
        }
    }

    // Write a dummy value so MLX sees an output dependency
    if (tgid == 0 && tid == 0) dummy_out[0] = 1.0f;
}
"#
);

// ── 2. turboquant_attention ─────────────────────────────────────

/// Tiled flash attention with quantized KV cache and online softmax.
///
/// MLX buffer order:
///   0: q [num_heads × head_dim] half
///   1: k_cache (quantized)
///   2: v_cache (quantized)
///   3: rotation_signs [head_dim] float
///   4: k_scale_buf [num_kv_heads × max_seq_len] float
///   5: v_scale_buf [num_kv_heads × max_seq_len] float
///   6: k_codebook [n_levels] float
///   7: v_codebook [n_levels] float
///   8: qjl_matrix [head_dim × head_dim] float
///   9: k_r_norms [num_kv_heads × max_seq_len] float
///  10: params [5] uint32 — {num_heads, num_kv_heads, head_dim, max_seq_len, seq_len, n_bits}
///  Output 0: output [num_heads × head_dim] half
pub const TURBOQUANT_ATTENTION: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline uint kv_cache_base_attn(uint kv_head, uint max_seq_len, uint head_dim, uint n_bits) {
    uint bytes_per_pos = (n_bits == 4) ? (head_dim / 2) : head_dim;
    return kv_head * max_seq_len * bytes_per_pos;
}

inline float read_quantized_tile_attn(threadgroup const char* tile,
                                      uint pos, uint dim, uint head_dim,
                                      uint n_bits, float deq_scale,
                                      device const float* codebook) {
    if (n_bits == 4) {
        uint packed_stride = head_dim / 2;
        uint byte_idx = pos * packed_stride + dim / 2;
        uchar packed = ((threadgroup const uchar*)tile)[byte_idx];
        uchar nibble = (dim % 2 == 0) ? (packed & 0xF) : (packed >> 4);
        return codebook[nibble] * deq_scale;
    } else {
        return float(tile[pos * head_dim + dim]) * deq_scale;
    }
}

inline void hadamard_rotate_inplace_attn(
    threadgroup float* shared_data,
    device const float* rotation_signs,
    uint head_dim, uint tid, uint tg_size)
{
    for (uint d = tid; d < head_dim; d += tg_size)
        shared_data[d] *= rotation_signs[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 1; step < head_dim; step *= 2) {
        for (uint i = tid; i < head_dim / 2; i += tg_size) {
            uint block = i / step;
            uint offset = i % step;
            uint j = block * (step * 2) + offset;
            float a = shared_data[j];
            float b = shared_data[j + step];
            shared_data[j] = a + b;
            shared_data[j + step] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = 1.0f / sqrt(float(head_dim));
    for (uint d = tid; d < head_dim; d += tg_size)
        shared_data[d] *= rotation_signs[d] * scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

[[kernel]] void turboquant_attention(
    device const half* q                [[buffer(0)]],
    device const char* k_cache          [[buffer(1)]],
    device const char* v_cache          [[buffer(2)]],
    device const float* rotation_signs  [[buffer(3)]],
    device const float* k_scale_buf     [[buffer(4)]],
    device const float* v_scale_buf     [[buffer(5)]],
    device const float* k_codebook      [[buffer(6)]],
    device const float* v_codebook      [[buffer(7)]],
    device const float* qjl_matrix      [[buffer(8)]],
    device const float* k_r_norms       [[buffer(9)]],
    device const uint* params           [[buffer(10)]],
    device half* output                 [[buffer(11)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint num_heads    = params[0];
    uint num_kv_heads = params[1];
    uint head_dim     = params[2];
    uint max_seq_len  = params[3];
    uint seq_len      = params[4];
    uint n_bits       = params[5];

    constexpr uint TILE = 32;
    constexpr uint MAX_DIM = 256;

    threadgroup float shared_q_rot[MAX_DIM];
    threadgroup float shared_s_q[MAX_DIM];
    threadgroup char  kv_tile_raw[TILE * MAX_DIM];
    threadgroup float tile_scales[TILE];
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
    uint q_base = head_idx * head_dim;
    uint kv_base = kv_cache_base_attn(kv_head, max_seq_len, head_dim, n_bits);

    // Load Q and rotate via butterfly
    for (uint d = tid; d < head_dim; d += tg_size)
        shared_q_rot[d] = float(q[q_base + d]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hadamard_rotate_inplace_attn(shared_q_rot, rotation_signs, head_dim, tid, tg_size);

    // Precompute S · q_rot for QJL correction (INT4 only)
    if (n_bits == 4) {
        for (uint out_d = tid; out_d < head_dim; out_d += tg_size) {
            float proj = 0.0f;
            uint row_base = out_d * head_dim;
            for (uint k = 0; k < head_dim; k++)
                proj += qjl_matrix[row_base + k] * shared_q_rot[k];
            shared_s_q[out_d] = proj;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Zero output accumulator
    for (uint d = tid; d < head_dim; d += tg_size)
        shared_output[d] = 0.0f;
    if (tid == 0) {
        softmax_max[0] = -INFINITY;
        softmax_sum[0] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint bytes_per_pos = (n_bits == 4) ? (head_dim / 2) : head_dim;

    // Tiled flash attention
    for (uint tile_start = 0; tile_start < seq_len; tile_start += TILE) {
        uint tile_end = min(tile_start + TILE, seq_len);
        uint actual_tile = tile_end - tile_start;

        // Cooperative load of K tile
        uint tile_bytes = actual_tile * bytes_per_pos;
        for (uint i = tid; i < tile_bytes; i += tg_size)
            kv_tile_raw[i] = ((device const char*)k_cache)[kv_base + tile_start * bytes_per_pos + i];
        for (uint i = tid; i < actual_tile; i += tg_size)
            tile_scales[i] = k_scale_buf[kv_head * max_seq_len + tile_start + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute QK^T
        for (uint p = 0; p < actual_tile; p++) {
            float k_deq = tile_scales[p];
            float partial_dot = 0.0f;
            for (uint d = tid; d < head_dim; d += tg_size) {
                float k_val = read_quantized_tile_attn(
                    kv_tile_raw, p, d, head_dim, n_bits, k_deq, k_codebook);
                partial_dot += shared_q_rot[d] * k_val;
            }

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

        // Per-tile online softmax
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

        float corr = tile_correction[0];
        for (uint d = tid; d < head_dim; d += tg_size)
            shared_output[d] *= corr;

        // Load V tile (reuse kv_tile_raw)
        for (uint i = tid; i < tile_bytes; i += tg_size)
            kv_tile_raw[i] = ((device const char*)v_cache)[kv_base + tile_start * bytes_per_pos + i];
        for (uint i = tid; i < actual_tile; i += tg_size)
            tile_scales[i] = v_scale_buf[kv_head * max_seq_len + tile_start + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate weighted V
        for (uint p = 0; p < actual_tile; p++) {
            float w = tile_scores[p];
            float v_deq = tile_scales[p];
            for (uint d = tid; d < head_dim; d += tg_size) {
                float v_val = read_quantized_tile_attn(
                    kv_tile_raw, p, d, head_dim, n_bits, v_deq, v_codebook);
                shared_output[d] += w * v_val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize
    float denom = max(softmax_sum[0], 1e-10f);
    for (uint d = tid; d < head_dim; d += tg_size)
        shared_output[d] /= denom;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Un-rotate
    hadamard_rotate_inplace_attn(shared_output, rotation_signs, head_dim, tid, tg_size);

    uint out_base = head_idx * head_dim;
    for (uint d = tid; d < head_dim; d += tg_size)
        output[out_base + d] = half(shared_output[d]);
}
"#;

// ── 3. turboquant_outlier_cache_write ───────────────────────────

/// Outlier-aware cache write: splits KV into outlier and non-outlier
/// channel groups, applies independent TurboQuant to each.
///
/// MLX buffer order:
///   0: kv_proj [num_kv_heads × head_dim] half
///   1: channel_indices [head_dim] uint
///   2: outlier_cache (rw)
///   3: non_outlier_cache (rw)
///   4: outlier_rotation_signs [d_outlier_padded] float
///   5: non_outlier_rotation_signs [d_non_padded] float
///   6: outlier_codebook [outlier_n_levels] float
///   7: outlier_boundaries [outlier_n_levels-1] float
///   8: non_outlier_codebook [non_outlier_n_levels] float
///   9: non_outlier_boundaries [non_outlier_n_levels-1] float
///  10: outlier_scale_buf (rw)
///  11: non_outlier_scale_buf (rw)
///  12: params [8] uint — {num_kv_heads, head_dim, max_seq_len, seq_pos,
///      n_outlier, d_outlier_padded, d_non_padded, outlier_n_levels, non_outlier_n_levels}
///  Output 0: dummy [1] float
pub const TURBOQUANT_OUTLIER_CACHE_WRITE: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline void hadamard_rotate_inplace_ocw(
    threadgroup float* shared_data,
    device const float* rotation_signs,
    uint head_dim, uint tid, uint tg_size)
{
    for (uint d = tid; d < head_dim; d += tg_size)
        shared_data[d] *= rotation_signs[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 1; step < head_dim; step *= 2) {
        for (uint i = tid; i < head_dim / 2; i += tg_size) {
            uint block = i / step;
            uint offset = i % step;
            uint j = block * (step * 2) + offset;
            float a = shared_data[j];
            float b = shared_data[j + step];
            shared_data[j] = a + b;
            shared_data[j + step] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = 1.0f / sqrt(float(head_dim));
    for (uint d = tid; d < head_dim; d += tg_size)
        shared_data[d] *= rotation_signs[d] * scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void quantize_group_ocw(
    threadgroup float* shared_vec,
    threadgroup float* shared_reduce,
    threadgroup char* shared_quant,
    device const float* codebook,
    device const float* boundaries,
    uint n_levels,
    device char* cache,
    device float* scale_buf,
    uint dim, uint head_idx, uint max_seq_len, uint seq_pos,
    uint bytes_per_pos, uint tid, uint tg_size)
{
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

    uint cache_base = head_idx * max_seq_len * bytes_per_pos + seq_pos * bytes_per_pos;
    for (uint d = tid * 2; d < dim; d += tg_size * 2) {
        uchar lo = uchar(shared_quant[d]     & 0xF);
        uchar hi = (d + 1 < dim) ? uchar(shared_quant[d + 1] & 0xF) : 0;
        ((device uchar*)cache)[cache_base + d / 2] = lo | (hi << 4);
    }
}

[[kernel]] void turboquant_outlier_cache_write(
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
    device const uint* params                       [[buffer(12)]],
    device float* dummy_out                         [[buffer(13)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint num_kv_heads      = params[0];
    uint head_dim          = params[1];
    uint max_seq_len       = params[2];
    uint seq_pos           = params[3];
    uint n_outlier         = params[4];
    uint d_outlier_padded  = params[5];
    uint d_non_padded      = params[6];
    uint outlier_n_levels  = params[7];
    uint non_outlier_n_levels = params[8];

    threadgroup float shared_outlier[256];
    threadgroup float shared_non_outlier[256];
    threadgroup float shared_reduce[256];
    threadgroup char shared_quant[256];

    uint head_idx = tgid;
    if (head_idx >= num_kv_heads) return;

    uint n_non = head_dim - n_outlier;
    uint input_base = head_idx * head_dim;

    // Extract outlier channels
    for (uint i = tid; i < d_outlier_padded; i += tg_size) {
        if (i < n_outlier) {
            uint src_idx = channel_indices[i];
            shared_outlier[i] = float(kv_proj[input_base + src_idx]);
        } else {
            shared_outlier[i] = 0.0f;
        }
    }

    // Extract non-outlier channels
    for (uint i = tid; i < d_non_padded; i += tg_size) {
        if (i < n_non) {
            uint src_idx = channel_indices[n_outlier + i];
            shared_non_outlier[i] = float(kv_proj[input_base + src_idx]);
        } else {
            shared_non_outlier[i] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Rotate and quantize outlier group
    hadamard_rotate_inplace_ocw(shared_outlier, outlier_rotation_signs, d_outlier_padded, tid, tg_size);
    quantize_group_ocw(
        shared_outlier, shared_reduce, shared_quant,
        outlier_codebook, outlier_boundaries, outlier_n_levels,
        outlier_cache, outlier_scale_buf,
        d_outlier_padded, head_idx, max_seq_len, seq_pos,
        d_outlier_padded / 2, tid, tg_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Rotate and quantize non-outlier group
    hadamard_rotate_inplace_ocw(shared_non_outlier, non_outlier_rotation_signs, d_non_padded, tid, tg_size);
    quantize_group_ocw(
        shared_non_outlier, shared_reduce, shared_quant,
        non_outlier_codebook, non_outlier_boundaries, non_outlier_n_levels,
        non_outlier_cache, non_outlier_scale_buf,
        d_non_padded, head_idx, max_seq_len, seq_pos,
        d_non_padded / 2, tid, tg_size);

    if (tgid == 0 && tid == 0) dummy_out[0] = 1.0f;
}
"#;

// ── 4. turboquant_outlier_attention ─────────────────────────────

/// Attention with dual-group quantized KV cache.
///
/// MLX buffer order:
///   0: q [num_heads × head_dim] half
///   1: k_outlier_cache
///   2: v_outlier_cache
///   3: k_non_outlier_cache
///   4: v_non_outlier_cache
///   5: channel_indices [head_dim] uint
///   6: outlier_rotation_signs [d_outlier_padded] float
///   7: non_outlier_rotation_signs [d_non_padded] float
///   8: outlier_codebook [outlier_n_levels] float
///   9: non_outlier_codebook [non_outlier_n_levels] float
///  10: k_outlier_scales [num_kv_heads × max_seq_len] float
///  11: v_outlier_scales [num_kv_heads × max_seq_len] float
///  12: k_non_outlier_scales [num_kv_heads × max_seq_len] float
///  13: v_non_outlier_scales [num_kv_heads × max_seq_len] float
///  14: params [8] uint — {num_heads, num_kv_heads, head_dim, max_seq_len,
///      seq_len, n_outlier, d_outlier_padded, d_non_padded}
///  Output 0: output [num_heads × head_dim] half
pub const TURBOQUANT_OUTLIER_ATTENTION: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline void hadamard_rotate_inplace_oa(
    threadgroup float* shared_data,
    device const float* rotation_signs,
    uint head_dim, uint tid, uint tg_size)
{
    for (uint d = tid; d < head_dim; d += tg_size)
        shared_data[d] *= rotation_signs[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 1; step < head_dim; step *= 2) {
        for (uint i = tid; i < head_dim / 2; i += tg_size) {
            uint block = i / step;
            uint offset = i % step;
            uint j = block * (step * 2) + offset;
            float a = shared_data[j];
            float b = shared_data[j + step];
            shared_data[j] = a + b;
            shared_data[j + step] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = 1.0f / sqrt(float(head_dim));
    for (uint d = tid; d < head_dim; d += tg_size)
        shared_data[d] *= rotation_signs[d] * scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline float read_quantized_tile_oa(threadgroup const char* tile,
                                    uint pos, uint dim, uint head_dim,
                                    float deq_scale,
                                    device const float* codebook) {
    uint packed_stride = head_dim / 2;
    uint byte_idx = pos * packed_stride + dim / 2;
    uchar packed = ((threadgroup const uchar*)tile)[byte_idx];
    uchar nibble = (dim % 2 == 0) ? (packed & 0xF) : (packed >> 4);
    return codebook[nibble] * deq_scale;
}

[[kernel]] void turboquant_outlier_attention(
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
    device const float* k_outlier_scales            [[buffer(10)]],
    device const float* v_outlier_scales            [[buffer(11)]],
    device const float* k_non_outlier_scales        [[buffer(12)]],
    device const float* v_non_outlier_scales        [[buffer(13)]],
    device const uint* params                       [[buffer(14)]],
    device half* output                             [[buffer(15)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint num_heads        = params[0];
    uint num_kv_heads     = params[1];
    uint head_dim         = params[2];
    uint max_seq_len      = params[3];
    uint seq_len          = params[4];
    uint n_outlier        = params[5];
    uint d_outlier_padded = params[6];
    uint d_non_padded     = params[7];

    constexpr uint TILE = 32;
    constexpr uint MAX_DIM = 256;
    constexpr uint MAX_PACKED = 128;

    threadgroup float shared_q_outlier[MAX_DIM];
    threadgroup float shared_q_non_outlier[MAX_DIM];
    threadgroup char  outlier_kv_tile[TILE * MAX_PACKED];
    threadgroup char  non_outlier_kv_tile[TILE * MAX_PACKED];
    threadgroup float o_tile_scales[TILE];
    threadgroup float n_tile_scales[TILE];
    threadgroup float shared_reduce[MAX_DIM];
    threadgroup float tile_scores[TILE];
    threadgroup float shared_output_outlier[MAX_DIM];
    threadgroup float shared_output_non_outlier[MAX_DIM];
    threadgroup float softmax_max[1];
    threadgroup float softmax_sum[1];
    threadgroup float tile_correction[1];

    uint head_idx = tgid;
    if (head_idx >= num_heads) return;

    uint heads_per_group = num_heads / num_kv_heads;
    uint kv_head = head_idx / heads_per_group;
    float scale = 1.0f / sqrt(float(head_dim));
    uint q_base = head_idx * head_dim;
    uint n_non = head_dim - n_outlier;

    // Load and rotate Q for both groups
    for (uint i = tid; i < d_outlier_padded; i += tg_size) {
        shared_q_outlier[i] = (i < n_outlier) ? float(q[q_base + channel_indices[i]]) : 0.0f;
    }
    for (uint i = tid; i < d_non_padded; i += tg_size) {
        shared_q_non_outlier[i] = (i < n_non) ? float(q[q_base + channel_indices[n_outlier + i]]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    hadamard_rotate_inplace_oa(shared_q_outlier, outlier_rotation_signs, d_outlier_padded, tid, tg_size);
    hadamard_rotate_inplace_oa(shared_q_non_outlier, non_outlier_rotation_signs, d_non_padded, tid, tg_size);

    // Zero accumulators
    for (uint d = tid; d < d_outlier_padded; d += tg_size) shared_output_outlier[d] = 0.0f;
    for (uint d = tid; d < d_non_padded; d += tg_size) shared_output_non_outlier[d] = 0.0f;
    if (tid == 0) {
        softmax_max[0] = -INFINITY;
        softmax_sum[0] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint o_bytes_per_pos = d_outlier_padded / 2;
    uint n_bytes_per_pos = d_non_padded / 2;
    uint o_base = kv_head * max_seq_len * o_bytes_per_pos;
    uint n_base = kv_head * max_seq_len * n_bytes_per_pos;

    for (uint tile_start = 0; tile_start < seq_len; tile_start += TILE) {
        uint tile_end = min(tile_start + TILE, seq_len);
        uint actual_tile = tile_end - tile_start;

        // Load K tiles
        uint o_tile_bytes = actual_tile * o_bytes_per_pos;
        for (uint i = tid; i < o_tile_bytes; i += tg_size)
            outlier_kv_tile[i] = k_outlier_cache[o_base + tile_start * o_bytes_per_pos + i];
        uint n_tile_bytes = actual_tile * n_bytes_per_pos;
        for (uint i = tid; i < n_tile_bytes; i += tg_size)
            non_outlier_kv_tile[i] = k_non_outlier_cache[n_base + tile_start * n_bytes_per_pos + i];
        for (uint i = tid; i < actual_tile; i += tg_size) {
            o_tile_scales[i] = k_outlier_scales[kv_head * max_seq_len + tile_start + i];
            n_tile_scales[i] = k_non_outlier_scales[kv_head * max_seq_len + tile_start + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute Q · K
        for (uint p = 0; p < actual_tile; p++) {
            float k_o_deq = o_tile_scales[p];
            float k_n_deq = n_tile_scales[p];
            float partial_dot = 0.0f;
            for (uint d = tid; d < d_outlier_padded; d += tg_size) {
                float k_val = read_quantized_tile_oa(outlier_kv_tile, p, d, d_outlier_padded, k_o_deq, outlier_codebook);
                partial_dot += shared_q_outlier[d] * k_val;
            }
            for (uint d = tid; d < d_non_padded; d += tg_size) {
                float k_val = read_quantized_tile_oa(non_outlier_kv_tile, p, d, d_non_padded, k_n_deq, non_outlier_codebook);
                partial_dot += shared_q_non_outlier[d] * k_val;
            }
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

        // Online softmax
        if (tid == 0) {
            float tm = -INFINITY;
            for (uint p = 0; p < actual_tile; p++) tm = max(tm, tile_scores[p]);
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

        float corr = tile_correction[0];
        for (uint d = tid; d < d_outlier_padded; d += tg_size) shared_output_outlier[d] *= corr;
        for (uint d = tid; d < d_non_padded; d += tg_size) shared_output_non_outlier[d] *= corr;

        // Load V tiles
        for (uint i = tid; i < o_tile_bytes; i += tg_size)
            outlier_kv_tile[i] = v_outlier_cache[o_base + tile_start * o_bytes_per_pos + i];
        for (uint i = tid; i < n_tile_bytes; i += tg_size)
            non_outlier_kv_tile[i] = v_non_outlier_cache[n_base + tile_start * n_bytes_per_pos + i];
        for (uint i = tid; i < actual_tile; i += tg_size) {
            o_tile_scales[i] = v_outlier_scales[kv_head * max_seq_len + tile_start + i];
            n_tile_scales[i] = v_non_outlier_scales[kv_head * max_seq_len + tile_start + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint p = 0; p < actual_tile; p++) {
            float w = tile_scores[p];
            float v_o_deq = o_tile_scales[p];
            for (uint d = tid; d < d_outlier_padded; d += tg_size) {
                float v_val = read_quantized_tile_oa(outlier_kv_tile, p, d, d_outlier_padded, v_o_deq, outlier_codebook);
                shared_output_outlier[d] += w * v_val;
            }
            float v_n_deq = n_tile_scales[p];
            for (uint d = tid; d < d_non_padded; d += tg_size) {
                float v_val = read_quantized_tile_oa(non_outlier_kv_tile, p, d, d_non_padded, v_n_deq, non_outlier_codebook);
                shared_output_non_outlier[d] += w * v_val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize
    float denom = max(softmax_sum[0], 1e-10f);
    for (uint d = tid; d < d_outlier_padded; d += tg_size) shared_output_outlier[d] /= denom;
    for (uint d = tid; d < d_non_padded; d += tg_size) shared_output_non_outlier[d] /= denom;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Un-rotate
    hadamard_rotate_inplace_oa(shared_output_outlier, outlier_rotation_signs, d_outlier_padded, tid, tg_size);
    hadamard_rotate_inplace_oa(shared_output_non_outlier, non_outlier_rotation_signs, d_non_padded, tid, tg_size);

    // Scatter back
    uint out_base = head_idx * head_dim;
    for (uint i = tid; i < n_outlier; i += tg_size)
        output[out_base + channel_indices[i]] = half(shared_output_outlier[i]);
    for (uint i = tid; i < n_non; i += tg_size)
        output[out_base + channel_indices[n_outlier + i]] = half(shared_output_non_outlier[i]);
}
"#;

// ── 5. PolarQuant INT4 matvec ───────────────────────────────────

/// PolarQuant INT4 matvec (decode path, M=1).
///
/// MLX buffer order:
///   0: A [1, K] half
///   1: B_packed [N, K/2] uchar
///   2: lut [16] half
///   3: norms [N] half
///   4: params [2] uint — {N, K}
///  Output 0: C [1, N] half
pub const POLARQUANT_MATVEC_INT4: &str = r#"
#include <metal_stdlib>
using namespace metal;

[[kernel]] void polarquant_matvec_int4(
    device const half *A            [[buffer(0)]],
    device const uchar *B_packed    [[buffer(1)]],
    constant half *lut              [[buffer(2)]],
    device const half *norms        [[buffer(3)]],
    device const uint *params       [[buffer(4)]],
    device half *C                  [[buffer(5)]],
    uint tid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint N = params[0];
    uint K = params[1];
    if (tid >= N) return;

    uint half_K = K / 2;
    uint row_offset = tid * half_K;
    float norm = float(norms[tid]);
    float acc = 0.0f;

    for (uint k = lane; k < half_K; k += 32) {
        uchar packed = B_packed[row_offset + k];
        uchar lo_idx = (packed >> 4) & 0xF;
        uchar hi_idx = packed & 0xF;
        float w0 = float(lut[lo_idx]) * norm;
        float w1 = float(lut[hi_idx]) * norm;
        uint k2 = k * 2;
        acc += float(A[k2])     * w0;
        acc += float(A[k2 + 1]) * w1;
    }

    acc = simd_sum(acc);
    if (lane == 0) C[tid] = half(acc);
}
"#;

// ── 6. PolarQuant INT4 matmul ───────────────────────────────────

/// PolarQuant INT4 tiled GEMM (prefill path, M>1).
///
/// MLX buffer order:
///   0: A [M, K] half
///   1: B_packed [N, K/2] uchar
///   2: lut [16] half
///   3: norms [N] half
///   4: params [3] uint — {M, N, K}
///  Output 0: C [M, N] half
pub const POLARQUANT_MATMUL_INT4: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr uint TILE_M = 8;
constant constexpr uint TILE_N = 32;
constant constexpr uint TILE_K = 32;

[[kernel]] void polarquant_matmul_int4(
    device const half *A            [[buffer(0)]],
    device const uchar *B_packed    [[buffer(1)]],
    constant half *lut              [[buffer(2)]],
    device const half *norms        [[buffer(3)]],
    device const uint *params       [[buffer(4)]],
    device half *C                  [[buffer(5)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 local_id [[thread_position_in_threadgroup]])
{
    uint M = params[0];
    uint N = params[1];
    uint K = params[2];

    uint local_row = local_id.y;
    uint local_col = local_id.x;
    uint row = group_id.y * TILE_M + local_row;
    uint col = group_id.x * TILE_N + local_col;

    threadgroup half tg_a[TILE_M * TILE_K];
    threadgroup half tg_b[TILE_N * TILE_K];

    uint thread_idx   = local_row * TILE_N + local_col;
    uint total_threads = TILE_M * TILE_N;
    uint half_K = K / 2;
    float acc = 0.0f;

    for (uint k_base = 0; k_base < K; k_base += TILE_K) {
        for (uint i = thread_idx; i < TILE_M * TILE_K; i += total_threads) {
            uint a_row = i / TILE_K;
            uint a_col = i % TILE_K;
            uint g_row = group_id.y * TILE_M + a_row;
            uint g_col = k_base + a_col;
            tg_a[i] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : half(0);
        }

        for (uint i = thread_idx; i < TILE_N * TILE_K; i += total_threads) {
            uint b_n = i / TILE_K;
            uint b_k = i % TILE_K;
            uint g_n = group_id.x * TILE_N + b_n;
            uint g_k = k_base + b_k;
            half val = half(0);
            if (g_n < N && g_k < K) {
                uint byte_idx = g_n * half_K + g_k / 2;
                uchar packed  = B_packed[byte_idx];
                uchar idx     = (g_k % 2 == 0) ? ((packed >> 4) & 0xF) : (packed & 0xF);
                val = lut[idx] * norms[g_n];
            }
            tg_b[i] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < M && col < N) {
            uint a_base = local_row * TILE_K;
            uint b_base = local_col * TILE_K;
            uint k_end  = min(TILE_K, K - k_base);
            for (uint k = 0; k < k_end; k++)
                acc += float(tg_a[a_base + k]) * float(tg_b[b_base + k]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) C[row * N + col] = half(acc);
}
"#;

// ── 7. PolarQuant INT8 matvec ───────────────────────────────────

/// PolarQuant INT8 matvec (decode path, M=1).
///
/// MLX buffer order:
///   0: A [1, K] half
///   1: B_packed [N, K] uchar
///   2: lut [256] half
///   3: norms [N] half
///   4: params [2] uint — {N, K}
///  Output 0: C [1, N] half
pub const POLARQUANT_MATVEC_INT8: &str = r#"
#include <metal_stdlib>
using namespace metal;

[[kernel]] void polarquant_matvec_int8(
    device const half *A            [[buffer(0)]],
    device const uchar *B_packed    [[buffer(1)]],
    constant half *lut              [[buffer(2)]],
    device const half *norms        [[buffer(3)]],
    device const uint *params       [[buffer(4)]],
    device half *C                  [[buffer(5)]],
    uint tid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint N = params[0];
    uint K = params[1];
    if (tid >= N) return;

    uint row_offset = tid * K;
    float norm = float(norms[tid]);
    float acc = 0.0f;

    for (uint k = lane; k < K; k += 32) {
        uchar idx = B_packed[row_offset + k];
        float w = float(lut[idx]) * norm;
        acc += float(A[k]) * w;
    }

    acc = simd_sum(acc);
    if (lane == 0) C[tid] = half(acc);
}
"#;

// ── 8. PolarQuant INT8 matmul ───────────────────────────────────

/// PolarQuant INT8 tiled GEMM (prefill path, M>1).
///
/// MLX buffer order:
///   0: A [M, K] half
///   1: B_packed [N, K] uchar
///   2: lut [256] half
///   3: norms [N] half
///   4: params [3] uint — {M, N, K}
///  Output 0: C [M, N] half
pub const POLARQUANT_MATMUL_INT8: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr uint TILE_M = 8;
constant constexpr uint TILE_N = 32;
constant constexpr uint TILE_K = 32;

[[kernel]] void polarquant_matmul_int8(
    device const half *A            [[buffer(0)]],
    device const uchar *B_packed    [[buffer(1)]],
    constant half *lut              [[buffer(2)]],
    device const half *norms        [[buffer(3)]],
    device const uint *params       [[buffer(4)]],
    device half *C                  [[buffer(5)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 local_id [[thread_position_in_threadgroup]])
{
    uint M = params[0];
    uint N = params[1];
    uint K = params[2];

    uint local_row = local_id.y;
    uint local_col = local_id.x;
    uint row = group_id.y * TILE_M + local_row;
    uint col = group_id.x * TILE_N + local_col;

    threadgroup half tg_a[TILE_M * TILE_K];
    threadgroup half tg_b[TILE_N * TILE_K];

    uint thread_idx    = local_row * TILE_N + local_col;
    uint total_threads = TILE_M * TILE_N;
    float acc = 0.0f;

    for (uint k_base = 0; k_base < K; k_base += TILE_K) {
        for (uint i = thread_idx; i < TILE_M * TILE_K; i += total_threads) {
            uint a_row = i / TILE_K;
            uint a_col = i % TILE_K;
            uint g_row = group_id.y * TILE_M + a_row;
            uint g_col = k_base + a_col;
            tg_a[i] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : half(0);
        }

        for (uint i = thread_idx; i < TILE_N * TILE_K; i += total_threads) {
            uint b_n = i / TILE_K;
            uint b_k = i % TILE_K;
            uint g_n = group_id.x * TILE_N + b_n;
            uint g_k = k_base + b_k;
            half val = half(0);
            if (g_n < N && g_k < K) {
                uchar idx = B_packed[g_n * K + g_k];
                val = lut[idx] * norms[g_n];
            }
            tg_b[i] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < M && col < N) {
            uint a_base = local_row * TILE_K;
            uint b_base = local_col * TILE_K;
            uint k_end  = min(TILE_K, K - k_base);
            for (uint k = 0; k < k_end; k++)
                acc += float(tg_a[a_base + k]) * float(tg_b[b_base + k]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) C[row * N + col] = half(acc);
}
"#;
