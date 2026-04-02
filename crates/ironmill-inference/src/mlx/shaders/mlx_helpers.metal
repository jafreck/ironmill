// Shared helper functions for MLX TurboQuant kernels.
//
// Prepended at compile time (via concat!(include_str!(...))) to each
// TurboQuant kernel source before it is handed to MLX's metal_kernel().
// Do NOT add [[kernel]] functions here — only inline utilities.

#include <metal_stdlib>
using namespace metal;

#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif
#ifndef HEAD_DIM_PACKED
#define HEAD_DIM_PACKED (HEAD_DIM / 2)
#endif

// ── Cache addressing ────────────────────────────────────────────

inline uint kv_cache_base(uint kv_head, uint max_seq_len, uint head_dim, uint n_bits) {
    uint bytes_per_pos = (n_bits == 4) ? (head_dim / 2) : head_dim;
    return kv_head * max_seq_len * bytes_per_pos;
}

// ── Quantized tile readers ──────────────────────────────────────

/// Read a dequantized scalar from a threadgroup-shared KV tile.
/// Supports both INT4 (packed nibble) and INT8 layouts.
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

/// INT4-only variant (no n_bits parameter). Used by outlier kernels
/// which always operate in INT4 mode.
inline float read_quantized_tile_int4(threadgroup const char* tile,
                                      uint pos, uint dim, uint head_dim,
                                      float deq_scale,
                                      device const float* codebook) {
    uint packed_stride = head_dim / 2;
    uint byte_idx = pos * packed_stride + dim / 2;
    uchar packed = ((threadgroup const uchar*)tile)[byte_idx];
    uchar nibble = (dim % 2 == 0) ? (packed & 0xF) : (packed >> 4);
    return codebook[nibble] * deq_scale;
}

/// Read K value + QJL sign from a threadgroup-shared tile (INT4).
/// Returns (k_mse_value, qjl_sign ±1.0).
inline float2 read_k_quantized_tile(threadgroup const char* tile,
                                    uint pos, uint dim, uint head_dim,
                                    float deq_scale,
                                    device const float* k_codebook) {
    uint packed_stride = head_dim / 2;
    uint byte_idx = pos * packed_stride + dim / 2;
    uchar packed = ((threadgroup const uchar*)tile)[byte_idx];
    uchar nibble = (dim % 2 == 0) ? (packed & 0xF) : (packed >> 4);
    uchar idx = nibble & 0x7;
    float sign_val = (nibble & 0x8) ? 1.0f : -1.0f;
    return float2(k_codebook[idx] * deq_scale, sign_val);
}

// ── Hadamard butterfly rotation ─────────────────────────────────

inline void hadamard_rotate_inplace(
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
