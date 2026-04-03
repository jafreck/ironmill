// Shared TurboQuant helper functions for Metal and MLX backends.
//
// This file is prepended (via include_str!) to both the Metal and MLX
// TurboQuant kernel sources at compile time.  Keep only inline utility
// functions here — no [[kernel]] entry points.

#ifndef TURBOQUANT_HELPERS_METAL
#define TURBOQUANT_HELPERS_METAL

#include <metal_stdlib>
using namespace metal;

#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif
#ifndef HEAD_DIM_PACKED
#define HEAD_DIM_PACKED (HEAD_DIM / 2)
#endif

// ── Cache addressing ────────────────────────────────────────────

/// Compute the byte offset for a KV head's cache region.
inline uint kv_cache_base(uint kv_head, uint max_seq_len, uint head_dim, uint n_bits) {
    uint bytes_per_pos = (n_bits == 4) ? (head_dim / 2) : head_dim;
    return kv_head * max_seq_len * bytes_per_pos;
}

// ── Quantized tile readers ──────────────────────────────────────

/// Read a dequantized scalar from a threadgroup-shared KV tile.
/// Supports both INT4 (packed nibble + codebook) and INT8 layouts.
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

/// INT4-only variant (no n_bits branch). Used by outlier kernels
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

/// INT4 reader for K cache with QJL sign packed in bit 3 of the nibble.
/// The lower 3 bits are the (b-1)-bit codebook index (Algorithm 2).
/// Returns the dequantized codebook value and writes the QJL sign (±1.0)
/// to `qjl_sign_out`.
inline float read_quantized_tile_int4_qjl(threadgroup const char* tile,
                                           uint pos, uint dim, uint head_dim,
                                           float deq_scale,
                                           device const float* codebook,
                                           thread float& qjl_sign_out) {
    uint packed_stride = head_dim / 2;
    uint byte_idx = pos * packed_stride + dim / 2;
    uchar packed = ((threadgroup const uchar*)tile)[byte_idx];
    uchar nibble = (dim % 2 == 0) ? (packed & 0xF) : (packed >> 4);
    uchar cb_index = nibble & 0x7;
    qjl_sign_out = (nibble & 0x8) ? 1.0f : -1.0f;
    return codebook[cb_index] * deq_scale;
}

// ── In-place Walsh-Hadamard butterfly transform ─────────────────
//
// Applies the randomized Hadamard rotation R = (1/√d)·D·H·D in-place
// on a threadgroup-shared buffer.  O(d log d) compute, O(d) storage for
// the sign vector (vs O(d²) for the dense matrix approach).
//
// `shared_data` must have at least `head_dim` elements.
// `rotation_signs` holds d floats (±1.0).

inline void hadamard_rotate_inplace(
    threadgroup float* shared_data,
    device const float* rotation_signs,
    uint head_dim, uint tid, uint tg_size)
{
    // Step 1: Apply diagonal sign matrix D
    for (uint d = tid; d < head_dim; d += tg_size)
        shared_data[d] *= rotation_signs[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: In-place butterfly (unnormalized Walsh-Hadamard)
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

    // Step 3: Apply D again and scale (R = D·H·D is self-inverse)
    float scale = 1.0f / sqrt(float(head_dim));
    for (uint d = tid; d < head_dim; d += tg_size)
        shared_data[d] *= rotation_signs[d] * scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

#endif // TURBOQUANT_HELPERS_METAL
