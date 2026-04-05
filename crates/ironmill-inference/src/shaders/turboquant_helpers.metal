// Shared TurboQuant helper functions for the Metal backend.
//
// This file is prepended (via include_str!) to TurboQuant kernel sources
// at compile time.  Keep only inline utility
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

// ── Packed cache primitives ──────────────────────────────────────
//
// All packing-format knowledge lives here.  Call sites never branch
// on n_bits for addressing or element access.

/// Bytes consumed by one position's worth of quantized elements.
inline uint bytes_per_pos(uint n_bits, uint head_dim) {
    return (n_bits < 8) ? (head_dim / (8 / n_bits)) : head_dim;
}

/// Compute the byte offset for a KV head's cache region.
inline uint kv_cache_base(uint kv_head, uint max_seq_len, uint head_dim, uint n_bits) {
    return kv_head * max_seq_len * bytes_per_pos(n_bits, head_dim);
}

/// Extract one raw unsigned element from a threadgroup-shared tile.
inline uchar read_raw_element(threadgroup const char* tile,
                              uint pos, uint dim, uint head_dim,
                              uint n_bits) {
    uint bpp = bytes_per_pos(n_bits, head_dim);
    uint byte_idx = pos * bpp + dim * n_bits / 8;
    uchar packed = ((threadgroup const uchar*)tile)[byte_idx];
    uint shift = (dim * n_bits) % 8;
    uchar mask = uchar((1u << n_bits) - 1);
    return (packed >> shift) & mask;
}

/// Extract one raw unsigned element from device memory.
inline uchar read_raw_device(device const char* buf, uint base,
                             uint pos, uint dim, uint head_dim,
                             uint n_bits) {
    uint bpp = bytes_per_pos(n_bits, head_dim);
    uint byte_idx = base + pos * bpp + dim * n_bits / 8;
    uchar packed = ((device const uchar*)buf)[byte_idx];
    uint shift = (dim * n_bits) % 8;
    uchar mask = uchar((1u << n_bits) - 1);
    return (packed >> shift) & mask;
}

/// Pack a tile of quantized indices from shared_quant[0..head_dim)
/// into device cache memory.  Handles nibble and byte layouts.
inline void write_packed_elements(threadgroup const char* shared_quant,
                                  device char* cache,
                                  uint cache_base,
                                  uint head_dim, uint n_bits,
                                  uint tid, uint tg_size) {
    if (n_bits == 4) {
        for (uint d = tid * 2; d < head_dim; d += tg_size * 2) {
            uchar lo = uchar(shared_quant[d]     & 0xF);
            uchar hi = (d + 1 < head_dim) ? uchar(shared_quant[d + 1] & 0xF) : 0;
            ((device uchar*)cache)[cache_base + d / 2] = lo | (hi << 4);
        }
    } else {
        for (uint d = tid; d < head_dim; d += tg_size) {
            ((device uchar*)cache)[cache_base + d] = uchar(shared_quant[d]);
        }
    }
}

// ── Quantized tile readers (Algorithm 2 — shared INT4/INT8) ─────

/// Read a V cache element: full b-bit codebook index → centroid × scale.
inline float read_v_tile(threadgroup const char* tile,
                         uint pos, uint dim, uint head_dim,
                         uint n_bits, float deq_scale,
                         device const float* codebook) {
    return codebook[read_raw_element(tile, pos, dim, head_dim, n_bits)] * deq_scale;
}

/// Read a K cache element: (b-1)-bit codebook index + 1-bit QJL sign.
inline float read_k_tile_qjl(threadgroup const char* tile,
                              uint pos, uint dim, uint head_dim,
                              uint n_bits, float deq_scale,
                              device const float* codebook,
                              thread float& qjl_sign_out) {
    uchar raw = read_raw_element(tile, pos, dim, head_dim, n_bits);
    uchar sign_mask = uchar(1) << (n_bits - 1);
    uchar cb_index = raw & (sign_mask - 1);
    qjl_sign_out = (raw & sign_mask) ? 1.0f : -1.0f;
    return codebook[cb_index] * deq_scale;
}

/// INT4-only V reader. Used by outlier kernels (always INT4).
inline float read_quantized_tile_int4(threadgroup const char* tile,
                                      uint pos, uint dim, uint head_dim,
                                      float deq_scale,
                                      device const float* codebook) {
    return read_v_tile(tile, pos, dim, head_dim, 4, deq_scale, codebook);
}

/// INT4-only K reader with QJL sign. Used by outlier kernels.
inline float read_quantized_tile_int4_qjl(threadgroup const char* tile,
                                           uint pos, uint dim, uint head_dim,
                                           float deq_scale,
                                           device const float* codebook,
                                           thread float& qjl_sign_out) {
    return read_k_tile_qjl(tile, pos, dim, head_dim, 4, deq_scale, codebook, qjl_sign_out);
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
