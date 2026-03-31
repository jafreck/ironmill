#include <metal_stdlib>
using namespace metal;

// RoPE: Apply rotary position embeddings to Q and K projections.
//
// Uses half-split pairing: pairs (i, i + half_dim) — NOT interleaved (2i, 2i+1).
// This matches Qwen3/Llama-style RoPE.
//
// Buffers:
//   buffer(0) qk:        [token_count × num_heads × head_dim]  half (in-place)
//   buffer(1) cos_cache:  [max_seq_len × half_head_dim]         half
//   buffer(2) sin_cache:  [max_seq_len × half_head_dim]         half
//   buffer(3) num_heads:   uint
//   buffer(4) head_dim:    uint
//   buffer(5) seq_offset:  uint  (starting position for decode)
//   buffer(6) token_count: uint
//
// Dispatch: one thread per (token, head, half_dim_pair).
//   Grid: [half_head_dim, num_heads, token_count]

kernel void rope(
    device half* qk                [[buffer(0)]],
    device const half* cos_cache   [[buffer(1)]],
    device const half* sin_cache   [[buffer(2)]],
    constant uint& num_heads       [[buffer(3)]],
    constant uint& head_dim        [[buffer(4)]],
    constant uint& seq_offset      [[buffer(5)]],
    constant uint& token_count     [[buffer(6)]],
    uint3 tid [[thread_position_in_grid]])
{
    uint d     = tid.x;  // dimension index within half_dim
    uint h     = tid.y;  // head index
    uint t     = tid.z;  // token index

    uint half_dim = head_dim / 2;
    if (d >= half_dim || h >= num_heads || t >= token_count) return;

    // Position for this token
    uint pos = seq_offset + t;

    // Indices into the cos/sin cache: [pos, d]
    uint cs_idx = pos * half_dim + d;
    float cos_val = float(cos_cache[cs_idx]);
    float sin_val = float(sin_cache[cs_idx]);

    // Indices into qk buffer: [t, h, head_dim]
    // Half-split: pair (d) with (d + half_dim)
    uint base = (t * num_heads + h) * head_dim;
    uint idx_lo = base + d;
    uint idx_hi = base + d + half_dim;

    float x0 = float(qk[idx_lo]);
    float x1 = float(qk[idx_hi]);

    // Apply rotation:
    //   out_lo = x0 * cos - x1 * sin
    //   out_hi = x0 * sin + x1 * cos
    qk[idx_lo] = half(x0 * cos_val - x1 * sin_val);
    qk[idx_hi] = half(x0 * sin_val + x1 * cos_val);
}
