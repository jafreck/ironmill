#include <metal_stdlib>
using namespace metal;

// Scatter Q/K/V projections into FP16 KV cache on GPU.
//
// Eliminates CPU readback + scatter loop by performing the copy entirely
// on the GPU command buffer.
//
// Buffers:
//   buffer(0) proj:        [token_count × num_kv_heads × head_dim]  half (input)
//   buffer(1) cache:       [num_kv_heads × max_seq_len × head_dim]  half (output)
//   buffer(2) seq_pos:     uint
//   buffer(3) token_count: uint
//   buffer(4) num_kv_heads: uint
//   buffer(5) head_dim:    uint
//   buffer(6) max_seq_len: uint
//
// Grid: (head_dim, num_kv_heads, token_count)

kernel void kv_scatter(
    device const half* proj      [[buffer(0)]],
    device half* cache           [[buffer(1)]],
    constant uint& seq_pos       [[buffer(2)]],
    constant uint& token_count   [[buffer(3)]],
    constant uint& num_kv_heads  [[buffer(4)]],
    constant uint& head_dim      [[buffer(5)]],
    constant uint& max_seq_len   [[buffer(6)]],
    uint3 tid                    [[thread_position_in_grid]])
{
    uint d = tid.x;
    uint head = tid.y;
    uint t = tid.z;
    if (t >= token_count || head >= num_kv_heads || d >= head_dim) return;

    uint src_idx = (t * num_kv_heads + head) * head_dim + d;
    // Ring buffer: wrap write position for sliding window layers where
    // max_seq_len == window_size. For full-attention layers seq_pos + t
    // never reaches max_seq_len, so the modulo is a no-op.
    uint write_pos = (seq_pos + t) % max_seq_len;
    uint dst_idx = (head * max_seq_len + write_pos) * head_dim + d;
    cache[dst_idx] = proj[src_idx];
}
