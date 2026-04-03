#include <metal_stdlib>
using namespace metal;

// Embedding lookup: for each token_id, copy the corresponding row from the
// embedding table into the output buffer.
//
// Buffers:
//   buffer(0) token_ids:       [token_count]                  uint
//   buffer(1) embedding_table: [vocab_size × hidden_size]     half
//   buffer(2) output:          [token_count × hidden_size]    half
//   buffer(3) hidden_size:     uint
//   buffer(4) token_count:     uint
//   buffer(5) vocab_size:     uint
//
// Dispatch: 2D grid [hidden_size, token_count].
//   tid.x = dimension index, tid.y = token index.

kernel void embedding_lookup(
    device const uint* token_ids          [[buffer(0)]],
    device const half* embedding_table    [[buffer(1)]],
    device half* output                   [[buffer(2)]],
    constant uint& hidden_size            [[buffer(3)]],
    constant uint& token_count            [[buffer(4)]],
    constant uint& vocab_size             [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint dim_idx   = tid.x;
    uint token_idx = tid.y;

    if (dim_idx >= hidden_size || token_idx >= token_count) return;

    uint token_id = token_ids[token_idx];
    if (token_id >= vocab_size) {
        output[token_idx * hidden_size + dim_idx] = half(0.0f);
        return;
    }

    // Copy one element from embedding_table[token_id][dim_idx]
    output[token_idx * hidden_size + dim_idx] =
        embedding_table[token_id * hidden_size + dim_idx];
}
