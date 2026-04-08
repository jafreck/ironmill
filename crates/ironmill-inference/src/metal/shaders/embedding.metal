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

// ── INT4 affine embedding lookup ────────────────────────────────
//
// Gathers embedding rows from a packed INT4 table and dequantizes to FP16.
// Each thread handles one element, identical to the FP16 embedding_lookup
// dispatch pattern but with on-the-fly dequant.
//
// Packed table layout: row-major [vocab_size, hidden_size/2] bytes,
// each byte packs two 4-bit values (lo nibble = even dim, hi = odd).
// Scales/zeros: [vocab_size, num_groups] FP16, group_size elements per group.
//
// Dispatch: (hidden_size, token_count, 1) threads.

kernel void affine_embedding_lookup_int4(
    device const uint *token_ids      [[buffer(0)]],
    device const uchar *packed_table  [[buffer(1)]],   // [vocab, hidden/2]
    device const half *scales         [[buffer(2)]],   // [vocab, num_groups]
    device const half *zeros          [[buffer(3)]],   // [vocab, num_groups]
    device half *output               [[buffer(4)]],   // [token_count, hidden]
    constant uint &hidden_size        [[buffer(5)]],
    constant uint &token_count        [[buffer(6)]],
    constant uint &vocab_size         [[buffer(7)]],
    constant uint &group_size         [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint dim_idx   = gid.x;
    uint token_idx = gid.y;
    if (dim_idx >= hidden_size || token_idx >= token_count) return;

    uint token_id = token_ids[token_idx];
    if (token_id >= vocab_size) {
        output[token_idx * hidden_size + dim_idx] = half(0.0f);
        return;
    }

    uint half_hidden = hidden_size / 2;
    uint byte_idx = token_id * half_hidden + dim_idx / 2;
    uchar packed = packed_table[byte_idx];
    uchar nibble = (dim_idx & 1) ? (packed >> 4) : (packed & 0x0F);

    uint group_idx = dim_idx / group_size;
    uint num_groups = (hidden_size + group_size - 1) / group_size;
    float s = float(scales[token_id * num_groups + group_idx]);
    float z = float(zeros[token_id * num_groups + group_idx]);

    float val = (float(nibble) - z) * s;
    output[token_idx * hidden_size + dim_idx] = half(val);
}
