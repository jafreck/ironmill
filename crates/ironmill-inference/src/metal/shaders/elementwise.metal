#include <metal_stdlib>
using namespace metal;

// Residual add: output[i] = a[i] + b[i]
//
// Buffers:
//   buffer(0) a:      [token_count × hidden_size]  half
//   buffer(1) b:      [token_count × hidden_size]  half
//   buffer(2) output: [token_count × hidden_size]  half
//   buffer(3) size:   uint (total element count = token_count * hidden_size)
//
// Dispatch: one thread per element.

kernel void residual_add(
    device const half* a     [[buffer(0)]],
    device const half* b     [[buffer(1)]],
    device half* output      [[buffer(2)]],
    constant uint& size      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= size) return;
    output[tid] = half(float(a[tid]) + float(b[tid]));
}

// Bias add (broadcast): data[i] += bias[i % hidden_size]
//
// Adds a per-channel bias vector [hidden_size] to a matrix [T × hidden_size],
// broadcasting the bias across all T tokens. Used for D2Quant Deviation-Aware
// Correction (DAC) which compensates for quantization-induced mean shift at
// the post-attention LayerNorm output.
//
// Buffers:
//   buffer(0) data:        [T × hidden_size]  half  (in-place)
//   buffer(1) bias:        [hidden_size]       half
//   buffer(2) hidden_size: uint
//   buffer(3) total_size:  uint (T * hidden_size)
//
// Dispatch: one thread per element.

kernel void bias_add(
    device half* data               [[buffer(0)]],
    device const half* bias         [[buffer(1)]],
    constant uint& hidden_size      [[buffer(2)]],
    constant uint& total_size       [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= total_size) return;
    uint h = tid % hidden_size;
    data[tid] = half(float(data[tid]) + float(bias[h]));
}

// Sigmoid gate (in-place): attn_out[i] *= sigmoid(gate[i])
//
// Used by Qwen3.5 attn_output_gate: the Q projection produces interleaved
// Q + gate values.  After attention, the output is gated element-wise with
// sigmoid(gate) before the O projection.
//
// Buffers:
//   buffer(0) attn_out: [token_count × dim]  half  (read-write, modified in place)
//   buffer(1) gate:     [token_count × dim]  half  (read-only)
//   buffer(2) size:     uint (total element count = token_count * dim)
//
// Dispatch: one thread per element.

kernel void sigmoid_gate_inplace(
    device half* attn_out     [[buffer(0)]],
    device const half* gate   [[buffer(1)]],
    constant uint& size       [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= size) return;
    float g = float(gate[tid]);
    float s = 1.0f / (1.0f + exp(-g));
    attn_out[tid] = half(float(attn_out[tid]) * s);
}

// In-place scalar multiply: data[i] *= scalar[0]
//
// Buffers:
//   buffer(0) data:   [size]  half  (read/write)
//   buffer(1) scalar: [1]     half
//   buffer(2) size:   uint
//
// Dispatch: one thread per element.

kernel void scale_buffer(
    device half* data            [[buffer(0)]],
    device const half* scalar    [[buffer(1)]],
    constant uint& size          [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= size) return;
    data[tid] = half(float(data[tid]) * float(scalar[0]));
}

// Buffer copy: output[i] = input[i]
//
// Buffers:
//   buffer(0) input:  [size]  half
//   buffer(1) output: [size]  half
//   buffer(2) size:   uint
//
// Dispatch: one thread per element.

kernel void copy_buffer(
    device const half* input  [[buffer(0)]],
    device half* output       [[buffer(1)]],
    constant uint& size       [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= size) return;
    output[tid] = input[tid];
}

// ── Q8 input quantization ──────────────────────────────────────
//
// Quantize an FP16 input vector [K] to INT8 with per-group scale factors.
// One threadgroup per group: find max abs, compute scale, quantize elements.
//
// Amortized across all projections that read the same input vector (Q, K, V,
// gate, up, etc.), saving ~2× float ops per INT4 matvec dispatch.
//
// Buffers:
//   buffer(0) input:      [K]              half   (FP16 input vector)
//   buffer(1) q8_data:    [K]              char   (INT8 quantized output)
//   buffer(2) q8_scales:  [K/group_size]   float  (per-group scale factors)
//   buffer(3) K:          uint
//   buffer(4) group_size: uint
//
// Dispatch: (K / group_size, 1, 1) threadgroups, (group_size, 1, 1) threads.

kernel void quantize_input_q8(
    device const half *input   [[buffer(0)]],
    device char *q8_data       [[buffer(1)]],
    device float *q8_scales    [[buffer(2)]],
    constant uint &K           [[buffer(3)]],
    constant uint &group_size  [[buffer(4)]],
    uint group_id  [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]])
{
    uint start = group_id * group_size;

    // Phase 1: find max abs in group via simdgroup reduction.
    float local_max = 0.0f;
    for (uint i = lane; i < group_size && (start + i) < K; i += tg_size) {
        local_max = fmax(local_max, fabs(float(input[start + i])));
    }
    local_max = simd_max(local_max);

    float scale = local_max / 127.0f;
    float inv_scale = (scale > 0.0f) ? 127.0f / local_max : 0.0f;

    if (lane == 0) {
        q8_scales[group_id] = scale;
    }

    // Phase 2: quantize elements.
    for (uint i = lane; i < group_size && (start + i) < K; i += tg_size) {
        float v = float(input[start + i]) * inv_scale;
        q8_data[start + i] = char(clamp(rint(v), -127.0f, 127.0f));
    }
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
