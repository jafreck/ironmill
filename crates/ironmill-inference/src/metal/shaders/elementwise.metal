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
