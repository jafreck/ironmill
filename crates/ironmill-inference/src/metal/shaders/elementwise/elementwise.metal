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
