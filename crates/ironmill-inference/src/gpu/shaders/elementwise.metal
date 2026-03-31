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
    output[tid] = a[tid] + b[tid];
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
