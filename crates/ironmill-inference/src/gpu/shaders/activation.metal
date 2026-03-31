#include <metal_stdlib>
using namespace metal;

// SiLU-gated activation: output[i] = silu(gate[i]) * up[i]
// where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// Buffers:
//   buffer(0) gate:   [token_count × intermediate_size]  half
//   buffer(1) up:     [token_count × intermediate_size]  half
//   buffer(2) output: [token_count × intermediate_size]  half
//   buffer(3) size:   uint (total element count = token_count * intermediate_size)
//
// Dispatch: one thread per element.

kernel void silu_gate(
    device const half* gate   [[buffer(0)]],
    device const half* up     [[buffer(1)]],
    device half* output       [[buffer(2)]],
    constant uint& size       [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= size) return;

    float g = float(gate[tid]);
    float u = float(up[tid]);

    // silu(g) = g * sigmoid(g) = g / (1 + exp(-g))
    float silu_g = g / (1.0f + exp(-g));

    output[tid] = half(silu_g * u);
}
