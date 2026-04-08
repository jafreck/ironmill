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

// GELU-gated activation: output[i] = gelu(gate[i]) * up[i]
// Uses gelu_pytorch_tanh approximation:
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
//
// Same buffer layout as silu_gate — drop-in replacement for Gemma 4.
kernel void gelu_gate(
    device const half* gate   [[buffer(0)]],
    device const half* up     [[buffer(1)]],
    device half* output       [[buffer(2)]],
    constant uint& size       [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= size) return;

    float g = float(gate[tid]);
    float u = float(up[tid]);

    const float kSqrt2OverPi = 0.7978845608f;
    float inner = kSqrt2OverPi * (g + 0.044715f * g * g * g);
    inner = clamp(inner, -10.0f, 10.0f);
    float gelu_g = 0.5f * g * (1.0f + precise::tanh(inner));

    output[tid] = half(gelu_g * u);
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
