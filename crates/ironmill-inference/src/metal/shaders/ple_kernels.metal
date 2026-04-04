#include <metal_stdlib>
using namespace metal;

// GELU activation with element-wise multiply:
//   output[i] = gelu(gate[i]) * input_slice[i]
//
// Uses the fast approximation: gelu(x) ≈ x * sigmoid(1.702 * x)
// This avoids the tanh approximation which can produce NaN with certain
// Metal GPU hardware due to intermediate overflow in x^3 computation.
//
// Buffers:
//   buffer(0) gate:   [size]  half  (contiguous, token_count × ple_hidden)
//   buffer(1) input:  [token_count × stride]  half  (strided, each row at stride elements apart)
//   buffer(2) output: [size]  half  (contiguous)
//   buffer(3) ple_hidden: uint  (elements per token in gate/output)
//   buffer(4) token_count: uint
//   buffer(5) input_stride: uint  (row width of input buffer, e.g. num_layers * ple_hidden)
//   buffer(6) input_offset: uint  (column offset into each input row for this layer's slice)
//
// Dispatch: one thread per element in output (token_count × ple_hidden).

kernel void gelu_gate(
    device const half* gate        [[buffer(0)]],
    device const half* input       [[buffer(1)]],
    device half* output            [[buffer(2)]],
    constant uint& ple_hidden      [[buffer(3)]],
    constant uint& token_count     [[buffer(4)]],
    constant uint& input_stride    [[buffer(5)]],
    constant uint& input_offset    [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = token_count * ple_hidden;
    if (tid >= total) return;

    // Map flat tid to (token, element_within_ple)
    uint token = tid / ple_hidden;
    uint elem = tid % ple_hidden;

    float g = float(gate[tid]);
    float inp = float(input[token * input_stride + input_offset + elem]);

    // GELU approximation: gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    // Clamp intermediate to avoid tanh overflow.
    float cube = g * g * g;
    const float kSqrt2OverPi = 0.7978845608f;
    float inner = kSqrt2OverPi * (g + 0.044715f * cube);
    inner = clamp(inner, -10.0f, 10.0f); // prevent tanh overflow
    float gelu_g = 0.5f * g * (1.0f + precise::tanh(inner));

    output[tid] = half(gelu_g * inp);
}

// Add and scale: output[i] = (a[i] + b[i]) * scale
//
// Buffers:
//   buffer(0) a:      [size]  half
//   buffer(1) b:      [size]  half
//   buffer(2) output: [size]  half
//   buffer(3) size:   uint
//   buffer(4) scale:  float
//
// Dispatch: one thread per element.

kernel void add_scale(
    device const half* a      [[buffer(0)]],
    device const half* b      [[buffer(1)]],
    device half* output       [[buffer(2)]],
    constant uint& size       [[buffer(3)]],
    constant float& scale     [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= size) return;
    output[tid] = half((float(a[tid]) + float(b[tid])) * scale);
}
