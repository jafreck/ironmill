#include <metal_stdlib>
using namespace metal;

// Fused logit softcapping: data[i] = softcap * tanh(data[i] / softcap)
//
// Buffers:
//   buffer(0) data:    [count] half (in-place)
//   buffer(1) softcap: float
//   buffer(2) count:   uint
//
// Dispatch: one thread per element.

kernel void fused_softcap(
    device half* data          [[buffer(0)]],
    constant float& softcap    [[buffer(1)]],
    constant uint& count       [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    float x = float(data[tid]);
    data[tid] = half(softcap * precise::tanh(x / softcap));
}
