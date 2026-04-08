#include <metal_stdlib>
using namespace metal;

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
