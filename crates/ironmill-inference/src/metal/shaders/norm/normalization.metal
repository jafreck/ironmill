#include <metal_stdlib>
#include "common/simdgroup_reduce.metal"
using namespace metal;

// RMSNorm: output[i] = (input[i] / rms) * weight[i]
// where rms = sqrt(mean(input^2) + eps)
//
// Buffers:
//   buffer(0) input:      [token_count × hidden_size]  half
//   buffer(1) weight:     [hidden_size]                half (gamma)
//   buffer(2) output:     [token_count × hidden_size]  half
//   buffer(3) hidden_size: uint
//   buffer(4) token_count: uint
//   buffer(5) eps:         float
//
// Dispatch: one threadgroup per token, hidden_size threads per group.
// Uses threadgroup reduction for sum-of-squares computation.

kernel void rms_norm(
    device const half* input       [[buffer(0)]],
    device const half* weight      [[buffer(1)]],
    device half* output            [[buffer(2)]],
    constant uint& hidden_size     [[buffer(3)]],
    constant uint& token_count     [[buffer(4)]],
    constant float& eps            [[buffer(5)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    // Shared memory for cross-simdgroup reduction (max 32 simdgroups = 1024 threads).
    threadgroup float sg_partial[32];

    uint token_idx = tgid;
    if (token_idx >= token_count) return;

    uint base = token_idx * hidden_size;

    // Step 1: Each thread accumulates sum-of-squares for its strided range
    float local_sum = 0.0f;
    for (uint i = tid; i < hidden_size; i += tg_size) {
        float val = float(input[base + i]);
        local_sum += val * val;
    }

    // Step 2: Two-level reduction — simd_sum within each simdgroup (zero
    // barriers), then a short cross-simdgroup reduce (≤32 iterations).
    float sum = threadgroup_reduce_sum(local_sum, tid, tg_size, sg_partial);

    // Step 3: Compute 1/rms from the reduced sum
    float rms_inv = rsqrt(sum / float(hidden_size) + eps);

    // Step 4: Normalize and scale
    for (uint i = tid; i < hidden_size; i += tg_size) {
        float val = float(input[base + i]);
        output[base + i] = half(val * rms_inv * float(weight[i]));
    }
}
