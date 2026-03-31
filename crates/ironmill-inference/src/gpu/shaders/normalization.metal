#include <metal_stdlib>
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
    // Shared memory for parallel reduction of sum-of-squares.
    // Max 4096 elements (16 KB for float) fits in 32 KB threadgroup budget.
    threadgroup float shared_sum[4096];

    uint token_idx = tgid;
    if (token_idx >= token_count) return;

    uint base = token_idx * hidden_size;

    // Step 1: Each thread accumulates sum-of-squares for its strided range
    float local_sum = 0.0f;
    for (uint i = tid; i < hidden_size; i += tg_size) {
        float val = float(input[base + i]);
        local_sum += val * val;
    }
    shared_sum[tid] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Parallel reduction to compute total sum-of-squares
    // Round tg_size up to next power of 2 for reduction
    uint reduce_size = 1;
    while (reduce_size < tg_size) reduce_size <<= 1;

    for (uint stride = reduce_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (tid + stride) < tg_size) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Step 3: Compute 1/rms from the reduced sum
    float rms_inv = rsqrt(shared_sum[0] / float(hidden_size) + eps);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Normalize and scale
    for (uint i = tid; i < hidden_size; i += tg_size) {
        float val = float(input[base + i]);
        output[base + i] = half(val * rms_inv * float(weight[i]));
    }
}
