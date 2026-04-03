#include <metal_stdlib>
using namespace metal;

// Fused residual add + RMSNorm: computes residual = a + b, then normalizes.
// Avoids writing residual to global memory and reading it back for a separate
// normalization pass — the intermediate sum is kept in registers.
//
// Outputs BOTH the normalized result AND the raw residual (for the next skip
// connection), so downstream layers can proceed without a separate copy.
//
// Buffers:
//   buffer(0) a:               [token_count × hidden_size]  half  (current hidden state)
//   buffer(1) b:               [token_count × hidden_size]  half  (skip connection / proj output)
//   buffer(2) weight:          [hidden_size]                half  (norm gamma)
//   buffer(3) normed_output:   [token_count × hidden_size]  half  (normalized for next matmul)
//   buffer(4) residual_output: [token_count × hidden_size]  half  (residual for next skip)
//   buffer(5) eps:             float
//   buffer(6) hidden_size:     uint
//   buffer(7) token_count:     uint
//
// Dispatch: one threadgroup per token, min(1024, hidden_size) threads per group.
// Uses strided loops and threadgroup reduction, matching the existing rms_norm
// kernel pattern for hidden_size > threadgroup size.

kernel void fused_residual_rms_norm(
    device const half* a               [[buffer(0)]],
    device const half* b               [[buffer(1)]],
    device const half* weight          [[buffer(2)]],
    device half* normed_output         [[buffer(3)]],
    device half* residual_output       [[buffer(4)]],
    constant float& eps                [[buffer(5)]],
    constant uint& hidden_size         [[buffer(6)]],
    constant uint& token_count         [[buffer(7)]],
    uint tid    [[thread_position_in_threadgroup]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    // Shared memory for parallel reduction of sum-of-squares.
    // Max 4096 elements (16 KB for float) fits in 32 KB threadgroup budget.
    threadgroup float shared_sum[4096];

    uint token_idx = tgid;
    if (token_idx >= token_count) return;

    uint base = token_idx * hidden_size;

    // Step 1: Compute residual = a + b, write to output, accumulate sum-of-squares
    float local_sum = 0.0f;
    for (uint i = tid; i < hidden_size; i += tg_size) {
        float val = float(a[base + i]) + float(b[base + i]);
        residual_output[base + i] = half(val);
        local_sum += val * val;
    }
    shared_sum[tid] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Parallel reduction to compute total sum-of-squares
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

    // Step 4: Normalize and scale — recompute residual in float to avoid
    // precision loss from the half round-trip through residual_output.
    for (uint i = tid; i < hidden_size; i += tg_size) {
        float val = float(a[base + i]) + float(b[base + i]);
        normed_output[base + i] = half(val * rms_inv * float(weight[i]));
    }
}
