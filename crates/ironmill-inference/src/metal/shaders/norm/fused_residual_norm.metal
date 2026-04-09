#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include "common/simdgroup_reduce.metal"
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

// Maximum input dimension that fits in threadgroup memory alongside
// tg_result (FRN_SIMDGROUPS * 64 * 4 = 2048 bytes) and sg_sq_partial
// (32 * 4 = 128 bytes). Total budget: 32768 bytes.
// 15296 half = 30592 bytes, + 2048 + 128 = 32768.
constant constexpr uint TG_INPUT_MAX = 15296;

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
    // Shared memory for cross-simdgroup reduction (max 32 simdgroups = 1024 threads).
    threadgroup float sg_partial[32];

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

    // Step 2: Two-level reduction — simd_sum within each simdgroup (zero
    // barriers), then a short cross-simdgroup reduce (≤32 iterations).
    float sum = threadgroup_reduce_sum(local_sum, tid, tg_size, sg_partial);

    // Step 3: Compute 1/rms from the reduced sum
    float rms_inv = rsqrt(sum / float(hidden_size) + eps);

    // Step 4: Normalize and scale — recompute residual in float to avoid
    // precision loss from the half round-trip through residual_output.
    for (uint i = tid; i < hidden_size; i += tg_size) {
        float val = float(a[base + i]) + float(b[base + i]);
        normed_output[base + i] = half(val * rms_inv * float(weight[i]));
    }
}


// ============================================================================
// Fused residual + RMSNorm + dense FP16 matvec.
//
// Combines three operations into a single dispatch:
//   1. residual[i] = a[i] + b[i]
//   2. normed[i] = residual[i] * rsqrt(mean(sq)) * norm_weight[i]
//   3. y[row] = dot(normed, w_packed[row, :])
//
// Key insight: for matvec (M=1), each threadgroup reads the ENTIRE input
// vector for one output row. Instead of reading pre-normed input, we compute
// (a+b), derive rms_inv via simd reduction, then use the normed values as
// input for the simdgroup_matrix dot product.
//
// The rms_inv scalar factors out of the dot product:
//   y[row] = rms_inv * Σ_i (a[i]+b[i]) * norm_weight[i] * w[row,i]
//
// So we compute the un-normalized dot product in the main loop, accumulate
// sum_sq in parallel, and multiply by rms_inv at the end.
//
// Threadgroup 0 writes residual_output = a + b AND normed_output = RMSNorm(a+b).
//
// Buffers:
//   buffer(0)  a:               [hidden_size] half
//   buffer(1)  b:               [hidden_size] half
//   buffer(2)  norm_weight:     [hidden_size] half (RMSNorm gamma)
//   buffer(3)  residual_output: [hidden_size] half (a+b for skip connection)
//   buffer(4)  w_packed:        [N/8, K/8, 8, 8] half (blocked dense weights)
//   buffer(5)  y:               [N] half (projection output)
//   buffer(6)  params:          uint[4] — (K, N, eps_bits, 0)
//   buffer(7)  normed_output:   [hidden_size] half (RMSNorm result for subsequent projections)
//
// Dispatch: ((N + 63) / 64, 1, 1) threadgroups, (256, 1, 1) threads.
// ============================================================================

constant constexpr uint FRN_ROWS_PER_TG = 64;
constant constexpr uint FRN_ROWS_PER_SG = 8;
constant constexpr uint FRN_TILE_K      = 8;
constant constexpr uint FRN_SIMDGROUPS  = 8;

kernel void fused_residual_norm_matvec(
    device const half* a               [[buffer(0)]],
    device const half* b               [[buffer(1)]],
    device const half* norm_weight      [[buffer(2)]],
    device half* residual_output       [[buffer(3)]],
    device const half* w_packed        [[buffer(4)]],
    device half* y                     [[buffer(5)]],
    constant uint* params              [[buffer(6)]],
    device half* normed_output         [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint sgid   [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]])
{
    uint K   = params[0];
    uint N   = params[1];
    float eps = as_type<float>(params[2]);

    uint row_base = tgid * FRN_ROWS_PER_TG + sgid * FRN_ROWS_PER_SG;
    if (row_base >= N) return;

    // Phase 1: Compute normed input in threadgroup memory.
    // All 256 threads cooperate to compute a+b, sum_sq, and normed values.
    if (K > TG_INPUT_MAX) return;
    threadgroup half tg_input[TG_INPUT_MAX];

    // Step 1a: Compute a+b and accumulate sum_sq (only for pass 1 in first SG).
    // All simdgroups participate in the reduction for rms_inv.
    threadgroup float sg_sq_partial[32];

    float local_sq = 0.0f;
    for (uint i = tid; i < K; i += 256) {
        float val = float(a[i]) + float(b[i]);
        local_sq += val * val;
        // Threadgroup 0 writes residual
        if (tgid == 0) {
            residual_output[i] = half(val);
        }
    }

    // Reduce sum_sq across all 256 threads
    float simd_sq = simd_sum(local_sq);
    if (lane == 0) sg_sq_partial[sgid] = simd_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < FRN_SIMDGROUPS; i++) total += sg_sq_partial[i];
        sg_sq_partial[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = rsqrt(sg_sq_partial[0] / float(K) + eps);

    // Step 1b: Write normed input to threadgroup memory in blocked format.
    // Also write normed_output for subsequent projections (threadgroup 0).
    for (uint i = tid; i < K; i += 256) {
        float val = float(a[i]) + float(b[i]);
        float normed = val * rms_inv * float(norm_weight[i]);
        tg_input[i] = half(normed);
        if (tgid == 0) {
            normed_output[i] = half(normed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Standard simdgroup matrix matvec using normed input from tg memory.
    uint n_blocks = K / FRN_TILE_K;
    uint row_block = row_base / 8;

    simdgroup_matrix<float, 8, 8> acc(0);

    for (uint kb = 0; kb < n_blocks; kb++) {
        simdgroup_matrix<half, 8, 8> w_T;
        simdgroup_load(w_T, w_packed + (row_block * n_blocks + kb) * 64, 8,
                       ulong2(0, 0), true);

        simdgroup_matrix<half, 8, 8> x_mat;
        simdgroup_load(x_mat, tg_input + kb * FRN_TILE_K, 0);

        simdgroup_multiply_accumulate(acc, x_mat, w_T, acc);
    }

    // Store results
    threadgroup float tg_result[FRN_SIMDGROUPS * 64];
    simdgroup_store(acc, tg_result + sgid * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        for (uint r = 0; r < FRN_ROWS_PER_SG; r++) {
            uint n_row = row_base + r;
            if (n_row < N) {
                y[n_row] = half(tg_result[sgid * 64 + r]);
            }
        }
    }
}
