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


// ============================================================================
// Fused residual + RMSNorm + affine INT4 matvec.
//
// Same concept as fused_residual_norm_matvec but for INT4 affine quantized
// weights. Uses the rms_inv factoring trick:
//   y[row] = rms_inv * sum_i (a[i]+b[i]) * norm_weight[i] * dequant(w[row,i])
//
// One threadgroup per output row, 32 threads (1 simdgroup).
// Each thread loops over K/2 packed bytes (2 nibbles each).
//
// Threadgroup 0 writes residual_output = a + b AND normed_output = RMSNorm(a+b).
//
// Buffers:
//   buffer(0)  a:               [K] half
//   buffer(1)  b:               [K] half
//   buffer(2)  norm_weight:     [K] half
//   buffer(3)  residual_output: [K] half
//   buffer(4)  B_packed:        blocked INT4 weights
//   buffer(5)  scales:          [N, num_groups] half
//   buffer(6)  zeros:           [N, num_groups] half
//   buffer(7)  C:               [N] half
//   buffer(8)  params:          uint[4] — (N, K, group_size, eps_bits)
//   buffer(9)  awq_scales:      [K] half or empty
//   buffer(10) has_awq:         uint
//   buffer(11) normed_output:   [K] half (RMSNorm result for subsequent projections)
//
// Dispatch: (N, 1, 1) threadgroups, (32, 1, 1) threads.
// ============================================================================

// ============================================================================
// TODO: This kernel remains at 32 threads/TG (scalar, 1 row per TG).
// It is a candidate for future AMX optimization, but the residual+norm
// computation must happen before the matvec can use normed input, making
// the 64-row AMX pattern less straightforward (each row needs its own rms_inv).

constant constexpr uint FRN_BLK_N = 64;
constant constexpr uint FRN_BLK_K = 8;

kernel void fused_residual_norm_affine_matvec_int4(
    device const half* a               [[buffer(0)]],
    device const half* b               [[buffer(1)]],
    device const half* norm_weight      [[buffer(2)]],
    device half* residual_output       [[buffer(3)]],
    device const uchar* B_packed       [[buffer(4)]],
    device const half* scales          [[buffer(5)]],
    device const half* zeros           [[buffer(6)]],
    device half* C                     [[buffer(7)]],
    constant uint* params              [[buffer(8)]],
    device const half* awq_scales      [[buffer(9)]],
    constant uint& has_awq             [[buffer(10)]],
    device half* normed_output         [[buffer(11)]],
    uint tid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint N = params[0];
    uint K = params[1];
    uint group_size = params[2];
    float eps = as_type<float>(params[3]);

    if (tid >= N) return;

    uint half_K = K / 2;
    uint num_groups = (K + group_size - 1) / group_size;
    uint scale_row = tid * num_groups;

    // Blocked layout addressing
    uint k_blocks      = (K + FRN_BLK_K - 1) / FRN_BLK_K;
    uint local_k_bytes = FRN_BLK_K / 2;
    uint block_bytes   = FRN_BLK_N * local_k_bytes;
    uint n_block = tid / FRN_BLK_N;
    uint n_local = tid % FRN_BLK_N;

    // First pass: compute sum_sq for RMSNorm AND the un-normalized dot product.
    // We factor rms_inv out: y = rms_inv * sum(normed_input * dequant_weight)
    // where normed_input = (a+b) * norm_weight (without rms_inv).
    float dot_acc = 0.0f;
    float sq_acc = 0.0f;

    for (uint k = lane; k < half_K; k += 32) {
        uint kb = k / local_k_bytes;
        uint b_idx = k % local_k_bytes;
        uint byte_idx = (n_block * k_blocks + kb) * block_bytes
                      + n_local * local_k_bytes + b_idx;

        uchar packed = B_packed[byte_idx];
        uchar lo = packed & 0x0F;
        uchar hi = (packed >> 4) & 0x0F;

        uint k2 = k * 2;
        uint g0 = k2 / group_size;
        uint g1 = (k2 + 1) / group_size;

        float s0 = float(scales[scale_row + g0]);
        float z0 = float(zeros[scale_row + g0]);
        float w0 = (float(lo) - z0) * s0;

        float s1 = float(scales[scale_row + g1]);
        float z1 = float(zeros[scale_row + g1]);
        float w1 = (float(hi) - z1) * s1;

        // Compute input values: (a + b) * norm_weight
        float val0 = float(a[k2]) + float(b[k2]);
        float val1 = float(a[k2 + 1]) + float(b[k2 + 1]);
        sq_acc += val0 * val0 + val1 * val1;

        float normed0 = val0 * float(norm_weight[k2]);
        float normed1 = val1 * float(norm_weight[k2 + 1]);

        if (has_awq) {
            dot_acc += (normed0 / float(awq_scales[k2])) * w0;
            dot_acc += (normed1 / float(awq_scales[k2 + 1])) * w1;
        } else {
            dot_acc += normed0 * w0;
            dot_acc += normed1 * w1;
        }
    }

    // Reduce across SIMD lanes
    sq_acc = simd_sum(sq_acc);
    dot_acc = simd_sum(dot_acc);

    float rms_inv;
    if (lane == 0) {
        rms_inv = rsqrt(sq_acc / float(K) + eps);
        C[tid] = half(dot_acc * rms_inv);
    }

    // Threadgroup 0 writes residual output AND normed output
    if (tid == 0) {
        rms_inv = simd_broadcast_first(rms_inv);
        for (uint i = lane; i < K; i += 32) {
            float val = float(a[i]) + float(b[i]);
            residual_output[i] = half(val);
            normed_output[i] = half(val * rms_inv * float(norm_weight[i]));
        }
    }
}
