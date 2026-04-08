#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ============================================================================
// Custom FP16 matrix-vector product using simdgroup matrix hardware.
//
// Computes y = x · W^T where x is [1, K] and W is [N, K] (row-major FP16).
// Weights must be pre-packed into blocked [N/8, K/8, 8, 8] FP16 format,
// which maps directly to 8×8 simdgroup_matrix tiles.
//
// Uses simdgroup_matrix_multiply_accumulate for hardware-accelerated 8×8
// matrix multiply with mixed-precision (half inputs, float accumulator).
// Loads weights transposed so the multiply contracts over K (not N).
// Weight loads through simdgroup_matrix are coalesced by hardware.
//
// Each threadgroup handles ROWS_PER_TG (64) output rows.
// Within the threadgroup, 8 simdgroups (32 threads each = 256 threads) each
// process 8 output rows using one simdgroup_matrix accumulator.
//
// Dispatch: ((N + 63) / 64, 1, 1) threadgroups, (256, 1, 1) threads per group.
// ============================================================================

constant constexpr uint ROWS_PER_TG   = 64;   // output rows per threadgroup
constant constexpr uint ROWS_PER_SG   = 8;    // output rows per simdgroup
constant constexpr uint TILE_K        = 8;     // K-tile size (matches simdgroup_matrix)
constant constexpr uint SIMD_SIZE     = 32;
constant constexpr uint SIMDGROUPS    = 8;     // ROWS_PER_TG / ROWS_PER_SG

kernel void matvec(
    device const half* x             [[buffer(0)]],   // [1, K] input vector
    device const half* w_packed      [[buffer(1)]],   // [N/8, K/8, 8, 8] blocked weights
    device half* y                   [[buffer(2)]],   // [1, N] output vector
    constant uint& N                 [[buffer(3)]],   // output dimension
    constant uint& K                 [[buffer(4)]],   // input dimension (inner)
    uint tgid   [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint sgid   [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]])
{
    // Which 8 output rows this simdgroup handles.
    uint row_base = tgid * ROWS_PER_TG + sgid * ROWS_PER_SG;
    if (row_base >= N) return;

    uint n_blocks = K / TILE_K;    // number of K-tiles
    uint row_block = row_base / 8; // block index along N

    // Hardware-accelerated accumulator (half inputs → float accumulator).
    simdgroup_matrix<float, 8, 8> acc(0);

    for (uint kb = 0; kb < n_blocks; kb++) {
        // Load weight tile transposed: w_T[k, m] = W[row_base + m, kb*8 + k].
        // Transpose converts the row-major 8×8 block into column-major so the
        // multiply below contracts over the K dimension correctly.
        simdgroup_matrix<half, 8, 8> w_T;
        simdgroup_load(w_T, w_packed + (row_block * n_blocks + kb) * 64, 8,
                       ulong2(0, 0), true);

        // Broadcast x[kb*8..kb*8+8] as identical rows (stride=0).
        // x_mat[r, k] = x[kb*8 + k] for all r.
        simdgroup_matrix<half, 8, 8> x_mat;
        simdgroup_load(x_mat, x + kb * TILE_K, 0);

        // acc[r, m] += Σ_k x_mat[r, k] × w_T[k, m]
        //            = Σ_k x[kb*8+k] × W[row_base+m, kb*8+k]
        // All rows of acc are identical; column m holds the partial dot
        // product for output row (row_base + m).
        simdgroup_multiply_accumulate(acc, x_mat, w_T, acc);
    }

    // All rows of acc are identical. Column m = y[row_base + m].
    threadgroup float tg_result[SIMDGROUPS * 64];
    simdgroup_store(acc, tg_result + sgid * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        for (uint r = 0; r < ROWS_PER_SG; r++) {
            uint n_row = row_base + r;
            if (n_row < N) {
                y[n_row] = half(tg_result[sgid * 64 + r]);
            }
        }
    }
}

// ============================================================================
// Custom FP16 matrix-matrix product using simdgroup matrix hardware.
//
// Computes C[M, N] = A[M, K] × W[N, K]^T for M>1 (prefill path).
// Weights must be pre-packed into blocked [N/8, K/8, 8, 8] FP16 format.
//
// Uses double-buffered threadgroup memory and simdgroup_matrix 8×8 tiles.
// 8 simdgroups each handle 8 rows of M, iterating over N in blocks of 8.
//
// Dispatch: ((M+63)/64, (N+63)/64, 1) threadgroups, (256, 1, 1) threads.
// ============================================================================

#include "common/matmul_tile_constants.h"
constant constexpr uint MATMUL_K_TILE  = 32;   // K-step (4 MMA ops per tile)
constant constexpr uint K_BLOCKS       = MATMUL_K_TILE / 8;  // 4 MMA ops per K-tile

kernel void matmul(
    device const half* A          [[buffer(0)]],   // [M, K] activations
    device const half* w_packed   [[buffer(1)]],   // [N/8, K/8, 8, 8] blocked weights
    device half* C                [[buffer(2)]],   // [M, N] output
    constant uint& M              [[buffer(3)]],
    constant uint& N              [[buffer(4)]],
    constant uint& K              [[buffer(5)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    uint tg_m = group_id.x * TM_TILE;   // M from x (fast-varying)
    uint tg_n = group_id.y * TN_TILE;   // N from y

    uint num_k_steps = (K + MATMUL_K_TILE - 1) / MATMUL_K_TILE;
    uint total_k_blocks = (K + 7) / 8;  // total BLK_K=8 blocks in K dimension

    // Accumulators: each simdgroup handles 8 rows of M × all TN_BLOCKS columns.
    simdgroup_matrix<float, 8, 8> acc[TN_BLOCKS];
    for (uint j = 0; j < TN_BLOCKS; j++) acc[j] = simdgroup_matrix<float, 8, 8>(0);

    threadgroup half tg_a[2][TM_TILE * MATMUL_K_TILE];     // [64×32] activations
    threadgroup half tg_bt[2][MATMUL_K_TILE * TN_STRIDE];  // [32×65] weights transposed

    // ---- Prologue: load first K-tile into buffer 0 ----
    {
        uint k_base = 0;
        // Load A tile
        for (uint i = tid; i < TM_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
            uint m = i / MATMUL_K_TILE;
            uint k = i % MATMUL_K_TILE;
            uint g_row = tg_m + m;
            uint g_col = k_base + k;
            tg_a[0][i] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : half(0);
        }
        // Load B tile (transposed from blocked format)
        for (uint i = tid; i < TN_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
            uint n = i / MATMUL_K_TILE;
            uint k = i % MATMUL_K_TILE;
            uint g_n = tg_n + n;
            uint g_k = k_base + k;
            half val = half(0);
            if (g_n < N && g_k < K) {
                uint n_block = g_n / 8;
                uint n_local = g_n % 8;
                uint k_block_idx = g_k / 8;
                uint k_in_block = g_k % 8;
                uint w_offset = (n_block * total_k_blocks + k_block_idx) * 64 + n_local * 8 + k_in_block;
                val = w_packed[w_offset];
            }
            tg_bt[0][k * TN_STRIDE + n] = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Main double-buffered loop ----
    for (uint kb = 0; kb < num_k_steps; kb++) {
        uint cur = kb & 1;
        uint nxt = 1 - cur;

        // Prefetch next K-tile into the alternate buffer.
        uint next_kb = kb + 1;
        if (next_kb < num_k_steps) {
            uint k_base = next_kb * MATMUL_K_TILE;
            for (uint i = tid; i < TM_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
                uint m = i / MATMUL_K_TILE;
                uint k = i % MATMUL_K_TILE;
                uint g_row = tg_m + m;
                uint g_col = k_base + k;
                tg_a[nxt][i] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : half(0);
            }
            for (uint i = tid; i < TN_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
                uint n = i / MATMUL_K_TILE;
                uint k = i % MATMUL_K_TILE;
                uint g_n = tg_n + n;
                uint g_k = k_base + k;
                half val = half(0);
                if (g_n < N && g_k < K) {
                    uint n_block = g_n / 8;
                    uint n_local = g_n % 8;
                    uint k_block_idx = g_k / 8;
                    uint k_in_block = g_k % 8;
                    uint w_offset = (n_block * total_k_blocks + k_block_idx) * 64 + n_local * 8 + k_in_block;
                    val = w_packed[w_offset];
                }
                tg_bt[nxt][k * TN_STRIDE + n] = val;
            }
        }

        // Compute: each simdgroup iterates over K_BLOCKS 8×8 MMA ops per tile
        for (uint kbi = 0; kbi < K_BLOCKS; kbi++) {
            simdgroup_matrix<half, 8, 8> a_mat;
            simdgroup_load(a_mat, tg_a[cur] + sgid * 8 * MATMUL_K_TILE + kbi * 8, MATMUL_K_TILE);
            for (uint j = 0; j < TN_BLOCKS; j++) {
                simdgroup_matrix<half, 8, 8> bt_mat;
                simdgroup_load(bt_mat, tg_bt[cur] + kbi * 8 * TN_STRIDE + j * 8, TN_STRIDE);
                simdgroup_multiply_accumulate(acc[j], a_mat, bt_mat, acc[j]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Store results ----
    threadgroup float tg_out[N_SIMDGROUPS * 8 * 8];

    for (uint j = 0; j < TN_BLOCKS; j++) {
        simdgroup_store(acc[j], tg_out + sgid * 64, 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = tid; i < TM_TILE * 8; i += THREADS_PER_TG) {
            uint local_m = i / 8;
            uint local_n = i % 8;
            uint out_row = tg_m + local_m;
            uint out_col = tg_n + j * 8 + local_n;
            if (out_row < M && out_col < N) {
                uint sg = local_m / 8;
                uint sg_row = local_m % 8;
                C[out_row * N + out_col] = half(tg_out[sg * 64 + sg_row * 8 + local_n]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}


// ============================================================================
// Batched FP16 matrix-vector product for GDN projections.
//
// Computes 4 independent matvecs in a single dispatch:
//   y_i = x · W_i^T   for i in {0, 1, 2, 3}
//
// All 4 projections share the same input x but have different weight matrices
// and output buffers. Threadgroups are partitioned across the 4 projections
// using cumulative N boundaries.
//
// Each threadgroup handles ROWS_PER_TG (64) output rows, identical to the
// single matvec kernel. The threadgroup index determines which projection
// and which row offset within that projection.
//
// Dispatch: (ceil(N0/64) + ceil(N1/64) + ceil(N2/64) + ceil(N3/64), 1, 1)
//           threadgroups, (256, 1, 1) threads per group.
//
// Buffers:
//   buffer(0)  x:       [1, K] input vector (shared)
//   buffer(1)  w0:      [N0/8, K/8, 8, 8] blocked weights (projection 0)
//   buffer(2)  w1:      [N1/8, K/8, 8, 8] blocked weights (projection 1)
//   buffer(3)  w2:      [N2/8, K/8, 8, 8] blocked weights (projection 2)
//   buffer(4)  w3:      [N3/8, K/8, 8, 8] blocked weights (projection 3)
//   buffer(5)  y0:      [1, N0] output (projection 0)
//   buffer(6)  y1:      [1, N1] output (projection 1)
//   buffer(7)  y2:      [1, N2] output (projection 2)
//   buffer(8)  y3:      [1, N3] output (projection 3)
//   buffer(9)  params:  uint[6] — (K, N0, N1, N2, N3, 0)
// ============================================================================

kernel void gdn_batched_matvec(
    device const half* x              [[buffer(0)]],
    device const half* w0             [[buffer(1)]],
    device const half* w1             [[buffer(2)]],
    device const half* w2             [[buffer(3)]],
    device const half* w3             [[buffer(4)]],
    device half* y0                   [[buffer(5)]],
    device half* y1                   [[buffer(6)]],
    device half* y2                   [[buffer(7)]],
    device half* y3                   [[buffer(8)]],
    constant uint* params             [[buffer(9)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint sgid   [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]])
{
    uint K  = params[0];
    uint N0 = params[1];
    uint N1 = params[2];
    uint N2 = params[3];
    uint N3 = params[4];

    // Determine which projection this threadgroup belongs to
    uint tg0 = (N0 + ROWS_PER_TG - 1) / ROWS_PER_TG;
    uint tg1 = (N1 + ROWS_PER_TG - 1) / ROWS_PER_TG;
    uint tg2 = (N2 + ROWS_PER_TG - 1) / ROWS_PER_TG;

    uint local_tgid;
    device const half* w_packed;
    device half* y;
    uint N;

    if (tgid < tg0) {
        local_tgid = tgid;
        w_packed = w0; y = y0; N = N0;
    } else if (tgid < tg0 + tg1) {
        local_tgid = tgid - tg0;
        w_packed = w1; y = y1; N = N1;
    } else if (tgid < tg0 + tg1 + tg2) {
        local_tgid = tgid - tg0 - tg1;
        w_packed = w2; y = y2; N = N2;
    } else {
        local_tgid = tgid - tg0 - tg1 - tg2;
        w_packed = w3; y = y3; N = N3;
    }

    // Standard matvec: 8 simdgroups × 8 rows = 64 output rows per threadgroup
    uint row_base = local_tgid * ROWS_PER_TG + sgid * ROWS_PER_SG;
    if (row_base >= N) return;

    uint n_blocks = K / TILE_K;
    uint row_block = row_base / 8;

    simdgroup_matrix<float, 8, 8> acc(0);

    for (uint kb = 0; kb < n_blocks; kb++) {
        simdgroup_matrix<half, 8, 8> w_T;
        simdgroup_load(w_T, w_packed + (row_block * n_blocks + kb) * 64, 8,
                       ulong2(0, 0), true);

        simdgroup_matrix<half, 8, 8> x_mat;
        simdgroup_load(x_mat, x + kb * TILE_K, 0);

        simdgroup_multiply_accumulate(acc, x_mat, w_T, acc);
    }

    threadgroup float tg_result[SIMDGROUPS * 64];
    simdgroup_store(acc, tg_result + sgid * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        for (uint r = 0; r < ROWS_PER_SG; r++) {
            uint n_row = row_base + r;
            if (n_row < N) {
                y[n_row] = half(tg_result[sgid * 64 + r]);
            }
        }
    }
}
