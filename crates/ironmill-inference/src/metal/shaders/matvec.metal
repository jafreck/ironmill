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
