#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Custom FP16 matrix-vector product with 8×8 tiling and SIMD reduction.
//
// Computes y = x · W^T where x is [1, K] and W is [N, K] (row-major FP16).
// Weights must be pre-packed into blocked [N/8, K/8, 8, 8] FP16 format.
//
// Each threadgroup handles ROWS_PER_TG (64) output rows.
// Within the threadgroup, 8 simdgroups (32 threads each = 256 threads) each
// process 8 output rows using standard SIMD ops with simd_sum reduction.
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

    // Each lane accumulates partial sums for its assigned output rows.
    // With 32 lanes and 8 output rows, we tile: each lane accumulates
    // across K, then we reduce within the simdgroup.
    //
    // Strategy: iterate over K in blocks of 8. For each K-block, each
    // lane loads one element of x and the corresponding 8×8 weight tile,
    // then accumulates dot products.

    // Accumulators for 8 output rows.
    float acc[ROWS_PER_SG] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Each lane processes K elements strided by SIMD_SIZE within each 8-wide
    // K-tile. Since TILE_K=8 < SIMD_SIZE=32, we assign lanes to K-tiles:
    // - 32 lanes cover 4 consecutive K-tiles (4 × 8 = 32 elements).
    // - Each lane handles one element per K-tile.
    //
    // Process K in chunks of 32 (4 K-tiles at a time).
    for (uint k_base = 0; k_base < K; k_base += SIMD_SIZE) {
        uint k = k_base + lane;
        float x_val = (k < K) ? float(x[k]) : 0.0f;

        // For each of the 8 output rows, load the weight and accumulate.
        uint k_block = k / TILE_K;         // which K-block this lane falls in
        uint k_local = k % TILE_K;         // position within the 8-wide tile

        for (uint r = 0; r < ROWS_PER_SG; r++) {
            uint n_row = row_base + r;
            if (n_row >= N || k >= K) continue;

            // Blocked layout: w_packed[row_block, k_block, r_local, k_local]
            // where r_local = n_row % 8
            // Linear index: ((row_block * n_blocks + k_block) * 8 + (n_row % 8)) * 8 + k_local
            uint w_idx = ((row_block * n_blocks + k_block) * 8 + r) * 8 + k_local;
            float w_val = float(w_packed[w_idx]);
            acc[r] += x_val * w_val;
        }
    }

    // Reduce across the simdgroup.
    for (uint r = 0; r < ROWS_PER_SG; r++) {
        acc[r] = simd_sum(acc[r]);
    }

    // Lane 0 writes the results.
    if (lane == 0) {
        for (uint r = 0; r < ROWS_PER_SG; r++) {
            uint n_row = row_base + r;
            if (n_row < N) {
                y[n_row] = half(acc[r]);
            }
        }
    }
}
