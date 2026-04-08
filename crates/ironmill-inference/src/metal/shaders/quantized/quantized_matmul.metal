#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ============================================================================
// PolarQuant: Quantized matmul kernels with LUT-based dequantization
//
// Weights stay packed in GPU memory; dequantization happens inline during
// the dot-product.  Two paths per bit-width:
//   - matvec (M=1): one threadgroup per output element, SIMD reduction
//   - matmul (M>1): tiled GEMM with threadgroup-shared tiles
// ============================================================================

// ── Matmul tuning parameters ──
// N_SIMDGROUPS controls the threadgroup size: N_SIMDGROUPS * 32 threads.
// Increasing to 16 (512 threads) doubles output rows per threadgroup but
// increases register pressure and threadgroup memory, potentially reducing
// occupancy (fewer concurrent threadgroups per GPU core).
//
// Trade-off:
//   8 simdgroups (256 threads): lower register pressure, higher occupancy
//  16 simdgroups (512 threads): larger tiles, better arithmetic intensity
//
// Profile on target hardware to determine the optimum.
#include "common/matmul_tile_constants.metal"
constant constexpr uint MATMUL_K_TILE  = 8;

// ── INT4 matvec (decode path, M=1) ──────────────────────────────
//
// One threadgroup (single SIMD group) per output element.
// Each lane processes K/(2·32) packed bytes, unpacks two nibbles per byte,
// performs LUT lookup, scales by per-row norm, and dot-products with A.
//
// Dispatch: (N, 1, 1) threadgroups, (32, 1, 1) threads per group.

kernel void polarquant_matvec_int4(
    device const half *A            [[buffer(0)]],   // [1, K]
    device const uchar *B_packed    [[buffer(1)]],   // [N, K/2]
    constant half *lut              [[buffer(2)]],   // [16] reconstruction levels
    device const half *norms        [[buffer(3)]],   // [N] row norms
    device half *C                  [[buffer(4)]],   // [1, N]
    constant uint &N                [[buffer(5)]],
    constant uint &K                [[buffer(6)]],
    uint tid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (tid >= N) return;

    uint half_K = K / 2;
    uint row_offset = tid * half_K;
    float norm = float(norms[tid]);

    float acc = 0.0f;

    // Each lane strides by 32 (SIMD width) over packed bytes
    for (uint k = lane; k < half_K; k += 32) {
        uchar packed = B_packed[row_offset + k];
        uchar hi_idx = (packed >> 4) & 0xF;
        uchar lo_idx = packed & 0xF;

        float w0 = float(lut[hi_idx]) * norm;
        float w1 = float(lut[lo_idx]) * norm;

        uint k2 = k * 2;
        acc += float(A[k2])     * w0;
        acc += float(A[k2 + 1]) * w1;
    }

    acc = simd_sum(acc);

    if (lane == 0) {
        C[tid] = half(acc);
    }
}

// ── INT4 tiled GEMM (prefill path, M>1) ─────────────────────────
//
// Uses simdgroup_matrix_multiply_accumulate for hardware-accelerated 8×8
// matrix multiply. 256 threads = 8 simdgroups, each handling 8 output rows.
//
// Dispatch: (ceil(M/TM_TILE), ceil(N/TN_TILE), 1) threadgroups,
//           (256, 1, 1) threads per group.

kernel void polarquant_matmul_int4(
    device const half *A            [[buffer(0)]],   // [M, K]
    device const uchar *B_packed    [[buffer(1)]],   // [N, K/2]
    constant half *lut              [[buffer(2)]],   // [16] reconstruction levels
    device const half *norms        [[buffer(3)]],   // [N] row norms
    device half *C                  [[buffer(4)]],   // [M, N]
    constant uint &M                [[buffer(5)]],
    constant uint &N                [[buffer(6)]],
    constant uint &K                [[buffer(7)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    uint tg_m = group_id.x * TM_TILE;
    uint tg_n = group_id.y * TN_TILE;

    threadgroup half tg_a[2][TM_TILE * MATMUL_K_TILE];
    threadgroup half tg_bt[2][MATMUL_K_TILE * TN_STRIDE];

    uint num_k_steps = (K + MATMUL_K_TILE - 1) / MATMUL_K_TILE;

    // Blocked layout constants for B-load
    uint k_blocks = num_k_steps;
    uint local_k_bytes = MATMUL_K_TILE / 2;  // 4 bytes per INT4 tile column
    uint block_bytes = TN_TILE * local_k_bytes;

    simdgroup_matrix<float, 8, 8> acc[TN_BLOCKS];
    for (uint j = 0; j < TN_BLOCKS; j++) acc[j] = simdgroup_matrix<float, 8, 8>(0);

    // Prologue: load first tile into buf[0]
    {
        uint k_base = 0;
        for (uint i = tid; i < TM_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
            uint m = i / MATMUL_K_TILE;
            uint k = i % MATMUL_K_TILE;
            uint g_row = tg_m + m;
            uint g_col = k_base + k;
            tg_a[0][i] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : half(0);
        }
        uint block_k = k_base / MATMUL_K_TILE;
        uint block_base = (group_id.y * k_blocks + block_k) * block_bytes;
        for (uint i = tid; i < TN_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
            uint n = i / MATMUL_K_TILE;
            uint k = i % MATMUL_K_TILE;
            uint g_n = tg_n + n;
            uint g_k = k_base + k;
            half val = half(0);
            if (g_n < N && g_k < K) {
                uint byte_idx = block_base + n * local_k_bytes + k / 2;
                uchar packed  = B_packed[byte_idx];
                uchar idx     = (k % 2 == 0) ? ((packed >> 4) & 0xF) : (packed & 0xF);
                val = lut[idx] * norms[g_n];
            }
            tg_bt[0][k * TN_STRIDE + n] = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Main loop: overlap load[next] with compute[current]
    for (uint t = 0; t < num_k_steps; t++) {
        uint cur = t % 2;
        uint nxt = (t + 1) % 2;

        // Load NEXT tile (if not the last step)
        if (t + 1 < num_k_steps) {
            uint k_base = (t + 1) * MATMUL_K_TILE;
            for (uint i = tid; i < TM_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
                uint m = i / MATMUL_K_TILE;
                uint k = i % MATMUL_K_TILE;
                uint g_row = tg_m + m;
                uint g_col = k_base + k;
                tg_a[nxt][i] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : half(0);
            }
            uint block_k = k_base / MATMUL_K_TILE;
            uint block_base = (group_id.y * k_blocks + block_k) * block_bytes;
            for (uint i = tid; i < TN_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
                uint n = i / MATMUL_K_TILE;
                uint k = i % MATMUL_K_TILE;
                uint g_n = tg_n + n;
                uint g_k = k_base + k;
                half val = half(0);
                if (g_n < N && g_k < K) {
                    uint byte_idx = block_base + n * local_k_bytes + k / 2;
                    uchar packed  = B_packed[byte_idx];
                    uchar idx     = (k % 2 == 0) ? ((packed >> 4) & 0xF) : (packed & 0xF);
                    val = lut[idx] * norms[g_n];
                }
                tg_bt[nxt][k * TN_STRIDE + n] = val;
            }
        }

        // Compute on CURRENT tile
        simdgroup_matrix<half, 8, 8> a_mat;
        simdgroup_load(a_mat, tg_a[cur] + sgid * 8 * MATMUL_K_TILE, MATMUL_K_TILE);
        for (uint j = 0; j < TN_BLOCKS; j++) {
            simdgroup_matrix<half, 8, 8> bt_mat;
            simdgroup_load(bt_mat, tg_bt[cur] + j * 8, TN_STRIDE);
            simdgroup_multiply_accumulate(acc[j], a_mat, bt_mat, acc[j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results via threadgroup memory
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

// ── INT8 matvec (decode path, M=1) ──────────────────────────────
//
// Same structure as INT4 matvec but one byte = one index (no nibble
// unpacking).  LUT has 256 entries for INT8.
//
// Dispatch: (N, 1, 1) threadgroups, (32, 1, 1) threads per group.

kernel void polarquant_matvec_int8(
    device const half *A            [[buffer(0)]],   // [1, K]
    device const uchar *B_packed    [[buffer(1)]],   // [N, K]
    constant half *lut              [[buffer(2)]],   // [256] reconstruction levels
    device const half *norms        [[buffer(3)]],   // [N] row norms
    device half *C                  [[buffer(4)]],   // [1, N]
    constant uint &N                [[buffer(5)]],
    constant uint &K                [[buffer(6)]],
    uint tid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (tid >= N) return;

    uint row_offset = tid * K;
    float norm = float(norms[tid]);

    float acc = 0.0f;

    for (uint k = lane; k < K; k += 32) {
        uchar idx = B_packed[row_offset + k];
        float w = float(lut[idx]) * norm;
        acc += float(A[k]) * w;
    }

    acc = simd_sum(acc);

    if (lane == 0) {
        C[tid] = half(acc);
    }
}

// ── INT8 tiled GEMM (prefill path, M>1) ─────────────────────────
//
// Uses simdgroup_matrix_multiply_accumulate for hardware-accelerated 8×8
// matrix multiply. 256 threads = 8 simdgroups, each handling 8 output rows.
//
// Dispatch: (ceil(M/TM_TILE), ceil(N/TN_TILE), 1) threadgroups,
//           (256, 1, 1) threads per group.

kernel void polarquant_matmul_int8(
    device const half *A            [[buffer(0)]],   // [M, K]
    device const uchar *B_packed    [[buffer(1)]],   // [N, K]
    constant half *lut              [[buffer(2)]],   // [256] reconstruction levels
    device const half *norms        [[buffer(3)]],   // [N] row norms
    device half *C                  [[buffer(4)]],   // [M, N]
    constant uint &M                [[buffer(5)]],
    constant uint &N                [[buffer(6)]],
    constant uint &K                [[buffer(7)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    uint tg_m = group_id.x * TM_TILE;
    uint tg_n = group_id.y * TN_TILE;

    threadgroup half tg_a[2][TM_TILE * MATMUL_K_TILE];
    threadgroup half tg_bt[2][MATMUL_K_TILE * TN_STRIDE];

    uint num_k_steps = (K + MATMUL_K_TILE - 1) / MATMUL_K_TILE;

    // Blocked layout constants for B-load
    uint k_blocks = num_k_steps;
    uint local_k_bytes = MATMUL_K_TILE;  // 8 bytes per INT8 tile column
    uint block_bytes = TN_TILE * local_k_bytes;

    simdgroup_matrix<float, 8, 8> acc[TN_BLOCKS];
    for (uint j = 0; j < TN_BLOCKS; j++) acc[j] = simdgroup_matrix<float, 8, 8>(0);

    // Prologue: load first tile into buf[0]
    {
        uint k_base = 0;
        for (uint i = tid; i < TM_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
            uint m = i / MATMUL_K_TILE;
            uint k = i % MATMUL_K_TILE;
            uint g_row = tg_m + m;
            uint g_col = k_base + k;
            tg_a[0][i] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : half(0);
        }
        uint block_k = k_base / MATMUL_K_TILE;
        uint block_base = (group_id.y * k_blocks + block_k) * block_bytes;
        for (uint i = tid; i < TN_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
            uint n = i / MATMUL_K_TILE;
            uint k = i % MATMUL_K_TILE;
            uint g_n = tg_n + n;
            uint g_k = k_base + k;
            half val = half(0);
            if (g_n < N && g_k < K) {
                uint byte_idx = block_base + n * local_k_bytes + k;
                uchar idx = B_packed[byte_idx];
                val = lut[idx] * norms[g_n];
            }
            tg_bt[0][k * TN_STRIDE + n] = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Main loop: overlap load[next] with compute[current]
    for (uint t = 0; t < num_k_steps; t++) {
        uint cur = t % 2;
        uint nxt = (t + 1) % 2;

        // Load NEXT tile (if not the last step)
        if (t + 1 < num_k_steps) {
            uint k_base = (t + 1) * MATMUL_K_TILE;
            for (uint i = tid; i < TM_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
                uint m = i / MATMUL_K_TILE;
                uint k = i % MATMUL_K_TILE;
                uint g_row = tg_m + m;
                uint g_col = k_base + k;
                tg_a[nxt][i] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : half(0);
            }
            uint block_k = k_base / MATMUL_K_TILE;
            uint block_base = (group_id.y * k_blocks + block_k) * block_bytes;
            for (uint i = tid; i < TN_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
                uint n = i / MATMUL_K_TILE;
                uint k = i % MATMUL_K_TILE;
                uint g_n = tg_n + n;
                uint g_k = k_base + k;
                half val = half(0);
                if (g_n < N && g_k < K) {
                    uint byte_idx = block_base + n * local_k_bytes + k;
                    uchar idx = B_packed[byte_idx];
                    val = lut[idx] * norms[g_n];
                }
                tg_bt[nxt][k * TN_STRIDE + n] = val;
            }
        }

        // Compute on CURRENT tile
        simdgroup_matrix<half, 8, 8> a_mat;
        simdgroup_load(a_mat, tg_a[cur] + sgid * 8 * MATMUL_K_TILE, MATMUL_K_TILE);
        for (uint j = 0; j < TN_BLOCKS; j++) {
            simdgroup_matrix<half, 8, 8> bt_mat;
            simdgroup_load(bt_mat, tg_bt[cur] + j * 8, TN_STRIDE);
            simdgroup_multiply_accumulate(acc[j], a_mat, bt_mat, acc[j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results via threadgroup memory
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
