#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ============================================================================
// Affine Quantized Matmul: Fused dequant + matmul for INT4/INT8
//
// Weights stay packed in GPU memory; affine dequantization happens inline
// during the dot-product:  w = (quantized - zero) * scale
//
// AWQ compensation (s^{-1}) is now fused into the preceding LayerNorm
// When AWQ scales are present (has_awq=1), the dequantized weight is
// divided by the per-column AWQ scale to compensate for activation-aware
// weight scaling:  w = (quantized - zero) * scale / awq_scale[col]
//
// Two paths per bit-width:
//   - matvec (M=1): one threadgroup per output row, SIMD reduction
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
constant constexpr uint N_SIMDGROUPS   = 8;
constant constexpr uint THREADS_PER_TG = N_SIMDGROUPS * 32;
constant constexpr uint TM_TILE        = N_SIMDGROUPS * 8;   // 64 for 8 SG, 128 for 16 SG
constant constexpr uint TN_TILE        = 64;
constant constexpr uint TN_STRIDE      = TN_TILE + 1;
constant constexpr uint MATMUL_K_TILE  = 32;
constant constexpr uint K_BLOCKS       = MATMUL_K_TILE / 8;  // 4 MMA ops per K-tile
constant constexpr uint TN_BLOCKS      = TN_TILE / 8;

// ── Blocked-layout constants (must match pack_quantized_blocked) ─
constant constexpr uint BLK_N = 64;
constant constexpr uint BLK_K = 8;

// ── AMX decode constants (shared by all AMX matvec kernels) ──
constant constexpr uint AMX_ROWS_PER_TG = 64;
constant constexpr uint AMX_ROWS_PER_SG = 8;
constant constexpr uint AMX_SIMDGROUPS  = 8;
constant constexpr uint AMX_TILE_K      = 128;
constant constexpr uint AMX_TG_SIZE     = 256;

// ── INT4 matvec (decode path, M=1) ──────────────────────────────
//
// One threadgroup per output row. Each lane processes K/(2·32) packed
// bytes, unpacks two nibbles, applies per-group affine dequant, and
// dot-products with A.
//
// B_packed is in blocked layout: [N_blocks, K_blocks, BLK_N, BLK_K/2]
// produced by pack_quantized_blocked().
//
// Dispatch: (N, 1, 1) threadgroups, (32, 1, 1) threads per group.

kernel void affine_matvec_int4(
    device const half *A            [[buffer(0)]],   // [1, K]
    device const uchar *B_packed    [[buffer(1)]],   // blocked [N_blk, K_blk, 64, 4]
    device const half *scales       [[buffer(2)]],   // [N, num_groups]
    device const half *zeros        [[buffer(3)]],   // [N, num_groups]
    device half *C                  [[buffer(4)]],   // [1, N]
    constant uint &N                [[buffer(5)]],
    constant uint &K                [[buffer(6)]],
    constant uint &group_size       [[buffer(7)]],
    device const half *awq_scales   [[buffer(8)]],   // [K] or empty
    constant uint &has_awq          [[buffer(9)]],
    uint tid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (tid >= N) return;

    uint half_K = K / 2;
    uint num_groups = (K + group_size - 1) / group_size;
    uint scale_row = tid * num_groups;

    // Blocked layout addressing
    uint k_blocks     = (K + BLK_K - 1) / BLK_K;
    uint local_k_bytes = BLK_K / 2;            // 4
    uint block_bytes   = BLK_N * local_k_bytes; // 256
    uint n_block = tid / BLK_N;
    uint n_local = tid % BLK_N;

    float acc = 0.0f;

    for (uint k = lane; k < half_K; k += 32) {
        uint kb = k / local_k_bytes;
        uint b  = k % local_k_bytes;
        uint byte_idx = (n_block * k_blocks + kb) * block_bytes
                      + n_local * local_k_bytes + b;

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

        if (has_awq) {
            acc += (float(A[k2]) / float(awq_scales[k2])) * w0;
            acc += (float(A[k2 + 1]) / float(awq_scales[k2 + 1])) * w1;
        } else {
            acc += float(A[k2])     * w0;
            acc += float(A[k2 + 1]) * w1;
        }
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

kernel void affine_matmul_int4(
    device const half *A            [[buffer(0)]],   // [M, K]
    device const uchar *B_packed    [[buffer(1)]],   // [N, K/2]
    device const half *scales       [[buffer(2)]],   // [N, num_groups]
    device const half *zeros        [[buffer(3)]],   // [N, num_groups]
    device half *C                  [[buffer(4)]],   // [M, N]
    constant uint &M                [[buffer(5)]],
    constant uint &N                [[buffer(6)]],
    constant uint &K                [[buffer(7)]],
    constant uint &group_size       [[buffer(8)]],
    device const half *awq_scales   [[buffer(9)]],
    constant uint &has_awq          [[buffer(10)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    uint tg_m = group_id.x * TM_TILE;
    uint tg_n = group_id.y * TN_TILE;

    threadgroup half tg_a[2][TM_TILE * MATMUL_K_TILE];
    threadgroup half tg_bt[2][MATMUL_K_TILE * TN_STRIDE];

    uint num_groups = (K + group_size - 1) / group_size;
    uint num_k_steps = (K + MATMUL_K_TILE - 1) / MATMUL_K_TILE;

    // Blocked layout constants for B-load (BLK_K=8 blocks, independent of K_TILE)
    uint total_k_blocks = (K + BLK_K - 1) / BLK_K;
    uint blk_bytes = BLK_N * (BLK_K / 2);  // bytes per blocked-layout block

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
            half a_val = (g_row < M && g_col < K) ? A[g_row * K + g_col] : half(0);
            if (has_awq && g_col < K) {
                a_val = half(float(a_val) / float(awq_scales[g_col]));
            }
            tg_a[0][i] = a_val;
        }
        for (uint i = tid; i < TN_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
            uint n = i / MATMUL_K_TILE;
            uint k = i % MATMUL_K_TILE;
            uint g_n = tg_n + n;
            uint g_k = k_base + k;
            half val = half(0);
            if (g_n < N && g_k < K) {
                uint n_blk = g_n / BLK_N;
                uint n_loc = g_n % BLK_N;
                uint k_blk = g_k / BLK_K;
                uint k_loc = g_k % BLK_K;
                uint byte_idx = (n_blk * total_k_blocks + k_blk) * blk_bytes
                              + n_loc * (BLK_K / 2) + k_loc / 2;
                uchar packed  = B_packed[byte_idx];
                uchar nibble  = (k_loc % 2 == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
                uint grp = g_k / group_size;
                float s = float(scales[g_n * num_groups + grp]);
                float z = float(zeros[g_n * num_groups + grp]);
                val = half((float(nibble) - z) * s);
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
                half a_val = (g_row < M && g_col < K) ? A[g_row * K + g_col] : half(0);
                if (has_awq && g_col < K) {
                    a_val = half(float(a_val) / float(awq_scales[g_col]));
                }
                tg_a[nxt][i] = a_val;
            }
            for (uint i = tid; i < TN_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
                uint n = i / MATMUL_K_TILE;
                uint k = i % MATMUL_K_TILE;
                uint g_n = tg_n + n;
                uint g_k = k_base + k;
                half val = half(0);
                if (g_n < N && g_k < K) {
                    uint n_blk = g_n / BLK_N;
                    uint n_loc = g_n % BLK_N;
                    uint k_blk = g_k / BLK_K;
                    uint k_loc = g_k % BLK_K;
                    uint byte_idx = (n_blk * total_k_blocks + k_blk) * blk_bytes
                                  + n_loc * (BLK_K / 2) + k_loc / 2;
                    uchar packed  = B_packed[byte_idx];
                    uchar nibble  = (k_loc % 2 == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
                    uint grp = g_k / group_size;
                    float s = float(scales[g_n * num_groups + grp]);
                    float z = float(zeros[g_n * num_groups + grp]);
                    val = half((float(nibble) - z) * s);
                }
                tg_bt[nxt][k * TN_STRIDE + n] = val;
            }
        }

        // Compute on CURRENT tile: K_BLOCKS MMA ops per K-tile
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
// One byte = one element (no nibble unpacking).
//
// B_packed is in blocked layout: [N_blocks, K_blocks, BLK_N, BLK_K]
// produced by pack_quantized_blocked().
//
// Dispatch: (N, 1, 1) threadgroups, (32, 1, 1) threads per group.

kernel void affine_matvec_int8(
    device const half *A            [[buffer(0)]],   // [1, K]
    device const uchar *B_packed    [[buffer(1)]],   // blocked [N_blk, K_blk, 64, 8]
    device const half *scales       [[buffer(2)]],   // [N, num_groups]
    device const half *zeros        [[buffer(3)]],   // [N, num_groups]
    device half *C                  [[buffer(4)]],   // [1, N]
    constant uint &N                [[buffer(5)]],
    constant uint &K                [[buffer(6)]],
    constant uint &group_size       [[buffer(7)]],
    device const half *awq_scales   [[buffer(8)]],
    constant uint &has_awq          [[buffer(9)]],
    uint tid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (tid >= N) return;

    uint num_groups = (K + group_size - 1) / group_size;
    uint scale_row = tid * num_groups;

    // Blocked layout addressing
    uint k_blocks      = (K + BLK_K - 1) / BLK_K;
    uint local_k_bytes = BLK_K;                // 8
    uint block_bytes   = BLK_N * local_k_bytes; // 512
    uint n_block = tid / BLK_N;
    uint n_local = tid % BLK_N;

    float acc = 0.0f;

    for (uint k = lane; k < K; k += 32) {
        uint kb = k / BLK_K;
        uint b  = k % BLK_K;
        uint byte_idx = (n_block * k_blocks + kb) * block_bytes
                      + n_local * local_k_bytes + b;

        uchar q = B_packed[byte_idx];
        uint grp = k / group_size;
        float s = float(scales[scale_row + grp]);
        float z = float(zeros[scale_row + grp]);
        float w = (float(q) - z) * s;
        float a_val = float(A[k]);
        if (has_awq) { a_val /= float(awq_scales[k]); }
        acc += a_val * w;
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

kernel void affine_matmul_int8(
    device const half *A            [[buffer(0)]],   // [M, K]
    device const uchar *B_packed    [[buffer(1)]],   // [N, K]
    device const half *scales       [[buffer(2)]],   // [N, num_groups]
    device const half *zeros        [[buffer(3)]],   // [N, num_groups]
    device half *C                  [[buffer(4)]],   // [M, N]
    constant uint &M                [[buffer(5)]],
    constant uint &N                [[buffer(6)]],
    constant uint &K                [[buffer(7)]],
    constant uint &group_size       [[buffer(8)]],
    device const half *awq_scales   [[buffer(9)]],
    constant uint &has_awq          [[buffer(10)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    uint tg_m = group_id.x * TM_TILE;
    uint tg_n = group_id.y * TN_TILE;

    threadgroup half tg_a[2][TM_TILE * MATMUL_K_TILE];
    threadgroup half tg_bt[2][MATMUL_K_TILE * TN_STRIDE];

    uint num_groups = (K + group_size - 1) / group_size;
    uint num_k_steps = (K + MATMUL_K_TILE - 1) / MATMUL_K_TILE;

    // Blocked layout constants for B-load (BLK_K=8 blocks, independent of K_TILE)
    uint total_k_blocks = (K + BLK_K - 1) / BLK_K;
    uint blk_bytes = BLK_N * BLK_K;  // bytes per blocked-layout block

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
            half a_val = (g_row < M && g_col < K) ? A[g_row * K + g_col] : half(0);
            if (has_awq && g_col < K) {
                a_val = half(float(a_val) / float(awq_scales[g_col]));
            }
            tg_a[0][i] = a_val;
        }
        for (uint i = tid; i < TN_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
            uint n = i / MATMUL_K_TILE;
            uint k = i % MATMUL_K_TILE;
            uint g_n = tg_n + n;
            uint g_k = k_base + k;
            half val = half(0);
            if (g_n < N && g_k < K) {
                uint n_blk = g_n / BLK_N;
                uint n_loc = g_n % BLK_N;
                uint k_blk = g_k / BLK_K;
                uint k_loc = g_k % BLK_K;
                uint byte_idx = (n_blk * total_k_blocks + k_blk) * blk_bytes
                              + n_loc * BLK_K + k_loc;
                uchar q = B_packed[byte_idx];
                uint grp = g_k / group_size;
                float s = float(scales[g_n * num_groups + grp]);
                float z = float(zeros[g_n * num_groups + grp]);
                val = half((float(q) - z) * s);
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
                half a_val = (g_row < M && g_col < K) ? A[g_row * K + g_col] : half(0);
                if (has_awq && g_col < K) {
                    a_val = half(float(a_val) / float(awq_scales[g_col]));
                }
                tg_a[nxt][i] = a_val;
            }
            for (uint i = tid; i < TN_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
                uint n = i / MATMUL_K_TILE;
                uint k = i % MATMUL_K_TILE;
                uint g_n = tg_n + n;
                uint g_k = k_base + k;
                half val = half(0);
                if (g_n < N && g_k < K) {
                    uint n_blk = g_n / BLK_N;
                    uint n_loc = g_n % BLK_N;
                    uint k_blk = g_k / BLK_K;
                    uint k_loc = g_k % BLK_K;
                    uint byte_idx = (n_blk * total_k_blocks + k_blk) * blk_bytes
                                  + n_loc * BLK_K + k_loc;
                    uchar q = B_packed[byte_idx];
                    uint grp = g_k / group_size;
                    float s = float(scales[g_n * num_groups + grp]);
                    float z = float(zeros[g_n * num_groups + grp]);
                    val = half((float(q) - z) * s);
                }
                tg_bt[nxt][k * TN_STRIDE + n] = val;
            }
        }

        // Compute on CURRENT tile: K_BLOCKS MMA ops per K-tile
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


// ── Batched INT4 AMX matvec for FFN gate+up projections (M=1) ──
//
// Computes gate = x · W_gate^T and up = x · W_up^T in a single dispatch.
// Each TG processes 64 rows of either gate or up projection using AMX.
//
// Threadgroup routing: tgid < N_gate_tgs → gate, else → up.
// N_gate_tgs = ceil(N_gate/64), N_up_tgs = ceil(N_up/64) (N_up == N_gate).
//
// Dispatch: (N_gate_tgs + N_up_tgs, 1, 1) threadgroups, (256, 1, 1) threads.

kernel void batched_affine_matvec_int4(
    device const half *A               [[buffer(0)]],    // [1, K] shared input
    device const uchar *B_gate_packed  [[buffer(1)]],    // gate weights
    device const half *scales_gate     [[buffer(2)]],
    device const half *zeros_gate      [[buffer(3)]],
    device half *C_gate               [[buffer(4)]],
    device const uchar *B_up_packed    [[buffer(5)]],    // up weights
    device const half *scales_up       [[buffer(6)]],
    device const half *zeros_up        [[buffer(7)]],
    device half *C_up                 [[buffer(8)]],
    constant uint &N_gate              [[buffer(9)]],
    constant uint &K                   [[buffer(10)]],
    constant uint &group_size          [[buffer(11)]],
    device const half *awq_scales      [[buffer(12)]],
    constant uint &has_awq             [[buffer(13)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint sgid   [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]])
{
    // Route TG to gate or up projection
    uint n_gate_tgs = (N_gate + AMX_ROWS_PER_TG - 1) / AMX_ROWS_PER_TG;
    device const uchar* B_packed;
    device const half* scales;
    device const half* zeros;
    device half* C;
    uint N_proj;
    uint row_base;

    if (tgid < n_gate_tgs) {
        row_base = tgid * AMX_ROWS_PER_TG;
        N_proj = N_gate;
        B_packed = B_gate_packed; scales = scales_gate; zeros = zeros_gate; C = C_gate;
    } else {
        row_base = (tgid - n_gate_tgs) * AMX_ROWS_PER_TG;
        N_proj = N_gate;  // N_up == N_gate
        B_packed = B_up_packed; scales = scales_up; zeros = zeros_up; C = C_up;
    }

    if (row_base >= N_proj) return;
    uint rows_this_tg = min(AMX_ROWS_PER_TG, N_proj - row_base);
    uint n_block_base = row_base / BLK_N;
    uint k_blocks_total = (K + BLK_K - 1) / BLK_K;
    uint num_groups = (K + group_size - 1) / group_size;

    threadgroup half tg_w[AMX_ROWS_PER_TG * AMX_TILE_K];
    threadgroup half tg_x[AMX_TILE_K];

    simdgroup_matrix<float, 8, 8> acc_mat(0);

    for (uint kt = 0; kt < K; kt += AMX_TILE_K) {
        uint tile_k = min(AMX_TILE_K, K - kt);

        for (uint i = tid; i < tile_k; i += AMX_TG_SIZE) {
            float val = float(A[kt + i]);
            if (has_awq) val /= float(awq_scales[kt + i]);
            tg_x[i] = half(val);
        }

        uint half_tile_k = tile_k / 2;
        uint total_pairs = rows_this_tg * half_tile_k;
        for (uint i = tid; i < total_pairs; i += AMX_TG_SIZE) {
            uint n_local = i / half_tile_k;
            uint kp      = i % half_tile_k;
            uint k_abs   = kt + kp * 2;
            uint n_abs   = row_base + n_local;

            uint kb_idx   = k_abs / BLK_K;
            uint k_local  = k_abs % BLK_K;
            uint byte_idx = (n_block_base * k_blocks_total + kb_idx) * (BLK_N * BLK_K / 2)
                          + n_local * (BLK_K / 2)
                          + k_local / 2;

            uchar packed = B_packed[byte_idx];

            uint g0 = k_abs / group_size;
            float s0 = float(scales[n_abs * num_groups + g0]);
            float z0 = float(zeros[n_abs * num_groups + g0]);
            float w0 = (float(packed & 0x0F) - z0) * s0;

            uint g1 = (k_abs + 1) / group_size;
            float s1 = (g1 == g0) ? s0 : float(scales[n_abs * num_groups + g1]);
            float z1 = (g1 == g0) ? z0 : float(zeros[n_abs * num_groups + g1]);
            float w1 = (float(packed >> 4) - z1) * s1;

            tg_w[n_local * AMX_TILE_K + kp * 2]     = half(w0);
            tg_w[n_local * AMX_TILE_K + kp * 2 + 1] = half(w1);
        }
        if (tile_k < AMX_TILE_K) {
            for (uint i = tid; i < rows_this_tg; i += AMX_TG_SIZE) {
                for (uint j = tile_k; j < AMX_TILE_K; j++) {
                    tg_w[i * AMX_TILE_K + j] = 0;
                }
            }
            for (uint i = tile_k + tid; i < AMX_TILE_K; i += AMX_TG_SIZE) {
                tg_x[i] = 0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint sg_row = sgid * AMX_ROWS_PER_SG;
        if (sg_row < rows_this_tg) {
            uint n_k_tiles = (tile_k + 7) / 8;
            for (uint kb = 0; kb < n_k_tiles; kb++) {
                simdgroup_matrix<half, 8, 8> w_T;
                simdgroup_load(w_T, tg_w + sg_row * AMX_TILE_K + kb * 8,
                               AMX_TILE_K, ulong2(0, 0), true);

                simdgroup_matrix<half, 8, 8> x_mat;
                simdgroup_load(x_mat, tg_x + kb * 8, 0);

                simdgroup_multiply_accumulate(acc_mat, x_mat, w_T, acc_mat);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint sg_row = sgid * AMX_ROWS_PER_SG;
    if (sg_row >= rows_this_tg) return;

    threadgroup float tg_result[AMX_SIMDGROUPS * 64];
    simdgroup_store(acc_mat, tg_result + sgid * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        for (uint r = 0; r < AMX_ROWS_PER_SG; r++) {
            uint n_row = row_base + sg_row + r;
            if (n_row < N_proj) {
                C[n_row] = half(tg_result[sgid * 64 + r]);
            }
        }
    }
}

// ── Batched INT4 AMX matvec for 4 GDN projections (M=1) ─────────
//
// Computes 4 independent matvecs in a single dispatch using AMX:
//   qkv = x · W_qkv^T   (ceil(N0/64) TGs)
//   z   = x · W_z^T      (ceil(N1/64) TGs)
//   a   = x · W_a^T      (ceil(N2/64) TGs)
//   b   = x · W_b^T      (ceil(N3/64) TGs)
//
// All projections share the same input x. Threadgroup index determines
// which projection via cumulative TG-count thresholds.
//
// Dispatch: (ceil(N0/64)+ceil(N1/64)+ceil(N2/64)+ceil(N3/64), 1, 1) TGs,
//           (256, 1, 1) threads.

struct GdnBatchedInt4Params {
    uint N0;           // qkv output dim
    uint N1;           // z output dim
    uint N2;           // a output dim
    uint N3;           // b output dim
    uint K;            // input dim (shared)
    uint group_size;
    uint has_awq;
};

kernel void gdn_batched_affine_matvec_int4(
    device const half *A               [[buffer(0)]],    // [1, K] shared input
    device const uchar *B0_packed      [[buffer(1)]],    // qkv weights
    device const half *scales0         [[buffer(2)]],
    device const half *zeros0          [[buffer(3)]],
    device half *C0                    [[buffer(4)]],
    device const uchar *B1_packed      [[buffer(5)]],    // z weights
    device const half *scales1         [[buffer(6)]],
    device const half *zeros1          [[buffer(7)]],
    device half *C1                    [[buffer(8)]],
    device const uchar *B2_packed      [[buffer(9)]],    // a weights
    device const half *scales2         [[buffer(10)]],
    device const half *zeros2          [[buffer(11)]],
    device half *C2                    [[buffer(12)]],
    device const uchar *B3_packed      [[buffer(13)]],    // b weights
    device const half *scales3         [[buffer(14)]],
    device const half *zeros3          [[buffer(15)]],
    device half *C3                    [[buffer(16)]],
    constant GdnBatchedInt4Params &params [[buffer(17)]],
    device const half *awq_scales      [[buffer(18)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint sgid   [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]])
{
    uint N0 = params.N0;
    uint N1 = params.N1;
    uint N2 = params.N2;
    uint N3 = params.N3;
    uint K  = params.K;
    uint group_size = params.group_size;
    uint has_awq = params.has_awq;

    // TG-count thresholds for routing
    uint t0_tgs = (N0 + AMX_ROWS_PER_TG - 1) / AMX_ROWS_PER_TG;
    uint t1_tgs = t0_tgs + (N1 + AMX_ROWS_PER_TG - 1) / AMX_ROWS_PER_TG;
    uint t2_tgs = t1_tgs + (N2 + AMX_ROWS_PER_TG - 1) / AMX_ROWS_PER_TG;

    // Route to correct projection and compute row_base
    uint N_proj;
    uint row_base;
    device const uchar* B_packed;
    device const half* scales;
    device const half* zeros;
    device half* C;

    if (tgid < t0_tgs) {
        row_base = tgid * AMX_ROWS_PER_TG;
        N_proj = N0;
        B_packed = B0_packed; scales = scales0; zeros = zeros0; C = C0;
    } else if (tgid < t1_tgs) {
        row_base = (tgid - t0_tgs) * AMX_ROWS_PER_TG;
        N_proj = N1;
        B_packed = B1_packed; scales = scales1; zeros = zeros1; C = C1;
    } else if (tgid < t2_tgs) {
        row_base = (tgid - t1_tgs) * AMX_ROWS_PER_TG;
        N_proj = N2;
        B_packed = B2_packed; scales = scales2; zeros = zeros2; C = C2;
    } else {
        row_base = (tgid - t2_tgs) * AMX_ROWS_PER_TG;
        N_proj = N3;
        B_packed = B3_packed; scales = scales3; zeros = zeros3; C = C3;
    }

    if (row_base >= N_proj) return;
    uint rows_this_tg = min(AMX_ROWS_PER_TG, N_proj - row_base);
    uint n_block_base = row_base / BLK_N;
    uint k_blocks_total = (K + BLK_K - 1) / BLK_K;
    uint num_groups = (K + group_size - 1) / group_size;

    threadgroup half tg_w[AMX_ROWS_PER_TG * AMX_TILE_K];
    threadgroup half tg_x[AMX_TILE_K];

    simdgroup_matrix<float, 8, 8> acc_mat(0);

    for (uint kt = 0; kt < K; kt += AMX_TILE_K) {
        uint tile_k = min(AMX_TILE_K, K - kt);

        for (uint i = tid; i < tile_k; i += AMX_TG_SIZE) {
            float val = float(A[kt + i]);
            if (has_awq) val /= float(awq_scales[kt + i]);
            tg_x[i] = half(val);
        }

        uint half_tile_k = tile_k / 2;
        uint total_pairs = rows_this_tg * half_tile_k;
        for (uint i = tid; i < total_pairs; i += AMX_TG_SIZE) {
            uint n_local = i / half_tile_k;
            uint kp      = i % half_tile_k;
            uint k_abs   = kt + kp * 2;
            uint n_abs   = row_base + n_local;

            uint kb_idx   = k_abs / BLK_K;
            uint k_local  = k_abs % BLK_K;
            uint byte_idx = (n_block_base * k_blocks_total + kb_idx) * (BLK_N * BLK_K / 2)
                          + n_local * (BLK_K / 2)
                          + k_local / 2;

            uchar packed = B_packed[byte_idx];

            uint g0 = k_abs / group_size;
            float s0 = float(scales[n_abs * num_groups + g0]);
            float z0 = float(zeros[n_abs * num_groups + g0]);
            float w0 = (float(packed & 0x0F) - z0) * s0;

            uint g1 = (k_abs + 1) / group_size;
            float s1 = (g1 == g0) ? s0 : float(scales[n_abs * num_groups + g1]);
            float z1 = (g1 == g0) ? z0 : float(zeros[n_abs * num_groups + g1]);
            float w1 = (float(packed >> 4) - z1) * s1;

            tg_w[n_local * AMX_TILE_K + kp * 2]     = half(w0);
            tg_w[n_local * AMX_TILE_K + kp * 2 + 1] = half(w1);
        }
        if (tile_k < AMX_TILE_K) {
            for (uint i = tid; i < rows_this_tg; i += AMX_TG_SIZE) {
                for (uint j = tile_k; j < AMX_TILE_K; j++) {
                    tg_w[i * AMX_TILE_K + j] = 0;
                }
            }
            for (uint i = tile_k + tid; i < AMX_TILE_K; i += AMX_TG_SIZE) {
                tg_x[i] = 0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint sg_row = sgid * AMX_ROWS_PER_SG;
        if (sg_row < rows_this_tg) {
            uint n_k_tiles = (tile_k + 7) / 8;
            for (uint kb = 0; kb < n_k_tiles; kb++) {
                simdgroup_matrix<half, 8, 8> w_T;
                simdgroup_load(w_T, tg_w + sg_row * AMX_TILE_K + kb * 8,
                               AMX_TILE_K, ulong2(0, 0), true);

                simdgroup_matrix<half, 8, 8> x_mat;
                simdgroup_load(x_mat, tg_x + kb * 8, 0);

                simdgroup_multiply_accumulate(acc_mat, x_mat, w_T, acc_mat);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint sg_row = sgid * AMX_ROWS_PER_SG;
    if (sg_row >= rows_this_tg) return;

    threadgroup float tg_result[AMX_SIMDGROUPS * 64];
    simdgroup_store(acc_mat, tg_result + sgid * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        for (uint r = 0; r < AMX_ROWS_PER_SG; r++) {
            uint n_row = row_base + sg_row + r;
            if (n_row < N_proj) {
                C[n_row] = half(tg_result[sgid * 64 + r]);
            }
        }
    }
}


// ============================================================================
// AMX-accelerated INT4 matvec (decode path, M=1)
//
// Two-phase approach per K-tile:
//   Phase 1: All 256 threads cooperatively dequant a [64 x TILE_K] INT4 tile
//            into FP16 threadgroup memory.
//   Phase 2: Each of 8 simdgroups uses simdgroup_matrix_multiply_accumulate
//            (Apple AMX hardware) on its 8-row slice.
//
// This replaces the scalar inner loop with hardware-accelerated 8x8 matrix
// multiply, yielding ~4x higher throughput at the cost of one threadgroup
// barrier per K-tile.
//
// Same buffer layout as affine_matvec_int4 for drop-in replacement.
// Dispatch: (ceil(N/64), 1, 1) threadgroups, (256, 1, 1) threads.
// ============================================================================

kernel void affine_matvec_int4_amx(
    device const half *A              [[buffer(0)]],
    device const uchar *B_packed      [[buffer(1)]],
    device const half *scales         [[buffer(2)]],
    device const half *zeros          [[buffer(3)]],
    device half *C                    [[buffer(4)]],
    constant uint &N                  [[buffer(5)]],
    constant uint &K                  [[buffer(6)]],
    constant uint &group_size         [[buffer(7)]],
    device const half *awq_scales     [[buffer(8)]],
    constant uint &has_awq            [[buffer(9)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint sgid   [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]])
{
    uint row_base = tgid * AMX_ROWS_PER_TG;
    if (row_base >= N) return;

    uint rows_this_tg = min(AMX_ROWS_PER_TG, N - row_base);
    uint n_block_base = row_base / BLK_N;
    uint k_blocks_total = (K + BLK_K - 1) / BLK_K;
    uint num_groups = (K + group_size - 1) / group_size;

    threadgroup half tg_w[AMX_ROWS_PER_TG * AMX_TILE_K]; // 16 KB
    threadgroup half tg_x[AMX_TILE_K];                    // 256 B

    simdgroup_matrix<float, 8, 8> acc_mat(0);

    for (uint kt = 0; kt < K; kt += AMX_TILE_K) {
        uint tile_k = min(AMX_TILE_K, K - kt);

        // -- Phase 1a: Load x into threadgroup memory --
        for (uint i = tid; i < tile_k; i += AMX_TG_SIZE) {
            float val = float(A[kt + i]);
            if (has_awq) val /= float(awq_scales[kt + i]);
            tg_x[i] = half(val);
        }

        // -- Phase 1b: Cooperative INT4 dequant --
        uint half_tile_k = tile_k / 2;
        uint total_pairs = rows_this_tg * half_tile_k;
        for (uint i = tid; i < total_pairs; i += AMX_TG_SIZE) {
            uint n_local = i / half_tile_k;
            uint kp      = i % half_tile_k;
            uint k_abs   = kt + kp * 2;
            uint n_abs   = row_base + n_local;

            uint kb_idx   = k_abs / BLK_K;
            uint k_local  = k_abs % BLK_K;
            uint byte_idx = (n_block_base * k_blocks_total + kb_idx) * (BLK_N * BLK_K / 2)
                          + n_local * (BLK_K / 2)
                          + k_local / 2;

            uchar packed = B_packed[byte_idx];

            uint g0 = k_abs / group_size;
            float s0 = float(scales[n_abs * num_groups + g0]);
            float z0 = float(zeros[n_abs * num_groups + g0]);
            float w0 = (float(packed & 0x0F) - z0) * s0;

            uint g1 = (k_abs + 1) / group_size;
            float s1 = (g1 == g0) ? s0 : float(scales[n_abs * num_groups + g1]);
            float z1 = (g1 == g0) ? z0 : float(zeros[n_abs * num_groups + g1]);
            float w1 = (float(packed >> 4) - z1) * s1;

            tg_w[n_local * AMX_TILE_K + kp * 2]     = half(w0);
            tg_w[n_local * AMX_TILE_K + kp * 2 + 1] = half(w1);
        }
        if (tile_k < AMX_TILE_K) {
            for (uint i = tid; i < rows_this_tg; i += AMX_TG_SIZE) {
                for (uint j = tile_k; j < AMX_TILE_K; j++) {
                    tg_w[i * AMX_TILE_K + j] = 0;
                }
            }
            for (uint i = tile_k + tid; i < AMX_TILE_K; i += AMX_TG_SIZE) {
                tg_x[i] = 0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -- Phase 2: simdgroup matrix multiply (AMX) --
        uint sg_row = sgid * AMX_ROWS_PER_SG;
        if (sg_row < rows_this_tg) {
            uint n_k_tiles = (tile_k + 7) / 8;
            for (uint kb = 0; kb < n_k_tiles; kb++) {
                simdgroup_matrix<half, 8, 8> w_T;
                simdgroup_load(w_T, tg_w + sg_row * AMX_TILE_K + kb * 8,
                               AMX_TILE_K, ulong2(0, 0), true);

                simdgroup_matrix<half, 8, 8> x_mat;
                simdgroup_load(x_mat, tg_x + kb * 8, 0);

                simdgroup_multiply_accumulate(acc_mat, x_mat, w_T, acc_mat);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // -- Extract and store results --
    uint sg_row = sgid * AMX_ROWS_PER_SG;
    if (sg_row >= rows_this_tg) return;

    threadgroup float tg_result[AMX_SIMDGROUPS * 64];
    simdgroup_store(acc_mat, tg_result + sgid * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        for (uint r = 0; r < AMX_ROWS_PER_SG; r++) {
            uint n_row = row_base + sg_row + r;
            if (n_row < N) {
                C[n_row] = half(tg_result[sgid * 64 + r]);
            }
        }
    }
}


// ============================================================================
// AMX-accelerated INT8 matvec (decode path, M=1)
//
// Same two-phase approach as affine_matvec_int4_amx but for INT8 weights.
// Phase 1: All 256 threads cooperatively dequant a [64 x TILE_K] INT8 tile.
// Phase 2: 8 simdgroups use simdgroup_matrix_multiply_accumulate (AMX).
//
// INT8 difference: each byte is one element (vs INT4 where each byte is two).
// Blocked layout: [N/BLK_N, K/BLK_K, BLK_N, BLK_K]
//
// Same buffer layout as affine_matvec_int8 for drop-in replacement.
// Dispatch: (ceil(N/64), 1, 1) threadgroups, (256, 1, 1) threads.
// ============================================================================

kernel void affine_matvec_int8_amx(
    device const half *A              [[buffer(0)]],
    device const uchar *B_packed      [[buffer(1)]],
    device const half *scales         [[buffer(2)]],
    device const half *zeros          [[buffer(3)]],
    device half *C                    [[buffer(4)]],
    constant uint &N                  [[buffer(5)]],
    constant uint &K                  [[buffer(6)]],
    constant uint &group_size         [[buffer(7)]],
    device const half *awq_scales     [[buffer(8)]],
    constant uint &has_awq            [[buffer(9)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint sgid   [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]])
{
    uint row_base = tgid * AMX_ROWS_PER_TG;
    if (row_base >= N) return;

    uint rows_this_tg = min(AMX_ROWS_PER_TG, N - row_base);
    uint n_block_base = row_base / BLK_N;
    uint k_blocks_total = (K + BLK_K - 1) / BLK_K;
    uint num_groups = (K + group_size - 1) / group_size;

    threadgroup half tg_w[AMX_ROWS_PER_TG * AMX_TILE_K]; // 16 KB
    threadgroup half tg_x[AMX_TILE_K];                    // 256 B

    simdgroup_matrix<float, 8, 8> acc_mat(0);

    for (uint kt = 0; kt < K; kt += AMX_TILE_K) {
        uint tile_k = min(AMX_TILE_K, K - kt);

        // -- Phase 1a: Load x into threadgroup memory --
        for (uint i = tid; i < tile_k; i += AMX_TG_SIZE) {
            float val = float(A[kt + i]);
            if (has_awq) val /= float(awq_scales[kt + i]);
            tg_x[i] = half(val);
        }

        // -- Phase 1b: Cooperative INT8 dequant --
        uint total_elems = rows_this_tg * tile_k;
        for (uint i = tid; i < total_elems; i += AMX_TG_SIZE) {
            uint n_local = i / tile_k;
            uint k_elem  = i % tile_k;
            uint k_abs   = kt + k_elem;
            uint n_abs   = row_base + n_local;

            uint kb_idx   = k_abs / BLK_K;
            uint k_local  = k_abs % BLK_K;
            uint byte_idx = (n_block_base * k_blocks_total + kb_idx) * (BLK_N * BLK_K)
                          + n_local * BLK_K
                          + k_local;

            uchar q = B_packed[byte_idx];
            uint grp = k_abs / group_size;
            float s = float(scales[n_abs * num_groups + grp]);
            float z = float(zeros[n_abs * num_groups + grp]);
            float w = (float(q) - z) * s;

            tg_w[n_local * AMX_TILE_K + k_elem] = half(w);
        }
        if (tile_k < AMX_TILE_K) {
            for (uint i = tid; i < rows_this_tg; i += AMX_TG_SIZE) {
                for (uint j = tile_k; j < AMX_TILE_K; j++) {
                    tg_w[i * AMX_TILE_K + j] = 0;
                }
            }
            for (uint i = tile_k + tid; i < AMX_TILE_K; i += AMX_TG_SIZE) {
                tg_x[i] = 0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -- Phase 2: simdgroup matrix multiply (AMX) --
        uint sg_row = sgid * AMX_ROWS_PER_SG;
        if (sg_row < rows_this_tg) {
            uint n_k_tiles = (tile_k + 7) / 8;
            for (uint kb = 0; kb < n_k_tiles; kb++) {
                simdgroup_matrix<half, 8, 8> w_T;
                simdgroup_load(w_T, tg_w + sg_row * AMX_TILE_K + kb * 8,
                               AMX_TILE_K, ulong2(0, 0), true);

                simdgroup_matrix<half, 8, 8> x_mat;
                simdgroup_load(x_mat, tg_x + kb * 8, 0);

                simdgroup_multiply_accumulate(acc_mat, x_mat, w_T, acc_mat);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // -- Extract and store results --
    uint sg_row = sgid * AMX_ROWS_PER_SG;
    if (sg_row >= rows_this_tg) return;

    threadgroup float tg_result[AMX_SIMDGROUPS * 64];
    simdgroup_store(acc_mat, tg_result + sgid * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        for (uint r = 0; r < AMX_ROWS_PER_SG; r++) {
            uint n_row = row_base + sg_row + r;
            if (n_row < N) {
                C[n_row] = half(tg_result[sgid * 64 + r]);
            }
        }
    }
}


// ============================================================================
// Fused FFN gate+up+activation for INT4 decode using AMX (M=1)
//
// Computes output[i] = act(x · W_gate^T[i]) * (x · W_up^T[i]) in one dispatch.
// Each TG processes 64 output rows. Two weight dequant phases per K-tile
// (gate + up) with two sets of MMA accumulators.
//
// Uses AMX_TILE_K=64 to fit both gate+up tiles in threadgroup memory (16KB).
// Dispatch: (ceil(N/64), 1, 1) threadgroups, (256, 1, 1) threads.
// ============================================================================

constant constexpr uint FUSED_FFN_TILE_K = 64;

kernel void fused_ffn_gate_up_act_int4(
    device const half *A              [[buffer(0)]],    // [1, K] shared input
    device const uchar *B_gate_packed [[buffer(1)]],    // gate weights (blocked)
    device const half *scales_gate    [[buffer(2)]],
    device const half *zeros_gate     [[buffer(3)]],
    device const uchar *B_up_packed   [[buffer(4)]],    // up weights (blocked)
    device const half *scales_up      [[buffer(5)]],
    device const half *zeros_up       [[buffer(6)]],
    device half *C                    [[buffer(7)]],    // [1, N] fused output
    constant uint &N                  [[buffer(8)]],
    constant uint &K                  [[buffer(9)]],
    constant uint &group_size         [[buffer(10)]],
    device const half *awq_scales     [[buffer(11)]],
    constant uint &has_awq            [[buffer(12)]],
    constant uint &use_gelu           [[buffer(13)]],   // 0=SiLU, 1=GELU-tanh
    uint tgid   [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint sgid   [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]])
{
    uint row_base = tgid * AMX_ROWS_PER_TG;
    if (row_base >= N) return;

    uint rows_this_tg = min(AMX_ROWS_PER_TG, N - row_base);
    uint n_block_base = row_base / BLK_N;
    uint k_blocks_total = (K + BLK_K - 1) / BLK_K;
    uint num_groups = (K + group_size - 1) / group_size;

    // Two weight tiles + one x tile, using FUSED_FFN_TILE_K to fit in 16KB
    threadgroup half tg_w_gate[AMX_ROWS_PER_TG * FUSED_FFN_TILE_K]; // 8 KB
    threadgroup half tg_w_up[AMX_ROWS_PER_TG * FUSED_FFN_TILE_K];   // 8 KB
    threadgroup half tg_x[FUSED_FFN_TILE_K];                          // 128 B

    simdgroup_matrix<float, 8, 8> gate_acc(0);
    simdgroup_matrix<float, 8, 8> up_acc(0);

    for (uint kt = 0; kt < K; kt += FUSED_FFN_TILE_K) {
        uint tile_k = min(FUSED_FFN_TILE_K, K - kt);

        // Load x
        for (uint i = tid; i < tile_k; i += AMX_TG_SIZE) {
            float val = float(A[kt + i]);
            if (has_awq) val /= float(awq_scales[kt + i]);
            tg_x[i] = half(val);
        }

        // Cooperative INT4 dequant for both gate and up weights
        uint half_tile_k = tile_k / 2;
        uint total_pairs = rows_this_tg * half_tile_k;
        for (uint i = tid; i < total_pairs; i += AMX_TG_SIZE) {
            uint n_local = i / half_tile_k;
            uint kp      = i % half_tile_k;
            uint k_abs   = kt + kp * 2;
            uint n_abs   = row_base + n_local;

            uint kb_idx   = k_abs / BLK_K;
            uint k_local  = k_abs % BLK_K;
            uint byte_idx = (n_block_base * k_blocks_total + kb_idx) * (BLK_N * BLK_K / 2)
                          + n_local * (BLK_K / 2)
                          + k_local / 2;

            // Gate weights
            uchar g_packed = B_gate_packed[byte_idx];
            uint g0 = k_abs / group_size;
            float sg0 = float(scales_gate[n_abs * num_groups + g0]);
            float zg0 = float(zeros_gate[n_abs * num_groups + g0]);
            float gw0 = (float(g_packed & 0x0F) - zg0) * sg0;

            uint g1 = (k_abs + 1) / group_size;
            float sg1 = (g1 == g0) ? sg0 : float(scales_gate[n_abs * num_groups + g1]);
            float zg1 = (g1 == g0) ? zg0 : float(zeros_gate[n_abs * num_groups + g1]);
            float gw1 = (float(g_packed >> 4) - zg1) * sg1;

            tg_w_gate[n_local * FUSED_FFN_TILE_K + kp * 2]     = half(gw0);
            tg_w_gate[n_local * FUSED_FFN_TILE_K + kp * 2 + 1] = half(gw1);

            // Up weights
            uchar u_packed = B_up_packed[byte_idx];
            float su0 = float(scales_up[n_abs * num_groups + g0]);
            float zu0 = float(zeros_up[n_abs * num_groups + g0]);
            float uw0 = (float(u_packed & 0x0F) - zu0) * su0;

            float su1 = (g1 == g0) ? su0 : float(scales_up[n_abs * num_groups + g1]);
            float zu1 = (g1 == g0) ? zu0 : float(zeros_up[n_abs * num_groups + g1]);
            float uw1 = (float(u_packed >> 4) - zu1) * su1;

            tg_w_up[n_local * FUSED_FFN_TILE_K + kp * 2]     = half(uw0);
            tg_w_up[n_local * FUSED_FFN_TILE_K + kp * 2 + 1] = half(uw1);
        }
        if (tile_k < FUSED_FFN_TILE_K) {
            for (uint i = tid; i < rows_this_tg; i += AMX_TG_SIZE) {
                for (uint j = tile_k; j < FUSED_FFN_TILE_K; j++) {
                    tg_w_gate[i * FUSED_FFN_TILE_K + j] = 0;
                    tg_w_up[i * FUSED_FFN_TILE_K + j] = 0;
                }
            }
            for (uint i = tile_k + tid; i < FUSED_FFN_TILE_K; i += AMX_TG_SIZE) {
                tg_x[i] = 0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: simdgroup MMA for both gate and up
        uint sg_row = sgid * AMX_ROWS_PER_SG;
        if (sg_row < rows_this_tg) {
            uint n_k_tiles = (tile_k + 7) / 8;
            for (uint kb = 0; kb < n_k_tiles; kb++) {
                simdgroup_matrix<half, 8, 8> x_mat;
                simdgroup_load(x_mat, tg_x + kb * 8, 0);

                simdgroup_matrix<half, 8, 8> wg_T;
                simdgroup_load(wg_T, tg_w_gate + sg_row * FUSED_FFN_TILE_K + kb * 8,
                               FUSED_FFN_TILE_K, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(gate_acc, x_mat, wg_T, gate_acc);

                simdgroup_matrix<half, 8, 8> wu_T;
                simdgroup_load(wu_T, tg_w_up + sg_row * FUSED_FFN_TILE_K + kb * 8,
                               FUSED_FFN_TILE_K, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(up_acc, x_mat, wu_T, up_acc);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Extract results and apply activation
    uint sg_row = sgid * AMX_ROWS_PER_SG;
    if (sg_row >= rows_this_tg) return;

    threadgroup float tg_gate_result[AMX_SIMDGROUPS * 64];
    threadgroup float tg_up_result[AMX_SIMDGROUPS * 64];
    simdgroup_store(gate_acc, tg_gate_result + sgid * 64, 8);
    simdgroup_store(up_acc, tg_up_result + sgid * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        for (uint r = 0; r < AMX_ROWS_PER_SG; r++) {
            uint n_row = row_base + sg_row + r;
            if (n_row < N) {
                float gate_val = tg_gate_result[sgid * 64 + r];
                float up_val = tg_up_result[sgid * 64 + r];
                float act;
                if (use_gelu) {
                    const float kSqrt2OverPi = 0.7978845608f;
                    float inner = kSqrt2OverPi * (gate_val + 0.044715f * gate_val * gate_val * gate_val);
                    inner = clamp(inner, -10.0f, 10.0f);
                    act = 0.5f * gate_val * (1.0f + precise::tanh(inner));
                } else {
                    act = gate_val / (1.0f + exp(-gate_val));
                }
                C[n_row] = half(act * up_val);
            }
        }
    }
}

// ── INT4×Q8 integer dot product matvec (decode path, M=1) ───────
//
// Uses pre-quantized INT8 input (from quantize_input_q8) to replace float
// dequant with integer multiply-add. The per-group Q8 scale is combined
// with the weight scale in the final reduction.
//
// B_packed is in blocked layout: [N_blocks, K_blocks, BLK_N, BLK_K/2]
// (same as affine_matvec_int4).
//
// Dispatch: (N, 1, 1) threadgroups, (32, 1, 1) threads per group.

// ============================================================================
// AMX-accelerated INT4×Q8 matvec (decode path, M=1)
//
// Two-phase approach per K-tile (same as affine_matvec_int4_amx):
//   Phase 1a: Dequant Q8 input to FP16 in threadgroup memory.
//   Phase 1b: All 256 threads cooperatively dequant a [64 x TILE_K] INT4 tile
//             into FP16 threadgroup memory (zero is integer-rounded).
//   Phase 2: Each of 8 simdgroups uses simdgroup_matrix_multiply_accumulate
//            (Apple AMX hardware) on its 8-row slice.
//
// Same buffer layout as the scalar affine_matvec_int4xq8 for drop-in replacement.
// Dispatch: (ceil(N/64), 1, 1) threadgroups, (256, 1, 1) threads.
// ============================================================================

kernel void affine_matvec_int4xq8(
    device const char *A_q8         [[buffer(0)]],   // [K] int8
    device const float *A_scales    [[buffer(1)]],   // [K/q8_group_size] float
    device const uchar *B_packed    [[buffer(2)]],   // blocked [N_blk, K_blk, 64, 4]
    device const half *w_scales     [[buffer(3)]],   // [N, num_groups]
    device const half *w_zeros      [[buffer(4)]],   // [N, num_groups]
    device half *C                  [[buffer(5)]],   // [1, N]
    constant uint &N                [[buffer(6)]],
    constant uint &K                [[buffer(7)]],
    constant uint &group_size       [[buffer(8)]],   // weight group size
    constant uint &q8_group_size    [[buffer(9)]],   // Q8 input group size
    uint tgid   [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint sgid   [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]])
{
    uint row_base = tgid * AMX_ROWS_PER_TG;
    if (row_base >= N) return;

    uint rows_this_tg = min(AMX_ROWS_PER_TG, N - row_base);
    uint n_block_base = row_base / BLK_N;
    uint k_blocks_total = (K + BLK_K - 1) / BLK_K;
    uint num_groups = (K + group_size - 1) / group_size;

    threadgroup half tg_w[AMX_ROWS_PER_TG * AMX_TILE_K]; // 16 KB
    threadgroup half tg_x[AMX_TILE_K];                    // 256 B

    simdgroup_matrix<float, 8, 8> acc_mat(0);

    for (uint kt = 0; kt < K; kt += AMX_TILE_K) {
        uint tile_k = min(AMX_TILE_K, K - kt);

        // -- Phase 1a: Load Q8 input into threadgroup FP16 --
        for (uint i = tid; i < tile_k; i += AMX_TG_SIZE) {
            uint k_abs = kt + i;
            float q8_val = float(A_q8[k_abs]);
            float a_scale = A_scales[k_abs / q8_group_size];
            tg_x[i] = half(q8_val * a_scale);
        }

        // -- Phase 1b: Cooperative INT4 dequant (integer-rounded zero) --
        uint half_tile_k = tile_k / 2;
        uint total_pairs = rows_this_tg * half_tile_k;
        for (uint i = tid; i < total_pairs; i += AMX_TG_SIZE) {
            uint n_local = i / half_tile_k;
            uint kp      = i % half_tile_k;
            uint k_abs   = kt + kp * 2;
            uint n_abs   = row_base + n_local;

            uint kb_idx   = k_abs / BLK_K;
            uint k_local  = k_abs % BLK_K;
            uint byte_idx = (n_block_base * k_blocks_total + kb_idx) * (BLK_N * BLK_K / 2)
                          + n_local * (BLK_K / 2)
                          + k_local / 2;

            uchar packed = B_packed[byte_idx];

            uint g0 = k_abs / group_size;
            float s0 = float(w_scales[n_abs * num_groups + g0]);
            int iz0 = int(rint(float(w_zeros[n_abs * num_groups + g0])));
            float w0 = float(int(packed & 0x0F) - iz0) * s0;

            uint g1 = (k_abs + 1) / group_size;
            float s1 = (g1 == g0) ? s0 : float(w_scales[n_abs * num_groups + g1]);
            int iz1 = (g1 == g0) ? iz0 : int(rint(float(w_zeros[n_abs * num_groups + g1])));
            float w1 = float(int(packed >> 4) - iz1) * s1;

            tg_w[n_local * AMX_TILE_K + kp * 2]     = half(w0);
            tg_w[n_local * AMX_TILE_K + kp * 2 + 1] = half(w1);
        }
        if (tile_k < AMX_TILE_K) {
            for (uint i = tid; i < rows_this_tg; i += AMX_TG_SIZE) {
                for (uint j = tile_k; j < AMX_TILE_K; j++) {
                    tg_w[i * AMX_TILE_K + j] = 0;
                }
            }
            for (uint i = tile_k + tid; i < AMX_TILE_K; i += AMX_TG_SIZE) {
                tg_x[i] = 0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -- Phase 2: simdgroup matrix multiply (AMX) --
        uint sg_row = sgid * AMX_ROWS_PER_SG;
        if (sg_row < rows_this_tg) {
            uint n_k_tiles = (tile_k + 7) / 8;
            for (uint kb = 0; kb < n_k_tiles; kb++) {
                simdgroup_matrix<half, 8, 8> w_T;
                simdgroup_load(w_T, tg_w + sg_row * AMX_TILE_K + kb * 8,
                               AMX_TILE_K, ulong2(0, 0), true);

                simdgroup_matrix<half, 8, 8> x_mat;
                simdgroup_load(x_mat, tg_x + kb * 8, 0);

                simdgroup_multiply_accumulate(acc_mat, x_mat, w_T, acc_mat);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // -- Extract and store results --
    uint sg_row = sgid * AMX_ROWS_PER_SG;
    if (sg_row >= rows_this_tg) return;

    threadgroup float tg_result[AMX_SIMDGROUPS * 64];
    simdgroup_store(acc_mat, tg_result + sgid * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        for (uint r = 0; r < AMX_ROWS_PER_SG; r++) {
            uint n_row = row_base + sg_row + r;
            if (n_row < N) {
                C[n_row] = half(tg_result[sgid * 64 + r]);
            }
        }
    }
}
