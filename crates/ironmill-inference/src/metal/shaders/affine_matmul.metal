#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Affine Quantized Matmul: Fused dequant + matmul for INT4/INT8
//
// Weights stay packed in GPU memory; affine dequantization happens inline
// during the dot-product:  w = (quantized - zero) * scale
//
// Two paths per bit-width:
//   - matvec (M=1): one threadgroup per output row, SIMD reduction
//   - matmul (M>1): tiled GEMM with threadgroup-shared tiles
// ============================================================================

constant constexpr uint TILE_M = 8;
constant constexpr uint TILE_N = 32;
constant constexpr uint TILE_K = 32;

// ── INT4 matvec (decode path, M=1) ──────────────────────────────
//
// One threadgroup per output row. Each lane processes K/(2·32) packed
// bytes, unpacks two nibbles, applies per-group affine dequant, and
// dot-products with A.
//
// Dispatch: (N, 1, 1) threadgroups, (32, 1, 1) threads per group.

kernel void affine_matvec_int4(
    device const half *A            [[buffer(0)]],   // [1, K]
    device const uchar *B_packed    [[buffer(1)]],   // [N, K/2]
    device const half *scales       [[buffer(2)]],   // [N, num_groups]
    device const half *zeros        [[buffer(3)]],   // [N, num_groups]
    device half *C                  [[buffer(4)]],   // [1, N]
    constant uint &N                [[buffer(5)]],
    constant uint &K                [[buffer(6)]],
    constant uint &group_size       [[buffer(7)]],
    uint tid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (tid >= N) return;

    uint half_K = K / 2;
    uint row_offset = tid * half_K;
    uint num_groups = (K + group_size - 1) / group_size;
    uint scale_row = tid * num_groups;

    float acc = 0.0f;

    for (uint k = lane; k < half_K; k += 32) {
        uchar packed = B_packed[row_offset + k];
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
// Dispatch: (ceil(N/TILE_N), ceil(M/TILE_M), 1) threadgroups,
//           (TILE_N, TILE_M, 1) threads per group.

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
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 local_id [[thread_position_in_threadgroup]])
{
    uint local_row = local_id.y;
    uint local_col = local_id.x;
    uint row = group_id.y * TILE_M + local_row;
    uint col = group_id.x * TILE_N + local_col;

    threadgroup half tg_a[TILE_M * TILE_K];
    threadgroup half tg_b[TILE_N * TILE_K];

    uint thread_idx   = local_row * TILE_N + local_col;
    uint total_threads = TILE_M * TILE_N;
    uint half_K = K / 2;
    uint num_groups = (K + group_size - 1) / group_size;

    float acc = 0.0f;

    for (uint k_base = 0; k_base < K; k_base += TILE_K) {
        // Load A tile
        for (uint i = thread_idx; i < TILE_M * TILE_K; i += total_threads) {
            uint a_row = i / TILE_K;
            uint a_col = i % TILE_K;
            uint g_row = group_id.y * TILE_M + a_row;
            uint g_col = k_base + a_col;
            tg_a[i] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : half(0);
        }

        // Load & dequantize B tile
        for (uint i = thread_idx; i < TILE_N * TILE_K; i += total_threads) {
            uint b_n = i / TILE_K;
            uint b_k = i % TILE_K;
            uint g_n = group_id.x * TILE_N + b_n;
            uint g_k = k_base + b_k;

            half val = half(0);
            if (g_n < N && g_k < K) {
                uint byte_idx = g_n * half_K + g_k / 2;
                uchar packed  = B_packed[byte_idx];
                uchar nibble  = (g_k % 2 == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);

                uint grp = g_k / group_size;
                float s = float(scales[g_n * num_groups + grp]);
                float z = float(zeros[g_n * num_groups + grp]);
                val = half((float(nibble) - z) * s);
            }
            tg_b[i] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < M && col < N) {
            uint a_base = local_row * TILE_K;
            uint b_base = local_col * TILE_K;
            uint k_end  = min(TILE_K, K - k_base);
            for (uint k = 0; k < k_end; k++) {
                acc += float(tg_a[a_base + k]) * float(tg_b[b_base + k]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = half(acc);
    }
}

// ── INT8 matvec (decode path, M=1) ──────────────────────────────
//
// One byte = one element (no nibble unpacking).
//
// Dispatch: (N, 1, 1) threadgroups, (32, 1, 1) threads per group.

kernel void affine_matvec_int8(
    device const half *A            [[buffer(0)]],   // [1, K]
    device const uchar *B_packed    [[buffer(1)]],   // [N, K]
    device const half *scales       [[buffer(2)]],   // [N, num_groups]
    device const half *zeros        [[buffer(3)]],   // [N, num_groups]
    device half *C                  [[buffer(4)]],   // [1, N]
    constant uint &N                [[buffer(5)]],
    constant uint &K                [[buffer(6)]],
    constant uint &group_size       [[buffer(7)]],
    uint tid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (tid >= N) return;

    uint row_offset = tid * K;
    uint num_groups = (K + group_size - 1) / group_size;
    uint scale_row = tid * num_groups;

    float acc = 0.0f;

    for (uint k = lane; k < K; k += 32) {
        uchar q = B_packed[row_offset + k];
        uint grp = k / group_size;
        float s = float(scales[scale_row + grp]);
        float z = float(zeros[scale_row + grp]);
        float w = (float(q) - z) * s;
        acc += float(A[k]) * w;
    }

    acc = simd_sum(acc);

    if (lane == 0) {
        C[tid] = half(acc);
    }
}

// ── INT8 tiled GEMM (prefill path, M>1) ─────────────────────────
//
// Dispatch: (ceil(N/TILE_N), ceil(M/TILE_M), 1) threadgroups,
//           (TILE_N, TILE_M, 1) threads per group.

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
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 local_id [[thread_position_in_threadgroup]])
{
    uint local_row = local_id.y;
    uint local_col = local_id.x;
    uint row = group_id.y * TILE_M + local_row;
    uint col = group_id.x * TILE_N + local_col;

    threadgroup half tg_a[TILE_M * TILE_K];
    threadgroup half tg_b[TILE_N * TILE_K];

    uint thread_idx    = local_row * TILE_N + local_col;
    uint total_threads = TILE_M * TILE_N;
    uint num_groups = (K + group_size - 1) / group_size;

    float acc = 0.0f;

    for (uint k_base = 0; k_base < K; k_base += TILE_K) {
        // Load A tile
        for (uint i = thread_idx; i < TILE_M * TILE_K; i += total_threads) {
            uint a_row = i / TILE_K;
            uint a_col = i % TILE_K;
            uint g_row = group_id.y * TILE_M + a_row;
            uint g_col = k_base + a_col;
            tg_a[i] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : half(0);
        }

        // Load & dequantize B tile
        for (uint i = thread_idx; i < TILE_N * TILE_K; i += total_threads) {
            uint b_n = i / TILE_K;
            uint b_k = i % TILE_K;
            uint g_n = group_id.x * TILE_N + b_n;
            uint g_k = k_base + b_k;

            half val = half(0);
            if (g_n < N && g_k < K) {
                uchar q = B_packed[g_n * K + g_k];
                uint grp = g_k / group_size;
                float s = float(scales[g_n * num_groups + grp]);
                float z = float(zeros[g_n * num_groups + grp]);
                val = half((float(q) - z) * s);
            }
            tg_b[i] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < M && col < N) {
            uint a_base = local_row * TILE_K;
            uint b_base = local_col * TILE_K;
            uint k_end  = min(TILE_K, K - k_base);
            for (uint k = 0; k < k_end; k++) {
                acc += float(tg_a[a_base + k]) * float(tg_b[b_base + k]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = half(acc);
    }
}
