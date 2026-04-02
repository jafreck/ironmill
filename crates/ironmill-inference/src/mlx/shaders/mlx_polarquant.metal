// PolarQuant Metal kernels for MLX backend.
//
// Contains all PolarQuant kernel variants:
//   - polarquant_matvec       (INT4, M=1 decode path)
//   - polarquant_matmul       (INT4, M>1 prefill path)
//   - polarquant_matvec_int4  (INT4, M=1 decode path, explicit bit-width)
//   - polarquant_matmul_int4  (INT4, M>1 prefill path, explicit bit-width)
//   - polarquant_matvec_int8  (INT8, M=1 decode path)
//   - polarquant_matmul_int8  (INT8, M>1 prefill path)
//
// This file is standalone — no shared helpers are prepended.

#include <metal_stdlib>
using namespace metal;

// ── polarquant_matvec (INT4, decode) ────────────────────────────

[[kernel]] void polarquant_matvec(
    device const half *A            [[buffer(0)]],
    device const uchar *B_packed    [[buffer(1)]],
    constant half *lut              [[buffer(2)]],
    device const half *norms        [[buffer(3)]],
    device const uint *params       [[buffer(4)]],
    device half *C                  [[buffer(5)]],
    uint tid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint N = params[0];
    uint K = params[1];
    if (tid >= N) return;

    uint half_K = K / 2;
    uint row_offset = tid * half_K;
    float norm = float(norms[tid]);
    float acc = 0.0f;

    for (uint k = lane; k < half_K; k += 32) {
        uchar packed = B_packed[row_offset + k];
        uchar lo_idx = (packed >> 4) & 0xF;
        uchar hi_idx = packed & 0xF;
        float w0 = float(lut[lo_idx]) * norm;
        float w1 = float(lut[hi_idx]) * norm;
        uint k2 = k * 2;
        acc += float(A[k2])     * w0;
        acc += float(A[k2 + 1]) * w1;
    }

    acc = simd_sum(acc);
    if (lane == 0) {
        C[tid] = half(acc);
    }
}

// ── polarquant_matmul (INT4, prefill) ───────────────────────────

constant constexpr uint TILE_M_mm = 8;
constant constexpr uint TILE_N_mm = 32;
constant constexpr uint TILE_K_mm = 32;

[[kernel]] void polarquant_matmul(
    device const half *A            [[buffer(0)]],
    device const uchar *B_packed    [[buffer(1)]],
    constant half *lut              [[buffer(2)]],
    device const half *norms        [[buffer(3)]],
    device const uint *params       [[buffer(4)]],
    device half *C                  [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 lid  [[thread_position_in_threadgroup]])
{
    uint M = params[0];
    uint N = params[1];
    uint K = params[2];

    uint col = tgid.x * TILE_N_mm + lid.x;
    uint row = tgid.y * TILE_M_mm + lid.y;

    if (row >= M || col >= N) return;

    uint half_K = K / 2;
    float norm = float(norms[col]);
    float acc = 0.0f;

    for (uint k = 0; k < half_K; k++) {
        uchar packed = B_packed[col * half_K + k];
        uchar lo_idx = (packed >> 4) & 0xF;
        uchar hi_idx = packed & 0xF;
        float w0 = float(lut[lo_idx]) * norm;
        float w1 = float(lut[hi_idx]) * norm;
        uint k2 = k * 2;
        acc += float(A[row * K + k2])     * w0;
        acc += float(A[row * K + k2 + 1]) * w1;
    }

    C[row * N + col] = half(acc);
}

// ── polarquant_matvec_int4 (decode) ─────────────────────────────

[[kernel]] void polarquant_matvec_int4(
    device const half *A            [[buffer(0)]],
    device const uchar *B_packed    [[buffer(1)]],
    constant half *lut              [[buffer(2)]],
    device const half *norms        [[buffer(3)]],
    device const uint *params       [[buffer(4)]],
    device half *C                  [[buffer(5)]],
    uint tid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint N = params[0];
    uint K = params[1];
    if (tid >= N) return;

    uint half_K = K / 2;
    uint row_offset = tid * half_K;
    float norm = float(norms[tid]);
    float acc = 0.0f;

    for (uint k = lane; k < half_K; k += 32) {
        uchar packed = B_packed[row_offset + k];
        uchar lo_idx = (packed >> 4) & 0xF;
        uchar hi_idx = packed & 0xF;
        float w0 = float(lut[lo_idx]) * norm;
        float w1 = float(lut[hi_idx]) * norm;
        uint k2 = k * 2;
        acc += float(A[k2])     * w0;
        acc += float(A[k2 + 1]) * w1;
    }

    acc = simd_sum(acc);
    if (lane == 0) C[tid] = half(acc);
}

// ── polarquant_matmul_int4 (prefill) ────────────────────────────

constant constexpr uint TILE_M_i4 = 8;
constant constexpr uint TILE_N_i4 = 32;
constant constexpr uint TILE_K_i4 = 32;

[[kernel]] void polarquant_matmul_int4(
    device const half *A            [[buffer(0)]],
    device const uchar *B_packed    [[buffer(1)]],
    constant half *lut              [[buffer(2)]],
    device const half *norms        [[buffer(3)]],
    device const uint *params       [[buffer(4)]],
    device half *C                  [[buffer(5)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 local_id [[thread_position_in_threadgroup]])
{
    uint M = params[0];
    uint N = params[1];
    uint K = params[2];

    uint local_row = local_id.y;
    uint local_col = local_id.x;
    uint row = group_id.y * TILE_M_i4 + local_row;
    uint col = group_id.x * TILE_N_i4 + local_col;

    threadgroup half tg_a[TILE_M_i4 * TILE_K_i4];
    threadgroup half tg_b[TILE_N_i4 * TILE_K_i4];

    uint thread_idx   = local_row * TILE_N_i4 + local_col;
    uint total_threads = TILE_M_i4 * TILE_N_i4;
    uint half_K = K / 2;
    float acc = 0.0f;

    for (uint k_base = 0; k_base < K; k_base += TILE_K_i4) {
        for (uint i = thread_idx; i < TILE_M_i4 * TILE_K_i4; i += total_threads) {
            uint a_row = i / TILE_K_i4;
            uint a_col = i % TILE_K_i4;
            uint g_row = group_id.y * TILE_M_i4 + a_row;
            uint g_col = k_base + a_col;
            tg_a[i] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : half(0);
        }

        for (uint i = thread_idx; i < TILE_N_i4 * TILE_K_i4; i += total_threads) {
            uint b_n = i / TILE_K_i4;
            uint b_k = i % TILE_K_i4;
            uint g_n = group_id.x * TILE_N_i4 + b_n;
            uint g_k = k_base + b_k;
            half val = half(0);
            if (g_n < N && g_k < K) {
                uint byte_idx = g_n * half_K + g_k / 2;
                uchar packed  = B_packed[byte_idx];
                uchar idx     = (g_k % 2 == 0) ? ((packed >> 4) & 0xF) : (packed & 0xF);
                val = lut[idx] * norms[g_n];
            }
            tg_b[i] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < M && col < N) {
            uint a_base = local_row * TILE_K_i4;
            uint b_base = local_col * TILE_K_i4;
            uint k_end  = min(TILE_K_i4, K - k_base);
            for (uint k = 0; k < k_end; k++)
                acc += float(tg_a[a_base + k]) * float(tg_b[b_base + k]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) C[row * N + col] = half(acc);
}

// ── polarquant_matvec_int8 (decode) ─────────────────────────────

[[kernel]] void polarquant_matvec_int8(
    device const half *A            [[buffer(0)]],
    device const uchar *B_packed    [[buffer(1)]],
    constant half *lut              [[buffer(2)]],
    device const half *norms        [[buffer(3)]],
    device const uint *params       [[buffer(4)]],
    device half *C                  [[buffer(5)]],
    uint tid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint N = params[0];
    uint K = params[1];
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
    if (lane == 0) C[tid] = half(acc);
}

// ── polarquant_matmul_int8 (prefill) ────────────────────────────

constant constexpr uint TILE_M_i8 = 8;
constant constexpr uint TILE_N_i8 = 32;
constant constexpr uint TILE_K_i8 = 32;

[[kernel]] void polarquant_matmul_int8(
    device const half *A            [[buffer(0)]],
    device const uchar *B_packed    [[buffer(1)]],
    constant half *lut              [[buffer(2)]],
    device const half *norms        [[buffer(3)]],
    device const uint *params       [[buffer(4)]],
    device half *C                  [[buffer(5)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 local_id [[thread_position_in_threadgroup]])
{
    uint M = params[0];
    uint N = params[1];
    uint K = params[2];

    uint local_row = local_id.y;
    uint local_col = local_id.x;
    uint row = group_id.y * TILE_M_i8 + local_row;
    uint col = group_id.x * TILE_N_i8 + local_col;

    threadgroup half tg_a[TILE_M_i8 * TILE_K_i8];
    threadgroup half tg_b[TILE_N_i8 * TILE_K_i8];

    uint thread_idx    = local_row * TILE_N_i8 + local_col;
    uint total_threads = TILE_M_i8 * TILE_N_i8;
    float acc = 0.0f;

    for (uint k_base = 0; k_base < K; k_base += TILE_K_i8) {
        for (uint i = thread_idx; i < TILE_M_i8 * TILE_K_i8; i += total_threads) {
            uint a_row = i / TILE_K_i8;
            uint a_col = i % TILE_K_i8;
            uint g_row = group_id.y * TILE_M_i8 + a_row;
            uint g_col = k_base + a_col;
            tg_a[i] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : half(0);
        }

        for (uint i = thread_idx; i < TILE_N_i8 * TILE_K_i8; i += total_threads) {
            uint b_n = i / TILE_K_i8;
            uint b_k = i % TILE_K_i8;
            uint g_n = group_id.x * TILE_N_i8 + b_n;
            uint g_k = k_base + b_k;
            half val = half(0);
            if (g_n < N && g_k < K) {
                uchar idx = B_packed[g_n * K + g_k];
                val = lut[idx] * norms[g_n];
            }
            tg_b[i] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < M && col < N) {
            uint a_base = local_row * TILE_K_i8;
            uint b_base = local_col * TILE_K_i8;
            uint k_end  = min(TILE_K_i8, K - k_base);
            for (uint k = 0; k < k_end; k++)
                acc += float(tg_a[a_base + k]) * float(tg_b[b_base + k]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) C[row * N + col] = half(acc);
}
