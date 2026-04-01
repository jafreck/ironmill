#include <metal_stdlib>
using namespace metal;

// ============================================================================
// PolarQuant: Quantized matmul kernels with LUT-based dequantization
//
// Weights stay packed in GPU memory; dequantization happens inline during
// the dot-product.  Two paths per bit-width:
//   - matvec (M=1): one threadgroup per output element, SIMD reduction
//   - matmul (M>1): tiled GEMM with threadgroup-shared tiles
// ============================================================================

constant constexpr uint TILE_M = 8;
constant constexpr uint TILE_N = 32;
constant constexpr uint TILE_K = 32;

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
        uchar lo_idx = packed & 0xF;
        uchar hi_idx = (packed >> 4) & 0xF;

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

// ── INT4 tiled GEMM (prefill path, M>1) ─────────────────────────
//
// Threadgroup size: (TILE_N, TILE_M, 1) = (32, 8, 1) = 256 threads.
// Each thread accumulates one element of the output tile.
//
// Per K-chunk the threadgroup cooperatively loads:
//   tg_a  [TILE_M × TILE_K]  — A activations (FP16)
//   tg_b  [TILE_N × TILE_K]  — B dequantized weights (FP16)
//
// Dispatch: (ceil(N/TILE_N), ceil(M/TILE_M), 1) threadgroups.

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
    uint2 local_id [[thread_position_in_threadgroup]])
{
    uint local_row = local_id.y;  // 0..TILE_M-1
    uint local_col = local_id.x;  // 0..TILE_N-1

    uint row = group_id.y * TILE_M + local_row;
    uint col = group_id.x * TILE_N + local_col;

    threadgroup half tg_a[TILE_M * TILE_K];
    threadgroup half tg_b[TILE_N * TILE_K];

    uint thread_idx   = local_row * TILE_N + local_col;
    uint total_threads = TILE_M * TILE_N;  // 256
    uint half_K = K / 2;

    float acc = 0.0f;

    for (uint k_base = 0; k_base < K; k_base += TILE_K) {
        // ---- Load A tile [TILE_M, TILE_K] ----
        for (uint i = thread_idx; i < TILE_M * TILE_K; i += total_threads) {
            uint a_row = i / TILE_K;
            uint a_col = i % TILE_K;
            uint g_row = group_id.y * TILE_M + a_row;
            uint g_col = k_base + a_col;
            tg_a[i] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : half(0);
        }

        // ---- Load & dequantize B tile [TILE_N, TILE_K] ----
        for (uint i = thread_idx; i < TILE_N * TILE_K; i += total_threads) {
            uint b_n = i / TILE_K;
            uint b_k = i % TILE_K;
            uint g_n = group_id.x * TILE_N + b_n;
            uint g_k = k_base + b_k;

            half val = half(0);
            if (g_n < N && g_k < K) {
                uint byte_idx = g_n * half_K + g_k / 2;
                uchar packed  = B_packed[byte_idx];
                uchar idx     = (g_k % 2 == 0) ? (packed & 0xF) : ((packed >> 4) & 0xF);
                val = lut[idx] * norms[g_n];
            }
            tg_b[i] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Accumulate dot product for this thread's output element ----
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
// Same tiling scheme as INT4 but B is [N, K] with one byte per element.
//
// Dispatch: (ceil(N/TILE_N), ceil(M/TILE_M), 1) threadgroups,
//           (TILE_N, TILE_M, 1) = (32, 8, 1) threads per group.

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

    float acc = 0.0f;

    for (uint k_base = 0; k_base < K; k_base += TILE_K) {
        // ---- Load A tile ----
        for (uint i = thread_idx; i < TILE_M * TILE_K; i += total_threads) {
            uint a_row = i / TILE_K;
            uint a_col = i % TILE_K;
            uint g_row = group_id.y * TILE_M + a_row;
            uint g_col = k_base + a_col;
            tg_a[i] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : half(0);
        }

        // ---- Load & dequantize B tile ----
        for (uint i = thread_idx; i < TILE_N * TILE_K; i += total_threads) {
            uint b_n = i / TILE_K;
            uint b_k = i % TILE_K;
            uint g_n = group_id.x * TILE_N + b_n;
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
