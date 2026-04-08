#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ============================================================================
// D2Quant 3-bit fused dequant + matmul
//
// Weights stored as packed 3-bit with dual-scale per-group parameters.
// Data layout: row-major (NOT blocked).
//
// Dequantization:
//   if mask[i] == 1 (outlier): w = (q - outlier_zero) * outlier_scale
//   else (normal):             w = (q - normal_zero)  * normal_scale
//
// Packing: 8 values → 3 bytes (24 bits), LSB-first.
//   group = i/8, offset = i%8
//   load 3 bytes at group*3, form 24-bit word, shift by offset*3, mask 0x07
//
// Two paths:
//   - matvec (M=1): one threadgroup per output row, SIMD reduction
//   - matmul (M>1): tiled GEMM with threadgroup-shared tiles
// ============================================================================

// ── D2Quant matvec (decode path, M=1) ───────────────────────────
//
// One threadgroup per output row. Each of 32 lanes processes K/32
// elements, unpacks 3-bit values, applies dual-scale dequant, and
// dot-products with A.
//
// Dispatch: (N, 1, 1) threadgroups, (32, 1, 1) threads per group.

kernel void d2quant_matvec_3bit(
    device const half *A              [[buffer(0)]],   // [1, K]
    device const uchar *B_packed      [[buffer(1)]],   // [N, ceil(K/8)*3] row-major
    device const float *normal_scale  [[buffer(2)]],   // [N, num_groups]
    device const float *normal_zero   [[buffer(3)]],   // [N, num_groups]
    device const float *outlier_scale [[buffer(4)]],   // [N, num_groups]
    device const float *outlier_zero  [[buffer(5)]],   // [N, num_groups]
    device const uchar *outlier_mask  [[buffer(6)]],   // [N, ceil(K/8)]
    device half *C                    [[buffer(7)]],   // [1, N]
    constant uint &N                  [[buffer(8)]],
    constant uint &K                  [[buffer(9)]],
    constant uint &group_size         [[buffer(10)]],
    uint tid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (tid >= N) return;

    uint num_groups = (K + group_size - 1) / group_size;
    uint bytes_per_row = ((K + 7) / 8) * 3;
    uint mask_bytes_per_row = (K + 7) / 8;

    uint data_row = tid * bytes_per_row;
    uint mask_row = tid * mask_bytes_per_row;
    uint scale_row = tid * num_groups;

    float acc = 0.0f;

    for (uint elem = lane; elem < K; elem += 32) {
        // Unpack 3-bit value
        uint group_of_8 = elem / 8;
        uint offset = elem % 8;
        uint byte_offset = data_row + group_of_8 * 3;
        uint bits = uint(B_packed[byte_offset])
                  | (uint(B_packed[byte_offset + 1]) << 8)
                  | (uint(B_packed[byte_offset + 2]) << 16);
        float q = float((bits >> (offset * 3)) & 0x07);

        // Read mask bit
        bool is_outlier = (outlier_mask[mask_row + elem / 8] >> (elem % 8)) & 1;

        // Select scale/zero
        uint grp = elem / group_size;
        float s = is_outlier ? outlier_scale[scale_row + grp] : normal_scale[scale_row + grp];
        float z = is_outlier ? outlier_zero[scale_row + grp]  : normal_zero[scale_row + grp];

        float w = (q - z) * s;
        acc += float(A[elem]) * w;
    }

    acc = simd_sum(acc);
    if (lane == 0) {
        C[tid] = half(acc);
    }
}

// ── Matmul tuning parameters (shared naming across all formats) ──

constant constexpr uint N_SIMDGROUPS   = 8;
constant constexpr uint THREADS_PER_TG = N_SIMDGROUPS * 32;
constant constexpr uint TM_TILE        = N_SIMDGROUPS * 8;   // 64
constant constexpr uint TN_TILE        = 64;
constant constexpr uint TN_STRIDE      = TN_TILE + 1;
constant constexpr uint MATMUL_K_TILE  = 32;
constant constexpr uint K_BLOCKS       = MATMUL_K_TILE / 8;  // 4 MMA ops per K-tile
constant constexpr uint TN_BLOCKS      = TN_TILE / 8;

// ── D2Quant tiled GEMM (prefill path, M>1) ──────────────────────
//
// Uses simdgroup_matrix_multiply_accumulate for hardware-accelerated 8×8
// matrix multiply. 256 threads = 8 simdgroups, each handling 8 output rows.
//
// B data is row-major packed 3-bit with dual-scale dequant.
//
// Dispatch: (ceil(M/TM_TILE), ceil(N/TN_TILE), 1) threadgroups,
//           (256, 1, 1) threads per group.

kernel void d2quant_matmul_3bit(
    device const half *A              [[buffer(0)]],   // [M, K]
    device const uchar *B_packed      [[buffer(1)]],   // [N, ceil(K/8)*3] row-major
    device const float *normal_scale  [[buffer(2)]],   // [N, num_groups]
    device const float *normal_zero   [[buffer(3)]],   // [N, num_groups]
    device const float *outlier_scale [[buffer(4)]],   // [N, num_groups]
    device const float *outlier_zero  [[buffer(5)]],   // [N, num_groups]
    device const uchar *outlier_mask  [[buffer(6)]],   // [N, ceil(K/8)]
    device half *C                    [[buffer(7)]],   // [M, N]
    constant uint &M                  [[buffer(8)]],
    constant uint &N                  [[buffer(9)]],
    constant uint &K                  [[buffer(10)]],
    constant uint &group_size         [[buffer(11)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    uint tg_m = group_id.x * TM_TILE;
    uint tg_n = group_id.y * TN_TILE;

    threadgroup half tg_a[2][TM_TILE * MATMUL_K_TILE];
    threadgroup half tg_bt[2][MATMUL_K_TILE * TN_STRIDE];

    uint num_groups_k = (K + group_size - 1) / group_size;
    uint num_k_steps = (K + MATMUL_K_TILE - 1) / MATMUL_K_TILE;
    uint bytes_per_row = ((K + 7) / 8) * 3;
    uint mask_bytes_per_row = (K + 7) / 8;

    simdgroup_matrix<float, 8, 8> acc[TN_BLOCKS];
    for (uint j = 0; j < TN_BLOCKS; j++) acc[j] = simdgroup_matrix<float, 8, 8>(0);

    // Prologue: load first tile into buf[0]
    {
        uint k_base = 0;
        // Load A tile
        for (uint i = tid; i < TM_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
            uint m = i / MATMUL_K_TILE;
            uint k = i % MATMUL_K_TILE;
            uint g_row = tg_m + m;
            uint g_col = k_base + k;
            half a_val = (g_row < M && g_col < K) ? A[g_row * K + g_col] : half(0);
            tg_a[0][i] = a_val;
        }
        // Load B tile (3-bit unpack + dual-scale dequant)
        for (uint i = tid; i < TN_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
            uint n = i / MATMUL_K_TILE;
            uint k = i % MATMUL_K_TILE;
            uint g_n = tg_n + n;
            uint g_k = k_base + k;
            half val = half(0);
            if (g_n < N && g_k < K) {
                uint group_of_8 = g_k / 8;
                uint offset = g_k % 8;
                uint byte_off = g_n * bytes_per_row + group_of_8 * 3;
                uint bits = uint(B_packed[byte_off])
                          | (uint(B_packed[byte_off + 1]) << 8)
                          | (uint(B_packed[byte_off + 2]) << 16);
                float q = float((bits >> (offset * 3)) & 0x07);
                bool is_out = (outlier_mask[g_n * mask_bytes_per_row + g_k / 8] >> (g_k % 8)) & 1;
                uint grp = g_k / group_size;
                float s = is_out ? outlier_scale[g_n * num_groups_k + grp] : normal_scale[g_n * num_groups_k + grp];
                float z = is_out ? outlier_zero[g_n * num_groups_k + grp]  : normal_zero[g_n * num_groups_k + grp];
                val = half((q - z) * s);
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
            // Load A tile
            for (uint i = tid; i < TM_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
                uint m = i / MATMUL_K_TILE;
                uint k = i % MATMUL_K_TILE;
                uint g_row = tg_m + m;
                uint g_col = k_base + k;
                half a_val = (g_row < M && g_col < K) ? A[g_row * K + g_col] : half(0);
                tg_a[nxt][i] = a_val;
            }
            // Load B tile (3-bit unpack + dual-scale dequant)
            for (uint i = tid; i < TN_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
                uint n = i / MATMUL_K_TILE;
                uint k = i % MATMUL_K_TILE;
                uint g_n = tg_n + n;
                uint g_k = k_base + k;
                half val = half(0);
                if (g_n < N && g_k < K) {
                    uint group_of_8 = g_k / 8;
                    uint offset = g_k % 8;
                    uint byte_off = g_n * bytes_per_row + group_of_8 * 3;
                    uint bits = uint(B_packed[byte_off])
                              | (uint(B_packed[byte_off + 1]) << 8)
                              | (uint(B_packed[byte_off + 2]) << 16);
                    float q = float((bits >> (offset * 3)) & 0x07);
                    bool is_out = (outlier_mask[g_n * mask_bytes_per_row + g_k / 8] >> (g_k % 8)) & 1;
                    uint grp = g_k / group_size;
                    float s = is_out ? outlier_scale[g_n * num_groups_k + grp] : normal_scale[g_n * num_groups_k + grp];
                    float z = is_out ? outlier_zero[g_n * num_groups_k + grp]  : normal_zero[g_n * num_groups_k + grp];
                    val = half((q - z) * s);
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

// ── D2Quant 3-bit embedding lookup ──────────────────────────────
//
// Gather rows from a D2Quant packed table and write FP16 output.
// Each thread processes one output element [token, col].
//
// Dispatch: (ceil(hidden/tg), token_count, 1) threadgroups, (tg, 1, 1) threads.

kernel void d2quant_embedding_lookup_3bit(
    device const uint *token_ids       [[buffer(0)]],    // [token_count]
    device const uchar *packed_table   [[buffer(1)]],    // [vocab, ceil(K/8)*3]
    device const float *normal_scale   [[buffer(2)]],    // [vocab, num_groups]
    device const float *normal_zero    [[buffer(3)]],    // [vocab, num_groups]
    device const float *outlier_scale  [[buffer(4)]],    // [vocab, num_groups]
    device const float *outlier_zero   [[buffer(5)]],    // [vocab, num_groups]
    device const uchar *outlier_mask   [[buffer(6)]],    // [vocab, ceil(K/8)]
    device half *output                [[buffer(7)]],    // [token_count, K]
    constant uint &hidden_size         [[buffer(8)]],    // K
    constant uint &token_count         [[buffer(9)]],
    constant uint &vocab_size          [[buffer(10)]],
    constant uint &group_size          [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint col = gid.x;
    uint tok = gid.y;
    if (col >= hidden_size || tok >= token_count) return;

    uint token_id = token_ids[tok];
    if (token_id >= vocab_size) {
        output[tok * hidden_size + col] = half(0.0f);
        return;
    }

    uint K = hidden_size;
    uint bytes_per_row = ((K + 7) / 8) * 3;
    uint mask_bytes_per_row = (K + 7) / 8;
    uint num_groups = (K + group_size - 1) / group_size;

    // Unpack 3-bit value at (token_id, col)
    uint group_of_8 = col / 8;
    uint offset = col % 8;
    uint byte_off = token_id * bytes_per_row + group_of_8 * 3;
    uint bits = uint(packed_table[byte_off])
              | (uint(packed_table[byte_off + 1]) << 8)
              | (uint(packed_table[byte_off + 2]) << 16);
    float q = float((bits >> (offset * 3)) & 0x07);

    // Read outlier mask bit
    bool is_outlier = (outlier_mask[token_id * mask_bytes_per_row + col / 8] >> (col % 8)) & 1;

    // Select scale/zero based on partition
    uint grp = col / group_size;
    uint param_idx = token_id * num_groups + grp;
    float s = is_outlier ? outlier_scale[param_idx] : normal_scale[param_idx];
    float z = is_outlier ? outlier_zero[param_idx]  : normal_zero[param_idx];

    output[tok * K + col] = half((q - z) * s);
}
