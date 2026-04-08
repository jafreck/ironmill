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
