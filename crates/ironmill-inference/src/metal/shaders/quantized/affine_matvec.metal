// ── INT4 matvec (decode path, M=1) ──────────────────────────────
//
// One threadgroup per output row. Each lane processes K/(8·32) uint32 words,
// unpacking 8 nibbles per load for 4× wider memory transactions.
//
// B_packed is in blocked layout: [N_blocks, K_blocks, BLK_N, BLK_K/2]
// produced by pack_quantized_blocked(). BLK_K=8 elements → 4 packed bytes
// = 1 uint32 per (n_local, k_block).
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

    uint num_groups = (K + group_size - 1) / group_size;
    uint scale_row = tid * num_groups;

    // Blocked layout: 1 uint32 per (n_local, k_block)
    uint k_blocks = (K + BLK_K - 1) / BLK_K;
    uint n_block = tid / BLK_N;
    uint n_local = tid % BLK_N;

    float acc = 0.0f;

    for (uint kb = lane; kb < k_blocks; kb += 32) {
        // Word-aligned load: 4 bytes = 8 nibbles
        uint word_idx = (n_block * k_blocks + kb) * BLK_N + n_local;
        uint packed4 = ((device const uint*)B_packed)[word_idx];

        // Pre-fetch scale/zero once per group (BLK_K=8 <= group_size)
        uint k_elem = kb * BLK_K;
        uint grp = k_elem / group_size;
        float s = float(scales[scale_row + grp]);
        float z = float(zeros[scale_row + grp]);

        // Unpack 8 nibbles and dequantize
        float w0 = (float(packed4 & 0xF) - z) * s;
        float w1 = (float((packed4 >> 4) & 0xF) - z) * s;
        float w2 = (float((packed4 >> 8) & 0xF) - z) * s;
        float w3 = (float((packed4 >> 12) & 0xF) - z) * s;
        float w4 = (float((packed4 >> 16) & 0xF) - z) * s;
        float w5 = (float((packed4 >> 20) & 0xF) - z) * s;
        float w6 = (float((packed4 >> 24) & 0xF) - z) * s;
        float w7 = (float((packed4 >> 28) & 0xF) - z) * s;

        if (has_awq) {
            acc += (float(A[k_elem])     / float(awq_scales[k_elem]))     * w0;
            acc += (float(A[k_elem + 1]) / float(awq_scales[k_elem + 1])) * w1;
            acc += (float(A[k_elem + 2]) / float(awq_scales[k_elem + 2])) * w2;
            acc += (float(A[k_elem + 3]) / float(awq_scales[k_elem + 3])) * w3;
            acc += (float(A[k_elem + 4]) / float(awq_scales[k_elem + 4])) * w4;
            acc += (float(A[k_elem + 5]) / float(awq_scales[k_elem + 5])) * w5;
            acc += (float(A[k_elem + 6]) / float(awq_scales[k_elem + 6])) * w6;
            acc += (float(A[k_elem + 7]) / float(awq_scales[k_elem + 7])) * w7;
        } else {
            acc += float(A[k_elem])     * w0 + float(A[k_elem + 1]) * w1
                 + float(A[k_elem + 2]) * w2 + float(A[k_elem + 3]) * w3
                 + float(A[k_elem + 4]) * w4 + float(A[k_elem + 5]) * w5
                 + float(A[k_elem + 6]) * w6 + float(A[k_elem + 7]) * w7;
        }
    }

    acc = simd_sum(acc);

    if (lane == 0) {
        C[tid] = half(acc);
    }
}

// ── INT4 matvec 2-row (decode path, M=1) ────────────────────────
//
// Two output rows per threadgroup, one per simdgroup. Halves TG count
// and improves L1 sharing of the input vector between simdgroups.
//
// Dispatch: (ceil(N/2), 1, 1) threadgroups, (64, 1, 1) threads per group.

kernel void affine_matvec_int4_2row(
    device const half *A            [[buffer(0)]],   // [1, K]
    device const uchar *B_packed    [[buffer(1)]],   // blocked [N_blk, K_blk, 64, 4]
    device const half *scales       [[buffer(2)]],   // [N, num_groups]
    device const half *zeros        [[buffer(3)]],   // [N, num_groups]
    device half *C                  [[buffer(4)]],   // [1, N]
    constant uint &N                [[buffer(5)]],
    constant uint &K                [[buffer(6)]],
    constant uint &group_size       [[buffer(7)]],
    device const half *awq_scales   [[buffer(8)]],
    constant uint &has_awq          [[buffer(9)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint row = tgid * 2 + sgid;
    if (row >= N) return;

    uint num_groups = (K + group_size - 1) / group_size;
    uint scale_row = row * num_groups;
    uint k_blocks = (K + BLK_K - 1) / BLK_K;
    uint n_block = row / BLK_N;
    uint n_local = row % BLK_N;

    float acc = 0.0f;

    for (uint kb = lane; kb < k_blocks; kb += 32) {
        uint word_idx = (n_block * k_blocks + kb) * BLK_N + n_local;
        uint packed4 = ((device const uint*)B_packed)[word_idx];

        uint k_elem = kb * BLK_K;
        uint grp = k_elem / group_size;
        float s = float(scales[scale_row + grp]);
        float z = float(zeros[scale_row + grp]);

        float w0 = (float(packed4 & 0xF) - z) * s;
        float w1 = (float((packed4 >> 4) & 0xF) - z) * s;
        float w2 = (float((packed4 >> 8) & 0xF) - z) * s;
        float w3 = (float((packed4 >> 12) & 0xF) - z) * s;
        float w4 = (float((packed4 >> 16) & 0xF) - z) * s;
        float w5 = (float((packed4 >> 20) & 0xF) - z) * s;
        float w6 = (float((packed4 >> 24) & 0xF) - z) * s;
        float w7 = (float((packed4 >> 28) & 0xF) - z) * s;

        if (has_awq) {
            acc += (float(A[k_elem])     / float(awq_scales[k_elem]))     * w0;
            acc += (float(A[k_elem + 1]) / float(awq_scales[k_elem + 1])) * w1;
            acc += (float(A[k_elem + 2]) / float(awq_scales[k_elem + 2])) * w2;
            acc += (float(A[k_elem + 3]) / float(awq_scales[k_elem + 3])) * w3;
            acc += (float(A[k_elem + 4]) / float(awq_scales[k_elem + 4])) * w4;
            acc += (float(A[k_elem + 5]) / float(awq_scales[k_elem + 5])) * w5;
            acc += (float(A[k_elem + 6]) / float(awq_scales[k_elem + 6])) * w6;
            acc += (float(A[k_elem + 7]) / float(awq_scales[k_elem + 7])) * w7;
        } else {
            acc += float(A[k_elem])     * w0 + float(A[k_elem + 1]) * w1
                 + float(A[k_elem + 2]) * w2 + float(A[k_elem + 3]) * w3
                 + float(A[k_elem + 4]) * w4 + float(A[k_elem + 5]) * w5
                 + float(A[k_elem + 6]) * w6 + float(A[k_elem + 7]) * w7;
        }
    }

    acc = simd_sum(acc);
    if (lane == 0) {
        C[row] = half(acc);
    }
}

// ── INT8 matvec (decode path, M=1) ──────────────────────────────
//
// One byte = one element. Uses uint32 loads for 4 elements per load.
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

    // Blocked layout: BLK_K=8 bytes per row per k-block = 2 uint32s
    uint k_blocks = (K + BLK_K - 1) / BLK_K;
    uint n_block = tid / BLK_N;
    uint n_local = tid % BLK_N;

    float acc = 0.0f;

    // Process 4 bytes (4 INT8 elements) per uint32 load.
    // Each k-block has BLK_K=8 bytes = 2 uint32 words per row.
    uint words_per_row = K / 4;
    for (uint w = lane; w < words_per_row; w += 32) {
        uint kb = (w * 4) / BLK_K;
        uint local_word = ((w * 4) % BLK_K) / 4;
        uint word_idx = (n_block * k_blocks + kb) * (BLK_N * 2)
                      + n_local * 2 + local_word;
        uint packed4 = ((device const uint*)B_packed)[word_idx];

        uint k_elem = w * 4;
        uint grp = k_elem / group_size;
        float s = float(scales[scale_row + grp]);
        float z = float(zeros[scale_row + grp]);

        float w0 = (float(packed4 & 0xFF) - z) * s;
        float w1 = (float((packed4 >> 8) & 0xFF) - z) * s;
        float w2 = (float((packed4 >> 16) & 0xFF) - z) * s;
        float w3 = (float((packed4 >> 24) & 0xFF) - z) * s;

        if (has_awq) {
            acc += (float(A[k_elem])     / float(awq_scales[k_elem]))     * w0;
            acc += (float(A[k_elem + 1]) / float(awq_scales[k_elem + 1])) * w1;
            acc += (float(A[k_elem + 2]) / float(awq_scales[k_elem + 2])) * w2;
            acc += (float(A[k_elem + 3]) / float(awq_scales[k_elem + 3])) * w3;
        } else {
            acc += float(A[k_elem])     * w0 + float(A[k_elem + 1]) * w1
                 + float(A[k_elem + 2]) * w2 + float(A[k_elem + 3]) * w3;
        }
    }

    acc = simd_sum(acc);

    if (lane == 0) {
        C[tid] = half(acc);
    }
}
