// ── INT4 matvec (decode path, M=1) ──────────────────────────────
//
// One threadgroup per output row. Each lane processes K/(8·32) uint32 words,
// unpacking 8 nibbles per load for 4× wider memory transactions.
//
// B_packed is in blocked layout: [N_blocks, K_blocks, BLK_N, BLK_K/2]
// produced by pack_quantized_blocked(). BLK_K=8 elements → 4 packed bytes
// = 1 uint32 per (n_local, k_block). The blocked layout guarantees 4-byte
// alignment for each (n_local, k_block) slot, so the uint32 cast is safe.
//
// Assumes group_size >= BLK_K (true for all practical group sizes: 32, 64, 128).
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


// ── INT4 matvec coalesced (decode path, M=1) ────────────────────
//
// 32 adjacent output rows per threadgroup. Each lane handles one row.
// All lanes in a simdgroup access CONSECUTIVE uint32s within the same
// k-block, achieving perfect memory coalescing (128 bytes per cache
// line transaction, 100% utilized vs 3% in the strided per-row kernel).
//
// Each lane iterates over ALL k-blocks sequentially — no simd_sum needed.
// The input vector A is broadcast-read by all lanes (L1 cached).
//
// Dispatch: (ceil(N/32), 1, 1) threadgroups, (32, 1, 1) threads.

kernel void affine_matvec_int4_coalesced(
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
    uint lane [[thread_index_in_simdgroup]])
{
    uint row = tgid * 32 + lane;
    if (row >= N) return;

    uint num_groups = (K + group_size - 1) / group_size;
    uint scale_row = row * num_groups;
    uint k_blocks = (K + BLK_K - 1) / BLK_K;
    uint n_block = row / BLK_N;
    uint n_local = row % BLK_N;

    // Base offset for this row's weight data across all k-blocks
    uint row_base = n_block * k_blocks * BLK_N + n_local;

    float acc = 0.0f;

    // Process 4 k-blocks per iteration for instruction-level parallelism.
    // All 32 lanes access consecutive addresses within each k-block (coalesced).
    uint kb = 0;
    uint k_blocks_4 = k_blocks & ~3u;  // round down to multiple of 4

    for (; kb < k_blocks_4; kb += 4) {
        uint base0 = row_base + kb * BLK_N;
        uint p0 = ((device const uint*)B_packed)[base0];
        uint p1 = ((device const uint*)B_packed)[base0 + BLK_N];
        uint p2 = ((device const uint*)B_packed)[base0 + BLK_N * 2];
        uint p3 = ((device const uint*)B_packed)[base0 + BLK_N * 3];

        uint k0 = kb * BLK_K;

        // Scale/zero for all 4 groups (may share if group_size >= 32)
        uint g0 = k0 / group_size;
        float s0 = float(scales[scale_row + g0]);
        float z0 = float(zeros[scale_row + g0]);

        uint g1 = (k0 + BLK_K) / group_size;
        float s1 = (g1 != g0) ? float(scales[scale_row + g1]) : s0;
        float z1 = (g1 != g0) ? float(zeros[scale_row + g1]) : z0;

        uint g2 = (k0 + BLK_K * 2) / group_size;
        float s2 = (g2 != g1) ? float(scales[scale_row + g2]) : s1;
        float z2 = (g2 != g1) ? float(zeros[scale_row + g2]) : z1;

        uint g3 = (k0 + BLK_K * 3) / group_size;
        float s3 = (g3 != g2) ? float(scales[scale_row + g3]) : s2;
        float z3 = (g3 != g2) ? float(zeros[scale_row + g3]) : z2;

        // Dequant and dot product for k-block 0 (8 elements)
        acc += float(A[k0])     * ((float(p0 & 0xF) - z0) * s0);
        acc += float(A[k0 + 1]) * ((float((p0 >> 4) & 0xF) - z0) * s0);
        acc += float(A[k0 + 2]) * ((float((p0 >> 8) & 0xF) - z0) * s0);
        acc += float(A[k0 + 3]) * ((float((p0 >> 12) & 0xF) - z0) * s0);
        acc += float(A[k0 + 4]) * ((float((p0 >> 16) & 0xF) - z0) * s0);
        acc += float(A[k0 + 5]) * ((float((p0 >> 20) & 0xF) - z0) * s0);
        acc += float(A[k0 + 6]) * ((float((p0 >> 24) & 0xF) - z0) * s0);
        acc += float(A[k0 + 7]) * ((float((p0 >> 28) & 0xF) - z0) * s0);

        // k-block 1
        uint k1 = k0 + BLK_K;
        acc += float(A[k1])     * ((float(p1 & 0xF) - z1) * s1);
        acc += float(A[k1 + 1]) * ((float((p1 >> 4) & 0xF) - z1) * s1);
        acc += float(A[k1 + 2]) * ((float((p1 >> 8) & 0xF) - z1) * s1);
        acc += float(A[k1 + 3]) * ((float((p1 >> 12) & 0xF) - z1) * s1);
        acc += float(A[k1 + 4]) * ((float((p1 >> 16) & 0xF) - z1) * s1);
        acc += float(A[k1 + 5]) * ((float((p1 >> 20) & 0xF) - z1) * s1);
        acc += float(A[k1 + 6]) * ((float((p1 >> 24) & 0xF) - z1) * s1);
        acc += float(A[k1 + 7]) * ((float((p1 >> 28) & 0xF) - z1) * s1);

        // k-block 2
        uint k2 = k0 + BLK_K * 2;
        acc += float(A[k2])     * ((float(p2 & 0xF) - z2) * s2);
        acc += float(A[k2 + 1]) * ((float((p2 >> 4) & 0xF) - z2) * s2);
        acc += float(A[k2 + 2]) * ((float((p2 >> 8) & 0xF) - z2) * s2);
        acc += float(A[k2 + 3]) * ((float((p2 >> 12) & 0xF) - z2) * s2);
        acc += float(A[k2 + 4]) * ((float((p2 >> 16) & 0xF) - z2) * s2);
        acc += float(A[k2 + 5]) * ((float((p2 >> 20) & 0xF) - z2) * s2);
        acc += float(A[k2 + 6]) * ((float((p2 >> 24) & 0xF) - z2) * s2);
        acc += float(A[k2 + 7]) * ((float((p2 >> 28) & 0xF) - z2) * s2);

        // k-block 3
        uint k3 = k0 + BLK_K * 3;
        acc += float(A[k3])     * ((float(p3 & 0xF) - z3) * s3);
        acc += float(A[k3 + 1]) * ((float((p3 >> 4) & 0xF) - z3) * s3);
        acc += float(A[k3 + 2]) * ((float((p3 >> 8) & 0xF) - z3) * s3);
        acc += float(A[k3 + 3]) * ((float((p3 >> 12) & 0xF) - z3) * s3);
        acc += float(A[k3 + 4]) * ((float((p3 >> 16) & 0xF) - z3) * s3);
        acc += float(A[k3 + 5]) * ((float((p3 >> 20) & 0xF) - z3) * s3);
        acc += float(A[k3 + 6]) * ((float((p3 >> 24) & 0xF) - z3) * s3);
        acc += float(A[k3 + 7]) * ((float((p3 >> 28) & 0xF) - z3) * s3);
    }

    // Handle remaining k-blocks (0-3)
    for (; kb < k_blocks; kb++) {
        uint word_idx = row_base + kb * BLK_N;
        uint packed4 = ((device const uint*)B_packed)[word_idx];
        uint k_elem = kb * BLK_K;
        uint grp = k_elem / group_size;
        float s = float(scales[scale_row + grp]);
        float z = float(zeros[scale_row + grp]);

        acc += float(A[k_elem])     * ((float(packed4 & 0xF) - z) * s);
        acc += float(A[k_elem + 1]) * ((float((packed4 >> 4) & 0xF) - z) * s);
        acc += float(A[k_elem + 2]) * ((float((packed4 >> 8) & 0xF) - z) * s);
        acc += float(A[k_elem + 3]) * ((float((packed4 >> 12) & 0xF) - z) * s);
        acc += float(A[k_elem + 4]) * ((float((packed4 >> 16) & 0xF) - z) * s);
        acc += float(A[k_elem + 5]) * ((float((packed4 >> 20) & 0xF) - z) * s);
        acc += float(A[k_elem + 6]) * ((float((packed4 >> 24) & 0xF) - z) * s);
        acc += float(A[k_elem + 7]) * ((float((packed4 >> 28) & 0xF) - z) * s);
    }

    C[row] = half(acc);
}

// ── INT4 superblock matvec (decode path, M=1) ──────────────────
//
// 8 output rows per threadgroup (2 simdgroups × 4 rows/sg). Each lane
// processes one uint32 (8 nibbles) per iteration, striding by 32 across K.
//
// Key optimizations:
// 1. Coalesced reads: consecutive lanes read consecutive uint32s within
//    superblock data — contiguous for group_size >= 256, near-contiguous
//    (4B header gaps) for group_size = 128
// 2. Inline scale/zero: fetched from superblock header, cached per group
// 3. Multi-row amortization: input load shared across 4 rows per simdgroup
//
// Dispatch: (ceil(N/8), 1, 1) threadgroups × (64, 1, 1) threads

kernel void superblock_matvec_int4(
    device const half *A            [[buffer(0)]],   // [1, K]
    device const uchar *W           [[buffer(1)]],   // [N, G, sb_bytes] superblocks
    device half *C                  [[buffer(2)]],   // [1, N]
    constant uint &N                [[buffer(3)]],
    constant uint &K                [[buffer(4)]],
    constant uint &group_size       [[buffer(5)]],
    device const half *awq_scales   [[buffer(6)]],   // [K] or dummy
    constant uint &has_awq          [[buffer(7)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    uint base_row = tgid * SB_ROWS_PER_TG + sgid * SB_ROWS_PER_SG;

    uint num_groups = K / group_size;
    uint sb_bytes = SB_HEADER_BYTES + group_size / 2;
    uint sb_stride = num_groups * sb_bytes;  // bytes per row

    float result[SB_ROWS_PER_SG] = {0};

    // Process K in chunks of 8 elements (1 uint32 = 8 nibbles) per lane,
    // 32 lanes process 256 elements per iteration.
    uint k_words = K / 8;  // total uint32 words across K dimension

    for (uint w = lane; w < k_words; w += 32) {
        uint k_elem = w * 8;

        // Which group and position within group
        uint g = k_elem / group_size;
        uint k_in_group = k_elem % group_size;
        uint word_in_data = k_in_group / 8;

        // Load 8 input values (shared across 4 rows)
        float a0 = float(A[k_elem]);
        float a1 = float(A[k_elem + 1]);
        float a2 = float(A[k_elem + 2]);
        float a3 = float(A[k_elem + 3]);
        float a4 = float(A[k_elem + 4]);
        float a5 = float(A[k_elem + 5]);
        float a6 = float(A[k_elem + 6]);
        float a7 = float(A[k_elem + 7]);

        if (has_awq) {
            a0 /= float(awq_scales[k_elem]);
            a1 /= float(awq_scales[k_elem + 1]);
            a2 /= float(awq_scales[k_elem + 2]);
            a3 /= float(awq_scales[k_elem + 3]);
            a4 /= float(awq_scales[k_elem + 4]);
            a5 /= float(awq_scales[k_elem + 5]);
            a6 /= float(awq_scales[k_elem + 6]);
            a7 /= float(awq_scales[k_elem + 7]);
        }

        for (uint r = 0; r < SB_ROWS_PER_SG; r++) {
            uint row = base_row + r;
            if (row >= N) break;

            // Superblock pointer: scale + zero + weight data inline
            device const uchar *sb = W + row * sb_stride + g * sb_bytes;
            float scale = float(*(device const half *)(sb));
            float zero  = float(*(device const half *)(sb + 2));

            // Read uint32 from data portion of the superblock
            uint packed4 = ((device const uint *)(sb + SB_HEADER_BYTES))[word_in_data];

            // Dequantize 8 nibbles
            float w0 = (float(packed4 & 0xF) - zero) * scale;
            float w1 = (float((packed4 >> 4) & 0xF) - zero) * scale;
            float w2 = (float((packed4 >> 8) & 0xF) - zero) * scale;
            float w3 = (float((packed4 >> 12) & 0xF) - zero) * scale;
            float w4 = (float((packed4 >> 16) & 0xF) - zero) * scale;
            float w5 = (float((packed4 >> 20) & 0xF) - zero) * scale;
            float w6 = (float((packed4 >> 24) & 0xF) - zero) * scale;
            float w7 = (float((packed4 >> 28) & 0xF) - zero) * scale;

            result[r] += a0*w0 + a1*w1 + a2*w2 + a3*w3
                       + a4*w4 + a5*w5 + a6*w6 + a7*w7;
        }
    }

    // Reduce across SIMD lanes
    for (uint r = 0; r < SB_ROWS_PER_SG; r++) {
        result[r] = simd_sum(result[r]);
        if (lane == 0 && base_row + r < N) {
            C[base_row + r] = half(result[r]);
        }
    }
}
