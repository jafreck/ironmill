// ── Batched INT4 matvec for FFN gate+up projections (M=1) ──────
//
// Computes 2 independent matvecs in a single dispatch:
//   gate = x · W_gate^T   (first N_gate threadgroups)
//   up   = x · W_up^T     (remaining N_up threadgroups)
//
// Both projections share the same input x. Threadgroup index determines
// which projection: tid < N_gate → gate, else → up.
//
// B_packed blocked layout guarantees 4-byte alignment per (n_local, k_block)
// slot, making the uint32 cast safe. Assumes group_size >= BLK_K.
//
// Dispatch: (N_gate + N_up, 1, 1) threadgroups, (32, 1, 1) threads.

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
    uint tid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    // Select which projection based on threadgroup index
    uint local_tid;
    device const uchar* B_packed;
    device const half* scales;
    device const half* zeros;
    device half* C;

    if (tid < N_gate) {
        local_tid = tid;
        B_packed = B_gate_packed;
        scales = scales_gate;
        zeros = zeros_gate;
        C = C_gate;
    } else {
        local_tid = tid - N_gate;
        B_packed = B_up_packed;
        scales = scales_up;
        zeros = zeros_up;
        C = C_up;
    }

    if (local_tid >= N_gate) return;

    uint num_groups = (K + group_size - 1) / group_size;
    uint scale_row = local_tid * num_groups;

    // Blocked layout: word-aligned addressing
    uint k_blocks = (K + BLK_K - 1) / BLK_K;
    uint n_block = local_tid / BLK_N;
    uint n_local = local_tid % BLK_N;

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
        C[local_tid] = half(acc);
    }
}

// ── Batched INT4 matvec for 4 GDN projections (M=1) ─────────────
//
// Computes 4 independent matvecs in a single dispatch:
//   qkv = x · W_qkv^T   (N0 threadgroups)
//   z   = x · W_z^T      (N1 threadgroups)
//   a   = x · W_a^T      (N2 threadgroups)
//   b   = x · W_b^T      (N3 threadgroups)
//
// All projections share the same input x. Threadgroup index determines
// which projection via cumulative thresholds in params.
//
// Dispatch: (N0 + N1 + N2 + N3, 1, 1) threadgroups, (32, 1, 1) threads.

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
    uint tid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint N0 = params.N0;
    uint N1 = params.N1;
    uint N2 = params.N2;
    uint K  = params.K;
    uint group_size = params.group_size;
    uint has_awq = params.has_awq;

    // Route threadgroup to the correct projection
    uint local_tid;
    uint N_proj;
    device const uchar* B_packed;
    device const half* scales;
    device const half* zeros;
    device half* C;

    uint t0 = N0;
    uint t1 = t0 + N1;
    uint t2 = t1 + N2;

    if (tid < t0) {
        local_tid = tid;
        N_proj = N0;
        B_packed = B0_packed; scales = scales0; zeros = zeros0; C = C0;
    } else if (tid < t1) {
        local_tid = tid - t0;
        N_proj = N1;
        B_packed = B1_packed; scales = scales1; zeros = zeros1; C = C1;
    } else if (tid < t2) {
        local_tid = tid - t1;
        N_proj = N2;
        B_packed = B2_packed; scales = scales2; zeros = zeros2; C = C2;
    } else {
        local_tid = tid - t2;
        N_proj = params.N3;
        B_packed = B3_packed; scales = scales3; zeros = zeros3; C = C3;
    }

    if (local_tid >= N_proj) return;

    uint num_groups = (K + group_size - 1) / group_size;
    uint scale_row = local_tid * num_groups;

    // Blocked layout: word-aligned addressing
    uint k_blocks = (K + BLK_K - 1) / BLK_K;
    uint n_block = local_tid / BLK_N;
    uint n_local = local_tid % BLK_N;

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
        C[local_tid] = half(acc);
    }
}

// ── Superblock batched INT4 matvec for FFN gate+up (M=1) ────────
//
// Dispatch: (N_gate + N_up, 1, 1) threadgroups, (32, 1, 1) threads.

kernel void superblock_batched_affine_matvec_int4(
    device const half *A            [[buffer(0)]],   // [1, K] shared input
    device const uchar *W_gate      [[buffer(1)]],   // gate superblocks
    device half *C_gate             [[buffer(2)]],
    device const uchar *W_up        [[buffer(3)]],   // up superblocks
    device half *C_up               [[buffer(4)]],
    constant uint &N_gate           [[buffer(5)]],
    constant uint &K                [[buffer(6)]],
    constant uint &group_size       [[buffer(7)]],
    device const half *awq_scales   [[buffer(8)]],
    constant uint &has_awq          [[buffer(9)]],
    uint tid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint local_tid;
    device const uchar *W;
    device half *C;

    if (tid < N_gate) {
        local_tid = tid;
        W = W_gate;
        C = C_gate;
    } else {
        local_tid = tid - N_gate;
        W = W_up;
        C = C_up;
    }

    if (local_tid >= N_gate) return;

    uint num_groups = K / group_size;
    uint sb_bytes = SB_HEADER_BYTES + group_size / 2;
    uint sb_stride = num_groups * sb_bytes;

    float acc = 0.0f;

    for (uint g = 0; g < num_groups; g++) {
        device const uchar *sb = W + local_tid * sb_stride + g * sb_bytes;
        float s = float(*(device const half *)(sb));
        float z = float(*(device const half *)(sb + 2));

        uint k_base = g * group_size;

        for (uint i = lane * BLK_K; i < group_size; i += 32 * BLK_K) {
            uint k_elem = k_base + i;
            uint word_idx = i / 8;
            uint packed4 = ((device const uint*)(sb + SB_HEADER_BYTES))[word_idx];

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
    }

    acc = simd_sum(acc);
    if (lane == 0) {
        C[local_tid] = half(acc);
    }
}

// ── Superblock GDN batched INT4 matvec for 4 projections (M=1) ──
//
// Superblock params struct replaces the separate N/K/group_size/has_awq params.
// Dispatch: (N0+N1+N2+N3, 1, 1) threadgroups, (32, 1, 1) threads.

kernel void superblock_gdn_batched_affine_matvec_int4(
    device const half *A            [[buffer(0)]],
    device const uchar *W0          [[buffer(1)]],   // qkv superblocks
    device half *C0                 [[buffer(2)]],
    device const uchar *W1          [[buffer(3)]],   // z superblocks
    device half *C1                 [[buffer(4)]],
    device const uchar *W2          [[buffer(5)]],   // a superblocks
    device half *C2                 [[buffer(6)]],
    device const uchar *W3          [[buffer(7)]],   // b superblocks
    device half *C3                 [[buffer(8)]],
    constant GdnBatchedInt4Params &params [[buffer(9)]],
    device const half *awq_scales   [[buffer(10)]],
    uint tid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint N0 = params.N0;
    uint N1 = params.N1;
    uint N2 = params.N2;
    uint K  = params.K;
    uint group_size = params.group_size;
    uint has_awq = params.has_awq;

    uint local_tid;
    uint N_proj;
    device const uchar *W;
    device half *C;

    uint t0 = N0;
    uint t1 = t0 + N1;
    uint t2 = t1 + N2;

    if (tid < t0) {
        local_tid = tid; N_proj = N0;
        W = W0; C = C0;
    } else if (tid < t1) {
        local_tid = tid - t0; N_proj = N1;
        W = W1; C = C1;
    } else if (tid < t2) {
        local_tid = tid - t1; N_proj = N2;
        W = W2; C = C2;
    } else {
        local_tid = tid - t2; N_proj = params.N3;
        W = W3; C = C3;
    }

    if (local_tid >= N_proj) return;

    uint num_groups = K / group_size;
    uint sb_bytes = SB_HEADER_BYTES + group_size / 2;
    uint sb_stride = num_groups * sb_bytes;

    float acc = 0.0f;

    for (uint g = 0; g < num_groups; g++) {
        device const uchar *sb = W + local_tid * sb_stride + g * sb_bytes;
        float s = float(*(device const half *)(sb));
        float z = float(*(device const half *)(sb + 2));

        uint k_base = g * group_size;

        for (uint i = lane * BLK_K; i < group_size; i += 32 * BLK_K) {
            uint k_elem = k_base + i;
            uint word_idx = i / 8;
            uint packed4 = ((device const uint*)(sb + SB_HEADER_BYTES))[word_idx];

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
    }

    acc = simd_sum(acc);
    if (lane == 0) {
        C[local_tid] = half(acc);
    }
}
