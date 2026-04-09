// ── INT4 superblock matvec (decode path, M=1) ──────────────────
//
// 8 output rows per threadgroup (2 simdgroups × 4 rows/sg). Each lane
// processes one uint32 (8 nibbles) per iteration, striding by 32 across K.
//
// Optimizations:
// 1. Power-of-2 fast path: bitwise shift/AND replaces integer division
// 2. Coalesced reads: consecutive lanes access consecutive superblock data
// 3. Multi-row amortization: input load shared across 4 rows per simdgroup
//
// Dispatch: (ceil(N/8), 1, 1) threadgroups × (64, 1, 1) threads

kernel void superblock_matvec_int4(
    device const half *A            [[buffer(0)]],   // [1, K]
    device const uchar *W           [[buffer(1)]],   // [N, G, sb_bytes] superblocks
    device half *C                  [[buffer(2)]],   // [1, N]
    constant uint &N                [[buffer(3)]],
    constant uint &K                [[buffer(4)]],
    device const half *awq_scales   [[buffer(5)]],   // [K] or dummy
    constant uint &has_awq          [[buffer(6)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    uint base_row = tgid * SB_ROWS_PER_TG + sgid * SB_ROWS_PER_SG;

    uint num_groups = K / GS;
    uint sb_bytes = SB_HEADER_BYTES + GS / 2;
    uint sb_stride = num_groups * sb_bytes;

    float result[SB_ROWS_PER_SG] = {0};

    uint k_words = K / 8;

    for (uint w = lane; w < k_words; w += 32) {
        uint k_elem = w * 8;

        uint g = k_elem / GS;
        uint word_in_data = (k_elem % GS) >> 3;

        // Load 8 input values (shared across all rows)
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

            device const uchar *sb = W + row * sb_stride + g * sb_bytes;
            float scale = float(*(device const half *)(sb));
            float zero  = float(*(device const half *)(sb + 2));

            uint packed4 = ((device const uint *)(sb + SB_HEADER_BYTES))[word_in_data];

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

    for (uint r = 0; r < SB_ROWS_PER_SG; r++) {
        result[r] = simd_sum(result[r]);
        if (lane == 0 && base_row + r < N) {
            C[base_row + r] = half(result[r]);
        }
    }
}

// ── INT8 superblock matvec (decode path, M=1) ──────────────────
//
// 8 output rows per threadgroup (2 simdgroups × 4 rows/sg).
// Each lane processes 4 elements (1 uint32) per iteration, striding by 32.
//
// INT8: 1 byte per element, so uint32 = 4 elements.
// sb_bytes = 4 + group_size (e.g. 132 for gs=128).
//
// Dispatch: (ceil(N/8), 1, 1) threadgroups × (64, 1, 1) threads

kernel void superblock_matvec_int8(
    device const half *A            [[buffer(0)]],
    device const uchar *W           [[buffer(1)]],
    device half *C                  [[buffer(2)]],
    constant uint &N                [[buffer(3)]],
    constant uint &K                [[buffer(4)]],
    device const half *awq_scales   [[buffer(5)]],
    constant uint &has_awq          [[buffer(6)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    uint base_row = tgid * SB_ROWS_PER_TG + sgid * SB_ROWS_PER_SG;

    uint num_groups = K / GS;
    uint sb_bytes = SB_HEADER_BYTES + GS;  // INT8: 1 byte per element
    uint sb_stride = num_groups * sb_bytes;

    float result[SB_ROWS_PER_SG] = {0};

    // Process K in chunks of 4 elements (1 uint32) per lane
    uint k_words = K / 4;  // total uint32 words across K dimension

    for (uint w = lane; w < k_words; w += 32) {
        uint k_elem = w * 4;
        uint g = k_elem / GS;
        uint k_in_group = k_elem % GS;
        uint word_in_data = k_in_group / 4;

        float a0 = float(A[k_elem]);
        float a1 = float(A[k_elem + 1]);
        float a2 = float(A[k_elem + 2]);
        float a3 = float(A[k_elem + 3]);

        if (has_awq) {
            a0 /= float(awq_scales[k_elem]);
            a1 /= float(awq_scales[k_elem + 1]);
            a2 /= float(awq_scales[k_elem + 2]);
            a3 /= float(awq_scales[k_elem + 3]);
        }

        for (uint r = 0; r < SB_ROWS_PER_SG; r++) {
            uint row = base_row + r;
            if (row >= N) break;

            device const uchar *sb = W + row * sb_stride + g * sb_bytes;
            float scale = float(*(device const half *)(sb));
            float zero  = float(*(device const half *)(sb + 2));

            uint packed4 = ((device const uint *)(sb + SB_HEADER_BYTES))[word_in_data];

            float w0 = (float(packed4 & 0xFF) - zero) * scale;
            float w1 = (float((packed4 >> 8) & 0xFF) - zero) * scale;
            float w2 = (float((packed4 >> 16) & 0xFF) - zero) * scale;
            float w3 = (float((packed4 >> 24) & 0xFF) - zero) * scale;

            result[r] += a0*w0 + a1*w1 + a2*w2 + a3*w3;
        }
    }

    for (uint r = 0; r < SB_ROWS_PER_SG; r++) {
        result[r] = simd_sum(result[r]);
        if (lane == 0 && base_row + r < N) {
            C[base_row + r] = half(result[r]);
        }
    }
}
