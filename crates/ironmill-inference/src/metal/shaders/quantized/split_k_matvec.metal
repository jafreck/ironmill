// ── Split-K INT4 superblock matvec (decode path, M=1) ──────────
//
// Single-dispatch split-K for improved GPU occupancy. Instead of
// launching separate partial + reduce dispatches (which add barrier
// overhead), this kernel uses wider threadgroups: each TG has
// SB_NUM_SIMDGROUPS * split_k SIMDgroups, where each pair of SIMDgroups
// handles the same 8 output rows but a different K-slice.
//
// After the K-slice computation, a cheap threadgroup_barrier + reduction
// in threadgroup memory produces the final half output in a single dispatch.
//
// Inner loop is identical to superblock_matvec_int4 (pre-scaled input
// trick, zero correction via scale * accum + bias * x_sum).
//
// Dispatch: (ceil(N/8), 1, 1) TGs × (64 * split_k, 1, 1) threads

kernel void split_k_matvec_int4(
    device const half *A            [[buffer(0)]],   // [1, K]
    device const uchar *W           [[buffer(1)]],   // [N, K/2] contiguous packed nibbles
    device half *C                  [[buffer(2)]],   // [1, N] — final half output
    constant uint &N                [[buffer(3)]],
    constant uint &K                [[buffer(4)]],
    device const half *awq_scales   [[buffer(5)]],   // [K] or dummy
    constant uint &has_awq          [[buffer(6)]],
    device const half *W_scales     [[buffer(7)]],   // [N, K/GS]
    device const half *W_zeros      [[buffer(8)]],   // [N, K/GS]
    constant uint &split_k          [[buffer(9)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    // Decompose SIMDgroup index: sg_local selects row group, sg_k selects K-slice
    uint sg_local = sgid % SB_NUM_SIMDGROUPS;   // [0, SB_NUM_SIMDGROUPS)
    uint sg_k     = sgid / SB_NUM_SIMDGROUPS;   // [0, split_k)
    uint base_row = tgid * SB_ROWS_PER_TG + sg_local * SB_ROWS_PER_SG;

    uint k_per_split = K / split_k;
    uint k_start = sg_k * k_per_split;
    uint k_end   = k_start + k_per_split;
    uint w_start = k_start / 8;
    uint w_end   = k_end / 8;

    uint num_groups = K / GS;
    uint sb_stride = num_groups * SB_BYTES_INT4;

    float result[SB_ROWS_PER_SG] = {0};

    for (uint w = w_start + lane; w < w_end; w += 32) {
        uint k_elem = w * 8;

        // Group index (compile-time shift since GS is #define'd)
        uint g = k_elem / GS;
        // uint16 index within the group's data section
        uint u16_in_data = (k_elem % GS) / 4;

        // ── Pre-scale 8 input values (2 × 4-element pattern) ──
        float v0 = float(A[k_elem]);
        float v1 = float(A[k_elem + 1]);
        float v2 = float(A[k_elem + 2]);
        float v3 = float(A[k_elem + 3]);
        float v4 = float(A[k_elem + 4]);
        float v5 = float(A[k_elem + 5]);
        float v6 = float(A[k_elem + 6]);
        float v7 = float(A[k_elem + 7]);

        if (has_awq) {
            v0 /= float(awq_scales[k_elem]);
            v1 /= float(awq_scales[k_elem + 1]);
            v2 /= float(awq_scales[k_elem + 2]);
            v3 /= float(awq_scales[k_elem + 3]);
            v4 /= float(awq_scales[k_elem + 4]);
            v5 /= float(awq_scales[k_elem + 5]);
            v6 /= float(awq_scales[k_elem + 6]);
            v7 /= float(awq_scales[k_elem + 7]);
        }

        float x_sum = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;

        // Pre-divide by positional powers: ÷1, ÷16, ÷256, ÷4096
        float xp0 = v0;
        float xp1 = v1 * (1.0f / 16.0f);
        float xp2 = v2 * (1.0f / 256.0f);
        float xp3 = v3 * (1.0f / 4096.0f);
        float xp4 = v4;
        float xp5 = v5 * (1.0f / 16.0f);
        float xp6 = v6 * (1.0f / 256.0f);
        float xp7 = v7 * (1.0f / 4096.0f);

        for (uint r = 0; r < SB_ROWS_PER_SG; r++) {
            uint row = base_row + r;
            if (row >= N) break;

            float scale = float(W_scales[row * num_groups + g]);
            float bias  = float(W_zeros[row * num_groups + g]);

            // Read 2 uint16s (= 8 nibbles) from contiguous data
            device const uint16_t *ws =
                (device const uint16_t *)(W + row * sb_stride + g * SB_BYTES_INT4) + u16_in_data;

            float accum = xp0 * float(ws[0] & 0x000f)
                        + xp1 * float(ws[0] & 0x00f0)
                        + xp2 * float(ws[0] & 0x0f00)
                        + xp3 * float(ws[0] & 0xf000)
                        + xp4 * float(ws[1] & 0x000f)
                        + xp5 * float(ws[1] & 0x00f0)
                        + xp6 * float(ws[1] & 0x0f00)
                        + xp7 * float(ws[1] & 0xf000);

            result[r] += scale * accum + bias * x_sum;
        }
    }

    // Reduce across SIMD lanes within each SIMDgroup
    for (uint r = 0; r < SB_ROWS_PER_SG; r++) {
        result[r] = simd_sum(result[r]);
    }

    // ── Intra-TG reduction across K-slices via threadgroup memory ──
    // Layout: partial[row_in_tg][k_slice], 256 bytes max (8 rows × 8 slices × float)
    threadgroup float partial[SB_ROWS_PER_TG][8];

    if (lane == 0) {
        for (uint r = 0; r < SB_ROWS_PER_SG; r++) {
            partial[sg_local * SB_ROWS_PER_SG + r][sg_k] = result[r];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First SB_NUM_SIMDGROUPS SIMDgroups (sg_k==0) perform final reduction
    if (sg_k == 0 && lane < SB_ROWS_PER_SG) {
        uint row = base_row + lane;
        if (row < N) {
            float sum = 0.0f;
            for (uint s = 0; s < split_k; s++) {
                sum += partial[sg_local * SB_ROWS_PER_SG + lane][s];
            }
            C[row] = half(sum);
        }
    }
}
