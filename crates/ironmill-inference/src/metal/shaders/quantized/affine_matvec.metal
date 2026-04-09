// ── INT4 superblock matvec (decode path, M=1) ──────────────────
//
// 8 output rows per threadgroup (2 simdgroups × 4 rows/sg).
// 16 elements per lane per iteration (4 × uint16 = 16 nibbles).
// MLX-style pre-scaled input trick with ÷1, ÷16, ÷256, ÷4096 pattern (×4).
//
// Pre-scaled trick: divide input by positional powers so that raw masked
// nibbles × pre-scaled input = correct product. Eliminates ALL bit-shift
// operations from the hot loop.
//
// Zero correction via scale * accum + bias * x_sum (1 FMA per word per row).
// 16 elements always stays within a single group (GS >= 32).
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
    uint sb_stride = num_groups * SB_BYTES_INT4;

    float result[SB_ROWS_PER_SG] = {0};

    uint k_words = K / 16;  // 16 elements (4 × uint16) per lane per iteration

    for (uint w = lane; w < k_words; w += 32) {
        uint k_elem = w * 16;

        // Group index (compile-time shift since GS is #define'd)
        uint g = k_elem / GS;
        // uint16 index within the group's data section (4 uint16s per 16 nibbles)
        uint u16_in_data = (k_elem % GS) / 4;

        // ── Pre-scale 16 input values (4 × 4-element pattern) ──
        float v0  = float(A[k_elem]);
        float v1  = float(A[k_elem + 1]);
        float v2  = float(A[k_elem + 2]);
        float v3  = float(A[k_elem + 3]);
        float v4  = float(A[k_elem + 4]);
        float v5  = float(A[k_elem + 5]);
        float v6  = float(A[k_elem + 6]);
        float v7  = float(A[k_elem + 7]);
        float v8  = float(A[k_elem + 8]);
        float v9  = float(A[k_elem + 9]);
        float v10 = float(A[k_elem + 10]);
        float v11 = float(A[k_elem + 11]);
        float v12 = float(A[k_elem + 12]);
        float v13 = float(A[k_elem + 13]);
        float v14 = float(A[k_elem + 14]);
        float v15 = float(A[k_elem + 15]);

        if (has_awq) {
            v0  /= float(awq_scales[k_elem]);
            v1  /= float(awq_scales[k_elem + 1]);
            v2  /= float(awq_scales[k_elem + 2]);
            v3  /= float(awq_scales[k_elem + 3]);
            v4  /= float(awq_scales[k_elem + 4]);
            v5  /= float(awq_scales[k_elem + 5]);
            v6  /= float(awq_scales[k_elem + 6]);
            v7  /= float(awq_scales[k_elem + 7]);
            v8  /= float(awq_scales[k_elem + 8]);
            v9  /= float(awq_scales[k_elem + 9]);
            v10 /= float(awq_scales[k_elem + 10]);
            v11 /= float(awq_scales[k_elem + 11]);
            v12 /= float(awq_scales[k_elem + 12]);
            v13 /= float(awq_scales[k_elem + 13]);
            v14 /= float(awq_scales[k_elem + 14]);
            v15 /= float(awq_scales[k_elem + 15]);
        }

        float x_sum = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7
                     + v8 + v9 + v10 + v11 + v12 + v13 + v14 + v15;

        // Pre-divide by positional powers: ÷1, ÷16, ÷256, ÷4096 (repeated 4×)
        float xp0  = v0;
        float xp1  = v1  * (1.0f / 16.0f);
        float xp2  = v2  * (1.0f / 256.0f);
        float xp3  = v3  * (1.0f / 4096.0f);
        float xp4  = v4;
        float xp5  = v5  * (1.0f / 16.0f);
        float xp6  = v6  * (1.0f / 256.0f);
        float xp7  = v7  * (1.0f / 4096.0f);
        float xp8  = v8;
        float xp9  = v9  * (1.0f / 16.0f);
        float xp10 = v10 * (1.0f / 256.0f);
        float xp11 = v11 * (1.0f / 4096.0f);
        float xp12 = v12;
        float xp13 = v13 * (1.0f / 16.0f);
        float xp14 = v14 * (1.0f / 256.0f);
        float xp15 = v15 * (1.0f / 4096.0f);

        for (uint r = 0; r < SB_ROWS_PER_SG; r++) {
            uint row = base_row + r;
            if (row >= N) break;

            device const uchar *sb = W + row * sb_stride + g * SB_BYTES_INT4;
            float scale = float(*(device const half *)(sb));
            float bias  = float(*(device const half *)(sb + 2));

            // Read 4 uint16s (= 16 nibbles) from data section — NO shifts needed
            device const uint16_t *ws =
                (device const uint16_t *)(sb + SB_HEADER_BYTES) + u16_in_data;

            float accum = xp0  * float(ws[0] & 0x000f)
                        + xp1  * float(ws[0] & 0x00f0)
                        + xp2  * float(ws[0] & 0x0f00)
                        + xp3  * float(ws[0] & 0xf000)
                        + xp4  * float(ws[1] & 0x000f)
                        + xp5  * float(ws[1] & 0x00f0)
                        + xp6  * float(ws[1] & 0x0f00)
                        + xp7  * float(ws[1] & 0xf000)
                        + xp8  * float(ws[2] & 0x000f)
                        + xp9  * float(ws[2] & 0x00f0)
                        + xp10 * float(ws[2] & 0x0f00)
                        + xp11 * float(ws[2] & 0xf000)
                        + xp12 * float(ws[3] & 0x000f)
                        + xp13 * float(ws[3] & 0x00f0)
                        + xp14 * float(ws[3] & 0x0f00)
                        + xp15 * float(ws[3] & 0xf000);

            result[r] += scale * accum + bias * x_sum;
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

// ── INT8 superblock matvec (decode path, M=1) ──────────────────
//
// 8 output rows per threadgroup (2 simdgroups × 4 rows/sg).
// Each lane processes 8 elements (2 uint32) per iteration, striding by 32.
//
// INT8: 1 byte per element, so 2 × uint32 = 8 elements.
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
    uint sb_stride = num_groups * SB_BYTES_INT8;

    float result[SB_ROWS_PER_SG] = {0};

    // Process K in chunks of 8 elements (2 uint32) per lane
    uint k_words = K / 8;

    for (uint w = lane; w < k_words; w += 32) {
        uint k_elem = w * 8;
        uint g = k_elem / GS;
        uint k_in_group = k_elem % GS;
        uint word_in_data = k_in_group / 4;

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

            device const uchar *sb = W + row * sb_stride + g * SB_BYTES_INT8;
            float scale = float(*(device const half *)(sb));
            float zero  = float(*(device const half *)(sb + 2));

            uint packed4a = ((device const uint *)(sb + SB_HEADER_BYTES))[word_in_data];
            uint packed4b = ((device const uint *)(sb + SB_HEADER_BYTES))[word_in_data + 1];

            float w0 = (float(packed4a & 0xFF) - zero) * scale;
            float w1 = (float((packed4a >> 8) & 0xFF) - zero) * scale;
            float w2 = (float((packed4a >> 16) & 0xFF) - zero) * scale;
            float w3 = (float((packed4a >> 24) & 0xFF) - zero) * scale;
            float w4 = (float(packed4b & 0xFF) - zero) * scale;
            float w5 = (float((packed4b >> 8) & 0xFF) - zero) * scale;
            float w6 = (float((packed4b >> 16) & 0xFF) - zero) * scale;
            float w7 = (float((packed4b >> 24) & 0xFF) - zero) * scale;

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
