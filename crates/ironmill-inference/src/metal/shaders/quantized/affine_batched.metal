// ── Superblock batched INT4 matvec for FFN gate+up (M=1) ────────
//
// Dispatch: (N_gate + N_up, 1, 1) threadgroups, (32, 1, 1) threads.

kernel void superblock_batched_affine_matvec_int4(
    device const half *A            [[buffer(0)]],   // [1, K] shared input
    device const uchar *W_gate      [[buffer(1)]],   // gate data [N, K/2]
    device half *C_gate             [[buffer(2)]],
    device const uchar *W_up        [[buffer(3)]],   // up data [N, K/2]
    device half *C_up               [[buffer(4)]],
    constant uint &N_gate           [[buffer(5)]],
    constant uint &K                [[buffer(6)]],
    device const half *awq_scales   [[buffer(7)]],
    constant uint &has_awq          [[buffer(8)]],
    device const half *gate_scales  [[buffer(9)]],   // [N, K/GS]
    device const half *gate_zeros   [[buffer(10)]],  // [N, K/GS]
    device const half *up_scales    [[buffer(11)]],   // [N, K/GS]
    device const half *up_zeros     [[buffer(12)]],   // [N, K/GS]
    uint tid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint local_tid;
    device const uchar *W;
    device half *C;
    device const half *sc;
    device const half *zr;

    if (tid < N_gate) {
        local_tid = tid;
        W = W_gate;
        C = C_gate;
        sc = gate_scales;
        zr = gate_zeros;
    } else {
        local_tid = tid - N_gate;
        W = W_up;
        C = C_up;
        sc = up_scales;
        zr = up_zeros;
    }

    if (local_tid >= N_gate) return;

    uint num_groups = K / GS;
    uint sb_stride = num_groups * SB_BYTES_INT4;

    float acc = 0.0f;

    uint k_words = K / 8;

    for (uint w = lane; w < k_words; w += 32) {
        uint k_elem = w * 8;
        uint g = k_elem / GS;
        uint word_idx = (k_elem % GS) / 8;

        float s = float(sc[local_tid * num_groups + g]);
        float z = float(zr[local_tid * num_groups + g]);

        uint packed4 = ((device const uint*)(W + local_tid * sb_stride + g * SB_BYTES_INT4))[word_idx];

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

// ── Superblock GDN batched INT4 matvec for 4 projections (M=1) ──
//
// Params struct replaces the separate N/K/has_awq params.
// Dispatch: (N0+N1+N2+N3, 1, 1) threadgroups, (32, 1, 1) threads.

kernel void superblock_gdn_batched_affine_matvec_int4(
    device const half *A            [[buffer(0)]],
    device const uchar *W0          [[buffer(1)]],   // qkv data
    device half *C0                 [[buffer(2)]],
    device const uchar *W1          [[buffer(3)]],   // z data
    device half *C1                 [[buffer(4)]],
    device const uchar *W2          [[buffer(5)]],   // a data
    device half *C2                 [[buffer(6)]],
    device const uchar *W3          [[buffer(7)]],   // b data
    device half *C3                 [[buffer(8)]],
    constant GdnBatchedInt4Params &params [[buffer(9)]],
    device const half *awq_scales   [[buffer(10)]],
    device const half *sc0          [[buffer(11)]],  // scales for each proj
    device const half *zr0          [[buffer(12)]],
    device const half *sc1          [[buffer(13)]],
    device const half *zr1          [[buffer(14)]],
    device const half *sc2          [[buffer(15)]],
    device const half *zr2          [[buffer(16)]],
    device const half *sc3          [[buffer(17)]],
    device const half *zr3          [[buffer(18)]],
    uint tid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint N0 = params.N0;
    uint N1 = params.N1;
    uint N2 = params.N2;
    uint K  = params.K;
    uint has_awq = params.has_awq;

    uint local_tid;
    uint N_proj;
    device const uchar *W;
    device half *C;
    device const half *sc;
    device const half *zr;

    uint t0 = N0;
    uint t1 = t0 + N1;
    uint t2 = t1 + N2;

    if (tid < t0) {
        local_tid = tid; N_proj = N0;
        W = W0; C = C0; sc = sc0; zr = zr0;
    } else if (tid < t1) {
        local_tid = tid - t0; N_proj = N1;
        W = W1; C = C1; sc = sc1; zr = zr1;
    } else if (tid < t2) {
        local_tid = tid - t1; N_proj = N2;
        W = W2; C = C2; sc = sc2; zr = zr2;
    } else {
        local_tid = tid - t2; N_proj = params.N3;
        W = W3; C = C3; sc = sc3; zr = zr3;
    }

    if (local_tid >= N_proj) return;

    uint num_groups = K / GS;
    uint sb_stride = num_groups * SB_BYTES_INT4;

    float acc = 0.0f;

    uint k_words = K / 8;

    for (uint w = lane; w < k_words; w += 32) {
        uint k_elem = w * 8;
        uint g = k_elem / GS;
        uint word_idx = (k_elem % GS) / 8;

        float s = float(sc[local_tid * num_groups + g]);
        float z = float(zr[local_tid * num_groups + g]);

        uint packed4 = ((device const uint*)(W + local_tid * sb_stride + g * SB_BYTES_INT4))[word_idx];

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

// ── Superblock batched INT4 matvec for QKV projections (M=1) ────
//
// Computes Q = x·W_q^T, K = x·W_k^T, V = x·W_v^T in a single
// dispatch. Supports different N per projection (GQA: N_k, N_v < N_q).
//
// Uses the same 8-row/64-thread optimized pattern as superblock_matvec_int4:
// 2 simdgroups × 4 rows/sg = 8 rows per TG, with pre-scale trick.
//
// Dispatch: (ceil(N_q/8)+ceil(N_k/8)+ceil(N_v/8), 1, 1) TGs × (64,1,1)

kernel void superblock_batched_qkv_matvec_int4(
    device const half *A            [[buffer(0)]],   // [1, K] shared input
    device const uchar *W_q         [[buffer(1)]],   // Q weight data [N_q, K/2]
    device half *C_q                [[buffer(2)]],
    device const uchar *W_k         [[buffer(3)]],   // K weight data [N_k, K/2]
    device half *C_k                [[buffer(4)]],
    device const uchar *W_v         [[buffer(5)]],   // V weight data [N_v, K/2]
    device half *C_v                [[buffer(6)]],
    constant QkvBatchedInt4Params &params [[buffer(7)]],
    device const half *awq_scales   [[buffer(8)]],
    device const half *q_scales     [[buffer(9)]],   // [N_q, K/GS]
    device const half *q_zeros      [[buffer(10)]],
    device const half *k_scales     [[buffer(11)]],  // [N_k, K/GS]
    device const half *k_zeros      [[buffer(12)]],
    device const half *v_scales     [[buffer(13)]],  // [N_v, K/GS]
    device const half *v_zeros      [[buffer(14)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    uint N_q = params.N_q;
    uint N_k = params.N_k;
    uint N_v = params.N_v;
    uint K   = params.K;
    uint has_awq = params.has_awq;

    // Route TG to projection: TGs are allocated in blocks of ceil(N/8)
    uint tg_q = (N_q + SB_ROWS_PER_TG - 1) / SB_ROWS_PER_TG;
    uint tg_k = (N_k + SB_ROWS_PER_TG - 1) / SB_ROWS_PER_TG;

    uint local_tg;
    uint N_proj;
    device const uchar *W;
    device half *C;
    device const half *sc;
    device const half *zr;

    if (tgid < tg_q) {
        local_tg = tgid; N_proj = N_q;
        W = W_q; C = C_q; sc = q_scales; zr = q_zeros;
    } else if (tgid < tg_q + tg_k) {
        local_tg = tgid - tg_q; N_proj = N_k;
        W = W_k; C = C_k; sc = k_scales; zr = k_zeros;
    } else {
        local_tg = tgid - tg_q - tg_k; N_proj = N_v;
        W = W_v; C = C_v; sc = v_scales; zr = v_zeros;
    }

    uint base_row = local_tg * SB_ROWS_PER_TG + sgid * SB_ROWS_PER_SG;

    uint num_groups = K / GS;
    uint sb_stride = num_groups * SB_BYTES_INT4;

    float result[SB_ROWS_PER_SG] = {0};

    uint k_words = K / 8;

    for (uint w = lane; w < k_words; w += 32) {
        uint k_elem = w * 8;
        uint g = k_elem / GS;
        uint u16_in_data = (k_elem % GS) / 4;

        // Pre-scale 8 input values
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
            if (row >= N_proj) break;

            float scale = float(sc[row * num_groups + g]);
            float bias  = float(zr[row * num_groups + g]);

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

    for (uint r = 0; r < SB_ROWS_PER_SG; r++) {
        result[r] = simd_sum(result[r]);
        if (lane == 0 && base_row + r < N_proj) {
            C[base_row + r] = half(result[r]);
        }
    }
}
