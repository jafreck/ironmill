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
    device const half *awq_scales   [[buffer(7)]],
    constant uint &has_awq          [[buffer(8)]],
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

    uint num_groups = K / GS;
    uint sb_stride = num_groups * SB_BYTES_INT4;

    float acc = 0.0f;

    for (uint g = 0; g < num_groups; g++) {
        device const uchar *sb = W + local_tid * sb_stride + g * SB_BYTES_INT4;
        float s = float(*(device const half *)(sb));
        float z = float(*(device const half *)(sb + 2));

        uint k_base = g * GS;

        for (uint i = lane * 16; i < GS; i += 32 * 16) {
            uint k_elem = k_base + i;
            uint w_idx0 = i / 8;
            uint packed4a = ((device const uint*)(sb + SB_HEADER_BYTES))[w_idx0];
            uint packed4b = ((device const uint*)(sb + SB_HEADER_BYTES))[w_idx0 + 1];

            float w0  = (float(packed4a & 0xF) - z) * s;
            float w1  = (float((packed4a >> 4) & 0xF) - z) * s;
            float w2  = (float((packed4a >> 8) & 0xF) - z) * s;
            float w3  = (float((packed4a >> 12) & 0xF) - z) * s;
            float w4  = (float((packed4a >> 16) & 0xF) - z) * s;
            float w5  = (float((packed4a >> 20) & 0xF) - z) * s;
            float w6  = (float((packed4a >> 24) & 0xF) - z) * s;
            float w7  = (float((packed4a >> 28) & 0xF) - z) * s;
            float w8  = (float(packed4b & 0xF) - z) * s;
            float w9  = (float((packed4b >> 4) & 0xF) - z) * s;
            float w10 = (float((packed4b >> 8) & 0xF) - z) * s;
            float w11 = (float((packed4b >> 12) & 0xF) - z) * s;
            float w12 = (float((packed4b >> 16) & 0xF) - z) * s;
            float w13 = (float((packed4b >> 20) & 0xF) - z) * s;
            float w14 = (float((packed4b >> 24) & 0xF) - z) * s;
            float w15 = (float((packed4b >> 28) & 0xF) - z) * s;

            if (has_awq) {
                acc += (float(A[k_elem])      / float(awq_scales[k_elem]))      * w0;
                acc += (float(A[k_elem + 1])  / float(awq_scales[k_elem + 1]))  * w1;
                acc += (float(A[k_elem + 2])  / float(awq_scales[k_elem + 2]))  * w2;
                acc += (float(A[k_elem + 3])  / float(awq_scales[k_elem + 3]))  * w3;
                acc += (float(A[k_elem + 4])  / float(awq_scales[k_elem + 4]))  * w4;
                acc += (float(A[k_elem + 5])  / float(awq_scales[k_elem + 5]))  * w5;
                acc += (float(A[k_elem + 6])  / float(awq_scales[k_elem + 6]))  * w6;
                acc += (float(A[k_elem + 7])  / float(awq_scales[k_elem + 7]))  * w7;
                acc += (float(A[k_elem + 8])  / float(awq_scales[k_elem + 8]))  * w8;
                acc += (float(A[k_elem + 9])  / float(awq_scales[k_elem + 9]))  * w9;
                acc += (float(A[k_elem + 10]) / float(awq_scales[k_elem + 10])) * w10;
                acc += (float(A[k_elem + 11]) / float(awq_scales[k_elem + 11])) * w11;
                acc += (float(A[k_elem + 12]) / float(awq_scales[k_elem + 12])) * w12;
                acc += (float(A[k_elem + 13]) / float(awq_scales[k_elem + 13])) * w13;
                acc += (float(A[k_elem + 14]) / float(awq_scales[k_elem + 14])) * w14;
                acc += (float(A[k_elem + 15]) / float(awq_scales[k_elem + 15])) * w15;
            } else {
                acc += float(A[k_elem])      * w0  + float(A[k_elem + 1])  * w1
                     + float(A[k_elem + 2])  * w2  + float(A[k_elem + 3])  * w3
                     + float(A[k_elem + 4])  * w4  + float(A[k_elem + 5])  * w5
                     + float(A[k_elem + 6])  * w6  + float(A[k_elem + 7])  * w7
                     + float(A[k_elem + 8])  * w8  + float(A[k_elem + 9])  * w9
                     + float(A[k_elem + 10]) * w10 + float(A[k_elem + 11]) * w11
                     + float(A[k_elem + 12]) * w12 + float(A[k_elem + 13]) * w13
                     + float(A[k_elem + 14]) * w14 + float(A[k_elem + 15]) * w15;
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
// Superblock params struct replaces the separate N/K/has_awq params.
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

    uint num_groups = K / GS;
    uint sb_stride = num_groups * SB_BYTES_INT4;

    float acc = 0.0f;

    for (uint g = 0; g < num_groups; g++) {
        device const uchar *sb = W + local_tid * sb_stride + g * SB_BYTES_INT4;
        float s = float(*(device const half *)(sb));
        float z = float(*(device const half *)(sb + 2));

        uint k_base = g * GS;

        for (uint i = lane * 16; i < GS; i += 32 * 16) {
            uint k_elem = k_base + i;
            uint w_idx0 = i / 8;
            uint packed4a = ((device const uint*)(sb + SB_HEADER_BYTES))[w_idx0];
            uint packed4b = ((device const uint*)(sb + SB_HEADER_BYTES))[w_idx0 + 1];

            float w0  = (float(packed4a & 0xF) - z) * s;
            float w1  = (float((packed4a >> 4) & 0xF) - z) * s;
            float w2  = (float((packed4a >> 8) & 0xF) - z) * s;
            float w3  = (float((packed4a >> 12) & 0xF) - z) * s;
            float w4  = (float((packed4a >> 16) & 0xF) - z) * s;
            float w5  = (float((packed4a >> 20) & 0xF) - z) * s;
            float w6  = (float((packed4a >> 24) & 0xF) - z) * s;
            float w7  = (float((packed4a >> 28) & 0xF) - z) * s;
            float w8  = (float(packed4b & 0xF) - z) * s;
            float w9  = (float((packed4b >> 4) & 0xF) - z) * s;
            float w10 = (float((packed4b >> 8) & 0xF) - z) * s;
            float w11 = (float((packed4b >> 12) & 0xF) - z) * s;
            float w12 = (float((packed4b >> 16) & 0xF) - z) * s;
            float w13 = (float((packed4b >> 20) & 0xF) - z) * s;
            float w14 = (float((packed4b >> 24) & 0xF) - z) * s;
            float w15 = (float((packed4b >> 28) & 0xF) - z) * s;

            if (has_awq) {
                acc += (float(A[k_elem])      / float(awq_scales[k_elem]))      * w0;
                acc += (float(A[k_elem + 1])  / float(awq_scales[k_elem + 1]))  * w1;
                acc += (float(A[k_elem + 2])  / float(awq_scales[k_elem + 2]))  * w2;
                acc += (float(A[k_elem + 3])  / float(awq_scales[k_elem + 3]))  * w3;
                acc += (float(A[k_elem + 4])  / float(awq_scales[k_elem + 4]))  * w4;
                acc += (float(A[k_elem + 5])  / float(awq_scales[k_elem + 5]))  * w5;
                acc += (float(A[k_elem + 6])  / float(awq_scales[k_elem + 6]))  * w6;
                acc += (float(A[k_elem + 7])  / float(awq_scales[k_elem + 7]))  * w7;
                acc += (float(A[k_elem + 8])  / float(awq_scales[k_elem + 8]))  * w8;
                acc += (float(A[k_elem + 9])  / float(awq_scales[k_elem + 9]))  * w9;
                acc += (float(A[k_elem + 10]) / float(awq_scales[k_elem + 10])) * w10;
                acc += (float(A[k_elem + 11]) / float(awq_scales[k_elem + 11])) * w11;
                acc += (float(A[k_elem + 12]) / float(awq_scales[k_elem + 12])) * w12;
                acc += (float(A[k_elem + 13]) / float(awq_scales[k_elem + 13])) * w13;
                acc += (float(A[k_elem + 14]) / float(awq_scales[k_elem + 14])) * w14;
                acc += (float(A[k_elem + 15]) / float(awq_scales[k_elem + 15])) * w15;
            } else {
                acc += float(A[k_elem])      * w0  + float(A[k_elem + 1])  * w1
                     + float(A[k_elem + 2])  * w2  + float(A[k_elem + 3])  * w3
                     + float(A[k_elem + 4])  * w4  + float(A[k_elem + 5])  * w5
                     + float(A[k_elem + 6])  * w6  + float(A[k_elem + 7])  * w7
                     + float(A[k_elem + 8])  * w8  + float(A[k_elem + 9])  * w9
                     + float(A[k_elem + 10]) * w10 + float(A[k_elem + 11]) * w11
                     + float(A[k_elem + 12]) * w12 + float(A[k_elem + 13]) * w13
                     + float(A[k_elem + 14]) * w14 + float(A[k_elem + 15]) * w15;
            }
        }
    }

    acc = simd_sum(acc);
    if (lane == 0) {
        C[local_tid] = half(acc);
    }
}
