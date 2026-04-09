// ── Superblock fused FFN gate+up+SiLU for INT4 decode (M=1) ─────
//
// Same fusion as fused_ffn_gate_up_act_int4 but reads from superblock layout.
// Gate and up projections each have their own superblock buffer.
//
// Buffer binding: W_gate and W_up are superblock buffers (inline scale/zero).
// Dispatch: (N, 1, 1) threadgroups, (32, 1, 1) threads.

kernel void superblock_fused_ffn_gate_up_act_int4(
    device const half *A            [[buffer(0)]],   // [1, K] shared input
    device const uchar *W_gate      [[buffer(1)]],   // gate superblocks
    device const uchar *W_up        [[buffer(2)]],   // up superblocks
    device half *C                  [[buffer(3)]],   // [1, N] fused output
    constant uint &N                [[buffer(4)]],
    constant uint &K                [[buffer(5)]],
    device const half *awq_scales   [[buffer(6)]],
    constant uint &has_awq          [[buffer(7)]],
    constant uint &use_gelu         [[buffer(8)]],
    uint tid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (tid >= N) return;

    uint num_groups = K / GS;
    uint sb_stride = num_groups * SB_BYTES_INT4;

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;

    for (uint g = 0; g < num_groups; g++) {
        device const uchar *sb_gate = W_gate + tid * sb_stride + g * SB_BYTES_INT4;
        device const uchar *sb_up   = W_up   + tid * sb_stride + g * SB_BYTES_INT4;

        float sg = float(*(device const half *)(sb_gate));
        float zg = float(*(device const half *)(sb_gate + 2));
        float su = float(*(device const half *)(sb_up));
        float zu = float(*(device const half *)(sb_up + 2));

        uint k_base = g * GS;

        for (uint i = lane * BLK_K; i < GS; i += 32 * BLK_K) {
            uint k_elem = k_base + i;
            uint gw_idx = (i / 2) / 4;
            uint uw_idx = gw_idx;
            uint g_packed4 = ((device const uint*)(sb_gate + SB_HEADER_BYTES))[gw_idx];
            uint u_packed4 = ((device const uint*)(sb_up + SB_HEADER_BYTES))[uw_idx];

            float gw0 = (float(g_packed4 & 0xF) - zg) * sg;
            float gw1 = (float((g_packed4 >> 4) & 0xF) - zg) * sg;
            float gw2 = (float((g_packed4 >> 8) & 0xF) - zg) * sg;
            float gw3 = (float((g_packed4 >> 12) & 0xF) - zg) * sg;
            float gw4 = (float((g_packed4 >> 16) & 0xF) - zg) * sg;
            float gw5 = (float((g_packed4 >> 20) & 0xF) - zg) * sg;
            float gw6 = (float((g_packed4 >> 24) & 0xF) - zg) * sg;
            float gw7 = (float((g_packed4 >> 28) & 0xF) - zg) * sg;

            float uw0 = (float(u_packed4 & 0xF) - zu) * su;
            float uw1 = (float((u_packed4 >> 4) & 0xF) - zu) * su;
            float uw2 = (float((u_packed4 >> 8) & 0xF) - zu) * su;
            float uw3 = (float((u_packed4 >> 12) & 0xF) - zu) * su;
            float uw4 = (float((u_packed4 >> 16) & 0xF) - zu) * su;
            float uw5 = (float((u_packed4 >> 20) & 0xF) - zu) * su;
            float uw6 = (float((u_packed4 >> 24) & 0xF) - zu) * su;
            float uw7 = (float((u_packed4 >> 28) & 0xF) - zu) * su;

            float x0, x1, x2, x3, x4, x5, x6, x7;
            if (has_awq) {
                x0 = float(A[k_elem])     / float(awq_scales[k_elem]);
                x1 = float(A[k_elem + 1]) / float(awq_scales[k_elem + 1]);
                x2 = float(A[k_elem + 2]) / float(awq_scales[k_elem + 2]);
                x3 = float(A[k_elem + 3]) / float(awq_scales[k_elem + 3]);
                x4 = float(A[k_elem + 4]) / float(awq_scales[k_elem + 4]);
                x5 = float(A[k_elem + 5]) / float(awq_scales[k_elem + 5]);
                x6 = float(A[k_elem + 6]) / float(awq_scales[k_elem + 6]);
                x7 = float(A[k_elem + 7]) / float(awq_scales[k_elem + 7]);
            } else {
                x0 = float(A[k_elem]);     x1 = float(A[k_elem + 1]);
                x2 = float(A[k_elem + 2]); x3 = float(A[k_elem + 3]);
                x4 = float(A[k_elem + 4]); x5 = float(A[k_elem + 5]);
                x6 = float(A[k_elem + 6]); x7 = float(A[k_elem + 7]);
            }

            gate_acc += x0*gw0 + x1*gw1 + x2*gw2 + x3*gw3
                      + x4*gw4 + x5*gw5 + x6*gw6 + x7*gw7;
            up_acc   += x0*uw0 + x1*uw1 + x2*uw2 + x3*uw3
                      + x4*uw4 + x5*uw5 + x6*uw6 + x7*uw7;
        }
    }

    gate_acc = simd_sum(gate_acc);
    up_acc   = simd_sum(up_acc);

    if (lane == 0) {
        float act;
        if (use_gelu) {
            const float kSqrt2OverPi = 0.7978845608f;
            float inner = kSqrt2OverPi * (gate_acc + 0.044715f * gate_acc * gate_acc * gate_acc);
            inner = clamp(inner, -10.0f, 10.0f);
            act = 0.5f * gate_acc * (1.0f + precise::tanh(inner));
        } else {
            act = gate_acc / (1.0f + exp(-gate_acc));
        }
        C[tid] = half(act * up_acc);
    }
}

// ── Superblock INT4×Q8 matvec (decode path, M=1) ─────────────────
//
// Same as affine_matvec_int4xq8 but reads from superblock layout.
// Dispatch: (N, 1, 1) threadgroups, (32, 1, 1) threads per group.

kernel void superblock_affine_matvec_int4xq8(
    device const char *A_q8         [[buffer(0)]],   // [K] int8
    device const float *A_scales    [[buffer(1)]],   // [K/q8_group_size] float
    device const uchar *W           [[buffer(2)]],   // [N, G, sb_bytes] superblocks
    device half *C                  [[buffer(3)]],   // [1, N]
    constant uint &N                [[buffer(4)]],
    constant uint &K                [[buffer(5)]],
    constant uint &q8_group_size    [[buffer(6)]],
    uint tid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (tid >= N) return;

    uint num_groups = K / GS;
    uint sb_stride = num_groups * SB_BYTES_INT4;

    float acc = 0.0f;

    for (uint g = 0; g < num_groups; g++) {
        device const uchar *sb = W + tid * sb_stride + g * SB_BYTES_INT4;
        float ws = float(*(device const half *)(sb));
        float wz = float(*(device const half *)(sb + 2));
        int iz = int(rint(wz));

        uint k_base = g * GS;

        for (uint i = lane * BLK_K; i < GS; i += 32 * BLK_K) {
            uint k_elem = k_base + i;
            uint word_idx = i / 8;
            uint packed4 = ((device const uint*)(sb + SB_HEADER_BYTES))[word_idx];

            int lo0 = int(packed4 & 0xF);
            int lo1 = int((packed4 >> 4) & 0xF);
            int lo2 = int((packed4 >> 8) & 0xF);
            int lo3 = int((packed4 >> 12) & 0xF);
            int lo4 = int((packed4 >> 16) & 0xF);
            int lo5 = int((packed4 >> 20) & 0xF);
            int lo6 = int((packed4 >> 24) & 0xF);
            int lo7 = int((packed4 >> 28) & 0xF);

            uint ag0 = k_elem / q8_group_size;
            uint ag4 = (k_elem + 4) / q8_group_size;
            float a_s0 = A_scales[ag0];
            float a_s4 = (ag4 != ag0) ? A_scales[ag4] : a_s0;

            acc += float(int(A_q8[k_elem]))     * float(lo0 - iz) * ws * a_s0;
            acc += float(int(A_q8[k_elem + 1])) * float(lo1 - iz) * ws * a_s0;
            acc += float(int(A_q8[k_elem + 2])) * float(lo2 - iz) * ws * a_s0;
            acc += float(int(A_q8[k_elem + 3])) * float(lo3 - iz) * ws * a_s0;
            acc += float(int(A_q8[k_elem + 4])) * float(lo4 - iz) * ws * a_s4;
            acc += float(int(A_q8[k_elem + 5])) * float(lo5 - iz) * ws * a_s4;
            acc += float(int(A_q8[k_elem + 6])) * float(lo6 - iz) * ws * a_s4;
            acc += float(int(A_q8[k_elem + 7])) * float(lo7 - iz) * ws * a_s4;
        }
    }

    acc = simd_sum(acc);
    if (lane == 0) {
        C[tid] = half(acc);
    }
}
