// ============================================================================
// Fused FFN gate+up+SiLU for INT4 decode (M=1)
//
// Computes output[i] = silu(x . W_gate^T[i]) * (x . W_up^T[i]) in one dispatch.
// Eliminates the intermediate gate/up buffer writes and the separate activation
// dispatch. Each threadgroup computes one output row by running BOTH the gate
// and up dot products, then applies gated activation inline.
//
// Same buffer layout as batched_affine_matvec_int4, plus an activation mode flag.
// B_packed blocked layout guarantees 4-byte alignment per (n_local, k_block)
// slot, making the uint32 cast safe. Assumes group_size >= BLK_K.
//
// Dispatch: (N, 1, 1) threadgroups, (32, 1, 1) threads.
// ============================================================================

kernel void fused_ffn_gate_up_act_int4(
    device const half *A              [[buffer(0)]],    // [1, K] shared input
    device const uchar *B_gate_packed [[buffer(1)]],    // gate weights (blocked)
    device const half *scales_gate    [[buffer(2)]],
    device const half *zeros_gate     [[buffer(3)]],
    device const uchar *B_up_packed   [[buffer(4)]],    // up weights (blocked)
    device const half *scales_up      [[buffer(5)]],
    device const half *zeros_up       [[buffer(6)]],
    device half *C                    [[buffer(7)]],    // [1, N] fused output
    constant uint &N                  [[buffer(8)]],
    constant uint &K                  [[buffer(9)]],
    constant uint &group_size         [[buffer(10)]],
    device const half *awq_scales     [[buffer(11)]],
    constant uint &has_awq            [[buffer(12)]],
    constant uint &use_gelu           [[buffer(13)]],   // 0=SiLU, 1=GELU-tanh
    uint tid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (tid >= N) return;

    uint num_groups = (K + group_size - 1) / group_size;
    uint scale_row = tid * num_groups;

    // Blocked layout: word-aligned addressing
    uint k_blocks = (K + BLK_K - 1) / BLK_K;
    uint n_block = tid / BLK_N;
    uint n_local = tid % BLK_N;

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;

    for (uint kb = lane; kb < k_blocks; kb += 32) {
        uint word_idx = (n_block * k_blocks + kb) * BLK_N + n_local;

        // Gate weight: vectorized uint32 load
        uint g_packed4 = ((device const uint*)B_gate_packed)[word_idx];
        // Up weight: vectorized uint32 load
        uint u_packed4 = ((device const uint*)B_up_packed)[word_idx];

        // Pre-fetch scale/zero per group
        uint k_elem = kb * BLK_K;
        uint grp = k_elem / group_size;

        float sg = float(scales_gate[scale_row + grp]);
        float zg = float(zeros_gate[scale_row + grp]);
        float su = float(scales_up[scale_row + grp]);
        float zu = float(zeros_up[scale_row + grp]);

        // Unpack 8 nibbles from each weight stream
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

        // Load 8 input values
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
            x0 = float(A[k_elem]);
            x1 = float(A[k_elem + 1]);
            x2 = float(A[k_elem + 2]);
            x3 = float(A[k_elem + 3]);
            x4 = float(A[k_elem + 4]);
            x5 = float(A[k_elem + 5]);
            x6 = float(A[k_elem + 6]);
            x7 = float(A[k_elem + 7]);
        }

        gate_acc += x0 * gw0 + x1 * gw1 + x2 * gw2 + x3 * gw3
                  + x4 * gw4 + x5 * gw5 + x6 * gw6 + x7 * gw7;
        up_acc   += x0 * uw0 + x1 * uw1 + x2 * uw2 + x3 * uw3
                  + x4 * uw4 + x5 * uw5 + x6 * uw6 + x7 * uw7;
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

// ---- INT4xQ8 integer dot product matvec (decode path, M=1) ----
//
// Uses pre-quantized INT8 input (from quantize_input_q8) to replace float
// dequant with integer multiply-add. The per-group Q8 scale is combined
// with the weight scale in the final reduction.
//
// Dispatch: (N, 1, 1) threadgroups, (32, 1, 1) threads per group.

kernel void affine_matvec_int4xq8(
    device const char *A_q8         [[buffer(0)]],   // [K] int8
    device const float *A_scales    [[buffer(1)]],   // [K/q8_group_size] float
    device const uchar *B_packed    [[buffer(2)]],   // blocked [N_blk, K_blk, 64, 4]
    device const half *w_scales     [[buffer(3)]],   // [N, num_groups]
    device const half *w_zeros      [[buffer(4)]],   // [N, num_groups]
    device half *C                  [[buffer(5)]],   // [1, N]
    constant uint &N                [[buffer(6)]],
    constant uint &K                [[buffer(7)]],
    constant uint &group_size       [[buffer(8)]],   // weight group size
    constant uint &q8_group_size    [[buffer(9)]],   // Q8 input group size
    uint tid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (tid >= N) return;

    uint num_groups = (K + group_size - 1) / group_size;
    uint scale_row = tid * num_groups;

    // Blocked layout: word-aligned addressing
    uint k_blocks = (K + BLK_K - 1) / BLK_K;
    uint n_block = tid / BLK_N;
    uint n_local = tid % BLK_N;

    float acc = 0.0f;
    uint prev_wg = 0xFFFFFFFF;
    float ws = 0.0f;
    float wz = 0.0f;
    int iz = 0;

    for (uint kb = lane; kb < k_blocks; kb += 32) {
        uint word_idx = (n_block * k_blocks + kb) * BLK_N + n_local;
        uint packed4 = ((device const uint*)B_packed)[word_idx];

        uint k_elem = kb * BLK_K;
        uint wg = k_elem / group_size;

        // Cache weight scale/zero per group
        if (wg != prev_wg) {
            ws = float(w_scales[scale_row + wg]);
            wz = float(w_zeros[scale_row + wg]);
            iz = int(rint(wz));
            prev_wg = wg;
        }

        // Unpack 8 nibbles
        int lo0 = int(packed4 & 0xF);
        int lo1 = int((packed4 >> 4) & 0xF);
        int lo2 = int((packed4 >> 8) & 0xF);
        int lo3 = int((packed4 >> 12) & 0xF);
        int lo4 = int((packed4 >> 16) & 0xF);
        int lo5 = int((packed4 >> 20) & 0xF);
        int lo6 = int((packed4 >> 24) & 0xF);
        int lo7 = int((packed4 >> 28) & 0xF);

        // Q8 input scale lookups
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

    acc = simd_sum(acc);

    if (lane == 0) {
        C[tid] = half(acc);
    }
}
