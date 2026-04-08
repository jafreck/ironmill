// ============================================================================
// Fused FFN gate+up+SiLU for INT4 decode (M=1)
//
// Computes output[i] = silu(x · W_gate^T[i]) * (x · W_up^T[i]) in one dispatch.
// Eliminates the intermediate gate/up buffer writes and the separate activation
// dispatch. Each threadgroup computes one output row by running BOTH the gate
// and up dot products, then applies gated activation inline.
//
// Same buffer layout as batched_affine_matvec_int4, plus an activation mode flag.
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

    uint half_K = K / 2;
    uint num_groups = (K + group_size - 1) / group_size;
    uint scale_row = tid * num_groups;

    // Blocked layout constants
    uint k_blocks      = (K + BLK_K - 1) / BLK_K;
    uint local_k_bytes = BLK_K / 2;
    uint block_bytes   = BLK_N * local_k_bytes;
    uint n_block = tid / BLK_N;
    uint n_local = tid % BLK_N;

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;

    for (uint k = lane; k < half_K; k += 32) {
        uint kb = k / local_k_bytes;
        uint b  = k % local_k_bytes;
        uint byte_idx = (n_block * k_blocks + kb) * block_bytes
                      + n_local * local_k_bytes + b;

        // Gate weight dequant
        uchar g_packed = B_gate_packed[byte_idx];
        uchar g_lo = g_packed & 0x0F;
        uchar g_hi = (g_packed >> 4) & 0x0F;

        // Up weight dequant
        uchar u_packed = B_up_packed[byte_idx];
        uchar u_lo = u_packed & 0x0F;
        uchar u_hi = (u_packed >> 4) & 0x0F;

        uint k2 = k * 2;
        uint g0 = k2 / group_size;
        uint g1 = (k2 + 1) / group_size;

        // Shared scale/zero (gate and up have same group structure)
        float sg0 = float(scales_gate[scale_row + g0]);
        float zg0 = float(zeros_gate[scale_row + g0]);
        float sg1 = float(scales_gate[scale_row + g1]);
        float zg1 = float(zeros_gate[scale_row + g1]);

        float su0 = float(scales_up[scale_row + g0]);
        float zu0 = float(zeros_up[scale_row + g0]);
        float su1 = float(scales_up[scale_row + g1]);
        float zu1 = float(zeros_up[scale_row + g1]);

        float gw0 = (float(g_lo) - zg0) * sg0;
        float gw1 = (float(g_hi) - zg1) * sg1;
        float uw0 = (float(u_lo) - zu0) * su0;
        float uw1 = (float(u_hi) - zu1) * su1;

        float x0, x1;
        if (has_awq) {
            x0 = float(A[k2])     / float(awq_scales[k2]);
            x1 = float(A[k2 + 1]) / float(awq_scales[k2 + 1]);
        } else {
            x0 = float(A[k2]);
            x1 = float(A[k2 + 1]);
        }

        gate_acc += x0 * gw0 + x1 * gw1;
        up_acc   += x0 * uw0 + x1 * uw1;
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

// ── INT4×Q8 integer dot product matvec (decode path, M=1) ───────
//
// Uses pre-quantized INT8 input (from quantize_input_q8) to replace float
// dequant with integer multiply-add. The per-group Q8 scale is combined
// with the weight scale in the final reduction.
//
// B_packed is in blocked layout: [N_blocks, K_blocks, BLK_N, BLK_K/2]
// (same as affine_matvec_int4).
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

    uint half_K = K / 2;
    uint num_groups = (K + group_size - 1) / group_size;
    uint scale_row = tid * num_groups;

    // Blocked layout addressing (same as affine_matvec_int4)
    uint k_blocks      = (K + BLK_K - 1) / BLK_K;
    uint local_k_bytes = BLK_K / 2;             // 4
    uint block_bytes   = BLK_N * local_k_bytes;  // 256
    uint n_block = tid / BLK_N;
    uint n_local = tid % BLK_N;

    float acc = 0.0f;
    uint prev_wg = 0xFFFFFFFF;
    float ws = 0.0f;
    float wz = 0.0f;

    for (uint k = lane; k < half_K; k += 32) {
        uint kb = k / local_k_bytes;
        uint b  = k % local_k_bytes;
        uint byte_idx = (n_block * k_blocks + kb) * block_bytes
                      + n_local * local_k_bytes + b;

        uchar packed = B_packed[byte_idx];
        int lo = int(packed & 0x0F);
        int hi = int((packed >> 4) & 0x0F);

        uint k2 = k * 2;
        uint wg0 = k2 / group_size;
        uint wg1 = (k2 + 1) / group_size;
        uint ag0 = k2 / q8_group_size;
        uint ag1 = (k2 + 1) / q8_group_size;

        // Weight group 0
        if (wg0 != prev_wg) {
            ws = float(w_scales[scale_row + wg0]);
            wz = float(w_zeros[scale_row + wg0]);
            prev_wg = wg0;
        }
        int iz = int(rint(wz));
        int w0 = lo - iz;
        float a_scale0 = A_scales[ag0];
        acc += float(int(A_q8[k2])) * float(w0) * ws * a_scale0;

        // Weight group 1 (may differ if crossing group boundary)
        if (wg1 != wg0) {
            ws = float(w_scales[scale_row + wg1]);
            wz = float(w_zeros[scale_row + wg1]);
            prev_wg = wg1;
            iz = int(rint(wz));
        }
        int w1 = hi - iz;
        float a_scale1 = A_scales[ag1];
        acc += float(int(A_q8[k2 + 1])) * float(w1) * ws * a_scale1;
    }

    acc = simd_sum(acc);

    if (lane == 0) {
        C[tid] = half(acc);
    }
}
