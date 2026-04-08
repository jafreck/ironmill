// ============================================================================
// Fused FFN gate+up+activation for INT4 decode using AMX (M=1)
//
// Computes output[i] = act(x · W_gate^T[i]) * (x · W_up^T[i]) in one dispatch.
// Each TG processes 64 output rows. Two weight dequant phases per K-tile
// (gate + up) with two sets of MMA accumulators.
//
// Uses AMX_TILE_K=64 to fit both gate+up tiles in threadgroup memory (16KB).
// Dispatch: (ceil(N/64), 1, 1) threadgroups, (256, 1, 1) threads.
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
    uint tgid   [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint sgid   [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]])
{
    uint row_base = tgid * AMX_ROWS_PER_TG;
    if (row_base >= N) return;

    uint rows_this_tg = min(AMX_ROWS_PER_TG, N - row_base);
    uint n_block_base = row_base / BLK_N;
    uint k_blocks_total = (K + BLK_K - 1) / BLK_K;
    uint num_groups = (K + group_size - 1) / group_size;

    // Two weight tiles + one x tile, using FUSED_FFN_TILE_K to fit in 16KB
    threadgroup half tg_w_gate[AMX_ROWS_PER_TG * FUSED_FFN_TILE_K]; // 8 KB
    threadgroup half tg_w_up[AMX_ROWS_PER_TG * FUSED_FFN_TILE_K];   // 8 KB
    threadgroup half tg_x[FUSED_FFN_TILE_K];                          // 128 B

    simdgroup_matrix<float, 8, 8> gate_acc(0);
    simdgroup_matrix<float, 8, 8> up_acc(0);

    for (uint kt = 0; kt < K; kt += FUSED_FFN_TILE_K) {
        uint tile_k = min(FUSED_FFN_TILE_K, K - kt);

        // Load x
        for (uint i = tid; i < tile_k; i += AMX_TG_SIZE) {
            float val = float(A[kt + i]);
            if (has_awq) val /= float(awq_scales[kt + i]);
            tg_x[i] = half(val);
        }

        // Cooperative INT4 dequant for both gate and up weights
        uint half_tile_k = tile_k / 2;
        uint total_pairs = rows_this_tg * half_tile_k;
        for (uint i = tid; i < total_pairs; i += AMX_TG_SIZE) {
            uint n_local = i / half_tile_k;
            uint kp      = i % half_tile_k;
            uint k_abs   = kt + kp * 2;
            uint n_abs   = row_base + n_local;

            uint kb_idx   = k_abs / BLK_K;
            uint k_local  = k_abs % BLK_K;
            uint byte_idx = (n_block_base * k_blocks_total + kb_idx) * (BLK_N * BLK_K / 2)
                          + n_local * (BLK_K / 2)
                          + k_local / 2;

            // Gate weights
            uchar g_packed = B_gate_packed[byte_idx];
            uint g0 = k_abs / group_size;
            float sg0 = float(scales_gate[n_abs * num_groups + g0]);
            float zg0 = float(zeros_gate[n_abs * num_groups + g0]);
            float gw0 = (float(g_packed & 0x0F) - zg0) * sg0;

            uint g1 = (k_abs + 1) / group_size;
            float sg1 = (g1 == g0) ? sg0 : float(scales_gate[n_abs * num_groups + g1]);
            float zg1 = (g1 == g0) ? zg0 : float(zeros_gate[n_abs * num_groups + g1]);
            float gw1 = (float(g_packed >> 4) - zg1) * sg1;

            tg_w_gate[n_local * FUSED_FFN_TILE_K + kp * 2]     = half(gw0);
            tg_w_gate[n_local * FUSED_FFN_TILE_K + kp * 2 + 1] = half(gw1);

            // Up weights
            uchar u_packed = B_up_packed[byte_idx];
            float su0 = float(scales_up[n_abs * num_groups + g0]);
            float zu0 = float(zeros_up[n_abs * num_groups + g0]);
            float uw0 = (float(u_packed & 0x0F) - zu0) * su0;

            float su1 = (g1 == g0) ? su0 : float(scales_up[n_abs * num_groups + g1]);
            float zu1 = (g1 == g0) ? zu0 : float(zeros_up[n_abs * num_groups + g1]);
            float uw1 = (float(u_packed >> 4) - zu1) * su1;

            tg_w_up[n_local * FUSED_FFN_TILE_K + kp * 2]     = half(uw0);
            tg_w_up[n_local * FUSED_FFN_TILE_K + kp * 2 + 1] = half(uw1);
        }
        if (tile_k < FUSED_FFN_TILE_K) {
            for (uint i = tid; i < rows_this_tg; i += AMX_TG_SIZE) {
                for (uint j = tile_k; j < FUSED_FFN_TILE_K; j++) {
                    tg_w_gate[i * FUSED_FFN_TILE_K + j] = 0;
                    tg_w_up[i * FUSED_FFN_TILE_K + j] = 0;
                }
            }
            for (uint i = tile_k + tid; i < FUSED_FFN_TILE_K; i += AMX_TG_SIZE) {
                tg_x[i] = 0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: simdgroup MMA for both gate and up
        uint sg_row = sgid * AMX_ROWS_PER_SG;
        if (sg_row < rows_this_tg) {
            uint n_k_tiles = (tile_k + 7) / 8;
            for (uint kb = 0; kb < n_k_tiles; kb++) {
                simdgroup_matrix<half, 8, 8> x_mat;
                simdgroup_load(x_mat, tg_x + kb * 8, 0);

                simdgroup_matrix<half, 8, 8> wg_T;
                simdgroup_load(wg_T, tg_w_gate + sg_row * FUSED_FFN_TILE_K + kb * 8,
                               FUSED_FFN_TILE_K, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(gate_acc, x_mat, wg_T, gate_acc);

                simdgroup_matrix<half, 8, 8> wu_T;
                simdgroup_load(wu_T, tg_w_up + sg_row * FUSED_FFN_TILE_K + kb * 8,
                               FUSED_FFN_TILE_K, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(up_acc, x_mat, wu_T, up_acc);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Extract results and apply activation
    uint sg_row = sgid * AMX_ROWS_PER_SG;
    if (sg_row >= rows_this_tg) return;

    threadgroup float tg_gate_result[AMX_SIMDGROUPS * 64];
    threadgroup float tg_up_result[AMX_SIMDGROUPS * 64];
    simdgroup_store(gate_acc, tg_gate_result + sgid * 64, 8);
    simdgroup_store(up_acc, tg_up_result + sgid * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        for (uint r = 0; r < AMX_ROWS_PER_SG; r++) {
            uint n_row = row_base + sg_row + r;
            if (n_row < N) {
                float gate_val = tg_gate_result[sgid * 64 + r];
                float up_val = tg_up_result[sgid * 64 + r];
                float act;
                if (use_gelu) {
                    const float kSqrt2OverPi = 0.7978845608f;
                    float inner = kSqrt2OverPi * (gate_val + 0.044715f * gate_val * gate_val * gate_val);
                    inner = clamp(inner, -10.0f, 10.0f);
                    act = 0.5f * gate_val * (1.0f + precise::tanh(inner));
                } else {
                    act = gate_val / (1.0f + exp(-gate_val));
                }
                C[n_row] = half(act * up_val);
            }
        }
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

// ============================================================================
// AMX-accelerated INT4×Q8 matvec (decode path, M=1)
//
// Two-phase approach per K-tile (same as affine_matvec_int4_amx):
//   Phase 1a: Dequant Q8 input to FP16 in threadgroup memory.
//   Phase 1b: All 256 threads cooperatively dequant a [64 x TILE_K] INT4 tile
//             into FP16 threadgroup memory (zero is integer-rounded).
//   Phase 2: Each of 8 simdgroups uses simdgroup_matrix_multiply_accumulate
//            (Apple AMX hardware) on its 8-row slice.
//
// Same buffer layout as the scalar affine_matvec_int4xq8 for drop-in replacement.
// Dispatch: (ceil(N/64), 1, 1) threadgroups, (256, 1, 1) threads.
// ============================================================================

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
    uint tgid   [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint sgid   [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]])
{
    uint row_base = tgid * AMX_ROWS_PER_TG;
    if (row_base >= N) return;

    uint rows_this_tg = min(AMX_ROWS_PER_TG, N - row_base);
    uint n_block_base = row_base / BLK_N;
    uint k_blocks_total = (K + BLK_K - 1) / BLK_K;
    uint num_groups = (K + group_size - 1) / group_size;

    threadgroup half tg_w[AMX_ROWS_PER_TG * AMX_TILE_K]; // 16 KB
    threadgroup half tg_x[AMX_TILE_K];                    // 256 B

    simdgroup_matrix<float, 8, 8> acc_mat(0);

    for (uint kt = 0; kt < K; kt += AMX_TILE_K) {
        uint tile_k = min(AMX_TILE_K, K - kt);

        // -- Phase 1a: Load Q8 input into threadgroup FP16 --
        for (uint i = tid; i < tile_k; i += AMX_TG_SIZE) {
            uint k_abs = kt + i;
            float q8_val = float(A_q8[k_abs]);
            float a_scale = A_scales[k_abs / q8_group_size];
            tg_x[i] = half(q8_val * a_scale);
        }

        // -- Phase 1b: Cooperative INT4 dequant (integer-rounded zero) --
        uint half_tile_k = tile_k / 2;
        uint total_pairs = rows_this_tg * half_tile_k;
        for (uint i = tid; i < total_pairs; i += AMX_TG_SIZE) {
            uint n_local = i / half_tile_k;
            uint kp      = i % half_tile_k;
            uint k_abs   = kt + kp * 2;
            uint n_abs   = row_base + n_local;

            uint kb_idx   = k_abs / BLK_K;
            uint k_local  = k_abs % BLK_K;
            uint byte_idx = (n_block_base * k_blocks_total + kb_idx) * (BLK_N * BLK_K / 2)
                          + n_local * (BLK_K / 2)
                          + k_local / 2;

            uchar packed = B_packed[byte_idx];

            uint g0 = k_abs / group_size;
            float s0 = float(w_scales[n_abs * num_groups + g0]);
            int iz0 = int(rint(float(w_zeros[n_abs * num_groups + g0])));
            float w0 = float(int(packed & 0x0F) - iz0) * s0;

            uint g1 = (k_abs + 1) / group_size;
            float s1 = (g1 == g0) ? s0 : float(w_scales[n_abs * num_groups + g1]);
            int iz1 = (g1 == g0) ? iz0 : int(rint(float(w_zeros[n_abs * num_groups + g1])));
            float w1 = float(int(packed >> 4) - iz1) * s1;

            tg_w[n_local * AMX_TILE_K + kp * 2]     = half(w0);
            tg_w[n_local * AMX_TILE_K + kp * 2 + 1] = half(w1);
        }
        if (tile_k < AMX_TILE_K) {
            for (uint i = tid; i < rows_this_tg; i += AMX_TG_SIZE) {
                for (uint j = tile_k; j < AMX_TILE_K; j++) {
                    tg_w[i * AMX_TILE_K + j] = 0;
                }
            }
            for (uint i = tile_k + tid; i < AMX_TILE_K; i += AMX_TG_SIZE) {
                tg_x[i] = 0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -- Phase 2: simdgroup matrix multiply (AMX) --
        uint sg_row = sgid * AMX_ROWS_PER_SG;
        if (sg_row < rows_this_tg) {
            uint n_k_tiles = (tile_k + 7) / 8;
            for (uint kb = 0; kb < n_k_tiles; kb++) {
                simdgroup_matrix<half, 8, 8> w_T;
                simdgroup_load(w_T, tg_w + sg_row * AMX_TILE_K + kb * 8,
                               AMX_TILE_K, ulong2(0, 0), true);

                simdgroup_matrix<half, 8, 8> x_mat;
                simdgroup_load(x_mat, tg_x + kb * 8, 0);

                simdgroup_multiply_accumulate(acc_mat, x_mat, w_T, acc_mat);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // -- Extract and store results --
    uint sg_row = sgid * AMX_ROWS_PER_SG;
    if (sg_row >= rows_this_tg) return;

    threadgroup float tg_result[AMX_SIMDGROUPS * 64];
    simdgroup_store(acc_mat, tg_result + sgid * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        for (uint r = 0; r < AMX_ROWS_PER_SG; r++) {
            uint n_row = row_base + sg_row + r;
            if (n_row < N) {
                C[n_row] = half(tg_result[sgid * 64 + r]);
            }
        }
    }
}
