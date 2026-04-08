// ── Batched INT4 AMX matvec for FFN gate+up projections (M=1) ──
//
// Computes gate = x · W_gate^T and up = x · W_up^T in a single dispatch.
// Each TG processes 64 rows of either gate or up projection using AMX.
//
// Threadgroup routing: tgid < N_gate_tgs → gate, else → up.
// N_gate_tgs = ceil(N_gate/64), N_up_tgs = ceil(N_up/64) (N_up == N_gate).
//
// Dispatch: (N_gate_tgs + N_up_tgs, 1, 1) threadgroups, (256, 1, 1) threads.

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
    uint tgid   [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint sgid   [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]])
{
    // Route TG to gate or up projection
    uint n_gate_tgs = (N_gate + AMX_ROWS_PER_TG - 1) / AMX_ROWS_PER_TG;
    device const uchar* B_packed;
    device const half* scales;
    device const half* zeros;
    device half* C;
    uint N_proj;
    uint row_base;

    if (tgid < n_gate_tgs) {
        row_base = tgid * AMX_ROWS_PER_TG;
        N_proj = N_gate;
        B_packed = B_gate_packed; scales = scales_gate; zeros = zeros_gate; C = C_gate;
    } else {
        row_base = (tgid - n_gate_tgs) * AMX_ROWS_PER_TG;
        N_proj = N_gate;  // N_up == N_gate
        B_packed = B_up_packed; scales = scales_up; zeros = zeros_up; C = C_up;
    }

    if (row_base >= N_proj) return;
    uint rows_this_tg = min(AMX_ROWS_PER_TG, N_proj - row_base);
    uint n_block_base = row_base / BLK_N;
    uint k_blocks_total = (K + BLK_K - 1) / BLK_K;
    uint num_groups = (K + group_size - 1) / group_size;

    threadgroup half tg_w[AMX_ROWS_PER_TG * AMX_TILE_K];
    threadgroup half tg_x[AMX_TILE_K];

    simdgroup_matrix<float, 8, 8> acc_mat(0);

    for (uint kt = 0; kt < K; kt += AMX_TILE_K) {
        uint tile_k = min(AMX_TILE_K, K - kt);

        for (uint i = tid; i < tile_k; i += AMX_TG_SIZE) {
            float val = float(A[kt + i]);
            if (has_awq) val /= float(awq_scales[kt + i]);
            tg_x[i] = half(val);
        }

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
            float s0 = float(scales[n_abs * num_groups + g0]);
            float z0 = float(zeros[n_abs * num_groups + g0]);
            float w0 = (float(packed & 0x0F) - z0) * s0;

            uint g1 = (k_abs + 1) / group_size;
            float s1 = (g1 == g0) ? s0 : float(scales[n_abs * num_groups + g1]);
            float z1 = (g1 == g0) ? z0 : float(zeros[n_abs * num_groups + g1]);
            float w1 = (float(packed >> 4) - z1) * s1;

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

    uint sg_row = sgid * AMX_ROWS_PER_SG;
    if (sg_row >= rows_this_tg) return;

    threadgroup float tg_result[AMX_SIMDGROUPS * 64];
    simdgroup_store(acc_mat, tg_result + sgid * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        for (uint r = 0; r < AMX_ROWS_PER_SG; r++) {
            uint n_row = row_base + sg_row + r;
            if (n_row < N_proj) {
                C[n_row] = half(tg_result[sgid * 64 + r]);
            }
        }
    }
}

// ── Batched INT4 AMX matvec for 4 GDN projections (M=1) ─────────
//
// Computes 4 independent matvecs in a single dispatch using AMX:
//   qkv = x · W_qkv^T   (ceil(N0/64) TGs)
//   z   = x · W_z^T      (ceil(N1/64) TGs)
//   a   = x · W_a^T      (ceil(N2/64) TGs)
//   b   = x · W_b^T      (ceil(N3/64) TGs)
//
// All projections share the same input x. Threadgroup index determines
// which projection via cumulative TG-count thresholds.
//
// Dispatch: (ceil(N0/64)+ceil(N1/64)+ceil(N2/64)+ceil(N3/64), 1, 1) TGs,
//           (256, 1, 1) threads.

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
    uint tgid   [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint sgid   [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]])
{
    uint N0 = params.N0;
    uint N1 = params.N1;
    uint N2 = params.N2;
    uint N3 = params.N3;
    uint K  = params.K;
    uint group_size = params.group_size;
    uint has_awq = params.has_awq;

    // TG-count thresholds for routing
    uint t0_tgs = (N0 + AMX_ROWS_PER_TG - 1) / AMX_ROWS_PER_TG;
    uint t1_tgs = t0_tgs + (N1 + AMX_ROWS_PER_TG - 1) / AMX_ROWS_PER_TG;
    uint t2_tgs = t1_tgs + (N2 + AMX_ROWS_PER_TG - 1) / AMX_ROWS_PER_TG;

    // Route to correct projection and compute row_base
    uint N_proj;
    uint row_base;
    device const uchar* B_packed;
    device const half* scales;
    device const half* zeros;
    device half* C;

    if (tgid < t0_tgs) {
        row_base = tgid * AMX_ROWS_PER_TG;
        N_proj = N0;
        B_packed = B0_packed; scales = scales0; zeros = zeros0; C = C0;
    } else if (tgid < t1_tgs) {
        row_base = (tgid - t0_tgs) * AMX_ROWS_PER_TG;
        N_proj = N1;
        B_packed = B1_packed; scales = scales1; zeros = zeros1; C = C1;
    } else if (tgid < t2_tgs) {
        row_base = (tgid - t1_tgs) * AMX_ROWS_PER_TG;
        N_proj = N2;
        B_packed = B2_packed; scales = scales2; zeros = zeros2; C = C2;
    } else {
        row_base = (tgid - t2_tgs) * AMX_ROWS_PER_TG;
        N_proj = N3;
        B_packed = B3_packed; scales = scales3; zeros = zeros3; C = C3;
    }

    if (row_base >= N_proj) return;
    uint rows_this_tg = min(AMX_ROWS_PER_TG, N_proj - row_base);
    uint n_block_base = row_base / BLK_N;
    uint k_blocks_total = (K + BLK_K - 1) / BLK_K;
    uint num_groups = (K + group_size - 1) / group_size;

    threadgroup half tg_w[AMX_ROWS_PER_TG * AMX_TILE_K];
    threadgroup half tg_x[AMX_TILE_K];

    simdgroup_matrix<float, 8, 8> acc_mat(0);

    for (uint kt = 0; kt < K; kt += AMX_TILE_K) {
        uint tile_k = min(AMX_TILE_K, K - kt);

        for (uint i = tid; i < tile_k; i += AMX_TG_SIZE) {
            float val = float(A[kt + i]);
            if (has_awq) val /= float(awq_scales[kt + i]);
            tg_x[i] = half(val);
        }

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
            float s0 = float(scales[n_abs * num_groups + g0]);
            float z0 = float(zeros[n_abs * num_groups + g0]);
            float w0 = (float(packed & 0x0F) - z0) * s0;

            uint g1 = (k_abs + 1) / group_size;
            float s1 = (g1 == g0) ? s0 : float(scales[n_abs * num_groups + g1]);
            float z1 = (g1 == g0) ? z0 : float(zeros[n_abs * num_groups + g1]);
            float w1 = (float(packed >> 4) - z1) * s1;

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

    uint sg_row = sgid * AMX_ROWS_PER_SG;
    if (sg_row >= rows_this_tg) return;

    threadgroup float tg_result[AMX_SIMDGROUPS * 64];
    simdgroup_store(acc_mat, tg_result + sgid * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        for (uint r = 0; r < AMX_ROWS_PER_SG; r++) {
            uint n_row = row_base + sg_row + r;
            if (n_row < N_proj) {
                C[n_row] = half(tg_result[sgid * 64 + r]);
            }
        }
    }
}
