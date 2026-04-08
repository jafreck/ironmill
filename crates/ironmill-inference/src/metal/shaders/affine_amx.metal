// ============================================================================
// AMX-accelerated INT4 matvec (decode path, M=1)
//
// Two-phase approach per K-tile:
//   Phase 1: All 256 threads cooperatively dequant a [64 x TILE_K] INT4 tile
//            into FP16 threadgroup memory.
//   Phase 2: Each of 8 simdgroups uses simdgroup_matrix_multiply_accumulate
//            (Apple AMX hardware) on its 8-row slice.
//
// This replaces the scalar inner loop with hardware-accelerated 8x8 matrix
// multiply, yielding ~4x higher throughput at the cost of one threadgroup
// barrier per K-tile.
//
// Same buffer layout as affine_matvec_int4 for drop-in replacement.
// Dispatch: (ceil(N/64), 1, 1) threadgroups, (256, 1, 1) threads.
// ============================================================================

kernel void affine_matvec_int4_amx(
    device const half *A              [[buffer(0)]],
    device const uchar *B_packed      [[buffer(1)]],
    device const half *scales         [[buffer(2)]],
    device const half *zeros          [[buffer(3)]],
    device half *C                    [[buffer(4)]],
    constant uint &N                  [[buffer(5)]],
    constant uint &K                  [[buffer(6)]],
    constant uint &group_size         [[buffer(7)]],
    device const half *awq_scales     [[buffer(8)]],
    constant uint &has_awq            [[buffer(9)]],
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

        // -- Phase 1a: Load x into threadgroup memory --
        for (uint i = tid; i < tile_k; i += AMX_TG_SIZE) {
            float val = float(A[kt + i]);
            if (has_awq) val /= float(awq_scales[kt + i]);
            tg_x[i] = half(val);
        }

        // -- Phase 1b: Cooperative INT4 dequant --
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


// ============================================================================
// AMX-accelerated INT8 matvec (decode path, M=1)
//
// Same two-phase approach as affine_matvec_int4_amx but for INT8 weights.
// Phase 1: All 256 threads cooperatively dequant a [64 x TILE_K] INT8 tile.
// Phase 2: 8 simdgroups use simdgroup_matrix_multiply_accumulate (AMX).
//
// INT8 difference: each byte is one element (vs INT4 where each byte is two).
// Blocked layout: [N/BLK_N, K/BLK_K, BLK_N, BLK_K]
//
// Same buffer layout as affine_matvec_int8 for drop-in replacement.
// Dispatch: (ceil(N/64), 1, 1) threadgroups, (256, 1, 1) threads.
// ============================================================================

kernel void affine_matvec_int8_amx(
    device const half *A              [[buffer(0)]],
    device const uchar *B_packed      [[buffer(1)]],
    device const half *scales         [[buffer(2)]],
    device const half *zeros          [[buffer(3)]],
    device half *C                    [[buffer(4)]],
    constant uint &N                  [[buffer(5)]],
    constant uint &K                  [[buffer(6)]],
    constant uint &group_size         [[buffer(7)]],
    device const half *awq_scales     [[buffer(8)]],
    constant uint &has_awq            [[buffer(9)]],
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

        // -- Phase 1a: Load x into threadgroup memory --
        for (uint i = tid; i < tile_k; i += AMX_TG_SIZE) {
            float val = float(A[kt + i]);
            if (has_awq) val /= float(awq_scales[kt + i]);
            tg_x[i] = half(val);
        }

        // -- Phase 1b: Cooperative INT8 dequant --
        uint total_elems = rows_this_tg * tile_k;
        for (uint i = tid; i < total_elems; i += AMX_TG_SIZE) {
            uint n_local = i / tile_k;
            uint k_elem  = i % tile_k;
            uint k_abs   = kt + k_elem;
            uint n_abs   = row_base + n_local;

            uint kb_idx   = k_abs / BLK_K;
            uint k_local  = k_abs % BLK_K;
            uint byte_idx = (n_block_base * k_blocks_total + kb_idx) * (BLK_N * BLK_K)
                          + n_local * BLK_K
                          + k_local;

            uchar q = B_packed[byte_idx];
            uint grp = k_abs / group_size;
            float s = float(scales[n_abs * num_groups + grp]);
            float z = float(zeros[n_abs * num_groups + grp]);
            float w = (float(q) - z) * s;

            tg_w[n_local * AMX_TILE_K + k_elem] = half(w);
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
