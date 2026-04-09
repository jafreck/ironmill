// ── INT4 superblock tiled GEMM (prefill path, M>1) ──────────────
//
// Same MMA-based tiled GEMM structure as affine_matmul_int4, but the B-tile
// load reads from superblock layout with inline scale/zero instead of
// blocked layout with separate scale/zero arrays.
//
// Dispatch: (ceil(M/TM_TILE), ceil(N/TN_TILE), 1) threadgroups,
//           (256, 1, 1) threads per group.

kernel void superblock_matmul_int4(
    device const half *A            [[buffer(0)]],   // [M, K]
    device const uchar *W           [[buffer(1)]],   // [N, K/2] contiguous data
    device half *C                  [[buffer(2)]],   // [M, N]
    constant uint &M                [[buffer(3)]],
    constant uint &N                [[buffer(4)]],
    constant uint &K                [[buffer(5)]],
    device const half *awq_scales   [[buffer(6)]],   // [K] or dummy
    constant uint &has_awq          [[buffer(7)]],
    device const half *W_scales     [[buffer(8)]],   // [N, K/GS]
    device const half *W_zeros      [[buffer(9)]],   // [N, K/GS]
    uint2 group_id [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    uint tg_m = group_id.x * TM_TILE;
    uint tg_n = group_id.y * TN_TILE;

    threadgroup half tg_a[2][TM_TILE * MATMUL_K_TILE];
    threadgroup half tg_bt[2][MATMUL_K_TILE * TN_STRIDE];

    uint num_groups = K / GS;
    uint num_k_steps = (K + MATMUL_K_TILE - 1) / MATMUL_K_TILE;

    uint sb_stride = num_groups * SB_BYTES_INT4;  // bytes per row

    simdgroup_matrix<float, 8, 8> acc[TN_BLOCKS];
    for (uint j = 0; j < TN_BLOCKS; j++) acc[j] = simdgroup_matrix<float, 8, 8>(0);

    // Prologue: load first tile into buf[0]
    {
        uint k_base = 0;
        // A-tile load (unchanged from original)
        for (uint i = tid; i < TM_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
            uint m = i / MATMUL_K_TILE;
            uint k = i % MATMUL_K_TILE;
            uint g_row = tg_m + m;
            uint g_col = k_base + k;
            half a_val = (g_row < M && g_col < K) ? A[g_row * K + g_col] : half(0);
            if (has_awq && g_col < K) {
                a_val = half(float(a_val) / float(awq_scales[g_col]));
            }
            tg_a[0][i] = a_val;
        }
        // B-tile load from superblock layout (vectorized word loads)
        for (uint i = tid; i < TN_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
            uint n = i / MATMUL_K_TILE;
            uint k = i % MATMUL_K_TILE;
            uint g_n = tg_n + n;
            uint g_k = k_base + k;
            half val = half(0);
            if (g_n < N && g_k < K) {
                uint sb_idx = g_k / GS;
                uint sb_offset = g_k % GS;
                device const uchar *data_ptr = W + g_n * sb_stride + sb_idx * SB_BYTES_INT4;
                uint word_offset = sb_offset / 8;
                uint nibble_pos = sb_offset % 8;
                uint packed4 = ((device const uint*)(data_ptr))[word_offset];
                uchar nibble = (packed4 >> (nibble_pos * 4)) & 0xF;
                float s = float(W_scales[g_n * num_groups + sb_idx]);
                float z = float(W_zeros[g_n * num_groups + sb_idx]);
                val = half((float(nibble) - z) * s);
            }
            tg_bt[0][k * TN_STRIDE + n] = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Main loop: overlap load[next] with compute[current]
    for (uint t = 0; t < num_k_steps; t++) {
        uint cur = t % 2;
        uint nxt = (t + 1) % 2;

        if (t + 1 < num_k_steps) {
            uint k_base = (t + 1) * MATMUL_K_TILE;
            for (uint i = tid; i < TM_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
                uint m = i / MATMUL_K_TILE;
                uint k = i % MATMUL_K_TILE;
                uint g_row = tg_m + m;
                uint g_col = k_base + k;
                half a_val = (g_row < M && g_col < K) ? A[g_row * K + g_col] : half(0);
                if (has_awq && g_col < K) {
                    a_val = half(float(a_val) / float(awq_scales[g_col]));
                }
                tg_a[nxt][i] = a_val;
            }
            for (uint i = tid; i < TN_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
                uint n = i / MATMUL_K_TILE;
                uint k = i % MATMUL_K_TILE;
                uint g_n = tg_n + n;
                uint g_k = k_base + k;
                half val = half(0);
                if (g_n < N && g_k < K) {
                    uint sb_idx = g_k / GS;
                    uint sb_offset = g_k % GS;
                    device const uchar *data_ptr = W + g_n * sb_stride + sb_idx * SB_BYTES_INT4;
                    uint word_offset = sb_offset / 8;
                    uint nibble_pos = sb_offset % 8;
                    uint packed4 = ((device const uint*)(data_ptr))[word_offset];
                    uchar nibble = (packed4 >> (nibble_pos * 4)) & 0xF;
                    float s = float(W_scales[g_n * num_groups + sb_idx]);
                    float z = float(W_zeros[g_n * num_groups + sb_idx]);
                    val = half((float(nibble) - z) * s);
                }
                tg_bt[nxt][k * TN_STRIDE + n] = val;
            }
        }

        for (uint kbi = 0; kbi < K_BLOCKS; kbi++) {
            simdgroup_matrix<half, 8, 8> a_mat;
            simdgroup_load(a_mat, tg_a[cur] + sgid * 8 * MATMUL_K_TILE + kbi * 8, MATMUL_K_TILE);
            for (uint j = 0; j < TN_BLOCKS; j++) {
                simdgroup_matrix<half, 8, 8> bt_mat;
                simdgroup_load(bt_mat, tg_bt[cur] + kbi * 8 * TN_STRIDE + j * 8, TN_STRIDE);
                simdgroup_multiply_accumulate(acc[j], a_mat, bt_mat, acc[j]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results via threadgroup memory
    threadgroup float tg_out[N_SIMDGROUPS * 8 * 8];

    for (uint j = 0; j < TN_BLOCKS; j++) {
        simdgroup_store(acc[j], tg_out + sgid * 64, 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = tid; i < TM_TILE * 8; i += THREADS_PER_TG) {
            uint local_m = i / 8;
            uint local_n = i % 8;
            uint out_row = tg_m + local_m;
            uint out_col = tg_n + j * 8 + local_n;
            if (out_row < M && out_col < N) {
                uint sg = local_m / 8;
                uint sg_row = local_m % 8;
                C[out_row * N + out_col] = half(tg_out[sg * 64 + sg_row * 8 + local_n]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ── INT8 superblock tiled GEMM (prefill path, M>1) ──────────────
kernel void superblock_matmul_int8(
    device const half *A            [[buffer(0)]],
    device const uchar *W           [[buffer(1)]],   // [N, K] contiguous data
    device half *C                  [[buffer(2)]],
    constant uint &M                [[buffer(3)]],
    constant uint &N                [[buffer(4)]],
    constant uint &K                [[buffer(5)]],
    device const half *awq_scales   [[buffer(6)]],
    constant uint &has_awq          [[buffer(7)]],
    device const half *W_scales     [[buffer(8)]],   // [N, K/GS]
    device const half *W_zeros      [[buffer(9)]],   // [N, K/GS]
    uint2 group_id [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    uint tg_m = group_id.x * TM_TILE;
    uint tg_n = group_id.y * TN_TILE;

    threadgroup half tg_a[2][TM_TILE * MATMUL_K_TILE];
    threadgroup half tg_bt[2][MATMUL_K_TILE * TN_STRIDE];

    uint num_groups = K / GS;
    uint num_k_steps = (K + MATMUL_K_TILE - 1) / MATMUL_K_TILE;

    uint sb_stride = num_groups * SB_BYTES_INT8;

    simdgroup_matrix<float, 8, 8> acc[TN_BLOCKS];
    for (uint j = 0; j < TN_BLOCKS; j++) acc[j] = simdgroup_matrix<float, 8, 8>(0);

    // Prologue
    {
        uint k_base = 0;
        for (uint i = tid; i < TM_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
            uint m = i / MATMUL_K_TILE;
            uint k = i % MATMUL_K_TILE;
            uint g_row = tg_m + m;
            uint g_col = k_base + k;
            half a_val = (g_row < M && g_col < K) ? A[g_row * K + g_col] : half(0);
            if (has_awq && g_col < K) {
                a_val = half(float(a_val) / float(awq_scales[g_col]));
            }
            tg_a[0][i] = a_val;
        }
        for (uint i = tid; i < TN_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
            uint n = i / MATMUL_K_TILE;
            uint k = i % MATMUL_K_TILE;
            uint g_n = tg_n + n;
            uint g_k = k_base + k;
            half val = half(0);
            if (g_n < N && g_k < K) {
                uint sb_idx = g_k / GS;
                uint sb_offset = g_k % GS;
                device const uchar *data_ptr = W + g_n * sb_stride + sb_idx * SB_BYTES_INT8;
                uint word_in_data = sb_offset / 4;
                uint byte_in_word = sb_offset % 4;
                uint packed4 = ((device const uint*)(data_ptr))[word_in_data];
                uchar q = (packed4 >> (byte_in_word * 8)) & 0xFF;
                float s = float(W_scales[g_n * num_groups + sb_idx]);
                float z = float(W_zeros[g_n * num_groups + sb_idx]);
                val = half((float(q) - z) * s);
            }
            tg_bt[0][k * TN_STRIDE + n] = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = 0; t < num_k_steps; t++) {
        uint cur = t % 2;
        uint nxt = (t + 1) % 2;

        if (t + 1 < num_k_steps) {
            uint k_base = (t + 1) * MATMUL_K_TILE;
            for (uint i = tid; i < TM_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
                uint m = i / MATMUL_K_TILE;
                uint k = i % MATMUL_K_TILE;
                uint g_row = tg_m + m;
                uint g_col = k_base + k;
                half a_val = (g_row < M && g_col < K) ? A[g_row * K + g_col] : half(0);
                if (has_awq && g_col < K) {
                    a_val = half(float(a_val) / float(awq_scales[g_col]));
                }
                tg_a[nxt][i] = a_val;
            }
            for (uint i = tid; i < TN_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
                uint n = i / MATMUL_K_TILE;
                uint k = i % MATMUL_K_TILE;
                uint g_n = tg_n + n;
                uint g_k = k_base + k;
                half val = half(0);
                if (g_n < N && g_k < K) {
                    uint sb_idx = g_k / GS;
                    uint sb_offset = g_k % GS;
                    device const uchar *data_ptr = W + g_n * sb_stride + sb_idx * SB_BYTES_INT8;
                    uint word_in_data = sb_offset / 4;
                    uint byte_in_word = sb_offset % 4;
                    uint packed4 = ((device const uint*)(data_ptr))[word_in_data];
                    uchar q = (packed4 >> (byte_in_word * 8)) & 0xFF;
                    float s = float(W_scales[g_n * num_groups + sb_idx]);
                    float z = float(W_zeros[g_n * num_groups + sb_idx]);
                    val = half((float(q) - z) * s);
                }
                tg_bt[nxt][k * TN_STRIDE + n] = val;
            }
        }

        for (uint kbi = 0; kbi < K_BLOCKS; kbi++) {
            simdgroup_matrix<half, 8, 8> a_mat;
            simdgroup_load(a_mat, tg_a[cur] + sgid * 8 * MATMUL_K_TILE + kbi * 8, MATMUL_K_TILE);
            for (uint j = 0; j < TN_BLOCKS; j++) {
                simdgroup_matrix<half, 8, 8> bt_mat;
                simdgroup_load(bt_mat, tg_bt[cur] + kbi * 8 * TN_STRIDE + j * 8, TN_STRIDE);
                simdgroup_multiply_accumulate(acc[j], a_mat, bt_mat, acc[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float tg_out[N_SIMDGROUPS * 8 * 8];
    for (uint j = 0; j < TN_BLOCKS; j++) {
        simdgroup_store(acc[j], tg_out + sgid * 64, 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < TM_TILE * 8; i += THREADS_PER_TG) {
            uint local_m = i / 8;
            uint local_n = i % 8;
            uint out_row = tg_m + local_m;
            uint out_col = tg_n + j * 8 + local_n;
            if (out_row < M && out_col < N) {
                uint sg = local_m / 8;
                uint sg_row = local_m % 8;
                C[out_row * N + out_col] = half(tg_out[sg * 64 + sg_row * 8 + local_n]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
