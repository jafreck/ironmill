// ── INT4 tiled GEMM (prefill path, M>1) ─────────────────────────
//
// Uses simdgroup_matrix_multiply_accumulate for hardware-accelerated 8×8
// matrix multiply. 256 threads = 8 simdgroups, each handling 8 output rows.
//
// Dispatch: (ceil(M/TM_TILE), ceil(N/TN_TILE), 1) threadgroups,
//           (256, 1, 1) threads per group.

kernel void affine_matmul_int4(
    device const half *A            [[buffer(0)]],   // [M, K]
    device const uchar *B_packed    [[buffer(1)]],   // [N, K/2]
    device const half *scales       [[buffer(2)]],   // [N, num_groups]
    device const half *zeros        [[buffer(3)]],   // [N, num_groups]
    device half *C                  [[buffer(4)]],   // [M, N]
    constant uint &M                [[buffer(5)]],
    constant uint &N                [[buffer(6)]],
    constant uint &K                [[buffer(7)]],
    constant uint &group_size       [[buffer(8)]],
    device const half *awq_scales   [[buffer(9)]],
    constant uint &has_awq          [[buffer(10)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    uint tg_m = group_id.x * TM_TILE;
    uint tg_n = group_id.y * TN_TILE;

    threadgroup half tg_a[2][TM_TILE * MATMUL_K_TILE];
    threadgroup half tg_bt[2][MATMUL_K_TILE * TN_STRIDE];

    uint num_groups = (K + group_size - 1) / group_size;
    uint num_k_steps = (K + MATMUL_K_TILE - 1) / MATMUL_K_TILE;

    // Blocked layout constants for B-load (BLK_K=8 blocks, independent of K_TILE)
    uint total_k_blocks = (K + BLK_K - 1) / BLK_K;
    uint blk_bytes = BLK_N * (BLK_K / 2);  // bytes per blocked-layout block

    simdgroup_matrix<float, 8, 8> acc[TN_BLOCKS];
    for (uint j = 0; j < TN_BLOCKS; j++) acc[j] = simdgroup_matrix<float, 8, 8>(0);

    // Prologue: load first tile into buf[0]
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
                uint n_blk = g_n / BLK_N;
                uint n_loc = g_n % BLK_N;
                uint k_blk = g_k / BLK_K;
                uint k_loc = g_k % BLK_K;
                // Word-aligned load: read the uint32 containing this nibble
                uint word_idx = (n_blk * total_k_blocks + k_blk) * BLK_N + n_loc;
                uint packed4 = ((device const uint*)B_packed)[word_idx];
                uint shift = k_loc * 4;
                uchar nibble = (packed4 >> shift) & 0xF;
                uint grp = g_k / group_size;
                float s = float(scales[g_n * num_groups + grp]);
                float z = float(zeros[g_n * num_groups + grp]);
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

        // Load NEXT tile (if not the last step)
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
                    uint n_blk = g_n / BLK_N;
                    uint n_loc = g_n % BLK_N;
                    uint k_blk = g_k / BLK_K;
                    uint k_loc = g_k % BLK_K;
                    uint word_idx = (n_blk * total_k_blocks + k_blk) * BLK_N + n_loc;
                    uint packed4 = ((device const uint*)B_packed)[word_idx];
                    uint shift = k_loc * 4;
                    uchar nibble  = (packed4 >> shift) & 0xF;
                    uint grp = g_k / group_size;
                    float s = float(scales[g_n * num_groups + grp]);
                    float z = float(zeros[g_n * num_groups + grp]);
                    val = half((float(nibble) - z) * s);
                }
                tg_bt[nxt][k * TN_STRIDE + n] = val;
            }
        }

        // Compute on CURRENT tile: K_BLOCKS MMA ops per K-tile
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

// ── INT8 tiled GEMM (prefill path, M>1) ─────────────────────────
//
// Uses simdgroup_matrix_multiply_accumulate for hardware-accelerated 8×8
// matrix multiply. 256 threads = 8 simdgroups, each handling 8 output rows.
//
// Dispatch: (ceil(M/TM_TILE), ceil(N/TN_TILE), 1) threadgroups,
//           (256, 1, 1) threads per group.

kernel void affine_matmul_int8(
    device const half *A            [[buffer(0)]],   // [M, K]
    device const uchar *B_packed    [[buffer(1)]],   // [N, K]
    device const half *scales       [[buffer(2)]],   // [N, num_groups]
    device const half *zeros        [[buffer(3)]],   // [N, num_groups]
    device half *C                  [[buffer(4)]],   // [M, N]
    constant uint &M                [[buffer(5)]],
    constant uint &N                [[buffer(6)]],
    constant uint &K                [[buffer(7)]],
    constant uint &group_size       [[buffer(8)]],
    device const half *awq_scales   [[buffer(9)]],
    constant uint &has_awq          [[buffer(10)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    uint tg_m = group_id.x * TM_TILE;
    uint tg_n = group_id.y * TN_TILE;

    threadgroup half tg_a[2][TM_TILE * MATMUL_K_TILE];
    threadgroup half tg_bt[2][MATMUL_K_TILE * TN_STRIDE];

    uint num_groups = (K + group_size - 1) / group_size;
    uint num_k_steps = (K + MATMUL_K_TILE - 1) / MATMUL_K_TILE;

    // Blocked layout constants for B-load (BLK_K=8 blocks, independent of K_TILE)
    uint total_k_blocks = (K + BLK_K - 1) / BLK_K;
    uint blk_bytes = BLK_N * BLK_K;  // bytes per blocked-layout block

    simdgroup_matrix<float, 8, 8> acc[TN_BLOCKS];
    for (uint j = 0; j < TN_BLOCKS; j++) acc[j] = simdgroup_matrix<float, 8, 8>(0);

    // Prologue: load first tile into buf[0]
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
                uint n_blk = g_n / BLK_N;
                uint n_loc = g_n % BLK_N;
                uint k_blk = g_k / BLK_K;
                uint k_loc = g_k % BLK_K;
                // Word-aligned load: read uint32 containing this byte
                uint word_idx = (n_blk * total_k_blocks + k_blk) * (BLK_N * 2)
                              + n_loc * 2 + k_loc / 4;
                uint packed4 = ((device const uint*)B_packed)[word_idx];
                uint shift = (k_loc % 4) * 8;
                uchar q = (packed4 >> shift) & 0xFF;
                uint grp = g_k / group_size;
                float s = float(scales[g_n * num_groups + grp]);
                float z = float(zeros[g_n * num_groups + grp]);
                val = half((float(q) - z) * s);
            }
            tg_bt[0][k * TN_STRIDE + n] = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Main loop: overlap load[next] with compute[current]
    for (uint t = 0; t < num_k_steps; t++) {
        uint cur = t % 2;
        uint nxt = (t + 1) % 2;

        // Load NEXT tile (if not the last step)
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
                    uint n_blk = g_n / BLK_N;
                    uint n_loc = g_n % BLK_N;
                    uint k_blk = g_k / BLK_K;
                    uint k_loc = g_k % BLK_K;
                    uint word_idx = (n_blk * total_k_blocks + k_blk) * (BLK_N * 2)
                                  + n_loc * 2 + k_loc / 4;
                    uint packed4 = ((device const uint*)B_packed)[word_idx];
                    uint shift = (k_loc % 4) * 8;
                    uchar q = (packed4 >> shift) & 0xFF;
                    uint grp = g_k / group_size;
                    float s = float(scales[g_n * num_groups + grp]);
                    float z = float(zeros[g_n * num_groups + grp]);
                    val = half((float(q) - z) * s);
                }
                tg_bt[nxt][k * TN_STRIDE + n] = val;
            }
        }

        // Compute on CURRENT tile: K_BLOCKS MMA ops per K-tile
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
    device const uchar *W           [[buffer(1)]],   // [N, G, sb_bytes] superblocks
    device half *C                  [[buffer(2)]],   // [M, N]
    constant uint &M                [[buffer(3)]],
    constant uint &N                [[buffer(4)]],
    constant uint &K                [[buffer(5)]],
    constant uint &group_size       [[buffer(6)]],
    device const half *awq_scales   [[buffer(7)]],   // [K] or dummy
    constant uint &has_awq          [[buffer(8)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    uint tg_m = group_id.x * TM_TILE;
    uint tg_n = group_id.y * TN_TILE;

    threadgroup half tg_a[2][TM_TILE * MATMUL_K_TILE];
    threadgroup half tg_bt[2][MATMUL_K_TILE * TN_STRIDE];

    uint num_groups = K / group_size;
    uint num_k_steps = (K + MATMUL_K_TILE - 1) / MATMUL_K_TILE;

    // Superblock layout constants
    uint sb_bytes = SB_HEADER_BYTES + group_size / 2;
    uint sb_stride = num_groups * sb_bytes;  // bytes per row

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
        // B-tile load from superblock layout
        for (uint i = tid; i < TN_TILE * MATMUL_K_TILE; i += THREADS_PER_TG) {
            uint n = i / MATMUL_K_TILE;
            uint k = i % MATMUL_K_TILE;
            uint g_n = tg_n + n;
            uint g_k = k_base + k;
            half val = half(0);
            if (g_n < N && g_k < K) {
                uint sb_idx = g_k / group_size;
                uint sb_offset = g_k % group_size;
                uint byte_in_sb = SB_HEADER_BYTES + sb_offset / 2;
                device const uchar *sb = W + g_n * sb_stride + sb_idx * sb_bytes;
                uchar packed = sb[byte_in_sb];
                uchar nibble = (sb_offset % 2 == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
                float s = float(*(device const half *)(sb));
                float z = float(*(device const half *)(sb + 2));
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
                    uint sb_idx = g_k / group_size;
                    uint sb_offset = g_k % group_size;
                    uint byte_in_sb = SB_HEADER_BYTES + sb_offset / 2;
                    device const uchar *sb = W + g_n * sb_stride + sb_idx * sb_bytes;
                    uchar packed = sb[byte_in_sb];
                    uchar nibble = (sb_offset % 2 == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
                    float s = float(*(device const half *)(sb));
                    float z = float(*(device const half *)(sb + 2));
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
