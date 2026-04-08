// ── Batched INT4 matvec for FFN gate+up projections (M=1) ──────
//
// Computes 2 independent matvecs in a single dispatch:
//   gate = x · W_gate^T   (first N_gate threadgroups)
//   up   = x · W_up^T     (remaining N_up threadgroups)
//
// Both projections share the same input x. Threadgroup index determines
// which projection: tid < N_gate → gate, else → up.
//
// Dispatch: (N_gate + N_up, 1, 1) threadgroups, (32, 1, 1) threads.

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
    uint tid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    // Select which projection based on threadgroup index
    uint local_tid;
    device const uchar* B_packed;
    device const half* scales;
    device const half* zeros;
    device half* C;

    if (tid < N_gate) {
        local_tid = tid;
        B_packed = B_gate_packed;
        scales = scales_gate;
        zeros = zeros_gate;
        C = C_gate;
    } else {
        local_tid = tid - N_gate;
        B_packed = B_up_packed;
        scales = scales_up;
        zeros = zeros_up;
        C = C_up;
    }

    if (local_tid >= N_gate) return;

    uint half_K = K / 2;
    uint num_groups = (K + group_size - 1) / group_size;
    uint scale_row = local_tid * num_groups;

    // Blocked layout addressing
    uint k_blocks      = (K + BLK_K - 1) / BLK_K;
    uint local_k_bytes = BLK_K / 2;
    uint block_bytes   = BLK_N * local_k_bytes;
    uint n_block = local_tid / BLK_N;
    uint n_local = local_tid % BLK_N;

    float acc = 0.0f;

    for (uint k = lane; k < half_K; k += 32) {
        uint kb = k / local_k_bytes;
        uint b  = k % local_k_bytes;
        uint byte_idx = (n_block * k_blocks + kb) * block_bytes
                      + n_local * local_k_bytes + b;

        uchar packed = B_packed[byte_idx];
        uchar lo = packed & 0x0F;
        uchar hi = (packed >> 4) & 0x0F;

        uint k2 = k * 2;
        uint g0 = k2 / group_size;
        uint g1 = (k2 + 1) / group_size;

        float s0 = float(scales[scale_row + g0]);
        float z0 = float(zeros[scale_row + g0]);
        float w0 = (float(lo) - z0) * s0;

        float s1 = float(scales[scale_row + g1]);
        float z1 = float(zeros[scale_row + g1]);
        float w1 = (float(hi) - z1) * s1;

        if (has_awq) {
            acc += (float(A[k2]) / float(awq_scales[k2])) * w0;
            acc += (float(A[k2 + 1]) / float(awq_scales[k2 + 1])) * w1;
        } else {
            acc += float(A[k2])     * w0;
            acc += float(A[k2 + 1]) * w1;
        }
    }

    acc = simd_sum(acc);

    if (lane == 0) {
        C[local_tid] = half(acc);
    }
}

// ── Batched INT4 matvec for 4 GDN projections (M=1) ─────────────
//
// Computes 4 independent matvecs in a single dispatch:
//   qkv = x · W_qkv^T   (N0 threadgroups)
//   z   = x · W_z^T      (N1 threadgroups)
//   a   = x · W_a^T      (N2 threadgroups)
//   b   = x · W_b^T      (N3 threadgroups)
//
// All projections share the same input x. Threadgroup index determines
// which projection via cumulative thresholds in params.
//
// Dispatch: (N0 + N1 + N2 + N3, 1, 1) threadgroups, (32, 1, 1) threads.

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
    uint tid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint N0 = params.N0;
    uint N1 = params.N1;
    uint N2 = params.N2;
    uint K  = params.K;
    uint group_size = params.group_size;
    uint has_awq = params.has_awq;

    // Route threadgroup to the correct projection
    uint local_tid;
    uint N_proj;
    device const uchar* B_packed;
    device const half* scales;
    device const half* zeros;
    device half* C;

    uint t0 = N0;
    uint t1 = t0 + N1;
    uint t2 = t1 + N2;

    if (tid < t0) {
        local_tid = tid;
        N_proj = N0;
        B_packed = B0_packed; scales = scales0; zeros = zeros0; C = C0;
    } else if (tid < t1) {
        local_tid = tid - t0;
        N_proj = N1;
        B_packed = B1_packed; scales = scales1; zeros = zeros1; C = C1;
    } else if (tid < t2) {
        local_tid = tid - t1;
        N_proj = N2;
        B_packed = B2_packed; scales = scales2; zeros = zeros2; C = C2;
    } else {
        local_tid = tid - t2;
        N_proj = params.N3;
        B_packed = B3_packed; scales = scales3; zeros = zeros3; C = C3;
    }

    if (local_tid >= N_proj) return;

    uint half_K = K / 2;
    uint num_groups = (K + group_size - 1) / group_size;
    uint scale_row = local_tid * num_groups;

    // Blocked layout addressing (same as affine_matvec_int4)
    uint k_blocks      = (K + BLK_K - 1) / BLK_K;
    uint local_k_bytes = BLK_K / 2;
    uint block_bytes   = BLK_N * local_k_bytes;
    uint n_block = local_tid / BLK_N;
    uint n_local = local_tid % BLK_N;

    float acc = 0.0f;

    for (uint k = lane; k < half_K; k += 32) {
        uint kb = k / local_k_bytes;
        uint b  = k % local_k_bytes;
        uint byte_idx = (n_block * k_blocks + kb) * block_bytes
                      + n_local * local_k_bytes + b;

        uchar packed = B_packed[byte_idx];
        uchar lo = packed & 0x0F;
        uchar hi = (packed >> 4) & 0x0F;

        uint k2 = k * 2;
        uint g0 = k2 / group_size;
        uint g1 = (k2 + 1) / group_size;

        float s0 = float(scales[scale_row + g0]);
        float z0 = float(zeros[scale_row + g0]);
        float w0 = (float(lo) - z0) * s0;

        float s1 = float(scales[scale_row + g1]);
        float z1 = float(zeros[scale_row + g1]);
        float w1 = (float(hi) - z1) * s1;

        if (has_awq) {
            acc += (float(A[k2]) / float(awq_scales[k2])) * w0;
            acc += (float(A[k2 + 1]) / float(awq_scales[k2 + 1])) * w1;
        } else {
            acc += float(A[k2])     * w0;
            acc += float(A[k2 + 1]) * w1;
        }
    }

    acc = simd_sum(acc);

    if (lane == 0) {
        C[local_tid] = half(acc);
    }
}
