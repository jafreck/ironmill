// TurboQuant outlier-aware cache write kernel for MLX backend.
//
// Splits KV into outlier and non-outlier channel groups, applies
// independent TurboQuant quantization to each group.
// Helpers (hadamard_rotate_inplace, etc.) are prepended at compile
// time from mlx_helpers.metal.

[[kernel]] void turboquant_outlier_cache_write(
    device const half* kv_proj                      [[buffer(0)]],
    device const uint* channel_indices              [[buffer(1)]],
    device char* outlier_cache                      [[buffer(2)]],
    device char* non_outlier_cache                  [[buffer(3)]],
    device const float* outlier_rotation_signs      [[buffer(4)]],
    device const float* non_outlier_rotation_signs  [[buffer(5)]],
    device const float* outlier_codebook            [[buffer(6)]],
    device const float* outlier_boundaries          [[buffer(7)]],
    device const float* non_outlier_codebook        [[buffer(8)]],
    device const float* non_outlier_boundaries      [[buffer(9)]],
    device float* outlier_scale_buf                 [[buffer(10)]],
    device float* non_outlier_scale_buf             [[buffer(11)]],
    device const float* outlier_qjl_matrix          [[buffer(12)]],
    device const float* non_outlier_qjl_matrix      [[buffer(13)]],
    device float* outlier_r_norms_buf               [[buffer(14)]],
    device float* non_outlier_r_norms_buf           [[buffer(15)]],
    device const uint* params                       [[buffer(16)]],
    device float* dummy_out                         [[buffer(17)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint num_kv_heads      = params[0];
    uint head_dim          = params[1];
    uint max_seq_len       = params[2];
    uint seq_pos           = params[3];
    uint n_outlier         = params[4];
    uint d_outlier_padded  = params[5];
    uint d_non_padded      = params[6];
    uint outlier_n_levels  = params[7];
    uint non_outlier_n_levels = params[8];
    uint is_k_cache        = params[9];

    threadgroup float shared_outlier[HEAD_DIM];
    threadgroup float shared_non_outlier[HEAD_DIM];
    threadgroup float shared_reduce[HEAD_DIM];
    threadgroup char shared_quant[HEAD_DIM];

    uint head_idx = tgid;
    if (head_idx >= num_kv_heads) return;

    uint n_non = head_dim - n_outlier;
    uint input_base = head_idx * head_dim;

    // Extract outlier channels
    for (uint i = tid; i < d_outlier_padded; i += tg_size) {
        if (i < n_outlier) {
            uint src_idx = channel_indices[i];
            shared_outlier[i] = float(kv_proj[input_base + src_idx]);
        } else {
            shared_outlier[i] = 0.0f;
        }
    }

    // Extract non-outlier channels
    for (uint i = tid; i < d_non_padded; i += tg_size) {
        if (i < n_non) {
            uint src_idx = channel_indices[n_outlier + i];
            shared_non_outlier[i] = float(kv_proj[input_base + src_idx]);
        } else {
            shared_non_outlier[i] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Outlier group: rotate → L2 norm → quantize → [QJL] → pack ──
    hadamard_rotate_inplace(shared_outlier, outlier_rotation_signs, d_outlier_padded, tid, tg_size);

    float local_sq = 0.0f;
    for (uint d = tid; d < d_outlier_padded; d += tg_size)
        local_sq += shared_outlier[d] * shared_outlier[d];
    shared_reduce[tid] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared_reduce[tid] += shared_reduce[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float o_l2 = sqrt(max(shared_reduce[0], 1e-20f));
    if (tid == 0)
        outlier_scale_buf[head_idx * max_seq_len + seq_pos] = o_l2;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float o_inv = 1.0f / max(o_l2, 1e-10f);
    uint o_nb = outlier_n_levels - 1;
    for (uint d = tid; d < d_outlier_padded; d += tg_size) {
        float normalized = shared_outlier[d] * o_inv;
        uint idx = 0;
        for (uint b = 0; b < o_nb; b++) {
            if (normalized >= outlier_boundaries[b]) idx = b + 1;
        }
        shared_quant[d] = char(idx);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (is_k_cache == 1) {
        float local_sq_e = 0.0f;
        for (uint d = tid; d < d_outlier_padded; d += tg_size) {
            float normalized = shared_outlier[d] * o_inv;
            float dequant_val = outlier_codebook[uint(shared_quant[d])];
            float residual = normalized - dequant_val;
            shared_outlier[d] = residual;
            local_sq_e += residual * residual;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        shared_reduce[tid] = local_sq_e;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) shared_reduce[tid] += shared_reduce[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float r_e = sqrt(max(shared_reduce[0], 1e-20f));
        if (tid == 0)
            outlier_r_norms_buf[head_idx * max_seq_len + seq_pos] = r_e;

        for (uint d = tid; d < d_outlier_padded; d += tg_size) {
            float proj = 0.0f;
            uint row_base = d * d_outlier_padded;
            for (uint k = 0; k < d_outlier_padded; k++) {
                proj += outlier_qjl_matrix[row_base + k] * shared_outlier[k];
            }
            uchar sign_bit = (proj >= 0.0f) ? uchar(0x8) : uchar(0x0);
            shared_quant[d] = char(uchar(shared_quant[d]) | sign_bit);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint o_bpp = d_outlier_padded / 2;
    uint o_cache_base = head_idx * max_seq_len * o_bpp + seq_pos * o_bpp;
    for (uint d = tid * 2; d < d_outlier_padded; d += tg_size * 2) {
        uchar lo = uchar(shared_quant[d]     & 0xF);
        uchar hi = (d + 1 < d_outlier_padded) ? uchar(shared_quant[d + 1] & 0xF) : 0;
        ((device uchar*)outlier_cache)[o_cache_base + d / 2] = lo | (hi << 4);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Non-outlier group: rotate → L2 norm → quantize → [QJL] → pack ──
    hadamard_rotate_inplace(shared_non_outlier, non_outlier_rotation_signs, d_non_padded, tid, tg_size);

    local_sq = 0.0f;
    for (uint d = tid; d < d_non_padded; d += tg_size)
        local_sq += shared_non_outlier[d] * shared_non_outlier[d];
    shared_reduce[tid] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared_reduce[tid] += shared_reduce[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float n_l2 = sqrt(max(shared_reduce[0], 1e-20f));
    if (tid == 0)
        non_outlier_scale_buf[head_idx * max_seq_len + seq_pos] = n_l2;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float n_inv = 1.0f / max(n_l2, 1e-10f);
    uint n_nb = non_outlier_n_levels - 1;
    for (uint d = tid; d < d_non_padded; d += tg_size) {
        float normalized = shared_non_outlier[d] * n_inv;
        uint idx = 0;
        for (uint b = 0; b < n_nb; b++) {
            if (normalized >= non_outlier_boundaries[b]) idx = b + 1;
        }
        shared_quant[d] = char(idx);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (is_k_cache == 1) {
        float local_sq_e = 0.0f;
        for (uint d = tid; d < d_non_padded; d += tg_size) {
            float normalized = shared_non_outlier[d] * n_inv;
            float dequant_val = non_outlier_codebook[uint(shared_quant[d])];
            float residual = normalized - dequant_val;
            shared_non_outlier[d] = residual;
            local_sq_e += residual * residual;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        shared_reduce[tid] = local_sq_e;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) shared_reduce[tid] += shared_reduce[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float r_e = sqrt(max(shared_reduce[0], 1e-20f));
        if (tid == 0)
            non_outlier_r_norms_buf[head_idx * max_seq_len + seq_pos] = r_e;

        for (uint d = tid; d < d_non_padded; d += tg_size) {
            float proj = 0.0f;
            uint row_base = d * d_non_padded;
            for (uint k = 0; k < d_non_padded; k++) {
                proj += non_outlier_qjl_matrix[row_base + k] * shared_non_outlier[k];
            }
            uchar sign_bit = (proj >= 0.0f) ? uchar(0x8) : uchar(0x0);
            shared_quant[d] = char(uchar(shared_quant[d]) | sign_bit);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint n_bpp = d_non_padded / 2;
    uint n_cache_base = head_idx * max_seq_len * n_bpp + seq_pos * n_bpp;
    for (uint d = tid * 2; d < d_non_padded; d += tg_size * 2) {
        uchar lo = uchar(shared_quant[d]     & 0xF);
        uchar hi = (d + 1 < d_non_padded) ? uchar(shared_quant[d + 1] & 0xF) : 0;
        ((device uchar*)non_outlier_cache)[n_cache_base + d / 2] = lo | (hi << 4);
    }

    if (tgid == 0 && tid == 0) dummy_out[0] = 1.0f;
}
