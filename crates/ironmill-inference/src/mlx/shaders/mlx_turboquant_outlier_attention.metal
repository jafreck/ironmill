// TurboQuant outlier-aware attention kernel for MLX backend.
//
// Performs attention with dual-group (outlier + non-outlier) quantized
// KV cache plus QJL correction on both groups.
// Helpers (hadamard_rotate_inplace, read_quantized_tile_int4,
// read_quantized_tile_int4) are prepended at compile time from src/shaders/turboquant_helpers.metal.

[[kernel]] void turboquant_outlier_attention(
    device const half* q                            [[buffer(0)]],
    device const char* k_outlier_cache              [[buffer(1)]],
    device const char* v_outlier_cache              [[buffer(2)]],
    device const char* k_non_outlier_cache          [[buffer(3)]],
    device const char* v_non_outlier_cache          [[buffer(4)]],
    device const uint* channel_indices              [[buffer(5)]],
    device const float* outlier_rotation_signs      [[buffer(6)]],
    device const float* non_outlier_rotation_signs  [[buffer(7)]],
    device const float* outlier_codebook            [[buffer(8)]],
    device const float* non_outlier_codebook        [[buffer(9)]],
    device const float* k_outlier_scales            [[buffer(10)]],
    device const float* v_outlier_scales            [[buffer(11)]],
    device const float* k_non_outlier_scales        [[buffer(12)]],
    device const float* v_non_outlier_scales        [[buffer(13)]],
    device const float* outlier_qjl_matrix          [[buffer(14)]],
    device const float* non_outlier_qjl_matrix      [[buffer(15)]],
    device const float* k_outlier_r_norms           [[buffer(16)]],
    device const float* k_non_outlier_r_norms       [[buffer(17)]],
    device const float* v_outlier_codebook          [[buffer(18)]],
    device const float* v_non_outlier_codebook      [[buffer(19)]],
    device const uint* params                       [[buffer(20)]],
    device half* output                             [[buffer(21)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint num_heads        = params[0];
    uint num_kv_heads     = params[1];
    uint head_dim         = params[2];
    uint max_seq_len      = params[3];
    uint seq_len          = params[4];
    uint n_outlier        = params[5];
    uint d_outlier_padded = params[6];
    uint d_non_padded     = params[7];

    constexpr uint TILE = 32;

    threadgroup float shared_q_outlier[HEAD_DIM];
    threadgroup float shared_q_non_outlier[HEAD_DIM];
    threadgroup float shared_s_q_outlier[HEAD_DIM];
    threadgroup float shared_s_q_non[HEAD_DIM];
    threadgroup char  outlier_kv_tile[TILE * HEAD_DIM_PACKED];
    threadgroup char  non_outlier_kv_tile[TILE * HEAD_DIM_PACKED];
    threadgroup float o_tile_scales[TILE];
    threadgroup float n_tile_scales[TILE];
    threadgroup float shared_reduce[HEAD_DIM];
    threadgroup float tile_scores[TILE];
    threadgroup float shared_output_outlier[HEAD_DIM];
    threadgroup float shared_output_non_outlier[HEAD_DIM];
    threadgroup float softmax_max[1];
    threadgroup float softmax_sum[1];
    threadgroup float tile_correction[1];

    uint head_idx = tgid;
    if (head_idx >= num_heads) return;

    uint heads_per_group = num_heads / num_kv_heads;
    uint kv_head = head_idx / heads_per_group;
    float scale = 1.0f / sqrt(float(head_dim));
    uint q_base = head_idx * head_dim;
    uint n_non = head_dim - n_outlier;

    // Load and rotate Q for both groups
    for (uint i = tid; i < d_outlier_padded; i += tg_size) {
        shared_q_outlier[i] = (i < n_outlier) ? float(q[q_base + channel_indices[i]]) : 0.0f;
    }
    for (uint i = tid; i < d_non_padded; i += tg_size) {
        shared_q_non_outlier[i] = (i < n_non) ? float(q[q_base + channel_indices[n_outlier + i]]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    hadamard_rotate_inplace(shared_q_outlier, outlier_rotation_signs, d_outlier_padded, tid, tg_size);
    hadamard_rotate_inplace(shared_q_non_outlier, non_outlier_rotation_signs, d_non_padded, tid, tg_size);

    // Precompute S · q projections for QJL correction
    for (uint out_d = tid; out_d < d_outlier_padded; out_d += tg_size) {
        float proj = 0.0f;
        uint row_base = out_d * d_outlier_padded;
        for (uint k = 0; k < d_outlier_padded; k++)
            proj += outlier_qjl_matrix[row_base + k] * shared_q_outlier[k];
        shared_s_q_outlier[out_d] = proj;
    }
    for (uint out_d = tid; out_d < d_non_padded; out_d += tg_size) {
        float proj = 0.0f;
        uint row_base = out_d * d_non_padded;
        for (uint k = 0; k < d_non_padded; k++)
            proj += non_outlier_qjl_matrix[row_base + k] * shared_q_non_outlier[k];
        shared_s_q_non[out_d] = proj;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float qjl_factor_o = sqrt(2.0f / 3.14159265f) / float(d_outlier_padded);
    float qjl_factor_n = sqrt(2.0f / 3.14159265f) / float(d_non_padded);

    // Zero accumulators
    for (uint d = tid; d < d_outlier_padded; d += tg_size) shared_output_outlier[d] = 0.0f;
    for (uint d = tid; d < d_non_padded; d += tg_size) shared_output_non_outlier[d] = 0.0f;
    if (tid == 0) {
        softmax_max[0] = -INFINITY;
        softmax_sum[0] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint o_bytes_per_pos = d_outlier_padded / 2;
    uint n_bytes_per_pos = d_non_padded / 2;
    uint o_base = kv_head * max_seq_len * o_bytes_per_pos;
    uint n_base = kv_head * max_seq_len * n_bytes_per_pos;

    for (uint tile_start = 0; tile_start < seq_len; tile_start += TILE) {
        uint tile_end = min(tile_start + TILE, seq_len);
        uint actual_tile = tile_end - tile_start;

        // Load K tiles
        uint o_tile_bytes = actual_tile * o_bytes_per_pos;
        for (uint i = tid; i < o_tile_bytes; i += tg_size)
            outlier_kv_tile[i] = k_outlier_cache[o_base + tile_start * o_bytes_per_pos + i];
        uint n_tile_bytes = actual_tile * n_bytes_per_pos;
        for (uint i = tid; i < n_tile_bytes; i += tg_size)
            non_outlier_kv_tile[i] = k_non_outlier_cache[n_base + tile_start * n_bytes_per_pos + i];
        for (uint i = tid; i < actual_tile; i += tg_size) {
            o_tile_scales[i] = k_outlier_scales[kv_head * max_seq_len + tile_start + i];
            n_tile_scales[i] = k_non_outlier_scales[kv_head * max_seq_len + tile_start + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute Q · K with QJL correction
        for (uint p = 0; p < actual_tile; p++) {
            float k_o_deq = o_tile_scales[p];
            float k_n_deq = n_tile_scales[p];
            float partial_dot = 0.0f;
            float partial_qjl_o = 0.0f;
            float partial_qjl_n = 0.0f;

            for (uint d = tid; d < d_outlier_padded; d += tg_size) {
                float k_val = read_quantized_tile_int4(outlier_kv_tile, p, d, d_outlier_padded, k_o_deq, outlier_codebook);
                partial_dot += shared_q_outlier[d] * k_val;
            }
            for (uint d = tid; d < d_non_padded; d += tg_size) {
                float k_val = read_quantized_tile_int4(non_outlier_kv_tile, p, d, d_non_padded, k_n_deq, non_outlier_codebook);
                partial_dot += shared_q_non_outlier[d] * k_val;
            }

            shared_reduce[tid] = partial_dot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint rs = 1;
            while (rs < tg_size) rs <<= 1;
            for (uint s = rs / 2; s > 0; s >>= 1) {
                if (tid < s && (tid + s) < tg_size)
                    shared_reduce[tid] += shared_reduce[tid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float base_score = shared_reduce[0];

            shared_reduce[tid] = partial_qjl_o;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = rs / 2; s > 0; s >>= 1) {
                if (tid < s && (tid + s) < tg_size)
                    shared_reduce[tid] += shared_reduce[tid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float qjl_o = shared_reduce[0];

            shared_reduce[tid] = partial_qjl_n;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = rs / 2; s > 0; s >>= 1) {
                if (tid < s && (tid + s) < tg_size)
                    shared_reduce[tid] += shared_reduce[tid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float qjl_n = shared_reduce[0];

            float corr_o = k_o_deq * k_outlier_r_norms[kv_head * max_seq_len + tile_start + p]
                         * qjl_factor_o * qjl_o;
            float corr_n = k_n_deq * k_non_outlier_r_norms[kv_head * max_seq_len + tile_start + p]
                         * qjl_factor_n * qjl_n;

            if (tid == 0) tile_scores[p] = (base_score + corr_o + corr_n) * scale;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Online softmax
        if (tid == 0) {
            float tm = -INFINITY;
            for (uint p = 0; p < actual_tile; p++) tm = max(tm, tile_scores[p]);
            float old_max = softmax_max[0];
            float new_max = max(old_max, tm);
            float corr = exp(old_max - new_max);
            tile_correction[0] = corr;
            softmax_max[0] = new_max;
            softmax_sum[0] = softmax_sum[0] * corr;
            for (uint p = 0; p < actual_tile; p++) {
                float w = exp(tile_scores[p] - new_max);
                tile_scores[p] = w;
                softmax_sum[0] += w;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float corr = tile_correction[0];
        for (uint d = tid; d < d_outlier_padded; d += tg_size) shared_output_outlier[d] *= corr;
        for (uint d = tid; d < d_non_padded; d += tg_size) shared_output_non_outlier[d] *= corr;

        // Load V tiles
        for (uint i = tid; i < o_tile_bytes; i += tg_size)
            outlier_kv_tile[i] = v_outlier_cache[o_base + tile_start * o_bytes_per_pos + i];
        for (uint i = tid; i < n_tile_bytes; i += tg_size)
            non_outlier_kv_tile[i] = v_non_outlier_cache[n_base + tile_start * n_bytes_per_pos + i];
        for (uint i = tid; i < actual_tile; i += tg_size) {
            o_tile_scales[i] = v_outlier_scales[kv_head * max_seq_len + tile_start + i];
            n_tile_scales[i] = v_non_outlier_scales[kv_head * max_seq_len + tile_start + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint p = 0; p < actual_tile; p++) {
            float w = tile_scores[p];
            float v_o_deq = o_tile_scales[p];
            for (uint d = tid; d < d_outlier_padded; d += tg_size) {
                float v_val = read_quantized_tile_int4(outlier_kv_tile, p, d, d_outlier_padded, v_o_deq, v_outlier_codebook);
                shared_output_outlier[d] += w * v_val;
            }
            float v_n_deq = n_tile_scales[p];
            for (uint d = tid; d < d_non_padded; d += tg_size) {
                float v_val = read_quantized_tile_int4(non_outlier_kv_tile, p, d, d_non_padded, v_n_deq, v_non_outlier_codebook);
                shared_output_non_outlier[d] += w * v_val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize
    float denom = max(softmax_sum[0], 1e-10f);
    for (uint d = tid; d < d_outlier_padded; d += tg_size) shared_output_outlier[d] /= denom;
    for (uint d = tid; d < d_non_padded; d += tg_size) shared_output_non_outlier[d] /= denom;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Un-rotate
    hadamard_rotate_inplace(shared_output_outlier, outlier_rotation_signs, d_outlier_padded, tid, tg_size);
    hadamard_rotate_inplace(shared_output_non_outlier, non_outlier_rotation_signs, d_non_padded, tid, tg_size);

    // Scatter back
    uint out_base = head_idx * head_dim;
    for (uint i = tid; i < n_outlier; i += tg_size)
        output[out_base + channel_indices[i]] = half(shared_output_outlier[i]);
    for (uint i = tid; i < n_non; i += tg_size)
        output[out_base + channel_indices[n_outlier + i]] = half(shared_output_non_outlier[i]);
}
