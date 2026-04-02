// TurboQuant tiled flash attention kernel for MLX backend.
//
// Performs attention with quantized KV cache and online softmax.
// Helpers (hadamard_rotate_inplace, kv_cache_base, read_quantized_tile)
// are prepended at compile time from src/shaders/turboquant_helpers.metal.

[[kernel]] void turboquant_attention(
    device const half* q                [[buffer(0)]],
    device const char* k_cache          [[buffer(1)]],
    device const char* v_cache          [[buffer(2)]],
    device const float* rotation_signs  [[buffer(3)]],
    device const float* k_scale_buf     [[buffer(4)]],
    device const float* v_scale_buf     [[buffer(5)]],
    device const float* k_codebook      [[buffer(6)]],
    device const float* v_codebook      [[buffer(7)]],
    device const float* qjl_matrix      [[buffer(8)]],
    device const float* k_r_norms       [[buffer(9)]],
    device const uint* params           [[buffer(10)]],
    device half* output                 [[buffer(11)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint num_heads    = params[0];
    uint num_kv_heads = params[1];
    uint head_dim     = params[2];
    uint max_seq_len  = params[3];
    uint seq_len      = params[4];
    uint n_bits       = params[5];

    constexpr uint TILE = 32;

    threadgroup float shared_q_rot[HEAD_DIM];
    threadgroup char  kv_tile_raw[TILE * HEAD_DIM];
    threadgroup float tile_scales[TILE];
    threadgroup float shared_reduce[HEAD_DIM];
    threadgroup float tile_scores[TILE];
    threadgroup float shared_output[HEAD_DIM];
    threadgroup float softmax_max[1];
    threadgroup float softmax_sum[1];
    threadgroup float tile_correction[1];

    uint head_idx = tgid;
    if (head_idx >= num_heads) return;

    uint heads_per_group = num_heads / num_kv_heads;
    uint kv_head = head_idx / heads_per_group;
    float scale = 1.0f / sqrt(float(head_dim));
    uint q_base = head_idx * head_dim;
    uint kv_base = kv_cache_base(kv_head, max_seq_len, head_dim, n_bits);

    // Load Q and rotate via butterfly
    for (uint d = tid; d < head_dim; d += tg_size)
        shared_q_rot[d] = float(q[q_base + d]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hadamard_rotate_inplace(shared_q_rot, rotation_signs, head_dim, tid, tg_size);

    // Zero output accumulator
    for (uint d = tid; d < head_dim; d += tg_size)
        shared_output[d] = 0.0f;
    if (tid == 0) {
        softmax_max[0] = -INFINITY;
        softmax_sum[0] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint bytes_per_pos = (n_bits == 4) ? (head_dim / 2) : head_dim;

    // Tiled flash attention
    for (uint tile_start = 0; tile_start < seq_len; tile_start += TILE) {
        uint tile_end = min(tile_start + TILE, seq_len);
        uint actual_tile = tile_end - tile_start;

        // Cooperative load of K tile
        uint tile_bytes = actual_tile * bytes_per_pos;
        for (uint i = tid; i < tile_bytes; i += tg_size)
            kv_tile_raw[i] = ((device const char*)k_cache)[kv_base + tile_start * bytes_per_pos + i];
        for (uint i = tid; i < actual_tile; i += tg_size)
            tile_scales[i] = k_scale_buf[kv_head * max_seq_len + tile_start + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute QK^T — standard codebook dequant for both INT4 and INT8
        for (uint p = 0; p < actual_tile; p++) {
            float k_deq = tile_scales[p];
            float partial_dot = 0.0f;

            for (uint d = tid; d < head_dim; d += tg_size) {
                float k_val = read_quantized_tile(
                    kv_tile_raw, p, d, head_dim, n_bits, k_deq, k_codebook);
                partial_dot += shared_q_rot[d] * k_val;
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

            if (tid == 0)
                tile_scores[p] = base_score * scale;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Per-tile online softmax
        if (tid == 0) {
            float tm = -INFINITY;
            for (uint p = 0; p < actual_tile; p++)
                tm = max(tm, tile_scores[p]);

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
        for (uint d = tid; d < head_dim; d += tg_size)
            shared_output[d] *= corr;

        // Load V tile (reuse kv_tile_raw)
        for (uint i = tid; i < tile_bytes; i += tg_size)
            kv_tile_raw[i] = ((device const char*)v_cache)[kv_base + tile_start * bytes_per_pos + i];
        for (uint i = tid; i < actual_tile; i += tg_size)
            tile_scales[i] = v_scale_buf[kv_head * max_seq_len + tile_start + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate weighted V
        for (uint p = 0; p < actual_tile; p++) {
            float w = tile_scores[p];
            float v_deq = tile_scales[p];
            for (uint d = tid; d < head_dim; d += tg_size) {
                float v_val = read_quantized_tile(
                    kv_tile_raw, p, d, head_dim, n_bits, v_deq, v_codebook);
                shared_output[d] += w * v_val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize
    float denom = max(softmax_sum[0], 1e-10f);
    for (uint d = tid; d < head_dim; d += tg_size)
        shared_output[d] /= denom;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Un-rotate
    hadamard_rotate_inplace(shared_output, rotation_signs, head_dim, tid, tg_size);

    uint out_base = head_idx * head_dim;
    for (uint d = tid; d < head_dim; d += tg_size)
        output[out_base + d] = half(shared_output[d]);
}
