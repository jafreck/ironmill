// TurboQuant cache write kernel for MLX backend.
//
// Rotates K/V via Hadamard butterfly, quantizes to INT4 or INT8,
// and writes to the quantized KV cache.
// Helpers (hadamard_rotate_inplace, kv_cache_base, etc.) are
// prepended at compile time from mlx_helpers.metal.

[[kernel]] void turboquant_cache_write(
    device const half* kv_proj          [[buffer(0)]],
    device const float* rotation_signs  [[buffer(1)]],
    device char* cache                  [[buffer(2)]],
    device float* scale_buf             [[buffer(3)]],
    device const float* codebook        [[buffer(4)]],
    device const float* boundaries      [[buffer(5)]],
    device const float* qjl_matrix      [[buffer(6)]],
    device uchar* qjl_signs_buf         [[buffer(7)]],
    device float* r_norms_buf           [[buffer(8)]],
    device const uint* params           [[buffer(9)]],
    device float* dummy_out             [[buffer(10)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint num_kv_heads = params[0];
    uint head_dim     = params[1];
    uint max_seq_len  = params[2];
    uint seq_pos      = params[3];
    uint n_bits       = params[4];
    uint n_levels     = params[5];
    uint is_k_cache   = params[6];

    threadgroup float shared_rotated[HEAD_DIM];
    threadgroup float shared_reduce[HEAD_DIM];
    threadgroup char shared_quant[HEAD_DIM];

    uint head_idx = tgid;
    if (head_idx >= num_kv_heads) return;

    // Load input into shared memory
    uint input_base = head_idx * head_dim;
    for (uint i = tid; i < head_dim; i += tg_size)
        shared_rotated[i] = float(kv_proj[input_base + i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hadamard rotation
    hadamard_rotate_inplace(shared_rotated, rotation_signs, head_dim, tid, tg_size);

    if (n_bits == 4) {
        // L2 norm for TurboQuant
        float local_sq = 0.0f;
        for (uint d = tid; d < head_dim; d += tg_size)
            local_sq += shared_rotated[d] * shared_rotated[d];
        shared_reduce[tid] = local_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) shared_reduce[tid] += shared_reduce[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float l2_norm = sqrt(max(shared_reduce[0], 1e-20f));

        if (tid == 0)
            scale_buf[head_idx * max_seq_len + seq_pos] = l2_norm;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Quantize via nearest codebook centroid
        float inv_norm = 1.0f / max(l2_norm, 1e-10f);
        uint n_boundaries = n_levels - 1;
        for (uint d = tid; d < head_dim; d += tg_size) {
            float normalized = shared_rotated[d] * inv_norm;
            uint idx = 0;
            for (uint b = 0; b < n_boundaries; b++) {
                if (normalized >= boundaries[b]) idx = b + 1;
            }
            shared_quant[d] = char(idx);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Stage 2 (K cache only): QJL residual correction (Algorithm 2)
        if (is_k_cache == 1) {
            float local_sq_e = 0.0f;
            for (uint d = tid; d < head_dim; d += tg_size) {
                float normalized = shared_rotated[d] * inv_norm;
                float dequant_val = codebook[uint(shared_quant[d])];
                float residual = normalized - dequant_val;
                shared_rotated[d] = residual;
                local_sq_e += residual * residual;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            shared_reduce[tid] = local_sq_e;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = tg_size / 2; s > 0; s >>= 1) {
                if (tid < s) shared_reduce[tid] += shared_reduce[tid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float r_e_norm = sqrt(max(shared_reduce[0], 1e-20f));
            if (tid == 0)
                r_norms_buf[head_idx * max_seq_len + seq_pos] = r_e_norm;

            for (uint d = tid; d < head_dim; d += tg_size) {
                float proj = 0.0f;
                uint row_base = d * head_dim;
                for (uint k = 0; k < head_dim; k++)
                    proj += qjl_matrix[row_base + k] * shared_rotated[k];
                uchar sign_bit = (proj >= 0.0f) ? uchar(0x8) : uchar(0x0);
                shared_quant[d] = char(uchar(shared_quant[d]) | sign_bit);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Pack INT4 nibbles
        uint packed_stride = head_dim / 2;
        uint cache_base = kv_cache_base(head_idx, max_seq_len, head_dim, 4)
                        + seq_pos * packed_stride;
        for (uint d = tid * 2; d < head_dim; d += tg_size * 2) {
            uchar lo = uchar(shared_quant[d]     & 0xF);
            uchar hi = uchar(shared_quant[d + 1] & 0xF);
            ((device uchar*)cache)[cache_base + d / 2] = lo | (hi << 4);
        }
    } else {
        // INT8: per-head absmax quantization
        float local_max = 0.0f;
        for (uint d = tid; d < head_dim; d += tg_size)
            local_max = max(local_max, fabs(shared_rotated[d]));
        shared_reduce[tid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint reduce_size = 1;
        while (reduce_size < tg_size) reduce_size <<= 1;
        for (uint stride = reduce_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride && (tid + stride) < tg_size)
                shared_reduce[tid] = max(shared_reduce[tid], shared_reduce[tid + stride]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float max_val = max(shared_reduce[0], 1e-10f);
        float dyn_inv_scale = 127.0f / max_val;
        float dyn_deq_scale = max_val / 127.0f;
        if (tid == 0)
            scale_buf[head_idx * max_seq_len + seq_pos] = dyn_deq_scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d = tid; d < head_dim; d += tg_size) {
            float scaled = clamp(shared_rotated[d] * dyn_inv_scale, -128.0f, 127.0f);
            cache[kv_cache_base(head_idx, max_seq_len, head_dim, 8)
                + seq_pos * head_dim + d] = char(rint(scaled));
        }
    }

    // Write a dummy value so MLX sees an output dependency
    if (tgid == 0 && tid == 0) dummy_out[0] = 1.0f;
}
