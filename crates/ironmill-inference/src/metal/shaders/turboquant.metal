// TurboQuant: Hadamard rotation + quantized KV cache (INT8 or INT4)
//
// Shared helpers (kv_cache_base, read_quantized_tile,
// hadamard_rotate_inplace) are prepended at compile time from
// src/shaders/turboquant_helpers.metal.

// ── Metal-only helper: device-pointer dequant (non-tiled access) ─

/// Read one dequantized cache value directly from device memory.
inline float read_quantized(device const char* cache, uint base,
                            uint pos, uint dim, uint head_dim,
                            uint n_bits, float deq_scale,
                            device const float* codebook) {
    if (n_bits == 4) {
        uint packed_stride = head_dim / 2;
        uint byte_idx = base + pos * packed_stride + dim / 2;
        uchar packed = ((device const uchar*)cache)[byte_idx];
        uchar nibble = (dim % 2 == 0) ? (packed & 0xF) : (packed >> 4);
        return codebook[nibble] * deq_scale;
    } else {
        return float(cache[base + pos * head_dim + dim]) * deq_scale;
    }
}

// === Fused Cache Write ===
//
// Rotates K/V via in-place Walsh-Hadamard butterfly, quantizes, writes to cache.
// Supports both INT8 (n_bits=8) and INT4 (n_bits=4) quantization.
// For K cache (is_k_cache=1), also computes QJL residual correction data.
//
// Buffers:
//   buffer(0)  kv_proj:          [num_kv_heads × head_dim]           half
//   buffer(1)  rotation_signs:   [head_dim]                          float (±1.0)
//   buffer(2)  cache:            see below                           char/packed
//   buffer(3)  num_kv_heads:     uint
//   buffer(4)  head_dim:         uint
//   buffer(5)  max_seq_len:      uint
//   buffer(6)  seq_pos:          uint  (current write position)
//   buffer(7)  inv_scale:        float (UNUSED — kept for API compat)
//   buffer(8)  n_bits:           uint  (4 or 8)
//   buffer(9)  scale_buf:        [num_kv_heads × max_seq_len] float
//   buffer(10) codebook:         [n_levels] float (INT4 codebook centroids)
//   buffer(11) boundaries:       [n_levels-1] float (INT4 boundary values)
//   buffer(12) n_levels:         uint (number of codebook levels; K=(b-1)-bit, V=b-bit)
//   buffer(13) qjl_matrix:       [head_dim × head_dim] float (QJL projection)
//   buffer(14) qjl_signs_buf:    UNUSED (signs packed in nibble bit 3; kept for API compat)
//   buffer(15) r_norms_buf:      [num_kv_heads × max_seq_len] float
//   buffer(16) is_k_cache:       uint (1 = K cache, 0 = V cache)
//
// Cache layout:
//   INT8: [num_kv_heads × max_seq_len × head_dim]       1 byte per element
//   INT4: [num_kv_heads × max_seq_len × head_dim/2]     2 elements packed per byte
//
// Dispatch: num_kv_heads threadgroups, head_dim threads per group.

kernel void turboquant_cache_write(
    device const half* kv_proj          [[buffer(0)]],
    device const float* rotation_signs  [[buffer(1)]],
    device char* cache                  [[buffer(2)]],
    constant uint& num_kv_heads         [[buffer(3)]],
    constant uint& head_dim             [[buffer(4)]],
    constant uint& max_seq_len          [[buffer(5)]],
    constant uint& seq_pos              [[buffer(6)]],
    constant float& inv_scale           [[buffer(7)]],  // UNUSED — kept for API compat
    constant uint& n_bits               [[buffer(8)]],
    device float* scale_buf             [[buffer(9)]],
    device const float* codebook        [[buffer(10)]],
    device const float* boundaries      [[buffer(11)]],
    constant uint& n_levels             [[buffer(12)]],
    device const float* qjl_matrix      [[buffer(13)]],
    device uchar* qjl_signs_buf         [[buffer(14)]],
    device float* r_norms_buf           [[buffer(15)]],
    constant uint& is_k_cache           [[buffer(16)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float shared_rotated[HEAD_DIM];
    threadgroup float shared_reduce[HEAD_DIM];
    threadgroup char shared_quant[HEAD_DIM];

    uint head_idx = tgid;
    if (head_idx >= num_kv_heads) return;

    // Step 1: Load input into shared memory as float
    uint input_base = head_idx * head_dim;
    for (uint i = tid; i < head_dim; i += tg_size) {
        shared_rotated[i] = float(kv_proj[input_base + i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: In-place Hadamard rotation
    hadamard_rotate_inplace(shared_rotated, rotation_signs, head_dim, tid, tg_size);

    // Step 3: Compute per-thread absmax (needed for INT8)
    float local_max = 0.0f;
    for (uint out_dim = tid; out_dim < head_dim; out_dim += tg_size) {
        local_max = max(local_max, fabs(shared_rotated[out_dim]));
    }

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

    if (n_bits == 4) {
        // ── TurboQuant Algorithm 2: Rotate → L2 norm → Quantize → QJL → Pack ──
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

        // QJL residual correction (K cache only, per Algorithm 2)
        // Sign bit is packed into bit 3 of each nibble (like outlier path).
        if (is_k_cache == 1) {
            // Compute quantization residual in normalized space
            for (uint d = tid; d < head_dim; d += tg_size) {
                float normalized = shared_rotated[d] * inv_norm;
                float dequant_val = codebook[uint(shared_quant[d])];
                shared_reduce[d] = normalized - dequant_val;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Compute ||residual||₂
            float local_rsq = 0.0f;
            for (uint d = tid; d < head_dim; d += tg_size)
                local_rsq += shared_reduce[d] * shared_reduce[d];
            shared_rotated[tid] = local_rsq;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = tg_size / 2; s > 0; s >>= 1) {
                if (tid < s) shared_rotated[tid] += shared_rotated[tid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float r_norm = sqrt(max(shared_rotated[0], 1e-20f));
            if (tid == 0)
                r_norms_buf[head_idx * max_seq_len + seq_pos] = r_norm;

            // Project residual with QJL matrix S, pack sign into nibble bit 3
            for (uint d = tid; d < head_dim; d += tg_size) {
                float proj = 0.0f;
                uint row_base = d * head_dim;
                for (uint k = 0; k < head_dim; k++)
                    proj += qjl_matrix[row_base + k] * shared_reduce[k];
                uchar sign_bit = (proj >= 0.0f) ? uchar(0x8) : uchar(0x0);
                shared_quant[d] = char(uchar(shared_quant[d]) | sign_bit);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Pack INT4 nibbles into cache bytes
        uint packed_stride = head_dim / 2;
        uint cache_base = kv_cache_base(head_idx, max_seq_len, head_dim, 4)
                        + seq_pos * packed_stride;
        for (uint d = tid * 2; d < head_dim; d += tg_size * 2) {
            uchar lo = uchar(shared_quant[d]     & 0xF);
            uchar hi = (d + 1 < head_dim) ? uchar(shared_quant[d + 1] & 0xF) : 0;
            ((device uchar*)cache)[cache_base + d / 2] = lo | (hi << 4);
        }
    } else {
        // INT8: per-head absmax (unchanged)
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
}


// === Fused TurboQuant Attention ===
//
// Full attention with quantized KV cache using online softmax.
// For INT4, includes QJL residual correction on K cache inner products.
//
// Buffers:
//   buffer(0)  q:               [num_heads × head_dim]              half
//   buffer(1)  k_cache:         quantized KV cache                  char/packed
//   buffer(2)  v_cache:         quantized KV cache                  char/packed
//   buffer(3)  rotation_signs:  [head_dim]                          float (±1.0)
//   buffer(4)  output:          [num_heads × head_dim]              half
//   buffer(5)  num_heads:       uint
//   buffer(6)  num_kv_heads:    uint
//   buffer(7)  head_dim:        uint
//   buffer(8)  max_seq_len:     uint
//   buffer(9)  seq_len:         uint
//   buffer(10) deq_scale:       float (UNUSED — API compat)
//   buffer(11) n_bits:          uint (4 or 8)
//   buffer(12) k_scale_buf:     [num_kv_heads × max_seq_len] float
//   buffer(13) v_scale_buf:     [num_kv_heads × max_seq_len] float
//   buffer(14) k_codebook:      K cache codebook ((b-1)-bit levels for TurboQuant_prod)
//   buffer(15) v_codebook:      V cache codebook (b-bit levels for TurboQuant_mse)
//   buffer(16) qjl_matrix:      [head_dim × head_dim] float
//   buffer(17) k_r_norms:       [num_kv_heads × max_seq_len] float
//
// Dispatch: num_heads threadgroups, min(head_dim, 1024) threads per group.

kernel void turboquant_attention(
    device const half* q                [[buffer(0)]],
    device const char* k_cache          [[buffer(1)]],
    device const char* v_cache          [[buffer(2)]],
    device const float* rotation_signs  [[buffer(3)]],
    device half* output                 [[buffer(4)]],
    constant uint& num_heads            [[buffer(5)]],
    constant uint& num_kv_heads         [[buffer(6)]],
    constant uint& head_dim             [[buffer(7)]],
    constant uint& max_seq_len          [[buffer(8)]],
    constant uint& seq_len              [[buffer(9)]],
    constant float& deq_scale           [[buffer(10)]],
    constant uint& n_bits               [[buffer(11)]],
    device const float* k_scale_buf     [[buffer(12)]],
    device const float* v_scale_buf     [[buffer(13)]],
    device const float* k_codebook      [[buffer(14)]],
    device const float* v_codebook      [[buffer(15)]],
    device const float* qjl_matrix      [[buffer(16)]],
    device const float* k_r_norms       [[buffer(17)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    // Shared memory sized exactly to HEAD_DIM (injected at compile time).
    //   shared_q_rot:  HEAD_DIM × 4     B
    //   shared_s_q:    HEAD_DIM × 4     B
    //   kv_tile_raw:   32 × HEAD_DIM    B (char; aliased K/V)
    //   tile_scales:   32 × 4           B
    //   shared_reduce: HEAD_DIM × 4     B
    //   tile_scores:   32 × 4           B
    //   shared_output: HEAD_DIM × 4     B
    //   softmax/corr:  3 × 4            B
    constexpr uint TILE = 32;

    threadgroup float shared_q_rot[HEAD_DIM];
    threadgroup float shared_s_q[HEAD_DIM];
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

    // ---- Step 1: Load Q and rotate via butterfly ----
    for (uint d = tid; d < head_dim; d += tg_size) {
        shared_q_rot[d] = float(q[q_base + d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hadamard_rotate_inplace(shared_q_rot, rotation_signs, head_dim, tid, tg_size);

    // Precompute S · Q_rot for QJL correction (INT4 K only)
    if (n_bits == 4) {
        for (uint out_d = tid; out_d < head_dim; out_d += tg_size) {
            float proj = 0.0f;
            uint row_base = out_d * head_dim;
            for (uint k = 0; k < head_dim; k++)
                proj += qjl_matrix[row_base + k] * shared_q_rot[k];
            shared_s_q[out_d] = proj;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // QJL correction coefficient: √(π/2) / d per Algorithm 2
    float qjl_factor = (n_bits == 4) ? (sqrt(3.14159265f / 2.0f) / float(head_dim)) : 0.0f;

    // Zero output accumulator
    for (uint d = tid; d < head_dim; d += tg_size) {
        shared_output[d] = 0.0f;
    }
    if (tid == 0) {
        softmax_max[0] = -INFINITY;
        softmax_sum[0] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint bytes_per_pos = (n_bits == 4) ? (head_dim / 2) : head_dim;

    // Precompute reduction-size constant for tree reduction
    uint rs = 1;
    while (rs < tg_size) rs <<= 1;

    // ---- Step 2: Tiled flash attention ----
    for (uint tile_start = 0; tile_start < seq_len; tile_start += TILE) {
        uint tile_end = min(tile_start + TILE, seq_len);
        uint actual_tile = tile_end - tile_start;

        // Cooperative load of K tile raw bytes into threadgroup SRAM
        uint tile_bytes = actual_tile * bytes_per_pos;
        for (uint i = tid; i < tile_bytes; i += tg_size) {
            kv_tile_raw[i] = k_cache[kv_base + tile_start * bytes_per_pos + i];
        }
        for (uint i = tid; i < actual_tile; i += tg_size) {
            tile_scales[i] = k_scale_buf[kv_head * max_seq_len + tile_start + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute QK^T for every position in the tile
        for (uint p = 0; p < actual_tile; p++) {
            float k_deq = tile_scales[p];
            float partial_dot = 0.0f;
            float partial_qjl = 0.0f;

            // INT4: use read_quantized_tile_int4_qjl to extract (b-1)-bit
            // codebook value and QJL sign from nibble simultaneously.
            // INT8: standard codebook dequant, no QJL.
            if (n_bits == 4) {
                for (uint d = tid; d < head_dim; d += tg_size) {
                    float qjl_sign;
                    float k_val = read_quantized_tile_int4_qjl(
                        kv_tile_raw, p, d, head_dim, k_deq,
                        k_codebook, qjl_sign);
                    partial_dot += shared_q_rot[d] * k_val;
                    partial_qjl += shared_s_q[d] * qjl_sign;
                }
            } else {
                for (uint d = tid; d < head_dim; d += tg_size) {
                    float k_val = read_quantized_tile(
                        kv_tile_raw, p, d, head_dim, n_bits, k_deq, k_codebook);
                    partial_dot += shared_q_rot[d] * k_val;
                }
            }

            // Tree reduction for base dot product
            shared_reduce[tid] = partial_dot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = rs / 2; s > 0; s >>= 1) {
                if (tid < s && (tid + s) < tg_size)
                    shared_reduce[tid] += shared_reduce[tid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float base_score = shared_reduce[0];

            // QJL inner product correction for INT4 K cache (Algorithm 2)
            // Correction = deq_scale * ||r|| * (√(π/2)/d) * Σ (S·Q)ᵢ · sign(S·r)ᵢ
            float qjl_correction = 0.0f;
            if (n_bits == 4) {
                uint abs_pos = tile_start + p;
                float r_norm = k_r_norms[kv_head * max_seq_len + abs_pos];

                shared_reduce[tid] = partial_qjl;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint s = rs / 2; s > 0; s >>= 1) {
                    if (tid < s && (tid + s) < tg_size)
                        shared_reduce[tid] += shared_reduce[tid + s];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                qjl_correction = k_deq * r_norm * qjl_factor * shared_reduce[0];
            }

            if (tid == 0)
                tile_scores[p] = (base_score + qjl_correction) * scale;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ---- Per-tile online softmax ----
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

        // Rescale accumulator
        float corr = tile_correction[0];
        for (uint d = tid; d < head_dim; d += tg_size)
            shared_output[d] *= corr;

        // Load V tile raw bytes (aliasing K tile memory)
        for (uint i = tid; i < tile_bytes; i += tg_size) {
            kv_tile_raw[i] = v_cache[kv_base + tile_start * bytes_per_pos + i];
        }
        for (uint i = tid; i < actual_tile; i += tg_size) {
            tile_scales[i] = v_scale_buf[kv_head * max_seq_len + tile_start + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate weighted V from tile
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

    // ---- Step 3: Normalize ----
    float denom = max(softmax_sum[0], 1e-10f);
    for (uint d = tid; d < head_dim; d += tg_size) {
        shared_output[d] /= denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Step 4: Un-rotate via butterfly (self-inverse) ----
    hadamard_rotate_inplace(shared_output, rotation_signs, head_dim, tid, tg_size);

    uint out_base = head_idx * head_dim;
    for (uint d = tid; d < head_dim; d += tg_size) {
        output[out_base + d] = half(shared_output[d]);
    }
}


// ============================================================================
// Outlier Channel Strategy (Section 4.3)
//
// Splits KV dimensions into outlier and non-outlier channel groups.
// Each group is independently rotated (Hadamard), quantized (Lloyd-Max),
// and stored in separate cache regions.
// ============================================================================

// Helper: quantize a sub-vector in shared memory using TurboQuant MSE.
// Writes L2 norm to scale_buf, packed INT4 indices to cache.
// `shared_vec` holds the rotated sub-vector of length `dim`.
// `shared_scratch` is scratch space of length >= dim.
// Returns via `shared_quant` starting at `quant_offset`.
inline void turboquant_quantize_group(
    threadgroup float* shared_vec,
    threadgroup float* shared_reduce,
    threadgroup char* shared_quant,
    device const float* codebook,
    device const float* boundaries,
    uint n_levels,
    device char* cache,
    device float* scale_buf,
    uint dim,
    uint head_idx,
    uint max_seq_len,
    uint seq_pos,
    uint bytes_per_pos,
    uint tid,
    uint tg_size)
{
    // L2 norm
    float local_sq = 0.0f;
    for (uint d = tid; d < dim; d += tg_size)
        local_sq += shared_vec[d] * shared_vec[d];
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

    // Quantize
    float inv_norm = 1.0f / max(l2_norm, 1e-10f);
    uint n_boundaries = n_levels - 1;
    for (uint d = tid; d < dim; d += tg_size) {
        float normalized = shared_vec[d] * inv_norm;
        uint idx = 0;
        for (uint b = 0; b < n_boundaries; b++) {
            if (normalized >= boundaries[b]) idx = b + 1;
        }
        shared_quant[d] = char(idx);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pack INT4
    uint cache_base = head_idx * max_seq_len * bytes_per_pos + seq_pos * bytes_per_pos;
    for (uint d = tid * 2; d < dim; d += tg_size * 2) {
        uchar lo = uchar(shared_quant[d]     & 0xF);
        uchar hi = (d + 1 < dim) ? uchar(shared_quant[d + 1] & 0xF) : 0;
        ((device uchar*)cache)[cache_base + d / 2] = lo | (hi << 4);
    }
}


// === Outlier Cache Write ===
//
// Splits KV vector into outlier and non-outlier groups, applies independent
// TurboQuant to each, stores results in separate cache buffers.
//
// Buffers:
//   buffer(0)  kv_proj:                  [num_kv_heads × head_dim] half
//   buffer(1)  channel_indices:          [head_dim] uint (outlier first, then non-outlier)
//   buffer(2)  outlier_cache:            packed 4-bit
//   buffer(3)  non_outlier_cache:        packed 4-bit
//   buffer(4)  outlier_rotation_signs:   [d_outlier_padded] float
//   buffer(5)  non_outlier_rotation_signs: [d_non_padded] float
//   buffer(6)  outlier_codebook:         [outlier_n_levels] float
//   buffer(7)  outlier_boundaries:       [outlier_n_levels-1] float
//   buffer(8)  non_outlier_codebook:     [non_outlier_n_levels] float
//   buffer(9)  non_outlier_boundaries:   [non_outlier_n_levels-1] float
//   buffer(10) outlier_scale_buf:        [num_kv_heads × max_seq_len] float
//   buffer(11) non_outlier_scale_buf:    [num_kv_heads × max_seq_len] float
//   buffer(12) num_kv_heads:             uint
//   buffer(13) head_dim:                 uint
//   buffer(14) max_seq_len:              uint
//   buffer(15) seq_pos:                  uint
//   buffer(16) n_outlier:                uint
//   buffer(17) d_outlier_padded:         uint
//   buffer(18) d_non_padded:             uint
//   buffer(19) outlier_n_levels:         uint
//   buffer(20) non_outlier_n_levels:     uint
//   buffer(21) is_k_cache:              uint (1=K, 0=V)
//   buffer(22) outlier_qjl_matrix:      [d_outlier_padded²] float
//   buffer(23) non_outlier_qjl_matrix:  [d_non_padded²] float
//   buffer(24) outlier_r_norms_buf:     [num_kv_heads × max_seq_len] float
//   buffer(25) non_outlier_r_norms_buf: [num_kv_heads × max_seq_len] float
//
// Dispatch: num_kv_heads threadgroups, max(d_outlier_padded, d_non_padded) threads.

kernel void turboquant_outlier_cache_write(
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
    constant uint& num_kv_heads                     [[buffer(12)]],
    constant uint& head_dim                         [[buffer(13)]],
    constant uint& max_seq_len                      [[buffer(14)]],
    constant uint& seq_pos                          [[buffer(15)]],
    constant uint& n_outlier                        [[buffer(16)]],
    constant uint& d_outlier_padded                 [[buffer(17)]],
    constant uint& d_non_padded                     [[buffer(18)]],
    constant uint& outlier_n_levels                 [[buffer(19)]],
    constant uint& non_outlier_n_levels             [[buffer(20)]],
    constant uint& is_k_cache                       [[buffer(21)]],
    device const float* outlier_qjl_matrix          [[buffer(22)]],
    device const float* non_outlier_qjl_matrix      [[buffer(23)]],
    device float* outlier_r_norms_buf               [[buffer(24)]],
    device float* non_outlier_r_norms_buf           [[buffer(25)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float shared_outlier[HEAD_DIM];
    threadgroup float shared_non_outlier[HEAD_DIM];
    threadgroup float shared_reduce[HEAD_DIM];
    threadgroup char shared_quant[HEAD_DIM];

    uint head_idx = tgid;
    if (head_idx >= num_kv_heads) return;

    uint n_non = head_dim - n_outlier;
    uint input_base = head_idx * head_dim;

    // Extract outlier channels (zero-pad to d_outlier_padded)
    for (uint i = tid; i < d_outlier_padded; i += tg_size) {
        if (i < n_outlier) {
            uint src_idx = channel_indices[i];
            shared_outlier[i] = float(kv_proj[input_base + src_idx]);
        } else {
            shared_outlier[i] = 0.0f;
        }
    }

    // Extract non-outlier channels (zero-pad to d_non_padded)
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

    // L2 norm
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

    // Quantize
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

    // QJL residual correction (K cache only)
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

    // Pack INT4 nibbles
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

    // L2 norm
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

    // Quantize
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

    // QJL residual correction (K cache only)
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

    // Pack INT4 nibbles
    uint n_bpp = d_non_padded / 2;
    uint n_cache_base = head_idx * max_seq_len * n_bpp + seq_pos * n_bpp;
    for (uint d = tid * 2; d < d_non_padded; d += tg_size * 2) {
        uchar lo = uchar(shared_quant[d]     & 0xF);
        uchar hi = (d + 1 < d_non_padded) ? uchar(shared_quant[d + 1] & 0xF) : 0;
        ((device uchar*)non_outlier_cache)[n_cache_base + d / 2] = lo | (hi << 4);
    }
}


// === Outlier Attention ===
//
// Attention with dual-group quantized KV cache. Each group is independently
// dequantized and un-rotated, then combined for attention scoring.
//
// Buffers mirror the cache write kernel, plus Q and output buffers.
//
// Dispatch: num_heads threadgroups, min(max(d_outlier_padded, d_non_padded), 1024) threads.

kernel void turboquant_outlier_attention(
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
    device half* output                             [[buffer(10)]],
    device const float* k_outlier_scales            [[buffer(11)]],
    device const float* v_outlier_scales            [[buffer(12)]],
    device const float* k_non_outlier_scales        [[buffer(13)]],
    device const float* v_non_outlier_scales        [[buffer(14)]],
    constant uint& num_heads                        [[buffer(15)]],
    constant uint& num_kv_heads                     [[buffer(16)]],
    constant uint& head_dim                         [[buffer(17)]],
    constant uint& max_seq_len                      [[buffer(18)]],
    constant uint& seq_len                          [[buffer(19)]],
    constant uint& n_outlier                        [[buffer(20)]],
    constant uint& d_outlier_padded                 [[buffer(21)]],
    constant uint& d_non_padded                     [[buffer(22)]],
    device const float* outlier_qjl_matrix          [[buffer(23)]],
    device const float* non_outlier_qjl_matrix      [[buffer(24)]],
    device const float* k_outlier_r_norms           [[buffer(25)]],
    device const float* k_non_outlier_r_norms       [[buffer(26)]],
    device const float* v_outlier_codebook          [[buffer(27)]],
    device const float* v_non_outlier_codebook      [[buffer(28)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    // Shared memory sized exactly to HEAD_DIM (injected at compile time).
    //   shared_q_outlier:     HEAD_DIM × 4         B
    //   shared_q_non_outlier: HEAD_DIM × 4         B
    //   shared_s_q_outlier:   HEAD_DIM × 4         B
    //   shared_s_q_non:       HEAD_DIM × 4         B
    //   outlier_kv_tile:      32 × HEAD_DIM_PACKED B (INT4 packed, aliased K/V)
    //   non_outlier_kv_tile:  32 × HEAD_DIM_PACKED B
    //   o/n_tile_scales:      2 × 32 × 4           B
    //   shared_reduce:        HEAD_DIM × 4         B
    //   tile_scores:          32 × 4               B
    //   output_outlier:       HEAD_DIM × 4         B
    //   output_non_outlier:   HEAD_DIM × 4         B
    //   softmax/corr:         3 × 4                B
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

    // ---- Load and rotate Q for both groups ----
    for (uint i = tid; i < d_outlier_padded; i += tg_size) {
        if (i < n_outlier) {
            shared_q_outlier[i] = float(q[q_base + channel_indices[i]]);
        } else {
            shared_q_outlier[i] = 0.0f;
        }
    }
    for (uint i = tid; i < d_non_padded; i += tg_size) {
        if (i < n_non) {
            shared_q_non_outlier[i] = float(q[q_base + channel_indices[n_outlier + i]]);
        } else {
            shared_q_non_outlier[i] = 0.0f;
        }
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

    // QJL correction coefficients: √(π/2) / d per Algorithm 2
    float qjl_factor_o = sqrt(3.14159265f / 2.0f) / float(d_outlier_padded);
    float qjl_factor_n = sqrt(3.14159265f / 2.0f) / float(d_non_padded);

    // Zero output accumulators
    for (uint d = tid; d < d_outlier_padded; d += tg_size)
        shared_output_outlier[d] = 0.0f;
    for (uint d = tid; d < d_non_padded; d += tg_size)
        shared_output_non_outlier[d] = 0.0f;
    if (tid == 0) {
        softmax_max[0] = -INFINITY;
        softmax_sum[0] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint o_bytes_per_pos = d_outlier_padded / 2;
    uint n_bytes_per_pos = d_non_padded / 2;
    uint o_base = kv_head * max_seq_len * o_bytes_per_pos;
    uint n_base = kv_head * max_seq_len * n_bytes_per_pos;

    uint rs = 1;
    while (rs < tg_size) rs <<= 1;

    // ---- Tiled flash attention ----
    for (uint tile_start = 0; tile_start < seq_len; tile_start += TILE) {
        uint tile_end = min(tile_start + TILE, seq_len);
        uint actual_tile = tile_end - tile_start;

        // Cooperative load of K tiles into threadgroup SRAM
        uint o_tile_bytes = actual_tile * o_bytes_per_pos;
        for (uint i = tid; i < o_tile_bytes; i += tg_size) {
            outlier_kv_tile[i] = k_outlier_cache[o_base + tile_start * o_bytes_per_pos + i];
        }
        uint n_tile_bytes = actual_tile * n_bytes_per_pos;
        for (uint i = tid; i < n_tile_bytes; i += tg_size) {
            non_outlier_kv_tile[i] = k_non_outlier_cache[n_base + tile_start * n_bytes_per_pos + i];
        }
        for (uint i = tid; i < actual_tile; i += tg_size) {
            o_tile_scales[i] = k_outlier_scales[kv_head * max_seq_len + tile_start + i];
            n_tile_scales[i] = k_non_outlier_scales[kv_head * max_seq_len + tile_start + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute Q · K dot product with QJL correction for each position in tile
        for (uint p = 0; p < actual_tile; p++) {
            float k_o_deq = o_tile_scales[p];
            float k_n_deq = n_tile_scales[p];

            float partial_dot = 0.0f;
            float partial_qjl_o = 0.0f;
            float partial_qjl_n = 0.0f;

            // Outlier K dequant with QJL sign extraction
            // K cache stores (b-1)-bit codebook index in bits [2:0] and
            // QJL sign in bit [3] of each nibble (Algorithm 2).
            for (uint d = tid; d < d_outlier_padded; d += tg_size) {
                float qjl_sign;
                float k_val = read_quantized_tile_int4_qjl(
                    outlier_kv_tile, p, d, d_outlier_padded, k_o_deq,
                    outlier_codebook, qjl_sign);
                partial_dot += shared_q_outlier[d] * k_val;
                partial_qjl_o += shared_s_q_outlier[d] * qjl_sign;
            }
            // Non-outlier K dequant with QJL sign extraction
            for (uint d = tid; d < d_non_padded; d += tg_size) {
                float qjl_sign;
                float k_val = read_quantized_tile_int4_qjl(
                    non_outlier_kv_tile, p, d, d_non_padded, k_n_deq,
                    non_outlier_codebook, qjl_sign);
                partial_dot += shared_q_non_outlier[d] * k_val;
                partial_qjl_n += shared_s_q_non[d] * qjl_sign;
            }

            // Reduce base dot product
            shared_reduce[tid] = partial_dot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = rs / 2; s > 0; s >>= 1) {
                if (tid < s && (tid + s) < tg_size)
                    shared_reduce[tid] += shared_reduce[tid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float base_score = shared_reduce[0];

            // Reduce outlier QJL dot
            shared_reduce[tid] = partial_qjl_o;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = rs / 2; s > 0; s >>= 1) {
                if (tid < s && (tid + s) < tg_size)
                    shared_reduce[tid] += shared_reduce[tid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float qjl_o = shared_reduce[0];

            // Reduce non-outlier QJL dot
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

        // ---- Per-tile online softmax ----
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

        // Rescale both accumulators
        float corr = tile_correction[0];
        for (uint d = tid; d < d_outlier_padded; d += tg_size)
            shared_output_outlier[d] *= corr;
        for (uint d = tid; d < d_non_padded; d += tg_size)
            shared_output_non_outlier[d] *= corr;

        // Load V tiles (aliasing K tile memory)
        for (uint i = tid; i < o_tile_bytes; i += tg_size) {
            outlier_kv_tile[i] = v_outlier_cache[o_base + tile_start * o_bytes_per_pos + i];
        }
        for (uint i = tid; i < n_tile_bytes; i += tg_size) {
            non_outlier_kv_tile[i] = v_non_outlier_cache[n_base + tile_start * n_bytes_per_pos + i];
        }
        for (uint i = tid; i < actual_tile; i += tg_size) {
            o_tile_scales[i] = v_outlier_scales[kv_head * max_seq_len + tile_start + i];
            n_tile_scales[i] = v_non_outlier_scales[kv_head * max_seq_len + tile_start + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate weighted V from both groups
        for (uint p = 0; p < actual_tile; p++) {
            float w = tile_scores[p];
            float v_o_deq = o_tile_scales[p];
            for (uint d = tid; d < d_outlier_padded; d += tg_size) {
                float v_val = read_quantized_tile(
                    outlier_kv_tile, p, d, d_outlier_padded, 4, v_o_deq, v_outlier_codebook);
                shared_output_outlier[d] += w * v_val;
            }
            float v_n_deq = n_tile_scales[p];
            for (uint d = tid; d < d_non_padded; d += tg_size) {
                float v_val = read_quantized_tile(
                    non_outlier_kv_tile, p, d, d_non_padded, 4, v_n_deq, v_non_outlier_codebook);
                shared_output_non_outlier[d] += w * v_val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Normalize ----
    float denom = max(softmax_sum[0], 1e-10f);
    for (uint d = tid; d < d_outlier_padded; d += tg_size)
        shared_output_outlier[d] /= denom;
    for (uint d = tid; d < d_non_padded; d += tg_size)
        shared_output_non_outlier[d] /= denom;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Un-rotate both groups ----
    hadamard_rotate_inplace(shared_output_outlier, outlier_rotation_signs, d_outlier_padded, tid, tg_size);
    hadamard_rotate_inplace(shared_output_non_outlier, non_outlier_rotation_signs, d_non_padded, tid, tg_size);

    // ---- Scatter back to original channel positions ----
    uint out_base = head_idx * head_dim;
    for (uint i = tid; i < n_outlier; i += tg_size) {
        uint dst_idx = channel_indices[i];
        output[out_base + dst_idx] = half(shared_output_outlier[i]);
    }
    for (uint i = tid; i < n_non; i += tg_size) {
        uint dst_idx = channel_indices[n_outlier + i];
        output[out_base + dst_idx] = half(shared_output_non_outlier[i]);
    }
}
