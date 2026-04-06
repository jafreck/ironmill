#include <metal_stdlib>
using namespace metal;

// ── GDN Conv1d + SiLU kernel ────────────────────────────────────
//
// For each channel in qkv_dim:
//   1. Shift conv_state left by 1, append new input value
//   2. Dot product: sum_k(conv_weight[ch, k] * state_window[k])
//      where state_window = [conv_state[ch, 0..kernel-2], input_qkv[ch]]
//   3. Apply SiLU activation: x * sigmoid(x)
//
// Buffers:
//   buffer(0) input_qkv:   [qkv_dim] half — raw projection output
//   buffer(1) conv_weight:  [qkv_dim * kernel_size] half — [ch, k] layout
//   buffer(2) conv_state:   [qkv_dim * conv_state_len] half — [ch, time] layout, read/write
//   buffer(3) output:       [qkv_dim] half — conv + silu result
//   buffer(4) params:       uint4 — (qkv_dim, kernel_size, conv_state_len, 0)
//
// Dispatch: one thread per channel (qkv_dim threads total).

kernel void gdn_conv1d_silu(
    device const half* input_qkv   [[buffer(0)]],
    device const half* conv_weight  [[buffer(1)]],
    device half* conv_state         [[buffer(2)]],
    device half* output             [[buffer(3)]],
    constant uint4& params          [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint qkv_dim = params.x;
    uint kernel_size = params.y;
    uint conv_state_len = params.z; // = kernel_size - 1

    if (tid >= qkv_dim) return;

    uint ch = tid;
    uint state_base = ch * conv_state_len;
    uint w_base = ch * kernel_size;

    // Match HF causal_conv1d_update: concat([state, new_val]), conv, update state.
    // State stores the last (kernel-1) values. The full window is [state..., new_val].
    // Dot product uses ALL kernel weights over the full window of kernel_size values.

    float val = 0.0f;
    // weight[0] * state[0], weight[1] * state[1], ..., weight[kernel-2] * state[kernel-2]
    for (uint j = 0; j < conv_state_len; j++) {
        val += float(conv_weight[w_base + j]) * float(conv_state[state_base + j]);
    }
    // weight[kernel-1] * new_value
    val += float(conv_weight[w_base + conv_state_len]) * float(input_qkv[ch]);

    // Update state: shift left, append new value
    for (uint k = 0; k < conv_state_len - 1; k++) {
        conv_state[state_base + k] = conv_state[state_base + k + 1];
    }
    conv_state[state_base + conv_state_len - 1] = input_qkv[ch];

    // SiLU activation
    float silu_val = val / (1.0f + exp(-val));
    output[ch] = half(silu_val);
}


// ── GDN Recurrent State Update kernel ───────────────────────────
//
// Per-head recurrent update for decode (single token):
//   1. Compute gates: beta = sigmoid(b), dt = softplus(a + dt_bias),
//      decay = exp(-exp(A_log) * dt)
//   2. Update state: S[h] = decay * S[h] + beta * outer(v[h], k[h])
//   3. Compute output: o[h] = S[h] @ q[h]
//
// Buffers:
//   buffer(0) conv_out:       [qkv_dim] half — post-conv SiLU output (q,k,v concatenated)
//   buffer(1) a_proj:         [num_v_heads] half — projected alpha
//   buffer(2) b_proj:         [num_v_heads] half — projected beta
//   buffer(3) a_log:          [num_v_heads] half — log-space decay param (model weight)
//   buffer(4) dt_bias:        [num_v_heads] half — softplus bias (model weight)
//   buffer(5) recurrent_state: [num_v_heads * v_head_dim * k_head_dim] half — read/write
//   buffer(6) output:         [value_dim] half — raw output before gating
//   buffer(7) params:         uint params[6] — (key_dim, value_dim, num_v_heads, k_head_dim, v_head_dim, 0)
//
// Dispatch: one threadgroup per head (num_v_heads threadgroups),
//           v_head_dim threads per threadgroup.
// Each thread handles one row of the state matrix for its head.

kernel void gdn_recurrent_update(
    device const half* conv_out        [[buffer(0)]],
    device const half* a_proj          [[buffer(1)]],
    device const half* b_proj          [[buffer(2)]],
    device const half* a_log           [[buffer(3)]],
    device const half* dt_bias         [[buffer(4)]],
    device float* recurrent_state      [[buffer(5)]],
    device half* output                [[buffer(6)]],
    constant uint* params              [[buffer(7)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint h_idx [[threadgroup_position_in_grid]])
{
    uint key_dim = params[0];
    uint value_dim = params[1];
    uint num_v_heads = params[2];
    uint k_head_dim = params[3];
    uint v_head_dim = params[4];
    uint num_k_heads = params[5];

    if (h_idx >= num_v_heads) return;
    uint vi = tid;
    if (vi >= v_head_dim) return;

    // GQA mapping: multiple value heads share the same Q/K head.
    uint k_idx = h_idx * num_k_heads / num_v_heads;

    // Split conv_out into q, k, v.
    device const half* q_head = conv_out + k_idx * k_head_dim;
    device const half* k_head = conv_out + key_dim + k_idx * k_head_dim;
    device const half* v_head = conv_out + 2 * key_dim + h_idx * v_head_dim;

    // L2-normalize Q and K per head (use_qk_l2norm_in_kernel=True in HF)
    float q_sq = 0.0f, k_sq = 0.0f;
    for (uint ki = 0; ki < k_head_dim; ki++) {
        float qv = float(q_head[ki]);
        float kv = float(k_head[ki]);
        q_sq += qv * qv;
        k_sq += kv * kv;
    }
    float q_inv = rsqrt(q_sq + 1e-6f);
    float k_inv = rsqrt(k_sq + 1e-6f);
    float scale = rsqrt(float(k_head_dim)); // query scaling

    // Gates
    float b_val = float(b_proj[h_idx]);
    float beta = 1.0f / (1.0f + exp(-b_val));
    float a_val = float(a_proj[h_idx]) + float(dt_bias[h_idx]);
    float dt = (a_val > 20.0f) ? a_val : log(1.0f + exp(a_val));
    float decay = exp(-exp(float(a_log[h_idx])) * dt);

    uint s_base = h_idx * v_head_dim * k_head_dim;
    uint row_base = s_base + vi * k_head_dim;

    // Delta rule: S = decay*S + k_norm ⊗ beta*(v - S_decayed @ k_norm)
    // Step 1: Decay + memory read
    float kv_mem = 0.0f;
    for (uint ki = 0; ki < k_head_dim; ki++) {
        float s_decayed = decay * recurrent_state[row_base + ki];
        kv_mem += s_decayed * float(k_head[ki]) * k_inv;
    }

    // Step 2: Delta correction
    float delta_vi = beta * (float(v_head[vi]) - kv_mem);

    // Step 3: State update + output (q scaled and normalized)
    float o_sum = 0.0f;
    for (uint ki = 0; ki < k_head_dim; ki++) {
        float s_decayed = decay * recurrent_state[row_base + ki];
        float k_n = float(k_head[ki]) * k_inv;
        float s_new = s_decayed + k_n * delta_vi;
        recurrent_state[row_base + ki] = s_new;
        o_sum += s_new * float(q_head[ki]) * q_inv * scale;
    }

    output[h_idx * v_head_dim + vi] = half(o_sum);
}


// ── GDN Prefill Conv1d + SiLU kernel ──────────────────────────────
//
// Batched version: processes ALL tokens sequentially per channel.
// One thread per channel — loops over token_count internally.
//
// Buffers:
//   buffer(0) all_qkv:      [token_count * qkv_dim] half — batched projection output
//   buffer(1) conv_weight:   [qkv_dim * kernel_size] half — [ch, k] layout
//   buffer(2) conv_state:    [qkv_dim * conv_state_len] half — [ch, time] layout, read/write
//   buffer(3) all_conv_out:  [token_count * qkv_dim] half — batched conv + silu result
//   buffer(4) params:        uint4 — (qkv_dim, kernel_size, conv_state_len, token_count)
//
// Dispatch: qkv_dim threads total, ONE dispatch for all tokens.

kernel void gdn_prefill_conv1d_silu(
    device const half* all_qkv       [[buffer(0)]],
    device const half* conv_weight    [[buffer(1)]],
    device half* conv_state           [[buffer(2)]],
    device half* all_conv_out         [[buffer(3)]],
    constant uint4& params            [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint qkv_dim = params.x;
    uint kernel_size = params.y;
    uint conv_state_len = params.z;
    uint token_count = params.w;

    uint ch = tid;
    if (ch >= qkv_dim) return;

    uint state_base = ch * conv_state_len;
    uint w_base = ch * kernel_size;

    for (uint t = 0; t < token_count; t++) {
        half new_val = all_qkv[t * qkv_dim + ch];

        // Conv: dot product over full window [state..., new_val] using ALL weights
        float val = 0.0f;
        for (uint j = 0; j < conv_state_len; j++) {
            val += float(conv_weight[w_base + j]) * float(conv_state[state_base + j]);
        }
        val += float(conv_weight[w_base + conv_state_len]) * float(new_val);

        // Update state: shift left, append new value
        for (uint k = 0; k < conv_state_len - 1; k++) {
            conv_state[state_base + k] = conv_state[state_base + k + 1];
        }
        conv_state[state_base + conv_state_len - 1] = new_val;

        // SiLU
        float silu_val = val / (1.0f + exp(-val));
        all_conv_out[t * qkv_dim + ch] = half(silu_val);
    }
}


// ── GDN Prefill Recurrent kernel ──────────────────────────────────
//
// Batched version: processes ALL tokens sequentially per head.
// One threadgroup per head, v_head_dim threads per threadgroup.
// Each thread handles one row of the state matrix and loops over tokens.
// Includes RMSNorm + silu(z) output gating (fused).
//
// Buffers:
//   buffer(0)  all_conv_out:    [token_count * qkv_dim] half — post-conv SiLU output
//   buffer(1)  all_a:           [token_count * num_v_heads] half — projected alpha
//   buffer(2)  all_b:           [token_count * num_v_heads] half — projected beta
//   buffer(3)  a_log:           [num_v_heads] half — log-space decay param
//   buffer(4)  dt_bias:         [num_v_heads] half — softplus bias
//   buffer(5)  norm_weight:     [v_head_dim] half — per-head RMSNorm weight
//   buffer(6)  all_z:           [token_count * value_dim] half — z projection
//   buffer(7)  recurrent_state: [num_v_heads * v_head_dim * k_head_dim] half
//   buffer(8)  all_output:      [token_count * value_dim] half — gated output
//   buffer(9)  params:          uint[8] — dims + eps
//
// Dispatch: num_v_heads threadgroups × v_head_dim threads.

kernel void gdn_prefill_recurrent(
    device const half* all_conv_out     [[buffer(0)]],
    device const half* all_a            [[buffer(1)]],
    device const half* all_b            [[buffer(2)]],
    device const half* a_log            [[buffer(3)]],
    device const half* dt_bias          [[buffer(4)]],
    device const half* norm_weight      [[buffer(5)]],
    device const half* all_z            [[buffer(6)]],
    device float* recurrent_state       [[buffer(7)]],
    device half* all_output             [[buffer(8)]],
    constant uint* params               [[buffer(9)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint h_idx [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint token_count = params[0];
    uint qkv_dim = params[1];
    uint key_dim = params[2];
    uint value_dim = params[3];
    uint num_v_heads = params[4];
    uint k_head_dim = params[5];
    uint v_head_dim = params[6];
    float eps = as_type<float>(params[7]);
    uint num_k_heads = params[8];

    if (h_idx >= num_v_heads) return;
    uint vi = tid;
    if (vi >= v_head_dim) return;

    // GQA mapping: multiple value heads share the same Q/K head.
    uint k_idx = h_idx * num_k_heads / num_v_heads;

    uint s_base = h_idx * v_head_dim * k_head_dim + vi * k_head_dim;

    // Shared memory for cross-simdgroup RMSNorm reduction
    threadgroup float sg_partial[32];

    for (uint t = 0; t < token_count; t++) {
        device const half* q_head = all_conv_out + t * qkv_dim + k_idx * k_head_dim;
        device const half* k_head = all_conv_out + t * qkv_dim + key_dim + k_idx * k_head_dim;
        device const half* v_head = all_conv_out + t * qkv_dim + 2 * key_dim + h_idx * v_head_dim;

        // L2-normalize Q and K per head
        float q_sq = 0.0f, k_sq = 0.0f;
        for (uint ki = 0; ki < k_head_dim; ki++) {
            float qv = float(q_head[ki]);
            float kv = float(k_head[ki]);
            q_sq += qv * qv;
            k_sq += kv * kv;
        }
        float q_inv = rsqrt(q_sq + 1e-6f);
        float k_inv = rsqrt(k_sq + 1e-6f);
        float scale = rsqrt(float(k_head_dim));

        // Gates
        float b_val = float(all_b[t * num_v_heads + h_idx]);
        float beta = 1.0f / (1.0f + exp(-b_val));
        float a_val = float(all_a[t * num_v_heads + h_idx]) + float(dt_bias[h_idx]);
        float dt = (a_val > 20.0f) ? a_val : log(1.0f + exp(a_val));
        float decay = exp(-exp(float(a_log[h_idx])) * dt);

        // Delta rule: memory read
        float kv_mem = 0.0f;
        for (uint ki = 0; ki < k_head_dim; ki++) {
            float s_decayed = decay * recurrent_state[s_base + ki];
            kv_mem += s_decayed * float(k_head[ki]) * k_inv;
        }

        // Delta correction + state update + output
        float delta_vi = beta * (float(v_head[vi]) - kv_mem);
        float o_sum = 0.0f;
        for (uint ki = 0; ki < k_head_dim; ki++) {
            float s_decayed = decay * recurrent_state[s_base + ki];
            float k_n = float(k_head[ki]) * k_inv;
            float s_new = s_decayed + k_n * delta_vi;
            recurrent_state[s_base + ki] = s_new;
            o_sum += s_new * float(q_head[ki]) * q_inv * scale;
        }

        // Per-head RMSNorm on output
        float sq = o_sum * o_sum;
        float simd_total = simd_sum(sq);
        uint sg_idx = tid / 32;
        if (tid % 32 == 0) sg_partial[sg_idx] = simd_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            uint num_sg = (tg_size + 31) / 32;
            float total = 0.0f;
            for (uint i = 0; i < num_sg; i++) total += sg_partial[i];
            sg_partial[0] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float rms_inv = rsqrt(sg_partial[0] / float(v_head_dim) + eps);
        float normed = o_sum * rms_inv * float(norm_weight[vi]);

        // silu(z) gating
        float z_val = float(all_z[t * value_dim + h_idx * v_head_dim + vi]);
        float z_silu = z_val / (1.0f + exp(-z_val));

        all_output[t * value_dim + h_idx * v_head_dim + vi] = half(normed * z_silu);

        threadgroup_barrier(mem_flags::mem_threadgroup); // sync before next token
    }
}


// ── GDN Output Gate kernel ──────────────────────────────────────
//
// Per-head: RMSNorm on raw output, then multiply by silu(z).
//
// Buffers:
//   buffer(0) raw_output:   [value_dim] half — from recurrent update
//   buffer(1) z:            [value_dim] half — z projection output
//   buffer(2) norm_weight:  [v_head_dim] half — per-head RMSNorm weight
//   buffer(3) output:       [value_dim] half — gated output
//   buffer(4) params:       uint4 — (num_v_heads, v_head_dim, eps_bits, 0)
//
// Dispatch: one threadgroup per head (num_v_heads threadgroups),
//           v_head_dim threads per threadgroup.

kernel void gdn_output_gate(
    device const half* raw_output   [[buffer(0)]],
    device const half* z            [[buffer(1)]],
    device const half* norm_weight  [[buffer(2)]],
    device half* output             [[buffer(3)]],
    constant uint4& params          [[buffer(4)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint h_idx [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint num_v_heads = params.x;
    uint v_head_dim = params.y;
    float eps = as_type<float>(params.z);

    if (h_idx >= num_v_heads) return;
    if (tid >= v_head_dim) return;

    uint head_base = h_idx * v_head_dim;

    // Shared memory for cross-simdgroup reduction.
    threadgroup float sg_partial[32];

    // Step 1: Compute sum of squares for RMSNorm.
    float val = float(raw_output[head_base + tid]);
    float sq = val * val;

    // Reduce across threads using simd_sum + threadgroup reduction.
    float simd_total = simd_sum(sq);
    uint sg_idx = tid / 32;
    if (tid % 32 == 0) sg_partial[sg_idx] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        uint num_sg = (tg_size + 31) / 32;
        float total = 0.0f;
        for (uint i = 0; i < num_sg; i++) total += sg_partial[i];
        sg_partial[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = rsqrt(sg_partial[0] / float(v_head_dim) + eps);

    // Step 2: RMSNorm + silu(z) gating.
    float normed = val * rms_inv * float(norm_weight[tid]);

    float z_val = float(z[head_base + tid]);
    float z_silu = z_val / (1.0f + exp(-z_val));

    output[head_base + tid] = half(normed * z_silu);
}
