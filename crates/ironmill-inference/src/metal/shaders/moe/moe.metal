#include <metal_stdlib>
using namespace metal;

// ── Row-wise softmax ────────────────────────────────────────────
//
// Computes softmax along the last dimension of a [token_count, width] matrix.
// Each threadgroup handles one row.
//
// Buffers:
//   buffer(0) data:        [token_count × width]  half  (read/write, in-place)
//   buffer(1) width:       uint
//   buffer(2) token_count: uint
//
// Dispatch: one threadgroup per token, threadgroup_size = min(width, 256).

kernel void moe_softmax(
    device half* data           [[buffer(0)]],
    constant uint& width        [[buffer(1)]],
    constant uint& token_count  [[buffer(2)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tg >= token_count) return;

    device half* row = data + tg * width;

    // 1. Find max for numerical stability
    float local_max = -INFINITY;
    for (uint i = tid; i < width; i += tg_size) {
        local_max = max(local_max, float(row[i]));
    }

    // Reduce max across threadgroup
    threadgroup float shared_max[256];
    shared_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < tg_size) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared_max[0];

    // 2. Compute exp(x - max) and local sum
    float local_sum = 0.0f;
    for (uint i = tid; i < width; i += tg_size) {
        float v = exp(float(row[i]) - row_max);
        row[i] = half(v);
        local_sum += v;
    }

    // Reduce sum across threadgroup
    threadgroup float shared_sum[256];
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < tg_size) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float total = shared_sum[0];

    // 3. Normalize
    float inv_sum = 1.0f / total;
    for (uint i = tid; i < width; i += tg_size) {
        row[i] = half(float(row[i]) * inv_sum);
    }
}

// ── GELU activation (non-gated, in-place) ───────────────────────
//
// Applies GELU to every element: output[i] = gelu(input[i])
// Uses tanh approximation with clamping for numerical stability.
//
// Buffers:
//   buffer(0) data: [size]  half  (read/write, in-place)
//   buffer(1) size: uint
//
// Dispatch: one thread per element.

kernel void moe_gelu(
    device half* data        [[buffer(0)]],
    constant uint& size      [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= size) return;
    float x = float(data[tid]);
    float cube = x * x * x;
    const float kSqrt2OverPi = 0.7978845608f;
    float inner = kSqrt2OverPi * (x + 0.044715f * cube);
    inner = clamp(inner, -10.0f, 10.0f);
    float gelu_x = 0.5f * x * (1.0f + precise::tanh(inner));
    data[tid] = half(gelu_x);
}

// ── Element-wise multiply (in-place) ────────────────────────────
//
// gate[i] *= up[i]
//
// Buffers:
//   buffer(0) gate: [size]  half  (read/write)
//   buffer(1) up:   [size]  half
//   buffer(2) size: uint
//
// Dispatch: one thread per element.

kernel void moe_mul(
    device half* gate         [[buffer(0)]],
    device const half* up     [[buffer(1)]],
    constant uint& size       [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= size) return;
    gate[tid] = half(float(gate[tid]) * float(up[tid]));
}

// ── MoE top-k weighted combine ──────────────────────────────────
//
// For each token, selects the top-k experts by routing weight,
// multiplies each expert output by its weight, and sums into the output.
//
// Buffers:
//   buffer(0) router_probs:  [token_count × num_experts]  half  (softmax output)
//   buffer(1) expert_outputs: [num_experts × token_count × hidden_size]  half
//   buffer(2) output:         [token_count × hidden_size]  half
//   buffer(3) num_experts:    uint
//   buffer(4) top_k:          uint
//   buffer(5) hidden_size:    uint
//   buffer(6) token_count:    uint
//
// Dispatch: one threadgroup per token, threadgroup_size = min(hidden_size, 256).
// Each threadgroup does its own top-k selection (num_experts is small, e.g. 4)
// then computes the weighted sum across hidden dimensions.

kernel void moe_weighted_combine(
    device const half* router_probs    [[buffer(0)]],
    device const half* expert_outputs  [[buffer(1)]],
    device half* output                [[buffer(2)]],
    constant uint& num_experts         [[buffer(3)]],
    constant uint& top_k               [[buffer(4)]],
    constant uint& hidden_size         [[buffer(5)]],
    constant uint& token_count         [[buffer(6)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tg >= token_count) return;

    // Each thread in the threadgroup can read the router probs for this token.
    // num_experts is small (e.g. 4-16), so we do top-k selection in shared mem.
    // Thread 0 computes top-k indices and weights, then all threads use them.
    threadgroup uint   top_indices[16]; // max 16 experts
    threadgroup float  top_weights[16];

    if (tid == 0) {
        device const half* probs = router_probs + tg * num_experts;

        // Simple insertion sort for top-k (num_experts is tiny).
        for (uint k = 0; k < top_k && k < num_experts; k++) {
            float best_val = -INFINITY;
            uint best_idx = 0;
            for (uint e = 0; e < num_experts; e++) {
                float v = float(probs[e]);
                // Check if already selected
                bool already = false;
                for (uint j = 0; j < k; j++) {
                    if (top_indices[j] == e) { already = true; break; }
                }
                if (!already && v > best_val) {
                    best_val = v;
                    best_idx = e;
                }
            }
            top_indices[k] = best_idx;
            top_weights[k] = best_val;
        }

        // Renormalize top-k weights so they sum to 1
        float wsum = 0.0f;
        for (uint k = 0; k < top_k && k < num_experts; k++) {
            wsum += top_weights[k];
        }
        if (wsum > 0.0f) {
            float inv = 1.0f / wsum;
            for (uint k = 0; k < top_k && k < num_experts; k++) {
                top_weights[k] *= inv;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread handles a subset of hidden dims
    uint actual_k = min(top_k, num_experts);
    for (uint d = tid; d < hidden_size; d += tg_size) {
        float acc = 0.0f;
        for (uint k = 0; k < actual_k; k++) {
            uint expert_idx = top_indices[k];
            float w = top_weights[k];
            // expert_outputs layout: [num_experts, token_count, hidden_size]
            uint offset = expert_idx * token_count * hidden_size + tg * hidden_size + d;
            acc += w * float(expert_outputs[offset]);
        }
        output[tg * hidden_size + d] = half(acc);
    }
}
