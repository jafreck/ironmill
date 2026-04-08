#include <metal_stdlib>
using namespace metal;

// Fused embedding lookup + RMSNorm: look up token embeddings and normalize
// in a single kernel, writing directly to the normalized output buffer.
//
// Eliminates the intermediate hidden_state write + read between the
// embedding lookup and first-layer RMSNorm.
//
// Each threadgroup handles one token. Threads cooperatively load the
// embedding row, compute the RMS norm, and write the normalized result.
//
// Buffers:
//   buffer(0) token_ids:       [token_count]                uint
//   buffer(1) embedding_table: [vocab_size × hidden_size]   half
//   buffer(2) norm_weight:     [hidden_size]                half
//   buffer(3) normed_output:   [token_count × hidden_size]  half
//   buffer(4) raw_output:      [token_count × hidden_size]  half (un-normed for residual)
//   buffer(5) hidden_size:     uint
//   buffer(6) token_count:     uint
//   buffer(7) vocab_size:      uint
//   buffer(8) eps:             float
//   buffer(9) embed_scale:     float  (sqrt(hidden_size) for Gemma, 1.0 otherwise)
//
// Dispatch: (token_count, 1, 1) threadgroups,
//           (min(hidden_size, 1024), 1, 1) threads per group.

kernel void fused_embedding_norm(
    device const uint* token_ids        [[buffer(0)]],
    device const half* embedding_table  [[buffer(1)]],
    device const half* norm_weight      [[buffer(2)]],
    device half* normed_output          [[buffer(3)]],
    device half* raw_output             [[buffer(4)]],
    constant uint& hidden_size          [[buffer(5)]],
    constant uint& token_count          [[buffer(6)]],
    constant uint& vocab_size           [[buffer(7)]],
    constant float& eps                 [[buffer(8)]],
    constant float& embed_scale         [[buffer(9)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float sg_partial[32];

    uint token_idx = tgid;
    if (token_idx >= token_count) return;

    uint token_id = token_ids[token_idx];
    uint emb_base = token_id * hidden_size;
    uint out_base = token_idx * hidden_size;
    bool valid = (token_id < vocab_size);

    // Step 1: Load embedding row, apply embed_scale, accumulate sum-of-squares
    float local_sum = 0.0f;
    for (uint i = tid; i < hidden_size; i += tg_size) {
        float val = valid ? float(embedding_table[emb_base + i]) * embed_scale : 0.0f;
        raw_output[out_base + i] = half(val);
        local_sum += val * val;
    }

    // Step 2: Reduce
    float simd_total = simd_sum(local_sum);
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

    float rms_inv = rsqrt(sg_partial[0] / float(hidden_size) + eps);

    // Step 3: Normalize and write
    for (uint i = tid; i < hidden_size; i += tg_size) {
        float val = valid ? float(embedding_table[emb_base + i]) * embed_scale : 0.0f;
        normed_output[out_base + i] = half(val * rms_inv * float(norm_weight[i]));
    }
}
