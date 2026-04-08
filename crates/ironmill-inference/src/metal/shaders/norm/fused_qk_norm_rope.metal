#include <metal_stdlib>
using namespace metal;

#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif

#ifndef ROTARY_DIM
#define ROTARY_DIM HEAD_DIM
#endif

// Fused QK-norm + partial RoPE: RMSNorm each head then apply rotary
// embeddings to the first ROTARY_DIM dimensions only.
//
// When ROTARY_DIM < HEAD_DIM (e.g. Qwen 3.5 with partial_rotary_factor=0.25),
// only the first ROTARY_DIM dims are rotated; the remaining dims are just
// normalized and written back unchanged.
//
// RoPE uses half-split pairing WITHIN the rotary dimensions:
//   pairs (d, d + ROTARY_DIM/2)  for d = 0..ROTARY_DIM/2-1.
//
// Dispatch: (num_heads_total, token_count, 1) threadgroups,
//           (min(HEAD_DIM, 1024), 1, 1) threads per group.
//
// The first `nh` threadgroups (per token) process Q heads,
// the next `nkv` process K heads.  norm_weight is [head_dim] and
// repeats across all heads of the same type.
//
// Buffers:
//   buffer(0)  q_proj:     [token_count × nh × head_dim]      half (in-place)
//   buffer(1)  k_proj:     [token_count × nkv × head_dim]     half (in-place)
//   buffer(2)  q_norm_w:   [head_dim]                          half
//   buffer(3)  k_norm_w:   [head_dim]                          half
//   buffer(4)  cos_cache:  [max_seq × ROTARY_DIM/2]            half
//   buffer(5)  sin_cache:  [max_seq × ROTARY_DIM/2]            half
//   buffer(6)  nh:          uint
//   buffer(7)  nkv:         uint
//   buffer(8)  head_dim:    uint
//   buffer(9)  seq_offset:  uint
//   buffer(10) token_count: uint
//   buffer(11) eps:         float

kernel void fused_qk_norm_rope(
    device half* q_proj          [[buffer(0)]],
    device half* k_proj          [[buffer(1)]],
    device const half* q_norm_w  [[buffer(2)]],
    device const half* k_norm_w  [[buffer(3)]],
    device const half* cos_cache [[buffer(4)]],
    device const half* sin_cache [[buffer(5)]],
    constant uint& nh            [[buffer(6)]],
    constant uint& nkv           [[buffer(7)]],
    constant uint& head_dim_u    [[buffer(8)]],
    constant uint& seq_offset    [[buffer(9)]],
    constant uint& token_count   [[buffer(10)]],
    constant float& eps          [[buffer(11)]],
    uint3 tid_3d  [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tg_size_3d [[threads_per_threadgroup]])
{
    uint tid = tid_3d.x;
    uint tg_size = tg_size_3d.x;
    uint head_slot = tgid.x;       // 0..nh+nkv-1
    uint token_idx = tgid.y;
    if (token_idx >= token_count) return;

    bool is_q = (head_slot < nh);
    uint head_idx = is_q ? head_slot : (head_slot - nh);
    uint num_heads = is_q ? nh : nkv;
    if (head_idx >= num_heads) return;

    device half* proj = is_q ? q_proj : k_proj;
    device const half* norm_w = is_q ? q_norm_w : k_norm_w;
    uint base = (token_idx * num_heads + head_idx) * HEAD_DIM;

    // Shared memory for cross-simdgroup reduction.
    threadgroup float sg_partial[32];

    // Step 1: sum of squares for RMSNorm (over ALL head dims)
    float local_sum = 0.0f;
    for (uint d = tid; d < HEAD_DIM; d += tg_size) {
        float v = float(proj[base + d]);
        local_sum += v * v;
    }

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

    float rms_inv = rsqrt(sg_partial[0] / float(HEAD_DIM) + eps);

    // Step 2: RMSNorm + partial RoPE
    uint half_rotary = ROTARY_DIM / 2;
    uint half_dim = HEAD_DIM / 2;
    uint pos = seq_offset + token_idx;

    // Part A: First ROTARY_DIM dims — normalize + RoPE rotation.
    // Pairing is (d, d + half_rotary) within the rotary dimensions.
    // Cache layout: [pos × half_dim + d] — same stride as standalone rope kernel.
    for (uint d = tid; d < half_rotary; d += tg_size) {
        float x_lo = float(proj[base + d]) * rms_inv * float(norm_w[d]);
        float x_hi = float(proj[base + d + half_rotary]) * rms_inv * float(norm_w[d + half_rotary]);

        uint cs_idx = pos * half_dim + d;
        float cos_val = float(cos_cache[cs_idx]);
        float sin_val = float(sin_cache[cs_idx]);

        proj[base + d]                = half(x_lo * cos_val - x_hi * sin_val);
        proj[base + d + half_rotary]  = half(x_lo * sin_val + x_hi * cos_val);
    }

    // Part B: Remaining dims — just normalize (no rotation).
    for (uint d = tid + ROTARY_DIM; d < HEAD_DIM; d += tg_size) {
        proj[base + d] = half(float(proj[base + d]) * rms_inv * float(norm_w[d]));
    }
}
