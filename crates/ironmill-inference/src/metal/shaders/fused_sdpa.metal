#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif

// Tile sizes — tunable via compile-time defines.
#ifndef SDPA_BR
#define SDPA_BR 32
#endif

#ifndef SDPA_BC
#define SDPA_BC 32
#endif

// simdgroup_matrix tile dimension (Apple GPU fixed at 8×8).
constant constexpr uint SM = 8;
constant constexpr uint SIMD_SIZE = 32;

// ============================================================================
// Fused Scaled Dot-Product Attention (FlashAttention-style)
//
// Keeps all intermediate QK^T scores and softmax state in SRAM via
// register tiling across simdgroup lanes.  Handles both prefill
// (token_count > 1) and decode (token_count == 1) in a single kernel.
//
// GQA-aware: one threadgroup per (kv_head_group, q_block). Within the
// threadgroup each simdgroup handles one Q head from the group.
//
// Online softmax with conditional rescaling (FA4 optimisation): the O
// accumulator is rescaled only when the running max actually changes,
// saving ~90 % of rescales for long sequences.
//
// Buffers:
//   buffer(0)  Q            [token_count × num_q_heads × head_dim]        half
//   buffer(1)  K            [num_kv_heads × max_seq_len × head_dim]       half (cache layout)
//   buffer(2)  V            [num_kv_heads × max_seq_len × head_dim]       half (cache layout)
//   buffer(3)  O            [token_count × num_q_heads × head_dim]        half
//   buffer(4)  seq_len      uint  (number of valid positions in cache)
//   buffer(5)  token_count  uint
//   buffer(6)  head_dim     uint  (must == HEAD_DIM)
//   buffer(7)  num_q_heads  uint
//   buffer(8)  num_kv_heads uint
//   buffer(9)  scale        float (1/sqrt(head_dim))
//   buffer(10) max_seq_len  uint  (cache stride)
//
// Dispatch:
//   grid:       (num_kv_groups, ceil(token_count / Br), 1)
//   threadgroup: (32 * heads_per_group, 1, 1)   — one simdgroup per Q head
// ============================================================================

kernel void fused_sdpa(
    device const half*   Q            [[buffer(0)]],
    device const half*   K            [[buffer(1)]],
    device const half*   V            [[buffer(2)]],
    device half*         O            [[buffer(3)]],
    constant uint&       seq_len      [[buffer(4)]],
    constant uint&       token_count  [[buffer(5)]],
    constant uint&       head_dim     [[buffer(6)]],
    constant uint&       num_q_heads  [[buffer(7)]],
    constant uint&       num_kv_heads [[buffer(8)]],
    constant float&      scale        [[buffer(9)]],
    constant uint&       max_seq_len  [[buffer(10)]],
    uint3 tgid       [[threadgroup_position_in_grid]],
    uint  simd_lane   [[thread_index_in_simdgroup]],
    uint  simd_id     [[simdgroup_index_in_threadgroup]])
{
    if (head_dim != HEAD_DIM) return;

    // ── Decode identifiers from grid / threadgroup ──────────────
    uint kv_group   = tgid.x;                    // which KV head group
    uint q_block_id = tgid.y;                    // which Q-token block
    uint heads_per_group = num_q_heads / num_kv_heads;

    // Each simdgroup handles one Q head within the group.
    uint local_head = simd_id;                   // 0 .. heads_per_group-1
    if (local_head >= heads_per_group) return;

    uint q_head   = kv_group * heads_per_group + local_head;
    uint kv_head  = kv_group;

    // Q-token range for this threadgroup.
    uint q_start = q_block_id * SDPA_BR;
    uint q_end   = min(q_start + SDPA_BR, token_count);
    uint actual_q = q_end - q_start;
    if (actual_q == 0) return;

    // ── Per-lane register accumulators ──────────────────────────
    // Each lane accumulates a slice of the output rows it is responsible
    // for.  With Br Q rows and HEAD_DIM columns, we stripe across lanes
    // within the simdgroup (SIMD_SIZE=32).
    //
    // For efficiency we keep per-row softmax state in registers and
    // broadcast via simd_shuffle when needed.

    // Per-row online-softmax state (only lanes 0..actual_q-1 are active
    // for the row-max / row-sum, but we initialise all to avoid branches
    // inside the KV-tile loop).
    float row_max[SDPA_BR];
    float row_sum[SDPA_BR];
    float o_acc[SDPA_BR * (HEAD_DIM / SIMD_SIZE)];

    for (uint r = 0; r < SDPA_BR; ++r) {
        row_max[r] = -INFINITY;
        row_sum[r] = 0.0f;
    }
    for (uint i = 0; i < SDPA_BR * (HEAD_DIM / SIMD_SIZE); ++i)
        o_acc[i] = 0.0f;

    // ── Preload Q tile into registers ───────────────────────────
    // Q layout: [token_count, num_q_heads, head_dim]
    // Each lane loads head_dim/SIMD_SIZE elements per row.
    float q_reg[SDPA_BR * (HEAD_DIM / SIMD_SIZE)];
    for (uint r = 0; r < actual_q; ++r) {
        uint q_row = q_start + r;
        uint q_base = (q_row * num_q_heads + q_head) * HEAD_DIM;
        for (uint c = 0; c < HEAD_DIM / SIMD_SIZE; ++c) {
            q_reg[r * (HEAD_DIM / SIMD_SIZE) + c] =
                float(Q[q_base + c * SIMD_SIZE + simd_lane]);
        }
    }
    // Zero remaining rows to avoid NaN in dot products.
    for (uint r = actual_q; r < SDPA_BR; ++r) {
        for (uint c = 0; c < HEAD_DIM / SIMD_SIZE; ++c)
            q_reg[r * (HEAD_DIM / SIMD_SIZE) + c] = 0.0f;
    }

    // ── Iterate over KV tiles ───────────────────────────────────
    // KV cache layout: [num_kv_heads, max_seq_len, head_dim]
    uint kv_base = kv_head * max_seq_len * HEAD_DIM;

    for (uint kv_start = 0; kv_start < seq_len; kv_start += SDPA_BC) {
        uint kv_end   = min(kv_start + SDPA_BC, seq_len);
        uint actual_kv = kv_end - kv_start;

        // -- Causal skip: if ALL Q rows are masked for this KV tile, skip.
        //    For decode (token_count == 1), seq_len already equals the
        //    causal length so no tile is fully masked.
        if (token_count > 1) {
            // Last Q position in this block
            uint last_q_in_block = seq_len - token_count + min(q_start + actual_q - 1, token_count - 1);
            // All KV positions >= last_q_in_block + 1 are fully masked for all Q rows in this block
            if (kv_start > last_q_in_block) break;
        }

        // -- Compute S = Q · K^T for this tile (Br × Bc) ----------------
        // Each lane computes one element of S per (r, p) pair using
        // simd_sum for the dot-product reduction across HEAD_DIM.
        float s_tile[SDPA_BR * SDPA_BC];

        for (uint p = 0; p < actual_kv; ++p) {
            uint kv_pos = kv_start + p;
            uint k_base = kv_base + kv_pos * HEAD_DIM;

            // Load one K vector striped across lanes.
            float k_val[HEAD_DIM / SIMD_SIZE];
            for (uint c = 0; c < HEAD_DIM / SIMD_SIZE; ++c)
                k_val[c] = float(K[k_base + c * SIMD_SIZE + simd_lane]);

            for (uint r = 0; r < SDPA_BR; ++r) {
                float dot = 0.0f;
                for (uint c = 0; c < HEAD_DIM / SIMD_SIZE; ++c)
                    dot += q_reg[r * (HEAD_DIM / SIMD_SIZE) + c] * k_val[c];
                dot = simd_sum(dot);

                // Apply causal mask for prefill.
                bool masked = false;
                if (token_count > 1) {
                    uint q_pos = q_start + r;
                    // Causal: query at position q_pos can attend to
                    // KV positions 0 .. (seq_len - token_count + q_pos).
                    uint causal_limit = seq_len - token_count + q_pos + 1;
                    masked = (r >= actual_q) || (kv_pos >= causal_limit);
                } else {
                    masked = (r >= actual_q);
                }

                s_tile[r * SDPA_BC + p] = masked ? -INFINITY : (dot * scale);
            }
        }
        // Pad unused KV positions with -inf.
        for (uint p = actual_kv; p < SDPA_BC; ++p)
            for (uint r = 0; r < SDPA_BR; ++r)
                s_tile[r * SDPA_BC + p] = -INFINITY;

        // -- Online softmax per Q row (lane 0 has the reduced values) ----
        // All lanes computed the same dot via simd_sum, so every lane has
        // identical s_tile values.  We can therefore compute per-row max /
        // sum redundantly on every lane (cheap — Bc iterations).

        for (uint r = 0; r < actual_q; ++r) {
            // Tile max.
            float tile_max = -INFINITY;
            for (uint p = 0; p < actual_kv; ++p)
                tile_max = max(tile_max, s_tile[r * SDPA_BC + p]);

            float old_max = row_max[r];
            float new_max = max(old_max, tile_max);

            // FA4 conditional rescale: only when max actually changes.
            if (new_max > old_max) {
                float corr = exp(old_max - new_max);
                row_sum[r] *= corr;
                // Rescale O accumulator for this row.
                for (uint c = 0; c < HEAD_DIM / SIMD_SIZE; ++c)
                    o_acc[r * (HEAD_DIM / SIMD_SIZE) + c] *= corr;
                row_max[r] = new_max;
            }

            // Exponentiate and accumulate denominator.
            float tile_sum = 0.0f;
            for (uint p = 0; p < actual_kv; ++p) {
                float w = exp(s_tile[r * SDPA_BC + p] - row_max[r]);
                s_tile[r * SDPA_BC + p] = w;
                tile_sum += w;
            }
            row_sum[r] += tile_sum;
        }

        // -- Accumulate O += S · V for this tile ─────────────────────────
        // V layout: [num_kv_heads, max_seq_len, head_dim]
        for (uint p = 0; p < actual_kv; ++p) {
            uint kv_pos = kv_start + p;
            uint v_base = kv_base + kv_pos * HEAD_DIM;

            // Load V vector striped across lanes.
            float v_val[HEAD_DIM / SIMD_SIZE];
            for (uint c = 0; c < HEAD_DIM / SIMD_SIZE; ++c)
                v_val[c] = float(V[v_base + c * SIMD_SIZE + simd_lane]);

            for (uint r = 0; r < actual_q; ++r) {
                float w = s_tile[r * SDPA_BC + p];
                if (w > 0.0f) {
                    for (uint c = 0; c < HEAD_DIM / SIMD_SIZE; ++c)
                        o_acc[r * (HEAD_DIM / SIMD_SIZE) + c] += w * v_val[c];
                }
            }
        }
    } // end KV tile loop

    // ── Normalise and write output ──────────────────────────────
    // O layout: [token_count, num_q_heads, head_dim]
    for (uint r = 0; r < actual_q; ++r) {
        float denom = max(row_sum[r], 1e-10f);
        uint o_base = ((q_start + r) * num_q_heads + q_head) * HEAD_DIM;
        for (uint c = 0; c < HEAD_DIM / SIMD_SIZE; ++c) {
            O[o_base + c * SIMD_SIZE + simd_lane] =
                half(o_acc[r * (HEAD_DIM / SIMD_SIZE) + c] / denom);
        }
    }
}
