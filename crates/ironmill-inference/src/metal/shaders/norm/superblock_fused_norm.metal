// ── Superblock fused residual+norm+affine matvec INT4 ───────────
//
// Extracted from fused_residual_norm.metal for per-GS compilation.
// build.rs prepends #define GS <value> before this file.
//
// Reads weights from superblock layout with inline scale/zero.
//
// Dispatch: (N, 1, 1) threadgroups, (32, 1, 1) threads.

#include <metal_stdlib>
using namespace metal;

#ifndef GS
#error "GS must be defined by build.rs"
#endif

constant constexpr uint FRN_SB_HEADER_BYTES = 0;  // separate arrays
constant constexpr uint FRN_SB_BYTES_INT4 = GS / 2;
constant constexpr uint FRN_BLK_K = 8;

kernel void superblock_fused_residual_norm_affine_matvec_int4(
    device const half* a               [[buffer(0)]],
    device const half* b               [[buffer(1)]],
    device const half* norm_weight      [[buffer(2)]],
    device half* residual_output       [[buffer(3)]],
    device const uchar* W              [[buffer(4)]],   // data [N, K/2]
    device half* C                     [[buffer(5)]],
    constant uint* params              [[buffer(6)]],
    device const half* awq_scales      [[buffer(7)]],
    constant uint& has_awq             [[buffer(8)]],
    device half* normed_output         [[buffer(9)]],
    device const half* W_scales        [[buffer(10)]],  // [N, K/GS]
    device const half* W_zeros         [[buffer(11)]],  // [N, K/GS]
    uint tid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint N = params[0];
    uint K = params[1];
    float eps = as_type<float>(params[2]);

    if (tid >= N) return;

    uint num_groups = K / GS;
    uint sb_stride = num_groups * FRN_SB_BYTES_INT4;

    float dot_acc = 0.0f;
    float sq_acc = 0.0f;

    uint k_words = K / 8;

    for (uint w = lane; w < k_words; w += 32) {
        uint k_elem = w * 8;
        uint g = k_elem / GS;
        uint word_idx = (k_elem % GS) / 8;

        device const uchar *data_ptr = W + tid * sb_stride + g * FRN_SB_BYTES_INT4;
        float s = float(W_scales[tid * num_groups + g]);
        float z = float(W_zeros[tid * num_groups + g]);

        uint packed4 = ((device const uint*)(data_ptr))[word_idx];

        float w0 = (float(packed4 & 0xF) - z) * s;
        float w1 = (float((packed4 >> 4) & 0xF) - z) * s;
        float w2 = (float((packed4 >> 8) & 0xF) - z) * s;
        float w3 = (float((packed4 >> 12) & 0xF) - z) * s;
        float w4 = (float((packed4 >> 16) & 0xF) - z) * s;
        float w5 = (float((packed4 >> 20) & 0xF) - z) * s;
        float w6 = (float((packed4 >> 24) & 0xF) - z) * s;
        float w7 = (float((packed4 >> 28) & 0xF) - z) * s;

        float val0 = float(a[k_elem]) + float(b[k_elem]);
        float val1 = float(a[k_elem + 1]) + float(b[k_elem + 1]);
        float val2 = float(a[k_elem + 2]) + float(b[k_elem + 2]);
        float val3 = float(a[k_elem + 3]) + float(b[k_elem + 3]);
        float val4 = float(a[k_elem + 4]) + float(b[k_elem + 4]);
        float val5 = float(a[k_elem + 5]) + float(b[k_elem + 5]);
        float val6 = float(a[k_elem + 6]) + float(b[k_elem + 6]);
        float val7 = float(a[k_elem + 7]) + float(b[k_elem + 7]);

        sq_acc += val0*val0 + val1*val1 + val2*val2 + val3*val3
                + val4*val4 + val5*val5 + val6*val6 + val7*val7;

        float normed0 = val0 * float(norm_weight[k_elem]);
        float normed1 = val1 * float(norm_weight[k_elem + 1]);
        float normed2 = val2 * float(norm_weight[k_elem + 2]);
        float normed3 = val3 * float(norm_weight[k_elem + 3]);
        float normed4 = val4 * float(norm_weight[k_elem + 4]);
        float normed5 = val5 * float(norm_weight[k_elem + 5]);
        float normed6 = val6 * float(norm_weight[k_elem + 6]);
        float normed7 = val7 * float(norm_weight[k_elem + 7]);

        if (has_awq) {
            dot_acc += (normed0 / float(awq_scales[k_elem]))     * w0;
            dot_acc += (normed1 / float(awq_scales[k_elem + 1])) * w1;
            dot_acc += (normed2 / float(awq_scales[k_elem + 2])) * w2;
            dot_acc += (normed3 / float(awq_scales[k_elem + 3])) * w3;
            dot_acc += (normed4 / float(awq_scales[k_elem + 4])) * w4;
            dot_acc += (normed5 / float(awq_scales[k_elem + 5])) * w5;
            dot_acc += (normed6 / float(awq_scales[k_elem + 6])) * w6;
            dot_acc += (normed7 / float(awq_scales[k_elem + 7])) * w7;
        } else {
            dot_acc += normed0*w0 + normed1*w1 + normed2*w2 + normed3*w3
                     + normed4*w4 + normed5*w5 + normed6*w6 + normed7*w7;
        }
    }

    sq_acc = simd_sum(sq_acc);
    dot_acc = simd_sum(dot_acc);

    float rms_inv;
    if (lane == 0) {
        rms_inv = rsqrt(sq_acc / float(K) + eps);
        C[tid] = half(dot_acc * rms_inv);
    }

    if (tid == 0) {
        rms_inv = simd_broadcast_first(rms_inv);
        for (uint i = lane; i < K; i += 32) {
            float val = float(a[i]) + float(b[i]);
            residual_output[i] = half(val);
            normed_output[i] = half(val * rms_inv * float(norm_weight[i]));
        }
    }
}
