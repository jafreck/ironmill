#pragma once
#include <metal_stdlib>
using namespace metal;

// Two-level simdgroup sum reduction.
// Requires `scratch` to point to threadgroup float[32] (supports up to 1024 threads).
// All threads in the threadgroup must call this function (contains barriers).
inline float threadgroup_reduce_sum(float val, uint tid, uint tg_size,
                                    threadgroup float* scratch) {
    float simd_total = simd_sum(val);
    uint sg_idx = tid / 32;
    if (tid % 32 == 0) scratch[sg_idx] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        uint num_sg = (tg_size + 31) / 32;
        float total = 0.0f;
        for (uint i = 0; i < num_sg; i++) total += scratch[i];
        scratch[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return scratch[0];
}
