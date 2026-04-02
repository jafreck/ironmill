#include <metal_stdlib>
using namespace metal;

// ============================================================================
// INT4 Affine Dequantization
//
// Unpacks 2 INT4 values from each byte, applies per-group affine
// dequantization:  output = (quantized - zero_point) * scale
//
// Packing convention: low nibble first (bits [3:0] = element 2i,
//                     bits [7:4] = element 2i+1).
//
// Dispatch: ceil(total_elements / 2) threads, (256, 1, 1) threadgroup.
// ============================================================================

kernel void int4_dequantize(
    device const uint8_t* quantized_data [[buffer(0)]],
    device const half* scales            [[buffer(1)]],
    device const half* zeros             [[buffer(2)]],
    device half* output                  [[buffer(3)]],
    constant uint& group_size            [[buffer(4)]],
    constant uint& total_elements        [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    // Each thread handles one packed byte → 2 output elements.
    uint byte_idx = gid;
    uint elem_idx = byte_idx * 2;
    if (elem_idx >= total_elements) return;

    uint8_t packed = quantized_data[byte_idx];
    uint8_t lo = packed & 0x0F;
    uint8_t hi = (packed >> 4) & 0x0F;

    // Low nibble → element at elem_idx
    uint group_lo = elem_idx / group_size;
    half scale_lo = scales[group_lo];
    half zero_lo = zeros[group_lo];
    output[elem_idx] = (half(lo) - zero_lo) * scale_lo;

    // High nibble → element at elem_idx + 1
    if (elem_idx + 1 < total_elements) {
        uint group_hi = (elem_idx + 1) / group_size;
        half scale_hi = scales[group_hi];
        half zero_hi = zeros[group_hi];
        output[elem_idx + 1] = (half(hi) - zero_hi) * scale_hi;
    }
}
