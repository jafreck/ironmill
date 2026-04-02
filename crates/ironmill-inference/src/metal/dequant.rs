//! CPU dequantization from quantized formats to FP16 byte arrays.
//!
//! Used by [`super::weights::MetalWeights::load`] to convert LUT-palettized
//! and affine-quantized tensors into dense FP16 buffers before uploading
//! to Metal.

use anyhow::bail;
use half::f16;
use mil_rs::ir::ScalarType;

use crate::dequant::{read_typed_f32, unpack_indices};

// ── Public API ───────────────────────────────────────────────────

/// Dequantize a LUT-encoded tensor to FP16 bytes.
///
/// Unpacks `n_bits`-wide indices, looks up reconstruction levels in `lut`,
/// applies inverse Hadamard rotation (when `polar_quant_seed` is present),
/// multiplies by per-row norms, and returns the result as little-endian
/// FP16 bytes.
pub fn dequant_lut_to_dense(
    indices: &[u8],
    lut: &[u8],
    lut_dtype: ScalarType,
    original_shape: &[usize],
    n_bits: u8,
    row_norms: &[u8],
    norms_dtype: ScalarType,
    polar_quant_seed: Option<u64>,
) -> anyhow::Result<Vec<u8>> {
    let total_elements: usize = original_shape.iter().product();
    let cols = *original_shape.last().expect("shape must be non-empty");
    let rows = if cols == 0 { 0 } else { total_elements / cols };

    // Only use padded dimensions when Hadamard rotation was applied
    // (rotation requires power-of-2 cols). Per-row quantization without
    // rotation packs indices at the original dimensions.
    let work_cols = if polar_quant_seed.is_some() {
        cols.next_power_of_two()
    } else {
        cols
    };
    let work_total = rows * work_cols;

    let entry_size = lut_dtype.byte_size();
    let norm_size = norms_dtype.byte_size();

    let unpacked = unpack_indices(indices, n_bits, work_total);
    let max_lut_idx = lut.len() / entry_size;

    // Reconstruct f32 values from LUT.
    let mut f32_values: Vec<f32> = Vec::with_capacity(work_total);
    for flat_idx in 0..work_total {
        let lut_idx = unpacked[flat_idx];
        if lut_idx >= max_lut_idx {
            bail!(
                "LUT index {lut_idx} out of bounds (max {max_lut_idx}) at element {flat_idx}/{work_total}"
            );
        }
        f32_values.push(read_typed_f32(lut, lut_idx * entry_size, lut_dtype)?);
    }

    // Apply inverse Hadamard rotation if a seed is present.
    if let Some(seed) = polar_quant_seed {
        mil_rs::ir::passes::rotation::unrotate_rows_hadamard(
            &mut f32_values,
            rows,
            work_cols,
            seed,
        );
    }

    // Trim padding and apply per-row norms, convert to FP16.
    let mut output = Vec::with_capacity(total_elements * 2);
    for row in 0..rows {
        let norm = read_typed_f32(row_norms, row * norm_size, norms_dtype)?;
        for col in 0..cols {
            let value = f32_values[row * work_cols + col];
            let result = f16::from_f32(value * norm);
            output.extend_from_slice(&result.to_le_bytes());
        }
    }

    Ok(output)
}

/// Dequantize an affine-quantized tensor to FP16 bytes.
///
/// Applies `(quantized - zero_point) * scale` element-wise. Each byte
/// of `quantized_data` is treated as a signed `i8` value.
pub fn dequant_affine(
    quantized_data: &[u8],
    scale: &[u8],
    zero_point: &[u8],
    scale_dtype: ScalarType,
    zero_point_dtype: ScalarType,
    axis: Option<usize>,
    shape: &[usize],
) -> anyhow::Result<Vec<u8>> {
    let total_elements: usize = shape.iter().product();
    let scale_elem_size = scale_dtype.byte_size();
    let zp_elem_size = zero_point_dtype.byte_size();

    let mut output = Vec::with_capacity(total_elements * 2);

    match axis {
        Some(ax) => {
            // Per-axis: scale and zero_point have one entry per slice along `ax`.
            let stride: usize = shape[ax + 1..].iter().product();
            let axis_size = shape[ax];

            for i in 0..total_elements {
                let axis_idx = (i / stride) % axis_size;
                let s = read_typed_f32(scale, axis_idx * scale_elem_size, scale_dtype)?;
                let z = read_typed_f32(zero_point, axis_idx * zp_elem_size, zero_point_dtype)?;
                let q = quantized_data[i] as i8 as f32;
                let result = f16::from_f32((q - z) * s);
                output.extend_from_slice(&result.to_le_bytes());
            }
        }
        None => {
            // Per-tensor: single scale and zero_point.
            let s = read_typed_f32(scale, 0, scale_dtype)?;
            let z = read_typed_f32(zero_point, 0, zero_point_dtype)?;

            for i in 0..total_elements {
                let q = quantized_data[i] as i8 as f32;
                let result = f16::from_f32((q - z) * s);
                output.extend_from_slice(&result.to_le_bytes());
            }
        }
    }

    Ok(output)
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: convert an `f32` to little-endian FP16 bytes.
    fn f16_bytes(v: f32) -> [u8; 2] {
        f16::from_f32(v).to_le_bytes()
    }

    /// Helper: read FP16 value back from a byte slice at a given element index.
    fn read_f16(data: &[u8], idx: usize) -> f32 {
        let off = idx * 2;
        f16::from_le_bytes([data[off], data[off + 1]]).to_f32()
    }

    // ── Index unpacking ──────────────────────────────────────────

    #[test]
    fn unpack_4bit_indices() {
        // Two values per byte: lo nibble first, hi nibble second.
        // Byte 0xA3 → indices 3 (0x03), 10 (0x0A)
        // Byte 0x51 → indices 1 (0x01), 5  (0x05)
        let packed = vec![0xA3, 0x51];
        let result = unpack_indices(&packed, 4, 4);
        assert_eq!(result, vec![10, 3, 5, 1]);
    }

    #[test]
    fn unpack_2bit_indices() {
        // Four values per byte, shifts 0/2/4/6.
        // Byte 0b_11_10_01_00 = 0xE4 → indices 0, 1, 2, 3
        let packed = vec![0xE4];
        let result = unpack_indices(&packed, 2, 4);
        assert_eq!(result, vec![3, 2, 1, 0]);
    }

    #[test]
    fn unpack_8bit_indices() {
        let packed = vec![7, 0, 15, 3];
        let result = unpack_indices(&packed, 8, 4);
        assert_eq!(result, vec![7, 0, 15, 3]);
    }

    #[test]
    fn unpack_truncates_to_total() {
        // Ask for fewer elements than the packed data contains.
        let packed = vec![0xA3, 0x51];
        let result = unpack_indices(&packed, 4, 3);
        assert_eq!(result, vec![10, 3, 5]);
    }

    // ── LUT dequantization ───────────────────────────────────────

    #[test]
    fn dequant_lut_4bit_fp16_2x4() {
        // 16-entry FP16 LUT: entry i = i as f16.
        let mut lut = Vec::new();
        for i in 0..16u32 {
            lut.extend_from_slice(&f16_bytes(i as f32));
        }

        // 2×4 matrix of 4-bit indices:
        //   row 0: [0, 1, 2, 3]
        //   row 1: [4, 5, 6, 7]
        let raw_indices: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let indices = mil_rs::ir::passes::polar_quantize::pack_indices(&raw_indices, 4);
        let shape = vec![2, 4];

        // Row norms (FP16): row 0 = 2.0, row 1 = 0.5.
        let mut norms = Vec::new();
        norms.extend_from_slice(&f16_bytes(2.0));
        norms.extend_from_slice(&f16_bytes(0.5));

        let output = dequant_lut_to_dense(
            &indices,
            &lut,
            ScalarType::Float16,
            &shape,
            4,
            &norms,
            ScalarType::Float16,
            None,
        )
        .unwrap();

        assert_eq!(output.len(), 2 * 4 * 2); // 8 elements × 2 bytes

        // Row 0: lut[i] * 2.0 → [0, 2, 4, 6]
        assert!((read_f16(&output, 0) - 0.0).abs() < 1e-3);
        assert!((read_f16(&output, 1) - 2.0).abs() < 1e-3);
        assert!((read_f16(&output, 2) - 4.0).abs() < 1e-3);
        assert!((read_f16(&output, 3) - 6.0).abs() < 1e-3);

        // Row 1: lut[i] * 0.5 → [2, 2.5, 3, 3.5]
        assert!((read_f16(&output, 4) - 2.0).abs() < 1e-3);
        assert!((read_f16(&output, 5) - 2.5).abs() < 1e-2);
        assert!((read_f16(&output, 6) - 3.0).abs() < 1e-3);
        assert!((read_f16(&output, 7) - 3.5).abs() < 1e-2);
    }

    #[test]
    fn dequant_lut_fp32_lut_and_norms() {
        // LUT stored as FP32, norms as FP32. Verify cross-dtype handling.
        let mut lut = Vec::new();
        for i in 0..4u32 {
            lut.extend_from_slice(&(i as f32).to_le_bytes());
        }

        // 2-bit indices for a 1×4 matrix: [0, 1, 2, 3]
        let raw_indices: Vec<usize> = vec![0, 1, 2, 3];
        let indices = mil_rs::ir::passes::polar_quantize::pack_indices(&raw_indices, 2);
        let shape = vec![1, 4];

        let mut norms = Vec::new();
        norms.extend_from_slice(&3.0f32.to_le_bytes());

        let output = dequant_lut_to_dense(
            &indices,
            &lut,
            ScalarType::Float32,
            &shape,
            2,
            &norms,
            ScalarType::Float32,
            None,
        )
        .unwrap();

        assert_eq!(output.len(), 4 * 2);

        // Expected: lut[i] * 3.0 → [0, 3, 6, 9]
        assert!((read_f16(&output, 0) - 0.0).abs() < 1e-3);
        assert!((read_f16(&output, 1) - 3.0).abs() < 1e-3);
        assert!((read_f16(&output, 2) - 6.0).abs() < 1e-2);
        assert!((read_f16(&output, 3) - 9.0).abs() < 1e-2);
    }

    // ── Affine dequantization ────────────────────────────────────

    #[test]
    fn dequant_affine_per_tensor_int8() {
        // 4 INT8 values: [-2, -1, 0, 1] stored as i8 → u8.
        let quantized: Vec<u8> = vec![(-2i8) as u8, (-1i8) as u8, 0u8, 1u8];
        let shape = vec![2, 2];

        // Per-tensor: scale = 0.5, zero_point = 0.0 (FP32).
        let scale = 0.5f32.to_le_bytes().to_vec();
        let zero_point = 0.0f32.to_le_bytes().to_vec();

        let output = dequant_affine(
            &quantized,
            &scale,
            &zero_point,
            ScalarType::Float32,
            ScalarType::Float32,
            None,
            &shape,
        )
        .unwrap();

        assert_eq!(output.len(), 4 * 2);

        // Expected: (q - 0) * 0.5 → [-1.0, -0.5, 0.0, 0.5]
        assert!((read_f16(&output, 0) - (-1.0)).abs() < 1e-3);
        assert!((read_f16(&output, 1) - (-0.5)).abs() < 1e-3);
        assert!((read_f16(&output, 2) - 0.0).abs() < 1e-3);
        assert!((read_f16(&output, 3) - 0.5).abs() < 1e-3);
    }

    #[test]
    fn dequant_affine_per_axis() {
        // 2×3 matrix, axis=0 → scale/zero_point have 2 entries.
        // Row 0 values: [10, 20, 30]  with scale=0.1, zp=10
        // Row 1 values: [5, 10, 15]   with scale=0.5, zp=5
        let quantized: Vec<u8> = vec![10, 20, 30, 5, 10, 15];
        let shape = vec![2, 3];

        let mut scale = Vec::new();
        scale.extend_from_slice(&f16_bytes(0.1));
        scale.extend_from_slice(&f16_bytes(0.5));

        let mut zero_point = Vec::new();
        zero_point.extend_from_slice(&f16_bytes(10.0));
        zero_point.extend_from_slice(&f16_bytes(5.0));

        let output = dequant_affine(
            &quantized,
            &scale,
            &zero_point,
            ScalarType::Float16,
            ScalarType::Float16,
            Some(0),
            &shape,
        )
        .unwrap();

        assert_eq!(output.len(), 6 * 2);

        // Row 0: (q - 10) * 0.1 → [0.0, 1.0, 2.0]
        assert!((read_f16(&output, 0) - 0.0).abs() < 1e-2);
        assert!((read_f16(&output, 1) - 1.0).abs() < 1e-2);
        assert!((read_f16(&output, 2) - 2.0).abs() < 1e-2);

        // Row 1: (q - 5) * 0.5 → [0.0, 2.5, 5.0]
        assert!((read_f16(&output, 3) - 0.0).abs() < 1e-2);
        assert!((read_f16(&output, 4) - 2.5).abs() < 1e-1);
        assert!((read_f16(&output, 5) - 5.0).abs() < 1e-1);
    }

    #[test]
    fn dequant_affine_with_nonzero_zero_point() {
        // Verify zero_point offset works correctly.
        let quantized: Vec<u8> = vec![128u8]; // 128 as u8, but interpreted as i8 = -128
        let shape = vec![1];

        let scale = 1.0f32.to_le_bytes().to_vec();
        let zero_point = 0.0f32.to_le_bytes().to_vec();

        let output = dequant_affine(
            &quantized,
            &scale,
            &zero_point,
            ScalarType::Float32,
            ScalarType::Float32,
            None,
            &shape,
        )
        .unwrap();

        // i8 interpretation: 128u8 = -128i8
        // (-128 - 0) * 1.0 = -128.0
        assert!((read_f16(&output, 0) - (-128.0)).abs() < 1.0);
    }
}
