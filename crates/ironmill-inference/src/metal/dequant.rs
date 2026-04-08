//! CPU dequantization from quantized formats to FP16 byte arrays.
//!
//! Used by [`super::weights::MetalWeights::load`] to convert LUT-palettized
//! and affine-quantized tensors into dense FP16 buffers before uploading
//! to Metal.
//!
//! Generic routines (affine, dual-scale, parameter conversion) are shared
//! via [`ironmill_core::dequant`].  This module re-exports them and adds
//! inference-specific variants (QuIP#, LUT-to-dense).

use anyhow::bail;
use half::f16;
use mil_rs::ir::ScalarType;

use crate::dequant::{read_typed_f32, unpack_indices};

// Re-export shared dequant functions from core so callers in this crate
// can continue to use `super::dequant::dequant_affine` etc.
pub use ironmill_core::dequant::{convert_params_to_f16, dequant_affine, dequant_dual_scale};

// ── E8 codebook constants ────────────────────────────────────────

/// Number of entries in the E8 lattice codebook.
const E8_CODEBOOK_SIZE: usize = 256;

/// Dimension of each E8 codebook entry (elements per group).
const E8_GROUP_DIM: usize = 8;

// ── Public API ───────────────────────────────────────────────────

/// Dequantize a QuIP#-encoded tensor (E8 codebook) to FP16 bytes.
///
/// Each u8 index selects one of 256 entries from the E8 codebook (8 elements
/// each). After reconstruction, applies inverse Hadamard rotation using
/// `quip_sharp_seed` and rescales by per-row norms.
pub fn dequant_quip_sharp(
    indices: &[u8],
    lut: &[u8],
    lut_dtype: ScalarType,
    original_shape: &[usize],
    row_norms: &[u8],
    norms_dtype: ScalarType,
    seed: u64,
) -> anyhow::Result<Vec<u8>> {
    let total_elements: usize = original_shape.iter().product();
    let cols = *original_shape
        .last()
        .ok_or_else(|| anyhow::anyhow!("shape must be non-empty"))?;
    let rows = if cols == 0 { 0 } else { total_elements / cols };

    // QuIP# operates on padded (power-of-2) columns.
    let padded_cols = cols.next_power_of_two();
    let groups_per_row = padded_cols / E8_GROUP_DIM;
    let entry_size = lut_dtype.byte_size();
    let norm_size = norms_dtype.byte_size();

    // Read the E8 codebook into f32.
    let mut codebook = vec![[0.0f32; E8_GROUP_DIM]; E8_CODEBOOK_SIZE];
    for (i, entry) in codebook.iter_mut().enumerate() {
        for (j, val) in entry.iter_mut().enumerate() {
            let offset = (i * E8_GROUP_DIM + j) * entry_size;
            *val = read_typed_f32(lut, offset, lut_dtype)?;
        }
    }

    // Reconstruct the padded weight matrix from codebook indices.
    let work_total = rows * padded_cols;
    let mut f32_values = vec![0.0f32; work_total];
    for r in 0..rows {
        for g in 0..groups_per_row {
            let idx_pos = r * groups_per_row + g;
            if idx_pos >= indices.len() {
                break;
            }
            let cb_idx = indices[idx_pos] as usize;
            let entry = &codebook[cb_idx.min(E8_CODEBOOK_SIZE - 1)];
            let base = r * padded_cols + g * E8_GROUP_DIM;
            for (k, &v) in entry.iter().enumerate() {
                f32_values[base + k] = v;
            }
        }
    }

    // Apply inverse Hadamard rotation.
    mil_rs::ir::passes::rotation::unrotate_rows_hadamard(&mut f32_values, rows, padded_cols, seed);

    // Trim padding and apply per-row norms, convert to FP16.
    let mut output = Vec::with_capacity(total_elements * 2);
    for row in 0..rows {
        let norm = read_typed_f32(row_norms, row * norm_size, norms_dtype)?;
        for col in 0..cols {
            let value = f32_values[row * padded_cols + col];
            let result = f16::from_f32(value * norm);
            output.extend_from_slice(&result.to_le_bytes());
        }
    }

    Ok(output)
}

/// Dequantize a LUT-encoded tensor to FP16 bytes.
///
/// Unpacks `n_bits`-wide indices, looks up reconstruction levels in `lut`,
/// applies inverse Hadamard rotation (when `polar_quant_seed` is present),
/// multiplies by per-row norms, and returns the result as little-endian
/// FP16 bytes.
#[allow(clippy::too_many_arguments)]
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
    let cols = *original_shape
        .last()
        .ok_or_else(|| anyhow::anyhow!("shape must be non-empty"))?;
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
    for (flat_idx, &lut_idx) in unpacked.iter().enumerate().take(work_total) {
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

    // ── LUT dequantization (inference-specific) ─────────────────

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

    // ── INT4 shader source (inference-specific) ────────────────

    #[test]
    fn int4_dequant_shader_source_is_valid_metal() {
        // Verify the shader source is included and contains the expected kernel.
        let src = include_str!("shaders/quantized/int4_dequant.metal");
        assert!(src.contains("kernel void int4_dequantize("));
        assert!(src.contains("buffer(0)"));
        assert!(src.contains("buffer(5)"));
        assert!(src.contains("thread_position_in_grid"));
        // Verify dequantization formula
        assert!(src.contains("(float(lo) - float(zero_lo)) * float(scale_lo)"));
        assert!(src.contains("(float(hi) - float(zero_hi)) * float(scale_hi)"));
    }
}
