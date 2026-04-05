//! Shared dequantization helpers used by the Metal backend.
//!
//! This module contains low-level routines for reading typed scalars
//! and unpacking bit-packed indices—common operations needed by any
//! backend that dequantizes on the CPU.

use anyhow::bail;
use half::f16;
use mil_rs::ir::ScalarType;

/// Read a single scalar from `data` at the given byte offset, returning `f32`.
pub(crate) fn read_typed_f32(
    data: &[u8],
    byte_offset: usize,
    dtype: ScalarType,
) -> anyhow::Result<f32> {
    match dtype {
        ScalarType::Float16 => {
            let bytes = [data[byte_offset], data[byte_offset + 1]];
            Ok(f16::from_le_bytes(bytes).to_f32())
        }
        ScalarType::Float32 => {
            let bytes = [
                data[byte_offset],
                data[byte_offset + 1],
                data[byte_offset + 2],
                data[byte_offset + 3],
            ];
            Ok(f32::from_le_bytes(bytes))
        }
        other => bail!("unsupported dtype for dequantization: {other:?}"),
    }
}

/// Unpack `n_bits`-wide indices from a byte array (MSB-first packing).
///
/// For 4-bit packing each byte holds two values: `lo | (hi << 4)`.
/// For 2-bit packing each byte holds four values, etc.
pub(crate) fn unpack_indices(packed: &[u8], n_bits: u8, total_elements: usize) -> Vec<usize> {
    if n_bits == 8 {
        return packed
            .iter()
            .take(total_elements)
            .map(|&b| b as usize)
            .collect();
    }

    let mask = ((1u16 << n_bits) - 1) as u16;
    let mut indices = Vec::with_capacity(total_elements);
    let mut bit_offset = 0usize;

    for _ in 0..total_elements {
        let byte_pos = bit_offset / 8;
        let bit_in_byte = bit_offset % 8;

        // Read up to 2 bytes to handle values that span byte boundaries.
        let hi = packed[byte_pos] as u16;
        let lo = if byte_pos + 1 < packed.len() {
            packed[byte_pos + 1] as u16
        } else {
            0
        };
        let word = (hi << 8) | lo;
        let shift = 16 - n_bits as usize - bit_in_byte;
        let idx = ((word >> shift) & mask) as usize;
        indices.push(idx);

        bit_offset += n_bits as usize;
    }

    indices
}

// ── Affine dequant helpers ───────────────────────────────────────

/// Dequantize an affine-quantized tensor to FP16 bytes (unsigned
/// interpretation).
///
/// Applies `(quantized - zero_point) * scale` element-wise, where each
/// byte of `data` is treated as an unsigned `u8` value.
pub(crate) fn dequant_affine_to_fp16(
    data: &[u8],
    scale: &[u8],
    zero_point: &[u8],
    scale_dtype: ScalarType,
    zero_point_dtype: ScalarType,
    _axis: Option<usize>,
    shape: &[usize],
) -> anyhow::Result<Vec<u8>> {
    let num_elements: usize = shape.iter().product();
    let mut output = Vec::with_capacity(num_elements * 2);

    let scale_elem_size = scale_dtype.byte_size();
    let zp_elem_size = zero_point_dtype.byte_size();

    for i in 0..num_elements {
        let q_val = data[i] as f32;
        let s_idx = if scale.len() / scale_elem_size > 1 {
            // Per-channel scale: index by row.
            let cols = if shape.len() > 1 { shape[1] } else { 1 };
            i / cols
        } else {
            0
        };
        let s = read_typed_f32(scale, s_idx * scale_elem_size, scale_dtype)?;
        let zp = if zero_point.is_empty() {
            0.0
        } else {
            let zp_idx = s_idx.min(zero_point.len() / zp_elem_size - 1);
            read_typed_f32(zero_point, zp_idx * zp_elem_size, zero_point_dtype)?
        };
        let val = (q_val - zp) * s;
        let h = f16::from_f32(val);
        output.extend_from_slice(&h.to_le_bytes());
    }

    Ok(output)
}

/// Dequantize an affine-quantized tensor using GPTQ `g_idx` group mapping.
///
/// Like [`dequant_affine_to_fp16`] but uses `g_idx[col]` instead of
/// `col / group_size` to determine which group's scale/zero_point applies
/// to each column. This is needed for GPTQ models with act-order
/// (`desc_act=True`), where columns were quantized in importance order
/// and `g_idx` records which group each column belongs to.
pub(crate) fn dequant_affine_with_g_idx(
    data: &[u8],
    scale: &[u8],
    zero_point: &[u8],
    scale_dtype: ScalarType,
    zero_point_dtype: ScalarType,
    shape: &[usize],
    _bit_width: u8,
    g_idx: &[u32],
) -> anyhow::Result<Vec<u8>> {
    let num_elements: usize = shape.iter().product();
    let cols = if shape.len() > 1 { shape[1] } else { 1 };
    let rows = num_elements / cols;
    let mut output = Vec::with_capacity(num_elements * 2);

    let scale_elem_size = scale_dtype.byte_size();
    let zp_elem_size = zero_point_dtype.byte_size();
    let n_scale_entries = scale.len() / scale_elem_size;
    let n_groups_per_row = if rows > 0 {
        n_scale_entries / rows
    } else {
        n_scale_entries
    };

    for row in 0..rows {
        for col in 0..cols {
            let i = row * cols + col;
            let q_val = data[i] as f32;

            // Use g_idx to determine the group for this column.
            let group = if col < g_idx.len() {
                g_idx[col] as usize
            } else {
                0
            };
            let s_idx = row * n_groups_per_row + group;

            let s = read_typed_f32(scale, s_idx * scale_elem_size, scale_dtype)?;
            let zp = if zero_point.is_empty() {
                0.0
            } else {
                let zp_idx = s_idx.min(zero_point.len() / zp_elem_size - 1);
                read_typed_f32(zero_point, zp_idx * zp_elem_size, zero_point_dtype)?
            };
            let val = (q_val - zp) * s;
            let h = f16::from_f32(val);
            output.extend_from_slice(&h.to_le_bytes());
        }
    }

    Ok(output)
}
///
/// Unpacks `n_bits`-wide indices, looks up values in `lut`, multiplies
/// by per-row norms, and returns FP16 bytes. Does **not** apply inverse
/// Hadamard rotation (ignores `polar_quant_seed`).
pub(crate) fn dequant_lut_to_fp16(
    indices: &[u8],
    lut: &[u8],
    lut_dtype: ScalarType,
    original_shape: &[usize],
    n_bits: u8,
    row_norms: &[u8],
    norms_dtype: ScalarType,
    _polar_quant_seed: Option<u64>,
) -> anyhow::Result<Vec<u8>> {
    let num_elements: usize = original_shape.iter().product();
    let cols = if original_shape.len() > 1 {
        original_shape[1]
    } else {
        num_elements
    };
    let rows = num_elements / cols;

    let palette_size = 1usize << n_bits;
    let lut_elem_size = lut_dtype.byte_size();

    let unpacked = unpack_indices(indices, n_bits, num_elements);

    let mut output = Vec::with_capacity(num_elements * 2);

    for row in 0..rows {
        let norm = read_typed_f32(row_norms, row * norms_dtype.byte_size(), norms_dtype)?;
        let lut_row_offset = row * palette_size * lut_elem_size;

        for col in 0..cols {
            let idx = unpacked[row * cols + col];
            let lut_val = read_typed_f32(lut, lut_row_offset + idx * lut_elem_size, lut_dtype)?;
            let val = lut_val * norm;
            let h = f16::from_f32(val);
            output.extend_from_slice(&h.to_le_bytes());
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequant_affine_with_g_idx_uses_correct_groups() {
        // 1×4 weight matrix with 2 groups.
        // g_idx remaps columns: col 0→group 1, col 1→group 0,
        // col 2→group 0, col 3→group 1.
        let shape = vec![1, 4];

        // Quantized data: all ones.
        let data: Vec<u8> = vec![1, 1, 1, 1];

        // 2 groups per row → 2 scale entries, 2 zp entries.
        // Group 0: scale=2.0, zp=0.0 → dequant(1) = (1 - 0) * 2 = 2.0
        // Group 1: scale=3.0, zp=0.0 → dequant(1) = (1 - 0) * 3 = 3.0
        let scale: Vec<u8> = [2.0f32, 3.0f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let zero_point: Vec<u8> = [0.0f32, 0.0f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let g_idx: Vec<u32> = vec![1, 0, 0, 1];

        let result = dequant_affine_with_g_idx(
            &data,
            &scale,
            &zero_point,
            ScalarType::Float32,
            ScalarType::Float32,
            &shape,
            8,
            &g_idx,
        )
        .unwrap();

        assert_eq!(result.len(), 4 * 2);

        let vals: Vec<f32> = result
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();

        // col 0 → group 1 → scale 3.0 → 3.0
        assert!((vals[0] - 3.0).abs() < 0.01, "col0={}", vals[0]);
        // col 1 → group 0 → scale 2.0 → 2.0
        assert!((vals[1] - 2.0).abs() < 0.01, "col1={}", vals[1]);
        // col 2 → group 0 → scale 2.0 → 2.0
        assert!((vals[2] - 2.0).abs() < 0.01, "col2={}", vals[2]);
        // col 3 → group 1 → scale 3.0 → 3.0
        assert!((vals[3] - 3.0).abs() < 0.01, "col3={}", vals[3]);
    }
}
