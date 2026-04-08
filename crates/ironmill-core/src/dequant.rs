//! Shared CPU dequantization routines.
//!
//! Generic, backend-agnostic helpers for converting quantized weight tensors
//! into dense FP16 byte arrays.  Used by both the compile pipeline (for GPU
//! bundle fallback dequant) and the inference runtime (for CPU-side
//! dequantization before Metal upload).

use anyhow::bail;
use half::f16;
use mil_rs::ir::ScalarType;

// ── Scalar helpers ──────────────────────────────────────────────────────

/// Read a single scalar from a byte buffer at the given offset, returning `f32`.
///
/// Supports `Float16` and `Float32` dtypes.
pub fn read_typed_f32(data: &[u8], byte_offset: usize, dtype: ScalarType) -> anyhow::Result<f32> {
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
pub fn unpack_indices(packed: &[u8], n_bits: u8, total_elements: usize) -> Vec<usize> {
    if n_bits == 8 {
        return packed
            .iter()
            .take(total_elements)
            .map(|&b| b as usize)
            .collect();
    }

    let mask = (1u16 << n_bits) - 1;
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

// ── Affine dequantization ───────────────────────────────────────────────

/// Dequantize an affine-quantized tensor to FP16 bytes.
///
/// Applies `(quantized - zero_point) * scale` element-wise.
/// For INT8 (`bit_width=8`), each byte is one element.
/// For INT4 (`bit_width=4`), elements are packed 2 per byte (low nibble first).
///
/// When `group_size` is `Some(gs)`, scales/zeros are per-group along the last
/// axis: there are `ceil(K / gs)` groups per row.
pub fn dequant_affine(
    quantized_data: &[u8],
    scale: &[u8],
    zero_point: &[u8],
    scale_dtype: ScalarType,
    zero_point_dtype: ScalarType,
    axis: Option<usize>,
    shape: &[usize],
    bit_width: u8,
    group_size: Option<usize>,
) -> anyhow::Result<Vec<u8>> {
    let total_elements: usize = shape.iter().product();
    let scale_elem_size = scale_dtype.byte_size();
    let zp_elem_size = zero_point_dtype.byte_size();

    // Helper: extract the i-th quantized value, handling INT4 packing.
    // Both INT4 and INT8 values are unsigned (0..qmax range).
    let read_q = |i: usize| -> f32 {
        if bit_width == 4 {
            let byte = quantized_data[i / 2];
            if i % 2 == 0 {
                (byte & 0x0F) as f32
            } else {
                ((byte >> 4) & 0x0F) as f32
            }
        } else {
            quantized_data[i] as f32
        }
    };

    let mut output = Vec::with_capacity(total_elements * 2);

    if let Some(gs) = group_size {
        // Per-group: scales/zeros have one entry per group of `gs` elements
        // along the last axis. Layout: [N, num_groups] where num_groups = ceil(K/gs).
        if shape.is_empty() {
            bail!("dequant_affine: shape must be non-empty");
        }
        let k = *shape.last().unwrap();
        let n: usize = shape[..shape.len().saturating_sub(1)]
            .iter()
            .product::<usize>()
            .max(1);
        let num_groups = k.div_ceil(gs);

        for row in 0..n {
            for col in 0..k {
                let group_idx = col / gs;
                let param_idx = row * num_groups + group_idx;
                let s = read_typed_f32(scale, param_idx * scale_elem_size, scale_dtype)?;
                let z = read_typed_f32(zero_point, param_idx * zp_elem_size, zero_point_dtype)?;
                let elem_idx = row * k + col;
                let q = read_q(elem_idx);
                let result = f16::from_f32((q - z) * s);
                output.extend_from_slice(&result.to_le_bytes());
            }
        }
    } else {
        match axis {
            Some(ax) => {
                if ax >= shape.len() {
                    bail!(
                        "dequant_affine: axis {} is out of bounds for shape with {} dimensions",
                        ax,
                        shape.len()
                    );
                }
                // Per-axis: scale and zero_point have one entry per slice along `ax`.
                let stride: usize = shape[ax + 1..].iter().product();
                let axis_size = shape[ax];

                for i in 0..total_elements {
                    let axis_idx = (i / stride) % axis_size;
                    let s = read_typed_f32(scale, axis_idx * scale_elem_size, scale_dtype)?;
                    let z = read_typed_f32(zero_point, axis_idx * zp_elem_size, zero_point_dtype)?;
                    let q = read_q(i);
                    let result = f16::from_f32((q - z) * s);
                    output.extend_from_slice(&result.to_le_bytes());
                }
            }
            None => {
                // Per-tensor: single scale and zero_point.
                let s = read_typed_f32(scale, 0, scale_dtype)?;
                let z = read_typed_f32(zero_point, 0, zero_point_dtype)?;

                for i in 0..total_elements {
                    let q = read_q(i);
                    let result = f16::from_f32((q - z) * s);
                    output.extend_from_slice(&result.to_le_bytes());
                }
            }
        }
    }

    Ok(output)
}

/// Dequantize an affine-quantized tensor using GPTQ `g_idx` group mapping.
///
/// Like [`dequant_affine`] but uses `g_idx[col]` instead of `col / group_size`
/// to determine which group's scale/zero_point applies to each column.
/// This is needed for GPTQ models with act-order (`desc_act=True`), where
/// columns were quantized in importance order and `g_idx` records which group
/// each column belongs to.
pub fn dequant_affine_with_g_idx(
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

// ── D2Quant dual-scale dequantization ───────────────────────────────────

/// Dequantize a D2Quant dual-scale tensor to FP16 bytes.
///
/// Each group has separate scale/zero for normal and outlier partitions,
/// selected by a per-weight bit mask. The formula for each weight `i` in
/// group `g` is:
/// - If `outlier_mask[i]` is 1: `(quantized[i] - outlier_zero[g]) * outlier_scale[g]`
/// - If `outlier_mask[i]` is 0: `(quantized[i] - normal_zero[g]) * normal_scale[g]`
pub fn dequant_dual_scale(
    quantized_data: &[u8],
    normal_scale: &[u8],
    normal_zero: &[u8],
    outlier_scale: &[u8],
    outlier_zero: &[u8],
    outlier_mask: &[u8],
    original_shape: &[usize],
    bit_width: u8,
    group_size: usize,
) -> anyhow::Result<Vec<u8>> {
    use mil_rs::ir::passes::d2quant::dual_scale::{unpack_2bit, unpack_3bit};

    let total_elements: usize = original_shape.iter().product();

    // Unpack quantized values based on bit width.
    let unpacked = match bit_width {
        2 => unpack_2bit(quantized_data, total_elements),
        3 => unpack_3bit(quantized_data, total_elements),
        _ => bail!("unsupported D2Quant bit_width: {bit_width} (expected 2 or 3)"),
    };

    let mut output = Vec::with_capacity(total_elements * 2);

    for (i, &q) in unpacked.iter().enumerate() {
        let group = i / group_size;
        let is_outlier = (outlier_mask[i / 8] >> (i % 8)) & 1 == 1;

        let (scale_buf, zero_buf) = if is_outlier {
            (outlier_scale, outlier_zero)
        } else {
            (normal_scale, normal_zero)
        };

        let s = read_typed_f32(scale_buf, group * 4, ScalarType::Float32)?;
        let z = read_typed_f32(zero_buf, group * 4, ScalarType::Float32)?;

        let val = (q as f32 - z) * s;
        let h = f16::from_f32(val);
        output.extend_from_slice(&h.to_le_bytes());
    }

    Ok(output)
}

// ── Parameter conversion ────────────────────────────────────────────────

/// Convert per-group quantization parameters (scale or zero_point) from
/// their native dtype to FP16 bytes suitable for upload to Metal.
///
/// For per-group INT4 quantization, there is one scale/zero per group.
/// The number of groups is `ceil(elements_along_axis / group_size)` for
/// per-axis quantization, or `ceil(total_elements / group_size)` for
/// per-tensor.
pub fn convert_params_to_f16(
    params: &[u8],
    dtype: ScalarType,
    _axis: Option<usize>,
    _shape: &[usize],
    _group_size: usize,
) -> anyhow::Result<Vec<u8>> {
    let elem_size = dtype.byte_size();
    let num_params = params.len() / elem_size;

    let mut output = Vec::with_capacity(num_params * 2);
    for i in 0..num_params {
        let val = read_typed_f32(params, i * elem_size, dtype)?;
        let fp16 = f16::from_f32(val);
        output.extend_from_slice(&fp16.to_le_bytes());
    }

    Ok(output)
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn f16_bytes(v: f32) -> [u8; 2] {
        f16::from_f32(v).to_le_bytes()
    }

    fn read_f16(data: &[u8], idx: usize) -> f32 {
        let off = idx * 2;
        f16::from_le_bytes([data[off], data[off + 1]]).to_f32()
    }

    // ── read_typed_f32 ──────────────────────────────────────────

    #[test]
    fn read_typed_f32_float16() {
        let bytes = f16_bytes(3.5);
        let val = read_typed_f32(&bytes, 0, ScalarType::Float16).unwrap();
        assert!((val - 3.5).abs() < 1e-3);
    }

    #[test]
    fn read_typed_f32_float32() {
        let bytes = 42.0f32.to_le_bytes();
        let val = read_typed_f32(&bytes, 0, ScalarType::Float32).unwrap();
        assert!((val - 42.0).abs() < 1e-6);
    }

    // ── unpack_indices ──────────────────────────────────────────

    #[test]
    fn unpack_4bit_indices() {
        let packed = vec![0xA3, 0x51];
        let result = unpack_indices(&packed, 4, 4);
        assert_eq!(result, vec![10, 3, 5, 1]);
    }

    #[test]
    fn unpack_2bit_indices() {
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

    // ── dequant_affine ──────────────────────────────────────────

    #[test]
    fn dequant_affine_per_tensor_int8() {
        let quantized: Vec<u8> = vec![0, 1, 128, 255];
        let shape = vec![2, 2];
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
            8,
            None,
        )
        .unwrap();

        assert_eq!(output.len(), 4 * 2);
        assert!((read_f16(&output, 0) - 0.0).abs() < 1e-3);
        assert!((read_f16(&output, 1) - 0.5).abs() < 1e-3);
        assert!((read_f16(&output, 2) - 64.0).abs() < 1e-1);
        assert!((read_f16(&output, 3) - 127.5).abs() < 1.0);
    }

    #[test]
    fn dequant_affine_per_axis() {
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
            8,
            None,
        )
        .unwrap();

        assert_eq!(output.len(), 6 * 2);
        assert!((read_f16(&output, 0) - 0.0).abs() < 1e-2);
        assert!((read_f16(&output, 1) - 1.0).abs() < 1e-2);
        assert!((read_f16(&output, 2) - 2.0).abs() < 1e-2);
        assert!((read_f16(&output, 3) - 0.0).abs() < 1e-2);
        assert!((read_f16(&output, 4) - 2.5).abs() < 1e-1);
        assert!((read_f16(&output, 5) - 5.0).abs() < 1e-1);
    }

    // ── dequant_affine_with_g_idx ───────────────────────────────

    #[test]
    fn dequant_affine_with_g_idx_uses_correct_groups() {
        let shape = vec![1, 4];
        let data: Vec<u8> = vec![1, 1, 1, 1];
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
        assert!((vals[0] - 3.0).abs() < 0.01);
        assert!((vals[1] - 2.0).abs() < 0.01);
        assert!((vals[2] - 2.0).abs() < 0.01);
        assert!((vals[3] - 3.0).abs() < 0.01);
    }

    // ── convert_params_to_f16 ───────────────────────────────────

    #[test]
    fn convert_fp32_params_to_f16() {
        let mut params = Vec::new();
        params.extend_from_slice(&0.5f32.to_le_bytes());
        params.extend_from_slice(&1.5f32.to_le_bytes());

        let result = convert_params_to_f16(&params, ScalarType::Float32, None, &[4], 2).unwrap();
        assert_eq!(result.len(), 4);
        assert!((read_f16(&result, 0) - 0.5).abs() < 1e-3);
        assert!((read_f16(&result, 1) - 1.5).abs() < 1e-3);
    }

    #[test]
    fn convert_fp16_params_to_f16() {
        let mut params = Vec::new();
        params.extend_from_slice(&f16_bytes(2.0));
        params.extend_from_slice(&f16_bytes(3.0));

        let result = convert_params_to_f16(&params, ScalarType::Float16, None, &[8], 4).unwrap();
        assert_eq!(result.len(), 4);
        assert!((read_f16(&result, 0) - 2.0).abs() < 1e-3);
        assert!((read_f16(&result, 1) - 3.0).abs() < 1e-3);
    }
}
