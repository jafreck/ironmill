//! Shared dequantization helpers used by both Metal and MLX backends.
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
