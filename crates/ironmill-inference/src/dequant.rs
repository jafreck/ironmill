//! Shared dequantization helpers used by the Metal backend.
//!
//! Low-level routines for reading typed scalars and unpacking bit-packed
//! indices are now shared via [`ironmill_core::dequant`]. This module
//! re-exports them for crate-internal use and provides the GPTQ `g_idx`
//! dequantization variant.

// Re-export shared helpers from core so existing crate-internal callers
// continue to compile without changing their import paths.
pub(crate) use ironmill_core::dequant::{
    dequant_affine_with_g_idx, read_typed_f32, unpack_indices,
};

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use mil_rs::ir::ScalarType;

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
