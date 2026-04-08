//! AWQ block-level calibration utilities.
//!
//! Pure CPU functions used by the `awq_block_calibrate` tool (and
//! potentially other calibration workflows):
//!
//! - **AWQ scale computation:** [`compute_awq_scales`], [`compute_channel_magnitudes`]
//! - **Quantize-dequantize round-trip:** [`quantize_dequant_scaled`]
//! - **MSE comparison:** [`mse_f16_bytes`]
//! - **Weight group definitions:** [`WeightGroup`], [`weight_groups`], [`ATTN_PROJS`], [`FFN_PROJS`]

use half::f16;
use rayon::prelude::*;

use super::quantized::quantize_affine_into;
use crate::error::{CompileError, Result};

pub use super::quantized::AwqTensorConfig;

// Re-export shared calibration primitives from core so that downstream
// callers (e.g. `awq_block_calibrate` example) continue to import from
// `ironmill_compile::weights::calibration::*`.
pub use ironmill_core::calibration::{
    ATTN_PROJS, FFN_PROJS, WeightGroup, compute_awq_scales, compute_channel_magnitudes,
    weight_groups,
};

/// Quantize-then-dequantize a weight matrix with AWQ scaling applied.
///
/// Returns the dequantized f32 weights with the inverse scale applied,
/// i.e. the "approximated FP16 weights" that would result from INT4
/// quantization. Rows are processed in parallel via rayon.
pub fn quantize_dequant_scaled(
    weights: &[f32],
    out_features: usize,
    in_features: usize,
    scales: &[f32],
    group_size: usize,
) -> Vec<f32> {
    debug_assert_eq!(weights.len(), out_features * in_features);
    let qmax = 15.0_f32; // INT4
    let n_groups = in_features.div_ceil(group_size);
    let mut result = vec![0.0f32; weights.len()];

    // Process rows in parallel — each row is independent.
    result
        .par_chunks_mut(in_features)
        .enumerate()
        .for_each(|(row, result_row)| {
            let weight_row = &weights[row * in_features..(row + 1) * in_features];
            let mut quant_buf = vec![0u8; group_size];

            for g in 0..n_groups {
                let g_start = g * group_size;
                let g_end = (g_start + group_size).min(in_features);
                let gsize = g_end - g_start;

                // Scale weights
                let group_vals: Vec<f32> = (g_start..g_end)
                    .map(|c| weight_row[c] * scales[c])
                    .collect();

                // Quantize
                let (scale, zp) = quantize_affine_into(&group_vals, qmax, &mut quant_buf[..gsize]);

                // Dequantize and undo scaling
                for (j, &qb_val) in quant_buf[..gsize].iter().enumerate() {
                    let dequant = (qb_val as f32 - zp) * scale;
                    let c = g_start + j;
                    result_row[c] = dequant / scales[c];
                }
            }
        });

    result
}

/// Compute MSE between two FP16 byte buffers.
pub fn mse_f16_bytes(a: &[u8], b: &[u8]) -> Result<f64> {
    if a.len() != b.len() {
        return Err(CompileError::Other(format!(
            "mse_f16_bytes: mismatched buffer lengths ({} vs {})",
            a.len(),
            b.len(),
        )));
    }
    let n = a.len() / 2;
    if n == 0 {
        return Ok(0.0);
    }
    let mut sum = 0.0_f64;
    for i in 0..n {
        let va = f16::from_le_bytes([a[i * 2], a[i * 2 + 1]]).to_f64();
        let vb = f16::from_le_bytes([b[i * 2], b[i * 2 + 1]]).to_f64();
        let d = va - vb;
        sum += d * d;
    }
    Ok(sum / n as f64)
}

/// Reinterpret `&[f16]` as raw little-endian bytes.
///
/// # Safety
/// `f16` is `#[repr(transparent)]` over `u16`, so the reinterpret is safe
/// on little-endian targets (all Apple Silicon / x86-64).
pub fn f16_slice_to_bytes(data: &[f16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 2);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Search for optimal per-group clip ranges on AWQ-scaled weights.
///
/// Thin wrapper around the internal clip search in `quantized.rs`.
/// See [`super::quantized::search_clip_ranges`] for the full algorithm.
pub fn search_clip_ranges(
    scaled_weights: &[f32],
    out_features: usize,
    in_features: usize,
    group_size: usize,
    qmax: f32,
    activations: &[f32],
    token_count: usize,
    clip_grid: usize,
    max_shrink: f32,
) -> Vec<f32> {
    super::quantized::search_clip_ranges(
        scaled_weights,
        out_features,
        in_features,
        group_size,
        qmax,
        activations,
        token_count,
        clip_grid,
        max_shrink,
    )
}

/// Apply per-group clipping to a weight matrix in-place.
///
/// Thin wrapper around the internal clip function in `quantized.rs`.
pub fn apply_clip(
    weights: &mut [f32],
    out_features: usize,
    in_features: usize,
    group_size: usize,
    clip_maxvals: &[f32],
) {
    super::quantized::apply_clip(weights, out_features, in_features, group_size, clip_maxvals);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_awq_scales_alpha_zero() {
        let mags = vec![1.0, 2.0, 3.0];
        let scales = compute_awq_scales(&mags, 0.0);
        assert_eq!(scales, vec![1.0; 3]);
    }

    #[test]
    fn test_awq_scales_nonzero() {
        let mags = vec![1.0, 4.0, 9.0];
        let scales = compute_awq_scales(&mags, 0.5);
        // All should be positive and normalised
        assert!(scales.iter().all(|&s| s > 0.0));
    }

    #[test]
    fn test_channel_magnitudes() {
        // 2 tokens, 3 features: [[1, -2, 3], [3, 2, -1]]
        let acts = vec![1.0, -2.0, 3.0, 3.0, 2.0, -1.0];
        let mags = compute_channel_magnitudes(&acts, 3);
        assert_eq!(mags.len(), 3);
        assert!((mags[0] - 2.0).abs() < 1e-6); // (1+3)/2
        assert!((mags[1] - 2.0).abs() < 1e-6); // (2+2)/2
        assert!((mags[2] - 2.0).abs() < 1e-6); // (3+1)/2
    }

    #[test]
    fn test_quantize_dequant_roundtrip() {
        let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let scales = vec![1.0; 4];
        let result = quantize_dequant_scaled(&weights, 2, 4, &scales, 4);
        assert_eq!(result.len(), 8);
        // Round-trip should be close to original (INT4 quantisation noise)
        for (orig, dq) in weights.iter().zip(result.iter()) {
            assert!((orig - dq).abs() < 0.15, "orig={orig}, dq={dq}");
        }
    }

    #[test]
    fn test_mse_f16_identical() {
        let data: Vec<u8> = [1.0_f32, 2.0, 3.0]
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect();
        assert!((mse_f16_bytes(&data, &data).unwrap()).abs() < 1e-12);
    }

    #[test]
    fn test_weight_groups_coverage() {
        let groups = weight_groups();
        let all_projs: Vec<&str> = groups.iter().flat_map(|g| &g.proj_names).copied().collect();
        for &p in ATTN_PROJS {
            assert!(all_projs.contains(&p), "missing {p}");
        }
        for &p in FFN_PROJS {
            assert!(all_projs.contains(&p), "missing {p}");
        }
    }
}
