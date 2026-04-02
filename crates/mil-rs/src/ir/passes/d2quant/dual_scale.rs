//! Core dual-scale quantization math for D2Quant sub-4-bit quantization.
//!
//! Within each weight group, weights are partitioned into "normal" and
//! "outlier" sets based on magnitude.  Each partition gets its own
//! scale / zero_point pair so that both the bulk of weights and the
//! extreme values are represented with higher precision than a single
//! shared scale could achieve.

/// Parameters produced by dual-scale quantization for a single group.
#[derive(Debug, Clone)]
pub struct DualScaleParams {
    /// Affine scale for the normal (low-magnitude) partition.
    pub normal_scale: f32,
    /// Affine zero-point for the normal partition.
    pub normal_zero: f32,
    /// Affine scale for the outlier (high-magnitude) partition.
    pub outlier_scale: f32,
    /// Affine zero-point for the outlier partition.
    pub outlier_zero: f32,
    /// Per-weight mask: `true` = outlier, `false` = normal.
    pub outlier_mask: Vec<bool>,
}

/// Quantize a group of weights using dual-scale partitioning.
///
/// 1. Sort by absolute magnitude to find the percentile threshold.
/// 2. Partition into normal (below threshold) and outlier (at or above).
/// 3. Compute separate min/max affine params for each partition.
/// 4. Quantize each weight using its partition's params.
///
/// `bits` must be 2 or 3.  `outlier_percentile` is in `[0, 1]` — e.g.
/// `0.99` means the top 1 % by magnitude are outliers.
pub fn dual_scale_quantize(
    weights: &[f32],
    bits: u8,
    outlier_percentile: f32,
) -> (Vec<u8>, DualScaleParams) {
    assert!(bits == 2 || bits == 3, "bits must be 2 or 3");
    assert!(
        (0.0..=1.0).contains(&outlier_percentile),
        "outlier_percentile must be in [0, 1]"
    );

    let max_val = (1u8 << bits) - 1; // 3 for 2-bit, 7 for 3-bit
    let levels = max_val as f32;

    if weights.is_empty() {
        return (
            Vec::new(),
            DualScaleParams {
                normal_scale: 1.0,
                normal_zero: 0.0,
                outlier_scale: 1.0,
                outlier_zero: 0.0,
                outlier_mask: Vec::new(),
            },
        );
    }

    // --- 1. Determine magnitude threshold ---
    let mut magnitudes: Vec<f32> = weights.iter().map(|w| w.abs()).collect();
    magnitudes.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    // Index of the percentile boundary (exclusive for normal partition).
    let threshold_idx = ((weights.len() as f32) * outlier_percentile)
        .floor()
        .min((weights.len() - 1) as f32) as usize;
    let threshold = magnitudes[threshold_idx];

    // --- 2. Partition ---
    let outlier_mask: Vec<bool> = weights.iter().map(|w| w.abs() >= threshold).collect();

    // Guard: if everything ended up in one partition (e.g. all identical
    // weights), fall back so that both partitions are well-defined.
    let has_normal = outlier_mask.iter().any(|&o| !o);
    let has_outlier = outlier_mask.iter().any(|&o| o);

    // Collect partition values.
    let normal_vals: Vec<f32> = weights
        .iter()
        .zip(&outlier_mask)
        .filter(|&(_, &is_out)| !is_out)
        .map(|(&w, _)| w)
        .collect();
    let outlier_vals: Vec<f32> = weights
        .iter()
        .zip(&outlier_mask)
        .filter(|&(_, &is_out)| is_out)
        .map(|(&w, _)| w)
        .collect();

    // --- 3. Compute affine params per partition ---
    let (normal_scale, normal_zero) = if has_normal {
        affine_params(&normal_vals, levels)
    } else {
        // Degenerate: use outlier params for both.
        affine_params(&outlier_vals, levels)
    };

    let (outlier_scale, outlier_zero) = if has_outlier {
        affine_params(&outlier_vals, levels)
    } else {
        // Degenerate: use normal params for both.
        affine_params(&normal_vals, levels)
    };

    // --- 4. Quantize ---
    let quantized: Vec<u8> = weights
        .iter()
        .zip(&outlier_mask)
        .map(|(&w, &is_out)| {
            let (scale, zp) = if is_out {
                (outlier_scale, outlier_zero)
            } else {
                (normal_scale, normal_zero)
            };
            let q = (w / scale + zp).round().clamp(0.0, levels);
            q as u8
        })
        .collect();

    let params = DualScaleParams {
        normal_scale,
        normal_zero,
        outlier_scale,
        outlier_zero,
        outlier_mask,
    };
    (quantized, params)
}

/// Dequantize values produced by [`dual_scale_quantize`].
pub fn dual_scale_dequantize(quantized: &[u8], params: &DualScaleParams, bits: u8) -> Vec<f32> {
    assert!(bits == 2 || bits == 3, "bits must be 2 or 3");
    let _max_val = (1u8 << bits) - 1;

    quantized
        .iter()
        .zip(&params.outlier_mask)
        .map(|(&q, &is_out)| {
            let (scale, zp) = if is_out {
                (params.outlier_scale, params.outlier_zero)
            } else {
                (params.normal_scale, params.normal_zero)
            };
            (q as f32 - zp) * scale
        })
        .collect()
}

// ── Bit-packing helpers ──────────────────────────────────────────────

/// Pack 2-bit values (0–3): 4 values per byte, LSB-first.
pub fn pack_2bit(values: &[u8]) -> Vec<u8> {
    values
        .chunks(4)
        .map(|chunk| {
            let mut byte = 0u8;
            for (i, &v) in chunk.iter().enumerate() {
                byte |= (v & 0x03) << (i * 2);
            }
            byte
        })
        .collect()
}

/// Unpack 2-bit values from packed bytes.
pub fn unpack_2bit(packed: &[u8], count: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(count);
    for &byte in packed {
        for shift in (0..8).step_by(2) {
            if out.len() >= count {
                break;
            }
            out.push((byte >> shift) & 0x03);
        }
    }
    out.truncate(count);
    out
}

/// Pack 3-bit values: 8 values → 3 bytes (24 bits), LSB-first.
pub fn pack_3bit(values: &[u8]) -> Vec<u8> {
    let mut packed = Vec::new();
    // Process 8 values at a time → 24 bits = 3 bytes.
    for chunk in values.chunks(8) {
        let mut bits: u32 = 0;
        for (i, &v) in chunk.iter().enumerate() {
            bits |= ((v & 0x07) as u32) << (i * 3);
        }
        // Emit 3 bytes (lowest 24 bits).
        packed.push((bits & 0xFF) as u8);
        packed.push(((bits >> 8) & 0xFF) as u8);
        packed.push(((bits >> 16) & 0xFF) as u8);
    }
    packed
}

/// Unpack 3-bit values from packed bytes.
pub fn unpack_3bit(packed: &[u8], count: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(count);
    // Process 3 bytes at a time → 8 values.
    for triple in packed.chunks(3) {
        let mut bits: u32 = 0;
        for (i, &b) in triple.iter().enumerate() {
            bits |= (b as u32) << (i * 8);
        }
        for j in 0..8 {
            if out.len() >= count {
                break;
            }
            out.push(((bits >> (j * 3)) & 0x07) as u8);
        }
    }
    out.truncate(count);
    out
}

/// Pack the outlier boolean mask: one bit per weight, LSB-first.
pub fn pack_mask(mask: &[bool]) -> Vec<u8> {
    mask.chunks(8)
        .map(|chunk| {
            let mut byte = 0u8;
            for (i, &is_out) in chunk.iter().enumerate() {
                if is_out {
                    byte |= 1 << i;
                }
            }
            byte
        })
        .collect()
}

/// Unpack an outlier mask from packed bytes.
pub fn unpack_mask(packed: &[u8], count: usize) -> Vec<bool> {
    let mut out = Vec::with_capacity(count);
    for &byte in packed {
        for i in 0..8 {
            if out.len() >= count {
                break;
            }
            out.push((byte >> i) & 1 == 1);
        }
    }
    out.truncate(count);
    out
}

// ── Internal helpers ─────────────────────────────────────────────────

/// Compute affine quantization parameters (scale, zero_point) from a set of
/// values, mapping `[min, max]` to `[0, levels]`.
fn affine_params(values: &[f32], levels: f32) -> (f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in values {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }

    if (max - min).abs() < f32::EPSILON {
        // Degenerate — all values are identical.
        let zp = (-min).round();
        return (1.0, zp);
    }

    let scale = (max - min) / levels;
    let zero_point = (-min / scale).round();
    (scale, zero_point)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Round-trip correctness ───────────────────────────────────────

    #[test]
    fn round_trip_2bit() {
        let weights: Vec<f32> = vec![
            -0.1, 0.0, 0.05, 0.1, -0.05, 0.02, -0.08, 0.03, // normal-ish
            -0.5, 0.4, -0.3, 0.2, 0.15, -0.12, 0.07, 0.01,
        ];
        let (quantized, params) = dual_scale_quantize(&weights, 2, 0.99);
        let recovered = dual_scale_dequantize(&quantized, &params, 2);

        assert_eq!(recovered.len(), weights.len());
        for (orig, recov) in weights.iter().zip(&recovered) {
            let err = (orig - recov).abs();
            // 2-bit has only 4 levels so per-partition step can be large.
            assert!(err < 1.0, "error {err} too large for {orig} → {recov}");
        }
    }

    #[test]
    fn round_trip_3bit() {
        let weights: Vec<f32> = vec![
            -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, -2.0, 0.25, -0.25, 0.75, -0.75, 1.25, -1.25, 0.1,
            -0.1,
        ];
        let (quantized, params) = dual_scale_quantize(&weights, 3, 0.99);
        let recovered = dual_scale_dequantize(&quantized, &params, 3);

        assert_eq!(recovered.len(), weights.len());
        for (orig, recov) in weights.iter().zip(&recovered) {
            let err = (orig - recov).abs();
            assert!(err < 1.0, "error {err} too large for {orig} → {recov}");
        }
    }

    // ── Dual-scale beats single-scale ────────────────────────────────

    fn single_scale_mse(weights: &[f32], bits: u8) -> f64 {
        let max_val = (1u8 << bits) - 1;
        let levels = max_val as f32;

        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &v in weights {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
        let scale = if (max - min).abs() < f32::EPSILON {
            1.0
        } else {
            (max - min) / levels
        };
        let zp = (-min / scale).round();

        let mut sum_sq = 0.0_f64;
        for &w in weights {
            let q = (w / scale + zp).round().clamp(0.0, levels) as u8;
            let deq = (q as f32 - zp) * scale;
            let err = (w - deq) as f64;
            sum_sq += err * err;
        }
        sum_sq / weights.len() as f64
    }

    fn dual_scale_mse(weights: &[f32], bits: u8) -> f64 {
        let (quantized, params) = dual_scale_quantize(weights, bits, 0.99);
        let recovered = dual_scale_dequantize(&quantized, &params, bits);

        let mut sum_sq = 0.0_f64;
        for (orig, recov) in weights.iter().zip(&recovered) {
            let err = (*orig - recov) as f64;
            sum_sq += err * err;
        }
        sum_sq / weights.len() as f64
    }

    #[test]
    fn dual_scale_2bit_lower_mse_than_single() {
        // Weights with a few outliers: most are in [-0.1, 0.1], a few at ±1.0.
        let mut weights: Vec<f32> = (0..128).map(|i| ((i as f32 / 128.0) - 0.5) * 0.2).collect();
        // Inject outliers.
        weights[0] = 1.0;
        weights[1] = -1.0;

        let single = single_scale_mse(&weights, 2);
        let dual = dual_scale_mse(&weights, 2);

        assert!(
            dual <= single,
            "dual-scale MSE ({dual:.6}) should be ≤ single-scale MSE ({single:.6})"
        );
    }

    #[test]
    fn dual_scale_3bit_lower_mse_than_single() {
        let mut weights: Vec<f32> = (0..128).map(|i| ((i as f32 / 128.0) - 0.5) * 0.2).collect();
        weights[0] = 2.0;
        weights[1] = -2.0;

        let single = single_scale_mse(&weights, 3);
        let dual = dual_scale_mse(&weights, 3);

        assert!(
            dual <= single,
            "dual-scale MSE ({dual:.6}) should be ≤ single-scale MSE ({single:.6})"
        );
    }

    // ── Bit-packing round-trips ──────────────────────────────────────

    #[test]
    fn pack_unpack_2bit() {
        let values: Vec<u8> = vec![0, 1, 2, 3, 3, 2, 1, 0, 1];
        let packed = pack_2bit(&values);
        let unpacked = unpack_2bit(&packed, values.len());
        assert_eq!(unpacked, values);
    }

    #[test]
    fn pack_unpack_3bit() {
        let values: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7, 3, 5];
        let packed = pack_3bit(&values);
        let unpacked = unpack_3bit(&packed, values.len());
        assert_eq!(unpacked, values);
    }

    #[test]
    fn pack_unpack_mask() {
        let mask = vec![
            true, false, false, true, true, true, false, false, true, false,
        ];
        let packed = pack_mask(&mask);
        let unpacked = unpack_mask(&packed, mask.len());
        assert_eq!(unpacked, mask);
    }

    // ── Edge cases ───────────────────────────────────────────────────

    #[test]
    fn empty_weights() {
        let (q, params) = dual_scale_quantize(&[], 2, 0.99);
        assert!(q.is_empty());
        assert!(params.outlier_mask.is_empty());
    }

    #[test]
    fn all_same_weights() {
        let weights = vec![0.5; 16];
        let (quantized, params) = dual_scale_quantize(&weights, 2, 0.99);
        let recovered = dual_scale_dequantize(&quantized, &params, 2);

        for recov in &recovered {
            assert!(
                (recov - 0.5).abs() < 1.0,
                "all-same should recover near 0.5, got {recov}"
            );
        }
    }

    #[test]
    fn single_weight() {
        let weights = vec![3.14];
        let (quantized, params) = dual_scale_quantize(&weights, 3, 0.99);
        let recovered = dual_scale_dequantize(&quantized, &params, 3);
        assert_eq!(recovered.len(), 1);
    }

    #[test]
    #[should_panic(expected = "bits must be 2 or 3")]
    fn invalid_bits_panics() {
        dual_scale_quantize(&[1.0], 4, 0.99);
    }
}
