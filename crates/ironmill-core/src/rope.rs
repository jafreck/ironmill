//! Shared RoPE (Rotary Position Embedding) cache generation.
//!
//! Used by both the ANE compile pipeline and the Metal inference engine
//! to precompute cos/sin frequency tables.

use half::f16;

/// Extracted RoPE cos/sin cache data: `(cos_values, sin_values, values_per_position)`.
pub type RopeCacheData = (Vec<f16>, Vec<f16>, usize);

/// Precompute RoPE cos/sin cache tables.
///
/// * `head_dim` – size of each attention head
/// * `max_pos` – maximum sequence position
/// * `theta` – RoPE base frequency (e.g. 10 000.0)
/// * `rotary_dim` – if `Some(d)` and `d < head_dim`, dimensions beyond
///   `d / 2` are filled with identity (cos = 1, sin = 0). Pass `None` for
///   full rotation (equivalent to `Some(head_dim)`).
///
/// Returns flat `[max_pos × half_dim]` cos and sin tables in fp16, plus
/// the `half_dim` value.
pub fn precompute_rope_cache(
    head_dim: usize,
    max_pos: usize,
    theta: f64,
    rotary_dim: Option<usize>,
) -> RopeCacheData {
    let half_dim = head_dim / 2;
    let rd = rotary_dim.unwrap_or(head_dim);
    let rotary_half = rd / 2;
    let mut cos_cache = Vec::with_capacity(max_pos * half_dim);
    let mut sin_cache = Vec::with_capacity(max_pos * half_dim);

    for pos in 0..max_pos {
        for i in 0..half_dim {
            if i < rotary_half {
                let freq = 1.0_f64 / theta.powf(2.0 * i as f64 / rd as f64);
                let angle = pos as f64 * freq;
                cos_cache.push(f16::from_f64(angle.cos()));
                sin_cache.push(f16::from_f64(angle.sin()));
            } else {
                // Non-rotated dimensions: identity
                cos_cache.push(f16::from_f64(1.0));
                sin_cache.push(f16::from_f64(0.0));
            }
        }
    }
    (cos_cache, sin_cache, half_dim)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_rotation_basic() {
        let (cos, sin, half) = precompute_rope_cache(4, 2, 10000.0, None);
        assert_eq!(half, 2);
        assert_eq!(cos.len(), 4); // 2 positions × 2 half-dims
        assert_eq!(sin.len(), 4);
        // pos=0 → angle=0 for all dims → cos=1, sin=0
        assert_eq!(cos[0], f16::from_f64(1.0));
        assert_eq!(sin[0], f16::from_f64(0.0));
    }

    #[test]
    fn partial_rotation_fills_identity() {
        let (cos, sin, half) = precompute_rope_cache(8, 1, 10000.0, Some(4));
        assert_eq!(half, 4);
        // First 2 dims (rotary_half = 4/2 = 2) are computed, last 2 are identity.
        assert_eq!(cos[2], f16::from_f64(1.0));
        assert_eq!(sin[2], f16::from_f64(0.0));
        assert_eq!(cos[3], f16::from_f64(1.0));
        assert_eq!(sin[3], f16::from_f64(0.0));
    }
}
