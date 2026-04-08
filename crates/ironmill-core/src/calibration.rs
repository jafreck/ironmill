//! Shared AWQ calibration primitives.
//!
//! Pure-math functions and domain constants for Activation-aware Weight
//! Quantization (AWQ) that are used by both the compile pipeline and the
//! inference-side calibration engine.

// ── Projection constants ────────────────────────────────────────────────

/// Projection names under `self_attn` in the HuggingFace naming convention.
pub const ATTN_PROJS: &[&str] = &["q_proj", "k_proj", "v_proj", "o_proj"];

/// Projection names under `mlp` in the HuggingFace naming convention.
pub const FFN_PROJS: &[&str] = &["gate_proj", "up_proj", "down_proj"];

// ── Weight groups ───────────────────────────────────────────────────────

/// A group of projections that share the same activation norm for AWQ scaling.
pub struct WeightGroup {
    /// Projection names in this group.
    pub proj_names: Vec<&'static str>,
    /// Which norm feeds this group: `"attn"` or `"ffn"`.
    pub norm_key: &'static str,
}

/// Standard AWQ weight groupings.
///
/// Q/K/V share a single alpha (all fed by attn_norm), O_proj gets its own
/// (different input distribution), gate/up share one (ffn_norm), and
/// down_proj gets its own.
pub fn weight_groups() -> Vec<WeightGroup> {
    vec![
        WeightGroup {
            proj_names: vec!["q_proj", "k_proj", "v_proj"],
            norm_key: "attn",
        },
        WeightGroup {
            proj_names: vec!["o_proj"],
            norm_key: "attn",
        },
        WeightGroup {
            proj_names: vec!["gate_proj", "up_proj"],
            norm_key: "ffn",
        },
        WeightGroup {
            proj_names: vec!["down_proj"],
            norm_key: "ffn",
        },
    ]
}

// ── Scale computation ───────────────────────────────────────────────────

/// Compute AWQ scales from activation magnitudes and an alpha value.
///
/// Matches the reference AWQ normalisation:
///   `scales[c] = x_max[c]^alpha`
///   `scales /= sqrt(max(scales) * min(scales))`
pub fn compute_awq_scales(x_max: &[f32], alpha: f32) -> Vec<f32> {
    if alpha == 0.0 {
        return vec![1.0; x_max.len()];
    }
    let mut scales: Vec<f32> = x_max.iter().map(|&m| m.powf(alpha).max(1e-4)).collect();
    let max_s = scales.iter().cloned().fold(0.0_f32, f32::max);
    let min_s = scales.iter().cloned().fold(f32::INFINITY, f32::min);
    let norm = (max_s * min_s).sqrt().max(1e-8);
    for s in &mut scales {
        *s /= norm;
    }
    scales
}

/// Compute per-channel mean absolute activation (x_max) from flat
/// `[tokens × features]` f32 data.
pub fn compute_channel_magnitudes(activations: &[f32], n_features: usize) -> Vec<f32> {
    if n_features == 0 || activations.is_empty() {
        return Vec::new();
    }
    let n_tokens = activations.len() / n_features;
    let mut mags = vec![0.0_f32; n_features];
    for t in 0..n_tokens {
        let row = &activations[t * n_features..(t + 1) * n_features];
        for (c, &val) in row.iter().enumerate() {
            mags[c] += val.abs();
        }
    }
    if n_tokens > 0 {
        let inv = 1.0 / n_tokens as f32;
        for m in &mut mags {
            *m *= inv;
        }
    }
    mags
}

// ── Tests ───────────────────────────────────────────────────────────────

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
    fn test_channel_magnitudes_empty() {
        assert!(compute_channel_magnitudes(&[], 3).is_empty());
        assert!(compute_channel_magnitudes(&[1.0, 2.0], 0).is_empty());
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
