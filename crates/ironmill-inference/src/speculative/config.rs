//! Configuration for EAGLE-3 speculative decoding.

/// Configuration for speculative decoding.
///
/// Controls the tree search depth, branching factor, and acceptance
/// threshold used by the [`SpeculativeEngine`](super::SpeculativeEngine).
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct SpecConfig {
    /// Maximum tokens the draft head may propose per speculation round.
    pub max_draft_depth: usize,
    /// Number of candidate continuations evaluated at each tree position.
    pub tree_width: usize,
    /// Minimum target-model probability for a draft token to be accepted
    /// via Leviathan et al. rejection sampling.
    pub acceptance_threshold: f32,
}

impl Default for SpecConfig {
    fn default() -> Self {
        Self {
            max_draft_depth: 5,
            tree_width: 3,
            acceptance_threshold: 0.1,
        }
    }
}

/// Configuration for the TurboSpec adaptive speculation controller.
///
/// TurboSpec dynamically tunes [`SpecConfig`] parameters (depth, width,
/// acceptance threshold) based on observed acceptance rates, using an
/// exponential moving average (EMA) to smooth noisy per-round signals.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TurboSpecConfig {
    /// Starting speculation depth.
    pub initial_depth: usize,
    /// Minimum allowed depth (controller won't go below this).
    pub min_depth: usize,
    /// Maximum allowed depth (controller won't go above this).
    pub max_depth: usize,
    /// EMA decay factor for acceptance rate smoothing (0 < α ≤ 1).
    pub ema_alpha: f32,
    /// Increase depth when smoothed acceptance rate exceeds this.
    pub depth_up_threshold: f32,
    /// Decrease depth when smoothed acceptance rate drops below this.
    pub depth_down_threshold: f32,
}

impl Default for TurboSpecConfig {
    fn default() -> Self {
        Self {
            initial_depth: 5,
            min_depth: 1,
            max_depth: 10,
            ema_alpha: 0.1,
            depth_up_threshold: 0.8,
            depth_down_threshold: 0.4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speculative_config_default_values() {
        let cfg = SpecConfig::default();
        assert_eq!(cfg.max_draft_depth, 5);
        assert_eq!(cfg.tree_width, 3);
        assert!((cfg.acceptance_threshold - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn speculative_config_custom() {
        let cfg = SpecConfig {
            max_draft_depth: 8,
            tree_width: 5,
            acceptance_threshold: 0.05,
        };
        assert_eq!(cfg.max_draft_depth, 8);
        assert_eq!(cfg.tree_width, 5);
        assert!((cfg.acceptance_threshold - 0.05).abs() < f32::EPSILON);
    }

    #[test]
    fn turbospec_config_default_values() {
        let cfg = TurboSpecConfig::default();
        assert_eq!(cfg.initial_depth, 5);
        assert_eq!(cfg.min_depth, 1);
        assert_eq!(cfg.max_depth, 10);
        assert!((cfg.ema_alpha - 0.1).abs() < f32::EPSILON);
        assert!((cfg.depth_up_threshold - 0.8).abs() < f32::EPSILON);
        assert!((cfg.depth_down_threshold - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn turbospec_config_custom() {
        let cfg = TurboSpecConfig {
            initial_depth: 3,
            min_depth: 2,
            max_depth: 8,
            ema_alpha: 0.2,
            depth_up_threshold: 0.9,
            depth_down_threshold: 0.3,
        };
        assert_eq!(cfg.initial_depth, 3);
        assert_eq!(cfg.min_depth, 2);
        assert_eq!(cfg.max_depth, 8);
        assert!((cfg.ema_alpha - 0.2).abs() < f32::EPSILON);
    }
}
