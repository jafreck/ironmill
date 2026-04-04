//! Configuration for EAGLE-3 speculative decoding.

/// Configuration for speculative decoding.
///
/// Controls the tree search depth, branching factor, and acceptance
/// threshold used by the [`SpeculativeEngine`](super::SpeculativeEngine).
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
}
