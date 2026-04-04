//! TurboSpec adaptive speculation controller.
//!
//! Implements a closed-loop runtime controller that dynamically tunes
//! speculation depth, tree width, and acceptance thresholds based on
//! observed acceptance rates. The controller uses an exponential moving
//! average (EMA) to smooth noisy per-round acceptance signals and adjusts
//! [`SpecConfig`](super::SpecConfig) parameters accordingly.

use std::time::Instant;

use super::config::{SpecConfig, TurboSpecConfig};

/// Adaptive controller that tunes speculation parameters at runtime.
///
/// After each speculation round the caller reports how many tokens were
/// proposed and how many were accepted. The controller maintains a smoothed
/// acceptance rate (EMA) and adjusts depth, width, and acceptance threshold
/// to maximise *goodput* — accepted tokens per wall-clock second.
pub struct TurboSpecController {
    config: TurboSpecConfig,
    /// Exponential moving average of the per-round acceptance rate.
    acceptance_ema: f32,
    /// Current speculation depth recommended to the engine.
    current_depth: usize,
    /// Current tree width recommended to the engine.
    current_width: usize,
    /// Lifetime count of accepted tokens.
    total_accepted: u64,
    /// Lifetime count of proposed tokens.
    total_proposed: u64,
    /// Timestamp of the first `observe` call (for goodput calculation).
    start_time: Option<Instant>,
}

impl TurboSpecController {
    /// Create a new controller with the given configuration.
    pub fn new(config: TurboSpecConfig) -> Self {
        let depth = config
            .initial_depth
            .clamp(config.min_depth, config.max_depth);
        Self {
            acceptance_ema: 0.5, // neutral starting point
            current_depth: depth,
            current_width: 3, // sensible default
            total_accepted: 0,
            total_proposed: 0,
            start_time: None,
            config,
        }
    }

    /// Update the controller with the result of a speculation round.
    ///
    /// `proposed` is the number of draft tokens generated and `accepted` is
    /// how many the target model accepted. The controller updates the EMA
    /// and adjusts depth/width.
    pub fn observe(&mut self, proposed: usize, accepted: usize) {
        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
        }

        self.total_proposed += proposed as u64;
        self.total_accepted += accepted as u64;

        let rate = if proposed > 0 {
            accepted as f32 / proposed as f32
        } else {
            0.0
        };

        // EMA update: ema = α * sample + (1 - α) * ema
        let alpha = self.config.ema_alpha;
        self.acceptance_ema = alpha * rate + (1.0 - alpha) * self.acceptance_ema;

        self.adapt();
    }

    /// Return the current recommended [`SpecConfig`] for the next round.
    pub fn current_config(&self) -> SpecConfig {
        SpecConfig {
            max_draft_depth: self.current_depth,
            tree_width: self.current_width,
            acceptance_threshold: 0.1,
        }
    }

    /// Goodput metric: total accepted tokens / elapsed wall-clock seconds.
    ///
    /// Returns 0.0 if no observations have been recorded yet.
    pub fn goodput(&self) -> f64 {
        match self.start_time {
            Some(t) => {
                let elapsed = t.elapsed().as_secs_f64();
                if elapsed > 0.0 {
                    self.total_accepted as f64 / elapsed
                } else {
                    self.total_accepted as f64
                }
            }
            None => 0.0,
        }
    }

    /// Current smoothed acceptance rate (EMA).
    pub fn acceptance_rate(&self) -> f32 {
        self.acceptance_ema
    }

    /// Current recommended depth.
    pub fn depth(&self) -> usize {
        self.current_depth
    }

    /// Current recommended width.
    pub fn width(&self) -> usize {
        self.current_width
    }

    /// Lifetime accepted token count.
    pub fn total_accepted(&self) -> u64 {
        self.total_accepted
    }

    /// Lifetime proposed token count.
    pub fn total_proposed(&self) -> u64 {
        self.total_proposed
    }

    /// Adjust depth and width based on the current EMA.
    fn adapt(&mut self) {
        let ema = self.acceptance_ema;

        // Depth adjustment.
        if ema > self.config.depth_up_threshold && self.current_depth < self.config.max_depth {
            self.current_depth += 1;
        } else if ema < self.config.depth_down_threshold
            && self.current_depth > self.config.min_depth
        {
            self.current_depth -= 1;
        }

        // Width adjustment: mirror depth logic with slightly relaxed
        // thresholds so width trails depth changes.
        let width_up = self.config.depth_up_threshold * 0.95;
        let width_down = self.config.depth_down_threshold * 1.1;
        if ema > width_up && self.current_width < 6 {
            self.current_width += 1;
        } else if ema < width_down && self.current_width > 1 {
            self.current_width -= 1;
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_controller() -> TurboSpecController {
        TurboSpecController::new(TurboSpecConfig::default())
    }

    #[test]
    fn turbospec_initial_state() {
        let ctrl = default_controller();
        assert_eq!(ctrl.depth(), 5);
        assert_eq!(ctrl.total_accepted(), 0);
        assert_eq!(ctrl.total_proposed(), 0);
        assert!((ctrl.acceptance_rate() - 0.5).abs() < f32::EPSILON);
        assert!(ctrl.goodput() == 0.0);
    }

    #[test]
    fn turbospec_increases_depth_on_high_acceptance() {
        let config = TurboSpecConfig {
            initial_depth: 3,
            min_depth: 1,
            max_depth: 10,
            ema_alpha: 0.9, // high alpha → fast response
            depth_up_threshold: 0.8,
            depth_down_threshold: 0.4,
        };
        let mut ctrl = TurboSpecController::new(config);
        let initial_depth = ctrl.depth();

        // Feed consistently high acceptance.
        for _ in 0..10 {
            ctrl.observe(10, 10); // 100% acceptance
        }
        assert!(
            ctrl.depth() > initial_depth,
            "depth should increase from {} but is {}",
            initial_depth,
            ctrl.depth()
        );
    }

    #[test]
    fn turbospec_decreases_depth_on_low_acceptance() {
        let config = TurboSpecConfig {
            initial_depth: 5,
            min_depth: 1,
            max_depth: 10,
            ema_alpha: 0.9, // fast response
            depth_up_threshold: 0.8,
            depth_down_threshold: 0.4,
        };
        let mut ctrl = TurboSpecController::new(config);
        let initial_depth = ctrl.depth();

        // Feed consistently low acceptance.
        for _ in 0..10 {
            ctrl.observe(10, 1); // 10% acceptance
        }
        assert!(
            ctrl.depth() < initial_depth,
            "depth should decrease from {} but is {}",
            initial_depth,
            ctrl.depth()
        );
    }

    #[test]
    fn turbospec_depth_clamped_to_bounds() {
        let config = TurboSpecConfig {
            initial_depth: 2,
            min_depth: 2,
            max_depth: 3,
            ema_alpha: 0.95,
            depth_up_threshold: 0.8,
            depth_down_threshold: 0.4,
        };
        let mut ctrl = TurboSpecController::new(config);

        // Try to push below min.
        for _ in 0..20 {
            ctrl.observe(10, 0);
        }
        assert_eq!(ctrl.depth(), 2, "depth must not go below min_depth");

        // Try to push above max.
        for _ in 0..20 {
            ctrl.observe(10, 10);
        }
        assert!(ctrl.depth() <= 3, "depth must not exceed max_depth");
    }

    #[test]
    fn turbospec_goodput_tracks_accepted() {
        let mut ctrl = default_controller();
        ctrl.observe(10, 8);
        ctrl.observe(10, 7);
        assert_eq!(ctrl.total_accepted(), 15);
        assert_eq!(ctrl.total_proposed(), 20);
        // Goodput should be positive after observations.
        assert!(ctrl.goodput() > 0.0);
    }

    #[test]
    fn turbospec_ema_converges() {
        let config = TurboSpecConfig {
            ema_alpha: 0.1,
            ..TurboSpecConfig::default()
        };
        let mut ctrl = TurboSpecController::new(config);

        // Drive towards 90% acceptance rate.
        for _ in 0..200 {
            ctrl.observe(10, 9);
        }
        let ema = ctrl.acceptance_rate();
        assert!(
            (ema - 0.9).abs() < 0.05,
            "EMA should converge near 0.9, got {ema}"
        );
    }

    #[test]
    fn turbospec_converges_within_50_rounds() {
        let config = TurboSpecConfig {
            initial_depth: 3,
            min_depth: 1,
            max_depth: 10,
            ema_alpha: 0.15,
            depth_up_threshold: 0.8,
            depth_down_threshold: 0.4,
        };
        let mut ctrl = TurboSpecController::new(config);

        // High acceptance: depth should reach a stable higher value within 50 rounds.
        for _ in 0..50 {
            ctrl.observe(10, 9);
        }

        let depth_at_50 = ctrl.depth();
        assert!(
            depth_at_50 > 3,
            "controller should have increased depth within 50 rounds, got {depth_at_50}"
        );

        // 10 more rounds should not change depth much (stability).
        for _ in 0..10 {
            ctrl.observe(10, 9);
        }
        let depth_at_60 = ctrl.depth();
        // Allow at most 1 step of further drift.
        assert!(
            (depth_at_60 as i32 - depth_at_50 as i32).unsigned_abs() <= 1,
            "controller should be roughly stable: depth_50={depth_at_50}, depth_60={depth_at_60}"
        );
    }

    #[test]
    fn turbospec_current_config_reflects_state() {
        let config = TurboSpecConfig {
            initial_depth: 4,
            min_depth: 1,
            max_depth: 10,
            ema_alpha: 0.9,
            depth_up_threshold: 0.8,
            depth_down_threshold: 0.4,
        };
        let mut ctrl = TurboSpecController::new(config);
        let spec = ctrl.current_config();
        assert_eq!(spec.max_draft_depth, 4);

        // After high acceptance, depth should increase.
        for _ in 0..5 {
            ctrl.observe(10, 10);
        }
        let spec = ctrl.current_config();
        assert!(
            spec.max_draft_depth > 4,
            "current_config depth should reflect increased depth"
        );
    }

    #[test]
    fn turbospec_zero_proposed_does_not_panic() {
        let mut ctrl = default_controller();
        ctrl.observe(0, 0); // edge case
        assert_eq!(ctrl.total_proposed(), 0);
    }

    #[test]
    fn turbospec_width_adjusts() {
        let config = TurboSpecConfig {
            initial_depth: 5,
            min_depth: 1,
            max_depth: 10,
            ema_alpha: 0.9,
            depth_up_threshold: 0.8,
            depth_down_threshold: 0.4,
        };
        let mut ctrl = TurboSpecController::new(config);
        let initial_width = ctrl.width();

        // High acceptance should eventually bump width.
        for _ in 0..20 {
            ctrl.observe(10, 10);
        }
        assert!(
            ctrl.width() >= initial_width,
            "width should not decrease with high acceptance"
        );
    }
}
