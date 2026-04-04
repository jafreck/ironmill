//! Progress reporting for long-running operations.

use std::time::Duration;

/// Sink for progress updates during long-running operations.
///
/// Implement this trait to receive progress notifications from pipeline
/// execution, model conversion, and other long-running tasks.
///
/// All methods have default no-op implementations, so callers can
/// override only the callbacks they care about.
pub trait ProgressSink: Send + Sync {
    /// Called when a new named stage begins.
    fn on_stage(&self, _name: &str) {}

    /// Called periodically with progress within the current stage.
    fn on_progress(&self, _current: usize, _total: usize, _message: &str) {}

    /// Called when a stage completes, with wall-clock elapsed time.
    fn on_stage_complete(&self, _name: &str, _elapsed: Duration) {}

    /// Called when a non-fatal warning is encountered.
    fn on_warning(&self, _message: &str) {}
}

/// No-op progress sink that silently discards all updates.
pub struct NullProgress;

impl ProgressSink for NullProgress {}
