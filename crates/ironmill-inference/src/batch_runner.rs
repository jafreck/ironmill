//! Managed batch inference loop (§4.14).
//!
//! [`BatchRunner`] wraps a [`BatchInferenceEngine`] and drives
//! continuous-batching generation with configurable scheduling.

use crate::engine::{BatchInferenceEngine, InferenceError};
use crate::generate::{GenerateEvent, GenerateRequest};

/// Handle to a submitted generation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SequenceHandle(u64);

/// Scheduling policy for the batch runner.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// First-come, first-served.
    Fcfs,
    /// Shortest sequence first.
    ShortestFirst,
}

/// Configuration for the batch runner.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct BatchRunnerConfig {
    /// Maximum KV cache pool size in bytes.
    pub kv_pool_size: usize,
    /// Maximum number of concurrent sequences.
    pub max_batch_size: usize,
    /// Scheduling policy.
    pub policy: SchedulingPolicy,
}

impl Default for BatchRunnerConfig {
    fn default() -> Self {
        Self {
            kv_pool_size: 1024 * 1024 * 1024, // 1 GB
            max_batch_size: 8,
            policy: SchedulingPolicy::Fcfs,
        }
    }
}

/// Managed batch inference loop.
pub struct BatchRunner {
    _engine: Box<dyn BatchInferenceEngine>,
    _config: BatchRunnerConfig,
    _next_handle: u64,
}

impl BatchRunner {
    /// Create a new batch runner with the given engine and configuration.
    pub fn new(engine: Box<dyn BatchInferenceEngine>, config: BatchRunnerConfig) -> Self {
        Self {
            _engine: engine,
            _config: config,
            _next_handle: 0,
        }
    }

    /// Submit a generation request and receive a handle for tracking.
    pub fn submit(&mut self, _request: GenerateRequest) -> Result<SequenceHandle, InferenceError> {
        Err(InferenceError::Other(anyhow::anyhow!(
            "BatchRunner not yet implemented"
        )))
    }

    /// Advance the batch by one decode step, returning events for each active sequence.
    pub fn step(&mut self) -> Result<Vec<(SequenceHandle, GenerateEvent)>, InferenceError> {
        Err(InferenceError::Other(anyhow::anyhow!(
            "BatchRunner not yet implemented"
        )))
    }

    /// Cancel a previously submitted sequence.
    pub fn cancel(&mut self, _handle: SequenceHandle) {
        // BatchRunner not yet implemented; cancel is a no-op.
    }

    /// Returns `true` if there are pending or active sequences.
    pub fn has_pending(&self) -> bool {
        false
    }

    /// Returns the number of currently active sequences.
    pub fn active_count(&self) -> usize {
        0
    }
}
