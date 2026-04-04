//! Continuous batching with vAttention-style KV cache memory management.
//!
//! This module provides a [`BatchScheduler`] for managing concurrent
//! inference sequences and a [`KvPool`] that tracks contiguous
//! sub-allocations within a virtual backing buffer (actual Metal buffer
//! operations happen in the inference engine implementation).

pub mod batch;
pub mod pool;
pub mod scheduler;
pub mod sequence;

pub use batch::InferenceBatch;
pub use pool::{KvAllocation, KvPool};
pub use scheduler::BatchScheduler;
pub use sequence::{SequenceState, SequenceStatus};
