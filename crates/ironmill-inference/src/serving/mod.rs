//! Continuous batching with vAttention-style KV cache memory management.
//!
//! This module provides a [`BatchScheduler`] for managing concurrent
//! inference sequences and a [`KvPool`] that tracks contiguous
//! sub-allocations within a virtual backing buffer (actual Metal buffer
//! operations happen in the inference engine implementation).

#![allow(dead_code)]

pub mod batch;
pub mod pool;
pub mod scheduler;
pub mod sequence;

#[allow(unused_imports)]
pub use batch::InferenceBatch;
#[allow(unused_imports)]
pub use pool::{KvAllocation, KvPool};
#[allow(unused_imports)]
pub use scheduler::BatchScheduler;
#[allow(unused_imports)]
pub use sequence::{SequenceState, SequenceStatus};
