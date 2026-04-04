//! Per-sequence state for continuous batching.

use crate::engine::SequenceId;
use crate::serving::pool::KvAllocation;

/// Lifecycle status of an inference sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    /// Prompt tokens are being ingested into the KV cache.
    Prefilling,
    /// Autoregressive token generation is in progress.
    Decoding,
    /// Generation has finished (EOS or caller removed).
    Completed,
}

/// State of a single inference sequence within the batch scheduler.
#[derive(Debug, Clone)]
pub struct SequenceState {
    /// Unique identifier for this sequence.
    pub id: SequenceId,
    /// Token IDs accumulated so far (prompt + generated).
    pub tokens: Vec<u32>,
    /// KV cache allocation for this sequence (snapshot from pool).
    pub kv_allocation: KvAllocation,
    /// Current lifecycle status.
    pub status: SequenceStatus,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serving_sequence_state_construction() {
        let state = SequenceState {
            id: 42,
            tokens: vec![1, 2, 3],
            kv_allocation: KvAllocation {
                offset: 0,
                capacity: 16,
                used: 3,
            },
            status: SequenceStatus::Prefilling,
        };
        assert_eq!(state.id, 42);
        assert_eq!(state.tokens.len(), 3);
        assert_eq!(state.status, SequenceStatus::Prefilling);
    }

    #[test]
    fn serving_sequence_status_transitions() {
        let mut status = SequenceStatus::Prefilling;
        assert_eq!(status, SequenceStatus::Prefilling);
        status = SequenceStatus::Decoding;
        assert_eq!(status, SequenceStatus::Decoding);
        status = SequenceStatus::Completed;
        assert_eq!(status, SequenceStatus::Completed);
    }
}
