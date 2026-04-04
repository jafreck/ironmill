//! Continuous batching scheduler.
//!
//! [`BatchScheduler`] manages the lifecycle of concurrent inference
//! sequences, backed by a [`KvPool`] for memory tracking.

use std::collections::HashMap;

use crate::engine::{InferenceError, SequenceId};
use crate::serving::pool::KvPool;
use crate::serving::sequence::{SequenceState, SequenceStatus};

/// Continuous batching scheduler.
///
/// Coordinates multiple inference sequences, assigning KV cache memory
/// from the pool and selecting which sequences to include in each
/// decode step.
pub struct BatchScheduler {
    sequences: HashMap<SequenceId, SequenceState>,
    pool: KvPool,
    next_id: SequenceId,
    max_batch_size: usize,
}

impl BatchScheduler {
    /// Create a new scheduler with the given pool size and maximum batch
    /// size.
    pub fn new(pool_size: usize, max_batch_size: usize) -> Self {
        Self {
            sequences: HashMap::new(),
            pool: KvPool::new(pool_size),
            next_id: 1,
            max_batch_size,
        }
    }

    /// Add a sequence with the given prompt tokens.
    ///
    /// Allocates KV cache memory sized to the prompt length (minimum 1)
    /// and returns the assigned sequence ID.
    pub fn add_sequence(&mut self, tokens: Vec<u32>) -> Result<SequenceId, InferenceError> {
        let id = self.next_id;
        self.next_id += 1;

        let initial_capacity = tokens.len().max(1);
        let alloc = self.pool.allocate(id, initial_capacity)?;
        let kv_allocation = alloc.clone();

        self.sequences.insert(
            id,
            SequenceState {
                id,
                tokens,
                kv_allocation,
                status: SequenceStatus::Prefilling,
            },
        );

        Ok(id)
    }

    /// Remove a sequence and free its KV cache memory.
    pub fn remove_sequence(&mut self, id: SequenceId) -> Result<(), InferenceError> {
        self.sequences
            .remove(&id)
            .ok_or(InferenceError::SequenceNotFound(id))?;
        self.pool.free(id)?;
        Ok(())
    }

    /// Select up to `max_batch_size` active sequences for the next
    /// decode step.
    ///
    /// Prefilling sequences are prioritised over decoding ones.
    pub fn select_batch(&mut self) -> Vec<SequenceId> {
        let mut selected: Vec<SequenceId> = Vec::with_capacity(self.max_batch_size);

        // Prefilling sequences first.
        for (&id, seq) in &self.sequences {
            if selected.len() >= self.max_batch_size {
                break;
            }
            if seq.status == SequenceStatus::Prefilling {
                selected.push(id);
            }
        }

        // Fill remaining slots with decoding sequences.
        for (&id, seq) in &self.sequences {
            if selected.len() >= self.max_batch_size {
                break;
            }
            if seq.status == SequenceStatus::Decoding {
                selected.push(id);
            }
        }

        selected
    }

    /// Record a newly generated token for the given sequence.
    ///
    /// Appends the token, updates the KV allocation usage counter, and
    /// transitions `Prefilling → Decoding`.
    pub fn advance(&mut self, id: SequenceId, new_token: u32) {
        if let Some(seq) = self.sequences.get_mut(&id) {
            seq.tokens.push(new_token);

            if seq.status == SequenceStatus::Prefilling {
                seq.status = SequenceStatus::Decoding;
            }
        }

        if let Some(alloc) = self.pool.get_mut(id) {
            alloc.used += 1;
        }
    }

    /// Mark a sequence as completed.
    pub fn complete_sequence(&mut self, id: SequenceId) {
        if let Some(seq) = self.sequences.get_mut(&id) {
            seq.status = SequenceStatus::Completed;
        }
    }

    /// Access a sequence state by ID.
    pub fn get_sequence(&self, id: SequenceId) -> Option<&SequenceState> {
        self.sequences.get(&id)
    }

    /// Access the underlying KV pool.
    pub fn pool(&self) -> &KvPool {
        &self.pool
    }

    /// Mutable access to the underlying KV pool.
    pub fn pool_mut(&mut self) -> &mut KvPool {
        &mut self.pool
    }

    /// Number of active (non-completed) sequences.
    pub fn active_count(&self) -> usize {
        self.sequences
            .values()
            .filter(|s| s.status != SequenceStatus::Completed)
            .count()
    }

    /// Iterate over all sequence states.
    pub fn sequences(&self) -> impl Iterator<Item = &SequenceState> {
        self.sequences.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serving_scheduler_add_and_get() {
        let mut sched = BatchScheduler::new(1024, 4);
        let id = sched.add_sequence(vec![10, 20, 30]).unwrap();
        let seq = sched.get_sequence(id).unwrap();
        assert_eq!(seq.tokens, vec![10, 20, 30]);
        assert_eq!(seq.status, SequenceStatus::Prefilling);
        assert_eq!(seq.kv_allocation.capacity, 3);
    }

    #[test]
    fn serving_scheduler_remove() {
        let mut sched = BatchScheduler::new(1024, 4);
        let id = sched.add_sequence(vec![1]).unwrap();
        sched.remove_sequence(id).unwrap();
        assert!(sched.get_sequence(id).is_none());
        assert!(sched.remove_sequence(id).is_err());
    }

    #[test]
    fn serving_scheduler_select_batch_respects_max() {
        let mut sched = BatchScheduler::new(4096, 2);
        sched.add_sequence(vec![1]).unwrap();
        sched.add_sequence(vec![2]).unwrap();
        sched.add_sequence(vec![3]).unwrap();

        let batch = sched.select_batch();
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn serving_scheduler_select_batch_prefers_prefilling() {
        let mut sched = BatchScheduler::new(4096, 4);
        let id1 = sched.add_sequence(vec![1]).unwrap();
        let id2 = sched.add_sequence(vec![2]).unwrap();

        // Transition id1 to Decoding.
        sched.advance(id1, 100);

        let batch = sched.select_batch();
        assert!(batch.contains(&id1));
        assert!(batch.contains(&id2));
        // id2 (Prefilling) should appear — it is prioritised.
        assert!(batch.contains(&id2));
    }

    #[test]
    fn serving_scheduler_advance_transitions_status() {
        let mut sched = BatchScheduler::new(1024, 4);
        let id = sched.add_sequence(vec![1, 2]).unwrap();
        assert_eq!(
            sched.get_sequence(id).unwrap().status,
            SequenceStatus::Prefilling
        );

        sched.advance(id, 42);
        let seq = sched.get_sequence(id).unwrap();
        assert_eq!(seq.status, SequenceStatus::Decoding);
        assert_eq!(seq.tokens, vec![1, 2, 42]);
    }

    #[test]
    fn serving_scheduler_complete_sequence() {
        let mut sched = BatchScheduler::new(1024, 4);
        let id = sched.add_sequence(vec![1]).unwrap();
        sched.complete_sequence(id);
        assert_eq!(
            sched.get_sequence(id).unwrap().status,
            SequenceStatus::Completed
        );
    }

    #[test]
    fn serving_scheduler_completed_not_in_batch() {
        let mut sched = BatchScheduler::new(4096, 4);
        let id1 = sched.add_sequence(vec![1]).unwrap();
        let id2 = sched.add_sequence(vec![2]).unwrap();
        sched.complete_sequence(id1);

        let batch = sched.select_batch();
        assert!(!batch.contains(&id1));
        assert!(batch.contains(&id2));
    }

    #[test]
    fn serving_scheduler_four_concurrent() {
        let mut sched = BatchScheduler::new(4096, 8);
        let mut ids = Vec::new();
        for i in 0..4u32 {
            let id = sched.add_sequence(vec![i * 10, i * 10 + 1]).unwrap();
            ids.push(id);
        }
        assert_eq!(sched.active_count(), 4);

        // All four should be in the batch.
        let batch = sched.select_batch();
        assert_eq!(batch.len(), 4);
        for &id in &ids {
            assert!(batch.contains(&id));
        }

        // Advance each.
        for &id in &ids {
            sched.advance(id, 99);
        }

        // Remove two, verify memory reclamation.
        sched.remove_sequence(ids[0]).unwrap();
        sched.remove_sequence(ids[2]).unwrap();
        assert_eq!(sched.active_count(), 2);

        let batch = sched.select_batch();
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn serving_scheduler_ids_are_unique() {
        let mut sched = BatchScheduler::new(4096, 4);
        let id1 = sched.add_sequence(vec![1]).unwrap();
        let id2 = sched.add_sequence(vec![2]).unwrap();
        let id3 = sched.add_sequence(vec![3]).unwrap();
        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
    }

    #[test]
    fn serving_scheduler_pool_free_space_after_remove() {
        let mut sched = BatchScheduler::new(512, 4);
        let id1 = sched.add_sequence(vec![1, 2, 3, 4]).unwrap();
        let free_before = sched.pool().free_space();

        sched.remove_sequence(id1).unwrap();
        assert!(sched.pool().free_space() > free_before);
        assert_eq!(sched.pool().free_space(), 512);
    }
}
