//! Batch assembly for kernel dispatch.
//!
//! [`InferenceBatch`] collects the per-sequence metadata needed to
//! dispatch a single batched decode step to the compute backend.

use std::collections::HashMap;

use crate::engine::SequenceId;
use crate::serving::pool::KvPool;
use crate::serving::sequence::{SequenceState, SequenceStatus};

/// A prepared batch of sequences for kernel dispatch.
///
/// Contains the per-sequence tokens, KV cache offsets, and sequence
/// lengths needed by the compute kernel. The inference engine converts
/// this into backend-specific dispatch arguments.
#[derive(Debug, Clone)]
pub struct InferenceBatch {
    /// IDs of sequences in this batch.
    pub sequence_ids: Vec<SequenceId>,
    /// Token to process this step for each sequence (last token for
    /// decoding, all tokens for prefilling).
    pub tokens: Vec<Vec<u32>>,
    /// KV cache byte offset for each sequence.
    pub kv_offsets: Vec<usize>,
    /// Current sequence length (tokens in KV cache) for each sequence.
    pub seq_lengths: Vec<usize>,
}

impl InferenceBatch {
    /// Assemble a batch from the selected sequence IDs.
    pub fn assemble(
        ids: &[SequenceId],
        sequences: &HashMap<SequenceId, SequenceState>,
        pool: &KvPool,
    ) -> Self {
        let mut batch = InferenceBatch {
            sequence_ids: Vec::with_capacity(ids.len()),
            tokens: Vec::with_capacity(ids.len()),
            kv_offsets: Vec::with_capacity(ids.len()),
            seq_lengths: Vec::with_capacity(ids.len()),
        };

        for &id in ids {
            let Some(seq) = sequences.get(&id) else {
                continue;
            };
            if seq.status == SequenceStatus::Completed {
                continue;
            }
            let Some(alloc) = pool.get(id) else {
                continue;
            };

            batch.sequence_ids.push(id);
            batch.kv_offsets.push(alloc.offset);
            batch.seq_lengths.push(seq.tokens.len());

            match seq.status {
                SequenceStatus::Prefilling => {
                    batch.tokens.push(seq.tokens.clone());
                }
                SequenceStatus::Decoding => {
                    // Only the last token is needed for autoregressive decode.
                    let last = seq.tokens.last().copied().unwrap_or(0);
                    batch.tokens.push(vec![last]);
                }
                SequenceStatus::Completed | SequenceStatus::Waiting => unreachable!(),
                _ => {}
            }
        }

        batch
    }

    /// Number of sequences in this batch.
    pub fn len(&self) -> usize {
        self.sequence_ids.len()
    }

    /// Whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.sequence_ids.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serving::pool::KvAllocation;

    fn make_state(id: SequenceId, tokens: Vec<u32>, status: SequenceStatus) -> SequenceState {
        SequenceState {
            id,
            tokens: tokens.clone(),
            kv_allocation: KvAllocation {
                offset: 0,
                capacity: tokens.len(),
                used: tokens.len(),
            },
            status,
        }
    }

    #[test]
    fn serving_batch_assemble_decoding() {
        let mut pool = KvPool::new(1024);
        pool.allocate(1, 64).unwrap();
        pool.allocate(2, 64).unwrap();

        let mut seqs = HashMap::new();
        seqs.insert(1, make_state(1, vec![10, 20], SequenceStatus::Decoding));
        seqs.insert(2, make_state(2, vec![30, 40, 50], SequenceStatus::Decoding));

        let batch = InferenceBatch::assemble(&[1, 2], &seqs, &pool);
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.sequence_ids, vec![1, 2]);
        // Decoding sequences emit only the last token.
        assert_eq!(batch.tokens[0], vec![20]);
        assert_eq!(batch.tokens[1], vec![50]);
    }

    #[test]
    fn serving_batch_assemble_prefilling() {
        let mut pool = KvPool::new(1024);
        pool.allocate(1, 64).unwrap();

        let mut seqs = HashMap::new();
        seqs.insert(1, make_state(1, vec![5, 6, 7], SequenceStatus::Prefilling));

        let batch = InferenceBatch::assemble(&[1], &seqs, &pool);
        assert_eq!(batch.len(), 1);
        // Prefilling sequences emit all tokens.
        assert_eq!(batch.tokens[0], vec![5, 6, 7]);
    }

    #[test]
    fn serving_batch_assemble_skips_missing() {
        let pool = KvPool::new(1024);
        let seqs: HashMap<SequenceId, SequenceState> = HashMap::new();

        let batch = InferenceBatch::assemble(&[1, 2], &seqs, &pool);
        assert!(batch.is_empty());
    }

    #[test]
    fn serving_batch_kv_offsets() {
        let mut pool = KvPool::new(1024);
        pool.allocate(1, 128).unwrap();
        pool.allocate(2, 256).unwrap();

        let mut seqs = HashMap::new();
        seqs.insert(1, make_state(1, vec![1], SequenceStatus::Decoding));
        seqs.insert(2, make_state(2, vec![2], SequenceStatus::Decoding));

        let batch = InferenceBatch::assemble(&[1, 2], &seqs, &pool);
        assert_eq!(batch.kv_offsets[0], 0);
        assert_eq!(batch.kv_offsets[1], 128);
    }
}
