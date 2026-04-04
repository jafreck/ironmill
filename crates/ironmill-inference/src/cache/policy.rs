//! Eviction policies for the prompt prefix cache.
//!
//! The primary policy is LRU (least-recently-used) eviction constrained by
//! a memory budget. When the cache exceeds its budget, the least-recently
//! accessed KV slices are evicted until usage drops below the limit.

use crate::cache::radix_tree::EvictionCandidate;

/// Eviction policy trait.
pub(crate) trait EvictionPolicy {
    /// Given a set of eviction candidates and a target number of bytes to
    /// free, return the indices (into `candidates`) to evict, in order.
    fn select_victims(&self, candidates: &[EvictionCandidate], bytes_to_free: usize) -> Vec<usize>;
}

/// Least-recently-used eviction: evict the oldest-accessed entries first.
#[derive(Debug, Default)]
pub struct LruPolicy;

impl EvictionPolicy for LruPolicy {
    fn select_victims(&self, candidates: &[EvictionCandidate], bytes_to_free: usize) -> Vec<usize> {
        if candidates.is_empty() || bytes_to_free == 0 {
            return Vec::new();
        }

        // Sort indices by last_access ascending (oldest first).
        let mut indices: Vec<usize> = (0..candidates.len()).collect();
        indices.sort_by_key(|&i| candidates[i].last_access);

        let mut freed = 0usize;
        let mut victims = Vec::new();
        for idx in indices {
            if freed >= bytes_to_free {
                break;
            }
            freed += candidates[idx].memory;
            victims.push(idx);
        }
        victims
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::{Duration, Instant};

    fn candidate(last_access: Instant, memory: usize) -> EvictionCandidate {
        EvictionCandidate {
            path: Vec::new(),
            last_access,
            memory,
        }
    }

    #[test]
    fn cache_lru_policy_evicts_oldest_first() {
        let policy = LruPolicy;

        let t1 = Instant::now();
        thread::sleep(Duration::from_millis(10));
        let t2 = Instant::now();
        thread::sleep(Duration::from_millis(10));
        let t3 = Instant::now();

        let candidates = vec![
            candidate(t2, 100), // middle
            candidate(t1, 200), // oldest
            candidate(t3, 150), // newest
        ];

        // Need to free 250 bytes: should pick oldest (200) then middle (100) = 300
        let victims = policy.select_victims(&candidates, 250);
        assert_eq!(victims.len(), 2);
        assert_eq!(victims[0], 1); // oldest
        assert_eq!(victims[1], 0); // middle
    }

    #[test]
    fn cache_lru_policy_no_eviction_when_zero_needed() {
        let policy = LruPolicy;
        let candidates = vec![candidate(Instant::now(), 100)];
        let victims = policy.select_victims(&candidates, 0);
        assert!(victims.is_empty());
    }

    #[test]
    fn cache_lru_policy_empty_candidates() {
        let policy = LruPolicy;
        let victims = policy.select_victims(&[], 1000);
        assert!(victims.is_empty());
    }
}
