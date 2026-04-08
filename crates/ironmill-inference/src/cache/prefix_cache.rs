//! Prompt prefix cache with LRU eviction.
//!
//! [`PrefixCache`] wraps a [`RadixTree`] for serving/multi-user scenarios,
//! while [`LinearPrefixCache`] offers a simpler flat list optimized for
//! single-user local inference.

use std::time::Instant;

use crate::cache::policy::{EvictionPolicy, LruPolicy};
use crate::cache::radix_tree::{KvCacheSlice, RadixTree};
use crate::engine::InferenceError;

// ---------------------------------------------------------------------------
// PrefixCache (radix-tree backed)
// ---------------------------------------------------------------------------

/// Radix-tree prompt cache with memory-budget eviction.
///
/// Best suited for serving scenarios where many concurrent users share
/// prompt prefixes (e.g. system prompts, few-shot examples).
#[derive(Debug)]
pub struct PrefixCache {
    tree: RadixTree,
    memory_budget: usize,
    current_memory: usize,
    policy: LruPolicy,
}

impl PrefixCache {
    /// Create a new prefix cache with the given memory budget (in bytes).
    pub fn new(memory_budget: usize) -> Self {
        Self {
            tree: RadixTree::new(),
            memory_budget,
            current_memory: 0,
            policy: LruPolicy,
        }
    }

    /// Find the longest cached prefix of `tokens`.
    ///
    /// Returns `(matched_len, kv_slices)` where `matched_len` is how many
    /// leading tokens were found in the cache, and `kv_slices` are the
    /// corresponding KV activations in order.
    pub fn lookup(&self, tokens: &[u32]) -> (usize, Vec<&KvCacheSlice>) {
        self.tree.lookup(tokens)
    }

    /// Insert KV cache data for a token sequence.
    ///
    /// If the insertion would exceed the memory budget, LRU eviction runs
    /// first.
    pub fn insert(&mut self, tokens: &[u32], kv_data: KvCacheSlice) -> Result<(), InferenceError> {
        let new_bytes = kv_data.memory_bytes();
        let displaced = self.tree.insert(tokens, kv_data)?;
        self.current_memory = self.current_memory.saturating_sub(displaced);
        self.current_memory += new_bytes;
        self.evict_to_budget();
        Ok(())
    }

    /// Evict least-recently-used entries until memory usage is within budget.
    pub fn evict_to_budget(&mut self) {
        while self.current_memory > self.memory_budget {
            let candidates = self.tree.collect_eviction_candidates();
            if candidates.is_empty() {
                break;
            }
            let overflow = self.current_memory.saturating_sub(self.memory_budget);
            let victims = self.policy.select_victims(&candidates, overflow);
            if victims.is_empty() {
                break;
            }
            for idx in victims {
                let freed = self.tree.remove_kv_at(&candidates[idx].path);
                self.current_memory = self.current_memory.saturating_sub(freed);
            }
        }
    }

    /// Current memory usage in bytes.
    pub fn current_memory(&self) -> usize {
        self.current_memory
    }

    /// Configured memory budget in bytes.
    pub fn memory_budget(&self) -> usize {
        self.memory_budget
    }
}

// ---------------------------------------------------------------------------
// LinearPrefixCache (flat list, single-user)
// ---------------------------------------------------------------------------

/// A simplified flat prefix cache for single-user local inference.
///
/// Stores the last N conversation prefixes as `(Vec<u32>, KvCacheSlice)`
/// entries. On each request, finds the longest matching prefix by linear
/// scan. Evicts oldest entries when the memory budget is exceeded.
#[derive(Debug)]
pub struct LinearPrefixCache {
    entries: Vec<LinearCacheEntry>,
    memory_budget: usize,
    current_memory: usize,
    max_entries: usize,
}

#[derive(Debug)]
struct LinearCacheEntry {
    tokens: Vec<u32>,
    kv: KvCacheSlice,
    last_access: Instant,
}

impl LinearPrefixCache {
    /// Create a new linear cache.
    ///
    /// * `memory_budget` — maximum bytes for cached KV data.
    /// * `max_entries` — maximum number of cached prefixes.
    pub fn new(memory_budget: usize, max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            memory_budget,
            current_memory: 0,
            max_entries,
        }
    }

    /// Find the longest cached prefix of `tokens`.
    ///
    /// Returns `(matched_len, Option<&KvCacheSlice>)`. The `matched_len`
    /// indicates how many leading tokens are covered by the cache hit.
    pub fn lookup(&mut self, tokens: &[u32]) -> (usize, Option<&KvCacheSlice>) {
        let mut best_idx = None;
        let mut best_len = 0usize;

        for (i, entry) in self.entries.iter().enumerate() {
            let common = entry
                .tokens
                .iter()
                .zip(tokens.iter())
                .take_while(|(a, b)| a == b)
                .count();
            // Only count a hit if the *entire* cached prefix matches
            // (i.e. the cached tokens are a prefix of the query).
            if common == entry.tokens.len() && common > best_len {
                best_len = common;
                best_idx = Some(i);
            }
        }

        if let Some(idx) = best_idx {
            self.entries[idx].last_access = Instant::now();
            (best_len, Some(&self.entries[idx].kv))
        } else {
            (0, None)
        }
    }

    /// Insert a prefix and its KV data.
    pub fn insert(&mut self, tokens: Vec<u32>, kv_data: KvCacheSlice) {
        // If an identical prefix already exists, replace it.
        if let Some(pos) = self.entries.iter().position(|e| e.tokens == tokens) {
            let old_mem = self.entries[pos].kv.memory_bytes();
            self.current_memory = self.current_memory.saturating_sub(old_mem);
            self.entries.remove(pos);
        }

        let new_bytes = kv_data.memory_bytes();
        self.current_memory += new_bytes;

        self.entries.push(LinearCacheEntry {
            tokens,
            kv: kv_data,
            last_access: Instant::now(),
        });

        self.evict_to_budget();
    }

    /// Evict oldest entries until both memory and count limits are satisfied.
    pub fn evict_to_budget(&mut self) {
        // Evict by count.
        while self.entries.len() > self.max_entries {
            self.evict_oldest();
        }
        // Evict by memory.
        while self.current_memory > self.memory_budget && !self.entries.is_empty() {
            self.evict_oldest();
        }
    }

    fn evict_oldest(&mut self) {
        if self.entries.is_empty() {
            return;
        }
        let Some(oldest_idx) = self
            .entries
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| e.last_access)
            .map(|(i, _)| i)
        else {
            return;
        };
        let freed = self.entries[oldest_idx].kv.memory_bytes();
        self.entries.remove(oldest_idx);
        self.current_memory = self.current_memory.saturating_sub(freed);
    }

    /// Current memory usage in bytes.
    pub fn current_memory(&self) -> usize {
        self.current_memory
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::radix_tree::{KvCacheSlice, KvLayerSlice};
    use std::thread;
    use std::time::Duration;

    fn make_kv(start_pos: usize, len: usize, bytes_per_layer: usize) -> KvCacheSlice {
        KvCacheSlice {
            layer_data: vec![KvLayerSlice {
                k_data: vec![0xAA; bytes_per_layer],
                v_data: vec![0xBB; bytes_per_layer],
            }],
            start_pos,
            len,
        }
    }

    // ── PrefixCache tests ────────────────────────────────────────

    #[test]
    fn cache_prefix_cache_hit_full_prompt() {
        let mut cache = PrefixCache::new(10_000);
        let tokens: Vec<u32> = (0..1024).collect();
        cache.insert(&tokens, make_kv(0, 1024, 200)).unwrap();

        let (matched, slices) = cache.lookup(&tokens);
        assert_eq!(matched, 1024, "repeated 1024-token prompt should fully hit");
        assert!(!slices.is_empty());
    }

    #[test]
    fn cache_prefix_cache_partial_prefix_hit() {
        let mut cache = PrefixCache::new(100_000);

        let prompt_a: Vec<u32> = (0..1000).collect();
        cache.insert(&prompt_a, make_kv(0, 1000, 200)).unwrap();

        // prompt_b shares 800 tokens with prompt_a, then diverges.
        let mut prompt_b: Vec<u32> = (0..800).collect();
        prompt_b.extend(2000..2200);

        let (matched, _) = cache.lookup(&prompt_b);
        assert_eq!(
            matched, 800,
            "should match the 800-token shared prefix, not the full 1000"
        );
    }

    #[test]
    fn cache_prefix_cache_lru_eviction() {
        // Budget = 500 bytes. Each entry = 200 bytes (1 layer, 100+100).
        let mut cache = PrefixCache::new(500);

        cache.insert(&[1, 2, 3], make_kv(0, 3, 100)).unwrap();
        thread::sleep(Duration::from_millis(10));
        cache.insert(&[4, 5, 6], make_kv(0, 3, 100)).unwrap();
        thread::sleep(Duration::from_millis(10));
        cache.insert(&[7, 8, 9], make_kv(0, 3, 100)).unwrap();

        // All three fit (200 * 3 = 600 > 500), so the oldest should be evicted.
        assert!(
            cache.current_memory() <= 500,
            "memory {} should be <= budget 500",
            cache.current_memory()
        );

        // The first entry's KV data should have been evicted.
        let (_, slices) = cache.lookup(&[1, 2, 3]);
        assert!(
            slices.is_empty(),
            "oldest entry's KV data should have been evicted"
        );
    }

    #[test]
    fn cache_prefix_cache_overwrite_memory_accounting() {
        let mut cache = PrefixCache::new(10_000);
        cache.insert(&[1, 2, 3], make_kv(0, 3, 100)).unwrap(); // 200 bytes
        assert_eq!(cache.current_memory(), 200);

        // Overwrite with larger KV data — memory should reflect only
        // the new entry, not old + new.
        cache.insert(&[1, 2, 3], make_kv(0, 3, 150)).unwrap(); // 300 bytes
        assert_eq!(
            cache.current_memory(),
            300,
            "overwrite should replace old memory, not accumulate"
        );
    }

    // ── LinearPrefixCache tests ──────────────────────────────────

    #[test]
    fn cache_linear_cache_exact_prefix_hit() {
        let mut cache = LinearPrefixCache::new(10_000, 10);
        let tokens: Vec<u32> = (0..1024).collect();
        cache.insert(tokens.clone(), make_kv(0, 1024, 200));

        let (matched, kv) = cache.lookup(&tokens);
        assert_eq!(matched, 1024);
        assert!(kv.is_some());
    }

    #[test]
    fn cache_linear_cache_prefix_match() {
        let mut cache = LinearPrefixCache::new(10_000, 10);
        let prefix: Vec<u32> = (0..800).collect();
        cache.insert(prefix, make_kv(0, 800, 200));

        let query: Vec<u32> = (0..1024).collect();
        let (matched, kv) = cache.lookup(&query);
        assert_eq!(matched, 800, "should match the 800-token cached prefix");
        assert!(kv.is_some());
    }

    #[test]
    fn cache_linear_cache_eviction_by_count() {
        let mut cache = LinearPrefixCache::new(100_000, 2);
        cache.insert(vec![1], make_kv(0, 1, 10));
        thread::sleep(Duration::from_millis(10));
        cache.insert(vec![2], make_kv(0, 1, 10));
        thread::sleep(Duration::from_millis(10));
        cache.insert(vec![3], make_kv(0, 1, 10));

        assert_eq!(cache.len(), 2, "max_entries=2, oldest should be evicted");
        let (m, _) = cache.lookup(&[1]);
        assert_eq!(m, 0, "entry [1] should have been evicted");
    }

    #[test]
    fn cache_linear_cache_eviction_by_memory() {
        // Budget = 300 bytes. Each entry = 200 bytes.
        let mut cache = LinearPrefixCache::new(300, 100);
        cache.insert(vec![1, 2], make_kv(0, 2, 100));
        thread::sleep(Duration::from_millis(10));
        cache.insert(vec![3, 4], make_kv(0, 2, 100));

        assert!(cache.current_memory() <= 300);
        assert_eq!(
            cache.len(),
            1,
            "only one entry should fit in 300-byte budget"
        );
    }

    #[test]
    fn cache_linear_cache_replace_existing() {
        let mut cache = LinearPrefixCache::new(10_000, 10);
        cache.insert(vec![1, 2, 3], make_kv(0, 3, 50));
        cache.insert(vec![1, 2, 3], make_kv(0, 3, 80));
        assert_eq!(cache.len(), 1);
        let (m, kv) = cache.lookup(&[1, 2, 3]);
        assert_eq!(m, 3);
        assert_eq!(kv.unwrap().layer_data[0].k_data.len(), 80);
    }
}
