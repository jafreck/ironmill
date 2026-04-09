//! Compiled program cache with disk persistence and LRU eviction.
//!
//! The ~119 per-process compile limit (ANE constraint #5) makes caching
//! essential. This module provides both in-memory deduplication and
//! disk-based persistence to bypass the limit on subsequent runs.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

/// Maximum compilations per process (ANE constraint #5).
pub const MAX_COMPILE_BUDGET: usize = 119;

/// Cache key: hash of MIL text + weight blob.
///
/// Each unique (architecture, weights) pair requires its own compiled
/// program because weights are baked at compile time (constraint #7).
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ProgramKey {
    /// Hash of the MIL text IR for this program.
    pub(crate) mil_text_hash: u64,
    /// Hash of the weight blob baked into this program.
    pub(crate) weight_hash: u64,
}

/// Compiled program cache with disk persistence and LRU eviction.
///
/// Serves two purposes:
/// 1. **In-process dedup** — avoid redundant compilation for identical programs
/// 2. **Disk persistence** — bypass per-process compile limit on subsequent runs
pub struct ProgramCache {
    /// In-memory cache: key → index into entries vec.
    index: HashMap<ProgramKey, usize>,
    /// LRU-ordered entries (most recently used at the end).
    entries: Vec<CacheEntry>,
    /// Optional disk directory for persistent cache.
    disk_dir: Option<PathBuf>,
    /// Maximum entries in memory.
    max_entries: usize,
    /// Number of compilations in this session.
    session_compile_count: usize,
}

struct CacheEntry {
    key: ProgramKey,
    /// Path to cached compiled program on disk (if disk caching enabled).
    disk_path: Option<PathBuf>,
}

impl ProgramCache {
    /// Create a new cache.
    ///
    /// If `disk_dir` is provided and exists, the cache scans it for
    /// previously cached programs and populates the in-memory index.
    pub fn new(disk_dir: Option<PathBuf>, max_entries: usize) -> Self {
        let mut cache = Self {
            index: HashMap::new(),
            entries: Vec::new(),
            disk_dir,
            max_entries,
            session_compile_count: 0,
        };

        // Discover existing cached programs on disk
        if let Some(dir) = cache.disk_dir.clone() {
            if dir.exists() {
                cache.discover_disk_entries(&dir);
            }
        }

        cache
    }

    /// Look up a cached program by key.
    /// Returns the disk path if found (updates LRU order).
    pub fn get(&mut self, key: &ProgramKey) -> Option<&PathBuf> {
        let &idx = self.index.get(key)?;

        // Verify disk path still exists; if not, remove the entry
        if let Some(ref path) = self.entries[idx].disk_path {
            if !path.exists() {
                self.remove_at(idx);
                return None;
            }
        }

        // Move to end (most recently used)
        self.move_to_end(idx);
        let last = self.entries.len() - 1;
        self.entries[last].disk_path.as_ref()
    }

    /// Record a compiled program in the cache.
    ///
    /// If the cache is at capacity, the least-recently-used entry is evicted
    /// first. The caller is responsible for writing compiled artifacts to
    /// `disk_path`.
    pub fn insert(&mut self, key: ProgramKey, disk_path: PathBuf) {
        // If key already exists, update it and move to end
        if let Some(&idx) = self.index.get(&key) {
            self.entries[idx].disk_path = Some(disk_path);
            self.move_to_end(idx);
            return;
        }

        // Evict if at capacity
        if self.entries.len() >= self.max_entries {
            self.evict(1);
        }

        let new_idx = self.entries.len();
        self.index.insert(key.clone(), new_idx);
        self.entries.push(CacheEntry {
            key,
            disk_path: Some(disk_path),
        });
    }

    /// Check if a key exists in the cache.
    pub fn contains(&self, key: &ProgramKey) -> bool {
        self.index.contains_key(key)
    }

    /// Evict the least-recently-used entries (from the front of `entries`).
    pub fn evict(&mut self, count: usize) {
        let to_remove = count.min(self.entries.len());
        if to_remove == 0 {
            return;
        }
        // Remove all evicted keys from the index.
        for entry in &self.entries[..to_remove] {
            self.index.remove(&entry.key);
        }
        // Drain evicted entries in one shot (O(n) shift instead of O(n²)).
        self.entries.drain(..to_remove);
        // Rebuild all indices after the bulk removal.
        for (i, entry) in self.entries.iter().enumerate() {
            self.index.insert(entry.key.clone(), i);
        }
    }

    /// Number of compilations performed this session.
    pub fn session_compile_count(&self) -> usize {
        self.session_compile_count
    }

    /// Increment the session compile count.
    #[allow(dead_code)]
    pub(crate) fn record_compilation(&mut self) {
        self.session_compile_count += 1;
    }

    /// Remaining compile budget (~119 limit).
    #[allow(dead_code)]
    pub(crate) fn remaining_budget(&self) -> usize {
        MAX_COMPILE_BUDGET.saturating_sub(self.session_compile_count)
    }

    /// Number of entries currently cached.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Generate a cache key from MIL text and weight data.
    ///
    /// **Note:** Uses `DefaultHasher`, which is not guaranteed to be stable
    /// across Rust versions. Disk-persisted caches may miss after a toolchain
    /// upgrade. This is acceptable because cache misses only trigger a
    /// recompilation (no correctness impact).
    pub fn make_key(mil_text: &str, weight_data: &[u8]) -> ProgramKey {
        let mut hasher = DefaultHasher::new();
        mil_text.hash(&mut hasher);
        let mil_text_hash = hasher.finish();

        let mut hasher = DefaultHasher::new();
        weight_data.hash(&mut hasher);
        let weight_hash = hasher.finish();

        ProgramKey {
            mil_text_hash,
            weight_hash,
        }
    }

    /// Get the disk path for a given key (for saving compiled artifacts).
    ///
    /// Returns `None` if disk caching is not enabled.
    pub fn disk_path_for(&self, key: &ProgramKey) -> Option<PathBuf> {
        let dir = self.disk_dir.as_ref()?;
        Some(dir.join(format!(
            "mil_{:016x}_w_{:016x}",
            key.mil_text_hash, key.weight_hash
        )))
    }

    /// Move entry at `idx` to the end of the entries vec (most recently used).
    fn move_to_end(&mut self, idx: usize) {
        if idx == self.entries.len() - 1 {
            return;
        }
        let entry = self.entries.remove(idx);
        // All entries after `idx` shifted left by 1
        for (_, i) in self.index.iter_mut() {
            if *i > idx {
                *i -= 1;
            }
        }
        let new_idx = self.entries.len();
        self.index.insert(entry.key.clone(), new_idx);
        self.entries.push(entry);
    }

    /// Remove entry at `idx`, cleaning up the index.
    fn remove_at(&mut self, idx: usize) {
        let entry = self.entries.remove(idx);
        self.index.remove(&entry.key);
        for (_, i) in self.index.iter_mut() {
            if *i > idx {
                *i -= 1;
            }
        }
    }

    /// Scan a disk directory for cached program directories and populate
    /// the in-memory index.
    fn discover_disk_entries(&mut self, dir: &std::path::Path) {
        let read_dir = match std::fs::read_dir(dir) {
            Ok(rd) => rd,
            Err(_) => return,
        };

        for entry in read_dir.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n,
                None => continue,
            };
            if let Some(key) = Self::parse_dir_name(name) {
                if self.entries.len() >= self.max_entries {
                    break;
                }
                let idx = self.entries.len();
                self.index.insert(key.clone(), idx);
                self.entries.push(CacheEntry {
                    key,
                    disk_path: Some(path),
                });
            }
        }
    }

    /// Parse a directory name like `mil_<16hex>_w_<16hex>` into a ProgramKey.
    fn parse_dir_name(name: &str) -> Option<ProgramKey> {
        let rest = name.strip_prefix("mil_")?;
        let (mil_hex, rest) = rest.split_once("_w_")?;
        if mil_hex.len() != 16 || rest.len() != 16 {
            return None;
        }
        let mil_text_hash = u64::from_str_radix(mil_hex, 16).ok()?;
        let weight_hash = u64::from_str_radix(rest, 16).ok()?;
        ProgramKey {
            mil_text_hash,
            weight_hash,
        }
        .into()
    }
}

impl Default for ProgramCache {
    fn default() -> Self {
        Self::new(None, 100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_empty_default() {
        let cache = ProgramCache::default();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn cache_insert_and_get() {
        let mut cache = ProgramCache::new(None, 10);
        let key = ProgramCache::make_key("model_a", b"weights_a");
        let path = PathBuf::from("/fake/path");
        cache.insert(key.clone(), path.clone());
        // get() checks disk_path.exists(), so with a fake path it returns None.
        // Use contains() to verify insertion instead.
        assert!(cache.contains(&key));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn cache_contains() {
        let mut cache = ProgramCache::new(None, 10);
        let key = ProgramCache::make_key("model_b", b"weights_b");
        assert!(!cache.contains(&key));
        cache.insert(key.clone(), PathBuf::from("/fake"));
        assert!(cache.contains(&key));
    }

    #[test]
    fn cache_miss() {
        let mut cache = ProgramCache::new(None, 10);
        let key = ProgramCache::make_key("missing", b"missing");
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn cache_lru_eviction() {
        let mut cache = ProgramCache::new(None, 3);
        let k1 = ProgramCache::make_key("m1", b"w1");
        let k2 = ProgramCache::make_key("m2", b"w2");
        let k3 = ProgramCache::make_key("m3", b"w3");
        let k4 = ProgramCache::make_key("m4", b"w4");

        cache.insert(k1.clone(), PathBuf::from("/p1"));
        cache.insert(k2.clone(), PathBuf::from("/p2"));
        cache.insert(k3.clone(), PathBuf::from("/p3"));
        assert_eq!(cache.len(), 3);

        // Inserting k4 should evict k1 (oldest)
        cache.insert(k4.clone(), PathBuf::from("/p4"));
        assert_eq!(cache.len(), 3);
        assert!(!cache.contains(&k1));
        assert!(cache.contains(&k2));
        assert!(cache.contains(&k3));
        assert!(cache.contains(&k4));
    }

    #[test]
    fn cache_make_key_deterministic() {
        let k1 = ProgramCache::make_key("same_model", b"same_weights");
        let k2 = ProgramCache::make_key("same_model", b"same_weights");
        assert_eq!(k1, k2);
    }

    #[test]
    fn cache_make_key_different() {
        let k1 = ProgramCache::make_key("model_a", b"weights_a");
        let k2 = ProgramCache::make_key("model_b", b"weights_b");
        assert_ne!(k1, k2);

        // Same model text, different weights
        let k3 = ProgramCache::make_key("model_a", b"weights_b");
        assert_ne!(k1, k3);
        assert_eq!(k1.mil_text_hash, k3.mil_text_hash);
        assert_ne!(k1.weight_hash, k3.weight_hash);
    }

    #[test]
    fn cache_budget_tracking() {
        let mut cache = ProgramCache::default();
        assert_eq!(cache.session_compile_count(), 0);
        cache.record_compilation();
        cache.record_compilation();
        assert_eq!(cache.session_compile_count(), 2);
    }

    #[test]
    fn cache_remaining_budget() {
        let mut cache = ProgramCache::default();
        assert_eq!(cache.remaining_budget(), MAX_COMPILE_BUDGET);
        cache.record_compilation();
        assert_eq!(cache.remaining_budget(), MAX_COMPILE_BUDGET - 1);

        // Exhaust budget
        for _ in 0..MAX_COMPILE_BUDGET {
            cache.record_compilation();
        }
        assert_eq!(cache.remaining_budget(), 0);
    }

    #[test]
    fn cache_disk_path_format() {
        let key = ProgramKey {
            mil_text_hash: 0x0123_4567_89ab_cdef,
            weight_hash: 0xfedcba9876543210,
        };
        let cache = ProgramCache::new(Some(PathBuf::from("/cache")), 10);
        let path = cache.disk_path_for(&key).unwrap();
        assert_eq!(
            path,
            PathBuf::from("/cache/mil_0123456789abcdef_w_fedcba9876543210")
        );

        // No disk dir → None
        let cache_no_disk = ProgramCache::new(None, 10);
        assert!(cache_no_disk.disk_path_for(&key).is_none());
    }

    #[test]
    fn cache_disk_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let key = ProgramCache::make_key("persistent_model", b"persistent_weights");

        // Create cache #1, insert entry, write a marker dir to disk
        {
            let mut cache = ProgramCache::new(Some(dir.path().to_path_buf()), 10);
            let disk_path = cache.disk_path_for(&key).unwrap();
            std::fs::create_dir_all(&disk_path).unwrap();
            cache.insert(key.clone(), disk_path);
            assert!(cache.contains(&key));
        }

        // Create cache #2 from the same disk dir — should discover the entry
        {
            let cache = ProgramCache::new(Some(dir.path().to_path_buf()), 10);
            assert!(cache.contains(&key));
            assert_eq!(cache.len(), 1);
        }
    }
}
