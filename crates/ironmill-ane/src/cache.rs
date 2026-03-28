//! Compiled program cache with disk persistence and LRU eviction.
//!
//! The ~119 per-process compile limit (ANE constraint #5) makes caching
//! essential. This module provides both in-memory deduplication and
//! disk-based persistence to bypass the limit on subsequent runs.

// TODO: Implement ProgramCache — see spec Task 6
