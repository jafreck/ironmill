//! Prompt prefix caching via radix-tree KV activation reuse.
//!
//! This module provides two cache implementations:
//!
//! - [`PrefixCache`] — radix-tree backed, suited for multi-user serving
//!   where many requests share prompt prefixes (system prompts, few-shot
//!   examples, etc.).
//! - [`LinearPrefixCache`] — flat-list backed, optimized for single-user
//!   local inference where the conversation history grows incrementally.
//!
//! Both caches store CPU-side copies of KV activations ([`KvCacheSlice`])
//! and support LRU eviction under a configurable memory budget.

pub mod policy;
pub mod prefix_cache;
pub mod radix_tree;

pub use policy::LruPolicy;
pub use prefix_cache::{LinearPrefixCache, PrefixCache};
pub use radix_tree::{KvCacheSlice, KvLayerSlice, RadixNode, RadixTree};
