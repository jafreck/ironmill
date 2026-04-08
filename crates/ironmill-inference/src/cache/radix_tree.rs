//! Radix tree for prompt-prefix KV cache lookups.
//!
//! Each path from the root to a leaf encodes a token sequence. Interior
//! nodes may also carry [`KvCacheSlice`] data so that partial prefix
//! matches can still reuse computed KV activations.

use std::cell::Cell;
use std::collections::HashMap;
use std::time::Instant;

use crate::engine::InferenceError;

// ---------------------------------------------------------------------------
// KV cache slice types
// ---------------------------------------------------------------------------

/// KV activations for a single transformer layer.
#[derive(Debug, Clone)]
pub struct KvLayerSlice {
    /// Key projection bytes (CPU copy).
    pub k_data: Vec<u8>,
    /// Value projection bytes (CPU copy).
    pub v_data: Vec<u8>,
}

impl KvLayerSlice {
    /// Total bytes consumed by this layer slice.
    pub fn memory_bytes(&self) -> usize {
        self.k_data.len() + self.v_data.len()
    }
}

/// CPU-side snapshot of KV cache activations across all layers for a
/// contiguous span of tokens.
#[derive(Debug, Clone)]
pub struct KvCacheSlice {
    /// Per-layer key/value data.
    pub layer_data: Vec<KvLayerSlice>,
    /// Starting position in the sequence.
    pub start_pos: usize,
    /// Number of tokens this slice covers.
    pub len: usize,
}

impl KvCacheSlice {
    /// Total bytes consumed by all layers.
    pub fn memory_bytes(&self) -> usize {
        self.layer_data.iter().map(|l| l.memory_bytes()).sum()
    }
}

// ---------------------------------------------------------------------------
// Radix node
// ---------------------------------------------------------------------------

/// A node in the radix tree. Each node stores a span of tokens (the edge
/// label) and optionally the KV cache data computed for that span.
#[derive(Debug)]
pub struct RadixNode {
    /// Children keyed by the *first* token of the child's `token_span`.
    pub children: HashMap<u32, Box<RadixNode>>,
    /// KV cache data for this node's token span (if cached).
    pub kv_slice: Option<KvCacheSlice>,
    /// Timestamp of the last access (for LRU eviction).
    ///
    /// Uses `Cell` for interior mutability so that lookups can update
    /// timestamps without requiring `&mut self` on the whole tree.
    pub last_access: Cell<Instant>,
    /// The token span labelling the edge from the parent to this node.
    pub token_span: Vec<u32>,
}

impl RadixNode {
    /// Create a new empty node with the given token span.
    pub fn new(token_span: Vec<u32>) -> Self {
        Self {
            children: HashMap::new(),
            kv_slice: None,
            last_access: Cell::new(Instant::now()),
            token_span,
        }
    }

    /// Memory consumed by this node's KV data (excludes children).
    pub fn local_memory(&self) -> usize {
        self.kv_slice.as_ref().map_or(0, |s| s.memory_bytes())
    }

    /// Total memory consumed by this subtree (node + all descendants).
    pub fn subtree_memory(&self) -> usize {
        let own = self.local_memory();
        let children: usize = self.children.values().map(|c| c.subtree_memory()).sum();
        own + children
    }
}

// ---------------------------------------------------------------------------
// Radix tree
// ---------------------------------------------------------------------------

/// A radix tree mapping token sequences to [`KvCacheSlice`] entries.
///
/// Supports longest-prefix lookup and insertion with automatic edge
/// splitting when a new sequence shares a partial prefix with an existing
/// edge.
#[derive(Debug)]
pub struct RadixTree {
    pub(crate) root: RadixNode,
}

impl RadixTree {
    /// Create an empty radix tree.
    pub fn new() -> Self {
        Self {
            root: RadixNode::new(Vec::new()),
        }
    }

    /// Find the longest cached prefix of `tokens`.
    ///
    /// Returns `(matched_len, kv_slices)` where `matched_len` is the number
    /// of leading tokens that match the tree structure, and `kv_slices` are
    /// references to the KV data along the matched path (in order).
    ///
    /// A partial edge match counts toward `matched_len` — e.g. if the tree
    /// has an edge covering 1000 tokens and only the first 800 match the
    /// query, `matched_len` will include those 800 tokens and the edge's
    /// KV data (which covers the full 1000 tokens) is included in `slices`.
    /// The caller should use only the first `matched_len` positions of the
    /// returned data.
    ///
    /// If `slices` is empty, there is no cached KV data along the matched
    /// path and the match should be treated as a cache miss.
    pub fn lookup(&self, tokens: &[u32]) -> (usize, Vec<&KvCacheSlice>) {
        let mut slices = Vec::new();
        let mut matched = 0;
        let mut node = &self.root;
        node.last_access.set(Instant::now());

        while matched < tokens.len() {
            let next_token = tokens[matched];
            let child = match node.children.get(&next_token) {
                Some(c) => c,
                None => break,
            };

            let span = &child.token_span;
            let remaining = &tokens[matched..];

            // How many tokens of this edge match?
            let common = span
                .iter()
                .zip(remaining.iter())
                .take_while(|(a, b)| a == b)
                .count();

            if common == 0 {
                break;
            }

            child.last_access.set(Instant::now());
            matched += common;
            if let Some(ref kv) = child.kv_slice {
                slices.push(kv);
            }

            if common < span.len() {
                // Partial edge match — we've counted the matching tokens
                // but cannot descend further into this edge's children.
                break;
            }

            // Full edge match — descend into the child.
            node = child;
        }

        (matched, slices)
    }

    /// Insert KV data for the given token sequence.
    ///
    /// If a prefix already exists in the tree, the new data covers only
    /// the *suffix* not already present. If an edge needs to be split
    /// (partial match), a new interior node is created.
    ///
    /// Returns the number of bytes freed by overwriting an existing node's
    /// KV data (0 if no overwrite occurred).
    pub fn insert(
        &mut self,
        tokens: &[u32],
        kv_data: KvCacheSlice,
    ) -> Result<usize, InferenceError> {
        self.insert_at(&mut 0, tokens, kv_data)
    }

    fn insert_at(
        &mut self,
        pos: &mut usize,
        tokens: &[u32],
        kv_data: KvCacheSlice,
    ) -> Result<usize, InferenceError> {
        let mut node = &mut self.root;
        node.last_access.set(Instant::now());

        while *pos < tokens.len() {
            let next_token = tokens[*pos];

            if let std::collections::hash_map::Entry::Vacant(e) = node.children.entry(next_token) {
                // No child for this token — create a leaf with remaining tokens.
                let span = tokens[*pos..].to_vec();
                let mut leaf = RadixNode::new(span);
                leaf.kv_slice = Some(kv_data);
                leaf.last_access.set(Instant::now());
                e.insert(Box::new(leaf));
                *pos = tokens.len();
                return Ok(0);
            }

            let child = node.children.get_mut(&next_token).ok_or_else(|| {
                InferenceError::runtime(format!(
                    "radix tree inconsistency: missing child for token {next_token}"
                ))
            })?;
            let span_len = child.token_span.len();
            let remaining = &tokens[*pos..];

            let common = child
                .token_span
                .iter()
                .zip(remaining.iter())
                .take_while(|(a, b)| a == b)
                .count();

            if common < span_len {
                // Need to split this edge.
                let old_span = child.token_span.clone();
                let old_kv = child.kv_slice.take();
                let old_children = std::mem::take(&mut child.children);
                let old_last = child.last_access.get();

                // The current child becomes the prefix node.
                child.token_span = old_span[..common].to_vec();
                child.kv_slice = None;
                child.last_access.set(Instant::now());

                // Create a node for the old suffix.
                let suffix_key = old_span[common];
                let mut suffix_node = RadixNode::new(old_span[common..].to_vec());
                suffix_node.kv_slice = old_kv;
                suffix_node.children = old_children;
                suffix_node.last_access.set(old_last);
                child.children.insert(suffix_key, Box::new(suffix_node));

                *pos += common;

                if *pos < tokens.len() {
                    // Insert remaining tokens as a new child of the split node.
                    let new_span = tokens[*pos..].to_vec();
                    let new_key = new_span[0];
                    let mut new_leaf = RadixNode::new(new_span);
                    new_leaf.kv_slice = Some(kv_data);
                    new_leaf.last_access.set(Instant::now());
                    child.children.insert(new_key, Box::new(new_leaf));
                    *pos = tokens.len();
                } else {
                    // The insertion ends exactly at the split point.
                    child.kv_slice = Some(kv_data);
                }
                return Ok(0);
            }

            // Full edge match — advance into child.
            child.last_access.set(Instant::now());
            *pos += span_len;

            // We need to move `node` to point at the child. We re-borrow.
            node = node.children.get_mut(&next_token).ok_or_else(|| {
                InferenceError::runtime(format!(
                    "radix tree inconsistency: missing child for token {next_token}"
                ))
            })?;
        }

        // tokens exactly matched an existing path — update KV data.
        let displaced = node.local_memory();
        node.kv_slice = Some(kv_data);
        Ok(displaced)
    }

    /// Collect all nodes (as mutable references) in the tree via DFS.
    /// Returns `(path_to_node, &mut RadixNode)` pairs for leaf/data nodes.
    pub(crate) fn collect_eviction_candidates(&mut self) -> Vec<EvictionCandidate> {
        let mut candidates = Vec::new();
        Self::collect_recursive(&mut self.root, &mut Vec::new(), &mut candidates);
        candidates
    }

    fn collect_recursive(
        node: &mut RadixNode,
        path: &mut Vec<u32>,
        candidates: &mut Vec<EvictionCandidate>,
    ) {
        if node.kv_slice.is_some() {
            candidates.push(EvictionCandidate {
                path: path.clone(),
                last_access: node.last_access.get(),
                memory: node.local_memory(),
            });
        }
        let keys: Vec<u32> = node.children.keys().copied().collect();
        for key in keys {
            path.push(key);
            if let Some(child) = node.children.get_mut(&key) {
                Self::collect_recursive(child, path, candidates);
            }
            path.pop();
        }
    }

    /// Remove KV data at the node identified by `path` (sequence of child
    /// keys from root). Returns the freed memory.
    pub(crate) fn remove_kv_at(&mut self, path: &[u32]) -> usize {
        let mut node = &mut self.root;
        for &key in path {
            match node.children.get_mut(&key) {
                Some(child) => node = child,
                None => return 0,
            }
        }
        let freed = node.local_memory();
        node.kv_slice = None;
        freed
    }
}

impl Default for RadixTree {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata for an eviction candidate (avoids holding mutable borrows).
#[derive(Debug)]
pub(crate) struct EvictionCandidate {
    pub path: Vec<u32>,
    pub last_access: Instant,
    pub memory: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn cache_radix_tree_insert_and_lookup() {
        let mut tree = RadixTree::new();

        // Insert [1, 2, 3, 4, 5]
        tree.insert(&[1, 2, 3, 4, 5], make_kv(0, 5, 100)).unwrap();

        // Exact match
        let (matched, slices) = tree.lookup(&[1, 2, 3, 4, 5]);
        assert_eq!(matched, 5);
        assert_eq!(slices.len(), 1);
        assert_eq!(slices[0].len, 5);

        // Prefix match
        let (matched, _) = tree.lookup(&[1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(matched, 5);

        // No match
        let (matched, _) = tree.lookup(&[9, 8, 7]);
        assert_eq!(matched, 0);
    }

    #[test]
    fn cache_radix_tree_shared_prefix() {
        let mut tree = RadixTree::new();

        // Insert [1, 2, 3, 4, 5]
        tree.insert(&[1, 2, 3, 4, 5], make_kv(0, 5, 100)).unwrap();

        // Insert [1, 2, 3, 6, 7] — shares 3-token prefix
        tree.insert(&[1, 2, 3, 6, 7], make_kv(0, 5, 100)).unwrap();

        // Lookup the first sequence
        let (matched, _) = tree.lookup(&[1, 2, 3, 4, 5]);
        assert_eq!(matched, 5);

        // Lookup the second sequence
        let (matched, _) = tree.lookup(&[1, 2, 3, 6, 7]);
        assert_eq!(matched, 5);

        // Lookup shared prefix only
        let (matched, _) = tree.lookup(&[1, 2, 3, 99, 99]);
        assert_eq!(matched, 3, "should match the 3-token shared prefix");
    }

    #[test]
    fn cache_radix_tree_edge_split() {
        let mut tree = RadixTree::new();

        tree.insert(&[10, 20, 30, 40], make_kv(0, 4, 50)).unwrap();
        tree.insert(&[10, 20, 50, 60], make_kv(0, 4, 50)).unwrap();

        let (m1, _) = tree.lookup(&[10, 20, 30, 40]);
        assert_eq!(m1, 4);

        let (m2, _) = tree.lookup(&[10, 20, 50, 60]);
        assert_eq!(m2, 4);

        // Partial prefix
        let (m3, _) = tree.lookup(&[10, 20, 99]);
        assert_eq!(m3, 2, "should match the 2-token shared prefix");
    }

    #[test]
    fn cache_radix_tree_memory_tracking() {
        let mut tree = RadixTree::new();
        tree.insert(&[1, 2, 3], make_kv(0, 3, 100)).unwrap();
        // 1 layer, 100 bytes k + 100 bytes v = 200
        assert_eq!(tree.root.subtree_memory(), 200);
    }

    #[test]
    fn cache_radix_tree_overwrite_existing() {
        let mut tree = RadixTree::new();
        tree.insert(&[1, 2, 3], make_kv(0, 3, 100)).unwrap();
        tree.insert(&[1, 2, 3], make_kv(0, 3, 200)).unwrap();
        let (matched, slices) = tree.lookup(&[1, 2, 3]);
        assert_eq!(matched, 3);
        assert_eq!(slices[0].layer_data[0].k_data.len(), 200);
    }
}
