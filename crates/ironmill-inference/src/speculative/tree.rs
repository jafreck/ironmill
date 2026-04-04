//! Tree-structured candidate generation and verification (P-EAGLE).
//!
//! Builds a tree of [`DraftCandidate`]s from draft-head logits and
//! verifies them against target-model probabilities using the
//! Leviathan et al. rejection sampling scheme.

use crate::speculative::config::SpecConfig;

/// A single candidate token in the draft tree.
#[derive(Debug, Clone)]
pub struct DraftCandidate {
    /// Token ID proposed by the draft head.
    pub token_id: u32,
    /// Log-probability assigned by the draft head.
    pub log_prob: f32,
    /// Index of the parent candidate in the tree (or `None` for root-level).
    pub parent_idx: Option<usize>,
    /// Depth of this candidate in the tree (0 = first draft token).
    pub depth: usize,
}

/// A tree of draft candidates produced by the EAGLE-3 draft head.
///
/// Candidates are stored in a flat `Vec` with parent pointers, enabling
/// efficient batch prefill through the target model.
#[derive(Debug)]
pub struct CandidateTree {
    pub candidates: Vec<DraftCandidate>,
}

impl CandidateTree {
    /// Build a candidate tree from draft logits.
    ///
    /// At each depth level, the top-`config.tree_width` tokens (by logit
    /// score) are expanded as children, up to `config.max_draft_depth` levels.
    ///
    /// `draft_logits_fn` is called for each depth level with the parent
    /// token ID and must return logits of length `vocab_size`. For the
    /// root level the parent is the last accepted token.
    pub fn build(
        config: &SpecConfig,
        _vocab_size: usize,
        mut draft_logits_fn: impl FnMut(u32, usize) -> Vec<f32>,
        root_token: u32,
    ) -> Self {
        let mut candidates = Vec::new();

        // Seed: the root_token's children form depth 0.
        let mut frontier: Vec<(u32, Option<usize>, usize)> = vec![(root_token, None, 0)];

        while let Some((parent_token, parent_idx, depth)) = frontier.pop() {
            if depth >= config.max_draft_depth {
                continue;
            }

            let logits = draft_logits_fn(parent_token, depth);
            let top_k = top_k_indices(&logits, config.tree_width);

            let log_probs = log_softmax(&logits);

            for &idx in &top_k {
                let candidate_idx = candidates.len();
                candidates.push(DraftCandidate {
                    token_id: idx as u32,
                    log_prob: log_probs[idx],
                    parent_idx,
                    depth,
                });
                // Only expand the best candidate at each depth to keep
                // the tree manageable (beam-like pruning).
                if idx == top_k[0] {
                    frontier.push((idx as u32, Some(candidate_idx), depth + 1));
                }
            }
        }

        Self { candidates }
    }

    /// Extract the token IDs for batch verification through the target model.
    pub fn token_ids(&self) -> Vec<u32> {
        self.candidates.iter().map(|c| c.token_id).collect()
    }

    /// Trace the path from a candidate back to the root, returning
    /// candidate indices from root to the given candidate (inclusive).
    pub fn path_to_root(&self, candidate_idx: usize) -> Vec<usize> {
        let mut path = Vec::new();
        let mut idx = Some(candidate_idx);
        while let Some(i) = idx {
            path.push(i);
            idx = self.candidates[i].parent_idx;
        }
        path.reverse();
        path
    }

    /// Return the longest accepted prefix using Leviathan et al. rejection
    /// sampling.
    ///
    /// `target_log_probs[i]` is the target model's log-probability for
    /// `candidates[i].token_id` at that position. Acceptance probability
    /// is `min(1, p_target / p_draft)` (computed in log-space).
    ///
    /// Returns the indices of accepted candidates (in tree order along the
    /// best path) and stops at the first rejection.
    pub fn verify(&self, target_log_probs: &[f32], config: &SpecConfig) -> Vec<usize> {
        if self.candidates.is_empty() {
            return Vec::new();
        }

        // Find the longest root-to-leaf path (greedy: follow best child).
        let best_leaf = self.best_leaf_index();
        let path = self.path_to_root(best_leaf);

        let mut accepted = Vec::new();
        for &idx in &path {
            let draft_log_p = self.candidates[idx].log_prob;
            let target_log_p = target_log_probs[idx];

            // Acceptance criterion: tokens below the threshold are rejected
            // outright; otherwise use min(1, p_target / p_draft).
            if target_log_p.exp() < config.acceptance_threshold {
                break;
            }

            // log(p_target / p_draft) = target_log_p - draft_log_p
            let log_ratio = target_log_p - draft_log_p;
            if log_ratio >= 0.0 {
                // p_target >= p_draft → always accept
                accepted.push(idx);
            } else {
                // Accept with probability p_target / p_draft.
                // Use a deterministic threshold for reproducibility in tests.
                let acceptance_prob = log_ratio.exp();
                let r = deterministic_threshold(idx);
                if r < acceptance_prob {
                    accepted.push(idx);
                } else {
                    break;
                }
            }
        }

        accepted
    }

    /// Find the leaf with the highest cumulative log-probability.
    fn best_leaf_index(&self) -> usize {
        if self.candidates.is_empty() {
            return 0;
        }

        // A leaf is a candidate that is not the parent of any other candidate.
        let mut is_parent = vec![false; self.candidates.len()];
        for c in &self.candidates {
            if let Some(p) = c.parent_idx {
                is_parent[p] = true;
            }
        }

        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;
        for (i, _) in self.candidates.iter().enumerate() {
            if is_parent[i] {
                continue;
            }
            let path = self.path_to_root(i);
            let score: f32 = path.iter().map(|&j| self.candidates[j].log_prob).sum();
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        best_idx
    }
}

/// Return the indices of the top-k largest values in `values`.
fn top_k_indices(values: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = values.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
    indexed.iter().take(k).map(|&(i, _)| i).collect()
}

/// Numerically-stable log-softmax.
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp: f32 = logits.iter().map(|&l| (l - max).exp()).sum::<f32>().ln();
    logits.iter().map(|&l| l - max - log_sum_exp).collect()
}

/// Deterministic pseudo-random threshold in [0, 1) seeded by candidate index.
/// Used for reproducible acceptance decisions in tests.
fn deterministic_threshold(seed: usize) -> f32 {
    let mut x = (seed as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x as u32 as f32) / (u32::MAX as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speculative_tree_build_basic() {
        let config = SpecConfig {
            max_draft_depth: 2,
            tree_width: 2,
            acceptance_threshold: 0.1,
        };
        let vocab_size = 4;

        let tree = CandidateTree::build(
            &config,
            vocab_size,
            |_parent, _depth| vec![0.1, 0.9, 0.5, 0.2],
            0,
        );

        // depth 0: 2 candidates from root; depth 1: 2 children of best at depth 0
        assert!(!tree.candidates.is_empty());
        assert!(tree.candidates.len() >= 4); // 2 at depth 0 + 2 at depth 1
    }

    #[test]
    fn speculative_tree_token_ids() {
        let config = SpecConfig {
            max_draft_depth: 1,
            tree_width: 3,
            acceptance_threshold: 0.1,
        };
        let tree = CandidateTree::build(&config, 5, |_, _| vec![0.0, 0.5, 0.3, 0.9, 0.1], 0);

        let ids = tree.token_ids();
        // Top-3 of [0.0, 0.5, 0.3, 0.9, 0.1] are indices 3, 1, 2
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&3));
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
    }

    #[test]
    fn speculative_tree_path_to_root() {
        let config = SpecConfig {
            max_draft_depth: 2,
            tree_width: 1,
            acceptance_threshold: 0.1,
        };
        let tree = CandidateTree::build(&config, 4, |_, _| vec![0.1, 0.9, 0.5, 0.2], 0);

        // With width=1, depth=2: candidate[0] is depth 0, candidate[1] is depth 1
        assert_eq!(tree.candidates.len(), 2);
        let path = tree.path_to_root(1);
        assert_eq!(path, vec![0, 1]);
    }

    #[test]
    fn speculative_tree_verify_all_accepted() {
        let config = SpecConfig {
            max_draft_depth: 2,
            tree_width: 1,
            acceptance_threshold: 0.01,
        };

        let tree = CandidateTree::build(&config, 4, |_, _| vec![0.1, 0.9, 0.5, 0.2], 0);

        // Target log-probs that are higher than draft → always accept.
        let target_log_probs: Vec<f32> = tree
            .candidates
            .iter()
            .map(|c| c.log_prob + 1.0) // target >> draft
            .collect();

        let accepted = tree.verify(&target_log_probs, &config);
        assert_eq!(
            accepted.len(),
            2,
            "all candidates on best path should be accepted"
        );
    }

    #[test]
    fn speculative_tree_verify_threshold_rejection() {
        let config = SpecConfig {
            max_draft_depth: 2,
            tree_width: 1,
            acceptance_threshold: 0.99, // very high threshold
        };

        let tree = CandidateTree::build(&config, 4, |_, _| vec![0.1, 0.2, 0.3, 0.4], 0);

        // Target probs are low → should be rejected by threshold.
        let target_log_probs: Vec<f32> = tree
            .candidates
            .iter()
            .map(|_| (-5.0f32)) // exp(-5) ≈ 0.007, well below 0.99 threshold
            .collect();

        let accepted = tree.verify(&target_log_probs, &config);
        assert!(accepted.is_empty(), "high threshold should reject all");
    }

    #[test]
    fn speculative_tree_empty_on_zero_depth() {
        let config = SpecConfig {
            max_draft_depth: 0,
            tree_width: 3,
            acceptance_threshold: 0.1,
        };
        let tree = CandidateTree::build(&config, 4, |_, _| vec![0.1, 0.9, 0.5, 0.2], 0);
        assert!(tree.candidates.is_empty());
    }

    #[test]
    fn speculative_top_k_indices() {
        let vals = vec![0.1, 0.9, 0.5, 0.2];
        let top2 = top_k_indices(&vals, 2);
        assert_eq!(top2, vec![1, 2]);
    }

    #[test]
    fn speculative_log_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0];
        let lsp = log_softmax(&logits);
        let sum: f32 = lsp.iter().map(|&l| l.exp()).sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax should sum to 1, got {sum}"
        );
    }
}
