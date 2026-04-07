//! Efficient bitset mask over a token vocabulary.
//!
//! [`TokenMask`] represents which tokens are allowed at a given decode step
//! during grammar-constrained generation. Internally it stores one bit per
//! token packed into `u64` words for cache-friendly bulk operations.

use std::fmt;

/// Efficient bitset over vocabulary for grammar-constrained generation.
///
/// Each bit represents whether a token is allowed (1) or masked out (0)
/// at the current decode step.
#[derive(Clone, PartialEq, Eq)]
pub struct TokenMask {
    bits: Vec<u64>,
    vocab_size: usize,
}

impl TokenMask {
    /// Create a mask with all tokens allowed.
    pub fn new(vocab_size: usize) -> Self {
        let num_words = vocab_size.div_ceil(64);
        let mut bits = vec![u64::MAX; num_words];
        // Clear unused high bits in the last word.
        let remainder = vocab_size % 64;
        if remainder > 0 && num_words > 0 {
            bits[num_words - 1] = (1u64 << remainder) - 1;
        }
        Self { bits, vocab_size }
    }

    /// Create a mask with no tokens allowed.
    pub fn allow_none(vocab_size: usize) -> Self {
        let num_words = vocab_size.div_ceil(64);
        Self {
            bits: vec![0u64; num_words],
            vocab_size,
        }
    }

    /// Check if a token is allowed.
    #[inline]
    pub fn is_allowed(&self, token_id: usize) -> bool {
        if token_id >= self.vocab_size {
            return false;
        }
        let word = token_id / 64;
        let bit = token_id % 64;
        (self.bits[word] >> bit) & 1 == 1
    }

    /// Set whether a token is allowed.
    #[inline]
    pub fn set_allowed(&mut self, token_id: usize, allowed: bool) {
        if token_id >= self.vocab_size {
            return;
        }
        let word = token_id / 64;
        let bit = token_id % 64;
        if allowed {
            self.bits[word] |= 1u64 << bit;
        } else {
            self.bits[word] &= !(1u64 << bit);
        }
    }

    /// Bitwise AND in-place: keep only tokens allowed in both masks.
    ///
    /// # Panics
    /// Panics if the two masks have different vocabulary sizes.
    pub fn and_inplace(&mut self, other: &TokenMask) {
        assert_eq!(
            self.vocab_size, other.vocab_size,
            "TokenMask::and_inplace: vocab size mismatch ({} vs {})",
            self.vocab_size, other.vocab_size,
        );
        for (a, b) in self.bits.iter_mut().zip(other.bits.iter()) {
            *a &= *b;
        }
    }

    /// Count the number of allowed tokens.
    pub fn count_allowed(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// The vocabulary size this mask covers.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

impl fmt::Debug for TokenMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TokenMask({}/{} allowed)",
            self.count_allowed(),
            self.vocab_size,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grammar_mask_new_all_allowed() {
        let mask = TokenMask::new(100);
        assert_eq!(mask.vocab_size(), 100);
        assert_eq!(mask.count_allowed(), 100);
        for i in 0..100 {
            assert!(mask.is_allowed(i));
        }
        assert!(!mask.is_allowed(100));
    }

    #[test]
    fn grammar_mask_allow_none() {
        let mask = TokenMask::allow_none(200);
        assert_eq!(mask.count_allowed(), 0);
        for i in 0..200 {
            assert!(!mask.is_allowed(i));
        }
    }

    #[test]
    fn grammar_mask_set_allowed() {
        let mut mask = TokenMask::allow_none(128);
        mask.set_allowed(0, true);
        mask.set_allowed(63, true);
        mask.set_allowed(64, true);
        mask.set_allowed(127, true);
        assert_eq!(mask.count_allowed(), 4);
        assert!(mask.is_allowed(0));
        assert!(mask.is_allowed(63));
        assert!(mask.is_allowed(64));
        assert!(mask.is_allowed(127));
        assert!(!mask.is_allowed(1));

        mask.set_allowed(63, false);
        assert!(!mask.is_allowed(63));
        assert_eq!(mask.count_allowed(), 3);
    }

    #[test]
    fn grammar_mask_and_inplace() {
        let mut a = TokenMask::new(64);
        let mut b = TokenMask::new(64);
        a.set_allowed(0, false);
        b.set_allowed(1, false);
        a.and_inplace(&b);
        assert!(!a.is_allowed(0));
        assert!(!a.is_allowed(1));
        assert!(a.is_allowed(2));
        assert_eq!(a.count_allowed(), 62);
    }

    #[test]
    fn grammar_mask_boundary_sizes() {
        // Exact multiple of 64.
        let mask = TokenMask::new(64);
        assert_eq!(mask.count_allowed(), 64);
        assert!(!mask.is_allowed(64));

        // One more than a multiple of 64.
        let mask = TokenMask::new(65);
        assert_eq!(mask.count_allowed(), 65);
        assert!(mask.is_allowed(64));
        assert!(!mask.is_allowed(65));

        // Size zero.
        let mask = TokenMask::new(0);
        assert_eq!(mask.count_allowed(), 0);
    }

    #[test]
    fn grammar_mask_out_of_range_ignored() {
        let mut mask = TokenMask::new(10);
        mask.set_allowed(100, true); // should be no-op
        assert!(!mask.is_allowed(100));
        assert_eq!(mask.count_allowed(), 10);
    }

    #[test]
    fn grammar_mask_debug_format() {
        let mask = TokenMask::new(100);
        let debug = format!("{mask:?}");
        assert!(debug.contains("100/100 allowed"));
    }
}
