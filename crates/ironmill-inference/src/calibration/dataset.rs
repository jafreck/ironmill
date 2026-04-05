//! Calibration dataset loader.
//!
//! Reuses the same pre-tokenized JSON format as [`PerplexityDataset`] in
//! `ironmill-bench`, so existing dataset files work for both perplexity
//! evaluation and calibration-based quantization.

use std::path::Path;

use serde::Deserialize;

/// A pre-tokenized calibration dataset.
///
/// JSON schema (matches `PerplexityDataset`):
/// ```json
/// {
///   "name": "wikitext2-qwen3",
///   "model": "Qwen/Qwen3-0.6B",
///   "vocab_size": 151936,
///   "seq_len": 512,
///   "num_sequences": 128,
///   "eos_token_id": 151643,
///   "sequences": [[101, 2003, ...], ...]
/// }
/// ```
#[derive(Debug, Deserialize)]
pub struct CalibrationDataset {
    /// Dataset name identifier.
    pub name: String,
    /// Model name this dataset was tokenized for.
    pub model: String,
    /// Vocabulary size of the tokenizer.
    pub vocab_size: usize,
    /// Sequence length (number of tokens per sequence).
    pub seq_len: usize,
    /// Total number of sequences in the dataset.
    pub num_sequences: usize,
    /// Optional; present in PerplexityDataset files but unused for calibration.
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    /// Pre-tokenized sequences, each containing `seq_len` token IDs.
    pub sequences: Vec<Vec<u32>>,
}

impl CalibrationDataset {
    /// Load from a pre-tokenized JSON file (same format as `PerplexityDataset`).
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        let ds: Self = serde_json::from_str(&data)?;
        ds.validate()?;
        Ok(ds)
    }

    /// Generate a random calibration dataset for CI/testing.
    ///
    /// Uses a deterministic RNG seeded by `seed` so results are reproducible.
    /// Token IDs are drawn uniformly from `[0, vocab_size)`.
    pub fn random(vocab_size: usize, seq_len: usize, n_sequences: usize, seed: u64) -> Self {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(seed);
        let sequences: Vec<Vec<u32>> = (0..n_sequences)
            .map(|_| {
                (0..seq_len)
                    .map(|_| rng.gen_range(0..vocab_size as u32))
                    .collect()
            })
            .collect();

        Self {
            name: "random".into(),
            model: "synthetic".into(),
            vocab_size,
            seq_len,
            num_sequences: n_sequences,
            eos_token_id: None,
            sequences,
        }
    }

    /// Iterate over batches of token sequences.
    ///
    /// Each yielded slice contains up to `batch_size` sequences.
    /// The last batch may be smaller than `batch_size`.
    pub fn iter_batches(&self, batch_size: usize) -> impl Iterator<Item = &[Vec<u32>]> {
        self.sequences.chunks(batch_size)
    }

    /// Validate internal consistency of the loaded dataset.
    fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.sequences.len() != self.num_sequences {
            return Err(format!(
                "num_sequences mismatch: header says {} but found {} sequences",
                self.num_sequences,
                self.sequences.len()
            )
            .into());
        }
        for (i, seq) in self.sequences.iter().enumerate() {
            if seq.len() != self.seq_len {
                return Err(format!(
                    "sequence {i} has {} tokens, expected {}",
                    seq.len(),
                    self.seq_len
                )
                .into());
            }
            for &tok in seq {
                if (tok as usize) >= self.vocab_size {
                    return Err(format!(
                        "sequence {i} contains token {tok} >= vocab_size {}",
                        self.vocab_size
                    )
                    .into());
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_dataset_dimensions() {
        let ds = CalibrationDataset::random(1000, 64, 8, 42);
        assert_eq!(ds.num_sequences, 8);
        assert_eq!(ds.sequences.len(), 8);
        for seq in &ds.sequences {
            assert_eq!(seq.len(), 64);
            for &tok in seq {
                assert!(tok < 1000, "token {tok} out of range");
            }
        }
    }

    #[test]
    fn random_is_deterministic() {
        let a = CalibrationDataset::random(500, 32, 4, 123);
        let b = CalibrationDataset::random(500, 32, 4, 123);
        assert_eq!(a.sequences, b.sequences);
    }

    #[test]
    fn random_different_seeds_differ() {
        let a = CalibrationDataset::random(500, 32, 4, 1);
        let b = CalibrationDataset::random(500, 32, 4, 2);
        assert_ne!(a.sequences, b.sequences);
    }

    #[test]
    fn iter_batches_exact() {
        let ds = CalibrationDataset::random(100, 8, 6, 0);
        let batches: Vec<_> = ds.iter_batches(3).collect();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[1].len(), 3);
    }

    #[test]
    fn iter_batches_remainder() {
        let ds = CalibrationDataset::random(100, 8, 7, 0);
        let batches: Vec<_> = ds.iter_batches(3).collect();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[1].len(), 3);
        assert_eq!(batches[2].len(), 1);
    }

    #[test]
    fn iter_batches_single() {
        let ds = CalibrationDataset::random(100, 8, 5, 0);
        let batches: Vec<_> = ds.iter_batches(100).collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 5);
    }

    #[test]
    fn deserialize_perplexity_format() {
        let json = r#"{
            "name": "wikitext2",
            "model": "test-model",
            "vocab_size": 100,
            "seq_len": 4,
            "num_sequences": 2,
            "eos_token_id": 0,
            "sequences": [[1,2,3,4],[5,6,7,8]]
        }"#;
        let ds: CalibrationDataset = serde_json::from_str(json).unwrap();
        assert_eq!(ds.name, "wikitext2");
        assert_eq!(ds.vocab_size, 100);
        assert_eq!(ds.eos_token_id, Some(0));
        assert_eq!(ds.sequences.len(), 2);
    }

    #[test]
    fn deserialize_without_eos_token() {
        let json = r#"{
            "name": "test",
            "model": "m",
            "vocab_size": 50,
            "seq_len": 2,
            "num_sequences": 1,
            "sequences": [[10,20]]
        }"#;
        let ds: CalibrationDataset = serde_json::from_str(json).unwrap();
        assert_eq!(ds.eos_token_id, None);
    }

    #[test]
    fn validate_num_sequences_mismatch() {
        let mut ds = CalibrationDataset::random(100, 4, 3, 0);
        ds.num_sequences = 5;
        let err = ds.validate().unwrap_err();
        assert!(err.to_string().contains("num_sequences mismatch"));
    }

    #[test]
    fn validate_seq_len_mismatch() {
        let mut ds = CalibrationDataset::random(100, 4, 2, 0);
        ds.sequences[1] = vec![1, 2]; // wrong length
        let err = ds.validate().unwrap_err();
        assert!(err.to_string().contains("tokens, expected"));
    }

    #[test]
    fn validate_token_out_of_range() {
        let mut ds = CalibrationDataset::random(100, 4, 1, 0);
        ds.sequences[0][2] = 999;
        let err = ds.validate().unwrap_err();
        assert!(err.to_string().contains("vocab_size"));
    }
}
