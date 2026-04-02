//! QuIP# (Quantization with Incoherence Processing, Sharp) primitives.
//!
//! This module contains the E8 lattice codebook used for 2-bit vector
//! quantization in the QuIP# scheme.  The codebook is deterministic and
//! mathematically defined — no training data is needed to construct it.
//!
//! # References
//!
//! * Tseng et al., "QuIP#: Even Better LLM Quantization with Hadamard
//!   Incoherence and Lattice Codebooks", 2024.

pub mod e8_lattice;

pub use e8_lattice::E8Codebook;
