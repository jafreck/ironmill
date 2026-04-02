//! SpinQuant rotation learning via the Cayley parameterization.
//!
//! The Cayley transform guarantees that learned rotation matrices remain
//! orthogonal throughout optimization, which is the key insight behind
//! SpinQuant's approach to quantization-aware rotation learning.

pub mod cayley;

pub use cayley::{CayleyOptimizer, CayleyRotation};
