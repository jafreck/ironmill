#![forbid(unsafe_code)]

//! Common runtime traits for ironmill backends.
//!
//! **Deprecated:** This crate re-exports types from [`ironmill_inference`].
//! New code should depend on `ironmill-inference` directly.

pub use ironmill_inference::{
    ElementType, InputFeatureDesc, RuntimeBackend, RuntimeModel, RuntimeTensor, build_dummy_inputs,
};
