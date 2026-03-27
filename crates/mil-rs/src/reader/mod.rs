//! Readers for various CoreML model formats.

pub mod mlmodel;

pub use mlmodel::{print_model_summary, read_mlmodel};
