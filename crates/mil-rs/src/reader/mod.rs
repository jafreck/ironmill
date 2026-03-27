//! Readers for various CoreML model formats.

pub mod mlmodel;
pub mod mlpackage;

pub use mlmodel::{print_model_summary, read_mlmodel};
pub use mlpackage::read_mlpackage;
