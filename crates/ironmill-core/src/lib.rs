#![warn(missing_docs)]
//! Shared types for the ironmill workspace.
//!
//! This crate provides bundle manifest schemas, weight provider traits,
//! model configuration types, and shared abstractions used across
//! the ironmill stack.

pub mod ane;
pub mod device;
pub mod gpu;
pub mod model_info;
pub mod tokenizer;
pub mod weights;

pub use device::Device;
pub use model_info::ModelInfo;
#[cfg(feature = "hf-tokenizer")]
pub use tokenizer::HfTokenizer;
pub use tokenizer::{ChatMessage, Tokenizer, TokenizerError};
