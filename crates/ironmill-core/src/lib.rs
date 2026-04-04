#![warn(missing_docs)]
//! Shared types for the ironmill workspace.
//!
//! This crate provides bundle manifest schemas, weight provider traits,
//! model configuration types, and the high-level Model API used across
//! the ironmill stack.

pub mod ane;
pub mod chat;
pub mod device;
pub mod error;
pub mod gen_params;
pub mod gpu;
pub mod model;
pub mod text_output;
pub mod tokenizer;
pub mod weights;

pub use chat::ChatSession;
pub use device::Device;
pub use error::ModelError;
pub use gen_params::GenParams;
pub use model::{Model, ModelBuilder};
pub use text_output::{TextChunk, TextOutput};
pub use tokenizer::{ChatMessage, Tokenizer, TokenizerError};
