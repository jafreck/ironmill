#![warn(missing_docs)]
//! High-level model loading and inference.
//!
//! `ironmill-torch` provides a PyTorch-level abstraction over the
//! lower-level `ironmill-compile` and `ironmill-inference` crates.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use ironmill_torch::{Model, GenParams};
//!
//! let mut model = Model::from_pretrained("./Qwen3-0.6B/")
//!     .build()?;
//!
//! let output = model.generate("What is Rust?", &GenParams::default())?;
//! println!("{}", output.text);
//! # Ok::<(), ironmill_torch::TorchError>(())
//! ```

mod chat;
mod error;
mod gen_params;
mod model;
mod text_output;

pub use chat::{ChatSession, ChatSessionBuilder};
pub use error::TorchError;
pub use gen_params::GenParams;
pub use model::{Model, ModelBuilder};
pub use text_output::{TextChunk, TextOutput, TextStream};

// Re-export commonly needed types from lower crates so users
// don't need to add them as direct dependencies.
pub use ironmill_core::{ChatMessage, Device, ModelInfo, Tokenizer};
