//! MLX GPU backend.
//!
//! Provides an FP16 inference backend using Apple's MLX framework with
//! lazy evaluation. All operations build a computation graph that is
//! materialized with a single `eval()` call.

pub mod config;
pub mod error;
pub mod inference;
pub mod weights;

pub use config::MlxConfig;
pub use error::MlxError;
pub use inference::{MlxArtifacts, MlxInference};

#[cfg(test)]
mod kernel_spike;
