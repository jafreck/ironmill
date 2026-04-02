//! Metal GPU inference backend.
//!
//! Uses MPS for optimized matrix operations and custom Metal compute
//! shaders for TurboQuant quantization, attention, and element-wise ops.

pub mod bundle;
pub mod config;
pub mod dequant;
pub mod error;
pub mod inference;
pub mod ops;
pub mod turboquant;
pub mod weights;

pub use config::MetalConfig;
pub use error::MetalError;
pub use inference::{MetalArtifacts, MetalInference};
