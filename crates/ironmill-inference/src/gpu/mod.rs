//! Metal GPU inference backend.
//!
//! Uses MPS for optimized matrix operations and custom Metal compute
//! shaders for TurboQuant quantization, attention, and element-wise ops.

pub mod config;
pub mod error;
pub mod inference;
pub mod ops;
pub mod turboquant;
pub mod weights;

pub use config::GpuConfig;
pub use error::GpuError;
pub use inference::GpuInference;
