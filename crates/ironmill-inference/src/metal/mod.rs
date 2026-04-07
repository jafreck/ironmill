//! Metal GPU inference backend.
//!
//! Uses MPS for optimized matrix operations and custom Metal compute
//! shaders for TurboQuant quantization, attention, and element-wise ops.

pub(crate) mod buffers;
pub mod bundle;
pub mod config;
pub mod dequant;
pub mod error;
pub mod inference;
pub(crate) mod kv_cache;
pub mod mla;
pub mod ops;
pub(crate) mod plan;
pub mod turboquant;
pub mod weights;

pub use config::{ClaConfig, GdnModelConfig, Gemma4Config, Gemma4LayerConfig, MetalConfig};
pub use error::MetalError;
pub use inference::{MetalArtifacts, MetalInference};
pub use mla::MlaConfig;
pub use ops::LinearKernelKind;
