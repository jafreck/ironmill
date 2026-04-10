//! Metal GPU inference backend.
//!
//! Uses MPS for optimized matrix operations and custom Metal compute
//! shaders for TurboQuant quantization, attention, and element-wise ops.

pub(crate) mod attention;
pub(crate) mod buffers;
pub mod bundle;
pub(crate) mod calibration;
pub mod config;
pub mod dequant;
pub(crate) mod engine;
pub mod error;
pub(crate) mod ffn;
pub(crate) mod gdn;
pub(crate) mod inference;
pub(crate) mod kv_cache;
pub(crate) mod loading;
pub mod mla;
pub mod ops;
pub(crate) mod pipeline;
pub(crate) mod plan;
pub(crate) mod ple;
#[cfg(feature = "profile-metal")]
pub mod profiling;
pub(crate) mod projection;
pub mod turboquant;
pub mod weights;

pub use config::{ClaConfig, GdnModelConfig, Gemma4Config, Gemma4LayerConfig, MetalConfig};
pub use engine::{MetalArtifacts, MetalInference};
pub use error::MetalError;
pub use mla::MlaConfig;
pub use ops::LinearKernelKind;

// Calibration API — import `GpuCalibrationEngine` to access calibration
// methods on `MetalInference` (DAC, block-level alpha search, weight swapping).
pub use calibration::{GpuCalibrationEngine, update_weight_buffer_f16};
