//! ANE (Apple Neural Engine) direct inference backend.
//!
//! Provides the `AneInference` autoregressive decode loop, the `AneDevice`
//! trait for hardware abstraction, and the high-level [`AneModel`] facade.

pub mod bundle_manifest;
pub mod decode;
#[allow(unsafe_code)]
pub mod device;
pub mod model;
pub mod turboquant;

pub use decode::AneInference;
pub use device::{AneDevice, HardwareAneDevice, HardwareProgram};
pub use model::{
    AneConfig, AneDirectBackend, AneModel, AneRuntimeModel, CompiledArtifacts, SubProgramArtifact,
};

// Re-export error/result from the parent module for local convenience.
pub(crate) use super::AneError;
pub(crate) type Result<T> = std::result::Result<T, AneError>;
