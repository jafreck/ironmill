//! Inference engine for ironmill (ANE + CoreML backends).
//!
//! This crate consolidates all inference-related code: autoregressive
//! decode loops, KV cache management, token sampling, and the runtime
//! types shared across backends.

#![deny(unsafe_code)]

#[cfg(not(target_os = "macos"))]
compile_error!("ironmill-inference only supports macOS");

pub mod ane;
pub mod coreml;
#[cfg(any(feature = "metal", feature = "mlx"))]
mod dequant;
pub mod engine;
#[cfg(feature = "metal")]
pub mod metal;
#[cfg(feature = "mlx")]
pub mod mlx;
pub mod sampling;
pub mod turboquant;
pub mod types;
#[cfg(any(feature = "metal", feature = "mlx"))]
mod weight_loading;

// Re-exports for convenience.
pub use ane::model::{AneConfig, AneDirectBackend, AneModel, AneRuntimeModel};
pub use engine::{InferenceEngine, InferenceError};
#[cfg(feature = "mlx")]
pub use mlx::{MlxArtifacts, MlxConfig, MlxInference};
pub use sampling::{DEFAULT_EOS_TOKENS, is_eos_token, sample_token};
pub use types::{
    ElementType, InputFeatureDesc, RuntimeBackend, RuntimeModel, RuntimeTensor, build_dummy_inputs,
};

/// CoreML runtime types re-exported for downstream consumers.
///
/// User-facing crates should depend on `ironmill-inference`, not
/// `ironmill-coreml-sys` directly.
pub mod coreml_runtime {
    pub use ironmill_coreml_sys::{
        ComputeUnits, ExtractedOutput, InputDescription, InputFeature, Model, MultiArrayDataType,
        OutputTensorData, PredictionInput, PredictionOutput, build_dummy_input,
    };
}

// ── ANE error type ───────────────────────────────────────────────
// The AneError is shared across the ane submodules and needs to be
// at the crate level since both ane::device and ane::decode use it.

use ironmill_iosurface::IOSurfaceError;

/// Errors from the ANE runtime backend.
#[derive(Debug, thiserror::Error)]
pub enum AneError {
    /// ANE compilation failed with a status code.
    #[error("ANE compilation failed (status {status:#x}): {context}")]
    CompileFailed { status: u32, context: String },

    /// ANE eval failed with a status code.
    #[error("ANE eval failed (status {status:#x}): {context}")]
    EvalFailed { status: u32, context: String },

    /// IOSurface creation or I/O failed.
    #[error("IOSurface error: {0}")]
    SurfaceError(String),

    /// The compile budget (~119 per process) has been exhausted.
    #[error("ANE compile budget exhausted ({used}/{limit} compilations used)")]
    BudgetExhausted { used: usize, limit: usize },

    /// A generic error from an underlying operation.
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

impl From<IOSurfaceError> for AneError {
    fn from(e: IOSurfaceError) -> Self {
        AneError::SurfaceError(e.to_string())
    }
}
