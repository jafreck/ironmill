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
pub mod engine;
pub mod sampling;
pub mod types;

// Re-exports for convenience.
#[cfg(feature = "compile")]
pub use ane::model::CompiledArtifacts;
pub use ane::model::{AneConfig, AneDirectBackend, AneModel, AneRuntimeModel, SubProgramArtifact};
pub use engine::{InferenceEngine, InferenceError};
pub use sampling::{is_eos_token, sample_token};
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

#[cfg(feature = "compile")]
impl From<ironmill_compile::ane::AneCompileError> for AneError {
    fn from(e: ironmill_compile::ane::AneCompileError) -> Self {
        use ironmill_compile::ane::AneCompileError;
        match e {
            AneCompileError::CompileFailed { status, context } => {
                AneError::CompileFailed { status, context }
            }
            AneCompileError::EvalFailed { status, context } => {
                AneError::EvalFailed { status, context }
            }
            AneCompileError::SurfaceError(msg) => AneError::SurfaceError(msg),
            AneCompileError::BudgetExhausted { used, limit } => {
                AneError::BudgetExhausted { used, limit }
            }
            AneCompileError::Other(e) => AneError::Other(e),
        }
    }
}

/// Result type alias for ANE operations.
pub type AneResult<T> = std::result::Result<T, AneError>;

// ── Model architecture config ────────────────────────────────────

/// Describes the architecture of a transformer model for inference.
pub struct ModelArchitecture {
    /// EOS token IDs for stopping generation.
    pub eos_tokens: Vec<u32>,
    /// Attention configuration.
    pub attention: AttentionConfig,
}

/// Attention configuration extracted from model architecture.
pub struct AttentionConfig {
    /// Number of query heads.
    pub num_heads: usize,
    /// Number of key/value heads (may differ from num_heads for GQA).
    pub num_kv_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
}

// ── ANE hardware profile ─────────────────────────────────────────

/// Hardware-level constants for the Apple Neural Engine.
pub struct AneHardwareProfile {
    /// Minimum spatial dimension for ANE I/O tensors.
    pub min_io_seq: usize,
    /// Minimum IOSurface allocation size (bytes).
    pub min_surface_bytes: usize,
    /// Maximum compilations per process (~119).
    pub compile_budget: usize,
    /// QoS level for load/unload operations.
    pub qos_level: u32,
}

impl Default for AneHardwareProfile {
    fn default() -> Self {
        Self {
            min_io_seq: 32,
            min_surface_bytes: 16384,
            compile_budget: 119,
            qos_level: 21,
        }
    }
}
