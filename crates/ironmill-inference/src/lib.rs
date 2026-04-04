//! Inference engine for ironmill (ANE + CoreML backends).
//!
//! This crate consolidates all inference-related code: autoregressive
//! decode loops, KV cache management, token sampling, and the runtime
//! types shared across backends.

#![deny(unsafe_code)]
#![warn(missing_docs)]

// Core — always available, any platform
pub mod batch_runner;
pub mod cache;
pub mod calibration;
pub mod engine;
pub mod generate;
pub mod grammar;
pub mod jit;

pub mod memory;
pub mod model_info;
pub mod sampling;
pub mod serving;
pub mod shader_cache;
pub mod speculative;
pub mod turboquant;
pub mod types;

// Platform-specific backends — feature + OS gated
#[cfg(all(feature = "ane", target_os = "macos"))]
pub mod ane;
#[cfg(all(feature = "coreml", target_os = "macos"))]
pub mod coreml;
#[cfg(all(any(feature = "metal", feature = "mlx"), target_os = "macos"))]
mod dequant;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;
#[cfg(all(feature = "mlx", target_os = "macos"))]
pub mod mlx;
#[cfg(all(any(feature = "metal", feature = "mlx"), target_os = "macos"))]
mod weight_loading;

// Re-exports for convenience.
pub use batch_runner::{BatchRunner, BatchRunnerConfig, SchedulingPolicy, SequenceHandle};
#[cfg(all(feature = "ane", target_os = "macos"))]
pub use ane::model::{AneConfig, AneDirectBackend, AneModel, AneRuntimeModel};
pub use cache::{KvCacheSlice, KvLayerSlice, LinearPrefixCache, LruPolicy, PrefixCache, RadixTree};
pub use engine::{
    BatchInferenceEngine, ConstrainedDecoder, InferenceEngine, InferenceError, SequenceId,
    prefill_with_cache,
};
pub use generate::{
    CancellationToken, FinishReason, GenerateError, GenerateEvent, GenerateRequest, GenerateResult,
    TokenStream, generate, generate_with_callback,
};
pub use grammar::{CompiledGrammar, GrammarState, TokenMask};
pub use memory::{KvQuantLevel, MemoryEstimator, MemoryUsage, QuantLevel};
#[cfg(all(feature = "mlx", target_os = "macos"))]
pub use mlx::{MlxArtifacts, MlxConfig, MlxInference};
pub use model_info::ModelInfo;
pub use sampling::{
    DEFAULT_EOS_TOKENS, Sampler, SamplerConfig, apply_token_mask, is_eos_token, sample_token,
};
pub use speculative::{
    DraftCandidate, DraftHead, MsaHeadWeights, SpecConfig, SpeculativeEngine, SpeculativeStreaming,
    StreamingConfig, speculative_decode,
};
pub use types::{
    ElementType, InputFeatureDesc, RuntimeBackend, RuntimeModel, RuntimeTensor, build_dummy_inputs,
};

/// CoreML runtime types re-exported for downstream consumers.
///
/// User-facing crates should depend on `ironmill-inference`, not
/// `ironmill-coreml-sys` directly.
#[cfg(all(feature = "coreml", target_os = "macos"))]
pub mod coreml_runtime {
    pub use ironmill_coreml_sys::{
        ComputeUnits, ExtractedOutput, InputDescription, InputFeature, Model, MultiArrayDataType,
        OutputTensorData, PredictionInput, PredictionOutput, build_dummy_input,
    };

    /// Build a [`PredictionInput`] from named f32 tensor slices.
    ///
    /// This is the common input-building pattern shared by framework bridge
    /// crates (`burn-coreml`, `candle-coreml`). Each entry is
    /// `(name, shape, data)`.
    pub fn build_f32_input(inputs: &[(&str, &[usize], &[f32])]) -> anyhow::Result<PredictionInput> {
        let mut pi = PredictionInput::new()?;
        for &(name, shape, data) in inputs {
            pi.add_multi_array(name, shape, MultiArrayDataType::Float32, data)?;
        }
        Ok(pi)
    }
}

// ── ANE error type ───────────────────────────────────────────────
// The AneError is shared across the ane submodules and needs to be
// at the crate level since both ane::device and ane::decode use it.

#[cfg(all(feature = "ane", target_os = "macos"))]
use ironmill_iosurface::IOSurfaceError;

/// Errors from the ANE runtime backend.
#[cfg(all(feature = "ane", target_os = "macos"))]
#[non_exhaustive]
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

#[cfg(all(feature = "ane", target_os = "macos"))]
impl From<IOSurfaceError> for AneError {
    fn from(e: IOSurfaceError) -> Self {
        AneError::SurfaceError(e.to_string())
    }
}
