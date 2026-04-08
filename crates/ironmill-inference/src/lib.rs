//! Inference engine for ironmill (Metal, ANE, and CoreML backends).
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
#[cfg(all(feature = "metal", target_os = "macos"))]
mod dequant;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod weight_loading;

// Re-exports for convenience.
#[cfg(all(feature = "ane", target_os = "macos"))]
pub use ane::model::{AneConfig, AneDirectBackend, AneModel, AneRuntimeModel};
pub use batch_runner::{BatchRunner, BatchRunnerConfig, SchedulingPolicy, SequenceHandle};
pub use cache::{KvCacheSlice, KvLayerSlice, LinearPrefixCache, LruPolicy, PrefixCache, RadixTree};
pub use engine::{
    BatchInferenceEngine, ConstrainedDecoder, InferenceEngine, InferenceError, SequenceId,
    prefill_with_cache,
};
#[cfg(feature = "async")]
pub use generate::generate_async;
pub use generate::{
    CancellationToken, FinishReason, GenerateError, GenerateEvent, GenerateRequest, GenerateResult,
    TokenStream, generate, generate_with_callback,
};
pub use grammar::{CompiledGrammar, GrammarState, TokenMask};
pub use memory::{KvQuantLevel, MemoryEstimator, MemoryUsage, QuantLevel};
#[allow(deprecated)]
pub use model_info::ModelInfo;
pub use sampling::{
    DEFAULT_EOS_TOKENS, Sampler, SamplerConfig, SamplingError, apply_token_mask, is_eos_token,
    sample_token,
};
pub use speculative::{
    DraftCandidate, DraftHead, MsaHeadWeights, SpecConfig, SpeculativeEngine, SpeculativeStreaming,
    StreamingConfig, speculative_decode,
};
pub use types::{ElementType, InputFeatureDesc, RuntimeBackend, RuntimeModel, RuntimeTensor};

/// **Internal** — CoreML sys-layer types re-exported for sibling crates.
///
/// These re-exports exist so that `burn-coreml` and `candle-coreml` can
/// consume `ironmill_coreml_sys` types without adding a direct dependency on
/// the sys crate.  **External consumers should not rely on this module.**
/// Its contents may change or be removed without a semver bump.
///
/// Prefer the [`ComputeDevice`] abstraction enum (defined below) over
/// [`ComputeUnits`] when writing new code.
#[doc(hidden)]
#[cfg(all(feature = "coreml", target_os = "macos"))]
pub mod coreml_runtime {
    #[doc(hidden)]
    use std::path::Path;

    pub use ironmill_coreml_sys::{
        ComputeUnits, CoreMlError, ExtractedOutput, InputDescription, InputFeature, Model,
        MultiArrayDataType, OutputTensorData, PredictionInput, PredictionOutput, build_dummy_input,
    };

    /// Build a [`PredictionInput`] from named f32 tensor slices.
    ///
    /// This is the common input-building pattern shared by framework bridge
    /// crates (`burn-coreml`, `candle-coreml`). Each entry is
    /// `(name, shape, data)`.
    pub fn build_f32_input(
        inputs: &[(&str, &[usize], &[f32])],
    ) -> Result<PredictionInput, CoreMlError> {
        let mut pi = PredictionInput::new()?;
        for &(name, shape, data) in inputs {
            pi.add_multi_array(name, shape, MultiArrayDataType::Float32, data)?;
        }
        Ok(pi)
    }

    // ── Shared inference session ─────────────────────────────────────

    /// Description of a model input tensor.
    #[derive(Debug, Clone)]
    pub struct SessionInputDesc {
        /// Input feature name.
        pub name: String,
        /// Expected tensor shape.
        pub shape: Vec<usize>,
    }

    /// An output tensor from CoreML inference.
    #[derive(Debug, Clone)]
    pub struct SessionOutput {
        /// Output feature name.
        pub name: String,
        /// Output tensor shape.
        pub shape: Vec<usize>,
        /// Flattened f32 data.
        pub data: Vec<f32>,
    }

    /// Shared CoreML inference session used by framework bridge crates.
    ///
    /// Wraps a loaded [`Model`] and provides load / describe / predict
    /// methods that both `burn-coreml` and `candle-coreml` delegate to.
    pub struct CoreMlSession {
        model: Model,
    }

    impl CoreMlSession {
        /// Load a compiled CoreML model (`.mlmodelc` or `.mlpackage`).
        pub fn load(path: &Path, compute_units: ComputeUnits) -> Result<Self, CoreMlError> {
            let model = Model::load(path, compute_units)?;
            Ok(Self { model })
        }

        /// Get descriptions of the model's expected inputs.
        pub fn input_description(&self) -> Result<Vec<SessionInputDesc>, CoreMlError> {
            let desc = self.model.input_description()?;
            Ok(desc
                .features
                .into_iter()
                .map(|f| SessionInputDesc {
                    name: f.name,
                    shape: f.shape,
                })
                .collect())
        }

        /// Run inference with f32 input tensors.
        ///
        /// Each input is a tuple of `(name, shape, data)`. Returns output
        /// tensors with names, shapes, and f32 data.
        pub fn predict(
            &self,
            inputs: &[(&str, &[usize], &[f32])],
        ) -> Result<Vec<SessionOutput>, CoreMlError> {
            let output = self.predict_raw(inputs)?;
            let extracted = self.model.extract_outputs(&output)?;

            Ok(extracted
                .into_iter()
                .map(|e| SessionOutput {
                    name: e.name,
                    shape: e.shape,
                    data: e.data,
                })
                .collect())
        }

        /// Run inference and return the raw [`PredictionOutput`].
        ///
        /// Use this when you need custom output extraction logic.
        pub fn predict_raw(
            &self,
            inputs: &[(&str, &[usize], &[f32])],
        ) -> Result<PredictionOutput, CoreMlError> {
            let pi = build_f32_input(inputs)?;
            self.model.predict(&pi)
        }
    }
}

// ── Abstraction over CoreML compute-unit selection ───────────────
/// Hardware target for CoreML model execution.
///
/// This is the stable, crate-owned replacement for the sys-layer
/// [`ironmill_coreml_sys::ComputeUnits`] enum.  New code should prefer this
/// type; the sys re-export in [`coreml_runtime`] is `#[doc(hidden)]` and may
/// be removed in a future release.
#[cfg(all(feature = "coreml", target_os = "macos"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputeDevice {
    /// CPU only — safest, always available.
    CpuOnly,
    /// CPU + GPU.
    CpuAndGpu,
    /// CPU + Apple Neural Engine.
    CpuAndNeuralEngine,
    /// All available compute units (CPU, GPU, ANE).
    All,
}

#[cfg(all(feature = "coreml", target_os = "macos"))]
impl ComputeDevice {
    /// Convert to the underlying sys-layer [`ComputeUnits`](ironmill_coreml_sys::ComputeUnits).
    #[doc(hidden)]
    pub fn to_compute_units(self) -> ironmill_coreml_sys::ComputeUnits {
        match self {
            ComputeDevice::CpuOnly => ironmill_coreml_sys::ComputeUnits::CpuOnly,
            ComputeDevice::CpuAndGpu => ironmill_coreml_sys::ComputeUnits::CpuAndGpu,
            ComputeDevice::CpuAndNeuralEngine => {
                ironmill_coreml_sys::ComputeUnits::CpuAndNeuralEngine
            }
            ComputeDevice::All => ironmill_coreml_sys::ComputeUnits::All,
        }
    }
}

#[cfg(all(feature = "coreml", target_os = "macos"))]
impl From<ComputeDevice> for ironmill_coreml_sys::ComputeUnits {
    fn from(d: ComputeDevice) -> Self {
        d.to_compute_units()
    }
}

#[cfg(all(feature = "coreml", target_os = "macos"))]
impl From<ironmill_coreml_sys::ComputeUnits> for ComputeDevice {
    fn from(cu: ironmill_coreml_sys::ComputeUnits) -> Self {
        match cu {
            ironmill_coreml_sys::ComputeUnits::CpuOnly => ComputeDevice::CpuOnly,
            ironmill_coreml_sys::ComputeUnits::CpuAndGpu => ComputeDevice::CpuAndGpu,
            ironmill_coreml_sys::ComputeUnits::CpuAndNeuralEngine => {
                ComputeDevice::CpuAndNeuralEngine
            }
            ironmill_coreml_sys::ComputeUnits::All => ComputeDevice::All,
        }
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
    CompileFailed {
        /// The ANE status code returned by the compiler.
        status: u32,
        /// Human-readable description of the failure.
        context: String,
    },

    /// ANE eval failed with a status code.
    #[error("ANE eval failed (status {status:#x}): {context}")]
    EvalFailed {
        /// The ANE status code returned during evaluation.
        status: u32,
        /// Human-readable description of the failure.
        context: String,
    },

    /// IOSurface creation or I/O failed.
    #[error("IOSurface error: {0}")]
    SurfaceError(#[from] IOSurfaceError),

    /// The compile budget (~119 per process) has been exhausted.
    #[error("ANE compile budget exhausted ({used}/{limit} compilations used)")]
    BudgetExhausted {
        /// Number of compilations already consumed.
        used: usize,
        /// Maximum compilations allowed per process.
        limit: usize,
    },

    /// A generic error from an underlying operation.
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}
