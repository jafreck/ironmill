//! Backend-agnostic compilation trait and supporting types.
//!
//! [`CompileTarget`] defines the interface that every compilation backend
//! (Metal-direct, CoreML, …) must implement. The trait is object-safe so
//! callers can work with `&dyn CompileTarget`.

use std::path::PathBuf;

use mil_rs::{ComputeUnit, PassPipeline, PipelineReport, ProgressSink};

use crate::error::CompileError;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// A compilation target that transforms weights for a specific runtime.
pub trait CompileTarget {
    /// Name of this compilation target (for logging/diagnostics).
    fn name(&self) -> &str;

    /// Compile a model from source weights to a runtime-ready artifact.
    fn compile(
        &self,
        source: &dyn mil_rs::weights::WeightProvider,
        config: &CompileConfig,
        progress: &dyn ProgressSink,
    ) -> Result<CompileOutput, CompileError>;

    /// Estimate the output artifact size without performing compilation.
    fn estimate_size(
        &self,
        _source: &dyn mil_rs::weights::WeightProvider,
        _config: &CompileConfig,
    ) -> Result<Option<usize>, CompileError> {
        Ok(None) // default: unknown
    }
}

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

/// Configuration for a single compilation run.
// NOTE: `PassPipeline` does not implement `Clone`, so neither does this struct.
#[non_exhaustive]
#[derive(Debug)]
pub struct CompileConfig {
    /// Directory or file path for the output artifact.
    pub output: PathBuf,
    /// Pass pipeline to run before emitting the artifact.
    pub pipeline: PassPipeline,
    /// Target compute unit (ANE, GPU, CPU, Any).
    pub compute_unit: ComputeUnit,
}

/// Result of a successful compilation.
#[non_exhaustive]
#[derive(Debug)]
pub struct CompileOutput {
    /// Path to the compiled artifact on disk.
    pub artifact: PathBuf,
    /// Report from the pass pipeline (timings, pass results).
    pub report: PipelineReport,
    /// Non-fatal warnings emitted during compilation.
    pub warnings: Vec<String>,
    /// Metadata embedded in (or alongside) the artifact.
    pub metadata: ArtifactMetadata,
}

/// Metadata describing a compiled artifact.
#[non_exhaustive]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ArtifactMetadata {
    /// Schema version of the artifact format.
    pub format_version: u32,
    /// ironmill version that produced this artifact.
    pub ironmill_version: String,
    /// Model architecture name (e.g. "llama-3.2-1b").
    pub architecture: String,
    /// Quantization scheme applied (e.g. "q4_k", "f16").
    pub quantization: String,
}
