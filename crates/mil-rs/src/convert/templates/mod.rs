//! Architecture templates that construct MIL IR programs from weight tensors.
//!
//! Each template knows how to build a complete MIL [`Program`] for a specific
//! model architecture (LLaMA, Qwen, Gemma, etc.) given a [`WeightProvider`]
//! that supplies the model configuration and weight tensors.

pub mod config;
pub mod llama;

use crate::MilError;
use crate::convert::onnx_graph::ConversionResult;
use crate::convert::weights::{Architecture, WeightProvider};

#[derive(Debug, Clone, Default)]
pub struct TemplateOptions {
    /// When true, emit ANE-optimized ops:
    /// - Conv2d (1×1) instead of Linear for projections
    /// - Decomposed RMSNorm via concat + layer_norm + slice
    /// - Static KV-cache state inputs per layer
    /// - Prefill / decode function split
    pub ane: bool,
}

/// Build a MIL IR [`Program`] from a weight provider by dispatching to the
/// appropriate architecture template.
///
/// Uses default options (no ANE lowering). For ANE-optimized output, use
/// [`weights_to_program_with_options`].
pub fn weights_to_program(provider: &dyn WeightProvider) -> Result<ConversionResult, MilError> {
    weights_to_program_with_options(provider, &TemplateOptions::default())
}

/// Build a MIL IR [`Program`] from a weight provider with explicit template options.
pub fn weights_to_program_with_options(
    provider: &dyn WeightProvider,
    options: &TemplateOptions,
) -> Result<ConversionResult, MilError> {
    match provider.config().architecture {
        Architecture::Llama => llama::build_program(provider, options),
        Architecture::Qwen => Err(MilError::Validation(
            "Qwen template not yet implemented".into(),
        )),
        Architecture::Gemma => Err(MilError::Validation(
            "Gemma template not yet implemented".into(),
        )),
    }
}
