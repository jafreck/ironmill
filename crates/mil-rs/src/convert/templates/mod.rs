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

/// Build a MIL IR [`Program`] from a weight provider by dispatching to the
/// appropriate architecture template.
pub fn weights_to_program(provider: &dyn WeightProvider) -> Result<ConversionResult, MilError> {
    match provider.config().architecture {
        Architecture::Llama => llama::build_program(provider),
        Architecture::Qwen => Err(MilError::Validation(
            "Qwen template not yet implemented".into(),
        )),
        Architecture::Gemma => Err(MilError::Validation(
            "Gemma template not yet implemented".into(),
        )),
    }
}
