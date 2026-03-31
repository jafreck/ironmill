//! GPU inference engine implementing the InferenceEngine trait.
//!
//! Full implementation is in the next task — this is the module
//! skeleton to allow the build to pass.

use super::config::GpuConfig;

/// Metal GPU inference engine.
///
/// Implements the full transformer decode pipeline using Metal compute
/// shaders for element-wise ops and MPS for matrix multiplication.
pub struct GpuInference {
    _config: GpuConfig,
}
