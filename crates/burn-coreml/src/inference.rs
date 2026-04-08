//! CoreML inference for running exported models on macOS.
//!
//! This module provides [`CoreMlInference`], a lightweight wrapper around
//! the shared [`CoreMlSession`] from `ironmill-inference`.

use std::path::Path;

use ironmill_inference::coreml_runtime::{ComputeUnits, CoreMlError, CoreMlSession};

pub use ironmill_inference::coreml_runtime::SessionInputDesc as InputDesc;
pub use ironmill_inference::coreml_runtime::SessionOutput as InferenceOutput;

/// A CoreML inference session for running exported models.
///
/// # Example
///
/// ```no_run
/// use burn_coreml::inference::CoreMlInference;
/// use ironmill_inference::coreml_runtime::{ComputeUnits, CoreMlError};
///
/// let session = CoreMlInference::load("model.mlmodelc", ComputeUnits::All)?;
/// let inputs = session.input_description()?;
/// println!("model expects {} inputs", inputs.len());
///
/// let outputs = session.predict(&[
///     ("input", &[1, 3, 224, 224], &vec![0.0f32; 1 * 3 * 224 * 224]),
/// ])?;
/// for output in &outputs {
///     println!("{}: shape {:?}, {} values", output.name, output.shape, output.data.len());
/// }
/// # Ok::<(), CoreMlError>(())
/// ```
pub struct CoreMlInference {
    session: CoreMlSession,
}

impl CoreMlInference {
    /// Load a compiled CoreML model (`.mlmodelc`).
    ///
    /// # Arguments
    ///
    /// * `path` — Path to a compiled `.mlmodelc` bundle
    /// * `compute_units` — Which hardware to use for inference
    pub fn load(path: impl AsRef<Path>, compute_units: ComputeUnits) -> Result<Self, CoreMlError> {
        let session = CoreMlSession::load(path.as_ref(), compute_units)?;
        Ok(Self { session })
    }

    /// Get descriptions of the model's expected inputs.
    pub fn input_description(&self) -> Result<Vec<InputDesc>, CoreMlError> {
        self.session.input_description()
    }

    /// Run inference with f32 input tensors.
    ///
    /// Each input is a tuple of `(name, shape, data)`. Returns output tensors
    /// with names, shapes, and f32 data.
    pub fn predict(
        &self,
        inputs: &[(&str, &[usize], &[f32])],
    ) -> Result<Vec<InferenceOutput>, CoreMlError> {
        self.session.predict(inputs)
    }

    /// Run inference and return the raw `PredictionOutput`.
    ///
    /// Use this when you need custom output extraction logic beyond what
    /// [`predict`](Self::predict) provides.
    pub fn predict_raw(
        &self,
        inputs: &[(&str, &[usize], &[f32])],
    ) -> Result<ironmill_inference::coreml_runtime::PredictionOutput, CoreMlError> {
        self.session.predict_raw(inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_nonexistent_path_returns_error() {
        let result = CoreMlInference::load("/nonexistent/model.mlmodelc", ComputeUnits::CpuOnly);
        assert!(result.is_err());
    }
}
