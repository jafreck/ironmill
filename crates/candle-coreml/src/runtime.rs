//! CoreML inference runtime (macOS only).
//!
//! Wraps the shared [`CoreMlSession`] from `ironmill-inference` and returns
//! raw f32 data + shapes that callers can convert into candle `Tensor` values.
//!
//! ```ignore
//! use candle_core::{Device, Tensor};
//! use candle_coreml::runtime::{CoreMlModel, OutputTensor};
//!
//! let model = CoreMlModel::load("model.mlmodelc", ComputeUnits::All)?;
//! let outputs = model.predict(&[("input", &[1, 3, 224, 224], &input_data)])?;
//! for out in &outputs {
//!     let tensor = Tensor::from_slice(&out.data, &out.shape, &Device::Cpu)?;
//! }
//! ```

use std::path::Path;

pub use ironmill_inference::coreml_runtime::ComputeUnits;
use ironmill_inference::coreml_runtime::CoreMlSession;

pub use ironmill_inference::coreml_runtime::SessionInputDesc as InputTensorDesc;
pub use ironmill_inference::coreml_runtime::SessionOutput as OutputTensor;

/// A CoreML model wrapper for inference.
///
/// Loads a compiled `.mlmodelc` or `.mlpackage` and runs predictions,
/// returning output data as [`OutputTensor`] structs.
///
/// # Example
///
/// ```no_run
/// use candle_coreml::runtime::{CoreMlModel, ComputeUnits};
///
/// let model = CoreMlModel::load("model.mlmodelc", ComputeUnits::All)?;
/// let desc = model.input_description()?;
/// println!("inputs: {desc:?}");
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct CoreMlModel {
    session: CoreMlSession,
}

impl CoreMlModel {
    /// Load a compiled `.mlmodelc` or `.mlpackage`.
    ///
    /// # Errors
    ///
    /// Returns an error if the path does not exist or the model cannot be loaded.
    pub fn load(path: impl AsRef<Path>, compute_units: ComputeUnits) -> anyhow::Result<Self> {
        let session = CoreMlSession::load(path.as_ref(), compute_units)?;
        Ok(Self { session })
    }

    /// Get the model's input descriptions.
    pub fn input_description(&self) -> anyhow::Result<Vec<InputTensorDesc>> {
        self.session.input_description()
    }

    /// Run inference with named f32 input tensors.
    ///
    /// Each input is a tuple of `(name, shape, data)`. Returns output tensors
    /// with f32 data that can be converted to candle `Tensor` values.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let input_data = vec![0.0f32; 1 * 3 * 224 * 224];
    /// let outputs = model.predict(&[("input", &[1, 3, 224, 224], &input_data)])?;
    /// for out in &outputs {
    ///     println!("{}: shape={:?}, len={}", out.name, out.shape, out.data.len());
    /// }
    /// ```
    pub fn predict(
        &self,
        inputs: &[(&str, &[usize], &[f32])],
    ) -> anyhow::Result<Vec<OutputTensor>> {
        self.session.predict(inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_nonexistent_returns_error() {
        let result = CoreMlModel::load("nonexistent.mlmodelc", ComputeUnits::CpuOnly);
        assert!(result.is_err());
    }
}
