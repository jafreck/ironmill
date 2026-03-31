//! CoreML inference runtime (macOS only).
//!
//! Wraps [`ironmill_coreml_sys::Model`] and returns raw f32 data + shapes that
//! callers can convert into candle `Tensor` values:
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

pub use ironmill_coreml_sys::ComputeUnits;
use ironmill_coreml_sys::{Model, MultiArrayDataType, PredictionInput};

/// Output tensor from CoreML inference.
#[derive(Debug, Clone)]
pub struct OutputTensor {
    /// Tensor name from the model.
    pub name: String,
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Flattened f32 data.
    pub data: Vec<f32>,
}

/// Description of a model input.
#[derive(Debug, Clone)]
pub struct InputTensorDesc {
    /// Input feature name.
    pub name: String,
    /// Expected input shape.
    pub shape: Vec<usize>,
}

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
    inner: Model,
}

impl CoreMlModel {
    /// Load a compiled `.mlmodelc` or `.mlpackage`.
    ///
    /// # Errors
    ///
    /// Returns an error if the path does not exist or the model cannot be loaded.
    pub fn load(path: impl AsRef<Path>, compute_units: ComputeUnits) -> anyhow::Result<Self> {
        let model = Model::load(path.as_ref(), compute_units)?;
        Ok(Self { inner: model })
    }

    /// Get the model's input descriptions.
    pub fn input_description(&self) -> anyhow::Result<Vec<InputTensorDesc>> {
        let desc = self.inner.input_description()?;
        Ok(desc
            .features
            .iter()
            .map(|f| InputTensorDesc {
                name: f.name.clone(),
                shape: f.shape.clone(),
            })
            .collect())
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
        let mut pred_input = PredictionInput::new();
        for &(name, shape, data) in inputs {
            pred_input.add_multi_array(name, shape, MultiArrayDataType::Float32, data)?;
        }

        let output = self.inner.predict(&pred_input)?;
        let tensor_data = self.inner.extract_outputs(&output)?;

        Ok(tensor_data
            .into_iter()
            .map(|t| OutputTensor {
                name: t.name,
                shape: t.shape,
                data: t.data,
            })
            .collect())
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
