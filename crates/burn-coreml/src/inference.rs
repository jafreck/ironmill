//! CoreML inference for running exported models on macOS.
//!
//! This module provides [`CoreMlInference`], a lightweight wrapper around
//! the CoreML runtime for running exported Burn models.

use std::path::Path;

use ironmill_coreml_sys::{ComputeUnits, Model, MultiArrayDataType, PredictionInput};

/// A CoreML inference session for running exported models.
///
/// # Example
///
/// ```no_run
/// use burn_coreml::inference::CoreMlInference;
/// use ironmill_coreml_sys::ComputeUnits;
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
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct CoreMlInference {
    model: Model,
}

/// Description of a model input tensor.
#[derive(Debug, Clone)]
pub struct InputDesc {
    /// Input feature name.
    pub name: String,
    /// Expected tensor shape.
    pub shape: Vec<usize>,
}

/// An output tensor from inference.
#[derive(Debug, Clone)]
pub struct InferenceOutput {
    /// Output feature name.
    pub name: String,
    /// Output tensor shape.
    pub shape: Vec<usize>,
    /// Output data as f32 values.
    pub data: Vec<f32>,
}

impl CoreMlInference {
    /// Load a compiled CoreML model (`.mlmodelc`).
    ///
    /// # Arguments
    ///
    /// * `path` — Path to a compiled `.mlmodelc` bundle
    /// * `compute_units` — Which hardware to use for inference
    pub fn load(path: impl AsRef<Path>, compute_units: ComputeUnits) -> anyhow::Result<Self> {
        let model = Model::load(path.as_ref(), compute_units)?;
        Ok(Self { model })
    }

    /// Get descriptions of the model's expected inputs.
    pub fn input_description(&self) -> anyhow::Result<Vec<InputDesc>> {
        let desc = self.model.input_description()?;
        Ok(desc
            .features
            .into_iter()
            .map(|f| InputDesc {
                name: f.name,
                shape: f.shape,
            })
            .collect())
    }

    /// Run inference with f32 input tensors.
    ///
    /// Each input is a tuple of `(name, shape, data)`. Returns output tensors
    /// with names, shapes, and f32 data.
    pub fn predict(
        &self,
        inputs: &[(&str, &[usize], &[f32])],
    ) -> anyhow::Result<Vec<InferenceOutput>> {
        let output = self.predict_raw(inputs)?;
        let extracted = output.extract_multi_arrays()?;

        Ok(extracted
            .into_iter()
            .map(|e| InferenceOutput {
                name: e.name,
                shape: e.shape,
                data: e.data,
            })
            .collect())
    }

    /// Run inference and return the raw `PredictionOutput`.
    ///
    /// Use this when you need custom output extraction logic beyond what
    /// [`predict`](Self::predict) provides.
    pub fn predict_raw(
        &self,
        inputs: &[(&str, &[usize], &[f32])],
    ) -> anyhow::Result<ironmill_coreml_sys::PredictionOutput> {
        let mut pi = PredictionInput::new();
        for &(name, shape, data) in inputs {
            pi.add_multi_array(name, shape, MultiArrayDataType::Float32, data)?;
        }
        self.model.predict(&pi)
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
