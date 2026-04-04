//! CoreML inference backend — wraps [`ironmill_coreml_sys`] as a [`RuntimeBackend`].

#[cfg(not(target_os = "macos"))]
compile_error!("CoreML runtime only supports macOS");

use ironmill_coreml_sys::{
    ComputeUnits, InputDescription, Model, MultiArrayDataType, PredictionInput,
};

use crate::types::{ElementType, InputFeatureDesc, RuntimeBackend, RuntimeModel, RuntimeTensor};

fn multi_array_dtype_to_element(dt: MultiArrayDataType) -> ElementType {
    match dt {
        MultiArrayDataType::Float32 => ElementType::Float32,
        MultiArrayDataType::Float16 => ElementType::Float16,
        MultiArrayDataType::Int32 => ElementType::Int32,
        MultiArrayDataType::Double => ElementType::Float64,
    }
}

/// Wraps a loaded CoreML [`Model`] to implement [`RuntimeModel`].
pub struct CoremlRuntimeModel {
    model: Model,
}

impl RuntimeModel for CoremlRuntimeModel {
    fn input_description(&self) -> Vec<InputFeatureDesc> {
        let desc = self
            .model
            .input_description()
            .unwrap_or(InputDescription { features: vec![] });
        desc.features
            .iter()
            .map(|f| InputFeatureDesc {
                name: f.name.clone(),
                shape: f.shape.clone(),
                dtype: multi_array_dtype_to_element(f.data_type),
            })
            .collect()
    }

    fn predict(
        &self,
        inputs: &[RuntimeTensor],
    ) -> Result<Vec<RuntimeTensor>, crate::engine::InferenceError> {
        let mut pi =
            PredictionInput::new().map_err(|e| crate::engine::InferenceError::Runtime(e.into()))?;
        for t in inputs {
            let data_f32: Vec<f32> = match t.dtype {
                ElementType::Float32 => t
                    .data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                other => {
                    return Err(crate::engine::InferenceError::runtime(format!(
                        "unsupported input dtype {:?} for CoreML prediction — only Float32 is supported",
                        other
                    )));
                }
            };
            pi.add_multi_array(&t.name, &t.shape, MultiArrayDataType::Float32, &data_f32)
                .map_err(|e| crate::engine::InferenceError::Runtime(e.into()))?;
        }
        // Run prediction and extract output features as RuntimeTensor values.
        let output = self
            .model
            .predict(&pi)
            .map_err(|e| crate::engine::InferenceError::Runtime(e.into()))?;

        // Extract multi-array outputs from the prediction result.
        let extracted = self
            .model
            .extract_outputs(&output)
            .map_err(|e| crate::engine::InferenceError::Runtime(e.into()))?;

        let results = extracted
            .into_iter()
            .map(|out| {
                // OutputTensorData contains f32 data; convert to raw bytes.
                let raw_bytes: Vec<u8> = out.data.iter().flat_map(|v| v.to_le_bytes()).collect();
                RuntimeTensor {
                    name: out.name,
                    data: raw_bytes,
                    shape: out.shape,
                    dtype: ElementType::Float32,
                }
            })
            .collect();

        Ok(results)
    }
}

/// CoreML backend that loads compiled `.mlmodelc` bundles.
pub struct CoremlBackend {
    pub compute_units: ComputeUnits,
}

impl CoremlBackend {
    pub fn new(compute_units: ComputeUnits) -> Self {
        Self { compute_units }
    }
}

impl RuntimeBackend for CoremlBackend {
    fn name(&self) -> &str {
        "coreml"
    }

    fn load(
        &self,
        model_path: &std::path::Path,
    ) -> Result<Box<dyn RuntimeModel>, crate::engine::InferenceError> {
        let model = Model::load(model_path, self.compute_units)
            .map_err(|e| crate::engine::InferenceError::Runtime(e.into()))?;
        Ok(Box::new(CoremlRuntimeModel { model }))
    }
}
