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

    fn predict(&self, inputs: &[RuntimeTensor]) -> anyhow::Result<Vec<RuntimeTensor>> {
        let mut pi = PredictionInput::new();
        for t in inputs {
            let data_f32: Vec<f32> = match t.dtype {
                ElementType::Float32 => t
                    .data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                _ => vec![0.0f32; t.numel()],
            };
            pi.add_multi_array(&t.name, &t.shape, MultiArrayDataType::Float32, &data_f32)?;
        }
        // Run prediction — output features not yet extracted as RuntimeTensor.
        let _output = self.model.predict(&pi)?;
        // Return empty for now; full output extraction requires iterating
        // over MLFeatureProvider which is model-specific.
        Ok(vec![])
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

    fn load(&self, model_path: &std::path::Path) -> anyhow::Result<Box<dyn RuntimeModel>> {
        let model = Model::load(model_path, self.compute_units)?;
        Ok(Box::new(CoremlRuntimeModel { model }))
    }
}
