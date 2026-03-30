#![forbid(unsafe_code)]

#[cfg(not(target_os = "macos"))]
compile_error!("ironmill-coreml only supports macOS");

// Re-export all types from ironmill-coreml-sys
pub use ironmill_coreml_sys::{
    ComputeUnits, ExtractedOutput, InputDescription, InputFeature, Model, MultiArrayDataType,
    OutputTensorData, PredictionInput, PredictionOutput, build_dummy_input,
};

// Re-export RuntimeBackend / RuntimeModel implementations from ironmill-inference.
pub use ironmill_inference::coreml::runtime::{CoremlBackend, CoremlRuntimeModel};
