//! Bidirectional conversion between protobuf types and the Rust-native MIL IR.
//!
//! - [`model_to_program`] converts a protobuf [`Model`](crate::proto::specification::Model) into
//!   a MIL IR [`Program`](crate::ir::Program).
//! - [`program_to_model`] converts a MIL IR [`Program`](crate::ir::Program) back into a protobuf
//!   [`Model`](crate::proto::specification::Model).
//! - [`onnx_to_mil`](crate::convert::onnx_to_mil) converts individual ONNX [`NodeProto`](crate::proto::onnx::NodeProto)
//!   operations into MIL IR [`Operation`](crate::ir::Operation)s.
//! - [`onnx_graph`](crate::convert::onnx_graph) converts an entire ONNX [`ModelProto`](crate::proto::onnx::ModelProto)
//!   into a MIL IR [`Program`](crate::ir::Program).

#[cfg(feature = "ane-direct")]
pub mod ir_to_mil_text;
pub mod ir_to_proto;
pub mod lora;
pub mod moe;
pub mod onnx_graph;
pub mod onnx_to_mil;
pub mod pipeline;
pub mod proto_to_ir;
pub mod weights;

pub use ir_to_proto::{
    LossFunction, UpdatableModelConfig, UpdateOptimizer, program_to_model,
    program_to_multi_function_model, program_to_updatable_model,
};
pub use lora::{
    LoraAdapter, detect_lora_adapters, lora_initializer_names, merge_lora, merge_lora_weights,
};
pub use onnx_graph::{
    ConversionConfig, detect_autoregressive_pattern, onnx_to_program, onnx_to_program_with_config,
};
pub use onnx_to_mil::convert_node;
pub use pipeline::{PipelineManifest, convert_pipeline, parse_pipeline_manifest};
pub use proto_to_ir::model_to_program;
pub use weights::gguf::GgufProvider;
pub use weights::safetensors::SafeTensorsProvider;
pub use weights::{Architecture, ModelConfig, WeightProvider, WeightTensor};
