//! The MIL Intermediate Representation.
//!
//! This module defines the core graph-based IR that sits between model input formats
//! (ONNX, SafeTensors, GGUF) and CoreML's protobuf output format.
//!
//! The design mirrors Apple's Model Intermediate Language (MIL) as documented in
//! `coremltools`, but implemented as idiomatic Rust with strong typing.

mod graph;
mod operation;
mod pass;
pub mod passes;
mod pipeline;
mod program;
mod tensor;
mod types;

pub use graph::Graph;
pub use operation::Operation;
pub use pass::Pass;
pub use passes::{
    AttentionFusionPass, ConstantFoldPass, ConvBatchNormFusionPass, ConvBatchNormWeightFoldPass,
    ConvReluFusionPass, DeadCodeEliminationPass, Fp16QuantizePass, Granularity,
    IdentityEliminationPass, Int8QuantizePass, LayoutOptimizationPass, LinearReluFusionPass,
    OpSubstitutionPass, PalettizePass, ShapeMaterializePass,
};
pub use pipeline::{PassPipeline, PassResult, PipelineReport};
pub use program::{Block, Function, Program};
pub use tensor::{ScalarType, TensorType};
pub use types::Value;
