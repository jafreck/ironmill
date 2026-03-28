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
pub use operation::{ComputeUnit, Operation};
pub use pass::Pass;
pub use passes::{
    AttentionFusionPass, CodebookOptimizationPass, ComputeUnitAnnotationPass, ConstantFoldPass,
    ConvBatchNormFusionPass, ConvBatchNormWeightFoldPass, ConvReluFusionPass,
    DeadCodeEliminationPass, ExpertQuantConfig, ExpertQuantStrategy, Fp16QuantizePass, Granularity,
    IdentityEliminationPass, Int8QuantizePass, KvCachePass, LayerScheduleConfig, LayerSchedulePass,
    LayerType, LayoutOptimizationPass, LinearReluFusionPass, MixedPrecisionConfig,
    MixedPrecisionPass, ModelSplitPass, OpPrecision, OpSplittingPass, OpSubstitutionPass,
    PalettizePass, PerExpertQuantPass, ShapeMaterializePass, SplitResult,
};
pub use pipeline::{PassPipeline, PassResult, PipelineConfig, PipelineReport};
pub use program::{Block, Function, Program};
pub use tensor::{ScalarType, TensorType};
pub use types::Value;
