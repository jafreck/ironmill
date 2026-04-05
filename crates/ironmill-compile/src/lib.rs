//! Compilation pipeline for ironmill.
//!
//! This crate brings together ANE lowering passes, CoreML build-time compilation,
//! architecture templates, and weight providers. It depends on [`mil_rs`] for the
//! core MIL IR but owns all backend-specific compilation logic.

#![deny(unsafe_code)]
#![warn(missing_docs)]

/// Backend-agnostic compilation trait and supporting types.
pub mod compile_target;
/// Compilation error types.
pub mod error;

/// ANE (Apple Neural Engine) compilation: passes, splitting, packing, validation, and caching.
pub mod ane;
/// Model conversion utilities: LoRA merging, MoE splitting, and pipeline orchestration.
pub mod convert;
/// CoreML compilation via `xcrun coremlcompiler` and the high-level build API.
pub mod coreml;
/// GPU-targeted compilation: PolarQuant passes → MilWeightProvider.
pub mod gpu;
/// Architecture templates that construct MIL IR programs from weight tensors.
pub mod templates;
/// Weight providers for SafeTensors and GGUF formats.
pub mod weights;

/// Re-exports from [`mil_rs`] for downstream consumers.
///
/// User-facing crates should depend on `ironmill-compile`, not `mil-rs`
/// directly. This module surfaces the subset of the MIL API needed for
/// model I/O, conversion, pass pipelines, and IR inspection.
pub mod mil {
    // Core IR types
    pub use mil_rs::ir::{
        Block, ComputeUnit, Function, Graph, Operation, Pass, PassPipeline, PassResult,
        PipelineReport, Program, ScalarType, TensorType, Value,
    };

    // Error
    pub use mil_rs::MilError;

    // Model I/O
    pub use mil_rs::{
        model_to_program, onnx_to_program, program_to_model, program_to_multi_function_model,
        program_to_updatable_model, read_mlmodel, read_mlpackage, read_onnx, read_onnx_with_dir,
        write_mlmodel, write_mlpackage,
    };

    // Conversion
    pub use mil_rs::{
        ConversionConfig, LossFunction, UpdatableModelConfig, UpdateOptimizer,
        onnx_to_program_with_config,
    };

    // Reader / inspection
    pub use mil_rs::reader::{print_model_summary, print_onnx_summary};

    /// Protobuf model types — kept for backward compatibility with ironmill-compile-ffi.
    // TODO(§3.6): deprecate and remove once downstream crates import from mil-rs directly.
    pub mod proto {
        /// CoreML specification protobuf types.
        pub mod specification {
            pub use mil_rs::proto::specification::Model;
        }
        /// ONNX protobuf types.
        pub mod onnx {
            pub use mil_rs::proto::onnx::ModelProto;
        }
    }

    /// MIL IR optimization and quantization passes.
    pub mod passes {
        pub use mil_rs::ir::passes::ConstantFoldPass;
        pub use mil_rs::ir::passes::DeadCodeEliminationPass;
        pub use mil_rs::ir::passes::IdentityEliminationPass;
        pub use mil_rs::ir::passes::PolarQuantPass;

        // TODO(§3.6): deprecate and remove — used by ironmill-bench quality module.
        pub use mil_rs::ir::passes::tensor_utils;
    }

    // Quantization passes (used by GPU compilation path)
    pub use mil_rs::ir::Granularity;
    pub use mil_rs::ir::passes::{AffineQuantizePass, BitWidth};

    /// LoRA and MoE conversion helpers.
    pub mod convert {
        pub use mil_rs::convert::lora::{
            LoraAdapter, LoraWeights, detect_lora_adapters, lora_initializer_names, merge_lora,
            merge_lora_weights,
        };
        pub use mil_rs::convert::moe::{
            ExpertDescriptor, ExpertFrequencyProfile, MoeFuseResult, MoeManifest, MoeSplitResult,
            MoeTopology, Stage, detect_moe, fuse_top_k_experts, split_moe,
        };
    }
}
