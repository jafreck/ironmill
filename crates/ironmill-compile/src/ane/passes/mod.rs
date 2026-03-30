//! ANE-specific optimization passes.
//!
//! These passes are backend-specific and live in the compilation crate rather
//! than in the generic `mil-rs` pass infrastructure.

pub mod ane_arg_promotion;
pub mod ane_attention_decompose;
pub mod ane_concat_elim;
pub mod ane_layout;
pub mod ane_matmul_to_conv;
pub mod ane_variable_naming;
pub mod codebook;
pub mod compute_unit;
pub mod kv_cache;
pub mod layer_schedule;
pub mod mixed_precision;
pub mod model_split;
pub mod op_split;
pub mod op_substitute;

pub use ane_arg_promotion::AneArgPromotionPass;
pub use ane_attention_decompose::AttentionDecomposePass;
pub use ane_concat_elim::AneConcatEliminationPass;
pub use ane_layout::AneLayoutPass;
pub use ane_matmul_to_conv::AneMatmulToConvPass;
pub use ane_variable_naming::AneVariableNamingPass;
pub use codebook::CodebookOptimizationPass;
pub use compute_unit::ComputeUnitAnnotationPass;
pub use kv_cache::KvCachePass;
pub use layer_schedule::{LayerScheduleConfig, LayerSchedulePass, LayerType};
pub use mixed_precision::{
    ExpertQuantConfig, ExpertQuantStrategy, MixedPrecisionConfig, MixedPrecisionPass, OpPrecision,
    PerExpertQuantPass,
};
pub use model_split::{ModelSplitPass, SplitResult};
pub use op_split::{OpSplittingPass, parse_memory_size};
pub use op_substitute::OpSubstitutionPass;
