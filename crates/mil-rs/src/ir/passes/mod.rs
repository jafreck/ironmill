//! Built-in optimization passes for the MIL IR.
//!
//! Each pass implements the [`Pass`](super::Pass) trait and transforms a
//! [`Program`](super::Program) in place.

pub mod attention_fusion;
pub mod bn_weight_fold;
pub mod codebook;
pub mod compute_unit;
pub mod constant_fold;
pub mod dead_code;
pub mod fp16_quantize;
pub mod identity_elim;
pub mod int8_quantize;
pub mod kmeans;
pub mod kv_cache;
pub mod layer_schedule;
pub mod layout_optimize;
pub mod mixed_precision;
pub mod model_split;
pub mod op_fusion;
pub mod op_split;
pub mod op_substitute;
pub mod palettize;
pub mod shape_materialize;
pub mod tensor_utils;
pub mod type_repropagate;

// ANE-specific passes (feature-gated)
#[cfg(feature = "ane-direct")]
pub mod ane_attention_decompose;
#[cfg(feature = "ane-direct")]
pub mod ane_concat_elim;
#[cfg(feature = "ane-direct")]
pub mod ane_layout;
#[cfg(feature = "ane-direct")]
pub mod ane_variable_naming;

pub use attention_fusion::{AttentionFusionPass, GqaFusionPass};
pub use bn_weight_fold::ConvBatchNormWeightFoldPass;
pub use codebook::CodebookOptimizationPass;
pub use compute_unit::ComputeUnitAnnotationPass;
pub use constant_fold::ConstantFoldPass;
pub use dead_code::DeadCodeEliminationPass;
pub use fp16_quantize::Fp16QuantizePass;
pub use identity_elim::IdentityEliminationPass;
pub use int8_quantize::{Granularity, Int8QuantizePass};
pub use kv_cache::KvCachePass;
pub use layer_schedule::{LayerScheduleConfig, LayerSchedulePass, LayerType};
pub use layout_optimize::LayoutOptimizationPass;
pub use mixed_precision::{
    ExpertQuantConfig, ExpertQuantStrategy, MixedPrecisionConfig, MixedPrecisionPass, OpPrecision,
    PerExpertQuantPass,
};
pub use model_split::{ModelSplitPass, SplitResult};
pub use op_fusion::{
    ConvBatchNormFusionPass, ConvReluFusionPass, GeluLinearFusionPass, LayerNormLinearFusionPass,
    LinearReluFusionPass, ResidualAddFusionPass,
};
pub use op_split::{OpSplittingPass, parse_memory_size};
pub use op_substitute::OpSubstitutionPass;
pub use palettize::{GroupedPalettizePass, PalettizePass};
pub use shape_materialize::{AutoregressiveShapeMaterializePass, ShapeMaterializePass};
pub use type_repropagate::TypeRepropagationPass;

#[cfg(feature = "ane-direct")]
pub use ane_attention_decompose::AttentionDecomposePass;
#[cfg(feature = "ane-direct")]
pub use ane_concat_elim::AneConcatEliminationPass;
#[cfg(feature = "ane-direct")]
pub use ane_layout::AneLayoutPass;
#[cfg(feature = "ane-direct")]
pub use ane_variable_naming::AneVariableNamingPass;

use super::program::Block;
use super::types::Value;

/// Replace every [`Value::Reference`] that points to `old_name` with one
/// pointing to `new_name`, across all operations in `block`.
pub fn replace_reference(block: &mut Block, old_name: &str, new_name: &str) {
    for op in &mut block.operations {
        for value in op.inputs.values_mut() {
            replace_in_value(value, old_name, new_name);
        }
    }
    // Also update block outputs.
    for out in &mut block.outputs {
        if out == old_name {
            *out = new_name.to_string();
        }
    }
}

/// Recursively replace references inside a [`Value`], handling nested lists.
fn replace_in_value(value: &mut Value, old_name: &str, new_name: &str) {
    match value {
        Value::Reference(name) if name == old_name => {
            *name = new_name.to_string();
        }
        Value::List(items) => {
            for item in items {
                replace_in_value(item, old_name, new_name);
            }
        }
        _ => {}
    }
}
