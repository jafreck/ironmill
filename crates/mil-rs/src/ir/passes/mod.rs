//! Built-in optimization passes for the MIL IR.
//!
//! Each pass implements the [`Pass`](super::Pass) trait and transforms a
//! [`Program`](super::Program) in place.
//!
//! ANE-specific passes (codebook, compute-unit, kv-cache, layer-schedule,
//! mixed-precision, model-split, op-split, op-substitute, and all `ane_*`
//! passes) have moved to the `ironmill-compile` crate.

pub mod attention_fusion;
pub mod beta_quantizer;
pub mod bn_weight_fold;
pub mod constant_fold;
pub mod dead_code;
pub mod fp16_quantize;
pub mod identity_elim;
pub mod int4_pack;
pub mod int8_quantize;
pub mod kmeans;
pub mod layout_optimize;
pub mod op_fusion;
pub mod palettize;
pub mod polar_quantize;
pub mod polar_rotation_fusion;
pub mod rotation;
pub mod shape_materialize;
pub mod tensor_utils;
pub mod type_repropagate;
pub(crate) mod util;

#[cfg(feature = "gptq")]
pub mod gptq;

pub use attention_fusion::{AttentionFusionPass, GqaFusionPass};
pub use bn_weight_fold::ConvBatchNormWeightFoldPass;
pub use constant_fold::ConstantFoldPass;
pub use dead_code::DeadCodeEliminationPass;
pub use fp16_quantize::Fp16QuantizePass;
pub use identity_elim::IdentityEliminationPass;
pub use int8_quantize::{Granularity, Int8QuantizePass};
pub use layout_optimize::LayoutOptimizationPass;
pub use op_fusion::{
    ConvBatchNormFusionPass, ConvReluFusionPass, GeluLinearFusionPass, LayerNormLinearFusionPass,
    LinearReluFusionPass, ResidualAddFusionPass,
};
pub use palettize::{GroupedPalettizePass, PalettizePass};
pub use polar_quantize::PolarQuantPass;
pub use polar_rotation_fusion::PolarRotationFusionPass;
pub use shape_materialize::{AutoregressiveShapeMaterializePass, ShapeMaterializePass};
pub use type_repropagate::TypeRepropagationPass;

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
