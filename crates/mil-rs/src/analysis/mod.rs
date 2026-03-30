//! Analysis passes for MIL IR programs.
//!
//! Read-only analysis that computes metrics from a [`Program`] without
//! modifying it.

pub mod arch;
pub mod flops;

use std::collections::HashMap;

use crate::ir::{Function, TensorType, Value};

/// Build a map from output names to their resolved types for all operations
/// and inputs in a function.
pub fn build_type_map(func: &Function) -> HashMap<String, TensorType> {
    let mut types = HashMap::new();

    for (name, ty) in &func.inputs {
        types.insert(name.clone(), ty.clone());
    }

    for op in &func.body.operations {
        for (i, out_name) in op.outputs.iter().enumerate() {
            if let Some(Some(ty)) = op.output_types.get(i) {
                types.insert(out_name.clone(), ty.clone());
            }
        }

        // For const ops, extract the type from the tensor value.
        if op.op_type == "const" {
            if let Some(Value::Tensor { shape, dtype, .. }) = op.inputs.get("val") {
                let ty = TensorType::new(*dtype, shape.clone());
                for out_name in &op.outputs {
                    types.entry(out_name.clone()).or_insert_with(|| ty.clone());
                }
            }
        }
    }

    types
}
