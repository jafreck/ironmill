//! Shared test helpers for MIL optimization pass tests.
//!
//! These helpers eliminate duplication across the per-pass `#[cfg(test)]`
//! modules. Import with `use super::test_helpers::*;` from any sibling
//! pass's test module.

use crate::ir::operation::Operation;
use crate::ir::tensor::ScalarType;
use crate::ir::types::{TensorData, Value};

/// Create FP32 tensor bytes from a slice of f32 values.
pub(crate) fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Build a `const` op with a pre-built tensor [`Value`].
pub(crate) fn const_tensor_op(name: &str, output: &str, value: Value) -> Operation {
    Operation::new("const", name)
        .with_input("val", value)
        .with_output(output)
}

/// Build a `const` op from raw f32 data and shape (convenience wrapper).
pub(crate) fn const_f32_tensor_op(
    name: &str,
    output: &str,
    data: &[f32],
    shape: Vec<usize>,
) -> Operation {
    const_tensor_op(
        name,
        output,
        Value::Tensor {
            data: TensorData::Inline(f32_bytes(data)),
            shape,
            dtype: ScalarType::Float32,
        },
    )
}
