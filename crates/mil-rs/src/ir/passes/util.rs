//! Shared graph-walk utilities used by multiple MIL optimization passes.
//!
//! These helpers query producer→consumer relationships in a [`Block`]'s
//! operation list without mutating anything.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::error::MilError;
use crate::ir::operation::Operation;
use crate::ir::program::Block;
use crate::ir::types::Value;
use crate::weights::WeightProvider;

/// Create a resolver closure from an optional provider Arc and a spill index.
/// Used by quantization passes to materialize External tensors.
/// Checks the spill index first (for data written between passes), then
/// falls back to the weight provider.
pub(crate) fn make_resolver<'a>(
    provider: &'a Option<Arc<dyn WeightProvider + Send + Sync>>,
    spill_index: &'a HashMap<String, PathBuf>,
) -> impl Fn(&str) -> Result<Vec<u8>, MilError> + 'a {
    move |key: &str| {
        // Check spill first (quantized data written between passes)
        if let Some(path) = spill_index.get(key) {
            return std::fs::read(path).map_err(MilError::Io);
        }
        // Fall back to weight provider (original weight data)
        let p = provider.as_ref().ok_or_else(|| {
            MilError::Validation(format!(
                "no weight provider attached; cannot resolve tensor '{key}'"
            ))
        })?;
        let tensor = p.tensor(key)?;
        Ok(tensor.data.into_owned())
    }
}

/// Returns `true` if `value` contains a [`Value::Reference`] to `name`,
/// recursing into nested [`Value::List`] variants.
pub(crate) fn references_name(value: &Value, name: &str) -> bool {
    match value {
        Value::Reference(n) => n == name,
        Value::List(items) => items.iter().any(|v| references_name(v, name)),
        _ => false,
    }
}

/// Check if a value name is only consumed by a single operation in the block.
///
/// `consumer_idx` is the index of the expected consumer; this function verifies
/// that no *other* operation (or block output) references `value_name`.
pub(crate) fn is_single_consumer(block: &Block, value_name: &str, consumer_idx: usize) -> bool {
    for (idx, op) in block.operations.iter().enumerate() {
        if idx == consumer_idx {
            continue;
        }
        for input_val in op.inputs.values() {
            if references_name(input_val, value_name) {
                return false;
            }
        }
    }
    // Also check block outputs — if the value is a block output it has
    // an external consumer and must not be fused away.
    !block.outputs.contains(&value_name.to_string())
}

/// Recursively visit every [`Value::Reference`] inside `value`, calling `cb`
/// for each referenced name. Handles nested [`Value::List`] variants.
pub(crate) fn collect_value_references(value: &Value, cb: &mut impl FnMut(&str)) {
    match value {
        Value::Reference(name) => cb(name),
        Value::List(items) => {
            for item in items {
                collect_value_references(item, cb);
            }
        }
        _ => {}
    }
}

/// Collect all [`Value::Reference`] names from an operation's inputs.
pub(crate) fn collect_op_references(op: &Operation) -> Vec<String> {
    let mut refs = Vec::new();
    for value in op.inputs.values() {
        collect_value_references(value, &mut |name| refs.push(name.to_string()));
    }
    refs
}

/// Build a map from each value name to the `(op_index, input_key)` pairs that
/// consume it. Useful for tracing producer→consumer relationships with full
/// input-key information.
pub(crate) fn build_consumer_map(ops: &[Operation]) -> HashMap<String, Vec<(usize, String)>> {
    let mut map: HashMap<String, Vec<(usize, String)>> = HashMap::new();
    for (idx, op) in ops.iter().enumerate() {
        for (key, val) in &op.inputs {
            collect_value_references(val, &mut |ref_name| {
                map.entry(ref_name.to_string())
                    .or_default()
                    .push((idx, key.clone()));
            });
        }
    }
    map
}

/// Build a map from each value name to the indices of operations that consume
/// it. A simpler variant of [`build_consumer_map`] that omits input-key info.
pub(crate) fn build_consumer_index_map(ops: &[Operation]) -> HashMap<String, Vec<usize>> {
    let mut map: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, op) in ops.iter().enumerate() {
        for ref_name in collect_op_references(op) {
            map.entry(ref_name).or_default().push(idx);
        }
    }
    map
}
