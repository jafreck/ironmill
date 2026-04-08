//! Shared graph-walk and mutation utilities used by multiple MIL optimization
//! passes.
//!
//! Graph-query helpers (producer→consumer relationships, reference walking)
//! and common mutation helpers (op insertion, reference patching, byte
//! conversion, index packing) live here to avoid duplication across passes.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use half::f16;

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

// ── Mutation helpers ────────────────────────────────────────────────────────

/// Insert new operations after their anchor positions.
///
/// Each entry `(idx, new_ops)` inserts the operations in `new_ops` immediately
/// after the operation at index `idx`, adjusting for the cumulative offset of
/// prior insertions. Each insertion adds `new_ops.len()` to the offset.
///
/// **Note:** the offset increment is hard-coded to `new_ops.len()` which
/// assumes every insertion batch inserts the same structural pattern (e.g.
/// always 2 ops per insertion for a dequant+mul pair). Callers that insert
/// varying numbers of ops per batch should verify correctness.
pub(crate) fn apply_insertions(ops: &mut Vec<Operation>, insertions: &[(usize, Vec<Operation>)]) {
    for (offset, (idx, new_ops)) in insertions.iter().enumerate() {
        let insert_at = idx + 1 + offset * 2;
        for (j, op) in new_ops.iter().enumerate() {
            ops.insert(insert_at + j, op.clone());
        }
    }
}

/// Rewire downstream references from original const outputs to new outputs,
/// then restore the new op's own `x` input back to the original.
///
/// For each `(old_name, new_name)` pair, all references to `old_name` in the
/// block are rewritten to `new_name`. Then, any `mul` op whose output matches
/// `new_name` has its `x` input restored to `old_name` (since `replace_reference`
/// would have rewritten it too).
pub(crate) fn patch_references(body: &mut Block, replacements: &[(String, String)]) {
    for (old_name, new_name) in replacements {
        super::replace_reference(body, old_name, new_name);
        // The mul op's `x` input was also rewritten — fix it back.
        for op in &mut body.operations {
            if op.op_type == "mul" && op.outputs.iter().any(|o| o == new_name) {
                if let Some(Value::Reference(r)) = op.inputs.get_mut("x") {
                    if r == new_name {
                        *r = old_name.clone();
                    }
                }
            }
        }
    }
}

// ── Byte conversion helpers ─────────────────────────────────────────────────

/// Convert raw FP16 little-endian bytes to `Vec<f32>`.
///
/// # Panics
///
/// Debug-asserts that `data.len()` is a multiple of 2.
pub(crate) fn fp16_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    debug_assert!(
        data.len() % 2 == 0,
        "FP16 tensor data length must be a multiple of 2"
    );
    data.chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

/// Pack assignment indices into n-bit packed bytes (MSB-first).
pub fn pack_indices(indices: &[usize], n_bits: u8) -> Vec<u8> {
    if n_bits == 8 {
        return indices.iter().map(|&i| i as u8).collect();
    }

    let mask = (1u16 << n_bits) - 1;
    let total_bits = indices.len() * n_bits as usize;
    let n_bytes = total_bits.div_ceil(8);
    // Allocate one extra byte so the last value's lo byte always has a
    // valid destination, then truncate back to the true output size.
    let mut packed = vec![0u8; n_bytes + 1];

    for (i, &idx) in indices.iter().enumerate() {
        let bit_offset = i * n_bits as usize;
        let byte_pos = bit_offset / 8;
        let bit_in_byte = bit_offset % 8;
        let val = (idx as u16) & mask;

        let shifted = val << (16 - n_bits as usize - bit_in_byte);
        let [hi, lo] = shifted.to_be_bytes();
        packed[byte_pos] |= hi;
        packed[byte_pos + 1] |= lo;
    }

    packed.truncate(n_bytes);
    packed
}
