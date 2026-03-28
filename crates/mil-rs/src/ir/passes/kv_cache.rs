//! KV cache layout pass — detect autoregressive attention patterns and insert
//! cache management ops for efficient inference on the Apple Neural Engine.
//!
//! Autoregressive (decoder) models consume and produce `past_key_values` tensors
//! across time steps. This pass:
//!
//! 1. **Detects** ops whose inputs/outputs reference KV cache tensors (names
//!    containing `past_key_values`, `cache`, `past_key`, `past_value`).
//! 2. **Inserts** ring-buffer cache management ops (`kv_cache_update`,
//!    `kv_cache_read`) with a configurable `max_seq_length`.
//! 3. **Materializes** cache tensor shapes so they are fully static — a
//!    prerequisite for ANE eligibility.
//! 4. **Annotates** cache tensors for NHWC conversion where applicable.
//! 5. **Supports GQA** cache layouts (fewer KV heads shared across Q heads) by
//!    propagating head counts from `grouped_query_attention` ops.

use std::collections::HashSet;

use crate::error::Result;
use crate::ir::operation::Operation;
use crate::ir::pass::Pass;
use crate::ir::program::{Block, Function, Program};
use crate::ir::types::Value;

use super::replace_reference;

/// Default maximum sequence length for KV cache ring buffers.
const DEFAULT_MAX_SEQ_LENGTH: usize = 2048;

/// Name fragments that identify KV cache tensors.
const CACHE_NAME_PATTERNS: &[&str] = &[
    "past_key_values",
    "past_key",
    "past_value",
    "key_cache",
    "value_cache",
    "cache",
];

/// Insert and manage KV cache ring buffers for autoregressive attention.
pub struct KvCachePass {
    /// Maximum sequence length for the statically-sized ring buffer.
    pub max_seq_length: usize,
}

impl KvCachePass {
    /// Create a new pass with the given maximum sequence length.
    pub fn new(max_seq_length: usize) -> Self {
        Self { max_seq_length }
    }
}

impl Default for KvCachePass {
    fn default() -> Self {
        Self {
            max_seq_length: DEFAULT_MAX_SEQ_LENGTH,
        }
    }
}

impl Pass for KvCachePass {
    fn name(&self) -> &str {
        "kv-cache"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            let cache_inputs = detect_cache_inputs(function);
            if cache_inputs.is_empty() {
                continue;
            }

            materialize_cache_input_shapes(function, &cache_inputs, self.max_seq_length);
            insert_cache_ops(&mut function.body, &cache_inputs, self.max_seq_length);
            annotate_cache_for_nhwc(&mut function.body, &cache_inputs);
            propagate_gqa_cache_layout(&mut function.body);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Detection: find function inputs that are KV cache tensors
// ---------------------------------------------------------------------------

/// Identify function inputs whose names match KV cache naming conventions.
fn detect_cache_inputs(function: &Function) -> Vec<CacheInput> {
    let mut results = Vec::new();
    for (name, _ty) in &function.inputs {
        if is_cache_name(name) {
            let role = if name.contains("key") {
                CacheRole::Key
            } else if name.contains("value") || name.contains("val") {
                CacheRole::Value
            } else {
                // Generic "cache" / "past_key_values" — treat as key by default,
                // the paired value will be detected separately.
                CacheRole::Key
            };
            results.push(CacheInput {
                name: name.clone(),
                role,
            });
        }
    }
    results
}

/// Returns `true` if `name` looks like a KV cache tensor name.
fn is_cache_name(name: &str) -> bool {
    let lower = name.to_lowercase();
    CACHE_NAME_PATTERNS
        .iter()
        .any(|pattern| lower.contains(pattern))
}

/// Role of a cache tensor in an attention layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CacheRole {
    Key,
    Value,
}

/// A function input identified as a KV cache tensor.
#[derive(Debug, Clone)]
struct CacheInput {
    name: String,
    role: CacheRole,
}

// ---------------------------------------------------------------------------
// Shape materialization for cache inputs
// ---------------------------------------------------------------------------

/// Replace dynamic dimensions in cache input shapes with concrete values.
///
/// Cache tensors typically have shape `[batch, num_heads, seq_len, head_dim]`.
/// The `seq_len` dimension (axis 2) is set to `max_seq_length`; other dynamic
/// dims get sensible defaults (batch=1).
fn materialize_cache_input_shapes(
    function: &mut Function,
    cache_inputs: &[CacheInput],
    max_seq_length: usize,
) {
    let cache_names: HashSet<&str> = cache_inputs.iter().map(|c| c.name.as_str()).collect();

    for (input_name, tensor_type) in &mut function.inputs {
        if !cache_names.contains(input_name.as_str()) {
            continue;
        }
        if tensor_type.is_static() {
            continue;
        }
        for (axis, dim) in tensor_type.shape.iter_mut().enumerate() {
            if dim.is_some() {
                continue;
            }
            // [batch, num_heads, seq_len, head_dim]
            *dim = Some(match axis {
                0 => 1,              // batch
                2 => max_seq_length, // sequence length
                _ => 64,             // heads / head_dim fallback
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Cache op insertion
// ---------------------------------------------------------------------------

/// Insert `kv_cache_read` and `kv_cache_update` ops around attention patterns
/// that consume cache inputs.
fn insert_cache_ops(block: &mut Block, cache_inputs: &[CacheInput], max_seq_length: usize) {
    let mut new_ops: Vec<(usize, Operation)> = Vec::new();
    let mut counter = 0;

    for ci in cache_inputs {
        // Find operations that consume this cache input.
        let consumers = find_cache_consumers(block, &ci.name);
        if consumers.is_empty() {
            continue;
        }

        let read_name = format!("{}_read", ci.name);
        let read_out = format!("{}_read_out", ci.name);

        // Insert a kv_cache_read before the first consumer.
        let insert_pos = consumers.iter().copied().min().unwrap_or(0);
        let read_op = Operation::new("kv_cache_read", &read_name)
            .with_input("cache", Value::Reference(ci.name.clone()))
            .with_output(&read_out)
            .with_attr("max_seq_length", Value::Int(max_seq_length as i64))
            .with_attr("role", Value::String(cache_role_str(ci.role).to_string()));

        new_ops.push((insert_pos, read_op));

        // Rewire consumers to read from the cache read output.
        replace_reference(block, &ci.name, &read_out);

        // Find operations that produce updated cache values (block outputs
        // matching cache naming conventions).
        if let Some(update_src) = find_cache_producer(block, &ci.name) {
            let update_name = format!("{}_update_{}", ci.name, counter);
            let update_out = format!("{}_update_out_{}", ci.name, counter);
            counter += 1;

            let update_op = Operation::new("kv_cache_update", &update_name)
                .with_input("cache", Value::Reference(read_out.clone()))
                .with_input("new_entry", Value::Reference(update_src.clone()))
                .with_output(&update_out)
                .with_attr("max_seq_length", Value::Int(max_seq_length as i64));

            // Insert after the producer op.
            let producer_pos = block
                .operations
                .iter()
                .position(|op| op.outputs.iter().any(|s| s.as_str() == update_src.as_str()))
                .map(|p| p + 1)
                .unwrap_or(block.operations.len());

            new_ops.push((producer_pos, update_op));

            // Rewire any block output that references the raw producer to
            // reference the update output instead.
            for out in &mut block.outputs {
                if *out == update_src {
                    *out = update_out.clone();
                }
            }
        }
    }

    // Insert ops in reverse index order to keep indices stable.
    new_ops.sort_by(|a, b| b.0.cmp(&a.0));
    for (pos, op) in new_ops {
        let clamped = pos.min(block.operations.len());
        block.operations.insert(clamped, op);
    }
}

/// Find indices of operations that consume `cache_name` as an input.
fn find_cache_consumers(block: &Block, cache_name: &str) -> Vec<usize> {
    block
        .operations
        .iter()
        .enumerate()
        .filter(|(_, op)| {
            op.inputs
                .values()
                .any(|v| matches!(v, Value::Reference(n) if n == cache_name))
        })
        .map(|(i, _)| i)
        .collect()
}

/// Find an operation whose output will become the updated cache value.
///
/// Heuristic: look for a `concat` or `matmul` op whose output name suggests
/// it is producing an updated key/value tensor, or find ops that produce values
/// referenced by block outputs with cache-like names.
fn find_cache_producer(block: &Block, cache_name: &str) -> Option<String> {
    // First, check block outputs for cache-like names.
    for out_name in &block.outputs {
        if is_cache_name(out_name) && out_name != cache_name {
            return Some(out_name.clone());
        }
    }
    // Look for concat ops that might be building updated caches.
    for op in &block.operations {
        if op.op_type == "concat" {
            for output in &op.outputs {
                if is_cache_name(output) {
                    return Some(output.clone());
                }
            }
        }
    }
    None
}

fn cache_role_str(role: CacheRole) -> &'static str {
    match role {
        CacheRole::Key => "key",
        CacheRole::Value => "value",
    }
}

// ---------------------------------------------------------------------------
// NHWC annotation for cache tensors
// ---------------------------------------------------------------------------

/// Annotate cache-related ops for NHWC layout conversion.
///
/// The ANE prefers NHWC layout. For 4-D cache tensors
/// `[batch, heads, seq, dim]`, the layout pass can transpose them to
/// `[batch, seq, dim, heads]`. We tag the ops so the layout pass can
/// recognise them.
fn annotate_cache_for_nhwc(block: &mut Block, cache_inputs: &[CacheInput]) {
    let cache_names: HashSet<&str> = cache_inputs.iter().map(|c| c.name.as_str()).collect();

    for op in &mut block.operations {
        if op.op_type != "kv_cache_read" && op.op_type != "kv_cache_update" {
            continue;
        }
        // Check if this cache op is associated with a 4-D cache tensor.
        let is_4d = if let Some(Value::Reference(ref_name)) = op.inputs.get("cache") {
            cache_names.contains(ref_name.as_str()) || ref_name.ends_with("_read_out")
        } else {
            false
        };
        if is_4d {
            op.attributes
                .insert("layout_hint".to_string(), Value::String("nhwc".to_string()));
        }
    }
}

// ---------------------------------------------------------------------------
// GQA cache layout propagation
// ---------------------------------------------------------------------------

/// When `grouped_query_attention` ops exist, propagate GQA metadata to
/// associated cache ops so downstream passes know the KV head count differs
/// from Q head count.
fn propagate_gqa_cache_layout(block: &mut Block) {
    // Collect the value names that GQA ops consume as K/V.
    let mut gqa_kv_refs: HashSet<String> = HashSet::new();
    for op in &block.operations {
        if op.op_type == "grouped_query_attention" {
            if let Some(Value::Reference(k_ref)) = op.inputs.get("K") {
                gqa_kv_refs.insert(k_ref.clone());
            }
            if let Some(Value::Reference(v_ref)) = op.inputs.get("V") {
                gqa_kv_refs.insert(v_ref.clone());
            }
        }
    }
    if gqa_kv_refs.is_empty() {
        return;
    }

    // Transitively expand: if a value in gqa_kv_refs is produced by an op
    // that consumes other values, add those source values too. This traces
    // through concat and similar ops back to cache reads.
    let mut expanded = gqa_kv_refs.clone();
    loop {
        let mut added = false;
        for op in &block.operations {
            let produces_gqa_input = op.outputs.iter().any(|out| expanded.contains(out));
            if produces_gqa_input {
                for v in op.inputs.values() {
                    if let Value::Reference(r) = v {
                        if expanded.insert(r.clone()) {
                            added = true;
                        }
                    }
                }
            }
        }
        if !added {
            break;
        }
    }

    // Tag cache ops whose outputs are in the expanded set.
    for op in &mut block.operations {
        if op.op_type != "kv_cache_read" && op.op_type != "kv_cache_update" {
            continue;
        }
        let feeds_gqa = op.outputs.iter().any(|out| expanded.contains(out));
        if feeds_gqa {
            op.attributes
                .insert("is_gqa".to_string(), Value::Bool(true));
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::program::{Function, Program};
    use crate::ir::tensor::{ScalarType, TensorType};

    /// Helper: build a minimal program with cache inputs and an attention pattern.
    fn make_cache_program() -> Program {
        let mut program = Program::new("1.0.0");
        let key_ty = TensorType::with_dynamic_shape(
            ScalarType::Float16,
            vec![None, Some(8), None, Some(64)],
        );
        let val_ty = TensorType::with_dynamic_shape(
            ScalarType::Float16,
            vec![None, Some(8), None, Some(64)],
        );
        let q_ty = TensorType::new(ScalarType::Float16, vec![1, 8, 1, 64]);

        let mut func = Function::new("main")
            .with_input("past_key", key_ty)
            .with_input("past_value", val_ty)
            .with_input("query", q_ty);

        // concat: past_key + new_key → updated_key_cache
        let concat_k = Operation::new("concat", "concat_k")
            .with_input("x", Value::Reference("past_key".into()))
            .with_input("y", Value::Reference("new_key_proj".into()))
            .with_output("updated_key_cache")
            .with_attr("axis", Value::Int(2));

        // concat: past_value + new_value → updated_value_cache
        let concat_v = Operation::new("concat", "concat_v")
            .with_input("x", Value::Reference("past_value".into()))
            .with_input("y", Value::Reference("new_val_proj".into()))
            .with_output("updated_value_cache")
            .with_attr("axis", Value::Int(2));

        // Dummy scaled_dot_product_attention consuming the cached K/V
        let attn = Operation::new("scaled_dot_product_attention", "attn_0")
            .with_input("Q", Value::Reference("query".into()))
            .with_input("K", Value::Reference("updated_key_cache".into()))
            .with_input("V", Value::Reference("updated_value_cache".into()))
            .with_output("attn_out");

        func.body.add_op(concat_k);
        func.body.add_op(concat_v);
        func.body.add_op(attn);
        func.body.outputs.push("attn_out".into());
        func.body.outputs.push("updated_key_cache".into());
        func.body.outputs.push("updated_value_cache".into());

        program.add_function(func);
        program
    }

    #[test]
    fn detects_cache_inputs() {
        let program = make_cache_program();
        let func = &program.functions["main"];
        let inputs = detect_cache_inputs(func);
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0].name, "past_key");
        assert_eq!(inputs[0].role, CacheRole::Key);
        assert_eq!(inputs[1].name, "past_value");
        assert_eq!(inputs[1].role, CacheRole::Value);
    }

    #[test]
    fn no_cache_inputs_is_noop() {
        let mut program = Program::new("1.0.0");
        let q_ty = TensorType::new(ScalarType::Float16, vec![1, 8, 10, 64]);
        let func = Function::new("main").with_input("query", q_ty);
        program.add_function(func);

        let pass = KvCachePass::default();
        pass.run(&mut program).unwrap();

        // No ops should be inserted.
        assert!(program.functions["main"].body.operations.is_empty());
    }

    #[test]
    fn materializes_dynamic_cache_shapes() {
        let mut program = make_cache_program();
        let pass = KvCachePass::new(1024);
        pass.run(&mut program).unwrap();

        let func = &program.functions["main"];
        for (name, ty) in &func.inputs {
            if is_cache_name(name) {
                assert!(
                    ty.is_static(),
                    "cache input '{name}' should be fully static after pass"
                );
                // seq_len axis should be max_seq_length
                assert_eq!(ty.shape[2], Some(1024));
            }
        }
    }

    #[test]
    fn inserts_cache_read_ops() {
        let mut program = make_cache_program();
        let pass = KvCachePass::default();
        pass.run(&mut program).unwrap();

        let ops = &program.functions["main"].body.operations;
        let read_ops: Vec<_> = ops
            .iter()
            .filter(|o| o.op_type == "kv_cache_read")
            .collect();

        assert_eq!(read_ops.len(), 2, "should insert 2 cache read ops");
        assert!(
            read_ops.iter().any(|o| o.name.contains("past_key")),
            "should have key cache read"
        );
        assert!(
            read_ops.iter().any(|o| o.name.contains("past_value")),
            "should have value cache read"
        );
    }

    #[test]
    fn inserts_cache_update_ops() {
        let mut program = make_cache_program();
        let pass = KvCachePass::default();
        pass.run(&mut program).unwrap();

        let ops = &program.functions["main"].body.operations;
        let update_ops: Vec<_> = ops
            .iter()
            .filter(|o| o.op_type == "kv_cache_update")
            .collect();

        assert_eq!(update_ops.len(), 2, "should insert 2 cache update ops");
        for op in &update_ops {
            assert_eq!(
                op.attributes.get("max_seq_length"),
                Some(&Value::Int(DEFAULT_MAX_SEQ_LENGTH as i64)),
            );
        }
    }

    #[test]
    fn cache_ops_have_max_seq_length_attr() {
        let mut program = make_cache_program();
        let pass = KvCachePass::new(4096);
        pass.run(&mut program).unwrap();

        let ops = &program.functions["main"].body.operations;
        for op in ops {
            if op.op_type == "kv_cache_read" || op.op_type == "kv_cache_update" {
                assert_eq!(
                    op.attributes.get("max_seq_length"),
                    Some(&Value::Int(4096)),
                    "op '{}' should have max_seq_length=4096",
                    op.name,
                );
            }
        }
    }

    #[test]
    fn annotates_cache_ops_for_nhwc() {
        let mut program = make_cache_program();
        let pass = KvCachePass::default();
        pass.run(&mut program).unwrap();

        let ops = &program.functions["main"].body.operations;
        let cache_ops: Vec<_> = ops
            .iter()
            .filter(|o| o.op_type == "kv_cache_read")
            .collect();

        for op in &cache_ops {
            assert_eq!(
                op.attributes.get("layout_hint"),
                Some(&Value::String("nhwc".to_string())),
                "cache read '{}' should have NHWC layout hint",
                op.name,
            );
        }
    }

    #[test]
    fn propagates_gqa_layout_to_cache_ops() {
        let mut program = Program::new("1.0.0");
        let key_ty = TensorType::new(ScalarType::Float16, vec![1, 2, 128, 64]);
        let val_ty = TensorType::new(ScalarType::Float16, vec![1, 2, 128, 64]);
        let q_ty = TensorType::new(ScalarType::Float16, vec![1, 8, 1, 64]);

        let mut func = Function::new("main")
            .with_input("past_key", key_ty)
            .with_input("past_value", val_ty)
            .with_input("query", q_ty);

        let concat_k = Operation::new("concat", "concat_k")
            .with_input("x", Value::Reference("past_key".into()))
            .with_input("y", Value::Reference("new_k".into()))
            .with_output("updated_key_cache");

        let concat_v = Operation::new("concat", "concat_v")
            .with_input("x", Value::Reference("past_value".into()))
            .with_input("y", Value::Reference("new_v".into()))
            .with_output("updated_value_cache");

        let gqa = Operation::new("grouped_query_attention", "gqa_0")
            .with_input("Q", Value::Reference("query".into()))
            .with_input("K", Value::Reference("updated_key_cache".into()))
            .with_input("V", Value::Reference("updated_value_cache".into()))
            .with_attr("is_gqa", Value::Bool(true))
            .with_output("gqa_out");

        func.body.add_op(concat_k);
        func.body.add_op(concat_v);
        func.body.add_op(gqa);
        func.body.outputs.push("gqa_out".into());
        func.body.outputs.push("updated_key_cache".into());
        func.body.outputs.push("updated_value_cache".into());

        program.add_function(func);

        let pass = KvCachePass::default();
        pass.run(&mut program).unwrap();

        // The cache read ops that feed into GQA should be tagged is_gqa=true.
        let ops = &program.functions["main"].body.operations;
        let gqa_cache_ops: Vec<_> = ops
            .iter()
            .filter(|o| {
                (o.op_type == "kv_cache_read" || o.op_type == "kv_cache_update")
                    && o.attributes.get("is_gqa") == Some(&Value::Bool(true))
            })
            .collect();

        assert!(
            !gqa_cache_ops.is_empty(),
            "cache ops feeding GQA should be tagged with is_gqa"
        );
    }

    #[test]
    fn is_cache_name_detects_variants() {
        assert!(is_cache_name("past_key_values.0.key"));
        assert!(is_cache_name("layer.3.past_key"));
        assert!(is_cache_name("decoder.past_value"));
        assert!(is_cache_name("key_cache_layer_0"));
        assert!(is_cache_name("value_cache"));
        assert!(is_cache_name("PAST_KEY_VALUES"));
        assert!(is_cache_name("model.cache.key"));
        assert!(!is_cache_name("query_projection"));
        assert!(!is_cache_name("hidden_states"));
        assert!(!is_cache_name("attention_weights"));
    }

    #[test]
    fn configurable_max_seq_length() {
        let pass = KvCachePass::new(512);
        assert_eq!(pass.max_seq_length, 512);

        let default = KvCachePass::default();
        assert_eq!(default.max_seq_length, DEFAULT_MAX_SEQ_LENGTH);
    }

    #[test]
    fn preserves_non_cache_ops() {
        let mut program = make_cache_program();

        // Add a non-cache op.
        let relu = Operation::new("relu", "relu_0")
            .with_input("x", Value::Reference("attn_out".into()))
            .with_output("relu_out");
        program.functions["main"].body.add_op(relu);

        let pass = KvCachePass::default();
        pass.run(&mut program).unwrap();

        let ops = &program.functions["main"].body.operations;
        assert!(
            ops.iter()
                .any(|o| o.op_type == "relu" && o.name == "relu_0"),
            "non-cache ops should be preserved"
        );
    }
}
