//! Codebook optimization pass for Residual Vector Quantization (RVQ).
//!
//! Neural audio codecs (Mimi, EnCodec) use RVQ with learned codebooks.
//! In ONNX these appear as gather/embedding patterns:
//!
//! ```text
//! codebook_0 = const(...)   # static codebook table
//! codebook_1 = const(...)
//! ...
//! emb_0 = gather(codebook_0, indices_0)
//! emb_1 = gather(codebook_1, indices_1)
//! sum_01 = add(emb_0, emb_1)
//! sum_012 = add(sum_01, emb_2)
//! ...
//! ```
//!
//! This pass detects the pattern and replaces it with a single fused
//! `codebook_gather` op that stores all codebooks as a stacked tensor,
//! enabling ANE-friendly static embedding table layout.

use mil_rs::error::Result;
use mil_rs::ir::Operation;
use mil_rs::ir::Pass;
use mil_rs::ir::ScalarType;
use mil_rs::ir::Value;
use mil_rs::ir::{Block, Program};

use mil_rs::ir::passes::replace_reference;

/// Byte-size threshold above which codebook weights are tagged for palettization.
const PALETTIZE_THRESHOLD_BYTES: usize = 4 * 1024; // 4 KiB

/// Fuse RVQ codebook gather+sum patterns into a single `codebook_gather` op.
///
/// The pass is a no-op when no RVQ patterns are found, making it safe to
/// include in the default pipeline.
pub struct CodebookOptimizationPass;

impl Pass for CodebookOptimizationPass {
    fn name(&self) -> &str {
        "codebook-optimization"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            optimize_codebook_patterns(&mut function.body);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Pattern detection helpers
// ---------------------------------------------------------------------------

/// A single codebook gather: a `gather` op whose data input (`x`) is produced
/// by a `const` op holding a static tensor (the codebook table).
struct CodebookGather {
    /// Index of the `const` op that holds the codebook tensor.
    const_idx: usize,
    /// Index of the `gather` op.
    gather_idx: usize,
    /// Output name of the `gather` op.
    gather_output: String,
    /// The indices input reference name (the quantized codes).
    indices_ref: String,
}

/// A complete RVQ decode pattern: multiple codebook gathers summed together.
struct RvqPattern {
    /// The individual codebook gathers in this pattern.
    gathers: Vec<CodebookGather>,
    /// Indices of the `add` ops that form the summation chain.
    add_indices: Vec<usize>,
    /// Output name of the final `add` op (the decoded output).
    final_output: String,
}

/// Find the index of the operation that produces `output_name`.
fn find_producer(block: &Block, output_name: &str) -> Option<usize> {
    block
        .operations
        .iter()
        .position(|op| op.outputs.first().map(|s| s.as_str()) == Some(output_name))
}

/// Find all operations that consume `output_name` via any input.
fn find_consumers(block: &Block, output_name: &str) -> Vec<usize> {
    block
        .operations
        .iter()
        .enumerate()
        .filter(|(_, op)| {
            op.inputs
                .values()
                .any(|v| matches!(v, Value::Reference(n) if n == output_name))
        })
        .map(|(i, _)| i)
        .collect()
}

/// Check if `value_name` is only consumed by `expected_consumer` and is not
/// a block output.
fn is_internal_single_consumer(block: &Block, value_name: &str, expected_consumer: usize) -> bool {
    if block.outputs.contains(&value_name.to_string()) {
        return false;
    }
    let consumers = find_consumers(block, value_name);
    consumers.len() == 1 && consumers[0] == expected_consumer
}

/// Try to identify a codebook gather at `gather_idx`:
/// a `gather` op whose `x` input comes from a `const` op with a tensor value.
fn identify_codebook_gather(block: &Block, gather_idx: usize) -> Option<CodebookGather> {
    let gather_op = block.operations.get(gather_idx)?;
    if gather_op.op_type != "gather" {
        return None;
    }

    let gather_output = gather_op.outputs.first()?.clone();

    // The data input for gather is "x".
    let data_ref = match gather_op.inputs.get("x") {
        Some(Value::Reference(name)) => name.clone(),
        _ => return None,
    };

    // The indices input is "indices".
    let indices_ref = match gather_op.inputs.get("indices") {
        Some(Value::Reference(name)) => name.clone(),
        _ => return None,
    };

    // The data producer must be a `const` op with a tensor value.
    let const_idx = find_producer(block, &data_ref)?;
    let const_op = &block.operations[const_idx];
    if const_op.op_type != "const" {
        return None;
    }

    // Verify the const op actually holds a tensor (in inputs or attributes).
    let has_tensor = const_op
        .inputs
        .get("val")
        .or_else(|| const_op.attributes.get("val"))
        .is_some_and(|v| matches!(v, Value::Tensor { .. }));

    if !has_tensor {
        return None;
    }

    Some(CodebookGather {
        const_idx,
        gather_idx,
        gather_output,
        indices_ref,
    })
}

/// Walk backward through a chain of `add` ops to collect all leaf gather
/// outputs and the intermediate add ops.
///
/// Starting from an `add` op, we recursively trace both inputs. If an input
/// is produced by another `add` op (that is single-consumer to this chain),
/// we recurse into it. Otherwise we expect it to be a gather output.
fn collect_add_chain_leaves(
    block: &Block,
    add_idx: usize,
    gather_outputs: &[String],
    add_indices: &mut Vec<usize>,
    leaf_gather_outputs: &mut Vec<String>,
) -> bool {
    let add_op = &block.operations[add_idx];
    if add_op.op_type != "add" {
        return false;
    }

    add_indices.push(add_idx);

    for input_key in &["x", "y"] {
        let ref_name = match add_op.inputs.get(*input_key) {
            Some(Value::Reference(name)) => name.clone(),
            _ => return false,
        };

        if gather_outputs.contains(&ref_name) {
            leaf_gather_outputs.push(ref_name);
        } else if let Some(producer_idx) = find_producer(block, &ref_name) {
            let producer_op = &block.operations[producer_idx];
            if producer_op.op_type == "add"
                && is_internal_single_consumer(block, &ref_name, add_idx)
            {
                if !collect_add_chain_leaves(
                    block,
                    producer_idx,
                    gather_outputs,
                    add_indices,
                    leaf_gather_outputs,
                ) {
                    return false;
                }
            } else {
                return false;
            }
        } else {
            return false;
        }
    }

    true
}

/// Detect RVQ decode patterns in the block.
///
/// Strategy:
/// 1. Find all codebook gathers (gather ops with static const data).
/// 2. For each `add` op, check if it is the root of a chain summing ≥ 2
///    codebook gather outputs.
/// 3. Collect non-overlapping patterns.
fn detect_rvq_patterns(block: &Block) -> Vec<RvqPattern> {
    // Step 1: identify all codebook gathers.
    let codebook_gathers: Vec<CodebookGather> = block
        .operations
        .iter()
        .enumerate()
        .filter_map(|(idx, _)| identify_codebook_gather(block, idx))
        .collect();

    if codebook_gathers.len() < 2 {
        return Vec::new();
    }

    let gather_output_names: Vec<String> = codebook_gathers
        .iter()
        .map(|g| g.gather_output.clone())
        .collect();

    // Step 2: find add ops that could be the root of a summation chain.
    // A root add op is one that is NOT consumed by another add op in the
    // chain (i.e., it is the outermost add).
    let add_indices: Vec<usize> = block
        .operations
        .iter()
        .enumerate()
        .filter(|(_, op)| op.op_type == "add")
        .map(|(i, _)| i)
        .collect();

    let mut patterns: Vec<RvqPattern> = Vec::new();
    let mut claimed_ops: Vec<usize> = Vec::new();

    // Try each add op as a potential root, preferring later (outermost) ones.
    for &root_add_idx in add_indices.iter().rev() {
        if claimed_ops.contains(&root_add_idx) {
            continue;
        }

        let mut chain_add_indices = Vec::new();
        let mut leaf_outputs = Vec::new();

        let ok = collect_add_chain_leaves(
            block,
            root_add_idx,
            &gather_output_names,
            &mut chain_add_indices,
            &mut leaf_outputs,
        );

        if !ok || leaf_outputs.len() < 2 {
            continue;
        }

        // Deduplicate (a gather output should not appear twice).
        leaf_outputs.sort();
        leaf_outputs.dedup();
        if leaf_outputs.len() < 2 {
            continue;
        }

        // Verify all intermediate add outputs are single-consumer within
        // the chain (except the root, whose output goes downstream).
        let root_output = block.operations[root_add_idx]
            .outputs
            .first()
            .cloned()
            .unwrap_or_default();

        let intermediates_safe = chain_add_indices.iter().all(|&ai| {
            if ai == root_add_idx {
                return true;
            }
            let out = match block.operations[ai].outputs.first() {
                Some(name) => name.as_str(),
                None => return false,
            };
            is_internal_single_consumer(block, out, root_add_idx)
                || chain_add_indices
                    .iter()
                    .any(|&other| other != ai && is_internal_single_consumer(block, out, other))
        });

        if !intermediates_safe {
            continue;
        }

        // Collect the matching codebook gathers.
        let matched_gathers: Vec<&CodebookGather> = codebook_gathers
            .iter()
            .filter(|g| leaf_outputs.contains(&g.gather_output))
            .collect();

        // Check no op index conflicts with already-claimed patterns.
        let all_indices: Vec<usize> = matched_gathers
            .iter()
            .flat_map(|g| [g.const_idx, g.gather_idx])
            .chain(chain_add_indices.iter().copied())
            .collect();

        if all_indices.iter().any(|i| claimed_ops.contains(i)) {
            continue;
        }

        // Verify each gather output is single-consumer into the add chain.
        let gathers_safe = matched_gathers.iter().all(|g| {
            if block.outputs.contains(&g.gather_output) {
                return false;
            }
            let consumers = find_consumers(block, &g.gather_output);
            consumers.len() == 1 && chain_add_indices.contains(&consumers[0])
        });

        if !gathers_safe {
            continue;
        }

        // Verify each const output is single-consumer to its gather.
        let consts_safe = matched_gathers.iter().all(|g| {
            let const_output = match block.operations[g.const_idx].outputs.first() {
                Some(name) => name.as_str(),
                None => return false,
            };
            is_internal_single_consumer(block, const_output, g.gather_idx)
        });

        if !consts_safe {
            continue;
        }

        claimed_ops.extend_from_slice(&all_indices);

        // Build the owned gathers for the pattern.
        let owned_gathers: Vec<CodebookGather> = matched_gathers
            .iter()
            .map(|g| CodebookGather {
                const_idx: g.const_idx,
                gather_idx: g.gather_idx,
                gather_output: g.gather_output.clone(),
                indices_ref: g.indices_ref.clone(),
            })
            .collect();

        patterns.push(RvqPattern {
            gathers: owned_gathers,
            add_indices: chain_add_indices,
            final_output: root_output,
        });
    }

    patterns
}

// ---------------------------------------------------------------------------
// Optimization
// ---------------------------------------------------------------------------

/// Stack multiple codebook tensors into a single tensor with an extra leading
/// dimension: [num_codebooks, codebook_size, embedding_dim].
///
/// Returns the stacked tensor data, the combined shape, and the scalar type.
/// Returns `None` if the codebooks have inconsistent shapes or types.
fn stack_codebook_tensors(block: &Block, gathers: &[CodebookGather]) -> Option<Value> {
    let mut all_data: Vec<u8> = Vec::new();
    let mut first_shape: Option<Vec<usize>> = None;
    let mut first_dtype: Option<ScalarType> = None;
    let num_codebooks = gathers.len();

    for g in gathers {
        let const_op = &block.operations[g.const_idx];
        let val = const_op
            .inputs
            .get("val")
            .or_else(|| const_op.attributes.get("val"))?;

        match val {
            Value::Tensor { data, shape, dtype } => {
                if let Some(ref fs) = first_shape {
                    if shape != fs {
                        return None; // inconsistent shapes
                    }
                } else {
                    first_shape = Some(shape.clone());
                }
                if let Some(fd) = first_dtype {
                    if *dtype != fd {
                        return None; // inconsistent dtypes
                    }
                } else {
                    first_dtype = Some(*dtype);
                }
                all_data.extend_from_slice(data);
            }
            _ => return None,
        }
    }

    let inner_shape = first_shape?;
    let dtype = first_dtype?;

    // Stacked shape: [num_codebooks, ...inner_shape]
    let mut stacked_shape = vec![num_codebooks];
    stacked_shape.extend_from_slice(&inner_shape);

    Some(Value::Tensor {
        data: all_data,
        shape: stacked_shape,
        dtype,
    })
}

/// Compute total codebook data size in bytes.
fn total_codebook_bytes(block: &Block, gathers: &[CodebookGather]) -> usize {
    gathers
        .iter()
        .filter_map(|g| {
            let const_op = &block.operations[g.const_idx];
            let val = const_op
                .inputs
                .get("val")
                .or_else(|| const_op.attributes.get("val"))?;
            match val {
                Value::Tensor { data, .. } => Some(data.len()),
                _ => None,
            }
        })
        .sum()
}

/// Optimize detected RVQ patterns in the block.
fn optimize_codebook_patterns(block: &mut Block) {
    let patterns = detect_rvq_patterns(block);
    if patterns.is_empty() {
        return;
    }

    let mut remove_indices: Vec<usize> = Vec::new();

    for (pat_num, pattern) in patterns.iter().enumerate() {
        let num_codebooks = pattern.gathers.len();

        // Stack codebook tensors into a single tensor.
        let stacked_codebooks = match stack_codebook_tensors(block, &pattern.gathers) {
            Some(v) => v,
            None => continue, // skip if shapes/types are inconsistent
        };

        // Determine if codebooks should be tagged for palettization.
        let should_palettize =
            total_codebook_bytes(block, &pattern.gathers) > PALETTIZE_THRESHOLD_BYTES;

        // Build indices list: references to each codebook's indices input.
        let indices_list: Vec<Value> = pattern
            .gathers
            .iter()
            .map(|g| Value::Reference(g.indices_ref.clone()))
            .collect();

        // Create the fused codebook_gather op.
        let fused_name = format!("codebook_gather_{pat_num}");
        let fused_output = format!("codebook_gather_out_{pat_num}");

        let mut fused_op = Operation::new("codebook_gather", &fused_name)
            .with_input("codebooks", stacked_codebooks)
            .with_input("indices", Value::List(indices_list))
            .with_output(&fused_output)
            .with_attr("num_codebooks", Value::Int(num_codebooks as i64));

        if should_palettize {
            fused_op
                .attributes
                .insert("palettize_hint".to_string(), Value::Bool(true));
        }

        // Rewire downstream references.
        replace_reference(block, &pattern.final_output, &fused_output);

        // Collect indices to remove.
        for g in &pattern.gathers {
            remove_indices.push(g.const_idx);
            remove_indices.push(g.gather_idx);
        }
        remove_indices.extend_from_slice(&pattern.add_indices);

        // Append fused op.
        block.operations.push(fused_op);
    }

    // Remove old ops in reverse index order.
    remove_indices.sort_unstable();
    remove_indices.dedup();
    for idx in remove_indices.into_iter().rev() {
        block.operations.remove(idx);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use mil_rs::ir::Function;

    /// Helper: build a minimal program with a single "main" function.
    fn program_with_block(block: Block) -> Program {
        let mut func = Function::new("main");
        func.body = block;
        let mut program = Program::new("1.0.0");
        program.add_function(func);
        program
    }

    fn block_ops(program: &Program) -> &[Operation] {
        &program.functions["main"].body.operations
    }

    /// Create FP32 tensor bytes from a slice of f32 values.
    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Build a codebook const op with the given shape and values.
    fn codebook_const(name: &str, output: &str, values: &[f32], shape: Vec<usize>) -> Operation {
        Operation::new("const", name)
            .with_input(
                "val",
                Value::Tensor {
                    data: f32_bytes(values),
                    shape,
                    dtype: ScalarType::Float32,
                },
            )
            .with_output(output)
    }

    /// Build a simple 2-codebook RVQ decode block:
    ///   const_0, const_1 → gather_0, gather_1 → add → output
    fn build_rvq_2_codebook_block() -> Block {
        // Each codebook: 4 entries × 2 dims = 8 floats
        let cb_values: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();

        let mut block = Block::new();

        // Codebook constants
        block.add_op(codebook_const("cb_0", "cb_0_out", &cb_values, vec![4, 2]));
        block.add_op(codebook_const("cb_1", "cb_1_out", &cb_values, vec![4, 2]));

        // Gather ops
        block.add_op(
            Operation::new("gather", "gather_0")
                .with_input("x", Value::Reference("cb_0_out".into()))
                .with_input("indices", Value::Reference("indices_0".into()))
                .with_output("emb_0"),
        );
        block.add_op(
            Operation::new("gather", "gather_1")
                .with_input("x", Value::Reference("cb_1_out".into()))
                .with_input("indices", Value::Reference("indices_1".into()))
                .with_output("emb_1"),
        );

        // Sum
        block.add_op(
            Operation::new("add", "sum_01")
                .with_input("x", Value::Reference("emb_0".into()))
                .with_input("y", Value::Reference("emb_1".into()))
                .with_output("rvq_output"),
        );

        block.outputs.push("rvq_output".into());
        block
    }

    // ---- Basic 2-codebook RVQ fusion ----------------------------------------

    #[test]
    fn two_codebook_rvq_fused() {
        let block = build_rvq_2_codebook_block();
        let mut program = program_with_block(block);

        CodebookOptimizationPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        // Should have a single codebook_gather op.
        assert_eq!(ops.len(), 1, "expected 1 fused op, got {ops:?}");
        assert_eq!(ops[0].op_type, "codebook_gather");
        assert_eq!(ops[0].name, "codebook_gather_0");

        // Verify num_codebooks attribute.
        assert_eq!(ops[0].attributes.get("num_codebooks"), Some(&Value::Int(2)));

        // Verify the stacked codebooks tensor input.
        match ops[0].inputs.get("codebooks") {
            Some(Value::Tensor { shape, dtype, .. }) => {
                assert_eq!(*shape, vec![2, 4, 2]); // [num_codebooks, entries, dims]
                assert_eq!(*dtype, ScalarType::Float32);
            }
            other => panic!("expected stacked codebook tensor, got {other:?}"),
        }

        // Verify indices list input.
        match ops[0].inputs.get("indices") {
            Some(Value::List(items)) => {
                assert_eq!(items.len(), 2);
                assert_eq!(items[0], Value::Reference("indices_0".into()));
                assert_eq!(items[1], Value::Reference("indices_1".into()));
            }
            other => panic!("expected indices list, got {other:?}"),
        }

        // Block output should reference the fused op's output.
        assert_eq!(
            program.functions["main"].body.outputs,
            vec!["codebook_gather_out_0"]
        );
    }

    // ---- Multi-codebook (4 levels) RVQ detection ----------------------------

    #[test]
    fn four_codebook_rvq_fused() {
        let cb_values: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();

        let mut block = Block::new();

        // 4 codebook constants + gathers
        for i in 0..4 {
            block.add_op(codebook_const(
                &format!("cb_{i}"),
                &format!("cb_{i}_out"),
                &cb_values,
                vec![4, 2],
            ));
        }
        for i in 0..4 {
            block.add_op(
                Operation::new("gather", &format!("gather_{i}"))
                    .with_input("x", Value::Reference(format!("cb_{i}_out")))
                    .with_input("indices", Value::Reference(format!("indices_{i}")))
                    .with_output(format!("emb_{i}")),
            );
        }

        // Chain of adds: ((emb_0 + emb_1) + emb_2) + emb_3
        block.add_op(
            Operation::new("add", "sum_01")
                .with_input("x", Value::Reference("emb_0".into()))
                .with_input("y", Value::Reference("emb_1".into()))
                .with_output("sum_01_out"),
        );
        block.add_op(
            Operation::new("add", "sum_012")
                .with_input("x", Value::Reference("sum_01_out".into()))
                .with_input("y", Value::Reference("emb_2".into()))
                .with_output("sum_012_out"),
        );
        block.add_op(
            Operation::new("add", "sum_0123")
                .with_input("x", Value::Reference("sum_012_out".into()))
                .with_input("y", Value::Reference("emb_3".into()))
                .with_output("rvq_output"),
        );

        block.outputs.push("rvq_output".into());

        let mut program = program_with_block(block);
        CodebookOptimizationPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 1, "expected 1 fused op, got {ops:?}");
        assert_eq!(ops[0].op_type, "codebook_gather");

        assert_eq!(ops[0].attributes.get("num_codebooks"), Some(&Value::Int(4)));

        match ops[0].inputs.get("codebooks") {
            Some(Value::Tensor { shape, .. }) => {
                assert_eq!(*shape, vec![4, 4, 2]);
            }
            other => panic!("expected stacked codebook tensor, got {other:?}"),
        }

        match ops[0].inputs.get("indices") {
            Some(Value::List(items)) => {
                assert_eq!(items.len(), 4);
            }
            other => panic!("expected indices list with 4 entries, got {other:?}"),
        }
    }

    // ---- No fusion when pattern is incomplete --------------------------------

    #[test]
    fn no_fusion_single_gather() {
        let cb_values: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();

        let mut block = Block::new();
        block.add_op(codebook_const("cb_0", "cb_0_out", &cb_values, vec![4, 2]));
        block.add_op(
            Operation::new("gather", "gather_0")
                .with_input("x", Value::Reference("cb_0_out".into()))
                .with_input("indices", Value::Reference("indices_0".into()))
                .with_output("emb_0"),
        );
        block.outputs.push("emb_0".into());

        let mut program = program_with_block(block);
        CodebookOptimizationPass.run(&mut program).unwrap();

        // Single gather — not an RVQ pattern, should not be fused.
        assert_eq!(block_ops(&program).len(), 2);
    }

    // ---- No fusion when gather output is a block output ---------------------

    #[test]
    fn no_fusion_when_gather_is_block_output() {
        let block = {
            let mut b = build_rvq_2_codebook_block();
            // Also export emb_0 as a block output → multi-consumer.
            b.outputs.push("emb_0".into());
            b
        };

        let mut program = program_with_block(block);
        CodebookOptimizationPass.run(&mut program).unwrap();

        // emb_0 is a block output, so the gather is multi-consumer → no fusion.
        assert_eq!(block_ops(&program).len(), 5);
    }

    // ---- Downstream ops are rewired correctly --------------------------------

    #[test]
    fn downstream_ops_rewired_after_fusion() {
        let mut block = build_rvq_2_codebook_block();

        // Add a downstream consumer of the RVQ output.
        block.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("rvq_output".into()))
                .with_output("relu_out"),
        );
        block.outputs.clear();
        block.outputs.push("relu_out".into());

        let mut program = program_with_block(block);
        CodebookOptimizationPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 2); // codebook_gather + relu

        // relu should reference the fused op's output.
        let relu_op = ops.iter().find(|op| op.op_type == "relu").unwrap();
        match relu_op.inputs.get("x") {
            Some(Value::Reference(name)) => {
                assert_eq!(name, "codebook_gather_out_0");
            }
            other => panic!("expected relu to reference fused output, got {other:?}"),
        }
    }

    // ---- Non-RVQ adds are left alone ----------------------------------------

    #[test]
    fn non_rvq_add_untouched() {
        let mut block = Block::new();

        // A normal add of two non-codebook values.
        block.add_op(
            Operation::new("add", "add_0")
                .with_input("x", Value::Reference("a".into()))
                .with_input("y", Value::Reference("b".into()))
                .with_output("sum_out"),
        );
        block.outputs.push("sum_out".into());

        let mut program = program_with_block(block);
        CodebookOptimizationPass.run(&mut program).unwrap();

        assert_eq!(block_ops(&program).len(), 1);
        assert_eq!(block_ops(&program)[0].op_type, "add");
    }

    // ---- Palettize hint is set for large codebooks ---------------------------

    #[test]
    fn palettize_hint_set_for_large_codebooks() {
        // Create codebooks larger than the threshold.
        // PALETTIZE_THRESHOLD_BYTES = 4096, so we need > 4096 bytes total.
        // Use 2 codebooks of 256 entries × 4 dims = 1024 floats each = 4096 bytes each.
        let cb_values: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();

        let mut block = Block::new();
        block.add_op(codebook_const("cb_0", "cb_0_out", &cb_values, vec![256, 4]));
        block.add_op(codebook_const("cb_1", "cb_1_out", &cb_values, vec![256, 4]));

        block.add_op(
            Operation::new("gather", "gather_0")
                .with_input("x", Value::Reference("cb_0_out".into()))
                .with_input("indices", Value::Reference("indices_0".into()))
                .with_output("emb_0"),
        );
        block.add_op(
            Operation::new("gather", "gather_1")
                .with_input("x", Value::Reference("cb_1_out".into()))
                .with_input("indices", Value::Reference("indices_1".into()))
                .with_output("emb_1"),
        );

        block.add_op(
            Operation::new("add", "sum_01")
                .with_input("x", Value::Reference("emb_0".into()))
                .with_input("y", Value::Reference("emb_1".into()))
                .with_output("rvq_output"),
        );
        block.outputs.push("rvq_output".into());

        let mut program = program_with_block(block);
        CodebookOptimizationPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 1);
        assert_eq!(
            ops[0].attributes.get("palettize_hint"),
            Some(&Value::Bool(true))
        );
    }

    // ---- No palettize hint for small codebooks ------------------------------

    #[test]
    fn no_palettize_hint_for_small_codebooks() {
        let block = build_rvq_2_codebook_block();
        let mut program = program_with_block(block);
        CodebookOptimizationPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 1);
        // Small codebooks → no palettize hint.
        assert_eq!(ops[0].attributes.get("palettize_hint"), None);
    }

    // ---- Inconsistent codebook shapes prevent fusion ------------------------

    #[test]
    fn inconsistent_shapes_no_fusion() {
        let cb_values_a: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let cb_values_b: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();

        let mut block = Block::new();
        block.add_op(codebook_const("cb_0", "cb_0_out", &cb_values_a, vec![4, 2]));
        block.add_op(codebook_const(
            "cb_1",
            "cb_1_out",
            &cb_values_b,
            vec![4, 3], // different embedding dim
        ));

        block.add_op(
            Operation::new("gather", "gather_0")
                .with_input("x", Value::Reference("cb_0_out".into()))
                .with_input("indices", Value::Reference("indices_0".into()))
                .with_output("emb_0"),
        );
        block.add_op(
            Operation::new("gather", "gather_1")
                .with_input("x", Value::Reference("cb_1_out".into()))
                .with_input("indices", Value::Reference("indices_1".into()))
                .with_output("emb_1"),
        );

        block.add_op(
            Operation::new("add", "sum_01")
                .with_input("x", Value::Reference("emb_0".into()))
                .with_input("y", Value::Reference("emb_1".into()))
                .with_output("rvq_output"),
        );
        block.outputs.push("rvq_output".into());

        let mut program = program_with_block(block);
        CodebookOptimizationPass.run(&mut program).unwrap();

        // Inconsistent shapes → pattern detected but stacking fails → no fusion.
        assert_eq!(block_ops(&program).len(), 5);
    }

    // ---- Pass is a no-op on empty programs ----------------------------------

    #[test]
    fn noop_on_empty_program() {
        let mut program = Program::new("1.0.0");
        let func = Function::new("main");
        program.add_function(func);

        CodebookOptimizationPass.run(&mut program).unwrap();
        assert_eq!(block_ops(&program).len(), 0);
    }
}
