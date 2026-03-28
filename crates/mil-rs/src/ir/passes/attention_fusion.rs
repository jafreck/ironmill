//! Attention pattern fusion pass — detect and fuse scaled dot-product
//! attention into a single `scaled_dot_product_attention` op.
//!
//! The standard attention pattern in MIL:
//!
//! ```text
//! scores = matmul(Q, transpose(K))
//! scores_scaled = real_div(scores, sqrt(d_k))   # or mul by 1/√d_k
//! attn_weights = softmax(scores_scaled)
//! output = matmul(attn_weights, V)
//! ```
//!
//! This pass fuses the entire chain into a single
//! `scaled_dot_product_attention` op with Q, K, V inputs. If no attention
//! pattern is found the pass is a no-op, making it safe to run always.

use crate::error::Result;
use crate::ir::pass::Pass;
use crate::ir::program::{Block, Program};
use crate::ir::types::Value;

use super::replace_reference;

/// Fuses scaled dot-product attention patterns into a single op.
pub struct AttentionFusionPass;

impl Pass for AttentionFusionPass {
    fn name(&self) -> &str {
        "attention-fusion"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            fuse_attention_pattern(&mut function.body);
        }
        Ok(())
    }
}

/// Indices of the ops that form a complete attention pattern.
struct AttentionPattern {
    /// Index of `transpose(K)`.
    transpose_idx: usize,
    /// Index of `matmul(Q, K^T)` — the scores matmul.
    scores_matmul_idx: usize,
    /// Index of `real_div` or `mul` — the scaling step.
    scale_idx: usize,
    /// Index of `softmax`.
    softmax_idx: usize,
    /// Index of `matmul(attn_weights, V)` — the output matmul.
    output_matmul_idx: usize,
    /// The Q input value reference name.
    q_ref: String,
    /// The K input value reference name (pre-transpose).
    k_ref: String,
    /// The V input value reference name.
    v_ref: String,
}

/// Follow a `Value::Reference` in an operation's input to find the index of
/// the producing operation in the block.
fn trace_input(block: &Block, op_idx: usize, input_name: &str) -> Option<usize> {
    let op = block.operations.get(op_idx)?;
    let ref_name = match op.inputs.get(input_name) {
        Some(Value::Reference(name)) => name,
        _ => return None,
    };
    block
        .operations
        .iter()
        .position(|o| o.outputs.first().map(|s| s.as_str()) == Some(ref_name.as_str()))
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

/// Attempt to match a full attention pattern anchored at the softmax at
/// `softmax_idx`. Returns `Some(AttentionPattern)` on success.
fn match_attention_at_softmax(block: &Block, softmax_idx: usize) -> Option<AttentionPattern> {
    let softmax_op = &block.operations[softmax_idx];
    if softmax_op.op_type != "softmax" {
        return None;
    }

    // --- Backward from softmax: input should be a scaling op ----------------
    let scale_idx = trace_input(block, softmax_idx, "x")?;
    let scale_op = &block.operations[scale_idx];
    if scale_op.op_type != "real_div" && scale_op.op_type != "mul" {
        return None;
    }

    // --- Backward from scale: first input should be scores matmul -----------
    let scores_matmul_idx = trace_input(block, scale_idx, "x")?;
    let scores_matmul_op = &block.operations[scores_matmul_idx];
    if scores_matmul_op.op_type != "matmul" {
        return None;
    }

    // --- One of the matmul inputs should be a transpose (K^T) ---------------
    // Convention: matmul(Q, K^T) — "x" is Q, "y" is K^T.
    let transpose_idx = trace_input(block, scores_matmul_idx, "y")?;
    let transpose_op = &block.operations[transpose_idx];
    if transpose_op.op_type != "transpose" {
        return None;
    }

    // Q reference: from scores matmul "x" input.
    let q_ref = match scores_matmul_op.inputs.get("x") {
        Some(Value::Reference(name)) => name.clone(),
        _ => return None,
    };

    // K reference: the transpose's own input.
    let k_ref = match transpose_op.inputs.get("x") {
        Some(Value::Reference(name)) => name.clone(),
        _ => return None,
    };

    // --- Forward from softmax: consumer should be the output matmul ---------
    let softmax_output = softmax_op.outputs.first()?;
    let consumers = find_consumers(block, softmax_output);
    if consumers.len() != 1 {
        return None;
    }
    let output_matmul_idx = consumers[0];
    let output_matmul_op = &block.operations[output_matmul_idx];
    if output_matmul_op.op_type != "matmul" {
        return None;
    }

    // The output matmul must consume softmax output in the "x" position,
    // and V in the "y" position.
    match output_matmul_op.inputs.get("x") {
        Some(Value::Reference(name)) if name == softmax_output => {}
        _ => return None,
    }

    let v_ref = match output_matmul_op.inputs.get("y") {
        Some(Value::Reference(name)) => name.clone(),
        _ => return None,
    };

    Some(AttentionPattern {
        transpose_idx,
        scores_matmul_idx,
        scale_idx,
        softmax_idx,
        output_matmul_idx,
        q_ref,
        k_ref,
        v_ref,
    })
}

/// Check that an intermediate value is consumed only by its expected single
/// consumer and is not a block output.
fn is_internal_single_consumer(block: &Block, value_name: &str, expected_consumer: usize) -> bool {
    if block.outputs.contains(&value_name.to_string()) {
        return false;
    }
    let consumers = find_consumers(block, value_name);
    consumers.len() == 1 && consumers[0] == expected_consumer
}

/// Validate that all intermediate values in the pattern are single-consumer,
/// so we can safely remove them.
fn pattern_is_safe_to_fuse(block: &Block, pat: &AttentionPattern) -> bool {
    let transpose_out = match block.operations[pat.transpose_idx].outputs.first() {
        Some(name) => name.as_str(),
        None => return false,
    };
    let scores_out = match block.operations[pat.scores_matmul_idx].outputs.first() {
        Some(name) => name.as_str(),
        None => return false,
    };
    let scale_out = match block.operations[pat.scale_idx].outputs.first() {
        Some(name) => name.as_str(),
        None => return false,
    };
    let softmax_out = match block.operations[pat.softmax_idx].outputs.first() {
        Some(name) => name.as_str(),
        None => return false,
    };

    is_internal_single_consumer(block, transpose_out, pat.scores_matmul_idx)
        && is_internal_single_consumer(block, scores_out, pat.scale_idx)
        && is_internal_single_consumer(block, scale_out, pat.softmax_idx)
        && is_internal_single_consumer(block, softmax_out, pat.output_matmul_idx)
}

/// Scan a block for attention patterns and fuse each one.
fn fuse_attention_pattern(block: &mut Block) {
    // Collect all fusible patterns before mutating.
    let softmax_indices: Vec<usize> = block
        .operations
        .iter()
        .enumerate()
        .filter(|(_, op)| op.op_type == "softmax")
        .map(|(i, _)| i)
        .collect();

    let mut patterns: Vec<AttentionPattern> = Vec::new();

    for si in softmax_indices {
        if let Some(pat) = match_attention_at_softmax(block, si) {
            if !pattern_is_safe_to_fuse(block, &pat) {
                continue;
            }
            // Ensure no op index overlaps with an already-claimed pattern.
            let indices = [
                pat.transpose_idx,
                pat.scores_matmul_idx,
                pat.scale_idx,
                pat.softmax_idx,
                pat.output_matmul_idx,
            ];
            let conflicts = patterns.iter().any(|prev| {
                let prev_indices = [
                    prev.transpose_idx,
                    prev.scores_matmul_idx,
                    prev.scale_idx,
                    prev.softmax_idx,
                    prev.output_matmul_idx,
                ];
                indices.iter().any(|i| prev_indices.contains(i))
            });
            if !conflicts {
                patterns.push(pat);
            }
        }
    }

    if patterns.is_empty() {
        return;
    }

    // Build fused ops and rewire references.
    use crate::ir::operation::Operation;

    for (fusion_num, pat) in patterns.iter().enumerate() {
        let output_matmul_output = match block.operations[pat.output_matmul_idx].outputs.first() {
            Some(name) => name.clone(),
            None => continue,
        };

        // Create the fused op. Its output replaces the output matmul's output.
        let fused_name = format!("attention_{fusion_num}");
        let fused_output = format!("attention_out_{fusion_num}");

        let fused_op = Operation::new("scaled_dot_product_attention", &fused_name)
            .with_input("Q", Value::Reference(pat.q_ref.clone()))
            .with_input("K", Value::Reference(pat.k_ref.clone()))
            .with_input("V", Value::Reference(pat.v_ref.clone()))
            .with_output(&fused_output);

        // Rewire downstream references: anything that pointed at the
        // output matmul's output now points at the fused op's output.
        replace_reference(block, &output_matmul_output, &fused_output);

        // Insert the fused op at the position of the output matmul (it will
        // survive removal since we remove by original indices).
        block.operations.push(fused_op);
    }

    // Collect all indices to remove, sort descending, and remove.
    let mut remove_indices: Vec<usize> = patterns
        .iter()
        .flat_map(|pat| {
            [
                pat.transpose_idx,
                pat.scores_matmul_idx,
                pat.scale_idx,
                pat.softmax_idx,
                pat.output_matmul_idx,
            ]
        })
        .collect();
    remove_indices.sort_unstable();
    remove_indices.dedup();
    for idx in remove_indices.into_iter().rev() {
        block.operations.remove(idx);
    }
}

/// Fuses grouped-query attention (GQA) patterns into a single op.
///
/// GQA is a variant of multi-head attention where fewer K/V heads are
/// shared across Q heads. The pattern includes a `repeat_interleave`,
/// `tile`, or `expand` op on K and/or V before the standard attention
/// pattern to broadcast the fewer KV heads to match Q's head count.
///
/// ```text
/// K_expanded = repeat_interleave(K)   # or tile / expand
/// V_expanded = repeat_interleave(V)
/// scores = matmul(Q, transpose(K_expanded))
/// scores_scaled = real_div(scores, sqrt(d_k))
/// attn_weights = softmax(scores_scaled)
/// output = matmul(attn_weights, V_expanded)
/// ```
pub struct GqaFusionPass;

impl Pass for GqaFusionPass {
    fn name(&self) -> &str {
        "gqa-fusion"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            fuse_gqa_pattern(&mut function.body);
        }
        Ok(())
    }
}

/// Broadcast/repeat op types used in GQA to expand K/V heads.
const GQA_BROADCAST_OPS: &[&str] = &["repeat_interleave", "tile", "expand"];

/// Indices of ops forming a GQA pattern.
struct GqaPattern {
    /// Optional: index of broadcast op for K.
    k_broadcast_idx: Option<usize>,
    /// Optional: index of broadcast op for V.
    v_broadcast_idx: Option<usize>,
    /// Index of `transpose(K)` or `transpose(K_expanded)`.
    transpose_idx: usize,
    /// Index of `matmul(Q, K^T)`.
    scores_matmul_idx: usize,
    /// Index of scaling op (`real_div` or `mul`).
    scale_idx: usize,
    /// Index of `softmax`.
    softmax_idx: usize,
    /// Index of `matmul(attn_weights, V)`.
    output_matmul_idx: usize,
    /// Original K input (pre-broadcast).
    k_ref: String,
    /// Original V input (pre-broadcast).
    v_ref: String,
    /// Q input reference.
    q_ref: String,
}

/// Try to match a GQA pattern anchored at a softmax op.
fn match_gqa_at_softmax(block: &Block, softmax_idx: usize) -> Option<GqaPattern> {
    let softmax_op = &block.operations[softmax_idx];
    if softmax_op.op_type != "softmax" {
        return None;
    }

    // Backward from softmax: scaling op.
    let scale_idx = trace_input(block, softmax_idx, "x")?;
    let scale_op = &block.operations[scale_idx];
    if scale_op.op_type != "real_div" && scale_op.op_type != "mul" {
        return None;
    }

    // Backward from scale: scores matmul.
    let scores_matmul_idx = trace_input(block, scale_idx, "x")?;
    let scores_matmul_op = &block.operations[scores_matmul_idx];
    if scores_matmul_op.op_type != "matmul" {
        return None;
    }

    // Y input of scores matmul should be transpose.
    let transpose_idx = trace_input(block, scores_matmul_idx, "y")?;
    let transpose_op = &block.operations[transpose_idx];
    if transpose_op.op_type != "transpose" {
        return None;
    }

    // Q reference.
    let q_ref = match scores_matmul_op.inputs.get("x") {
        Some(Value::Reference(name)) => name.clone(),
        _ => return None,
    };

    // Check if transpose input comes from a broadcast op (K path).
    let transpose_input_idx = trace_input(block, transpose_idx, "x")?;
    let transpose_input_op = &block.operations[transpose_input_idx];
    let (k_broadcast_idx, k_ref) =
        if GQA_BROADCAST_OPS.contains(&transpose_input_op.op_type.as_str()) {
            // K goes through a broadcast before transpose.
            let original_k = match transpose_input_op.inputs.get("x") {
                Some(Value::Reference(name)) => name.clone(),
                _ => return None,
            };
            (Some(transpose_input_idx), original_k)
        } else {
            // No broadcast — this is regular attention, not GQA.
            return None;
        };

    // Forward from softmax: output matmul.
    let softmax_output = softmax_op.outputs.first()?;
    let consumers = find_consumers(block, softmax_output);
    if consumers.len() != 1 {
        return None;
    }
    let output_matmul_idx = consumers[0];
    let output_matmul_op = &block.operations[output_matmul_idx];
    if output_matmul_op.op_type != "matmul" {
        return None;
    }

    match output_matmul_op.inputs.get("x") {
        Some(Value::Reference(name)) if name == softmax_output => {}
        _ => return None,
    }

    // Check V path: the output matmul's "y" input may come from a broadcast.
    let v_input_ref = match output_matmul_op.inputs.get("y") {
        Some(Value::Reference(name)) => name.clone(),
        _ => return None,
    };

    let v_producer = block
        .operations
        .iter()
        .enumerate()
        .find(|(_, op)| op.outputs.first().map(|s| s.as_str()) == Some(v_input_ref.as_str()));

    let (v_broadcast_idx, v_ref) = match v_producer {
        Some((idx, op)) if GQA_BROADCAST_OPS.contains(&op.op_type.as_str()) => {
            let original_v = match op.inputs.get("x") {
                Some(Value::Reference(name)) => name.clone(),
                _ => return None,
            };
            (Some(idx), original_v)
        }
        _ => {
            // V has no broadcast but K does — still a valid GQA variant
            // (some implementations only broadcast K).
            (None, v_input_ref)
        }
    };

    Some(GqaPattern {
        k_broadcast_idx,
        v_broadcast_idx,
        transpose_idx,
        scores_matmul_idx,
        scale_idx,
        softmax_idx,
        output_matmul_idx,
        k_ref,
        v_ref,
        q_ref,
    })
}

/// Collect all op indices belonging to a GQA pattern.
fn gqa_pattern_indices(pat: &GqaPattern) -> Vec<usize> {
    let mut indices = vec![
        pat.transpose_idx,
        pat.scores_matmul_idx,
        pat.scale_idx,
        pat.softmax_idx,
        pat.output_matmul_idx,
    ];
    if let Some(idx) = pat.k_broadcast_idx {
        indices.push(idx);
    }
    if let Some(idx) = pat.v_broadcast_idx {
        indices.push(idx);
    }
    indices
}

/// Validate all intermediates in the GQA pattern are single-consumer.
fn gqa_pattern_is_safe_to_fuse(block: &Block, pat: &GqaPattern) -> bool {
    // Check broadcast → transpose chain for K.
    if let Some(k_bc_idx) = pat.k_broadcast_idx {
        let k_bc_out = match block.operations[k_bc_idx].outputs.first() {
            Some(name) => name.as_str(),
            None => return false,
        };
        if !is_internal_single_consumer(block, k_bc_out, pat.transpose_idx) {
            return false;
        }
    }

    let transpose_out = match block.operations[pat.transpose_idx].outputs.first() {
        Some(name) => name.as_str(),
        None => return false,
    };
    let scores_out = match block.operations[pat.scores_matmul_idx].outputs.first() {
        Some(name) => name.as_str(),
        None => return false,
    };
    let scale_out = match block.operations[pat.scale_idx].outputs.first() {
        Some(name) => name.as_str(),
        None => return false,
    };
    let softmax_out = match block.operations[pat.softmax_idx].outputs.first() {
        Some(name) => name.as_str(),
        None => return false,
    };

    if !is_internal_single_consumer(block, transpose_out, pat.scores_matmul_idx) {
        return false;
    }
    if !is_internal_single_consumer(block, scores_out, pat.scale_idx) {
        return false;
    }
    if !is_internal_single_consumer(block, scale_out, pat.softmax_idx) {
        return false;
    }
    if !is_internal_single_consumer(block, softmax_out, pat.output_matmul_idx) {
        return false;
    }

    // Check V broadcast if present.
    if let Some(v_bc_idx) = pat.v_broadcast_idx {
        let v_bc_out = match block.operations[v_bc_idx].outputs.first() {
            Some(name) => name.as_str(),
            None => return false,
        };
        if !is_internal_single_consumer(block, v_bc_out, pat.output_matmul_idx) {
            return false;
        }
    }

    true
}

/// Scan for GQA patterns and fuse each into a single
/// `grouped_query_attention` op.
fn fuse_gqa_pattern(block: &mut Block) {
    let softmax_indices: Vec<usize> = block
        .operations
        .iter()
        .enumerate()
        .filter(|(_, op)| op.op_type == "softmax")
        .map(|(i, _)| i)
        .collect();

    let mut patterns: Vec<GqaPattern> = Vec::new();

    for si in softmax_indices {
        if let Some(pat) = match_gqa_at_softmax(block, si) {
            if !gqa_pattern_is_safe_to_fuse(block, &pat) {
                continue;
            }
            let indices = gqa_pattern_indices(&pat);
            let conflicts = patterns.iter().any(|prev| {
                let prev_indices = gqa_pattern_indices(prev);
                indices.iter().any(|i| prev_indices.contains(i))
            });
            if !conflicts {
                patterns.push(pat);
            }
        }
    }

    if patterns.is_empty() {
        return;
    }

    use crate::ir::operation::Operation;

    // Collect fused ops and replacements first to avoid corrupting
    // earlier fused ops when multiple patterns share input references.
    let mut fused_ops = Vec::new();
    let mut replacements = Vec::new();

    for (fusion_num, pat) in patterns.iter().enumerate() {
        let output_matmul_output = match block.operations[pat.output_matmul_idx].outputs.first() {
            Some(name) => name.clone(),
            None => continue,
        };

        let fused_name = format!("gqa_{fusion_num}");
        let fused_output = format!("gqa_out_{fusion_num}");

        let fused_op = Operation::new("grouped_query_attention", &fused_name)
            .with_input("Q", Value::Reference(pat.q_ref.clone()))
            .with_input("K", Value::Reference(pat.k_ref.clone()))
            .with_input("V", Value::Reference(pat.v_ref.clone()))
            .with_attr("is_gqa", Value::Bool(true))
            .with_output(&fused_output);

        replacements.push((output_matmul_output, fused_output));
        fused_ops.push(fused_op);
    }

    for (old_name, new_name) in &replacements {
        replace_reference(block, old_name, new_name);
    }

    for fused_op in fused_ops {
        block.operations.push(fused_op);
    }

    let mut remove_indices: Vec<usize> = patterns.iter().flat_map(gqa_pattern_indices).collect();
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
    use crate::ir::operation::Operation;
    use crate::ir::program::Function;

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

    /// Build the canonical attention pattern:
    ///   transpose(K) → matmul(Q, K^T) → real_div(scores, scale) → softmax → matmul(attn, V)
    fn build_attention_block() -> Block {
        let mut block = Block::new();
        block.add_op(
            Operation::new("transpose", "transpose_k")
                .with_input("x", Value::Reference("K".into()))
                .with_output("K_T"),
        );
        block.add_op(
            Operation::new("matmul", "scores_matmul")
                .with_input("x", Value::Reference("Q".into()))
                .with_input("y", Value::Reference("K_T".into()))
                .with_output("scores"),
        );
        block.add_op(
            Operation::new("real_div", "scale")
                .with_input("x", Value::Reference("scores".into()))
                .with_input("y", Value::Float(8.0))
                .with_output("scores_scaled"),
        );
        block.add_op(
            Operation::new("softmax", "softmax_0")
                .with_input("x", Value::Reference("scores_scaled".into()))
                .with_output("attn_weights"),
        );
        block.add_op(
            Operation::new("matmul", "output_matmul")
                .with_input("x", Value::Reference("attn_weights".into()))
                .with_input("y", Value::Reference("V".into()))
                .with_output("attn_output"),
        );
        block.outputs.push("attn_output".into());
        block
    }

    // ---- Full attention pattern fused --------------------------------------

    #[test]
    fn full_attention_pattern_fused() {
        let block = build_attention_block();
        let mut program = program_with_block(block);
        AttentionFusionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 1, "expected a single fused op, got {ops:?}");
        assert_eq!(ops[0].op_type, "scaled_dot_product_attention");

        // Verify inputs.
        assert_eq!(ops[0].inputs.get("Q"), Some(&Value::Reference("Q".into())));
        assert_eq!(ops[0].inputs.get("K"), Some(&Value::Reference("K".into())));
        assert_eq!(ops[0].inputs.get("V"), Some(&Value::Reference("V".into())));

        // Block output should reference the fused op's output.
        let block_outputs = &program.functions["main"].body.outputs;
        assert_eq!(block_outputs, &ops[0].outputs);
    }

    // ---- Attention with mul scaling ----------------------------------------

    #[test]
    fn attention_pattern_with_mul_scaling_fused() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("transpose", "transpose_k")
                .with_input("x", Value::Reference("K".into()))
                .with_output("K_T"),
        );
        block.add_op(
            Operation::new("matmul", "scores_matmul")
                .with_input("x", Value::Reference("Q".into()))
                .with_input("y", Value::Reference("K_T".into()))
                .with_output("scores"),
        );
        // Use mul instead of real_div for scaling.
        block.add_op(
            Operation::new("mul", "scale")
                .with_input("x", Value::Reference("scores".into()))
                .with_input("y", Value::Float(0.125))
                .with_output("scores_scaled"),
        );
        block.add_op(
            Operation::new("softmax", "softmax_0")
                .with_input("x", Value::Reference("scores_scaled".into()))
                .with_output("attn_weights"),
        );
        block.add_op(
            Operation::new("matmul", "output_matmul")
                .with_input("x", Value::Reference("attn_weights".into()))
                .with_input("y", Value::Reference("V".into()))
                .with_output("attn_output"),
        );
        block.outputs.push("attn_output".into());

        let mut program = program_with_block(block);
        AttentionFusionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op_type, "scaled_dot_product_attention");
    }

    // ---- Partial pattern: missing softmax → no fusion ----------------------

    #[test]
    fn partial_pattern_no_softmax_no_fusion() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("transpose", "transpose_k")
                .with_input("x", Value::Reference("K".into()))
                .with_output("K_T"),
        );
        block.add_op(
            Operation::new("matmul", "scores_matmul")
                .with_input("x", Value::Reference("Q".into()))
                .with_input("y", Value::Reference("K_T".into()))
                .with_output("scores"),
        );
        block.add_op(
            Operation::new("real_div", "scale")
                .with_input("x", Value::Reference("scores".into()))
                .with_input("y", Value::Float(8.0))
                .with_output("scores_scaled"),
        );
        // No softmax — directly feed into output matmul.
        block.add_op(
            Operation::new("matmul", "output_matmul")
                .with_input("x", Value::Reference("scores_scaled".into()))
                .with_input("y", Value::Reference("V".into()))
                .with_output("attn_output"),
        );
        block.outputs.push("attn_output".into());

        let mut program = program_with_block(block);
        AttentionFusionPass.run(&mut program).unwrap();

        // No softmax → pass is a no-op.
        assert_eq!(block_ops(&program).len(), 4);
    }

    // ---- Multiple attention heads fused independently ----------------------

    #[test]
    fn multiple_attention_heads_fused() {
        let mut block = Block::new();

        // Head 0
        block.add_op(
            Operation::new("transpose", "transpose_k_0")
                .with_input("x", Value::Reference("K0".into()))
                .with_output("K_T_0"),
        );
        block.add_op(
            Operation::new("matmul", "scores_matmul_0")
                .with_input("x", Value::Reference("Q0".into()))
                .with_input("y", Value::Reference("K_T_0".into()))
                .with_output("scores_0"),
        );
        block.add_op(
            Operation::new("real_div", "scale_0")
                .with_input("x", Value::Reference("scores_0".into()))
                .with_input("y", Value::Float(8.0))
                .with_output("scores_scaled_0"),
        );
        block.add_op(
            Operation::new("softmax", "softmax_0")
                .with_input("x", Value::Reference("scores_scaled_0".into()))
                .with_output("attn_weights_0"),
        );
        block.add_op(
            Operation::new("matmul", "output_matmul_0")
                .with_input("x", Value::Reference("attn_weights_0".into()))
                .with_input("y", Value::Reference("V0".into()))
                .with_output("attn_output_0"),
        );

        // Head 1
        block.add_op(
            Operation::new("transpose", "transpose_k_1")
                .with_input("x", Value::Reference("K1".into()))
                .with_output("K_T_1"),
        );
        block.add_op(
            Operation::new("matmul", "scores_matmul_1")
                .with_input("x", Value::Reference("Q1".into()))
                .with_input("y", Value::Reference("K_T_1".into()))
                .with_output("scores_1"),
        );
        block.add_op(
            Operation::new("real_div", "scale_1")
                .with_input("x", Value::Reference("scores_1".into()))
                .with_input("y", Value::Float(8.0))
                .with_output("scores_scaled_1"),
        );
        block.add_op(
            Operation::new("softmax", "softmax_1")
                .with_input("x", Value::Reference("scores_scaled_1".into()))
                .with_output("attn_weights_1"),
        );
        block.add_op(
            Operation::new("matmul", "output_matmul_1")
                .with_input("x", Value::Reference("attn_weights_1".into()))
                .with_input("y", Value::Reference("V1".into()))
                .with_output("attn_output_1"),
        );

        // Combine heads (not part of attention pattern).
        block.add_op(
            Operation::new("concat", "concat_heads")
                .with_input(
                    "x",
                    Value::List(vec![
                        Value::Reference("attn_output_0".into()),
                        Value::Reference("attn_output_1".into()),
                    ]),
                )
                .with_output("combined"),
        );
        block.outputs.push("combined".into());

        let mut program = program_with_block(block);
        AttentionFusionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        // 2 fused attention ops + 1 concat = 3 ops.
        assert_eq!(
            ops.len(),
            3,
            "expected 3 ops after fusing two heads, got {ops:?}"
        );

        let attn_ops: Vec<_> = ops
            .iter()
            .filter(|op| op.op_type == "scaled_dot_product_attention")
            .collect();
        assert_eq!(attn_ops.len(), 2);

        let concat_op = ops.iter().find(|op| op.op_type == "concat").unwrap();
        assert_eq!(concat_op.name, "concat_heads");
    }

    // ---- Non-attention softmax → not fused ---------------------------------

    #[test]
    fn non_attention_softmax_not_fused() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("linear", "linear_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("logits"),
        );
        block.add_op(
            Operation::new("softmax", "softmax_0")
                .with_input("x", Value::Reference("logits".into()))
                .with_output("probs"),
        );
        block.outputs.push("probs".into());

        let mut program = program_with_block(block);
        AttentionFusionPass.run(&mut program).unwrap();

        // Softmax input is linear, not real_div/mul → no fusion.
        let ops = block_ops(&program);
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].op_type, "linear");
        assert_eq!(ops[1].op_type, "softmax");
    }

    // ---- Downstream ops rewired after fusion -------------------------------

    #[test]
    fn downstream_ops_rewired_after_attention_fusion() {
        let mut block = build_attention_block();
        // Add a downstream op that consumes the attention output.
        block.add_op(
            Operation::new("linear", "proj")
                .with_input("x", Value::Reference("attn_output".into()))
                .with_output("proj_output"),
        );
        // Change block output to the projection.
        block.outputs = vec!["proj_output".into()];

        let mut program = program_with_block(block);
        AttentionFusionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].op_type, "linear");
        assert_eq!(ops[1].op_type, "scaled_dot_product_attention");

        // The linear should now reference the fused op's output.
        let fused_output = ops[1]
            .outputs
            .first()
            .expect("fused op should have an output");
        if let Some(Value::Reference(name)) = ops[0].inputs.get("x") {
            assert_eq!(name, fused_output);
        } else {
            panic!("expected linear input to be a reference to fused output");
        }
    }

    // ---- Multi-consumer intermediate blocks fusion -------------------------

    #[test]
    fn no_fusion_when_intermediate_has_multiple_consumers() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("transpose", "transpose_k")
                .with_input("x", Value::Reference("K".into()))
                .with_output("K_T"),
        );
        block.add_op(
            Operation::new("matmul", "scores_matmul")
                .with_input("x", Value::Reference("Q".into()))
                .with_input("y", Value::Reference("K_T".into()))
                .with_output("scores"),
        );
        block.add_op(
            Operation::new("real_div", "scale")
                .with_input("x", Value::Reference("scores".into()))
                .with_input("y", Value::Float(8.0))
                .with_output("scores_scaled"),
        );
        block.add_op(
            Operation::new("softmax", "softmax_0")
                .with_input("x", Value::Reference("scores_scaled".into()))
                .with_output("attn_weights"),
        );
        block.add_op(
            Operation::new("matmul", "output_matmul")
                .with_input("x", Value::Reference("attn_weights".into()))
                .with_input("y", Value::Reference("V".into()))
                .with_output("attn_output"),
        );
        // Extra consumer of scores_scaled → intermediate is multi-consumer.
        block.add_op(
            Operation::new("add", "extra_consumer")
                .with_input("x", Value::Reference("scores_scaled".into()))
                .with_input("y", Value::Float(1.0))
                .with_output("extra_out"),
        );
        block.outputs.push("attn_output".into());
        block.outputs.push("extra_out".into());

        let mut program = program_with_block(block);
        AttentionFusionPass.run(&mut program).unwrap();

        // scores_scaled has two consumers → no fusion.
        assert_eq!(block_ops(&program).len(), 6);
    }

    // ---- GQA fusion --------------------------------------------------------

    /// Build a GQA pattern with repeat_interleave on K and V.
    fn build_gqa_block() -> Block {
        let mut block = Block::new();
        // K broadcast: repeat_interleave to expand KV heads.
        block.add_op(
            Operation::new("repeat_interleave", "repeat_k")
                .with_input("x", Value::Reference("K".into()))
                .with_output("K_expanded"),
        );
        // V broadcast.
        block.add_op(
            Operation::new("repeat_interleave", "repeat_v")
                .with_input("x", Value::Reference("V".into()))
                .with_output("V_expanded"),
        );
        block.add_op(
            Operation::new("transpose", "transpose_k")
                .with_input("x", Value::Reference("K_expanded".into()))
                .with_output("K_T"),
        );
        block.add_op(
            Operation::new("matmul", "scores_matmul")
                .with_input("x", Value::Reference("Q".into()))
                .with_input("y", Value::Reference("K_T".into()))
                .with_output("scores"),
        );
        block.add_op(
            Operation::new("real_div", "scale")
                .with_input("x", Value::Reference("scores".into()))
                .with_input("y", Value::Float(8.0))
                .with_output("scores_scaled"),
        );
        block.add_op(
            Operation::new("softmax", "softmax_0")
                .with_input("x", Value::Reference("scores_scaled".into()))
                .with_output("attn_weights"),
        );
        block.add_op(
            Operation::new("matmul", "output_matmul")
                .with_input("x", Value::Reference("attn_weights".into()))
                .with_input("y", Value::Reference("V_expanded".into()))
                .with_output("gqa_output"),
        );
        block.outputs.push("gqa_output".into());
        block
    }

    #[test]
    fn gqa_pattern_fused() {
        let block = build_gqa_block();
        let mut program = program_with_block(block);
        GqaFusionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 1, "expected a single fused GQA op, got {ops:?}");
        assert_eq!(ops[0].op_type, "grouped_query_attention");
        assert_eq!(ops[0].inputs.get("Q"), Some(&Value::Reference("Q".into())));
        assert_eq!(ops[0].inputs.get("K"), Some(&Value::Reference("K".into())));
        assert_eq!(ops[0].inputs.get("V"), Some(&Value::Reference("V".into())));
        assert_eq!(ops[0].attributes.get("is_gqa"), Some(&Value::Bool(true)));
    }

    #[test]
    fn gqa_with_tile_broadcast() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("tile", "tile_k")
                .with_input("x", Value::Reference("K".into()))
                .with_output("K_expanded"),
        );
        block.add_op(
            Operation::new("tile", "tile_v")
                .with_input("x", Value::Reference("V".into()))
                .with_output("V_expanded"),
        );
        block.add_op(
            Operation::new("transpose", "transpose_k")
                .with_input("x", Value::Reference("K_expanded".into()))
                .with_output("K_T"),
        );
        block.add_op(
            Operation::new("matmul", "scores_matmul")
                .with_input("x", Value::Reference("Q".into()))
                .with_input("y", Value::Reference("K_T".into()))
                .with_output("scores"),
        );
        block.add_op(
            Operation::new("real_div", "scale")
                .with_input("x", Value::Reference("scores".into()))
                .with_input("y", Value::Float(8.0))
                .with_output("scores_scaled"),
        );
        block.add_op(
            Operation::new("softmax", "softmax_0")
                .with_input("x", Value::Reference("scores_scaled".into()))
                .with_output("attn_weights"),
        );
        block.add_op(
            Operation::new("matmul", "output_matmul")
                .with_input("x", Value::Reference("attn_weights".into()))
                .with_input("y", Value::Reference("V_expanded".into()))
                .with_output("gqa_output"),
        );
        block.outputs.push("gqa_output".into());

        let mut program = program_with_block(block);
        GqaFusionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op_type, "grouped_query_attention");
    }

    #[test]
    fn gqa_k_only_broadcast() {
        let mut block = Block::new();
        // Only K is broadcast, V is used directly.
        block.add_op(
            Operation::new("repeat_interleave", "repeat_k")
                .with_input("x", Value::Reference("K".into()))
                .with_output("K_expanded"),
        );
        block.add_op(
            Operation::new("transpose", "transpose_k")
                .with_input("x", Value::Reference("K_expanded".into()))
                .with_output("K_T"),
        );
        block.add_op(
            Operation::new("matmul", "scores_matmul")
                .with_input("x", Value::Reference("Q".into()))
                .with_input("y", Value::Reference("K_T".into()))
                .with_output("scores"),
        );
        block.add_op(
            Operation::new("real_div", "scale")
                .with_input("x", Value::Reference("scores".into()))
                .with_input("y", Value::Float(8.0))
                .with_output("scores_scaled"),
        );
        block.add_op(
            Operation::new("softmax", "softmax_0")
                .with_input("x", Value::Reference("scores_scaled".into()))
                .with_output("attn_weights"),
        );
        block.add_op(
            Operation::new("matmul", "output_matmul")
                .with_input("x", Value::Reference("attn_weights".into()))
                .with_input("y", Value::Reference("V".into()))
                .with_output("gqa_output"),
        );
        block.outputs.push("gqa_output".into());

        let mut program = program_with_block(block);
        GqaFusionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op_type, "grouped_query_attention");
        // K should reference the original K, not K_expanded.
        assert_eq!(ops[0].inputs.get("K"), Some(&Value::Reference("K".into())));
        // V should reference V directly since no broadcast.
        assert_eq!(ops[0].inputs.get("V"), Some(&Value::Reference("V".into())));
    }

    #[test]
    fn regular_attention_not_matched_by_gqa() {
        // Standard attention (no broadcast) should NOT be matched by GQA pass.
        let block = build_attention_block();
        let mut program = program_with_block(block);
        GqaFusionPass.run(&mut program).unwrap();

        // No fusion — all 5 ops remain.
        assert_eq!(block_ops(&program).len(), 5);
    }

    #[test]
    fn gqa_downstream_rewired() {
        let mut block = build_gqa_block();
        block.add_op(
            Operation::new("linear", "proj")
                .with_input("x", Value::Reference("gqa_output".into()))
                .with_output("proj_output"),
        );
        block.outputs = vec!["proj_output".into()];

        let mut program = program_with_block(block);
        GqaFusionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 2);
        let linear_op = ops.iter().find(|op| op.op_type == "linear").unwrap();
        let gqa_op = ops
            .iter()
            .find(|op| op.op_type == "grouped_query_attention")
            .unwrap();
        let gqa_output = gqa_op.outputs.first().unwrap();
        assert_eq!(
            linear_op.inputs.get("x"),
            Some(&Value::Reference(gqa_output.clone()))
        );
    }

    // ---- Multi-pattern GQA: shared K/V must not corrupt earlier fused ops ---

    #[test]
    fn multi_pattern_gqa_shared_kv_no_corruption() {
        let mut block = Block::new();

        // Head 0: uses shared K and V via broadcast
        block.add_op(
            Operation::new("repeat_interleave", "repeat_k_0")
                .with_input("x", Value::Reference("K_shared".into()))
                .with_output("K_expanded_0"),
        );
        block.add_op(
            Operation::new("repeat_interleave", "repeat_v_0")
                .with_input("x", Value::Reference("V_shared".into()))
                .with_output("V_expanded_0"),
        );
        block.add_op(
            Operation::new("transpose", "transpose_k_0")
                .with_input("x", Value::Reference("K_expanded_0".into()))
                .with_output("K_T_0"),
        );
        block.add_op(
            Operation::new("matmul", "scores_matmul_0")
                .with_input("x", Value::Reference("Q0".into()))
                .with_input("y", Value::Reference("K_T_0".into()))
                .with_output("scores_0"),
        );
        block.add_op(
            Operation::new("real_div", "scale_0")
                .with_input("x", Value::Reference("scores_0".into()))
                .with_input("y", Value::Float(8.0))
                .with_output("scores_scaled_0"),
        );
        block.add_op(
            Operation::new("softmax", "softmax_0")
                .with_input("x", Value::Reference("scores_scaled_0".into()))
                .with_output("attn_weights_0"),
        );
        block.add_op(
            Operation::new("matmul", "output_matmul_0")
                .with_input("x", Value::Reference("attn_weights_0".into()))
                .with_input("y", Value::Reference("V_expanded_0".into()))
                .with_output("gqa_output_0"),
        );

        // Head 1: uses the SAME shared K and V via broadcast
        block.add_op(
            Operation::new("repeat_interleave", "repeat_k_1")
                .with_input("x", Value::Reference("K_shared".into()))
                .with_output("K_expanded_1"),
        );
        block.add_op(
            Operation::new("repeat_interleave", "repeat_v_1")
                .with_input("x", Value::Reference("V_shared".into()))
                .with_output("V_expanded_1"),
        );
        block.add_op(
            Operation::new("transpose", "transpose_k_1")
                .with_input("x", Value::Reference("K_expanded_1".into()))
                .with_output("K_T_1"),
        );
        block.add_op(
            Operation::new("matmul", "scores_matmul_1")
                .with_input("x", Value::Reference("Q1".into()))
                .with_input("y", Value::Reference("K_T_1".into()))
                .with_output("scores_1"),
        );
        block.add_op(
            Operation::new("real_div", "scale_1")
                .with_input("x", Value::Reference("scores_1".into()))
                .with_input("y", Value::Float(8.0))
                .with_output("scores_scaled_1"),
        );
        block.add_op(
            Operation::new("softmax", "softmax_1")
                .with_input("x", Value::Reference("scores_scaled_1".into()))
                .with_output("attn_weights_1"),
        );
        block.add_op(
            Operation::new("matmul", "output_matmul_1")
                .with_input("x", Value::Reference("attn_weights_1".into()))
                .with_input("y", Value::Reference("V_expanded_1".into()))
                .with_output("gqa_output_1"),
        );

        // Downstream concat consuming both heads
        block.add_op(
            Operation::new("concat", "concat_heads")
                .with_input(
                    "x",
                    Value::List(vec![
                        Value::Reference("gqa_output_0".into()),
                        Value::Reference("gqa_output_1".into()),
                    ]),
                )
                .with_output("combined"),
        );
        block.outputs.push("combined".into());

        let mut program = program_with_block(block);
        GqaFusionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        // 2 fused GQA ops + 1 concat = 3 ops
        assert_eq!(
            ops.len(),
            3,
            "expected 3 ops after fusing two GQA heads, got {ops:?}"
        );

        let gqa_ops: Vec<_> = ops
            .iter()
            .filter(|op| op.op_type == "grouped_query_attention")
            .collect();
        assert_eq!(gqa_ops.len(), 2);

        // Both fused ops must reference the shared K and V, NOT each other's outputs.
        for gqa_op in &gqa_ops {
            assert_eq!(
                gqa_op.inputs.get("K"),
                Some(&Value::Reference("K_shared".into())),
                "GQA op {} K input corrupted",
                gqa_op.name,
            );
            assert_eq!(
                gqa_op.inputs.get("V"),
                Some(&Value::Reference("V_shared".into())),
                "GQA op {} V input corrupted",
                gqa_op.name,
            );
        }

        // Concat must reference fused outputs, not the old gqa_output_* names.
        let concat_op = ops.iter().find(|op| op.op_type == "concat").unwrap();
        if let Some(Value::List(refs)) = concat_op.inputs.get("x") {
            for r in refs {
                if let Value::Reference(name) = r {
                    assert!(
                        name.starts_with("gqa_out_"),
                        "concat input should reference fused output, got {name}"
                    );
                }
            }
        } else {
            panic!("concat should have a list input");
        }
    }
}
