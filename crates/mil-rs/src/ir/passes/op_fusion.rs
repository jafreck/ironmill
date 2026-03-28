//! Op fusion passes — fuse adjacent operations into single fused ops.
//!
//! These passes implement well-documented ML compiler optimizations that
//! improve both GPU and Apple Neural Engine (ANE) performance by reducing
//! memory traffic and kernel-launch overhead.

use crate::error::Result;
use crate::ir::pass::Pass;
use crate::ir::program::{Block, Program};
use crate::ir::types::Value;

use super::replace_reference;

/// Fuses Conv + BatchNorm into a single Conv with adjusted weights.
///
/// Conv+BN fusion is mathematically equivalent: the BN parameters
/// (scale, bias, mean, variance, epsilon) are folded into the conv
/// weights and bias at conversion time.
///
/// This pass performs the graph-level fusion — it removes the `batch_norm`
/// op and marks the `conv` with a `has_fused_bn` attribute. Actual weight
/// folding requires tensor arithmetic and is handled separately.
pub struct ConvBatchNormFusionPass;

impl Pass for ConvBatchNormFusionPass {
    fn name(&self) -> &str {
        "conv-batchnorm-fusion"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            fuse_activation_pattern(
                &mut function.body,
                "conv",
                "batch_norm",
                FusionKind::BatchNorm,
            );
        }
        Ok(())
    }
}

/// Fuses Conv + Relu into a single Conv op with a fused activation attribute.
///
/// Many ML runtimes (including CoreML/ANE) have fused conv+relu
/// implementations that are faster than separate ops.
pub struct ConvReluFusionPass;

impl Pass for ConvReluFusionPass {
    fn name(&self) -> &str {
        "conv-relu-fusion"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            fuse_activation_pattern(&mut function.body, "conv", "relu", FusionKind::Activation);
        }
        Ok(())
    }
}

/// Fuses Linear + Relu into a single Linear op with a fused activation attribute.
pub struct LinearReluFusionPass;

impl Pass for LinearReluFusionPass {
    fn name(&self) -> &str {
        "linear-relu-fusion"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            fuse_activation_pattern(&mut function.body, "linear", "relu", FusionKind::Activation);
        }
        Ok(())
    }
}

/// Fuses LayerNorm + Linear into a single LayerNorm with a fused linear attribute.
///
/// This pattern is extremely common in transformer architectures, where
/// layer normalization is immediately followed by a linear projection.
/// Fusing eliminates the intermediate materialization.
pub struct LayerNormLinearFusionPass;

impl Pass for LayerNormLinearFusionPass {
    fn name(&self) -> &str {
        "layernorm-linear-fusion"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            fuse_activation_pattern(
                &mut function.body,
                "layer_norm",
                "linear",
                FusionKind::Linear,
            );
        }
        Ok(())
    }
}

/// Fuses GELU + Linear into a single GELU with a fused linear attribute.
///
/// Complements the GELU expansion in `op_substitute`: this pass runs
/// *before* substitution and fuses the high-level `gelu` op with a
/// following `linear`, avoiding both the intermediate write and the
/// subsequent expansion into 13 ops.
pub struct GeluLinearFusionPass;

impl Pass for GeluLinearFusionPass {
    fn name(&self) -> &str {
        "gelu-linear-fusion"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            fuse_activation_pattern(&mut function.body, "gelu", "linear", FusionKind::Linear);
        }
        Ok(())
    }
}

/// Fuses skip-connection add ops into `residual_add` ops.
///
/// Detects the residual/skip-connection pattern common in ResNets and
/// transformers: `output = F(x) + x`, where one input to an `add` op
/// is also an input (directly or transitively) to the computation chain
/// producing the other input.
pub struct ResidualAddFusionPass;

impl Pass for ResidualAddFusionPass {
    fn name(&self) -> &str {
        "residual-add-fusion"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            fuse_residual_add_pattern(&mut function.body);
        }
        Ok(())
    }
}

/// The kind of fusion to apply when merging two ops.
enum FusionKind {
    /// Fuse an activation function (e.g., relu) — sets `fused_activation`.
    Activation,
    /// Fuse batch normalization — sets `has_fused_bn`.
    BatchNorm,
    /// Fuse a following linear op — sets `has_fused_linear`.
    Linear,
}

/// Check if a value name is only consumed by a single operation in the block.
///
/// `consumer_idx` is the index of the expected consumer; this function verifies
/// that no *other* operation (or block output) references `value_name`.
fn is_single_consumer(block: &Block, value_name: &str, consumer_idx: usize) -> bool {
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

/// Returns `true` if `value` contains a [`Value::Reference`] to `name`.
fn references_name(value: &Value, name: &str) -> bool {
    match value {
        Value::Reference(n) => n == name,
        Value::List(items) => items.iter().any(|v| references_name(v, name)),
        _ => false,
    }
}

/// Returns `true` if the operation is tagged as a causal convolution.
fn is_causal_conv(op: &crate::ir::operation::Operation) -> bool {
    matches!(op.attributes.get("causal"), Some(Value::Bool(true)))
}

/// Generic fusion of a *producer* op followed by a *consumer* op.
///
/// Scans the block for pairs where `consumer.inputs["x"]` references
/// `producer.outputs[0]`, and the producer output is single-consumer.
fn fuse_activation_pattern(
    block: &mut Block,
    producer_type: &str,
    consumer_type: &str,
    kind: FusionKind,
) {
    // Collect fusion candidates: (producer_idx, consumer_idx).
    let mut fusions: Vec<(usize, usize)> = Vec::new();

    for (ci, consumer) in block.operations.iter().enumerate() {
        if consumer.op_type != consumer_type {
            continue;
        }
        // The canonical input for relu / batch_norm is "x".
        let input_ref = match consumer.inputs.get("x") {
            Some(Value::Reference(name)) => name.clone(),
            _ => continue,
        };

        // Find the producer whose output matches the consumer's input.
        let producer = block.operations.iter().enumerate().find(|(_, op)| {
            op.op_type == producer_type
                && op.outputs.first().map(|s| s.as_str()) == Some(&input_ref)
        });

        let (pi, _) = match producer {
            Some(pair) => pair,
            None => continue,
        };

        if !is_single_consumer(block, &input_ref, ci) {
            continue;
        }

        // Don't fuse a producer that was already claimed by an earlier fusion.
        if fusions.iter().any(|(p, _)| *p == pi) {
            continue;
        }

        // Defensive guard: prevent fusing same-type ops (e.g., conv→conv) with
        // mismatched causality. No current fusion passes match this pattern, but
        // this protects against future same-type fusion passes.
        if producer_type == consumer_type {
            let p_causal = is_causal_conv(&block.operations[pi]);
            let c_causal = is_causal_conv(&block.operations[ci]);
            if p_causal != c_causal {
                continue;
            }
        }

        fusions.push((pi, ci));
    }

    if fusions.is_empty() {
        return;
    }

    // Apply fusions. We need to rewire references before removing ops.
    for &(pi, ci) in &fusions {
        let consumer_output = block.operations[ci].outputs[0].clone();
        let producer_output = block.operations[pi].outputs[0].clone();

        // Replace downstream references to the consumer output with the
        // producer output (the surviving op).
        replace_reference(block, &consumer_output, &producer_output);

        // Mark the producer with fusion metadata.
        let attr_value = match kind {
            FusionKind::Activation => {
                let consumer_op_type = block.operations[ci].op_type.clone();
                (
                    "fused_activation".to_string(),
                    Value::String(consumer_op_type),
                )
            }
            FusionKind::BatchNorm => ("has_fused_bn".to_string(), Value::Bool(true)),
            FusionKind::Linear => ("has_fused_linear".to_string(), Value::Bool(true)),
        };
        block.operations[pi]
            .attributes
            .insert(attr_value.0, attr_value.1);
    }

    // Remove consumed ops (in reverse index order to keep indices stable).
    let mut remove_indices: Vec<usize> = fusions.iter().map(|&(_, ci)| ci).collect();
    remove_indices.sort_unstable();
    for idx in remove_indices.into_iter().rev() {
        block.operations.remove(idx);
    }
}

/// Trace backwards from a value through the block's operations, collecting
/// all transitive input references up to `max_depth` levels. Returns `true`
/// if `target` is found among the transitive inputs of `start`.
fn has_transitive_input(block: &Block, start: &str, target: &str, max_depth: usize) -> bool {
    if max_depth == 0 {
        return false;
    }
    // Find the producer of `start`.
    let producer = block
        .operations
        .iter()
        .find(|op| op.outputs.first().map(|s| s.as_str()) == Some(start));

    let producer_op = match producer {
        Some(op) => op,
        None => return false,
    };

    for input_val in producer_op.inputs.values() {
        if let Value::Reference(name) = input_val {
            if name == target {
                return true;
            }
            if has_transitive_input(block, name, target, max_depth - 1) {
                return true;
            }
        }
    }
    false
}

/// Scan the block for residual add patterns and mark them.
///
/// A residual add is an `add` op where one input is a "skip connection" —
/// i.e., the same value also feeds (directly or transitively) into the
/// computation chain that produces the other input.
fn fuse_residual_add_pattern(block: &mut Block) {
    let mut residual_indices: Vec<usize> = Vec::new();

    for (ai, add_op) in block.operations.iter().enumerate() {
        if add_op.op_type != "add" {
            continue;
        }

        let x_ref = match add_op.inputs.get("x") {
            Some(Value::Reference(name)) => name.clone(),
            _ => continue,
        };
        let y_ref = match add_op.inputs.get("y") {
            Some(Value::Reference(name)) => name.clone(),
            _ => continue,
        };

        // Check both directions: is y a transitive input of x's producer,
        // or is x a transitive input of y's producer?
        const MAX_TRACE_DEPTH: usize = 16;
        let is_residual = has_transitive_input(block, &x_ref, &y_ref, MAX_TRACE_DEPTH)
            || has_transitive_input(block, &y_ref, &x_ref, MAX_TRACE_DEPTH);

        if is_residual {
            residual_indices.push(ai);
        }
    }

    // Mark each identified add as a residual_add.
    for &ai in &residual_indices {
        block.operations[ai].op_type = "residual_add".to_string();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::operation::Operation;
    use crate::ir::program::Function;

    /// Helper: build a minimal program with a single "main" function whose
    /// body is the given block.
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

    // ---- Conv + Relu fusion ------------------------------------------------

    #[test]
    fn conv_relu_fused() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("conv", "conv_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("conv_out"),
        );
        block.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("conv_out".into()))
                .with_output("relu_out"),
        );
        block.outputs.push("relu_out".into());

        let mut program = program_with_block(block);
        ConvReluFusionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op_type, "conv");
        assert_eq!(ops[0].name, "conv_0");
        assert_eq!(
            ops[0].attributes.get("fused_activation"),
            Some(&Value::String("relu".into()))
        );
        // Block output should now reference conv_out.
        assert_eq!(program.functions["main"].body.outputs, vec!["conv_out"]);
    }

    // ---- Conv + BatchNorm fusion -------------------------------------------

    #[test]
    fn conv_batchnorm_fused() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("conv", "conv_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("conv_out"),
        );
        block.add_op(
            Operation::new("batch_norm", "bn_0")
                .with_input("x", Value::Reference("conv_out".into()))
                .with_output("bn_out"),
        );
        block.outputs.push("bn_out".into());

        let mut program = program_with_block(block);
        ConvBatchNormFusionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op_type, "conv");
        assert_eq!(
            ops[0].attributes.get("has_fused_bn"),
            Some(&Value::Bool(true))
        );
        assert_eq!(program.functions["main"].body.outputs, vec!["conv_out"]);
    }

    // ---- Linear + Relu fusion ----------------------------------------------

    #[test]
    fn linear_relu_fused() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("linear", "linear_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("linear_out"),
        );
        block.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("linear_out".into()))
                .with_output("relu_out"),
        );
        block.outputs.push("relu_out".into());

        let mut program = program_with_block(block);
        LinearReluFusionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op_type, "linear");
        assert_eq!(
            ops[0].attributes.get("fused_activation"),
            Some(&Value::String("relu".into()))
        );
        assert_eq!(program.functions["main"].body.outputs, vec!["linear_out"]);
    }

    // ---- Multi-consumer: no fusion -----------------------------------------

    #[test]
    fn no_fusion_when_multi_consumer() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("conv", "conv_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("conv_out"),
        );
        // relu consumes conv_out…
        block.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("conv_out".into()))
                .with_output("relu_out"),
        );
        // …but so does add_0 → conv_out has multiple consumers.
        block.add_op(
            Operation::new("add", "add_0")
                .with_input("x", Value::Reference("conv_out".into()))
                .with_input("y", Value::Reference("relu_out".into()))
                .with_output("add_out"),
        );
        block.outputs.push("add_out".into());

        let mut program = program_with_block(block);
        ConvReluFusionPass.run(&mut program).unwrap();

        // Nothing should be fused — all 3 ops remain.
        assert_eq!(block_ops(&program).len(), 3);
    }

    // ---- Block-output consumer: no fusion ----------------------------------

    #[test]
    fn no_fusion_when_producer_is_block_output() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("conv", "conv_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("conv_out"),
        );
        block.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("conv_out".into()))
                .with_output("relu_out"),
        );
        // conv_out is also a block output → external consumer.
        block.outputs.push("conv_out".into());
        block.outputs.push("relu_out".into());

        let mut program = program_with_block(block);
        ConvReluFusionPass.run(&mut program).unwrap();

        assert_eq!(block_ops(&program).len(), 2);
    }

    // ---- Non-adjacent: relu does not directly consume conv -----------------

    #[test]
    fn no_fusion_when_not_direct_consumer() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("conv", "conv_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("conv_out"),
        );
        // An intervening op sits between conv and relu.
        block.add_op(
            Operation::new("add", "add_0")
                .with_input("x", Value::Reference("conv_out".into()))
                .with_input("y", Value::Reference("conv_out".into()))
                .with_output("add_out"),
        );
        block.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("add_out".into()))
                .with_output("relu_out"),
        );
        block.outputs.push("relu_out".into());

        let mut program = program_with_block(block);
        ConvReluFusionPass.run(&mut program).unwrap();

        // relu's input is add_out, not conv_out → no conv+relu fusion.
        assert_eq!(block_ops(&program).len(), 3);
    }

    // ---- Multiple fusions in one block -------------------------------------

    #[test]
    fn multiple_fusions_in_one_block() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("conv", "conv_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("conv_out_0"),
        );
        block.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("conv_out_0".into()))
                .with_output("relu_out_0"),
        );
        block.add_op(
            Operation::new("conv", "conv_1")
                .with_input("x", Value::Reference("relu_out_0".into()))
                .with_output("conv_out_1"),
        );
        block.add_op(
            Operation::new("relu", "relu_1")
                .with_input("x", Value::Reference("conv_out_1".into()))
                .with_output("relu_out_1"),
        );
        block.outputs.push("relu_out_1".into());

        let mut program = program_with_block(block);
        ConvReluFusionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].name, "conv_0");
        assert_eq!(ops[1].name, "conv_1");

        for op in ops {
            assert_eq!(
                op.attributes.get("fused_activation"),
                Some(&Value::String("relu".into()))
            );
        }
    }

    // ---- Downstream rewiring -----------------------------------------------

    #[test]
    fn downstream_ops_rewired_after_fusion() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("conv", "conv_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("conv_out"),
        );
        block.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("conv_out".into()))
                .with_output("relu_out"),
        );
        block.add_op(
            Operation::new("softmax", "softmax_0")
                .with_input("x", Value::Reference("relu_out".into()))
                .with_output("softmax_out"),
        );
        block.outputs.push("softmax_out".into());

        let mut program = program_with_block(block);
        ConvReluFusionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].name, "conv_0");
        assert_eq!(ops[1].name, "softmax_0");
        // softmax should now reference conv_out (not relu_out).
        if let Some(Value::Reference(name)) = ops[1].inputs.get("x") {
            assert_eq!(name, "conv_out");
        } else {
            panic!("expected softmax input to be a reference");
        }
    }

    // ---- Causal convolution fusion tests -----------------------------------

    #[test]
    fn causal_conv_relu_fused() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("conv", "conv_0")
                .with_input("x", Value::Reference("input".into()))
                .with_attr("causal", Value::Bool(true))
                .with_attr("pad", Value::Int(2))
                .with_output("conv_out"),
        );
        block.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("conv_out".into()))
                .with_output("relu_out"),
        );
        block.outputs.push("relu_out".into());

        let mut program = program_with_block(block);
        ConvReluFusionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 1, "causal conv + relu should still fuse");
        assert_eq!(ops[0].op_type, "conv");
        assert_eq!(
            ops[0].attributes.get("fused_activation"),
            Some(&Value::String("relu".into()))
        );
        assert_eq!(
            ops[0].attributes.get("causal"),
            Some(&Value::Bool(true)),
            "causal attribute must be preserved after fusion"
        );
    }

    #[test]
    fn causal_conv_batchnorm_fused() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("conv", "conv_0")
                .with_input("x", Value::Reference("input".into()))
                .with_attr("causal", Value::Bool(true))
                .with_output("conv_out"),
        );
        block.add_op(
            Operation::new("batch_norm", "bn_0")
                .with_input("x", Value::Reference("conv_out".into()))
                .with_output("bn_out"),
        );
        block.outputs.push("bn_out".into());

        let mut program = program_with_block(block);
        ConvBatchNormFusionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 1, "causal conv + BN should still fuse");
        assert_eq!(
            ops[0].attributes.get("has_fused_bn"),
            Some(&Value::Bool(true))
        );
        assert_eq!(
            ops[0].attributes.get("causal"),
            Some(&Value::Bool(true)),
            "causal attribute must survive BN fusion"
        );
    }

    #[test]
    fn non_causal_conv_relu_still_fuses() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("conv", "conv_0")
                .with_input("x", Value::Reference("input".into()))
                .with_attr("causal", Value::Bool(false))
                .with_output("conv_out"),
        );
        block.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("conv_out".into()))
                .with_output("relu_out"),
        );
        block.outputs.push("relu_out".into());

        let mut program = program_with_block(block);
        ConvReluFusionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 1, "non-causal conv + relu should fuse normally");
        assert_eq!(
            ops[0].attributes.get("fused_activation"),
            Some(&Value::String("relu".into()))
        );
    }

    #[test]
    fn conv_without_causal_attr_still_fuses() {
        // Convs that predate the causal attribute (no `causal` key at all)
        // must continue to fuse as before.
        let mut block = Block::new();
        block.add_op(
            Operation::new("conv", "conv_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("conv_out"),
        );
        block.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("conv_out".into()))
                .with_output("relu_out"),
        );
        block.outputs.push("relu_out".into());

        let mut program = program_with_block(block);
        ConvReluFusionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 1, "conv without causal attr should still fuse");
    }

    // ---- LayerNorm + Linear fusion -----------------------------------------

    #[test]
    fn layernorm_linear_fused() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("layer_norm", "ln_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("ln_out"),
        );
        block.add_op(
            Operation::new("linear", "linear_0")
                .with_input("x", Value::Reference("ln_out".into()))
                .with_output("linear_out"),
        );
        block.outputs.push("linear_out".into());

        let mut program = program_with_block(block);
        LayerNormLinearFusionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op_type, "layer_norm");
        assert_eq!(ops[0].name, "ln_0");
        assert_eq!(
            ops[0].attributes.get("has_fused_linear"),
            Some(&Value::Bool(true))
        );
        assert_eq!(program.functions["main"].body.outputs, vec!["ln_out"]);
    }

    #[test]
    fn layernorm_linear_no_fusion_when_multi_consumer() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("layer_norm", "ln_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("ln_out"),
        );
        block.add_op(
            Operation::new("linear", "linear_0")
                .with_input("x", Value::Reference("ln_out".into()))
                .with_output("linear_out"),
        );
        // ln_out is also used elsewhere.
        block.add_op(
            Operation::new("add", "add_0")
                .with_input("x", Value::Reference("ln_out".into()))
                .with_input("y", Value::Reference("linear_out".into()))
                .with_output("add_out"),
        );
        block.outputs.push("add_out".into());

        let mut program = program_with_block(block);
        LayerNormLinearFusionPass.run(&mut program).unwrap();

        assert_eq!(block_ops(&program).len(), 3);
    }

    // ---- GELU + Linear fusion ----------------------------------------------

    #[test]
    fn gelu_linear_fused() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("gelu", "gelu_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("gelu_out"),
        );
        block.add_op(
            Operation::new("linear", "linear_0")
                .with_input("x", Value::Reference("gelu_out".into()))
                .with_output("linear_out"),
        );
        block.outputs.push("linear_out".into());

        let mut program = program_with_block(block);
        GeluLinearFusionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op_type, "gelu");
        assert_eq!(ops[0].name, "gelu_0");
        assert_eq!(
            ops[0].attributes.get("has_fused_linear"),
            Some(&Value::Bool(true))
        );
        assert_eq!(program.functions["main"].body.outputs, vec!["gelu_out"]);
    }

    #[test]
    fn gelu_linear_downstream_rewired() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("gelu", "gelu_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("gelu_out"),
        );
        block.add_op(
            Operation::new("linear", "linear_0")
                .with_input("x", Value::Reference("gelu_out".into()))
                .with_output("linear_out"),
        );
        block.add_op(
            Operation::new("softmax", "softmax_0")
                .with_input("x", Value::Reference("linear_out".into()))
                .with_output("softmax_out"),
        );
        block.outputs.push("softmax_out".into());

        let mut program = program_with_block(block);
        GeluLinearFusionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].name, "gelu_0");
        assert_eq!(ops[1].name, "softmax_0");
        if let Some(Value::Reference(name)) = ops[1].inputs.get("x") {
            assert_eq!(name, "gelu_out");
        } else {
            panic!("expected softmax input to be a reference");
        }
    }

    // ---- Residual add fusion -----------------------------------------------

    #[test]
    fn residual_add_fused() {
        let mut block = Block::new();
        // Skip connection pattern: input → conv → relu → add(relu_out, input)
        block.add_op(
            Operation::new("conv", "conv_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("conv_out"),
        );
        block.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("conv_out".into()))
                .with_output("relu_out"),
        );
        block.add_op(
            Operation::new("add", "add_0")
                .with_input("x", Value::Reference("relu_out".into()))
                .with_input("y", Value::Reference("input".into()))
                .with_output("add_out"),
        );
        block.outputs.push("add_out".into());

        let mut program = program_with_block(block);
        ResidualAddFusionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 3);
        assert_eq!(ops[2].op_type, "residual_add");
        assert_eq!(ops[2].name, "add_0");
    }

    #[test]
    fn residual_add_reversed_inputs() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("conv", "conv_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("conv_out"),
        );
        block.add_op(
            Operation::new("add", "add_0")
                .with_input("x", Value::Reference("input".into()))
                .with_input("y", Value::Reference("conv_out".into()))
                .with_output("add_out"),
        );
        block.outputs.push("add_out".into());

        let mut program = program_with_block(block);
        ResidualAddFusionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops[1].op_type, "residual_add");
    }

    #[test]
    fn non_residual_add_not_fused() {
        let mut block = Block::new();
        // Two independent inputs added — not a residual pattern.
        block.add_op(
            Operation::new("conv", "conv_0")
                .with_input("x", Value::Reference("input_a".into()))
                .with_output("conv_out"),
        );
        block.add_op(
            Operation::new("linear", "linear_0")
                .with_input("x", Value::Reference("input_b".into()))
                .with_output("linear_out"),
        );
        block.add_op(
            Operation::new("add", "add_0")
                .with_input("x", Value::Reference("conv_out".into()))
                .with_input("y", Value::Reference("linear_out".into()))
                .with_output("add_out"),
        );
        block.outputs.push("add_out".into());

        let mut program = program_with_block(block);
        ResidualAddFusionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops[2].op_type, "add");
    }
}
