//! Model splitting pass for speculative decoding workflows.
//!
//! Partitions a single model into draft and verifier variants at conversion
//! time.  The **draft** model contains only the first N transformer layers
//! (for fast speculative decoding) while the **verifier** retains all layers.
//!
//! Because this pass produces *two* programs from one input it cannot satisfy
//! the standard [`Pass`] trait which mutates a single program in place.
//! Instead it exposes a dedicated [`ModelSplitPass::split`] method that
//! returns a [`SplitResult`].

use std::collections::HashSet;

use mil_rs::error::{MilError, Result};
use mil_rs::ir::Operation;
use mil_rs::ir::Value;
use mil_rs::ir::{Block, Function, Program};

/// Result of splitting a model into draft and verifier variants.
#[derive(Debug, Clone)]
pub struct SplitResult {
    /// Draft model — first N transformer layers only.
    pub draft: Program,
    /// Verifier model — the full, unmodified model.
    pub verifier: Program,
}

/// Splits a [`Program`] into draft/verifier variants for speculative decoding.
///
/// The pass detects transformer layer boundaries by looking for repeated
/// `layer_norm` operations (which typically demarcate the end of each
/// transformer block: attention → FFN → layernorm).  Given a requested
/// `draft_layers` count, the draft program keeps only the operations up to
/// and including the Nth layer boundary, rewiring outputs appropriately.
///
/// The verifier program is always a clone of the full input program.
///
/// # Quantization
///
/// Different quantization strategies for draft vs. verifier (e.g. aggressive
/// 2-bit palettization on the draft, FP16 on the verifier) are configured
/// externally via the CLI — this pass only handles the structural split.
pub struct ModelSplitPass {
    /// How many transformer layers to keep in the draft model.
    draft_layers: usize,
}

impl ModelSplitPass {
    /// Create a new pass that will produce a draft model with `draft_layers`
    /// transformer layers.
    pub fn new(draft_layers: usize) -> Self {
        Self { draft_layers }
    }

    /// Split `program` into draft and verifier variants.
    ///
    /// Returns an error if the program has no functions or if fewer than
    /// `draft_layers` transformer layer boundaries are detected.
    pub fn split(&self, program: &Program) -> Result<SplitResult> {
        if self.draft_layers == 0 {
            return Err(MilError::Validation(
                "draft_layers must be at least 1".into(),
            ));
        }

        if program.functions.is_empty() {
            return Err(MilError::Validation(
                "cannot split an empty program with no functions".into(),
            ));
        }

        // The verifier is always the full model.
        let verifier = program.clone();

        // Build the draft by truncating each function.
        let mut draft = Program::new(program.version.clone());

        for (_name, function) in &program.functions {
            let draft_fn = self.build_draft_function(function)?;
            draft.add_function(draft_fn);
        }

        Ok(SplitResult { draft, verifier })
    }

    /// Build a truncated draft variant of a single function.
    fn build_draft_function(&self, function: &Function) -> Result<Function> {
        let boundaries = detect_layer_boundaries(&function.body);

        if boundaries.len() < self.draft_layers {
            return Err(MilError::Validation(format!(
                "requested {} draft layers but only {} transformer layer \
                 boundary(ies) detected in function '{}'",
                self.draft_layers,
                boundaries.len(),
                function.name,
            )));
        }

        // The cut point is the op index *after* the last op in the Nth layer.
        let cut = boundaries[self.draft_layers - 1] + 1;
        let kept_ops: Vec<Operation> = function.body.operations[..cut].to_vec();

        // Determine which values are defined by the kept ops so we can set
        // sensible block outputs.
        let defined: HashSet<&str> = kept_ops
            .iter()
            .flat_map(|op| op.outputs.iter().map(|s| s.as_str()))
            .collect();

        // The draft outputs are:
        // 1. Any original block output that is still defined, OR
        // 2. The outputs of the last kept operation (fallback).
        let mut draft_outputs: Vec<String> = function
            .body
            .outputs
            .iter()
            .filter(|o| defined.contains(o.as_str()))
            .cloned()
            .collect();

        if draft_outputs.is_empty() {
            // Use the last operation's outputs.
            if let Some(last) = kept_ops.last() {
                draft_outputs = last.outputs.clone();
            }
        }

        let mut draft_fn = Function::new(function.name.clone());
        draft_fn.inputs = function.inputs.clone();
        draft_fn.body = Block::with_operations(kept_ops, draft_outputs);

        Ok(draft_fn)
    }
}

// ---------------------------------------------------------------------------
// Layer boundary detection
// ---------------------------------------------------------------------------

/// Op types that signal the end of a transformer layer.
///
/// Transformer blocks almost universally end with a layer-normalisation op.
/// We also recognise `instance_norm` and `rms_norm` which appear in some
/// architectures (e.g. LLaMA uses RMSNorm).
const LAYER_BOUNDARY_OPS: &[&str] = &["layer_norm", "instance_norm", "rms_norm"];

/// Op types that are part of the attention sub-block.
const ATTENTION_OPS: &[&str] = &["matmul", "linear", "einsum"];

/// Detect indices of operations that mark the end of a transformer layer.
///
/// A "layer boundary" is a normalisation op (`layer_norm`, `instance_norm`,
/// `rms_norm`) that follows at least one attention-related op (`matmul`,
/// `linear`, `einsum`) since the previous boundary (or the start of the
/// block).  This heuristic avoids counting embedding-stage norms or other
/// stray normalisations as layer boundaries.
fn detect_layer_boundaries(block: &Block) -> Vec<usize> {
    let mut boundaries = Vec::new();
    let mut saw_attention_op = false;

    for (idx, op) in block.operations.iter().enumerate() {
        let ty = op.op_type.as_str();

        if ATTENTION_OPS.contains(&ty) {
            saw_attention_op = true;
        }

        if LAYER_BOUNDARY_OPS.contains(&ty) && saw_attention_op {
            boundaries.push(idx);
            saw_attention_op = false;
        }
    }

    boundaries
}

/// Collect all value names referenced by a [`Value`].
fn _collect_refs(value: &Value, out: &mut HashSet<String>) {
    match value {
        Value::Reference(name) => {
            out.insert(name.clone());
        }
        Value::List(items) => {
            for item in items {
                _collect_refs(item, out);
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use mil_rs::ir::Operation;
    use mil_rs::ir::Value;

    /// Helper: create a minimal transformer-like function with `n_layers`
    /// repeated blocks of [linear, matmul, softmax, linear, layer_norm].
    fn transformer_function(name: &str, n_layers: usize) -> Function {
        let mut func = Function::new(name);
        let mut prev_out = "input".to_string();

        for layer in 0..n_layers {
            // Attention sub-block
            let qkv = Operation::new("linear", format!("layer{layer}_qkv"))
                .with_input("x", Value::Reference(prev_out.clone()))
                .with_output(format!("layer{layer}_qkv_out"));
            let attn = Operation::new("matmul", format!("layer{layer}_attn"))
                .with_input("x", Value::Reference(format!("layer{layer}_qkv_out")))
                .with_output(format!("layer{layer}_attn_out"));
            let sm = Operation::new("softmax", format!("layer{layer}_softmax"))
                .with_input("x", Value::Reference(format!("layer{layer}_attn_out")))
                .with_output(format!("layer{layer}_sm_out"));
            // FFN
            let ffn = Operation::new("linear", format!("layer{layer}_ffn"))
                .with_input("x", Value::Reference(format!("layer{layer}_sm_out")))
                .with_output(format!("layer{layer}_ffn_out"));
            // LayerNorm
            let ln = Operation::new("layer_norm", format!("layer{layer}_ln"))
                .with_input("x", Value::Reference(format!("layer{layer}_ffn_out")))
                .with_output(format!("layer{layer}_ln_out"));

            func.body.add_op(qkv);
            func.body.add_op(attn);
            func.body.add_op(sm);
            func.body.add_op(ffn);
            func.body.add_op(ln);

            prev_out = format!("layer{layer}_ln_out");
        }

        func.body.outputs.push(prev_out);
        func
    }

    fn program_with_transformer(n_layers: usize) -> Program {
        let mut program = Program::new("1.0.0");
        program.add_function(transformer_function("main", n_layers));
        program
    }

    #[test]
    fn split_produces_draft_and_verifier() {
        let program = program_with_transformer(6);
        let pass = ModelSplitPass::new(2);
        let result = pass.split(&program).unwrap();

        // Verifier should be the full program.
        assert_eq!(
            result.verifier.functions["main"].body.operations.len(),
            6 * 5
        );

        // Draft should have only 2 layers worth of ops (2 * 5 = 10).
        assert_eq!(result.draft.functions["main"].body.operations.len(), 10);
    }

    #[test]
    fn draft_outputs_are_set() {
        let program = program_with_transformer(4);
        let pass = ModelSplitPass::new(2);
        let result = pass.split(&program).unwrap();

        let draft_fn = &result.draft.functions["main"];
        // The draft should have outputs from the last kept layer_norm.
        assert!(!draft_fn.body.outputs.is_empty());
        assert_eq!(draft_fn.body.outputs[0], "layer1_ln_out");
    }

    #[test]
    fn draft_preserves_inputs() {
        let mut program = program_with_transformer(4);
        // Add an input to the function.
        let func = program.functions.get_mut("main").unwrap();
        func.inputs.push((
            "tokens".into(),
            mil_rs::ir::TensorType::new(mil_rs::ir::ScalarType::Float32, vec![1, 128]),
        ));

        let pass = ModelSplitPass::new(2);
        let result = pass.split(&program).unwrap();

        assert_eq!(result.draft.functions["main"].inputs.len(), 1);
        assert_eq!(result.draft.functions["main"].inputs[0].0, "tokens");
    }

    #[test]
    fn error_when_too_few_layers() {
        let program = program_with_transformer(2);
        let pass = ModelSplitPass::new(5);
        let result = pass.split(&program);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("5 draft layers"));
        assert!(err.contains("2 transformer layer boundary"));
    }

    #[test]
    fn error_on_zero_draft_layers() {
        let program = program_with_transformer(4);
        let pass = ModelSplitPass::new(0);
        let result = pass.split(&program);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("at least 1"));
    }

    #[test]
    fn detect_boundaries_skips_non_attention_norms() {
        // A layer_norm that is NOT preceded by an attention op should be
        // ignored (e.g. an embedding-stage norm).
        let mut block = Block::new();
        // Embedding norm (no attention op before it).
        block.add_op(Operation::new("layer_norm", "embed_ln").with_output("embed_ln_out"));
        // Layer 0
        block.add_op(Operation::new("matmul", "l0_attn").with_output("l0_attn_out"));
        block.add_op(Operation::new("layer_norm", "l0_ln").with_output("l0_ln_out"));
        // Layer 1
        block.add_op(Operation::new("linear", "l1_ffn").with_output("l1_ffn_out"));
        block.add_op(Operation::new("layer_norm", "l1_ln").with_output("l1_ln_out"));

        let boundaries = detect_layer_boundaries(&block);
        // embed_ln should NOT be counted — only l0_ln (idx 2) and l1_ln (idx 4).
        assert_eq!(boundaries, vec![2, 4]);
    }

    #[test]
    fn verifier_is_full_clone() {
        let program = program_with_transformer(4);
        let pass = ModelSplitPass::new(2);
        let result = pass.split(&program).unwrap();

        assert_eq!(
            result.verifier.functions["main"].body.operations.len(),
            program.functions["main"].body.operations.len(),
        );
        assert_eq!(result.verifier.version, program.version);
    }

    #[test]
    fn split_with_rms_norm_boundaries() {
        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");

        // Layer using rms_norm (LLaMA-style)
        func.body.add_op(
            Operation::new("linear", "l0_qkv")
                .with_input("x", Value::Reference("input".into()))
                .with_output("l0_qkv_out"),
        );
        func.body.add_op(
            Operation::new("matmul", "l0_attn")
                .with_input("x", Value::Reference("l0_qkv_out".into()))
                .with_output("l0_attn_out"),
        );
        func.body.add_op(
            Operation::new("rms_norm", "l0_rms")
                .with_input("x", Value::Reference("l0_attn_out".into()))
                .with_output("l0_rms_out"),
        );
        // Layer 1
        func.body.add_op(
            Operation::new("linear", "l1_qkv")
                .with_input("x", Value::Reference("l0_rms_out".into()))
                .with_output("l1_qkv_out"),
        );
        func.body.add_op(
            Operation::new("matmul", "l1_attn")
                .with_input("x", Value::Reference("l1_qkv_out".into()))
                .with_output("l1_attn_out"),
        );
        func.body.add_op(
            Operation::new("rms_norm", "l1_rms")
                .with_input("x", Value::Reference("l1_attn_out".into()))
                .with_output("l1_rms_out"),
        );

        func.body.outputs.push("l1_rms_out".into());
        program.add_function(func);

        let pass = ModelSplitPass::new(1);
        let result = pass.split(&program).unwrap();
        assert_eq!(result.draft.functions["main"].body.operations.len(), 3);
        assert_eq!(result.draft.functions["main"].body.outputs[0], "l0_rms_out");
    }

    #[test]
    fn error_on_empty_program() {
        let program = Program::new("1.0.0");
        let pass = ModelSplitPass::new(2);
        let result = pass.split(&program);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("empty program"));
    }

    #[test]
    fn split_all_layers_equals_full_model() {
        let program = program_with_transformer(3);
        let pass = ModelSplitPass::new(3);
        let result = pass.split(&program).unwrap();

        assert_eq!(
            result.draft.functions["main"].body.operations.len(),
            result.verifier.functions["main"].body.operations.len(),
        );
    }
}
