//! Shape materialization pass — replaces dynamic dimensions with concrete values.

use std::collections::HashMap;

use crate::error::Result;
use crate::ir::pass::Pass;
use crate::ir::program::Program;

/// Name fragments that identify sequence-length inputs in autoregressive models.
const SEQ_LEN_PATTERNS: &[&str] = &["seq_len", "sequence_length", "position_ids", "input_ids"];

/// Name fragments that identify KV cache inputs (used for shape inference).
const CACHE_PATTERNS: &[&str] = &[
    "past_key_values",
    "past_key",
    "past_value",
    "key_cache",
    "value_cache",
];

/// Replaces dynamic dimensions with concrete values in function inputs.
///
/// ANE requires all tensor shapes to be fully static. This pass takes
/// a map of `input_name → concrete_shape` and replaces any `None`
/// dimensions in matching function inputs.
///
/// # Example
/// ```
/// use mil_rs::ir::passes::ShapeMaterializePass;
/// let pass = ShapeMaterializePass::new()
///     .with_shape("input", vec![1, 3, 224, 224]);
/// ```
pub struct ShapeMaterializePass {
    /// Map of input name → target shape.
    shapes: HashMap<String, Vec<usize>>,
}

impl ShapeMaterializePass {
    /// Create a new pass with an empty shape map.
    pub fn new() -> Self {
        Self {
            shapes: HashMap::new(),
        }
    }

    /// Set the concrete shape for a named input.
    pub fn with_shape(mut self, name: impl Into<String>, shape: Vec<usize>) -> Self {
        self.shapes.insert(name.into(), shape);
        self
    }
}

impl Default for ShapeMaterializePass {
    fn default() -> Self {
        Self::new()
    }
}

impl Pass for ShapeMaterializePass {
    fn name(&self) -> &str {
        "shape-materialization"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            for (input_name, tensor_type) in &mut function.inputs {
                let target = match self.shapes.get(input_name.as_str()) {
                    Some(s) => s,
                    None => continue,
                };

                // Verify rank matches.
                if tensor_type.rank() != target.len() {
                    return Err(crate::error::MilError::Validation(format!(
                        "shape materialization: input '{}' has rank {} but target shape has rank {}",
                        input_name,
                        tensor_type.rank(),
                        target.len(),
                    )));
                }

                // Warn if all dimensions are already static.
                if tensor_type.is_static() {
                    tracing::warn!(
                        "shape-materialization: input '{}' already has a fully static shape, skipping",
                        input_name,
                    );
                    continue;
                }

                // Replace None dimensions with concrete values.
                for (dim, &concrete) in tensor_type.shape.iter_mut().zip(target.iter()) {
                    if dim.is_none() {
                        *dim = Some(concrete);
                    }
                }
            }
        }
        Ok(())
    }
}

/// Materializes dynamic dimensions in autoregressive model inputs.
///
/// For programs tagged with `autoregressive = "true"`, this pass automatically
/// fixes dynamic sequence-length dimensions in inputs like `input_ids`,
/// `position_ids`, and cache tensors. This is required for ANE, which needs
/// fully static shapes.
///
/// Sequence-length inputs (e.g., `input_ids`, `position_ids`) get their
/// dynamic dimensions set to `1` (single-token decode step). Cache inputs
/// get their sequence dimension (axis 2) set to `max_seq_length`.
///
/// # Example
/// ```
/// use mil_rs::ir::passes::AutoregressiveShapeMaterializePass;
/// let pass = AutoregressiveShapeMaterializePass::new(2048);
/// ```
pub struct AutoregressiveShapeMaterializePass {
    /// Maximum sequence length for cache dimensions.
    max_seq_length: usize,
}

impl AutoregressiveShapeMaterializePass {
    /// Create a new pass with the given maximum sequence length.
    pub fn new(max_seq_length: usize) -> Self {
        Self { max_seq_length }
    }
}

impl Default for AutoregressiveShapeMaterializePass {
    fn default() -> Self {
        Self {
            max_seq_length: 2048,
        }
    }
}

impl Pass for AutoregressiveShapeMaterializePass {
    fn name(&self) -> &str {
        "ar-shape-materialization"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        if !program.is_autoregressive() {
            return Ok(());
        }

        for function in program.functions.values_mut() {
            for (input_name, tensor_type) in &mut function.inputs {
                if tensor_type.is_static() {
                    continue;
                }

                let lower = input_name.to_lowercase();

                if is_seq_len_input(&lower) {
                    // Sequence-length inputs: fix dynamic dims to 1 (decode step).
                    materialize_seq_len_dims(tensor_type);
                } else if is_cache_input(&lower) {
                    // Cache inputs: batch=1, seq_len axis=max_seq_length.
                    materialize_cache_dims(tensor_type, self.max_seq_length);
                }
            }
        }
        Ok(())
    }
}

/// Returns `true` if the lowercased name matches a sequence-length input.
fn is_seq_len_input(lower: &str) -> bool {
    SEQ_LEN_PATTERNS.iter().any(|p| lower.contains(p))
}

/// Returns `true` if the lowercased name matches a KV cache input.
fn is_cache_input(lower: &str) -> bool {
    CACHE_PATTERNS.iter().any(|p| lower.contains(p))
}

/// Replace `None` dimensions with `1` for sequence-length inputs.
///
/// For inputs like `input_ids [batch, seq_len]`, the decode step uses
/// a single token, so dynamic dims become `1`.
fn materialize_seq_len_dims(tensor_type: &mut crate::ir::tensor::TensorType) {
    for dim in &mut tensor_type.shape {
        if dim.is_none() {
            *dim = Some(1);
        }
    }
}

/// Replace `None` dimensions in cache tensors with concrete values.
///
/// Cache tensors typically have shape `[batch, num_heads, seq_len, head_dim]`.
/// - axis 0 (batch): set to `1`
/// - axis 2 (seq_len): set to `max_seq_length`
/// - other dynamic axes: set to `64` as a sensible default
fn materialize_cache_dims(tensor_type: &mut crate::ir::tensor::TensorType, max_seq_length: usize) {
    for (i, dim) in tensor_type.shape.iter_mut().enumerate() {
        if dim.is_none() {
            *dim = Some(match i {
                0 => 1,              // batch
                2 => max_seq_length, // seq_len
                _ => 64,             // num_heads or head_dim
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::program::Function;
    use crate::ir::tensor::{ScalarType, TensorType};

    #[test]
    fn replaces_dynamic_dims_with_concrete_values() {
        let mut program = Program::new("1.0.0");
        let input_ty =
            TensorType::with_dynamic_shape(ScalarType::Float32, vec![None, Some(3), None, None]);
        let func = Function::new("main").with_input("image", input_ty);
        program.add_function(func);

        let pass = ShapeMaterializePass::new().with_shape("image", vec![1, 3, 224, 224]);
        pass.run(&mut program).unwrap();

        let shape = &program.functions["main"].inputs[0].1.shape;
        assert_eq!(shape, &vec![Some(1), Some(3), Some(224), Some(224)]);
    }

    #[test]
    fn errors_on_rank_mismatch() {
        let mut program = Program::new("1.0.0");
        let input_ty =
            TensorType::with_dynamic_shape(ScalarType::Float32, vec![None, Some(3), None]);
        let func = Function::new("main").with_input("image", input_ty);
        program.add_function(func);

        let pass = ShapeMaterializePass::new().with_shape("image", vec![1, 3, 224, 224]);
        let result = pass.run(&mut program);

        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("rank"), "error should mention rank: {msg}");
    }

    #[test]
    fn noop_when_all_dims_static() {
        let mut program = Program::new("1.0.0");
        let input_ty = TensorType::new(ScalarType::Float32, vec![1, 3, 224, 224]);
        let func = Function::new("main").with_input("image", input_ty);
        program.add_function(func);

        let pass = ShapeMaterializePass::new().with_shape("image", vec![1, 3, 224, 224]);
        pass.run(&mut program).unwrap();

        let shape = &program.functions["main"].inputs[0].1.shape;
        assert_eq!(shape, &vec![Some(1), Some(3), Some(224), Some(224)]);
    }

    #[test]
    fn handles_multiple_inputs() {
        let mut program = Program::new("1.0.0");
        let img_ty =
            TensorType::with_dynamic_shape(ScalarType::Float32, vec![None, Some(3), None, None]);
        let mask_ty = TensorType::with_dynamic_shape(ScalarType::Float32, vec![None, None]);
        let func = Function::new("main")
            .with_input("image", img_ty)
            .with_input("mask", mask_ty);
        program.add_function(func);

        let pass = ShapeMaterializePass::new()
            .with_shape("image", vec![1, 3, 224, 224])
            .with_shape("mask", vec![1, 50]);
        pass.run(&mut program).unwrap();

        let inputs = &program.functions["main"].inputs;
        assert_eq!(
            inputs[0].1.shape,
            vec![Some(1), Some(3), Some(224), Some(224)]
        );
        assert_eq!(inputs[1].1.shape, vec![Some(1), Some(50)]);
    }

    #[test]
    fn ignores_inputs_not_in_shape_map() {
        let mut program = Program::new("1.0.0");
        let img_ty =
            TensorType::with_dynamic_shape(ScalarType::Float32, vec![None, Some(3), None, None]);
        let other_ty = TensorType::with_dynamic_shape(ScalarType::Float32, vec![None, None]);
        let func = Function::new("main")
            .with_input("image", img_ty)
            .with_input("other", other_ty);
        program.add_function(func);

        // Only provide shape for "image", not "other".
        let pass = ShapeMaterializePass::new().with_shape("image", vec![1, 3, 224, 224]);
        pass.run(&mut program).unwrap();

        let inputs = &program.functions["main"].inputs;
        assert_eq!(
            inputs[0].1.shape,
            vec![Some(1), Some(3), Some(224), Some(224)]
        );
        // "other" should remain untouched.
        assert_eq!(inputs[1].1.shape, vec![None, None]);
    }

    // -----------------------------------------------------------------------
    // AutoregressiveShapeMaterializePass tests
    // -----------------------------------------------------------------------

    #[test]
    fn ar_pass_materializes_seq_len_inputs() {
        let mut program = Program::new("1.0.0");
        program.set_attribute("autoregressive", "true");

        let func = Function::new("main")
            .with_input(
                "input_ids",
                TensorType::with_dynamic_shape(ScalarType::Int32, vec![None, None]),
            )
            .with_input(
                "position_ids",
                TensorType::with_dynamic_shape(ScalarType::Int32, vec![None, None]),
            );
        program.add_function(func);

        let pass = AutoregressiveShapeMaterializePass::new(2048);
        pass.run(&mut program).unwrap();

        let inputs = &program.functions["main"].inputs;
        // Sequence-length inputs should get all dynamic dims set to 1.
        assert_eq!(inputs[0].1.shape, vec![Some(1), Some(1)]);
        assert_eq!(inputs[1].1.shape, vec![Some(1), Some(1)]);
    }

    #[test]
    fn ar_pass_materializes_cache_inputs() {
        let mut program = Program::new("1.0.0");
        program.set_attribute("autoregressive", "true");

        let func = Function::new("main")
            .with_input(
                "past_key_values.0.key",
                TensorType::with_dynamic_shape(
                    ScalarType::Float32,
                    vec![None, Some(8), None, Some(64)],
                ),
            )
            .with_input(
                "past_value",
                TensorType::with_dynamic_shape(
                    ScalarType::Float32,
                    vec![None, Some(8), None, Some(64)],
                ),
            );
        program.add_function(func);

        let pass = AutoregressiveShapeMaterializePass::new(1024);
        pass.run(&mut program).unwrap();

        let inputs = &program.functions["main"].inputs;
        // Cache inputs: batch=1, seq_len=max_seq_length, others keep static values.
        assert_eq!(
            inputs[0].1.shape,
            vec![Some(1), Some(8), Some(1024), Some(64)]
        );
        assert_eq!(
            inputs[1].1.shape,
            vec![Some(1), Some(8), Some(1024), Some(64)]
        );
    }

    #[test]
    fn ar_pass_noop_for_non_ar_programs() {
        let mut program = Program::new("1.0.0");
        // No autoregressive attribute set.
        let func = Function::new("main").with_input(
            "input_ids",
            TensorType::with_dynamic_shape(ScalarType::Int32, vec![None, None]),
        );
        program.add_function(func);

        let pass = AutoregressiveShapeMaterializePass::new(2048);
        pass.run(&mut program).unwrap();

        // Shapes should remain dynamic.
        let inputs = &program.functions["main"].inputs;
        assert_eq!(inputs[0].1.shape, vec![None, None]);
    }

    #[test]
    fn ar_pass_skips_static_inputs() {
        let mut program = Program::new("1.0.0");
        program.set_attribute("autoregressive", "true");

        let func = Function::new("main").with_input(
            "input_ids",
            TensorType::new(ScalarType::Int32, vec![1, 128]),
        );
        program.add_function(func);

        let pass = AutoregressiveShapeMaterializePass::new(2048);
        pass.run(&mut program).unwrap();

        // Already-static shapes should be unchanged.
        let inputs = &program.functions["main"].inputs;
        assert_eq!(inputs[0].1.shape, vec![Some(1), Some(128)]);
    }

    #[test]
    fn ar_pass_leaves_non_ar_inputs_untouched() {
        let mut program = Program::new("1.0.0");
        program.set_attribute("autoregressive", "true");

        let func = Function::new("main")
            .with_input(
                "input_ids",
                TensorType::with_dynamic_shape(ScalarType::Int32, vec![None, None]),
            )
            .with_input(
                "pixel_values",
                TensorType::with_dynamic_shape(
                    ScalarType::Float32,
                    vec![None, Some(3), None, None],
                ),
            );
        program.add_function(func);

        let pass = AutoregressiveShapeMaterializePass::new(2048);
        pass.run(&mut program).unwrap();

        let inputs = &program.functions["main"].inputs;
        // input_ids gets materialized.
        assert_eq!(inputs[0].1.shape, vec![Some(1), Some(1)]);
        // pixel_values doesn't match AR patterns, left untouched.
        assert_eq!(inputs[1].1.shape, vec![None, Some(3), None, None]);
    }
}
