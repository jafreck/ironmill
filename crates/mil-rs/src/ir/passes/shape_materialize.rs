//! Shape materialization pass — replaces dynamic dimensions with concrete values.

use std::collections::HashMap;

use crate::error::Result;
use crate::ir::pass::Pass;
use crate::ir::program::Program;

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
                    eprintln!(
                        "warning: shape-materialization: input '{}' already has a fully static shape, skipping",
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
}
