//! Compute unit annotation pass.
//!
//! Assigns a preferred [`ComputeUnit`] to each operation based on the
//! shape-aware ANE constraint database from `validate.rs`. Operations
//! that pass all ANE eligibility checks are annotated with `Ane`; ops
//! that are supported but have performance annotations are sent to `Gpu`;
//! unsupported ops fall back to `Cpu`; and unknown ops get `Any`.

use crate::ane::validate::{
    build_type_map, check_performance, check_shape_constraints, is_ane_supported,
};
use mil_rs::error::Result;
use mil_rs::ir::ComputeUnit;
use mil_rs::ir::Pass;
use mil_rs::ir::Program;

/// Annotates every operation with its preferred [`ComputeUnit`].
///
/// The pass is non-destructive — it only sets the `compute_unit` attribute on
/// each [`Operation`] without modifying the graph structure.
pub struct ComputeUnitAnnotationPass;

impl Pass for ComputeUnitAnnotationPass {
    fn name(&self) -> &str {
        "compute-unit-annotation"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            let type_map = build_type_map(function);

            for op in &mut function.body.operations {
                // const ops are always compatible — skip annotation.
                if op.op_type == "const" {
                    op.set_compute_unit(ComputeUnit::Any);
                    continue;
                }

                // 1. Check the op-type allowlist.
                if !is_ane_supported(&op.op_type) {
                    // Unknown op — fall back to CPU.
                    op.set_compute_unit(ComputeUnit::Cpu);
                    continue;
                }

                // 2. Check shape-aware constraints.
                let (shape_rejection, _annotations) = check_shape_constraints(op, &type_map);
                if shape_rejection.is_some() {
                    // Shape constraints violated — route to GPU.
                    op.set_compute_unit(ComputeUnit::Gpu);
                    continue;
                }

                // 3. Check for performance annotations (suboptimal ANE patterns).
                let perf_annotations = check_performance(op, &type_map);
                if !perf_annotations.is_empty() {
                    // Technically ANE-eligible but may perform poorly — prefer GPU.
                    op.set_compute_unit(ComputeUnit::Gpu);
                    continue;
                }

                // Fully ANE-eligible.
                op.set_compute_unit(ComputeUnit::Ane);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mil_rs::ir::Function;
    use mil_rs::ir::Operation;
    use mil_rs::ir::ScalarType;
    use mil_rs::ir::TensorType;
    use mil_rs::ir::Value;

    #[test]
    fn annotates_relu_as_ane() {
        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");

        let input_type = TensorType::new(ScalarType::Float16, vec![1, 32, 64, 64]);
        func.inputs.push(("input".to_string(), input_type));

        let relu = Operation::new("relu", "relu_0")
            .with_input("x", Value::Reference("input".into()))
            .with_output("relu_out");
        func.body.add_op(relu);
        func.body.outputs.push("relu_out".into());
        program.add_function(func);

        ComputeUnitAnnotationPass.run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.compute_unit(), Some(ComputeUnit::Ane));
    }

    #[test]
    fn annotates_unsupported_op_as_cpu() {
        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");

        let op = Operation::new("custom_unknown_op", "custom_0")
            .with_input("x", Value::Reference("input".into()))
            .with_output("out");
        func.body.add_op(op);
        func.body.outputs.push("out".into());
        program.add_function(func);

        ComputeUnitAnnotationPass.run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.compute_unit(), Some(ComputeUnit::Cpu));
    }

    #[test]
    fn annotates_const_as_any() {
        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");

        let const_op = Operation::new("const", "const_0")
            .with_input(
                "val",
                Value::Tensor {
                    data: vec![0u8; 4].into(),
                    shape: vec![1],
                    dtype: ScalarType::Float32,
                },
            )
            .with_output("const_out");
        func.body.add_op(const_op);
        func.body.outputs.push("const_out".into());
        program.add_function(func);

        ComputeUnitAnnotationPass.run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.compute_unit(), Some(ComputeUnit::Any));
    }

    #[test]
    fn annotates_conv_with_large_kernel_as_gpu() {
        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");

        let input_type = TensorType::new(ScalarType::Float16, vec![1, 32, 64, 64]);
        func.inputs.push(("input".to_string(), input_type));

        // weight shape [64, 32, 32, 32] — kernel 32×32 exceeds ANE limit of 16.
        let weight_type = TensorType::new(ScalarType::Float16, vec![64, 32, 32, 32]);
        let weight_const = Operation::new("const", "weight")
            .with_input(
                "val",
                Value::Tensor {
                    data: vec![0u8; 64 * 32 * 32 * 32 * 2].into(),
                    shape: vec![64, 32, 32, 32],
                    dtype: ScalarType::Float16,
                },
            )
            .with_output("weight_out");
        let mut weight_const = weight_const;
        weight_const.output_types = vec![Some(weight_type)];

        let conv = Operation::new("conv", "conv_0")
            .with_input("x", Value::Reference("input".into()))
            .with_input("weight", Value::Reference("weight_out".into()))
            .with_output("conv_out");

        func.body.add_op(weight_const);
        func.body.add_op(conv);
        func.body.outputs.push("conv_out".into());
        program.add_function(func);

        ComputeUnitAnnotationPass.run(&mut program).unwrap();

        let conv_op = &program.functions["main"].body.operations[1];
        assert_eq!(conv_op.op_type, "conv");
        assert_eq!(conv_op.compute_unit(), Some(ComputeUnit::Gpu));
    }
}
