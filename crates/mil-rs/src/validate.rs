//! ANE (Apple Neural Engine) compatibility validation.
//!
//! Analyzes a MIL IR [`Program`] and reports which operations are expected to
//! run on the Neural Engine versus falling back to CPU or GPU. The validator
//! is **informational only** — it never modifies the program.

use crate::ir::{Operation, Program, ScalarType, TensorType};

/// Maximum size for any single tensor dimension on the ANE.
const ANE_MAX_DIM: usize = 16384;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of ANE compatibility analysis for a [`Program`].
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Operations compatible with the ANE.
    pub ane_compatible: Vec<OpReport>,
    /// Operations that will fall back to CPU or GPU.
    pub fallback_ops: Vec<OpReport>,
    /// Warnings and recommendations.
    pub warnings: Vec<String>,
    /// Overall ANE compatibility percentage (0.0–100.0).
    ///
    /// `const` ops are excluded from the calculation because they represent
    /// static data and are always compatible.
    pub compatibility_pct: f64,
}

/// Per-operation compatibility report.
#[derive(Debug, Clone)]
pub struct OpReport {
    /// Operation name in the graph.
    pub name: String,
    /// MIL operation type.
    pub op_type: String,
    /// Why this op falls back (empty for compatible ops).
    pub reason: Option<String>,
}

// ---------------------------------------------------------------------------
// Core validation
// ---------------------------------------------------------------------------

/// Validate a MIL [`Program`] for ANE compatibility.
///
/// Iterates every operation across all functions and checks:
/// 1. Whether the `op_type` is in the ANE-supported set.
/// 2. Whether function inputs contain dynamic (unknown) dimensions.
/// 3. Whether data types are ANE-friendly.
/// 4. Whether tensor dimensions exceed the ANE per-dimension limit.
pub fn validate_ane_compatibility(program: &Program) -> ValidationReport {
    let mut ane_compatible = Vec::new();
    let mut fallback_ops = Vec::new();
    let mut warnings = Vec::new();

    // Check function-level properties (inputs).
    for (func_name, func) in &program.functions {
        for (input_name, ty) in &func.inputs {
            check_tensor_type(func_name, input_name, ty, &mut warnings);
        }

        // Check each operation.
        for op in &func.body.operations {
            let report = check_operation(op);
            if report.reason.is_some() {
                fallback_ops.push(report);
            } else {
                ane_compatible.push(report);
            }
        }
    }

    // Compute percentage, excluding `const` ops from the denominator.
    let non_const_compatible = ane_compatible
        .iter()
        .filter(|r| r.op_type != "const")
        .count();
    let non_const_fallback = fallback_ops
        .iter()
        .filter(|r| r.op_type != "const")
        .count();
    let non_const_total = non_const_compatible + non_const_fallback;

    let compatibility_pct = if non_const_total == 0 {
        100.0
    } else {
        (non_const_compatible as f64 / non_const_total as f64) * 100.0
    };

    ValidationReport {
        ane_compatible,
        fallback_ops,
        warnings,
        compatibility_pct,
    }
}

/// Print a human-readable validation report to stdout.
pub fn print_validation_report(report: &ValidationReport) {
    println!("ANE Compatibility Report");
    println!("========================");
    println!(
        "  Compatible ops : {}",
        report.ane_compatible.len()
    );
    println!(
        "  Fallback ops   : {}",
        report.fallback_ops.len()
    );
    println!(
        "  Compatibility  : {:.1}%",
        report.compatibility_pct
    );

    if !report.fallback_ops.is_empty() {
        println!();
        println!("Fallback operations (will run on CPU/GPU):");
        for op in &report.fallback_ops {
            let reason = op.reason.as_deref().unwrap_or("unknown");
            println!("  • {} ({}): {}", op.name, op.op_type, reason);
        }
    }

    if !report.warnings.is_empty() {
        println!();
        println!("Warnings:");
        for w in &report.warnings {
            println!("  ⚠ {w}");
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check a single operation for ANE compatibility.
fn check_operation(op: &Operation) -> OpReport {
    let reason = if !is_ane_supported(&op.op_type) {
        Some(format!(
            "op '{}' is not supported on the Neural Engine",
            op.op_type
        ))
    } else {
        None
    };

    OpReport {
        name: op.name.clone(),
        op_type: op.op_type.clone(),
        reason,
    }
}

/// Check a function input's tensor type for ANE-unfriendly properties and
/// push any warnings to the list.
fn check_tensor_type(
    func_name: &str,
    input_name: &str,
    ty: &TensorType,
    warnings: &mut Vec<String>,
) {
    // Dynamic dimensions.
    if !ty.is_static() {
        warnings.push(format!(
            "function '{func_name}' input '{input_name}' has dynamic dimensions — \
             ANE requires fully static shapes"
        ));
    }

    // Per-dimension size limit.
    for (i, dim) in ty.shape.iter().enumerate() {
        if let Some(d) = dim {
            if *d > ANE_MAX_DIM {
                warnings.push(format!(
                    "function '{func_name}' input '{input_name}' dim {i} = {d} \
                     exceeds ANE limit of {ANE_MAX_DIM}"
                ));
            }
        }
    }

    // Data type check.
    if !is_ane_dtype(ty.scalar_type) {
        warnings.push(format!(
            "function '{func_name}' input '{input_name}' uses {:?} — \
             ANE supports Float16, Float32, and Int8",
            ty.scalar_type
        ));
    }
}

/// Returns `true` if the MIL operation type is known to run on the ANE.
pub fn is_ane_supported(op_type: &str) -> bool {
    matches!(
        op_type,
        "conv"
            | "matmul"
            | "linear"
            | "relu"
            | "sigmoid"
            | "tanh"
            | "softmax"
            | "batch_norm"
            | "add"
            | "mul"
            | "sub"
            | "real_div"
            | "concat"
            | "reshape"
            | "transpose"
            | "max_pool"
            | "avg_pool"
            | "reduce_mean"
            | "layer_norm"
            | "squeeze"
            | "expand_dims"
            | "pad"
            | "cast"
            | "const"
            | "clip"
            | "gather"
            | "split"
            | "slice_by_index"
            | "pow"
            | "sqrt"
            | "select"
            | "scaled_dot_product_attention"
    )
}

/// Returns `true` if the scalar type is ANE-friendly.
fn is_ane_dtype(scalar_type: ScalarType) -> bool {
    matches!(
        scalar_type,
        ScalarType::Float16 | ScalarType::Float32 | ScalarType::Int8
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Block, Function, Operation, Program};

    /// Build a minimal program with the given operations in a single function.
    fn make_program(ops: Vec<Operation>) -> Program {
        let mut func = Function::new("main");
        func.inputs
            .push(("input".into(), TensorType::new(ScalarType::Float32, vec![1, 3, 224, 224])));
        func.body = Block {
            operations: ops,
            outputs: vec!["out".into()],
        };
        let mut program = Program::new("1.0.0");
        program.add_function(func);
        program
    }

    #[test]
    fn all_compatible_ops() {
        let ops = vec![
            Operation::new("relu", "relu_0").with_output("r0"),
            Operation::new("sigmoid", "sigmoid_0").with_output("s0"),
            Operation::new("add", "add_0").with_output("out"),
        ];
        let report = validate_ane_compatibility(&make_program(ops));

        assert_eq!(report.ane_compatible.len(), 3);
        assert!(report.fallback_ops.is_empty());
        assert!((report.compatibility_pct - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn unsupported_ops_in_fallback() {
        let ops = vec![
            Operation::new("relu", "relu_0").with_output("r0"),
            Operation::new("erf", "erf_0").with_output("e0"),
            Operation::new("custom_op", "custom_0").with_output("out"),
        ];
        let report = validate_ane_compatibility(&make_program(ops));

        assert_eq!(report.ane_compatible.len(), 1);
        assert_eq!(report.fallback_ops.len(), 2);

        let names: Vec<&str> = report.fallback_ops.iter().map(|r| r.op_type.as_str()).collect();
        assert!(names.contains(&"erf"));
        assert!(names.contains(&"custom_op"));

        // 1 compatible out of 3 non-const ops ≈ 33.3%
        assert!((report.compatibility_pct - 100.0 / 3.0).abs() < 0.1);
    }

    #[test]
    fn dynamic_shapes_produce_warning() {
        let mut func = Function::new("main");
        func.inputs.push((
            "input".into(),
            TensorType::with_dynamic_shape(
                ScalarType::Float32,
                vec![None, Some(3), Some(224), Some(224)],
            ),
        ));
        func.body = Block {
            operations: vec![Operation::new("relu", "relu_0").with_output("out")],
            outputs: vec!["out".into()],
        };

        let mut program = Program::new("1.0.0");
        program.add_function(func);

        let report = validate_ane_compatibility(&program);
        assert!(
            report.warnings.iter().any(|w| w.contains("dynamic")),
            "expected a dynamic-shape warning, got: {:?}",
            report.warnings
        );
    }

    #[test]
    fn empty_program() {
        let program = Program::new("1.0.0");
        let report = validate_ane_compatibility(&program);

        assert!(report.ane_compatible.is_empty());
        assert!(report.fallback_ops.is_empty());
        assert!(report.warnings.is_empty());
        assert!((report.compatibility_pct - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn const_ops_excluded_from_percentage() {
        let ops = vec![
            Operation::new("const", "weight").with_output("w"),
            Operation::new("const", "bias").with_output("b"),
            Operation::new("relu", "relu_0").with_output("out"),
        ];
        let report = validate_ane_compatibility(&make_program(ops));

        assert_eq!(report.ane_compatible.len(), 3);
        // Only 1 non-const op, and it's compatible → 100%.
        assert!((report.compatibility_pct - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn const_ops_counted_as_compatible() {
        let ops = vec![
            Operation::new("const", "c0").with_output("c0_out"),
        ];
        let report = validate_ane_compatibility(&make_program(ops));

        assert_eq!(report.ane_compatible.len(), 1);
        assert_eq!(report.ane_compatible[0].op_type, "const");
        assert!(report.fallback_ops.is_empty());
    }

    #[test]
    fn oversized_dimension_produces_warning() {
        let mut func = Function::new("main");
        func.inputs.push((
            "big".into(),
            TensorType::new(ScalarType::Float32, vec![1, 3, 32768]),
        ));
        func.body = Block {
            operations: vec![Operation::new("relu", "r0").with_output("out")],
            outputs: vec!["out".into()],
        };

        let mut program = Program::new("1.0.0");
        program.add_function(func);

        let report = validate_ane_compatibility(&program);
        assert!(
            report.warnings.iter().any(|w| w.contains("exceeds ANE limit")),
            "expected an oversized-dimension warning, got: {:?}",
            report.warnings
        );
    }

    #[test]
    fn unsupported_dtype_produces_warning() {
        let mut func = Function::new("main");
        func.inputs.push((
            "input".into(),
            TensorType::new(ScalarType::Float64, vec![1, 10]),
        ));
        func.body = Block {
            operations: vec![Operation::new("relu", "r0").with_output("out")],
            outputs: vec!["out".into()],
        };

        let mut program = Program::new("1.0.0");
        program.add_function(func);

        let report = validate_ane_compatibility(&program);
        assert!(
            report.warnings.iter().any(|w| w.contains("Float64")),
            "expected a dtype warning, got: {:?}",
            report.warnings
        );
    }

    #[test]
    fn expand_dims_is_ane_supported() {
        assert!(is_ane_supported("expand_dims"));
    }

    #[test]
    fn slice_by_index_is_ane_supported() {
        assert!(is_ane_supported("slice_by_index"));
    }
}
