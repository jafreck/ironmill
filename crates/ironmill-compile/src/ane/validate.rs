//! ANE (Apple Neural Engine) compatibility validation.
//!
//! Analyzes a MIL IR [`Program`] and reports which operations are expected to
//! run on the Neural Engine versus falling back to CPU or GPU. The validator
//! is **informational only** — it never modifies the program.
//!
//! In addition to the static op-type allowlist, the validator performs
//! shape-aware checks informed by Orion research — ANE eligibility depends on
//! tensor shapes, data types, and memory alignment, not just op type.

use std::collections::HashMap;

use serde::Serialize;

use mil_rs::ir::{Function, Operation, Program, ScalarType, TensorType, Value};

/// Maximum size for any single tensor dimension on the ANE.
const ANE_MAX_DIM: usize = 16384;

/// Maximum convolution kernel dimension on the ANE (16×16).
const ANE_MAX_CONV_KERNEL: usize = 16;

/// Maximum inner dimension for matmul/linear ops on the ANE.
const ANE_MAX_INNER_DIM: usize = 16384;

/// ANE channel alignment requirement (elements).
const ANE_CHANNEL_ALIGN: usize = 32;

/// ANE memory alignment requirement (bytes).
const ANE_BYTE_ALIGNMENT: usize = 64;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of ANE compatibility analysis for a [`Program`].
#[derive(Debug, Clone, Serialize)]
pub struct ValidationReport {
    /// Operations compatible with the ANE.
    pub ane_compatible: Vec<OpReport>,
    /// Operations that will fall back to CPU or GPU.
    pub fallback_ops: Vec<OpReport>,
    /// Warnings and recommendations.
    pub warnings: Vec<String>,
    /// Overall ANE compatibility percentage (0.0–100.0) by op count.
    ///
    /// `const` ops are excluded from the calculation because they represent
    /// static data and are always compatible.
    pub compatibility_pct: f64,
    /// Estimated percentage of compute (FLOPs) that can run on the ANE.
    pub ane_compute_pct: f64,
    /// Total estimated FLOPs across all non-const operations.
    pub total_estimated_flops: u64,
    /// Estimated FLOPs for ANE-eligible operations.
    pub ane_estimated_flops: u64,
}

/// Per-operation compatibility report.
#[derive(Debug, Clone, Serialize)]
pub struct OpReport {
    /// Operation name in the graph.
    pub name: String,
    /// MIL operation type.
    pub op_type: String,
    /// Why this op falls back (empty for compatible ops).
    pub reason: Option<String>,
    /// Performance annotations for ops that are technically supported but may
    /// perform poorly on the ANE.
    pub performance_annotations: Vec<String>,
    /// Estimated FLOPs for this operation (0 when shape info is unavailable).
    pub estimated_flops: u64,
    /// Whether this operation is eligible to run on the ANE.
    pub ane_eligible: bool,
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
/// 5. Shape-aware constraints (kernel sizes, inner dimensions, alignment).
/// 6. Performance annotations for ops with suboptimal ANE patterns.
pub fn validate_ane_compatibility(program: &Program) -> ValidationReport {
    let mut ane_compatible = Vec::new();
    let mut fallback_ops = Vec::new();
    let mut warnings = Vec::new();

    // Check function-level properties (inputs).
    for (func_name, func) in &program.functions {
        for (input_name, ty) in &func.inputs {
            check_tensor_type(func_name, input_name, ty, &mut warnings);
        }

        let type_map = build_type_map(func);

        // Check each operation.
        for op in &func.body.operations {
            let report = check_operation(op, &type_map);
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
    let non_const_fallback = fallback_ops.iter().filter(|r| r.op_type != "const").count();
    let non_const_total = non_const_compatible + non_const_fallback;

    let compatibility_pct = if non_const_total == 0 {
        100.0
    } else {
        (non_const_compatible as f64 / non_const_total as f64) * 100.0
    };

    // Compute FLOP-based ANE vs CPU/GPU split.
    let ane_estimated_flops: u64 = ane_compatible
        .iter()
        .filter(|r| r.op_type != "const")
        .map(|r| r.estimated_flops)
        .sum();
    let fallback_flops: u64 = fallback_ops
        .iter()
        .filter(|r| r.op_type != "const")
        .map(|r| r.estimated_flops)
        .sum();
    let total_estimated_flops = ane_estimated_flops + fallback_flops;

    let ane_compute_pct = if total_estimated_flops == 0 {
        100.0
    } else {
        (ane_estimated_flops as f64 / total_estimated_flops as f64) * 100.0
    };

    ValidationReport {
        ane_compatible,
        fallback_ops,
        warnings,
        compatibility_pct,
        ane_compute_pct,
        total_estimated_flops,
        ane_estimated_flops,
    }
}

/// Print a human-readable validation report to stdout.
pub fn print_validation_report(report: &ValidationReport) {
    println!("ANE Compatibility Report");
    println!("========================");
    println!("  Compatible ops : {}", report.ane_compatible.len());
    println!("  Fallback ops   : {}", report.fallback_ops.len());
    println!("  Compatibility  : {:.1}%", report.compatibility_pct);
    println!(
        "  ANE compute    : {:.1}% of estimated FLOPs",
        report.ane_compute_pct
    );

    if report.total_estimated_flops > 0 {
        println!(
            "  Total FLOPs    : {} (ANE: {}, fallback: {})",
            report.total_estimated_flops,
            report.ane_estimated_flops,
            report.total_estimated_flops - report.ane_estimated_flops,
        );
    }

    if !report.fallback_ops.is_empty() {
        println!();
        println!("Fallback operations (will run on CPU/GPU):");
        for op in &report.fallback_ops {
            let reason = op.reason.as_deref().unwrap_or("unknown");
            println!("  • {} ({}): {}", op.name, op.op_type, reason);
        }
    }

    // Performance annotations.
    let annotated: Vec<&OpReport> = report
        .ane_compatible
        .iter()
        .chain(report.fallback_ops.iter())
        .filter(|r| !r.performance_annotations.is_empty())
        .collect();

    if !annotated.is_empty() {
        println!();
        println!("Performance annotations:");
        for op in annotated {
            for ann in &op.performance_annotations {
                println!("  ⚡ {} ({}): {}", op.name, op.op_type, ann);
            }
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

/// Serialize a validation report to a JSON string.
pub fn validation_report_to_json(report: &ValidationReport) -> String {
    serde_json::to_string_pretty(report).expect("report serialization should not fail")
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check a single operation for ANE compatibility, including shape-aware
/// constraints and performance annotations.
fn check_operation(op: &Operation, type_map: &HashMap<String, TensorType>) -> OpReport {
    let mut reason = None;
    let mut performance_annotations = Vec::new();

    // 1. Op-type allowlist check.
    if !is_ane_supported(&op.op_type) {
        reason = Some(format!(
            "op '{}' is not supported on the Neural Engine",
            op.op_type
        ));
    }

    // 2. Shape-aware constraint checks (only for ops that passed the allowlist).
    if reason.is_none() {
        let (shape_reason, annotations) = check_shape_constraints(op, type_map);
        if let Some(r) = shape_reason {
            reason = Some(r);
        }
        performance_annotations.extend(annotations);
    }

    // 3. Performance annotations (independent of eligibility).
    performance_annotations.extend(check_performance(op, type_map));

    let estimated_flops = estimate_flops(op, type_map);
    let ane_eligible = reason.is_none();

    OpReport {
        name: op.name.clone(),
        op_type: op.op_type.clone(),
        reason,
        performance_annotations,
        estimated_flops,
        ane_eligible,
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
    // Verified against Apple's private ANE compiler via ane_op_probe.
    // 59 ops are also eval-verified (numerical correctness confirmed).
    // See docs/research/ane-op-support-matrix.md for full results.
    matches!(
        op_type,
        // Arithmetic & elementwise
        "add"
            | "sub"
            | "mul"
            | "real_div"
            | "maximum"
            | "minimum"
            | "floor_div"
            | "abs"
            | "sign"
            | "sqrt"
            | "square"
            | "exp"
            | "exp2"
            | "erf"
            | "ceil"
            | "floor"
            | "round"
            | "atan"
            | "pow"
            | "clip"
            // Activations
            | "relu"
            | "relu6"
            | "sigmoid"
            | "tanh"
            | "softmax"
            | "silu"
            | "softsign"
            | "softplus"
            // Comparison (returns bool)
            | "greater"
            | "greater_equal"
            | "less"
            | "less_equal"
            | "equal"
            | "not_equal"
            // Reductions
            | "reduce_sum"
            | "reduce_mean"
            | "reduce_max"
            | "reduce_min"
            | "reduce_l2_norm"
            | "reduce_l1_norm"
            | "reduce_log_sum"
            | "reduce_log_sum_exp"
            | "reduce_sum_square"
            // Linear algebra & normalization
            | "conv"
            | "matmul"
            | "linear"
            | "layer_norm"
            // Shape manipulation
            | "reshape"
            | "transpose"
            | "concat"
            | "slice_by_index"
            | "slice_by_size"
            | "split"
            | "expand_dims"
            | "squeeze"
            | "tile"
            | "reverse"
            | "identity"
            // Conditional & logical
            | "select"
            | "logical_not"
            // Type casting & quantization
            | "cast"
            | "dequantize"
            // Constants & attention
            | "const"
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
// Type map
// ---------------------------------------------------------------------------

/// Build a map from output value name to [`TensorType`] using function inputs
/// and operation output types.
pub fn build_type_map(func: &Function) -> HashMap<String, TensorType> {
    let mut types = HashMap::new();

    for (name, ty) in &func.inputs {
        types.insert(name.clone(), ty.clone());
    }

    for op in &func.body.operations {
        for (i, out_name) in op.outputs.iter().enumerate() {
            if let Some(Some(ty)) = op.output_types.get(i) {
                types.insert(out_name.clone(), ty.clone());
            }
        }

        // For const ops, extract the type from the tensor value.
        if op.op_type == "const" {
            if let Some(Value::Tensor { shape, dtype, .. }) = op.inputs.get("val") {
                let ty = TensorType::new(*dtype, shape.clone());
                for out_name in &op.outputs {
                    types.entry(out_name.clone()).or_insert_with(|| ty.clone());
                }
            }
        }
    }

    types
}

/// Resolve the shape of a named input to an operation via the type map.
fn resolve_input_shape(
    op: &Operation,
    input_name: &str,
    type_map: &HashMap<String, TensorType>,
) -> Option<Vec<Option<usize>>> {
    match op.inputs.get(input_name)? {
        Value::Reference(name) => type_map.get(name).map(|t| t.shape.clone()),
        Value::Tensor { shape, .. } => Some(shape.iter().map(|&d| Some(d)).collect()),
        _ => None,
    }
}

/// Get the byte size of the element type for a named input.
fn resolve_input_dtype_bytes(
    op: &Operation,
    input_name: &str,
    type_map: &HashMap<String, TensorType>,
) -> usize {
    let dtype = match op.inputs.get(input_name) {
        Some(Value::Reference(name)) => type_map.get(name).map(|t| t.scalar_type),
        Some(Value::Tensor { dtype, .. }) => Some(*dtype),
        _ => None,
    };

    match dtype {
        Some(ScalarType::Float16 | ScalarType::Int16 | ScalarType::UInt16) => 2,
        Some(ScalarType::Float32 | ScalarType::Int32 | ScalarType::UInt32) => 4,
        Some(ScalarType::Float64 | ScalarType::Int64 | ScalarType::UInt64) => 8,
        Some(ScalarType::Int8 | ScalarType::UInt8 | ScalarType::Bool) => 1,
        None => 4,
    }
}

/// Extract an integer list from an operation attribute.
fn get_attr_int_list(op: &Operation, attr_name: &str) -> Option<Vec<i64>> {
    match op.attributes.get(attr_name)? {
        Value::List(items) => items
            .iter()
            .map(|v| match v {
                Value::Int(n) => Some(*n),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Shape-aware constraint checks
// ---------------------------------------------------------------------------

/// Check shape-aware ANE constraints for a single operation.
///
/// Returns a rejection reason (if the op cannot run on ANE due to shape
/// constraints) and a list of performance annotations.
pub fn check_shape_constraints(
    op: &Operation,
    type_map: &HashMap<String, TensorType>,
) -> (Option<String>, Vec<String>) {
    match op.op_type.as_str() {
        "conv" => check_conv_constraints(op, type_map),
        "matmul" | "linear" => check_matmul_constraints(op, type_map),
        _ => check_alignment_constraints(op, type_map),
    }
}

/// Conv: kernel ≤ 16×16, input channels aligned to 32.
fn check_conv_constraints(
    op: &Operation,
    type_map: &HashMap<String, TensorType>,
) -> (Option<String>, Vec<String>) {
    let mut annotations = Vec::new();

    // Kernel dimensions from the weight tensor: [C_out, C_in/groups, K_h, K_w].
    let weight_shape = resolve_input_shape(op, "weight", type_map);
    if let Some(ref shape) = weight_shape {
        if shape.len() >= 4 {
            if let (Some(kh), Some(kw)) = (shape[2], shape[3]) {
                if kh > ANE_MAX_CONV_KERNEL || kw > ANE_MAX_CONV_KERNEL {
                    return (
                        Some(format!(
                            "conv kernel {kh}\u{00d7}{kw} exceeds ANE limit of \
                             {ANE_MAX_CONV_KERNEL}\u{00d7}{ANE_MAX_CONV_KERNEL}"
                        )),
                        annotations,
                    );
                }
            }
        }
    }

    // Input channel alignment.
    let input_shape = resolve_input_shape(op, "x", type_map);
    if let Some(ref shape) = input_shape {
        if shape.len() >= 2 {
            if let Some(channels) = shape[1] {
                if channels % ANE_CHANNEL_ALIGN != 0 {
                    annotations.push(format!(
                        "input channels ({channels}) not aligned to \
                         {ANE_CHANNEL_ALIGN} \u{2014} may reduce ANE throughput"
                    ));
                }
            }
        }
    }

    (None, annotations)
}

/// MatMul/Linear: inner dimension ≤ 16384.
fn check_matmul_constraints(
    op: &Operation,
    type_map: &HashMap<String, TensorType>,
) -> (Option<String>, Vec<String>) {
    let annotations = Vec::new();

    // For matmul, x is [..., M, K] — inner dim is the last dim of x.
    if op.op_type == "matmul" {
        let x_shape = resolve_input_shape(op, "x", type_map);
        if let Some(ref shape) = x_shape {
            if let Some(&Some(inner)) = shape.last() {
                if inner > ANE_MAX_INNER_DIM {
                    return (
                        Some(format!(
                            "inner dimension ({inner}) exceeds ANE limit of {ANE_MAX_INNER_DIM}"
                        )),
                        annotations,
                    );
                }
            }
        }
    }

    // For linear, weight shape is [out_features, in_features].
    if op.op_type == "linear" {
        let weight_shape = resolve_input_shape(op, "weight", type_map);
        if let Some(ref shape) = weight_shape {
            if shape.len() >= 2 {
                if let Some(in_features) = shape[1] {
                    if in_features > ANE_MAX_INNER_DIM {
                        return (
                            Some(format!(
                                "inner dimension ({in_features}) exceeds ANE limit of \
                                 {ANE_MAX_INNER_DIM}"
                            )),
                            annotations,
                        );
                    }
                }
            }
        }
    }

    (None, annotations)
}

/// Check memory alignment constraints for the ANE.
///
/// Verifies that the innermost tensor dimension results in byte-aligned
/// memory accesses (aligned to 64 bytes).
fn check_alignment_constraints(
    op: &Operation,
    type_map: &HashMap<String, TensorType>,
) -> (Option<String>, Vec<String>) {
    let mut annotations = Vec::new();

    let input_shape = resolve_input_shape(op, "x", type_map);
    if let Some(ref shape) = input_shape {
        if let Some(&Some(innermost)) = shape.last() {
            let elem_bytes = resolve_input_dtype_bytes(op, "x", type_map);
            let stride_bytes = innermost * elem_bytes;
            if stride_bytes % ANE_BYTE_ALIGNMENT != 0 {
                annotations.push(format!(
                    "innermost stride ({stride_bytes} bytes) not aligned to \
                     {ANE_BYTE_ALIGNMENT} bytes \u{2014} may cause slower ANE memory access"
                ));
            }
        }
    }

    (None, annotations)
}

// ---------------------------------------------------------------------------
// Performance annotations
// ---------------------------------------------------------------------------

/// Generate performance annotations for ops that are technically supported
/// but may perform poorly on the ANE.
pub fn check_performance(op: &Operation, type_map: &HashMap<String, TensorType>) -> Vec<String> {
    match op.op_type.as_str() {
        "gather" => check_gather_performance(op, type_map),
        "transpose" => check_transpose_performance(op),
        "reshape" => check_reshape_performance(op, type_map),
        _ => Vec::new(),
    }
}

/// Large gather operations with non-contiguous access patterns perform poorly
/// on the ANE.
fn check_gather_performance(op: &Operation, type_map: &HashMap<String, TensorType>) -> Vec<String> {
    let mut annotations = Vec::new();

    let input_shape = resolve_input_shape(op, "x", type_map);
    if let Some(ref shape) = input_shape {
        let total_elements: usize = shape.iter().filter_map(|d| *d).product();
        if total_elements > 65536 {
            annotations.push(format!(
                "large gather over {total_elements} elements \u{2014} \
                 non-contiguous memory access may cause ANE stalls"
            ));
        }
    }

    let indices_shape = resolve_input_shape(op, "indices", type_map);
    if let Some(ref shape) = indices_shape {
        let total_indices: usize = shape.iter().filter_map(|d| *d).product();
        if total_indices > 32768 {
            annotations.push(format!(
                "gather with {total_indices} indices \u{2014} consider splitting \
                 for better ANE utilization"
            ));
        }
    }

    annotations
}

/// Transpose ops that move the channel dimension may force expensive data
/// reformatting on the ANE.
fn check_transpose_performance(op: &Operation) -> Vec<String> {
    let mut annotations = Vec::new();

    if let Some(Value::List(perm)) = op.attributes.get("perm") {
        if perm.len() >= 2 {
            if let Some(Value::Int(dest)) = perm.get(1) {
                if *dest != 1 {
                    annotations.push(
                        "transpose moves channel dimension \u{2014} may force \
                         expensive reformatting on ANE"
                            .to_string(),
                    );
                }
            }
        }
    }

    annotations
}

/// Reshapes of very large tensors can cause memory copy overhead on the ANE.
fn check_reshape_performance(
    op: &Operation,
    type_map: &HashMap<String, TensorType>,
) -> Vec<String> {
    let mut annotations = Vec::new();

    let input_shape = resolve_input_shape(op, "x", type_map);
    if let Some(ref shape) = input_shape {
        let total_elements: usize = shape.iter().filter_map(|d| *d).product();
        if total_elements > 1_000_000 {
            annotations.push(format!(
                "reshape of large tensor ({total_elements} elements) \u{2014} \
                 may cause memory copy overhead on ANE"
            ));
        }
    }

    annotations
}

// ---------------------------------------------------------------------------
// FLOPs estimation
// ---------------------------------------------------------------------------

/// Estimate FLOPs for a single operation based on its type and available shape
/// information. Returns 0 when shapes are unknown or for zero-cost ops.
fn estimate_flops(op: &Operation, type_map: &HashMap<String, TensorType>) -> u64 {
    match op.op_type.as_str() {
        "conv" => estimate_conv_flops(op, type_map),
        "matmul" => estimate_matmul_flops(op, type_map),
        "linear" => estimate_linear_flops(op, type_map),
        "relu" | "sigmoid" | "tanh" | "clip" | "sqrt" | "pow" | "cast" => {
            estimate_elementwise_flops(op, type_map)
        }
        "softmax" => estimate_elementwise_flops(op, type_map).saturating_mul(5),
        "batch_norm" | "layer_norm" => estimate_elementwise_flops(op, type_map).saturating_mul(4),
        "add" | "mul" | "sub" | "real_div" | "select" => estimate_elementwise_flops(op, type_map),
        "reduce_mean" => estimate_elementwise_flops(op, type_map),
        "max_pool" | "avg_pool" => estimate_pool_flops(op, type_map),
        "scaled_dot_product_attention" => estimate_attention_flops(op, type_map),
        // Memory/shape ops: no compute.
        "const" | "reshape" | "transpose" | "concat" | "squeeze" | "expand_dims" | "pad"
        | "split" | "slice_by_index" | "gather" => 0,
        _ => 0,
    }
}

/// Conv FLOPs = 2 * H_out * W_out * K_h * K_w * C_in_per_group * C_out.
fn estimate_conv_flops(op: &Operation, type_map: &HashMap<String, TensorType>) -> u64 {
    let weight_shape = resolve_input_shape(op, "weight", type_map);
    let (c_out, c_in_per_group, kh, kw) = match weight_shape {
        Some(ref s) if s.len() >= 4 => match (s[0], s[1], s[2], s[3]) {
            (Some(co), Some(ci), Some(kh), Some(kw)) => (co, ci, kh, kw),
            _ => return 0,
        },
        _ => return 0,
    };

    let (h_out, w_out) = output_spatial_dims(op, type_map).unwrap_or((0, 0));
    if h_out == 0 || w_out == 0 {
        return 0;
    }

    2u64.saturating_mul(h_out as u64)
        .saturating_mul(w_out as u64)
        .saturating_mul(kh as u64)
        .saturating_mul(kw as u64)
        .saturating_mul(c_in_per_group as u64)
        .saturating_mul(c_out as u64)
}

/// MatMul FLOPs = 2 * M * N * K.
fn estimate_matmul_flops(op: &Operation, type_map: &HashMap<String, TensorType>) -> u64 {
    let x_shape = resolve_input_shape(op, "x", type_map);
    let y_shape = resolve_input_shape(op, "y", type_map);

    let k = x_shape.as_ref().and_then(|s| s.last().copied()).flatten();
    let m = x_shape
        .as_ref()
        .and_then(|s| if s.len() >= 2 { s[s.len() - 2] } else { None });
    let n = y_shape.as_ref().and_then(|s| s.last().copied()).flatten();

    match (m, k, n) {
        (Some(m), Some(k), Some(n)) => 2u64
            .saturating_mul(m as u64)
            .saturating_mul(k as u64)
            .saturating_mul(n as u64),
        _ => 0,
    }
}

/// Linear FLOPs = 2 * batch * in_features * out_features.
fn estimate_linear_flops(op: &Operation, type_map: &HashMap<String, TensorType>) -> u64 {
    let weight_shape = resolve_input_shape(op, "weight", type_map);
    let input_shape = resolve_input_shape(op, "x", type_map);

    let (out_features, in_features) = match weight_shape {
        Some(ref s) if s.len() >= 2 => match (s[0], s[1]) {
            (Some(o), Some(i)) => (o, i),
            _ => return 0,
        },
        _ => return 0,
    };

    let batch: usize = input_shape
        .as_ref()
        .map(|s| {
            s.iter()
                .rev()
                .skip(1)
                .filter_map(|d| *d)
                .product::<usize>()
                .max(1)
        })
        .unwrap_or(1);

    2u64.saturating_mul(batch as u64)
        .saturating_mul(in_features as u64)
        .saturating_mul(out_features as u64)
}

/// Element-wise FLOPs = number of output elements.
fn estimate_elementwise_flops(op: &Operation, type_map: &HashMap<String, TensorType>) -> u64 {
    if let Some(Some(ty)) = op.output_types.first() {
        return ty.shape.iter().filter_map(|d| *d).product::<usize>() as u64;
    }
    if let Some(shape) = resolve_input_shape(op, "x", type_map) {
        return shape.iter().filter_map(|d| *d).product::<usize>() as u64;
    }
    0
}

/// Pool FLOPs = output_elements * kernel_size.
fn estimate_pool_flops(op: &Operation, type_map: &HashMap<String, TensorType>) -> u64 {
    let output_elements = estimate_elementwise_flops(op, type_map);
    let kernel_size = get_attr_int_list(op, "kernel_sizes")
        .map(|ks| ks.iter().product::<i64>().unsigned_abs())
        .unwrap_or(1);
    output_elements.saturating_mul(kernel_size)
}

/// SDPA FLOPs = 4 * batch_heads * seq_len^2 * head_dim.
fn estimate_attention_flops(op: &Operation, type_map: &HashMap<String, TensorType>) -> u64 {
    let q_shape = resolve_input_shape(op, "query", type_map);
    match q_shape {
        Some(ref s) if s.len() >= 3 => {
            let dims: Vec<usize> = s.iter().filter_map(|d| *d).collect();
            if dims.len() >= 3 {
                let batch_heads: u64 = dims[..dims.len() - 2].iter().map(|&d| d as u64).product();
                let seq_len = dims[dims.len() - 2] as u64;
                let head_dim = dims[dims.len() - 1] as u64;
                4u64.saturating_mul(batch_heads)
                    .saturating_mul(seq_len)
                    .saturating_mul(seq_len)
                    .saturating_mul(head_dim)
            } else {
                0
            }
        }
        _ => 0,
    }
}

/// Try to extract spatial output dimensions (H, W) for conv/pool ops.
fn output_spatial_dims(
    op: &Operation,
    type_map: &HashMap<String, TensorType>,
) -> Option<(usize, usize)> {
    if let Some(Some(ty)) = op.output_types.first() {
        if ty.shape.len() >= 4 {
            if let (Some(h), Some(w)) = (ty.shape[2], ty.shape[3]) {
                return Some((h, w));
            }
        }
    }
    // Fall back to input spatial dims (approximation ignoring stride/padding).
    let input_shape = resolve_input_shape(op, "x", type_map)?;
    if input_shape.len() >= 4 {
        if let (Some(h), Some(w)) = (input_shape[2], input_shape[3]) {
            return Some((h, w));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use mil_rs::ir::{Block, Function, Operation, Program};

    /// Build a minimal program with the given operations in a single function.
    fn make_program(ops: Vec<Operation>) -> Program {
        let mut func = Function::new("main");
        func.inputs.push((
            "input".into(),
            TensorType::new(ScalarType::Float32, vec![1, 3, 224, 224]),
        ));
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
            Operation::new("scatter", "scatter_0").with_output("s0"),
            Operation::new("custom_op", "custom_0").with_output("out"),
        ];
        let report = validate_ane_compatibility(&make_program(ops));

        assert_eq!(report.ane_compatible.len(), 1);
        assert_eq!(report.fallback_ops.len(), 2);

        let names: Vec<&str> = report
            .fallback_ops
            .iter()
            .map(|r| r.op_type.as_str())
            .collect();
        assert!(names.contains(&"scatter"));
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
        let ops = vec![Operation::new("const", "c0").with_output("c0_out")];
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
            report
                .warnings
                .iter()
                .any(|w| w.contains("exceeds ANE limit")),
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

    // -----------------------------------------------------------------------
    // Shape-aware validation tests
    // -----------------------------------------------------------------------

    #[test]
    fn conv_oversized_kernel_rejected() {
        // Weight shape: [64, 3, 32, 32] — 32×32 kernel exceeds 16×16 limit.
        let weight = Operation::new("const", "weight")
            .with_output("w")
            .with_input(
                "val",
                Value::Tensor {
                    data: vec![0; 64 * 3 * 32 * 32 * 4],
                    shape: vec![64, 3, 32, 32],
                    dtype: ScalarType::Float32,
                },
            );

        let conv = Operation::new("conv", "conv_0")
            .with_input("x", Value::Reference("input".into()))
            .with_input("weight", Value::Reference("w".into()))
            .with_output("out");

        let report = validate_ane_compatibility(&make_program(vec![weight, conv]));

        assert_eq!(report.fallback_ops.len(), 1);
        assert!(
            report.fallback_ops[0]
                .reason
                .as_ref()
                .unwrap()
                .contains("kernel"),
            "expected kernel rejection, got: {:?}",
            report.fallback_ops[0].reason
        );
    }

    #[test]
    fn conv_valid_kernel_accepted() {
        // Weight shape: [64, 3, 3, 3] — 3×3 kernel within 16×16 limit.
        let weight = Operation::new("const", "weight")
            .with_output("w")
            .with_input(
                "val",
                Value::Tensor {
                    data: vec![0; 64 * 3 * 3 * 3 * 4],
                    shape: vec![64, 3, 3, 3],
                    dtype: ScalarType::Float32,
                },
            );

        let conv = Operation::new("conv", "conv_0")
            .with_input("x", Value::Reference("input".into()))
            .with_input("weight", Value::Reference("w".into()))
            .with_output("out");

        let report = validate_ane_compatibility(&make_program(vec![weight, conv]));

        assert!(
            report.ane_compatible.iter().any(|r| r.op_type == "conv"),
            "expected conv to be compatible"
        );
    }

    #[test]
    fn conv_unaligned_channels_annotation() {
        // Input has 3 channels, not aligned to 32.
        let weight = Operation::new("const", "weight")
            .with_output("w")
            .with_input(
                "val",
                Value::Tensor {
                    data: vec![0; 64 * 3 * 3 * 3 * 4],
                    shape: vec![64, 3, 3, 3],
                    dtype: ScalarType::Float32,
                },
            );

        let conv = Operation::new("conv", "conv_0")
            .with_input("x", Value::Reference("input".into()))
            .with_input("weight", Value::Reference("w".into()))
            .with_output("out");

        let report = validate_ane_compatibility(&make_program(vec![weight, conv]));

        let conv_report = report
            .ane_compatible
            .iter()
            .find(|r| r.op_type == "conv")
            .expect("conv should be compatible");
        assert!(
            conv_report
                .performance_annotations
                .iter()
                .any(|a| a.contains("not aligned")),
            "expected channel alignment annotation, got: {:?}",
            conv_report.performance_annotations
        );
    }

    #[test]
    fn matmul_oversized_inner_dim_rejected() {
        // Input x with inner dim 32768 > 16384 limit.
        let mut func = Function::new("main");
        func.inputs.push((
            "input".into(),
            TensorType::new(ScalarType::Float32, vec![1, 128, 32768]),
        ));
        func.body = Block {
            operations: vec![
                Operation::new("matmul", "mm_0")
                    .with_input("x", Value::Reference("input".into()))
                    .with_input("y", Value::Reference("other".into()))
                    .with_output("out"),
            ],
            outputs: vec!["out".into()],
        };

        let mut program = Program::new("1.0.0");
        program.add_function(func);

        let report = validate_ane_compatibility(&program);

        assert_eq!(report.fallback_ops.len(), 1);
        assert!(
            report.fallback_ops[0]
                .reason
                .as_ref()
                .unwrap()
                .contains("inner dimension"),
            "expected inner dim rejection, got: {:?}",
            report.fallback_ops[0].reason
        );
    }

    #[test]
    fn matmul_valid_inner_dim_accepted() {
        let mut func = Function::new("main");
        func.inputs.push((
            "input".into(),
            TensorType::new(ScalarType::Float32, vec![1, 128, 512]),
        ));
        func.body = Block {
            operations: vec![
                Operation::new("matmul", "mm_0")
                    .with_input("x", Value::Reference("input".into()))
                    .with_output("out"),
            ],
            outputs: vec!["out".into()],
        };

        let mut program = Program::new("1.0.0");
        program.add_function(func);

        let report = validate_ane_compatibility(&program);

        assert!(
            report.ane_compatible.iter().any(|r| r.op_type == "matmul"),
            "expected matmul to be compatible"
        );
    }

    // -----------------------------------------------------------------------
    // FLOPs estimation tests
    // -----------------------------------------------------------------------

    #[test]
    fn flops_estimated_for_conv() {
        // Weight: [64, 3, 3, 3], input spatial: 224×224.
        // FLOPs = 2 * 224 * 224 * 3 * 3 * 3 * 64 = 173,408,256.
        let weight = Operation::new("const", "weight")
            .with_output("w")
            .with_input(
                "val",
                Value::Tensor {
                    data: vec![0; 64 * 3 * 3 * 3 * 4],
                    shape: vec![64, 3, 3, 3],
                    dtype: ScalarType::Float32,
                },
            );

        let conv = Operation::new("conv", "conv_0")
            .with_input("x", Value::Reference("input".into()))
            .with_input("weight", Value::Reference("w".into()))
            .with_output("out");

        let report = validate_ane_compatibility(&make_program(vec![weight, conv]));

        let conv_report = report
            .ane_compatible
            .iter()
            .find(|r| r.op_type == "conv")
            .expect("conv should be compatible");
        assert_eq!(conv_report.estimated_flops, 2 * 224 * 224 * 3 * 3 * 3 * 64);
    }

    #[test]
    fn flops_zero_for_reshape() {
        let ops = vec![
            Operation::new("reshape", "reshape_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("out"),
        ];
        let report = validate_ane_compatibility(&make_program(ops));

        assert_eq!(report.ane_compatible[0].estimated_flops, 0);
    }

    #[test]
    fn compute_split_reflects_flops() {
        // One compatible conv with FLOPs, one unsupported op with no FLOPs.
        let weight = Operation::new("const", "weight")
            .with_output("w")
            .with_input(
                "val",
                Value::Tensor {
                    data: vec![0; 64 * 3 * 3 * 3 * 4],
                    shape: vec![64, 3, 3, 3],
                    dtype: ScalarType::Float32,
                },
            );

        let conv = Operation::new("conv", "conv_0")
            .with_input("x", Value::Reference("input".into()))
            .with_input("weight", Value::Reference("w".into()))
            .with_output("c_out");

        let unsupported = Operation::new("erf", "erf_0").with_output("out");

        let report = validate_ane_compatibility(&make_program(vec![weight, conv, unsupported]));

        assert!(report.total_estimated_flops > 0);
        assert!(report.ane_estimated_flops > 0);
        // All FLOPs from the conv (compatible) — ANE compute should be 100%.
        assert!((report.ane_compute_pct - 100.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Performance annotation tests
    // -----------------------------------------------------------------------

    #[test]
    fn large_gather_produces_annotation() {
        let mut func = Function::new("main");
        func.inputs.push((
            "big_tensor".into(),
            TensorType::new(ScalarType::Float32, vec![1, 256, 512]),
        ));
        func.body = Block {
            operations: vec![
                Operation::new("gather", "gather_0")
                    .with_input("x", Value::Reference("big_tensor".into()))
                    .with_output("out"),
            ],
            outputs: vec!["out".into()],
        };

        let mut program = Program::new("1.0.0");
        program.add_function(func);

        let report = validate_ane_compatibility(&program);

        // gather is not ANE-supported (confirmed by compiler probe), so it
        // should appear in fallback_ops.
        assert!(
            report.fallback_ops.iter().any(|r| r.op_type == "gather"),
            "gather should be in fallback ops"
        );
    }

    #[test]
    fn op_report_has_ane_eligible_field() {
        let ops = vec![
            Operation::new("relu", "relu_0").with_output("r0"),
            Operation::new("scatter", "scatter_0").with_output("out"),
        ];
        let report = validate_ane_compatibility(&make_program(ops));

        assert!(report.ane_compatible[0].ane_eligible);
        assert!(!report.fallback_ops[0].ane_eligible);
    }

    // -----------------------------------------------------------------------
    // JSON output tests
    // -----------------------------------------------------------------------

    #[test]
    fn json_report_is_valid() {
        let ops = vec![
            Operation::new("relu", "relu_0").with_output("r0"),
            Operation::new("erf", "erf_0").with_output("out"),
        ];
        let report = validate_ane_compatibility(&make_program(ops));

        let json = validation_report_to_json(&report);
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("JSON should be valid");

        assert!(parsed.get("ane_compatible").is_some());
        assert!(parsed.get("fallback_ops").is_some());
        assert!(parsed.get("compatibility_pct").is_some());
        assert!(parsed.get("ane_compute_pct").is_some());
        assert!(parsed.get("total_estimated_flops").is_some());
        assert!(parsed.get("ane_estimated_flops").is_some());

        // Check per-op fields.
        let first_op = &parsed["ane_compatible"][0];
        assert!(first_op.get("performance_annotations").is_some());
        assert!(first_op.get("estimated_flops").is_some());
        assert!(first_op.get("ane_eligible").is_some());
    }
}
