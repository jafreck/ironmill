//! Convert ONNX [`NodeProto`] operations into MIL IR [`Operation`]s.
//!
//! This module provides a pure data/lookup mapping from ONNX operator types
//! to their equivalent MIL IR operations. Each converter function takes an
//! ONNX [`NodeProto`] and produces one or more MIL [`Operation`]s with the
//! appropriate named inputs, outputs, and attributes.
//!
//! Inputs are mapped from ONNX positional arguments to MIL named arguments
//! using [`Value::Reference`] — the graph builder resolves these references
//! to concrete values.

use std::collections::HashMap;

use crate::error::{MilError, Result};
use crate::ir::{Operation, ScalarType, Value};
use crate::proto::onnx::{NodeProto, TensorProto};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Convert an ONNX [`NodeProto`] into one or more MIL IR [`Operation`]s.
///
/// Returns a `Vec` because some ONNX ops expand to multiple MIL ops.
///
/// # Errors
///
/// Returns [`MilError::UnsupportedOp`] if the ONNX op type is not recognized.
pub fn convert_node(node: &NodeProto) -> Result<Vec<Operation>> {
    match node.op_type.as_str() {
        // P0 — Essential ops
        "Conv" => convert_conv(node),
        "MatMul" => convert_matmul(node),
        "Gemm" => convert_gemm(node),
        "Relu" => convert_unary(node, "relu"),
        "Add" => convert_binary(node, "add"),
        "Mul" => convert_binary(node, "mul"),
        "Reshape" => convert_reshape(node),
        "Transpose" => convert_transpose(node),
        "Softmax" => convert_softmax(node),
        "BatchNormalization" => convert_batch_norm(node),
        "MaxPool" => convert_pool(node, "max_pool"),
        "AveragePool" => convert_pool(node, "avg_pool"),
        "GlobalAveragePool" => convert_global_pool(node, "avg_pool"),
        "Concat" => convert_concat(node),
        "Flatten" => convert_flatten(node),

        // P1 — Important ops
        "Sigmoid" => convert_unary(node, "sigmoid"),
        "Tanh" => convert_unary(node, "tanh"),
        "Clip" => convert_clip(node),
        "Gather" => convert_gather(node),
        "Unsqueeze" => convert_unsqueeze(node),
        "Squeeze" => convert_squeeze(node),
        "Slice" => convert_slice(node),
        "Pad" => convert_pad(node),
        "ReduceMean" => convert_reduce_mean(node),
        "LayerNormalization" => convert_layer_norm(node),
        "Cast" => convert_cast(node),
        "Constant" => convert_constant(node),

        // P2 — Additional ops
        "Shape" => convert_unary(node, "shape"),
        "Split" => convert_split(node),
        "Where" => convert_where(node),
        "Pow" => convert_binary(node, "pow"),
        "Sqrt" => convert_unary(node, "sqrt"),
        "Div" => convert_binary(node, "real_div"),
        "Sub" => convert_binary(node, "sub"),
        "Erf" => convert_unary(node, "erf"),
        "ConvTranspose" => convert_conv_transpose(node),
        "Resize" => convert_resize(node),

        // Dropout is an identity in inference mode.
        "Dropout" => convert_identity(node),

        other => Err(MilError::UnsupportedOp(other.to_string())),
    }
}

// ---------------------------------------------------------------------------
// Attribute extraction helpers
// ---------------------------------------------------------------------------

/// Extract an integer attribute from an ONNX node by name.
pub fn get_int_attr(node: &NodeProto, name: &str) -> Option<i64> {
    node.attribute.iter().find(|a| a.name == name).map(|a| a.i)
}

/// Extract a float attribute from an ONNX node by name.
pub fn get_float_attr(node: &NodeProto, name: &str) -> Option<f32> {
    node.attribute.iter().find(|a| a.name == name).map(|a| a.f)
}

/// Extract an integer list attribute from an ONNX node by name.
pub fn get_int_list_attr(node: &NodeProto, name: &str) -> Option<Vec<i64>> {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.ints.clone())
}

/// Extract a float list attribute from an ONNX node by name.
pub fn get_float_list_attr(node: &NodeProto, name: &str) -> Option<Vec<f32>> {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.floats.clone())
}

/// Extract a string attribute from an ONNX node by name.
pub fn get_string_attr(node: &NodeProto, name: &str) -> Option<String> {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| String::from_utf8_lossy(&a.s).into_owned())
}

/// Extract a tensor attribute from an ONNX node by name.
pub fn get_tensor_attr<'a>(node: &'a NodeProto, name: &str) -> Option<&'a TensorProto> {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .and_then(|a| a.t.as_ref())
}

/// Map ONNX positional inputs to MIL named inputs.
///
/// ONNX uses positional arguments while MIL uses named arguments. This zips
/// the node's input list with the given parameter names, creating
/// [`Value::Reference`] entries for each non-empty input.
pub fn positional_to_named(node: &NodeProto, names: &[&str]) -> HashMap<String, Value> {
    let mut map = HashMap::new();
    for (i, name) in names.iter().enumerate() {
        if let Some(input) = node.input.get(i) {
            if !input.is_empty() {
                map.insert(name.to_string(), Value::Reference(input.clone()));
            }
        }
    }
    map
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Derive a MIL operation name from the ONNX node.
///
/// Prefers the node's `name` field; falls back to the first output name.
fn op_name(node: &NodeProto) -> String {
    if !node.name.is_empty() {
        node.name.clone()
    } else {
        node.output.first().cloned().unwrap_or_default()
    }
}

/// Attach all output names from the ONNX node to a MIL operation.
fn with_outputs(mut op: Operation, node: &NodeProto) -> Operation {
    for out in &node.output {
        op = op.with_output(out.clone());
    }
    op
}

/// Create a 1-D int32 tensor [`Value`] from a slice of i64 values.
///
/// CoreML MIL ops expect many parameters (perm, strides, pad, axes, etc.)
/// as 1-D int32 tensors rather than lists.
fn int_tensor_value(values: &[i64]) -> Value {
    let data: Vec<u8> = values
        .iter()
        .flat_map(|&v| (v as i32).to_le_bytes())
        .collect();
    Value::Tensor {
        data,
        shape: vec![values.len()],
        dtype: ScalarType::Int32,
    }
}

/// Create a [`Value::List`] of [`Value::Float`] from a slice.
fn float_list_value(values: &[f32]) -> Value {
    Value::List(values.iter().map(|&v| Value::Float(v as f64)).collect())
}

/// Check whether ONNX padding represents a causal (left-only) convolution.
///
/// ONNX `pads` are laid out as `[begin_d0, begin_d1, …, end_d0, end_d1, …]`.
/// For a causal 1-D conv the pattern is `[0, K-1, 0, 0]` — only the
/// second-to-last spatial dimension gets left padding equal to
/// `(kernel_size - 1) * dilation` and zero right padding.  For 2-D the
/// equivalent is `[0, K_w-1, 0, 0]` (height already zero) or the more
/// general form where each spatial dim has left = (ks-1)*d, right = 0.
fn is_causal_padding(pads: &[i64], kernel_shape: &[i64], dilations: &[i64]) -> bool {
    let ndim = kernel_shape.len();
    // pads must have 2 * ndim entries: [begin_d0 .. begin_dN, end_d0 .. end_dN].
    if pads.len() != 2 * ndim {
        return false;
    }
    // At least one spatial dimension must have non-zero causal padding.
    let mut has_causal_dim = false;
    for i in 0..ndim {
        let dilation = if i < dilations.len() { dilations[i] } else { 1 };
        let expected_left = (kernel_shape[i] - 1) * dilation;
        let left = pads[i];
        let right = pads[ndim + i];
        if left == expected_left && right == 0 && expected_left > 0 {
            has_causal_dim = true;
        } else if left != 0 || right != 0 {
            // Non-causal non-zero padding in this dimension.
            return false;
        }
    }
    has_causal_dim
}

/// Map an ONNX `TensorProto::DataType` integer to a MIL scalar-type name.
fn onnx_dtype_to_mil(dtype: i32) -> Result<&'static str> {
    match dtype {
        1 => Ok("float32"),
        2 => Ok("uint8"),
        3 => Ok("int8"),
        4 => Ok("uint16"),
        5 => Ok("int16"),
        6 => Ok("int32"),
        7 => Ok("int64"),
        9 => Ok("bool"),
        10 => Ok("float16"),
        11 => Ok("float64"),
        12 => Ok("uint32"),
        13 => Ok("uint64"),
        _ => Err(MilError::UnsupportedOp(format!(
            "unsupported ONNX data type: {dtype}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// P0 — Essential ops (~80% of models)
// ---------------------------------------------------------------------------

/// Convert a generic unary op (relu, sigmoid, tanh, sqrt, erf, shape).
fn convert_unary(node: &NodeProto, mil_op: &str) -> Result<Vec<Operation>> {
    let inputs = positional_to_named(node, &["x"]);
    let mut op = Operation::new(mil_op, op_name(node));
    op.inputs = inputs;
    Ok(vec![with_outputs(op, node)])
}

/// Convert a node to an identity op (pass-through).
///
/// Used for ops like Dropout that are identity in inference mode.
/// Only the first output is wired; extra outputs (e.g. mask) are ignored.
fn convert_identity(node: &NodeProto) -> Result<Vec<Operation>> {
    let inputs = positional_to_named(node, &["x"]);
    let mut op = Operation::new("identity", op_name(node));
    op.inputs = inputs;
    // Only take the first output (the pass-through data).
    if let Some(out) = node.output.first() {
        op = op.with_output(out.clone());
    }
    Ok(vec![op])
}

/// Convert a generic binary op (add, mul, sub, div, pow).
fn convert_binary(node: &NodeProto, mil_op: &str) -> Result<Vec<Operation>> {
    let inputs = positional_to_named(node, &["x", "y"]);
    let mut op = Operation::new(mil_op, op_name(node));
    op.inputs = inputs;
    Ok(vec![with_outputs(op, node)])
}

/// Conv: map kernel_shape, strides, pads, dilations, group attributes.
fn convert_conv(node: &NodeProto) -> Result<Vec<Operation>> {
    let inputs = positional_to_named(node, &["x", "weight", "bias"]);
    let mut op = Operation::new("conv", op_name(node));
    op.inputs = inputs;

    let strides = get_int_list_attr(node, "strides").unwrap_or_else(|| vec![1, 1]);
    op = op.with_attr("strides", int_tensor_value(&strides));

    let dilations = get_int_list_attr(node, "dilations").unwrap_or_else(|| vec![1, 1]);
    op = op.with_attr("dilations", int_tensor_value(&dilations));

    let group = get_int_attr(node, "group").unwrap_or(1);
    op = op.with_attr("groups", Value::Int(group));

    // MIL requires pad_type; use "custom" when explicit pads are provided.
    let pads = get_int_list_attr(node, "pads");
    let pad_type = if pads.is_some() {
        "custom"
    } else {
        let auto_pad = node
            .attribute
            .iter()
            .find(|a| a.name == "auto_pad")
            .and_then(|a| {
                if a.s.is_empty() {
                    None
                } else {
                    Some(String::from_utf8_lossy(&a.s).to_string())
                }
            });
        match auto_pad.as_deref() {
            Some("SAME_UPPER") | Some("SAME_LOWER") => "same",
            _ => "valid",
        }
    };
    op = op.with_attr("pad_type", Value::String(pad_type.to_string()));

    // ONNX pads are [top, left, bottom, right]; MIL expects the same order.
    let pad = pads.unwrap_or_else(|| vec![0, 0, 0, 0]);
    op = op.with_attr("pad", int_tensor_value(&pad));

    // Detect causal (left-only) padding used in streaming TTS/audio models.
    // Causal convolution: for each spatial dimension the left pad equals
    // kernel_size - 1 and the right pad is 0.
    let kernel_shape = get_int_list_attr(node, "kernel_shape");
    if pad_type == "custom" {
        if let Some(ref ks) = kernel_shape {
            let is_causal = is_causal_padding(&pad, ks, &dilations);
            op = op.with_attr("causal", Value::Bool(is_causal));
        }
    }

    Ok(vec![with_outputs(op, node)])
}

/// MatMul: direct mapping with default transpose flags.
fn convert_matmul(node: &NodeProto) -> Result<Vec<Operation>> {
    let inputs = positional_to_named(node, &["x", "y"]);
    let mut op = Operation::new("matmul", op_name(node));
    op.inputs = inputs;
    op = op.with_attr("transpose_x", Value::Bool(false));
    op = op.with_attr("transpose_y", Value::Bool(false));
    Ok(vec![with_outputs(op, node)])
}

/// Gemm: map alpha, beta, transA, transB to linear.
fn convert_gemm(node: &NodeProto) -> Result<Vec<Operation>> {
    let inputs = positional_to_named(node, &["x", "weight", "bias"]);
    let mut op = Operation::new("linear", op_name(node));
    op.inputs = inputs;
    // ONNX Gemm's alpha, beta, transA, transB are not supported by CoreML's
    // linear op. Most ONNX models use default values (alpha=1, beta=1,
    // transA=0, transB=1) which matches CoreML's linear semantics.
    Ok(vec![with_outputs(op, node)])
}

/// Reshape: map shape input.
fn convert_reshape(node: &NodeProto) -> Result<Vec<Operation>> {
    let inputs = positional_to_named(node, &["x", "shape"]);
    let mut op = Operation::new("reshape", op_name(node));
    op.inputs = inputs;

    if let Some(allowzero) = get_int_attr(node, "allowzero") {
        op = op.with_attr("allowzero", Value::Bool(allowzero != 0));
    }

    Ok(vec![with_outputs(op, node)])
}

/// Transpose: map perm attribute.
fn convert_transpose(node: &NodeProto) -> Result<Vec<Operation>> {
    let inputs = positional_to_named(node, &["x"]);
    let mut op = Operation::new("transpose", op_name(node));
    op.inputs = inputs;

    if let Some(perm) = get_int_list_attr(node, "perm") {
        op = op.with_attr("perm", int_tensor_value(&perm));
    }

    Ok(vec![with_outputs(op, node)])
}

/// Softmax: map axis attribute.
fn convert_softmax(node: &NodeProto) -> Result<Vec<Operation>> {
    let inputs = positional_to_named(node, &["x"]);
    let mut op = Operation::new("softmax", op_name(node));
    op.inputs = inputs;

    if let Some(axis) = get_int_attr(node, "axis") {
        op = op.with_attr("axis", Value::Int(axis));
    }

    Ok(vec![with_outputs(op, node)])
}

/// BatchNormalization: map epsilon, momentum; ONNX inputs → MIL named inputs.
fn convert_batch_norm(node: &NodeProto) -> Result<Vec<Operation>> {
    let inputs = positional_to_named(node, &["x", "gamma", "beta", "mean", "variance"]);
    let mut op = Operation::new("batch_norm", op_name(node));
    op.inputs = inputs;

    if let Some(epsilon) = get_float_attr(node, "epsilon") {
        op = op.with_attr("epsilon", Value::Float(epsilon as f64));
    }
    // momentum is an ONNX training-only parameter; CoreML MIL does not
    // recognise it, so we intentionally skip it.

    Ok(vec![with_outputs(op, node)])
}

/// MaxPool / AveragePool: map kernel_shape, strides, pads.
fn convert_pool(node: &NodeProto, mil_op: &str) -> Result<Vec<Operation>> {
    let inputs = positional_to_named(node, &["x"]);
    let mut op = Operation::new(mil_op, op_name(node));
    op.inputs = inputs;

    if let Some(kernel_shape) = get_int_list_attr(node, "kernel_shape") {
        op = op.with_attr("kernel_sizes", int_tensor_value(&kernel_shape));
    }
    if let Some(strides) = get_int_list_attr(node, "strides") {
        op = op.with_attr("strides", int_tensor_value(&strides));
    }
    if let Some(pads) = get_int_list_attr(node, "pads") {
        op = op.with_attr("pad", int_tensor_value(&pads));
    } else {
        // CoreML requires an explicit pad even for valid padding.
        op = op.with_attr("pad", int_tensor_value(&[0, 0, 0, 0]));
    }
    if let Some(ceil_mode) = get_int_attr(node, "ceil_mode") {
        op = op.with_attr("ceil_mode", Value::Bool(ceil_mode != 0));
    } else {
        op = op.with_attr("ceil_mode", Value::Bool(false));
    }

    // MIL requires pad_type for pool ops.
    let pad_type = if get_int_list_attr(node, "pads").is_some() {
        "custom"
    } else {
        "valid"
    };
    op = op.with_attr("pad_type", Value::String(pad_type.to_string()));

    // avg_pool requires exclude_padding_from_average.
    if mil_op == "avg_pool" {
        let count_include_pad = get_int_attr(node, "count_include_pad").unwrap_or(0);
        op = op.with_attr(
            "exclude_padding_from_average",
            Value::Bool(count_include_pad == 0),
        );
    }

    Ok(vec![with_outputs(op, node)])
}

/// GlobalAveragePool: maps to avg_pool with a `global` flag.
fn convert_global_pool(node: &NodeProto, mil_op: &str) -> Result<Vec<Operation>> {
    let inputs = positional_to_named(node, &["x"]);
    let mut op = Operation::new(mil_op, op_name(node));
    op.inputs = inputs;
    // Global pooling: kernel covers the entire spatial extent. We set
    // kernel_sizes in propagate_output_types once the input shape is known.
    // Mark as global via an internal attribute (skipped in proto serialization).
    op = op.with_attr("global_pool", Value::Bool(true));
    Ok(vec![with_outputs(op, node)])
}

/// Concat: all ONNX inputs gathered into a single `values` list.
fn convert_concat(node: &NodeProto) -> Result<Vec<Operation>> {
    let values = Value::List(
        node.input
            .iter()
            .filter(|s| !s.is_empty())
            .map(|s| Value::Reference(s.clone()))
            .collect(),
    );

    let mut op = Operation::new("concat", op_name(node));
    op.inputs.insert("values".to_string(), values);

    if let Some(axis) = get_int_attr(node, "axis") {
        op = op.with_attr("axis", Value::Int(axis));
    }
    op = op.with_attr("interleave", Value::Bool(false));

    Ok(vec![with_outputs(op, node)])
}

/// Flatten → reshape with `flatten_axis` attribute. The graph builder
/// computes the concrete output shape.
fn convert_flatten(node: &NodeProto) -> Result<Vec<Operation>> {
    let inputs = positional_to_named(node, &["x"]);
    let mut op = Operation::new("reshape", op_name(node));
    op.inputs = inputs;

    let axis = get_int_attr(node, "axis").unwrap_or(1);
    op = op.with_attr("flatten_axis", Value::Int(axis));

    Ok(vec![with_outputs(op, node)])
}

// ---------------------------------------------------------------------------
// P1 — Important ops
// ---------------------------------------------------------------------------

/// Clip: clamp values to [min, max]. Handles both attribute (opset < 11)
/// and input (opset ≥ 11) forms.
fn convert_clip(node: &NodeProto) -> Result<Vec<Operation>> {
    let mut op = Operation::new("clip", op_name(node));

    // Input form (opset ≥ 11): [input, min, max]
    if let Some(x) = node.input.first().filter(|s| !s.is_empty()) {
        op.inputs
            .insert("x".to_string(), Value::Reference(x.clone()));
    }
    // CoreML MIL uses "alpha" for min and "beta" for max.
    if let Some(min_val) = node.input.get(1).filter(|s| !s.is_empty()) {
        op.inputs
            .insert("alpha".to_string(), Value::Reference(min_val.clone()));
    }
    if let Some(max_val) = node.input.get(2).filter(|s| !s.is_empty()) {
        op.inputs
            .insert("beta".to_string(), Value::Reference(max_val.clone()));
    }

    // Attribute form (opset < 11)
    if let Some(min) = get_float_attr(node, "min") {
        op = op.with_attr("alpha", Value::Float(min as f64));
    }
    if let Some(max) = get_float_attr(node, "max") {
        op = op.with_attr("beta", Value::Float(max as f64));
    }

    Ok(vec![with_outputs(op, node)])
}

/// Gather: map axis attribute.
fn convert_gather(node: &NodeProto) -> Result<Vec<Operation>> {
    let inputs = positional_to_named(node, &["x", "indices"]);
    let mut op = Operation::new("gather", op_name(node));
    op.inputs = inputs;

    if let Some(axis) = get_int_attr(node, "axis") {
        op = op.with_attr("axis", Value::Int(axis));
    }

    Ok(vec![with_outputs(op, node)])
}

/// Unsqueeze: handles both attribute (opset < 13) and input (opset ≥ 13) forms.
fn convert_unsqueeze(node: &NodeProto) -> Result<Vec<Operation>> {
    let mut op = Operation::new("expand_dims", op_name(node));

    if let Some(x) = node.input.first().filter(|s| !s.is_empty()) {
        op.inputs
            .insert("x".to_string(), Value::Reference(x.clone()));
    }

    // Input form (opset ≥ 13)
    if let Some(axes_input) = node.input.get(1).filter(|s| !s.is_empty()) {
        op.inputs
            .insert("axes".to_string(), Value::Reference(axes_input.clone()));
    }
    // Attribute form (opset < 13)
    if let Some(axes) = get_int_list_attr(node, "axes") {
        op = op.with_attr("axes", int_tensor_value(&axes));
    }

    Ok(vec![with_outputs(op, node)])
}

/// Squeeze: handles both attribute (opset < 13) and input (opset ≥ 13) forms.
fn convert_squeeze(node: &NodeProto) -> Result<Vec<Operation>> {
    let mut op = Operation::new("squeeze", op_name(node));

    if let Some(x) = node.input.first().filter(|s| !s.is_empty()) {
        op.inputs
            .insert("x".to_string(), Value::Reference(x.clone()));
    }

    if let Some(axes_input) = node.input.get(1).filter(|s| !s.is_empty()) {
        op.inputs
            .insert("axes".to_string(), Value::Reference(axes_input.clone()));
    }
    if let Some(axes) = get_int_list_attr(node, "axes") {
        op = op.with_attr("axes", int_tensor_value(&axes));
    }

    Ok(vec![with_outputs(op, node)])
}

/// Slice: ONNX inputs are [data, starts, ends, axes, steps].
fn convert_slice(node: &NodeProto) -> Result<Vec<Operation>> {
    let inputs = positional_to_named(node, &["x", "begin", "end", "axes", "strides"]);
    let mut op = Operation::new("slice_by_index", op_name(node));
    op.inputs = inputs;
    Ok(vec![with_outputs(op, node)])
}

/// Pad: handles both attribute (opset < 11) and input (opset ≥ 11) forms.
fn convert_pad(node: &NodeProto) -> Result<Vec<Operation>> {
    let mut op = Operation::new("pad", op_name(node));

    if let Some(x) = node.input.first().filter(|s| !s.is_empty()) {
        op.inputs
            .insert("x".to_string(), Value::Reference(x.clone()));
    }

    // Input form (opset ≥ 11): [data, pads, constant_value, axes]
    if let Some(pads) = node.input.get(1).filter(|s| !s.is_empty()) {
        op.inputs
            .insert("pad".to_string(), Value::Reference(pads.clone()));
    }
    if let Some(cv) = node.input.get(2).filter(|s| !s.is_empty()) {
        op.inputs
            .insert("constant_val".to_string(), Value::Reference(cv.clone()));
    }

    // Attribute form (opset < 11)
    if let Some(pads) = get_int_list_attr(node, "pads") {
        op = op.with_attr("pad", int_tensor_value(&pads));
    }
    if let Some(mode) = get_string_attr(node, "mode") {
        op = op.with_attr("mode", Value::String(mode));
    }
    if let Some(value) = get_float_attr(node, "value") {
        op = op.with_attr("constant_val", Value::Float(value as f64));
    }

    Ok(vec![with_outputs(op, node)])
}

/// ReduceMean: handles both attribute (opset ≤ 17) and input (opset 18+) axes.
fn convert_reduce_mean(node: &NodeProto) -> Result<Vec<Operation>> {
    let mut op = Operation::new("reduce_mean", op_name(node));

    if let Some(x) = node.input.first().filter(|s| !s.is_empty()) {
        op.inputs
            .insert("x".to_string(), Value::Reference(x.clone()));
    }

    // Attribute form
    if let Some(axes) = get_int_list_attr(node, "axes") {
        op = op.with_attr("axes", int_tensor_value(&axes));
    }
    // Input form (opset 18+)
    if let Some(axes_input) = node.input.get(1).filter(|s| !s.is_empty()) {
        op.inputs
            .insert("axes".to_string(), Value::Reference(axes_input.clone()));
    }

    if let Some(keepdims) = get_int_attr(node, "keepdims") {
        op = op.with_attr("keep_dims", Value::Bool(keepdims != 0));
    }

    Ok(vec![with_outputs(op, node)])
}

/// LayerNormalization: map epsilon; axis → axes list.
fn convert_layer_norm(node: &NodeProto) -> Result<Vec<Operation>> {
    let inputs = positional_to_named(node, &["x", "gamma", "beta"]);
    let mut op = Operation::new("layer_norm", op_name(node));
    op.inputs = inputs;

    if let Some(axis) = get_int_attr(node, "axis") {
        op = op.with_attr("axes", Value::List(vec![Value::Int(axis)]));
    }
    if let Some(epsilon) = get_float_attr(node, "epsilon") {
        op = op.with_attr("epsilon", Value::Float(epsilon as f64));
    }

    Ok(vec![with_outputs(op, node)])
}

/// Cast: map ONNX `to` data-type integer to a MIL scalar-type name.
fn convert_cast(node: &NodeProto) -> Result<Vec<Operation>> {
    let inputs = positional_to_named(node, &["x"]);
    let mut op = Operation::new("cast", op_name(node));
    op.inputs = inputs;

    if let Some(to) = get_int_attr(node, "to") {
        let dtype = onnx_dtype_to_mil(to as i32)?;
        op = op.with_attr("dtype", Value::String(dtype.to_string()));
    }

    Ok(vec![with_outputs(op, node)])
}

/// Constant: extract value from attributes. Tensor metadata is stored as
/// attributes; the graph builder extracts the actual tensor data.
fn convert_constant(node: &NodeProto) -> Result<Vec<Operation>> {
    let mut op = Operation::new("const", op_name(node));

    if let Some(tensor) = get_tensor_attr(node, "value") {
        let dtype = super::onnx_graph::onnx_dtype_to_scalar(tensor.data_type)?;
        let shape: Vec<usize> = tensor.dims.iter().map(|&d| d as usize).collect();
        let raw_bytes = super::onnx_graph::extract_tensor_raw_data(tensor, dtype);
        op = op.with_attr(
            "val",
            Value::Tensor {
                data: raw_bytes,
                shape,
                dtype,
            },
        );
    } else if let Some(f) = get_float_attr(node, "value_float") {
        op = op.with_attr("val", Value::Float(f as f64));
    } else if let Some(i) = get_int_attr(node, "value_int") {
        op = op.with_attr("val", Value::Int(i));
    } else if let Some(floats) = get_float_list_attr(node, "value_floats") {
        op = op.with_attr("val", float_list_value(&floats));
    } else if let Some(ints) = get_int_list_attr(node, "value_ints") {
        op = op.with_attr("val", int_tensor_value(&ints));
    }

    Ok(vec![with_outputs(op, node)])
}

// ---------------------------------------------------------------------------
// P2 — Additional ops
// ---------------------------------------------------------------------------

/// Split: handles both attribute and input forms for split sizes.
fn convert_split(node: &NodeProto) -> Result<Vec<Operation>> {
    let mut op = Operation::new("split", op_name(node));

    if let Some(x) = node.input.first().filter(|s| !s.is_empty()) {
        op.inputs
            .insert("x".to_string(), Value::Reference(x.clone()));
    }

    // Input form (opset ≥ 13): split sizes as second input
    if let Some(split_input) = node.input.get(1).filter(|s| !s.is_empty()) {
        op.inputs.insert(
            "split_sizes".to_string(),
            Value::Reference(split_input.clone()),
        );
    }
    // Attribute form
    if let Some(split) = get_int_list_attr(node, "split") {
        op = op.with_attr("split_sizes", int_tensor_value(&split));
    }
    if let Some(axis) = get_int_attr(node, "axis") {
        op = op.with_attr("axis", Value::Int(axis));
    }
    if let Some(num_outputs) = get_int_attr(node, "num_outputs") {
        op = op.with_attr("num_splits", Value::Int(num_outputs));
    }

    Ok(vec![with_outputs(op, node)])
}

/// Where → select: map condition, a, b inputs.
fn convert_where(node: &NodeProto) -> Result<Vec<Operation>> {
    let inputs = positional_to_named(node, &["cond", "a", "b"]);
    let mut op = Operation::new("select", op_name(node));
    op.inputs = inputs;
    Ok(vec![with_outputs(op, node)])
}

/// ConvTranspose: map strides, pads, dilations, group, output_shape.
fn convert_conv_transpose(node: &NodeProto) -> Result<Vec<Operation>> {
    let inputs = positional_to_named(node, &["x", "weight", "bias"]);
    let mut op = Operation::new("conv_transpose", op_name(node));
    op.inputs = inputs;

    if let Some(strides) = get_int_list_attr(node, "strides") {
        op = op.with_attr("strides", int_tensor_value(&strides));
    }
    if let Some(pads) = get_int_list_attr(node, "pads") {
        op = op.with_attr("pad", int_tensor_value(&pads));
    }
    if let Some(dilations) = get_int_list_attr(node, "dilations") {
        op = op.with_attr("dilations", int_tensor_value(&dilations));
    }
    if let Some(group) = get_int_attr(node, "group") {
        op = op.with_attr("groups", Value::Int(group));
    }
    if let Some(output_shape) = get_int_list_attr(node, "output_shape") {
        op = op.with_attr("output_shape", int_tensor_value(&output_shape));
    }
    if let Some(output_padding) = get_int_list_attr(node, "output_padding") {
        op = op.with_attr("output_padding", int_tensor_value(&output_padding));
    }

    Ok(vec![with_outputs(op, node)])
}

/// Resize → upsample_bilinear: map scale/size inputs and coordinate mode.
fn convert_resize(node: &NodeProto) -> Result<Vec<Operation>> {
    let mut op = Operation::new("upsample_bilinear", op_name(node));

    // ONNX Resize inputs: [X, roi, scales, sizes]
    if let Some(x) = node.input.first().filter(|s| !s.is_empty()) {
        op.inputs
            .insert("x".to_string(), Value::Reference(x.clone()));
    }
    if let Some(roi) = node.input.get(1).filter(|s| !s.is_empty()) {
        op.inputs
            .insert("roi".to_string(), Value::Reference(roi.clone()));
    }
    if let Some(scales) = node.input.get(2).filter(|s| !s.is_empty()) {
        op.inputs
            .insert("scales".to_string(), Value::Reference(scales.clone()));
    }
    if let Some(sizes) = node.input.get(3).filter(|s| !s.is_empty()) {
        op.inputs
            .insert("sizes".to_string(), Value::Reference(sizes.clone()));
    }

    if let Some(mode) = get_string_attr(node, "mode") {
        op = op.with_attr("mode", Value::String(mode));
    }
    if let Some(coord) = get_string_attr(node, "coordinate_transformation_mode") {
        let align = coord == "align_corners";
        op = op.with_attr("align_corners", Value::Bool(align));
    }

    Ok(vec![with_outputs(op, node)])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::onnx::{AttributeProto, NodeProto, TensorProto};

    // -- Test helpers -------------------------------------------------------

    fn make_node(op_type: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    fn make_int_attr(name: &str, value: i64) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            i: value,
            ..Default::default()
        }
    }

    fn make_float_attr(name: &str, value: f32) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            f: value,
            ..Default::default()
        }
    }

    fn make_int_list_attr(name: &str, values: &[i64]) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            ints: values.to_vec(),
            ..Default::default()
        }
    }

    fn make_string_attr(name: &str, value: &str) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            s: value.as_bytes().to_vec(),
            ..Default::default()
        }
    }

    fn make_tensor_attr(name: &str, dims: &[i64], data_type: i32) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            t: Some(TensorProto {
                dims: dims.to_vec(),
                data_type,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    // -- Attribute extraction -----------------------------------------------

    #[test]
    fn test_get_int_attr() {
        let mut node = make_node("Test", &[], &[]);
        node.attribute.push(make_int_attr("group", 4));
        assert_eq!(get_int_attr(&node, "group"), Some(4));
        assert_eq!(get_int_attr(&node, "missing"), None);
    }

    #[test]
    fn test_get_float_attr() {
        let mut node = make_node("Test", &[], &[]);
        node.attribute.push(make_float_attr("epsilon", 1e-5));
        let val = get_float_attr(&node, "epsilon").unwrap();
        assert!((val - 1e-5).abs() < 1e-10);
        assert_eq!(get_float_attr(&node, "missing"), None);
    }

    #[test]
    fn test_get_int_list_attr() {
        let mut node = make_node("Test", &[], &[]);
        node.attribute.push(make_int_list_attr("strides", &[2, 2]));
        assert_eq!(get_int_list_attr(&node, "strides"), Some(vec![2, 2]));
        assert_eq!(get_int_list_attr(&node, "missing"), None);
    }

    #[test]
    fn test_get_string_attr() {
        let mut node = make_node("Test", &[], &[]);
        node.attribute.push(make_string_attr("mode", "constant"));
        assert_eq!(get_string_attr(&node, "mode"), Some("constant".to_string()));
        assert_eq!(get_string_attr(&node, "missing"), None);
    }

    #[test]
    fn test_get_tensor_attr() {
        let mut node = make_node("Test", &[], &[]);
        node.attribute.push(make_tensor_attr("value", &[2, 3], 1));
        let t = get_tensor_attr(&node, "value").unwrap();
        assert_eq!(t.dims, vec![2, 3]);
        assert_eq!(t.data_type, 1);
        assert!(get_tensor_attr(&node, "missing").is_none());
    }

    // -- positional_to_named ------------------------------------------------

    #[test]
    fn test_positional_to_named() {
        let node = make_node("Conv", &["input", "weight", "bias"], &["out"]);
        let map = positional_to_named(&node, &["x", "weight", "bias"]);
        assert_eq!(map.len(), 3);
        assert!(matches!(map.get("x"), Some(Value::Reference(s)) if s == "input"));
        assert!(matches!(map.get("weight"), Some(Value::Reference(s)) if s == "weight"));
        assert!(matches!(map.get("bias"), Some(Value::Reference(s)) if s == "bias"));
    }

    #[test]
    fn test_positional_to_named_skips_empty() {
        let node = make_node("Conv", &["input", "weight", ""], &["out"]);
        let map = positional_to_named(&node, &["x", "weight", "bias"]);
        assert_eq!(map.len(), 2);
        assert!(!map.contains_key("bias"));
    }

    #[test]
    fn test_positional_to_named_fewer_inputs() {
        let node = make_node("Conv", &["input", "weight"], &["out"]);
        let map = positional_to_named(&node, &["x", "weight", "bias"]);
        assert_eq!(map.len(), 2);
        assert!(!map.contains_key("bias"));
    }

    // -- Dispatch -----------------------------------------------------------

    #[test]
    fn test_dispatch_p0_ops() {
        let ops = [
            ("Conv", "conv"),
            ("MatMul", "matmul"),
            ("Gemm", "linear"),
            ("Relu", "relu"),
            ("Add", "add"),
            ("Mul", "mul"),
            ("Reshape", "reshape"),
            ("Transpose", "transpose"),
            ("Softmax", "softmax"),
            ("BatchNormalization", "batch_norm"),
            ("MaxPool", "max_pool"),
            ("AveragePool", "avg_pool"),
            ("Concat", "concat"),
            ("Flatten", "reshape"),
        ];
        for (onnx, mil) in ops {
            let node = make_node(onnx, &["x"], &["out"]);
            let result = convert_node(&node).unwrap();
            assert_eq!(
                result[0].op_type, mil,
                "ONNX {onnx} should map to MIL {mil}"
            );
        }
    }

    #[test]
    fn test_dispatch_p1_ops() {
        let ops = [
            ("Sigmoid", "sigmoid"),
            ("Tanh", "tanh"),
            ("Clip", "clip"),
            ("Gather", "gather"),
            ("Unsqueeze", "expand_dims"),
            ("Squeeze", "squeeze"),
            ("Slice", "slice_by_index"),
            ("Pad", "pad"),
            ("ReduceMean", "reduce_mean"),
            ("LayerNormalization", "layer_norm"),
            ("Cast", "cast"),
            ("Constant", "const"),
        ];
        for (onnx, mil) in ops {
            let node = make_node(onnx, &["x"], &["out"]);
            let result = convert_node(&node).unwrap();
            assert_eq!(
                result[0].op_type, mil,
                "ONNX {onnx} should map to MIL {mil}"
            );
        }
    }

    #[test]
    fn test_dispatch_p2_ops() {
        let ops = [
            ("Shape", "shape"),
            ("Split", "split"),
            ("Where", "select"),
            ("Pow", "pow"),
            ("Sqrt", "sqrt"),
            ("Div", "real_div"),
            ("Sub", "sub"),
            ("Erf", "erf"),
            ("ConvTranspose", "conv_transpose"),
            ("Resize", "upsample_bilinear"),
        ];
        for (onnx, mil) in ops {
            let node = make_node(onnx, &["x"], &["out"]);
            let result = convert_node(&node).unwrap();
            assert_eq!(
                result[0].op_type, mil,
                "ONNX {onnx} should map to MIL {mil}"
            );
        }
    }

    #[test]
    fn test_unsupported_op() {
        let node = make_node("FakeOp", &["x"], &["out"]);
        let result = convert_node(&node);
        assert!(matches!(result, Err(MilError::UnsupportedOp(s)) if s == "FakeOp"));
    }

    // -- Individual P0 ops --------------------------------------------------

    #[test]
    fn test_conv_full() {
        let mut node = make_node("Conv", &["input", "W", "B"], &["Y"]);
        node.attribute.push(make_int_list_attr("strides", &[2, 2]));
        node.attribute
            .push(make_int_list_attr("pads", &[1, 1, 1, 1]));
        node.attribute
            .push(make_int_list_attr("dilations", &[1, 1]));
        node.attribute.push(make_int_attr("group", 1));
        node.attribute
            .push(make_int_list_attr("kernel_shape", &[3, 3]));

        let ops = convert_node(&node).unwrap();
        assert_eq!(ops.len(), 1);
        let op = &ops[0];
        assert_eq!(op.op_type, "conv");
        assert_eq!(op.outputs, vec!["Y"]);
        assert!(matches!(op.inputs.get("x"), Some(Value::Reference(s)) if s == "input"));
        assert!(matches!(op.inputs.get("weight"), Some(Value::Reference(s)) if s == "W"));
        assert!(matches!(op.inputs.get("bias"), Some(Value::Reference(s)) if s == "B"));
        assert!(op.attributes.contains_key("strides"));
        assert!(op.attributes.contains_key("pad"));
        assert!(op.attributes.contains_key("groups"));
        assert!(op.attributes.contains_key("pad_type"));
    }

    #[test]
    fn test_conv_no_bias() {
        let node = make_node("Conv", &["input", "W"], &["Y"]);
        let ops = convert_node(&node).unwrap();
        let op = &ops[0];
        assert!(op.inputs.contains_key("x"));
        assert!(op.inputs.contains_key("weight"));
        assert!(!op.inputs.contains_key("bias"));
        // groups and pad_type are always set
        assert!(op.attributes.contains_key("groups"));
        assert!(op.attributes.contains_key("pad_type"));
    }

    #[test]
    fn test_gemm() {
        let mut node = make_node("Gemm", &["A", "B", "C"], &["Y"]);
        node.attribute.push(make_float_attr("alpha", 1.0));
        node.attribute.push(make_float_attr("beta", 1.0));
        node.attribute.push(make_int_attr("transB", 1));

        let ops = convert_node(&node).unwrap();
        let op = &ops[0];
        assert_eq!(op.op_type, "linear");
        // alpha, beta, transA, transB are ONNX-only; not emitted to MIL.
        assert!(op.attributes.is_empty());
    }

    #[test]
    fn test_batch_norm() {
        let mut node = make_node(
            "BatchNormalization",
            &["X", "scale", "B", "mean", "var"],
            &["Y"],
        );
        node.attribute.push(make_float_attr("epsilon", 1e-5));

        let ops = convert_node(&node).unwrap();
        let op = &ops[0];
        assert_eq!(op.op_type, "batch_norm");
        assert_eq!(op.inputs.len(), 5);
        assert!(matches!(op.inputs.get("x"), Some(Value::Reference(s)) if s == "X"));
        assert!(matches!(op.inputs.get("variance"), Some(Value::Reference(s)) if s == "var"));
    }

    #[test]
    fn test_concat_variadic() {
        let mut node = make_node("Concat", &["a", "b", "c"], &["out"]);
        node.attribute.push(make_int_attr("axis", 1));

        let ops = convert_node(&node).unwrap();
        let op = &ops[0];
        assert_eq!(op.op_type, "concat");
        match op.inputs.get("values") {
            Some(Value::List(refs)) => assert_eq!(refs.len(), 3),
            other => panic!("expected List of 3 references, got {other:?}"),
        }
        assert!(matches!(op.attributes.get("axis"), Some(Value::Int(1))));
    }

    #[test]
    fn test_flatten_default_axis() {
        let node = make_node("Flatten", &["X"], &["out"]);
        let ops = convert_node(&node).unwrap();
        let op = &ops[0];
        assert_eq!(op.op_type, "reshape");
        assert!(matches!(
            op.attributes.get("flatten_axis"),
            Some(Value::Int(1))
        ));
    }

    #[test]
    fn test_flatten_custom_axis() {
        let mut node = make_node("Flatten", &["X"], &["out"]);
        node.attribute.push(make_int_attr("axis", 2));
        let ops = convert_node(&node).unwrap();
        let op = &ops[0];
        assert!(matches!(
            op.attributes.get("flatten_axis"),
            Some(Value::Int(2))
        ));
    }

    #[test]
    fn test_pool_with_attrs() {
        let mut node = make_node("MaxPool", &["X"], &["Y"]);
        node.attribute
            .push(make_int_list_attr("kernel_shape", &[2, 2]));
        node.attribute.push(make_int_list_attr("strides", &[2, 2]));
        node.attribute
            .push(make_int_list_attr("pads", &[0, 0, 0, 0]));

        let ops = convert_node(&node).unwrap();
        let op = &ops[0];
        assert_eq!(op.op_type, "max_pool");
        assert!(op.attributes.contains_key("kernel_sizes"));
        assert!(op.attributes.contains_key("strides"));
        assert!(op.attributes.contains_key("pad"));
    }

    #[test]
    fn test_transpose_perm() {
        let mut node = make_node("Transpose", &["X"], &["Y"]);
        node.attribute.push(make_int_list_attr("perm", &[0, 2, 1]));

        let ops = convert_node(&node).unwrap();
        let op = &ops[0];
        assert_eq!(op.op_type, "transpose");
        assert!(op.attributes.contains_key("perm"));
    }

    #[test]
    fn test_softmax_axis() {
        let mut node = make_node("Softmax", &["X"], &["Y"]);
        node.attribute.push(make_int_attr("axis", -1));

        let ops = convert_node(&node).unwrap();
        let op = &ops[0];
        assert_eq!(op.op_type, "softmax");
        assert!(matches!(op.attributes.get("axis"), Some(Value::Int(-1))));
    }

    // -- Individual P1 ops --------------------------------------------------

    #[test]
    fn test_cast() {
        let mut node = make_node("Cast", &["X"], &["Y"]);
        node.attribute.push(make_int_attr("to", 1)); // FLOAT
        let ops = convert_node(&node).unwrap();
        let op = &ops[0];
        assert_eq!(op.op_type, "cast");
        assert!(matches!(op.attributes.get("dtype"), Some(Value::String(s)) if s == "float32"));
    }

    #[test]
    fn test_cast_unsupported_dtype() {
        let mut node = make_node("Cast", &["X"], &["Y"]);
        node.attribute.push(make_int_attr("to", 99));
        let result = convert_node(&node);
        assert!(result.is_err());
    }

    #[test]
    fn test_constant_float() {
        let mut node = make_node("Constant", &[], &["val"]);
        node.attribute.push(make_float_attr("value_float", 3.14));
        let ops = convert_node(&node).unwrap();
        let op = &ops[0];
        assert_eq!(op.op_type, "const");
        match op.attributes.get("val") {
            Some(Value::Float(f)) => assert!((*f - 3.14_f64).abs() < 0.01),
            other => panic!("expected Float, got {other:?}"),
        }
    }

    #[test]
    fn test_constant_tensor() {
        let mut node = make_node("Constant", &[], &["val"]);
        node.attribute.push(make_tensor_attr("value", &[2, 3], 1));
        let ops = convert_node(&node).unwrap();
        let op = &ops[0];
        assert_eq!(op.op_type, "const");
        assert!(
            op.attributes.contains_key("val"),
            "tensor constants should store data in 'val' attribute"
        );
        if let Some(Value::Tensor { shape, dtype, .. }) = op.attributes.get("val") {
            assert_eq!(shape, &[2, 3]);
            assert_eq!(*dtype, crate::ir::ScalarType::Float32);
        } else {
            panic!(
                "expected Value::Tensor for val, got {:?}",
                op.attributes.get("val")
            );
        }
    }

    #[test]
    fn test_gather_axis() {
        let mut node = make_node("Gather", &["data", "indices"], &["out"]);
        node.attribute.push(make_int_attr("axis", 1));
        let ops = convert_node(&node).unwrap();
        let op = &ops[0];
        assert_eq!(op.op_type, "gather");
        assert!(matches!(op.attributes.get("axis"), Some(Value::Int(1))));
    }

    #[test]
    fn test_unsqueeze_attr_form() {
        let mut node = make_node("Unsqueeze", &["X"], &["Y"]);
        node.attribute.push(make_int_list_attr("axes", &[0, 2]));
        let ops = convert_node(&node).unwrap();
        let op = &ops[0];
        assert_eq!(op.op_type, "expand_dims");
        assert!(op.attributes.contains_key("axes"));
    }

    #[test]
    fn test_unsqueeze_input_form() {
        let node = make_node("Unsqueeze", &["X", "axes_tensor"], &["Y"]);
        let ops = convert_node(&node).unwrap();
        let op = &ops[0];
        assert_eq!(op.op_type, "expand_dims");
        assert!(matches!(
            op.inputs.get("axes"),
            Some(Value::Reference(s)) if s == "axes_tensor"
        ));
    }

    #[test]
    fn test_reduce_mean_attr_form() {
        let mut node = make_node("ReduceMean", &["X"], &["Y"]);
        node.attribute.push(make_int_list_attr("axes", &[2, 3]));
        node.attribute.push(make_int_attr("keepdims", 1));
        let ops = convert_node(&node).unwrap();
        let op = &ops[0];
        assert_eq!(op.op_type, "reduce_mean");
        assert!(op.attributes.contains_key("axes"));
        assert!(matches!(
            op.attributes.get("keep_dims"),
            Some(Value::Bool(true))
        ));
    }

    #[test]
    fn test_layer_norm() {
        let mut node = make_node("LayerNormalization", &["X", "gamma", "beta"], &["Y"]);
        node.attribute.push(make_int_attr("axis", -1));
        node.attribute.push(make_float_attr("epsilon", 1e-5));
        let ops = convert_node(&node).unwrap();
        let op = &ops[0];
        assert_eq!(op.op_type, "layer_norm");
        assert_eq!(op.inputs.len(), 3);
        assert!(op.attributes.contains_key("axes"));
        assert!(op.attributes.contains_key("epsilon"));
    }

    #[test]
    fn test_clip_input_form() {
        let node = make_node("Clip", &["X", "min", "max"], &["Y"]);
        let ops = convert_node(&node).unwrap();
        let op = &ops[0];
        assert_eq!(op.op_type, "clip");
        assert!(matches!(op.inputs.get("x"), Some(Value::Reference(s)) if s == "X"));
        assert!(matches!(op.inputs.get("alpha"), Some(Value::Reference(s)) if s == "min"));
        assert!(matches!(op.inputs.get("beta"), Some(Value::Reference(s)) if s == "max"));
    }

    #[test]
    fn test_slice() {
        let node = make_node(
            "Slice",
            &["data", "starts", "ends", "axes", "steps"],
            &["out"],
        );
        let ops = convert_node(&node).unwrap();
        let op = &ops[0];
        assert_eq!(op.op_type, "slice_by_index");
        assert_eq!(op.inputs.len(), 5);
    }

    // -- Individual P2 ops --------------------------------------------------

    #[test]
    fn test_where() {
        let node = make_node("Where", &["cond", "A", "B"], &["out"]);
        let ops = convert_node(&node).unwrap();
        let op = &ops[0];
        assert_eq!(op.op_type, "select");
        assert!(matches!(op.inputs.get("cond"), Some(Value::Reference(s)) if s == "cond"));
        assert!(matches!(op.inputs.get("a"), Some(Value::Reference(s)) if s == "A"));
        assert!(matches!(op.inputs.get("b"), Some(Value::Reference(s)) if s == "B"));
    }

    #[test]
    fn test_conv_transpose() {
        let mut node = make_node("ConvTranspose", &["X", "W"], &["Y"]);
        node.attribute.push(make_int_list_attr("strides", &[2, 2]));
        node.attribute.push(make_int_attr("group", 1));
        let ops = convert_node(&node).unwrap();
        let op = &ops[0];
        assert_eq!(op.op_type, "conv_transpose");
        assert!(op.attributes.contains_key("strides"));
        assert!(matches!(op.attributes.get("groups"), Some(Value::Int(1))));
    }

    #[test]
    fn test_resize_with_mode() {
        let mut node = make_node("Resize", &["X", "", "scales"], &["Y"]);
        node.attribute.push(make_string_attr("mode", "linear"));
        node.attribute.push(make_string_attr(
            "coordinate_transformation_mode",
            "align_corners",
        ));
        let ops = convert_node(&node).unwrap();
        let op = &ops[0];
        assert_eq!(op.op_type, "upsample_bilinear");
        assert!(matches!(op.inputs.get("scales"), Some(Value::Reference(s)) if s == "scales"));
        assert!(!op.inputs.contains_key("roi")); // empty input skipped
        assert!(matches!(
            op.attributes.get("align_corners"),
            Some(Value::Bool(true))
        ));
    }

    #[test]
    fn test_split_with_axis() {
        let mut node = make_node("Split", &["X"], &["out1", "out2"]);
        node.attribute.push(make_int_attr("axis", 1));
        node.attribute.push(make_int_list_attr("split", &[3, 5]));
        let ops = convert_node(&node).unwrap();
        let op = &ops[0];
        assert_eq!(op.op_type, "split");
        assert_eq!(op.outputs, vec!["out1", "out2"]);
        assert!(matches!(op.attributes.get("axis"), Some(Value::Int(1))));
        assert!(op.attributes.contains_key("split_sizes"));
    }

    // -- Naming & outputs ---------------------------------------------------

    #[test]
    fn test_outputs_attached() {
        let node = make_node("Relu", &["x"], &["y1", "y2"]);
        let ops = convert_node(&node).unwrap();
        assert_eq!(ops[0].outputs, vec!["y1", "y2"]);
    }

    #[test]
    fn test_op_name_from_node_name() {
        let mut node = make_node("Relu", &["x"], &["out"]);
        node.name = "my_relu".to_string();
        let ops = convert_node(&node).unwrap();
        assert_eq!(ops[0].name, "my_relu");
    }

    #[test]
    fn test_op_name_fallback_to_output() {
        let node = make_node("Relu", &["x"], &["relu_out"]);
        let ops = convert_node(&node).unwrap();
        assert_eq!(ops[0].name, "relu_out");
    }

    // -- dtype mapping ------------------------------------------------------

    #[test]
    fn test_onnx_dtype_to_mil() {
        assert_eq!(onnx_dtype_to_mil(1).unwrap(), "float32");
        assert_eq!(onnx_dtype_to_mil(7).unwrap(), "int64");
        assert_eq!(onnx_dtype_to_mil(10).unwrap(), "float16");
        assert_eq!(onnx_dtype_to_mil(9).unwrap(), "bool");
        assert!(onnx_dtype_to_mil(99).is_err());
    }

    // -- Op name mapping: ANE-specific names --------------------------------

    #[test]
    fn unsqueeze_converts_to_expand_dims() {
        let node = make_node("Unsqueeze", &["x", "axes"], &["out"]);
        let ops = convert_node(&node).unwrap();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op_type, "expand_dims");
    }

    #[test]
    fn slice_converts_to_slice_by_index() {
        let node = make_node("Slice", &["data", "starts", "ends"], &["out"]);
        let ops = convert_node(&node).unwrap();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op_type, "slice_by_index");
    }

    // -- Causal convolution detection ---------------------------------------

    #[test]
    fn test_conv_causal_1d() {
        // kernel_size=3, dilation=1 → left=2, right=0 → pads=[0, 2, 0, 0]
        let mut node = make_node("Conv", &["input", "W"], &["Y"]);
        node.attribute
            .push(make_int_list_attr("kernel_shape", &[1, 3]));
        node.attribute
            .push(make_int_list_attr("pads", &[0, 2, 0, 0]));
        node.attribute
            .push(make_int_list_attr("dilations", &[1, 1]));

        let ops = convert_node(&node).unwrap();
        let op = &ops[0];
        assert_eq!(
            op.attributes.get("causal"),
            Some(&Value::Bool(true)),
            "expected causal=true for left-only padding"
        );
    }

    #[test]
    fn test_conv_causal_dilated() {
        // kernel_size=3, dilation=2 → left=(3-1)*2=4, right=0 → pads=[0, 4, 0, 0]
        let mut node = make_node("Conv", &["input", "W"], &["Y"]);
        node.attribute
            .push(make_int_list_attr("kernel_shape", &[1, 3]));
        node.attribute
            .push(make_int_list_attr("pads", &[0, 4, 0, 0]));
        node.attribute
            .push(make_int_list_attr("dilations", &[1, 2]));

        let ops = convert_node(&node).unwrap();
        let op = &ops[0];
        assert_eq!(
            op.attributes.get("causal"),
            Some(&Value::Bool(true)),
            "expected causal=true for dilated causal conv"
        );
    }

    #[test]
    fn test_conv_symmetric_padding_not_causal() {
        // Symmetric padding [1, 1, 1, 1] should NOT be flagged causal.
        let mut node = make_node("Conv", &["input", "W"], &["Y"]);
        node.attribute
            .push(make_int_list_attr("kernel_shape", &[3, 3]));
        node.attribute
            .push(make_int_list_attr("pads", &[1, 1, 1, 1]));
        node.attribute
            .push(make_int_list_attr("dilations", &[1, 1]));

        let ops = convert_node(&node).unwrap();
        let op = &ops[0];
        assert_eq!(
            op.attributes.get("causal"),
            Some(&Value::Bool(false)),
            "symmetric padding should not be causal"
        );
    }

    #[test]
    fn test_conv_no_kernel_shape_no_causal_attr() {
        // Without kernel_shape, causal cannot be determined.
        let mut node = make_node("Conv", &["input", "W"], &["Y"]);
        node.attribute
            .push(make_int_list_attr("pads", &[1, 1, 1, 1]));

        let ops = convert_node(&node).unwrap();
        let op = &ops[0];
        assert!(
            !op.attributes.contains_key("causal"),
            "causal attr should not be set without kernel_shape"
        );
    }

    #[test]
    fn test_conv_auto_pad_no_causal_attr() {
        // auto_pad modes don't get a causal attribute (pad_type != "custom").
        let mut node = make_node("Conv", &["input", "W"], &["Y"]);
        node.attribute
            .push(make_string_attr("auto_pad", "SAME_UPPER"));
        node.attribute
            .push(make_int_list_attr("kernel_shape", &[3, 3]));

        let ops = convert_node(&node).unwrap();
        let op = &ops[0];
        assert!(
            !op.attributes.contains_key("causal"),
            "causal attr should not be set for auto_pad modes"
        );
    }
}
