//! Convert an ONNX [`ModelProto`] graph into a MIL IR [`Program`].
//!
//! This module implements the top-level ONNX → MIL conversion pipeline:
//!
//! 1. Extract opset version from the model metadata.
//! 2. Identify real graph inputs (excluding initializers).
//! 3. Lower ONNX initializers (weights/biases) to MIL `const` operations.
//! 4. Walk ONNX nodes in topological order, converting each via [`convert_node`].
//! 5. Assemble the resulting operations into a MIL [`Program`].
//!
//! Unsupported operations are collected as warnings rather than aborting the
//! entire conversion, so callers can inspect partial results.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::convert::onnx_to_mil::convert_node;
use crate::error::{MilError, Result};
use crate::ir::{Block, Function, Operation, Program, ScalarType, TensorType, Value};
use crate::proto::onnx::{
    GraphProto, ModelProto, TensorProto, TypeProto, ValueInfoProto, tensor_shape_proto::dimension,
    type_proto,
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// The result of converting an ONNX model, including the MIL [`Program`] and
/// any warnings encountered during conversion (e.g. unsupported ops).
#[derive(Debug)]
pub struct ConversionResult {
    /// The converted MIL program.
    pub program: Program,
    /// Warnings collected during conversion (unsupported ops, skipped nodes, etc.).
    pub warnings: Vec<String>,
}

/// Convert an ONNX [`ModelProto`] into a MIL IR [`Program`].
///
/// This is the main entry point for ONNX → CoreML conversion. The model's
/// graph is converted into a single MIL function named `"main"`.
///
/// # Errors
///
/// Returns [`MilError::Validation`] if the model has no graph.
pub fn onnx_to_program(model: &ModelProto) -> Result<ConversionResult> {
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| MilError::Validation("ONNX model has no graph".into()))?;

    let opset = extract_opset_version(model);
    let mut warnings = Vec::new();
    let function = convert_graph(graph, opset, &mut warnings)?;

    let mut program = Program::new("1.0.0");
    program.add_function(function);

    Ok(ConversionResult { program, warnings })
}

// ---------------------------------------------------------------------------
// Graph conversion
// ---------------------------------------------------------------------------

/// Convert an ONNX [`GraphProto`] into a MIL [`Function`].
fn convert_graph(graph: &GraphProto, _opset: i64, warnings: &mut Vec<String>) -> Result<Function> {
    // 1. Collect initializer names for fast lookup.
    let initializer_names: HashSet<&str> =
        graph.initializer.iter().map(|t| t.name.as_str()).collect();

    // 2. Real inputs are graph inputs that are NOT initializers.
    let real_inputs: Vec<&ValueInfoProto> = graph
        .input
        .iter()
        .filter(|vi| !initializer_names.contains(vi.name.as_str()))
        .collect();

    // 3. Build function with typed inputs.
    let mut function = Function::new("main");
    for input in &real_inputs {
        match value_info_to_tensor_type(input) {
            Ok(ty) => {
                function = function.with_input(&input.name, ty);
            }
            Err(e) => {
                warnings.push(format!("skipping input '{}': {e}", input.name));
            }
        }
    }

    // Build a type map from ONNX value_info, graph inputs, and graph outputs
    // so we can attach output types to every operation.
    let mut onnx_type_map: HashMap<String, TensorType> = HashMap::new();
    for vi in graph
        .input
        .iter()
        .chain(graph.output.iter())
        .chain(graph.value_info.iter())
    {
        if let Ok(tt) = value_info_to_tensor_type(vi) {
            onnx_type_map.insert(vi.name.clone(), tt);
        }
    }
    // Initializers also have known types.  int64 is narrowed to int32
    // for CoreML compatibility, so update the type accordingly.
    for tensor in &graph.initializer {
        if let Ok(mut dtype) = onnx_dtype_to_scalar(tensor.data_type) {
            if dtype == ScalarType::Int64 {
                dtype = ScalarType::Int32;
            }
            let shape: Vec<Option<usize>> = tensor.dims.iter().map(|&d| Some(d as usize)).collect();
            onnx_type_map.insert(
                tensor.name.clone(),
                TensorType::with_dynamic_shape(dtype, shape),
            );
        }
    }

    // 4. Convert initializers to const operations.
    let mut block = Block::new();
    for tensor in &graph.initializer {
        match initializer_to_const(tensor) {
            Ok(mut op) => {
                stamp_output_types(&mut op, &onnx_type_map);
                block.add_op(op);
            }
            Err(e) => {
                warnings.push(format!("skipping initializer '{}': {e}", tensor.name));
            }
        }
    }

    // 5. Topologically sort the nodes, then convert each one.
    let sorted_nodes = topological_sort(&graph.node);
    for node_idx in &sorted_nodes {
        let node = &graph.node[*node_idx];
        match convert_node(node) {
            Ok(ops) => {
                for mut op in ops {
                    stamp_output_types(&mut op, &onnx_type_map);
                    block.add_op(op);
                }
            }
            Err(MilError::UnsupportedOp(op_type)) => {
                warnings.push(format!("unsupported op: {op_type}"));
                // Emit a placeholder so downstream references still resolve.
                let mut placeholder = Operation::new("unsupported", node.name.clone());
                placeholder
                    .attributes
                    .insert("original_op".into(), Value::String(op_type));
                for out in &node.output {
                    placeholder = placeholder.with_output(out.clone());
                }
                stamp_output_types(&mut placeholder, &onnx_type_map);
                block.add_op(placeholder);
            }
            Err(e) => {
                warnings.push(format!(
                    "error converting node '{}' ({}): {e}",
                    node.name, node.op_type
                ));
            }
        }
    }

    // 6. Set block outputs from graph outputs.
    for output in &graph.output {
        block.outputs.push(output.name.clone());
    }

    // 7. Propagate shapes for ops that still have None output_types.
    //    Without this, shape-changing ops (conv, pool) get unknown
    //    dimensions in the proto, which crashes the BNNS backend.
    propagate_output_types(&function.inputs, &mut block);

    function.body = block;
    Ok(function)
}

/// Fill in `op.output_types` from the ONNX type map for every output that
/// currently has `None` as its type.
fn stamp_output_types(op: &mut Operation, type_map: &HashMap<String, TensorType>) {
    // Ensure output_types vec is the right length.
    op.output_types.resize(op.outputs.len(), None);
    for (i, name) in op.outputs.iter().enumerate() {
        if op.output_types[i].is_none() {
            if let Some(tt) = type_map.get(name) {
                op.output_types[i] = Some(tt.clone());
            }
        }
    }
}

/// Propagate output types through the block for ops that still have `None`.
///
/// Builds a type map from function inputs and ops whose output types are
/// already known, then infers missing types using op-specific shape rules.
fn propagate_output_types(func_inputs: &[(String, TensorType)], block: &mut Block) {
    // Seed the type map with function inputs.
    let mut type_map: HashMap<String, TensorType> = HashMap::new();
    for (name, tt) in func_inputs {
        type_map.insert(name.clone(), tt.clone());
    }

    // Also collect const tensor values so reshape can read target shapes.
    // Stored as raw usize but preserving bit patterns for signed values
    // (reshape uses 0 = keep dim, -1 = infer dim).
    let mut const_values: HashMap<String, Vec<usize>> = HashMap::new();

    for op in &block.operations {
        for (i, name) in op.outputs.iter().enumerate() {
            if let Some(Some(tt)) = op.output_types.get(i) {
                type_map.insert(name.clone(), tt.clone());
            } else if op.op_type == "const" {
                let val = op.inputs.get("val").or_else(|| op.attributes.get("val"));
                if let Some(Value::Tensor { shape, dtype, data }) = val {
                    type_map.insert(name.clone(), TensorType::new(*dtype, shape.clone()));
                    if *dtype == ScalarType::Int32 && data.len() % 4 == 0 {
                        let dims: Vec<usize> = data
                            .chunks_exact(4)
                            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64 as usize)
                            .collect();
                        const_values.insert(name.clone(), dims);
                    } else if *dtype == ScalarType::Int64 && data.len() % 8 == 0 {
                        let dims: Vec<usize> = data
                            .chunks_exact(8)
                            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize)
                            .collect();
                        const_values.insert(name.clone(), dims);
                    }
                }
            }
        }
    }

    // Second pass: infer missing output types and fill in global pool params.
    for op in &mut block.operations {
        // For global pool ops, set kernel_sizes from the input shape.
        if matches!(op.op_type.as_str(), "max_pool" | "avg_pool")
            && op
                .attributes
                .get("global_pool")
                .is_some_and(|v| matches!(v, Value::Bool(true)))
            && !op.inputs.contains_key("kernel_sizes")
        {
            if let Some(Value::Reference(n)) = op.inputs.get("x") {
                if let Some(in_tt) = type_map.get(n) {
                    if in_tt.shape.len() == 4 {
                        let h = in_tt.shape[2].unwrap_or(1);
                        let w = in_tt.shape[3].unwrap_or(1);
                        let to_i32_tensor = |vals: &[i32]| -> Value {
                            let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
                            Value::Tensor {
                                data,
                                shape: vec![vals.len()],
                                dtype: ScalarType::Int32,
                            }
                        };
                        op.inputs
                            .insert("kernel_sizes".into(), to_i32_tensor(&[h as i32, w as i32]));
                        op.inputs.insert("strides".into(), to_i32_tensor(&[1, 1]));
                        op.inputs
                            .insert("pad_type".into(), Value::String("valid".into()));
                        op.inputs.insert("pad".into(), to_i32_tensor(&[0, 0, 0, 0]));
                        op.inputs
                            .insert("exclude_padding_from_average".into(), Value::Bool(true));
                        op.inputs.insert("ceil_mode".into(), Value::Bool(false));
                    }
                }
            }
        }

        // For flatten (reshape with flatten_axis), compute and insert the
        // target shape from the input dimensions.
        if op.op_type == "reshape"
            && op.attributes.contains_key("flatten_axis")
            && !op.inputs.contains_key("shape")
        {
            if let Some(Value::Reference(n)) = op.inputs.get("x") {
                if let Some(in_tt) = type_map.get(n) {
                    let axis = op
                        .attributes
                        .get("flatten_axis")
                        .and_then(|v| {
                            if let Value::Int(a) = v {
                                Some(*a as usize)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(1);
                    let dims: Vec<usize> = in_tt.shape.iter().map(|d| d.unwrap_or(1)).collect();
                    let batch: usize = dims[..axis].iter().product();
                    let rest: usize = dims[axis..].iter().product();
                    let shape_data: Vec<u8> = [batch as i32, rest as i32]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect();
                    op.inputs.insert(
                        "shape".into(),
                        Value::Tensor {
                            data: shape_data,
                            shape: vec![2],
                            dtype: ScalarType::Int32,
                        },
                    );
                }
            }
        }

        for (i, out_name) in op.outputs.iter().enumerate() {
            if op.output_types.get(i).is_some_and(|ot| ot.is_some()) {
                continue;
            }

            // Resolve the type of a value from the type map.
            let resolve = |v: &Value| -> Option<&TensorType> {
                match v {
                    Value::Reference(name) => type_map.get(name),
                    Value::List(items) => items.iter().find_map(|item| {
                        if let Value::Reference(name) = item {
                            type_map.get(name)
                        } else {
                            None
                        }
                    }),
                    _ => None,
                }
            };

            let inferred = match op.op_type.as_str() {
                "conv" => infer_conv_output(op, &type_map),
                "max_pool" | "avg_pool" => infer_pool_output(op, &type_map),
                "reshape" | "flatten" => infer_reshape_output(op, &type_map, &const_values),
                "concat" => infer_concat_output(op, &type_map),
                _ => {
                    // Element-wise / pass-through: output type = input type.
                    ["x", "data", "input", "values"]
                        .iter()
                        .filter_map(|&p| op.inputs.get(p))
                        .find_map(&resolve)
                        .or_else(|| op.inputs.values().find_map(resolve))
                        .cloned()
                }
            };

            if let Some(out_tt) = inferred {
                type_map.insert(out_name.clone(), out_tt.clone());
                while op.output_types.len() <= i {
                    op.output_types.push(None);
                }
                op.output_types[i] = Some(out_tt);
            }
        }
    }
}

/// Infer conv output shape: `[N, out_channels, out_h, out_w]`.
fn infer_conv_output(op: &Operation, type_map: &HashMap<String, TensorType>) -> Option<TensorType> {
    let in_tt = op.inputs.get("x").and_then(|v| {
        if let Value::Reference(n) = v {
            type_map.get(n)
        } else {
            None
        }
    })?;
    let w_tt = op.inputs.get("weight").and_then(|v| {
        if let Value::Reference(n) = v {
            type_map.get(n)
        } else {
            None
        }
    })?;

    let in_shape = &in_tt.shape; // [N, C_in, H, W]
    let w_shape = &w_tt.shape; // [C_out, C_in/groups, kH, kW]
    if in_shape.len() != 4 || w_shape.len() != 4 {
        return Some(in_tt.clone());
    }

    let out_channels = w_shape[0]?;
    let in_h = in_shape[2]?;
    let in_w = in_shape[3]?;
    let k_h = w_shape[2]?;
    let k_w = w_shape[3]?;

    let strides = get_int_list(op, "strides").unwrap_or_else(|| vec![1, 1]);
    let dilations = get_int_list(op, "dilations").unwrap_or_else(|| vec![1, 1]);
    let pads = get_int_list(op, "pad").unwrap_or_else(|| vec![0, 0, 0, 0]);

    let pad_top = *pads.first().unwrap_or(&0) as usize;
    let pad_bottom = *pads.get(1).unwrap_or(&0) as usize;
    let pad_left = *pads.get(2).unwrap_or(&0) as usize;
    let pad_right = *pads.get(3).unwrap_or(&0) as usize;
    let stride_h = *strides.first().unwrap_or(&1) as usize;
    let stride_w = *strides.get(1).unwrap_or(&1) as usize;
    let dil_h = *dilations.first().unwrap_or(&1) as usize;
    let dil_w = *dilations.get(1).unwrap_or(&1) as usize;

    // Check pad_type for "same" padding.
    let is_same = op
        .inputs
        .get("pad_type")
        .is_some_and(|v| matches!(v, Value::String(s) if s == "same"));

    let (out_h, out_w) = if is_same {
        (in_h.div_ceil(stride_h), in_w.div_ceil(stride_w))
    } else {
        let eff_kh = dil_h * (k_h - 1) + 1;
        let eff_kw = dil_w * (k_w - 1) + 1;
        (
            (in_h + pad_top + pad_bottom).saturating_sub(eff_kh) / stride_h + 1,
            (in_w + pad_left + pad_right).saturating_sub(eff_kw) / stride_w + 1,
        )
    };

    Some(TensorType::new(
        in_tt.scalar_type,
        vec![in_shape[0].unwrap_or(1), out_channels, out_h, out_w],
    ))
}

/// Infer pool output shape: channels preserved, spatial dims shrink.
fn infer_pool_output(op: &Operation, type_map: &HashMap<String, TensorType>) -> Option<TensorType> {
    let in_tt = op.inputs.get("x").and_then(|v| {
        if let Value::Reference(n) = v {
            type_map.get(n)
        } else {
            None
        }
    })?;
    let in_shape = &in_tt.shape;
    if in_shape.len() != 4 {
        return Some(in_tt.clone());
    }

    let in_c = in_shape[1]?;
    let in_h = in_shape[2]?;
    let in_w = in_shape[3]?;

    let is_global = op
        .attributes
        .get("global_pool")
        .is_some_and(|v| matches!(v, Value::Bool(true)));

    let kernels = if is_global {
        // Global pool: kernel covers entire spatial extent → output is 1×1.
        Some(vec![in_h as i64, in_w as i64])
    } else {
        get_int_list(op, "kernel_sizes")
    }
    .unwrap_or_else(|| vec![1, 1]);
    let strides = get_int_list(op, "strides").unwrap_or_else(|| vec![1, 1]);
    let pads = get_int_list(op, "pad").unwrap_or_else(|| vec![0, 0, 0, 0]);

    let k_h = *kernels.first().unwrap_or(&1) as usize;
    let k_w = *kernels.get(1).unwrap_or(&1) as usize;
    let stride_h = *strides.first().unwrap_or(&1) as usize;
    let stride_w = *strides.get(1).unwrap_or(&1) as usize;
    let pad_top = *pads.first().unwrap_or(&0) as usize;
    let pad_bottom = *pads.get(1).unwrap_or(&0) as usize;
    let pad_left = *pads.get(2).unwrap_or(&0) as usize;
    let pad_right = *pads.get(3).unwrap_or(&0) as usize;

    let out_h = (in_h + pad_top + pad_bottom).saturating_sub(k_h) / stride_h + 1;
    let out_w = (in_w + pad_left + pad_right).saturating_sub(k_w) / stride_w + 1;

    Some(TensorType::new(
        in_tt.scalar_type,
        vec![in_shape[0].unwrap_or(1), in_c, out_h, out_w],
    ))
}

/// Infer reshape/flatten output from the `shape` input constant.
fn infer_reshape_output(
    op: &Operation,
    type_map: &HashMap<String, TensorType>,
    const_values: &HashMap<String, Vec<usize>>,
) -> Option<TensorType> {
    let in_tt = op.inputs.get("x").and_then(|v| {
        if let Value::Reference(n) = v {
            type_map.get(n)
        } else {
            None
        }
    })?;

    // Read the target shape from the const tensor's raw int data.
    if let Some(Value::Reference(shape_name)) = op.inputs.get("shape") {
        // Look up the const op's raw tensor to get signed values (may contain 0 / -1).
        let raw_dims = const_values.get(shape_name);
        if let Some(dims) = raw_dims {
            let in_numel: usize = in_tt.shape.iter().map(|d| d.unwrap_or(1)).product();
            let mut out: Vec<usize> = dims.clone();
            let mut infer_idx: Option<usize> = None;
            let mut known_product: usize = 1;

            for (i, &d) in dims.iter().enumerate() {
                let signed = d as i64;
                if signed == 0 {
                    // 0 means "keep this dim from input".
                    out[i] = in_tt.shape.get(i).and_then(|s| *s).unwrap_or(1);
                    known_product *= out[i];
                } else if signed == -1 {
                    infer_idx = Some(i);
                } else {
                    known_product *= d;
                }
            }

            if let Some(idx) = infer_idx {
                out[idx] = if known_product > 0 {
                    in_numel / known_product
                } else {
                    1
                };
            }

            if !out.is_empty() {
                return Some(TensorType::new(in_tt.scalar_type, out));
            }
        }
    }

    Some(in_tt.clone())
}

/// Infer concat output: sum along concat axis, other dims preserved.
fn infer_concat_output(
    op: &Operation,
    type_map: &HashMap<String, TensorType>,
) -> Option<TensorType> {
    let values = op.inputs.get("values")?;
    let Value::List(refs) = values else {
        return None;
    };

    let input_types: Vec<&TensorType> = refs
        .iter()
        .filter_map(|v| {
            if let Value::Reference(name) = v {
                type_map.get(name)
            } else {
                None
            }
        })
        .collect();

    let first = input_types.first()?;
    let axis = op
        .inputs
        .get("axis")
        .or_else(|| op.attributes.get("axis"))
        .and_then(|v| {
            if let Value::Int(a) = v {
                Some(*a as usize)
            } else {
                None
            }
        })
        .unwrap_or(0);

    let mut out_shape: Vec<usize> = first.shape.iter().map(|d| d.unwrap_or(1)).collect();
    if axis < out_shape.len() {
        out_shape[axis] = input_types
            .iter()
            .map(|tt| tt.shape.get(axis).and_then(|d| *d).unwrap_or(0))
            .sum();
    }

    Some(TensorType::new(first.scalar_type, out_shape))
}

/// Extract an integer list from an op input (e.g. strides, pads).
fn get_int_list(op: &Operation, name: &str) -> Option<Vec<i64>> {
    let val = op.inputs.get(name).or_else(|| op.attributes.get(name))?;
    match val {
        Value::List(items) => {
            let ints: Vec<i64> = items
                .iter()
                .filter_map(|v| {
                    if let Value::Int(i) = v {
                        Some(*i)
                    } else {
                        None
                    }
                })
                .collect();
            if ints.is_empty() { None } else { Some(ints) }
        }
        _ => None,
    }
}

/// Convert an ONNX [`TensorProto`] (initializer/weight) to a MIL `const` [`Operation`].
///
/// CoreML MIL does not support int64 tensors in most operations, so int64
/// initializers are automatically narrowed to int32.
fn initializer_to_const(tensor: &TensorProto) -> Result<Operation> {
    let mut dtype = onnx_dtype_to_scalar(tensor.data_type)?;
    let shape: Vec<usize> = tensor.dims.iter().map(|&d| d as usize).collect();

    let mut raw_bytes = extract_tensor_raw_data(tensor, dtype);

    // Narrow int64 → int32 for CoreML compatibility.
    if dtype == ScalarType::Int64 {
        raw_bytes = raw_bytes
            .chunks_exact(8)
            .flat_map(|c| {
                let v = i64::from_le_bytes(c.try_into().unwrap());
                (v as i32).to_le_bytes()
            })
            .collect();
        dtype = ScalarType::Int32;
    }

    let mut op = Operation::new("const", &tensor.name);
    op = op.with_output(&tensor.name);
    op.attributes.insert(
        "val".into(),
        Value::Tensor {
            data: raw_bytes,
            shape: shape.clone(),
            dtype,
        },
    );
    Ok(op)
}

/// Extract raw bytes from an ONNX [`TensorProto`].
///
/// ONNX tensors store data either in `raw_data` or in typed fields
/// (`float_data`, `int32_data`, etc.). This function normalises both
/// representations into a single `Vec<u8>`.
pub(crate) fn extract_tensor_raw_data(tensor: &TensorProto, dtype: ScalarType) -> Vec<u8> {
    if !tensor.raw_data.is_empty() {
        return tensor.raw_data.clone();
    }

    match dtype {
        ScalarType::Float32 => tensor
            .float_data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        ScalarType::Float64 => tensor
            .double_data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        ScalarType::Int32 => tensor
            .int32_data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        ScalarType::Int64 => tensor
            .int64_data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        ScalarType::UInt64 => tensor
            .uint64_data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        // For types stored in int32_data (uint8, int8, uint16, int16, float16, bool).
        ScalarType::UInt8 | ScalarType::Bool => tensor
            .int32_data
            .iter()
            .flat_map(|v| (*v as u8).to_le_bytes())
            .collect(),
        ScalarType::Int8 => tensor
            .int32_data
            .iter()
            .flat_map(|v| (*v as i8).to_le_bytes())
            .collect(),
        ScalarType::UInt16 => tensor
            .int32_data
            .iter()
            .flat_map(|v| (*v as u16).to_le_bytes())
            .collect(),
        ScalarType::Int16 | ScalarType::Float16 => tensor
            .int32_data
            .iter()
            .flat_map(|v| (*v as i16).to_le_bytes())
            .collect(),
        ScalarType::UInt32 => tensor
            .int32_data
            .iter()
            .flat_map(|v| (*v as u32).to_le_bytes())
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// Type conversion helpers
// ---------------------------------------------------------------------------

/// Map an ONNX `TensorProto::DataType` integer to a MIL [`ScalarType`].
pub(crate) fn onnx_dtype_to_scalar(dtype: i32) -> Result<ScalarType> {
    match dtype {
        1 => Ok(ScalarType::Float32),
        2 => Ok(ScalarType::UInt8),
        3 => Ok(ScalarType::Int8),
        4 => Ok(ScalarType::UInt16),
        5 => Ok(ScalarType::Int16),
        6 => Ok(ScalarType::Int32),
        7 => Ok(ScalarType::Int64),
        9 => Ok(ScalarType::Bool),
        10 => Ok(ScalarType::Float16),
        11 => Ok(ScalarType::Float64),
        12 => Ok(ScalarType::UInt32),
        13 => Ok(ScalarType::UInt64),
        other => Err(MilError::UnsupportedOp(format!(
            "unsupported ONNX data type: {other}"
        ))),
    }
}

/// Convert an ONNX [`ValueInfoProto`] to a MIL [`TensorType`].
fn value_info_to_tensor_type(info: &ValueInfoProto) -> Result<TensorType> {
    let type_proto = info
        .r#type
        .as_ref()
        .ok_or_else(|| MilError::Validation(format!("input '{}' has no type info", info.name)))?;

    tensor_type_from_type_proto(type_proto)
}

/// Extract a [`TensorType`] from an ONNX [`TypeProto`].
fn tensor_type_from_type_proto(tp: &TypeProto) -> Result<TensorType> {
    let tensor = match &tp.value {
        Some(type_proto::Value::TensorType(t)) => t,
        _ => {
            return Err(MilError::Validation(
                "expected tensor type in TypeProto".into(),
            ));
        }
    };

    let scalar_type = onnx_dtype_to_scalar(tensor.elem_type)?;

    let shape: Vec<Option<usize>> = match &tensor.shape {
        Some(shape_proto) => shape_proto
            .dim
            .iter()
            .map(|d| match &d.value {
                Some(dimension::Value::DimValue(v)) if *v >= 0 => Some(*v as usize),
                _ => None, // dynamic or symbolic dimension
            })
            .collect(),
        None => Vec::new(),
    };

    Ok(TensorType::with_dynamic_shape(scalar_type, shape))
}

// ---------------------------------------------------------------------------
// Opset version extraction
// ---------------------------------------------------------------------------

/// Extract the default opset version from the model's `opset_import`.
///
/// The default domain is represented by an empty string. If no default is
/// found, returns `0`.
fn extract_opset_version(model: &ModelProto) -> i64 {
    model
        .opset_import
        .iter()
        .find(|os| os.domain.is_empty())
        .map(|os| os.version)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Topological sort
// ---------------------------------------------------------------------------

/// Topologically sort ONNX nodes by their data dependencies.
///
/// Returns a vector of indices into the input `nodes` slice in valid
/// execution order. Nodes that are already in topological order pass
/// through with minimal overhead.
fn topological_sort(nodes: &[crate::proto::onnx::NodeProto]) -> Vec<usize> {
    // Build name → producing-node-index map.
    let mut producer: HashMap<&str, usize> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for out in &node.output {
            if !out.is_empty() {
                producer.insert(out.as_str(), i);
            }
        }
    }

    // Build adjacency list and in-degree count.
    let n = nodes.len();
    let mut in_degree = vec![0u32; n];
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (i, node) in nodes.iter().enumerate() {
        for inp in &node.input {
            if let Some(&prod) = producer.get(inp.as_str()) {
                if prod != i {
                    dependents[prod].push(i);
                    in_degree[i] += 1;
                }
            }
        }
    }

    // Kahn's algorithm.
    let mut queue: VecDeque<usize> = VecDeque::new();
    for (i, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            queue.push_back(i);
        }
    }

    let mut order = Vec::with_capacity(n);
    while let Some(idx) = queue.pop_front() {
        order.push(idx);
        for &dep in &dependents[idx] {
            in_degree[dep] -= 1;
            if in_degree[dep] == 0 {
                queue.push_back(dep);
            }
        }
    }

    // If some nodes were not reached (cycle or disconnected), append them.
    if order.len() < n {
        let in_order: HashSet<usize> = order.iter().copied().collect();
        for i in 0..n {
            if !in_order.contains(&i) {
                order.push(i);
            }
        }
    }

    order
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::read_onnx;

    /// Path to the ONNX test fixtures relative to the workspace root.
    fn fixture(name: &str) -> std::path::PathBuf {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        manifest.join("../../tests/fixtures").join(name)
    }

    #[test]
    fn mnist_conversion() {
        let model = read_onnx(fixture("mnist.onnx")).unwrap();
        let result = onnx_to_program(&model).unwrap();
        let program = &result.program;

        // Should produce a program with one function.
        assert_eq!(program.functions.len(), 1);
        let main = program.main().expect("should have a main function");
        assert_eq!(main.name, "main");

        // MNIST has a single input "Input3" with shape [1, 1, 28, 28].
        assert!(
            !main.inputs.is_empty(),
            "main function should have at least one input"
        );
        let (input_name, input_ty) = &main.inputs[0];
        assert_eq!(input_name, "Input3");
        assert_eq!(input_ty.rank(), 4);
        assert_eq!(input_ty.scalar_type, ScalarType::Float32);

        // Block should contain operations (initializer consts + graph nodes).
        assert!(
            !main.body.operations.is_empty(),
            "block should have operations"
        );

        // Block should have outputs.
        assert!(
            !main.body.outputs.is_empty(),
            "block should declare outputs"
        );

        // Print warnings for visibility.
        if !result.warnings.is_empty() {
            eprintln!("MNIST conversion warnings:");
            for w in &result.warnings {
                eprintln!("  - {w}");
            }
        }
    }

    #[test]
    fn squeezenet_conversion() {
        let model = read_onnx(fixture("squeezenet1.1.onnx")).unwrap();
        let result = onnx_to_program(&model).unwrap();
        let program = &result.program;

        let main = program.main().expect("should have a main function");

        // SqueezeNet has a single image input.
        assert!(!main.inputs.is_empty(), "should have inputs");
        let (_, input_ty) = &main.inputs[0];
        assert_eq!(input_ty.rank(), 4, "input should be 4-D (NCHW)");

        // Should have many operations (weights + computation nodes).
        assert!(
            main.body.operations.len() > 50,
            "SqueezeNet should have many operations, got {}",
            main.body.operations.len()
        );

        if !result.warnings.is_empty() {
            eprintln!("SqueezeNet conversion warnings:");
            for w in &result.warnings {
                eprintln!("  - {w}");
            }
        }
    }

    #[test]
    fn empty_graph() {
        let model = ModelProto {
            graph: Some(GraphProto::default()),
            ..Default::default()
        };
        let result = onnx_to_program(&model).unwrap();
        let main = result.program.main().expect("should have main");
        assert!(main.inputs.is_empty());
        assert!(main.body.operations.is_empty());
        assert!(main.body.outputs.is_empty());
    }

    #[test]
    fn no_graph_is_error() {
        let model = ModelProto::default();
        assert!(onnx_to_program(&model).is_err());
    }

    #[test]
    fn initializers_become_const_ops() {
        let model = read_onnx(fixture("mnist.onnx")).unwrap();
        let result = onnx_to_program(&model).unwrap();
        let main = result.program.main().unwrap();

        let const_ops: Vec<_> = main
            .body
            .operations
            .iter()
            .filter(|op| op.op_type == "const")
            .collect();

        let graph = model.graph.as_ref().unwrap();
        assert_eq!(
            const_ops.len(),
            graph.initializer.len(),
            "each initializer should produce a const op"
        );

        // Every const op should have a `val` attribute with tensor data.
        for op in &const_ops {
            assert!(
                op.attributes.contains_key("val"),
                "const op '{}' should have a 'val' attribute",
                op.name
            );
        }
    }

    #[test]
    fn real_inputs_exclude_initializers() {
        let model = read_onnx(fixture("mnist.onnx")).unwrap();
        let graph = model.graph.as_ref().unwrap();

        let init_names: HashSet<&str> = graph.initializer.iter().map(|t| t.name.as_str()).collect();

        let result = onnx_to_program(&model).unwrap();
        let main = result.program.main().unwrap();

        // No function input should share a name with an initializer.
        for (name, _) in &main.inputs {
            assert!(
                !init_names.contains(name.as_str()),
                "function input '{name}' should not be an initializer"
            );
        }
    }

    #[test]
    fn opset_version_extracted() {
        let model = read_onnx(fixture("mnist.onnx")).unwrap();
        let opset = extract_opset_version(&model);
        assert!(opset > 0, "opset version should be > 0, got {opset}");
    }

    #[test]
    fn operation_count_reasonable() {
        let model = read_onnx(fixture("mnist.onnx")).unwrap();
        let graph = model.graph.as_ref().unwrap();
        let result = onnx_to_program(&model).unwrap();
        let main = result.program.main().unwrap();

        let non_const_ops: usize = main
            .body
            .operations
            .iter()
            .filter(|op| op.op_type != "const")
            .count();

        // Each ONNX node produces at least one operation, possibly more.
        assert!(
            non_const_ops >= graph.node.len(),
            "should have at least as many non-const ops ({non_const_ops}) \
             as ONNX nodes ({})",
            graph.node.len()
        );
    }

    #[test]
    fn topological_sort_preserves_valid_order() {
        use crate::proto::onnx::NodeProto;

        // Two nodes already in order: A -> B.
        let nodes = vec![
            NodeProto {
                input: vec!["x".into()],
                output: vec!["a_out".into()],
                op_type: "Relu".into(),
                ..Default::default()
            },
            NodeProto {
                input: vec!["a_out".into()],
                output: vec!["b_out".into()],
                op_type: "Relu".into(),
                ..Default::default()
            },
        ];

        let order = topological_sort(&nodes);
        assert_eq!(order, vec![0, 1]);
    }

    #[test]
    fn topological_sort_reorders_when_needed() {
        use crate::proto::onnx::NodeProto;

        // Two nodes in REVERSE order: B depends on A's output but comes first.
        let nodes = vec![
            NodeProto {
                input: vec!["a_out".into()],
                output: vec!["b_out".into()],
                op_type: "Relu".into(),
                ..Default::default()
            },
            NodeProto {
                input: vec!["x".into()],
                output: vec!["a_out".into()],
                op_type: "Relu".into(),
                ..Default::default()
            },
        ];

        let order = topological_sort(&nodes);
        // Node 1 (producer of a_out) must come before node 0 (consumer).
        assert_eq!(order, vec![1, 0]);
    }

    #[test]
    fn onnx_dtype_mapping() {
        assert_eq!(onnx_dtype_to_scalar(1).unwrap(), ScalarType::Float32);
        assert_eq!(onnx_dtype_to_scalar(7).unwrap(), ScalarType::Int64);
        assert_eq!(onnx_dtype_to_scalar(9).unwrap(), ScalarType::Bool);
        assert_eq!(onnx_dtype_to_scalar(10).unwrap(), ScalarType::Float16);
        assert!(onnx_dtype_to_scalar(99).is_err());
    }
}
