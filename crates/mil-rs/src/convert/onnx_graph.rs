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

    // 4. Convert initializers to const operations.
    let mut block = Block::new();
    for tensor in &graph.initializer {
        match initializer_to_const(tensor) {
            Ok(op) => block.add_op(op),
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
                for op in ops {
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

    function.body = block;
    Ok(function)
}

// ---------------------------------------------------------------------------
// Initializer → const op
// ---------------------------------------------------------------------------

/// Convert an ONNX [`TensorProto`] (initializer/weight) to a MIL `const` [`Operation`].
fn initializer_to_const(tensor: &TensorProto) -> Result<Operation> {
    let dtype = onnx_dtype_to_scalar(tensor.data_type)?;
    let shape: Vec<usize> = tensor.dims.iter().map(|&d| d as usize).collect();

    let raw_bytes = extract_tensor_raw_data(tensor, dtype);

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
    op.attributes.insert(
        "shape".into(),
        Value::List(shape.iter().map(|&d| Value::Int(d as i64)).collect()),
    );
    Ok(op)
}

/// Extract raw bytes from an ONNX [`TensorProto`].
///
/// ONNX tensors store data either in `raw_data` or in typed fields
/// (`float_data`, `int32_data`, etc.). This function normalises both
/// representations into a single `Vec<u8>`.
fn extract_tensor_raw_data(tensor: &TensorProto, dtype: ScalarType) -> Vec<u8> {
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
fn onnx_dtype_to_scalar(dtype: i32) -> Result<ScalarType> {
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
