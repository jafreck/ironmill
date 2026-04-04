//! Convert protobuf `Model` → MIL IR `Program`.

use std::collections::HashMap;

use indexmap::IndexMap;

use crate::error::{MilError, Result};
use crate::ir::ScalarType;
use crate::ir::{Block, Function, Operation, Program, TensorData, TensorType, Value};
use crate::proto::mil_spec;
use crate::proto::specification::{Model, model};

/// Convert a protobuf [`Model`] into a MIL IR [`Program`].
///
/// Only handles ML Program models (the modern MIL format). Returns an error
/// for legacy NeuralNetwork models or other model types.
pub fn model_to_program(model: &Model) -> Result<Program> {
    let proto_program = match &model.r#type {
        Some(model::Type::MlProgram(program)) => program,
        Some(_) => {
            return Err(MilError::UnsupportedOp(
                "only ML Program models are supported; this model uses a different type"
                    .to_string(),
            ));
        }
        None => {
            return Err(MilError::Protobuf(
                "model has no type field set".to_string(),
            ));
        }
    };

    convert_program(proto_program)
}

// ---------------------------------------------------------------------------
// Program / Function / Block
// ---------------------------------------------------------------------------

fn convert_program(proto: &mil_spec::Program) -> Result<Program> {
    let version = proto.version.to_string();

    let mut functions = IndexMap::new();
    for (name, proto_fn) in &proto.functions {
        let function = convert_function(name, proto_fn)?;
        functions.insert(name.clone(), function);
    }

    Ok(Program {
        version,
        functions,
        attributes: std::collections::HashMap::new(),
    })
}

fn convert_function(name: &str, proto: &mil_spec::Function) -> Result<Function> {
    let mut inputs = Vec::new();
    for nvt in &proto.inputs {
        let tensor_type = convert_named_value_type_to_tensor(nvt)?;
        inputs.push((nvt.name.clone(), tensor_type));
    }

    // MIL functions use "block_specializations" keyed by opset. Pick the
    // default block — the one matching the function's `opset` field, or just
    // the first block if there's only one.
    let block = if let Some(b) = proto.block_specializations.get(&proto.opset) {
        convert_block(b)?
    } else if proto.block_specializations.len() == 1 {
        let b = proto.block_specializations.values().next().ok_or_else(|| {
            MilError::Protobuf(format!(
                "function '{}': block specialization disappeared unexpectedly",
                name
            ))
        })?;
        convert_block(b)?
    } else if proto.block_specializations.is_empty() {
        Block::new()
    } else {
        return Err(MilError::Protobuf(format!(
            "function '{}': cannot determine which block specialization to use (opset '{}')",
            name, proto.opset
        )));
    };

    Ok(Function {
        name: name.to_string(),
        inputs,
        body: block,
    })
}

fn convert_block(proto: &mil_spec::Block) -> Result<Block> {
    let mut operations = Vec::new();
    for proto_op in &proto.operations {
        operations.push(convert_operation(proto_op)?);
    }

    Ok(Block {
        operations,
        outputs: proto.outputs.clone(),
    })
}

// ---------------------------------------------------------------------------
// Operation
// ---------------------------------------------------------------------------

fn convert_operation(proto: &mil_spec::Operation) -> Result<Operation> {
    let output_names: Vec<String> = proto.outputs.iter().map(|nvt| nvt.name.clone()).collect();

    // Preserve output types from the proto NamedValueType entries.
    let output_types: Vec<Option<TensorType>> = proto
        .outputs
        .iter()
        .map(|nvt| {
            nvt.r#type.as_ref().and_then(|vt| match &vt.r#type {
                Some(mil_spec::value_type::Type::TensorType(tt)) => convert_tensor_type(tt).ok(),
                _ => None,
            })
        })
        .collect();

    // Derive a stable name: use the first output name (which is the SSA name
    // in MIL) or fall back to the op type with a placeholder.
    let name = output_names
        .first()
        .cloned()
        .unwrap_or_else(|| format!("{}_unnamed", proto.r#type));

    let mut inputs: HashMap<String, Value> = HashMap::new();
    for (param_name, argument) in &proto.inputs {
        inputs.insert(param_name.clone(), convert_argument(argument)?);
    }

    let mut attributes: HashMap<String, Value> = HashMap::new();
    for (attr_name, attr_val) in &proto.attributes {
        attributes.insert(attr_name.clone(), convert_value(attr_val)?);
    }

    // Extract compute_unit from attributes into the dedicated field.
    let compute_unit = attributes.remove("compute_unit").and_then(|v| match v {
        Value::String(s) => s.parse::<crate::ir::ComputeUnit>().ok(),
        _ => None,
    });

    Ok(Operation {
        op_type: proto.r#type.clone(),
        name,
        inputs,
        outputs: output_names,
        output_types,
        attributes,
        compute_unit,
    })
}

// ---------------------------------------------------------------------------
// Arguments & Values
// ---------------------------------------------------------------------------

/// Convert a proto `Argument` (which wraps one or more bindings) into an IR
/// `Value`.  When there are multiple bindings we produce a `Value::List`.
fn convert_argument(arg: &mil_spec::Argument) -> Result<Value> {
    let values: Vec<Value> = arg
        .arguments
        .iter()
        .map(convert_binding)
        .collect::<Result<Vec<_>>>()?;

    match values.len() {
        0 => Err(MilError::Protobuf("argument has no bindings".to_string())),
        1 => Ok(values
            .into_iter()
            .next()
            .ok_or_else(|| MilError::Protobuf("argument has empty binding list".to_string()))?),
        _ => Ok(Value::List(values)),
    }
}

fn convert_binding(binding: &mil_spec::argument::Binding) -> Result<Value> {
    use mil_spec::argument::binding::Binding;

    match &binding.binding {
        Some(Binding::Name(name)) => Ok(Value::Reference(name.clone())),
        Some(Binding::Value(v)) => convert_value(v),
        None => Err(MilError::Protobuf("argument binding is empty".to_string())),
    }
}

/// Convert a proto `Value` into an IR `Value`.
fn convert_value(proto: &mil_spec::Value) -> Result<Value> {
    use mil_spec::value;

    match &proto.value {
        Some(value::Value::ImmediateValue(imm)) => {
            convert_immediate_value(imm, proto.r#type.as_ref())
        }
        Some(value::Value::BlobFileValue(blob)) => {
            // Represent blob references as a string for now.
            Ok(Value::String(format!(
                "blob:{}@{}",
                blob.file_name, blob.offset
            )))
        }
        None => {
            // A Value with no payload but a type set is used for type
            // annotations (e.g. on function inputs).
            if let Some(vt) = &proto.r#type {
                convert_value_type_to_value(vt)
            } else {
                Err(MilError::Protobuf(
                    "value has neither payload nor type".to_string(),
                ))
            }
        }
    }
}

fn convert_immediate_value(
    imm: &mil_spec::value::ImmediateValue,
    value_type: Option<&mil_spec::ValueType>,
) -> Result<Value> {
    use mil_spec::value::immediate_value;

    match &imm.value {
        Some(immediate_value::Value::Tensor(tv)) => convert_tensor_value(tv, value_type),
        Some(immediate_value::Value::List(lv)) => {
            let items = lv
                .values
                .iter()
                .map(convert_value)
                .collect::<Result<Vec<_>>>()?;
            Ok(Value::List(items))
        }
        Some(immediate_value::Value::Tuple(tv)) => {
            let items = tv
                .values
                .iter()
                .map(convert_value)
                .collect::<Result<Vec<_>>>()?;
            Ok(Value::List(items))
        }
        Some(immediate_value::Value::Dictionary(_)) => Err(MilError::UnsupportedOp(
            "dictionary immediate values are not yet supported".to_string(),
        )),
        None => Err(MilError::Protobuf(
            "immediate value has no payload".to_string(),
        )),
    }
}

/// Convert a proto `TensorValue` into an IR `Value`.
///
/// Scalar tensors (single-element) are unwrapped into the corresponding
/// scalar `Value` variant. Multi-element tensors become `Value::List`.
/// Raw-byte tensors are reconstructed into `Value::Tensor` using the
/// accompanying `ValueType` for dtype and shape information.
fn convert_tensor_value(
    tv: &mil_spec::TensorValue,
    value_type: Option<&mil_spec::ValueType>,
) -> Result<Value> {
    use mil_spec::tensor_value;

    match &tv.value {
        Some(tensor_value::Value::Floats(f)) => {
            // When type info indicates a tensor with non-scalar shape,
            // reconstruct as Value::Tensor so that FP32 LUTs (e.g.
            // constexpr_lut_to_dense) round-trip correctly instead of
            // being deserialized as Value::List. Scalars (rank-0) stay
            // as Value::Float for attribute compatibility.
            if let Some(tt) = value_type.and_then(|vt| match &vt.r#type {
                Some(mil_spec::value_type::Type::TensorType(tt)) => Some(tt),
                _ => None,
            }) {
                let shape: Vec<usize> = tt
                    .dimensions
                    .iter()
                    .filter_map(|d| match &d.dimension {
                        Some(mil_spec::dimension::Dimension::Constant(c)) => Some(c.size as usize),
                        _ => None,
                    })
                    .collect();
                // Only reconstruct as Tensor for non-scalar shapes (rank >= 1).
                if !shape.is_empty() {
                    let dtype = convert_data_type(tt.data_type)?;
                    let data: Vec<u8> = f.values.iter().flat_map(|v| v.to_le_bytes()).collect();
                    return Ok(Value::Tensor {
                        data: TensorData::Inline(data),
                        shape,
                        dtype,
                    });
                }
            }
            match f.values.as_slice() {
                [v] => Ok(Value::Float(*v as f64)),
                vals => Ok(Value::List(
                    vals.iter().map(|v| Value::Float(*v as f64)).collect(),
                )),
            }
        }
        Some(tensor_value::Value::Doubles(d)) => match d.values.as_slice() {
            [v] => Ok(Value::Float(*v)),
            vals => Ok(Value::List(vals.iter().map(|v| Value::Float(*v)).collect())),
        },
        Some(tensor_value::Value::Ints(i)) => match i.values.as_slice() {
            [v] => Ok(Value::Int(*v as i64)),
            vals => Ok(Value::List(
                vals.iter().map(|v| Value::Int(*v as i64)).collect(),
            )),
        },
        Some(tensor_value::Value::LongInts(l)) => match l.values.as_slice() {
            [v] => Ok(Value::Int(*v)),
            vals => Ok(Value::List(vals.iter().map(|v| Value::Int(*v)).collect())),
        },
        Some(tensor_value::Value::Bools(b)) => match b.values.as_slice() {
            [v] => Ok(Value::Bool(*v)),
            vals => Ok(Value::List(vals.iter().map(|v| Value::Bool(*v)).collect())),
        },
        Some(tensor_value::Value::Strings(s)) => match s.values.as_slice() {
            [v] => Ok(Value::String(v.clone())),
            vals => Ok(Value::List(
                vals.iter().map(|v| Value::String(v.clone())).collect(),
            )),
        },
        Some(tensor_value::Value::Bytes(b)) => {
            // Reconstruct Value::Tensor from raw bytes using the accompanying
            // type info for dtype and shape.
            let tt = value_type
                .and_then(|vt| match &vt.r#type {
                    Some(mil_spec::value_type::Type::TensorType(tt)) => Some(tt),
                    _ => None,
                })
                .ok_or_else(|| {
                    MilError::Protobuf(
                        "raw byte tensor value has no accompanying tensor type".to_string(),
                    )
                })?;

            let dtype = convert_data_type(tt.data_type)?;
            let shape: Vec<usize> = tt
                .dimensions
                .iter()
                .filter_map(|d| match &d.dimension {
                    Some(mil_spec::dimension::Dimension::Constant(c)) => Some(c.size as usize),
                    _ => None,
                })
                .collect();

            Ok(Value::Tensor {
                data: TensorData::Inline(b.values.clone()),
                shape,
                dtype,
            })
        }
        None => Err(MilError::Protobuf(
            "tensor value has no payload".to_string(),
        )),
    }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

fn convert_value_type_to_value(vt: &mil_spec::ValueType) -> Result<Value> {
    use mil_spec::value_type;

    match &vt.r#type {
        Some(value_type::Type::TensorType(tt)) => Ok(Value::Type(convert_tensor_type(tt)?)),
        Some(_) => Err(MilError::UnsupportedOp(
            "only tensor value types are currently supported".to_string(),
        )),
        None => Err(MilError::Protobuf(
            "value type has no inner type set".to_string(),
        )),
    }
}

fn convert_named_value_type_to_tensor(nvt: &mil_spec::NamedValueType) -> Result<TensorType> {
    let vt = nvt
        .r#type
        .as_ref()
        .ok_or_else(|| MilError::Protobuf(format!("named value '{}' has no type", nvt.name)))?;

    match &vt.r#type {
        Some(mil_spec::value_type::Type::TensorType(tt)) => convert_tensor_type(tt),
        Some(_) => Err(MilError::UnsupportedOp(format!(
            "named value '{}': only tensor types are supported as function inputs",
            nvt.name
        ))),
        None => Err(MilError::Protobuf(format!(
            "named value '{}': value type has no inner type",
            nvt.name
        ))),
    }
}

/// Convert a proto `TensorType` into an IR `TensorType`.
fn convert_tensor_type(proto: &mil_spec::TensorType) -> Result<TensorType> {
    let scalar_type = convert_data_type(proto.data_type)?;

    let shape = proto
        .dimensions
        .iter()
        .map(convert_dimension)
        .collect::<Vec<_>>();

    Ok(TensorType::with_dynamic_shape(scalar_type, shape))
}

fn convert_dimension(dim: &mil_spec::Dimension) -> Option<usize> {
    use mil_spec::dimension;

    match &dim.dimension {
        Some(dimension::Dimension::Constant(c)) => Some(c.size as usize),
        Some(dimension::Dimension::Unknown(_)) | None => None,
    }
}

/// Map proto `DataType` i32 → IR `ScalarType`.
fn convert_data_type(dt: i32) -> Result<ScalarType> {
    match dt {
        x if x == mil_spec::DataType::Float16 as i32 => Ok(ScalarType::Float16),
        x if x == mil_spec::DataType::Float32 as i32 => Ok(ScalarType::Float32),
        x if x == mil_spec::DataType::Float64 as i32 => Ok(ScalarType::Float64),
        x if x == mil_spec::DataType::Int8 as i32 => Ok(ScalarType::Int8),
        x if x == mil_spec::DataType::Int16 as i32 => Ok(ScalarType::Int16),
        x if x == mil_spec::DataType::Int32 as i32 => Ok(ScalarType::Int32),
        x if x == mil_spec::DataType::Int64 as i32 => Ok(ScalarType::Int64),
        x if x == mil_spec::DataType::Uint8 as i32 => Ok(ScalarType::UInt8),
        x if x == mil_spec::DataType::Uint16 as i32 => Ok(ScalarType::UInt16),
        x if x == mil_spec::DataType::Uint32 as i32 => Ok(ScalarType::UInt32),
        x if x == mil_spec::DataType::Uint64 as i32 => Ok(ScalarType::UInt64),
        x if x == mil_spec::DataType::Bool as i32 => Ok(ScalarType::Bool),
        _ => Err(MilError::UnsupportedOp(format!(
            "unsupported proto data type: {}",
            dt
        ))),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a minimal proto Model wrapping an MlProgram.
    fn make_ml_program_model(program: mil_spec::Program) -> Model {
        Model {
            specification_version: 7,
            description: None,
            is_updatable: false,
            r#type: Some(model::Type::MlProgram(program)),
        }
    }

    #[test]
    fn rejects_non_ml_program_model() {
        use crate::proto::specification::NeuralNetwork;

        let model = Model {
            specification_version: 1,
            description: None,
            is_updatable: false,
            r#type: Some(model::Type::NeuralNetwork(NeuralNetwork::default())),
        };
        let err = model_to_program(&model).unwrap_err();
        assert!(
            err.to_string().contains("different type"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_model_with_no_type() {
        let model = Model {
            specification_version: 1,
            description: None,
            is_updatable: false,
            r#type: None,
        };
        let err = model_to_program(&model).unwrap_err();
        assert!(
            err.to_string().contains("no type"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn converts_empty_program() {
        let proto_program = mil_spec::Program {
            version: 1,
            functions: HashMap::new(),
            doc_string: String::new(),
            attributes: HashMap::new(),
        };
        let model = make_ml_program_model(proto_program);
        let program = model_to_program(&model).unwrap();

        assert_eq!(program.version, "1");
        assert!(program.functions.is_empty());
    }

    #[test]
    fn converts_program_with_simple_function() {
        use mil_spec::*;

        // Build a proto function with one input and a block with one relu op.
        let input_tensor = TensorType {
            data_type: DataType::Float32 as i32,
            rank: 4,
            dimensions: vec![
                Dimension {
                    dimension: Some(dimension::Dimension::Constant(
                        dimension::ConstantDimension { size: 1 },
                    )),
                },
                Dimension {
                    dimension: Some(dimension::Dimension::Constant(
                        dimension::ConstantDimension { size: 3 },
                    )),
                },
                Dimension {
                    dimension: Some(dimension::Dimension::Constant(
                        dimension::ConstantDimension { size: 224 },
                    )),
                },
                Dimension {
                    dimension: Some(dimension::Dimension::Constant(
                        dimension::ConstantDimension { size: 224 },
                    )),
                },
            ],
            attributes: HashMap::new(),
        };

        let relu_op = Operation {
            r#type: "relu".to_string(),
            inputs: {
                let mut m = HashMap::new();
                m.insert(
                    "x".to_string(),
                    Argument {
                        arguments: vec![argument::Binding {
                            binding: Some(argument::binding::Binding::Name("input".to_string())),
                        }],
                    },
                );
                m
            },
            outputs: vec![NamedValueType {
                name: "relu_out".to_string(),
                r#type: Some(ValueType {
                    r#type: Some(value_type::Type::TensorType(input_tensor.clone())),
                }),
            }],
            blocks: vec![],
            attributes: HashMap::new(),
        };

        let block = Block {
            inputs: vec![],
            outputs: vec!["relu_out".to_string()],
            operations: vec![relu_op],
            attributes: HashMap::new(),
        };

        let function = Function {
            inputs: vec![NamedValueType {
                name: "input".to_string(),
                r#type: Some(ValueType {
                    r#type: Some(value_type::Type::TensorType(input_tensor)),
                }),
            }],
            opset: "CoreML6".to_string(),
            block_specializations: {
                let mut m = HashMap::new();
                m.insert("CoreML6".to_string(), block);
                m
            },
            attributes: HashMap::new(),
        };

        let proto_program = mil_spec::Program {
            version: 1,
            functions: {
                let mut m = HashMap::new();
                m.insert("main".to_string(), function);
                m
            },
            doc_string: String::new(),
            attributes: HashMap::new(),
        };

        let model = make_ml_program_model(proto_program);
        let program = model_to_program(&model).unwrap();

        assert_eq!(program.version, "1");
        assert_eq!(program.functions.len(), 1);

        let main = &program.functions["main"];
        assert_eq!(main.name, "main");
        assert_eq!(main.inputs.len(), 1);
        assert_eq!(main.inputs[0].0, "input");
        assert_eq!(main.inputs[0].1.scalar_type, ScalarType::Float32);
        assert_eq!(
            main.inputs[0].1.shape,
            vec![Some(1), Some(3), Some(224), Some(224)]
        );

        assert_eq!(main.body.operations.len(), 1);
        let op = &main.body.operations[0];
        assert_eq!(op.op_type, "relu");
        assert_eq!(op.outputs, vec!["relu_out"]);
        assert!(matches!(op.inputs.get("x"), Some(crate::ir::Value::Reference(r)) if r == "input"));

        assert_eq!(main.body.outputs, vec!["relu_out"]);
    }

    #[test]
    fn converts_dynamic_dimensions() {
        let tt = mil_spec::TensorType {
            data_type: mil_spec::DataType::Float16 as i32,
            rank: 3,
            dimensions: vec![
                mil_spec::Dimension {
                    dimension: Some(mil_spec::dimension::Dimension::Unknown(
                        mil_spec::dimension::UnknownDimension { variadic: false },
                    )),
                },
                mil_spec::Dimension {
                    dimension: Some(mil_spec::dimension::Dimension::Constant(
                        mil_spec::dimension::ConstantDimension { size: 768 },
                    )),
                },
                mil_spec::Dimension { dimension: None },
            ],
            attributes: HashMap::new(),
        };

        let ir_tt = convert_tensor_type(&tt).unwrap();
        assert_eq!(ir_tt.scalar_type, ScalarType::Float16);
        assert_eq!(ir_tt.shape, vec![None, Some(768), None]);
        assert!(!ir_tt.is_static());
    }

    #[test]
    fn converts_immediate_scalar_values() {
        use mil_spec::{TensorValue, tensor_value};

        // Float scalar
        let float_tv = TensorValue {
            value: Some(tensor_value::Value::Floats(tensor_value::RepeatedFloats {
                values: vec![3.14],
            })),
        };
        let v = convert_tensor_value(&float_tv, None).unwrap();
        assert!(matches!(v, Value::Float(f) if (f - 3.14_f64).abs() < 1e-5));

        // Int scalar
        let int_tv = TensorValue {
            value: Some(tensor_value::Value::Ints(tensor_value::RepeatedInts {
                values: vec![42],
            })),
        };
        let v = convert_tensor_value(&int_tv, None).unwrap();
        assert!(matches!(v, Value::Int(42)));

        // Bool scalar
        let bool_tv = TensorValue {
            value: Some(tensor_value::Value::Bools(tensor_value::RepeatedBools {
                values: vec![true],
            })),
        };
        let v = convert_tensor_value(&bool_tv, None).unwrap();
        assert!(matches!(v, Value::Bool(true)));
    }

    #[test]
    fn converts_immediate_vector_values() {
        use mil_spec::{TensorValue, tensor_value};

        let float_tv = TensorValue {
            value: Some(tensor_value::Value::Ints(tensor_value::RepeatedInts {
                values: vec![1, 2, 3],
            })),
        };
        let v = convert_tensor_value(&float_tv, None).unwrap();
        match v {
            Value::List(items) => {
                assert_eq!(items.len(), 3);
                assert!(matches!(items[0], Value::Int(1)));
                assert!(matches!(items[1], Value::Int(2)));
                assert!(matches!(items[2], Value::Int(3)));
            }
            other => panic!("expected List, got {other:?}"),
        }
    }
}
