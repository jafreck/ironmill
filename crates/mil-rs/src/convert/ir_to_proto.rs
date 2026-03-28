//! Convert MIL IR `Program` → protobuf `Model`.

use std::collections::HashMap;

use crate::error::Result;
use crate::ir::ScalarType;
use crate::ir::{Block, Function, Operation, Program, TensorType, Value};
use crate::proto::mil_spec;
use crate::proto::specification::{Model, model};

/// Convert a MIL IR [`Program`] back into a protobuf [`Model`].
///
/// Creates an ML Program model wrapping the converted program.
/// `spec_version` sets the `specification_version` field on the
/// resulting [`Model`] (use 7 or 8 for modern CoreML specs).
pub fn program_to_model(program: &Program, spec_version: i32) -> Result<Model> {
    let proto_program = convert_program(program)?;

    Ok(Model {
        specification_version: spec_version,
        description: None,
        is_updatable: false,
        r#type: Some(model::Type::MlProgram(proto_program)),
    })
}

// ---------------------------------------------------------------------------
// Program / Function / Block
// ---------------------------------------------------------------------------

fn convert_program(program: &Program) -> Result<mil_spec::Program> {
    let version: i64 = program.version.parse().unwrap_or(1);

    let mut functions = HashMap::new();
    for (name, func) in &program.functions {
        functions.insert(name.clone(), convert_function(func)?);
    }

    Ok(mil_spec::Program {
        version,
        functions,
        doc_string: String::new(),
        attributes: HashMap::new(),
    })
}

fn convert_function(func: &Function) -> Result<mil_spec::Function> {
    // Build a type map from function inputs so we can propagate output types.
    let mut type_map: HashMap<String, mil_spec::ValueType> = HashMap::new();
    let inputs = func
        .inputs
        .iter()
        .map(|(name, tt)| {
            let vt = mil_spec::ValueType {
                r#type: Some(mil_spec::value_type::Type::TensorType(convert_tensor_type(
                    tt,
                ))),
            };
            type_map.insert(name.clone(), vt.clone());
            mil_spec::NamedValueType {
                name: name.clone(),
                r#type: Some(vt),
            }
        })
        .collect();

    let block = convert_block(&func.body, &mut type_map)?;

    let opset = "CoreML6".to_string();
    let mut block_specializations = HashMap::new();
    block_specializations.insert(opset.clone(), block);

    Ok(mil_spec::Function {
        inputs,
        opset,
        block_specializations,
        attributes: HashMap::new(),
    })
}

fn convert_block(
    block: &Block,
    type_map: &mut HashMap<String, mil_spec::ValueType>,
) -> Result<mil_spec::Block> {
    let operations = block
        .operations
        .iter()
        .map(|op| convert_operation(op, type_map))
        .collect::<Result<Vec<_>>>()?;

    Ok(mil_spec::Block {
        inputs: vec![],
        outputs: block.outputs.clone(),
        operations,
        attributes: HashMap::new(),
    })
}

// ---------------------------------------------------------------------------
// Operation
// ---------------------------------------------------------------------------

fn convert_operation(
    op: &Operation,
    type_map: &mut HashMap<String, mil_spec::ValueType>,
) -> Result<mil_spec::Operation> {
    let mut inputs = HashMap::new();
    for (param, value) in &op.inputs {
        inputs.insert(param.clone(), convert_value_to_argument(value)?);
    }

    // Determine output type for this operation: prefer stored types, fall back
    // to inference from the type_map.
    let inferred_type = infer_output_type(op, type_map);

    let outputs: Vec<mil_spec::NamedValueType> = op
        .outputs
        .iter()
        .enumerate()
        .map(|(i, name)| {
            // Use the stored type for this specific output if available,
            // otherwise fall back to the inferred type.
            let vt = op
                .output_types
                .get(i)
                .and_then(|ot| ot.as_ref())
                .map(|tt| mil_spec::ValueType {
                    r#type: Some(mil_spec::value_type::Type::TensorType(convert_tensor_type(
                        tt,
                    ))),
                })
                .or_else(|| inferred_type.clone());

            // Register in the type map for downstream ops.
            if let Some(ref v) = vt {
                type_map.insert(name.clone(), v.clone());
            }

            mil_spec::NamedValueType {
                name: name.clone(),
                r#type: vt,
            }
        })
        .collect();

    let mut attributes = HashMap::new();
    for (attr_name, attr_val) in &op.attributes {
        if op.op_type == "const" {
            // For const ops, all attributes stay as proto attributes.
            attributes.insert(attr_name.clone(), convert_value_to_proto(attr_val)?);
            continue;
        }
        // Skip internal-only attributes from optimization passes.
        if matches!(
            attr_name.as_str(),
            "fused_activation" | "has_fused_bn" | "original_op" | "kernel_shape"
        ) {
            continue;
        }
        // Non-const: MIL expects parameters as proto inputs.
        inputs.insert(attr_name.clone(), convert_value_to_argument(attr_val)?);
    }

    Ok(mil_spec::Operation {
        r#type: op.op_type.clone(),
        inputs,
        outputs,
        blocks: vec![],
        attributes,
    })
}

/// Infer the output type of an operation from its inputs and the type map.
fn infer_output_type(
    op: &Operation,
    type_map: &HashMap<String, mil_spec::ValueType>,
) -> Option<mil_spec::ValueType> {
    // For const ops, derive the type from the val attribute/input.
    if op.op_type == "const" {
        let val = op.inputs.get("val").or_else(|| op.attributes.get("val"));
        return val.and_then(value_type_for);
    }

    // Try to resolve a type from a reference Value.
    let resolve = |v: &Value| -> Option<mil_spec::ValueType> {
        match v {
            Value::Reference(name) => type_map.get(name).cloned(),
            Value::List(items) => {
                // For list inputs (e.g., concat's "values"), use the
                // first reference's type.
                items.iter().find_map(|item| {
                    if let Value::Reference(name) = item {
                        type_map.get(name).cloned()
                    } else {
                        None
                    }
                })
            }
            _ => None,
        }
    };

    // For most ops, the output type matches the primary input's type.
    let primary_params = ["x", "data", "input", "values"];
    let first_input_type = primary_params
        .iter()
        .filter_map(|&param| op.inputs.get(param))
        .find_map(resolve);

    if first_input_type.is_some() {
        return first_input_type;
    }

    // Fallback: try any input.
    op.inputs.values().find_map(resolve)
}

// ---------------------------------------------------------------------------
// Arguments & Values
// ---------------------------------------------------------------------------

/// Convert an IR `Value` into a proto `Argument`.
fn convert_value_to_argument(value: &Value) -> Result<mil_spec::Argument> {
    // A list of references becomes multiple bindings (e.g., concat's "values").
    if let Value::List(items) = value {
        if items.iter().all(|v| matches!(v, Value::Reference(_))) {
            let bindings = items
                .iter()
                .map(|v| {
                    let Value::Reference(name) = v else {
                        unreachable!()
                    };
                    mil_spec::argument::Binding {
                        binding: Some(mil_spec::argument::binding::Binding::Name(name.clone())),
                    }
                })
                .collect();
            return Ok(mil_spec::Argument {
                arguments: bindings,
            });
        }
    }

    let binding = match value {
        Value::Reference(name) => mil_spec::argument::binding::Binding::Name(name.clone()),
        other => {
            let proto_val = convert_value_to_proto(other)?;
            mil_spec::argument::binding::Binding::Value(proto_val)
        }
    };

    Ok(mil_spec::Argument {
        arguments: vec![mil_spec::argument::Binding {
            binding: Some(binding),
        }],
    })
}

/// Build a scalar (rank-0) `ValueType` for the given `DataType`.
fn scalar_value_type(dt: mil_spec::DataType) -> mil_spec::ValueType {
    mil_spec::ValueType {
        r#type: Some(mil_spec::value_type::Type::TensorType(
            mil_spec::TensorType {
                data_type: dt as i32,
                rank: 0,
                dimensions: vec![],
                attributes: HashMap::new(),
            },
        )),
    }
}

/// Derive the `ValueType` that describes an IR `Value`.
fn value_type_for(value: &Value) -> Option<mil_spec::ValueType> {
    match value {
        Value::Int(_) => Some(scalar_value_type(mil_spec::DataType::Int32)),
        Value::Float(_) => Some(scalar_value_type(mil_spec::DataType::Float32)),
        Value::Bool(_) => Some(scalar_value_type(mil_spec::DataType::Bool)),
        Value::String(_) => Some(scalar_value_type(mil_spec::DataType::String)),
        Value::Tensor { shape, dtype, .. } => {
            let data_type = convert_scalar_type(*dtype) as i32;
            let dimensions = shape
                .iter()
                .map(|&d| mil_spec::Dimension {
                    dimension: Some(mil_spec::dimension::Dimension::Constant(
                        mil_spec::dimension::ConstantDimension { size: d as u64 },
                    )),
                })
                .collect::<Vec<_>>();
            Some(mil_spec::ValueType {
                r#type: Some(mil_spec::value_type::Type::TensorType(
                    mil_spec::TensorType {
                        data_type,
                        rank: dimensions.len() as i64,
                        dimensions,
                        attributes: HashMap::new(),
                    },
                )),
            })
        }
        Value::List(items) => {
            // Infer element type from the first item; fall back to int32.
            let elem_type = items
                .first()
                .and_then(value_type_for)
                .unwrap_or_else(|| scalar_value_type(mil_spec::DataType::Int32));
            Some(mil_spec::ValueType {
                r#type: Some(mil_spec::value_type::Type::ListType(Box::new(
                    mil_spec::ListType {
                        r#type: Some(Box::new(elem_type)),
                        length: Some(mil_spec::Dimension {
                            dimension: Some(mil_spec::dimension::Dimension::Constant(
                                mil_spec::dimension::ConstantDimension {
                                    size: items.len() as u64,
                                },
                            )),
                        }),
                    },
                ))),
            })
        }
        // Type-only values and references have their type handled separately.
        Value::Type(_) | Value::Reference(_) => None,
    }
}

/// Try to encode a list of scalar `Value`s as a 1D `TensorValue`.
///
/// Returns `Some((tensor_value, data_type))` if all items are the same
/// scalar type (Int, Float, or Bool), `None` otherwise.
fn try_list_as_tensor(items: &[Value]) -> Option<(mil_spec::TensorValue, mil_spec::DataType)> {
    if items.is_empty() {
        return None;
    }

    // Check if all items are ints.
    if items.iter().all(|v| matches!(v, Value::Int(_))) {
        let ints: Vec<i32> = items
            .iter()
            .map(|v| {
                let Value::Int(i) = v else { unreachable!() };
                *i as i32
            })
            .collect();
        return Some((
            mil_spec::TensorValue {
                value: Some(mil_spec::tensor_value::Value::Ints(
                    mil_spec::tensor_value::RepeatedInts { values: ints },
                )),
            },
            mil_spec::DataType::Int32,
        ));
    }

    // Check if all items are floats.
    if items.iter().all(|v| matches!(v, Value::Float(_))) {
        let floats: Vec<f32> = items
            .iter()
            .map(|v| {
                let Value::Float(f) = v else { unreachable!() };
                *f as f32
            })
            .collect();
        return Some((
            mil_spec::TensorValue {
                value: Some(mil_spec::tensor_value::Value::Floats(
                    mil_spec::tensor_value::RepeatedFloats { values: floats },
                )),
            },
            mil_spec::DataType::Float32,
        ));
    }

    // Check if all items are bools.
    if items.iter().all(|v| matches!(v, Value::Bool(_))) {
        let bools: Vec<bool> = items
            .iter()
            .map(|v| {
                let Value::Bool(b) = v else { unreachable!() };
                *b
            })
            .collect();
        return Some((
            mil_spec::TensorValue {
                value: Some(mil_spec::tensor_value::Value::Bools(
                    mil_spec::tensor_value::RepeatedBools { values: bools },
                )),
            },
            mil_spec::DataType::Bool,
        ));
    }

    None
}

/// Encode a `Value::Tensor` into the appropriate typed `TensorValue`.
///
/// Uses typed repeated fields (floats, ints, etc.) so that `coremlcompiler`
/// can verify element counts against the declared tensor type.
fn convert_tensor_data(value: &Value) -> mil_spec::TensorValue {
    let Value::Tensor { data, dtype, .. } = value else {
        unreachable!("convert_tensor_data called with non-Tensor value");
    };

    let tv_value = match dtype {
        ScalarType::Float32 => {
            let floats: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            mil_spec::tensor_value::Value::Floats(mil_spec::tensor_value::RepeatedFloats {
                values: floats,
            })
        }
        ScalarType::Float64 => {
            let doubles: Vec<f64> = data
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            mil_spec::tensor_value::Value::Doubles(mil_spec::tensor_value::RepeatedDoubles {
                values: doubles,
            })
        }
        ScalarType::Int32 | ScalarType::UInt32 => {
            let ints: Vec<i32> = data
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            mil_spec::tensor_value::Value::Ints(mil_spec::tensor_value::RepeatedInts {
                values: ints,
            })
        }
        ScalarType::Int64 | ScalarType::UInt64 => {
            let longs: Vec<i64> = data
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            mil_spec::tensor_value::Value::LongInts(mil_spec::tensor_value::RepeatedLongInts {
                values: longs,
            })
        }
        ScalarType::Bool => {
            let bools: Vec<bool> = data.iter().map(|&b| b != 0).collect();
            mil_spec::tensor_value::Value::Bools(mil_spec::tensor_value::RepeatedBools {
                values: bools,
            })
        }
        // For types without a dedicated repeated field (fp16, int8, etc.)
        // fall back to raw bytes.
        _ => mil_spec::tensor_value::Value::Bytes(mil_spec::tensor_value::RepeatedBytes {
            values: data.clone(),
        }),
    };

    mil_spec::TensorValue {
        value: Some(tv_value),
    }
}

/// Convert an IR `Value` into a proto `Value`.
fn convert_value_to_proto(value: &Value) -> Result<mil_spec::Value> {
    use mil_spec::value;

    let (proto_value, proto_type) = match value {
        Value::Reference(name) => {
            // References shouldn't normally appear as standalone proto Values,
            // but we handle it gracefully by encoding as a string immediate.
            let tv = mil_spec::TensorValue {
                value: Some(mil_spec::tensor_value::Value::Strings(
                    mil_spec::tensor_value::RepeatedStrings {
                        values: vec![name.clone()],
                    },
                )),
            };
            (
                Some(value::Value::ImmediateValue(value::ImmediateValue {
                    value: Some(value::immediate_value::Value::Tensor(tv)),
                })),
                None,
            )
        }
        Value::Int(v) => {
            let tv = mil_spec::TensorValue {
                value: Some(mil_spec::tensor_value::Value::Ints(
                    mil_spec::tensor_value::RepeatedInts {
                        values: vec![*v as i32],
                    },
                )),
            };
            (
                Some(value::Value::ImmediateValue(value::ImmediateValue {
                    value: Some(value::immediate_value::Value::Tensor(tv)),
                })),
                value_type_for(value),
            )
        }
        Value::Float(v) => {
            let tv = mil_spec::TensorValue {
                value: Some(mil_spec::tensor_value::Value::Floats(
                    mil_spec::tensor_value::RepeatedFloats {
                        values: vec![*v as f32],
                    },
                )),
            };
            (
                Some(value::Value::ImmediateValue(value::ImmediateValue {
                    value: Some(value::immediate_value::Value::Tensor(tv)),
                })),
                value_type_for(value),
            )
        }
        Value::Bool(v) => {
            let tv = mil_spec::TensorValue {
                value: Some(mil_spec::tensor_value::Value::Bools(
                    mil_spec::tensor_value::RepeatedBools { values: vec![*v] },
                )),
            };
            (
                Some(value::Value::ImmediateValue(value::ImmediateValue {
                    value: Some(value::immediate_value::Value::Tensor(tv)),
                })),
                value_type_for(value),
            )
        }
        Value::String(s) => {
            let tv = mil_spec::TensorValue {
                value: Some(mil_spec::tensor_value::Value::Strings(
                    mil_spec::tensor_value::RepeatedStrings {
                        values: vec![s.clone()],
                    },
                )),
            };
            (
                Some(value::Value::ImmediateValue(value::ImmediateValue {
                    value: Some(value::immediate_value::Value::Tensor(tv)),
                })),
                value_type_for(value),
            )
        }
        Value::List(items) => {
            // In MIL, homogeneous lists of scalars are represented as 1D
            // tensors. Only use a proto ListValue for heterogeneous/nested
            // lists or lists of references.
            if let Some((tv, dt)) = try_list_as_tensor(items) {
                let len = items.len();
                let tensor_type = mil_spec::ValueType {
                    r#type: Some(mil_spec::value_type::Type::TensorType(
                        mil_spec::TensorType {
                            data_type: dt as i32,
                            rank: 1,
                            dimensions: vec![mil_spec::Dimension {
                                dimension: Some(mil_spec::dimension::Dimension::Constant(
                                    mil_spec::dimension::ConstantDimension { size: len as u64 },
                                )),
                            }],
                            attributes: HashMap::new(),
                        },
                    )),
                };
                (
                    Some(value::Value::ImmediateValue(value::ImmediateValue {
                        value: Some(value::immediate_value::Value::Tensor(tv)),
                    })),
                    Some(tensor_type),
                )
            } else {
                let proto_items = items
                    .iter()
                    .map(convert_value_to_proto)
                    .collect::<Result<Vec<_>>>()?;
                (
                    Some(value::Value::ImmediateValue(value::ImmediateValue {
                        value: Some(value::immediate_value::Value::List(mil_spec::ListValue {
                            values: proto_items,
                        })),
                    })),
                    value_type_for(value),
                )
            }
        }
        Value::Type(tt) => {
            // Type-only value — no immediate payload, just the type field.
            (
                None,
                Some(mil_spec::ValueType {
                    r#type: Some(mil_spec::value_type::Type::TensorType(convert_tensor_type(
                        tt,
                    ))),
                }),
            )
        }
        Value::Tensor { .. } => {
            // Encode tensor using the typed storage that matches its dtype,
            // so coremlcompiler can verify element counts correctly.
            let tv = convert_tensor_data(value);
            (
                Some(value::Value::ImmediateValue(value::ImmediateValue {
                    value: Some(value::immediate_value::Value::Tensor(tv)),
                })),
                value_type_for(value),
            )
        }
    };

    Ok(mil_spec::Value {
        doc_string: String::new(),
        r#type: proto_type,
        value: proto_value,
    })
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Convert an IR `TensorType` into a proto `TensorType`.
fn convert_tensor_type(tt: &TensorType) -> mil_spec::TensorType {
    let data_type = convert_scalar_type(tt.scalar_type) as i32;

    let dimensions = tt
        .shape
        .iter()
        .map(|dim| match dim {
            Some(size) => mil_spec::Dimension {
                dimension: Some(mil_spec::dimension::Dimension::Constant(
                    mil_spec::dimension::ConstantDimension { size: *size as u64 },
                )),
            },
            None => mil_spec::Dimension {
                dimension: Some(mil_spec::dimension::Dimension::Unknown(
                    mil_spec::dimension::UnknownDimension { variadic: false },
                )),
            },
        })
        .collect::<Vec<_>>();

    mil_spec::TensorType {
        data_type,
        rank: dimensions.len() as i64,
        dimensions,
        attributes: HashMap::new(),
    }
}

/// Map IR `ScalarType` → proto `DataType`.
fn convert_scalar_type(st: ScalarType) -> mil_spec::DataType {
    match st {
        ScalarType::Float16 => mil_spec::DataType::Float16,
        ScalarType::Float32 => mil_spec::DataType::Float32,
        ScalarType::Float64 => mil_spec::DataType::Float64,
        ScalarType::Int8 => mil_spec::DataType::Int8,
        ScalarType::Int16 => mil_spec::DataType::Int16,
        ScalarType::Int32 => mil_spec::DataType::Int32,
        ScalarType::Int64 => mil_spec::DataType::Int64,
        ScalarType::UInt8 => mil_spec::DataType::Uint8,
        ScalarType::UInt16 => mil_spec::DataType::Uint16,
        ScalarType::UInt32 => mil_spec::DataType::Uint32,
        ScalarType::UInt64 => mil_spec::DataType::Uint64,
        ScalarType::Bool => mil_spec::DataType::Bool,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::model_to_program;

    #[test]
    fn round_trip_empty_program() {
        let program = Program::new("1");
        let model = program_to_model(&program, 7).unwrap();
        let recovered = model_to_program(&model).unwrap();

        assert_eq!(recovered.version, "1");
        assert!(recovered.functions.is_empty());
    }

    #[test]
    fn round_trip_simple_program() {
        let input_ty = TensorType::new(ScalarType::Float32, vec![1, 3, 224, 224]);
        let relu = Operation::new("relu", "relu_out")
            .with_input("x", Value::Reference("input".to_string()))
            .with_output("relu_out");

        let mut block = Block::new();
        block.add_op(relu);
        block.outputs.push("relu_out".into());

        let func = Function {
            name: "main".to_string(),
            inputs: vec![("input".to_string(), input_ty.clone())],
            body: block,
        };

        let mut program = Program::new("1");
        program.add_function(func);

        // IR → Proto → IR round-trip
        let model = program_to_model(&program, 7).unwrap();
        let recovered = model_to_program(&model).unwrap();

        assert_eq!(recovered.version, "1");
        assert_eq!(recovered.functions.len(), 1);

        let main = &recovered.functions["main"];
        assert_eq!(main.inputs.len(), 1);
        assert_eq!(main.inputs[0].0, "input");
        assert_eq!(main.inputs[0].1, input_ty);

        assert_eq!(main.body.operations.len(), 1);
        assert_eq!(main.body.operations[0].op_type, "relu");
        assert_eq!(main.body.outputs, vec!["relu_out"]);

        // The input reference should survive the round-trip.
        assert!(matches!(
            main.body.operations[0].inputs.get("x"),
            Some(Value::Reference(r)) if r == "input"
        ));
    }

    #[test]
    fn round_trip_with_dynamic_dimensions() {
        let input_ty =
            TensorType::with_dynamic_shape(ScalarType::Float16, vec![None, Some(768), None]);
        let func = Function {
            name: "encode".to_string(),
            inputs: vec![("tokens".to_string(), input_ty.clone())],
            body: Block::new(),
        };

        let mut program = Program::new("1");
        program.add_function(func);

        let model = program_to_model(&program, 8).unwrap();
        let recovered = model_to_program(&model).unwrap();

        let enc = &recovered.functions["encode"];
        assert_eq!(enc.inputs[0].1, input_ty);
        assert!(!enc.inputs[0].1.is_static());
    }

    #[test]
    fn round_trip_scalar_attributes() {
        let op = Operation::new("const", "c0")
            .with_attr("val", Value::Int(42))
            .with_attr("eps", Value::Float(1e-5))
            .with_attr("training", Value::Bool(false))
            .with_output("c0");

        let mut block = Block::new();
        block.add_op(op);
        block.outputs.push("c0".into());

        let func = Function {
            name: "main".to_string(),
            inputs: vec![],
            body: block,
        };

        let mut program = Program::new("1");
        program.add_function(func);

        let model = program_to_model(&program, 7).unwrap();
        let recovered = model_to_program(&model).unwrap();

        // const ops keep their params as proto attributes, so they
        // round-trip back as attributes in the IR.
        let attrs = &recovered.functions["main"].body.operations[0].attributes;
        assert!(matches!(attrs.get("val"), Some(Value::Int(42))));
        assert!(matches!(attrs.get("training"), Some(Value::Bool(false))));
        // Float round-trips through f32, so check approximate equality.
        if let Some(Value::Float(f)) = attrs.get("eps") {
            assert!((*f - 1e-5_f64).abs() < 1e-7, "eps = {f}");
        } else {
            panic!("expected Float for eps, got {:?}", attrs.get("eps"));
        }
    }

    #[test]
    fn model_specification_version_preserved() {
        let program = Program::new("1");
        let model = program_to_model(&program, 8).unwrap();
        assert_eq!(model.specification_version, 8);
    }
}
