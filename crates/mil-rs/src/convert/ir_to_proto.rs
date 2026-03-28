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
    let inputs = func
        .inputs
        .iter()
        .map(|(name, tt)| mil_spec::NamedValueType {
            name: name.clone(),
            r#type: Some(mil_spec::ValueType {
                r#type: Some(mil_spec::value_type::Type::TensorType(convert_tensor_type(
                    tt,
                ))),
            }),
        })
        .collect();

    let block = convert_block(&func.body)?;

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

fn convert_block(block: &Block) -> Result<mil_spec::Block> {
    let operations = block
        .operations
        .iter()
        .map(convert_operation)
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

fn convert_operation(op: &Operation) -> Result<mil_spec::Operation> {
    let mut inputs = HashMap::new();
    for (param, value) in &op.inputs {
        inputs.insert(param.clone(), convert_value_to_argument(value)?);
    }

    let outputs = op
        .outputs
        .iter()
        .map(|name| mil_spec::NamedValueType {
            name: name.clone(),
            // Output type info is not preserved in the IR — omit it.
            r#type: None,
        })
        .collect();

    let mut attributes = HashMap::new();
    for (attr_name, attr_val) in &op.attributes {
        attributes.insert(attr_name.clone(), convert_value_to_proto(attr_val)?);
    }

    Ok(mil_spec::Operation {
        r#type: op.op_type.clone(),
        inputs,
        outputs,
        blocks: vec![],
        attributes,
    })
}

// ---------------------------------------------------------------------------
// Arguments & Values
// ---------------------------------------------------------------------------

/// Convert an IR `Value` into a proto `Argument`.
fn convert_value_to_argument(value: &Value) -> Result<mil_spec::Argument> {
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
                None,
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
                None,
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
                None,
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
                None,
            )
        }
        Value::List(items) => {
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
                None,
            )
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
        Value::Tensor { data, .. } => {
            // Raw tensor data — encode as a bytes immediate.
            let tv = mil_spec::TensorValue {
                value: Some(mil_spec::tensor_value::Value::Bytes(
                    mil_spec::tensor_value::RepeatedBytes {
                        values: data.clone(),
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
