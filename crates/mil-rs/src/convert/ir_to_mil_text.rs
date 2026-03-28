//! Convert ironmill MIL IR to ANE MIL text format.
//!
//! This emitter produces the text-based MIL format consumed by `_ANECompiler`,
//! as reverse-engineered by Orion and maderix/ANE.
//!
//! Key differences from the CoreML protobuf emitter (`ir_to_proto`):
//! - Text format (UTF-8) instead of binary protobuf
//! - Tensor layout must be `[1, C, 1, S]`
//! - Weights use BLOBFILE with byte offsets
//! - I/O variables must be alphabetically ordered
//! - Const bools must be named const refs, not inline literals

use std::collections::HashMap;
use std::fmt::Write;

use crate::error::Result;
use crate::ir::{Function, Operation, Program, ScalarType, TensorType, Value};

/// Configuration for the MIL text emitter.
pub struct MilTextConfig {
    /// Tensor layout convention for the ANE.
    pub layout: AneLayout,
    /// Whether to allow int4 types in the output.
    pub enable_int4: bool,
}

impl Default for MilTextConfig {
    fn default() -> Self {
        Self {
            layout: AneLayout::Nchw,
            enable_int4: false,
        }
    }
}

/// ANE tensor layout convention.
pub enum AneLayout {
    /// `[1, C, 1, S]` — default ANE layout.
    Nchw,
}

/// A weight blob entry extracted during emission.
pub struct WeightBlobEntry {
    /// Name of the weight (matches the const op name).
    pub name: String,
    /// Raw tensor data bytes.
    pub data: Vec<u8>,
    /// Byte offset within the weight blob file.
    pub offset: u64,
    /// Element data type.
    pub dtype: ScalarType,
    /// Tensor dimensions.
    pub shape: Vec<usize>,
}

/// Convert a [`Program`] to ANE MIL text format.
///
/// Returns the MIL text string and a list of [`WeightBlobEntry`]s that must be
/// written to the companion `weights.blob` file.
pub fn program_to_mil_text(
    program: &Program,
    config: &MilTextConfig,
) -> Result<(String, Vec<WeightBlobEntry>)> {
    let mut emitter = MilTextEmitter::new(config);
    emitter.emit_program(program)?;
    Ok((emitter.output, emitter.weight_entries))
}

// Internal attributes that should not appear in the text output.
const SKIP_ATTRIBUTES: &[&str] = &["compute_unit", "fused_activation", "has_fused_bn"];

/// Accumulates MIL text output and tracks weight blob entries.
struct MilTextEmitter<'a> {
    config: &'a MilTextConfig,
    output: String,
    weight_entries: Vec<WeightBlobEntry>,
    /// Running byte offset into the weight blob file.
    weight_offset: u64,
    /// Maps original variable names → emitted names for I/O renaming.
    rename_map: HashMap<String, String>,
}

impl<'a> MilTextEmitter<'a> {
    fn new(config: &'a MilTextConfig) -> Self {
        Self {
            config,
            output: String::new(),
            weight_entries: Vec::new(),
            weight_offset: 0,
            rename_map: HashMap::new(),
        }
    }

    fn emit_program(&mut self, program: &Program) -> Result<()> {
        let version = if program.version.contains('.') {
            // Strip patch version: "1.0.0" → "1.0"
            let parts: Vec<&str> = program.version.splitn(3, '.').collect();
            if parts.len() >= 2 {
                format!("{}.{}", parts[0], parts[1])
            } else {
                program.version.clone()
            }
        } else {
            program.version.clone()
        };

        writeln!(self.output, "program({version})").unwrap();

        for func in program.functions.values() {
            self.emit_function(func)?;
        }

        Ok(())
    }

    fn emit_function(&mut self, func: &Function) -> Result<()> {
        // Build I/O rename maps for alphabetical ordering.
        self.rename_map.clear();
        for (i, (name, _ty)) in func.inputs.iter().enumerate() {
            let new_name = format!("a_input{i}");
            self.rename_map.insert(name.clone(), new_name);
        }
        for (i, output_name) in func.body.outputs.iter().enumerate() {
            let new_name = format!("z_output{i}");
            self.rename_map.insert(output_name.clone(), new_name);
        }

        // Emit function header.
        writeln!(self.output, "func {}(", func.name).unwrap();

        for (i, (name, ty)) in func.inputs.iter().enumerate() {
            let emitted_name = self.resolve_name(name);
            let type_str = self.format_tensor_type(ty);
            if i > 0 {
                writeln!(self.output, ",").unwrap();
            }
            write!(self.output, "    %{emitted_name}: {type_str}").unwrap();
        }
        writeln!(self.output).unwrap();

        // Emit return type.
        let return_types: Vec<String> = func
            .body
            .outputs
            .iter()
            .filter_map(|out_name| self.find_output_type(func, out_name))
            .map(|ty| self.format_tensor_type(&ty))
            .collect();
        let return_str = return_types.join(", ");
        writeln!(self.output, ") -> ({return_str}) {{").unwrap();

        // Emit operations.
        for op in &func.body.operations {
            self.emit_operation(op)?;
        }

        // Emit block outputs.
        let outputs: Vec<String> = func
            .body
            .outputs
            .iter()
            .map(|name| format!("%{}", self.resolve_name(name)))
            .collect();
        writeln!(self.output, "}} -> ({})", outputs.join(", ")).unwrap();

        Ok(())
    }

    fn emit_operation(&mut self, op: &Operation) -> Result<()> {
        if op.op_type == "const" {
            return self.emit_const_op(op);
        }

        let output_name = op
            .outputs
            .first()
            .map(|n| self.resolve_name(n))
            .unwrap_or_else(|| op.name.clone());

        let mut params = Vec::new();
        // Sort input keys for deterministic output.
        let mut input_keys: Vec<&String> = op.inputs.keys().collect();
        input_keys.sort();
        for key in input_keys {
            let val = &op.inputs[key];
            params.push(format!("{}={}", key, self.format_value(val)));
        }

        // Emit non-skipped attributes.
        let mut attr_keys: Vec<&String> = op
            .attributes
            .keys()
            .filter(|k| !SKIP_ATTRIBUTES.contains(&k.as_str()))
            .collect();
        attr_keys.sort();
        for key in attr_keys {
            let val = &op.attributes[key];
            params.push(format!("{}={}", key, self.format_value(val)));
        }

        writeln!(
            self.output,
            "    %{output_name} = {}({})",
            op.op_type,
            params.join(", ")
        )
        .unwrap();

        Ok(())
    }

    fn emit_const_op(&mut self, op: &Operation) -> Result<()> {
        let output_name = op
            .outputs
            .first()
            .map(|n| self.resolve_name(n))
            .unwrap_or_else(|| op.name.clone());

        // Check if this is a weight tensor or a scalar const.
        if let Some(Value::Tensor { data, shape, dtype }) = op.inputs.get("val") {
            // Weight tensor → collect as blob entry.
            let offset = self.weight_offset;
            let aligned_len = align_up(data.len() as u64, 64);
            self.weight_entries.push(WeightBlobEntry {
                name: op.name.clone(),
                data: data.clone(),
                offset,
                dtype: *dtype,
                shape: shape.clone(),
            });
            self.weight_offset += aligned_len;

            let mut params = Vec::new();
            params.push(format!("name=\"{}\"", op.name));
            params.push(format!(
                "val=blob(file=\"weights.blob\", offset=uint64({offset}))"
            ));
            writeln!(
                self.output,
                "    %{output_name} = const({params})",
                params = params.join(", ")
            )
            .unwrap();
        } else if let Some(val) = op.inputs.get("val") {
            // Scalar const → emit inline.
            let mut params = Vec::new();
            // Include name attribute for const ops if present.
            if op.attributes.contains_key("name") || !op.name.is_empty() {
                // Only include name for weight-like consts that have meaningful names.
            }
            params.push(format!("val={}", self.format_value(val)));
            writeln!(
                self.output,
                "    %{output_name} = const({params})",
                params = params.join(", ")
            )
            .unwrap();
        } else {
            // No val input — emit an empty const (shouldn't happen normally).
            writeln!(self.output, "    %{output_name} = const()").unwrap();
        }

        Ok(())
    }

    /// Format a [`Value`] for MIL text output.
    fn format_value(&self, value: &Value) -> String {
        match value {
            Value::Reference(name) => {
                let resolved = self.resolve_name(name);
                format!("%{resolved}")
            }
            Value::Int(n) => n.to_string(),
            Value::Float(f) => format_float(*f),
            Value::Bool(b) => if *b { "true" } else { "false" }.to_string(),
            Value::String(s) => format!("\"{s}\""),
            Value::List(items) => {
                let parts: Vec<String> = items.iter().map(|v| self.format_value(v)).collect();
                format!("[{}]", parts.join(", "))
            }
            Value::Type(ty) => self.format_tensor_type(ty),
            Value::Tensor { .. } => {
                // Inline tensor data shouldn't appear in regular value positions;
                // this is handled by emit_const_op. Fallback representation:
                "<tensor>".to_string()
            }
        }
    }

    /// Format a [`TensorType`] as MIL text: `tensor<fp16, [1, 768, 1, 32]>`.
    fn format_tensor_type(&self, ty: &TensorType) -> String {
        let dtype = format_scalar_type(ty.scalar_type, self.config.enable_int4);
        let dims: Vec<String> = ty
            .shape
            .iter()
            .map(|d: &Option<usize>| match d {
                Some(n) => n.to_string(),
                None => "?".to_string(),
            })
            .collect();
        format!("tensor<{dtype}, [{}]>", dims.join(", "))
    }

    /// Resolve a variable name through the rename map.
    fn resolve_name(&self, name: &str) -> String {
        self.rename_map
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_string())
    }

    /// Find the output type for a given output name by searching operations.
    fn find_output_type(&self, func: &Function, output_name: &str) -> Option<TensorType> {
        for op in &func.body.operations {
            for (i, out) in op.outputs.iter().enumerate() {
                if out == output_name {
                    if let Some(Some(ty)) = op.output_types.get(i) {
                        return Some(ty.clone());
                    }
                }
            }
        }
        None
    }
}

/// Format a [`ScalarType`] as a MIL type string.
fn format_scalar_type(st: ScalarType, _enable_int4: bool) -> &'static str {
    match st {
        ScalarType::Float16 => "fp16",
        ScalarType::Float32 => "fp32",
        ScalarType::Float64 => "fp64",
        ScalarType::Int8 => "int8",
        ScalarType::Int16 => "int16",
        ScalarType::Int32 => "int32",
        ScalarType::Int64 => "int64",
        ScalarType::UInt8 => "uint8",
        ScalarType::UInt16 => "uint16",
        ScalarType::UInt32 => "uint32",
        ScalarType::UInt64 => "uint64",
        ScalarType::Bool => "bool",
    }
}

/// Format a float with enough precision to round-trip.
fn format_float(f: f64) -> String {
    if f == f.floor() && f.abs() < 1e15 {
        // Emit as integer-like when there's no fractional part.
        format!("{f:.1}")
    } else {
        format!("{f}")
    }
}

/// Align a byte count up to the given alignment boundary.
fn align_up(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Block, Function, Operation};

    /// Helper: build a simple program with one function.
    fn make_program(func: Function) -> Program {
        let mut program = Program::new("1.0.0");
        program.add_function(func);
        program
    }

    /// Helper: build a const operation for a scalar value.
    fn scalar_const(name: &str, val: Value) -> Operation {
        Operation::new("const", name)
            .with_input("val", val)
            .with_output(name)
    }

    /// Helper: build a const operation for a weight tensor.
    fn weight_const(name: &str, data: Vec<u8>, shape: Vec<usize>, dtype: ScalarType) -> Operation {
        let mut op = Operation::new("const", name).with_output(name);
        op.inputs
            .insert("val".to_string(), Value::Tensor { data, shape, dtype });
        op
    }

    #[test]
    fn mil_text_simple_program() {
        // %z = add(x=%a, y=%b) with two fp16 inputs.
        let input_ty = TensorType::new(ScalarType::Float16, vec![1, 768, 1, 32]);
        let mut func = Function::new("main")
            .with_input("x", input_ty.clone())
            .with_input("y", input_ty.clone());

        let add_op = Operation::new("add", "add_0")
            .with_input("x", Value::Reference("x".to_string()))
            .with_input("y", Value::Reference("y".to_string()))
            .with_output("z");

        func.body.add_op(add_op);
        func.body.outputs.push("z".to_string());

        // Attach output type to the add op so return type is emitted.
        func.body.operations[0].output_types = vec![Some(input_ty.clone())];

        let program = make_program(func);
        let config = MilTextConfig::default();
        let (text, weights) = program_to_mil_text(&program, &config).unwrap();

        assert!(text.contains("program(1.0)"));
        assert!(text.contains("func main("));
        assert!(text.contains("%a_input0: tensor<fp16, [1, 768, 1, 32]>"));
        assert!(text.contains("%a_input1: tensor<fp16, [1, 768, 1, 32]>"));
        assert!(text.contains("-> (tensor<fp16, [1, 768, 1, 32]>)"));
        assert!(text.contains("%z_output0 = add(x=%a_input0, y=%a_input1)"));
        assert!(text.contains("} -> (%z_output0)"));
        assert!(weights.is_empty());
    }

    #[test]
    fn mil_text_variable_naming() {
        // Verify that inputs are renamed to a_input{N} and outputs to z_output{N}.
        let ty = TensorType::new(ScalarType::Float32, vec![1, 3, 1, 224]);
        let mut func = Function::new("main")
            .with_input("image", ty.clone())
            .with_input("mask", ty.clone());

        let relu_op = Operation::new("relu", "relu_0")
            .with_input("x", Value::Reference("image".to_string()))
            .with_output("relu_out");

        let sigmoid_op = Operation::new("sigmoid", "sig_0")
            .with_input("x", Value::Reference("mask".to_string()))
            .with_output("sig_out");

        func.body.add_op(relu_op);
        func.body.add_op(sigmoid_op);
        func.body.outputs.push("relu_out".to_string());
        func.body.outputs.push("sig_out".to_string());

        func.body.operations[0].output_types = vec![Some(ty.clone())];
        func.body.operations[1].output_types = vec![Some(ty.clone())];

        let program = make_program(func);
        let config = MilTextConfig::default();
        let (text, _) = program_to_mil_text(&program, &config).unwrap();

        // Inputs renamed.
        assert!(text.contains("%a_input0: tensor<fp32, [1, 3, 1, 224]>"));
        assert!(text.contains("%a_input1: tensor<fp32, [1, 3, 1, 224]>"));

        // Outputs renamed.
        assert!(text.contains("} -> (%z_output0, %z_output1)"));

        // References to inputs are also renamed.
        assert!(text.contains("x=%a_input0"));
        assert!(text.contains("x=%a_input1"));
    }

    #[test]
    fn mil_text_weight_offsets() {
        let ty = TensorType::new(ScalarType::Float16, vec![1, 64, 1, 32]);
        let mut func = Function::new("main").with_input("x", ty.clone());

        // Two weight tensors of different sizes.
        let w1_data = vec![0u8; 100];
        let w2_data = vec![0u8; 200];
        let w1 = weight_const(
            "layer0_w",
            w1_data.clone(),
            vec![64, 32],
            ScalarType::Float16,
        );
        let w2 = weight_const(
            "layer1_w",
            w2_data.clone(),
            vec![64, 64],
            ScalarType::Float16,
        );

        let add_op = Operation::new("add", "add_0")
            .with_input("x", Value::Reference("x".to_string()))
            .with_input("y", Value::Reference("layer0_w".to_string()))
            .with_output("out");

        func.body.add_op(w1);
        func.body.add_op(w2);
        func.body.add_op(add_op);
        func.body.outputs.push("out".to_string());
        func.body.operations[2].output_types = vec![Some(ty.clone())];

        let program = make_program(func);
        let config = MilTextConfig::default();
        let (text, weights) = program_to_mil_text(&program, &config).unwrap();

        // Two weight entries collected.
        assert_eq!(weights.len(), 2);
        assert_eq!(weights[0].name, "layer0_w");
        assert_eq!(weights[0].offset, 0);
        assert_eq!(weights[0].data.len(), 100);
        // Second weight is aligned to 64 bytes: align_up(100, 64) = 128.
        assert_eq!(weights[1].name, "layer1_w");
        assert_eq!(weights[1].offset, 128);
        assert_eq!(weights[1].data.len(), 200);

        // Blob references in text.
        assert!(text.contains("offset=uint64(0)"));
        assert!(text.contains("offset=uint64(128)"));
        assert!(text.contains("file=\"weights.blob\""));
    }

    #[test]
    fn mil_text_const_bool() {
        let ty = TensorType::new(ScalarType::Float16, vec![1, 64, 1, 32]);
        let mut func = Function::new("main").with_input("x", ty.clone());

        // Bool consts must be emitted as named const refs.
        let bool_const = scalar_const("my_flag", Value::Bool(true));
        let noop = Operation::new("identity", "id_0")
            .with_input("x", Value::Reference("x".to_string()))
            .with_output("out");

        func.body.add_op(bool_const);
        func.body.add_op(noop);
        func.body.outputs.push("out".to_string());
        func.body.operations[1].output_types = vec![Some(ty.clone())];

        let program = make_program(func);
        let config = MilTextConfig::default();
        let (text, _) = program_to_mil_text(&program, &config).unwrap();

        // Bool const emitted as named ref, not inline.
        assert!(text.contains("%my_flag = const(val=true)"));
    }

    #[test]
    fn mil_text_layout_1c1s() {
        // Verify tensor type formatting with the NCHW layout convention.
        let ty = TensorType::new(ScalarType::Float16, vec![1, 768, 1, 32]);
        let mut func = Function::new("main").with_input("x", ty.clone());

        let relu = Operation::new("relu", "relu_0")
            .with_input("x", Value::Reference("x".to_string()))
            .with_output("y");
        func.body.add_op(relu);
        func.body.outputs.push("y".to_string());
        func.body.operations[0].output_types = vec![Some(ty.clone())];

        let program = make_program(func);
        let config = MilTextConfig::default();
        let (text, _) = program_to_mil_text(&program, &config).unwrap();

        assert!(text.contains("tensor<fp16, [1, 768, 1, 32]>"));
    }

    #[test]
    fn mil_text_round_trip() {
        // Complex program: conv → add bias → relu pipeline.
        let input_ty = TensorType::new(ScalarType::Float16, vec![1, 768, 1, 32]);
        let out_ty = TensorType::new(ScalarType::Float16, vec![1, 256, 1, 32]);
        let mut func = Function::new("main")
            .with_input("input0", input_ty.clone())
            .with_input(
                "input1",
                TensorType::new(ScalarType::Float16, vec![1, 1, 1, 32]),
            );

        // Weight const.
        let w = weight_const(
            "layer0_w",
            vec![0u8; 512],
            vec![256, 768, 1, 1],
            ScalarType::Float16,
        );
        // Bias const.
        let bias = weight_const("layer0_b", vec![0u8; 64], vec![256], ScalarType::Float16);
        // String attribute const.
        let pad_const = scalar_const("pad_mode", Value::String("valid".to_string()));
        // Conv op.
        let mut conv = Operation::new("conv", "conv_0")
            .with_input("x", Value::Reference("input0".to_string()))
            .with_input("weight", Value::Reference("layer0_w".to_string()))
            .with_input("pad_type", Value::String("valid".to_string()))
            .with_output("conv_out");
        conv.output_types = vec![Some(out_ty.clone())];
        // Add op.
        let mut add = Operation::new("add", "add_0")
            .with_input("x", Value::Reference("conv_out".to_string()))
            .with_input("y", Value::Reference("layer0_b".to_string()))
            .with_output("add_out");
        add.output_types = vec![Some(out_ty.clone())];
        // Relu op.
        let mut relu = Operation::new("relu", "relu_0")
            .with_input("x", Value::Reference("add_out".to_string()))
            .with_output("final_out");
        relu.output_types = vec![Some(out_ty.clone())];

        func.body.add_op(w);
        func.body.add_op(bias);
        func.body.add_op(pad_const);
        func.body.add_op(conv);
        func.body.add_op(add);
        func.body.add_op(relu);
        func.body.outputs.push("final_out".to_string());

        let program = make_program(func);
        let config = MilTextConfig::default();
        let (text, weights) = program_to_mil_text(&program, &config).unwrap();

        // Structure checks.
        assert!(text.starts_with("program(1.0)\n"));
        assert!(text.contains("func main("));
        assert!(text.contains("%a_input0: tensor<fp16, [1, 768, 1, 32]>"));
        assert!(text.contains("%a_input1: tensor<fp16, [1, 1, 1, 32]>"));
        assert!(text.contains("-> (tensor<fp16, [1, 256, 1, 32]>)"));

        // Weight blob refs.
        assert!(text.contains("blob(file=\"weights.blob\", offset=uint64(0))"));
        assert!(text.contains("blob(file=\"weights.blob\", offset=uint64(512))"));

        // Regular ops reference renamed inputs.
        assert!(text.contains("x=%a_input0"));
        assert!(text.contains("pad_type=\"valid\""));
        assert!(text.contains("relu(x="));
        assert!(text.contains("} -> (%z_output0)"));

        // Weights collected.
        assert_eq!(weights.len(), 2);
        assert_eq!(weights[0].name, "layer0_w");
        assert_eq!(weights[1].name, "layer0_b");
    }

    #[test]
    fn mil_text_scalar_types() {
        assert_eq!(format_scalar_type(ScalarType::Float16, false), "fp16");
        assert_eq!(format_scalar_type(ScalarType::Float32, false), "fp32");
        assert_eq!(format_scalar_type(ScalarType::Float64, false), "fp64");
        assert_eq!(format_scalar_type(ScalarType::Int8, false), "int8");
        assert_eq!(format_scalar_type(ScalarType::Int32, false), "int32");
        assert_eq!(format_scalar_type(ScalarType::Bool, false), "bool");
        assert_eq!(format_scalar_type(ScalarType::UInt8, false), "uint8");
    }

    #[test]
    fn mil_text_format_values() {
        let config = MilTextConfig::default();
        let emitter = MilTextEmitter::new(&config);

        assert_eq!(emitter.format_value(&Value::Int(42)), "42");
        assert_eq!(emitter.format_value(&Value::Float(3.14)), "3.14");
        assert_eq!(emitter.format_value(&Value::Float(1.0)), "1.0");
        assert_eq!(emitter.format_value(&Value::Bool(true)), "true");
        assert_eq!(emitter.format_value(&Value::Bool(false)), "false");
        assert_eq!(
            emitter.format_value(&Value::String("hello".into())),
            "\"hello\""
        );
        assert_eq!(
            emitter.format_value(&Value::List(vec![Value::Int(1), Value::Int(2)])),
            "[1, 2]"
        );
        assert_eq!(
            emitter.format_value(&Value::Reference("foo".into())),
            "%foo"
        );
    }

    #[test]
    fn mil_text_align_up() {
        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(64, 64), 64);
        assert_eq!(align_up(65, 64), 128);
        assert_eq!(align_up(100, 64), 128);
        assert_eq!(align_up(128, 64), 128);
    }

    #[test]
    fn mil_text_dynamic_dims() {
        let ty =
            TensorType::with_dynamic_shape(ScalarType::Float16, vec![Some(1), None, Some(1), None]);
        let config = MilTextConfig::default();
        let emitter = MilTextEmitter::new(&config);
        assert_eq!(
            emitter.format_tensor_type(&ty),
            "tensor<fp16, [1, ?, 1, ?]>"
        );
    }
}
