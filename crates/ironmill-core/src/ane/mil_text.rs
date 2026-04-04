//! Convert ironmill MIL IR to ANE MIL text format.
//!
//! This emitter produces the text-based MIL format consumed by `_ANECompiler`,
//! matching the syntax from the Orion project's `core/mil_builder.m`.
//!
//! Key format features:
//! - `program(1.3)` with `[buildInfo = ...]` attribute block
//! - `func main<ios18>(TYPE name, ...) { ... } -> (%out);`
//! - `TYPE %name = op(args)[name=string("...")];` for operations
//! - Typed const values: `fp16()`, `int32()`, `bool()`, `string()`
//! - `BLOBFILE(path=string("@model_path/..."), offset=uint64(N))` for weights

use std::collections::HashMap;
use std::fmt::Write;

use mil_rs::error::{MilError, Result};
use mil_rs::ir::{Function, Operation, Program, ScalarType, TensorType, Value};

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
    /// MIL text path (e.g., `@model_path/weights/layer0_w.bin`).
    pub path: String,
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
/// written to the companion weight blob files.
pub fn program_to_mil_text(
    program: &Program,
    config: &MilTextConfig,
) -> Result<(String, Vec<WeightBlobEntry>)> {
    let mut emitter = MilTextEmitter::new(config);
    emitter.emit_program(program)?;
    Ok((emitter.output, emitter.weight_entries))
}

/// Byte offset into each per-weight BLOBFILE. The ANE weight blob file format
/// reserves a 64-byte header before the raw tensor data begins.
const WEIGHT_BLOB_HEADER_BYTES: u64 = 64;

/// MIL spec version emitted in `program(...)`.
const MIL_SPEC_VERSION: &str = "1.3";

/// CoreML compiler MIL component version (`coremlc-component-MIL`).
const COREMLC_MIL_VERSION: &str = "3510.2.1";

/// CoreML compiler version (`coremlc-version`).
const COREMLC_VERSION: &str = "3505.4.1";

/// CoreML Tools version (`coremltools-version`).
const COREMLTOOLS_VERSION: &str = "9.0";

// Internal attributes that should not appear in the text output.
const SKIP_ATTRIBUTES: &[&str] = &["compute_unit", "fused_activation", "has_fused_bn"];

/// Accumulates MIL text output and tracks weight blob entries.
struct MilTextEmitter<'a> {
    config: &'a MilTextConfig,
    output: String,
    weight_entries: Vec<WeightBlobEntry>,
    /// Maps original variable names → emitted names for I/O renaming.
    rename_map: HashMap<String, String>,
    /// Maps variable names → their tensor types (for output type inference).
    type_map: HashMap<String, TensorType>,
}

impl<'a> MilTextEmitter<'a> {
    fn new(config: &'a MilTextConfig) -> Self {
        Self {
            config,
            output: String::new(),
            weight_entries: Vec::new(),
            rename_map: HashMap::new(),
            type_map: HashMap::new(),
        }
    }

    fn emit_program(&mut self, program: &Program) -> Result<()> {
        // Always emit the ANE-compatible MIL spec version.
        let _ = &program.version; // acknowledge the field
        writeln!(self.output, "program({})", MIL_SPEC_VERSION).unwrap();

        // Required buildInfo attribute block.
        write!(
            self.output,
            "[buildInfo = dict<string, string>({{{{\"coremlc-component-MIL\", \"{COREMLC_MIL_VERSION}\"}}, \
             {{\"coremlc-version\", \"{COREMLC_VERSION}\"}}, \
             {{\"coremltools-component-milinternal\", \"\"}}, \
             {{\"coremltools-version\", \"{COREMLTOOLS_VERSION}\"}}}})]",
        )
        .unwrap();
        self.output.push('\n');

        writeln!(self.output, "{{").unwrap();

        for func in program.functions.values() {
            self.emit_function(func)?;
        }

        writeln!(self.output, "}}").unwrap();
        Ok(())
    }

    fn emit_function(&mut self, func: &Function) -> Result<()> {
        // Build I/O rename maps for alphabetical ordering.
        self.rename_map.clear();
        self.type_map.clear();
        for (i, (name, ty)) in func.inputs.iter().enumerate() {
            let new_name = format!("a_input{i}");
            self.rename_map.insert(name.clone(), new_name);
            self.type_map.insert(name.clone(), ty.clone());
        }
        for (i, output_name) in func.body.outputs.iter().enumerate() {
            let new_name = format!("z_output{i}");
            self.rename_map.insert(output_name.clone(), new_name);
        }

        // Emit: func main<ios18>(TYPE name, TYPE name, ...) {
        write!(self.output, "    func {}<ios18>(", func.name).unwrap();
        for (i, (name, ty)) in func.inputs.iter().enumerate() {
            if i > 0 {
                write!(self.output, ", ").unwrap();
            }
            let emitted_name = self.resolve_name(name);
            let type_str = self.format_tensor_type(ty);
            write!(self.output, "{type_str} {emitted_name}").unwrap();
        }
        writeln!(self.output, ") {{").unwrap();

        // Emit operations.
        for op in &func.body.operations {
            self.emit_operation(op)?;
        }

        // Emit block return: } -> (out1, out2);
        let outputs: Vec<String> = func
            .body
            .outputs
            .iter()
            .map(|name| self.resolve_name(name))
            .collect();
        writeln!(self.output, "    }} -> ({});", outputs.join(", ")).unwrap();

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

        // Determine output type. If not explicitly stored, try to infer from
        // input types (most element-wise ops preserve the type of their first
        // tensor input).
        let output_type = op
            .output_types
            .first()
            .and_then(|t| t.as_ref())
            .cloned()
            .or_else(|| self.infer_output_type_from_inputs(op));
        let type_str = match &output_type {
            Some(ty) => self.format_tensor_type(ty),
            None => {
                eprintln!(
                    "error: could not infer output type for op '{}' (type: {})",
                    op.name, op.op_type
                );
                return Err(MilError::Validation(format!(
                    "cannot infer output type for op '{}' (type: {})",
                    op.name, op.op_type
                )));
            }
        };

        // Register this output's type for downstream inference.
        if let (Some(out_name), Some(ty)) = (op.outputs.first(), &output_type) {
            self.type_map.insert(out_name.clone(), ty.clone());
        }

        let mut params = Vec::new();
        // Emit input keys with "x" first (ANE expects the primary data
        // input before other parameters), then remaining keys sorted.
        let mut input_keys: Vec<&String> = op.inputs.keys().collect();
        input_keys.sort_by(|a, b| {
            // "x" sorts first, then alphabetical.
            match (a.as_str(), b.as_str()) {
                ("x", _) => std::cmp::Ordering::Less,
                (_, "x") => std::cmp::Ordering::Greater,
                _ => a.cmp(b),
            }
        });
        for key in input_keys {
            let val = &op.inputs[key];
            params.push(format!("{}={}", key, self.format_value(val)));
        }

        // Emit non-skipped attributes as params.
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

        // TYPE %name = op(params)[name=string("name")];
        writeln!(
            self.output,
            "        {type_str} {output_name} = {op_type}({params})[name=string(\"{output_name}\")];",
            op_type = op.op_type,
            params = params.join(", "),
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
        // ONNX-converted models put the value in `attributes["val"]`,
        // while hand-built programs use `inputs["val"]`.
        let val = op.inputs.get("val").or_else(|| op.attributes.get("val"));
        if let Some(Value::Tensor { data, shape, dtype }) = val {
            // Register the type for downstream ops.
            if let Some(out_name) = op.outputs.first() {
                self.type_map
                    .insert(out_name.clone(), TensorType::new(*dtype, shape.clone()));
            }

            // Integer tensors (axes, shapes, strides) are small parameter
            // metadata — emit inline, not as BLOBFILE. ANE expects small
            // parameter tensors to be inline values.
            let is_int_param = matches!(
                dtype,
                ScalarType::Int8
                    | ScalarType::Int16
                    | ScalarType::Int32
                    | ScalarType::Int64
                    | ScalarType::UInt8
                    | ScalarType::UInt16
                    | ScalarType::UInt32
                    | ScalarType::UInt64
            );

            if is_int_param {
                // Emit inline: tensor<int32, [N]>([v1, v2, ...])
                let type_str = self.format_tensor_type_from(shape, *dtype);
                let values = format_tensor_elements(
                    data.as_bytes().expect("tensor not materialized"),
                    *dtype,
                );
                let num_elements: usize = shape.iter().product();

                writeln!(
                    self.output,
                    "        {type_str} {output_name} = const()[name=string(\"{output_name}\"), val=tensor<{dtype_str}, [{num_elements}]>([{values}])];",
                    dtype_str = format_scalar_type(*dtype, self.config.enable_int4),
                )
                .unwrap();
            } else {
                // FP weight tensor → collect as blob entry.
                let weight_path = format!("@model_path/weights/{}.bin", op.name);
                self.weight_entries.push(WeightBlobEntry {
                    name: op.name.clone(),
                    path: weight_path.clone(),
                    data: data.as_bytes().expect("tensor not materialized").to_vec(),
                    offset: WEIGHT_BLOB_HEADER_BYTES,
                    dtype: *dtype,
                    shape: shape.clone(),
                });

                let emit_shape = to_ane_weight_shape(shape);
                let type_str = self.format_tensor_type_from(&emit_shape, *dtype);

                writeln!(
                    self.output,
                    "        {type_str} {output_name} = const()[name=string(\"{output_name}\"), val={type_str}(BLOBFILE(path=string(\"{weight_path}\"), offset=uint64({WEIGHT_BLOB_HEADER_BYTES})))];",
                )
                .unwrap();
            }
        } else if let Some(val) = op.inputs.get("val").or_else(|| op.attributes.get("val")) {
            // Scalar/list const → emit with typed value.
            let (type_str, val_str) = self.format_typed_const_value(val);

            // TYPE %name = const()[name=string("name"), val=TYPED_VALUE];
            writeln!(
                self.output,
                "        {type_str} {output_name} = const()[name=string(\"{output_name}\"), val={val_str}];",
            )
            .unwrap();
        } else {
            // No val input — emit an empty const.
            writeln!(
                self.output,
                "        fp16 {output_name} = const()[name=string(\"{output_name}\")];",
            )
            .unwrap();
        }

        Ok(())
    }

    /// Format a [`Value`] for operation input parameters.
    fn format_value(&self, value: &Value) -> String {
        match value {
            Value::Reference(name) => self.resolve_name(name),
            Value::Int(n) => format!("int32({n})"),
            Value::Float(f) => format!("fp16({})", format_float(*f)),
            Value::Bool(b) => {
                format!("bool({})", if *b { "true" } else { "false" })
            }
            Value::String(s) => format!("string(\"{s}\")"),
            Value::List(items) if items.iter().all(|v| matches!(v, Value::Int(_))) => {
                let vals: Vec<String> = items
                    .iter()
                    .map(|v| match v {
                        Value::Int(n) => n.to_string(),
                        _ => "0".into(),
                    })
                    .collect();
                let n = vals.len();
                format!("tensor<int32, [{n}]>([{}])", vals.join(","))
            }
            Value::List(items) => {
                let parts: Vec<String> = items.iter().map(|v| self.format_value(v)).collect();
                format!("[{}]", parts.join(", "))
            }
            Value::Type(ty) => self.format_tensor_type(ty),
            Value::Tensor { data, shape, dtype } => {
                // Emit as inline typed tensor literal with 1-D shape.
                // Op arguments (e.g., reshape's `shape`, reduce's `axes`)
                // are always 1-D vectors. Flatten multi-dim shapes to [N].
                let dtype_str = format_scalar_type(*dtype, self.config.enable_int4);
                let num_elements: usize = shape.iter().product();
                let type_str = format!("tensor<{dtype_str}, [{num_elements}]>");

                // Decode the raw bytes into element values.
                let values = format_tensor_elements(
                    data.as_bytes().expect("tensor not materialized"),
                    *dtype,
                );
                format!("{type_str}([{values}])")
            }
            _ => panic!("unsupported Value variant: {value:?}"),
        }
    }

    /// Format a value with its type prefix for const `val=` attributes.
    ///
    /// Returns `(type_string, typed_value_string)`.
    fn format_typed_const_value(&self, val: &Value) -> (String, String) {
        match val {
            Value::Float(f) => ("fp16".into(), format!("fp16({})", format_float(*f))),
            Value::Int(n) => ("int32".into(), format!("int32({n})")),
            Value::Bool(b) => (
                "bool".into(),
                format!("bool({})", if *b { "true" } else { "false" }),
            ),
            Value::String(s) => ("string".into(), format!("string(\"{s}\")")),
            Value::List(items) if items.iter().all(|v| matches!(v, Value::Int(_))) => {
                let vals: Vec<String> = items
                    .iter()
                    .map(|v| match v {
                        Value::Int(n) => n.to_string(),
                        _ => "0".into(),
                    })
                    .collect();
                let n = vals.len();
                (
                    format!("tensor<int32, [{n}]>"),
                    format!("tensor<int32, [{n}]>([{}])", vals.join(",")),
                )
            }
            _ => ("fp16".into(), self.format_value(val)),
        }
    }

    /// Format a [`TensorType`] as MIL text: `tensor<fp16, [1,768,1,32]>`.
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
        format!("tensor<{dtype}, [{}]>", dims.join(","))
    }

    /// Format a tensor type from raw shape and dtype (for weight consts).
    fn format_tensor_type_from(&self, shape: &[usize], dtype: ScalarType) -> String {
        let dtype_str = format_scalar_type(dtype, self.config.enable_int4);
        let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        format!("tensor<{dtype_str}, [{}]>", dims.join(","))
    }

    /// Resolve a variable name through the rename map.
    fn resolve_name(&self, name: &str) -> String {
        self.rename_map
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_string())
    }

    /// Infer output type from input references using the type map.
    /// Most element-wise ops (add, mul, relu, etc.) preserve the type
    /// of their first tensor input.
    fn infer_output_type_from_inputs(&self, op: &Operation) -> Option<TensorType> {
        // First, check if any input is a reference we can look up in the type map.
        // Prefer "x" input (the primary operand for most ops).
        if let Some(Value::Reference(name)) = op.inputs.get("x") {
            if let Some(ty) = self.type_map.get(name) {
                // For reduce ops, collapse the reduced axis.
                if op.op_type.starts_with("reduce_") {
                    return Some(self.reduce_output_type(ty, op));
                }
                return Some(ty.clone());
            }
        }
        // Try any reference input.
        for val in op.inputs.values() {
            if let Value::Reference(name) = val {
                if let Some(ty) = self.type_map.get(name) {
                    return Some(ty.clone());
                }
            }
        }
        // Check for inline Tensor values.
        for val in op.inputs.values() {
            if let Value::Tensor { shape, dtype, .. } = val {
                return Some(TensorType::new(*dtype, shape.clone()));
            }
        }
        None
    }

    /// Compute the output type for a reduce op by collapsing the reduced
    /// axes to size 1 (when keep_dims=true, which is the default).
    fn reduce_output_type(&self, input_ty: &TensorType, op: &Operation) -> TensorType {
        // Extract axes from inputs (const reference) or attributes.
        let axes = self.extract_reduce_axes(op);
        if axes.is_empty() {
            return input_ty.clone();
        }

        let mut shape = input_ty.shape.clone();
        let rank = shape.len() as i64;
        for &axis in &axes {
            let normalized = if axis < 0 {
                let n = rank + axis;
                if n < 0 {
                    eprintln!(
                        "warning: reduce op '{}' has axis {} out of range for rank {}; \
                         returning input type unchanged",
                        op.name, axis, rank
                    );
                    return input_ty.clone();
                }
                n as usize
            } else {
                axis as usize
            };
            if normalized >= shape.len() {
                eprintln!(
                    "warning: reduce op '{}' has axis {} (normalized {}) out of range for rank {}; \
                     returning input type unchanged",
                    op.name, axis, normalized, rank
                );
                return input_ty.clone();
            }
            shape[normalized] = Some(1);
        }

        TensorType::with_dynamic_shape(input_ty.scalar_type, shape)
    }

    /// Extract the axes values from a reduce op, resolving const references.
    fn extract_reduce_axes(&self, op: &Operation) -> Vec<i64> {
        // Check inputs first, then attributes.
        let axes_val = op.inputs.get("axes").or_else(|| op.attributes.get("axes"));

        match axes_val {
            Some(Value::Reference(name)) => {
                // Look up the const op's value in our type_map... but type_map
                // only stores types, not values. We can't resolve the actual
                // axes value from a const reference at emit time. Fall back
                // to not adjusting the shape.
                // However, we stored the axes data in the weight entries.
                // For simplicity, just check common patterns.
                let _ = name;
                vec![]
            }
            Some(Value::List(items)) => items
                .iter()
                .filter_map(|v| match v {
                    Value::Int(n) => Some(*n),
                    _ => None,
                })
                .collect(),
            Some(Value::Tensor { data, dtype, .. }) if *dtype == ScalarType::Int32 => data
                .as_bytes()
                .expect("tensor not materialized")
                .chunks_exact(4)
                .map(|c: &[u8]| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                .collect(),
            _ => vec![],
        }
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
        _ => panic!("unsupported scalar type: {st:?}"),
    }
}

/// Format a float with enough precision to round-trip.
fn format_float(f: f64) -> String {
    if f == f.floor() && f.abs() < 1e15 {
        format!("{f:.1}")
    } else {
        format!("{f}")
    }
}

/// Convert a weight tensor shape to ANE-compatible 4D.
///
/// ANE requires all tensor type declarations in MIL text to be 4D.
/// Weight shapes like `[1024]` or `[1024,1024]` need to be padded
/// to `[1,1024,1,1]` or `[1,1024,1,1024]`.
fn to_ane_weight_shape(shape: &[usize]) -> Vec<usize> {
    match shape.len() {
        0 => vec![1, 1, 1, 1],
        1 => vec![1, shape[0], 1, 1],
        2 => vec![1, shape[0], 1, shape[1]],
        3 => vec![1, shape[0], shape[1], shape[2]],
        4 => shape.to_vec(),
        _ => {
            // 5D+ — collapse trailing dims.
            let mut s = vec![1usize; 4];
            s[1] = shape[0];
            s[3] = shape[1..].iter().product();
            s
        }
    }
}

/// Format raw tensor bytes as a comma-separated list of element values.
///
/// Used for inline tensor literals in op arguments (e.g., reshape shape).
fn format_tensor_elements(data: &[u8], dtype: ScalarType) -> String {
    let elem_size = dtype.byte_size();
    debug_assert!(
        elem_size <= 1 || data.len() % elem_size == 0,
        "data length {} is not divisible by dtype size {} for {:?}",
        data.len(),
        elem_size,
        dtype,
    );

    match dtype {
        ScalarType::Int32 => data
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]).to_string())
            .collect::<Vec<_>>()
            .join(","),
        ScalarType::Int64 => data
            .chunks_exact(8)
            .map(|b| {
                i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]).to_string()
            })
            .collect::<Vec<_>>()
            .join(","),
        ScalarType::Float32 => data
            .chunks_exact(4)
            .map(|b| format_float(f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64))
            .collect::<Vec<_>>()
            .join(","),
        ScalarType::Float16 => data
            .chunks_exact(2)
            .map(|b| format_float(half::f16::from_le_bytes([b[0], b[1]]).to_f64()))
            .collect::<Vec<_>>()
            .join(","),
        ScalarType::Int8 => data
            .iter()
            .map(|&b| (b as i8).to_string())
            .collect::<Vec<_>>()
            .join(","),
        ScalarType::UInt8 => data
            .iter()
            .map(|b| b.to_string())
            .collect::<Vec<_>>()
            .join(","),
        ScalarType::UInt16 => data
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]).to_string())
            .collect::<Vec<_>>()
            .join(","),
        ScalarType::UInt32 => data
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]).to_string())
            .collect::<Vec<_>>()
            .join(","),
        ScalarType::UInt64 => data
            .chunks_exact(8)
            .map(|b| {
                u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]).to_string()
            })
            .collect::<Vec<_>>()
            .join(","),
        ScalarType::Bool => data
            .iter()
            .map(|&b| if b != 0 { "true" } else { "false" })
            .collect::<Vec<_>>()
            .join(","),
        ScalarType::Float64 => data
            .chunks_exact(8)
            .map(|b| {
                format_float(f64::from_le_bytes([
                    b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                ]))
            })
            .collect::<Vec<_>>()
            .join(","),
        ScalarType::Int16 => data
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]).to_string())
            .collect::<Vec<_>>()
            .join(","),
        _ => panic!("unsupported scalar type for tensor elements: {dtype:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mil_rs::ir::{Function, Operation, TensorData};

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
        op.inputs.insert(
            "val".to_string(),
            Value::Tensor {
                data: data.into(),
                shape,
                dtype,
            },
        );
        op
    }

    #[test]
    fn mil_text_simple_program() {
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
        func.body.operations[0].output_types = vec![Some(input_ty.clone())];

        let program = make_program(func);
        let config = MilTextConfig::default();
        let (text, weights) = program_to_mil_text(&program, &config).unwrap();

        assert!(text.contains("program(1.3)"), "should use version 1.3");
        assert!(
            text.contains("[buildInfo = "),
            "should have buildInfo block"
        );
        assert!(
            text.contains("func main<ios18>("),
            "should have platform spec"
        );
        // Parameters: TYPE name (no % prefix in signature).
        assert!(text.contains("tensor<fp16, [1,768,1,32]> a_input0"));
        assert!(text.contains("tensor<fp16, [1,768,1,32]> a_input1"));
        // Op with type prefix, name attribute, and semicolon.
        assert!(text.contains(
            "tensor<fp16, [1,768,1,32]> z_output0 = add(x=a_input0, y=a_input1)[name=string(\"z_output0\")];"
        ));
        // Block return with semicolon.
        assert!(text.contains("} -> (z_output0);"));
        assert!(weights.is_empty());
    }

    #[test]
    fn mil_text_variable_naming() {
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

        // Inputs renamed (TYPE name format, no %).
        assert!(text.contains("tensor<fp32, [1,3,1,224]> a_input0"));
        assert!(text.contains("tensor<fp32, [1,3,1,224]> a_input1"));

        // Outputs renamed with semicolon.
        assert!(text.contains("} -> (z_output0, z_output1);"));

        // References to inputs are also renamed.
        assert!(text.contains("x=a_input0"));
        assert!(text.contains("x=a_input1"));
    }

    #[test]
    fn mil_text_weight_offsets() {
        let ty = TensorType::new(ScalarType::Float16, vec![1, 64, 1, 32]);
        let mut func = Function::new("main").with_input("x", ty.clone());

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

        // Two weight entries collected — each with its own BLOBFILE, offset=64.
        assert_eq!(weights.len(), 2);
        assert_eq!(weights[0].name, "layer0_w");
        assert_eq!(weights[0].offset, 64);
        assert_eq!(weights[0].data.len(), 100);
        assert_eq!(weights[0].path, "@model_path/weights/layer0_w.bin");
        assert_eq!(weights[1].name, "layer1_w");
        assert_eq!(weights[1].offset, 64);
        assert_eq!(weights[1].data.len(), 200);
        assert_eq!(weights[1].path, "@model_path/weights/layer1_w.bin");

        // BLOBFILE references in text (all offset=64).
        assert!(text.contains(
            "BLOBFILE(path=string(\"@model_path/weights/layer0_w.bin\"), offset=uint64(64))"
        ));
        assert!(text.contains(
            "BLOBFILE(path=string(\"@model_path/weights/layer1_w.bin\"), offset=uint64(64))"
        ));
    }

    #[test]
    fn mil_text_const_bool() {
        let ty = TensorType::new(ScalarType::Float16, vec![1, 64, 1, 32]);
        let mut func = Function::new("main").with_input("x", ty.clone());

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

        // Bool const with typed syntax.
        assert!(
            text.contains("bool my_flag = const()[name=string(\"my_flag\"), val=bool(true)];"),
            "got: {text}"
        );
    }

    #[test]
    fn mil_text_layout_1c1s() {
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

        assert!(text.contains("tensor<fp16, [1,768,1,32]>"));
    }

    #[test]
    fn mil_text_round_trip() {
        let input_ty = TensorType::new(ScalarType::Float16, vec![1, 768, 1, 32]);
        let out_ty = TensorType::new(ScalarType::Float16, vec![1, 256, 1, 32]);
        let mut func = Function::new("main")
            .with_input("input0", input_ty.clone())
            .with_input(
                "input1",
                TensorType::new(ScalarType::Float16, vec![1, 1, 1, 32]),
            );

        let w = weight_const(
            "layer0_w",
            vec![0u8; 512],
            vec![256, 768, 1, 1],
            ScalarType::Float16,
        );
        let bias = weight_const("layer0_b", vec![0u8; 64], vec![256], ScalarType::Float16);
        let pad_const = scalar_const("pad_mode", Value::String("valid".to_string()));
        let mut conv = Operation::new("conv", "conv_0")
            .with_input("x", Value::Reference("input0".to_string()))
            .with_input("weight", Value::Reference("layer0_w".to_string()))
            .with_input("pad_type", Value::String("valid".to_string()))
            .with_output("conv_out");
        conv.output_types = vec![Some(out_ty.clone())];
        let mut add = Operation::new("add", "add_0")
            .with_input("x", Value::Reference("conv_out".to_string()))
            .with_input("y", Value::Reference("layer0_b".to_string()))
            .with_output("add_out");
        add.output_types = vec![Some(out_ty.clone())];
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
        assert!(text.starts_with("program(1.3)\n"));
        assert!(text.contains("[buildInfo = "));
        assert!(text.contains("func main<ios18>("));
        assert!(text.contains("tensor<fp16, [1,768,1,32]> a_input0"));
        assert!(text.contains("tensor<fp16, [1,1,1,32]> a_input1"));

        // Weight BLOBFILE refs (each with offset=64, separate files).
        assert!(text.contains(
            "BLOBFILE(path=string(\"@model_path/weights/layer0_w.bin\"), offset=uint64(64))"
        ));
        assert!(text.contains(
            "BLOBFILE(path=string(\"@model_path/weights/layer0_b.bin\"), offset=uint64(64))"
        ));

        // Regular ops reference renamed inputs.
        assert!(text.contains("x=a_input0"));
        assert!(text.contains("pad_type=string(\"valid\")"));
        assert!(text.contains("relu(x="));
        // Semicolons on ops.
        assert!(text.contains(")[name=string(\"conv_out\")];"));
        assert!(text.contains("} -> (z_output0);"));

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

        assert_eq!(emitter.format_value(&Value::Int(42)), "int32(42)");
        assert_eq!(emitter.format_value(&Value::Float(3.14)), "fp16(3.14)");
        assert_eq!(emitter.format_value(&Value::Float(1.0)), "fp16(1.0)");
        assert_eq!(emitter.format_value(&Value::Bool(true)), "bool(true)");
        assert_eq!(emitter.format_value(&Value::Bool(false)), "bool(false)");
        assert_eq!(
            emitter.format_value(&Value::String("hello".into())),
            "string(\"hello\")"
        );
        assert_eq!(
            emitter.format_value(&Value::List(vec![Value::Int(1), Value::Int(2)])),
            "tensor<int32, [2]>([1,2])"
        );
        assert_eq!(emitter.format_value(&Value::Reference("foo".into())), "foo");

        // Tensor values should produce inline tensor literals.
        assert_eq!(
            emitter.format_value(&Value::Tensor {
                data: TensorData::Inline(vec![1, 0, 0, 0, 1, 0, 0, 0, 128, 0, 0, 0]), // [1, 1, 128] as int32
                shape: vec![3],
                dtype: ScalarType::Int32,
            }),
            "tensor<int32, [3]>([1,1,128])"
        );
    }

    #[test]
    fn mil_text_dynamic_dims() {
        let ty =
            TensorType::with_dynamic_shape(ScalarType::Float16, vec![Some(1), None, Some(1), None]);
        let config = MilTextConfig::default();
        let emitter = MilTextEmitter::new(&config);
        assert_eq!(emitter.format_tensor_type(&ty), "tensor<fp16, [1,?,1,?]>");
    }
}
