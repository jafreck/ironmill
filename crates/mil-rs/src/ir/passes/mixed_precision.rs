//! Mixed-precision quantization pass.
//!
//! Applies different quantization strategies (FP16, INT8, or none) to different
//! operations based on a user-provided configuration. This enables quality-preserving
//! mixed-precision inference where, for example, attention layers stay in FP16
//! while FFN layers are quantized to INT8.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use super::tensor_utils::tensor_as_f32_slice;
use crate::error::{MilError, Result};
use crate::ir::pass::Pass;
use crate::ir::program::Program;
use crate::ir::tensor::{ScalarType, TensorType};
use crate::ir::types::Value;

/// Per-operation quantization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpPrecision {
    /// No quantization — keep original precision.
    None,
    /// FP16 quantization (truncation from FP32).
    Fp16,
    /// INT8 affine quantization (weight-only).
    Int8,
}

/// A single rule that maps an operation name/type pattern to a precision.
#[derive(Debug, Clone)]
pub struct PrecisionRule {
    /// Glob pattern to match against operation names or op types.
    pub pattern: String,
    /// The precision to apply to matching operations.
    pub precision: OpPrecision,
}

/// Configuration for mixed-precision quantization.
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Ordered list of rules. First matching rule wins.
    pub rules: Vec<PrecisionRule>,
    /// Default precision for operations that don't match any rule.
    pub default: OpPrecision,
}

/// Raw TOML deserialization structure.
#[derive(Debug, Deserialize)]
struct RawConfig {
    rules: HashMap<String, String>,
}

impl MixedPrecisionConfig {
    /// Load configuration from a TOML file.
    ///
    /// Expected format:
    /// ```toml
    /// [rules]
    /// "attention.*" = "fp16"
    /// "ffn.*" = "int8"
    /// default = "fp16"
    /// ```
    pub fn from_toml_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            MilError::Validation(format!("failed to read mixed-precision config: {e}"))
        })?;
        Self::from_toml_str(&content)
    }

    /// Parse configuration from a TOML string.
    pub fn from_toml_str(s: &str) -> Result<Self> {
        let raw: RawConfig = toml::from_str(s)
            .map_err(|e| MilError::Validation(format!("invalid mixed-precision config: {e}")))?;

        let mut rules = Vec::new();
        let mut default = OpPrecision::None;

        for (pattern, precision_str) in &raw.rules {
            let precision = parse_precision(precision_str)?;
            if pattern == "default" {
                default = precision;
            } else {
                rules.push(PrecisionRule {
                    pattern: pattern.clone(),
                    precision,
                });
            }
        }

        Ok(Self { rules, default })
    }

    /// Create a preset config: attention in FP16, everything else in INT8.
    pub fn preset_fp16_int8() -> Self {
        Self {
            rules: vec![
                PrecisionRule {
                    pattern: "*attention*".to_string(),
                    precision: OpPrecision::Fp16,
                },
                PrecisionRule {
                    pattern: "*attn*".to_string(),
                    precision: OpPrecision::Fp16,
                },
                PrecisionRule {
                    pattern: "*softmax*".to_string(),
                    precision: OpPrecision::Fp16,
                },
                PrecisionRule {
                    pattern: "*layer_norm*".to_string(),
                    precision: OpPrecision::Fp16,
                },
                PrecisionRule {
                    pattern: "*layernorm*".to_string(),
                    precision: OpPrecision::Fp16,
                },
            ],
            default: OpPrecision::Int8,
        }
    }

    /// Determine the precision for an operation by matching its name and op_type
    /// against rules. First matching rule wins; falls back to default.
    fn resolve(&self, op_name: &str, op_type: &str) -> OpPrecision {
        for rule in &self.rules {
            if glob_match::glob_match(&rule.pattern, op_name)
                || glob_match::glob_match(&rule.pattern, op_type)
            {
                return rule.precision;
            }
        }
        self.default
    }
}

fn parse_precision(s: &str) -> Result<OpPrecision> {
    match s.to_lowercase().as_str() {
        "fp16" => Ok(OpPrecision::Fp16),
        "int8" => Ok(OpPrecision::Int8),
        "none" => Ok(OpPrecision::None),
        other => Err(MilError::Validation(format!(
            "unknown precision '{other}': expected 'fp16', 'int8', or 'none'"
        ))),
    }
}

/// Mixed-precision quantization pass.
///
/// Applies per-operation quantization based on name/type pattern matching.
/// Operations matching FP16 rules get FP16 treatment; those matching INT8
/// rules get INT8 weight quantization; unmatched ops use the default.
pub struct MixedPrecisionPass {
    config: MixedPrecisionConfig,
}

impl MixedPrecisionPass {
    /// Create a new mixed-precision pass from a config.
    pub fn new(config: MixedPrecisionConfig) -> Self {
        Self { config }
    }

    /// Create from a TOML config file path.
    pub fn from_config_file(path: &Path) -> Result<Self> {
        let config = MixedPrecisionConfig::from_toml_file(path)?;
        Ok(Self { config })
    }

    /// Create the preset "attention FP16, everything else INT8" pass.
    pub fn preset_fp16_int8() -> Self {
        Self {
            config: MixedPrecisionConfig::preset_fp16_int8(),
        }
    }

    /// Apply FP16 quantization to a single value (mirrors `fp16_quantize::quantize_value`).
    fn quantize_value_fp16(value: &mut Value) {
        match value {
            Value::Tensor {
                data,
                shape: _,
                dtype,
            } if *dtype == ScalarType::Float32 => {
                *data = fp32_to_fp16_bytes(data);
                *dtype = ScalarType::Float16;
            }
            Value::Type(ty) if ty.scalar_type == ScalarType::Float32 => {
                ty.scalar_type = ScalarType::Float16;
            }
            Value::List(items) => {
                for item in items {
                    Self::quantize_value_fp16(item);
                }
            }
            _ => {}
        }
    }
}

impl Pass for MixedPrecisionPass {
    fn name(&self) -> &str {
        "mixed-precision-quantization"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            for op in &mut function.body.operations {
                let precision = self.config.resolve(&op.name, &op.op_type);

                match precision {
                    OpPrecision::None => {}
                    OpPrecision::Fp16 => {
                        // Apply FP16 quantization to this operation's values.
                        for value in op.inputs.values_mut() {
                            Self::quantize_value_fp16(value);
                        }
                        for value in op.attributes.values_mut() {
                            Self::quantize_value_fp16(value);
                        }
                        for tt in op.output_types.iter_mut().flatten() {
                            if tt.scalar_type == ScalarType::Float32 {
                                tt.scalar_type = ScalarType::Float16;
                            }
                        }
                        // For const ops, ensure output_types is set.
                        if op.op_type == "const" {
                            let needs_type = op.output_types.first().is_none_or(|ot| ot.is_none());
                            if needs_type {
                                let val = op.inputs.get("val").or_else(|| op.attributes.get("val"));
                                if let Some(Value::Tensor {
                                    shape,
                                    dtype: ScalarType::Float16,
                                    ..
                                }) = val
                                {
                                    let tt = TensorType::new(ScalarType::Float16, shape.clone());
                                    if let Some(slot) = op.output_types.get_mut(0) {
                                        *slot = Some(tt);
                                    } else {
                                        op.output_types.push(Some(tt));
                                    }
                                }
                            }
                        }
                    }
                    OpPrecision::Int8 => {
                        // Apply INT8 quantization to const FP32 tensors only.
                        if op.op_type != "const" {
                            continue;
                        }

                        let in_inputs = matches!(
                            op.inputs.get("val"),
                            Some(Value::Tensor {
                                dtype: ScalarType::Float32,
                                ..
                            })
                        );
                        let in_attrs = !in_inputs
                            && matches!(
                                op.attributes.get("val"),
                                Some(Value::Tensor {
                                    dtype: ScalarType::Float32,
                                    ..
                                })
                            );

                        if !in_inputs && !in_attrs {
                            continue;
                        }

                        let val = if in_inputs {
                            op.inputs.remove("val").unwrap()
                        } else {
                            op.attributes.remove("val").unwrap()
                        };

                        if let Value::Tensor {
                            data,
                            shape,
                            dtype: _,
                        } = val
                        {
                            let floats = tensor_as_f32_slice(&data);
                            let (quantized, scale, zero_point) = quantize_f32_to_uint8(&floats);

                            let quantized_val = Value::Tensor {
                                data: quantized,
                                shape: shape.clone(),
                                dtype: ScalarType::UInt8,
                            };

                            op.op_type = "constexpr_affine_dequantize".to_string();
                            op.inputs.remove("val");
                            op.attributes.remove("val");
                            op.attributes
                                .insert("quantized_data".to_string(), quantized_val);
                            op.attributes
                                .insert("scale".to_string(), Value::Float(scale as f64));
                            op.attributes
                                .insert("zero_point".to_string(), Value::Float(zero_point as f64));
                            op.attributes.insert("axis".to_string(), Value::Int(0));

                            let out_type = TensorType::new(ScalarType::Float32, shape);
                            if let Some(slot) = op.output_types.get_mut(0) {
                                *slot = Some(out_type);
                            } else {
                                op.output_types.push(Some(out_type));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Convert raw FP32 bytes to FP16 bytes.
fn fp32_to_fp16_bytes(data: &[u8]) -> Vec<u8> {
    use half::f16;
    debug_assert!(data.len() % 4 == 0);
    let mut out = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(4) {
        let f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let h = f16::from_f32(f);
        out.extend_from_slice(&h.to_le_bytes());
    }
    out
}

/// Quantize f32 slice to UINT8 using min/max affine quantization.
fn quantize_f32_to_uint8(values: &[f32]) -> (Vec<u8>, f32, f32) {
    if values.is_empty() {
        return (Vec::new(), 1.0, 0.0);
    }

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in values {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }

    let (scale, zp_float) = if (max - min).abs() < f32::EPSILON {
        let zp = (-min).round();
        (1.0_f32, zp)
    } else {
        let s = (max - min) / 255.0;
        let zp = (-min / s).round();
        (s, zp)
    };

    let quantized: Vec<u8> = values
        .iter()
        .map(|&x| {
            let q = (x / scale + zp_float).round().clamp(0.0, 255.0);
            q as u8
        })
        .collect();

    (quantized, scale, zp_float)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::operation::Operation;
    use crate::ir::program::Function;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn const_tensor_op(name: &str, output: &str, value: Value) -> Operation {
        Operation::new("const", name)
            .with_input("val", value)
            .with_output(output)
    }

    #[test]
    fn config_from_toml() {
        let toml = r#"
[rules]
"*attention*" = "fp16"
"*ffn*" = "int8"
default = "fp16"
"#;
        let config = MixedPrecisionConfig::from_toml_str(toml).unwrap();
        assert_eq!(config.default, OpPrecision::Fp16);
        assert!(!config.rules.is_empty());
    }

    #[test]
    fn config_resolve_matches_name() {
        let config = MixedPrecisionConfig {
            rules: vec![
                PrecisionRule {
                    pattern: "*attention*".to_string(),
                    precision: OpPrecision::Fp16,
                },
                PrecisionRule {
                    pattern: "*ffn*".to_string(),
                    precision: OpPrecision::Int8,
                },
            ],
            default: OpPrecision::None,
        };

        assert_eq!(
            config.resolve("layer0.attention.qkv", "matmul"),
            OpPrecision::Fp16
        );
        assert_eq!(
            config.resolve("layer0.ffn.linear1", "linear"),
            OpPrecision::Int8
        );
        assert_eq!(
            config.resolve("layer0.norm", "layer_norm"),
            OpPrecision::None
        );
    }

    #[test]
    fn config_resolve_matches_op_type() {
        let config = MixedPrecisionConfig {
            rules: vec![PrecisionRule {
                pattern: "softmax".to_string(),
                precision: OpPrecision::Fp16,
            }],
            default: OpPrecision::Int8,
        };

        assert_eq!(config.resolve("op_42", "softmax"), OpPrecision::Fp16);
        assert_eq!(config.resolve("op_99", "conv"), OpPrecision::Int8);
    }

    #[test]
    fn preset_fp16_int8_config() {
        let config = MixedPrecisionConfig::preset_fp16_int8();
        assert_eq!(config.default, OpPrecision::Int8);
        assert_eq!(
            config.resolve("transformer.attention.query", "matmul"),
            OpPrecision::Fp16
        );
        assert_eq!(
            config.resolve("encoder.attn_proj", "linear"),
            OpPrecision::Fp16
        );
        assert_eq!(config.resolve("ffn.linear1", "linear"), OpPrecision::Int8);
    }

    #[test]
    fn mixed_precision_applies_fp16_to_matching_ops() {
        let fp32_data = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let tensor_val = Value::Tensor {
            data: fp32_data,
            shape: vec![4],
            dtype: ScalarType::Float32,
        };

        let config = MixedPrecisionConfig {
            rules: vec![PrecisionRule {
                pattern: "*attention*".to_string(),
                precision: OpPrecision::Fp16,
            }],
            default: OpPrecision::None,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_tensor_op("attention.weight", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        MixedPrecisionPass::new(config).run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        match op.inputs.get("val") {
            Some(Value::Tensor { dtype, data, .. }) => {
                assert_eq!(*dtype, ScalarType::Float16);
                assert_eq!(data.len(), 8); // 4 elements * 2 bytes
            }
            other => panic!("expected FP16 Tensor, got {other:?}"),
        }
    }

    #[test]
    fn mixed_precision_applies_int8_to_matching_ops() {
        let fp32_data = f32_bytes(&[0.0, 1.0, 2.0, 3.0]);
        let tensor_val = Value::Tensor {
            data: fp32_data,
            shape: vec![4],
            dtype: ScalarType::Float32,
        };

        let config = MixedPrecisionConfig {
            rules: vec![PrecisionRule {
                pattern: "*ffn*".to_string(),
                precision: OpPrecision::Int8,
            }],
            default: OpPrecision::None,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_tensor_op("ffn.weight", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        MixedPrecisionPass::new(config).run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "constexpr_affine_dequantize");
        assert!(op.attributes.contains_key("quantized_data"));
        assert!(op.attributes.contains_key("scale"));
        assert!(op.attributes.contains_key("zero_point"));
    }

    #[test]
    fn mixed_precision_leaves_none_ops_unchanged() {
        let fp32_data = f32_bytes(&[1.0, 2.0]);
        let tensor_val = Value::Tensor {
            data: fp32_data.clone(),
            shape: vec![2],
            dtype: ScalarType::Float32,
        };

        let config = MixedPrecisionConfig {
            rules: vec![],
            default: OpPrecision::None,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_tensor_op("weight", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        MixedPrecisionPass::new(config).run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "const");
        match op.inputs.get("val") {
            Some(Value::Tensor { dtype, data, .. }) => {
                assert_eq!(*dtype, ScalarType::Float32);
                assert_eq!(*data, fp32_data);
            }
            other => panic!("expected FP32 Tensor, got {other:?}"),
        }
    }

    #[test]
    fn invalid_precision_returns_error() {
        let toml = r#"
[rules]
"*attention*" = "fp64"
"#;
        let result = MixedPrecisionConfig::from_toml_str(toml);
        assert!(result.is_err());
    }
}
