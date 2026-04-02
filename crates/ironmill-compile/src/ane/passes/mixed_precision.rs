//! Mixed-precision quantization pass.
//!
//! Applies different quantization strategies (FP16, INT8, or none) to different
//! operations based on a user-provided configuration. This enables quality-preserving
//! mixed-precision inference where, for example, attention layers stay in FP16
//! while FFN layers are quantized to INT8.
//!
//! Also provides [`PerExpertQuantPass`] for Mixture-of-Experts models, applying
//! different compression levels to different experts based on activation frequency.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use mil_rs::error::{MilError, Result};
use mil_rs::ir::Pass;
use mil_rs::ir::Program;
use mil_rs::ir::Value;
use mil_rs::ir::passes::kmeans::kmeans;
use mil_rs::ir::passes::tensor_utils::tensor_as_f32_slice;
use mil_rs::ir::{ScalarType, TensorType};

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
                            match op.inputs.remove("val") {
                                Some(v) => v,
                                None => continue,
                            }
                        } else {
                            match op.attributes.remove("val") {
                                Some(v) => v,
                                None => continue,
                            }
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

// ---------------------------------------------------------------------------
// Per-expert quantization for Mixture-of-Experts models
// ---------------------------------------------------------------------------

/// Quantization strategy for a single expert (or layer class).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExpertQuantStrategy {
    /// FP16 — high accuracy, moderate compression.
    Fp16,
    /// INT8 affine weight quantization.
    Int8,
    /// N-bit palettization via k-means (2 or 4 bits).
    #[serde(rename = "palettize_4bit")]
    Palettize4Bit,
    #[serde(rename = "palettize_2bit")]
    Palettize2Bit,
    /// Leave unchanged.
    None,
}

/// Per-expert quantization settings for a single expert.
#[derive(Debug, Clone, Deserialize)]
struct ExpertEntry {
    /// Expert index.
    expert_id: usize,
    /// Quantization strategy.
    strategy: ExpertQuantStrategy,
    /// Activation frequency (0.0–1.0), informational / for diagnostics.
    #[serde(default)]
    #[allow(dead_code)]
    activation_frequency: f64,
}

/// Top-level JSON schema for the per-expert configuration file.
#[derive(Debug, Clone, Deserialize)]
struct RawExpertConfig {
    /// Strategy for shared/non-expert layers (default: fp16).
    #[serde(default = "default_shared_strategy")]
    shared_strategy: ExpertQuantStrategy,
    /// Per-expert entries.
    experts: Vec<ExpertEntry>,
}

fn default_shared_strategy() -> ExpertQuantStrategy {
    ExpertQuantStrategy::Fp16
}

/// Configuration for per-expert quantization.
///
/// Maps each expert ID to a quantization strategy and defines a strategy
/// for shared (always-active) layers such as embeddings, routers, and norms.
///
/// # JSON format
///
/// ```json
/// {
///   "shared_strategy": "fp16",
///   "experts": [
///     { "expert_id": 0, "strategy": "fp16",          "activation_frequency": 0.45 },
///     { "expert_id": 1, "strategy": "int8",           "activation_frequency": 0.30 },
///     { "expert_id": 2, "strategy": "palettize_4bit", "activation_frequency": 0.15 },
///     { "expert_id": 3, "strategy": "palettize_2bit", "activation_frequency": 0.05 }
///   ]
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ExpertQuantConfig {
    /// Quantization strategy for shared (non-expert) layers.
    pub shared_strategy: ExpertQuantStrategy,
    /// Per-expert strategy map: expert_id → strategy.
    pub expert_strategies: HashMap<usize, ExpertQuantStrategy>,
    /// Default strategy for experts not explicitly listed.
    pub default_expert_strategy: ExpertQuantStrategy,
}

impl ExpertQuantConfig {
    /// Load from a JSON file.
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            MilError::Validation(format!("failed to read expert quant config: {e}"))
        })?;
        Self::from_json_str(&content)
    }

    /// Parse from a JSON string.
    pub fn from_json_str(s: &str) -> Result<Self> {
        let raw: RawExpertConfig = serde_json::from_str(s)
            .map_err(|e| MilError::Validation(format!("invalid expert quant config: {e}")))?;

        let mut expert_strategies = HashMap::new();
        for entry in &raw.experts {
            expert_strategies.insert(entry.expert_id, entry.strategy);
        }

        Ok(Self {
            shared_strategy: raw.shared_strategy,
            expert_strategies,
            default_expert_strategy: ExpertQuantStrategy::Int8,
        })
    }

    /// Create a preset: hot experts in FP16, cold experts in 4-bit palettization,
    /// shared layers in FP16.
    ///
    /// `hot_experts` — expert IDs considered frequently activated.
    pub fn preset_hot_cold(hot_experts: &[usize], total_experts: usize) -> Self {
        let mut expert_strategies = HashMap::new();
        for id in 0..total_experts {
            let strategy = if hot_experts.contains(&id) {
                ExpertQuantStrategy::Fp16
            } else {
                ExpertQuantStrategy::Palettize4Bit
            };
            expert_strategies.insert(id, strategy);
        }
        Self {
            shared_strategy: ExpertQuantStrategy::Fp16,
            expert_strategies,
            default_expert_strategy: ExpertQuantStrategy::Palettize4Bit,
        }
    }

    /// Resolve the strategy for a given op by name.
    fn resolve(&self, op_name: &str) -> ExpertQuantStrategy {
        let name_lower = op_name.to_lowercase();

        if let Some(expert_id) = extract_expert_id(&name_lower) {
            return self
                .expert_strategies
                .get(&expert_id)
                .copied()
                .unwrap_or(self.default_expert_strategy);
        }

        // Router / gate ops are treated as shared.
        if name_lower.contains("gate") || name_lower.contains("router") {
            return self.shared_strategy;
        }

        self.shared_strategy
    }
}

/// Extract an expert ID from an operation name.
///
/// Matches patterns like `expert_0`, `experts.1`, `expert-2`, `expert3`.
/// This is a local copy of the logic in `convert::moe` so that this module
/// remains self-contained (no cross-crate dependency on the convert layer).
fn extract_expert_id(name: &str) -> Option<usize> {
    let pos = name.find("expert")?;
    let after = &name[pos + 6..];
    let after = after.strip_prefix('s').unwrap_or(after);
    let after = after
        .strip_prefix('.')
        .or_else(|| after.strip_prefix('_'))
        .or_else(|| after.strip_prefix('-'))
        .unwrap_or(after);
    let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse().ok()
}

/// Per-expert quantization pass for Mixture-of-Experts models.
///
/// Detects expert ops by name pattern and applies the configured quantization
/// strategy per expert. Shared layers (embeddings, routers, norms) get a
/// separate strategy (typically FP16 for stability).
pub struct PerExpertQuantPass {
    config: ExpertQuantConfig,
}

impl PerExpertQuantPass {
    /// Create from an [`ExpertQuantConfig`].
    pub fn new(config: ExpertQuantConfig) -> Self {
        Self { config }
    }

    /// Create from a JSON config file.
    pub fn from_config_file(path: &Path) -> Result<Self> {
        let config = ExpertQuantConfig::from_json_file(path)?;
        Ok(Self { config })
    }

    /// Apply FP16 quantization to a single value.
    fn apply_fp16(value: &mut Value) {
        MixedPrecisionPass::quantize_value_fp16(value);
    }

    /// Apply palettization to a const FP32/FP16 tensor op in-place.
    fn apply_palettize(op: &mut mil_rs::ir::Operation, n_bits: u8) {
        if op.op_type != "const" {
            return;
        }

        let in_inputs = matches!(
            op.inputs.get("val"),
            Some(Value::Tensor {
                dtype: ScalarType::Float32 | ScalarType::Float16,
                ..
            })
        );
        let in_attrs = !in_inputs
            && matches!(
                op.attributes.get("val"),
                Some(Value::Tensor {
                    dtype: ScalarType::Float32 | ScalarType::Float16,
                    ..
                })
            );
        if !in_inputs && !in_attrs {
            return;
        }

        let val = if in_inputs {
            match op.inputs.remove("val") {
                Some(v) => v,
                None => return,
            }
        } else {
            match op.attributes.remove("val") {
                Some(v) => v,
                None => return,
            }
        };

        if let Value::Tensor { data, shape, dtype } = val {
            let floats = match dtype {
                ScalarType::Float32 => tensor_as_f32_slice(&data).to_vec(),
                ScalarType::Float16 => fp16_bytes_to_f32(&data),
                _ => return,
            };

            let k = 1usize << n_bits;
            let (centroids, assignments) = kmeans(&floats, k, 100);

            let lut_bytes: Vec<u8> = centroids.iter().flat_map(|c| c.to_le_bytes()).collect();
            let packed_indices = pack_indices(&assignments, n_bits);

            let shape_u32: Vec<u8> = shape
                .iter()
                .flat_map(|&d| (d as u32).to_le_bytes())
                .collect();

            op.op_type = "constexpr_lut_to_dense".to_string();
            op.inputs.remove("val");
            op.attributes.remove("val");
            op.attributes.insert(
                "lut".to_string(),
                Value::Tensor {
                    data: lut_bytes,
                    shape: vec![k],
                    dtype: ScalarType::Float32,
                },
            );
            op.attributes.insert(
                "indices".to_string(),
                Value::Tensor {
                    data: packed_indices,
                    shape: shape.clone(),
                    dtype: ScalarType::UInt8,
                },
            );
            op.attributes.insert(
                "shape".to_string(),
                Value::Tensor {
                    data: shape_u32,
                    shape: vec![shape.len()],
                    dtype: ScalarType::UInt32,
                },
            );

            let out_type = TensorType::new(ScalarType::Float32, shape);
            if let Some(slot) = op.output_types.get_mut(0) {
                *slot = Some(out_type);
            } else {
                op.output_types.push(Some(out_type));
            }
        }
    }
}

impl Pass for PerExpertQuantPass {
    fn name(&self) -> &str {
        "per-expert-quantization"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            for op in &mut function.body.operations {
                let strategy = self.config.resolve(&op.name);

                match strategy {
                    ExpertQuantStrategy::None => {}
                    ExpertQuantStrategy::Fp16 => {
                        for value in op.inputs.values_mut() {
                            Self::apply_fp16(value);
                        }
                        for value in op.attributes.values_mut() {
                            Self::apply_fp16(value);
                        }
                        for tt in op.output_types.iter_mut().flatten() {
                            if tt.scalar_type == ScalarType::Float32 {
                                tt.scalar_type = ScalarType::Float16;
                            }
                        }
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
                    ExpertQuantStrategy::Int8 => {
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
                            match op.inputs.remove("val") {
                                Some(v) => v,
                                None => continue,
                            }
                        } else {
                            match op.attributes.remove("val") {
                                Some(v) => v,
                                None => continue,
                            }
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
                    ExpertQuantStrategy::Palettize4Bit => {
                        Self::apply_palettize(op, 4);
                    }
                    ExpertQuantStrategy::Palettize2Bit => {
                        Self::apply_palettize(op, 2);
                    }
                }
            }
        }
        Ok(())
    }
}

/// Convert FP16 raw bytes to f32 values.
fn fp16_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    use half::f16;
    data.chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

/// Pack index values into n-bit packed bytes.
fn pack_indices(indices: &[usize], n_bits: u8) -> Vec<u8> {
    let n_bits = n_bits as usize;
    let total_bits = indices.len() * n_bits;
    let n_bytes = total_bits.div_ceil(8);
    let mut packed = vec![0u8; n_bytes];
    for (i, &idx) in indices.iter().enumerate() {
        let bit_offset = i * n_bits;
        let byte_pos = bit_offset / 8;
        let bit_pos = bit_offset % 8;
        let val = (idx as u16) & ((1 << n_bits) - 1);
        let wide = (val as u32) << bit_pos;
        packed[byte_pos] |= wide as u8;
        if bit_pos + n_bits > 8 && byte_pos + 1 < n_bytes {
            packed[byte_pos + 1] |= (wide >> 8) as u8;
        }
        if bit_pos + n_bits > 16 && byte_pos + 2 < n_bytes {
            packed[byte_pos + 2] |= (wide >> 16) as u8;
        }
    }
    packed
}

#[cfg(test)]
mod tests {
    use super::*;
    use mil_rs::ir::Function;
    use mil_rs::ir::Operation;

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

    // ── Per-expert quantization tests ────────────────────────────────────

    #[test]
    fn expert_quant_config_from_json() {
        let json = r#"{
            "shared_strategy": "fp16",
            "experts": [
                { "expert_id": 0, "strategy": "fp16", "activation_frequency": 0.45 },
                { "expert_id": 1, "strategy": "int8", "activation_frequency": 0.30 },
                { "expert_id": 2, "strategy": "palettize_4bit", "activation_frequency": 0.15 },
                { "expert_id": 3, "strategy": "palettize_2bit", "activation_frequency": 0.05 }
            ]
        }"#;
        let config = ExpertQuantConfig::from_json_str(json).unwrap();
        assert_eq!(config.shared_strategy, ExpertQuantStrategy::Fp16);
        assert_eq!(config.expert_strategies.len(), 4);
        assert_eq!(config.expert_strategies[&0], ExpertQuantStrategy::Fp16);
        assert_eq!(
            config.expert_strategies[&2],
            ExpertQuantStrategy::Palettize4Bit
        );
    }

    #[test]
    fn expert_quant_config_default_shared() {
        let json = r#"{ "experts": [] }"#;
        let config = ExpertQuantConfig::from_json_str(json).unwrap();
        assert_eq!(config.shared_strategy, ExpertQuantStrategy::Fp16);
    }

    #[test]
    fn expert_quant_resolve_expert_op() {
        let config = ExpertQuantConfig {
            shared_strategy: ExpertQuantStrategy::Fp16,
            expert_strategies: {
                let mut m = HashMap::new();
                m.insert(0, ExpertQuantStrategy::Fp16);
                m.insert(1, ExpertQuantStrategy::Int8);
                m
            },
            default_expert_strategy: ExpertQuantStrategy::Palettize4Bit,
        };

        assert_eq!(
            config.resolve("layer.expert_0.w1"),
            ExpertQuantStrategy::Fp16
        );
        assert_eq!(
            config.resolve("layer.experts.1.fc"),
            ExpertQuantStrategy::Int8
        );
        // Unknown expert falls back to default_expert_strategy.
        assert_eq!(
            config.resolve("layer.expert_5.down"),
            ExpertQuantStrategy::Palettize4Bit
        );
    }

    #[test]
    fn expert_quant_resolve_shared_op() {
        let config = ExpertQuantConfig {
            shared_strategy: ExpertQuantStrategy::Fp16,
            expert_strategies: HashMap::new(),
            default_expert_strategy: ExpertQuantStrategy::Int8,
        };

        assert_eq!(config.resolve("embed.weight"), ExpertQuantStrategy::Fp16);
        assert_eq!(config.resolve("router.gate"), ExpertQuantStrategy::Fp16);
        assert_eq!(
            config.resolve("layer_norm.weight"),
            ExpertQuantStrategy::Fp16
        );
    }

    #[test]
    fn expert_quant_preset_hot_cold() {
        let config = ExpertQuantConfig::preset_hot_cold(&[0, 2], 4);
        assert_eq!(config.expert_strategies[&0], ExpertQuantStrategy::Fp16);
        assert_eq!(
            config.expert_strategies[&1],
            ExpertQuantStrategy::Palettize4Bit
        );
        assert_eq!(config.expert_strategies[&2], ExpertQuantStrategy::Fp16);
        assert_eq!(
            config.expert_strategies[&3],
            ExpertQuantStrategy::Palettize4Bit
        );
        assert_eq!(config.shared_strategy, ExpertQuantStrategy::Fp16);
    }

    #[test]
    fn per_expert_pass_applies_fp16_to_hot_expert() {
        let fp32_data = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let tensor_val = Value::Tensor {
            data: fp32_data,
            shape: vec![4],
            dtype: ScalarType::Float32,
        };

        let config = ExpertQuantConfig {
            shared_strategy: ExpertQuantStrategy::None,
            expert_strategies: {
                let mut m = HashMap::new();
                m.insert(0, ExpertQuantStrategy::Fp16);
                m
            },
            default_expert_strategy: ExpertQuantStrategy::None,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body.add_op(const_tensor_op(
            "layer.expert_0.weight",
            "w_out",
            tensor_val,
        ));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        PerExpertQuantPass::new(config).run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        match op.inputs.get("val") {
            Some(Value::Tensor { dtype, data, .. }) => {
                assert_eq!(*dtype, ScalarType::Float16);
                assert_eq!(data.len(), 8);
            }
            other => panic!("expected FP16 Tensor, got {other:?}"),
        }
    }

    #[test]
    fn per_expert_pass_applies_int8_to_medium_expert() {
        let fp32_data = f32_bytes(&[0.0, 1.0, 2.0, 3.0]);
        let tensor_val = Value::Tensor {
            data: fp32_data,
            shape: vec![4],
            dtype: ScalarType::Float32,
        };

        let config = ExpertQuantConfig {
            shared_strategy: ExpertQuantStrategy::None,
            expert_strategies: {
                let mut m = HashMap::new();
                m.insert(1, ExpertQuantStrategy::Int8);
                m
            },
            default_expert_strategy: ExpertQuantStrategy::None,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body.add_op(const_tensor_op(
            "layer.expert_1.weight",
            "w_out",
            tensor_val,
        ));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        PerExpertQuantPass::new(config).run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "constexpr_affine_dequantize");
        assert!(op.attributes.contains_key("quantized_data"));
        assert!(op.attributes.contains_key("scale"));
    }

    #[test]
    fn per_expert_pass_applies_palettize_to_cold_expert() {
        let fp32_data = f32_bytes(&[0.1, 0.5, 0.9, 1.2, 0.3, 0.7, 1.1, 0.4]);
        let tensor_val = Value::Tensor {
            data: fp32_data,
            shape: vec![8],
            dtype: ScalarType::Float32,
        };

        let config = ExpertQuantConfig {
            shared_strategy: ExpertQuantStrategy::None,
            expert_strategies: {
                let mut m = HashMap::new();
                m.insert(2, ExpertQuantStrategy::Palettize4Bit);
                m
            },
            default_expert_strategy: ExpertQuantStrategy::None,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body.add_op(const_tensor_op(
            "layer.expert_2.weight",
            "w_out",
            tensor_val,
        ));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        PerExpertQuantPass::new(config).run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "constexpr_lut_to_dense");
        assert!(op.attributes.contains_key("lut"));
        assert!(op.attributes.contains_key("indices"));
        assert!(op.attributes.contains_key("shape"));
    }

    #[test]
    fn per_expert_pass_applies_shared_strategy() {
        let fp32_data = f32_bytes(&[1.0, 2.0]);
        let tensor_val = Value::Tensor {
            data: fp32_data,
            shape: vec![2],
            dtype: ScalarType::Float32,
        };

        let config = ExpertQuantConfig {
            shared_strategy: ExpertQuantStrategy::Fp16,
            expert_strategies: HashMap::new(),
            default_expert_strategy: ExpertQuantStrategy::Int8,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_tensor_op("layer_norm.weight", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        PerExpertQuantPass::new(config).run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        // Shared layer → FP16.
        match op.inputs.get("val") {
            Some(Value::Tensor { dtype, .. }) => {
                assert_eq!(*dtype, ScalarType::Float16);
            }
            other => panic!("expected FP16 Tensor, got {other:?}"),
        }
    }

    #[test]
    fn per_expert_pass_mixed_expert_program() {
        // Build a small program with shared + 2 experts, each with a different strategy.
        let config = ExpertQuantConfig {
            shared_strategy: ExpertQuantStrategy::Fp16,
            expert_strategies: {
                let mut m = HashMap::new();
                m.insert(0, ExpertQuantStrategy::Int8);
                m.insert(1, ExpertQuantStrategy::Palettize4Bit);
                m
            },
            default_expert_strategy: ExpertQuantStrategy::None,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");

        // Shared const.
        func.body.add_op(const_tensor_op(
            "embed.weight",
            "embed_out",
            Value::Tensor {
                data: f32_bytes(&[1.0, 2.0]),
                shape: vec![2],
                dtype: ScalarType::Float32,
            },
        ));
        // Expert 0 const.
        func.body.add_op(const_tensor_op(
            "expert_0.weight",
            "e0_out",
            Value::Tensor {
                data: f32_bytes(&[3.0, 4.0, 5.0, 6.0]),
                shape: vec![4],
                dtype: ScalarType::Float32,
            },
        ));
        // Expert 1 const.
        func.body.add_op(const_tensor_op(
            "expert_1.weight",
            "e1_out",
            Value::Tensor {
                data: f32_bytes(&[0.1, 0.5, 0.9, 1.2, 0.3, 0.7, 1.1, 0.4]),
                shape: vec![8],
                dtype: ScalarType::Float32,
            },
        ));
        func.body.outputs.extend(
            ["embed_out", "e0_out", "e1_out"]
                .iter()
                .map(|s| s.to_string()),
        );
        program.add_function(func);

        PerExpertQuantPass::new(config).run(&mut program).unwrap();

        let ops = &program.functions["main"].body.operations;

        // Shared → FP16.
        assert_eq!(ops[0].op_type, "const");
        match ops[0].inputs.get("val") {
            Some(Value::Tensor { dtype, .. }) => assert_eq!(*dtype, ScalarType::Float16),
            other => panic!("embed: expected FP16, got {other:?}"),
        }

        // Expert 0 → INT8.
        assert_eq!(ops[1].op_type, "constexpr_affine_dequantize");

        // Expert 1 → 4-bit palettization.
        assert_eq!(ops[2].op_type, "constexpr_lut_to_dense");
    }

    #[test]
    fn extract_expert_id_patterns() {
        assert_eq!(extract_expert_id("expert_0_w1"), Some(0));
        assert_eq!(extract_expert_id("experts.1.fc"), Some(1));
        assert_eq!(extract_expert_id("block_expert-2"), Some(2));
        assert_eq!(extract_expert_id("expert3_linear"), Some(3));
        assert_eq!(extract_expert_id("layer.experts.15.down"), Some(15));
        assert_eq!(extract_expert_id("no_match_here"), None);
        assert_eq!(extract_expert_id("expert"), None);
    }

    #[test]
    fn per_expert_pass_name() {
        let config = ExpertQuantConfig {
            shared_strategy: ExpertQuantStrategy::Fp16,
            expert_strategies: HashMap::new(),
            default_expert_strategy: ExpertQuantStrategy::Int8,
        };
        let pass = PerExpertQuantPass::new(config);
        assert_eq!(pass.name(), "per-expert-quantization");
    }

    #[test]
    fn invalid_expert_config_json() {
        let json = r#"{ "experts": [{ "expert_id": 0 }] }"#;
        let result = ExpertQuantConfig::from_json_str(json);
        assert!(result.is_err(), "missing strategy field should error");
    }
}
