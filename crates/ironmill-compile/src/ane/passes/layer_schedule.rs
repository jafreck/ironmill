//! Layer-wise pipeline scheduling pass.
//!
//! Groups operations by logical layer type (convolution blocks, attention blocks,
//! FFN blocks, normalization layers) and applies per-layer-type optimization
//! strategies. This enables coarser-grained mixed-precision control where, for
//! example, all attention layers use FP16 while FFN layers use INT8.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use mil_rs::error::{MilError, Result};
use mil_rs::ir::Pass;
use mil_rs::ir::Program;

use crate::ane::passes::mixed_precision::{
    MixedPrecisionConfig, MixedPrecisionPass, OpPrecision, PrecisionRule,
};

// ── Layer type classification ─────────────────────────────────────────

/// The logical layer type assigned to a group of operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerType {
    /// Convolution block (conv, optionally followed by batch_norm and/or relu).
    Conv,
    /// Attention block (Q/K/V projections + matmul + softmax).
    Attention,
    /// Feed-forward network block (linear → activation → linear).
    Ffn,
    /// Normalization layer (layer_norm, rms_norm, batch_norm, etc.).
    Norm,
    /// Operations that don't fit a recognized pattern.
    Other,
}

impl LayerType {
    /// Parse from a lowercase string (used in TOML config).
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "conv" => Ok(Self::Conv),
            "attention" => Ok(Self::Attention),
            "ffn" => Ok(Self::Ffn),
            "norm" => Ok(Self::Norm),
            "other" => Ok(Self::Other),
            unknown => Err(MilError::Validation(format!(
                "unknown layer type '{unknown}': expected 'conv', 'attention', 'ffn', 'norm', or 'other'"
            ))),
        }
    }
}

/// A detected logical layer — a contiguous group of ops sharing a layer type.
#[derive(Debug, Clone)]
pub struct DetectedLayer {
    /// The layer type.
    pub layer_type: LayerType,
    /// Index range of operations within the function body (start inclusive, end exclusive).
    pub op_range: (usize, usize),
}

// ── Layer detection ───────────────────────────────────────────────────

/// Op-type constants used during detection.
const CONV_OPS: &[&str] = &["conv", "conv_transpose"];
const NORM_OPS: &[&str] = &[
    "layer_norm",
    "rms_norm",
    "batch_norm",
    "instance_norm",
    "group_norm",
];
const ACTIVATION_OPS: &[&str] = &["relu", "gelu", "silu", "sigmoid", "tanh", "swish"];
const LINEAR_OPS: &[&str] = &["linear", "matmul"];
const ATTENTION_INDICATOR_OPS: &[&str] = &["softmax"];

/// Classify a single op_type into a coarse category for pattern matching.
fn op_category(op_type: &str) -> &'static str {
    if CONV_OPS.contains(&op_type) {
        "conv"
    } else if NORM_OPS.contains(&op_type) {
        "norm"
    } else if ACTIVATION_OPS.contains(&op_type) {
        "activation"
    } else if LINEAR_OPS.contains(&op_type) {
        "linear"
    } else if ATTENTION_INDICATOR_OPS.contains(&op_type) {
        "softmax"
    } else {
        "other"
    }
}

/// Detect logical layers in a flat sequence of operations.
///
/// Scans operations for recognisable patterns:
/// - **Conv block**: `conv` optionally followed by `batch_norm` and/or an activation.
/// - **Attention block**: A run containing at least two linear/matmul ops *and* a softmax.
/// - **FFN block**: `linear → activation → linear` (without softmax).
/// - **Norm**: Standalone normalization ops not already consumed by another pattern.
/// - **Other**: Anything left over.
pub fn detect_layers(ops: &[mil_rs::ir::Operation]) -> Vec<DetectedLayer> {
    let n = ops.len();
    if n == 0 {
        return Vec::new();
    }

    // Per-op tag: None means not yet assigned.
    let mut assigned: Vec<Option<LayerType>> = vec![None; n];

    // ── Pass 1: Attention blocks ──────────────────────────────────────
    // Scan for runs that contain ≥2 linear/matmul ops AND a softmax.
    // Use a sliding-window approach: find softmax, then expand outward to
    // capture surrounding linear/matmul ops (Q, K, V projections and
    // output projection).
    for (i, op) in ops.iter().enumerate() {
        if op.op_type == "softmax" && assigned[i].is_none() {
            // Expand backward to find the start of the attention block.
            let mut start = i;
            let mut linear_count = 0u32;
            while start > 0 {
                let prev = start - 1;
                if assigned[prev].is_some() {
                    break;
                }
                let cat = op_category(&ops[prev].op_type);
                if cat == "linear" || cat == "softmax" || cat == "activation" || cat == "other" {
                    if cat == "linear" {
                        linear_count += 1;
                    }
                    start = prev;
                } else {
                    break;
                }
            }

            // Expand forward past the softmax.
            let mut end = i + 1;
            while end < n {
                if assigned[end].is_some() {
                    break;
                }
                let cat = op_category(&ops[end].op_type);
                if cat == "linear" || cat == "activation" || cat == "other" {
                    if cat == "linear" {
                        linear_count += 1;
                    }
                    end += 1;
                } else {
                    break;
                }
            }

            // Require ≥ 2 linear/matmul ops alongside the softmax.
            if linear_count >= 2 {
                for slot in &mut assigned[start..end] {
                    *slot = Some(LayerType::Attention);
                }
            }
        }
    }

    // ── Pass 2: Conv blocks ───────────────────────────────────────────
    {
        let mut i = 0;
        while i < n {
            if assigned[i].is_none() && CONV_OPS.contains(&ops[i].op_type.as_str()) {
                let start = i;
                i += 1;
                // Optional batch_norm
                if i < n && assigned[i].is_none() && ops[i].op_type == "batch_norm" {
                    i += 1;
                }
                // Optional activation
                if i < n
                    && assigned[i].is_none()
                    && ACTIVATION_OPS.contains(&ops[i].op_type.as_str())
                {
                    i += 1;
                }
                for slot in &mut assigned[start..i] {
                    *slot = Some(LayerType::Conv);
                }
            } else {
                i += 1;
            }
        }
    }

    // ── Pass 3: FFN blocks (linear → activation → linear) ────────────
    {
        let mut i = 0;
        while i + 2 < n {
            if assigned[i].is_none()
                && LINEAR_OPS.contains(&ops[i].op_type.as_str())
                && assigned[i + 1].is_none()
                && ACTIVATION_OPS.contains(&ops[i + 1].op_type.as_str())
                && assigned[i + 2].is_none()
                && LINEAR_OPS.contains(&ops[i + 2].op_type.as_str())
            {
                assigned[i] = Some(LayerType::Ffn);
                assigned[i + 1] = Some(LayerType::Ffn);
                assigned[i + 2] = Some(LayerType::Ffn);
                i += 3;
            } else {
                i += 1;
            }
        }
    }

    // ── Pass 4: Norm layers ──────────────────────────────────────────
    for i in 0..n {
        if assigned[i].is_none() && NORM_OPS.contains(&ops[i].op_type.as_str()) {
            assigned[i] = Some(LayerType::Norm);
        }
    }

    // ── Pass 5: Absorb const ops into adjacent compute layers ────────
    // Weight tensors typically appear as `const` ops immediately before
    // the compute op that consumes them. Assign unassigned const ops to
    // the next assigned layer type (forward look-ahead).
    for i in 0..n {
        if assigned[i].is_none() && ops[i].op_type == "const" {
            // Look ahead for the next assigned op.
            for j in (i + 1)..n {
                if let Some(ty) = assigned[j] {
                    assigned[i] = Some(ty);
                    break;
                }
            }
        }
    }

    // ── Pass 6: Everything else → Other ──────────────────────────────
    for slot in &mut assigned {
        if slot.is_none() {
            *slot = Some(LayerType::Other);
        }
    }

    // ── Merge contiguous ranges of the same type ─────────────────────
    let mut layers = Vec::new();
    let mut start = 0;
    while start < n {
        // Every slot filled by pass 6 above.
        let ty = assigned[start].expect("assigned by pass 6");
        let mut end = start + 1;
        while end < n && assigned[end] == Some(ty) {
            end += 1;
        }
        layers.push(DetectedLayer {
            layer_type: ty,
            op_range: (start, end),
        });
        start = end;
    }

    layers
}

// ── Per-layer strategy configuration ──────────────────────────────────

/// TOML-level configuration for layer-wise scheduling.
///
/// Example:
/// ```toml
/// [layer_strategies]
/// attention = "fp16"
/// ffn = "int8"
/// conv = "fp16"
/// norm = "fp16"
/// other = "none"
/// ```
#[derive(Debug, Clone)]
pub struct LayerScheduleConfig {
    /// Maps each layer type to a quantization precision.
    pub strategies: HashMap<LayerType, OpPrecision>,
}

/// Raw TOML deserialization helper.
#[derive(Debug, Deserialize)]
struct RawLayerConfig {
    layer_strategies: HashMap<String, String>,
}

impl LayerScheduleConfig {
    /// Load from a TOML file.
    pub fn from_toml_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            MilError::Validation(format!("failed to read layer-schedule config: {e}"))
        })?;
        Self::from_toml_str(&content)
    }

    /// Parse from a TOML string.
    pub fn from_toml_str(s: &str) -> Result<Self> {
        let raw: RawLayerConfig = toml::from_str(s)
            .map_err(|e| MilError::Validation(format!("invalid layer-schedule config: {e}")))?;

        let mut strategies = HashMap::new();
        for (layer_str, precision_str) in &raw.layer_strategies {
            let layer_type = LayerType::from_str(layer_str)?;
            let precision = parse_precision(precision_str)?;
            strategies.insert(layer_type, precision);
        }

        Ok(Self { strategies })
    }

    /// Build from inline TOML params (as stored in `PassConfig.params`).
    pub fn from_params(params: &HashMap<String, toml::Value>) -> Result<Self> {
        let mut strategies = HashMap::new();
        if let Some(table) = params.get("layer_strategies").and_then(|v| v.as_table()) {
            for (layer_str, precision_val) in table {
                let layer_type = LayerType::from_str(layer_str)?;
                let precision_str = precision_val.as_str().ok_or_else(|| {
                    MilError::Validation(format!(
                        "layer_strategies.{layer_str}: expected string value"
                    ))
                })?;
                let precision = parse_precision(precision_str)?;
                strategies.insert(layer_type, precision);
            }
        }
        Ok(Self { strategies })
    }

    /// Look up the precision for a given layer type.
    /// Returns `OpPrecision::None` if no strategy is configured.
    pub fn precision_for(&self, layer_type: LayerType) -> OpPrecision {
        self.strategies
            .get(&layer_type)
            .copied()
            .unwrap_or(OpPrecision::None)
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

// ── Pass implementation ───────────────────────────────────────────────

/// Layer-wise pipeline scheduling pass.
///
/// 1. Detects logical layer boundaries in each function.
/// 2. Assigns a `LayerType` to every operation.
/// 3. Applies per-layer-type precision strategies by delegating to the
///    existing [`MixedPrecisionPass`] infrastructure, using layer-type–aware
///    precision rules built from the detected layer assignments.
pub struct LayerSchedulePass {
    config: LayerScheduleConfig,
}

impl LayerSchedulePass {
    /// Create from an explicit config.
    pub fn new(config: LayerScheduleConfig) -> Self {
        Self { config }
    }

    /// Create from a TOML config file.
    pub fn from_config_file(path: &Path) -> Result<Self> {
        let config = LayerScheduleConfig::from_toml_file(path)?;
        Ok(Self { config })
    }

    /// Build from pipeline pass params.
    pub fn from_params(params: &HashMap<String, toml::Value>) -> Result<Self> {
        let config = LayerScheduleConfig::from_params(params)?;
        Ok(Self { config })
    }
}

impl Pass for LayerSchedulePass {
    fn name(&self) -> &str {
        "layer-schedule"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            let layers = detect_layers(&function.body.operations);

            // Tag every op with its layer-type precision by building a
            // per-op precision map, then applying the same quantisation
            // logic that MixedPrecisionPass uses.
            let mut op_precision: Vec<OpPrecision> =
                Vec::with_capacity(function.body.operations.len());

            for layer in &layers {
                let prec = self.config.precision_for(layer.layer_type);
                for _ in layer.op_range.0..layer.op_range.1 {
                    op_precision.push(prec);
                }
            }

            // Build a one-shot MixedPrecisionConfig with per-op name rules
            // derived from the detected layers. This reuses the existing
            // quantisation logic without duplicating it.
            let mut rules = Vec::new();
            for (i, &prec) in op_precision.iter().enumerate() {
                if prec == OpPrecision::None {
                    continue;
                }
                let op_name = &function.body.operations[i].name;
                rules.push(PrecisionRule {
                    pattern: op_name.clone(),
                    precision: prec,
                });
            }

            let mp_config = MixedPrecisionConfig {
                rules,
                default: OpPrecision::None,
            };
            let mp_pass = MixedPrecisionPass::new(mp_config);

            // MixedPrecisionPass::run works on the full program, but we've
            // scoped the rules to only this function's op names, so other
            // functions won't be affected. We must however pass the whole
            // program. To avoid double-processing, we run it once after
            // building rules for all functions.
            //
            // Actually, we run per-function below — see the alternative
            // approach at the end of this loop.
            drop(mp_pass);
        }

        // Build a single MixedPrecisionConfig covering all functions.
        let mut all_rules = Vec::new();
        for function in program.functions.values() {
            let layers = detect_layers(&function.body.operations);
            for layer in &layers {
                let prec = self.config.precision_for(layer.layer_type);
                if prec == OpPrecision::None {
                    continue;
                }
                for idx in layer.op_range.0..layer.op_range.1 {
                    let op_name = &function.body.operations[idx].name;
                    all_rules.push(PrecisionRule {
                        pattern: op_name.clone(),
                        precision: prec,
                    });
                }
            }
        }

        if !all_rules.is_empty() {
            let mp_config = MixedPrecisionConfig {
                rules: all_rules,
                default: OpPrecision::None,
            };
            let mp_pass = MixedPrecisionPass::new(mp_config);
            mp_pass.run(program)?;
        }

        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use mil_rs::ir::Function;
    use mil_rs::ir::Operation;
    use mil_rs::ir::ScalarType;
    use mil_rs::ir::Value;

    /// Helper: create a simple op with only a type and name.
    fn simple_op(op_type: &str, name: &str) -> Operation {
        Operation::new(op_type, name).with_output(format!("{name}_out"))
    }

    fn const_f32_op(name: &str, output: &str, values: &[f32]) -> Operation {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Operation::new("const", name)
            .with_input(
                "val",
                Value::Tensor {
                    data,
                    shape: vec![values.len()],
                    dtype: ScalarType::Float32,
                },
            )
            .with_output(output)
    }

    // ── Layer detection tests ─────────────────────────────────────────

    #[test]
    fn detect_conv_bn_relu_cluster() {
        let ops = vec![
            simple_op("conv", "conv_0"),
            simple_op("batch_norm", "bn_0"),
            simple_op("relu", "relu_0"),
        ];
        let layers = detect_layers(&ops);
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].layer_type, LayerType::Conv);
        assert_eq!(layers[0].op_range, (0, 3));
    }

    #[test]
    fn detect_conv_without_bn() {
        let ops = vec![simple_op("conv", "conv_0"), simple_op("relu", "relu_0")];
        let layers = detect_layers(&ops);
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].layer_type, LayerType::Conv);
        assert_eq!(layers[0].op_range, (0, 2));
    }

    #[test]
    fn detect_standalone_conv() {
        let ops = vec![simple_op("conv", "conv_0")];
        let layers = detect_layers(&ops);
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].layer_type, LayerType::Conv);
    }

    #[test]
    fn detect_attention_block() {
        // Q, K, V projections + matmul + softmax + output proj
        let ops = vec![
            simple_op("matmul", "q_proj"),
            simple_op("matmul", "k_proj"),
            simple_op("matmul", "v_proj"),
            simple_op("softmax", "attn_softmax"),
            simple_op("matmul", "attn_out"),
        ];
        let layers = detect_layers(&ops);
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].layer_type, LayerType::Attention);
        assert_eq!(layers[0].op_range, (0, 5));
    }

    #[test]
    fn detect_ffn_block() {
        let ops = vec![
            simple_op("linear", "ffn_up"),
            simple_op("gelu", "ffn_act"),
            simple_op("linear", "ffn_down"),
        ];
        let layers = detect_layers(&ops);
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].layer_type, LayerType::Ffn);
        assert_eq!(layers[0].op_range, (0, 3));
    }

    #[test]
    fn detect_norm_layer() {
        let ops = vec![simple_op("layer_norm", "ln_0")];
        let layers = detect_layers(&ops);
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].layer_type, LayerType::Norm);
    }

    #[test]
    fn detect_rms_norm() {
        let ops = vec![simple_op("rms_norm", "rmsnorm_0")];
        let layers = detect_layers(&ops);
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].layer_type, LayerType::Norm);
    }

    #[test]
    fn detect_mixed_sequence() {
        // norm → attention → norm → ffn
        let ops = vec![
            simple_op("layer_norm", "ln_0"),
            simple_op("matmul", "q_proj"),
            simple_op("matmul", "k_proj"),
            simple_op("softmax", "attn_softmax"),
            simple_op("matmul", "out_proj"),
            simple_op("layer_norm", "ln_1"),
            simple_op("linear", "ffn_up"),
            simple_op("relu", "ffn_act"),
            simple_op("linear", "ffn_down"),
        ];
        let layers = detect_layers(&ops);

        let types: Vec<LayerType> = layers.iter().map(|l| l.layer_type).collect();
        assert_eq!(
            types,
            vec![
                LayerType::Norm,
                LayerType::Attention,
                LayerType::Norm,
                LayerType::Ffn,
            ]
        );
    }

    #[test]
    fn detect_other_ops() {
        let ops = vec![
            simple_op("reshape", "reshape_0"),
            simple_op("transpose", "transpose_0"),
        ];
        let layers = detect_layers(&ops);
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].layer_type, LayerType::Other);
    }

    #[test]
    fn empty_ops_returns_empty() {
        let layers = detect_layers(&[]);
        assert!(layers.is_empty());
    }

    // ── Config tests ──────────────────────────────────────────────────

    #[test]
    fn config_from_toml() {
        let toml = r#"
[layer_strategies]
attention = "fp16"
ffn = "int8"
conv = "fp16"
norm = "fp16"
"#;
        let config = LayerScheduleConfig::from_toml_str(toml).unwrap();
        assert_eq!(
            config.precision_for(LayerType::Attention),
            OpPrecision::Fp16
        );
        assert_eq!(config.precision_for(LayerType::Ffn), OpPrecision::Int8);
        assert_eq!(config.precision_for(LayerType::Conv), OpPrecision::Fp16);
        assert_eq!(config.precision_for(LayerType::Norm), OpPrecision::Fp16);
        assert_eq!(config.precision_for(LayerType::Other), OpPrecision::None);
    }

    #[test]
    fn config_from_params() {
        let mut params = HashMap::new();
        let mut table = toml::value::Table::new();
        table.insert(
            "attention".to_string(),
            toml::Value::String("fp16".to_string()),
        );
        table.insert("ffn".to_string(), toml::Value::String("int8".to_string()));
        params.insert("layer_strategies".to_string(), toml::Value::Table(table));

        let config = LayerScheduleConfig::from_params(&params).unwrap();
        assert_eq!(
            config.precision_for(LayerType::Attention),
            OpPrecision::Fp16
        );
        assert_eq!(config.precision_for(LayerType::Ffn), OpPrecision::Int8);
    }

    #[test]
    fn config_invalid_layer_type() {
        let toml = r#"
[layer_strategies]
transformer = "fp16"
"#;
        let result = LayerScheduleConfig::from_toml_str(toml);
        assert!(result.is_err());
    }

    #[test]
    fn config_invalid_precision() {
        let toml = r#"
[layer_strategies]
attention = "fp64"
"#;
        let result = LayerScheduleConfig::from_toml_str(toml);
        assert!(result.is_err());
    }

    // ── Pass integration tests ────────────────────────────────────────

    #[test]
    fn pass_applies_fp16_to_conv_layer() {
        let mut strategies = HashMap::new();
        strategies.insert(LayerType::Conv, OpPrecision::Fp16);
        let config = LayerScheduleConfig { strategies };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_f32_op("conv_weight", "w_out", &[1.0, 2.0, 3.0, 4.0]));
        func.body.add_op(simple_op("conv", "conv_0"));
        func.body.add_op(simple_op("relu", "relu_0"));
        func.body.outputs.push("relu_0_out".into());
        program.add_function(func);

        let pass = LayerSchedulePass::new(config);
        pass.run(&mut program).unwrap();

        // The const op for the conv weight should have been converted to FP16.
        let const_op = &program.functions["main"].body.operations[0];
        if let Some(Value::Tensor { dtype, .. }) = const_op.inputs.get("val") {
            assert_eq!(*dtype, ScalarType::Float16);
        }
    }

    #[test]
    fn pass_applies_int8_to_ffn_layer() {
        let mut strategies = HashMap::new();
        strategies.insert(LayerType::Ffn, OpPrecision::Int8);
        let config = LayerScheduleConfig { strategies };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_f32_op("ffn_weight", "w_out", &[0.0, 1.0, 2.0, 3.0]));
        func.body.add_op(simple_op("linear", "ffn_up"));
        func.body.add_op(simple_op("gelu", "ffn_act"));
        func.body.add_op(simple_op("linear", "ffn_down"));
        func.body.outputs.push("ffn_down_out".into());
        program.add_function(func);

        let pass = LayerSchedulePass::new(config);
        pass.run(&mut program).unwrap();

        // The const op should have been rewritten to INT8 dequantize.
        let const_op = &program.functions["main"].body.operations[0];
        assert_eq!(const_op.op_type, "constexpr_affine_dequantize");
    }

    #[test]
    fn pass_leaves_unconfigured_layers_unchanged() {
        let mut strategies = HashMap::new();
        strategies.insert(LayerType::Attention, OpPrecision::Fp16);
        let config = LayerScheduleConfig { strategies };

        let fp32_data: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body.add_op(
            Operation::new("const", "other_weight")
                .with_input(
                    "val",
                    Value::Tensor {
                        data: fp32_data.clone(),
                        shape: vec![2],
                        dtype: ScalarType::Float32,
                    },
                )
                .with_output("w_out"),
        );
        func.body.add_op(simple_op("reshape", "reshape_0"));
        func.body.outputs.push("reshape_0_out".into());
        program.add_function(func);

        let pass = LayerSchedulePass::new(config);
        pass.run(&mut program).unwrap();

        // Should be unchanged — no attention ops here.
        let const_op = &program.functions["main"].body.operations[0];
        assert_eq!(const_op.op_type, "const");
        if let Some(Value::Tensor { dtype, data, .. }) = const_op.inputs.get("val") {
            assert_eq!(*dtype, ScalarType::Float32);
            assert_eq!(*data, fp32_data);
        }
    }

    #[test]
    fn pass_name() {
        let config = LayerScheduleConfig {
            strategies: HashMap::new(),
        };
        let pass = LayerSchedulePass::new(config);
        assert_eq!(pass.name(), "layer-schedule");
    }
}
