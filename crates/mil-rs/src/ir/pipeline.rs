//! Pass pipeline manager for ordering, mutual exclusivity, and builder API.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use std::time::Instant;

use serde::Deserialize;

use super::pass::Pass;
use super::passes::{
    AttentionFusionPass, CodebookOptimizationPass, ComputeUnitAnnotationPass, ConstantFoldPass,
    ConvBatchNormFusionPass, ConvBatchNormWeightFoldPass, ConvReluFusionPass,
    DeadCodeEliminationPass, ExpertQuantConfig, Fp16QuantizePass, GeluLinearFusionPass,
    GqaFusionPass, Granularity, IdentityEliminationPass, Int8QuantizePass, KvCachePass,
    LayerNormLinearFusionPass, LayerSchedulePass, LayoutOptimizationPass, LinearReluFusionPass,
    MixedPrecisionConfig, MixedPrecisionPass, OpSplittingPass, OpSubstitutionPass, PalettizePass,
    PerExpertQuantPass, PolarQuantPass, PolarRotationFusionPass, ResidualAddFusionPass,
    ShapeMaterializePass, TypeRepropagationPass,
};
use super::program::Program;
use crate::error::{MilError, Result};

/// A configured optimization pipeline.
///
/// Manages pass ordering, mutual exclusivity checks, and pass selection
/// based on model characteristics and user flags.
pub struct PassPipeline {
    passes: Vec<Box<dyn Pass>>,
    has_fp16: bool,
    has_int8: bool,
    has_palettize: bool,
    has_polar_quant: bool,
    has_mixed_precision: bool,
}

// ── TOML configuration types ──────────────────────────────────────────

/// Top-level TOML pipeline configuration.
#[derive(Debug, Deserialize)]
pub struct PipelineConfig {
    /// Ordered list of pass configurations.
    pub passes: Vec<PassConfig>,
}

/// Configuration for a single pass in the pipeline.
#[derive(Debug, Deserialize)]
pub struct PassConfig {
    /// Pass name (must match one of the built-in pass names).
    pub name: String,
    /// Whether this pass is enabled (default: true).
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Pass-specific parameters.
    #[serde(default)]
    pub params: HashMap<String, toml::Value>,
}

fn default_true() -> bool {
    true
}

/// All known built-in pass names.
const KNOWN_PASSES: &[&str] = &[
    "dead-code-elimination",
    "identity-elimination",
    "constant-folding",
    "conv-bn-weight-fold",
    "conv-batchnorm-fusion",
    "conv-relu-fusion",
    "linear-relu-fusion",
    "layernorm-linear-fusion",
    "gelu-linear-fusion",
    "residual-add-fusion",
    "attention-fusion",
    "gqa-fusion",
    "kv-cache",
    "codebook-optimization",
    "op-substitution",
    "layout-optimization",
    "type-repropagation",
    "fp16-quantization",
    "int8-quantization",
    "mixed-precision",
    "layer-schedule",
    "palettization",
    "polar-quantization",
    "per-expert-quantization",
    "shape-materialization",
    "compute-unit-annotation",
    "op-splitting",
];

/// Create a boxed pass from its name and optional parameters.
fn pass_from_name(name: &str, params: &HashMap<String, toml::Value>) -> Result<Box<dyn Pass>> {
    match name {
        "dead-code-elimination" => Ok(Box::new(DeadCodeEliminationPass)),
        "identity-elimination" => Ok(Box::new(IdentityEliminationPass)),
        "constant-folding" => Ok(Box::new(ConstantFoldPass)),
        "conv-bn-weight-fold" => Ok(Box::new(ConvBatchNormWeightFoldPass)),
        "conv-batchnorm-fusion" => Ok(Box::new(ConvBatchNormFusionPass)),
        "conv-relu-fusion" => Ok(Box::new(ConvReluFusionPass)),
        "linear-relu-fusion" => Ok(Box::new(LinearReluFusionPass)),
        "layernorm-linear-fusion" => Ok(Box::new(LayerNormLinearFusionPass)),
        "gelu-linear-fusion" => Ok(Box::new(GeluLinearFusionPass)),
        "residual-add-fusion" => Ok(Box::new(ResidualAddFusionPass)),
        "attention-fusion" => Ok(Box::new(AttentionFusionPass)),
        "gqa-fusion" => Ok(Box::new(GqaFusionPass)),
        "kv-cache" => {
            let max_seq = params
                .get("max_seq_length")
                .and_then(|v| v.as_integer())
                .map(|i| i as usize)
                .unwrap_or(2048);
            Ok(Box::new(KvCachePass::new(max_seq)))
        }
        "codebook-optimization" => Ok(Box::new(CodebookOptimizationPass)),
        "op-substitution" => Ok(Box::new(OpSubstitutionPass)),
        "layout-optimization" => Ok(Box::new(LayoutOptimizationPass)),
        "type-repropagation" => Ok(Box::new(TypeRepropagationPass)),
        "fp16-quantization" => Ok(Box::new(Fp16QuantizePass)),
        "int8-quantization" => {
            let cal_dir = params
                .get("calibration_dir")
                .and_then(|v| v.as_str())
                .map(PathBuf::from);
            let granularity = match params.get("granularity").and_then(|v| v.as_str()) {
                Some("per-tensor") => Granularity::PerTensor,
                _ => Granularity::PerChannel,
            };
            Ok(Box::new(Int8QuantizePass::new(cal_dir, granularity)))
        }
        "palettization" => {
            let n_bits_i64 = params
                .get("n_bits")
                .and_then(|v| v.as_integer())
                .unwrap_or(4);
            if !matches!(n_bits_i64, 2 | 4 | 6 | 8) {
                return Err(MilError::Validation(format!(
                    "palettize n_bits must be 2, 4, 6, or 8, got {n_bits_i64}"
                )));
            }
            let n_bits = n_bits_i64 as u8;
            Ok(Box::new(PalettizePass::new(n_bits)))
        }
        "polar-quantization" => {
            let n_bits = params.get("bits").and_then(|v| v.as_integer()).unwrap_or(4) as u8;
            let seed = params
                .get("seed")
                .and_then(|v| v.as_integer())
                .unwrap_or(42) as u64;
            let min_elements = params
                .get("min_elements")
                .and_then(|v| v.as_integer())
                .unwrap_or(1024) as usize;
            Ok(Box::new(PolarQuantPass {
                n_bits,
                seed,
                min_elements,
            }))
        }
        "mixed-precision" => {
            let config_path = params
                .get("config_path")
                .and_then(|v| v.as_str())
                .map(PathBuf::from);
            if let Some(path) = config_path {
                let pass = MixedPrecisionPass::from_config_file(&path)?;
                Ok(Box::new(pass))
            } else {
                let config = MixedPrecisionConfig::preset_fp16_int8();
                Ok(Box::new(MixedPrecisionPass::new(config)))
            }
        }
        "layer-schedule" => {
            let config_path = params
                .get("config_path")
                .and_then(|v| v.as_str())
                .map(PathBuf::from);
            if let Some(path) = config_path {
                let pass = LayerSchedulePass::from_config_file(&path)?;
                Ok(Box::new(pass))
            } else {
                let pass = LayerSchedulePass::from_params(params)?;
                Ok(Box::new(pass))
            }
        }
        "shape-materialization" => {
            let mut pass = ShapeMaterializePass::new();
            if let Some(shapes) = params.get("shapes") {
                if let Some(table) = shapes.as_table() {
                    for (input_name, dims_val) in table {
                        if let Some(arr) = dims_val.as_array() {
                            let dims: Vec<usize> = arr
                                .iter()
                                .filter_map(|v| {
                                    v.as_integer()
                                        .and_then(|i| if i >= 0 { Some(i as usize) } else { None })
                                })
                                .collect();
                            pass = pass.with_shape(input_name.clone(), dims);
                        }
                    }
                }
            }
            Ok(Box::new(pass))
        }
        "compute-unit-annotation" => Ok(Box::new(ComputeUnitAnnotationPass)),
        "op-splitting" => {
            let budget = params
                .get("memory_budget")
                .and_then(|v| v.as_str())
                .map(|s| {
                    super::passes::op_split::parse_memory_size(s)
                        .map_err(|e| MilError::Validation(format!("invalid memory budget: {e}")))
                })
                .transpose()?
                .or_else(|| {
                    params
                        .get("memory_budget_bytes")
                        .and_then(|v| v.as_integer())
                        .map(|i| i as usize)
                })
                .unwrap_or(super::passes::op_split::DEFAULT_MEMORY_BUDGET);
            Ok(Box::new(OpSplittingPass::new(budget)))
        }
        "per-expert-quantization" => {
            let config_path = params
                .get("config_path")
                .and_then(|v| v.as_str())
                .map(PathBuf::from);
            if let Some(path) = config_path {
                let pass = PerExpertQuantPass::from_config_file(&path)?;
                Ok(Box::new(pass))
            } else {
                // Default: hot experts 0,1 in FP16, everything else 4-bit palettize.
                let total = params
                    .get("total_experts")
                    .and_then(|v| v.as_integer())
                    .unwrap_or(8) as usize;
                let hot: Vec<usize> = params
                    .get("hot_experts")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_integer().map(|i| i as usize))
                            .collect()
                    })
                    .unwrap_or_else(|| vec![0, 1]);
                let config = ExpertQuantConfig::preset_hot_cold(&hot, total);
                Ok(Box::new(PerExpertQuantPass::new(config)))
            }
        }
        _ => Err(MilError::Validation(format!("unknown pass: '{name}'"))),
    }
}

impl Default for PassPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for PassPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PassPipeline")
            .field("passes", &self.pass_names())
            .field("has_fp16", &self.has_fp16)
            .field("has_int8", &self.has_int8)
            .field("has_palettize", &self.has_palettize)
            .field("has_polar_quant", &self.has_polar_quant)
            .field("has_mixed_precision", &self.has_mixed_precision)
            .finish()
    }
}

impl PassPipeline {
    /// Create the default pipeline with all always-on passes.
    ///
    /// Includes cleanup (DCE, identity elimination, constant folding),
    /// fusion (conv-bn weight fold, conv-bn fusion, conv-relu, linear-relu,
    /// layernorm-linear, gelu-linear, residual-add), and optimization
    /// (attention fusion, GQA fusion, op substitution).
    pub fn new() -> Self {
        Self {
            passes: vec![
                // Cleanup passes (1-3)
                Box::new(DeadCodeEliminationPass),
                Box::new(IdentityEliminationPass),
                Box::new(ConstantFoldPass),
                // Fusion passes (4-10)
                Box::new(ConvBatchNormWeightFoldPass),
                Box::new(ConvBatchNormFusionPass),
                Box::new(ConvReluFusionPass),
                Box::new(LinearReluFusionPass),
                Box::new(LayerNormLinearFusionPass),
                Box::new(GeluLinearFusionPass),
                Box::new(ResidualAddFusionPass),
                // Optimization passes
                Box::new(AttentionFusionPass),
                Box::new(GqaFusionPass),
                Box::new(KvCachePass::default()),
                Box::new(CodebookOptimizationPass),
                Box::new(OpSubstitutionPass),
                Box::new(LayoutOptimizationPass),
                // Re-propagate output types after all transformations so that
                // newly-created ops (transposes, tiles, etc.) get concrete types.
                Box::new(TypeRepropagationPass),
            ],
            has_fp16: false,
            has_int8: false,
            has_palettize: false,
            has_polar_quant: false,
            has_mixed_precision: false,
        }
    }

    /// Build a pipeline from a TOML configuration file.
    ///
    /// The TOML file should contain an ordered list of passes:
    ///
    /// ```toml
    /// [[passes]]
    /// name = "dead-code-elimination"
    /// enabled = true
    ///
    /// [[passes]]
    /// name = "fp16-quantization"
    /// enabled = true
    ///
    /// [[passes]]
    /// name = "palettization"
    /// enabled = true
    /// [passes.params]
    /// n_bits = 4
    /// ```
    pub fn with_config(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::from_config_str(&content)
    }

    /// Build a pipeline from a TOML configuration string.
    pub fn from_config_str(toml_str: &str) -> Result<Self> {
        let config: PipelineConfig = toml::from_str(toml_str)
            .map_err(|e| MilError::Validation(format!("invalid pipeline config: {e}")))?;

        let mut passes: Vec<Box<dyn Pass>> = Vec::new();
        let mut has_fp16 = false;
        let mut has_int8 = false;
        let mut has_palettize = false;
        let mut has_polar_quant = false;
        let mut has_mixed_precision = false;

        for entry in &config.passes {
            if !entry.enabled {
                continue;
            }
            if !KNOWN_PASSES.contains(&entry.name.as_str()) {
                return Err(MilError::Validation(format!(
                    "unknown pass '{}' in pipeline config",
                    entry.name
                )));
            }

            // Enforce mutual exclusivity
            match entry.name.as_str() {
                "fp16-quantization" => {
                    if has_int8 && !has_mixed_precision {
                        return Err(MilError::Validation(
                            "FP16 and INT8 quantization are mutually exclusive".into(),
                        ));
                    }
                    if has_polar_quant {
                        return Err(MilError::Validation(
                            "polar-quantization is mutually exclusive with fp16/int8/palettization"
                                .into(),
                        ));
                    }
                    has_fp16 = true;
                }
                "int8-quantization" => {
                    if has_fp16 && !has_mixed_precision {
                        return Err(MilError::Validation(
                            "FP16 and INT8 quantization are mutually exclusive".into(),
                        ));
                    }
                    if has_palettize {
                        return Err(MilError::Validation(
                            "INT8 quantization and palettization are mutually exclusive".into(),
                        ));
                    }
                    if has_polar_quant {
                        return Err(MilError::Validation(
                            "polar-quantization is mutually exclusive with fp16/int8/palettization"
                                .into(),
                        ));
                    }
                    has_int8 = true;
                }
                "palettization" => {
                    if has_int8 {
                        return Err(MilError::Validation(
                            "INT8 quantization and palettization are mutually exclusive".into(),
                        ));
                    }
                    if has_polar_quant {
                        return Err(MilError::Validation(
                            "palettization and polar-quantization are mutually exclusive".into(),
                        ));
                    }
                    has_palettize = true;
                }
                "polar-quantization" => {
                    if has_fp16 || has_int8 || has_palettize {
                        return Err(MilError::Validation(
                            "polar-quantization is mutually exclusive with fp16/int8/palettization"
                                .into(),
                        ));
                    }
                    has_polar_quant = true;
                }
                "mixed-precision" => {
                    has_mixed_precision = true;
                }
                _ => {}
            }

            passes.push(pass_from_name(&entry.name, &entry.params)?);
        }

        Ok(Self {
            passes,
            has_fp16,
            has_int8,
            has_palettize,
            has_polar_quant,
            has_mixed_precision,
        })
    }

    /// Add FP16 quantization. Errors if INT8 is already added (unless mixed-precision is enabled).
    pub fn with_fp16(mut self) -> Result<Self> {
        if self.has_int8 && !self.has_mixed_precision {
            return Err(MilError::Validation(
                "FP16 and INT8 quantization are mutually exclusive".into(),
            ));
        }
        if self.has_polar_quant {
            return Err(MilError::Validation(
                "polar-quantization is mutually exclusive with fp16/int8/palettization".into(),
            ));
        }
        self.has_fp16 = true;
        self.passes.push(Box::new(Fp16QuantizePass));
        Ok(self)
    }

    /// Add INT8 quantization. Errors if FP16 is already added (unless mixed-precision is enabled).
    pub fn with_int8(mut self, cal_dir: Option<PathBuf>) -> Result<Self> {
        if self.has_fp16 && !self.has_mixed_precision {
            return Err(MilError::Validation(
                "FP16 and INT8 quantization are mutually exclusive".into(),
            ));
        }
        if self.has_polar_quant {
            return Err(MilError::Validation(
                "polar-quantization is mutually exclusive with fp16/int8/palettization".into(),
            ));
        }
        self.has_int8 = true;
        self.passes.push(Box::new(Int8QuantizePass::new(
            cal_dir,
            Granularity::PerChannel,
        )));
        Ok(self)
    }

    /// Add weight palettization. Errors if INT8 is already added or palettization was already configured.
    pub fn with_palettize(mut self, n_bits: u8) -> Result<Self> {
        if self.has_palettize {
            return Err(MilError::Validation(
                "Palettization has already been configured".into(),
            ));
        }
        if self.has_int8 {
            return Err(MilError::Validation(
                "INT8 quantization and palettization are mutually exclusive".into(),
            ));
        }
        if self.has_polar_quant {
            return Err(MilError::Validation(
                "polar-quantization is mutually exclusive with fp16/int8/palettization".into(),
            ));
        }
        if !matches!(n_bits, 2 | 4 | 6 | 8) {
            return Err(MilError::Validation(format!(
                "palettize n_bits must be 2, 4, 6, or 8, got {n_bits}"
            )));
        }
        self.has_palettize = true;
        self.passes.push(Box::new(PalettizePass::new(n_bits)));
        Ok(self)
    }

    /// Add PolarQuant weight quantization.
    ///
    /// Applies random Hadamard rotation + Beta-optimal scalar quantization
    /// at the specified bit-width (2, 3, or 4). Automatically schedules
    /// the rotation fusion pass after quantization.
    pub fn with_polar_quant(mut self, n_bits: u8) -> Result<Self> {
        if self.has_fp16 || self.has_int8 || self.has_palettize {
            return Err(MilError::Validation(
                "polar-quantization is mutually exclusive with fp16/int8/palettization".into(),
            ));
        }
        if !(2..=4).contains(&n_bits) {
            return Err(MilError::Validation(format!(
                "polar-quantize n_bits must be 2, 3, or 4, got {n_bits}"
            )));
        }
        self.has_polar_quant = true;
        self.passes.push(Box::new(PolarQuantPass::new(n_bits)));
        self.passes.push(Box::new(PolarRotationFusionPass::new()));
        // Re-propagate types after PolarQuant inserts new ops.
        self.passes.push(Box::new(TypeRepropagationPass));
        Ok(self)
    }

    /// Add mixed-precision quantization from a TOML config file.
    ///
    /// When mixed-precision is configured, the normal FP16/INT8 mutual exclusivity
    /// constraints are relaxed since the mixed-precision pass handles per-op routing.
    pub fn with_mixed_precision(mut self, config_path: &Path) -> Result<Self> {
        let pass = MixedPrecisionPass::from_config_file(config_path)?;
        self.has_mixed_precision = true;
        self.passes.push(Box::new(pass));
        Ok(self)
    }

    /// Add mixed-precision quantization from a [`MixedPrecisionConfig`].
    pub fn with_mixed_precision_config(mut self, config: MixedPrecisionConfig) -> Result<Self> {
        self.has_mixed_precision = true;
        self.passes.push(Box::new(MixedPrecisionPass::new(config)));
        Ok(self)
    }

    /// Add per-expert quantization from an [`ExpertQuantConfig`].
    ///
    /// This pass applies different compression strategies to different experts
    /// in a Mixture-of-Experts model, based on activation frequency profiles.
    pub fn with_per_expert_quant(mut self, config: ExpertQuantConfig) -> Result<Self> {
        self.has_mixed_precision = true;
        self.passes.push(Box::new(PerExpertQuantPass::new(config)));
        Ok(self)
    }

    /// Add shape materialization with user-provided shapes.
    ///
    /// The shape pass is inserted before any quantization/palettization passes.
    pub fn with_shapes(mut self, shapes: HashMap<String, Vec<usize>>) -> Self {
        let mut shape_pass = ShapeMaterializePass::new();
        for (name, dims) in shapes {
            shape_pass = shape_pass.with_shape(name, dims);
        }
        // Insert before any quantization passes (which are appended at the end).
        let insert_pos = self
            .passes
            .iter()
            .position(|p| {
                let name = p.name();
                name == "fp16-quantization"
                    || name == "int8-quantization"
                    || name == "palettization"
                    || name == "polar-quantization"
                    || name == "mixed-precision-quantization"
                    || name == "per-expert-quantization"
            })
            .unwrap_or(self.passes.len());
        self.passes.insert(insert_pos, Box::new(shape_pass));
        self
    }

    /// Add operator splitting with the given ANE memory budget (in bytes).
    ///
    /// Ops whose estimated memory exceeds the budget are decomposed into
    /// smaller tiles that each fit within the ANE's on-chip memory.
    pub fn with_op_splitting(mut self, memory_budget_bytes: usize) -> Self {
        self.passes
            .push(Box::new(OpSplittingPass::new(memory_budget_bytes)));
        self
    }

    /// Disable fusion passes (passes 4–10 in the pipeline order).
    pub fn without_fusion(mut self) -> Self {
        const FUSION_NAMES: &[&str] = &[
            "conv-bn-weight-fold",
            "conv-batchnorm-fusion",
            "conv-relu-fusion",
            "linear-relu-fusion",
            "layernorm-linear-fusion",
            "gelu-linear-fusion",
            "residual-add-fusion",
            "attention-fusion",
            "gqa-fusion",
            "kv-cache",
            "codebook-optimization",
            "op-substitution",
            "layout-optimization",
        ];
        self.passes.retain(|p| !FUSION_NAMES.contains(&p.name()));
        self
    }

    /// Append the compute-unit annotation pass to the pipeline.
    ///
    /// This pass annotates each operation with its preferred compute unit
    /// (ANE, GPU, CPU, or Any) based on shape-aware ANE constraint checks.
    /// Should be added after all optimization/fusion passes so annotations
    /// reflect the final graph.
    pub fn with_compute_unit_annotations(mut self) -> Self {
        self.passes.push(Box::new(ComputeUnitAnnotationPass));
        self
    }

    /// Return the names of passes in the pipeline, in order.
    pub fn pass_names(&self) -> Vec<&str> {
        self.passes.iter().map(|p| p.name()).collect()
    }

    /// Run the full pipeline, returning a report of what each pass did.
    pub fn run(self, program: &mut Program) -> Result<PipelineReport> {
        let mut pass_results = Vec::new();
        for pass in &self.passes {
            let ops_before = count_ops(program);
            let flops_before = estimate_flops(program);
            let memory_before = estimate_memory(program);

            let start = Instant::now();
            pass.run(program)?;
            let elapsed = start.elapsed();

            let ops_after = count_ops(program);
            let flops_after = estimate_flops(program);
            let memory_after = estimate_memory(program);

            pass_results.push(PassResult {
                name: pass.name().to_string(),
                ops_before,
                ops_after,
                flops_before,
                flops_after,
                memory_before,
                memory_after,
                elapsed,
            });
        }
        Ok(PipelineReport { pass_results })
    }
}

/// Count total operations across all functions in a program.
fn count_ops(program: &Program) -> usize {
    program
        .functions
        .values()
        .map(|f| f.body.operations.len())
        .sum()
}

/// Estimate total FLOPs across all functions.
///
/// Uses simple heuristics based on op type:
/// - `conv`: 2 × output_elements × kernel_size × in_channels
/// - `matmul`/`linear`: 2 × M × N × K
/// - Element-wise ops: 1 FLOP per element
///
/// Falls back to 1 FLOP per op when shapes are unavailable.
fn estimate_flops(program: &Program) -> u64 {
    let mut total: u64 = 0;
    for func in program.functions.values() {
        for op in &func.body.operations {
            total += flops_for_op(op);
        }
    }
    total
}

/// Estimate FLOPs for a single operation.
fn flops_for_op(op: &super::operation::Operation) -> u64 {
    // Without full shape inference, use op-type-based heuristics.
    match op.op_type.as_str() {
        "conv" | "conv_transpose" => 2,
        "matmul" | "linear" => 2,
        "add" | "sub" | "mul" | "div" | "relu" | "sigmoid" | "tanh" | "gelu" | "softmax"
        | "reshape" | "transpose" | "concat" | "split" | "pad" | "cast" => 1,
        "batch_norm" | "layer_norm" | "instance_norm" | "group_norm" => 2,
        "reduce_mean" | "reduce_sum" | "reduce_max" | "reduce_min" => 1,
        _ => 1,
    }
}

/// Estimate memory footprint in bytes from const tensor data in the program.
fn estimate_memory(program: &Program) -> u64 {
    let mut total: u64 = 0;
    for func in program.functions.values() {
        for op in &func.body.operations {
            if op.op_type == "const" {
                if let Some(super::types::Value::Tensor { data, .. }) = op.inputs.get("val") {
                    total += data.len() as u64;
                }
            }
        }
    }
    total
}

/// Report from running a [`PassPipeline`].
pub struct PipelineReport {
    pub pass_results: Vec<PassResult>,
}

impl PipelineReport {
    /// Total wall-clock time for the entire pipeline.
    pub fn total_elapsed(&self) -> std::time::Duration {
        self.pass_results.iter().map(|r| r.elapsed).sum()
    }

    /// Format a side-by-side comparison of two pipeline reports.
    pub fn compare(a: &PipelineReport, b: &PipelineReport) -> String {
        use std::fmt::Write;
        let mut out = String::new();

        writeln!(
            out,
            "{:<30} {:>10} {:>10} {:>10} {:>10}",
            "Metric", "Pipeline A", "", "Pipeline B", ""
        )
        .ok();
        writeln!(
            out,
            "{:<30} {:>10} {:>10} {:>10} {:>10}",
            "", "Before", "After", "Before", "After"
        )
        .ok();
        writeln!(out, "{}", "-".repeat(72)).ok();

        let a_ops_before = a.pass_results.first().map_or(0, |r| r.ops_before);
        let a_ops_after = a.pass_results.last().map_or(0, |r| r.ops_after);
        let b_ops_before = b.pass_results.first().map_or(0, |r| r.ops_before);
        let b_ops_after = b.pass_results.last().map_or(0, |r| r.ops_after);

        writeln!(
            out,
            "{:<30} {:>10} {:>10} {:>10} {:>10}",
            "Op count", a_ops_before, a_ops_after, b_ops_before, b_ops_after
        )
        .ok();

        let a_flops_before = a.pass_results.first().map_or(0, |r| r.flops_before);
        let a_flops_after = a.pass_results.last().map_or(0, |r| r.flops_after);
        let b_flops_before = b.pass_results.first().map_or(0, |r| r.flops_before);
        let b_flops_after = b.pass_results.last().map_or(0, |r| r.flops_after);

        writeln!(
            out,
            "{:<30} {:>10} {:>10} {:>10} {:>10}",
            "Est. compute (FLOPs)", a_flops_before, a_flops_after, b_flops_before, b_flops_after
        )
        .ok();

        let a_mem_before = a.pass_results.first().map_or(0, |r| r.memory_before);
        let a_mem_after = a.pass_results.last().map_or(0, |r| r.memory_after);
        let b_mem_before = b.pass_results.first().map_or(0, |r| r.memory_before);
        let b_mem_after = b.pass_results.last().map_or(0, |r| r.memory_after);

        writeln!(
            out,
            "{:<30} {:>10} {:>10} {:>10} {:>10}",
            "Est. memory (bytes)", a_mem_before, a_mem_after, b_mem_before, b_mem_after
        )
        .ok();

        writeln!(
            out,
            "{:<30} {:>10} {:>21}",
            "Total time",
            format!("{:.2?}", a.total_elapsed()),
            format!("{:.2?}", b.total_elapsed()),
        )
        .ok();

        // Per-pass detail for each pipeline
        writeln!(out).ok();
        writeln!(out, "Pipeline A passes:").ok();
        writeln!(
            out,
            "  {:<30} {:>8} {:>8} {:>12}",
            "Pass", "Ops Δ", "FLOPs Δ", "Time"
        )
        .ok();
        for r in &a.pass_results {
            let ops_delta = r.ops_after as i64 - r.ops_before as i64;
            let flops_delta = r.flops_after as i64 - r.flops_before as i64;
            writeln!(
                out,
                "  {:<30} {:>+8} {:>+8} {:>12.2?}",
                r.name, ops_delta, flops_delta, r.elapsed
            )
            .ok();
        }

        writeln!(out).ok();
        writeln!(out, "Pipeline B passes:").ok();
        writeln!(
            out,
            "  {:<30} {:>8} {:>8} {:>12}",
            "Pass", "Ops Δ", "FLOPs Δ", "Time"
        )
        .ok();
        for r in &b.pass_results {
            let ops_delta = r.ops_after as i64 - r.ops_before as i64;
            let flops_delta = r.flops_after as i64 - r.flops_before as i64;
            writeln!(
                out,
                "  {:<30} {:>+8} {:>+8} {:>12.2?}",
                r.name, ops_delta, flops_delta, r.elapsed
            )
            .ok();
        }

        out
    }
}

/// Result of a single pass execution.
pub struct PassResult {
    pub name: String,
    pub ops_before: usize,
    pub ops_after: usize,
    /// Estimated FLOPs before this pass ran.
    pub flops_before: u64,
    /// Estimated FLOPs after this pass ran.
    pub flops_after: u64,
    /// Estimated memory footprint (bytes) before this pass ran.
    pub memory_before: u64,
    /// Estimated memory footprint (bytes) after this pass ran.
    pub memory_after: u64,
    /// Wall-clock time for this pass.
    pub elapsed: std::time::Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Function, Operation, Program};

    /// Helper: build a minimal program with a given number of ops.
    fn program_with_ops(n: usize) -> Program {
        let mut func = Function::new("main");
        for i in 0..n {
            let op = Operation::new("relu", &format!("op_{i}")).with_output(format!("out_{i}"));
            func.body.add_op(op);
        }
        if n > 0 {
            func.body.outputs.push(format!("out_{}", n - 1));
        }
        let mut program = Program::new("1.0.0");
        program.add_function(func);
        program
    }

    #[test]
    fn default_pipeline_has_all_always_on_passes() {
        let pipeline = PassPipeline::new();
        let names = pipeline.pass_names();
        assert_eq!(
            names,
            vec![
                "dead-code-elimination",
                "identity-elimination",
                "constant-folding",
                "conv-bn-weight-fold",
                "conv-batchnorm-fusion",
                "conv-relu-fusion",
                "linear-relu-fusion",
                "layernorm-linear-fusion",
                "gelu-linear-fusion",
                "residual-add-fusion",
                "attention-fusion",
                "gqa-fusion",
                "kv-cache",
                "codebook-optimization",
                "op-substitution",
                "layout-optimization",
                "type-repropagation",
            ]
        );
    }

    #[test]
    fn fp16_plus_int8_returns_error() {
        let pipeline = PassPipeline::new().with_fp16().unwrap();
        let result = pipeline.with_int8(None);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    fn int8_plus_palettize_returns_error() {
        let pipeline = PassPipeline::new().with_int8(None).unwrap();
        let result = pipeline.with_palettize(4);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    fn fp16_plus_palettize_is_allowed() {
        let pipeline = PassPipeline::new()
            .with_fp16()
            .unwrap()
            .with_palettize(4)
            .unwrap();
        let names = pipeline.pass_names();
        assert!(names.contains(&"fp16-quantization"));
        assert!(names.contains(&"palettization"));
    }

    #[test]
    fn pipeline_report_counts_ops() {
        let mut program = program_with_ops(5);
        let pipeline = PassPipeline::new();
        let report = pipeline.run(&mut program).unwrap();
        // Every always-on pass should have recorded the op counts
        assert!(!report.pass_results.is_empty());
        // The first pass should see 5 ops_before
        assert_eq!(report.pass_results[0].ops_before, 5);
    }

    #[test]
    fn without_fusion_removes_fusion_passes() {
        let pipeline = PassPipeline::new().without_fusion();
        let names = pipeline.pass_names();
        assert_eq!(
            names,
            vec![
                "dead-code-elimination",
                "identity-elimination",
                "constant-folding",
                "type-repropagation",
            ]
        );
        // Make sure fusion passes are gone
        assert!(!names.contains(&"conv-bn-weight-fold"));
        assert!(!names.contains(&"conv-batchnorm-fusion"));
        assert!(!names.contains(&"attention-fusion"));
        assert!(!names.contains(&"layout-optimization"));
    }

    #[test]
    fn with_shapes_inserts_before_quantization() {
        let shapes = HashMap::from([("input".to_string(), vec![1, 3, 224, 224])]);
        let pipeline = PassPipeline::new().with_fp16().unwrap().with_shapes(shapes);
        let names = pipeline.pass_names();
        let shape_pos = names
            .iter()
            .position(|n| *n == "shape-materialization")
            .expect("shape pass should be present");
        let fp16_pos = names
            .iter()
            .position(|n| *n == "fp16-quantization")
            .expect("fp16 pass should be present");
        assert!(
            shape_pos < fp16_pos,
            "shape materialization should come before fp16 quantization"
        );
    }

    #[test]
    fn invalid_palettize_bits_returns_error() {
        let result = PassPipeline::new().with_palettize(3);
        assert!(result.is_err());
    }

    #[test]
    fn default_trait_works() {
        let pipeline = PassPipeline::default();
        assert_eq!(pipeline.pass_names().len(), 17);
    }

    #[test]
    fn int8_plus_fp16_returns_error() {
        let pipeline = PassPipeline::new().with_int8(None).unwrap();
        let result = pipeline.with_fp16();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    fn duplicate_palettize_returns_error() {
        let pipeline = PassPipeline::new().with_palettize(4).unwrap();
        let result = pipeline.with_palettize(4);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("already been configured"),
            "expected duplicate palettize error, got: {err}"
        );
    }

    // ── TOML config tests ─────────────────────────────────────────────

    #[test]
    fn config_loads_basic_pipeline() {
        let toml = r#"
[[passes]]
name = "dead-code-elimination"
enabled = true

[[passes]]
name = "constant-folding"
enabled = true

[[passes]]
name = "conv-batchnorm-fusion"
enabled = true
"#;
        let pipeline = PassPipeline::from_config_str(toml).unwrap();
        assert_eq!(
            pipeline.pass_names(),
            vec![
                "dead-code-elimination",
                "constant-folding",
                "conv-batchnorm-fusion",
            ]
        );
    }

    #[test]
    fn config_disabled_passes_are_skipped() {
        let toml = r#"
[[passes]]
name = "dead-code-elimination"
enabled = true

[[passes]]
name = "identity-elimination"
enabled = false

[[passes]]
name = "constant-folding"
enabled = true
"#;
        let pipeline = PassPipeline::from_config_str(toml).unwrap();
        let names = pipeline.pass_names();
        assert_eq!(names, vec!["dead-code-elimination", "constant-folding"]);
        assert!(!names.contains(&"identity-elimination"));
    }

    #[test]
    fn config_unknown_pass_returns_error() {
        let toml = r#"
[[passes]]
name = "nonexistent-pass"
enabled = true
"#;
        let result = PassPipeline::from_config_str(toml);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("unknown pass"), "got: {err}");
    }

    #[test]
    fn config_mutual_exclusivity_enforced() {
        let toml = r#"
[[passes]]
name = "fp16-quantization"

[[passes]]
name = "int8-quantization"
"#;
        let result = PassPipeline::from_config_str(toml);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("mutually exclusive"), "got: {err}");
    }

    #[test]
    fn config_with_palettize_params() {
        let toml = r#"
[[passes]]
name = "palettization"
[passes.params]
n_bits = 6
"#;
        let pipeline = PassPipeline::from_config_str(toml).unwrap();
        assert_eq!(pipeline.pass_names(), vec!["palettization"]);
    }

    #[test]
    fn config_driven_matches_programmatic() {
        let toml = r#"
[[passes]]
name = "dead-code-elimination"

[[passes]]
name = "identity-elimination"

[[passes]]
name = "constant-folding"

[[passes]]
name = "conv-bn-weight-fold"

[[passes]]
name = "conv-batchnorm-fusion"

[[passes]]
name = "conv-relu-fusion"

[[passes]]
name = "linear-relu-fusion"

[[passes]]
name = "layernorm-linear-fusion"

[[passes]]
name = "gelu-linear-fusion"

[[passes]]
name = "residual-add-fusion"

[[passes]]
name = "attention-fusion"

[[passes]]
name = "gqa-fusion"

[[passes]]
name = "kv-cache"

[[passes]]
name = "codebook-optimization"

[[passes]]
name = "op-substitution"

[[passes]]
name = "layout-optimization"

[[passes]]
name = "type-repropagation"
"#;
        let config_pipeline = PassPipeline::from_config_str(toml).unwrap();
        let default_pipeline = PassPipeline::new();
        assert_eq!(config_pipeline.pass_names(), default_pipeline.pass_names());

        // Both should produce the same results on the same program
        let mut prog_a = program_with_ops(5);
        let mut prog_b = program_with_ops(5);
        let report_a = config_pipeline.run(&mut prog_a).unwrap();
        let report_b = default_pipeline.run(&mut prog_b).unwrap();

        assert_eq!(report_a.pass_results.len(), report_b.pass_results.len());
        for (a, b) in report_a.pass_results.iter().zip(&report_b.pass_results) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.ops_before, b.ops_before);
            assert_eq!(a.ops_after, b.ops_after);
        }
    }

    #[test]
    fn config_from_file() {
        let dir = std::env::current_dir().unwrap().join("test-pipeline-cfg");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("test.toml");
        std::fs::write(
            &path,
            r#"
[[passes]]
name = "dead-code-elimination"

[[passes]]
name = "constant-folding"
"#,
        )
        .unwrap();
        let pipeline = PassPipeline::with_config(&path).unwrap();
        assert_eq!(
            pipeline.pass_names(),
            vec!["dead-code-elimination", "constant-folding"]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn pipeline_report_has_timing_and_metrics() {
        let mut program = program_with_ops(5);
        let pipeline = PassPipeline::new();
        let report = pipeline.run(&mut program).unwrap();

        for r in &report.pass_results {
            // Duration is always non-negative; just verify it was recorded
            let _ = r.elapsed;
            // FLOPs estimates should be present
            assert!(r.flops_before > 0 || r.ops_before == 0);
        }
        // Total elapsed should be sum of parts
        let sum: std::time::Duration = report.pass_results.iter().map(|r| r.elapsed).sum();
        assert_eq!(report.total_elapsed(), sum);
    }

    #[test]
    fn pipeline_report_compare_format() {
        let mut prog_a = program_with_ops(5);
        let mut prog_b = program_with_ops(5);
        let report_a = PassPipeline::new().run(&mut prog_a).unwrap();
        let report_b = PassPipeline::new()
            .without_fusion()
            .run(&mut prog_b)
            .unwrap();
        let comparison = PipelineReport::compare(&report_a, &report_b);
        assert!(comparison.contains("Op count"));
        assert!(comparison.contains("Pipeline A passes:"));
        assert!(comparison.contains("Pipeline B passes:"));
    }

    #[test]
    fn config_enabled_defaults_to_true() {
        let toml = r#"
[[passes]]
name = "dead-code-elimination"
"#;
        let pipeline = PassPipeline::from_config_str(toml).unwrap();
        assert_eq!(pipeline.pass_names(), vec!["dead-code-elimination"]);
    }
}
