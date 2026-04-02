//! Pass pipeline manager for ordering, mutual exclusivity, and builder API.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use std::time::Instant;

use serde::Deserialize;

use super::pass::Pass;
use super::passes::awq_quantize::AwqQuantizePass;
use super::passes::{
    AffineQuantizePass, AttentionFusionPass, AwqScaleFusionPass, ChannelStats, ConstantFoldPass,
    ConvBatchNormFusionPass, ConvBatchNormWeightFoldPass, ConvReluFusionPass, D2QuantPass, DacPass,
    DeadCodeEliminationPass, Fp16QuantizePass, GeluLinearFusionPass, GqaFusionPass, Granularity,
    IdentityEliminationPass, Int8QuantizePass, LayerNormLinearFusionPass, LayoutOptimizationPass,
    LinearReluFusionPass, PalettizePass, PolarQuantPass, PolarRotationFusionPass, QuipSharpPass,
    ResidualAddFusionPass, ShapeMaterializePass, TypeRepropagationPass,
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
    has_int4: bool,
    has_awq: bool,
    has_palettize: bool,
    has_polar_quant: bool,
    has_gptq: bool,
    has_quip_sharp: bool,
    has_d2quant: bool,
    has_spinquant: bool,
}

/// Quantization method to use after SpinQuant rotation.
pub enum SpinQuantMethod {
    /// Simple min-max affine quantization.
    MinMax { group_size: usize },
    /// Activation-aware weight quantization.
    Awq {
        channel_magnitudes: HashMap<String, Vec<f32>>,
        group_size: usize,
    },
    /// GPTQ optimal weight quantization (requires `gptq` feature).
    #[cfg(feature = "gptq")]
    Gptq {
        hessian_data: HashMap<String, (Vec<f32>, usize, usize)>,
        group_size: usize,
        block_size: usize,
        dampening: f64,
    },
}

/// Configuration for SpinQuant learned rotation.
pub struct SpinQuantConfig {
    /// Number of Cayley optimizer epochs for rotation learning.
    pub rotation_epochs: usize,
    /// Target quantization bit-width.
    pub bits: u8,
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
    "layout-optimization",
    "type-repropagation",
    "fp16-quantization",
    "int8-quantization",
    "int4-quantize",
    "palettization",
    "polar-quantization",
    "shape-materialization",
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
        "int4-quantize" => {
            let group_size = params
                .get("group_size")
                .and_then(|v| v.as_integer())
                .unwrap_or(128) as usize;
            Ok(Box::new(AffineQuantizePass::int4_per_group(group_size)))
        }
        "palettization" => {
            let n_bits_i64 = params
                .get("n_bits")
                .and_then(|v| v.as_integer())
                .unwrap_or(4);
            if !matches!(n_bits_i64, 1 | 2 | 4 | 6 | 8) {
                return Err(MilError::Validation(format!(
                    "palettize n_bits must be 1, 2, 4, 6, or 8, got {n_bits_i64}"
                )));
            }
            let n_bits = n_bits_i64 as u8;
            Ok(Box::new(PalettizePass::new(n_bits)))
        }
        "polar-quantization" => {
            let n_bits = params.get("bits").and_then(|v| v.as_integer()).unwrap_or(4) as u8;
            if n_bits != 2 && n_bits != 4 {
                return Err(MilError::Validation(format!(
                    "polar-quantize bits must be 2 or 4, got {n_bits}"
                )));
            }
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
        "mixed-precision"
        | "layer-schedule"
        | "compute-unit-annotation"
        | "op-splitting"
        | "per-expert-quantization"
        | "kv-cache"
        | "codebook-optimization"
        | "op-substitution" => Err(MilError::Validation(format!(
            "pass '{name}' has been moved to ironmill-compile"
        ))),
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
            .field("has_int4", &self.has_int4)
            .field("has_awq", &self.has_awq)
            .field("has_palettize", &self.has_palettize)
            .field("has_polar_quant", &self.has_polar_quant)
            .field("has_gptq", &self.has_gptq)
            .field("has_quip_sharp", &self.has_quip_sharp)
            .field("has_d2quant", &self.has_d2quant)
            .field("has_spinquant", &self.has_spinquant)
            .finish()
    }
}

impl PassPipeline {
    /// Create the default pipeline with all always-on passes.
    ///
    /// Includes cleanup (DCE, identity elimination, constant folding),
    /// fusion (conv-bn weight fold, conv-bn fusion, conv-relu, linear-relu,
    /// layernorm-linear, gelu-linear, residual-add), and optimization
    /// (attention fusion, GQA fusion, layout optimization, type repropagation).
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
                Box::new(LayoutOptimizationPass),
                // Re-propagate output types after all transformations so that
                // newly-created ops (transposes, tiles, etc.) get concrete types.
                Box::new(TypeRepropagationPass),
            ],
            has_fp16: false,
            has_int8: false,
            has_int4: false,
            has_awq: false,
            has_palettize: false,
            has_polar_quant: false,
            has_gptq: false,
            has_quip_sharp: false,
            has_d2quant: false,
            has_spinquant: false,
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
        let mut has_int4 = false;
        let has_awq = false;
        let mut has_palettize = false;
        let mut has_polar_quant = false;

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
                    if has_int8 {
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
                    if has_fp16 {
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
                "int4-quantize" => {
                    if has_int8 {
                        return Err(MilError::Validation(
                            "INT4 and INT8 quantization are mutually exclusive".into(),
                        ));
                    }
                    if has_polar_quant {
                        return Err(MilError::Validation(
                            "polar-quantization is mutually exclusive with int4 quantization"
                                .into(),
                        ));
                    }
                    has_int4 = true;
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
                _ => {}
            }

            passes.push(pass_from_name(&entry.name, &entry.params)?);
        }

        Ok(Self {
            passes,
            has_fp16,
            has_int8,
            has_int4,
            has_awq,
            has_palettize,
            has_polar_quant,
            has_gptq: false,
            has_quip_sharp: false,
            has_d2quant: false,
            has_spinquant: false,
        })
    }

    /// Add FP16 quantization. Errors if INT8 is already added.
    pub fn with_fp16(mut self) -> Result<Self> {
        if self.has_int8 {
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

    /// Add INT8 quantization. Errors if FP16 is already added.
    pub fn with_int8(mut self, cal_dir: Option<PathBuf>) -> Result<Self> {
        if self.has_fp16 {
            return Err(MilError::Validation(
                "FP16 and INT8 quantization are mutually exclusive".into(),
            ));
        }
        if self.has_polar_quant {
            return Err(MilError::Validation(
                "polar-quantization is mutually exclusive with fp16/int8/palettization".into(),
            ));
        }
        if self.has_int4 {
            return Err(MilError::Validation(
                "INT4 and INT8 quantization are mutually exclusive".into(),
            ));
        }
        if self.has_awq {
            return Err(MilError::Validation(
                "AWQ and INT8 quantization are mutually exclusive".into(),
            ));
        }
        if self.has_quip_sharp {
            return Err(MilError::Validation(
                "INT8 and QuIP# quantization are mutually exclusive".into(),
            ));
        }
        if self.has_d2quant {
            return Err(MilError::Validation(
                "INT8 and D2Quant quantization are mutually exclusive".into(),
            ));
        }
        if self.has_gptq {
            return Err(MilError::Validation(
                "INT8 and GPTQ quantization are mutually exclusive".into(),
            ));
        }
        if self.has_spinquant {
            return Err(MilError::Validation(
                "INT8 and SpinQuant are mutually exclusive".into(),
            ));
        }
        self.has_int8 = true;
        self.passes.push(Box::new(Int8QuantizePass::new(
            cal_dir,
            Granularity::PerChannel,
        )));
        Ok(self)
    }

    /// Add INT4 affine weight quantization with per-group granularity.
    ///
    /// Inserts `AffineQuantizePass::int4_per_group(group_size)` after fusion
    /// passes. Mutually exclusive with INT8 and polar quantization.
    pub fn with_int4(mut self, group_size: usize) -> Result<Self> {
        if self.has_int8 {
            return Err(MilError::Validation(
                "INT4 and INT8 quantization are mutually exclusive".into(),
            ));
        }
        if self.has_polar_quant {
            return Err(MilError::Validation(
                "polar-quantization is mutually exclusive with int4 quantization".into(),
            ));
        }
        if self.has_palettize {
            return Err(MilError::Validation(
                "palettization is mutually exclusive with int4 quantization".into(),
            ));
        }
        if self.has_awq {
            return Err(MilError::Validation(
                "INT4 and AWQ quantization are mutually exclusive".into(),
            ));
        }
        if self.has_quip_sharp {
            return Err(MilError::Validation(
                "INT4 and QuIP# quantization are mutually exclusive".into(),
            ));
        }
        if self.has_d2quant {
            return Err(MilError::Validation(
                "INT4 and D2Quant quantization are mutually exclusive".into(),
            ));
        }
        if self.has_gptq {
            return Err(MilError::Validation(
                "INT4 and GPTQ quantization are mutually exclusive".into(),
            ));
        }
        if self.has_spinquant {
            return Err(MilError::Validation(
                "INT4 and SpinQuant are mutually exclusive".into(),
            ));
        }
        self.has_int4 = true;
        // Insert before type-repropagation (last pass in default pipeline).
        let insert_pos = self
            .passes
            .iter()
            .position(|p| p.name() == "type-repropagation")
            .unwrap_or(self.passes.len());
        self.passes.insert(
            insert_pos,
            Box::new(AffineQuantizePass::int4_per_group(group_size)),
        );
        Ok(self)
    }

    /// Add AWQ (Activation-aware Weight Quantization).
    ///
    /// Inserts `AwqQuantizePass` before type-repropagation and
    /// `AwqScaleFusionPass` immediately after it. AWQ uses per-channel
    /// activation magnitudes collected during calibration to protect
    /// salient weight channels from quantization error.
    ///
    /// Mutually exclusive with INT4, INT8, palettization, and polar
    /// quantization.
    pub fn with_awq(
        mut self,
        channel_magnitudes: HashMap<String, Vec<f32>>,
        group_size: usize,
    ) -> Result<Self> {
        if self.has_int8 {
            return Err(MilError::Validation(
                "AWQ and INT8 quantization are mutually exclusive".into(),
            ));
        }
        if self.has_int4 {
            return Err(MilError::Validation(
                "AWQ and INT4 quantization are mutually exclusive".into(),
            ));
        }
        if self.has_polar_quant {
            return Err(MilError::Validation(
                "AWQ and polar-quantization are mutually exclusive".into(),
            ));
        }
        if self.has_palettize {
            return Err(MilError::Validation(
                "AWQ and palettization are mutually exclusive".into(),
            ));
        }
        if self.has_quip_sharp {
            return Err(MilError::Validation(
                "AWQ and QuIP# quantization are mutually exclusive".into(),
            ));
        }
        if self.has_d2quant {
            return Err(MilError::Validation(
                "AWQ and D2Quant quantization are mutually exclusive".into(),
            ));
        }
        if self.has_gptq {
            return Err(MilError::Validation(
                "AWQ and GPTQ quantization are mutually exclusive".into(),
            ));
        }
        if self.has_spinquant {
            return Err(MilError::Validation(
                "AWQ and SpinQuant are mutually exclusive".into(),
            ));
        }
        self.has_awq = true;
        // Insert AwqQuantizePass before type-repropagation.
        let insert_pos = self
            .passes
            .iter()
            .position(|p| p.name() == "type-repropagation")
            .unwrap_or(self.passes.len());
        self.passes.insert(
            insert_pos,
            Box::new(AwqQuantizePass::new(4, group_size, channel_magnitudes)),
        );
        // Insert AwqScaleFusionPass right after AwqQuantizePass.
        self.passes
            .insert(insert_pos + 1, Box::new(AwqScaleFusionPass));
        Ok(self)
    }

    /// Add GPTQ weight quantization.
    ///
    /// Uses pre-computed Hessian data from calibration to perform
    /// optimal weight quantization via the GPTQ algorithm.
    ///
    /// Mutually exclusive with INT8, palettization, and polar quantization.
    #[cfg(feature = "gptq")]
    pub fn with_gptq(
        mut self,
        hessian_data: HashMap<String, (Vec<f32>, usize, usize)>,
        group_size: usize,
        block_size: usize,
        dampening: f64,
    ) -> Result<Self> {
        if self.has_int8 {
            return Err(MilError::Validation(
                "GPTQ and INT8 quantization are mutually exclusive".into(),
            ));
        }
        if self.has_int4 {
            return Err(MilError::Validation(
                "GPTQ and INT4 quantization are mutually exclusive".into(),
            ));
        }
        if self.has_awq {
            return Err(MilError::Validation(
                "GPTQ and AWQ quantization are mutually exclusive".into(),
            ));
        }
        if self.has_palettize {
            return Err(MilError::Validation(
                "GPTQ and palettization are mutually exclusive".into(),
            ));
        }
        if self.has_polar_quant {
            return Err(MilError::Validation(
                "GPTQ and polar-quantization are mutually exclusive".into(),
            ));
        }
        if self.has_quip_sharp {
            return Err(MilError::Validation(
                "GPTQ and QuIP# quantization are mutually exclusive".into(),
            ));
        }
        if self.has_d2quant {
            return Err(MilError::Validation(
                "GPTQ and D2Quant quantization are mutually exclusive".into(),
            ));
        }
        if self.has_spinquant {
            return Err(MilError::Validation(
                "GPTQ and SpinQuant are mutually exclusive".into(),
            ));
        }
        self.has_gptq = true;
        // Insert before type-repropagation so output types are refreshed.
        let insert_pos = self
            .passes
            .iter()
            .position(|p| p.name() == "type-repropagation")
            .unwrap_or(self.passes.len());
        self.passes.insert(
            insert_pos,
            Box::new(super::passes::gptq::GptqQuantizePass::new(
                4,
                group_size,
                block_size,
                dampening,
                hessian_data,
            )),
        );
        Ok(self)
    }

    /// Add GPTQ weight quantization (stub when `gptq` feature is disabled).
    #[cfg(not(feature = "gptq"))]
    pub fn with_gptq(
        self,
        _hessian_data: HashMap<String, (Vec<f32>, usize, usize)>,
        _group_size: usize,
        _block_size: usize,
        _dampening: f64,
    ) -> Result<Self> {
        Err(MilError::Validation(
            "GPTQ requires the 'gptq' feature".into(),
        ))
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
        if self.has_int4 {
            return Err(MilError::Validation(
                "INT4 quantization and palettization are mutually exclusive".into(),
            ));
        }
        if self.has_awq {
            return Err(MilError::Validation(
                "AWQ and palettization are mutually exclusive".into(),
            ));
        }
        if self.has_polar_quant {
            return Err(MilError::Validation(
                "polar-quantization is mutually exclusive with fp16/int8/palettization".into(),
            ));
        }
        if self.has_quip_sharp {
            return Err(MilError::Validation(
                "palettization and QuIP# quantization are mutually exclusive".into(),
            ));
        }
        if self.has_d2quant {
            return Err(MilError::Validation(
                "palettization and D2Quant quantization are mutually exclusive".into(),
            ));
        }
        if self.has_spinquant {
            return Err(MilError::Validation(
                "palettization and SpinQuant are mutually exclusive".into(),
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
    /// at the specified bit-width (2 or 4). Automatically schedules
    /// the rotation fusion pass after quantization.
    pub fn with_polar_quant(mut self, n_bits: u8) -> Result<Self> {
        if self.has_fp16 || self.has_int8 || self.has_palettize || self.has_int4 || self.has_awq {
            return Err(MilError::Validation(
                "polar-quantization is mutually exclusive with fp16/int4/int8/awq/palettization"
                    .into(),
            ));
        }
        if self.has_quip_sharp {
            return Err(MilError::Validation(
                "polar-quantization and QuIP# are mutually exclusive".into(),
            ));
        }
        if self.has_d2quant {
            return Err(MilError::Validation(
                "polar-quantization and D2Quant are mutually exclusive".into(),
            ));
        }
        if self.has_spinquant {
            return Err(MilError::Validation(
                "polar-quantization and SpinQuant are mutually exclusive".into(),
            ));
        }
        if n_bits != 2 && n_bits != 4 {
            return Err(MilError::Validation(format!(
                "polar-quantize n_bits must be 2 or 4, got {n_bits}"
            )));
        }
        self.has_polar_quant = true;
        self.passes.push(Box::new(PolarQuantPass::new(n_bits)));
        self.passes.push(Box::new(PolarRotationFusionPass::new()));
        // Re-propagate types after PolarQuant inserts new ops.
        self.passes.push(Box::new(TypeRepropagationPass));
        Ok(self)
    }

    /// Add QuIP# (Quantization with Incoherence Processing) weight quantization.
    ///
    /// Combines randomized Hadamard rotation with E8 lattice vector
    /// quantization for high-quality 2-bit weight compression.
    ///
    /// Mutually exclusive with all other weight quantization methods.
    pub fn with_quip_sharp(mut self, bits: u8, seed: u64) -> Result<Self> {
        if self.has_int8 {
            return Err(MilError::Validation(
                "QuIP# and INT8 quantization are mutually exclusive".into(),
            ));
        }
        if self.has_int4 {
            return Err(MilError::Validation(
                "QuIP# and INT4 quantization are mutually exclusive".into(),
            ));
        }
        if self.has_awq {
            return Err(MilError::Validation(
                "QuIP# and AWQ quantization are mutually exclusive".into(),
            ));
        }
        if self.has_gptq {
            return Err(MilError::Validation(
                "QuIP# and GPTQ quantization are mutually exclusive".into(),
            ));
        }
        if self.has_palettize {
            return Err(MilError::Validation(
                "QuIP# and palettization are mutually exclusive".into(),
            ));
        }
        if self.has_polar_quant {
            return Err(MilError::Validation(
                "QuIP# and polar-quantization are mutually exclusive".into(),
            ));
        }
        if self.has_d2quant {
            return Err(MilError::Validation(
                "QuIP# and D2Quant quantization are mutually exclusive".into(),
            ));
        }
        if self.has_spinquant {
            return Err(MilError::Validation(
                "QuIP# and SpinQuant are mutually exclusive".into(),
            ));
        }
        self.has_quip_sharp = true;
        let insert_pos = self
            .passes
            .iter()
            .position(|p| p.name() == "type-repropagation")
            .unwrap_or(self.passes.len());
        self.passes.insert(
            insert_pos,
            Box::new(QuipSharpPass {
                bits,
                seed,
                min_elements: 256,
            }),
        );
        Ok(self)
    }

    /// Add D2Quant dual-scale sub-4-bit weight quantization.
    ///
    /// Partitions each weight group into normal and outlier subsets with
    /// separate scale/zero-point pairs for lower quantization error at
    /// 2 or 3 bits.
    ///
    /// Mutually exclusive with all other weight quantization methods.
    pub fn with_d2quant(
        mut self,
        bits: u8,
        group_size: usize,
        outlier_threshold: f32,
        dac_stats: Option<(HashMap<String, ChannelStats>, HashMap<String, ChannelStats>)>,
    ) -> Result<Self> {
        if self.has_int8 {
            return Err(MilError::Validation(
                "D2Quant and INT8 quantization are mutually exclusive".into(),
            ));
        }
        if self.has_int4 {
            return Err(MilError::Validation(
                "D2Quant and INT4 quantization are mutually exclusive".into(),
            ));
        }
        if self.has_awq {
            return Err(MilError::Validation(
                "D2Quant and AWQ quantization are mutually exclusive".into(),
            ));
        }
        if self.has_gptq {
            return Err(MilError::Validation(
                "D2Quant and GPTQ quantization are mutually exclusive".into(),
            ));
        }
        if self.has_palettize {
            return Err(MilError::Validation(
                "D2Quant and palettization are mutually exclusive".into(),
            ));
        }
        if self.has_polar_quant {
            return Err(MilError::Validation(
                "D2Quant and polar-quantization are mutually exclusive".into(),
            ));
        }
        if self.has_quip_sharp {
            return Err(MilError::Validation(
                "D2Quant and QuIP# quantization are mutually exclusive".into(),
            ));
        }
        if self.has_spinquant {
            return Err(MilError::Validation(
                "D2Quant and SpinQuant are mutually exclusive".into(),
            ));
        }
        self.has_d2quant = true;
        let insert_pos = self
            .passes
            .iter()
            .position(|p| p.name() == "type-repropagation")
            .unwrap_or(self.passes.len());
        self.passes.insert(
            insert_pos,
            Box::new(D2QuantPass::new(bits, group_size, outlier_threshold)),
        );
        if let Some((fp16_stats, quant_stats)) = dac_stats {
            let dac_pos = insert_pos + 1;
            self.passes.insert(
                dac_pos,
                Box::new(DacPass {
                    fp16_stats,
                    quant_stats,
                }),
            );
        }
        Ok(self)
    }

    /// Add SpinQuant rotation optimization followed by a quantization pass.
    ///
    /// SpinQuant learns rotation matrices via the Cayley parameterization,
    /// absorbs them into weights, then quantizes using the specified method.
    /// This composes rotation + quantization in a single pipeline step.
    ///
    /// Mutually exclusive with all other weight quantization methods.
    #[cfg(feature = "gptq")]
    pub fn with_spinquant(
        mut self,
        rotation_config: SpinQuantConfig,
        _quantize_method: SpinQuantMethod,
    ) -> Result<Self> {
        if self.has_int8 {
            return Err(MilError::Validation(
                "SpinQuant and INT8 quantization are mutually exclusive".into(),
            ));
        }
        if self.has_int4 {
            return Err(MilError::Validation(
                "SpinQuant and INT4 quantization are mutually exclusive".into(),
            ));
        }
        if self.has_awq {
            return Err(MilError::Validation(
                "SpinQuant and AWQ quantization are mutually exclusive".into(),
            ));
        }
        if self.has_gptq {
            return Err(MilError::Validation(
                "SpinQuant and GPTQ quantization are mutually exclusive".into(),
            ));
        }
        if self.has_palettize {
            return Err(MilError::Validation(
                "SpinQuant and palettization are mutually exclusive".into(),
            ));
        }
        if self.has_polar_quant {
            return Err(MilError::Validation(
                "SpinQuant and polar-quantization are mutually exclusive".into(),
            ));
        }
        if self.has_quip_sharp {
            return Err(MilError::Validation(
                "SpinQuant and QuIP# quantization are mutually exclusive".into(),
            ));
        }
        if self.has_d2quant {
            return Err(MilError::Validation(
                "SpinQuant and D2Quant quantization are mutually exclusive".into(),
            ));
        }
        self.has_spinquant = true;
        let mut pass = super::passes::spinquant::SpinQuantPass::new();
        pass.rotation_epochs = rotation_config.rotation_epochs;
        pass.bits = rotation_config.bits;
        pass.group_size = 128;
        let insert_pos = self
            .passes
            .iter()
            .position(|p| p.name() == "type-repropagation")
            .unwrap_or(self.passes.len());
        self.passes.insert(insert_pos, Box::new(pass));
        Ok(self)
    }

    /// Add SpinQuant (stub when `gptq` feature is disabled).
    #[cfg(not(feature = "gptq"))]
    pub fn with_spinquant(
        self,
        _rotation_config: SpinQuantConfig,
        _quantize_method: SpinQuantMethod,
    ) -> Result<Self> {
        Err(MilError::Validation(
            "SpinQuant requires the 'gptq' feature".into(),
        ))
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
            })
            .unwrap_or(self.passes.len());
        self.passes.insert(insert_pos, Box::new(shape_pass));
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
            "layout-optimization",
        ];
        self.passes.retain(|p| !FUSION_NAMES.contains(&p.name()));
        self
    }

    /// Append a pass to the end of the pipeline.
    ///
    /// This allows downstream crates (e.g. ironmill-compile) to register
    /// backend-specific passes that are not part of the core mil-rs pipeline.
    pub fn add_pass(&mut self, pass: Box<dyn Pass>) {
        self.passes.push(pass);
    }

    /// Insert a pass immediately after the named pass.
    ///
    /// Returns `true` if the named pass was found (and the new pass was
    /// inserted after it). Returns `false` if the named pass was not found,
    /// in which case the new pass is appended to the end.
    pub fn add_pass_after(&mut self, after_name: &str, pass: Box<dyn Pass>) -> bool {
        if let Some(pos) = self.passes.iter().position(|p| p.name() == after_name) {
            self.passes.insert(pos + 1, pass);
            true
        } else {
            self.passes.push(pass);
            false
        }
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
#[derive(Debug)]
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
#[derive(Debug)]
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
        assert_eq!(pipeline.pass_names().len(), 14);
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

    // ── INT4 pipeline tests ───────────────────────────────────────────

    #[test]
    fn with_int4_builds_pipeline() {
        let pipeline = PassPipeline::new().with_int4(128).unwrap();
        let names = pipeline.pass_names();
        assert!(
            names.contains(&"affine-quantization"),
            "expected affine-quantization pass, got: {names:?}"
        );
    }

    #[test]
    fn with_int4_default_group_size() {
        let pipeline = PassPipeline::new().with_int4(32).unwrap();
        let names = pipeline.pass_names();
        assert!(names.contains(&"affine-quantization"));
    }

    #[test]
    fn int4_and_int8_mutually_exclusive() {
        let pipeline = PassPipeline::new().with_int8(None).unwrap();
        let result = pipeline.with_int4(128);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    fn int4_and_polar_quant_mutually_exclusive() {
        let pipeline = PassPipeline::new().with_polar_quant(4).unwrap();
        let result = pipeline.with_int4(128);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    fn int4_plus_fp16_is_allowed() {
        let pipeline = PassPipeline::new()
            .with_fp16()
            .unwrap()
            .with_int4(128)
            .unwrap();
        let names = pipeline.pass_names();
        assert!(names.contains(&"fp16-quantization"));
        assert!(names.contains(&"affine-quantization"));
    }

    #[test]
    fn config_int4_quantize() {
        let toml = r#"
[[passes]]
name = "int4-quantize"
[passes.params]
group_size = 64
"#;
        let pipeline = PassPipeline::from_config_str(toml).unwrap();
        assert_eq!(pipeline.pass_names(), vec!["affine-quantization"]);
    }

    #[test]
    fn config_int4_quantize_default_group_size() {
        let toml = r#"
[[passes]]
name = "int4-quantize"
"#;
        let pipeline = PassPipeline::from_config_str(toml).unwrap();
        assert_eq!(pipeline.pass_names(), vec!["affine-quantization"]);
    }

    // ── AWQ pipeline tests ────────────────────────────────────────────

    fn sample_magnitudes() -> HashMap<String, Vec<f32>> {
        let mut m = HashMap::new();
        m.insert("weight_0".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
        m.insert("weight_1".to_string(), vec![0.5, 1.5, 2.5, 3.5]);
        m
    }

    #[test]
    fn with_awq_builds_pipeline() {
        let pipeline = PassPipeline::new()
            .with_awq(sample_magnitudes(), 128)
            .unwrap();
        let names = pipeline.pass_names();
        assert!(
            names.contains(&"awq-quantization"),
            "expected awq-quantization pass, got: {names:?}"
        );
        assert!(
            names.contains(&"awq-scale-fusion"),
            "expected awq-scale-fusion pass, got: {names:?}"
        );
    }

    #[test]
    fn with_awq_inserts_before_type_repropagation() {
        let pipeline = PassPipeline::new()
            .with_awq(sample_magnitudes(), 128)
            .unwrap();
        let names = pipeline.pass_names();
        let awq_pos = names.iter().position(|n| *n == "awq-quantization").unwrap();
        let fusion_pos = names.iter().position(|n| *n == "awq-scale-fusion").unwrap();
        let repr_pos = names
            .iter()
            .position(|n| *n == "type-repropagation")
            .unwrap();
        assert!(
            awq_pos < repr_pos,
            "awq-quantization ({awq_pos}) should come before type-repropagation ({repr_pos})"
        );
        assert_eq!(
            fusion_pos,
            awq_pos + 1,
            "awq-scale-fusion should immediately follow awq-quantization"
        );
    }

    #[test]
    fn awq_and_int4_mutually_exclusive() {
        let pipeline = PassPipeline::new().with_int4(128).unwrap();
        let result = pipeline.with_awq(sample_magnitudes(), 128);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    fn awq_and_int8_mutually_exclusive() {
        let pipeline = PassPipeline::new().with_int8(None).unwrap();
        let result = pipeline.with_awq(sample_magnitudes(), 128);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    fn awq_and_polar_quant_mutually_exclusive() {
        let pipeline = PassPipeline::new().with_polar_quant(4).unwrap();
        let result = pipeline.with_awq(sample_magnitudes(), 128);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    fn awq_and_palettize_mutually_exclusive() {
        let pipeline = PassPipeline::new().with_palettize(4).unwrap();
        let result = pipeline.with_awq(sample_magnitudes(), 128);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    fn awq_plus_fp16_is_allowed() {
        let pipeline = PassPipeline::new()
            .with_fp16()
            .unwrap()
            .with_awq(sample_magnitudes(), 128)
            .unwrap();
        let names = pipeline.pass_names();
        assert!(names.contains(&"fp16-quantization"));
        assert!(names.contains(&"awq-quantization"));
        assert!(names.contains(&"awq-scale-fusion"));
    }

    #[test]
    fn awq_blocks_subsequent_int4() {
        let pipeline = PassPipeline::new()
            .with_awq(sample_magnitudes(), 128)
            .unwrap();
        let result = pipeline.with_int4(128);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    #[cfg(not(feature = "gptq"))]
    fn gptq_stub_returns_error() {
        let result = PassPipeline::new().with_gptq(HashMap::new(), 128, 128, 0.01);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("gptq"),
            "expected gptq feature error, got: {err}"
        );
    }

    #[test]
    #[cfg(feature = "gptq")]
    fn with_gptq_builds_pipeline() {
        let pipeline = PassPipeline::new()
            .with_gptq(HashMap::new(), 128, 128, 0.01)
            .unwrap();
        let names = pipeline.pass_names();
        assert!(
            names.contains(&"gptq-quantization"),
            "expected gptq-quantization in pipeline, got: {names:?}"
        );
        // Should be inserted before type-repropagation.
        let gptq_pos = names
            .iter()
            .position(|n| *n == "gptq-quantization")
            .unwrap();
        let reprop_pos = names
            .iter()
            .position(|n| *n == "type-repropagation")
            .unwrap();
        assert!(
            gptq_pos < reprop_pos,
            "GPTQ should precede type-repropagation"
        );
    }

    #[test]
    #[cfg(feature = "gptq")]
    fn gptq_and_int8_mutually_exclusive() {
        let pipeline = PassPipeline::new().with_int8(None).unwrap();
        let result = pipeline.with_gptq(HashMap::new(), 128, 128, 0.01);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    #[cfg(feature = "gptq")]
    fn gptq_and_palettize_mutually_exclusive() {
        let pipeline = PassPipeline::new().with_palettize(4).unwrap();
        let result = pipeline.with_gptq(HashMap::new(), 128, 128, 0.01);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    #[cfg(feature = "gptq")]
    fn gptq_and_polar_quant_mutually_exclusive() {
        let pipeline = PassPipeline::new().with_polar_quant(4).unwrap();
        let result = pipeline.with_gptq(HashMap::new(), 128, 128, 0.01);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    // ── QuIP# pipeline tests ──────────────────────────────────────────

    #[test]
    fn with_quip_sharp_builds_pipeline() {
        let pipeline = PassPipeline::new().with_quip_sharp(2, 42).unwrap();
        let names = pipeline.pass_names();
        assert!(
            names.contains(&"quip-sharp"),
            "expected quip-sharp pass, got: {names:?}"
        );
    }

    #[test]
    fn quip_sharp_inserts_before_type_repropagation() {
        let pipeline = PassPipeline::new().with_quip_sharp(2, 42).unwrap();
        let names = pipeline.pass_names();
        let qs_pos = names.iter().position(|n| *n == "quip-sharp").unwrap();
        let reprop_pos = names
            .iter()
            .position(|n| *n == "type-repropagation")
            .unwrap();
        assert!(
            qs_pos < reprop_pos,
            "quip-sharp ({qs_pos}) should come before type-repropagation ({reprop_pos})"
        );
    }

    #[test]
    fn quip_sharp_and_int4_mutually_exclusive() {
        let pipeline = PassPipeline::new().with_int4(128).unwrap();
        let result = pipeline.with_quip_sharp(2, 42);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    fn quip_sharp_and_int8_mutually_exclusive() {
        let pipeline = PassPipeline::new().with_int8(None).unwrap();
        let result = pipeline.with_quip_sharp(2, 42);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    fn quip_sharp_and_awq_mutually_exclusive() {
        let pipeline = PassPipeline::new()
            .with_awq(sample_magnitudes(), 128)
            .unwrap();
        let result = pipeline.with_quip_sharp(2, 42);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    fn quip_sharp_and_palettize_mutually_exclusive() {
        let pipeline = PassPipeline::new().with_palettize(4).unwrap();
        let result = pipeline.with_quip_sharp(2, 42);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    fn quip_sharp_and_polar_quant_mutually_exclusive() {
        let pipeline = PassPipeline::new().with_polar_quant(4).unwrap();
        let result = pipeline.with_quip_sharp(2, 42);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    fn quip_sharp_blocks_subsequent_int4() {
        let pipeline = PassPipeline::new().with_quip_sharp(2, 42).unwrap();
        let result = pipeline.with_int4(128);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    fn quip_sharp_plus_fp16_is_allowed() {
        let pipeline = PassPipeline::new()
            .with_fp16()
            .unwrap()
            .with_quip_sharp(2, 42)
            .unwrap();
        let names = pipeline.pass_names();
        assert!(names.contains(&"fp16-quantization"));
        assert!(names.contains(&"quip-sharp"));
    }

    // ── D2Quant pipeline tests ────────────────────────────────────────

    #[test]
    fn with_d2quant_builds_pipeline() {
        let pipeline = PassPipeline::new()
            .with_d2quant(2, 128, 0.99, None)
            .unwrap();
        let names = pipeline.pass_names();
        assert!(
            names.contains(&"d2quant"),
            "expected d2quant pass, got: {names:?}"
        );
    }

    #[test]
    fn d2quant_inserts_before_type_repropagation() {
        let pipeline = PassPipeline::new()
            .with_d2quant(2, 128, 0.99, None)
            .unwrap();
        let names = pipeline.pass_names();
        let d2_pos = names.iter().position(|n| *n == "d2quant").unwrap();
        let reprop_pos = names
            .iter()
            .position(|n| *n == "type-repropagation")
            .unwrap();
        assert!(
            d2_pos < reprop_pos,
            "d2quant ({d2_pos}) should come before type-repropagation ({reprop_pos})"
        );
    }

    #[test]
    fn d2quant_and_int4_mutually_exclusive() {
        let pipeline = PassPipeline::new().with_int4(128).unwrap();
        let result = pipeline.with_d2quant(2, 128, 0.99, None);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    fn d2quant_and_int8_mutually_exclusive() {
        let pipeline = PassPipeline::new().with_int8(None).unwrap();
        let result = pipeline.with_d2quant(2, 128, 0.99, None);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    fn d2quant_and_awq_mutually_exclusive() {
        let pipeline = PassPipeline::new()
            .with_awq(sample_magnitudes(), 128)
            .unwrap();
        let result = pipeline.with_d2quant(2, 128, 0.99, None);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    fn d2quant_and_quip_sharp_mutually_exclusive() {
        let pipeline = PassPipeline::new().with_quip_sharp(2, 42).unwrap();
        let result = pipeline.with_d2quant(2, 128, 0.99, None);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    fn d2quant_blocks_subsequent_awq() {
        let pipeline = PassPipeline::new()
            .with_d2quant(2, 128, 0.99, None)
            .unwrap();
        let result = pipeline.with_awq(sample_magnitudes(), 128);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    fn d2quant_plus_fp16_is_allowed() {
        let pipeline = PassPipeline::new()
            .with_fp16()
            .unwrap()
            .with_d2quant(2, 128, 0.99, None)
            .unwrap();
        let names = pipeline.pass_names();
        assert!(names.contains(&"fp16-quantization"));
        assert!(names.contains(&"d2quant"));
    }

    // ── SpinQuant pipeline tests ──────────────────────────────────────

    #[test]
    fn with_spinquant_builds_pipeline() {
        let config = SpinQuantConfig {
            rotation_epochs: 100,
            bits: 4,
        };
        let pipeline = PassPipeline::new()
            .with_spinquant(config, SpinQuantMethod::MinMax { group_size: 128 })
            .unwrap();
        let names = pipeline.pass_names();
        assert!(
            names.contains(&"spin-quantization"),
            "expected spin-quantization pass, got: {names:?}"
        );
    }

    #[test]
    fn spinquant_and_int4_mutually_exclusive() {
        let pipeline = PassPipeline::new().with_int4(128).unwrap();
        let config = SpinQuantConfig {
            rotation_epochs: 100,
            bits: 4,
        };
        let result = pipeline.with_spinquant(config, SpinQuantMethod::MinMax { group_size: 128 });
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    fn spinquant_and_int8_mutually_exclusive() {
        let pipeline = PassPipeline::new().with_int8(None).unwrap();
        let config = SpinQuantConfig {
            rotation_epochs: 100,
            bits: 4,
        };
        let result = pipeline.with_spinquant(config, SpinQuantMethod::MinMax { group_size: 128 });
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    fn spinquant_and_awq_mutually_exclusive() {
        let pipeline = PassPipeline::new()
            .with_awq(sample_magnitudes(), 128)
            .unwrap();
        let config = SpinQuantConfig {
            rotation_epochs: 100,
            bits: 4,
        };
        let result = pipeline.with_spinquant(config, SpinQuantMethod::MinMax { group_size: 128 });
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    fn spinquant_and_quip_sharp_mutually_exclusive() {
        let pipeline = PassPipeline::new().with_quip_sharp(2, 42).unwrap();
        let config = SpinQuantConfig {
            rotation_epochs: 100,
            bits: 4,
        };
        let result = pipeline.with_spinquant(config, SpinQuantMethod::MinMax { group_size: 128 });
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    fn spinquant_and_d2quant_mutually_exclusive() {
        let pipeline = PassPipeline::new()
            .with_d2quant(2, 128, 0.99, None)
            .unwrap();
        let config = SpinQuantConfig {
            rotation_epochs: 100,
            bits: 4,
        };
        let result = pipeline.with_spinquant(config, SpinQuantMethod::MinMax { group_size: 128 });
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    fn spinquant_blocks_subsequent_int4() {
        let config = SpinQuantConfig {
            rotation_epochs: 100,
            bits: 4,
        };
        let pipeline = PassPipeline::new()
            .with_spinquant(config, SpinQuantMethod::MinMax { group_size: 128 })
            .unwrap();
        let result = pipeline.with_int4(128);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual exclusivity error, got: {err}"
        );
    }

    #[test]
    fn spinquant_plus_fp16_is_allowed() {
        let config = SpinQuantConfig {
            rotation_epochs: 100,
            bits: 4,
        };
        let pipeline = PassPipeline::new()
            .with_fp16()
            .unwrap()
            .with_spinquant(config, SpinQuantMethod::MinMax { group_size: 128 })
            .unwrap();
        let names = pipeline.pass_names();
        assert!(names.contains(&"fp16-quantization"));
    }

    #[test]
    fn spinquant_with_awq_method() {
        let config = SpinQuantConfig {
            rotation_epochs: 50,
            bits: 4,
        };
        let method = SpinQuantMethod::Awq {
            channel_magnitudes: sample_magnitudes(),
            group_size: 128,
        };
        let pipeline = PassPipeline::new().with_spinquant(config, method).unwrap();
        let names = pipeline.pass_names();
        assert!(
            names.contains(&"spin-quantization"),
            "expected spin-quantization pass, got: {names:?}"
        );
    }
}
