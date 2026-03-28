//! Pass pipeline manager for ordering, mutual exclusivity, and builder API.

use std::collections::HashMap;
use std::path::PathBuf;

use super::pass::Pass;
use super::passes::{
    AttentionFusionPass, ConstantFoldPass, ConvBatchNormFusionPass, ConvBatchNormWeightFoldPass,
    ConvReluFusionPass, DeadCodeEliminationPass, Fp16QuantizePass, Granularity,
    IdentityEliminationPass, Int8QuantizePass, LinearReluFusionPass, OpSubstitutionPass,
    PalettizePass, ShapeMaterializePass,
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
            .finish()
    }
}

impl PassPipeline {
    /// Create the default pipeline with all always-on passes.
    ///
    /// Includes cleanup (DCE, identity elimination, constant folding),
    /// fusion (conv-bn weight fold, conv-bn fusion, conv-relu, linear-relu),
    /// and optimization (attention fusion, op substitution).
    pub fn new() -> Self {
        Self {
            passes: vec![
                // Cleanup passes (1-3)
                Box::new(DeadCodeEliminationPass),
                Box::new(IdentityEliminationPass),
                Box::new(ConstantFoldPass),
                // Fusion passes (4-7)
                Box::new(ConvBatchNormWeightFoldPass),
                Box::new(ConvBatchNormFusionPass),
                Box::new(ConvReluFusionPass),
                Box::new(LinearReluFusionPass),
                // Optimization passes (8-9)
                Box::new(AttentionFusionPass),
                Box::new(OpSubstitutionPass),
                // NOTE: LayoutOptimizationPass disabled — it inserts transpose ops
                // that cause CoreML to segfault at runtime. Re-enable once transpose
                // serialization (output_types + perm encoding) is fixed.
            ],
            has_fp16: false,
            has_int8: false,
            has_palettize: false,
        }
    }

    /// Add FP16 quantization. Errors if INT8 is already added.
    pub fn with_fp16(mut self) -> Result<Self> {
        if self.has_int8 {
            return Err(MilError::Validation(
                "FP16 and INT8 quantization are mutually exclusive".into(),
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
        if !matches!(n_bits, 2 | 4 | 6 | 8) {
            return Err(MilError::Validation(format!(
                "palettize n_bits must be 2, 4, 6, or 8, got {n_bits}"
            )));
        }
        self.has_palettize = true;
        self.passes.push(Box::new(PalettizePass::new(n_bits)));
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
            "attention-fusion",
            "op-substitution",
            "layout-optimization",
        ];
        self.passes.retain(|p| !FUSION_NAMES.contains(&p.name()));
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
            pass.run(program)?;
            let ops_after = count_ops(program);
            pass_results.push(PassResult {
                name: pass.name().to_string(),
                ops_before,
                ops_after,
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

/// Report from running a [`PassPipeline`].
pub struct PipelineReport {
    pub pass_results: Vec<PassResult>,
}

/// Result of a single pass execution.
pub struct PassResult {
    pub name: String,
    pub ops_before: usize,
    pub ops_after: usize,
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
        // layout-optimization is intentionally excluded — see note in new()
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
                "attention-fusion",
                "op-substitution",
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
        assert_eq!(pipeline.pass_names().len(), 9);
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
}
