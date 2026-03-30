//! Multi-source pipeline conversion.
//!
//! Converts a set of related model files (ONNX, SafeTensors, or GGUF) into
//! coordinated `.mlpackage` outputs based on a TOML manifest that describes
//! the pipeline topology.
//!
//! # Manifest format
//!
//! ```toml
//! [pipeline]
//! name = "my-pipeline"
//!
//! [[stages]]
//! name = "encoder"
//! onnx = "encoder.onnx"
//! quantize = "fp16"
//!
//! [[stages]]
//! name = "decoder"
//! onnx = "decoder.onnx"
//! quantize = "int8"
//! depends_on = ["encoder"]
//!
//! # SafeTensors-based stage
//! [[stages]]
//! name = "transformer"
//! safetensors = "model.safetensors"
//! component = "transformer"
//! quantize = "fp16"
//!
//! # GGUF-based stage
//! [[stages]]
//! name = "embeddings"
//! gguf = "model.gguf"
//! component = "embeddings"
//! quantize = "none"
//! ```

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::templates::weights_to_program_component;
use mil_rs::convert::onnx_graph::onnx_to_program;
use mil_rs::error::{MilError, Result};
use mil_rs::ir::{PassPipeline, Program};
use mil_rs::reader::read_onnx;
use mil_rs::writer::write_mlpackage;

// ---------------------------------------------------------------------------
// TOML manifest types
// ---------------------------------------------------------------------------

/// Top-level pipeline manifest parsed from a TOML file.
#[derive(Debug, Clone, Deserialize)]
pub struct PipelineManifest {
    /// Pipeline metadata.
    pub pipeline: PipelineMeta,
    /// Ordered list of conversion stages.
    pub stages: Vec<StageConfig>,
}

/// Pipeline-level metadata.
#[derive(Debug, Clone, Deserialize)]
pub struct PipelineMeta {
    /// Human-readable pipeline name.
    pub name: String,
}

/// Configuration for a single stage in the pipeline.
#[derive(Debug, Clone, Deserialize)]
pub struct StageConfig {
    /// Unique stage name (used as the output `.mlpackage` stem).
    pub name: String,
    /// Path to the ONNX file (relative to the manifest directory).
    /// Exactly one of `onnx`, `safetensors`, or `gguf` must be specified.
    pub onnx: Option<String>,
    /// Path to a SafeTensors file (relative to the manifest directory).
    pub safetensors: Option<String>,
    /// Path to a GGUF file (relative to the manifest directory).
    pub gguf: Option<String>,
    /// Which model component to extract when using `safetensors` or `gguf`.
    /// Valid values: `"embeddings"`, `"transformer"`, `"lm_head"`.
    /// If not specified, the full model is built.
    pub component: Option<String>,
    /// Quantization mode: `"none"`, `"fp16"`, or `"int8"` (default: `"none"`).
    #[serde(default = "default_quantize")]
    pub quantize: String,
    /// Optional calibration data directory for int8 quantization.
    pub cal_data: Option<PathBuf>,
    /// Optional palettization bit-width (2, 4, 6, or 8).
    pub palettize: Option<u8>,
    /// Disable fusion passes for this stage.
    #[serde(default)]
    pub no_fusion: bool,
    /// Names of stages this stage depends on (for topology ordering).
    #[serde(default)]
    pub depends_on: Vec<String>,
}

fn default_quantize() -> String {
    "none".into()
}

// ---------------------------------------------------------------------------
// Output manifest types (JSON)
// ---------------------------------------------------------------------------

/// Runtime manifest written alongside the `.mlpackage` outputs.
///
/// Describes the pipeline topology so downstream consumers know how to
/// orchestrate inference across the individual models.
#[derive(Debug, Clone, Serialize)]
pub struct PipelineOutputManifest {
    /// Pipeline name from the TOML manifest.
    pub name: String,
    /// Ordered list of stage descriptors.
    pub stages: Vec<StageDescriptor>,
}

/// Descriptor for one stage in the output manifest.
#[derive(Debug, Clone, Serialize)]
pub struct StageDescriptor {
    /// Stage name.
    pub name: String,
    /// Relative path to the `.mlpackage` output.
    pub package: String,
    /// Input tensor descriptors.
    pub inputs: Vec<TensorDescriptor>,
    /// Output tensor descriptors.
    pub outputs: Vec<TensorDescriptor>,
    /// Names of stages this stage depends on.
    pub depends_on: Vec<String>,
}

/// Describes a single tensor in the stage I/O.
#[derive(Debug, Clone, Serialize)]
pub struct TensorDescriptor {
    /// Tensor name.
    pub name: String,
    /// Element type (e.g. `"Float32"`, `"Float16"`).
    pub scalar_type: String,
    /// Shape dimensions. `null` entries are dynamic.
    pub shape: Vec<Option<usize>>,
}

// ---------------------------------------------------------------------------
// Conversion result
// ---------------------------------------------------------------------------

/// Result of converting a full pipeline.
#[derive(Debug)]
pub struct PipelineConversionResult {
    /// The output manifest describing the pipeline topology.
    pub manifest: PipelineOutputManifest,
    /// Per-stage warnings collected during ONNX conversion.
    pub stage_warnings: Vec<(String, Vec<String>)>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse a pipeline TOML manifest from a string.
pub fn parse_pipeline_manifest(toml_str: &str) -> Result<PipelineManifest> {
    toml::from_str(toml_str)
        .map_err(|e| MilError::Validation(format!("failed to parse pipeline manifest: {e}")))
}

/// Convert a multi-source pipeline according to the given manifest.
///
/// Each stage is independently converted and optimized:
/// 1. Read the source model (ONNX, SafeTensors, or GGUF).
/// 2. Convert to MIL IR via the appropriate converter.
/// 3. Build and run a [`PassPipeline`] with the stage's settings.
/// 4. Convert to protobuf and write as `.mlpackage`.
///
/// After all stages are converted the output tensor shapes/types of each
/// stage are validated against the input expectations of dependent stages.
/// A JSON manifest describing the pipeline topology is written alongside
/// the `.mlpackage` files.
///
/// # Arguments
///
/// * `manifest` – parsed pipeline manifest
/// * `base_dir` – directory containing the source files (manifest paths are relative to this)
/// * `output_dir` – directory where `.mlpackage` outputs and the manifest are written
pub fn convert_pipeline(
    manifest: &PipelineManifest,
    base_dir: &Path,
    output_dir: &Path,
) -> Result<PipelineConversionResult> {
    validate_manifest(manifest)?;

    std::fs::create_dir_all(output_dir)?;

    let mut stage_programs: Vec<(String, Program)> = Vec::new();
    let mut stage_warnings: Vec<(String, Vec<String>)> = Vec::new();

    // Resolve conversion order respecting dependencies.
    let order = topological_order(&manifest.stages)?;

    for idx in &order {
        let stage = &manifest.stages[*idx];

        let (mut program, warnings) = convert_stage(stage, base_dir)?;

        stage_warnings.push((stage.name.clone(), warnings));

        // Build per-stage pass pipeline
        let pipeline = build_stage_pipeline(stage, base_dir)?;
        let _report = pipeline.run(&mut program)?;

        // Convert to protobuf and write .mlpackage
        let model = mil_rs::convert::ir_to_proto::program_to_model(&program, 9).map_err(|e| {
            MilError::Validation(format!(
                "stage '{}': failed to convert program to model: {e}",
                stage.name
            ))
        })?;

        let pkg_name = format!("{}.mlpackage", stage.name);
        let pkg_path = output_dir.join(&pkg_name);
        write_mlpackage(&model, &pkg_path)?;

        stage_programs.push((stage.name.clone(), program));
    }

    // Validate inter-stage tensor compatibility
    validate_inter_stage_tensors(manifest, &stage_programs)?;

    // Build and write the output manifest
    let output_manifest = build_output_manifest(manifest, &stage_programs);
    let manifest_json = serde_json::to_string_pretty(&output_manifest)
        .map_err(|e| MilError::Validation(format!("failed to serialize pipeline manifest: {e}")))?;
    std::fs::write(output_dir.join("pipeline.json"), manifest_json)?;

    Ok(PipelineConversionResult {
        manifest: output_manifest,
        stage_warnings,
    })
}

/// Convert a single stage from its source format to a MIL IR program.
/// Returns `(program, warnings)`.
fn convert_stage(stage: &StageConfig, base_dir: &Path) -> Result<(Program, Vec<String>)> {
    if let Some(onnx_path) = &stage.onnx {
        // ONNX source
        let full_path = base_dir.join(onnx_path);
        let onnx_model = read_onnx(&full_path).map_err(|e| {
            MilError::Validation(format!(
                "stage '{}': failed to read ONNX file '{}': {e}",
                stage.name,
                full_path.display()
            ))
        })?;

        let result = onnx_to_program(&onnx_model).map_err(|e| {
            MilError::Validation(format!(
                "stage '{}': ONNX conversion failed: {e}",
                stage.name
            ))
        })?;

        Ok((result.program, result.warnings))
    } else if let Some(st_path) = &stage.safetensors {
        // SafeTensors source
        let full_path = base_dir.join(st_path);
        let provider = crate::weights::SafeTensorsProvider::load(&full_path).map_err(|e| {
            MilError::Validation(format!(
                "stage '{}': failed to load SafeTensors '{}': {e}",
                stage.name,
                full_path.display()
            ))
        })?;

        let result =
            weights_to_program_component(&provider, stage.component.as_deref()).map_err(|e| {
                MilError::Validation(format!(
                    "stage '{}': SafeTensors conversion failed: {e}",
                    stage.name
                ))
            })?;

        Ok((result.program, result.warnings))
    } else if let Some(gguf_path) = &stage.gguf {
        // GGUF source
        let full_path = base_dir.join(gguf_path);
        let provider = crate::weights::GgufProvider::load(&full_path).map_err(|e| {
            MilError::Validation(format!(
                "stage '{}': failed to load GGUF '{}': {e}",
                stage.name,
                full_path.display()
            ))
        })?;

        let result =
            weights_to_program_component(&provider, stage.component.as_deref()).map_err(|e| {
                MilError::Validation(format!(
                    "stage '{}': GGUF conversion failed: {e}",
                    stage.name
                ))
            })?;

        Ok((result.program, result.warnings))
    } else {
        Err(MilError::Validation(format!(
            "stage '{}': no source specified (need exactly one of 'onnx', 'safetensors', or 'gguf')",
            stage.name
        )))
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Validate the manifest for structural issues (duplicate names, unknown deps, cycles,
/// source field constraints).
fn validate_manifest(manifest: &PipelineManifest) -> Result<()> {
    if manifest.stages.is_empty() {
        return Err(MilError::Validation(
            "pipeline manifest has no stages".into(),
        ));
    }

    let names: HashSet<&str> = manifest.stages.iter().map(|s| s.name.as_str()).collect();
    if names.len() != manifest.stages.len() {
        return Err(MilError::Validation(
            "pipeline manifest has duplicate stage names".into(),
        ));
    }

    for stage in &manifest.stages {
        // Validate exactly one source is specified.
        let source_count = [
            stage.onnx.is_some(),
            stage.safetensors.is_some(),
            stage.gguf.is_some(),
        ]
        .iter()
        .filter(|&&b| b)
        .count();

        if source_count == 0 {
            return Err(MilError::Validation(format!(
                "stage '{}': no source specified (need exactly one of 'onnx', 'safetensors', or 'gguf')",
                stage.name
            )));
        }
        if source_count > 1 {
            return Err(MilError::Validation(format!(
                "stage '{}': multiple sources specified (need exactly one of 'onnx', 'safetensors', or 'gguf')",
                stage.name
            )));
        }

        // Validate component is only used with weight-based sources.
        if stage.component.is_some() && stage.onnx.is_some() {
            return Err(MilError::Validation(format!(
                "stage '{}': 'component' is only valid with 'safetensors' or 'gguf' sources",
                stage.name
            )));
        }

        // Validate component values.
        if let Some(comp) = &stage.component {
            match comp.as_str() {
                "embeddings" | "transformer" | "lm_head" => {}
                other => {
                    return Err(MilError::Validation(format!(
                        "stage '{}': unsupported component '{other}' (expected 'embeddings', 'transformer', or 'lm_head')",
                        stage.name
                    )));
                }
            }
        }

        for dep in &stage.depends_on {
            if !names.contains(dep.as_str()) {
                return Err(MilError::Validation(format!(
                    "stage '{}' depends on unknown stage '{dep}'",
                    stage.name
                )));
            }
            if dep == &stage.name {
                return Err(MilError::Validation(format!(
                    "stage '{}' depends on itself",
                    stage.name
                )));
            }
        }

        match stage.quantize.as_str() {
            "none" | "fp16" | "int8" => {}
            other => {
                return Err(MilError::Validation(format!(
                    "stage '{}': unsupported quantize value '{other}' (expected 'none', 'fp16', or 'int8')",
                    stage.name
                )));
            }
        }
    }

    // Check for cycles via topological sort.
    topological_order(&manifest.stages)?;

    Ok(())
}

/// Compute a topological ordering of stages based on `depends_on` edges.
fn topological_order(stages: &[StageConfig]) -> Result<Vec<usize>> {
    let name_to_idx: HashMap<&str, usize> = stages
        .iter()
        .enumerate()
        .map(|(i, s)| (s.name.as_str(), i))
        .collect();

    let n = stages.len();
    let mut in_degree = vec![0usize; n];
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (i, stage) in stages.iter().enumerate() {
        for dep in &stage.depends_on {
            if let Some(&dep_idx) = name_to_idx.get(dep.as_str()) {
                adj[dep_idx].push(i);
                in_degree[i] += 1;
            }
        }
    }

    let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    // Sort for deterministic ordering.
    queue.sort_unstable();
    let mut order = Vec::with_capacity(n);

    while let Some(node) = queue.pop() {
        order.push(node);
        // Process in reverse to maintain stable ordering with the pop-from-end.
        let mut children = adj[node].clone();
        children.sort_unstable();
        children.reverse();
        for child in children {
            in_degree[child] -= 1;
            if in_degree[child] == 0 {
                queue.push(child);
                queue.sort_unstable();
            }
        }
    }

    if order.len() != n {
        return Err(MilError::Validation(
            "pipeline has a dependency cycle".into(),
        ));
    }

    Ok(order)
}

/// Build a [`PassPipeline`] configured for a single stage.
fn build_stage_pipeline(stage: &StageConfig, base_dir: &Path) -> Result<PassPipeline> {
    let mut pipeline = PassPipeline::new();

    if stage.no_fusion {
        pipeline = pipeline.without_fusion();
    }

    match stage.quantize.as_str() {
        "fp16" => {
            pipeline = pipeline.with_fp16()?;
        }
        "int8" => {
            let cal_data = stage.cal_data.as_ref().map(|p| base_dir.join(p));
            pipeline = pipeline.with_int8(cal_data)?;
        }
        _ => {}
    }

    if let Some(bits) = stage.palettize {
        pipeline = pipeline.with_palettize(bits)?;
    }

    Ok(pipeline)
}

/// Validate that output tensors of depended-upon stages are compatible with
/// the inputs of dependent stages.
///
/// This is a best-effort check: if a stage declares `depends_on`, we verify
/// that at least the tensor names referenced in the dependent stage's inputs
/// exist among the outputs of its dependencies (when both sides have typed
/// metadata). Shape and type mismatches are reported as errors.
fn validate_inter_stage_tensors(
    manifest: &PipelineManifest,
    stage_programs: &[(String, Program)],
) -> Result<()> {
    let program_by_name: HashMap<&str, &Program> = stage_programs
        .iter()
        .map(|(name, prog)| (name.as_str(), prog))
        .collect();

    for stage in &manifest.stages {
        if stage.depends_on.is_empty() {
            continue;
        }

        // Collect all output tensor types from dependencies.
        let mut dep_outputs: HashMap<String, mil_rs::ir::TensorType> = HashMap::new();
        for dep_name in &stage.depends_on {
            if let Some(dep_prog) = program_by_name.get(dep_name.as_str()) {
                if let Some(func) = dep_prog.main() {
                    let output_types: HashMap<&String, &mil_rs::ir::TensorType> = func
                        .body
                        .operations
                        .iter()
                        .flat_map(|op| op.outputs.iter().zip(&op.output_types))
                        .filter_map(|(name, ty)| ty.as_ref().map(|t| (name, t)))
                        .collect();

                    for output_name in &func.body.outputs {
                        if let Some(&ty) = output_types.get(output_name) {
                            dep_outputs.insert(output_name.clone(), ty.clone());
                        }
                    }
                }
            }
        }

        // Check whether the current stage's inputs have compatible types.
        if let Some(prog) = program_by_name.get(stage.name.as_str()) {
            if let Some(func) = prog.main() {
                for (input_name, input_ty) in &func.inputs {
                    if let Some(dep_ty) = dep_outputs.get(input_name) {
                        // Validate scalar type match.
                        if dep_ty.scalar_type != input_ty.scalar_type {
                            return Err(MilError::Validation(format!(
                                "stage '{}': input '{input_name}' expects {:?} but dependency produces {:?}",
                                stage.name, input_ty.scalar_type, dep_ty.scalar_type
                            )));
                        }
                        // Validate rank match.
                        if dep_ty.rank() != input_ty.rank() {
                            return Err(MilError::Validation(format!(
                                "stage '{}': input '{input_name}' expects rank {} but dependency produces rank {}",
                                stage.name,
                                input_ty.rank(),
                                dep_ty.rank()
                            )));
                        }
                        // Validate static dimensions match where both are known.
                        for (dim_idx, (dep_dim, in_dim)) in
                            dep_ty.shape.iter().zip(input_ty.shape.iter()).enumerate()
                        {
                            if let (Some(d), Some(i)) = (dep_dim, in_dim) {
                                if d != i {
                                    return Err(MilError::Validation(format!(
                                        "stage '{}': input '{input_name}' dimension {dim_idx} expects {i} but dependency produces {d}",
                                        stage.name
                                    )));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Build the JSON output manifest from the converted programs.
fn build_output_manifest(
    manifest: &PipelineManifest,
    stage_programs: &[(String, Program)],
) -> PipelineOutputManifest {
    let program_by_name: HashMap<&str, &Program> = stage_programs
        .iter()
        .map(|(name, prog)| (name.as_str(), prog))
        .collect();

    let stages = manifest
        .stages
        .iter()
        .map(|stage| {
            let (inputs, outputs) = if let Some(prog) = program_by_name.get(stage.name.as_str()) {
                if let Some(func) = prog.main() {
                    let ins: Vec<TensorDescriptor> = func
                        .inputs
                        .iter()
                        .map(|(name, ty)| TensorDescriptor {
                            name: name.clone(),
                            scalar_type: format!("{:?}", ty.scalar_type),
                            shape: ty.shape.clone(),
                        })
                        .collect();

                    let output_types: HashMap<&String, &mil_rs::ir::TensorType> = func
                        .body
                        .operations
                        .iter()
                        .flat_map(|op| op.outputs.iter().zip(&op.output_types))
                        .filter_map(|(name, ty)| ty.as_ref().map(|t| (name, t)))
                        .collect();

                    let outs: Vec<TensorDescriptor> = func
                        .body
                        .outputs
                        .iter()
                        .map(|out_name| {
                            if let Some(&ty) = output_types.get(out_name) {
                                TensorDescriptor {
                                    name: out_name.clone(),
                                    scalar_type: format!("{:?}", ty.scalar_type),
                                    shape: ty.shape.clone(),
                                }
                            } else {
                                // Fallback: output name without type info.
                                TensorDescriptor {
                                    name: out_name.clone(),
                                    scalar_type: "Unknown".into(),
                                    shape: Vec::new(),
                                }
                            }
                        })
                        .collect();

                    (ins, outs)
                } else {
                    (Vec::new(), Vec::new())
                }
            } else {
                (Vec::new(), Vec::new())
            };

            StageDescriptor {
                name: stage.name.clone(),
                package: format!("{}.mlpackage", stage.name),
                inputs,
                outputs,
                depends_on: stage.depends_on.clone(),
            }
        })
        .collect();

    PipelineOutputManifest {
        name: manifest.pipeline.name.clone(),
        stages,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a StageConfig with only ONNX source (backward-compat pattern).
    fn onnx_stage(name: &str, onnx: &str) -> StageConfig {
        StageConfig {
            name: name.into(),
            onnx: Some(onnx.into()),
            safetensors: None,
            gguf: None,
            component: None,
            quantize: "none".into(),
            cal_data: None,
            palettize: None,
            no_fusion: false,
            depends_on: vec![],
        }
    }

    #[test]
    fn parse_simple_manifest() {
        let toml = r#"
[pipeline]
name = "test-pipeline"

[[stages]]
name = "encoder"
onnx = "encoder.onnx"
quantize = "fp16"

[[stages]]
name = "decoder"
onnx = "decoder.onnx"
depends_on = ["encoder"]
"#;
        let manifest = parse_pipeline_manifest(toml).unwrap();
        assert_eq!(manifest.pipeline.name, "test-pipeline");
        assert_eq!(manifest.stages.len(), 2);
        assert_eq!(manifest.stages[0].name, "encoder");
        assert_eq!(manifest.stages[0].quantize, "fp16");
        assert_eq!(manifest.stages[1].name, "decoder");
        assert_eq!(manifest.stages[1].quantize, "none");
        assert_eq!(manifest.stages[1].depends_on, vec!["encoder"]);
    }

    #[test]
    fn parse_manifest_with_all_fields() {
        let toml = r#"
[pipeline]
name = "full-pipeline"

[[stages]]
name = "stage_a"
onnx = "a.onnx"
quantize = "int8"
palettize = 4
no_fusion = true
depends_on = []
"#;
        let manifest = parse_pipeline_manifest(toml).unwrap();
        let stage = &manifest.stages[0];
        assert_eq!(stage.quantize, "int8");
        assert_eq!(stage.palettize, Some(4));
        assert!(stage.no_fusion);
        assert!(stage.depends_on.is_empty());
    }

    #[test]
    fn parse_manifest_safetensors_stage() {
        let toml = r#"
[pipeline]
name = "weight-pipeline"

[[stages]]
name = "transformer"
safetensors = "model.safetensors"
component = "transformer"
quantize = "fp16"
"#;
        let manifest = parse_pipeline_manifest(toml).unwrap();
        let stage = &manifest.stages[0];
        assert_eq!(stage.safetensors, Some("model.safetensors".into()));
        assert_eq!(stage.component, Some("transformer".into()));
        assert!(stage.onnx.is_none());
        assert!(stage.gguf.is_none());
    }

    #[test]
    fn parse_manifest_gguf_stage() {
        let toml = r#"
[pipeline]
name = "gguf-pipeline"

[[stages]]
name = "embeddings"
gguf = "model.gguf"
component = "embeddings"
"#;
        let manifest = parse_pipeline_manifest(toml).unwrap();
        let stage = &manifest.stages[0];
        assert_eq!(stage.gguf, Some("model.gguf".into()));
        assert_eq!(stage.component, Some("embeddings".into()));
        assert!(stage.onnx.is_none());
        assert!(stage.safetensors.is_none());
    }

    #[test]
    fn parse_invalid_toml() {
        let result = parse_pipeline_manifest("not valid toml {{{}}}");
        assert!(result.is_err());
    }

    #[test]
    fn validate_empty_stages() {
        let manifest = PipelineManifest {
            pipeline: PipelineMeta {
                name: "empty".into(),
            },
            stages: vec![],
        };
        let err = validate_manifest(&manifest).unwrap_err();
        assert!(err.to_string().contains("no stages"));
    }

    #[test]
    fn validate_duplicate_names() {
        let manifest = PipelineManifest {
            pipeline: PipelineMeta { name: "dup".into() },
            stages: vec![onnx_stage("a", "a.onnx"), onnx_stage("a", "b.onnx")],
        };
        let err = validate_manifest(&manifest).unwrap_err();
        assert!(err.to_string().contains("duplicate"));
    }

    #[test]
    fn validate_unknown_dependency() {
        let mut stage = onnx_stage("a", "a.onnx");
        stage.depends_on = vec!["nonexistent".into()];
        let manifest = PipelineManifest {
            pipeline: PipelineMeta {
                name: "bad-dep".into(),
            },
            stages: vec![stage],
        };
        let err = validate_manifest(&manifest).unwrap_err();
        assert!(err.to_string().contains("unknown stage"));
    }

    #[test]
    fn validate_self_dependency() {
        let mut stage = onnx_stage("a", "a.onnx");
        stage.depends_on = vec!["a".into()];
        let manifest = PipelineManifest {
            pipeline: PipelineMeta {
                name: "self-dep".into(),
            },
            stages: vec![stage],
        };
        let err = validate_manifest(&manifest).unwrap_err();
        assert!(err.to_string().contains("depends on itself"));
    }

    #[test]
    fn validate_invalid_quantize() {
        let mut stage = onnx_stage("a", "a.onnx");
        stage.quantize = "fp64".into();
        let manifest = PipelineManifest {
            pipeline: PipelineMeta {
                name: "bad-q".into(),
            },
            stages: vec![stage],
        };
        let err = validate_manifest(&manifest).unwrap_err();
        assert!(err.to_string().contains("unsupported quantize"));
    }

    #[test]
    fn validate_no_source() {
        let manifest = PipelineManifest {
            pipeline: PipelineMeta {
                name: "no-src".into(),
            },
            stages: vec![StageConfig {
                name: "a".into(),
                onnx: None,
                safetensors: None,
                gguf: None,
                component: None,
                quantize: "none".into(),
                cal_data: None,
                palettize: None,
                no_fusion: false,
                depends_on: vec![],
            }],
        };
        let err = validate_manifest(&manifest).unwrap_err();
        assert!(err.to_string().contains("no source specified"));
    }

    #[test]
    fn validate_multiple_sources() {
        let manifest = PipelineManifest {
            pipeline: PipelineMeta {
                name: "multi-src".into(),
            },
            stages: vec![StageConfig {
                name: "a".into(),
                onnx: Some("a.onnx".into()),
                safetensors: Some("a.safetensors".into()),
                gguf: None,
                component: None,
                quantize: "none".into(),
                cal_data: None,
                palettize: None,
                no_fusion: false,
                depends_on: vec![],
            }],
        };
        let err = validate_manifest(&manifest).unwrap_err();
        assert!(err.to_string().contains("multiple sources"));
    }

    #[test]
    fn validate_component_with_onnx_rejected() {
        let mut stage = onnx_stage("a", "a.onnx");
        stage.component = Some("transformer".into());
        let manifest = PipelineManifest {
            pipeline: PipelineMeta {
                name: "bad-comp".into(),
            },
            stages: vec![stage],
        };
        let err = validate_manifest(&manifest).unwrap_err();
        assert!(err.to_string().contains("only valid with"));
    }

    #[test]
    fn validate_invalid_component() {
        let manifest = PipelineManifest {
            pipeline: PipelineMeta {
                name: "bad-comp".into(),
            },
            stages: vec![StageConfig {
                name: "a".into(),
                onnx: None,
                safetensors: Some("model.safetensors".into()),
                gguf: None,
                component: Some("invalid_component".into()),
                quantize: "none".into(),
                cal_data: None,
                palettize: None,
                no_fusion: false,
                depends_on: vec![],
            }],
        };
        let err = validate_manifest(&manifest).unwrap_err();
        assert!(err.to_string().contains("unsupported component"));
    }

    #[test]
    fn validate_component_with_safetensors_accepted() {
        let manifest = PipelineManifest {
            pipeline: PipelineMeta {
                name: "ok-comp".into(),
            },
            stages: vec![StageConfig {
                name: "a".into(),
                onnx: None,
                safetensors: Some("model.safetensors".into()),
                gguf: None,
                component: Some("transformer".into()),
                quantize: "none".into(),
                cal_data: None,
                palettize: None,
                no_fusion: false,
                depends_on: vec![],
            }],
        };
        assert!(validate_manifest(&manifest).is_ok());
    }

    #[test]
    fn topological_order_no_deps() {
        let stages = vec![onnx_stage("a", "a.onnx"), onnx_stage("b", "b.onnx")];
        let order = topological_order(&stages).unwrap();
        assert_eq!(order.len(), 2);
    }

    #[test]
    fn topological_order_with_deps() {
        let mut a = onnx_stage("a", "a.onnx");
        a.depends_on = vec!["b".into()];
        let b = onnx_stage("b", "b.onnx");
        let stages = vec![a, b];
        let order = topological_order(&stages).unwrap();
        // "b" (index 1) must come before "a" (index 0)
        let pos_a = order.iter().position(|&x| x == 0).unwrap();
        let pos_b = order.iter().position(|&x| x == 1).unwrap();
        assert!(pos_b < pos_a);
    }

    #[test]
    fn topological_order_cycle() {
        let mut a = onnx_stage("a", "a.onnx");
        a.depends_on = vec!["b".into()];
        let mut b = onnx_stage("b", "b.onnx");
        b.depends_on = vec!["a".into()];
        let stages = vec![a, b];
        let err = topological_order(&stages).unwrap_err();
        assert!(err.to_string().contains("cycle"));
    }

    #[test]
    fn build_output_manifest_structure() {
        use mil_rs::ir::{Function, Program, ScalarType, TensorType};

        let manifest = PipelineManifest {
            pipeline: PipelineMeta {
                name: "test".into(),
            },
            stages: vec![onnx_stage("enc", "enc.onnx"), {
                let mut s = onnx_stage("dec", "dec.onnx");
                s.depends_on = vec!["enc".into()];
                s
            }],
        };

        let input_ty = TensorType::new(ScalarType::Float32, vec![1, 3, 224, 224]);
        let func = Function::new("main").with_input("image", input_ty);
        let mut prog = Program::new("1.0.0");
        prog.add_function(func);

        let programs = vec![("enc".into(), prog.clone()), ("dec".into(), prog)];

        let out = build_output_manifest(&manifest, &programs);
        assert_eq!(out.name, "test");
        assert_eq!(out.stages.len(), 2);
        assert_eq!(out.stages[0].name, "enc");
        assert_eq!(out.stages[0].package, "enc.mlpackage");
        assert_eq!(out.stages[1].depends_on, vec!["enc"]);
        assert_eq!(out.stages[0].inputs.len(), 1);
        assert_eq!(out.stages[0].inputs[0].name, "image");
        assert_eq!(out.stages[0].inputs[0].scalar_type, "Float32");
    }
}
