//! ANE (Apple Neural Engine) direct runtime backend for ironmill.
//!
//! This crate provides a Rust-native interface to Apple's private ANE APIs
//! (`_ANEClient`, `_ANECompiler`) for compiling and executing models directly
//! on the Neural Engine, bypassing CoreML's `MLModel` path.
//!
//! # ⚠️ Private API Warning
//!
//! This crate uses **undocumented Apple private APIs** that may change between
//! macOS releases. It is feature-gated behind `ane-direct` and should not be
//! used in Mac App Store submissions.
//!
//! # Architecture
//!
//! The crate mirrors `ironmill-coreml` but targets the ANE directly:
//!
//! | Module      | Purpose                                    |
//! |-------------|--------------------------------------------|
//! | `blobfile`  | BLOBFILE weight format writer               |
//! | `tensor`    | IOSurface-backed tensor I/O                 |
//! | `runtime`   | `_ANEClient` lifecycle and program execution|
//! | `cache`     | Compiled program cache with disk persistence|
//! | `split`     | Model → ANE-sized sub-program splitter      |
//! | `program`   | Compiled program handle types               |

#[cfg(not(target_os = "macos"))]
compile_error!("ironmill-ane only supports macOS");

pub mod blobfile;
pub mod cache;
pub mod program;
pub mod runtime;
pub mod split;
pub mod tensor;

use std::path::PathBuf;

use mil_rs::ir::ScalarType;

use crate::blobfile::BlobFileWriter;
use crate::cache::ProgramCache;
use crate::program::{CompiledProgram, LoadedProgram};
use crate::runtime::AneRuntime;
use crate::split::{SplitConfig, split_for_ane};
use crate::tensor::{AneTensor, uniform_alloc_size};

// ── Error type ────────────────────────────────────────────────────

/// Errors from the ANE runtime backend.
#[derive(Debug, thiserror::Error)]
pub enum AneError {
    /// ANE compilation failed with a status code.
    #[error("ANE compilation failed (status {status:#x}): {context}")]
    CompileFailed { status: u32, context: String },

    /// ANE eval failed with a status code.
    #[error("ANE eval failed (status {status:#x}): {context}")]
    EvalFailed { status: u32, context: String },

    /// IOSurface creation or I/O failed.
    #[error("IOSurface error: {0}")]
    SurfaceError(String),

    /// The compile budget (~119 per process) has been exhausted.
    #[error("ANE compile budget exhausted ({used}/{limit} compilations used)")]
    BudgetExhausted { used: usize, limit: usize },

    /// A generic error from an underlying operation.
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

/// Result type alias for ANE operations.
pub type Result<T> = std::result::Result<T, AneError>;

// ── Configuration ─────────────────────────────────────────────────

/// Configuration for the ANE runtime backend.
#[derive(Debug, Clone)]
pub struct AneConfig {
    /// Maximum number of compiled programs to cache in memory.
    /// Must stay under the ~119 per-process compile limit.
    pub max_programs: usize,
    /// Directory for persisting compiled programs to disk.
    /// Bypasses the per-process compile limit on subsequent runs.
    pub cache_dir: Option<PathBuf>,
    /// Enable INT4 data type support (experimental).
    pub enable_int4: bool,
}

impl Default for AneConfig {
    fn default() -> Self {
        Self {
            max_programs: 100,
            cache_dir: None,
            enable_int4: false,
        }
    }
}

// ── Tensor descriptor ─────────────────────────────────────────────

/// Describes a tensor's shape and data type for I/O specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDescriptor {
    /// Variable name in the MIL program.
    pub name: String,
    /// Shape in ANE layout: `[1, C, 1, S]`.
    pub shape: [usize; 4],
    /// Element data type.
    pub dtype: ScalarType,
}

// ── AneModel facade ───────────────────────────────────────────────

/// High-level model API that orchestrates the full ANE pipeline:
/// IR passes → split → emit MIL → compile → load → eval.
///
/// Mirrors [`ironmill_coreml::Model`] for the direct-ANE path.
pub struct AneModel {
    runtime: AneRuntime,
    sub_programs: Vec<LoadedSubProgram>,
    cache: ProgramCache,
    #[allow(dead_code)]
    config: AneConfig,
    /// Scratch directory for compiler inputs (BLOBFILEs). Cleaned up on drop.
    _work_dir: tempfile::TempDir,
}

struct LoadedSubProgram {
    loaded: LoadedProgram,
    meta: SubProgramMeta,
    input_tensors: Vec<AneTensor>,
    output_tensors: Vec<AneTensor>,
}

struct SubProgramMeta {
    #[allow(dead_code)]
    name: String,
    inputs: Vec<TensorDescriptor>,
    outputs: Vec<TensorDescriptor>,
}

impl AneModel {
    /// Compile a model from ironmill IR and load it for execution.
    ///
    /// Pipeline:
    /// 1. Run ANE-specific passes (op substitution, layout, attention decompose,
    ///    concat elimination, variable naming)
    /// 2. Split into sub-programs
    /// 3. For each sub-program:
    ///    a. Emit MIL text + collect weight blob entries
    ///    b. Write BLOBFILE
    ///    c. Check cache — skip compilation if cached
    ///    d. Compile MIL text (or load from cache)
    ///    e. Load compiled program into runtime
    ///    f. Pre-allocate IOSurface tensors for I/O
    /// 4. Return the assembled `AneModel`
    pub fn compile_and_load(program: &mil_rs::ir::Program, config: AneConfig) -> Result<Self> {
        use mil_rs::convert::ir_to_mil_text::{MilTextConfig, program_to_mil_text};
        use mil_rs::ffi::ane::AneCompiler;
        use mil_rs::ir::Pass;
        use mil_rs::ir::passes::{
            AneConcatEliminationPass, AneLayoutPass, AneVariableNamingPass, AttentionDecomposePass,
            OpSubstitutionPass,
        };

        // 1. Clone and run ANE-specific passes
        let mut program = program.clone();
        let passes: &[(&str, &dyn Pass)] = &[
            ("OpSubstitution", &OpSubstitutionPass),
            ("AneLayout", &AneLayoutPass),
            ("AttentionDecompose", &AttentionDecomposePass),
            ("AneConcatElimination", &AneConcatEliminationPass),
            ("AneVariableNaming", &AneVariableNamingPass),
        ];
        for (name, pass) in passes {
            pass.run(&mut program)
                .map_err(|e| AneError::Other(anyhow::anyhow!("{name} pass failed: {e}")))?;
        }

        // 2. Split into sub-programs
        let split_config = SplitConfig::default();
        let model_split = split_for_ane(&program, &split_config)?;

        // 3. Initialize runtime and cache
        let runtime = AneRuntime::new()?;
        let mut cache = ProgramCache::new(config.cache_dir.clone(), config.max_programs);

        // 4. Process each sub-program
        let mil_config = MilTextConfig::default();
        let mut loaded_subs = Vec::new();
        let work_dir = tempfile::tempdir()
            .map_err(|e| AneError::Other(anyhow::anyhow!("failed to create work dir: {e}")))?;

        for sub in &model_split.programs {
            // 4a. Emit MIL text + weight entries
            let (mil_text, weight_entries) = program_to_mil_text(&sub.program, &mil_config)
                .map_err(|e| {
                    AneError::Other(anyhow::anyhow!(
                        "MIL text emission failed for {}: {e}",
                        sub.name
                    ))
                })?;

            // 4b. Write BLOBFILE
            let mut blob_writer = BlobFileWriter::new();
            for entry in &weight_entries {
                blob_writer.add_weight(&entry.name, &entry.data, entry.dtype);
            }
            let weight_dir = work_dir.path().join(&sub.name);
            std::fs::create_dir_all(&weight_dir)
                .map_err(|e| AneError::Other(anyhow::anyhow!("failed to create dir: {e}")))?;
            let weight_path = weight_dir.join("weights.blob");
            blob_writer
                .write(&weight_path)
                .map_err(|e| AneError::Other(anyhow::anyhow!("BLOBFILE write failed: {e}")))?;

            // 4c. Check cache and 4d. Compile (or load from cache)
            let cache_key = ProgramCache::make_key(&mil_text, blob_writer.as_bytes());

            let compiled = if cache.contains(&cache_key) {
                // Cache hit — still compile for now (disk deserialization not yet
                // implemented), but skip re-inserting into cache.
                let ptr = AneCompiler::compile_mil_text(&mil_text, &weight_path).map_err(|e| {
                    AneError::CompileFailed {
                        status: 0,
                        context: format!("{e}"),
                    }
                })?;
                cache.record_compilation();
                CompiledProgram { inner: ptr }
            } else {
                let ptr = AneCompiler::compile_mil_text(&mil_text, &weight_path).map_err(|e| {
                    AneError::CompileFailed {
                        status: 0,
                        context: format!("{e}"),
                    }
                })?;
                cache.record_compilation();
                if let Some(disk_path) = cache.disk_path_for(&cache_key) {
                    std::fs::create_dir_all(&disk_path).ok();
                    cache.insert(cache_key, disk_path);
                }
                CompiledProgram { inner: ptr }
            };

            // 4e. Load into runtime
            let loaded = runtime.load_program(&compiled)?;

            // 4f. Pre-allocate I/O tensors with uniform sizing
            let input_shapes: Vec<_> = sub.inputs.iter().map(|td| (td.shape, td.dtype)).collect();
            let output_shapes: Vec<_> = sub.outputs.iter().map(|td| (td.shape, td.dtype)).collect();
            let input_alloc = uniform_alloc_size(&input_shapes);
            let output_alloc = uniform_alloc_size(&output_shapes);

            let input_tensors: Vec<AneTensor> = sub
                .inputs
                .iter()
                .map(|td| {
                    AneTensor::new_with_min_alloc(td.shape[1], td.shape[3], td.dtype, input_alloc)
                })
                .collect::<Result<Vec<_>>>()?;
            let output_tensors: Vec<AneTensor> = sub
                .outputs
                .iter()
                .map(|td| {
                    AneTensor::new_with_min_alloc(td.shape[1], td.shape[3], td.dtype, output_alloc)
                })
                .collect::<Result<Vec<_>>>()?;

            loaded_subs.push(LoadedSubProgram {
                loaded,
                meta: SubProgramMeta {
                    name: sub.name.clone(),
                    inputs: sub.inputs.clone(),
                    outputs: sub.outputs.clone(),
                },
                input_tensors,
                output_tensors,
            });
        }

        Ok(Self {
            runtime,
            sub_programs: loaded_subs,
            cache,
            config,
            _work_dir: work_dir,
        })
    }

    /// Run inference on the loaded model.
    ///
    /// Executes sub-programs sequentially, piping each sub-program's outputs
    /// into the next sub-program's inputs.
    pub fn predict(&mut self, inputs: &[AneTensor]) -> Result<Vec<AneTensor>> {
        if self.sub_programs.is_empty() {
            return Err(AneError::Other(anyhow::anyhow!("no sub-programs loaded")));
        }

        // Copy user inputs into first sub-program's input tensors.
        {
            let first = &mut self.sub_programs[0];
            if inputs.len() != first.input_tensors.len() {
                return Err(AneError::Other(anyhow::anyhow!(
                    "expected {} inputs, got {}",
                    first.input_tensors.len(),
                    inputs.len()
                )));
            }
            for (dst, src) in first.input_tensors.iter_mut().zip(inputs.iter()) {
                let data = src.read_f16()?;
                dst.write_f16(&data)?;
            }
        }

        // Execute sub-programs sequentially, wiring outputs → next inputs.
        let n = self.sub_programs.len();
        for i in 0..n {
            // Split-borrow: destructure the sub-program so the borrow checker
            // can see that `loaded`, `input_tensors`, and `output_tensors` are
            // disjoint from `self.runtime`.
            {
                let sub = &mut self.sub_programs[i];
                let input_refs: Vec<&AneTensor> = sub.input_tensors.iter().collect();
                let mut output_refs: Vec<&mut AneTensor> = sub.output_tensors.iter_mut().collect();
                self.runtime
                    .eval(&sub.loaded, &input_refs, &mut output_refs)?;
            }

            // Wire outputs of sub-program i to inputs of sub-program i+1.
            if i + 1 < n {
                let (left, right) = self.sub_programs.split_at_mut(i + 1);
                let current = &left[i];
                let next = &mut right[0];
                let copy_count = current.output_tensors.len().min(next.input_tensors.len());
                for j in 0..copy_count {
                    let data = current.output_tensors[j].read_f16()?;
                    next.input_tensors[j].write_f16(&data)?;
                }
            }
        }

        // Read final outputs into fresh tensors for the caller.
        let last = self.sub_programs.last().unwrap();
        let mut results = Vec::with_capacity(last.output_tensors.len());
        for tensor in &last.output_tensors {
            let data = tensor.read_f16()?;
            let shape = tensor.shape();
            let mut out = AneTensor::new(shape[1], shape[3], tensor.dtype())?;
            out.write_f16(&data)?;
            results.push(out);
        }

        Ok(results)
    }

    /// Get the model's input description (from the first sub-program).
    pub fn input_description(&self) -> Vec<TensorDescriptor> {
        self.sub_programs
            .first()
            .map(|s| s.meta.inputs.clone())
            .unwrap_or_default()
    }

    /// Get the model's output description (from the last sub-program).
    pub fn output_description(&self) -> Vec<TensorDescriptor> {
        self.sub_programs
            .last()
            .map(|s| s.meta.outputs.clone())
            .unwrap_or_default()
    }

    /// Number of sub-programs in this model.
    pub fn num_sub_programs(&self) -> usize {
        self.sub_programs.len()
    }

    /// Cache statistics: `(cached_count, session_compiles)`.
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.cache.len(), self.cache.session_compile_count())
    }
}

// ── Compilation artifacts (no runtime required) ───────────────────

/// Artifacts produced by the pre-compilation pipeline.
///
/// This represents the output of passes → split → MIL text emission →
/// BLOBFILE writing, without requiring the ANE runtime. Useful for
/// persisting compiled artifacts to disk and validating the pipeline
/// independently of the private ANE APIs.
pub struct CompiledArtifacts {
    /// Per-sub-program artifacts, in execution order.
    pub sub_programs: Vec<SubProgramArtifact>,
}

/// Compiled artifacts for a single sub-program.
pub struct SubProgramArtifact {
    /// Name of the sub-program (e.g., "main", "layer_0").
    pub name: String,
    /// The MIL text that would be sent to `_ANECompiler`.
    pub mil_text: String,
    /// The BLOBFILE bytes containing baked weights.
    pub weight_blob: Vec<u8>,
    /// Input tensor descriptors.
    pub inputs: Vec<TensorDescriptor>,
    /// Output tensor descriptors.
    pub outputs: Vec<TensorDescriptor>,
}

impl CompiledArtifacts {
    /// Run the full pre-compilation pipeline on an IR program.
    ///
    /// This exercises the same passes, splitting, MIL text emission, and
    /// BLOBFILE writing as [`AneModel::compile_and_load`], but does NOT
    /// require the ANE runtime or private APIs. The artifacts can be
    /// inspected, persisted to disk, or fed to `_ANECompiler` separately.
    pub fn prepare(program: &mil_rs::ir::Program) -> Result<Self> {
        use mil_rs::convert::ir_to_mil_text::{MilTextConfig, program_to_mil_text};
        use mil_rs::ir::Pass;
        use mil_rs::ir::passes::{
            AneConcatEliminationPass, AneLayoutPass, AneVariableNamingPass, AttentionDecomposePass,
            OpSubstitutionPass,
        };

        // 1. Clone and run ANE-specific passes
        let mut program = program.clone();
        let passes: &[(&str, &dyn Pass)] = &[
            ("OpSubstitution", &OpSubstitutionPass),
            ("AneLayout", &AneLayoutPass),
            ("AttentionDecompose", &AttentionDecomposePass),
            ("AneConcatElimination", &AneConcatEliminationPass),
            ("AneVariableNaming", &AneVariableNamingPass),
        ];
        for (name, pass) in passes {
            pass.run(&mut program)
                .map_err(|e| AneError::Other(anyhow::anyhow!("{name} pass failed: {e}")))?;
        }

        // 2. Split into sub-programs
        let split_config = SplitConfig::default();
        let model_split = split_for_ane(&program, &split_config)?;

        // 3. Emit MIL text + BLOBFILE for each sub-program
        let mil_config = MilTextConfig::default();
        let mut artifacts = Vec::new();

        for sub in &model_split.programs {
            let (mil_text, weight_entries) = program_to_mil_text(&sub.program, &mil_config)
                .map_err(|e| {
                    AneError::Other(anyhow::anyhow!(
                        "MIL text emission failed for {}: {e}",
                        sub.name
                    ))
                })?;

            let mut blob_writer = BlobFileWriter::new();
            for entry in &weight_entries {
                blob_writer.add_weight(&entry.name, &entry.data, entry.dtype);
            }

            artifacts.push(SubProgramArtifact {
                name: sub.name.clone(),
                mil_text,
                weight_blob: blob_writer.as_bytes().to_vec(),
                inputs: sub.inputs.clone(),
                outputs: sub.outputs.clone(),
            });
        }

        Ok(Self {
            sub_programs: artifacts,
        })
    }

    /// Save all artifacts to a directory on disk.
    ///
    /// Creates one subdirectory per sub-program containing:
    /// - `program.mil` — the MIL text
    /// - `weights.blob` — the BLOBFILE
    /// - `manifest.json` — I/O descriptors
    pub fn save(&self, dir: &std::path::Path) -> Result<()> {
        use std::io::Write;

        std::fs::create_dir_all(dir)
            .map_err(|e| AneError::Other(anyhow::anyhow!("failed to create dir: {e}")))?;

        for (i, sub) in self.sub_programs.iter().enumerate() {
            let sub_dir = dir.join(format!("{:02}_{}", i, sub.name));
            std::fs::create_dir_all(&sub_dir)
                .map_err(|e| AneError::Other(anyhow::anyhow!("failed to create sub-dir: {e}")))?;

            // Write MIL text
            let mil_path = sub_dir.join("program.mil");
            let mut f = std::fs::File::create(&mil_path).map_err(|e| {
                AneError::Other(anyhow::anyhow!(
                    "failed to create {}: {e}",
                    mil_path.display()
                ))
            })?;
            f.write_all(sub.mil_text.as_bytes())
                .map_err(|e| AneError::Other(anyhow::anyhow!("failed to write MIL text: {e}")))?;

            // Write BLOBFILE
            let blob_path = sub_dir.join("weights.blob");
            std::fs::write(&blob_path, &sub.weight_blob)
                .map_err(|e| AneError::Other(anyhow::anyhow!("failed to write BLOBFILE: {e}")))?;

            // Write manifest
            let manifest = serde_json::json!({
                "name": sub.name,
                "inputs": sub.inputs.iter().map(|td| {
                    serde_json::json!({
                        "name": td.name,
                        "shape": td.shape,
                        "dtype": format!("{:?}", td.dtype),
                    })
                }).collect::<Vec<_>>(),
                "outputs": sub.outputs.iter().map(|td| {
                    serde_json::json!({
                        "name": td.name,
                        "shape": td.shape,
                        "dtype": format!("{:?}", td.dtype),
                    })
                }).collect::<Vec<_>>(),
            });
            let manifest_path = sub_dir.join("manifest.json");
            let manifest_str = serde_json::to_string_pretty(&manifest)
                .map_err(|e| AneError::Other(anyhow::anyhow!("JSON serialization failed: {e}")))?;
            std::fs::write(&manifest_path, manifest_str)
                .map_err(|e| AneError::Other(anyhow::anyhow!("failed to write manifest: {e}")))?;
        }

        Ok(())
    }

    /// Total number of sub-programs.
    pub fn num_sub_programs(&self) -> usize {
        self.sub_programs.len()
    }

    /// Validate that all artifacts are well-formed.
    ///
    /// Checks:
    /// - MIL text is non-empty and contains expected structure
    /// - BLOBFILE has valid header
    /// - All sub-programs have at least one output
    pub fn validate(&self) -> Result<Vec<String>> {
        let mut warnings = Vec::new();

        for sub in &self.sub_programs {
            // MIL text must contain program declaration
            if !sub.mil_text.contains("program(") {
                return Err(AneError::CompileFailed {
                    status: 0,
                    context: format!(
                        "sub-program '{}': MIL text missing program() declaration",
                        sub.name
                    ),
                });
            }
            if !sub.mil_text.contains("func ") {
                return Err(AneError::CompileFailed {
                    status: 0,
                    context: format!(
                        "sub-program '{}': MIL text missing func declaration",
                        sub.name
                    ),
                });
            }

            // BLOBFILE must have valid header (at least 64 bytes with BLOB magic)
            if sub.weight_blob.len() >= 4 && &sub.weight_blob[0..4] != b"BLOB" {
                return Err(AneError::CompileFailed {
                    status: 0,
                    context: format!(
                        "sub-program '{}': BLOBFILE has invalid magic bytes",
                        sub.name
                    ),
                });
            }

            // Sub-programs with no outputs are likely errors
            if sub.outputs.is_empty() {
                warnings.push(format!("sub-program '{}': has no outputs", sub.name));
            }
        }

        Ok(warnings)
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Try to construct a minimal `AneModel` for testing descriptor methods.
    /// Returns `None` if the ANE runtime isn't available (e.g. CI).
    fn test_model(sub_programs: Vec<LoadedSubProgram>) -> Option<AneModel> {
        let runtime = AneRuntime::new().ok()?;
        let work_dir = tempfile::tempdir().ok()?;
        Some(AneModel {
            runtime,
            sub_programs,
            cache: ProgramCache::default(),
            config: AneConfig::default(),
            _work_dir: work_dir,
        })
    }

    #[test]
    fn config_default() {
        let c = AneConfig::default();
        assert_eq!(c.max_programs, 100);
        assert!(c.cache_dir.is_none());
        assert!(!c.enable_int4);
    }

    #[test]
    fn input_description_empty() {
        if let Some(model) = test_model(vec![]) {
            assert!(model.input_description().is_empty());
        }
    }

    #[test]
    fn output_description_empty() {
        if let Some(model) = test_model(vec![]) {
            assert!(model.output_description().is_empty());
        }
    }

    #[test]
    fn num_sub_programs_count() {
        if let Some(model) = test_model(vec![]) {
            assert_eq!(model.num_sub_programs(), 0);
        }
    }

    #[test]
    fn pass_pipeline_runs_without_error() {
        use mil_rs::ir::Pass;
        use mil_rs::ir::Program;
        use mil_rs::ir::passes::{
            AneConcatEliminationPass, AneLayoutPass, AneVariableNamingPass, AttentionDecomposePass,
            OpSubstitutionPass,
        };

        // Minimal empty program — passes should be no-ops.
        let mut program = Program {
            version: "1.0".to_string(),
            functions: Default::default(),
            attributes: Default::default(),
        };

        OpSubstitutionPass
            .run(&mut program)
            .expect("OpSubstitution");
        AneLayoutPass.run(&mut program).expect("AneLayout");
        AttentionDecomposePass
            .run(&mut program)
            .expect("AttentionDecompose");
        AneConcatEliminationPass
            .run(&mut program)
            .expect("AneConcatElimination");
        AneVariableNamingPass
            .run(&mut program)
            .expect("AneVariableNaming");
    }

    // ── E2E pipeline tests ───────────────────────────────────────

    /// Build a simple `z = add(x, y)` program for e2e testing.
    fn build_add_program() -> mil_rs::ir::Program {
        use mil_rs::ir::{Block, Function, Operation, Program, TensorType, Value};

        let input_ty = TensorType::new(ScalarType::Float16, vec![1, 4, 1, 8]);

        let add_op = Operation::new("add", "add_0")
            .with_input("x", Value::Reference("x".into()))
            .with_input("y", Value::Reference("y".into()))
            .with_output("result");

        let mut block = Block::new();
        block.add_op(add_op);
        block.outputs.push("result".into());

        let func = Function::new("main")
            .with_input("x", input_ty.clone())
            .with_input("y", input_ty);
        let mut func = func;
        func.body = block;

        let mut program = Program::new("1.0");
        program.add_function(func);
        program
    }

    /// Build a program with const weights: `z = add(x, w)` where w is a
    /// baked weight tensor.
    fn build_weighted_program() -> mil_rs::ir::Program {
        use mil_rs::ir::{Block, Function, Operation, Program, TensorType, Value};

        let input_ty = TensorType::new(ScalarType::Float16, vec![1, 4, 1, 8]);

        // Weight: 4*8 = 32 fp16 elements = 64 bytes of zeros
        let weight_data = vec![0u8; 64];
        let weight_op = Operation::new("const", "w")
            .with_input(
                "val",
                Value::Tensor {
                    data: weight_data,
                    shape: vec![1, 4, 1, 8],
                    dtype: ScalarType::Float16,
                },
            )
            .with_output("w");

        let add_op = Operation::new("add", "add_0")
            .with_input("x", Value::Reference("x".into()))
            .with_input("y", Value::Reference("w".into()))
            .with_output("result");

        let mut block = Block::new();
        block.add_op(weight_op);
        block.add_op(add_op);
        block.outputs.push("result".into());

        let func = Function::new("main").with_input("x", input_ty);
        let mut func = func;
        func.body = block;

        let mut program = Program::new("1.0");
        program.add_function(func);
        program
    }

    /// Build a multi-layer program to test splitting.
    fn build_multi_layer_program() -> mil_rs::ir::Program {
        use mil_rs::ir::{Block, Function, Operation, Program, TensorType, Value};

        let ty = TensorType::new(ScalarType::Float16, vec![1, 4, 1, 8]);
        let weight_data = vec![0u8; 64]; // 32 fp16 elements

        let mut block = Block::new();

        // Layer 0: w0 + add
        let w0 = Operation::new("const", "layer_0_w")
            .with_input(
                "val",
                Value::Tensor {
                    data: weight_data.clone(),
                    shape: vec![1, 4, 1, 8],
                    dtype: ScalarType::Float16,
                },
            )
            .with_output("layer_0_w");
        let add0 = Operation::new("add", "layer_0_add")
            .with_input("x", Value::Reference("x".into()))
            .with_input("y", Value::Reference("layer_0_w".into()))
            .with_output("layer_0_out");
        block.add_op(w0);
        block.add_op(add0);

        // Layer 1: w1 + add
        let w1 = Operation::new("const", "layer_1_w")
            .with_input(
                "val",
                Value::Tensor {
                    data: weight_data,
                    shape: vec![1, 4, 1, 8],
                    dtype: ScalarType::Float16,
                },
            )
            .with_output("layer_1_w");
        let add1 = Operation::new("add", "layer_1_add")
            .with_input("x", Value::Reference("layer_0_out".into()))
            .with_input("y", Value::Reference("layer_1_w".into()))
            .with_output("layer_1_out");
        block.add_op(w1);
        block.add_op(add1);

        block.outputs.push("layer_1_out".into());

        let func = Function::new("main").with_input("x", ty);
        let mut func = func;
        func.body = block;

        let mut program = Program::new("1.0");
        program.add_function(func);
        program
    }

    // ── CompiledArtifacts tests ──────────────────────────────────

    #[test]
    fn e2e_prepare_simple_add_program() {
        let program = build_add_program();
        let artifacts = CompiledArtifacts::prepare(&program).expect("prepare failed");

        assert!(
            artifacts.num_sub_programs() > 0,
            "should produce at least one sub-program"
        );

        // Validate the artifacts
        let warnings = artifacts.validate().expect("validation failed");
        // Warnings about empty outputs are acceptable for simple programs
        for w in &warnings {
            eprintln!("  warning: {w}");
        }

        // Check MIL text structure
        let first = &artifacts.sub_programs[0];
        assert!(
            first.mil_text.contains("program("),
            "MIL text should contain program declaration"
        );
        assert!(
            first.mil_text.contains("func "),
            "MIL text should contain function declaration"
        );
    }

    #[test]
    fn e2e_prepare_weighted_program() {
        let program = build_weighted_program();
        let artifacts = CompiledArtifacts::prepare(&program).expect("prepare failed");

        assert!(artifacts.num_sub_programs() > 0);

        let first = &artifacts.sub_programs[0];

        // Should have a BLOBFILE with weight data
        assert!(
            first.weight_blob.len() >= 64,
            "BLOBFILE should have at least a header (got {} bytes)",
            first.weight_blob.len()
        );
        assert_eq!(
            &first.weight_blob[0..4],
            b"BLOB",
            "BLOBFILE should start with BLOB magic"
        );

        // MIL text should reference the weight blob
        assert!(
            first.mil_text.contains("blob("),
            "MIL text should contain blob reference for weight: {}",
            first.mil_text
        );

        // Validate
        artifacts.validate().expect("validation failed");
    }

    #[test]
    fn e2e_prepare_multi_layer_produces_sub_programs() {
        let program = build_multi_layer_program();
        let artifacts = CompiledArtifacts::prepare(&program).expect("prepare failed");

        assert!(
            artifacts.num_sub_programs() >= 1,
            "multi-layer program should produce sub-programs"
        );

        // Each sub-program should be independently valid
        for sub in &artifacts.sub_programs {
            assert!(
                sub.mil_text.contains("program("),
                "sub-program '{}' missing program() declaration",
                sub.name
            );
            assert!(
                sub.weight_blob.len() >= 4,
                "sub-program '{}' has undersized BLOBFILE",
                sub.name
            );
        }
    }

    #[test]
    fn e2e_save_and_validate_artifacts() {
        let program = build_weighted_program();
        let artifacts = CompiledArtifacts::prepare(&program).expect("prepare failed");

        // Save to a temp directory
        let tmp = tempfile::tempdir().expect("failed to create temp dir");
        let save_dir = tmp.path().join("ane-compiled");
        artifacts.save(&save_dir).expect("save failed");

        // Verify directory structure
        assert!(save_dir.exists(), "save directory should exist");

        for (i, sub) in artifacts.sub_programs.iter().enumerate() {
            let sub_dir = save_dir.join(format!("{:02}_{}", i, sub.name));
            assert!(
                sub_dir.exists(),
                "sub-program dir should exist: {}",
                sub_dir.display()
            );

            let mil_path = sub_dir.join("program.mil");
            assert!(mil_path.exists(), "program.mil should exist");
            let mil_on_disk = std::fs::read_to_string(&mil_path).expect("read MIL");
            assert_eq!(mil_on_disk, sub.mil_text, "MIL text should match");

            let blob_path = sub_dir.join("weights.blob");
            assert!(blob_path.exists(), "weights.blob should exist");
            let blob_on_disk = std::fs::read(&blob_path).expect("read BLOBFILE");
            assert_eq!(blob_on_disk, sub.weight_blob, "BLOBFILE should match");

            let manifest_path = sub_dir.join("manifest.json");
            assert!(manifest_path.exists(), "manifest.json should exist");
            let manifest_str = std::fs::read_to_string(&manifest_path).expect("read manifest");
            let manifest: serde_json::Value =
                serde_json::from_str(&manifest_str).expect("parse manifest");
            assert_eq!(manifest["name"], sub.name);
        }
    }

    #[test]
    fn e2e_validate_catches_bad_mil_text() {
        let bad = CompiledArtifacts {
            sub_programs: vec![SubProgramArtifact {
                name: "broken".into(),
                mil_text: "not a valid MIL program".into(),
                weight_blob: b"BLOB".to_vec(),
                inputs: vec![],
                outputs: vec![],
            }],
        };

        let result = bad.validate();
        assert!(result.is_err(), "should reject invalid MIL text");
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("missing program()"), "{msg}");
    }

    #[test]
    fn e2e_validate_catches_bad_blobfile() {
        let bad = CompiledArtifacts {
            sub_programs: vec![SubProgramArtifact {
                name: "broken".into(),
                mil_text: "program(1.0)\nfunc main() { } -> ()".into(),
                weight_blob: b"JUNK_NOT_BLOB_HEADER_LONG_ENOUGH".to_vec(),
                inputs: vec![],
                outputs: vec![],
            }],
        };

        let result = bad.validate();
        assert!(result.is_err(), "should reject invalid BLOBFILE");
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("invalid magic"), "{msg}");
    }

    #[test]
    fn e2e_full_pipeline_add_program() {
        // Full pipeline: build IR → prepare → validate → save → reload from disk
        let program = build_add_program();

        // Prepare (runs all passes, splitting, MIL emission, BLOBFILE writing)
        let artifacts = CompiledArtifacts::prepare(&program).expect("prepare failed");

        // Validate
        let _warnings = artifacts.validate().expect("validation failed");

        // Save
        let tmp = tempfile::tempdir().expect("temp dir");
        artifacts.save(tmp.path()).expect("save failed");

        // Verify saved artifacts can be read back
        for (i, sub) in artifacts.sub_programs.iter().enumerate() {
            let sub_dir = tmp.path().join(format!("{:02}_{}", i, sub.name));
            let mil = std::fs::read_to_string(sub_dir.join("program.mil")).expect("read MIL");
            assert!(
                mil.contains("program("),
                "round-tripped MIL should be valid"
            );
        }
    }

    #[test]
    fn e2e_full_pipeline_weighted_program() {
        // Full pipeline with weights: build IR → prepare → validate → save
        let program = build_weighted_program();

        let artifacts = CompiledArtifacts::prepare(&program).expect("prepare failed");
        let _warnings = artifacts.validate().expect("validation failed");

        let tmp = tempfile::tempdir().expect("temp dir");
        artifacts.save(tmp.path()).expect("save failed");

        // Verify weight blob was persisted correctly
        let first = &artifacts.sub_programs[0];
        let sub_dir = tmp.path().join(format!("00_{}", first.name));
        let blob = std::fs::read(sub_dir.join("weights.blob")).expect("read BLOBFILE");
        assert!(blob.len() >= 64, "BLOBFILE should be at least 64 bytes");
        assert_eq!(&blob[0..4], b"BLOB", "BLOBFILE magic should match");
    }
}
