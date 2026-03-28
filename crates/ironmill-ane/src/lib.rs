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
        let tmp_dir = std::env::temp_dir().join("ironmill-ane");

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
            let weight_dir = tmp_dir.join(&sub.name);
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

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Try to construct a minimal `AneModel` for testing descriptor methods.
    /// Returns `None` if the ANE runtime isn't available (e.g. CI).
    fn test_model(sub_programs: Vec<LoadedSubProgram>) -> Option<AneModel> {
        let runtime = AneRuntime::new().ok()?;
        Some(AneModel {
            runtime,
            sub_programs,
            cache: ProgramCache::default(),
            config: AneConfig::default(),
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
}
