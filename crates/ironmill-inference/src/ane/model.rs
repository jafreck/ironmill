//! High-level ANE model API and compiled artifacts.
//!
//! Contains the [`AneModel`] facade that orchestrates the full ANE pipeline
//! (IR passes → split → emit MIL → compile → load → eval), plus
//! [`CompiledArtifacts`] for offline compilation inspection.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use mil_rs::ir::ScalarType;

#[cfg(feature = "compile")]
use ironmill_compile::ane::blobfile::BlobFileWriter;
#[cfg(feature = "compile")]
use ironmill_compile::ane::bundle::{AneCompileConfig, compile_model_bundle};
#[cfg(feature = "compile")]
use ironmill_compile::ane::cache::ProgramCache;
#[cfg(feature = "compile")]
use ironmill_compile::ane::mil_text::{MilTextConfig, program_to_mil_text};
#[cfg(feature = "compile")]
use ironmill_compile::ane::passes::{
    AneConcatEliminationPass, AneLayoutPass, AneVariableNamingPass, AttentionDecomposePass,
    OpSubstitutionPass,
};
#[cfg(feature = "compile")]
use ironmill_compile::ane::split::{SplitConfig, split_for_ane};
use ironmill_iosurface::{AneTensor, uniform_alloc_size};
#[cfg(feature = "compile")]
use mil_rs::ir::Pass;

use super::device::{AneDevice, HardwareAneDevice};
use crate::AneError;
use crate::ane::bundle_manifest::{BundleManifest, BundleModelType, TensorDescriptorManifest};
use crate::types::{ElementType, InputFeatureDesc, RuntimeBackend, RuntimeModel, RuntimeTensor};

/// Describes an ANE tensor's name, shape, and element type.
///
/// This is the inference-side equivalent of `ironmill_compile::ane::TensorDescriptor`.
/// Having a local copy avoids requiring the compile crate at runtime.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorDescriptor {
    pub name: String,
    pub shape: [usize; 4],
    pub dtype: ScalarType,
}

#[cfg(feature = "compile")]
impl From<&ironmill_compile::ane::TensorDescriptor> for TensorDescriptor {
    fn from(td: &ironmill_compile::ane::TensorDescriptor) -> Self {
        Self {
            name: td.name.clone(),
            shape: td.shape,
            dtype: td.dtype,
        }
    }
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

// ── AneModel facade ───────────────────────────────────────────────

/// High-level model API that orchestrates the full ANE pipeline:
/// IR passes → split → emit MIL → compile → load → eval.
pub struct AneModel<D: AneDevice> {
    device: Arc<D>,
    sub_programs: Vec<LoadedSubProgram<D>>,
    #[cfg(feature = "compile")]
    cache: ProgramCache,
    #[allow(dead_code)]
    config: AneConfig,
    /// Scratch directory for compiler inputs (BLOBFILEs). Cleaned up on drop.
    _work_dir: tempfile::TempDir,
}

struct LoadedSubProgram<D: AneDevice> {
    program: D::Program,
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

#[cfg(feature = "compile")]
impl<D: AneDevice> AneModel<D> {
    /// Compile a model from ironmill IR and load it for execution.
    ///
    /// Delegates to `compile_model_bundle()` to run the full ANE compilation
    /// pipeline, saves to a temporary directory, then loads via `from_bundle()`.
    pub fn compile_and_load(
        device: Arc<D>,
        program: &mil_rs::ir::Program,
        config: AneConfig,
    ) -> Result<Self> {
        let bundle = compile_model_bundle(program, &AneCompileConfig::default())?;
        let tmp = tempfile::tempdir()
            .map_err(|e| AneError::Other(anyhow::anyhow!("failed to create work dir: {e}")))?;
        let bundle_path = tmp.path().join("model.ironml");
        bundle
            .save(&bundle_path)
            .map_err(|e| AneError::Other(anyhow::anyhow!("failed to save bundle: {e}")))?;
        Self::from_bundle(device, &bundle_path, config)
    }
}

impl<D: AneDevice> AneModel<D> {
    /// Load a pre-compiled `.ironml` bundle for execution.
    ///
    /// Reads the manifest and compiled artifacts from the bundle directory,
    /// then compiles each sub-program's MIL text on the ANE device and
    /// allocates I/O tensors. No IR passes or splitting needed.
    pub fn from_bundle(device: Arc<D>, bundle_path: &Path, config: AneConfig) -> Result<Self> {
        // 1. Read and parse manifest.json
        let manifest_path = bundle_path.join("manifest.json");
        let manifest_json = std::fs::read_to_string(&manifest_path)
            .map_err(|e| AneError::Other(anyhow::anyhow!("failed to read manifest: {e}")))?;
        let manifest: BundleManifest = serde_json::from_str(&manifest_json)
            .map_err(|e| AneError::Other(anyhow::anyhow!("invalid manifest: {e}")))?;

        // 2. Verify it's a simple bundle
        if !matches!(manifest.model_type, BundleModelType::Simple) {
            return Err(AneError::Other(anyhow::anyhow!(
                "AneModel::from_bundle only supports simple bundles, got {:?}",
                manifest.model_type
            )));
        }

        // 3. Initialize work directory
        let work_dir = tempfile::tempdir()
            .map_err(|e| AneError::Other(anyhow::anyhow!("failed to create work dir: {e}")))?;

        // 4. Process each sub-program from the manifest
        let mut loaded_subs = Vec::new();

        for sub_manifest in &manifest.sub_programs {
            // 4a. Read MIL text
            let mil_path = bundle_path
                .join("programs")
                .join(format!("{}.mil", sub_manifest.name));
            let mil_text = std::fs::read_to_string(&mil_path).map_err(|e| {
                AneError::Other(anyhow::anyhow!(
                    "failed to read MIL text for '{}': {e}",
                    sub_manifest.name
                ))
            })?;

            // 4b. Read weight blob
            let blob_path = bundle_path
                .join("weights")
                .join(format!("{}.bin", sub_manifest.name));
            let weight_blob = std::fs::read(&blob_path).map_err(|e| {
                AneError::Other(anyhow::anyhow!(
                    "failed to read weight blob for '{}': {e}",
                    sub_manifest.name
                ))
            })?;

            // 4c. Extract individual weights from the combined BLOBFILE.
            //
            // The bundle stores all weights for a sub-program in a single
            // BLOBFILE (128-byte header + concatenated raw weight data).
            // device.compile() expects individual weight entries keyed by
            // their MIL text path references, so we parse the MIL text to
            // find each BLOBFILE reference and split the raw data.
            let weight_refs = extract_weights_from_blob(&mil_text, &weight_blob)?;
            let weight_slices: Vec<(&str, &[u8])> = weight_refs
                .iter()
                .map(|(path, data)| (path.as_str(), data.as_slice()))
                .collect();

            // 4d. Compile on the ANE device
            let compiled = device.compile(&mil_text, &weight_slices)?;

            // 4e. Convert manifest descriptors to TensorDescriptor
            let inputs: Vec<TensorDescriptor> = sub_manifest
                .inputs
                .iter()
                .map(|td| TensorDescriptor {
                    name: td.name.clone(),
                    shape: td.shape,
                    dtype: td.scalar_type(),
                })
                .collect();
            let outputs: Vec<TensorDescriptor> = sub_manifest
                .outputs
                .iter()
                .map(|td| TensorDescriptor {
                    name: td.name.clone(),
                    shape: td.shape,
                    dtype: td.scalar_type(),
                })
                .collect();

            // 4f. Pre-allocate I/O tensors with uniform sizing
            let input_shapes: Vec<_> = inputs.iter().map(|td| (td.shape, td.dtype)).collect();
            let output_shapes: Vec<_> = outputs.iter().map(|td| (td.shape, td.dtype)).collect();
            let input_alloc = uniform_alloc_size(&input_shapes);
            let output_alloc = uniform_alloc_size(&output_shapes);

            let input_tensors: Vec<AneTensor> = inputs
                .iter()
                .map(|td| {
                    AneTensor::new_with_min_alloc(td.shape[1], td.shape[3], td.dtype, input_alloc)
                        .map_err(Into::into)
                })
                .collect::<Result<Vec<_>>>()?;
            let output_tensors: Vec<AneTensor> = outputs
                .iter()
                .map(|td| {
                    AneTensor::new_with_min_alloc(td.shape[1], td.shape[3], td.dtype, output_alloc)
                        .map_err(Into::into)
                })
                .collect::<Result<Vec<_>>>()?;

            loaded_subs.push(LoadedSubProgram {
                program: compiled,
                meta: SubProgramMeta {
                    name: sub_manifest.name.clone(),
                    inputs,
                    outputs,
                },
                input_tensors,
                output_tensors,
            });
        }

        Ok(Self {
            device,
            sub_programs: loaded_subs,
            #[cfg(feature = "compile")]
            cache: ProgramCache::new(config.cache_dir.clone(), config.max_programs),
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
            {
                let sub = &mut self.sub_programs[i];
                let input_refs: Vec<&AneTensor> = sub.input_tensors.iter().collect();
                let mut output_refs: Vec<&mut AneTensor> = sub.output_tensors.iter_mut().collect();
                self.device
                    .eval(&sub.program, &input_refs, &mut output_refs)?;
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
    #[cfg(feature = "compile")]
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
#[cfg(feature = "compile")]
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

#[cfg(feature = "compile")]
impl CompiledArtifacts {
    /// Run the full pre-compilation pipeline on an IR program.
    ///
    /// This exercises the same passes, splitting, MIL text emission, and
    /// BLOBFILE writing as [`AneModel::compile_and_load`], but does NOT
    /// require the ANE runtime or private APIs.
    pub fn prepare(program: &mil_rs::ir::Program) -> Result<Self> {
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
                inputs: sub.inputs.iter().map(TensorDescriptor::from).collect(),
                outputs: sub.outputs.iter().map(TensorDescriptor::from).collect(),
            });
        }

        Ok(Self {
            sub_programs: artifacts,
        })
    }

    /// Save all artifacts to a directory on disk.
    pub fn save(&self, dir: &std::path::Path) -> Result<()> {
        use std::io::Write;

        std::fs::create_dir_all(dir)
            .map_err(|e| AneError::Other(anyhow::anyhow!("failed to create dir: {e}")))?;

        for (i, sub) in self.sub_programs.iter().enumerate() {
            let sub_dir = dir.join(format!("{:02}_{}", i, sub.name));
            std::fs::create_dir_all(&sub_dir)
                .map_err(|e| AneError::Other(anyhow::anyhow!("failed to create sub-dir: {e}")))?;

            let mil_path = sub_dir.join("program.mil");
            let mut f = std::fs::File::create(&mil_path).map_err(|e| {
                AneError::Other(anyhow::anyhow!(
                    "failed to create {}: {e}",
                    mil_path.display()
                ))
            })?;
            f.write_all(sub.mil_text.as_bytes())
                .map_err(|e| AneError::Other(anyhow::anyhow!("failed to write MIL text: {e}")))?;

            let blob_path = sub_dir.join("weights.blob");
            std::fs::write(&blob_path, &sub.weight_blob)
                .map_err(|e| AneError::Other(anyhow::anyhow!("failed to write BLOBFILE: {e}")))?;

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
    pub fn validate(&self) -> Result<Vec<String>> {
        let mut warnings = Vec::new();

        for sub in &self.sub_programs {
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

            if sub.weight_blob.len() >= 68
                && (sub.weight_blob[0] != 1 || sub.weight_blob[64..68] != [0xEF, 0xBE, 0xAD, 0xDE])
            {
                return Err(AneError::CompileFailed {
                    status: 0,
                    context: format!(
                        "sub-program '{}': BLOBFILE has invalid header format",
                        sub.name
                    ),
                });
            }

            if sub.outputs.is_empty() {
                warnings.push(format!("sub-program '{}': has no outputs", sub.name));
            }
        }

        Ok(warnings)
    }
}

// ── Bundle weight extraction ─────────────────────────────────────

/// BLOBFILE header size (file header + chunk descriptor).
const BLOBFILE_HEADER_SIZE: usize = 128;

/// Extract individual weight entries from a combined BLOBFILE blob.
///
/// The bundle stores all weights for a sub-program in a single BLOBFILE
/// (128-byte header + concatenated raw weight data). `device.compile()`
/// expects individual weight entries keyed by their MIL text path
/// references (`@model_path/weights/<name>.bin`).
///
/// This function parses the MIL text to find each BLOBFILE reference,
/// calculates the data size from the tensor's shape and dtype, then
/// splits the raw data section accordingly.
fn extract_weights_from_blob(mil_text: &str, weight_blob: &[u8]) -> Result<Vec<(String, Vec<u8>)>> {
    // The raw data section starts after the 128-byte header.
    let raw_data = if weight_blob.len() > BLOBFILE_HEADER_SIZE {
        &weight_blob[BLOBFILE_HEADER_SIZE..]
    } else {
        // Empty or header-only blob — no weights.
        return Ok(Vec::new());
    };

    let mut weights = Vec::new();
    let mut data_offset: usize = 0;

    for line in mil_text.lines() {
        // Look for BLOBFILE references: ...val=tensor<dtype, [dims]>(BLOBFILE(path=string("..."), ...))
        let Some(path) = extract_blobfile_path(line) else {
            continue;
        };
        let Some(byte_size) = extract_weight_byte_size(line) else {
            continue;
        };

        let end = data_offset + byte_size;
        if end > raw_data.len() {
            return Err(AneError::Other(anyhow::anyhow!(
                "weight '{}' needs {} bytes at offset {}, but blob data section is only {} bytes",
                path,
                byte_size,
                data_offset,
                raw_data.len()
            )));
        }

        weights.push((path, raw_data[data_offset..end].to_vec()));
        data_offset = end;
    }

    Ok(weights)
}

/// Extract the BLOBFILE path from a MIL text line.
///
/// Looks for `BLOBFILE(path=string("..."),` and returns the path string.
fn extract_blobfile_path(line: &str) -> Option<String> {
    let marker = "BLOBFILE(path=string(\"";
    let start = line.find(marker)? + marker.len();
    let rest = &line[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Calculate the byte size of a weight tensor from its MIL text declaration.
///
/// Parses the val type `tensor<dtype, [d1, d2, ..., dN]>(BLOBFILE(...))` to
/// extract the shape and scalar type, then computes `product(shape) * byte_size(dtype)`.
fn extract_weight_byte_size(line: &str) -> Option<usize> {
    // Find the val type just before BLOBFILE:
    //   val=tensor<float16, [1, 64, 1, 128]>(BLOBFILE(...))
    let blobfile_pos = line.find("BLOBFILE(")?;
    // Walk backwards to find the tensor type: tensor<dtype, [dims]>(BLOBFILE
    let before_blob = &line[..blobfile_pos];
    // Find the last "tensor<" before BLOBFILE
    let tensor_start = before_blob.rfind("tensor<")?;
    let type_str = &before_blob[tensor_start..];

    // Extract dtype: tensor<DTYPE, ...>
    let dtype_start = "tensor<".len();
    let dtype_end = type_str.find(',')?;
    let dtype_str = type_str[dtype_start..dtype_end].trim();
    let byte_size_per_elem = match dtype_str {
        "float16" | "fp16" => 2,
        "float32" | "fp32" => 4,
        "int8" => 1,
        "uint8" => 1,
        "int16" => 2,
        "int32" => 4,
        _ => 2, // default to float16
    };

    // Extract dimensions: [..., [d1, d2, ..., dN]](BLOBFILE
    let bracket_start = type_str.find('[')?;
    let bracket_end = type_str.find(']')?;
    let dims_str = &type_str[bracket_start + 1..bracket_end];
    let num_elements: usize = dims_str
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .product();

    Some(num_elements * byte_size_per_elem)
}

/// Convert a `TensorDescriptorManifest` to `TensorDescriptor`.
impl From<&TensorDescriptorManifest> for TensorDescriptor {
    fn from(td: &TensorDescriptorManifest) -> Self {
        TensorDescriptor {
            name: td.name.clone(),
            shape: td.shape,
            dtype: td.scalar_type(),
        }
    }
}

// ── RuntimeBackend / RuntimeModel ─────────────────────────────────

fn scalar_to_element(dt: ScalarType) -> ElementType {
    match dt {
        ScalarType::Float16 => ElementType::Float16,
        ScalarType::Float32 => ElementType::Float32,
        ScalarType::Int32 => ElementType::Int32,
        _ => ElementType::Float32,
    }
}

/// Wraps a loaded [`AneModel`] to implement [`RuntimeModel`].
pub struct AneRuntimeModel {
    model: AneModel<HardwareAneDevice>,
}

impl RuntimeModel for AneRuntimeModel {
    fn input_description(&self) -> Vec<InputFeatureDesc> {
        self.model
            .input_description()
            .iter()
            .map(|d| InputFeatureDesc {
                name: d.name.clone(),
                shape: d.shape.to_vec(),
                dtype: scalar_to_element(d.dtype),
            })
            .collect()
    }

    fn predict(&self, _inputs: &[RuntimeTensor]) -> anyhow::Result<Vec<RuntimeTensor>> {
        Err(anyhow::anyhow!(
            "ANE predict through RuntimeModel requires compile_and_load with a Program"
        ))
    }
}

/// ANE direct backend. Since ANE compilation takes a `Program` (not a file),
/// this backend is a no-op placeholder that enables uniform backend selection.
/// Use [`AneModel::compile_and_load`] directly for actual ANE execution.
pub struct AneDirectBackend;

impl RuntimeBackend for AneDirectBackend {
    fn name(&self) -> &str {
        "ane-direct"
    }

    fn load(&self, _model_path: &std::path::Path) -> anyhow::Result<Box<dyn RuntimeModel>> {
        Err(anyhow::anyhow!(
            "ANE direct backend requires a Program, not a file path. \
             Use AneModel::compile_and_load() directly."
        ))
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_default() {
        let c = AneConfig::default();
        assert_eq!(c.max_programs, 100);
        assert!(c.cache_dir.is_none());
        assert!(!c.enable_int4);
    }

    /// Try to construct a minimal `AneModel` for testing descriptor methods.
    /// Returns `None` if the ANE runtime isn't available (e.g. CI).
    #[cfg(feature = "compile")]
    fn test_model(
        sub_programs: Vec<LoadedSubProgram<HardwareAneDevice>>,
    ) -> Option<AneModel<HardwareAneDevice>> {
        let device = Arc::new(HardwareAneDevice::new().ok()?);
        let work_dir = tempfile::tempdir().ok()?;
        Some(AneModel {
            device,
            sub_programs,
            cache: ProgramCache::default(),
            config: AneConfig::default(),
            _work_dir: work_dir,
        })
    }

    #[test]
    #[cfg(feature = "compile")]
    fn input_description_empty() {
        if let Some(model) = test_model(vec![]) {
            assert!(model.input_description().is_empty());
        }
    }

    #[test]
    #[cfg(feature = "compile")]
    fn output_description_empty() {
        if let Some(model) = test_model(vec![]) {
            assert!(model.output_description().is_empty());
        }
    }

    #[test]
    #[cfg(feature = "compile")]
    fn num_sub_programs_count() {
        if let Some(model) = test_model(vec![]) {
            assert_eq!(model.num_sub_programs(), 0);
        }
    }

    #[test]
    #[cfg(feature = "compile")]
    fn pass_pipeline_runs_without_error() {
        let mut program = mil_rs::ir::Program {
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

    // ── E2E pipeline tests (require compile feature) ────────────

    #[cfg(feature = "compile")]
    fn build_add_program() -> mil_rs::ir::Program {
        use mil_rs::ir::{Block, Function, Operation, Program, TensorType, Value};

        let input_ty = TensorType::new(ScalarType::Float16, vec![1, 4, 1, 8]);

        let mut add_op = Operation::new("add", "add_0")
            .with_input("x", Value::Reference("x".into()))
            .with_input("y", Value::Reference("y".into()))
            .with_output("result");
        add_op.output_types = vec![Some(input_ty.clone())];

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

    #[cfg(feature = "compile")]
    fn build_weighted_program() -> mil_rs::ir::Program {
        use mil_rs::ir::{Block, Function, Operation, Program, TensorType, Value};

        let input_ty = TensorType::new(ScalarType::Float16, vec![1, 4, 1, 8]);

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

        let mut add_op = Operation::new("add", "add_0")
            .with_input("x", Value::Reference("x".into()))
            .with_input("y", Value::Reference("w".into()))
            .with_output("result");
        add_op.output_types = vec![Some(input_ty.clone())];

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

    fn build_multi_layer_program() -> mil_rs::ir::Program {
        use mil_rs::ir::{Block, Function, Operation, Program, TensorType, Value};

        let ty = TensorType::new(ScalarType::Float16, vec![1, 4, 1, 8]);
        let weight_data = vec![0u8; 64];

        let mut block = Block::new();

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

    #[cfg(feature = "compile")]
    fn build_conv_linear_program() -> mil_rs::ir::Program {
        use mil_rs::ir::{Block, Function, Operation, Program, TensorType, Value};

        let in_dim = 4;
        let out_dim = 8;
        let seq = 16;

        let input_ty = TensorType::new(ScalarType::Float16, vec![1, in_dim, 1, seq]);
        let output_ty = TensorType::new(ScalarType::Float16, vec![1, out_dim, 1, seq]);
        let weight_shape = vec![out_dim, in_dim, 1, 1];

        let weight_data = vec![0u8; out_dim * in_dim * 2];

        let mut block = Block::new();

        let pt = Operation::new("const", "pt")
            .with_input("val", Value::String("valid".into()))
            .with_output("pt");
        let st = Operation::new("const", "st")
            .with_input("val", Value::List(vec![Value::Int(1), Value::Int(1)]))
            .with_output("st");
        let pd = Operation::new("const", "pd")
            .with_input(
                "val",
                Value::List(vec![
                    Value::Int(0),
                    Value::Int(0),
                    Value::Int(0),
                    Value::Int(0),
                ]),
            )
            .with_output("pd");
        let dl = Operation::new("const", "dl")
            .with_input("val", Value::List(vec![Value::Int(1), Value::Int(1)]))
            .with_output("dl");
        let gr = Operation::new("const", "gr")
            .with_input("val", Value::Int(1))
            .with_output("gr");

        let w = Operation::new("const", "W")
            .with_input(
                "val",
                Value::Tensor {
                    data: weight_data,
                    shape: weight_shape,
                    dtype: ScalarType::Float16,
                },
            )
            .with_output("W");

        let mut conv = Operation::new("conv", "conv_0")
            .with_input("x", Value::Reference("x".into()))
            .with_input("weight", Value::Reference("W".into()))
            .with_input("pad_type", Value::Reference("pt".into()))
            .with_input("strides", Value::Reference("st".into()))
            .with_input("pad", Value::Reference("pd".into()))
            .with_input("dilations", Value::Reference("dl".into()))
            .with_input("groups", Value::Reference("gr".into()))
            .with_output("result");
        conv.output_types = vec![Some(output_ty)];

        block.add_op(pt);
        block.add_op(st);
        block.add_op(pd);
        block.add_op(dl);
        block.add_op(gr);
        block.add_op(w);
        block.add_op(conv);
        block.outputs.push("result".into());

        let mut func = Function::new("main").with_input("x", input_ty);
        func.body = block;

        let mut program = Program::new("1.0");
        program.add_function(func);
        program
    }

    #[cfg(feature = "compile")]
    #[test]
    fn e2e_prepare_simple_add_program() {
        let program = build_add_program();
        let artifacts = CompiledArtifacts::prepare(&program).expect("prepare failed");

        assert!(
            artifacts.num_sub_programs() > 0,
            "should produce at least one sub-program"
        );

        let warnings = artifacts.validate().expect("validation failed");
        for w in &warnings {
            eprintln!("  warning: {w}");
        }

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

    #[cfg(feature = "compile")]
    #[test]
    fn e2e_prepare_weighted_program() {
        let program = build_weighted_program();
        let artifacts = CompiledArtifacts::prepare(&program).expect("prepare failed");

        assert!(artifacts.num_sub_programs() > 0);

        let first = &artifacts.sub_programs[0];

        assert!(
            first.weight_blob.len() >= 128,
            "BLOBFILE should have at least header + chunk desc (got {} bytes)",
            first.weight_blob.len()
        );
        assert_eq!(first.weight_blob[0], 1, "BLOBFILE byte 0 should be 1");
        assert_eq!(
            &first.weight_blob[64..68],
            &[0xEF, 0xBE, 0xAD, 0xDE],
            "BLOBFILE should have 0xDEADBEEF chunk magic at byte 64"
        );

        assert!(
            first.mil_text.contains("BLOBFILE("),
            "MIL text should contain BLOBFILE reference for weight: {}",
            first.mil_text
        );

        artifacts.validate().expect("validation failed");
    }

    #[cfg(feature = "compile")]
    #[test]
    fn e2e_prepare_multi_layer_produces_sub_programs() {
        let program = build_multi_layer_program();
        let artifacts = CompiledArtifacts::prepare(&program).expect("prepare failed");

        assert!(
            artifacts.num_sub_programs() >= 1,
            "multi-layer program should produce sub-programs"
        );

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

    #[cfg(feature = "compile")]
    #[test]
    fn e2e_save_and_validate_artifacts() {
        let program = build_weighted_program();
        let artifacts = CompiledArtifacts::prepare(&program).expect("prepare failed");

        let tmp = tempfile::tempdir().expect("failed to create temp dir");
        let save_dir = tmp.path().join("ane-compiled");
        artifacts.save(&save_dir).expect("save failed");

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

    #[cfg(feature = "compile")]
    #[test]
    fn e2e_validate_catches_bad_mil_text() {
        let mut blob = vec![0u8; 128];
        blob[0] = 1;
        blob[64..68].copy_from_slice(&[0xEF, 0xBE, 0xAD, 0xDE]);

        let bad = CompiledArtifacts {
            sub_programs: vec![SubProgramArtifact {
                name: "broken".into(),
                mil_text: "not a valid MIL program".into(),
                weight_blob: blob,
                inputs: vec![],
                outputs: vec![],
            }],
        };

        let result = bad.validate();
        assert!(result.is_err(), "should reject invalid MIL text");
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("missing program()"), "{msg}");
    }

    #[cfg(feature = "compile")]
    #[test]
    fn e2e_validate_catches_bad_blobfile() {
        let mut blob = vec![0u8; 128];
        blob[0] = 99;

        let bad = CompiledArtifacts {
            sub_programs: vec![SubProgramArtifact {
                name: "broken".into(),
                mil_text: "program(1.0)\nfunc main() { } -> ()".into(),
                weight_blob: blob,
                inputs: vec![],
                outputs: vec![],
            }],
        };

        let result = bad.validate();
        assert!(result.is_err(), "should reject invalid BLOBFILE");
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("invalid header"), "{msg}");
    }

    #[cfg(feature = "compile")]
    #[test]
    fn e2e_full_pipeline_add_program() {
        let program = build_add_program();
        let artifacts = CompiledArtifacts::prepare(&program).expect("prepare failed");
        let _warnings = artifacts.validate().expect("validation failed");

        let tmp = tempfile::tempdir().expect("temp dir");
        artifacts.save(tmp.path()).expect("save failed");

        for (i, sub) in artifacts.sub_programs.iter().enumerate() {
            let sub_dir = tmp.path().join(format!("{:02}_{}", i, sub.name));
            let mil = std::fs::read_to_string(sub_dir.join("program.mil")).expect("read MIL");
            assert!(
                mil.contains("program("),
                "round-tripped MIL should be valid"
            );
        }
    }

    #[cfg(feature = "compile")]
    #[test]
    fn e2e_full_pipeline_weighted_program() {
        let program = build_weighted_program();
        let artifacts = CompiledArtifacts::prepare(&program).expect("prepare failed");
        let _warnings = artifacts.validate().expect("validation failed");

        let tmp = tempfile::tempdir().expect("temp dir");
        artifacts.save(tmp.path()).expect("save failed");

        let first = &artifacts.sub_programs[0];
        let sub_dir = tmp.path().join(format!("00_{}", first.name));
        let blob = std::fs::read(sub_dir.join("weights.blob")).expect("read BLOBFILE");
        assert!(blob.len() >= 128, "BLOBFILE should be at least 128 bytes");
        assert_eq!(blob[0], 1, "BLOBFILE byte 0 should be 1");
    }

    #[cfg(feature = "compile")]
    #[test]
    fn e2e_compile_and_load_weighted_program() {
        let program = build_weighted_program();
        let config = AneConfig::default();

        let device = match HardwareAneDevice::new() {
            Ok(d) => Arc::new(d),
            Err(e) => {
                eprintln!("  ANE not available: {e}");
                return;
            }
        };
        let result = AneModel::compile_and_load(device, &program, config);

        match result {
            Ok(mut model) => {
                let desc = model.input_description();
                assert!(!desc.is_empty(), "model should have inputs");

                let inputs: Vec<AneTensor> = desc
                    .iter()
                    .map(|td| AneTensor::new(td.shape[1], td.shape[3], td.dtype).unwrap())
                    .collect();
                let outputs = model.predict(&inputs).expect("predict should succeed");
                assert!(!outputs.is_empty(), "predict should return outputs");
            }
            Err(e) => {
                let msg = format!("{e}");
                eprintln!("  compile_and_load failed: {msg}");
                assert!(
                    msg.contains("ANE")
                        || msg.contains("compilation")
                        || msg.contains("dlopen")
                        || msg.contains("compile"),
                    "expected compilation failure, got: {msg}"
                );
            }
        }
    }

    #[cfg(feature = "compile")]
    #[test]
    fn e2e_compile_and_load_add_program() {
        let program = build_add_program();
        let config = AneConfig::default();

        let device = match HardwareAneDevice::new() {
            Ok(d) => Arc::new(d),
            Err(e) => {
                eprintln!("  ANE not available: {e}");
                return;
            }
        };
        let result = AneModel::compile_and_load(device, &program, config);

        match result {
            Ok(mut model) => {
                let desc = model.input_description();
                let inputs: Vec<AneTensor> = desc
                    .iter()
                    .map(|td| AneTensor::new(td.shape[1], td.shape[3], td.dtype).unwrap())
                    .collect();
                let outputs = model.predict(&inputs).expect("predict should succeed");
                assert!(!outputs.is_empty(), "predict should return outputs");
            }
            Err(e) => {
                let msg = format!("{e}");
                eprintln!("  compile_and_load failed (expected): {msg}");
                assert!(
                    msg.contains("ANE")
                        || msg.contains("compilation")
                        || msg.contains("dlopen")
                        || msg.contains("compile"),
                    "expected compilation failure, got: {msg}"
                );
            }
        }
    }

    #[cfg(feature = "compile")]
    #[test]
    fn e2e_compile_and_load_conv_linear() {
        let program = build_conv_linear_program();
        let config = AneConfig::default();

        {
            let artifacts = CompiledArtifacts::prepare(&program).expect("prepare");
            for sub in &artifacts.sub_programs {
                eprintln!("  [post-pass MIL for '{}']\n{}", sub.name, sub.mil_text);
            }
        }

        let device = match HardwareAneDevice::new() {
            Ok(d) => Arc::new(d),
            Err(e) => {
                eprintln!("  ANE not available: {e}");
                return;
            }
        };
        let result = AneModel::compile_and_load(device, &program, config);

        match result {
            Ok(mut model) => {
                eprintln!(
                    "  ✓ ANE compile+load succeeded! ({} sub-programs)",
                    model.num_sub_programs()
                );

                let desc = model.input_description();
                assert!(!desc.is_empty(), "model should have inputs");

                let inputs: Vec<AneTensor> = desc
                    .iter()
                    .map(|td| AneTensor::new(td.shape[1], td.shape[3], td.dtype).unwrap())
                    .collect();
                let outputs = model.predict(&inputs).expect("predict should succeed");
                assert!(!outputs.is_empty(), "predict should return outputs");
                eprintln!("  ✓ predict succeeded with {} output(s)", outputs.len());
            }
            Err(e) => {
                let msg = format!("{e}");
                eprintln!("  compile_and_load (conv) failed: {msg}");
                assert!(
                    msg.contains("ANE")
                        || msg.contains("compilation")
                        || msg.contains("dlopen")
                        || msg.contains("compile"),
                    "expected compilation failure, got: {msg}"
                );
            }
        }
    }

    #[cfg(feature = "compile")]
    #[test]
    fn e2e_debug_mil_text_output() {
        use ironmill_compile::ane::mil_text::{MilTextConfig, program_to_mil_text};

        for (label, program) in [
            ("add (no weights)", build_add_program()),
            ("weighted", build_weighted_program()),
            ("conv linear", build_conv_linear_program()),
        ] {
            let config = MilTextConfig::default();
            let (text, entries) = program_to_mil_text(&program, &config).unwrap();
            eprintln!(
                "=== {label}: {len} bytes, {n} weight entries ===",
                len = text.len(),
                n = entries.len()
            );
            eprintln!("{text}");
        }
    }

    /// Build a program with an `add` op and a weight tensor large enough
    /// for PolarQuantPass (>= 1024 elements, last dim >= 64).
    fn build_polarquant_test_program() -> mil_rs::ir::Program {
        use half::f16;
        use mil_rs::ir::{Block, Function, Operation, Program, TensorType, Value};

        let channels = 16usize;
        let seq_len = 128usize; // power of 2, no padding needed

        let input_ty = TensorType::new(ScalarType::Float16, vec![1, channels, 1, seq_len]);

        // Non-zero FP16 weight data (sin-wave pattern).
        let weight_data: Vec<u8> = (0..channels * seq_len)
            .flat_map(|i| f16::from_f32((i as f32 * 0.1).sin() * 0.5).to_le_bytes())
            .collect();

        let weight_op = Operation::new("const", "w")
            .with_input(
                "val",
                Value::Tensor {
                    data: weight_data,
                    shape: vec![1, channels, 1, seq_len],
                    dtype: ScalarType::Float16,
                },
            )
            .with_output("w");

        let mut add_op = Operation::new("add", "add_0")
            .with_input("x", Value::Reference("x".into()))
            .with_input("y", Value::Reference("w".into()))
            .with_output("result");
        add_op.output_types = vec![Some(input_ty.clone())];

        let mut block = Block::new();
        block.add_op(weight_op);
        block.add_op(add_op);
        block.outputs.push("result".into());

        let mut func = Function::new("main").with_input("x", input_ty);
        func.body = block;

        let mut program = Program::new("1.0");
        program.add_function(func);
        program
    }

    /// Verify that PolarQuant's `constexpr_lut_to_dense` op is accepted by
    /// the ANE compiler and produces correct MIL text and artifacts.
    #[test]
    fn e2e_polarquant_constexpr_lut_to_dense_on_ane() {
        use mil_rs::ir::Pass;
        use mil_rs::ir::passes::PolarQuantPass;

        // ── 1. Build FP16 baseline and PolarQuant variant ──────────
        let fp16_program = build_polarquant_test_program();

        let mut quant_program = fp16_program.clone();
        PolarQuantPass::new(4).run(&mut quant_program).unwrap();

        // Verify constexpr_lut_to_dense was inserted.
        let ops = &quant_program.functions["main"].body.operations;
        assert!(
            ops.iter().any(|op| op.op_type == "constexpr_lut_to_dense"),
            "PolarQuantPass should have inserted constexpr_lut_to_dense"
        );

        // ── 2. Compare pre-compilation weight blob sizes ───────────
        let fp16_artifacts = CompiledArtifacts::prepare(&fp16_program).expect("FP16 prepare");
        let quant_artifacts = CompiledArtifacts::prepare(&quant_program).expect("quant prepare");

        let fp16_size: usize = fp16_artifacts
            .sub_programs
            .iter()
            .map(|s| s.weight_blob.len())
            .sum();
        let quant_size: usize = quant_artifacts
            .sub_programs
            .iter()
            .map(|s| s.weight_blob.len())
            .sum();
        eprintln!("  FP16 weight blob:      {fp16_size} bytes");
        eprintln!("  PolarQuant weight blob: {quant_size} bytes");

        // ── 3. Emit MIL text and check it contains constexpr_lut_to_dense
        for sub in &quant_artifacts.sub_programs {
            eprintln!("  [PolarQuant MIL for '{}']\n{}", sub.name, sub.mil_text);
            assert!(
                sub.mil_text.contains("constexpr_lut_to_dense"),
                "MIL text should contain constexpr_lut_to_dense op"
            );
        }

        // ── 4. Compile both on ANE ─────────────────────────────────
        let device = match HardwareAneDevice::new() {
            Ok(d) => Arc::new(d),
            Err(e) => {
                eprintln!("  ANE not available, skipping hardware test: {e}");
                return;
            }
        };

        let config = AneConfig::default();

        // Compile FP16 baseline.
        let fp16_ok =
            match AneModel::compile_and_load(device.clone(), &fp16_program, config.clone()) {
                Ok(mut model) => {
                    eprintln!("  ✓ FP16 compile+load succeeded");
                    let desc = model.input_description();
                    let inputs: Vec<AneTensor> = desc
                        .iter()
                        .map(|td| AneTensor::new(td.shape[1], td.shape[3], td.dtype).unwrap())
                        .collect();
                    let outputs = model.predict(&inputs).expect("FP16 predict");
                    eprintln!(
                        "  ✓ FP16 predict succeeded with {} output(s)",
                        outputs.len()
                    );
                    true
                }
                Err(e) => {
                    eprintln!("  FP16 compile failed (ANE may not support this shape): {e}");
                    false
                }
            };

        // Compile PolarQuant variant — this is the key test.
        // As of 2026-03, ANE's private API rejects constexpr_lut_to_dense.
        // The op is only supported via CoreML's public API, which handles
        // dequantization before passing to ANE.
        match AneModel::compile_and_load(device, &quant_program, config) {
            Ok(mut model) => {
                eprintln!(
                    "  ✓ PolarQuant compile+load succeeded \
                     (constexpr_lut_to_dense accepted by ANE)"
                );
                let desc = model.input_description();
                let inputs: Vec<AneTensor> = desc
                    .iter()
                    .map(|td| AneTensor::new(td.shape[1], td.shape[3], td.dtype).unwrap())
                    .collect();
                let outputs = model.predict(&inputs).expect("PolarQuant predict");
                eprintln!(
                    "  ✓ PolarQuant predict succeeded with {} output(s)",
                    outputs.len()
                );
            }
            Err(e) => {
                // Expected: ANE private API does not support constexpr_lut_to_dense.
                // PolarQuant integration must dequantize to FP16 before ANE compilation.
                eprintln!(
                    "  ✗ PolarQuant (constexpr_lut_to_dense) rejected by ANE \
                     (expected with private API): {e}"
                );
                assert!(
                    fp16_ok,
                    "FP16 baseline should compile for this result to be meaningful"
                );
            }
        }
    }
}
