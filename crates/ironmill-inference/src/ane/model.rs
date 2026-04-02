//! High-level ANE model API.
//!
//! Contains the [`AneModel`] facade for loading and evaluating pre-compiled
//! `.ironml` bundles on the ANE.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use mil_rs::ir::ScalarType;

use ironmill_iosurface::{AneTensor, uniform_alloc_size};

use super::device::{AneDevice, HardwareAneDevice};
use crate::AneError;
use crate::ane::bundle_manifest::{BundleManifest, BundleModelType, TensorDescriptorManifest};
use crate::types::{ElementType, InputFeatureDesc, RuntimeBackend, RuntimeModel, RuntimeTensor};

/// Describes an ANE tensor's name, shape, and element type.
///
/// This is the inference-side equivalent of the compile-side TensorDescriptor.
/// Having a local copy avoids requiring the compile crate at runtime.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorDescriptor {
    pub name: String,
    pub shape: [usize; 4],
    pub dtype: ScalarType,
}

impl From<&ironmill_core::ane::TensorDescriptor> for TensorDescriptor {
    fn from(td: &ironmill_core::ane::TensorDescriptor) -> Self {
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
    /// ANE QoS level for compile/load/eval.
    /// Defaults to user-interactive (33) for interactive inference.
    pub qos: u32,
    /// Enable ANE hardware performance profiling.
    /// When set, `eval_with_stats()` is used instead of `eval()` and a
    /// per-layer timing summary is printed after each decode step.
    /// Has zero overhead when disabled (the normal eval path is used).
    pub enable_profiling: bool,
    /// Enable chained (pipelined) execution of layer programs.
    /// When set, the decode loop attempts to build a `ChainingRequest`
    /// that pipelines all pre_attn/post_attn programs, eliminating
    /// per-layer CPU↔ANE roundtrips. Falls back to per-layer eval if
    /// chaining fails. **Experimental — the ANE chaining API is
    /// undocumented and runtime behavior may vary by hardware.**
    pub enable_chaining: bool,
    /// Enable MIL program fusion to reduce per-eval dispatch overhead.
    ///
    /// When set, fused programs are used where available:
    /// - Cache-write K/V fusion (already the default — `build_cache_write_program`
    ///   produces a single program for both K and V).
    /// - FFN fusion: gate_proj + SiLU + up_proj + down_proj combined into a
    ///   single program, saving 2-3 eval dispatches per layer.
    ///
    /// Fused programs may have higher op counts and larger I/O tensor
    /// allocations. Falls back to unfused execution if a fused program
    /// fails to compile or exceeds ANE limits.
    pub enable_fusion: bool,
    /// Enable experimental hybrid ANE↔GPU execution via shared events.
    ///
    /// When set, the decode loop attempts to coordinate ANE and Metal GPU
    /// work using `MTLSharedEvent`-backed `SharedSignalEvent`/`SharedWaitEvent`
    /// fences. This removes the CPU from the critical path between GPU
    /// projections and ANE attention.
    ///
    /// **Requires** shared event support verified by probe 15
    /// (`probe_shared_events`). Falls back to the standard per-layer
    /// eval path if shared events are not available or the handoff fails.
    ///
    /// **Highly experimental — disabled by default.**
    pub enable_hybrid: bool,
}

impl Default for AneConfig {
    fn default() -> Self {
        Self {
            max_programs: 100,
            cache_dir: None,
            enable_int4: false,
            // QoSMapper::ane_user_interactive_task_qos() == 33
            qos: 33,
            enable_profiling: false,
            enable_chaining: false,
            enable_fusion: false,
            enable_hybrid: false,
        }
    }
}

// ── AneModel facade ───────────────────────────────────────────────

/// High-level model API for loading and evaluating pre-compiled ANE bundles.
pub struct AneModel<D: AneDevice> {
    device: Arc<D>,
    sub_programs: Vec<LoadedSubProgram<D>>,
    /// ANE QoS level for eval calls.
    qos: u32,
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
    inputs: Vec<TensorDescriptor>,
    outputs: Vec<TensorDescriptor>,
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
            let compiled = device.compile(&mil_text, &weight_slices, config.qos)?;

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
                meta: SubProgramMeta { inputs, outputs },
                input_tensors,
                output_tensors,
            });
        }

        Ok(Self {
            device,
            sub_programs: loaded_subs,
            qos: config.qos,
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
                    .eval(&sub.program, &input_refs, &mut output_refs, self.qos)?;
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
        other => {
            eprintln!(
                "warning: unknown dtype '{other}' in tensor type annotation, defaulting to fp16 (2 bytes)"
            );
            2
        }
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
        // TODO: ANE RuntimeModel::predict is not yet implemented
        Err(anyhow::anyhow!(
            "ANE predict through RuntimeModel is not yet implemented — \
             use AneModel::from_bundle() and AneModel::predict() directly for ANE inference"
        ))
    }
}

/// ANE direct backend. Since ANE requires a pre-compiled bundle,
/// this backend is a no-op placeholder that enables uniform backend selection.
/// Use [`AneModel::from_bundle`] directly for actual ANE execution.
pub struct AneDirectBackend;

impl RuntimeBackend for AneDirectBackend {
    fn name(&self) -> &str {
        "ane-direct"
    }

    fn load(&self, _model_path: &std::path::Path) -> anyhow::Result<Box<dyn RuntimeModel>> {
        Err(anyhow::anyhow!(
            "ANE direct backend requires a pre-compiled bundle. \
             Use AneModel::from_bundle() directly."
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
        assert_eq!(c.qos, 33);
        assert!(!c.enable_profiling);
        assert!(!c.enable_chaining);
        assert!(!c.enable_hybrid);
    }
}
