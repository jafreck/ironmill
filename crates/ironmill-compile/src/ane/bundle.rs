//! `.ironml` bundle format — compiled ANE model artifacts.
//!
//! This module defines the on-disk artifact types and manifest format for
//! compiled ANE models. Bundles are directory trees containing MIL programs,
//! weight blobs, and a JSON manifest that captures all metadata needed for
//! runtime loading.

use std::fs;
use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use mil_rs::ir::{Pass, Program};

use crate::ane::TensorDescriptor;
use crate::ane::blobfile::BlobFileWriter;
use crate::ane::mil_text::{MilTextConfig, program_to_mil_text};
use crate::ane::packing::InputPacking;
use crate::ane::passes::{
    AneConcatEliminationPass, AneLayoutPass, AneVariableNamingPass, AttentionDecomposePass,
    OpSubstitutionPass,
};
use crate::ane::split::{SplitConfig, split_for_ane};

// ---------------------------------------------------------------------------
// Bundle types
// ---------------------------------------------------------------------------

/// Compiled ANE model — ready for runtime loading.
/// Used by the "simple" compilation path (AneModel).
#[derive(Debug)]
pub struct AneModelBundle {
    pub sub_programs: Vec<SubProgramBundle>,
}

/// Single ANE sub-program artifact.
#[derive(Debug)]
pub struct SubProgramBundle {
    pub name: String,
    pub mil_text: String,
    pub weight_blob: Vec<u8>,
    pub inputs: Vec<BundleTensorDescriptor>,
    pub outputs: Vec<BundleTensorDescriptor>,
    pub input_packing: Option<BundleInputPacking>,
}

/// Tensor descriptor for bundle manifest (serializable).
///
/// This is the bundle-format equivalent of the compile-crate's
/// [`TensorDescriptor`]. Uses a string dtype for JSON compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleTensorDescriptor {
    pub name: String,
    pub shape: [usize; 4],
    /// Scalar type as a string, e.g. `"Float16"`, `"Float32"`.
    pub dtype: String,
}

/// Input packing metadata for bundle manifest (serializable).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleInputPacking {
    /// Spatial offset for each original input within the packed tensor.
    pub offsets: Vec<usize>,
    /// Spatial size (S dimension) of each original input.
    pub sizes: Vec<usize>,
}

/// Compiled autoregressive decode model.
#[derive(Debug)]
pub struct AneDecodeBundle {
    pub architecture: BundleArchitecture,
    pub rope_cos: Vec<u8>,
    pub rope_sin: Vec<u8>,
    pub embedding_weights: Vec<u8>,
    pub lm_head: LmHeadBundle,
    pub final_norm_weight: Option<Vec<u8>>,
    pub layers: Vec<LayerBundle>,
}

/// Architecture metadata for decode bundles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleArchitecture {
    pub vocab_size: usize,
    pub eos_tokens: Vec<u32>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub rope_theta: f64,
    pub max_seq_len: usize,
    pub qk_norm: bool,
}

/// Single transformer layer artifact.
#[derive(Debug)]
pub struct LayerBundle {
    pub index: usize,
    pub pre_attn: SubProgramBundle,
    pub post_attn: Option<SubProgramBundle>,
    pub fp16_attn: Option<SubProgramBundle>,
    pub cache_write_fused: bool,
    pub donor_compatible: bool,
}

/// LM head bundle — can be ANE-chunked or CPU fallback.
#[derive(Debug)]
pub enum LmHeadBundle {
    /// ANE-accelerated chunked lm_head.
    Ane {
        chunks: Vec<SubProgramBundle>,
        vocab_size: usize,
        hidden_size: usize,
    },
    /// CPU fallback: raw fp16 weight bytes `[vocab_size, hidden_size]`.
    Cpu {
        weight_data: Vec<u8>,
        vocab_size: usize,
        hidden_size: usize,
    },
}

// ---------------------------------------------------------------------------
// Manifest types
// ---------------------------------------------------------------------------

/// Top-level manifest for `.ironml` bundles.
#[derive(Debug, Serialize, Deserialize)]
pub struct BundleManifest {
    pub format_version: u32,
    pub model_type: BundleModelType,
    /// Per-sub-program metadata (for simple bundles).
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub sub_programs: Vec<SubProgramManifest>,
    /// Decode-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode: Option<DecodeManifest>,
}

/// Discriminator for the kind of model stored in a bundle.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BundleModelType {
    Simple,
    Decode,
}

/// Manifest entry for a single ANE sub-program.
#[derive(Debug, Serialize, Deserialize)]
pub struct SubProgramManifest {
    pub name: String,
    pub inputs: Vec<BundleTensorDescriptor>,
    pub outputs: Vec<BundleTensorDescriptor>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_packing: Option<BundleInputPacking>,
}

/// Decode-specific section of the bundle manifest.
#[derive(Debug, Serialize, Deserialize)]
pub struct DecodeManifest {
    pub architecture: BundleArchitecture,
    pub lm_head: LmHeadManifest,
    pub layers: Vec<LayerManifest>,
}

/// Manifest entry for the LM head.
#[derive(Debug, Serialize, Deserialize)]
pub struct LmHeadManifest {
    /// `"ane"` or `"cpu"`.
    pub kind: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    /// ANE chunk sub-program manifests (empty for CPU fallback).
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub chunks: Vec<SubProgramManifest>,
}

/// Manifest entry for a single transformer layer.
#[derive(Debug, Serialize, Deserialize)]
pub struct LayerManifest {
    pub index: usize,
    pub pre_attn: SubProgramManifest,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub post_attn: Option<SubProgramManifest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fp16_attn: Option<SubProgramManifest>,
    pub cache_write_fused: bool,
    pub donor_compatible: bool,
}

// ---------------------------------------------------------------------------
// Conversions from compile-crate types
// ---------------------------------------------------------------------------

impl From<&TensorDescriptor> for BundleTensorDescriptor {
    fn from(td: &TensorDescriptor) -> Self {
        Self {
            name: td.name.clone(),
            shape: td.shape,
            dtype: format!("{:?}", td.dtype),
        }
    }
}

impl From<&InputPacking> for BundleInputPacking {
    fn from(ip: &InputPacking) -> Self {
        Self {
            offsets: ip.offsets.clone(),
            sizes: ip.sizes.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Simple-path compilation
// ---------------------------------------------------------------------------

/// Configuration for simple model compilation.
#[derive(Debug, Default)]
pub struct AneCompileConfig {
    /// Enable INT4 data type support.
    pub enable_int4: bool,
}

/// Compile a MIL IR program into an ANE model bundle.
///
/// Runs the full ANE compilation pipeline:
/// 1. ANE-specific passes (op substitution, layout, attention decompose,
///    concat elimination, variable naming)
/// 2. Split into sub-programs
/// 3. Emit MIL text + weight blobs for each sub-program
///
/// The returned bundle can be saved to disk with [`AneModelBundle::save()`]
/// and loaded by the inference engine without any compile-crate dependency.
pub fn compile_model_bundle(
    program: &Program,
    config: &AneCompileConfig,
) -> Result<AneModelBundle> {
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
            .map_err(|e| anyhow::anyhow!("{name} pass failed: {e}"))?;
    }

    // 2. Split into sub-programs
    let split_config = SplitConfig::default();
    let model_split = split_for_ane(&program, &split_config)?;

    // 3. Emit MIL text + weight blob for each sub-program
    let mil_config = MilTextConfig {
        enable_int4: config.enable_int4,
        ..MilTextConfig::default()
    };
    let mut sub_programs = Vec::with_capacity(model_split.programs.len());

    for sub in &model_split.programs {
        let (mil_text, weight_entries) = program_to_mil_text(&sub.program, &mil_config)
            .map_err(|e| anyhow::anyhow!("MIL text emission failed for {}: {e}", sub.name))?;

        let mut blob_writer = BlobFileWriter::new();
        for entry in &weight_entries {
            blob_writer.add_weight(&entry.name, &entry.data, entry.dtype);
        }

        sub_programs.push(SubProgramBundle {
            name: sub.name.clone(),
            mil_text,
            weight_blob: blob_writer.as_bytes().to_vec(),
            inputs: sub
                .inputs
                .iter()
                .map(BundleTensorDescriptor::from)
                .collect(),
            outputs: sub
                .outputs
                .iter()
                .map(BundleTensorDescriptor::from)
                .collect(),
            input_packing: None,
        });
    }

    Ok(AneModelBundle { sub_programs })
}

// ---------------------------------------------------------------------------
// Save helpers
// ---------------------------------------------------------------------------

/// Write a sub-program's MIL text and weight blob into the given directories,
/// and return a [`SubProgramManifest`] entry.
fn save_sub_program(
    sp: &SubProgramBundle,
    programs_dir: &Path,
    weights_dir: &Path,
) -> Result<SubProgramManifest> {
    fs::write(programs_dir.join(format!("{}.mil", sp.name)), &sp.mil_text)?;
    fs::write(
        weights_dir.join(format!("{}.bin", sp.name)),
        &sp.weight_blob,
    )?;

    Ok(SubProgramManifest {
        name: sp.name.clone(),
        inputs: sp.inputs.clone(),
        outputs: sp.outputs.clone(),
        input_packing: sp.input_packing.clone(),
    })
}

// ---------------------------------------------------------------------------
// Save implementations
// ---------------------------------------------------------------------------

impl AneModelBundle {
    /// Save this bundle as a `.ironml` directory.
    ///
    /// Directory layout:
    /// ```text
    /// <path>/
    ///   manifest.json
    ///   programs/
    ///     <name>.mil
    ///   weights/
    ///     <name>.bin
    /// ```
    pub fn save(&self, path: &Path) -> Result<()> {
        let programs_dir = path.join("programs");
        let weights_dir = path.join("weights");
        fs::create_dir_all(&programs_dir)?;
        fs::create_dir_all(&weights_dir)?;

        let mut sub_manifests = Vec::with_capacity(self.sub_programs.len());
        for sp in &self.sub_programs {
            sub_manifests.push(save_sub_program(sp, &programs_dir, &weights_dir)?);
        }

        let manifest = BundleManifest {
            format_version: 1,
            model_type: BundleModelType::Simple,
            sub_programs: sub_manifests,
            decode: None,
        };

        let json = serde_json::to_string_pretty(&manifest)?;
        fs::write(path.join("manifest.json"), json)?;
        Ok(())
    }
}

impl AneDecodeBundle {
    /// Save this bundle as a `.ironml` directory.
    ///
    /// Directory layout:
    /// ```text
    /// <path>/
    ///   manifest.json
    ///   programs/
    ///     layer_<i>_pre_attn.mil
    ///     layer_<i>_post_attn.mil   (if present)
    ///     layer_<i>_fp16_attn.mil   (if present)
    ///     lm_head_chunk_<i>.mil     (if ANE lm_head)
    ///   weights/
    ///     layer_<i>_pre_attn.bin
    ///     ...
    ///   cpu_weights/
    ///     embedding.bin
    ///     lm_head.bin               (if CPU fallback)
    ///     rope_cos.bin
    ///     rope_sin.bin
    ///     final_norm.bin            (if present)
    /// ```
    pub fn save(&self, path: &Path) -> Result<()> {
        let programs_dir = path.join("programs");
        let weights_dir = path.join("weights");
        let cpu_weights_dir = path.join("cpu_weights");
        fs::create_dir_all(&programs_dir)?;
        fs::create_dir_all(&weights_dir)?;
        fs::create_dir_all(&cpu_weights_dir)?;

        // --- CPU weights ---
        fs::write(
            cpu_weights_dir.join("embedding.bin"),
            &self.embedding_weights,
        )?;
        fs::write(cpu_weights_dir.join("rope_cos.bin"), &self.rope_cos)?;
        fs::write(cpu_weights_dir.join("rope_sin.bin"), &self.rope_sin)?;
        if let Some(norm) = &self.final_norm_weight {
            fs::write(cpu_weights_dir.join("final_norm.bin"), norm)?;
        }

        // --- LM head ---
        let lm_head_manifest = match &self.lm_head {
            LmHeadBundle::Ane {
                chunks,
                vocab_size,
                hidden_size,
            } => {
                let mut chunk_manifests = Vec::with_capacity(chunks.len());
                for (i, chunk) in chunks.iter().enumerate() {
                    let sp = chunk.clone_with_name(format!("lm_head_chunk_{i}"));
                    chunk_manifests.push(save_sub_program(&sp, &programs_dir, &weights_dir)?);
                }
                LmHeadManifest {
                    kind: "ane".to_string(),
                    vocab_size: *vocab_size,
                    hidden_size: *hidden_size,
                    chunks: chunk_manifests,
                }
            }
            LmHeadBundle::Cpu {
                weight_data,
                vocab_size,
                hidden_size,
            } => {
                fs::write(cpu_weights_dir.join("lm_head.bin"), weight_data)?;
                LmHeadManifest {
                    kind: "cpu".to_string(),
                    vocab_size: *vocab_size,
                    hidden_size: *hidden_size,
                    chunks: Vec::new(),
                }
            }
        };

        // --- Layers ---
        let mut layer_manifests = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let pre_attn = save_sub_program(&layer.pre_attn, &programs_dir, &weights_dir)?;

            let post_attn = layer
                .post_attn
                .as_ref()
                .map(|sp| save_sub_program(sp, &programs_dir, &weights_dir))
                .transpose()?;

            let fp16_attn = layer
                .fp16_attn
                .as_ref()
                .map(|sp| save_sub_program(sp, &programs_dir, &weights_dir))
                .transpose()?;

            layer_manifests.push(LayerManifest {
                index: layer.index,
                pre_attn,
                post_attn,
                fp16_attn,
                cache_write_fused: layer.cache_write_fused,
                donor_compatible: layer.donor_compatible,
            });
        }

        // --- Manifest ---
        let manifest = BundleManifest {
            format_version: 1,
            model_type: BundleModelType::Decode,
            sub_programs: Vec::new(),
            decode: Some(DecodeManifest {
                architecture: self.architecture.clone(),
                lm_head: lm_head_manifest,
                layers: layer_manifests,
            }),
        };

        let json = serde_json::to_string_pretty(&manifest)?;
        fs::write(path.join("manifest.json"), json)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SubProgramBundle helpers
// ---------------------------------------------------------------------------

impl SubProgramBundle {
    /// Create a copy of this sub-program with a different name.
    /// Used when saving ANE lm_head chunks with positional names.
    fn clone_with_name(&self, name: String) -> SubProgramBundle {
        SubProgramBundle {
            name,
            mil_text: self.mil_text.clone(),
            weight_blob: self.weight_blob.clone(),
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
            input_packing: self.input_packing.clone(),
        }
    }
}
