//! `.ironml` bundle format — compiled ANE model artifacts.
//!
//! This module defines the on-disk artifact types and manifest format for
//! compiled ANE models. Bundles are directory trees containing MIL programs,
//! weight blobs, and a JSON manifest that captures all metadata needed for
//! runtime loading.
//!
//! The [`compile_decode_bundle`] function runs the full decode-path compilation
//! pipeline, producing an [`AneDecodeBundle`] ready for serialization.

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::Path;

use anyhow::Result;
use half::f16;

use mil_rs::ir::passes::{
    AutoregressiveShapeMaterializePass, DeadCodeEliminationPass, TypeRepropagationPass,
};
use mil_rs::ir::{Pass, Program};

use crate::ane::blobfile::BlobFileWriter;
use crate::ane::decode_compile::{
    CacheWriteConfig, CpuWeight, apply_min_seq_padding, convert_f32_consts_to_f16,
    extract_1d_weight, extract_cpu_weight, extract_qk_norm_weights, extract_rope_caches,
    inject_cache_write_ops, inject_ffn_residual, precompute_rope_cache, prune_unreferenced_inputs,
    replace_gather_with_inputs,
};
use crate::ane::passes::{
    AneArgPromotionPass, AneConcatEliminationPass, AneLayoutPass, AneMatmulToConvPass,
    AneVariableNamingPass, AttentionDecomposePass, OpSubstitutionPass,
};
use crate::ane::split::{ModelSplit, SplitConfig, SubProgram, split_for_ane};
use ironmill_core::ane::mil_text::{MilTextConfig, program_to_mil_text};

// ---------------------------------------------------------------------------
// Bundle types
// ---------------------------------------------------------------------------

/// Compiled ANE model — ready for runtime loading.
/// Used by the "simple" compilation path (AneModel).
#[derive(Debug)]
pub struct AneModelBundle {
    /// Ordered list of compiled sub-programs that compose the model.
    pub sub_programs: Vec<SubProgramBundle>,
}

/// Single ANE sub-program artifact.
#[derive(Debug)]
pub struct SubProgramBundle {
    /// Sub-program name (e.g., `"layer_0"`, `"embedding"`).
    pub name: String,
    /// Serialized MIL text representation of this sub-program.
    pub mil_text: String,
    /// BLOBFILE weight data baked into this sub-program.
    pub weight_blob: Vec<u8>,
    /// Input tensor descriptors consumed by this sub-program.
    pub inputs: Vec<BundleTensorDescriptor>,
    /// Output tensor descriptors produced by this sub-program.
    pub outputs: Vec<BundleTensorDescriptor>,
    /// Optional packing scheme applied to input tensors.
    pub input_packing: Option<BundleInputPacking>,
}

use ironmill_core::ane::bundle::{
    BundleArchitecture, BundleInputPacking, BundleManifest, BundleModelType,
    BundleTensorDescriptor, DecodeManifest, LayerManifest, LmHeadKind, LmHeadManifest,
    SubProgramManifest,
};

/// Compiled autoregressive decode model.
#[derive(Debug)]
pub struct AneDecodeBundle {
    /// Model architecture metadata (family, dimensions, etc.).
    pub architecture: BundleArchitecture,
    /// Pre-computed RoPE cosine embeddings (fp16 bytes).
    pub rope_cos: Vec<u8>,
    /// Pre-computed RoPE sine embeddings (fp16 bytes).
    pub rope_sin: Vec<u8>,
    /// Token embedding weight table (fp16 bytes).
    pub embedding_weights: Vec<u8>,
    /// Language model head (ANE-chunked or CPU fallback).
    pub lm_head: LmHeadBundle,
    /// Optional final layer-norm weight (e.g., for RMSNorm before lm_head).
    pub final_norm_weight: Option<Vec<u8>>,
    /// Per-layer compiled transformer bundles, in order.
    pub layers: Vec<LayerBundle>,
}

/// Single transformer layer artifact.
#[derive(Debug)]
pub struct LayerBundle {
    /// Zero-based layer index within the model.
    pub index: usize,
    /// Pre-attention sub-program (QKV projections + norms).
    pub pre_attn: SubProgramBundle,
    /// Optional post-attention sub-program (output projection + FFN).
    pub post_attn: Option<SubProgramBundle>,
    /// Optional FP16 attention sub-program for higher-precision layers.
    pub fp16_attn: Option<SubProgramBundle>,
    /// Whether cache-write ops are fused into the pre-attention sub-program.
    pub cache_write_fused: bool,
    /// Whether this layer is compatible with donor weight injection.
    pub donor_compatible: bool,
}

/// LM head bundle — can be ANE-chunked or CPU fallback.
#[non_exhaustive]
#[derive(Debug)]
pub enum LmHeadBundle {
    /// ANE-accelerated chunked lm_head.
    Ane {
        /// Compiled sub-program chunks that partition the vocabulary.
        chunks: Vec<SubProgramBundle>,
        /// Total vocabulary size across all chunks.
        vocab_size: usize,
        /// Hidden dimension of the model.
        hidden_size: usize,
    },
    /// CPU fallback: raw fp16 weight bytes `[vocab_size, hidden_size]`.
    Cpu {
        /// Raw fp16 weight data for CPU matrix multiplication.
        weight_data: Vec<u8>,
        /// Total vocabulary size.
        vocab_size: usize,
        /// Hidden dimension of the model.
        hidden_size: usize,
    },
}

// ---------------------------------------------------------------------------
// Decode compilation config
// ---------------------------------------------------------------------------

/// Configuration for autoregressive decode model compilation.
pub struct AneDecodeConfig {
    /// Maximum sequence length for autoregressive shape materialization.
    pub max_seq_len: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of KV heads (for GQA).
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// RoPE theta parameter.
    pub rope_theta: f64,
    /// EOS token IDs.
    pub eos_tokens: Vec<u32>,
    /// Enable cache-write fusion into pre_attn sub-programs.
    pub fuse_cache_write: bool,
    /// Enable QJL correction in cache-write fusion.
    pub enable_qjl: bool,
}

// ---------------------------------------------------------------------------
// Compile decode bundle
// ---------------------------------------------------------------------------

/// Compile a MIL program into an [`AneDecodeBundle`].
///
/// This runs the full decode-path compilation pipeline (ANE passes,
/// splitting, weight extraction, MIL text emission) but does NOT call
/// `device.compile()`. The resulting bundle contains MIL text + weight
/// blobs ready for ANE device compilation at load time.
///
/// # Errors
///
/// Returns an error if external tensors cannot be materialized (e.g. the
/// weight provider is missing or a key lookup fails).
pub fn compile_decode_bundle(
    program: &mil_rs::ir::Program,
    config: &AneDecodeConfig,
) -> Result<AneDecodeBundle> {
    // 0. Clone and materialize all external tensors up front so that
    //    every downstream `as_bytes()` call is guaranteed to succeed.
    let mut program = program.clone();
    program
        .materialize_all()
        .map_err(|e| anyhow::anyhow!("tensor materialization failed: {e}"))?;

    // 1. Extract pre-pass data from the (now-materialized) program.
    let original_arch = mil_rs::analysis::arch::detect_model_arch(&program);
    let (rope_cos_cache, rope_sin_cache, _rope_cache_dim) = extract_rope_caches(&program)
        .unwrap_or_else(|| {
            eprintln!(
                "Warning: RoPE caches not found in program; using precomputed fallback \
                 (head_dim={}, max_seq_len={}, theta={})",
                config.head_dim, config.max_seq_len, config.rope_theta
            );
            precompute_rope_cache(
                config.head_dim,
                config.max_seq_len,
                config.rope_theta as f32,
            )
        });
    let has_qk_norm = extract_qk_norm_weights(&program).is_some();

    // 2. Prepare program for single-token decode.
    if !program.is_autoregressive() {
        program.set_attribute("autoregressive", "true");
    }
    materialize_single_token_shapes(&mut program);

    let ar_shape_pass = AutoregressiveShapeMaterializePass::new(config.max_seq_len);
    let passes: &[(&str, &dyn Pass)] = &[
        ("ArShapeMaterialize", &ar_shape_pass),
        ("OpSubstitution", &OpSubstitutionPass),
        ("AneLayout", &AneLayoutPass),
        ("AneArgPromotion", &AneArgPromotionPass),
        ("TypeRepropagate", &TypeRepropagationPass),
    ];
    for (name, pass) in passes {
        pass.run(&mut program)
            .map_err(|e| anyhow::anyhow!("{name} pass failed: {e}"))?;
    }
    replace_gather_with_inputs(&mut program);

    // 2. Split into sub-programs and run post-split transforms.
    let split_config = SplitConfig {
        split_attention: true,
        emit_attention: false,
        ..Default::default()
    };
    let mut model_split = split_for_ane(&program, &split_config)?;
    prune_subprogram_inputs(&mut model_split.programs);
    pad_ane_seq_dims(&mut model_split.programs);
    run_post_split_passes(&mut model_split.programs)?;
    let cache_write_fused = try_fuse_cache_writes(&mut model_split, config)?;

    // 3. Classify sub-programs and extract CPU weights.
    let classified = classify_subprograms(&model_split.programs)?;
    let weights = extract_bundle_weights(&classified)?;

    // 4. Resolve architecture parameters.
    let (num_heads, num_kv_heads, head_dim, vocab_size, hidden_size) =
        if let Some(ref arch) = original_arch {
            (
                arch.num_heads,
                arch.num_kv_heads,
                arch.head_dim,
                arch.vocab_size,
                arch.hidden_size,
            )
        } else {
            let vocab = weights.lm_head_weight.shape[0];
            let hidden = weights.lm_head_weight.shape[1];
            (
                config.num_heads,
                config.num_kv_heads,
                config.head_dim,
                vocab,
                hidden,
            )
        };

    // 5. Emit MIL text + weight blobs for each layer.
    let mil_config = MilTextConfig::default();
    let layer_bundles = build_layer_bundles(&classified, &mil_config, cache_write_fused)?;

    // 6. Assemble the bundle.
    let rope_cos_bytes: Vec<u8> = rope_cos_cache
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let rope_sin_bytes: Vec<u8> = rope_sin_cache
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let final_norm_bytes = weights
        .final_norm_weight
        .map(|w| w.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());

    Ok(AneDecodeBundle {
        architecture: BundleArchitecture {
            vocab_size,
            eos_tokens: config.eos_tokens.clone(),
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_size,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_seq_len,
            qk_norm: has_qk_norm,
        },
        rope_cos: rope_cos_bytes,
        rope_sin: rope_sin_bytes,
        embedding_weights: weights.embed_weight.data,
        lm_head: LmHeadBundle::Cpu {
            weight_data: weights.lm_head_weight.data,
            vocab_size: weights.lm_head_weight.shape[0],
            hidden_size: weights.lm_head_weight.shape[1],
        },
        final_norm_weight: final_norm_bytes,
        layers: layer_bundles,
    })
}

// ---------------------------------------------------------------------------
// Phase helpers for compile_decode_bundle
// ---------------------------------------------------------------------------

/// Materialize all dynamic dimensions to `Some(1)` for single-token decode.
fn materialize_single_token_shapes(program: &mut Program) {
    for func in program.functions.values_mut() {
        for (_, ty) in &mut func.inputs {
            for dim in &mut ty.shape {
                if dim.is_none() {
                    *dim = Some(1);
                }
            }
        }
        for op in &mut func.body.operations {
            for t in op.output_types.iter_mut().flatten() {
                for dim in &mut t.shape {
                    if dim.is_none() {
                        *dim = Some(1);
                    }
                }
            }
        }
    }
}

/// Prune unreferenced inputs from each sub-program after splitting.
fn prune_subprogram_inputs(programs: &mut [SubProgram]) {
    for sub in programs.iter_mut() {
        prune_unreferenced_inputs(&mut sub.program);
        if let Some(func) = sub.program.main() {
            let func_input_names: HashSet<String> =
                func.inputs.iter().map(|(n, _)| n.clone()).collect();
            sub.inputs.retain(|td| func_input_names.contains(&td.name));
        }
    }
}

/// Pad all shapes so the sequence (last) dimension is at least 32.
///
/// ANE rejects I/O tensors with C>768 and S<32, so we enforce S≥32 on
/// function inputs, operation output types, reshape constants, and the
/// sub-program tensor descriptors.
fn pad_ane_seq_dims(programs: &mut [SubProgram]) {
    const ANE_MIN_SEQ: usize = 32;
    for sub in programs.iter_mut() {
        if let Some(func) = sub.program.functions.values_mut().next() {
            for (_, ty) in &mut func.inputs {
                if ty.shape.len() == 4 {
                    if let Some(s) = ty.shape[3] {
                        if s < ANE_MIN_SEQ {
                            ty.shape[3] = Some(ANE_MIN_SEQ);
                        }
                    }
                }
            }
            for op in &mut func.body.operations {
                for t in op.output_types.iter_mut().flatten() {
                    if t.shape.len() == 4 {
                        if let Some(s) = t.shape[3] {
                            if s < ANE_MIN_SEQ {
                                t.shape[3] = Some(ANE_MIN_SEQ);
                            }
                        }
                    }
                }
                if op.op_type == "reshape" {
                    pad_reshape_shape_constant(op, ANE_MIN_SEQ);
                }
            }
        }
        for td in &mut sub.inputs {
            if td.shape[3] < ANE_MIN_SEQ {
                td.shape[3] = ANE_MIN_SEQ;
            }
        }
        for td in &mut sub.outputs {
            if td.shape[3] < ANE_MIN_SEQ {
                td.shape[3] = ANE_MIN_SEQ;
            }
        }
    }
}

/// Update a reshape op's shape constant so the last dim is at least `min_seq`.
fn pad_reshape_shape_constant(op: &mut mil_rs::ir::Operation, min_seq: usize) {
    if let Some(mil_rs::ir::Value::Tensor { shape, data, dtype }) = op.inputs.get_mut("shape") {
        if shape.len() == 1 && *dtype == mil_rs::ir::ScalarType::Int32 {
            let ndims = shape[0];
            if let Some(data) = data.as_bytes_mut() {
                if ndims >= 4 && data.len() >= ndims * 4 {
                    let last_offset = (ndims - 1) * 4;
                    let last_val = i32::from_le_bytes([
                        data[last_offset],
                        data[last_offset + 1],
                        data[last_offset + 2],
                        data[last_offset + 3],
                    ]);
                    if last_val > 0 && (last_val as usize) < min_seq {
                        let b = (min_seq as i32).to_le_bytes();
                        data[last_offset] = b[0];
                        data[last_offset + 1] = b[1];
                        data[last_offset + 2] = b[2];
                        data[last_offset + 3] = b[3];
                    }
                }
            }
        }
    }
}

/// Run per-sub-program passes after splitting: matmul→conv, DCE, FFN residual.
fn run_post_split_passes(programs: &mut [SubProgram]) -> Result<()> {
    for sub in programs.iter_mut() {
        AneMatmulToConvPass
            .run(&mut sub.program)
            .map_err(|e| anyhow::anyhow!("MatmulToConv failed for {}: {e}", sub.name))?;
    }
    for sub in programs.iter_mut() {
        DeadCodeEliminationPass
            .run(&mut sub.program)
            .map_err(|e| anyhow::anyhow!("DCE failed for {}: {e}", sub.name))?;
    }
    for sub in programs.iter_mut() {
        if sub.name.ends_with("_post_attn") {
            inject_ffn_residual(&mut sub.program);
        }
    }
    Ok(())
}

/// Attempt cache-write fusion on all pre_attn sub-programs.
///
/// Returns `true` if fusion was applied, `false` otherwise.
fn try_fuse_cache_writes(model_split: &mut ModelSplit, config: &AneDecodeConfig) -> Result<bool> {
    if !config.fuse_cache_write {
        return Ok(false);
    }

    let cw_config = CacheWriteConfig {
        num_heads: config.num_heads,
        num_kv_heads: config.num_kv_heads,
        head_dim: config.head_dim,
    };
    let kv_ch = config.num_kv_heads * config.head_dim;
    let q_ch = config.num_heads * config.head_dim;

    let all_injectable = model_split.programs.iter().all(|sub| {
        if !sub.name.ends_with("_pre_attn") {
            return true;
        }
        let func = match sub.program.main() {
            Some(f) => f,
            None => return false,
        };
        if func.body.outputs.len() < 3 {
            return false;
        }
        let mut found_q = false;
        let mut found_k = false;
        let mut found_v = false;
        for name in &func.body.outputs {
            let lower = name.to_lowercase();
            if lower.contains("k_proj") {
                found_k = true;
            } else if lower.contains("v_proj") {
                found_v = true;
            } else if lower.contains("q_proj") {
                found_q = true;
            }
        }
        if !found_q || !found_k || !found_v {
            found_q = false;
            found_k = false;
            found_v = false;
            for td in &sub.outputs {
                if td.shape[1] == q_ch && !found_q {
                    found_q = true;
                } else if td.shape[1] == kv_ch {
                    if !found_k {
                        found_k = true;
                    } else if !found_v {
                        found_v = true;
                    }
                }
            }
        }
        found_q && found_k && found_v
    });

    let has_pre_attn = model_split
        .programs
        .iter()
        .any(|s| s.name.ends_with("_pre_attn"));

    if all_injectable && has_pre_attn {
        for sub in &mut model_split.programs {
            if sub.name.ends_with("_pre_attn") {
                inject_cache_write_ops(sub, &cw_config)?;
            }
        }
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Classified sub-programs, grouped by their role in the decode pipeline.
struct ClassifiedSubprograms<'a> {
    embedding: &'a SubProgram,
    lm_head: &'a SubProgram,
    pre_attn: BTreeMap<usize, &'a SubProgram>,
    post_attn: BTreeMap<usize, &'a SubProgram>,
    fp16_attn: BTreeMap<usize, &'a SubProgram>,
    layer_numbers: Vec<usize>,
}

/// Classify sub-programs by name into embedding, lm_head, and per-layer maps.
fn classify_subprograms(programs: &[SubProgram]) -> Result<ClassifiedSubprograms<'_>> {
    let mut embedding: Option<&SubProgram> = None;
    let mut lm_head: Option<&SubProgram> = None;
    let mut pre_attn: BTreeMap<usize, &SubProgram> = BTreeMap::new();
    let mut post_attn: BTreeMap<usize, &SubProgram> = BTreeMap::new();
    let mut fp16_attn: BTreeMap<usize, &SubProgram> = BTreeMap::new();

    for sub in programs {
        if sub.name == "embedding" {
            embedding = Some(sub);
        } else if sub.name == "lm_head" {
            lm_head = Some(sub);
        } else if sub.name.ends_with("_pre_attn") {
            if let Some(n) = parse_layer_index(&sub.name, "_pre_attn") {
                pre_attn.insert(n, sub);
            }
        } else if sub.name.ends_with("_fp16_attn") {
            if let Some(n) = parse_layer_index(&sub.name, "_fp16_attn") {
                fp16_attn.insert(n, sub);
            }
        } else if sub.name.ends_with("_post_attn") {
            if let Some(n) = parse_layer_index(&sub.name, "_post_attn") {
                post_attn.insert(n, sub);
            }
        } else if sub.name.starts_with("layer_") {
            if let Some(n) = sub.name.strip_prefix("layer_").and_then(|s| s.parse().ok()) {
                pre_attn.insert(n, sub);
            }
        }
    }

    let embedding = embedding.ok_or_else(|| anyhow::anyhow!("no embedding sub-program found"))?;
    let lm_head = lm_head.ok_or_else(|| anyhow::anyhow!("no lm_head sub-program found"))?;

    let layer_numbers: Vec<usize> = pre_attn.keys().copied().collect();
    if layer_numbers.is_empty() {
        anyhow::bail!("no layer sub-programs found after attention splitting");
    }

    Ok(ClassifiedSubprograms {
        embedding,
        lm_head,
        pre_attn,
        post_attn,
        fp16_attn,
        layer_numbers,
    })
}

/// Parse a layer index from a sub-program name like `"layer_3_pre_attn"`.
fn parse_layer_index(name: &str, suffix: &str) -> Option<usize> {
    name.strip_suffix(suffix)
        .and_then(|s| s.strip_prefix("layer_"))
        .and_then(|s| s.parse().ok())
}

/// Extracted CPU weights from the classified sub-programs.
struct ExtractedWeights {
    embed_weight: CpuWeight,
    lm_head_weight: CpuWeight,
    final_norm_weight: Option<Vec<f16>>,
}

/// Extract embedding, lm_head, and final-norm weights from the classified sub-programs.
fn extract_bundle_weights(classified: &ClassifiedSubprograms<'_>) -> Result<ExtractedWeights> {
    let embed_weight = extract_cpu_weight(classified.embedding, "embed")
        .ok_or_else(|| anyhow::anyhow!("could not extract embedding weight"))?;

    let lm_head_weight = extract_cpu_weight(classified.lm_head, "lm_head").unwrap_or_else(|| {
        // Tied embeddings: lm_head reuses embedding weight.
        eprintln!("Warning: lm_head weight not found; falling back to tied embeddings");
        CpuWeight {
            data: embed_weight.data.clone(),
            shape: embed_weight.shape,
        }
    });

    let final_norm_weight = extract_1d_weight(classified.lm_head, "norm");

    Ok(ExtractedWeights {
        embed_weight,
        lm_head_weight,
        final_norm_weight,
    })
}

/// Build [`LayerBundle`]s by emitting MIL text and weight blobs for each layer.
fn build_layer_bundles(
    classified: &ClassifiedSubprograms<'_>,
    mil_config: &MilTextConfig,
    cache_write_fused: bool,
) -> Result<Vec<LayerBundle>> {
    let mut bundles = Vec::with_capacity(classified.layer_numbers.len());

    for &layer_n in &classified.layer_numbers {
        let pre_sub = classified
            .pre_attn
            .get(&layer_n)
            .ok_or_else(|| anyhow::anyhow!("missing pre_attn for layer {layer_n}"))?;
        let pre_attn_bundle = emit_sub_program_bundle(pre_sub, mil_config)?;

        let post_attn_bundle = classified
            .post_attn
            .get(&layer_n)
            .map(|s| emit_sub_program_bundle(s, mil_config))
            .transpose()?;

        let fp16_attn_bundle = classified
            .fp16_attn
            .get(&layer_n)
            .map(|s| emit_sub_program_bundle(s, mil_config))
            .transpose()?;

        let donor_compatible = layer_n > *classified.layer_numbers.first().unwrap_or(&0);

        bundles.push(LayerBundle {
            index: layer_n,
            pre_attn: pre_attn_bundle,
            post_attn: post_attn_bundle,
            fp16_attn: fp16_attn_bundle,
            cache_write_fused,
            donor_compatible,
        });
    }

    Ok(bundles)
}

/// Emit MIL text + weight blob for a single sub-program, producing a
/// [`SubProgramBundle`].
///
/// Runs per-sub-program passes (f32→f16 conversion, layout fixups,
/// padding, variable naming) before emission.
///
/// # Panics
///
/// Panics if any tensor data is `TensorData::External` (not materialized).
/// The caller ([`compile_decode_bundle`] / [`compile_model_bundle`]) must
/// ensure all tensors are materialized before invoking this function.
fn emit_sub_program_bundle(
    sub: &SubProgram,
    mil_config: &MilTextConfig,
) -> Result<SubProgramBundle> {
    let mut program = sub.program.clone();

    convert_f32_consts_to_f16(&mut program);
    prune_unreferenced_inputs(&mut program);

    AneLayoutPass.run(&mut program)?;
    apply_min_seq_padding(&mut program, 32);
    TypeRepropagationPass.run(&mut program)?;

    // Rename variables to ANE-friendly names.
    AneVariableNamingPass.run(&mut program)?;

    let (mil_text, weight_entries) = program_to_mil_text(&program, mil_config)
        .map_err(|e| anyhow::anyhow!("MIL text emission failed for {}: {e}", sub.name))?;

    // Build the weight blob.
    let mut blob = BlobFileWriter::new();
    for entry in &weight_entries {
        blob.add_weight(&entry.name, &entry.data, entry.dtype);
    }

    // Derive I/O descriptors from the processed program.
    let main_fn = program.main().ok_or_else(|| {
        anyhow::anyhow!(
            "program has no main function for sub-program '{}'",
            sub.name
        )
    })?;

    let inputs: Vec<BundleTensorDescriptor> = main_fn
        .inputs
        .iter()
        .map(|(name, ty)| {
            let mut shape = [1usize; 4];
            for (i, d) in ty.shape.iter().enumerate().take(4) {
                shape[i] = d.unwrap_or(1);
            }
            BundleTensorDescriptor {
                name: name.clone(),
                shape,
                dtype: ty.scalar_type,
            }
        })
        .collect();

    let outputs: Vec<BundleTensorDescriptor> = main_fn
        .body
        .outputs
        .iter()
        .filter_map(|out_name| {
            main_fn
                .body
                .operations
                .iter()
                .find(|op| op.outputs.iter().any(|o| o == out_name))
                .and_then(|op| {
                    let idx = op.outputs.iter().position(|o| o == out_name)?;
                    let ty = op.output_types.get(idx)?.as_ref()?;
                    let mut shape = [1usize; 4];
                    for (i, d) in ty.shape.iter().enumerate().take(4) {
                        shape[i] = d.unwrap_or(1);
                    }
                    Some(BundleTensorDescriptor {
                        name: out_name.clone(),
                        shape,
                        dtype: ty.scalar_type,
                    })
                })
        })
        .collect();

    Ok(SubProgramBundle {
        name: sub.name.clone(),
        mil_text,
        weight_blob: blob.as_bytes(),
        inputs,
        outputs,
        input_packing: None,
    })
}

// ---------------------------------------------------------------------------
// Conversions from compile-crate types
// ---------------------------------------------------------------------------

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
///
/// # Errors
///
/// Returns an error if external tensors cannot be materialized (e.g. the
/// weight provider is missing or a key lookup fails).
pub fn compile_model_bundle(
    program: &Program,
    config: &AneCompileConfig,
) -> Result<AneModelBundle> {
    // 1. Clone, materialize external tensors, and run ANE-specific passes.
    let mut program = program.clone();
    program
        .materialize_all()
        .map_err(|e| anyhow::anyhow!("tensor materialization failed: {e}"))?;
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
                    kind: LmHeadKind::Ane,
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
                    kind: LmHeadKind::Cpu,
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
