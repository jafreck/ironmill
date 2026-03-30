//! Autoregressive inference engine for ANE-direct execution.
//!
//! [`AneInference`] manages stateful autoregressive inference with
//! per-layer sub-programs split at the attention boundary. It supports
//! two modes:
//!
//! - **Baseline (FP16):** Runs the model's FP16 attention sub-programs
//!   with a CPU-managed FP16 KV cache.
//! - **TurboQuant (INT8):** Replaces FP16 attention with TurboQuant's
//!   INT8 cache-write + attention pipeline, halving KV cache memory.
//!
//! Both paths share the same `decode()` control flow — only the
//! attention + cache management diverges (~20 lines in the per-layer loop).

use half::f16;
use ironmill_ane_sys::AneCompiler;
use ironmill_compile::ane::mil_text::{MilTextConfig, program_to_mil_text};
use ironmill_compile::ane::passes::{
    AneArgPromotionPass, AneLayoutPass, AneMatmulToConvPass, AneVariableNamingPass,
    OpSubstitutionPass,
};
use ironmill_compile::ane::split::{SplitConfig, split_for_ane};
use mil_rs::ir::Pass;
use mil_rs::ir::ScalarType;
use mil_rs::ir::passes::{
    AutoregressiveShapeMaterializePass, DeadCodeEliminationPass, TypeRepropagationPass,
};

use super::runtime::AneRuntime;
use super::turboquant::mil_emitter;
use super::turboquant::mil_emitter::MIN_IO_SEQ;
use super::turboquant::{TurboQuantConfig, TurboQuantModel};
use crate::ane::{AneError, Result};
use ironmill_ane_sys::CompiledProgram;
use ironmill_iosurface::{AneTensor, uniform_alloc_size};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Autoregressive inference engine for ANE-direct execution.
///
/// Manages per-layer sub-programs (split at attention boundary),
/// FP16 or INT8 KV caches, and the decode/generate control flow.
///
/// Embedding and lm_head run on **CPU** (embedding is a gather, lm_head
/// is a single-vector matmul — both too small to benefit from ANE and
/// contain ops ANE doesn't support). Per-layer pre_attn/post_attn
/// sub-programs run on ANE.
pub struct AneInference {
    /// Embedding weight table: [vocab_size, hidden_size] as fp16 bytes.
    embed_weight: CpuWeight,
    /// Per-layer: (pre_attn, post_attn) sub-programs.
    /// The fp16_attn sub-program is only present in baseline mode.
    layers: Vec<LayerPrograms>,
    /// LM head projection. ANE-accelerated when possible, CPU fallback otherwise.
    lm_head: LmHead,
    /// Optional TurboQuant model (replaces layer attention when enabled).
    turboquant: Option<TurboQuantModel>,
    /// FP16 KV caches (used when TurboQuant is disabled).
    /// Per-layer (K, V) tensors. `None` when TurboQuant manages the cache.
    fp16_kv_caches: Option<Vec<(AneTensor, AneTensor)>>,
    /// Compiled FP16 attention program (shared across all layers).
    /// Only present when TurboQuant is NOT configured.
    fp16_attn_compiled: Option<CompiledProgram>,
    /// Staging tensor for Q input to FP16 attention.
    fp16_attn_q_staging: Option<AneTensor>,
    /// Staging tensor for FP16 attention output.
    fp16_attn_out_staging: Option<AneTensor>,
    /// Runtime handle.
    runtime: AneRuntime,
    /// Current sequence position.
    seq_pos: usize,
    /// Number of KV heads (for FP16 cache management).
    #[allow(dead_code)]
    num_kv_heads: usize,
    /// Head dimension.
    #[allow(dead_code)]
    head_dim: usize,
    /// Maximum sequence length for FP16 caches.
    #[allow(dead_code)]
    max_seq_len: usize,
    /// Pre-extracted RoPE cos/sin cache tables (from model consts).
    /// Flat f16 arrays: `[num_positions * values_per_position]`.
    /// Indexed as `cache[pos * rope_cache_dim .. (pos+1) * rope_cache_dim]`.
    #[allow(dead_code)]
    rope_cos_cache: Vec<f16>,
    #[allow(dead_code)]
    rope_sin_cache: Vec<f16>,
    /// Number of f16 values per position in the RoPE cache.
    #[allow(dead_code)]
    rope_cache_dim: usize,
}

/// CPU-side weight tensor for embedding/lm_head.
struct CpuWeight {
    /// Raw fp16 bytes, row-major.
    data: Vec<u8>,
    /// [rows, cols].
    shape: [usize; 2],
}

/// Per-layer compiled sub-programs.
struct LayerPrograms {
    pre_attn: LoadedSubProgram,
    /// FP16 attention sub-program. Only compiled in baseline mode;
    /// in TurboQuant mode this is `None`. Retained for future use
    /// (model-extracted attention path).
    #[allow(dead_code)]
    fp16_attn: Option<LoadedSubProgram>,
    /// Post-attention sub-program. `None` for layers where all ops
    /// fall within the attention cluster (e.g., position-only layers).
    post_attn: Option<LoadedSubProgram>,
    /// Input mapping for fp16_attn (which tensor indices are Q/K/V/cos/sin).
    /// Retained for future use (model-extracted attention path).
    #[allow(dead_code)]
    attn_input_map: Option<AttnInputMap>,
}

/// Mapping from logical Q/K/V/cos/sin roles to fp16_attn input tensor indices.
///
/// After stripping gather ops from the attention cluster, the fp16_attn
/// sub-program's inputs are determined by `build_sub_program` in alphabetical
/// order. This map records which index is which so `decode()` writes
/// the correct data to the correct input tensor.
///
/// Retained for future use (model-extracted attention path).
#[allow(dead_code)]
struct AttnInputMap {
    q_idx: usize,
    k_idx: usize,
    v_idx: usize,
    /// Indices of cos cache inputs (RoPE gathered values).
    rope_cos_indices: Vec<usize>,
    /// Indices of sin cache inputs (RoPE gathered values).
    rope_sin_indices: Vec<usize>,
}

/// A compiled sub-program with pre-allocated I/O tensors.
///
/// Programs are compiled upfront but kept unloaded. During decode,
/// each program is loaded → evaluated → unloaded on demand via
/// `runtime.eval_compiled()`, keeping at most ~3 programs loaded
/// simultaneously.
struct LoadedSubProgram {
    compiled: CompiledProgram,
    input_tensors: Vec<AneTensor>,
    output_tensors: Vec<AneTensor>,
    /// If inputs were spatially packed, stores the packing metadata.
    input_packing: Option<ironmill_compile::ane::packing::InputPacking>,
}

/// LM head projection — ANE-accelerated or CPU fallback.
enum LmHead {
    /// ANE-accelerated: chunked conv1×1 across multiple ANE programs.
    Ane(AneLmHead),
    /// CPU fallback: scalar matmul (used when ANE compilation fails).
    Cpu(CpuWeight),
}

// ---------------------------------------------------------------------------
// ANE-accelerated LM head
// ---------------------------------------------------------------------------

/// Maximum output channels per ANE lm_head chunk.
/// ANE rejects tensor dimensions > 16384.
const LM_HEAD_MAX_CHUNK_CH: usize = 16384;

/// Minimum spatial dimension for ANE I/O tensors (same as TurboQuant).
const LM_HEAD_MIN_SEQ: usize = 32;

/// ANE-accelerated lm_head projection via chunked conv1×1.
///
/// Splits the `[vocab_size, hidden_size]` weight matrix into chunks of
/// ≤16384 output channels. Each chunk is compiled as a separate conv1×1
/// ANE program. At inference time, all chunks are evaluated and their
/// outputs concatenated to produce the full logits vector.
#[allow(dead_code)]
struct AneLmHead {
    chunks: Vec<LmHeadChunk>,
    vocab_size: usize,
    hidden_size: usize,
    runtime: AneRuntime,
}

/// One chunk of the ANE lm_head — a conv1×1 with ≤16384 output channels.
struct LmHeadChunk {
    compiled: CompiledProgram,
    input_tensor: AneTensor,
    output_tensor: AneTensor,
    out_channels: usize,
}

impl AneLmHead {
    /// Compile the chunked ANE lm_head from a CPU weight tensor.
    ///
    /// Splits `[vocab_size, hidden_size]` into chunks of ≤16384 output
    /// channels, compiles each as a conv1×1 ANE program with BLOBFILE
    /// weights, and pre-allocates I/O tensors.
    ///
    /// Uses donor/patch optimization: all full-size chunks share the same
    /// MIL text (same `hidden_size` × `LM_HEAD_MAX_CHUNK_CH` conv1×1).
    /// Only the first full-size chunk is compiled; the rest reuse its
    /// compiled `net.plist` via `AneCompiler::patch_weights`.
    fn compile(weight: &CpuWeight) -> Result<Self> {
        let [vocab_size, hidden_size] = weight.shape;
        let runtime = AneRuntime::new()?;

        let num_chunks = vocab_size.div_ceil(LM_HEAD_MAX_CHUNK_CH);
        let mut chunks: Vec<LmHeadChunk> = Vec::with_capacity(num_chunks);

        let bytes_per_elem = 2; // fp16
        let row_bytes = hidden_size * bytes_per_elem;

        // Track the donor for full-size chunks (all have out_ch == LM_HEAD_MAX_CHUNK_CH).
        // The last chunk may be smaller — it gets its own compilation.
        let mut full_chunk_donor_idx: Option<usize> = None;

        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * LM_HEAD_MAX_CHUNK_CH;
            let end = (start + LM_HEAD_MAX_CHUNK_CH).min(vocab_size);
            let out_ch = end - start;

            // Extract weight chunk: rows [start..end], each row = hidden_size fp16 values.
            let chunk_data = weight.data[start * row_bytes..end * row_bytes].to_vec();

            // Generate MIL text for conv1×1 with BLOBFILE weight.
            let mil_text = emit_lm_head_chunk_mil(hidden_size, out_ch);
            let weight_path = "@model_path/weights/weight.bin";

            let is_full_size = out_ch == LM_HEAD_MAX_CHUNK_CH;

            let compiled = if let Some(donor_idx) = full_chunk_donor_idx.filter(|_| is_full_size) {
                // Patch from the donor — skip compilation.
                let donor = &chunks[donor_idx].compiled;
                AneCompiler::patch_weights(donor, &mil_text, &[(weight_path, &chunk_data)])
                    .map_err(|e| AneError::CompileFailed {
                        status: 0,
                        context: format!(
                            "lm_head chunk {chunk_idx} ({out_ch} ch) donor patch failed: {e}"
                        ),
                    })?
            } else {
                // Full compile (first full-size chunk, or the smaller tail chunk).
                AneCompiler::compile_mil_text(&mil_text, &[(weight_path, &chunk_data)]).map_err(
                    |e| AneError::CompileFailed {
                        status: 0,
                        context: format!(
                            "lm_head chunk {chunk_idx} ({out_ch} ch) compilation failed: {e}"
                        ),
                    },
                )?
            };
            // Unload to free ANE slots — lm_head uses eval_compiled (called
            // 10 times per token, load/unload overhead is ~1ms total).
            runtime.unload_compiled(&compiled);

            // Set the donor index once the first full-size chunk is compiled.
            if is_full_size && full_chunk_donor_idx.is_none() {
                full_chunk_donor_idx = Some(chunks.len());
            }

            // ANE requires all I/O tensors in one eval to have the same alloc size.
            let alloc = uniform_alloc_size(&[
                ([1, hidden_size, 1, LM_HEAD_MIN_SEQ], ScalarType::Float16),
                ([1, out_ch, 1, LM_HEAD_MIN_SEQ], ScalarType::Float16),
            ]);

            let input_tensor = AneTensor::new_with_min_alloc(
                hidden_size,
                LM_HEAD_MIN_SEQ,
                ScalarType::Float16,
                alloc,
            )?;
            let output_tensor =
                AneTensor::new_with_min_alloc(out_ch, LM_HEAD_MIN_SEQ, ScalarType::Float16, alloc)?;

            chunks.push(LmHeadChunk {
                compiled,
                input_tensor,
                output_tensor,
                out_channels: out_ch,
            });
        }

        let patched_chunks = if num_chunks > 1 {
            num_chunks
                - 1
                - if vocab_size % LM_HEAD_MAX_CHUNK_CH != 0 {
                    1
                } else {
                    0
                }
        } else {
            0
        };
        eprintln!(
            "ANE lm_head: {vocab_size} vocab × {hidden_size} hidden → {num_chunks} chunks \
             (max {LM_HEAD_MAX_CHUNK_CH} channels each, {patched_chunks} patched)"
        );

        Ok(Self {
            chunks,
            vocab_size,
            hidden_size,
            runtime,
        })
    }

    /// Run the lm_head projection on ANE, returning logits as f32.
    fn forward(&mut self, hidden: &[f16]) -> Result<Vec<f32>> {
        let mut logits = Vec::with_capacity(self.vocab_size);

        for chunk in &mut self.chunks {
            // Write hidden state to input tensor (column 0, zero-padded).
            write_f16_padded(&mut chunk.input_tensor, hidden)?;

            // Eval conv1×1 on ANE (load → eval → unload to stay within budget).
            self.runtime.eval_compiled(
                &chunk.compiled,
                &[&chunk.input_tensor],
                &mut [&mut chunk.output_tensor],
            )?;

            // Read output logits from column 0 of each channel.
            let output = read_f16_channels(&chunk.output_tensor)?;
            logits.extend(output.iter().take(chunk.out_channels).map(|v| v.to_f32()));
        }

        Ok(logits)
    }
}

/// Generate MIL text for a single lm_head chunk (conv1×1).
///
/// The weight is delivered via BLOBFILE at `@model_path/weights/weight.bin`.
/// Input: `[1, hidden_size, 1, 32]` fp16.
/// Output: `[1, out_channels, 1, 32]` fp16.
fn emit_lm_head_chunk_mil(hidden_size: usize, out_channels: usize) -> String {
    let s = LM_HEAD_MIN_SEQ;
    // Reuse the same program wrapper format as turboquant_mil.
    format!(
        "program(1.3)\n\
         [buildInfo = dict<string, string>(\
         {{{{\"coremlc-component-MIL\", \"3510.2.1\"}}, \
         {{\"coremlc-version\", \"3505.4.1\"}}, \
         {{\"coremltools-component-milinternal\", \"\"}}, \
         {{\"coremltools-version\", \"9.0\"}}}})]\n\
         {{\n\
             func main<ios18>(tensor<fp16, [1,{hidden_size},1,{s}]> a_input0) {{\n\
                 string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n\
                 tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n\
                 tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n\
                 tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n\
                 int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n\
                 tensor<fp16, [{out_channels},{hidden_size},1,1]> weight = const()\
                     [name=string(\"weight\"), val=tensor<fp16, [{out_channels},{hidden_size},1,1]>\
                     (BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n\
                 tensor<fp16, [1,{out_channels},1,{s}]> z_output0 = conv(\
                     x=a_input0, weight=weight, pad_type=pt, strides=st, pad=pd, \
                     dilations=dl, groups=gr)[name=string(\"z_output0\")];\n\
             }} -> (z_output0);\n\
         }}"
    )
}
// Construction
// ---------------------------------------------------------------------------

impl AneInference {
    /// Build from a Program. When `turbo_config` is Some, TurboQuant
    /// replaces the FP16 attention sub-programs (they are not compiled,
    /// saving ANE compile budget).
    ///
    /// The program must be a transformer with attention-splittable layers.
    pub fn compile(
        program: &mil_rs::ir::Program,
        turbo_config: Option<TurboQuantConfig>,
    ) -> Result<Self> {
        // 0. Extract RoPE cos/sin cache from the original program before
        //    passes modify shapes. These tables are used by CPU-side RoPE
        //    when gather ops are stripped from fp16_attn sub-programs.
        let (rope_cos_cache, rope_sin_cache, rope_cache_dim) = extract_rope_caches(program)
            .unwrap_or_else(|| {
                // No extractable RoPE cache (model may not use RoPE, or
                // caches are runtime inputs). Precompute a default.
                let head_dim = 64; // will be refined later
                let max_pos = 2048;
                let theta = 500_000.0f32;
                precompute_rope_cache(head_dim, max_pos, theta)
            });

        // 1. Run ANE-specific passes.
        // Note: AneConcatEliminationPass is NOT run here on the full program.
        // RoPE (RotaryEmbedding) ops contain concat and are stripped by the
        // attention-boundary split below. Concat validation runs per
        // sub-program after splitting.
        let mut program = program.clone();
        // Ensure autoregressive tagging (required for ArShapeMaterialize).
        if !program.is_autoregressive() {
            program.set_attribute("autoregressive", "true");
        }

        // Materialize all dynamic dims to 1 for single-token decode
        // BEFORE AneLayoutPass so the layout gets static shapes.
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

        let ar_shape_pass = AutoregressiveShapeMaterializePass::new(2048);
        let passes: &[(&str, &dyn Pass)] = &[
            ("ArShapeMaterialize", &ar_shape_pass),
            ("OpSubstitution", &OpSubstitutionPass),
            ("AneLayout", &AneLayoutPass),
            ("AneArgPromotion", &AneArgPromotionPass),
            ("TypeRepropagate", &TypeRepropagationPass),
            // AttentionDecompose skipped: decomposing grouped_query_attention
            // expands layers from ~12 to ~106 ops, exceeding the ANE program
            // budget. The structural split handles fused GQA ops directly.
            ("AneVariableNaming", &AneVariableNamingPass),
        ];
        for (name, pass) in passes {
            pass.run(&mut program)
                .map_err(|e| AneError::Other(anyhow::anyhow!("{name} pass failed: {e}")))?;
        }

        // ANE rejects IOSurface-backed tensors when C > ~768 and S < 32.
        // After AneLayoutPass, tensors are in [1, C, 1, S] format.
        // Padding is applied PER SUB-PROGRAM after splitting (below),
        // NOT on the full program — fp16_attn sub-programs need original
        // shapes for correct attention dimension semantics.

        // 2. Split with attention boundary.
        // This strips attention + RoPE ops (which contain concat) from
        // the sub-programs that will be compiled for ANE.
        let split_config = SplitConfig {
            split_attention: true,
            // Emit fp16_attn sub-programs in FP16 baseline mode.
            // Gather ops (RoPE cos/sin lookup) are stripped from the
            // attention cluster by strip_gather_ops() in the splitter;
            // the gathered values are filled by the CPU at decode time.
            //
            // With lazy load/unload (loadWithQoS/unloadWithQoS), we can
            // emit attention programs without exceeding the ANE slot limit:
            // at most 3 programs are loaded simultaneously during decode.
            emit_attention: false, // FP16 attention needs KV cache inputs (full seq_len)
            // not single-token K/V (S=1 with C>768 violates ANE constraint)
            ..Default::default()
        };
        let mut model_split = split_for_ane(&program, &split_config)?;

        // 2a. Pad S≥32 on non-attention sub-programs only.
        // fp16_attn sub-programs need original shapes for correct
        // per-head attention dimensions (reshape [1,C,1,1] → [1,H,D,1]).
        const ANE_MIN_SEQ: usize = 32;
        for sub in &mut model_split.programs {
            if sub.name.ends_with("_fp16_attn") {
                continue;
            }
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
                }
            }
            // Also update the sub-program's TensorDescriptors.
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

        // 2b. Pack inputs spatially where possible.
        // Must run BEFORE matmul→conv since packing adds slice_by_size ops,
        // not matmuls. Stores packing metadata per sub-program name.
        let mut packing_map: std::collections::HashMap<
            String,
            ironmill_compile::ane::packing::InputPacking,
        > = std::collections::HashMap::new();
        for sub in &mut model_split.programs {
            if let Some(packing) = ironmill_compile::ane::packing::pack_inputs(sub) {
                packing_map.insert(sub.name.clone(), packing);
            }
        }

        // 2b. Convert matmul → 1×1 conv on each sub-program.
        // This must run AFTER splitting so the structural attention splitter
        // can still find matmul ops for Q/K/V projection identification.
        for sub in &mut model_split.programs {
            AneMatmulToConvPass.run(&mut sub.program).map_err(|e| {
                AneError::Other(anyhow::anyhow!("MatmulToConv failed for {}: {e}", sub.name))
            })?;
        }

        // 2c. Run dead code elimination on each sub-program to remove
        // orphaned ops (e.g., position ID reformatting that only fed RoPE).
        for sub in &mut model_split.programs {
            DeadCodeEliminationPass.run(&mut sub.program).map_err(|e| {
                AneError::Other(anyhow::anyhow!("DCE failed for {}: {e}", sub.name))
            })?;
        }

        // Note: concat IS supported by ANE (eval-verified, see ane-op-support-matrix.md).
        // The previous concat validation was based on a false assumption.

        // 3. Initialize runtime.
        let runtime = AneRuntime::new()?;
        let mil_config = MilTextConfig::default();

        // 4. Classify sub-programs by name.
        let mut embedding_sp = None;
        let mut lm_head_sp = None;
        // Collect layer sub-programs indexed by layer number.
        let mut pre_attn_map: std::collections::BTreeMap<
            usize,
            &ironmill_compile::ane::split::SubProgram,
        > = std::collections::BTreeMap::new();
        let mut post_attn_map: std::collections::BTreeMap<
            usize,
            &ironmill_compile::ane::split::SubProgram,
        > = std::collections::BTreeMap::new();
        let mut fp16_attn_map: std::collections::BTreeMap<
            usize,
            &ironmill_compile::ane::split::SubProgram,
        > = std::collections::BTreeMap::new();

        for sub in &model_split.programs {
            if sub.name == "embedding" {
                embedding_sp = Some(sub);
            } else if sub.name == "lm_head" {
                lm_head_sp = Some(sub);
            } else if sub.name.ends_with("_pre_attn") {
                // Extract layer number from "layer_N_pre_attn"
                if let Some(n) = sub
                    .name
                    .strip_suffix("_pre_attn")
                    .and_then(|s| s.strip_prefix("layer_"))
                    .and_then(|s| s.parse::<usize>().ok())
                {
                    pre_attn_map.insert(n, sub);
                }
            } else if sub.name.ends_with("_fp16_attn") {
                if let Some(n) = sub
                    .name
                    .strip_suffix("_fp16_attn")
                    .and_then(|s| s.strip_prefix("layer_"))
                    .and_then(|s| s.parse::<usize>().ok())
                {
                    fp16_attn_map.insert(n, sub);
                }
            } else if sub.name.ends_with("_post_attn") {
                if let Some(n) = sub
                    .name
                    .strip_suffix("_post_attn")
                    .and_then(|s| s.strip_prefix("layer_"))
                    .and_then(|s| s.parse::<usize>().ok())
                {
                    post_attn_map.insert(n, sub);
                }
            }
        }

        let embedding_sub = embedding_sp
            .ok_or_else(|| AneError::Other(anyhow::anyhow!("no embedding sub-program found")))?;
        let lm_head_sub = lm_head_sp
            .ok_or_else(|| AneError::Other(anyhow::anyhow!("no lm_head sub-program found")))?;

        // Determine layers: a layer exists if it has at least a pre_attn.
        let layer_numbers: Vec<usize> = pre_attn_map.keys().copied().collect();
        let num_layers = layer_numbers.len();
        if num_layers == 0 {
            return Err(AneError::Other(anyhow::anyhow!(
                "no layer sub-programs found after attention splitting"
            )));
        }

        // 5. Extract CPU weights for embedding; build ANE or CPU lm_head.
        let embed_weight = extract_cpu_weight(embedding_sub, "embed").ok_or_else(|| {
            AneError::Other(anyhow::anyhow!(
                "could not extract embedding weight from sub-program"
            ))
        })?;
        let lm_head_cpu_weight = extract_cpu_weight(lm_head_sub, "lm_head").unwrap_or_else(|| {
            // Tied embeddings: lm_head reuses embedding weight.
            CpuWeight {
                data: embed_weight.data.clone(),
                shape: embed_weight.shape,
            }
        });

        // Try to compile lm_head as chunked ANE conv1×1. Falls back to CPU
        // if ANE compilation fails (e.g., unsupported shape or ANE unavailable).
        let lm_head = match AneLmHead::compile(&lm_head_cpu_weight) {
            Ok(ane_lm_head) => LmHead::Ane(ane_lm_head),
            Err(e) => {
                eprintln!("warning: ANE lm_head compilation failed, falling back to CPU: {e}");
                LmHead::Cpu(lm_head_cpu_weight)
            }
        };

        // 6. Compile and load per-layer sub-programs for ANE.
        //
        // Donor/patch optimization: all layers share the same MIL text
        // structure (same ops, shapes, BLOBFILE paths) — only weight data
        // differs. Compile the first layer's sub-programs normally (the
        // "donor"), then use AneCompiler::patch_weights for layers 1+.
        // This copies the donor's compiled net.plist and loads with new
        // weights, bypassing compileWithQoS: entirely. Saves ~56
        // compilations for a 29-layer model (2 instead of 58).
        let mut layers: Vec<LayerPrograms> = Vec::with_capacity(num_layers);
        for (i, &layer_n) in layer_numbers.iter().enumerate() {
            let pre_sub = pre_attn_map.get(&layer_n).ok_or_else(|| {
                AneError::Other(anyhow::anyhow!("missing pre_attn for layer {layer_n}"))
            })?;
            let pre_packing = packing_map.remove(&pre_sub.name);
            let pre = if i == 0 {
                // First layer: compile normally — this becomes the donor.
                compile_and_load_sub(pre_sub, &runtime, &mil_config, pre_packing)?
            } else {
                // Layers 1+: patch weights from the layer-0 donor.
                match compile_and_load_sub_with_donor(
                    pre_sub,
                    &layers[0].pre_attn.compiled,
                    &runtime,
                    &mil_config,
                    pre_packing,
                ) {
                    Ok(loaded) => loaded,
                    Err(e) => {
                        eprintln!(
                            "warning: layer {layer_n} pre_attn donor patch failed, \
                             falling back to full compile: {e}"
                        );
                        let pre_packing_retry = packing_map.remove(&pre_sub.name);
                        compile_and_load_sub(pre_sub, &runtime, &mil_config, pre_packing_retry)?
                    }
                }
            };
            let post = if let Some(post_sub) = post_attn_map.get(&layer_n) {
                let post_packing = packing_map.remove(&post_sub.name);
                if i == 0 {
                    Some(compile_and_load_sub(
                        post_sub,
                        &runtime,
                        &mil_config,
                        post_packing,
                    )?)
                } else if let Some(ref donor_post) = layers[0].post_attn {
                    match compile_and_load_sub_with_donor(
                        post_sub,
                        &donor_post.compiled,
                        &runtime,
                        &mil_config,
                        post_packing,
                    ) {
                        Ok(loaded) => Some(loaded),
                        Err(e) => {
                            eprintln!(
                                "warning: layer {layer_n} post_attn donor patch failed, \
                                 falling back to full compile: {e}"
                            );
                            let post_packing_retry = packing_map.remove(&post_sub.name);
                            Some(compile_and_load_sub(
                                post_sub,
                                &runtime,
                                &mil_config,
                                post_packing_retry,
                            )?)
                        }
                    }
                } else {
                    // Layer 0 had no post_attn — no donor available.
                    Some(compile_and_load_sub(
                        post_sub,
                        &runtime,
                        &mil_config,
                        post_packing,
                    )?)
                }
            } else {
                None
            };
            let fp16_attn = if let Some(attn_sub) = fp16_attn_map.get(&layer_n) {
                let attn_packing = packing_map.remove(&attn_sub.name);
                match compile_and_load_sub(attn_sub, &runtime, &mil_config, attn_packing) {
                    Ok(loaded) => Some(loaded),
                    Err(e) => {
                        eprintln!(
                            "warning: layer {layer_n} fp16_attn compilation failed, \
                             falling back to Q pass-through: {e}"
                        );
                        None
                    }
                }
            } else {
                None
            };
            // Compute input mapping for fp16_attn (which tensor indices
            // correspond to Q/K/V vs RoPE cos/sin gathered values).
            let attn_input_map = if let Some(attn_sub) = fp16_attn_map.get(&layer_n) {
                if fp16_attn.is_some() {
                    Some(compute_attn_input_map(&attn_sub.inputs))
                } else {
                    None
                }
            } else {
                None
            };
            layers.push(LayerPrograms {
                pre_attn: pre,
                fp16_attn,
                post_attn: post,
                attn_input_map,
            });
        }

        // Report donor/patch savings.
        if num_layers > 1 {
            let patched = (num_layers - 1) * if layers[0].post_attn.is_some() { 2 } else { 1 };
            eprintln!(
                "donor/patch: compiled 1 donor layer, patched {patched} sub-programs \
                 ({} compilations saved)",
                patched
            );
        }

        // 6b. Unload fp16_attn programs to free ANE slots.
        // pre_attn and post_attn stay loaded (they fit within the ~55 budget).
        // fp16_attn programs are loaded on demand via eval_compiled().
        for layer in &layers {
            if let Some(ref attn) = layer.fp16_attn {
                runtime.unload_compiled(&attn.compiled);
            }
        }

        // 7. Set up TurboQuant or FP16 caches.
        let (
            turboquant,
            fp16_kv_caches,
            fp16_attn_compiled,
            fp16_attn_q_staging,
            fp16_attn_out_staging,
            num_kv_heads,
            head_dim,
            max_seq_len,
        ) = if let Some(tq_config) = turbo_config {
            let nkv = tq_config.num_kv_heads;
            let hd = tq_config.head_dim;
            let msl = tq_config.max_seq_len;
            let tq = TurboQuantModel::compile(tq_config)?;
            (Some(tq), None, None, None, None, nkv, hd, msl)
        } else {
            // FP16 baseline: detect architecture from program.
            let arch = mil_rs::analysis::arch::detect_model_arch(&program);
            let (nh, nkv, hd, msl) = if let Some(ref a) = arch {
                (a.num_heads, a.num_kv_heads, a.head_dim, 2048)
            } else {
                // Fallback: infer from pre_attn output shapes.
                let pre_sps: Vec<&ironmill_compile::ane::split::SubProgram> =
                    pre_attn_map.values().copied().collect();
                let kv_channels = infer_kv_heads_from_sub(&pre_sps);
                let hd = infer_head_dim_from_sub(&pre_sps);
                let nkv = kv_channels / hd.max(1);
                let q_channels = pre_sps
                    .first()
                    .map(|sp| sp.outputs[0].shape[1])
                    .unwrap_or(kv_channels);
                (q_channels / hd.max(1), nkv, hd, 2048)
            };
            let kv_channels = nkv * hd;
            let q_channels = nh * hd;

            // Compile hand-written FP16 attention MIL (shared across layers).
            let mil = mil_emitter::emit_fp16_attention_mil(nh, nkv, hd, msl, msl);
            let dump_path = "/tmp/ironmill_debug_fp16_attn_handwritten.mil";
            let _ = std::fs::write(dump_path, &mil);
            let attn_compiled =
                AneCompiler::compile_mil_text(&mil, &[]).map_err(|e| AneError::CompileFailed {
                    status: 0,
                    context: format!(
                        "FP16 attention compilation failed: {e} (MIL dumped to {dump_path})"
                    ),
                })?;
            // Keep loaded — it's only 1 program, well within budget.
            // Using eval_raw avoids the ~5ms loadWithQoS cost per layer.

            // Uniform alloc: all tensors in one eval must share the same alloc.
            let attn_alloc = uniform_alloc_size(&[
                ([1, q_channels, 1, MIN_IO_SEQ], ScalarType::Float16),
                ([1, kv_channels, 1, msl], ScalarType::Float16),
                ([1, kv_channels, 1, msl], ScalarType::Float16),
            ]);

            let mut caches = Vec::with_capacity(num_layers);
            for _ in 0..num_layers {
                let k = AneTensor::new_with_min_alloc(
                    kv_channels,
                    msl,
                    ScalarType::Float16,
                    attn_alloc,
                )?;
                let v = AneTensor::new_with_min_alloc(
                    kv_channels,
                    msl,
                    ScalarType::Float16,
                    attn_alloc,
                )?;
                caches.push((k, v));
            }

            let q_staging = AneTensor::new_with_min_alloc(
                q_channels,
                MIN_IO_SEQ,
                ScalarType::Float16,
                attn_alloc,
            )?;
            let out_staging = AneTensor::new_with_min_alloc(
                q_channels,
                MIN_IO_SEQ,
                ScalarType::Float16,
                attn_alloc,
            )?;

            (
                None,
                Some(caches),
                Some(attn_compiled),
                Some(q_staging),
                Some(out_staging),
                nkv,
                hd,
                msl,
            )
        };

        Ok(Self {
            embed_weight,
            layers,
            lm_head,
            turboquant,
            fp16_kv_caches,
            fp16_attn_compiled,
            fp16_attn_q_staging,
            fp16_attn_out_staging,
            runtime,
            seq_pos: 0,
            num_kv_heads,
            head_dim,
            max_seq_len,
            rope_cos_cache,
            rope_sin_cache,
            rope_cache_dim,
        })
    }

    // -----------------------------------------------------------------------
    // Decode
    // -----------------------------------------------------------------------

    /// Process one token, return logits as f32.
    pub fn decode(&mut self, token_id: u32) -> Result<Vec<f32>> {
        let profiling = self.seq_pos == 0 && std::env::var("IRONMILL_PROFILE").is_ok();
        let t_total = if profiling {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // 1. Embedding: CPU gather from weight table.
        let t0 = if profiling {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let embed_out = cpu_embedding_lookup(&self.embed_weight, token_id)?;
        let d_embed = t0.map(|t| t.elapsed());

        // 2. Per-layer: pre_attn → attention → post_attn
        let num_layers = self.layers.len();
        let mut hidden = embed_out;
        let mut d_pre_attn = std::time::Duration::ZERO;
        let mut d_read_qkv = std::time::Duration::ZERO;
        let mut d_attn = std::time::Duration::ZERO;
        let mut d_post_attn = std::time::Duration::ZERO;

        for layer_idx in 0..num_layers {
            // Pre-attention: norm → Q/K/V projection
            let t0 = if profiling {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let layer = &mut self.layers[layer_idx];
            write_f16_padded(&mut layer.pre_attn.input_tensors[0], &hidden)?;
            {
                let in_refs: Vec<&AneTensor> = layer.pre_attn.input_tensors.iter().collect();
                let mut out_refs: Vec<&mut AneTensor> =
                    layer.pre_attn.output_tensors.iter_mut().collect();
                self.runtime
                    .eval_raw(
                        layer.pre_attn.compiled.as_raw_ptr(),
                        &in_refs,
                        &mut out_refs,
                    )
                    .map_err(|e| {
                        AneError::Other(anyhow::anyhow!(
                            "layer {layer_idx} pre_attn eval failed: {e}"
                        ))
                    })?;
            }
            if let Some(t) = t0 {
                d_pre_attn += t.elapsed();
            }

            // Read Q, K_proj, V_proj from pre_attn outputs.
            // Convention: outputs are [Q, K_proj, V_proj, residual_hidden]
            // or fewer if the model merges them.
            let num_pre_outputs = layer.pre_attn.output_tensors.len();

            // Attention (divergent path)
            let attn_out_data = if let Some(tq) = &mut self.turboquant {
                let t0 = if profiling {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let q = &layer.pre_attn.output_tensors[0];
                let k_proj = if num_pre_outputs > 1 {
                    &layer.pre_attn.output_tensors[1]
                } else {
                    &layer.pre_attn.output_tensors[0]
                };
                let v_proj = if num_pre_outputs > 2 {
                    &layer.pre_attn.output_tensors[2]
                } else {
                    &layer.pre_attn.output_tensors[0]
                };
                let attn_tensor = tq
                    .step_attention(layer_idx, q, k_proj, v_proj)
                    .map_err(|e| {
                        AneError::Other(anyhow::anyhow!(
                            "layer {layer_idx} turboquant step_attention failed: {e}"
                        ))
                    })?;
                let result = read_f16_channels(&attn_tensor)?;
                if let Some(t) = t0 {
                    d_attn += t.elapsed();
                }
                result
            } else if let Some(ref mut caches) = self.fp16_kv_caches {
                let t0 = if profiling {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let q_data = read_f16_channels(&layer.pre_attn.output_tensors[0])?;
                let k_data = if num_pre_outputs > 1 {
                    read_f16_channels(&layer.pre_attn.output_tensors[1])?
                } else {
                    q_data.clone()
                };
                let v_data = if num_pre_outputs > 2 {
                    read_f16_channels(&layer.pre_attn.output_tensors[2])?
                } else {
                    q_data.clone()
                };
                if let Some(t) = t0 {
                    d_read_qkv += t.elapsed();
                }

                let t0 = if profiling {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                // Write K/V to persistent FP16 cache at current position.
                let k_elements = k_data.len();
                let v_elements = v_data.len();
                let elem_offset_k = self.seq_pos * k_elements;
                let elem_offset_v = self.seq_pos * v_elements;
                caches[layer_idx].0.write_f16_at(elem_offset_k, &k_data)?;
                caches[layer_idx].1.write_f16_at(elem_offset_v, &v_data)?;

                // Hand-written FP16 attention: Q + K_cache + V_cache → attn_output
                let result = if let Some(ref compiled) = self.fp16_attn_compiled {
                    let q_staging = self.fp16_attn_q_staging.as_mut().unwrap();
                    let out_staging = self.fp16_attn_out_staging.as_mut().unwrap();
                    write_f16_padded(q_staging, &q_data)?;
                    let (k_cache, v_cache) = (&caches[layer_idx].0, &caches[layer_idx].1);
                    self.runtime
                        .eval_raw(
                            compiled.as_raw_ptr(),
                            &[q_staging, k_cache, v_cache],
                            &mut [out_staging],
                        )
                        .map_err(|e| {
                            AneError::Other(anyhow::anyhow!(
                                "layer {layer_idx} fp16_attn eval failed: {e}"
                            ))
                        })?;
                    read_f16_channels(out_staging)?
                } else {
                    // No compiled attention — return Q as pass-through.
                    q_data
                };
                if let Some(t) = t0 {
                    d_attn += t.elapsed();
                }
                result
            } else {
                return Err(AneError::Other(anyhow::anyhow!(
                    "no attention backend configured"
                )));
            };

            // Post-attention: O proj → residual → FFN → residual
            let t0 = if profiling {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let layer = &mut self.layers[layer_idx];
            if let Some(ref mut post_attn) = layer.post_attn {
                if let Some(ref packing) = post_attn.input_packing {
                    // Packed: write all logical inputs into the single tensor.
                    if packing.offsets.len() > 1 && num_pre_outputs > 3 {
                        let residual = read_f16_channels(&layer.pre_attn.output_tensors[3])?;
                        // Build the packed buffer directly (can't use write_packed_inputs
                        // because residual is a temporary we need to own).
                        let [_, channels, _, total_s] = post_attn.input_tensors[0].shape();
                        let mut packed = vec![f16::ZERO; channels * total_s];
                        let c0 = attn_out_data.len().min(channels);
                        for ch in 0..c0 {
                            packed[ch * total_s + packing.offsets[0]] = attn_out_data[ch];
                        }
                        let c1 = residual.len().min(channels);
                        for ch in 0..c1 {
                            packed[ch * total_s + packing.offsets[1]] = residual[ch];
                        }
                        post_attn.input_tensors[0].write_f16(&packed)?;
                    } else {
                        ironmill_compile::ane::packing::write_packed_inputs(
                            &mut post_attn.input_tensors[0],
                            &[&attn_out_data],
                            packing,
                        )?;
                    }
                } else {
                    write_f16_padded(&mut post_attn.input_tensors[0], &attn_out_data)?;
                    if post_attn.input_tensors.len() > 1 && num_pre_outputs > 3 {
                        let residual = read_f16_channels(&layer.pre_attn.output_tensors[3])?;
                        write_f16_padded(&mut post_attn.input_tensors[1], &residual)?;
                    }
                }
                {
                    let in_refs: Vec<&AneTensor> = post_attn.input_tensors.iter().collect();
                    let mut out_refs: Vec<&mut AneTensor> =
                        post_attn.output_tensors.iter_mut().collect();
                    self.runtime
                        .eval_raw(post_attn.compiled.as_raw_ptr(), &in_refs, &mut out_refs)
                        .map_err(|e| {
                            AneError::Other(anyhow::anyhow!(
                                "layer {layer_idx} post_attn eval failed: {e}"
                            ))
                        })?;
                }
                hidden = read_f16_channels(&post_attn.output_tensors[0])?;
            } else {
                // No post_attn sub-program — use attention output directly.
                hidden = attn_out_data;
            }
            if let Some(t) = t0 {
                d_post_attn += t.elapsed();
            }
        }

        // Advance TurboQuant sequence position after all layers.
        if let Some(tq) = &mut self.turboquant {
            tq.advance_seq_pos();
        }

        self.seq_pos += 1;

        // 3. LM head: ANE conv1×1 (chunked) or CPU fallback.
        let t0 = if profiling {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let logits = match &mut self.lm_head {
            LmHead::Ane(ane_lm_head) => ane_lm_head.forward(&hidden),
            LmHead::Cpu(weight) => cpu_lm_head_matmul(weight, &hidden),
        };
        let d_lm_head = t0.map(|t| t.elapsed());

        if profiling {
            let total = t_total.unwrap().elapsed();
            eprintln!(
                "[profile] decode token 0: embed={:.2}ms pre_attn={:.1}ms read_qkv={:.1}ms attn={:.1}ms post_attn={:.1}ms lm_head={:.1}ms total={:.1}ms",
                d_embed.unwrap().as_secs_f64() * 1000.0,
                d_pre_attn.as_secs_f64() * 1000.0,
                d_read_qkv.as_secs_f64() * 1000.0,
                d_attn.as_secs_f64() * 1000.0,
                d_post_attn.as_secs_f64() * 1000.0,
                d_lm_head.unwrap().as_secs_f64() * 1000.0,
                total.as_secs_f64() * 1000.0,
            );
        }

        logits
    }

    // -----------------------------------------------------------------------
    // Generation
    // -----------------------------------------------------------------------

    /// Prefill: process all prompt tokens sequentially, populating the
    /// KV cache. Returns logits for the last prompt token.
    ///
    /// Batch prefill (processing multiple tokens in one ANE eval) is not
    /// supported initially — the attention sub-programs are compiled for
    /// single-token decode shapes.
    pub fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<Vec<f32>> {
        if prompt_tokens.is_empty() {
            return Err(AneError::Other(anyhow::anyhow!(
                "prompt_tokens must not be empty"
            )));
        }

        let mut logits = Vec::new();
        for &token_id in prompt_tokens {
            logits = self.decode(token_id)?;
        }
        Ok(logits)
    }

    /// Generate tokens autoregressively.
    ///
    /// Calls `prefill()` then loops `decode()` with sampling.
    /// Sampling: greedy (temperature=0), temperature scaling (temperature>0).
    /// EOS token detection stops generation.
    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        temperature: f32,
    ) -> Result<Vec<u32>> {
        let mut logits = self.prefill(prompt_tokens)?;
        let mut generated = Vec::with_capacity(max_tokens);

        for _ in 0..max_tokens {
            let token_id = sample_token(&logits, temperature);

            // EOS detection (common EOS token IDs).
            if is_eos_token(token_id) {
                break;
            }

            generated.push(token_id);
            logits = self.decode(token_id)?;
        }

        Ok(generated)
    }

    /// Reset all state for a new conversation.
    pub fn reset(&mut self) {
        self.seq_pos = 0;
        if let Some(tq) = &mut self.turboquant {
            tq.reset();
        }
        // FP16 cache data will be overwritten on subsequent writes;
        // no need to zero it.
    }

    /// Current sequence position.
    pub fn seq_pos(&self) -> usize {
        self.seq_pos
    }

    /// Whether TurboQuant mode is active.
    pub fn is_turboquant(&self) -> bool {
        self.turboquant.is_some()
    }

    /// Whether the lm_head runs on ANE (true) or CPU (false).
    pub fn is_ane_lm_head(&self) -> bool {
        matches!(self.lm_head, LmHead::Ane(_))
    }
}

impl Drop for AneInference {
    fn drop(&mut self) {
        // Unload all compiled programs from the ANE to free execution slots
        // and compile budget for subsequent model compilations in the same process.
        for layer in &self.layers {
            self.runtime.unload_compiled(&layer.pre_attn.compiled);
            if let Some(ref attn) = layer.fp16_attn {
                self.runtime.unload_compiled(&attn.compiled);
            }
            if let Some(ref post) = layer.post_attn {
                self.runtime.unload_compiled(&post.compiled);
            }
        }
        if let Some(ref compiled) = self.fp16_attn_compiled {
            self.runtime.unload_compiled(compiled);
        }
        if let LmHead::Ane(ref ane_lm) = self.lm_head {
            for chunk in &ane_lm.chunks {
                self.runtime.unload_compiled(&chunk.compiled);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Padded I/O helpers for ANE minimum-S constraint
// ---------------------------------------------------------------------------

/// Write `data` (C elements for one token) into an ANE tensor with shape
/// `[1, C, 1, S]` at sequence position 0. ANE uses NCHW layout, so element
/// `(c, s)` is at flat index `c * S + s`. The remaining S-1 columns are
/// zero-padded.
fn write_f16_padded(tensor: &mut AneTensor, data: &[f16]) -> Result<()> {
    let [_, channels, _, seq_len] = tensor.shape();
    if seq_len == 1 {
        // No padding needed — direct write.
        return Ok(tensor.write_f16(data)?);
    }
    let total = channels * seq_len;
    let mut padded = vec![f16::ZERO; total];
    let c = data.len().min(channels);
    for i in 0..c {
        padded[i * seq_len] = data[i]; // column 0 of each channel row
    }
    Ok(tensor.write_f16(&padded)?)
}

/// Read C elements (one token at column 0) from an ANE tensor with shape
/// `[1, C, 1, S]`. Inverse of `write_f16_padded`.
fn read_f16_channels(tensor: &AneTensor) -> Result<Vec<f16>> {
    let [_, channels, _, seq_len] = tensor.shape();
    if seq_len == 1 {
        return Ok(tensor.read_f16()?);
    }
    let full = tensor.read_f16()?;
    let mut out = Vec::with_capacity(channels);
    for c in 0..channels {
        out.push(full[c * seq_len]); // column 0 of each channel row
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Compilation helpers
// ---------------------------------------------------------------------------

/// Compile a single sub-program, pre-allocating I/O tensors.
///
/// The program is auto-loaded by `compile_mil_text`; the caller is
/// responsible for calling `runtime.unload_compiled()` afterwards.
fn compile_and_load_sub(
    sub: &ironmill_compile::ane::split::SubProgram,
    _runtime: &AneRuntime,
    mil_config: &MilTextConfig,
    input_packing: Option<ironmill_compile::ane::packing::InputPacking>,
) -> Result<LoadedSubProgram> {
    // ANE requires Float16. Convert any Float32 const ops to Float16.
    let mut program = sub.program.clone();
    convert_f32_consts_to_f16(&mut program);

    let (mil_text, weight_entries) = program_to_mil_text(&program, mil_config).map_err(|e| {
        AneError::Other(anyhow::anyhow!(
            "MIL text emission failed for {}: {e}",
            sub.name
        ))
    })?;

    // Collect weight data for the compiler's weight dict.
    let weight_refs: Vec<(String, Vec<u8>)> = weight_entries
        .iter()
        .map(|e| (e.path.clone(), e.data.clone()))
        .collect();
    let weight_slices: Vec<(&str, &[u8])> = weight_refs
        .iter()
        .map(|(path, data)| (path.as_str(), data.as_slice()))
        .collect();

    let compiled = AneCompiler::compile_mil_text(&mil_text, &weight_slices).map_err(|e| {
        // Collect op types for diagnosis.
        let op_types: Vec<&str> = program
            .main()
            .map(|f| f.body.operations.iter().map(|op| op.op_type.as_str()).collect())
            .unwrap_or_default();
        // Dump MIL text for debugging.
        let dump_path = format!("/tmp/ironmill_debug_{}.mil", sub.name);
        let _ = std::fs::write(&dump_path, &mil_text);
        AneError::CompileFailed {
            status: 0,
            context: format!(
                "{} compilation failed: {e} (ops: {:?}, mil_text: {} bytes → {dump_path}, weights: {} entries)",
                sub.name,
                op_types,
                mil_text.len(),
                weight_slices.len(),
            ),
        }
    })?;

    // Pre-allocate I/O tensors with uniform sizing.
    // Note: we do NOT call runtime.load_program() — the program is
    // auto-loaded by compile_mil_text and will be unloaded in bulk
    // after all layers are compiled.
    let input_shapes: Vec<_> = sub.inputs.iter().map(|td| (td.shape, td.dtype)).collect();
    let output_shapes: Vec<_> = sub.outputs.iter().map(|td| (td.shape, td.dtype)).collect();
    let input_alloc = uniform_alloc_size(&input_shapes);
    let output_alloc = uniform_alloc_size(&output_shapes);

    let input_tensors: Vec<AneTensor> = sub
        .inputs
        .iter()
        .map(|td| {
            AneTensor::new_with_min_alloc(td.shape[1], td.shape[3], td.dtype, input_alloc)
                .map_err(Into::into)
        })
        .collect::<Result<Vec<_>>>()?;
    let output_tensors: Vec<AneTensor> = sub
        .outputs
        .iter()
        .map(|td| {
            AneTensor::new_with_min_alloc(td.shape[1], td.shape[3], td.dtype, output_alloc)
                .map_err(Into::into)
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(LoadedSubProgram {
        compiled,
        input_tensors,
        output_tensors,
        input_packing,
    })
}

/// Compile a sub-program by patching weights from a donor program.
///
/// Uses `AneCompiler::patch_weights` to copy the donor's compiled
/// `net.plist` and load with new weights — **no compilation**. This does
/// not consume a compile budget slot.
///
/// The donor must have been compiled from the same MIL text structure
/// (same ops and shapes, different weight data).
fn compile_and_load_sub_with_donor(
    sub: &ironmill_compile::ane::split::SubProgram,
    donor: &CompiledProgram,
    _runtime: &AneRuntime,
    mil_config: &MilTextConfig,
    input_packing: Option<ironmill_compile::ane::packing::InputPacking>,
) -> Result<LoadedSubProgram> {
    let mut program = sub.program.clone();
    convert_f32_consts_to_f16(&mut program);

    let (mil_text, weight_entries) = program_to_mil_text(&program, mil_config).map_err(|e| {
        AneError::Other(anyhow::anyhow!(
            "MIL text emission failed for {}: {e}",
            sub.name
        ))
    })?;

    let weight_refs: Vec<(String, Vec<u8>)> = weight_entries
        .iter()
        .map(|e| (e.path.clone(), e.data.clone()))
        .collect();
    let weight_slices: Vec<(&str, &[u8])> = weight_refs
        .iter()
        .map(|(path, data)| (path.as_str(), data.as_slice()))
        .collect();

    let compiled = AneCompiler::patch_weights(donor, &mil_text, &weight_slices).map_err(|e| {
        AneError::CompileFailed {
            status: 0,
            context: format!(
                "{} donor patch failed: {e} (weights: {} entries)",
                sub.name,
                weight_slices.len(),
            ),
        }
    })?;

    let input_shapes: Vec<_> = sub.inputs.iter().map(|td| (td.shape, td.dtype)).collect();
    let output_shapes: Vec<_> = sub.outputs.iter().map(|td| (td.shape, td.dtype)).collect();
    let input_alloc = uniform_alloc_size(&input_shapes);
    let output_alloc = uniform_alloc_size(&output_shapes);

    let input_tensors: Vec<AneTensor> = sub
        .inputs
        .iter()
        .map(|td| {
            AneTensor::new_with_min_alloc(td.shape[1], td.shape[3], td.dtype, input_alloc)
                .map_err(Into::into)
        })
        .collect::<Result<Vec<_>>>()?;
    let output_tensors: Vec<AneTensor> = sub
        .outputs
        .iter()
        .map(|td| {
            AneTensor::new_with_min_alloc(td.shape[1], td.shape[3], td.dtype, output_alloc)
                .map_err(Into::into)
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(LoadedSubProgram {
        compiled,
        input_tensors,
        output_tensors,
        input_packing,
    })
}

/// Infer the number of KV heads from pre_attn sub-program output shapes.
fn infer_kv_heads_from_sub(pre_attn_sps: &[&ironmill_compile::ane::split::SubProgram]) -> usize {
    // The KV projection outputs have fewer channels than Q in GQA models.
    // Find the minimum channel count among outputs (excluding very small ones
    // like scalars/position IDs). This gives kv_channels = num_kv_heads * head_dim.
    if let Some(sp) = pre_attn_sps.first() {
        let min_channels = sp
            .outputs
            .iter()
            .map(|td| td.shape[1])
            .filter(|&c| c > 1) // skip scalars
            .min();
        if let Some(c) = min_channels {
            return c;
        }
    }
    1
}

/// Infer head_dim from pre_attn sub-program output shapes.
fn infer_head_dim_from_sub(_pre_attn_sps: &[&ironmill_compile::ane::split::SubProgram]) -> usize {
    // Without explicit arch info, assume head_dim = 64 (common default).
    64
}

// ---------------------------------------------------------------------------
// CPU weight extraction and computation
// ---------------------------------------------------------------------------

/// Extract the largest weight tensor from a sub-program for CPU execution.
///
/// Walks the sub-program's ops looking for `const` ops with tensor values.
/// Returns the largest one (embedding table or lm_head weight), converted
/// to fp16 if needed.
fn extract_cpu_weight(
    sub: &ironmill_compile::ane::split::SubProgram,
    _label: &str,
) -> Option<CpuWeight> {
    use mil_rs::ir::Value;

    let func = sub.program.main()?;
    let mut best: Option<(usize, Vec<u8>, [usize; 2], ScalarType)> = None;

    for op in &func.body.operations {
        if op.op_type == "const" {
            let tensor = op.inputs.get("val").or_else(|| op.attributes.get("val"));
            if let Some(Value::Tensor { data, shape, dtype }) = tensor {
                if shape.len() >= 2 && data.len() > best.as_ref().map_or(0, |b| b.0) {
                    best = Some((data.len(), data.clone(), [shape[0], shape[1]], *dtype));
                }
            }
        }
    }

    best.map(|(_, data, shape, dtype)| {
        // Convert to fp16 if needed (CPU embedding/lm_head work in fp16).
        let fp16_data = if dtype == ScalarType::Float32 {
            data.chunks_exact(4)
                .flat_map(|b| {
                    let v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    f16::from_f32(v).to_le_bytes()
                })
                .collect()
        } else {
            data
        };
        CpuWeight {
            data: fp16_data,
            shape,
        }
    })
}

/// CPU embedding lookup: gather row `token_id` from the weight table.
/// Returns the row as fp16 values.
fn cpu_embedding_lookup(weight: &CpuWeight, token_id: u32) -> Result<Vec<f16>> {
    let [vocab_size, hidden_size] = weight.shape;
    let idx = token_id as usize;
    if idx >= vocab_size {
        return Err(AneError::Other(anyhow::anyhow!(
            "token_id {token_id} >= vocab_size {vocab_size}"
        )));
    }

    let bytes_per_elem = 2; // fp16
    let row_start = idx * hidden_size * bytes_per_elem;
    let row_end = row_start + hidden_size * bytes_per_elem;

    if row_end > weight.data.len() {
        return Err(AneError::Other(anyhow::anyhow!(
            "embedding weight data too short: need {} bytes, have {}",
            row_end,
            weight.data.len()
        )));
    }

    let row_bytes = &weight.data[row_start..row_end];
    let row: Vec<f16> = row_bytes
        .chunks_exact(2)
        .map(|b| f16::from_le_bytes([b[0], b[1]]))
        .collect();

    Ok(row)
}

/// CPU lm_head: matmul hidden × weight^T → logits as f32.
/// Weight shape: [vocab_size, hidden_size], hidden shape: [hidden_size].
fn cpu_lm_head_matmul(weight: &CpuWeight, hidden: &[f16]) -> Result<Vec<f32>> {
    let [vocab_size, hidden_size] = weight.shape;
    if hidden.len() < hidden_size {
        return Err(AneError::Other(anyhow::anyhow!(
            "hidden size mismatch: expected {hidden_size}, got {}",
            hidden.len()
        )));
    }

    let bytes_per_elem = 2; // fp16
    let mut logits = vec![0.0f32; vocab_size];

    for (v, logit) in logits.iter_mut().enumerate().take(vocab_size) {
        let row_start = v * hidden_size * bytes_per_elem;
        let mut dot = 0.0f32;
        for (h, &hval) in hidden.iter().enumerate().take(hidden_size) {
            let w_offset = row_start + h * bytes_per_elem;
            let w = f16::from_le_bytes([weight.data[w_offset], weight.data[w_offset + 1]]);
            dot += w.to_f32() * hval.to_f32();
        }
        *logit = dot;
    }

    Ok(logits)
}

/// Convert Float32 const ops to Float16, materialize dynamic shapes,
/// and decompose unsupported ops for ANE compatibility.
fn convert_f32_consts_to_f16(program: &mut mil_rs::ir::Program) {
    use mil_rs::ir::{Operation, Value};

    for func in program.functions.values_mut() {
        // First pass: decompose unsupported ops.
        let mut new_ops = Vec::with_capacity(func.body.operations.len());
        for op in &func.body.operations {
            if op.op_type == "reciprocal" {
                // reciprocal(x) → real_div(1, x)
                // ANE doesn't support reciprocal but supports real_div.
                let x_input = op
                    .inputs
                    .get("x")
                    .cloned()
                    .unwrap_or(Value::Reference("unknown".into()));
                let mut div_op = Operation::new("real_div", &op.name)
                    .with_input("x", Value::Float(1.0))
                    .with_input("y", x_input);
                for out_name in &op.outputs {
                    div_op = div_op.with_output(out_name);
                }
                div_op.output_types = op.output_types.clone();
                new_ops.push(div_op);
            } else {
                new_ops.push(op.clone());
            }
        }
        func.body.operations = new_ops;

        // Second pass: convert dtypes and materialize shapes.
        for op in &mut func.body.operations {
            for val in op.inputs.values_mut().chain(op.attributes.values_mut()) {
                if let Value::Tensor {
                    data,
                    shape: _,
                    dtype,
                } = val
                {
                    if *dtype == ScalarType::Float32 {
                        let f32_values: Vec<f32> = data
                            .chunks_exact(4)
                            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                            .collect();
                        let f16_bytes: Vec<u8> = f32_values
                            .iter()
                            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
                            .collect();
                        *data = f16_bytes;
                        *dtype = ScalarType::Float16;
                    }
                }
            }
            for t in op.output_types.iter_mut().flatten() {
                if t.scalar_type == ScalarType::Float32 {
                    t.scalar_type = ScalarType::Float16;
                }
                for dim in &mut t.shape {
                    if dim.is_none() {
                        *dim = Some(1);
                    }
                }
            }
        }
        for (_, ty) in &mut func.inputs {
            if ty.scalar_type == ScalarType::Float32 {
                ty.scalar_type = ScalarType::Float16;
            }
            for dim in &mut ty.shape {
                if dim.is_none() {
                    *dim = Some(1);
                }
            }
        }
    }
}

// Sampling functions are in crate::sampling
use crate::sampling::{is_eos_token, sample_token};

// ---------------------------------------------------------------------------
// RoPE cache extraction and precomputation
// ---------------------------------------------------------------------------

/// Extracted RoPE cos/sin cache data.
type RopeCacheData = (Vec<f16>, Vec<f16>, usize);

/// Extract RoPE cos/sin cache data from model const ops.
///
/// Walks the program looking for `gather` ops whose names contain "cos"
/// or "sin" (produced by RotaryEmbedding decomposition). Traces back to
/// the const table input and extracts it.
///
/// Returns `(cos_values, sin_values, values_per_position)` where each
/// value array is flat `[num_positions * values_per_position]` in fp16.
fn extract_rope_caches(program: &mil_rs::ir::Program) -> Option<RopeCacheData> {
    use mil_rs::ir::Value;
    let func = program.main()?;

    // Map output_name → (data, shape, dtype) for const ops.
    let mut const_map: std::collections::HashMap<&str, (&[u8], &[usize], ScalarType)> =
        std::collections::HashMap::new();
    for op in &func.body.operations {
        if op.op_type == "const" {
            let tensor = op.inputs.get("val").or_else(|| op.attributes.get("val"));
            if let Some(Value::Tensor { data, shape, dtype }) = tensor {
                for out in &op.outputs {
                    const_map.insert(out.as_str(), (data.as_slice(), shape.as_slice(), *dtype));
                }
            }
        }
    }

    let mut cos_result: Option<(Vec<f16>, usize, usize)> = None;
    let mut sin_result: Option<(Vec<f16>, usize, usize)> = None;

    for op in &func.body.operations {
        if op.op_type != "gather" {
            continue;
        }
        let name = op.name.to_ascii_lowercase();
        let is_cos = name.contains("cos");
        let is_sin = name.contains("sin");
        if !is_cos && !is_sin {
            continue;
        }

        // Trace the gather's "x" input back to a const table.
        if let Some(Value::Reference(cache_ref)) = op.inputs.get("x") {
            if let Some(&(data, shape, dtype)) = const_map.get(cache_ref.as_str()) {
                if shape.len() < 2 {
                    continue;
                }
                let f16_values: Vec<f16> = match dtype {
                    ScalarType::Float32 => data
                        .chunks_exact(4)
                        .map(|b| f16::from_f32(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
                        .collect(),
                    ScalarType::Float16 => data
                        .chunks_exact(2)
                        .map(|b| f16::from_le_bytes([b[0], b[1]]))
                        .collect(),
                    _ => continue,
                };
                let num_pos = shape[0];
                let dim = shape[1];

                if is_cos && cos_result.is_none() {
                    cos_result = Some((f16_values.clone(), num_pos, dim));
                }
                if is_sin && sin_result.is_none() {
                    sin_result = Some((f16_values, num_pos, dim));
                }
            }
        }
    }

    match (cos_result, sin_result) {
        (Some((cos, _num_pos, dim)), Some((sin, _, _))) => Some((cos, sin, dim)),
        _ => None,
    }
}

/// Precompute RoPE cos/sin cache tables from scratch.
///
/// Used as a fallback when the model's const tables can't be extracted.
fn precompute_rope_cache(head_dim: usize, max_pos: usize, theta: f32) -> RopeCacheData {
    let half_dim = head_dim / 2;
    let mut cos_cache = Vec::with_capacity(max_pos * half_dim);
    let mut sin_cache = Vec::with_capacity(max_pos * half_dim);

    for pos in 0..max_pos {
        for i in 0..half_dim {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            cos_cache.push(f16::from_f32(angle.cos()));
            sin_cache.push(f16::from_f32(angle.sin()));
        }
    }
    (cos_cache, sin_cache, half_dim)
}

// ---------------------------------------------------------------------------
// Attention input mapping
// ---------------------------------------------------------------------------

/// Compute the [`AttnInputMap`] for an fp16_attn sub-program.
///
/// After stripping gather ops, the fp16_attn sub-program has inputs for
/// Q, K, V projections AND gathered cos/sin values. The inputs are sorted
/// alphabetically by name. This function identifies which index is which
/// by examining input names and channel counts.
fn compute_attn_input_map(inputs: &[ironmill_compile::ane::TensorDescriptor]) -> AttnInputMap {
    let mut cos_indices = Vec::new();
    let mut sin_indices = Vec::new();
    let mut qkv_candidates: Vec<(usize, &ironmill_compile::ane::TensorDescriptor)> = Vec::new();

    for (i, td) in inputs.iter().enumerate() {
        let name = td.name.to_ascii_lowercase();
        if name.contains("cos") {
            cos_indices.push(i);
        } else if name.contains("sin") {
            sin_indices.push(i);
        } else {
            qkv_candidates.push((i, td));
        }
    }

    // Sort QKV candidates: largest channels first (Q has more channels
    // than K/V in GQA models). For equal channels (MHA), distinguish
    // by name: "q" before "k" before "v".
    qkv_candidates.sort_by(|a, b| {
        let ch_cmp = b.1.shape[1].cmp(&a.1.shape[1]);
        if ch_cmp != std::cmp::Ordering::Equal {
            return ch_cmp;
        }
        let a_name = a.1.name.to_ascii_lowercase();
        let b_name = b.1.name.to_ascii_lowercase();
        let rank = |n: &str| -> u8 {
            if n.contains("q_proj") || n.contains("_q_") {
                0
            } else if n.contains("k_proj") || n.contains("_k_") {
                1
            } else {
                2
            }
        };
        rank(&a_name).cmp(&rank(&b_name))
    });

    AttnInputMap {
        q_idx: qkv_candidates.first().map_or(0, |&(i, _)| i),
        k_idx: qkv_candidates.get(1).map_or(1, |&(i, _)| i),
        v_idx: qkv_candidates.get(2).map_or(2, |&(i, _)| i),
        rope_cos_indices: cos_indices,
        rope_sin_indices: sin_indices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lm_head_mil_is_valid_program() {
        let mil = emit_lm_head_chunk_mil(1024, 8192);
        assert!(mil.starts_with("program(1.3)"));
        assert!(mil.contains("func main<ios18>"));
        assert!(mil.contains("a_input0"));
        assert!(mil.contains("z_output0"));
        assert!(mil.contains("conv("));
        assert!(mil.contains("[8192,1024,1,1]"));
        assert!(
            mil.contains("BLOBFILE"),
            "weight should use BLOBFILE reference"
        );
    }

    #[test]
    fn lm_head_mil_small_chunk() {
        let mil = emit_lm_head_chunk_mil(512, 256);
        assert!(mil.contains("[256,512,1,1]"));
        assert!(mil.contains("conv("));
    }

    #[test]
    fn lm_head_chunk_count() {
        // 151936 vocab / 16384 max = 10 chunks (9×16384 + 1×4480)
        let vocab_size: usize = 151936;
        let num_chunks = vocab_size.div_ceil(LM_HEAD_MAX_CHUNK_CH);
        assert_eq!(num_chunks, 10);

        let last_chunk_size = vocab_size - (num_chunks - 1) * LM_HEAD_MAX_CHUNK_CH;
        assert_eq!(last_chunk_size, 4480);
    }
}
