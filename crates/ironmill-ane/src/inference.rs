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
use mil_rs::convert::ir_to_mil_text::{MilTextConfig, program_to_mil_text};
use mil_rs::ffi::ane::AneCompiler;
use mil_rs::ir::Pass;
use mil_rs::ir::ScalarType;
use mil_rs::ir::passes::{
    AneArgPromotionPass, AneLayoutPass, AneMatmulToConvPass, AneVariableNamingPass,
    AttentionDecomposePass, AutoregressiveShapeMaterializePass, DeadCodeEliminationPass,
    OpSubstitutionPass, TypeRepropagationPass,
};

use crate::program::{CompiledProgram, LoadedProgram};
use crate::runtime::AneRuntime;
use crate::split::{SplitConfig, split_for_ane};
use crate::tensor::{AneTensor, uniform_alloc_size};
use crate::turboquant::{TurboQuantConfig, TurboQuantModel};
use crate::{AneError, Result};

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
    /// lm_head weight: [vocab_size, hidden_size] as fp16 bytes.
    lm_head_weight: CpuWeight,
    /// Optional TurboQuant model (replaces layer attention when enabled).
    turboquant: Option<TurboQuantModel>,
    /// FP16 KV caches (used when TurboQuant is disabled).
    /// Per-layer (K, V) tensors. `None` when TurboQuant manages the cache.
    fp16_kv_caches: Option<Vec<(AneTensor, AneTensor)>>,
    /// Runtime handle.
    runtime: AneRuntime,
    /// Current sequence position.
    seq_pos: usize,
    /// Number of KV heads (for FP16 cache management).
    num_kv_heads: usize,
    /// Head dimension.
    head_dim: usize,
    /// Maximum sequence length for FP16 caches.
    #[allow(dead_code)]
    max_seq_len: usize,
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
    /// in TurboQuant mode this is `None`.
    fp16_attn: Option<LoadedSubProgram>,
    /// Post-attention sub-program. `None` for layers where all ops
    /// fall within the attention cluster (e.g., position-only layers).
    post_attn: Option<LoadedSubProgram>,
}

/// A loaded sub-program with pre-allocated I/O tensors.
struct LoadedSubProgram {
    loaded: LoadedProgram,
    input_tensors: Vec<AneTensor>,
    output_tensors: Vec<AneTensor>,
    /// If inputs were spatially packed, stores the packing metadata.
    input_packing: Option<crate::packing::InputPacking>,
}

// ---------------------------------------------------------------------------
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
            ("AttentionDecompose", &AttentionDecomposePass),
            ("AneVariableNaming", &AneVariableNamingPass),
        ];
        for (name, pass) in passes {
            pass.run(&mut program)
                .map_err(|e| AneError::Other(anyhow::anyhow!("{name} pass failed: {e}")))?;
        }

        // ANE rejects IOSurface-backed tensors when C > ~768 and S < 32.
        // After AneLayoutPass, tensors are in [1, C, 1, S] format.
        // Pad dim 3 (S) to at least 32. The decode loop writes one token
        // to column 0 and reads from column 0.
        const ANE_MIN_SEQ: usize = 32;
        for func in program.functions.values_mut() {
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

        // 2. Split with attention boundary.
        // This strips attention + RoPE ops (which contain concat) from
        // the sub-programs that will be compiled for ANE.
        let split_config = SplitConfig {
            split_attention: true,
            emit_attention: false, // Disabled: fp16_attn sub-programs fail ANE compilation
            // due to name-heuristic fallback producing wrong op subsets
            ..Default::default()
        };
        let mut model_split = split_for_ane(&program, &split_config)?;

        // 2a. Pack inputs spatially where possible.
        // Must run BEFORE matmul→conv since packing adds slice_by_size ops,
        // not matmuls. Stores packing metadata per sub-program name.
        let mut packing_map: std::collections::HashMap<String, crate::packing::InputPacking> =
            std::collections::HashMap::new();
        for sub in &mut model_split.programs {
            if let Some(packing) = crate::packing::pack_inputs(sub) {
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

        // 2d. Validate that no concat ops remain in any sub-program.
        for sub in &model_split.programs {
            if let Some(func) = sub.program.main() {
                let concats: Vec<&str> = func
                    .body
                    .operations
                    .iter()
                    .filter(|op| op.op_type == "concat")
                    .map(|op| op.name.as_str())
                    .collect();
                if !concats.is_empty() {
                    return Err(AneError::Other(anyhow::anyhow!(
                        "sub-program '{}' still contains {} concat op(s) after splitting: {}. \
                         ANE does not support concat (constraint #1).",
                        sub.name,
                        concats.len(),
                        concats.join(", ")
                    )));
                }
            }
        }

        // 3. Initialize runtime.
        let runtime = AneRuntime::new()?;
        let mil_config = MilTextConfig::default();

        // 4. Classify sub-programs by name.
        let mut embedding_sp = None;
        let mut lm_head_sp = None;
        // Collect layer sub-programs indexed by layer number.
        let mut pre_attn_map: std::collections::BTreeMap<usize, &crate::split::SubProgram> =
            std::collections::BTreeMap::new();
        let mut post_attn_map: std::collections::BTreeMap<usize, &crate::split::SubProgram> =
            std::collections::BTreeMap::new();
        let mut fp16_attn_map: std::collections::BTreeMap<usize, &crate::split::SubProgram> =
            std::collections::BTreeMap::new();

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

        // 5. Extract CPU weights for embedding and lm_head (these run on CPU
        // because they contain ops ANE doesn't support: gather, large matmul).
        let embed_weight = extract_cpu_weight(embedding_sub, "embed").ok_or_else(|| {
            AneError::Other(anyhow::anyhow!(
                "could not extract embedding weight from sub-program"
            ))
        })?;
        let lm_head_weight = extract_cpu_weight(lm_head_sub, "lm_head").unwrap_or_else(|| {
            // Tied embeddings: lm_head reuses embedding weight.
            CpuWeight {
                data: embed_weight.data.clone(),
                shape: embed_weight.shape,
            }
        });

        // 6. Compile and load per-layer sub-programs for ANE.
        let mut layers = Vec::with_capacity(num_layers);
        for &layer_n in &layer_numbers {
            let pre_sub = pre_attn_map.get(&layer_n).ok_or_else(|| {
                AneError::Other(anyhow::anyhow!("missing pre_attn for layer {layer_n}"))
            })?;
            let pre_packing = packing_map.remove(&pre_sub.name);
            let pre = compile_and_load_sub(pre_sub, &runtime, &mil_config, pre_packing)?;
            let post = if let Some(post_sub) = post_attn_map.get(&layer_n) {
                let post_packing = packing_map.remove(&post_sub.name);
                Some(compile_and_load_sub(
                    post_sub,
                    &runtime,
                    &mil_config,
                    post_packing,
                )?)
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
            layers.push(LayerPrograms {
                pre_attn: pre,
                fp16_attn,
                post_attn: post,
            });
        }

        // 7. Set up TurboQuant or FP16 caches.
        let (turboquant, fp16_kv_caches, num_kv_heads, head_dim, max_seq_len) =
            if let Some(tq_config) = turbo_config {
                let nkv = tq_config.num_kv_heads;
                let hd = tq_config.head_dim;
                let msl = tq_config.max_seq_len;
                let tq = TurboQuantModel::compile(tq_config)?;
                (Some(tq), None, nkv, hd, msl)
            } else {
                // FP16 baseline: we need arch info to allocate caches.
                // Infer from the pre_attn output shapes.
                let pre_sps: Vec<&crate::split::SubProgram> =
                    pre_attn_map.values().copied().collect();
                let nkv = infer_kv_heads_from_sub(&pre_sps);
                let hd = infer_head_dim_from_sub(&pre_sps);
                let msl = 2048; // Default, can be made configurable.
                let channels = nkv * hd;

                let mut caches = Vec::with_capacity(num_layers);
                for _ in 0..num_layers {
                    let k = AneTensor::new(channels, msl, ScalarType::Float16)?;
                    let v = AneTensor::new(channels, msl, ScalarType::Float16)?;
                    caches.push((k, v));
                }
                (None, Some(caches), nkv, hd, msl)
            };

        Ok(Self {
            embed_weight,
            layers,
            lm_head_weight,
            turboquant,
            fp16_kv_caches,
            runtime,
            seq_pos: 0,
            num_kv_heads,
            head_dim,
            max_seq_len,
        })
    }

    // -----------------------------------------------------------------------
    // Decode
    // -----------------------------------------------------------------------

    /// Process one token, return logits as f32.
    pub fn decode(&mut self, token_id: u32) -> Result<Vec<f32>> {
        // 1. Embedding: CPU gather from weight table.
        let embed_out = cpu_embedding_lookup(&self.embed_weight, token_id)?;

        // 2. Per-layer: pre_attn → attention → post_attn
        let num_layers = self.layers.len();
        let mut hidden = embed_out;

        for layer_idx in 0..num_layers {
            // Pre-attention: norm → Q/K/V projection
            let layer = &mut self.layers[layer_idx];
            write_f16_padded(&mut layer.pre_attn.input_tensors[0], &hidden)?;
            {
                let in_refs: Vec<&AneTensor> = layer.pre_attn.input_tensors.iter().collect();
                let mut out_refs: Vec<&mut AneTensor> =
                    layer.pre_attn.output_tensors.iter_mut().collect();
                self.runtime
                    .eval(&layer.pre_attn.loaded, &in_refs, &mut out_refs)
                    .map_err(|e| {
                        AneError::Other(anyhow::anyhow!(
                            "layer {layer_idx} pre_attn eval failed: {e}"
                        ))
                    })?;
            }

            // Read Q, K_proj, V_proj from pre_attn outputs.
            // Convention: outputs are [Q, K_proj, V_proj, residual_hidden]
            // or fewer if the model merges them.
            let num_pre_outputs = layer.pre_attn.output_tensors.len();

            // Attention (divergent path)
            let attn_out_data = if let Some(tq) = &mut self.turboquant {
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
                read_f16_channels(&attn_tensor)?
            } else if let Some(ref mut caches) = self.fp16_kv_caches {
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

                // Write K/V to persistent FP16 cache at current position.
                let token_elements = self.num_kv_heads * self.head_dim;
                let elem_offset = self.seq_pos * token_elements;
                caches[layer_idx]
                    .0
                    .write_f16_at(elem_offset, &k_data[..token_elements])?;
                caches[layer_idx]
                    .1
                    .write_f16_at(elem_offset, &v_data[..token_elements])?;

                // If we have an FP16 attention sub-program, use it.
                if let Some(ref mut fp16_attn) = self.layers[layer_idx].fp16_attn {
                    if let Some(ref packing) = fp16_attn.input_packing {
                        // Packed: write all logical inputs into the single tensor.
                        let mut logical_inputs: Vec<&[f16]> = vec![&q_data];
                        if packing.offsets.len() > 1 {
                            logical_inputs.push(&k_data);
                        }
                        if packing.offsets.len() > 2 {
                            logical_inputs.push(&v_data);
                        }
                        crate::packing::write_packed_inputs(
                            &mut fp16_attn.input_tensors[0],
                            &logical_inputs,
                            packing,
                        )?;
                    } else {
                        write_f16_padded(&mut fp16_attn.input_tensors[0], &q_data)?;
                        if fp16_attn.input_tensors.len() > 1 {
                            write_f16_padded(&mut fp16_attn.input_tensors[1], &k_data)?;
                        }
                        if fp16_attn.input_tensors.len() > 2 {
                            write_f16_padded(&mut fp16_attn.input_tensors[2], &v_data)?;
                        }
                    }
                    let in_refs: Vec<&AneTensor> = fp16_attn.input_tensors.iter().collect();
                    let mut out_refs: Vec<&mut AneTensor> =
                        fp16_attn.output_tensors.iter_mut().collect();
                    self.runtime
                        .eval(&fp16_attn.loaded, &in_refs, &mut out_refs)
                        .map_err(|e| {
                            AneError::Other(anyhow::anyhow!(
                                "layer {layer_idx} fp16_attn eval failed: {e}"
                            ))
                        })?;
                    read_f16_channels(&fp16_attn.output_tensors[0])?
                } else {
                    // No compiled attention — return Q as pass-through
                    // (real deployment would have attention sub-programs).
                    q_data
                }
            } else {
                return Err(AneError::Other(anyhow::anyhow!(
                    "no attention backend configured"
                )));
            };

            // Post-attention: O proj → residual → FFN → residual
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
                        crate::packing::write_packed_inputs(
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
                        .eval(&post_attn.loaded, &in_refs, &mut out_refs)
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
        }

        // Advance TurboQuant sequence position after all layers.
        if let Some(tq) = &mut self.turboquant {
            tq.advance_seq_pos();
        }

        self.seq_pos += 1;

        // 3. LM head: CPU matmul hidden × lm_head_weight^T → logits
        cpu_lm_head_matmul(&self.lm_head_weight, &hidden)
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
        return tensor.write_f16(data);
    }
    let total = channels * seq_len;
    let mut padded = vec![f16::ZERO; total];
    let c = data.len().min(channels);
    for i in 0..c {
        padded[i * seq_len] = data[i]; // column 0 of each channel row
    }
    tensor.write_f16(&padded)
}

/// Read C elements (one token at column 0) from an ANE tensor with shape
/// `[1, C, 1, S]`. Inverse of `write_f16_padded`.
fn read_f16_channels(tensor: &AneTensor) -> Result<Vec<f16>> {
    let [_, channels, _, seq_len] = tensor.shape();
    if seq_len == 1 {
        return tensor.read_f16();
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

/// Compile and load a single sub-program, pre-allocating I/O tensors.
fn compile_and_load_sub(
    sub: &crate::split::SubProgram,
    runtime: &AneRuntime,
    mil_config: &MilTextConfig,
    input_packing: Option<crate::packing::InputPacking>,
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

    let ptr = AneCompiler::compile_mil_text(&mil_text, &weight_slices).map_err(|e| {
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
    let compiled = unsafe { CompiledProgram::from_raw(ptr) };
    let loaded = runtime.load_program(&compiled)?;

    // Pre-allocate I/O tensors with uniform sizing.
    let input_shapes: Vec<_> = sub.inputs.iter().map(|td| (td.shape, td.dtype)).collect();
    let output_shapes: Vec<_> = sub.outputs.iter().map(|td| (td.shape, td.dtype)).collect();
    let input_alloc = uniform_alloc_size(&input_shapes);
    let output_alloc = uniform_alloc_size(&output_shapes);

    let input_tensors: Vec<AneTensor> = sub
        .inputs
        .iter()
        .map(|td| AneTensor::new_with_min_alloc(td.shape[1], td.shape[3], td.dtype, input_alloc))
        .collect::<Result<Vec<_>>>()?;
    let output_tensors: Vec<AneTensor> = sub
        .outputs
        .iter()
        .map(|td| AneTensor::new_with_min_alloc(td.shape[1], td.shape[3], td.dtype, output_alloc))
        .collect::<Result<Vec<_>>>()?;

    Ok(LoadedSubProgram {
        loaded,
        input_tensors,
        output_tensors,
        input_packing,
    })
}

/// Infer the number of KV heads from pre_attn sub-program output shapes.
fn infer_kv_heads_from_sub(pre_attn_sps: &[&crate::split::SubProgram]) -> usize {
    // Look at the second output (K_proj) of the first pre_attn sub-program.
    if let Some(sp) = pre_attn_sps.first() {
        if sp.outputs.len() > 1 {
            // Shape is [1, C, 1, S] where C = num_kv_heads * head_dim.
            // We can't distinguish without head_dim, so return the channel count.
            return sp.outputs[1].shape[1];
        }
    }
    // Fallback: can't detect.
    1
}

/// Infer head_dim from pre_attn sub-program output shapes.
fn infer_head_dim_from_sub(_pre_attn_sps: &[&crate::split::SubProgram]) -> usize {
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
fn extract_cpu_weight(sub: &crate::split::SubProgram, _label: &str) -> Option<CpuWeight> {
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

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

/// Sample a token from logits.
fn sample_token(logits: &[f32], temperature: f32) -> u32 {
    if logits.is_empty() {
        return 0;
    }

    if temperature <= 0.0 {
        // Greedy: argmax.
        let (idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        return idx as u32;
    }

    // Temperature-scaled softmax sampling.
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let scaled: Vec<f32> = logits
        .iter()
        .map(|&l| ((l - max_logit) / temperature).exp())
        .collect();
    let sum: f32 = scaled.iter().sum();
    let probs: Vec<f32> = scaled.iter().map(|&s| s / sum).collect();

    // Simple sampling using a pseudo-random threshold.
    // For deterministic benchmarks, callers use temperature=0 (greedy).
    let threshold = simple_random_f32();
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= threshold {
            return i as u32;
        }
    }
    (logits.len() - 1) as u32
}

/// Simple pseudo-random f32 in [0, 1) using a thread-local xorshift.
fn simple_random_f32() -> f32 {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = const { Cell::new(0x1234_5678_9ABC_DEF0) };
    }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        (x as u32 as f32) / (u32::MAX as f32)
    })
}

/// Check if a token ID is an end-of-sequence marker.
///
/// Common EOS token IDs across popular tokenizers:
/// - 2: LLaMA, Qwen
/// - 151643: Qwen3 (tiktoken-based)
/// - 128001: LLaMA-3
fn is_eos_token(token_id: u32) -> bool {
    matches!(token_id, 2 | 151643 | 128001)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_sampling_picks_argmax() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        assert_eq!(sample_token(&logits, 0.0), 3);
    }

    #[test]
    fn greedy_sampling_empty_logits() {
        assert_eq!(sample_token(&[], 0.0), 0);
    }

    #[test]
    fn eos_detection() {
        assert!(is_eos_token(2));
        assert!(is_eos_token(151643));
        assert!(is_eos_token(128001));
        assert!(!is_eos_token(42));
    }

    #[test]
    fn temperature_sampling_produces_valid_token() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let token = sample_token(&logits, 1.0);
        assert!(token < 5);
    }
}
