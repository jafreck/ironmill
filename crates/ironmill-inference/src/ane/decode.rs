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

use super::device::AneDevice;
use super::turboquant::TurboQuantConfig;
use super::turboquant::TurboQuantModel;
use super::turboquant::mil_emitter;
use super::turboquant::mil_emitter::MIN_IO_SEQ;
use crate::ane::{AneError, Result};
use ironmill_core::ane::mil_text::{MilTextConfig, program_to_mil_text};
use mil_rs::ir::ScalarType;
use std::sync::Arc;

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
pub struct AneInference<D: AneDevice> {
    /// Embedding weight table: [vocab_size, hidden_size] as fp16 bytes.
    embed_weight: CpuWeight,
    /// Per-layer: (pre_attn, post_attn) sub-programs.
    /// The fp16_attn sub-program is only present in baseline mode.
    layers: Vec<LayerPrograms<D>>,
    /// LM head projection. ANE-accelerated when possible, CPU fallback otherwise.
    lm_head: LmHead<D>,
    /// Final RMSNorm weight applied before lm_head. Shape: [hidden_size].
    final_norm_weight: Option<Vec<f16>>,
    /// Optional TurboQuant model (replaces layer attention when enabled).
    turboquant: Option<TurboQuantModel<D>>,
    /// FP16 KV caches (used when TurboQuant is disabled).
    /// Per-layer (K, V) tensors. `None` when TurboQuant manages the cache.
    fp16_kv_caches: Option<Vec<(AneTensor, AneTensor)>>,
    /// Compiled FP16 attention program (shared across all layers).
    /// Only present when TurboQuant is NOT configured.
    fp16_attn_program: Option<D::Program>,
    /// Staging tensor for Q input to FP16 attention.
    fp16_attn_q_staging: Option<AneTensor>,
    /// Staging tensor for FP16 attention output.
    fp16_attn_out_staging: Option<AneTensor>,
    /// Causal mask tensor for FP16 attention. Shape: [1, 1, 1, max_seq_len].
    /// Updated each step: 0.0 for positions ≤ seq_pos, -inf for future.
    fp16_attn_mask: Option<AneTensor>,
    /// Device handle.
    device: Arc<D>,
    /// ANE QoS level for eval calls.
    qos: u32,
    /// Current sequence position.
    seq_pos: usize,
    /// Number of KV heads (for FP16 cache management).
    num_kv_heads: usize,
    /// Head dimension.
    head_dim: usize,
    /// Maximum sequence length for FP16 caches.
    max_seq_len: usize,
    /// Pre-extracted RoPE cos/sin cache tables (from model consts).
    /// Flat f16 arrays: `[num_positions * values_per_position]`.
    /// Indexed as `cache[pos * rope_cache_dim .. (pos+1) * rope_cache_dim]`.
    rope_cos_cache: Vec<f16>,
    rope_sin_cache: Vec<f16>,
    /// Number of f16 values per position in the RoPE cache.
    rope_cache_dim: usize,
    /// Whether cache-write ops were fused into pre_attn sub-programs.
    cache_write_fused: bool,
    /// Per-head Q/K normalization weights (Qwen3 feature).
    /// Shape: [head_dim] each. Applied per-head after Q/K projection.
    qk_norm_weights: Option<Vec<(Vec<f16>, Vec<f16>)>>,
}

/// CPU-side weight tensor for embedding/lm_head.
struct CpuWeight {
    /// Raw fp16 bytes, row-major.
    data: Vec<u8>,
    /// [rows, cols].
    shape: [usize; 2],
}

/// Per-layer compiled sub-programs.
struct LayerPrograms<D: AneDevice> {
    pre_attn: LoadedSubProgram<D>,
    /// FP16 attention sub-program. Only compiled in baseline mode;
    /// in TurboQuant mode this is `None`.
    fp16_attn: Option<LoadedSubProgram<D>>,
    /// Post-attention sub-program. `None` for layers where all ops
    /// fall within the attention cluster (e.g., position-only layers).
    post_attn: Option<LoadedSubProgram<D>>,
    /// Input mapping for fp16_attn (which tensor indices are Q/K/V/cos/sin).
    attn_input_map: Option<AttnInputMap>,
}

/// Mapping from logical Q/K/V/cos/sin roles to fp16_attn input tensor indices.
///
/// After stripping gather ops from the attention cluster, the fp16_attn
/// sub-program's inputs are determined by `build_sub_program` in alphabetical
/// order. This map records which index is which so `decode()` writes
/// the correct data to the correct input tensor.
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
/// Programs are compiled and loaded upfront via `device.compile()`.
/// During decode, each program is evaluated via `device.eval()`.
/// `D::Program`'s `Drop` releases hardware resources automatically.
struct LoadedSubProgram<D: AneDevice> {
    program: D::Program,
    input_tensors: Vec<AneTensor>,
    output_tensors: Vec<AneTensor>,
    /// If inputs were spatially packed, stores the packing metadata.
    input_packing: Option<super::bundle_manifest::InputPacking>,
}

/// LM head projection — ANE-accelerated or CPU fallback.
enum LmHead<D: AneDevice> {
    /// ANE-accelerated: chunked conv1×1 across multiple ANE programs.
    Ane(AneLmHead<D>),
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
struct AneLmHead<D: AneDevice> {
    chunks: Vec<LmHeadChunk<D>>,
    vocab_size: usize,
    qos: u32,
    device: Arc<D>,
}

/// One chunk of the ANE lm_head — a conv1×1 with ≤16384 output channels.
struct LmHeadChunk<D: AneDevice> {
    program: D::Program,
    input_tensor: AneTensor,
    output_tensor: AneTensor,
    out_channels: usize,
}

impl<D: AneDevice> AneLmHead<D> {
    /// Compile the chunked ANE lm_head from a CPU weight tensor.
    ///
    /// Splits `[vocab_size, hidden_size]` into chunks of ≤16384 output
    /// channels, compiles each as a conv1×1 ANE program with BLOBFILE
    /// weights, and pre-allocates I/O tensors.
    ///
    /// Uses donor/patch optimization: all full-size chunks share the same
    /// MIL text (same `hidden_size` × `LM_HEAD_MAX_CHUNK_CH` conv1×1).
    /// Only the first full-size chunk is compiled; the rest reuse its
    /// compiled `net.plist` via `model::patch_weights`.
    fn compile(device: Arc<D>, weight: &CpuWeight, qos: u32) -> Result<Self> {
        let [vocab_size, hidden_size] = weight.shape;

        let num_chunks = vocab_size.div_ceil(LM_HEAD_MAX_CHUNK_CH);
        let mut chunks: Vec<LmHeadChunk<D>> = Vec::with_capacity(num_chunks);

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
                let donor = &chunks[donor_idx].program;
                device.compile_patched(donor, &mil_text, &[(weight_path, &chunk_data)], qos)?
            } else {
                device.compile(&mil_text, &[(weight_path, &chunk_data)], qos)?
            };
            // DON'T unload here — patches need the donor's net.plist to exist.
            // Programs will be unloaded in AneInference::Drop.

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
                program: compiled,
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
            qos,
            device,
        })
    }

    /// Run the lm_head projection on ANE, returning logits as f32.
    fn forward(&mut self, hidden: &[f16]) -> Result<Vec<f32>> {
        let mut logits = Vec::with_capacity(self.vocab_size);

        for chunk in &mut self.chunks {
            // Write hidden state to input tensor (column 0, zero-padded).
            write_f16_padded(&mut chunk.input_tensor, hidden)?;

            // Eval conv1×1 on ANE (program stays loaded).
            self.device.eval(
                &chunk.program,
                &[&chunk.input_tensor],
                &mut [&mut chunk.output_tensor],
                self.qos,
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

impl<D: AneDevice> AneInference<D> {
    /// Load a pre-compiled `.ironml` decode bundle.
    ///
    /// The bundle must have been created by `compile_decode_bundle()` and
    /// saved via `AneDecodeBundle::save()`. This method reads the manifest,
    /// CPU weights, and per-layer MIL programs, then compiles them on the
    /// ANE device and sets up runtime state (KV caches, etc.).
    pub fn from_bundle(
        device: Arc<D>,
        bundle_path: &std::path::Path,
        turbo_config: Option<TurboQuantConfig>,
        qos: u32,
    ) -> Result<Self> {
        use super::bundle_manifest::BundleManifest;

        // 1. Read manifest.
        let manifest_json = std::fs::read_to_string(bundle_path.join("manifest.json"))
            .map_err(|e| AneError::Other(anyhow::anyhow!("failed to read manifest: {e}")))?;
        let manifest: BundleManifest = serde_json::from_str(&manifest_json)
            .map_err(|e| AneError::Other(anyhow::anyhow!("invalid manifest: {e}")))?;
        let decode = manifest
            .decode
            .ok_or_else(|| AneError::Other(anyhow::anyhow!("not a decode bundle")))?;
        let arch = &decode.architecture;

        let programs_dir = bundle_path.join("programs");
        let weights_dir = bundle_path.join("weights");

        // 2. Load CPU weights.
        let cpu_weights_dir = bundle_path.join("cpu_weights");
        let embed_data = std::fs::read(cpu_weights_dir.join("embedding.bin")).map_err(|e| {
            AneError::Other(anyhow::anyhow!("failed to read embedding weights: {e}"))
        })?;
        let embed_weight = CpuWeight {
            data: embed_data,
            shape: [arch.vocab_size, arch.hidden_size],
        };

        // Tied embeddings fallback: if lm_head.bin doesn't exist, reuse embed.
        let lm_head_data = std::fs::read(cpu_weights_dir.join("lm_head.bin"))
            .unwrap_or_else(|_| embed_weight.data.clone());
        let lm_head_cpu_weight = CpuWeight {
            data: lm_head_data,
            shape: [arch.vocab_size, arch.hidden_size],
        };

        let final_norm_weight = std::fs::read(cpu_weights_dir.join("final_norm.bin"))
            .ok()
            .map(|data| {
                data.chunks_exact(2)
                    .map(|c| f16::from_le_bytes([c[0], c[1]]))
                    .collect::<Vec<f16>>()
            });

        // Load RoPE caches.
        let rope_cos_data = std::fs::read(cpu_weights_dir.join("rope_cos.bin"))
            .map_err(|e| AneError::Other(anyhow::anyhow!("failed to read rope_cos: {e}")))?;
        let rope_sin_data = std::fs::read(cpu_weights_dir.join("rope_sin.bin"))
            .map_err(|e| AneError::Other(anyhow::anyhow!("failed to read rope_sin: {e}")))?;
        let rope_cos_cache: Vec<f16> = rope_cos_data
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]))
            .collect();
        let rope_sin_cache: Vec<f16> = rope_sin_data
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]))
            .collect();
        let rope_cache_dim = if rope_cos_cache.is_empty() {
            0
        } else {
            arch.head_dim / 2
        };

        // 3. Compile LM head (try ANE, fall back to CPU).
        let lm_head = match AneLmHead::compile(Arc::clone(&device), &lm_head_cpu_weight, qos) {
            Ok(ane_lm_head) => LmHead::Ane(ane_lm_head),
            Err(e) => {
                eprintln!("warning: ANE lm_head compilation failed: {e}");
                LmHead::Cpu(lm_head_cpu_weight)
            }
        };

        // 4. Compile per-layer sub-programs from bundle.
        let num_layers = decode.layers.len();

        // Pre_attn min output alloc for TurboQuant compatibility.
        let pre_attn_min_output_alloc = if let Some(ref tc) = turbo_config {
            let kv_ch = tc.num_kv_heads * tc.head_dim;
            let q_ch = tc.num_heads * tc.head_dim;
            uniform_alloc_size(&[
                ([1, kv_ch, 1, MIN_IO_SEQ], ScalarType::Float16),
                ([1, kv_ch, 1, MIN_IO_SEQ], ScalarType::Float16),
                ([1, 1, tc.head_dim, tc.head_dim], ScalarType::Float16),
                ([1, q_ch, 1, MIN_IO_SEQ], ScalarType::Float16),
                ([1, kv_ch, 1, tc.max_seq_len], ScalarType::Int8),
                ([1, kv_ch, 1, tc.max_seq_len], ScalarType::Int8),
                ([1, 1, tc.head_dim, tc.head_dim], ScalarType::Float16),
            ])
        } else {
            0
        };

        let cache_write_fused = decode.layers.first().is_some_and(|l| l.cache_write_fused);

        let mut layers: Vec<LayerPrograms<D>> = Vec::with_capacity(num_layers);
        for (i, layer_manifest) in decode.layers.iter().enumerate() {
            let pre = if i == 0 || !layer_manifest.donor_compatible {
                // First layer or non-donor-compatible: full compile.
                compile_sub_from_bundle(
                    &layer_manifest.pre_attn,
                    &programs_dir,
                    &weights_dir,
                    &*device,
                    pre_attn_min_output_alloc,
                    qos,
                )?
            } else {
                // Layers 1+: patch weights from layer-0 donor.
                match compile_sub_from_bundle_with_donor(
                    &layer_manifest.pre_attn,
                    &layers[0].pre_attn.program,
                    &programs_dir,
                    &weights_dir,
                    &*device,
                    pre_attn_min_output_alloc,
                    qos,
                ) {
                    Ok(loaded) => loaded,
                    Err(e) => {
                        eprintln!(
                            "warning: layer {} pre_attn donor patch failed, \
                             falling back to full compile: {e}",
                            layer_manifest.index
                        );
                        compile_sub_from_bundle(
                            &layer_manifest.pre_attn,
                            &programs_dir,
                            &weights_dir,
                            &*device,
                            pre_attn_min_output_alloc,
                            qos,
                        )?
                    }
                }
            };

            let post = if let Some(ref post_manifest) = layer_manifest.post_attn {
                if i == 0 || !layer_manifest.donor_compatible {
                    Some(compile_sub_from_bundle(
                        post_manifest,
                        &programs_dir,
                        &weights_dir,
                        &*device,
                        0,
                        qos,
                    )?)
                } else if let Some(ref donor_post) = layers[0].post_attn {
                    match compile_sub_from_bundle_with_donor(
                        post_manifest,
                        &donor_post.program,
                        &programs_dir,
                        &weights_dir,
                        &*device,
                        0,
                        qos,
                    ) {
                        Ok(loaded) => Some(loaded),
                        Err(e) => {
                            eprintln!(
                                "warning: layer {} post_attn donor patch failed, \
                                 falling back to full compile: {e}",
                                layer_manifest.index
                            );
                            Some(compile_sub_from_bundle(
                                post_manifest,
                                &programs_dir,
                                &weights_dir,
                                &*device,
                                0,
                                qos,
                            )?)
                        }
                    }
                } else {
                    Some(compile_sub_from_bundle(
                        post_manifest,
                        &programs_dir,
                        &weights_dir,
                        &*device,
                        0,
                        qos,
                    )?)
                }
            } else {
                None
            };

            let fp16_attn = if let Some(ref attn_manifest) = layer_manifest.fp16_attn {
                match compile_sub_from_bundle(
                    attn_manifest,
                    &programs_dir,
                    &weights_dir,
                    &*device,
                    0,
                    qos,
                ) {
                    Ok(loaded) => Some(loaded),
                    Err(e) => {
                        eprintln!(
                            "warning: layer {} fp16_attn compilation failed: {e}",
                            layer_manifest.index
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
                attn_input_map: None,
            });
        }

        if num_layers > 1 {
            let patched = (num_layers - 1) * if layers[0].post_attn.is_some() { 2 } else { 1 };
            eprintln!(
                "donor/patch: compiled 1 donor layer, patched {patched} sub-programs \
                 ({} compilations saved)",
                patched
            );
        }

        // 5. Set up TurboQuant or FP16 caches (same as compile()).
        let (
            turboquant,
            fp16_kv_caches,
            fp16_attn_program,
            fp16_attn_q_staging,
            fp16_attn_out_staging,
            fp16_attn_mask,
            num_kv_heads,
            head_dim,
            max_seq_len,
        ) = if let Some(tq_config) = turbo_config {
            let nkv = tq_config.num_kv_heads;
            let hd = tq_config.head_dim;
            let msl = tq_config.max_seq_len;
            let tq = TurboQuantModel::compile(Arc::clone(&device), tq_config)?;
            (Some(tq), None, None, None, None, None, nkv, hd, msl)
        } else {
            let nh = arch.num_heads;
            let nkv = arch.num_kv_heads;
            let hd = arch.head_dim;
            let msl = arch.max_seq_len;
            let kv_channels = nkv * hd;
            let q_channels = nh * hd;

            // Compile hand-written FP16 attention MIL (shared across layers).
            let attn_program = mil_emitter::build_fp16_attention_program(nh, nkv, hd, msl, msl);
            let mil_config = MilTextConfig::default();
            let (mil, _) = program_to_mil_text(&attn_program, &mil_config).map_err(|e| {
                AneError::Other(anyhow::anyhow!("FP16 attention MIL text failed: {e}"))
            })?;
            let attn_compiled = device.compile(&mil, &[], qos)?;

            let attn_alloc = uniform_alloc_size(&[
                ([1, q_channels, 1, MIN_IO_SEQ], ScalarType::Float16),
                ([1, kv_channels, 1, msl], ScalarType::Float16),
                ([1, kv_channels, 1, msl], ScalarType::Float16),
                ([1, 1, 1, msl], ScalarType::Float16),
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
            let mut mask_tensor =
                AneTensor::new_with_min_alloc(1, msl, ScalarType::Float16, attn_alloc)?;
            let neg_inf = f16::NEG_INFINITY;
            let mask_data = vec![neg_inf; msl];
            mask_tensor.write_f16(&mask_data)?;

            (
                None,
                Some(caches),
                Some(attn_compiled),
                Some(q_staging),
                Some(out_staging),
                Some(mask_tensor),
                nkv,
                hd,
                msl,
            )
        };

        // TODO: Load per-layer QK norm weights from bundle.
        // QK norm weights need to be serialized into the bundle by
        // compile_decode_bundle. For now, set to None.
        let qk_norm_weights: Option<Vec<(Vec<f16>, Vec<f16>)>> = None;

        Ok(Self {
            embed_weight,
            layers,
            lm_head,
            final_norm_weight,
            turboquant,
            fp16_kv_caches,
            fp16_attn_program,
            fp16_attn_q_staging,
            fp16_attn_out_staging,
            fp16_attn_mask,
            device,
            qos,
            seq_pos: 0,
            num_kv_heads,
            head_dim,
            max_seq_len,
            rope_cos_cache,
            rope_sin_cache,
            rope_cache_dim,
            cache_write_fused,
            qk_norm_weights,
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

        // Update causal mask: unmask current position.
        // mask[seq_pos] = 0.0 (allow attention), rest stays -inf.
        if let Some(ref mut mask) = self.fp16_attn_mask {
            mask.write_f16_at(self.seq_pos, &[f16::ZERO])?;
        }
        // Update TurboQuant causal mask (for QJL correction path).
        if let Some(ref mut tq) = self.turboquant {
            tq.update_mask()?;
        }

        let mut d_pre_attn = std::time::Duration::ZERO;
        let mut d_read_qkv = std::time::Duration::ZERO;
        let mut d_attn = std::time::Duration::ZERO;
        let mut d_post_attn = std::time::Duration::ZERO;

        let effective_layers = std::env::var("IRONMILL_MAX_LAYERS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(num_layers)
            .min(num_layers);

        for layer_idx in 0..effective_layers {
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
                self.device
                    .eval(&layer.pre_attn.program, &in_refs, &mut out_refs, self.qos)
                    .map_err(|e| {
                        AneError::Other(anyhow::anyhow!(
                            "layer {layer_idx} pre_attn eval failed: {e}"
                        ))
                    })?;
            }
            if let Some(t) = t0 {
                d_pre_attn += t.elapsed();
            }

            // Read Q, K_quant, V_quant from pre_attn outputs.
            // With cache-write fusion, outputs are [Q, K_quant, V_quant, ...]
            // K_quant/V_quant are already rotated + quantized (fp16 with INT8-range values).
            // When QJL is enabled, an extra output carries the original K_proj.
            // Read Q/K/V from pre_attn outputs.
            // Outputs are sorted lexicographically by name (k_proj < q_proj < v_proj),
            // NOT in Q/K/V order. Identify Q by channel count: Q has num_heads * head_dim
            // channels while K/V have num_kv_heads * head_dim (smaller for GQA models).
            let num_pre_outputs = layer.pre_attn.output_tensors.len();

            // Attention (divergent path)
            let attn_out_data = if let Some(tq) = &mut self.turboquant {
                let t0 = if profiling {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let q = &layer.pre_attn.output_tensors[0];
                let attn_tensor = if self.cache_write_fused {
                    // Fused path: pre_attn already produced K_quant/V_quant
                    let k_quant = if num_pre_outputs > 1 {
                        &layer.pre_attn.output_tensors[1]
                    } else {
                        &layer.pre_attn.output_tensors[0]
                    };
                    let v_quant = if num_pre_outputs > 2 {
                        &layer.pre_attn.output_tensors[2]
                    } else {
                        &layer.pre_attn.output_tensors[0]
                    };
                    // When QJL is enabled, output[3] is the original K_proj
                    let k_original = if tq.config().enable_qjl && num_pre_outputs > 3 {
                        Some(&layer.pre_attn.output_tensors[3])
                    } else {
                        None
                    };
                    tq.step_attention_fused(layer_idx, q, k_quant, v_quant, k_original)
                } else {
                    // Non-fused path: pre_attn outputs raw K_proj/V_proj
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
                    tq.step_attention(layer_idx, q, k_proj, v_proj)
                };
                let attn_tensor = attn_tensor.map_err(|e| {
                    AneError::Other(anyhow::anyhow!(
                        "layer {layer_idx} turboquant attention failed: {e}"
                    ))
                })?;
                let result = read_f16_channels(&attn_tensor)?;
                if let Some(t) = t0 {
                    d_attn += t.elapsed();
                }
                result
            } else if let Some(ref mut attn) = layer.fp16_attn {
                // Per-layer fp16_attn from the splitter: execute directly.
                // Inputs match pre_attn outputs (same ANE layout, same shapes).
                let t0 = if profiling {
                    Some(std::time::Instant::now())
                } else {
                    None
                };

                if let Some(ref map) = layer.attn_input_map {
                    // Copy Q/K/V from pre_attn outputs to fp16_attn inputs.
                    let q_src = layer.pre_attn.output_tensors[map.q_idx].read_f16()?;
                    attn.input_tensors[map.q_idx].write_f16(&q_src)?;

                    let k_src = layer.pre_attn.output_tensors[map.k_idx].read_f16()?;
                    attn.input_tensors[map.k_idx].write_f16(&k_src)?;

                    let v_src = layer.pre_attn.output_tensors[map.v_idx].read_f16()?;
                    attn.input_tensors[map.v_idx].write_f16(&v_src)?;

                    // Fill RoPE cos/sin from CPU cache.
                    for &cos_idx in &map.rope_cos_indices {
                        let cos_slice = &self.rope_cos_cache[self.seq_pos * self.rope_cache_dim
                            ..(self.seq_pos + 1) * self.rope_cache_dim];
                        write_f16_padded(&mut attn.input_tensors[cos_idx], cos_slice)?;
                    }
                    for &sin_idx in &map.rope_sin_indices {
                        let sin_slice = &self.rope_sin_cache[self.seq_pos * self.rope_cache_dim
                            ..(self.seq_pos + 1) * self.rope_cache_dim];
                        write_f16_padded(&mut attn.input_tensors[sin_idx], sin_slice)?;
                    }
                } else {
                    // No input map — copy pre_attn outputs to fp16_attn inputs.
                    for (i, src) in layer.pre_attn.output_tensors.iter().enumerate() {
                        if i < attn.input_tensors.len() {
                            let data = src.read_f16()?;
                            attn.input_tensors[i].write_f16(&data)?;
                        }
                    }
                }

                // Eval fp16_attn on ANE.
                {
                    let in_refs: Vec<&AneTensor> = attn.input_tensors.iter().collect();
                    let mut out_refs: Vec<&mut AneTensor> =
                        attn.output_tensors.iter_mut().collect();
                    self.device
                        .eval(&attn.program, &in_refs, &mut out_refs, self.qos)
                        .map_err(|e| {
                            AneError::Other(anyhow::anyhow!(
                                "layer {layer_idx} fp16_attn eval failed: {e}"
                            ))
                        })?;
                }

                let result = read_f16_channels(&attn.output_tensors[0])?;
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
                // Outputs are lexicographically sorted: [k_proj, q_proj, v_proj]
                // Index 0 = K, index 1 = Q, index 2 = V
                // ANE layout pass doubles channel dimensions. Truncate to the
                // real model dimensions (num_heads * head_dim for Q,
                // num_kv_heads * head_dim for K/V).
                let _real_q_ch = (self.num_kv_heads * self.head_dim).max(self.head_dim); // at least head_dim
                // Q has more heads than K/V in GQA — compute from num_kv_heads ratio
                let gqa = if self.num_kv_heads > 0 {
                    // num_heads / num_kv_heads, but we only have num_kv_heads stored
                    // Infer: Q output channels / K output channels = GQA ratio
                    let q_ane_ch = layer
                        .pre_attn
                        .output_tensors
                        .get(1)
                        .map(|t| t.shape()[1])
                        .unwrap_or(0);
                    let k_ane_ch = layer.pre_attn.output_tensors[0].shape()[1];
                    if k_ane_ch > 0 { q_ane_ch / k_ane_ch } else { 1 }
                } else {
                    1
                };
                let real_kv_ch = self.num_kv_heads * self.head_dim;
                let real_q_ch = real_kv_ch * gqa;

                let mut q_data = if num_pre_outputs > 1 {
                    read_f16_channels(&layer.pre_attn.output_tensors[1])?
                } else {
                    read_f16_channels(&layer.pre_attn.output_tensors[0])?
                };
                let mut k_data = read_f16_channels(&layer.pre_attn.output_tensors[0])?;
                let mut v_data = if num_pre_outputs > 2 {
                    read_f16_channels(&layer.pre_attn.output_tensors[2])?
                } else {
                    k_data.clone()
                };
                // Truncate to real model dimensions (undo ANE layout doubling).
                q_data.truncate(real_q_ch);
                k_data.truncate(real_kv_ch);
                v_data.truncate(real_kv_ch);

                // Apply per-head QK normalization (Qwen3 feature).
                if let Some(ref norms) = self.qk_norm_weights {
                    if let Some((q_norm_w, k_norm_w)) = norms.get(layer_idx) {
                        apply_per_head_rms_norm(&mut q_data, q_norm_w, self.head_dim);
                        apply_per_head_rms_norm(&mut k_data, k_norm_w, self.head_dim);
                    }
                }

                // Apply RoPE rotation on CPU. Pre_attn outputs unrotated Q/K
                // because RoPE gather ops were stripped during compilation.
                if self.rope_cache_dim > 0 {
                    let pos = self.seq_pos;
                    let cos_start = pos * self.rope_cache_dim;
                    let cos_end = cos_start + self.rope_cache_dim;
                    let cos = &self.rope_cos_cache[cos_start..cos_end];
                    let sin = &self.rope_sin_cache[cos_start..cos_end];
                    apply_rope_rotation(&mut q_data, cos, sin, self.head_dim);
                    apply_rope_rotation(&mut k_data, cos, sin, self.head_dim);
                }
                if let Some(t) = t0 {
                    d_read_qkv += t.elapsed();
                }

                let t0 = if profiling {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                // Write K/V to persistent FP16 cache at current position.
                // Cache shape is [1, channels, 1, max_seq_len] in NCHW layout.
                // Element (c, s) is at flat index c * max_seq_len + s.
                // Each channel gets one value at the current sequence position.
                let k_elements = k_data.len();
                let v_elements = v_data.len();
                for (c, &val) in k_data.iter().enumerate().take(k_elements) {
                    caches[layer_idx]
                        .0
                        .write_f16_at(c * self.max_seq_len + self.seq_pos, &[val])?;
                }
                for (c, &val) in v_data.iter().enumerate().take(v_elements) {
                    caches[layer_idx]
                        .1
                        .write_f16_at(c * self.max_seq_len + self.seq_pos, &[val])?;
                }

                // Hand-written FP16 attention: Q + K_cache + V_cache + mask → attn_output
                let result = if let Some(ref fp16_program) = self.fp16_attn_program {
                    let q_staging = self.fp16_attn_q_staging.as_mut().unwrap();
                    let out_staging = self.fp16_attn_out_staging.as_mut().unwrap();
                    let mask = self.fp16_attn_mask.as_ref().unwrap();
                    write_f16_padded(q_staging, &q_data)?;
                    let (k_cache, v_cache) = (&caches[layer_idx].0, &caches[layer_idx].1);
                    self.device
                        .eval(
                            fp16_program,
                            &[q_staging, k_cache, v_cache, mask],
                            &mut [out_staging],
                            self.qos,
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
                        super::bundle_manifest::write_packed_inputs(
                            &mut post_attn.input_tensors[0],
                            &[&attn_out_data],
                            packing,
                        )?;
                    }
                } else {
                    write_f16_padded(&mut post_attn.input_tensors[0], &attn_out_data)?;
                    // Write residual (pre-layer hidden state) to the second input.
                    // The residual flows directly from the decode loop's hidden state,
                    // not from a pre_attn output, since the 3-way split (pre_attn/
                    // attention/post_attn) doesn't pass the residual through pre_attn.
                    if post_attn.input_tensors.len() > 1 {
                        write_f16_padded(&mut post_attn.input_tensors[1], &hidden)?;
                    }
                }
                {
                    let in_refs: Vec<&AneTensor> = post_attn.input_tensors.iter().collect();
                    let mut out_refs: Vec<&mut AneTensor> =
                        post_attn.output_tensors.iter_mut().collect();
                    self.device
                        .eval(&post_attn.program, &in_refs, &mut out_refs, self.qos)
                        .map_err(|e| {
                            AneError::Other(anyhow::anyhow!(
                                "layer {layer_idx} post_attn eval failed: {e}"
                            ))
                        })?;
                }
                hidden = read_f16_channels(&post_attn.output_tensors[0])?;

                // Debug: trace last few layers for explosion diagnosis
                if self.seq_pos == 0
                    && layer_idx >= 24
                    && std::env::var("IRONMILL_TRACE_LAST").is_ok()
                {
                    let f32s: Vec<f32> = hidden.iter().map(|x| x.to_f32()).collect();
                    let absmax = f32s.iter().map(|x| x.abs()).fold(0f32, f32::max);
                    let has_nan = f32s.iter().any(|x| x.is_nan());
                    let has_inf = f32s.iter().any(|x| x.is_infinite());
                    eprintln!(
                        "  [L{:2}] absmax={:.2} nan={} inf={} first3=[{:.4},{:.4},{:.4}]",
                        layer_idx, absmax, has_nan, has_inf, f32s[0], f32s[1], f32s[2]
                    );
                }
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

        // 3. Final RMSNorm + LM head.
        let t0 = if profiling {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Apply final RMSNorm before projecting to vocab logits.
        if let Some(norm_w) = &self.final_norm_weight {
            cpu_rms_norm(&mut hidden, norm_w);
        }

        // Debug: trace post-norm hidden and logits
        if self.seq_pos <= 3 && std::env::var("IRONMILL_TRACE_LAST").is_ok() {
            let f32s: Vec<f32> = hidden.iter().map(|x| x.to_f32()).collect();
            let absmax = f32s.iter().map(|x| x.abs()).fold(0f32, f32::max);
            eprintln!(
                "  [norm] absmax={:.4} first3=[{:.4},{:.4},{:.4}]",
                absmax, f32s[0], f32s[1], f32s[2]
            );
        }

        let logits = match &mut self.lm_head {
            LmHead::Ane(ane_lm_head) => ane_lm_head.forward(&hidden),
            LmHead::Cpu(weight) => cpu_lm_head_matmul(weight, &hidden),
        };

        if self.seq_pos <= 3 && std::env::var("IRONMILL_TRACE_LAST").is_ok() {
            if let Ok(ref l) = logits {
                let max = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let argmax = l
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                eprintln!(
                    "  [logits] max={:.4} argmax={} len={}",
                    max,
                    argmax,
                    l.len()
                );
            }
        }

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
            if is_eos_token(token_id, DEFAULT_EOS_TOKENS) {
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
        // Re-initialize the causal mask to all -inf.
        if let Some(ref mut mask) = self.fp16_attn_mask {
            let msl = self.max_seq_len;
            let mask_data = vec![f16::NEG_INFINITY; msl];
            let _ = mask.write_f16(&mask_data);
        }
        // Zero the FP16 KV caches for the new sequence.
        if let Some(ref mut caches) = self.fp16_kv_caches {
            for (k, v) in caches.iter_mut() {
                let k_size = k.shape()[1] * k.shape()[3];
                let v_size = v.shape()[1] * v.shape()[3];
                let _ = k.write_f16(&vec![f16::ZERO; k_size]);
                let _ = v.write_f16(&vec![f16::ZERO; v_size]);
            }
        }
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

// ---------------------------------------------------------------------------
// RoPE rotation helper
// ---------------------------------------------------------------------------

/// Apply rotary position encoding to a Q or K vector on CPU.
///
/// `data` is `[num_heads * head_dim]` values. `cos`/`sin` are
/// `[head_dim / 2]` values for the current position. For each head,
/// interleaved dimension pairs `(2i, 2i+1)` are rotated.
fn apply_rope_rotation(data: &mut [f16], cos: &[f16], sin: &[f16], head_dim: usize) {
    let half_dim = head_dim / 2;
    let num_heads = data.len() / head_dim;
    for h in 0..num_heads {
        let base = h * head_dim;
        for i in 0..half_dim.min(cos.len()) {
            let x0 = data[base + i].to_f32();
            let x1 = data[base + i + half_dim].to_f32();
            let c = cos[i].to_f32();
            let s = sin[i].to_f32();
            data[base + i] = f16::from_f32(x0 * c - x1 * s);
            data[base + i + half_dim] = f16::from_f32(x0 * s + x1 * c);
        }
    }
}

// ---------------------------------------------------------------------------
// QK normalization helper
// ---------------------------------------------------------------------------

/// Apply per-head RMSNorm to a Q or K vector on CPU.
///
/// `data` is `[num_heads * head_dim]`. `weight` is `[head_dim]`.
/// Each head's `head_dim` values are independently RMS-normalized
/// then multiplied by the weight.
fn apply_per_head_rms_norm(data: &mut [f16], weight: &[f16], head_dim: usize) {
    let eps = 1e-6f32;
    let num_heads = data.len() / head_dim;
    for h in 0..num_heads {
        let base = h * head_dim;
        let n = head_dim.min(weight.len());
        let mut sum_sq = 0.0f32;
        for i in 0..n {
            let v = data[base + i].to_f32();
            sum_sq += v * v;
        }
        let rms = (sum_sq / n as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        for i in 0..n {
            let normed = data[base + i].to_f32() * inv_rms * weight[i].to_f32();
            data[base + i] = f16::from_f32(normed);
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

// ---------------------------------------------------------------------------
// Bundle loading helpers
// ---------------------------------------------------------------------------

/// Compile a sub-program from bundle artifacts (MIL text + weight blob).
///
/// Reads the MIL text from `programs/<name>.mil` and the combined weight
/// BLOBFILE from `weights/<name>.bin`, extracts individual weight entries,
/// and compiles via `device.compile()`.
fn compile_sub_from_bundle<D: AneDevice>(
    manifest: &super::bundle_manifest::SubProgramManifest,
    programs_dir: &std::path::Path,
    weights_dir: &std::path::Path,
    device: &D,
    min_output_alloc: usize,
    qos: u32,
) -> Result<LoadedSubProgram<D>> {
    let mil_text = std::fs::read_to_string(programs_dir.join(format!("{}.mil", manifest.name)))
        .map_err(|e| {
            AneError::Other(anyhow::anyhow!(
                "failed to read MIL text for {}: {e}",
                manifest.name
            ))
        })?;

    let weight_blob =
        std::fs::read(weights_dir.join(format!("{}.bin", manifest.name))).unwrap_or_default();

    let weight_entries =
        super::bundle_manifest::extract_weight_entries_from_bundle(&mil_text, &weight_blob);
    let weight_slices: Vec<(&str, &[u8])> = weight_entries
        .iter()
        .map(|(path, data)| (path.as_str(), data.as_slice()))
        .collect();

    let compiled = device.compile(&mil_text, &weight_slices, qos)?;

    allocate_io_from_manifest(manifest, compiled, min_output_alloc)
}

/// Compile a sub-program from bundle artifacts using donor/patch optimization.
fn compile_sub_from_bundle_with_donor<D: AneDevice>(
    manifest: &super::bundle_manifest::SubProgramManifest,
    donor: &D::Program,
    programs_dir: &std::path::Path,
    weights_dir: &std::path::Path,
    device: &D,
    min_output_alloc: usize,
    qos: u32,
) -> Result<LoadedSubProgram<D>> {
    let mil_text = std::fs::read_to_string(programs_dir.join(format!("{}.mil", manifest.name)))
        .map_err(|e| {
            AneError::Other(anyhow::anyhow!(
                "failed to read MIL text for {}: {e}",
                manifest.name
            ))
        })?;

    let weight_blob =
        std::fs::read(weights_dir.join(format!("{}.bin", manifest.name))).unwrap_or_default();

    let weight_entries =
        super::bundle_manifest::extract_weight_entries_from_bundle(&mil_text, &weight_blob);
    let weight_slices: Vec<(&str, &[u8])> = weight_entries
        .iter()
        .map(|(path, data)| (path.as_str(), data.as_slice()))
        .collect();

    let compiled = device.compile_patched(donor, &mil_text, &weight_slices, qos)?;

    allocate_io_from_manifest(manifest, compiled, min_output_alloc)
}

/// Allocate I/O tensors based on manifest descriptors and build a
/// `LoadedSubProgram`.
fn allocate_io_from_manifest<D: AneDevice>(
    manifest: &super::bundle_manifest::SubProgramManifest,
    program: D::Program,
    min_output_alloc: usize,
) -> Result<LoadedSubProgram<D>> {
    let input_shapes: Vec<_> = manifest
        .inputs
        .iter()
        .map(|td| (td.shape, td.scalar_type()))
        .collect();
    let output_shapes: Vec<_> = manifest
        .outputs
        .iter()
        .map(|td| (td.shape, td.scalar_type()))
        .collect();

    let input_alloc = uniform_alloc_size(&input_shapes);
    let output_alloc = uniform_alloc_size(&output_shapes).max(min_output_alloc);

    let input_tensors: Vec<AneTensor> = manifest
        .inputs
        .iter()
        .map(|td| {
            AneTensor::new_with_min_alloc(td.shape[1], td.shape[3], td.scalar_type(), input_alloc)
                .map_err(Into::into)
        })
        .collect::<Result<Vec<_>>>()?;

    let output_tensors: Vec<AneTensor> = manifest
        .outputs
        .iter()
        .map(|td| {
            AneTensor::new_with_min_alloc(td.shape[1], td.shape[3], td.scalar_type(), output_alloc)
                .map_err(Into::into)
        })
        .collect::<Result<Vec<_>>>()?;

    let input_packing = manifest
        .input_packing
        .clone()
        .map(super::bundle_manifest::InputPacking::from);

    Ok(LoadedSubProgram {
        program,
        input_tensors,
        output_tensors,
        input_packing,
    })
}

/// CPU RMSNorm: normalize hidden state in-place using the given weight.
/// RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
fn cpu_rms_norm(hidden: &mut [f16], weight: &[f16]) {
    let eps = 1e-6f32;
    let n = hidden.len().min(weight.len());

    // Compute RMS in f32 for numerical stability.
    let mut sum_sq = 0.0f32;
    for &h in hidden.iter().take(n) {
        let v = h.to_f32();
        sum_sq += v * v;
    }
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    for i in 0..n {
        let normed = hidden[i].to_f32() * inv_rms * weight[i].to_f32();
        hidden[i] = f16::from_f32(normed);
    }
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
// Sampling functions are in crate::sampling
use crate::sampling::{DEFAULT_EOS_TOKENS, is_eos_token, sample_token};
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
