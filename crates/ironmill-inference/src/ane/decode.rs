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
use super::model::AneConfig;
use super::turboquant::TurboQuantConfig;
use super::turboquant::TurboQuantModel;
use super::turboquant::mil_emitter;
use super::turboquant::mil_emitter::MIN_IO_SEQ;
use crate::ane::{AneError, Result};
use crate::engine::{InferenceEngine, InferenceError};
use crate::types::Logits;
use ironmill_core::ane::mil_text::{MilTextConfig, program_to_mil_text};
use ironmill_core::model_info::ModelInfo;
use mil_rs::ir::ScalarType;
use mil_rs::weights::Architecture;
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
    /// ANE hardware profiling enabled. When `true`, `eval_with_stats()`
    /// is used and per-layer HW timing is printed after each decode.
    enable_profiling: bool,
    /// Chained (pipelined) execution enabled. When `true`, the decode
    /// loop attempts to dispatch all ANE programs via `ChainingRequest`
    /// to eliminate per-layer CPU↔ANE roundtrips. Falls back to
    /// per-layer eval on failure.
    enable_chaining: bool,
    /// Program fusion enabled. When `true` and a fused FFN program is
    /// available in TurboQuant, `step_ffn()` is used instead of the
    /// bundle's `post_attn` program. Falls back to `post_attn` when the
    /// fused FFN program is not compiled or not available.
    ///
    /// Cache-write K/V fusion is always active (the cache-write program
    /// inherently fuses both K and V quantization into one dispatch).
    enable_fusion: bool,
    /// Hybrid ANE↔GPU execution enabled. When `true`, the decode loop
    /// attempts to coordinate GPU projections and ANE attention via
    /// `MTLSharedEvent`-backed fences, removing the CPU from the
    /// critical path. Falls back to standard per-layer eval on failure.
    /// **Highly experimental.**
    enable_hybrid: bool,
    /// Pre-allocated scratch buffers for the decode loop.
    scratch: ScratchBuffers,
    /// Model metadata for the InferenceEngine trait.
    model_info: ModelInfo,
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

/// Pre-allocated scratch buffers to eliminate per-token heap allocations.
///
/// Sized at model construction time based on max tensor dimensions.
/// Reused across `decode()` calls via `std::mem::take` / put-back pattern.
struct ScratchBuffers {
    /// Scratch for `write_f16_padded()` zero-padded writes.
    padded: Vec<f16>,
    /// Scratch for post-attn packed residual writes.
    packed_residual: Vec<f16>,
    /// Pre-allocated zero buffer for cache/mask reset.
    zeros: Vec<f16>,
    /// Pre-allocated -inf buffer for mask reset.
    neg_inf_mask: Vec<f16>,
    /// Reusable hidden state / embedding buffer.
    hidden: Vec<f16>,
    /// Reusable attention output buffer.
    attn_out: Vec<f16>,
    /// Reusable Q projection buffer (FP16 cache path).
    q_data: Vec<f16>,
    /// Reusable K projection buffer (FP16 cache path).
    k_data: Vec<f16>,
    /// Reusable V projection buffer (FP16 cache path).
    v_data: Vec<f16>,
    /// Reusable residual read buffer (packed post-attn path).
    residual: Vec<f16>,
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
    /// Scratch for `write_f16_padded` in `forward_inner`.
    padded_scratch: Vec<f16>,
    /// Scratch for `read_f16_channels` chunk reads in `forward_inner`.
    chunk_read_buf: Vec<f16>,
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
            ])?;

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

        // Compute scratch sizes from the chunks.
        let max_padded = hidden_size * LM_HEAD_MIN_SEQ;
        let max_chunk_ch = chunks.iter().map(|c| c.out_channels).max().unwrap_or(0);

        Ok(Self {
            chunks,
            vocab_size,
            qos,
            device,
            padded_scratch: vec![f16::ZERO; max_padded],
            chunk_read_buf: Vec::with_capacity(max_chunk_ch),
        })
    }

    /// Run the lm_head projection on ANE, returning logits as f32.
    fn forward(&mut self, hidden: &[f16]) -> Result<Vec<f32>> {
        self.forward_inner(hidden, false, &mut 0)
    }

    /// Like [`forward`] but optionally collects hardware timing.
    fn forward_profiled(&mut self, hidden: &[f16], hw_ns: &mut u64) -> Result<Vec<f32>> {
        self.forward_inner(hidden, true, hw_ns)
    }

    fn forward_inner(
        &mut self,
        hidden: &[f16],
        enable_profiling: bool,
        hw_ns: &mut u64,
    ) -> Result<Vec<f32>> {
        let mut logits = Vec::with_capacity(self.vocab_size);
        // Take scratch buffers out of self to avoid borrow conflicts with self.chunks.
        let mut padded = std::mem::take(&mut self.padded_scratch);
        let mut chunk_buf = std::mem::take(&mut self.chunk_read_buf);

        for chunk in &mut self.chunks {
            // Write hidden state to input tensor (column 0, zero-padded).
            write_f16_padded(&mut chunk.input_tensor, hidden, &mut padded)?;

            // Eval conv1×1 on ANE (program stays loaded).
            ane_eval(
                &*self.device,
                &chunk.program,
                &[&chunk.input_tensor],
                &mut [&mut chunk.output_tensor],
                self.qos,
                enable_profiling,
                hw_ns,
            )?;

            // Read output logits from column 0 of each channel.
            read_f16_channels(&chunk.output_tensor, &mut chunk_buf)?;
            logits.extend(
                chunk_buf
                    .iter()
                    .take(chunk.out_channels)
                    .map(|v| v.to_f32()),
            );
        }

        // Put scratch buffers back for reuse on next call.
        self.padded_scratch = padded;
        self.chunk_read_buf = chunk_buf;

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

/// The default perf_stats_mask used for hardware profiling (all bits set).
const PROFILE_PERF_MASK: u32 = 0xFFFFFFFF;

/// Run an ANE eval, optionally collecting hardware performance stats.
///
/// When `enable_profiling` is `false` this is a plain `device.eval()` call
/// with zero extra overhead (a single predictable branch).
fn ane_eval<D: AneDevice>(
    device: &D,
    program: &D::Program,
    inputs: &[&AneTensor],
    outputs: &mut [&mut AneTensor],
    qos: u32,
    enable_profiling: bool,
    hw_ns_accum: &mut u64,
) -> Result<()> {
    if enable_profiling {
        if let Some(stats) =
            device.eval_with_stats(program, inputs, outputs, qos, PROFILE_PERF_MASK)?
        {
            *hw_ns_accum += stats.hw_execution_time_ns;
        }
    } else {
        device.eval(program, inputs, outputs, qos)?;
    }
    Ok(())
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
            .map_err(|e| AneError::ManifestError(format!("failed to read manifest: {e}")))?;
        let manifest: BundleManifest = serde_json::from_str(&manifest_json)
            .map_err(|e| AneError::ManifestError(format!("invalid manifest: {e}")))?;
        let decode = manifest
            .decode
            .ok_or_else(|| AneError::ManifestError("not a decode bundle".into()))?;
        let arch = &decode.architecture;

        let programs_dir = bundle_path.join("programs");
        let weights_dir = bundle_path.join("weights");
        let cpu_weights_dir = bundle_path.join("cpu_weights");

        // 2. Load CPU weights and caches.
        let (
            embed_weight,
            lm_head,
            final_norm_weight,
            rope_cos_cache,
            rope_sin_cache,
            rope_cache_dim,
        ) = Self::load_cpu_weights_and_caches(&cpu_weights_dir, arch, Arc::clone(&device), qos)?;

        // 3. Compile per-layer sub-programs from bundle.
        let num_layers = decode.layers.len();
        let (layers, cache_write_fused) = Self::compile_layer_programs(
            &decode.layers,
            &programs_dir,
            &weights_dir,
            &*device,
            turbo_config.as_ref(),
            num_layers,
            qos,
        )?;

        // 4. Set up TurboQuant or FP16 caches.
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
        ) = Self::setup_execution_mode(Arc::clone(&device), turbo_config, arch, num_layers, qos)?;

        // 5. Load per-layer QK norm weights.
        let qk_norm_weights =
            Self::load_qk_norm_weights(&cpu_weights_dir, arch, head_dim, num_layers)?;

        // 6. Size and allocate scratch buffers.
        let scratch = Self::compute_scratch_buffers(
            &layers,
            fp16_attn_q_staging.as_ref(),
            fp16_attn_out_staging.as_ref(),
            num_kv_heads,
            head_dim,
            max_seq_len,
            arch.hidden_size,
        );

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
            enable_profiling: false,
            enable_chaining: false,
            enable_fusion: false,
            enable_hybrid: false,
            scratch,
            model_info: ModelInfo {
                architecture: Architecture::Llama,
                num_layers,
                hidden_size: arch.hidden_size,
                vocab_size: arch.vocab_size,
                max_context_len: arch.max_seq_len,
                weight_quantization: String::from("fp16"),
                eos_tokens: arch.eos_tokens.clone(),
                param_count_m: 0.0, // not available from bundle manifest
                uses_gqa: arch.num_kv_heads < arch.num_heads,
                uses_mla: false,
                head_dim: arch.head_dim,
                num_attention_heads: arch.num_heads,
                num_kv_heads: arch.num_kv_heads,
            },
        })
    }

    /// Load embedding, lm_head, final_norm, and RoPE caches from the CPU weights directory.
    fn load_cpu_weights_and_caches(
        cpu_weights_dir: &std::path::Path,
        arch: &super::bundle_manifest::ArchConfig,
        device: Arc<D>,
        qos: u32,
    ) -> Result<(
        CpuWeight,
        LmHead<D>,
        Option<Vec<f16>>,
        Vec<f16>,
        Vec<f16>,
        usize,
    )> {
        let embed_data = std::fs::read(cpu_weights_dir.join("embedding.bin"))
            .map_err(|e| AneError::IoError(format!("failed to read embedding weights: {e}")))?;
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
            .map_err(|e| AneError::IoError(format!("failed to read rope_cos: {e}")))?;
        let rope_sin_data = std::fs::read(cpu_weights_dir.join("rope_sin.bin"))
            .map_err(|e| AneError::IoError(format!("failed to read rope_sin: {e}")))?;
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

        // Compile LM head (try ANE, fall back to CPU).
        let lm_head = match AneLmHead::compile(device, &lm_head_cpu_weight, qos) {
            Ok(ane_lm_head) => LmHead::Ane(ane_lm_head),
            Err(e) => {
                eprintln!("warning: ANE lm_head compilation failed: {e}");
                LmHead::Cpu(lm_head_cpu_weight)
            }
        };

        Ok((
            embed_weight,
            lm_head,
            final_norm_weight,
            rope_cos_cache,
            rope_sin_cache,
            rope_cache_dim,
        ))
    }

    /// Compile per-layer sub-programs (pre_attn, post_attn, fp16_attn) from
    /// the bundle, using donor-patching where possible for layers 1+.
    fn compile_layer_programs(
        layer_manifests: &[super::bundle_manifest::LayerManifest],
        programs_dir: &std::path::Path,
        weights_dir: &std::path::Path,
        device: &D,
        turbo_config: Option<&TurboQuantConfig>,
        num_layers: usize,
        qos: u32,
    ) -> Result<(Vec<LayerPrograms<D>>, bool)> {
        let pre_attn_min_output_alloc = if let Some(tc) = turbo_config {
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
            ])?
        } else {
            0
        };

        let cache_write_fused = layer_manifests.first().is_some_and(|l| l.cache_write_fused);

        let mut layers: Vec<LayerPrograms<D>> = Vec::with_capacity(num_layers);
        for (i, layer_manifest) in layer_manifests.iter().enumerate() {
            let pre = if i == 0 || !layer_manifest.donor_compatible {
                // First layer or non-donor-compatible: full compile.
                compile_sub_from_bundle(
                    &layer_manifest.pre_attn,
                    programs_dir,
                    weights_dir,
                    device,
                    pre_attn_min_output_alloc,
                    qos,
                )?
            } else {
                // Layers 1+: patch weights from layer-0 donor.
                match compile_sub_from_bundle_with_donor(
                    &layer_manifest.pre_attn,
                    &layers[0].pre_attn.program,
                    programs_dir,
                    weights_dir,
                    device,
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
                            programs_dir,
                            weights_dir,
                            device,
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
                        programs_dir,
                        weights_dir,
                        device,
                        0,
                        qos,
                    )?)
                } else if let Some(ref donor_post) = layers[0].post_attn {
                    match compile_sub_from_bundle_with_donor(
                        post_manifest,
                        &donor_post.program,
                        programs_dir,
                        weights_dir,
                        device,
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
                                programs_dir,
                                weights_dir,
                                device,
                                0,
                                qos,
                            )?)
                        }
                    }
                } else {
                    Some(compile_sub_from_bundle(
                        post_manifest,
                        programs_dir,
                        weights_dir,
                        device,
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
                    programs_dir,
                    weights_dir,
                    device,
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

        Ok((layers, cache_write_fused))
    }

    /// Set up TurboQuant or FP16 attention caches and compile the shared
    /// FP16 attention program when needed.
    fn setup_execution_mode(
        device: Arc<D>,
        turbo_config: Option<TurboQuantConfig>,
        arch: &super::bundle_manifest::ArchConfig,
        num_layers: usize,
        qos: u32,
    ) -> Result<(
        Option<TurboQuantModel<D>>,
        Option<Vec<(AneTensor, AneTensor)>>,
        Option<D::Program>,
        Option<AneTensor>,
        Option<AneTensor>,
        Option<AneTensor>,
        usize,
        usize,
        usize,
    )> {
        if let Some(tq_config) = turbo_config {
            let nkv = tq_config.num_kv_heads;
            let hd = tq_config.head_dim;
            let msl = tq_config.max_seq_len;
            let tq = TurboQuantModel::compile(Arc::clone(&device), tq_config)?;
            Ok((Some(tq), None, None, None, None, None, nkv, hd, msl))
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
                AneError::Validation(format!("FP16 attention MIL text failed: {e}"))
            })?;
            let attn_compiled = device.compile(&mil, &[], qos)?;

            let attn_alloc = uniform_alloc_size(&[
                ([1, q_channels, 1, MIN_IO_SEQ], ScalarType::Float16),
                ([1, kv_channels, 1, msl], ScalarType::Float16),
                ([1, kv_channels, 1, msl], ScalarType::Float16),
                ([1, 1, 1, msl], ScalarType::Float16),
            ])?;

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

            Ok((
                None,
                Some(caches),
                Some(attn_compiled),
                Some(q_staging),
                Some(out_staging),
                Some(mask_tensor),
                nkv,
                hd,
                msl,
            ))
        }
    }

    /// Load and validate per-layer QK normalization weights when the model
    /// architecture requires them.
    fn load_qk_norm_weights(
        cpu_weights_dir: &std::path::Path,
        arch: &super::bundle_manifest::ArchConfig,
        head_dim: usize,
        num_layers: usize,
    ) -> Result<Option<Vec<(Vec<f16>, Vec<f16>)>>> {
        if !arch.qk_norm {
            return Ok(None);
        }

        let q_path = cpu_weights_dir.join("qk_norm_q.bin");
        let k_path = cpu_weights_dir.join("qk_norm_k.bin");
        match (std::fs::read(&q_path), std::fs::read(&k_path)) {
            (Ok(q_data), Ok(k_data)) => {
                let q_all: Vec<f16> = q_data
                    .chunks_exact(2)
                    .map(|c| f16::from_le_bytes([c[0], c[1]]))
                    .collect();
                let k_all: Vec<f16> = k_data
                    .chunks_exact(2)
                    .map(|c| f16::from_le_bytes([c[0], c[1]]))
                    .collect();
                let hd = head_dim;
                if hd == 0 || q_all.len() % hd != 0 || k_all.len() % hd != 0 {
                    return Err(AneError::Validation(format!(
                        "QK norm weight size not divisible by head_dim \
                         (q={}, k={}, head_dim={})",
                        q_all.len(),
                        k_all.len(),
                        hd,
                    )));
                }
                if q_all.len() != k_all.len() {
                    return Err(AneError::Validation(format!(
                        "QK norm Q/K size mismatch: q={} vs k={} \
                         (expected equal lengths for {} layers × head_dim={})",
                        q_all.len(),
                        k_all.len(),
                        num_layers,
                        hd,
                    )));
                }
                let loaded_layers = q_all.len() / hd;
                if loaded_layers != num_layers {
                    return Err(AneError::Validation(format!(
                        "QK norm layer count mismatch: weights have {} layers \
                         but model has {} layers",
                        loaded_layers, num_layers,
                    )));
                }
                let mut norms = Vec::with_capacity(loaded_layers);
                for i in 0..loaded_layers {
                    let q_w = q_all[i * hd..(i + 1) * hd].to_vec();
                    let k_w = k_all[i * hd..(i + 1) * hd].to_vec();
                    norms.push((q_w, k_w));
                }
                Ok(Some(norms))
            }
            _ => Err(AneError::ManifestError(
                "model requires QK norm but qk_norm_q.bin / qk_norm_k.bin \
                 not found in bundle"
                    .into(),
            )),
        }
    }

    /// Compute scratch buffer sizes by scanning all layer tensors and allocate
    /// the reusable scratch vectors.
    fn compute_scratch_buffers(
        layers: &[LayerPrograms<D>],
        fp16_attn_q_staging: Option<&AneTensor>,
        fp16_attn_out_staging: Option<&AneTensor>,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        hidden_size: usize,
    ) -> ScratchBuffers {
        let mut max_padded = 0usize;
        let mut max_out_channels = 0usize;
        let mut max_packed_residual = 0usize;
        for layer in layers {
            for t in &layer.pre_attn.input_tensors {
                max_padded = max_padded.max(t.shape()[1] * t.shape()[3]);
            }
            for t in &layer.pre_attn.output_tensors {
                max_out_channels = max_out_channels.max(t.shape()[1]);
            }
            if let Some(ref post) = layer.post_attn {
                for t in &post.input_tensors {
                    let sz = t.shape()[1] * t.shape()[3];
                    max_padded = max_padded.max(sz);
                    max_packed_residual = max_packed_residual.max(sz);
                }
                for t in &post.output_tensors {
                    max_out_channels = max_out_channels.max(t.shape()[1]);
                }
            }
            if let Some(ref attn) = layer.fp16_attn {
                for t in &attn.input_tensors {
                    max_padded = max_padded.max(t.shape()[1] * t.shape()[3]);
                }
                for t in &attn.output_tensors {
                    max_out_channels = max_out_channels.max(t.shape()[1]);
                }
            }
        }
        // Also account for FP16 attention staging tensors.
        if let Some(qs) = fp16_attn_q_staging {
            max_padded = max_padded.max(qs.shape()[1] * qs.shape()[3]);
        }
        if let Some(os) = fp16_attn_out_staging {
            max_out_channels = max_out_channels.max(os.shape()[1]);
        }

        let kv_cache_size = num_kv_heads * head_dim * max_seq_len;

        ScratchBuffers {
            padded: vec![f16::ZERO; max_padded],
            packed_residual: vec![f16::ZERO; max_packed_residual],
            zeros: vec![f16::ZERO; kv_cache_size.max(max_seq_len)],
            neg_inf_mask: vec![f16::NEG_INFINITY; max_seq_len],
            hidden: Vec::with_capacity(hidden_size),
            attn_out: Vec::with_capacity(max_out_channels),
            q_data: Vec::with_capacity(max_out_channels),
            k_data: Vec::with_capacity(max_out_channels),
            v_data: Vec::with_capacity(max_out_channels),
            residual: Vec::with_capacity(max_out_channels),
        }
    }

    /// Enable or disable ANE hardware profiling.
    ///
    /// When enabled, each `decode()` call collects per-layer HW execution
    /// time from the ANE and prints a profiling summary.
    /// When disabled (the default), the normal eval path is used with
    /// zero overhead.
    pub fn set_profiling(&mut self, enable: bool) {
        self.enable_profiling = enable;
    }

    /// Enable or disable chained (pipelined) ANE execution.
    ///
    /// When enabled, the decode loop attempts to dispatch layer programs
    /// via `ChainingRequest` to eliminate per-layer CPU↔ANE roundtrips.
    /// Falls back to per-layer eval if chaining fails.
    /// Disabled by default. **Experimental.**
    pub fn set_chaining(&mut self, enable: bool) {
        self.enable_chaining = enable;
    }

    /// Enable or disable MIL program fusion.
    ///
    /// When enabled and a fused FFN program is available in TurboQuant,
    /// `step_ffn()` is used instead of the bundle's `post_attn` eval.
    /// Falls back to `post_attn` when the fused FFN is not compiled.
    ///
    /// Cache-write K/V fusion is always active regardless of this flag
    /// (the cache-write program inherently fuses both K and V).
    pub fn set_fusion(&mut self, enable: bool) {
        self.enable_fusion = enable;
    }

    /// Enable or disable experimental hybrid ANE↔GPU execution.
    ///
    /// When enabled, the decode loop attempts to use `MTLSharedEvent`-backed
    /// fences to coordinate GPU Q/K/V projections with ANE attention,
    /// removing the CPU from the critical synchronization path.
    ///
    /// **Prerequisites:**
    /// - Shared event support must be verified by probe 15
    ///   (`probe_shared_events` in `ironmill-ane-sys`).
    /// - Metal GPU infrastructure must be available.
    ///
    /// Falls back to standard per-layer eval if shared events are
    /// unavailable or the handoff fails. **Highly experimental.**
    pub fn set_hybrid(&mut self, enable: bool) {
        self.enable_hybrid = enable;
    }

    /// Apply runtime flags from an [`AneConfig`].
    ///
    /// Wires `enable_profiling`, `enable_chaining`, `enable_fusion`, and
    /// `enable_hybrid` from the config into this inference instance.
    /// Call after [`from_bundle`](Self::from_bundle) to activate the
    /// settings the user requested.
    pub fn configure(&mut self, config: &AneConfig) {
        self.enable_profiling = config.enable_profiling;
        self.enable_chaining = config.enable_chaining;
        self.enable_fusion = config.enable_fusion;
        self.enable_hybrid = config.enable_hybrid;
    }

    /// Attempt a hybrid ANE↔GPU decode step for a single layer.
    ///
    /// The intended pipeline:
    ///   1. GPU computes Q/K/V projections → signals shared event at value N
    ///   2. ANE waits for value N → runs attention → signals at value N+1
    ///   3. GPU waits for value N+1 → continues with post-attention
    ///
    /// This eliminates the CPU from the GPU→ANE→GPU synchronization path.
    ///
    /// # Current status
    ///
    /// This is a placeholder that logs the attempt and returns `Err` to
    /// trigger the fallback path. The actual Metal↔ANE handoff requires:
    /// - A compiled Metal compute pipeline for Q/K/V projections
    /// - An `MTLSharedEvent` bridged to `SharedSignalEvent`/`SharedWaitEvent`
    /// - An `AneRequest::with_shared_events()` submission
    ///
    /// These components depend on probe 15 results confirming that
    /// `MTLSharedEvent` is accepted by the ANE shared event API.
    fn try_hybrid_layer(&mut self, _hidden: &[f16], _layer_idx: usize) -> Result<Vec<f16>> {
        // TODO: Implement Metal↔ANE shared-event handoff.
        //
        // Required components (not yet available):
        //   1. A compiled Metal compute pipeline for Q/K/V projections
        //   2. An `MTLSharedEvent` bridged to `SharedSignalEvent`/`SharedWaitEvent`
        //   3. An `AneRequest::with_shared_events()` submission
        //
        // Sketch:
        //   let shared_event = self.hybrid_shared_event()?;
        //   let signal_val = (self.seq_pos as u64) * 2;
        //   self.gpu_encode_qkv_projection(hidden, layer_idx, shared_event, signal_val)?;
        //   let wait = SharedWaitEvent::new(signal_val, shared_event)?;
        //   let signal = SharedSignalEvent::new(signal_val + 1, 0, 0, shared_event)?;
        //   let events = SharedEvents::new(signal_array, wait_array)?;
        //   let req = AneRequest::with_shared_events(..., events.as_raw())?;
        //   self.gpu_encode_post_attn(layer_idx, shared_event, signal_val + 1)?;
        //
        // Blocked on: probe 15 validation of MTLSharedEvent↔ANE interop.
        Err(AneError::Other(anyhow::anyhow!(
            "hybrid ANE↔GPU execution not yet implemented — \
             requires MTLSharedEvent↔ANE interop (probe 15) and \
             compiled Metal Q/K/V projection pipelines"
        )))
    }

    // -----------------------------------------------------------------------
    // Chained execution (experimental)
    // -----------------------------------------------------------------------

    /// Attempt chained (pipelined) execution of all layer sub-programs.
    ///
    /// When successful, this dispatches all pre_attn and post_attn programs
    /// as a single chained ANE request, eliminating per-layer CPU↔ANE
    /// roundtrips. Returns the final hidden state on success.
    ///
    /// # Limitations
    ///
    /// Full end-to-end chaining is not yet possible because the attention
    /// step between pre_attn and post_attn requires CPU-managed state
    /// (KV cache, RoPE, etc.). This method currently chains only the
    /// pre_attn programs across layers as a proof-of-concept. The
    /// attention and post_attn steps still execute per-layer.
    ///
    /// The `_ANEChainingRequest` API is undocumented — this is a best-effort
    /// implementation that may fail on some hardware or macOS versions.
    fn try_chained_pre_attn(&mut self, _hidden: &[f16], _effective_layers: usize) -> Result<()> {
        // TODO: Implement within-layer program chaining.
        //
        // Cross-layer pre_attn chaining is architecturally unsound: each
        // layer's pre_attn input depends on the previous layer's complete
        // output (attention + FFN + both residuals). Pre_attn programs
        // cannot be chained across layers.
        //
        // Viable chaining targets:
        //   - Within-layer: pre_attn → attention → post_attn for one layer
        //   - Across-layer with full pipeline: all sub-programs for each layer
        //
        // The _ANEChainingRequest API is undocumented. Probes 13 & 14
        // (ane_probe.rs) explore the API mechanics. Once validated, rewrite
        // this to chain within-layer programs instead.
        Err(AneError::Other(anyhow::anyhow!(
            "cross-layer pre_attn chaining not yet implemented — \
             requires within-layer chaining strategy and validated \
             _ANEChainingRequest API (probes 13 & 14)"
        )))
    }

    // -----------------------------------------------------------------------
    // Decode helpers
    // -----------------------------------------------------------------------

    /// Execute the pre-attention sub-program (norm → Q/K/V projection) for one layer.
    fn run_pre_attn_layer(
        &mut self,
        layer_idx: usize,
        hidden: &[f16],
        hw_profiling: bool,
        hw_pre_attn_ns: &mut u64,
    ) -> Result<()> {
        let layer = &mut self.layers[layer_idx];
        write_f16_padded(
            &mut layer.pre_attn.input_tensors[0],
            hidden,
            &mut self.scratch.padded,
        )?;
        {
            let in_refs: Vec<&AneTensor> = layer.pre_attn.input_tensors.iter().collect();
            let mut out_refs: Vec<&mut AneTensor> =
                layer.pre_attn.output_tensors.iter_mut().collect();
            ane_eval(
                &*self.device,
                &layer.pre_attn.program,
                &in_refs,
                &mut out_refs,
                self.qos,
                hw_profiling,
                hw_pre_attn_ns,
            )
            .map_err(|e| {
                AneError::Validation(format!("layer {layer_idx} pre_attn eval failed: {e}"))
            })?;
        }
        Ok(())
    }

    /// Execute the attention step for one layer.
    ///
    /// Dispatches to the appropriate attention backend:
    /// - TurboQuant INT8 attention (when `self.turboquant` is configured)
    /// - Per-layer FP16 attention sub-program (when `layer.fp16_attn` exists)
    /// - Shared FP16 attention with CPU-managed KV cache (fallback)
    ///
    /// Writes the attention output into `attn_out`. The `q_buf`/`k_buf`/`v_buf`
    /// scratch buffers are only used by the shared FP16 cache path.
    fn run_attention_layer(
        &mut self,
        layer_idx: usize,
        attn_out: &mut Vec<f16>,
        q_buf: &mut Vec<f16>,
        k_buf: &mut Vec<f16>,
        v_buf: &mut Vec<f16>,
        profiling: bool,
        hw_profiling: bool,
        hw_attn_ns: &mut u64,
        d_attn: &mut std::time::Duration,
        d_read_qkv: &mut std::time::Duration,
    ) -> Result<()> {
        let layer = &mut self.layers[layer_idx];
        let num_pre_outputs = layer.pre_attn.output_tensors.len();

        if let Some(tq) = &mut self.turboquant {
            // TurboQuant INT8 attention path.
            let t0 = if profiling {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let q = &layer.pre_attn.output_tensors[0];
            let attn_tensor = if self.cache_write_fused {
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
                let k_original = if tq.config().enable_qjl && num_pre_outputs > 3 {
                    Some(&layer.pre_attn.output_tensors[3])
                } else {
                    None
                };
                tq.step_attention_fused(layer_idx, q, k_quant, v_quant, k_original)
            } else {
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
                AneError::Validation(format!(
                    "layer {layer_idx} turboquant attention failed: {e}"
                ))
            })?;
            read_f16_channels(&attn_tensor, attn_out)?;
            if let Some(t) = t0 {
                *d_attn += t.elapsed();
            }
        } else if let Some(ref mut attn) = layer.fp16_attn {
            // Per-layer FP16 attention sub-program path.
            let t0 = if profiling {
                Some(std::time::Instant::now())
            } else {
                None
            };

            if let Some(ref map) = layer.attn_input_map {
                layer.pre_attn.output_tensors[map.q_idx].read_f16_into(&mut self.scratch.q_data)?;
                attn.input_tensors[map.q_idx].write_f16(&self.scratch.q_data)?;

                layer.pre_attn.output_tensors[map.k_idx].read_f16_into(&mut self.scratch.k_data)?;
                attn.input_tensors[map.k_idx].write_f16(&self.scratch.k_data)?;

                layer.pre_attn.output_tensors[map.v_idx].read_f16_into(&mut self.scratch.v_data)?;
                attn.input_tensors[map.v_idx].write_f16(&self.scratch.v_data)?;

                for &cos_idx in &map.rope_cos_indices {
                    let cos_slice = &self.rope_cos_cache[self.seq_pos * self.rope_cache_dim
                        ..(self.seq_pos + 1) * self.rope_cache_dim];
                    write_f16_padded(
                        &mut attn.input_tensors[cos_idx],
                        cos_slice,
                        &mut self.scratch.padded,
                    )?;
                }
                for &sin_idx in &map.rope_sin_indices {
                    let sin_slice = &self.rope_sin_cache[self.seq_pos * self.rope_cache_dim
                        ..(self.seq_pos + 1) * self.rope_cache_dim];
                    write_f16_padded(
                        &mut attn.input_tensors[sin_idx],
                        sin_slice,
                        &mut self.scratch.padded,
                    )?;
                }
            } else {
                for (i, src) in layer.pre_attn.output_tensors.iter().enumerate() {
                    if i < attn.input_tensors.len() {
                        src.read_f16_into(&mut self.scratch.q_data)?;
                        attn.input_tensors[i].write_f16(&self.scratch.q_data)?;
                    }
                }
            }

            {
                let in_refs: Vec<&AneTensor> = attn.input_tensors.iter().collect();
                let mut out_refs: Vec<&mut AneTensor> = attn.output_tensors.iter_mut().collect();
                ane_eval(
                    &*self.device,
                    &attn.program,
                    &in_refs,
                    &mut out_refs,
                    self.qos,
                    hw_profiling,
                    hw_attn_ns,
                )
                .map_err(|e| {
                    AneError::Validation(format!("layer {layer_idx} fp16_attn eval failed: {e}"))
                })?;
            }

            read_f16_channels(&attn.output_tensors[0], attn_out)?;
            if let Some(t) = t0 {
                *d_attn += t.elapsed();
            }
        } else if let Some(ref mut caches) = self.fp16_kv_caches {
            // Shared FP16 attention with CPU-managed KV cache.
            let t0 = if profiling {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let _real_q_ch = (self.num_kv_heads * self.head_dim).max(self.head_dim);
            let gqa = if self.num_kv_heads > 0 {
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

            if num_pre_outputs > 1 {
                read_f16_channels(&layer.pre_attn.output_tensors[1], q_buf)?;
            } else {
                read_f16_channels(&layer.pre_attn.output_tensors[0], q_buf)?;
            }
            read_f16_channels(&layer.pre_attn.output_tensors[0], k_buf)?;
            if num_pre_outputs > 2 {
                read_f16_channels(&layer.pre_attn.output_tensors[2], v_buf)?;
            } else {
                v_buf.clear();
                v_buf.extend_from_slice(k_buf);
            }
            q_buf.truncate(real_q_ch);
            k_buf.truncate(real_kv_ch);
            v_buf.truncate(real_kv_ch);

            if let Some(ref norms) = self.qk_norm_weights {
                if let Some((q_norm_w, k_norm_w)) = norms.get(layer_idx) {
                    apply_per_head_rms_norm(q_buf, q_norm_w, self.head_dim);
                    apply_per_head_rms_norm(k_buf, k_norm_w, self.head_dim);
                }
            }

            if self.rope_cache_dim > 0 {
                let pos = self.seq_pos;
                let cos_start = pos * self.rope_cache_dim;
                let cos_end = cos_start + self.rope_cache_dim;
                let cos = &self.rope_cos_cache[cos_start..cos_end];
                let sin = &self.rope_sin_cache[cos_start..cos_end];
                apply_rope_rotation(q_buf, cos, sin, self.head_dim);
                apply_rope_rotation(k_buf, cos, sin, self.head_dim);
            }
            if let Some(t) = t0 {
                *d_read_qkv += t.elapsed();
            }

            let t0 = if profiling {
                Some(std::time::Instant::now())
            } else {
                None
            };
            caches[layer_idx]
                .0
                .write_column_f16(self.seq_pos, k_buf, self.max_seq_len)?;
            caches[layer_idx]
                .1
                .write_column_f16(self.seq_pos, v_buf, self.max_seq_len)?;

            if let Some(ref fp16_program) = self.fp16_attn_program {
                let q_staging = self.fp16_attn_q_staging.as_mut().ok_or_else(|| {
                    AneError::Other(anyhow::anyhow!("fp16_attn_q_staging not initialized"))
                })?;
                let out_staging = self.fp16_attn_out_staging.as_mut().ok_or_else(|| {
                    AneError::Other(anyhow::anyhow!("fp16_attn_out_staging not initialized"))
                })?;
                let mask = self.fp16_attn_mask.as_ref().ok_or_else(|| {
                    AneError::Other(anyhow::anyhow!("fp16_attn_mask not initialized"))
                })?;
                write_f16_padded(q_staging, q_buf, &mut self.scratch.padded)?;
                let (k_cache, v_cache) = (&caches[layer_idx].0, &caches[layer_idx].1);
                ane_eval(
                    &*self.device,
                    fp16_program,
                    &[q_staging, k_cache, v_cache, mask],
                    &mut [out_staging],
                    self.qos,
                    hw_profiling,
                    hw_attn_ns,
                )
                .map_err(|e| {
                    AneError::Validation(format!("layer {layer_idx} fp16_attn eval failed: {e}"))
                })?;
                read_f16_channels(out_staging, attn_out)?;
            } else {
                std::mem::swap(attn_out, q_buf);
            }
            if let Some(t) = t0 {
                *d_attn += t.elapsed();
            }
        } else {
            return Err(AneError::Validation(
                "no attention backend configured".into(),
            ));
        }
        Ok(())
    }

    /// Apply final RMSNorm and project hidden state to vocabulary logits.
    fn decode_lm_head(
        &mut self,
        hidden: &mut Vec<f16>,
        hw_profiling: bool,
        hw_lm_head_ns: &mut u64,
    ) -> Result<Vec<f32>> {
        if let Some(norm_w) = &self.final_norm_weight {
            cpu_rms_norm(hidden, norm_w);
        }

        if self.seq_pos <= 3 && std::env::var("IRONMILL_TRACE_LAST").is_ok() {
            let f32s: Vec<f32> = hidden.iter().map(|x| x.to_f32()).collect();
            let absmax = f32s.iter().map(|x| x.abs()).fold(0f32, f32::max);
            eprintln!(
                "  [norm] absmax={:.4} first3=[{:.4},{:.4},{:.4}]",
                absmax, f32s[0], f32s[1], f32s[2]
            );
        }

        let logits = match &mut self.lm_head {
            LmHead::Ane(ane_lm_head) if hw_profiling => {
                ane_lm_head.forward_profiled(hidden, hw_lm_head_ns)
            }
            LmHead::Ane(ane_lm_head) => ane_lm_head.forward(hidden),
            LmHead::Cpu(weight) => cpu_lm_head_matmul(weight, hidden),
        };

        if self.seq_pos <= 3 && std::env::var("IRONMILL_TRACE_LAST").is_ok() {
            if let Ok(ref l) = logits {
                let max = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let argmax = l
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
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

        logits
    }

    /// Print profiling and HW profiling statistics for one decode step.
    fn decode_print_profiling(
        t_total: Option<std::time::Instant>,
        d_embed: Option<std::time::Duration>,
        d_pre_attn: std::time::Duration,
        d_read_qkv: std::time::Duration,
        d_attn: std::time::Duration,
        d_post_attn: std::time::Duration,
        d_lm_head: Option<std::time::Duration>,
        hw_profiling: bool,
        seq_pos: usize,
        hw_pre_attn_ns: u64,
        hw_attn_ns: u64,
        hw_post_attn_ns: u64,
        hw_lm_head_ns: u64,
    ) {
        let total = t_total.map(|t| t.elapsed()).unwrap_or_default();
        eprintln!(
            "[profile] decode token 0: embed={:.2}ms pre_attn={:.1}ms read_qkv={:.1}ms attn={:.1}ms post_attn={:.1}ms lm_head={:.1}ms total={:.1}ms",
            d_embed.unwrap_or_default().as_secs_f64() * 1000.0,
            d_pre_attn.as_secs_f64() * 1000.0,
            d_read_qkv.as_secs_f64() * 1000.0,
            d_attn.as_secs_f64() * 1000.0,
            d_post_attn.as_secs_f64() * 1000.0,
            d_lm_head.unwrap_or_default().as_secs_f64() * 1000.0,
            total.as_secs_f64() * 1000.0,
        );

        if hw_profiling {
            let hw_total = hw_pre_attn_ns + hw_attn_ns + hw_post_attn_ns + hw_lm_head_ns;
            eprintln!(
                "[hw_profile] token {}: pre_attn={:.1}µs attn={:.1}µs post_attn={:.1}µs lm_head={:.1}µs total={:.1}µs",
                seq_pos,
                hw_pre_attn_ns as f64 / 1000.0,
                hw_attn_ns as f64 / 1000.0,
                hw_post_attn_ns as f64 / 1000.0,
                hw_lm_head_ns as f64 / 1000.0,
                hw_total as f64 / 1000.0,
            );
        }
    }

    /// Execute the post-attention sub-program (O proj + residual + FFN) for one layer.
    fn run_post_attn_layer(
        &mut self,
        layer_idx: usize,
        attn_out_data: &[f16],
        hidden: &[f16],
        num_pre_outputs: usize,
        hw_profiling: bool,
        hw_post_attn_ns: &mut u64,
    ) -> Result<Vec<f16>> {
        let layer = &mut self.layers[layer_idx];
        if let Some(ref mut post_attn) = layer.post_attn {
            if let Some(ref packing) = post_attn.input_packing {
                // Packed: write all logical inputs into the single tensor.
                if packing.offsets.len() > 1 && num_pre_outputs > 3 {
                    let mut residual_buf = Vec::new();
                    read_f16_channels(&layer.pre_attn.output_tensors[3], &mut residual_buf)?;
                    let [_, channels, _, total_s] = post_attn.input_tensors[0].shape();
                    let packed = &mut self.scratch.packed_residual;
                    packed.clear();
                    packed.resize(channels * total_s, f16::ZERO);
                    let c0 = attn_out_data.len().min(channels);
                    for ch in 0..c0 {
                        packed[ch * total_s + packing.offsets[0]] = attn_out_data[ch];
                    }
                    let c1 = residual_buf.len().min(channels);
                    for ch in 0..c1 {
                        packed[ch * total_s + packing.offsets[1]] = residual_buf[ch];
                    }
                    post_attn.input_tensors[0].write_f16(packed)?;
                } else {
                    super::bundle_manifest::write_packed_inputs(
                        &mut post_attn.input_tensors[0],
                        &[attn_out_data],
                        packing,
                    )?;
                }
            } else {
                write_f16_padded(
                    &mut post_attn.input_tensors[0],
                    attn_out_data,
                    &mut self.scratch.padded,
                )?;
                if post_attn.input_tensors.len() > 1 {
                    write_f16_padded(
                        &mut post_attn.input_tensors[1],
                        hidden,
                        &mut self.scratch.padded,
                    )?;
                }
            }
            {
                let in_refs: Vec<&AneTensor> = post_attn.input_tensors.iter().collect();
                let mut out_refs: Vec<&mut AneTensor> =
                    post_attn.output_tensors.iter_mut().collect();
                ane_eval(
                    &*self.device,
                    &post_attn.program,
                    &in_refs,
                    &mut out_refs,
                    self.qos,
                    hw_profiling,
                    hw_post_attn_ns,
                )
                .map_err(|e| {
                    AneError::Validation(format!("layer {layer_idx} post_attn eval failed: {e}"))
                })?;
            }
            let mut new_hidden = Vec::new();
            read_f16_channels(&post_attn.output_tensors[0], &mut new_hidden)?;

            // Debug: trace last few layers for explosion diagnosis
            if self.seq_pos == 0 && layer_idx >= 24 && std::env::var("IRONMILL_TRACE_LAST").is_ok()
            {
                let f32s: Vec<f32> = new_hidden.iter().map(|x| x.to_f32()).collect();
                let absmax = f32s.iter().map(|x| x.abs()).fold(0f32, f32::max);
                let has_nan = f32s.iter().any(|x| x.is_nan());
                let has_inf = f32s.iter().any(|x| x.is_infinite());
                eprintln!(
                    "  [L{:2}] absmax={:.2} nan={} inf={} first3=[{:.4},{:.4},{:.4}]",
                    layer_idx, absmax, has_nan, has_inf, f32s[0], f32s[1], f32s[2]
                );
            }
            Ok(new_hidden)
        } else {
            Ok(attn_out_data.to_vec())
        }
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
        // Swap scratch buffers out of self for use as locals (avoids borrow
        // conflicts with other self fields in the decode loop).
        let mut hidden = std::mem::take(&mut self.scratch.hidden);
        cpu_embedding_lookup(&self.embed_weight, token_id, &mut hidden)?;
        let d_embed = t0.map(|t| t.elapsed());

        // Scratch buffers swapped out for the duration of decode().
        let mut attn_out = std::mem::take(&mut self.scratch.attn_out);
        let mut q_buf = std::mem::take(&mut self.scratch.q_data);
        let mut k_buf = std::mem::take(&mut self.scratch.k_data);
        let mut v_buf = std::mem::take(&mut self.scratch.v_data);
        let residual_buf = std::mem::take(&mut self.scratch.residual);

        // 2. Per-layer: pre_attn → attention → post_attn
        let num_layers = self.layers.len();

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

        // HW profiling accumulators (only non-zero when enable_profiling is true).
        let hw_profiling = self.enable_profiling;
        let mut hw_pre_attn_ns: u64 = 0;
        let mut hw_attn_ns: u64 = 0;
        let mut hw_post_attn_ns: u64 = 0;
        let mut hw_lm_head_ns: u64 = 0;

        let effective_layers = std::env::var("IRONMILL_MAX_LAYERS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(num_layers)
            .min(num_layers);

        // Experimental: attempt chained pre_attn execution.
        // When enabled, all pre_attn sub-programs are dispatched as a
        // pipeline, eliminating per-layer CPU↔ANE roundtrips for the
        // norm+projection phase. Falls back to per-layer eval on failure.
        let chained_pre_attn_ok = if self.enable_chaining {
            match self.try_chained_pre_attn(&hidden, effective_layers) {
                Ok(()) => true,
                Err(e) => {
                    eprintln!("[chaining] pre_attn chain failed, falling back: {e}");
                    false
                }
            }
        } else {
            false
        };

        for layer_idx in 0..effective_layers {
            // Experimental: attempt hybrid ANE↔GPU execution for this layer.
            if self.enable_hybrid {
                match self.try_hybrid_layer(&hidden, layer_idx) {
                    Ok(layer_out) => {
                        hidden = layer_out;
                        continue;
                    }
                    Err(_e) => {
                        if layer_idx == 0 {
                            eprintln!(
                                "[hybrid] ANE↔GPU handoff not available, \
                                 falling back to standard eval"
                            );
                        }
                    }
                }
            }

            // Pre-attention: norm → Q/K/V projection.
            // Skip if chained pre_attn already dispatched all layers.
            if !chained_pre_attn_ok {
                let t0 = if profiling {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                self.run_pre_attn_layer(layer_idx, &hidden, hw_profiling, &mut hw_pre_attn_ns)?;
                if let Some(t) = t0 {
                    d_pre_attn += t.elapsed();
                }
            }

            // Attention (divergent path) — writes into attn_out scratch buffer.
            let num_pre_outputs = self.layers[layer_idx].pre_attn.output_tensors.len();
            self.run_attention_layer(
                layer_idx,
                &mut attn_out,
                &mut q_buf,
                &mut k_buf,
                &mut v_buf,
                profiling,
                hw_profiling,
                &mut hw_attn_ns,
                &mut d_attn,
                &mut d_read_qkv,
            )?;

            // Post-attention: O proj → residual → FFN → residual
            let t0 = if profiling {
                Some(std::time::Instant::now())
            } else {
                None
            };
            hidden = self.run_post_attn_layer(
                layer_idx,
                &attn_out,
                &hidden,
                num_pre_outputs,
                hw_profiling,
                &mut hw_post_attn_ns,
            )?;
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
        let logits = self.decode_lm_head(&mut hidden, hw_profiling, &mut hw_lm_head_ns);
        let d_lm_head = t0.map(|t| t.elapsed());

        if profiling {
            Self::decode_print_profiling(
                t_total,
                d_embed,
                d_pre_attn,
                d_read_qkv,
                d_attn,
                d_post_attn,
                d_lm_head,
                hw_profiling,
                self.seq_pos - 1,
                hw_pre_attn_ns,
                hw_attn_ns,
                hw_post_attn_ns,
                hw_lm_head_ns,
            );
        } else if hw_profiling {
            Self::decode_print_profiling(
                None,
                None,
                d_pre_attn,
                d_read_qkv,
                d_attn,
                d_post_attn,
                None,
                hw_profiling,
                self.seq_pos - 1,
                hw_pre_attn_ns,
                hw_attn_ns,
                hw_post_attn_ns,
                hw_lm_head_ns,
            );
        }

        // Put scratch buffers back for reuse on next decode() call.
        self.scratch.hidden = hidden;
        self.scratch.attn_out = attn_out;
        self.scratch.q_data = q_buf;
        self.scratch.k_data = k_buf;
        self.scratch.v_data = v_buf;
        self.scratch.residual = residual_buf;

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
            return Err(AneError::Validation(
                "prompt_tokens must not be empty".into(),
            ));
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
            let token_id = sample_token(&logits, temperature)
                .ok_or_else(|| AneError::Validation("sampling produced empty logits".into()))?;

            // EOS detection — use model-specific EOS tokens.
            let eos = &self.model_info.eos_tokens;
            let eos_tokens = if eos.is_empty() {
                DEFAULT_EOS_TOKENS
            } else {
                eos.as_slice()
            };
            if is_eos_token(token_id, eos_tokens) {
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
        // Re-initialize the causal mask to all -inf (pre-allocated buffer).
        if let Some(ref mut mask) = self.fp16_attn_mask {
            let msl = self.max_seq_len;
            let _ = mask.write_f16(&self.scratch.neg_inf_mask[..msl]);
        }
        // Zero the FP16 KV caches using pre-allocated zero buffer.
        if let Some(ref mut caches) = self.fp16_kv_caches {
            for (k, v) in caches.iter_mut() {
                let k_size = k.shape()[1] * k.shape()[3];
                let v_size = v.shape()[1] * v.shape()[3];
                let _ = k.write_f16(&self.scratch.zeros[..k_size]);
                let _ = v.write_f16(&self.scratch.zeros[..v_size]);
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
// InferenceEngine implementation
// ---------------------------------------------------------------------------

impl<D: AneDevice + Send + 'static> InferenceEngine for AneInference<D> {
    fn prefill(&mut self, tokens: &[u32]) -> std::result::Result<Logits, InferenceError> {
        AneInference::prefill(self, tokens).map_err(|e| InferenceError::Runtime(e.into()))
    }

    fn decode_step(&mut self, token: u32) -> std::result::Result<Logits, InferenceError> {
        self.decode(token)
            .map_err(|e| InferenceError::Runtime(e.into()))
    }

    fn reset(&mut self) {
        AneInference::reset(self);
    }

    fn seq_pos(&self) -> usize {
        AneInference::seq_pos(self)
    }

    fn truncate_to(&mut self, pos: usize) -> Result<(), InferenceError> {
        if pos > self.seq_pos {
            return Err(InferenceError::runtime(format!(
                "cannot truncate forward: pos {pos} > seq_pos {}",
                self.seq_pos
            )));
        }
        self.seq_pos = pos;
        // FP16 KV caches are pre-allocated with max_seq_len and written
        // positionally, so truncation only needs to reset the write pointer.
        // The stale data beyond `pos` will be overwritten on subsequent decodes.
        // TurboQuant caches don't support partial truncation — a full reset
        // is required, which means a subsequent prefill must replay from scratch.
        if let Some(ref mut tq) = self.turboquant {
            tq.reset();
        }
        Ok(())
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    fn model_info(&self) -> Result<&ModelInfo, InferenceError> {
        Ok(&self.model_info)
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
/// zero-padded. Uses `scratch` to avoid per-call heap allocation.
fn write_f16_padded(tensor: &mut AneTensor, data: &[f16], scratch: &mut Vec<f16>) -> Result<()> {
    let [_, channels, _, seq_len] = tensor.shape();
    if seq_len == 1 {
        // No padding needed — direct write.
        return Ok(tensor.write_f16(data)?);
    }
    let total = channels * seq_len;
    scratch.clear();
    scratch.resize(total, f16::ZERO);
    let c = data.len().min(channels);
    for i in 0..c {
        scratch[i * seq_len] = data[i]; // column 0 of each channel row
    }
    Ok(tensor.write_f16(scratch)?)
}

/// Read C elements (one token at column 0) from an ANE tensor with shape
/// `[1, C, 1, S]`. Inverse of `write_f16_padded`. Writes into `out` to
/// avoid per-call heap allocation. Uses `read_column0_f16_into()` to read
/// directly into the output buffer with zero intermediate allocations.
fn read_f16_channels(tensor: &AneTensor, out: &mut Vec<f16>) -> Result<()> {
    tensor.read_column0_f16_into(out)?;
    Ok(())
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
            AneError::IoError(format!(
                "failed to read MIL text for {}: {e}",
                manifest.name
            ))
        })?;

    let weight_blob = match std::fs::read(weights_dir.join(format!("{}.bin", manifest.name))) {
        Ok(data) => data,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(e) => {
            return Err(AneError::IoError(format!(
                "failed to read weights for {}: {e}",
                manifest.name
            )));
        }
    };

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
            AneError::IoError(format!(
                "failed to read MIL text for {}: {e}",
                manifest.name
            ))
        })?;

    let weight_blob = match std::fs::read(weights_dir.join(format!("{}.bin", manifest.name))) {
        Ok(data) => data,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(e) => {
            return Err(AneError::IoError(format!(
                "failed to read weights for {}: {e}",
                manifest.name
            )));
        }
    };

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

    let input_alloc = uniform_alloc_size(&input_shapes)?;
    let output_alloc = uniform_alloc_size(&output_shapes)?.max(min_output_alloc);

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
/// Writes the row as fp16 values into `out` to avoid per-call allocation.
fn cpu_embedding_lookup(weight: &CpuWeight, token_id: u32, out: &mut Vec<f16>) -> Result<()> {
    let [vocab_size, hidden_size] = weight.shape;
    let idx = token_id as usize;
    if idx >= vocab_size {
        return Err(AneError::Validation(format!(
            "token_id {token_id} >= vocab_size {vocab_size}"
        )));
    }

    let bytes_per_elem = 2; // fp16
    let row_start = idx * hidden_size * bytes_per_elem;
    let row_end = row_start + hidden_size * bytes_per_elem;

    if row_end > weight.data.len() {
        return Err(AneError::Validation(format!(
            "embedding weight data too short: need {} bytes, have {}",
            row_end,
            weight.data.len()
        )));
    }

    let row_bytes = &weight.data[row_start..row_end];
    out.clear();
    out.extend(
        row_bytes
            .chunks_exact(2)
            .map(|b| f16::from_le_bytes([b[0], b[1]])),
    );
    Ok(())
}

/// CPU lm_head: matmul hidden × weight^T → logits as f32.
/// Weight shape: [vocab_size, hidden_size], hidden shape: [hidden_size].
fn cpu_lm_head_matmul(weight: &CpuWeight, hidden: &[f16]) -> Result<Vec<f32>> {
    let [vocab_size, hidden_size] = weight.shape;
    if hidden.len() < hidden_size {
        return Err(AneError::Validation(format!(
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
