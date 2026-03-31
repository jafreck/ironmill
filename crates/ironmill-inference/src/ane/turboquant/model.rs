//! TurboQuant KV cache compression configuration and KV cache management.
//!
//! The KV cache stores INT8 values (1 byte/element) produced by rotation +
//! Beta-optimal scalar quantization. The ANE attention program casts INT8→FP16
//! inline, eliminating CPU format conversions while retaining 2× memory savings.

use ironmill_ane_sys::AneCompiler;
use mil_rs::ir::ScalarType;
use mil_rs::ir::passes::beta_quantizer::{beta_optimal_boundaries, beta_optimal_levels};
use mil_rs::ir::passes::rotation::{rotate_rows_hadamard, unrotate_rows_hadamard};

use half::f16;

use super::mil_emitter;
use super::mil_emitter::MIN_IO_SEQ;
use crate::ane::runtime::AneRuntime;
use crate::ane::{AneError, Result};
use ironmill_ane_sys::LoadedProgram;
use ironmill_iosurface::AneTensor;

/// Configuration for TurboQuant KV cache compression.
///
/// Controls runtime KV cache quantization using rotation + Beta-optimal
/// scalar quantization. Storage format is INT8 (1 byte/element);
/// `n_bits` controls the number of distinct quantization levels.
#[derive(Clone)]
pub struct TurboQuantConfig {
    /// Number of quantization bits (1, 2, 4, 6, or 8).
    /// Controls quality via 2^n_bits distinct Beta-optimal levels
    /// mapped into the [-128, 127] INT8 range.
    pub n_bits: u8,
    /// Maximum sequence length for cache allocation.
    pub max_seq_len: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of KV heads (may differ from num_heads for GQA).
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Hadamard rotation seed (deterministic, shared with dequant).
    pub rotation_seed: u64,
    /// Enable QJL 1-bit bias correction.
    pub enable_qjl: bool,
}

const VALID_N_BITS: &[u8] = &[1, 2, 4, 6, 8];

impl TurboQuantConfig {
    /// Create a new TurboQuantConfig, validating parameters.
    pub fn new(
        n_bits: u8,
        max_seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        num_layers: usize,
    ) -> Result<Self> {
        if !VALID_N_BITS.contains(&n_bits) {
            return Err(AneError::Other(anyhow::anyhow!(
                "invalid n_bits {n_bits}: must be one of {VALID_N_BITS:?}"
            )));
        }
        if max_seq_len == 0 {
            return Err(AneError::Other(anyhow::anyhow!("max_seq_len must be > 0")));
        }
        if num_heads == 0 {
            return Err(AneError::Other(anyhow::anyhow!("num_heads must be > 0")));
        }
        if num_kv_heads == 0 {
            return Err(AneError::Other(anyhow::anyhow!("num_kv_heads must be > 0")));
        }
        if head_dim == 0 {
            return Err(AneError::Other(anyhow::anyhow!("head_dim must be > 0")));
        }
        if num_layers == 0 {
            return Err(AneError::Other(anyhow::anyhow!("num_layers must be > 0")));
        }
        if num_heads % num_kv_heads != 0 {
            return Err(AneError::Other(anyhow::anyhow!(
                "num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )));
        }

        Ok(Self {
            n_bits,
            max_seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            num_layers,
            rotation_seed: 42,
            enable_qjl: false,
        })
    }

    /// Enable or disable QJL 1-bit bias correction.
    pub fn with_qjl(mut self, enable: bool) -> Self {
        self.enable_qjl = enable;
        self
    }

    /// Set the Hadamard rotation seed.
    pub fn with_rotation_seed(mut self, seed: u64) -> Self {
        self.rotation_seed = seed;
        self
    }

    /// Construct a `TurboQuantConfig` from detected model architecture.
    ///
    /// Uses 8-bit quantization by default.
    pub fn from_arch(arch: &mil_rs::analysis::arch::ModelArch, max_seq_len: usize) -> Result<Self> {
        Self::new(
            8,
            max_seq_len,
            arch.num_heads,
            arch.num_kv_heads,
            arch.head_dim,
            arch.num_layers,
        )
    }
}

/// Manages per-layer INT8 KV caches with TurboQuant compression.
///
/// Cache tensors store INT8 quantized values (1 byte/element). The ANE
/// attention program casts INT8→FP16 inline before dequantization.
#[allow(dead_code)]
pub struct KvCacheManager {
    config: TurboQuantConfig,
    /// Per-layer K caches: [num_kv_heads * head_dim, max_seq_len] as INT8.
    k_caches: Vec<AneTensor>,
    /// Per-layer V caches (same format).
    v_caches: Vec<AneTensor>,
    /// Current sequence position (next write index).
    seq_pos: usize,
    /// Precomputed Beta-optimal quantization levels [2^n_bits].
    quant_levels: Vec<f32>,
    /// Precomputed quantization boundaries [2^n_bits - 1].
    quant_boundaries: Vec<f32>,
    /// Precomputed Hadamard rotation matrix [head_dim × head_dim].
    rotation_matrix: Vec<f32>,
    /// Dequantization scale: 1.0 / inv_scale.
    deq_scale: f32,
    /// Optional: per-layer QJL residual sign caches (fp16 ±1).
    qjl_sign_caches: Option<Vec<AneTensor>>,
}

impl KvCacheManager {
    /// Create a new `KvCacheManager`, allocating per-layer INT8 KV cache
    /// tensors and precomputing quantization tables.
    pub fn new(config: TurboQuantConfig) -> Result<Self> {
        Self::new_with_alloc(config, 0)
    }

    /// Create with a minimum allocation size for uniform ANE eval compatibility.
    pub fn new_with_alloc(config: TurboQuantConfig, min_alloc: usize) -> Result<Self> {
        let channels = config.num_kv_heads * config.head_dim;

        let mut k_caches = Vec::with_capacity(config.num_layers);
        let mut v_caches = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            k_caches.push(AneTensor::new_with_min_alloc(
                channels,
                config.max_seq_len,
                ScalarType::Int8,
                min_alloc,
            )?);
            v_caches.push(AneTensor::new_with_min_alloc(
                channels,
                config.max_seq_len,
                ScalarType::Int8,
                min_alloc,
            )?);
        }

        let quant_levels = beta_optimal_levels(config.head_dim, config.n_bits);
        let quant_boundaries = beta_optimal_boundaries(config.head_dim, config.n_bits);

        // Precompute Hadamard rotation matrix by rotating an identity matrix.
        let dim = config.head_dim;
        let mut rotation_matrix = vec![0.0f32; dim * dim];
        for i in 0..dim {
            rotation_matrix[i * dim + i] = 1.0;
        }
        rotate_rows_hadamard(&mut rotation_matrix, dim, dim, config.rotation_seed);

        // Precompute dequantization scale.
        let max_level = quant_levels.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let deq_scale = if max_level == 0.0 {
            1.0
        } else {
            max_level / 127.0
        };

        let qjl_sign_caches = if config.enable_qjl {
            let mut caches = Vec::with_capacity(config.num_layers);
            for _ in 0..config.num_layers {
                caches.push(AneTensor::new_with_min_alloc(
                    channels,
                    config.max_seq_len,
                    ScalarType::Float16,
                    min_alloc,
                )?);
            }
            Some(caches)
        } else {
            None
        };

        Ok(Self {
            config,
            k_caches,
            v_caches,
            seq_pos: 0,
            quant_levels,
            quant_boundaries,
            rotation_matrix,
            deq_scale,
            qjl_sign_caches,
        })
    }

    /// Write one token's worth of INT8-quantized K and V data into `layer`'s
    /// caches at the current sequence position.
    ///
    /// When QJL is enabled, also computes and stores residual signs:
    /// `sign(K_original - K_dequantized)` as ±1 fp16.
    ///
    /// **Does not advance `seq_pos`.** Call [`advance_seq_pos`] once after
    /// all layers have been updated for a given token.
    pub fn update_cache(
        &mut self,
        layer: usize,
        k_quantized: &[u8],
        v_quantized: &[u8],
        k_original: Option<&[f16]>,
    ) -> Result<()> {
        if layer >= self.config.num_layers {
            return Err(AneError::Other(anyhow::anyhow!(
                "layer index {layer} out of range (num_layers = {})",
                self.config.num_layers
            )));
        }

        let token_elements = self.config.num_kv_heads * self.config.head_dim;
        if k_quantized.len() != token_elements {
            return Err(AneError::Other(anyhow::anyhow!(
                "k_quantized length {} != expected {} (num_kv_heads * head_dim)",
                k_quantized.len(),
                token_elements
            )));
        }
        if v_quantized.len() != token_elements {
            return Err(AneError::Other(anyhow::anyhow!(
                "v_quantized length {} != expected {} (num_kv_heads * head_dim)",
                v_quantized.len(),
                token_elements
            )));
        }

        if self.seq_pos >= self.config.max_seq_len {
            return Err(AneError::Other(anyhow::anyhow!(
                "KV cache full: seq_pos {} >= max_seq_len {}",
                self.seq_pos,
                self.config.max_seq_len
            )));
        }

        let byte_offset = self.seq_pos * token_elements;
        self.k_caches[layer].write_bytes_at(byte_offset, k_quantized)?;
        self.v_caches[layer].write_bytes_at(byte_offset, v_quantized)?;

        // QJL residual sign computation
        if let (Some(sign_caches), Some(k_orig)) = (&mut self.qjl_sign_caches, k_original) {
            let signs = compute_qjl_signs(
                k_quantized,
                k_orig,
                self.config.num_kv_heads,
                self.config.head_dim,
                self.deq_scale,
                self.config.rotation_seed,
            );
            let elem_offset = self.seq_pos * token_elements;
            sign_caches[layer].write_f16_at(elem_offset, &signs)?;
        }

        Ok(())
    }

    /// Return references to the QJL sign caches for a given layer.
    /// Returns `None` if QJL is not enabled.
    pub fn qjl_sign_tensors(&self, layer: usize) -> Option<&AneTensor> {
        self.qjl_sign_caches.as_ref().map(|c| &c[layer])
    }

    /// Advance the sequence position after all layers have been updated
    /// for the current token. Must be called exactly once per token.
    pub fn advance_seq_pos(&mut self) {
        self.seq_pos += 1;
    }

    /// Return references to the K and V cache tensors for a given layer.
    pub fn cache_tensors(&self, layer: usize) -> (&AneTensor, &AneTensor) {
        (&self.k_caches[layer], &self.v_caches[layer])
    }

    /// Current number of tokens stored (next write index).
    pub fn seq_len(&self) -> usize {
        self.seq_pos
    }

    /// Reset the sequence position to zero. Tensor data is not cleared —
    /// it will be overwritten on subsequent `update_cache` calls.
    pub fn reset(&mut self) {
        self.seq_pos = 0;
    }

    /// Write one token's cached K/V data directly from cache-write output
    /// IOSurface tensors into the persistent INT8 cache.
    ///
    /// Performs an IOSurface-to-IOSurface strided copy with FP16→INT8
    /// conversion in a single locked pass per tensor, eliminating the
    /// intermediate `Vec<f16>` and `Vec<u8>` allocations from the old
    /// `read_column0_f16` → convert → `write_bytes_at` path.
    ///
    /// When QJL is enabled, `k_original` must be provided for residual
    /// sign computation.
    ///
    /// **Does not advance `seq_pos`.** Call [`advance_seq_pos`] once after
    /// all layers have been updated for a given token.
    pub fn update_cache_direct(
        &mut self,
        layer: usize,
        cw_k_output: &AneTensor,
        cw_v_output: &AneTensor,
        k_original: Option<&[f16]>,
    ) -> Result<()> {
        if layer >= self.config.num_layers {
            return Err(AneError::Other(anyhow::anyhow!(
                "layer index {layer} out of range (num_layers = {})",
                self.config.num_layers
            )));
        }

        if self.seq_pos >= self.config.max_seq_len {
            return Err(AneError::Other(anyhow::anyhow!(
                "KV cache full: seq_pos {} >= max_seq_len {}",
                self.seq_pos,
                self.config.max_seq_len
            )));
        }

        let token_elements = self.config.num_kv_heads * self.config.head_dim;
        let byte_offset = self.seq_pos * token_elements;

        // Direct IOSurface-to-IOSurface copy with FP16→INT8 conversion.
        cw_k_output.copy_column0_fp16_as_int8_to(&mut self.k_caches[layer], byte_offset)?;
        cw_v_output.copy_column0_fp16_as_int8_to(&mut self.v_caches[layer], byte_offset)?;

        // QJL residual sign computation (still requires CPU read-back).
        if let (Some(sign_caches), Some(k_orig)) = (&mut self.qjl_sign_caches, k_original) {
            // Read back the quantized bytes we just wrote for dequant comparison.
            let k_quantized = self.k_caches[layer].read_bytes_at(byte_offset, token_elements)?;
            let signs = compute_qjl_signs(
                &k_quantized,
                k_orig,
                self.config.num_kv_heads,
                self.config.head_dim,
                self.deq_scale,
                self.config.rotation_seed,
            );
            let elem_offset = self.seq_pos * token_elements;
            sign_caches[layer].write_f16_at(elem_offset, &signs)?;
        }

        Ok(())
    }
}

/// Compute QJL residual signs: `sign(K_original - K_dequantized)`.
///
/// Dequantization pipeline: int8 → f32 → scale → un-rotate.
/// Returns ±1 as fp16 for each element.
fn compute_qjl_signs(
    k_quantized: &[u8],
    k_original: &[f16],
    num_kv_heads: usize,
    head_dim: usize,
    deq_scale: f32,
    rotation_seed: u64,
) -> Vec<f16> {
    let total = num_kv_heads * head_dim;

    // 1. Cast INT8 to f32 and scale (undo quantization, still rotated)
    let mut k_deq_rotated = vec![0.0f32; total];
    for i in 0..total {
        k_deq_rotated[i] = (k_quantized[i] as i8) as f32 * deq_scale;
    }

    // 2. Un-rotate per head
    unrotate_rows_hadamard(&mut k_deq_rotated, num_kv_heads, head_dim, rotation_seed);

    // 3. Compute sign(original - dequantized)
    let mut signs = vec![f16::ZERO; total];
    for i in 0..total {
        let orig = k_original[i].to_f32();
        let diff = orig - k_deq_rotated[i];
        signs[i] = if diff >= 0.0 {
            f16::from_f32(1.0)
        } else {
            f16::from_f32(-1.0)
        };
    }

    signs
}

// ---------------------------------------------------------------------------
// TurboQuantModel — inference loop orchestrator
// ---------------------------------------------------------------------------

/// Orchestrates TurboQuant INT8 KV cache inference on the ANE.
///
/// Holds compiled cache-write (rotate + quantize) and attention (dequant +
/// attention) sub-programs together with a [`KvCacheManager`] and an
/// [`AneRuntime`] handle. The KV cache stores INT8 values (1 byte/element);
/// the ANE attention program casts INT8→FP16 inline before dequantization.
///
/// All TQ-facing tensors share a single uniform allocation size
/// (`tq_alloc_size`), enabling zero-copy IOSurface passing from pre_attn
/// outputs directly into cache-write and attention eval calls.
#[allow(dead_code)]
pub struct TurboQuantModel {
    config: TurboQuantConfig,
    cache: KvCacheManager,
    /// Compiled cache-write sub-program (rotate + quantize K/V).
    cache_write_program: LoadedProgram,
    /// Compiled attention sub-program (dequant + Q-rotate + attention + output un-rotate).
    attention_program: LoadedProgram,
    /// Optional QJL correction program.
    qjl_program: Option<LoadedProgram>,
    /// Rotation matrix as IOSurface tensor (shared by cache-write and attention).
    /// Cache-write uses it to rotate K/V before quantization.
    /// Attention uses it to rotate Q and un-rotate the output.
    rotation_tensor: AneTensor,
    /// Uniform allocation size shared by all TQ input tensors.
    /// Pre-attn outputs must be allocated with this size for zero-copy.
    tq_alloc_size: usize,
    /// Uniform allocation size for cache-write outputs (may be smaller).
    cw_output_alloc_size: usize,
    /// Pre-allocated cache-write output for K (reused across calls).
    cw_k_output: AneTensor,
    /// Pre-allocated cache-write output for V (reused across calls).
    cw_v_output: AneTensor,
    /// The ANE runtime handle.
    runtime: AneRuntime,
}

impl TurboQuantModel {
    /// Compile and load the TurboQuant sub-programs onto the ANE.
    ///
    /// 1. Creates an [`AneRuntime`].
    /// 2. Emits + compiles the cache-write MIL program.
    /// 3. Emits + compiles the attention MIL program.
    /// 4. Optionally compiles the QJL correction program.
    /// 5. Allocates the [`KvCacheManager`].
    pub fn compile(config: TurboQuantConfig) -> Result<Self> {
        let runtime = AneRuntime::new()?;
        let head_dim = config.head_dim;

        // --- cache-write sub-program ---
        let (cw_mil, cw_weights) = mil_emitter::emit_cache_write_mil(&config);
        // Weights are delivered as function inputs, not BLOBFILE — pass empty weights
        let cw_compiled =
            AneCompiler::compile_mil_text(&cw_mil, &[]).map_err(|e| AneError::CompileFailed {
                status: 0,
                context: format!("cache-write compilation failed: {e}"),
            })?;
        let cache_write_program = runtime.load_program(&cw_compiled)?;

        // --- attention sub-program ---
        let deq_scale = mil_emitter::compute_deq_scale(head_dim, config.n_bits);
        let attn_config = mil_emitter::AttentionMilConfig {
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim,
            max_seq_len: config.max_seq_len,
            seq_len: config.max_seq_len,
            dequant_scale: Some(deq_scale),
            unrotation_seed: Some(config.rotation_seed),
            cache_int8: true,
        };
        let (attn_mil, _attn_weights) = mil_emitter::emit_attention_mil(&attn_config);
        let attn_compiled =
            AneCompiler::compile_mil_text(&attn_mil, &[]).map_err(|e| AneError::CompileFailed {
                status: 0,
                context: format!("attention compilation failed: {e}"),
            })?;
        let attention_program = runtime.load_program(&attn_compiled)?;

        // --- QJL correction sub-program (optional) ---
        let qjl_program = if config.enable_qjl {
            let (qjl_mil, _qjl_weights) =
                mil_emitter::emit_qjl_correction_mil(&config, config.max_seq_len);
            let qjl_compiled = AneCompiler::compile_mil_text(&qjl_mil, &[]).map_err(|e| {
                AneError::CompileFailed {
                    status: 0,
                    context: format!("QJL correction compilation failed: {e}"),
                }
            })?;
            Some(runtime.load_program(&qjl_compiled)?)
        } else {
            None
        };

        // --- Compute uniform alloc size (ANE requires all input tensors in
        //     one eval to share the same allocation size) ---
        //
        // Use a single alloc size across both cache-write and attention
        // input tensors. Pre-attn outputs are allocated with this same size,
        // enabling zero-copy IOSurface passing (no staging copies).
        let kv_ch = config.num_kv_heads * head_dim;
        let q_ch = config.num_heads * head_dim;

        let tq_alloc_size = ironmill_iosurface::uniform_alloc_size(&[
            // Cache-write inputs
            ([1, kv_ch, 1, MIN_IO_SEQ], ScalarType::Float16), // K_proj
            ([1, kv_ch, 1, MIN_IO_SEQ], ScalarType::Float16), // V_proj
            ([1, 1, head_dim, head_dim], ScalarType::Float16), // rotation matrix
            // Attention inputs (rotation_matrix is shared, same shape)
            ([1, q_ch, 1, MIN_IO_SEQ], ScalarType::Float16), // Q
            ([1, kv_ch, 1, config.max_seq_len], ScalarType::Int8), // K cache (INT8)
            ([1, kv_ch, 1, config.max_seq_len], ScalarType::Int8), // V cache (INT8)
        ]);

        // Cache-write outputs only need to be uniform with each other.
        let cw_output_alloc_size = ironmill_iosurface::uniform_alloc_size(&[
            ([1, kv_ch, 1, MIN_IO_SEQ], ScalarType::Float16),
            ([1, kv_ch, 1, MIN_IO_SEQ], ScalarType::Float16),
        ]);

        // --- KV cache with uniform alloc ---
        let cache = KvCacheManager::new_with_alloc(config.clone(), tq_alloc_size)?;

        // --- Rotation matrix IOSurface tensor (shared by cache-write and attention) ---
        // Q-rotation approach: the attention program uses the SAME rotation
        // matrix as cache-write (to rotate Q and un-rotate output), not the
        // inverse. This eliminates the need for a separate unrotation tensor.
        let rot_data = &cw_weights[0].1;
        let mut rotation_tensor =
            AneTensor::new_with_min_alloc(head_dim, head_dim, ScalarType::Float16, tq_alloc_size)?;
        rotation_tensor.write_bytes_at(0, rot_data)?;

        // Pre-allocate output tensors for step_attention() reuse.
        // These are overwritten on every eval call, so sharing across layers is safe.
        let kv_channels = config.num_kv_heads * head_dim;
        let cw_k_output = AneTensor::new_with_min_alloc(
            kv_channels,
            MIN_IO_SEQ,
            ScalarType::Float16,
            cw_output_alloc_size,
        )?;
        let cw_v_output = AneTensor::new_with_min_alloc(
            kv_channels,
            MIN_IO_SEQ,
            ScalarType::Float16,
            cw_output_alloc_size,
        )?;

        Ok(Self {
            config,
            cache,
            cache_write_program,
            attention_program,
            qjl_program,
            rotation_tensor,
            tq_alloc_size,
            cw_output_alloc_size,
            cw_k_output,
            cw_v_output,
            runtime,
        })
    }

    /// Run one token through all layers of the model.
    ///
    /// For each layer: quantize K/V → write to cache → run attention.
    /// The `projections` slice must contain one `(Q, K_proj, V_proj)` tuple
    /// per layer.
    pub fn step(
        &mut self,
        projections: &[(AneTensor, AneTensor, AneTensor)],
    ) -> Result<Vec<AneTensor>> {
        if projections.len() != self.config.num_layers {
            return Err(AneError::Other(anyhow::anyhow!(
                "expected {} layer projections, got {}",
                self.config.num_layers,
                projections.len()
            )));
        }

        let mut outputs = Vec::with_capacity(self.config.num_layers);
        for (layer, (q, k_proj, v_proj)) in projections.iter().enumerate() {
            let attn_out = self.step_attention(layer, q, k_proj, v_proj)?;
            outputs.push(attn_out);
        }

        // Advance sequence position once after all layers processed this token.
        self.cache.advance_seq_pos();

        Ok(outputs)
    }

    /// Process one token through cache-write and attention for a single layer.
    ///
    /// Takes Q, K_proj, V_proj as fp16 [`AneTensor`]s (single token).
    /// Returns attention output as fp16 [`AneTensor`].
    ///
    /// Uses direct IOSurface-to-IOSurface copy for the cache update,
    /// eliminating intermediate heap allocations.
    ///
    /// **Note:** Does not advance `seq_pos`. Use [`step`] for full-model
    /// inference, which advances after all layers.
    pub fn step_attention(
        &mut self,
        layer: usize,
        q: &AneTensor,
        k_proj: &AneTensor,
        v_proj: &AneTensor,
    ) -> Result<AneTensor> {
        // 1. Cache-write: K_proj, V_proj, rotation_matrix → K_quant, V_quant
        //    Pre-attn outputs share tq_alloc_size, so no staging copy needed.
        self.runtime.eval(
            &self.cache_write_program,
            &[k_proj, v_proj, &self.rotation_tensor],
            &mut [&mut self.cw_k_output, &mut self.cw_v_output],
        )?;

        // 2. Direct IOSurface-to-IOSurface copy: cache-write output → KV cache.
        //    Reads FP16 column 0 from cache-write output, converts to INT8,
        //    writes directly into cache IOSurface. Zero heap allocations.
        let k_original = if self.config.enable_qjl {
            Some(k_proj.read_f16()?)
        } else {
            None
        };
        self.cache.update_cache_direct(
            layer,
            &self.cw_k_output,
            &self.cw_v_output,
            k_original.as_deref(),
        )?;

        // 3. Attention: Q + cached K/V + rotation_matrix → output
        //    Q shares tq_alloc_size with caches, so no staging copy needed.
        //    The rotation matrix is used to rotate Q and un-rotate the output
        //    (O(1) per token, instead of un-rotating the entire cache).
        let q_channels = self.config.num_heads * self.config.head_dim;
        let (k_cache, v_cache) = self.cache.cache_tensors(layer);
        let mut attn_out = AneTensor::new_with_min_alloc(
            q_channels,
            MIN_IO_SEQ,
            ScalarType::Float16,
            self.tq_alloc_size,
        )?;
        self.runtime.eval(
            &self.attention_program,
            &[q, k_cache, v_cache, &self.rotation_tensor],
            &mut [&mut attn_out],
        )?;

        Ok(attn_out)
    }

    /// Process one token through attention for a single layer (fused mode).
    ///
    /// In fused mode, the cache-write ops (rotation + quantization) have
    /// been injected into the pre_attn sub-program, so K_quant and V_quant
    /// arrive already quantized. This method skips the cache-write ANE eval
    /// and goes directly to cache update + attention.
    ///
    /// `k_quant` and `v_quant` must be fp16 tensors with INT8-range values
    /// (output of the fused cache-write chain in pre_attn).
    ///
    /// When QJL is enabled, `k_original` provides the unquantized K values
    /// for residual sign computation.
    pub fn step_attention_fused(
        &mut self,
        layer: usize,
        q: &AneTensor,
        k_quant: &AneTensor,
        v_quant: &AneTensor,
        k_original: Option<&AneTensor>,
    ) -> Result<AneTensor> {
        // 1. Direct IOSurface-to-IOSurface copy: pre_attn quantized output → KV cache.
        let k_orig_data = if self.config.enable_qjl {
            let t = k_original.ok_or_else(|| {
                AneError::Other(anyhow::anyhow!(
                    "QJL enabled but k_original not provided in fused mode"
                ))
            })?;
            Some(t.read_f16()?)
        } else {
            None
        };
        self.cache
            .update_cache_direct(layer, k_quant, v_quant, k_orig_data.as_deref())?;

        // 2. Attention: Q + cached K/V + rotation_matrix → output
        let q_channels = self.config.num_heads * self.config.head_dim;
        let (k_cache, v_cache) = self.cache.cache_tensors(layer);
        let mut attn_out = AneTensor::new_with_min_alloc(
            q_channels,
            MIN_IO_SEQ,
            ScalarType::Float16,
            self.tq_alloc_size,
        )?;
        self.runtime.eval(
            &self.attention_program,
            &[q, k_cache, v_cache, &self.rotation_tensor],
            &mut [&mut attn_out],
        )?;

        Ok(attn_out)
    }

    /// Reset cache for a new conversation.
    pub fn reset(&mut self) {
        self.cache.reset();
    }

    /// Current sequence length.
    pub fn seq_len(&self) -> usize {
        self.cache.seq_len()
    }

    /// Advance the sequence position after all layers have been updated
    /// for the current token via per-layer `step_attention` calls.
    ///
    /// This is not needed when using `step()` (which advances internally),
    /// but is required when calling `step_attention()` per-layer from an
    /// external inference loop like [`AneInference`].
    pub fn advance_seq_pos(&mut self) {
        self.cache.advance_seq_pos();
    }

    /// Get a reference to the config.
    pub fn config(&self) -> &TurboQuantConfig {
        &self.config
    }

    /// Required uniform allocation size for external tensors.
    ///
    /// All pre-attn output tensors (Q, K_proj, V_proj) must be allocated
    /// with at least this size. This enables zero-copy IOSurface passing
    /// directly into cache-write and attention eval calls.
    pub fn alloc_size(&self) -> usize {
        self.tq_alloc_size
    }

    /// Required uniform allocation sizes for external tensors.
    ///
    /// Returns `(tq_alloc_size, tq_alloc_size)`. Both values are the
    /// same unified alloc size — kept as a pair for backward compatibility.
    pub fn alloc_sizes(&self) -> (usize, usize) {
        (self.tq_alloc_size, self.tq_alloc_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mil_rs::ir::ScalarType;

    fn test_config() -> TurboQuantConfig {
        TurboQuantConfig {
            n_bits: 8,
            max_seq_len: 64,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 8,
            num_layers: 2,
            rotation_seed: 42,
            enable_qjl: false,
        }
    }

    #[test]
    fn update_cache_direct_matches_update_cache() {
        let config = test_config();
        let channels = config.num_kv_heads * config.head_dim; // 32
        let mut cache_old = KvCacheManager::new(config.clone()).unwrap();
        let mut cache_new = KvCacheManager::new(config.clone()).unwrap();

        // Create a mock cache-write output: FP16 [1, channels, 1, 32]
        // with INT8-range values in column 0.
        let mut cw_k = AneTensor::new(channels, 32, ScalarType::Float16).unwrap();
        let mut cw_v = AneTensor::new(channels, 32, ScalarType::Float16).unwrap();

        let mut k_data = vec![f16::ZERO; channels * 32];
        let mut v_data = vec![f16::ZERO; channels * 32];
        for c in 0..channels {
            // Place INT8-range values in column 0 (offset c * 32).
            k_data[c * 32] = f16::from_f32((c as i32 - 16) as f32);
            v_data[c * 32] = f16::from_f32((c as i32 * 2 - 32) as f32);
        }
        cw_k.write_f16(&k_data).unwrap();
        cw_v.write_f16(&v_data).unwrap();

        // Old path: read → convert → write.
        let k_f16 = cw_k.read_column0_f16().unwrap();
        let k_bytes: Vec<u8> = k_f16.iter().map(|v| (v.to_f32() as i8) as u8).collect();
        let v_f16 = cw_v.read_column0_f16().unwrap();
        let v_bytes: Vec<u8> = v_f16.iter().map(|v| (v.to_f32() as i8) as u8).collect();
        cache_old.update_cache(0, &k_bytes, &v_bytes, None).unwrap();

        // New path: direct.
        cache_new
            .update_cache_direct(0, &cw_k, &cw_v, None)
            .unwrap();

        // Compare: read back from both caches.
        let byte_offset = 0; // seq_pos = 0
        let old_k = cache_old.k_caches[0]
            .read_bytes_at(byte_offset, channels)
            .unwrap();
        let new_k = cache_new.k_caches[0]
            .read_bytes_at(byte_offset, channels)
            .unwrap();
        assert_eq!(old_k, new_k, "K cache: direct path must match old path");

        let old_v = cache_old.v_caches[0]
            .read_bytes_at(byte_offset, channels)
            .unwrap();
        let new_v = cache_new.v_caches[0]
            .read_bytes_at(byte_offset, channels)
            .unwrap();
        assert_eq!(old_v, new_v, "V cache: direct path must match old path");
    }

    #[test]
    fn update_cache_direct_multiple_positions() {
        let config = test_config();
        let channels = config.num_kv_heads * config.head_dim;
        let mut cache = KvCacheManager::new(config).unwrap();

        for pos in 0..4 {
            let mut cw_k = AneTensor::new(channels, 32, ScalarType::Float16).unwrap();
            let mut cw_v = AneTensor::new(channels, 32, ScalarType::Float16).unwrap();

            let mut k_data = vec![f16::ZERO; channels * 32];
            let mut v_data = vec![f16::ZERO; channels * 32];
            for c in 0..channels {
                k_data[c * 32] = f16::from_f32((pos * channels + c) as f32);
                v_data[c * 32] = f16::from_f32(-((pos * channels + c) as f32));
            }
            cw_k.write_f16(&k_data).unwrap();
            cw_v.write_f16(&v_data).unwrap();

            cache.update_cache_direct(0, &cw_k, &cw_v, None).unwrap();
            cache.advance_seq_pos();
        }

        assert_eq!(cache.seq_len(), 4);

        // Verify token 2's data.
        let byte_offset = 2 * channels;
        let k_bytes = cache.k_caches[0]
            .read_bytes_at(byte_offset, channels)
            .unwrap();
        for c in 0..channels {
            let expected = (2 * channels + c) as i8 as u8;
            assert_eq!(
                k_bytes[c], expected,
                "K cache mismatch at channel {c} for token 2"
            );
        }
    }

    #[test]
    fn update_cache_direct_layer_bounds() {
        let config = test_config();
        let channels = config.num_kv_heads * config.head_dim;
        let mut cache = KvCacheManager::new(config).unwrap();
        let cw_k = AneTensor::new(channels, 32, ScalarType::Float16).unwrap();
        let cw_v = AneTensor::new(channels, 32, ScalarType::Float16).unwrap();

        // Layer 2 is out of bounds (num_layers = 2, valid indices are 0 and 1).
        assert!(cache.update_cache_direct(2, &cw_k, &cw_v, None).is_err());
    }
}
