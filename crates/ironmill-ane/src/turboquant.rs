//! TurboQuant INT8 KV cache compression configuration and KV cache management.

use mil_rs::ffi::ane::AneCompiler;
use mil_rs::ir::ScalarType;
use mil_rs::ir::passes::beta_quantizer::{beta_optimal_boundaries, beta_optimal_levels};
use mil_rs::ir::passes::rotation::{rotate_rows_hadamard, unrotate_rows_hadamard};

use half::f16;

use crate::program::LoadedProgram;
use crate::runtime::AneRuntime;
use crate::tensor::AneTensor;
use crate::turboquant_mil;
use crate::turboquant_mil::MIN_IO_SEQ;
use crate::{AneError, Result};

/// Configuration for TurboQuant INT8 KV cache compression.
///
/// Controls runtime KV cache quantization using rotation + Beta-optimal
/// scalar quantization. Storage format is always INT8 (1 byte/element);
/// `n_bits` controls the number of distinct quantization levels within
/// the INT8 range.
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

/// Manages per-layer INT8 KV caches with TurboQuant quantization.
#[allow(dead_code)]
pub struct KvCacheManager {
    config: TurboQuantConfig,
    /// Per-layer K caches: [num_kv_heads, max_seq_len, head_dim] as INT8.
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

    /// Write one token's worth of quantized K and V data into `layer`'s
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
/// [`AneRuntime`] handle.
#[allow(dead_code)]
pub struct TurboQuantModel {
    config: TurboQuantConfig,
    cache: KvCacheManager,
    /// Compiled cache-write sub-program (rotate + quantize K/V to INT8).
    cache_write_program: LoadedProgram,
    /// Compiled attention sub-program (dequant + attention).
    attention_program: LoadedProgram,
    /// Optional QJL correction program.
    qjl_program: Option<LoadedProgram>,
    /// Rotation matrix as IOSurface tensor (input to cache-write program).
    rotation_tensor: AneTensor,
    /// Inverse rotation matrix as IOSurface tensor (input to attention program).
    unrotation_tensor: AneTensor,
    /// Uniform allocation size for cache-write program inputs/outputs.
    cw_alloc_size: usize,
    /// Uniform allocation size for attention program inputs/outputs.
    attn_alloc_size: usize,
    /// Staging buffers for external tensors that need alloc-size alignment.
    /// ANE requires all input IOSurfaces in a single eval call to have
    /// the same allocation size. Pre-attn outputs may have a different
    /// alloc size, so we copy into these staging buffers before eval.
    cw_k_staging: AneTensor,
    cw_v_staging: AneTensor,
    attn_q_staging: AneTensor,
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
        let (cw_mil, cw_weights) = turboquant_mil::emit_cache_write_mil(&config);
        // Weights are delivered as function inputs, not BLOBFILE — pass empty weights
        let cw_compiled =
            AneCompiler::compile_mil_text(&cw_mil, &[]).map_err(|e| AneError::CompileFailed {
                status: 0,
                context: format!("cache-write compilation failed: {e}"),
            })?;
        let cache_write_program = runtime.load_program(&cw_compiled)?;

        // --- attention sub-program ---
        let (attn_mil, attn_weights) =
            turboquant_mil::emit_attention_mil(&config, config.max_seq_len);
        let attn_compiled =
            AneCompiler::compile_mil_text(&attn_mil, &[]).map_err(|e| AneError::CompileFailed {
                status: 0,
                context: format!("attention compilation failed: {e}"),
            })?;
        let attention_program = runtime.load_program(&attn_compiled)?;

        // --- QJL correction sub-program (optional) ---
        let qjl_program = if config.enable_qjl {
            let (qjl_mil, _qjl_weights) =
                turboquant_mil::emit_qjl_correction_mil(&config, config.max_seq_len);
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

        // --- Compute uniform alloc sizes (ANE requires all tensors in one eval
        //     to have the same allocation) ---
        let kv_ch = config.num_kv_heads * head_dim;
        let q_ch = config.num_heads * head_dim;

        // Cache-write inputs: K[kv_ch,S] fp16, V[kv_ch,S] fp16, R[hd,hd] fp16
        // Cache-write outputs: K_q[kv_ch,S] fp16, V_q[kv_ch,S] fp16
        let cw_alloc_size = crate::tensor::uniform_alloc_size(&[
            ([1, kv_ch, 1, MIN_IO_SEQ], ScalarType::Float16),
            ([1, kv_ch, 1, MIN_IO_SEQ], ScalarType::Float16),
            ([1, 1, head_dim, head_dim], ScalarType::Float16),
        ]);

        // Attention inputs: Q[q_ch,S] fp16, K_cache[kv_ch,max_seq] i8,
        //   V_cache[kv_ch,max_seq] i8, R_inv[hd,hd] fp16
        // Attention output: out[q_ch,S] fp16
        let attn_alloc_size = crate::tensor::uniform_alloc_size(&[
            ([1, q_ch, 1, MIN_IO_SEQ], ScalarType::Float16),
            ([1, kv_ch, 1, config.max_seq_len], ScalarType::Int8),
            ([1, kv_ch, 1, config.max_seq_len], ScalarType::Int8),
            ([1, 1, head_dim, head_dim], ScalarType::Float16),
        ]);

        // --- KV cache with uniform attention alloc ---
        let cache = KvCacheManager::new_with_alloc(config.clone(), attn_alloc_size)?;

        // --- Rotation matrix IOSurface tensors ---
        let rot_data = &cw_weights[0].1;
        let mut rotation_tensor =
            AneTensor::new_with_min_alloc(head_dim, head_dim, ScalarType::Float16, cw_alloc_size)?;
        rotation_tensor.write_bytes_at(0, rot_data)?;

        let unrot_data = &attn_weights[0].1;
        let mut unrotation_tensor = AneTensor::new_with_min_alloc(
            head_dim,
            head_dim,
            ScalarType::Float16,
            attn_alloc_size,
        )?;
        unrotation_tensor.write_bytes_at(0, unrot_data)?;

        // Pre-allocate staging buffers for external tensors (Q, K_proj, V_proj)
        // that arrive from pre-attn sub-programs with a different alloc size.
        let kv_channels = config.num_kv_heads * head_dim;
        let q_channels = config.num_heads * head_dim;
        let cw_k_staging = AneTensor::new_with_min_alloc(
            kv_channels,
            MIN_IO_SEQ,
            ScalarType::Float16,
            cw_alloc_size,
        )?;
        let cw_v_staging = AneTensor::new_with_min_alloc(
            kv_channels,
            MIN_IO_SEQ,
            ScalarType::Float16,
            cw_alloc_size,
        )?;
        let attn_q_staging = AneTensor::new_with_min_alloc(
            q_channels,
            MIN_IO_SEQ,
            ScalarType::Float16,
            attn_alloc_size,
        )?;

        // Pre-allocate output tensors for step_attention() reuse.
        // These are overwritten on every eval call, so sharing across layers is safe.
        let cw_k_output = AneTensor::new_with_min_alloc(
            kv_channels,
            MIN_IO_SEQ,
            ScalarType::Float16,
            cw_alloc_size,
        )?;
        let cw_v_output = AneTensor::new_with_min_alloc(
            kv_channels,
            MIN_IO_SEQ,
            ScalarType::Float16,
            cw_alloc_size,
        )?;

        Ok(Self {
            config,
            cache,
            cache_write_program,
            attention_program,
            qjl_program,
            rotation_tensor,
            unrotation_tensor,
            cw_alloc_size,
            attn_alloc_size,
            cw_k_staging,
            cw_v_staging,
            attn_q_staging,
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
    /// **Note:** Does not advance `seq_pos`. Use [`step`] for full-model
    /// inference, which advances after all layers.
    pub fn step_attention(
        &mut self,
        layer: usize,
        q: &AneTensor,
        k_proj: &AneTensor,
        v_proj: &AneTensor,
    ) -> Result<AneTensor> {
        // Copy column 0 directly from external tensors into staging buffers.
        // ANE requires all input IOSurfaces in a single eval to have the same
        // allocation size. Pre-attn outputs may have a different alloc size.
        // Direct IOSurface-to-IOSurface strided copy avoids CPU Vec intermediaries.
        self.cw_k_staging.copy_column0_from(k_proj)?;
        self.cw_v_staging.copy_column0_from(v_proj)?;

        // 1. Cache-write: K_proj, V_proj, rotation_matrix → K_quant, V_quant
        self.runtime.eval(
            &self.cache_write_program,
            &[
                &self.cw_k_staging,
                &self.cw_v_staging,
                &self.rotation_tensor,
            ],
            &mut [&mut self.cw_k_output, &mut self.cw_v_output],
        )?;

        // 2. CPU cache interception: read fp16 values (rounded to integer),
        //    convert to INT8 bytes, write to persistent cache.
        //    Output is S=32 padded; read only column 0 to get actual values.
        let k_f16 = self.cw_k_output.read_column0_f16()?;
        let k_bytes: Vec<u8> = k_f16.iter().map(|v| (v.to_f32() as i8) as u8).collect();
        let v_f16 = self.cw_v_output.read_column0_f16()?;
        let v_bytes: Vec<u8> = v_f16.iter().map(|v| (v.to_f32() as i8) as u8).collect();

        // Read original K_proj for QJL residual sign computation
        let k_original = if self.config.enable_qjl {
            Some(k_proj.read_f16()?)
        } else {
            None
        };

        self.cache
            .update_cache(layer, &k_bytes, &v_bytes, k_original.as_deref())?;

        // 3. Attention: Q + cached K/V + unrotation_matrix → output
        self.attn_q_staging.copy_column0_from(q)?;
        let q_channels = self.config.num_heads * self.config.head_dim;
        let (k_cache, v_cache) = self.cache.cache_tensors(layer);
        let mut attn_out = AneTensor::new_with_min_alloc(
            q_channels,
            MIN_IO_SEQ,
            ScalarType::Float16,
            self.attn_alloc_size,
        )?;
        self.runtime.eval(
            &self.attention_program,
            &[
                &self.attn_q_staging,
                k_cache,
                v_cache,
                &self.unrotation_tensor,
            ],
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

    /// Required uniform allocation sizes for external tensors.
    ///
    /// Returns `(cache_write_alloc, attention_alloc)`. All user-provided
    /// tensors in a `step_attention` call must use these sizes:
    /// - K_proj, V_proj: use `cache_write_alloc`
    /// - Q: use `attention_alloc`
    pub fn alloc_sizes(&self) -> (usize, usize) {
        (self.cw_alloc_size, self.attn_alloc_size)
    }
}
