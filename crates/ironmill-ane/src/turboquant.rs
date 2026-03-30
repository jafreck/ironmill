//! TurboQuant INT8 KV cache compression configuration and KV cache management.

use mil_rs::ffi::ane::AneCompiler;
use mil_rs::ir::ScalarType;
use mil_rs::ir::passes::beta_quantizer::{beta_optimal_boundaries, beta_optimal_levels};

use crate::program::{CompiledProgram, LoadedProgram};
use crate::runtime::AneRuntime;
use crate::tensor::AneTensor;
use crate::turboquant_mil;
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
    /// Precomputed Hadamard rotation signs for the seed.
    rotation_signs: Vec<f32>,
    /// Optional: per-layer QJL residual sign caches (fp16 ±1).
    qjl_sign_caches: Option<Vec<AneTensor>>,
}

impl KvCacheManager {
    /// Create a new `KvCacheManager`, allocating per-layer INT8 KV cache
    /// tensors and precomputing quantization tables.
    pub fn new(config: TurboQuantConfig) -> Result<Self> {
        let channels = config.num_kv_heads * config.head_dim;

        let mut k_caches = Vec::with_capacity(config.num_layers);
        let mut v_caches = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            k_caches.push(AneTensor::new(
                channels,
                config.max_seq_len,
                ScalarType::Int8,
            )?);
            v_caches.push(AneTensor::new(
                channels,
                config.max_seq_len,
                ScalarType::Int8,
            )?);
        }

        let quant_levels = beta_optimal_levels(config.head_dim, config.n_bits);
        let quant_boundaries = beta_optimal_boundaries(config.head_dim, config.n_bits);

        // TODO: extract rotation signs from Hadamard transform once the
        // `generate_signs` helper in `mil_rs::ir::passes::rotation` is made public.
        let rotation_signs = Vec::new();

        let qjl_sign_caches = if config.enable_qjl {
            let mut caches = Vec::with_capacity(config.num_layers);
            for _ in 0..config.num_layers {
                caches.push(AneTensor::new(
                    channels,
                    config.max_seq_len,
                    ScalarType::Float16,
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
            rotation_signs,
            qjl_sign_caches,
        })
    }

    /// Write one token's worth of quantized K and V data into `layer`'s
    /// caches at the current sequence position, then advance the position.
    pub fn update_cache(
        &mut self,
        layer: usize,
        k_quantized: &[u8],
        v_quantized: &[u8],
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
        self.seq_pos += 1;

        Ok(())
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
    /// The ANE runtime handle.
    runtime: AneRuntime,
}

impl TurboQuantModel {
    /// Compile and load the TurboQuant sub-programs onto the ANE.
    ///
    /// 1. Creates an [`AneRuntime`].
    /// 2. Emits + compiles the cache-write MIL program.
    /// 3. Emits + compiles the attention MIL program.
    /// 4. Allocates the [`KvCacheManager`].
    pub fn compile(config: TurboQuantConfig) -> Result<Self> {
        let runtime = AneRuntime::new()?;

        // --- cache-write sub-program ---
        let (cw_mil, cw_weights) = turboquant_mil::emit_cache_write_mil(&config);
        let cw_weight_refs: Vec<(&str, &[u8])> = cw_weights
            .iter()
            .map(|(name, data)| (name.as_str(), data.as_slice()))
            .collect();
        let cw_ptr = AneCompiler::compile_mil_text(&cw_mil, &cw_weight_refs).map_err(|e| {
            AneError::CompileFailed {
                status: 0,
                context: format!("cache-write compilation failed: {e}"),
            }
        })?;
        let cw_compiled = unsafe { CompiledProgram::from_raw(cw_ptr) };
        let cache_write_program = runtime.load_program(&cw_compiled)?;

        // --- attention sub-program ---
        let (attn_mil, attn_weights) =
            turboquant_mil::emit_attention_mil(&config, config.max_seq_len);
        let attn_weight_refs: Vec<(&str, &[u8])> = attn_weights
            .iter()
            .map(|(name, data)| (name.as_str(), data.as_slice()))
            .collect();
        let attn_ptr =
            AneCompiler::compile_mil_text(&attn_mil, &attn_weight_refs).map_err(|e| {
                AneError::CompileFailed {
                    status: 0,
                    context: format!("attention compilation failed: {e}"),
                }
            })?;
        let attn_compiled = unsafe { CompiledProgram::from_raw(attn_ptr) };
        let attention_program = runtime.load_program(&attn_compiled)?;

        // --- KV cache ---
        let cache = KvCacheManager::new(config.clone())?;

        Ok(Self {
            config,
            cache,
            cache_write_program,
            attention_program,
            qjl_program: None,
            runtime,
        })
    }

    /// Process one token through cache-write and attention for a single layer.
    ///
    /// Takes Q, K_proj, V_proj as fp16 [`AneTensor`]s (single token).
    /// Returns attention output as fp16 [`AneTensor`].
    pub fn step_attention(
        &mut self,
        layer: usize,
        q: &AneTensor,
        k_proj: &AneTensor,
        v_proj: &AneTensor,
    ) -> Result<AneTensor> {
        let channels = self.config.num_kv_heads * self.config.head_dim;

        // 1. Cache-write: K_proj, V_proj → K_quant, V_quant (INT8)
        let mut k_quant = AneTensor::new(channels, 1, ScalarType::Int8)?;
        let mut v_quant = AneTensor::new(channels, 1, ScalarType::Int8)?;
        self.runtime.eval(
            &self.cache_write_program,
            &[k_proj, v_proj],
            &mut [&mut k_quant, &mut v_quant],
        )?;

        // 2. CPU cache interception: copy INT8 bytes into persistent cache
        let k_bytes = k_quant.read_bytes_at(0, channels)?;
        let v_bytes = v_quant.read_bytes_at(0, channels)?;
        self.cache.update_cache(layer, &k_bytes, &v_bytes)?;

        // 3. Attention: Q + cached K/V → output
        let (k_cache, v_cache) = self.cache.cache_tensors(layer);
        let q_channels = self.config.num_heads * self.config.head_dim;
        let mut attn_out = AneTensor::new(q_channels, 1, ScalarType::Float16)?;
        self.runtime.eval(
            &self.attention_program,
            &[q, k_cache, v_cache],
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

    /// Get a reference to the config.
    pub fn config(&self) -> &TurboQuantConfig {
        &self.config
    }
}
