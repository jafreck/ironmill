//! Metal GPU inference engine — struct definition and trait impls.
//!
//! Contains [`MetalInference`], [`MetalArtifacts`], and the
//! [`InferenceEngine`] / [`CalibratingEngine`] trait implementations.
//! Loading, calibration, and pipeline code live in sibling modules.

use ironmill_metal_sys::{MetalBuffer, MetalDevice};
use mil_rs::weights::ModelConfig;

use super::buffers::{IntermediateBuffers, MpsMatmulCache};
use super::config::{Gemma4Config, MetalConfig};
use super::error::MetalError;
use super::gdn::GdnState;
use super::kv_cache::Fp16KvCache;
use super::mla::{MlaConfig, MlaKvCache};
use super::plan::{LayerPlan, ModelPlan};
use super::turboquant::{MetalKvCache, MetalTurboQuantModel};
use super::weights::MetalWeights;
use crate::calibration::ActivationHook;
use crate::engine::{InferenceEngine, InferenceError};
use crate::types::Logits;
use ironmill_core::model_info::ModelInfo;

// ── Public artifacts type for load() ────────────────────────────

/// Artifacts passed to [`MetalInference::load`] via the type-erased
/// [`InferenceEngine`] interface.
pub struct MetalArtifacts<'a> {
    /// Weight provider for loading model tensors.
    pub weights: &'a dyn mil_rs::weights::WeightProvider,
    /// Metal backend configuration.
    pub config: MetalConfig,
}

// ── MetalInference ────────────────────────────────────────────────

/// Metal GPU inference engine.
///
/// Implements the full transformer decode pipeline using Metal compute
/// shaders for element-wise ops and MPS for matrix multiplication.
pub struct MetalInference {
    pub(crate) device: MetalDevice,
    pub(crate) queue: ironmill_metal_sys::CommandQueue,
    pub(crate) pipelines: Option<super::ops::MetalPipelines>,
    /// Second pipeline set for global layers with a different HEAD_DIM (Gemma 4).
    pub(crate) global_pipelines: Option<super::ops::MetalPipelines>,
    /// The head_dim that `global_pipelines` was compiled for (0 if not set).
    pub(crate) global_head_dim: usize,
    pub(crate) weights: Option<MetalWeights>,
    pub(crate) turboquant: Option<MetalTurboQuantModel>,
    pub(crate) kv_cache: Option<MetalKvCache>,
    pub(crate) fp16_kv_cache: Option<Fp16KvCache>,
    pub(crate) intermediate_buffers: Option<IntermediateBuffers>,
    pub(crate) rope_cos: Option<MetalBuffer>,
    pub(crate) rope_sin: Option<MetalBuffer>,
    /// RoPE tables for global (full_attention) layers with different theta.
    pub(crate) global_rope_cos: Option<MetalBuffer>,
    pub(crate) global_rope_sin: Option<MetalBuffer>,
    /// Unit-weight buffer for scale-free RMSNorm (e.g., Gemma 4 V-norm).
    /// Contains [1.0; max_head_dim] in FP16.
    pub(crate) unit_norm_weight: Option<MetalBuffer>,
    /// MPS matmul cache for prefill (variable token_count > 1).
    pub(crate) decode_matmuls: Option<MpsMatmulCache>,
    /// MPS matmul cache for single-token decode (token_count=1), preserved
    /// across prefill→decode transitions so it never needs rebuilding.
    pub(crate) decode_matmuls_t1: Option<MpsMatmulCache>,
    pub(crate) config: MetalConfig,
    pub(crate) model_config: Option<ModelConfig>,
    /// MLA configuration (set when the model uses Multi-Head Latent Attention).
    pub(crate) mla_config: Option<MlaConfig>,
    /// Gemma 4 per-layer attention configuration.
    pub(crate) gemma4_config: Option<Gemma4Config>,
    /// MLA compressed KV cache (used instead of fp16_kv_cache for MLA models).
    pub(crate) mla_kv_cache: Option<MlaKvCache>,
    /// Pre-allocated buffer for FP16 logits readback.
    pub(crate) logits_fp16_buf: Vec<u8>,
    /// Pre-allocated buffer for serializing token IDs to GPU.
    pub(crate) token_bytes_buf: Vec<u8>,
    pub(crate) seq_pos: usize,
    /// Cached model info, populated during `load()`.
    pub(crate) model_info: Option<ModelInfo>,
    /// GDN (Gated Delta Network) recurrent state for linear-attention layers.
    pub(crate) gdn_state: Option<GdnState>,
    /// Per-layer execution plan, resolved at load time.
    pub(crate) layer_plans: Vec<LayerPlan>,
    /// Model-level execution plan, resolved at load time.
    pub(crate) model_plan: Option<ModelPlan>,
    /// D2Quant Deviation-Aware Correction (DAC) per-layer bias vectors.
    /// Each buffer is `[hidden_size]` FP16. Added to post-attention LayerNorm
    /// output to compensate for quantization-induced mean shift.
    pub(crate) dac_biases: Option<Vec<MetalBuffer>>,
    /// Reusable staging buffers for [`GpuCalibrationEngine::run_single_layer`].
    /// Lazily allocated on first calibration call to avoid polluting the
    /// default engine construction path.
    pub(crate) calib_scratch: Option<CalibScratchBuffers>,
    /// Recommended total threadgroups to saturate this GPU (cached at init).
    pub(crate) gpu_max_threadgroups: usize,
}

/// Reusable Shared-mode buffers for `run_single_layer` staging.
///
/// Avoids allocating two new Metal buffers per GPU forward during
/// calibration alpha search (~1000+ calls per model).
pub(crate) struct CalibScratchBuffers {
    /// Shared-mode buffer for uploading input hidden state to the GPU.
    pub staging_in: MetalBuffer,
    /// Shared-mode buffer for reading back output hidden state.
    pub readback: MetalBuffer,
    /// Capacity in bytes of each buffer.
    pub capacity: usize,
}

impl MetalInference {
    /// Access compiled pipelines — returns `NotLoaded` if `load()` hasn't been called yet.
    pub(crate) fn pipelines(&self) -> Result<&super::ops::MetalPipelines, InferenceError> {
        self.pipelines.as_ref().ok_or(InferenceError::NotLoaded)
    }

    /// Create a new GPU inference engine (device + queue + shader pipelines).
    pub fn new(config: MetalConfig) -> Result<Self, MetalError> {
        config
            .validate()
            .map_err(|e| MetalError::Config(e.to_string()))?;
        let device = MetalDevice::system_default().map_err(MetalError::Metal)?;
        let queue = device.create_command_queue().map_err(MetalError::Metal)?;
        // Pipelines are compiled in load() once head_dim is known.
        let gpu_max_threadgroups = device.recommended_max_working_set_threadgroups();
        Ok(Self {
            device,
            queue,
            pipelines: None,
            global_pipelines: None,
            global_head_dim: 0,
            weights: None,
            turboquant: None,
            kv_cache: None,
            fp16_kv_cache: None,
            intermediate_buffers: None,
            rope_cos: None,
            rope_sin: None,
            global_rope_cos: None,
            global_rope_sin: None,
            unit_norm_weight: None,
            decode_matmuls: None,
            decode_matmuls_t1: None,
            config,
            model_config: None,
            mla_config: None,
            gemma4_config: None,
            mla_kv_cache: None,
            logits_fp16_buf: Vec::new(),
            token_bytes_buf: Vec::new(),
            seq_pos: 0,
            model_info: None,
            gdn_state: None,
            layer_plans: Vec::new(),
            model_plan: None,
            dac_biases: None,
            calib_scratch: None,
            gpu_max_threadgroups,
        })
    }

    // ── Memory query ─────────────────────────────────────────────

    /// Returns the current Metal device allocation size in bytes.
    pub fn gpu_allocated_bytes(&self) -> usize {
        self.device.current_allocated_size()
    }
}

// ── InferenceEngine trait impl ─────────────────────────────────

impl InferenceEngine for MetalInference {
    fn decode_step(&mut self, token: u32) -> Result<Logits, InferenceError> {
        self.run_pipeline(&[token])
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<Logits, InferenceError> {
        if tokens.is_empty() {
            return Err(InferenceError::Decode("empty prefill tokens".into()));
        }

        // With dynamic buffer resizing, process all tokens in one shot
        // unless the caller explicitly set a chunk size.
        let chunk_size = self.config.prefill_chunk_size.unwrap_or(tokens.len());
        let chunk_size = chunk_size.max(1);

        // SWA: prefill chunks must not exceed the smallest window_size so
        // that ring-buffer writes and attention reads stay in bounds.
        let chunk_size = if let Some(ref mc) = self.model_config {
            let min_ws = (0..mc.num_hidden_layers)
                .map(|l| self.config.layer_window_size(l))
                .filter(|&ws| ws > 0)
                .min();
            match min_ws {
                Some(ws) => chunk_size.min(ws),
                None => chunk_size,
            }
        } else {
            chunk_size
        };

        let chunks: Vec<&[u32]> = tokens.chunks(chunk_size).collect();
        let n_chunks = chunks.len();

        if n_chunks <= 1 {
            // Single chunk — no pipelining needed.
            return self.run_pipeline(chunks[0]);
        }

        // Pipelined chunked prefill: submit non-last chunks with
        // skip_wait=true so the GPU executes while the CPU encodes the
        // next chunk. Alternate token ID buffers to avoid write-after-read
        // hazards on the Shared buffer.
        for (i, chunk) in chunks[..n_chunks - 1].iter().enumerate() {
            let use_alt = i % 2 != 0;
            self.run_pipeline_inner(chunk, true, use_alt, true)?;
        }

        // Last chunk: use the opposite buffer parity, wait for completion,
        // and read back logits.
        let last_alt = (n_chunks - 1) % 2 != 0;
        self.run_pipeline_inner(chunks[n_chunks - 1], false, last_alt, false)
    }

    fn reset(&mut self) {
        self.seq_pos = 0;
        if let Some(kv) = self.kv_cache.as_mut() {
            kv.reset();
        }
        if let Some(fp16_kv) = self.fp16_kv_cache.as_mut() {
            fp16_kv.reset();
        }
        if let Some(mla_kv) = self.mla_kv_cache.as_mut() {
            mla_kv.reset();
        }
        if let Some(gdn) = self.gdn_state.as_ref() {
            if let Err(e) = gdn.reset() {
                eprintln!("warning: GDN state reset failed: {e}");
            }
        }
    }

    fn seq_pos(&self) -> usize {
        self.seq_pos
    }

    fn truncate_to(&mut self, pos: usize) -> Result<(), InferenceError> {
        if pos > self.seq_pos {
            return Err(InferenceError::runtime(format!(
                "cannot truncate forward: pos {pos} > seq_pos {}",
                self.seq_pos
            )));
        }
        self.seq_pos = pos;
        if let Some(kv) = self.kv_cache.as_mut() {
            kv.truncate_to(pos);
        }
        if let Some(fp16_kv) = self.fp16_kv_cache.as_mut() {
            fp16_kv.truncate_to(pos);
        }
        if let Some(mla_kv) = self.mla_kv_cache.as_mut() {
            mla_kv.truncate_to(pos);
        }
        Ok(())
    }

    fn model_info(&self) -> Result<&ModelInfo, InferenceError> {
        self.model_info.as_ref().ok_or(InferenceError::NotLoaded)
    }
}

// ── CalibratingEngine trait impl ───────────────────────────────

impl crate::calibration::CalibratingEngine for MetalInference {
    fn prefill_with_hooks(
        &mut self,
        tokens: &[u32],
        hooks: &mut dyn ActivationHook,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.prefill_with_hooks(tokens, hooks)?;
        Ok(())
    }

    fn reset(&mut self) {
        InferenceEngine::reset(self);
    }
}
