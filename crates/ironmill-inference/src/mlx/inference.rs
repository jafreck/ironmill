//! MLX inference engine implementing the [`InferenceEngine`] trait.
//!
//! Runs the full LLaMA-family transformer decode pipeline using MLX
//! built-in operations with lazy evaluation. A single `eval()` call
//! at the end materializes the logits.

use std::any::Any;
use std::time::Instant;

use ironmill_mlx_sys::{
    MlxArray, MlxDtype, MlxStream, add, broadcast_to, concat, expand_dims, matmul, multiply,
    reshape, rms_norm, rope, scaled_dot_product_attention, silu, slice, transpose, transpose_axes,
};
use mil_rs::weights::WeightProvider;

use super::config::MlxConfig;
use super::error::MlxError;
use super::turboquant::{MlxKvCache, MlxTurboQuantModel};
use super::weights::{MlxWeightBuffer, MlxWeights};
use crate::engine::{InferenceEngine, InferenceError};
use crate::types::Logits;

// ── PolarQuant kernel source ────────────────────────────────────

/// All PolarQuant kernel variants, loaded from `shaders/mlx_polarquant.metal`.
/// Individual kernels are selected by name when calling `metal_kernel()`.
const POLARQUANT_SOURCE: &str = include_str!("shaders/mlx_polarquant.metal");

// ── Public artifacts type for load() ────────────────────────────

/// Artifacts passed to [`MlxInference::load`] via the type-erased
/// [`InferenceEngine`] interface.
pub struct MlxArtifacts<'a> {
    /// Weight provider supplying named tensors.
    pub weights: &'a dyn WeightProvider,
    /// MLX backend configuration.
    pub config: MlxConfig,
}

// ── MlxInference ────────────────────────────────────────────────

/// MLX GPU inference engine.
///
/// Implements the full transformer decode pipeline using MLX lazy
/// operations, materializing results with a single `eval()` call.
pub struct MlxInference {
    stream: Option<MlxStream>,
    weights: Option<MlxWeights>,
    config: MlxConfig,
    /// Per-layer K cache: `[batch=1, num_kv_heads, seq_len, head_dim]` FP16.
    k_cache: Vec<Option<MlxArray>>,
    /// Per-layer V cache: `[batch=1, num_kv_heads, seq_len, head_dim]` FP16.
    v_cache: Vec<Option<MlxArray>>,
    seq_pos: usize,
    /// TurboQuant model state (initialized when `enable_turboquant` is true).
    tq_model: Option<MlxTurboQuantModel>,
    /// TurboQuant quantized KV cache (initialized when `enable_turboquant` is true).
    tq_cache: Option<MlxKvCache>,
    /// Profiling: total eval() calls in the current pipeline invocation.
    profile_eval_count: usize,
    /// Profiling: cumulative time spent in eval() calls.
    profile_eval_duration: std::time::Duration,
}

impl MlxInference {
    /// Create a new MLX inference engine with the given configuration.
    pub fn new(config: MlxConfig) -> Result<Self, MlxError> {
        config.validate().map_err(MlxError::Config)?;
        Ok(Self {
            stream: None,
            weights: None,
            config,
            k_cache: Vec::new(),
            v_cache: Vec::new(),
            seq_pos: 0,
            tq_model: None,
            tq_cache: None,
            profile_eval_count: 0,
            profile_eval_duration: std::time::Duration::ZERO,
        })
    }

    /// Load weights from a provider (convenience method that avoids
    /// the `&dyn Any` lifetime constraints of `InferenceEngine::load`).
    pub fn load_weights(
        &mut self,
        weights: &dyn WeightProvider,
        config: MlxConfig,
    ) -> Result<(), MlxError> {
        self.config = config;

        let stream = MlxStream::default_gpu()?;
        let loaded = MlxWeights::load(weights, &stream)?;
        let num_layers = loaded.config.num_hidden_layers;

        self.k_cache = vec![None; num_layers];
        self.v_cache = vec![None; num_layers];

        if self.config.enable_turboquant {
            let tq_model = MlxTurboQuantModel::new(&loaded.config, &self.config, &stream)?;
            let tq_cache = MlxKvCache::new(
                &tq_model,
                loaded.config.num_key_value_heads,
                self.config.max_seq_len,
                loaded.config.head_dim,
                num_layers,
                &stream,
            )?;
            self.tq_model = Some(tq_model);
            self.tq_cache = Some(tq_cache);
        }

        self.stream = Some(stream);
        self.weights = Some(loaded);
        self.seq_pos = 0;

        Ok(())
    }

    /// Run the transformer pipeline for a batch of tokens.
    ///
    /// This builds a lazy MLX computation graph and evaluates it with
    /// a single `eval()` at the end. When `use_async_final` is `true`,
    /// the final evaluation uses `async_eval()` to overlap with the
    /// caller's next graph construction (used for prefill chunking).
    fn run_pipeline(&mut self, tokens: &[u32], use_async_final: bool) -> Result<Logits, MlxError> {
        // Reset profiling counters for this invocation.
        self.profile_eval_count = 0;
        self.profile_eval_duration = std::time::Duration::ZERO;

        // Take the stream and weights out temporarily to avoid borrow conflicts
        // between immutable references to them and mutable access to KV caches.
        let stream = self
            .stream
            .take()
            .ok_or_else(|| MlxError::WeightLoading("stream not initialized".into()))?;
        let weights = self
            .weights
            .take()
            .ok_or_else(|| MlxError::WeightLoading("weights not loaded".into()))?;
        let tq_model = self.tq_model.take();

        let result = self.run_pipeline_inner(
            tokens,
            &stream,
            &weights,
            tq_model.as_ref(),
            use_async_final,
        );

        // Put them back regardless of outcome.
        self.stream = Some(stream);
        self.weights = Some(weights);
        self.tq_model = tq_model;

        // Log profiling stats if enabled.
        if self.config.profile {
            eprintln!(
                "[mlx-profile] eval calls: {}, eval time: {:.3}ms, async_final: {}",
                self.profile_eval_count,
                self.profile_eval_duration.as_secs_f64() * 1000.0,
                use_async_final,
            );
        }

        result
    }

    /// Inner pipeline logic operating on borrowed stream/weights, allowing
    /// `&mut self` access to KV caches simultaneously.
    fn run_pipeline_inner(
        &mut self,
        tokens: &[u32],
        stream: &MlxStream,
        weights: &MlxWeights,
        tq_model: Option<&MlxTurboQuantModel>,
        use_async_final: bool,
    ) -> Result<Logits, MlxError> {
        let mc = &weights.config;
        let num_tokens = tokens.len();
        let hidden_size = mc.hidden_size;
        let num_heads = mc.num_attention_heads;
        let num_kv_heads = mc.num_key_value_heads;
        let head_dim = mc.head_dim;
        let rms_eps = mc.rms_norm_eps as f32;
        let rope_theta = mc.rope_theta as f32;
        let num_groups = num_heads / num_kv_heads;

        // ── Embedding lookup via slice ──────────────────────────
        // Build embedding for each token by slicing rows from the table.
        let mut hidden = self.gather_embeddings(&weights.embedding, tokens, stream)?;

        // ── Per-layer transformer block ─────────────────────────
        let use_turboquant = self.config.enable_turboquant && tq_model.is_some();
        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            // ── Lazy region A: RMSNorm + Q/K/V projections + RoPE ──
            let normed = rms_norm(&hidden, &layer.input_norm, rms_eps, stream)?;

            let q = self.weight_matmul(&normed, &layer.q_proj, stream)?;
            let k = self.weight_matmul(&normed, &layer.k_proj, stream)?;
            let v = self.weight_matmul(&normed, &layer.v_proj, stream)?;

            // Reshape to [batch=1, num_tokens, num_heads, head_dim]
            let q = reshape(
                &q,
                &[1, num_tokens as i32, num_heads as i32, head_dim as i32],
                stream,
            )?;
            let k = reshape(
                &k,
                &[1, num_tokens as i32, num_kv_heads as i32, head_dim as i32],
                stream,
            )?;
            let v = reshape(
                &v,
                &[1, num_tokens as i32, num_kv_heads as i32, head_dim as i32],
                stream,
            )?;

            // Optional Q/K normalization (Qwen3 QK norm)
            let q = if let Some(ref q_norm_w) = layer.q_norm {
                self.per_head_rms_norm(
                    &q, q_norm_w, rms_eps, num_heads, head_dim, num_tokens, stream,
                )?
            } else {
                q
            };
            let k = if let Some(ref k_norm_w) = layer.k_norm {
                self.per_head_rms_norm(
                    &k,
                    k_norm_w,
                    rms_eps,
                    num_kv_heads,
                    head_dim,
                    num_tokens,
                    stream,
                )?
            } else {
                k
            };

            // RoPE: expects [batch, seq, heads, head_dim]
            let offset = self.seq_pos as i32;
            let q = rope(&q, head_dim as i32, false, rope_theta, 1.0, offset, stream)?;
            let k = rope(&k, head_dim as i32, false, rope_theta, 1.0, offset, stream)?;

            let attn_out = if use_turboquant {
                // ── TurboQuant path: quantized KV cache + custom attention ──
                let tq = tq_model.unwrap();
                let tq_cache = self
                    .tq_cache
                    .as_ref()
                    .expect("tq_cache must be initialized when turboquant is enabled");

                // Flatten K/V from [1, T, num_kv_heads, head_dim] to
                // [num_kv_heads × head_dim] for cache write (single token decode).
                let k_flat = reshape(&k, &[(num_kv_heads * head_dim) as i32], stream)?;
                let v_flat = reshape(&v, &[(num_kv_heads * head_dim) as i32], stream)?;

                // Dispatch cache writes: K then V (lazy)
                let k_dummy = tq_cache.write_kv(layer_idx, &k_flat, true, tq, stream)?;
                let v_dummy = tq_cache.write_kv(layer_idx, &v_flat, false, tq, stream)?;

                // Materialize cache writes before attend() reads them.
                // A previous batched-eval optimization deferred this eval,
                // but that caused attend() to read stale/uninitialized cache
                // data when eval_interval > 1. Per-layer eval is required
                // for correctness. See turboquant_eval_interval doc comment.
                let eval_start = if self.config.profile {
                    Some(Instant::now())
                } else {
                    None
                };
                ironmill_mlx_sys::stream::eval(&[&k_dummy, &v_dummy])?;
                if let Some(t) = eval_start {
                    self.profile_eval_duration += t.elapsed();
                }
                self.profile_eval_count += 1;

                // Flatten Q from [1, T, num_heads, head_dim] to [num_heads × head_dim]
                let q_flat = reshape(&q, &[(num_heads * head_dim) as i32], stream)?;

                // Dispatch attention kernel
                let attn_flat = tq_cache.attend(layer_idx, &q_flat, num_heads, tq, stream)?;

                // Reshape back to [T, num_heads * head_dim]
                reshape(
                    &attn_flat,
                    &[num_tokens as i32, (num_heads * head_dim) as i32],
                    stream,
                )?
            } else {
                // ── FP16 SDPA path (original) ──
                let q = self.swap_dims_1_2(&q, stream)?;
                let k = self.swap_dims_1_2(&k, stream)?;
                let v = self.swap_dims_1_2(&v, stream)?;

                let (k_full, v_full) = self.update_kv_cache(layer_idx, &k, &v, stream)?;

                let k_for_attn = if num_groups > 1 {
                    self.expand_kv_for_gqa(&k_full, num_heads, num_kv_heads, stream)?
                } else {
                    k_full
                };
                let v_for_attn = if num_groups > 1 {
                    self.expand_kv_for_gqa(&v_full, num_heads, num_kv_heads, stream)?
                } else {
                    v_full
                };

                let scale = 1.0 / (head_dim as f32).sqrt();
                let attn_out = scaled_dot_product_attention(
                    &q,
                    &k_for_attn,
                    &v_for_attn,
                    scale,
                    None,
                    stream,
                )?;

                let attn_out = self.swap_dims_1_2(&attn_out, stream)?;
                reshape(
                    &attn_out,
                    &[num_tokens as i32, (num_heads * head_dim) as i32],
                    stream,
                )?
            };

            // ── Lazy region B: O proj + residual + FFN ──
            let attn_proj = self.weight_matmul(&attn_out, &layer.o_proj, stream)?;
            let hidden_post_attn = add(&hidden, &attn_proj, stream)?;

            let normed_ff = rms_norm(&hidden_post_attn, &layer.post_attn_norm, rms_eps, stream)?;

            let gate = self.weight_matmul(&normed_ff, &layer.gate_proj, stream)?;
            let up = self.weight_matmul(&normed_ff, &layer.up_proj, stream)?;
            let gate_activated = silu(&gate, stream)?;
            let ff_mid = multiply(&gate_activated, &up, stream)?;
            let ff_out = self.weight_matmul(&ff_mid, &layer.down_proj, stream)?;

            hidden = add(&hidden_post_attn, &ff_out, stream)?;
        }

        // ── Final norm + LM head ────────────────────────────────
        let final_normed = rms_norm(&hidden, &weights.final_norm, rms_eps, stream)?;

        // For decode, we only need the last token's logits
        let last_hidden = if num_tokens > 1 {
            slice(
                &final_normed,
                &[(num_tokens - 1) as i32, 0],
                &[num_tokens as i32, hidden_size as i32],
                &[1, 1],
                stream,
            )?
        } else {
            final_normed
        };

        // LM head matmul: [1, hidden_size] @ [hidden_size, vocab_size]^T
        let logits_arr = self.weight_matmul(&last_hidden, &weights.lm_head, stream)?;

        // ── Single eval() to materialize ────────────────────────
        if use_async_final {
            self.profiled_async_eval(&[&logits_arr])?;
        } else {
            self.profiled_eval(&[&logits_arr])?;
        }

        // Read logits back to CPU
        // SAFETY: logits_arr is f32 dtype (produced by lm_head matmul with f32 cast)
        #[allow(unsafe_code)]
        let logits_f32: &[f32] = unsafe { logits_arr.as_contiguous_slice()? };
        let logits = logits_f32.to_vec();

        // Advance sequence position
        self.seq_pos += num_tokens;
        if let Some(ref mut tq_cache) = self.tq_cache {
            tq_cache.advance_by(num_tokens);
        }

        Ok(logits)
    }

    /// Gather embedding rows for the given token IDs (lazy).
    ///
    /// For single tokens, slices one row. For multiple tokens, slices each
    /// row lazily and concatenates via `concat` — no eval needed.
    fn gather_embeddings(
        &self,
        embedding: &MlxArray,
        tokens: &[u32],
        stream: &MlxStream,
    ) -> Result<MlxArray, MlxError> {
        let shape = embedding.shape();
        let hidden_size = if shape.len() >= 2 { shape[1] } else { shape[0] };
        let num_tokens = tokens.len();

        if num_tokens == 1 {
            // Single token: slice one row from the embedding table.
            let t = tokens[0] as i32;
            let row = slice(
                embedding,
                &[t, 0],
                &[t + 1, hidden_size as i32],
                &[1, 1],
                stream,
            )?;
            let row = reshape(&row, &[1, hidden_size as i32], stream)?;
            return Ok(row);
        }

        // Multiple tokens: lazily slice each row, then concat along axis 0.
        let rows: Vec<MlxArray> = tokens
            .iter()
            .map(|&t| {
                let t = t as i32;
                slice(
                    embedding,
                    &[t, 0],
                    &[t + 1, hidden_size as i32],
                    &[1, 1],
                    stream,
                )
                .map_err(MlxError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let row_refs: Vec<&MlxArray> = rows.iter().collect();
        let gathered = concat(&row_refs, 0, stream)?;
        Ok(gathered)
    }

    /// Perform matmul with a weight buffer (dense or quantized).
    ///
    /// For dense weights: `x @ W^T` (weights are stored as `[out, in]`).
    /// For quantized weights: dispatch via custom Metal kernel.
    fn weight_matmul(
        &self,
        x: &MlxArray,
        weight: &MlxWeightBuffer,
        stream: &MlxStream,
    ) -> Result<MlxArray, MlxError> {
        match weight {
            MlxWeightBuffer::Dense(w) => {
                // Weights are [out_features, in_features], we need x @ W^T.
                let wt = transpose(w, stream)?;
                let result = matmul(x, &wt, stream)?;
                Ok(result)
            }
            MlxWeightBuffer::Quantized(qw) => {
                // Dispatch quantized matmul via custom Metal kernel.
                // This kernel was validated by the MLX-KERNEL-SPIKE task.
                self.quantized_matmul(x, qw, stream)
            }
        }
    }

    /// Quantized matmul using a custom Metal kernel for PolarQuant weights.
    ///
    /// For M=1 (decode): dispatches the SIMD-group matvec kernel — one
    /// threadgroup per output row with 32 threads for lane-strided reduction.
    ///
    /// For M>1 (prefill): dispatches a tiled GEMM kernel — (32, 8) thread
    /// tiles over the output matrix.
    fn quantized_matmul(
        &self,
        x: &MlxArray,
        qw: &super::weights::MlxQuantizedWeight,
        stream: &MlxStream,
    ) -> Result<MlxArray, MlxError> {
        let x_shape = x.shape();
        let m = if x_shape.len() >= 2 { x_shape[0] } else { 1 };
        let n = qw.n;
        let k = qw.k;

        let output_shape = [m, n];
        let output_shapes: &[&[usize]] = &[&output_shape];
        let output_dtypes = [MlxDtype::Float16];

        // Create a small uint32 array for dimension parameters.
        if m == 1 {
            // Decode path: one SIMD group per output row.
            let params_data = [n as u32, k as u32];
            let params_bytes: Vec<u8> = params_data.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let params = MlxArray::from_data_copy(&params_bytes, &[2], MlxDtype::Uint32, stream)?;
            let inputs: &[&MlxArray] = &[x, &qw.indices, &qw.lut, &qw.norms, &params];

            let result = ironmill_mlx_sys::metal_kernel(&ironmill_mlx_sys::MetalKernelParams {
                name: "polarquant_matvec",
                inputs,
                outputs: &[],
                source: POLARQUANT_SOURCE,
                grid: [n, 1, 1],
                threadgroup: [32, 1, 1],
                output_shapes,
                output_dtypes: &output_dtypes,
                stream,
            })?;
            return result.into_iter().next().ok_or_else(|| {
                MlxError::WeightLoading("quantized matvec returned no outputs".into())
            });
        }

        // Prefill path: tiled GEMM.
        let params_data = [m as u32, n as u32, k as u32];
        let params_bytes: Vec<u8> = params_data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let params = MlxArray::from_data_copy(&params_bytes, &[3], MlxDtype::Uint32, stream)?;
        let inputs: &[&MlxArray] = &[x, &qw.indices, &qw.lut, &qw.norms, &params];

        let result = ironmill_mlx_sys::metal_kernel(&ironmill_mlx_sys::MetalKernelParams {
            name: "polarquant_matmul",
            inputs,
            outputs: &[],
            source: POLARQUANT_SOURCE,
            grid: [(n + 31) / 32, (m + 7) / 8, 1],
            threadgroup: [32, 8, 1],
            output_shapes,
            output_dtypes: &output_dtypes,
            stream,
        })?;
        result
            .into_iter()
            .next()
            .ok_or_else(|| MlxError::WeightLoading("quantized matmul returned no outputs".into()))
    }

    /// Swap dimensions 1 and 2 of a 4D tensor: `[B, A, C, D] → [B, C, A, D]` (lazy).
    fn swap_dims_1_2(&self, x: &MlxArray, stream: &MlxStream) -> Result<MlxArray, MlxError> {
        let result = transpose_axes(x, &[0, 2, 1, 3], stream)?;
        Ok(result)
    }

    /// Update KV cache for a layer by concatenating new K/V along the seq dimension.
    ///
    /// K/V shapes: `[1, num_kv_heads, new_tokens, head_dim]`
    /// Cache shapes: `[1, num_kv_heads, cached_tokens, head_dim]`
    fn update_kv_cache(
        &mut self,
        layer_idx: usize,
        k_new: &MlxArray,
        v_new: &MlxArray,
        stream: &MlxStream,
    ) -> Result<(MlxArray, MlxArray), MlxError> {
        // Concatenation along seq dim (axis 2).
        // Since we don't have a concat op, we eval both, merge on CPU,
        // and create a new array. This is correct if not optimal.
        let k_full = self.concat_cache(layer_idx, true, k_new, stream)?;
        let v_full = self.concat_cache(layer_idx, false, v_new, stream)?;

        self.k_cache[layer_idx] = Some(k_full.clone());
        self.v_cache[layer_idx] = Some(v_full.clone());

        Ok((k_full, v_full))
    }

    /// Concatenate a new tensor along axis 2 with the existing cache (lazy).
    fn concat_cache(
        &self,
        layer_idx: usize,
        is_k: bool,
        new: &MlxArray,
        stream: &MlxStream,
    ) -> Result<MlxArray, MlxError> {
        let cache_vec = if is_k { &self.k_cache } else { &self.v_cache };

        match &cache_vec[layer_idx] {
            None => {
                // First token(s): the new tensor IS the cache.
                Ok(new.clone())
            }
            Some(existing) => {
                let result = concat(&[existing, new], 2, stream)?;
                Ok(result)
            }
        }
    }

    /// Expand KV heads for Grouped Query Attention (lazy).
    ///
    /// Repeats each KV head `num_groups` times along the head dimension
    /// so that K/V have the same number of heads as Q.
    /// Uses expand_dims + broadcast_to + reshape — all lazy MLX ops.
    fn expand_kv_for_gqa(
        &self,
        kv: &MlxArray,
        num_heads: usize,
        num_kv_heads: usize,
        stream: &MlxStream,
    ) -> Result<MlxArray, MlxError> {
        let num_groups = num_heads / num_kv_heads;
        if num_groups == 1 {
            return Ok(kv.clone());
        }

        // kv shape: [B, num_kv_heads, S, D]
        // Insert a group dim: [B, num_kv_heads, 1, S, D]
        let expanded = expand_dims(kv, &[2], stream)?;

        // Broadcast the group dim: [B, num_kv_heads, num_groups, S, D]
        // We need the concrete shape for broadcast_to. Since B=1 is fixed
        // and the other dims are known from the model config, read shape
        // metadata (which doesn't require eval — shape is tracked lazily).
        let kv_shape = kv.shape();
        let (b, kv_h, s, d) = (kv_shape[0], kv_shape[1], kv_shape[2], kv_shape[3]);
        let tiled = broadcast_to(
            &expanded,
            &[b as i32, kv_h as i32, num_groups as i32, s as i32, d as i32],
            stream,
        )?;

        // Merge kv_heads and groups: [B, num_heads, S, D]
        let result = reshape(
            &tiled,
            &[b as i32, num_heads as i32, s as i32, d as i32],
            stream,
        )?;
        Ok(result)
    }

    /// Per-head RMS normalization for QK norm (Qwen3).
    ///
    /// Input shape: `[B, T, H, D]`. The norm weight is `[D]`.
    /// We reshape to `[B*T*H, D]`, apply rms_norm, reshape back.
    fn per_head_rms_norm(
        &self,
        x: &MlxArray,
        weight: &MlxArray,
        eps: f32,
        num_heads: usize,
        head_dim: usize,
        num_tokens: usize,
        stream: &MlxStream,
    ) -> Result<MlxArray, MlxError> {
        // Reshape [1, T, H, D] → [T*H, D]
        let flat = reshape(
            x,
            &[(num_tokens * num_heads) as i32, head_dim as i32],
            stream,
        )?;
        let normed = rms_norm(&flat, weight, eps, stream)?;
        // Reshape back to [1, T, H, D]
        let result = reshape(
            &normed,
            &[1, num_tokens as i32, num_heads as i32, head_dim as i32],
            stream,
        )?;
        Ok(result)
    }

    // ── Profiled eval helpers (added by MLX-OPTIMIZE) ───────────

    /// Synchronous `eval()` with optional profiling instrumentation.
    fn profiled_eval(&mut self, outputs: &[&MlxArray]) -> Result<(), MlxError> {
        let start = if self.config.profile {
            Some(Instant::now())
        } else {
            None
        };

        ironmill_mlx_sys::stream::eval(outputs)?;

        if let Some(start) = start {
            self.profile_eval_duration += start.elapsed();
        }
        self.profile_eval_count += 1;
        Ok(())
    }

    /// Asynchronous `async_eval()` with optional profiling instrumentation.
    ///
    /// Only the call count is tracked — not duration — because `async_eval()`
    /// returns immediately after dispatching work to the GPU. Any wall-clock
    /// measurement would reflect dispatch overhead only, not actual GPU
    /// execution time.
    fn profiled_async_eval(&mut self, outputs: &[&MlxArray]) -> Result<(), MlxError> {
        ironmill_mlx_sys::stream::async_eval(outputs)?;

        self.profile_eval_count += 1;
        Ok(())
    }
}

// ── InferenceEngine implementation ──────────────────────────────

impl InferenceEngine for MlxInference {
    fn load(&mut self, artifacts: &dyn Any) -> Result<(), InferenceError> {
        let mlx_artifacts = artifacts
            .downcast_ref::<MlxArtifacts<'_>>()
            .ok_or_else(|| {
                InferenceError::Runtime("MlxInference::load expects MlxArtifacts".into())
            })?;

        self.config = mlx_artifacts.config.clone();

        // Initialize MLX stream.
        let stream = MlxStream::default_gpu()
            .map_err(|e| InferenceError::Runtime(format!("failed to create MLX stream: {e}")))?;

        // Load weights into MLX arrays.
        let weights = MlxWeights::load(mlx_artifacts.weights, &stream)
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;

        let num_layers = weights.config.num_hidden_layers;

        // Initialize empty KV cache vectors (FP16 path).
        self.k_cache = vec![None; num_layers];
        self.v_cache = vec![None; num_layers];

        // Initialize TurboQuant if enabled.
        if self.config.enable_turboquant {
            let tq_model = MlxTurboQuantModel::new(&weights.config, &self.config, &stream)
                .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            let tq_cache = MlxKvCache::new(
                &tq_model,
                weights.config.num_key_value_heads,
                self.config.max_seq_len,
                weights.config.head_dim,
                num_layers,
                &stream,
            )
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            self.tq_model = Some(tq_model);
            self.tq_cache = Some(tq_cache);
        }

        self.stream = Some(stream);
        self.weights = Some(weights);
        self.seq_pos = 0;

        Ok(())
    }

    fn decode_step(&mut self, token: u32) -> Result<Logits, InferenceError> {
        self.run_pipeline(&[token], false)
            .map_err(InferenceError::from)
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<Logits, InferenceError> {
        if tokens.is_empty() {
            return Err(InferenceError::Decode("empty prefill tokens".into()));
        }

        let chunk_size = self.config.prefill_chunk_size.unwrap_or(tokens.len());
        let chunk_size = chunk_size.max(1);
        let use_async = self.config.async_prefill && self.config.prefill_chunk_size.is_some();

        let chunks: Vec<&[u32]> = tokens.chunks(chunk_size).collect();
        let num_chunks = chunks.len();
        let mut last_logits = None;

        for (i, chunk) in chunks.iter().enumerate() {
            let is_last = i + 1 == num_chunks;
            // For non-final chunks with async_prefill, use async_eval to
            // overlap GPU execution of this chunk with graph construction
            // of the next. The final chunk always uses sync eval.
            let async_final = use_async && !is_last;
            let logits = self
                .run_pipeline(chunk, async_final)
                .map_err(InferenceError::from)?;
            last_logits = Some(logits);
        }

        last_logits.ok_or_else(|| InferenceError::Decode("no chunks processed".into()))
    }

    fn reset(&mut self) {
        self.seq_pos = 0;
        for slot in &mut self.k_cache {
            *slot = None;
        }
        for slot in &mut self.v_cache {
            *slot = None;
        }
        if let Some(ref mut tq_cache) = self.tq_cache {
            tq_cache.reset();
        }
        // Optionally release pooled Metal buffers.
        if self.config.clear_cache_on_reset {
            let _ = ironmill_mlx_sys::stream::metal_clear_cache();
        }
    }
}
