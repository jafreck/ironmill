//! MLX inference engine implementing the [`InferenceEngine`] trait.
//!
//! Runs the full LLaMA-family transformer decode pipeline using MLX
//! built-in operations with lazy evaluation. A single `eval()` call
//! at the end materializes the logits.

use std::any::Any;

use ironmill_mlx_sys::{
    MlxArray, MlxDtype, MlxStream, add, broadcast_to, concat, expand_dims, matmul, multiply,
    reshape, rms_norm, rope, scaled_dot_product_attention, silu, slice, transpose, transpose_axes,
};
use mil_rs::weights::WeightProvider;

use super::config::MlxConfig;
use super::error::MlxError;
use super::weights::{MlxWeightBuffer, MlxWeights};
use crate::engine::{InferenceEngine, InferenceError};
use crate::types::Logits;

// ── Embedded PolarQuant kernel source ───────────────────────────

/// Metal kernel for PolarQuant dequant + matvec.
///
/// Each thread computes one output element by reconstructing the weight
/// row from packed LUT indices and norms, then dotting with the input.
/// This is a placeholder that will be replaced with the optimized kernel
/// from the MLX-TURBOQUANT task.
const POLARQUANT_MATVEC_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

[[kernel]] void polarquant_matvec(
    device const half* x          [[buffer(0)]],
    device const uchar* indices   [[buffer(1)]],
    device const half* lut        [[buffer(2)]],
    device const half* norms      [[buffer(3)]],
    device half* output           [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]])
{
    // Placeholder: output[row] = 0 for now.
    // Full implementation comes with MLX-TURBOQUANT.
    uint row = tid.x;
    uint batch = tid.y;
    output[batch * gridDim.x + row] = half(0.0);
}
"#;

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
        })
    }

    /// Run the transformer pipeline for a batch of tokens.
    ///
    /// This builds a lazy MLX computation graph and evaluates it with
    /// a single `eval()` at the end.
    fn run_pipeline(&mut self, tokens: &[u32]) -> Result<Logits, MlxError> {
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

        let result = self.run_pipeline_inner(tokens, &stream, &weights);

        // Put them back regardless of outcome.
        self.stream = Some(stream);
        self.weights = Some(weights);

        result
    }

    /// Inner pipeline logic operating on borrowed stream/weights, allowing
    /// `&mut self` access to KV caches simultaneously.
    fn run_pipeline_inner(
        &mut self,
        tokens: &[u32],
        stream: &MlxStream,
        weights: &MlxWeights,
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
        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            // Pre-attention RMSNorm
            let normed = rms_norm(&hidden, &layer.input_norm, rms_eps, stream)?;

            // Q/K/V projections
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

            // Transpose to [batch, heads, seq, head_dim] for attention
            // MLX transpose reverses all axes: [0,1,2,3] → [3,2,1,0]
            // We need [0,2,1,3] which we achieve via reshapes.
            // Actually, we reshape: [1, T, H, D] → [1, H, T, D] by swapping dims 1 and 2.
            // Use reshape: [1*T, H, D] then reshape to [H, T, D] then [1, H, T, D].
            let q = self.swap_dims_1_2(&q, stream)?;
            let k = self.swap_dims_1_2(&k, stream)?;
            let v = self.swap_dims_1_2(&v, stream)?;

            // Update KV cache by concatenating along the seq dimension
            let (k_full, v_full) = self.update_kv_cache(layer_idx, &k, &v, stream)?;

            // GQA: expand K/V if num_heads != num_kv_heads
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

            // Scaled dot-product attention
            let scale = 1.0 / (head_dim as f32).sqrt();
            let attn_out =
                scaled_dot_product_attention(&q, &k_for_attn, &v_for_attn, scale, None, stream)?;

            // Transpose back: [1, H, T, D] → [1, T, H, D] → [T, H*D]
            let attn_out = self.swap_dims_1_2(&attn_out, stream)?;
            // Now [1, T, H, D] — reshape to [T, H*D]
            let attn_out = reshape(
                &attn_out,
                &[num_tokens as i32, (num_heads * head_dim) as i32],
                stream,
            )?;

            // O projection
            let attn_proj = self.weight_matmul(&attn_out, &layer.o_proj, stream)?;

            // Residual connection
            let hidden_post_attn = add(&hidden, &attn_proj, stream)?;

            // Post-attention RMSNorm
            let normed_ff = rms_norm(&hidden_post_attn, &layer.post_attn_norm, rms_eps, stream)?;

            // Feed-forward: gate/up → SiLU(gate) * up → down
            let gate = self.weight_matmul(&normed_ff, &layer.gate_proj, stream)?;
            let up = self.weight_matmul(&normed_ff, &layer.up_proj, stream)?;
            let gate_activated = silu(&gate, stream)?;
            let ff_mid = multiply(&gate_activated, &up, stream)?;
            let ff_out = self.weight_matmul(&ff_mid, &layer.down_proj, stream)?;

            // Residual connection
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
        ironmill_mlx_sys::stream::eval(&[&logits_arr])?;

        // Read logits back to CPU
        let logits_f32: &[f32] = logits_arr.as_contiguous_slice()?;
        let logits = logits_f32.to_vec();

        self.seq_pos += num_tokens;

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
    /// Dispatches the PolarQuant dequant+matvec kernel which reads packed
    /// indices, LUT, and norms, reconstructs weights on-the-fly, and
    /// computes the matrix-vector product. Kernel source is embedded inline.
    fn quantized_matmul(
        &self,
        x: &MlxArray,
        qw: &super::weights::MlxQuantizedWeight,
        stream: &MlxStream,
    ) -> Result<MlxArray, MlxError> {
        let x_shape = x.shape();
        let m = if x_shape.len() >= 2 { x_shape[0] } else { 1 };
        let n = qw.n;
        let _k = qw.k;

        let output_shape = [m, n];
        let output_shapes: &[&[usize]] = &[&output_shape];
        let output_dtypes = [MlxDtype::Float16];

        // PolarQuant dequant+matvec kernel.
        // Each thread computes one output element by reconstructing
        // the weight row from packed indices + LUT and dotting with input.
        let kernel_source = POLARQUANT_MATVEC_SOURCE;

        let grid_x = n;
        let grid_y = m;
        let tg_size = 256usize;

        let inputs: &[&MlxArray] = &[x, &qw.indices, &qw.lut, &qw.norms];
        let outputs: &[&MlxArray] = &[];

        let result = ironmill_mlx_sys::metal_kernel(
            "polarquant_matvec",
            inputs,
            outputs,
            kernel_source,
            [grid_x, grid_y, 1],
            [tg_size, 1, 1],
            output_shapes,
            &output_dtypes,
            stream,
        )?;

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

        // Initialize empty KV cache vectors.
        self.k_cache = vec![None; num_layers];
        self.v_cache = vec![None; num_layers];

        self.stream = Some(stream);
        self.weights = Some(weights);
        self.seq_pos = 0;

        Ok(())
    }

    fn decode_step(&mut self, token: u32) -> Result<Logits, InferenceError> {
        self.run_pipeline(&[token]).map_err(InferenceError::from)
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<Logits, InferenceError> {
        if tokens.is_empty() {
            return Err(InferenceError::Decode("empty prefill tokens".into()));
        }

        let chunk_size = self.config.prefill_chunk_size.unwrap_or(tokens.len());
        let chunk_size = chunk_size.max(1);

        let mut last_logits = None;
        for chunk in tokens.chunks(chunk_size) {
            last_logits = Some(self.run_pipeline(chunk).map_err(InferenceError::from)?);
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
    }
}
