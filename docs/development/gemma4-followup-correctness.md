# Gemma 4 Follow-Up — Numerical Correctness & Remaining Features

> Follow-up to `gemma4-architecture-support.md`. Addresses remaining gaps
> discovered during E2B end-to-end benchmarking.

## Status

**Pending** — E2B compiles, loads, and runs at 26.7 tok/s decode with all
major features active. FP16 PPL is finite but very high (~10¹⁷ vs expected
~42 on 128-token wikitext2). Root causes identified below.

## Benchmark Baseline (gemma4/base branch)

| Metric | FP16 | TQ-INT8 | TQ-INT4 |
|--------|------|---------|---------|
| Decode tok/s | 26.7 | 23.0 | 23.2 |
| GPU Memory | 14,038 MB | 14,030 MB | 14,026 MB |
| PPL (512-tok) | 1.37×10¹⁷ | NaN | NaN |
| Load time | 7.9s | 7.5s | 7.3s |

## Root Causes of High PPL

### 1. Missing Gemma 4-Specific Layer Norms — ⚠️ Critical

The E2B checkpoint contains layer norms not in the original spec:

- `model.layers.{i}.pre_feedforward_layernorm.weight` — RMSNorm before FFN
- `model.layers.{i}.post_feedforward_layernorm.weight` — RMSNorm after FFN

The HuggingFace reference (`Gemma4TextDecoderLayer`) applies these as:
```python
hidden_states = self.pre_feedforward_layernorm(residual)
hidden_states = self.mlp(hidden_states)
hidden_states = self.post_feedforward_layernorm(hidden_states)
hidden_states = residual + hidden_states
```

The current ironmill implementation uses a single `post_attention_layernorm`
before the MLP and a fused residual+norm after. The pre/post FFN layernorms
are missing entirely.

**Impact**: Every layer's MLP input and output are unnormalized, causing
accumulating numerical drift across 35 layers.

**Fix**: Load `pre_feedforward_layernorm` and `post_feedforward_layernorm`
weights in `LoadedLayer`. Apply pre-FFN norm before `encode_ffn_block` and
post-FFN norm after. This changes the residual flow from:
```
norm → MLP → residual_add
```
to:
```
pre_ffn_norm → MLP → post_ffn_norm → residual_add
```

### 2. Missing `layer_scalar` — ⚠️ Moderate

Each layer has a `model.layers.{i}.layer_scalar` tensor (shape `[1]`).
Per the HF reference, this scales the layer output before the residual add.
Not loaded or applied.

**Fix**: Load as `Option<MetalBuffer>` in `LoadedLayer`. After the post-FFN
norm (or PLE), multiply the layer contribution by `layer_scalar` before the
residual add.

### 3. QK Normalization Not Wired for Gemma 4 — ⚠️ Moderate

The checkpoint has `self_attn.q_norm.weight` and `self_attn.k_norm.weight`
per layer. The inference engine supports QK normalization (used by Qwen3),
and the weights are loaded into `LoadedLayer::q_norm` / `k_norm`. However,
the `encode_qk_norm_and_rope` dispatch only activates the fused QK-norm path
when BOTH `q_norm` and `k_norm` are present.

**Verify**: Confirm that `q_norm` and `k_norm` ARE loaded for Gemma 4 layers
and that the fused path is taken. If the weights load as `Option<MetalBuffer>`
and both are `Some`, the existing code should work. If they're not loaded
(e.g., the weight name doesn't match the expected pattern), add loading.

### 4. Embedding Norm Factor — ⚡ Low (likely correct)

Gemma models multiply embeddings by `sqrt(hidden_size)`. The compile-side
`emit_embedding_norm` handles this. Verify the inference-side
`fused_embedding_norm` kernel also applies this scaling. For E2B:
`sqrt(1536) ≈ 39.19`.

### 5. TurboQuant with head_dim=256 — 🔴 Blocked

TurboQuant KV cache quantization produces NaN for Gemma 4 because
`TurboQuantMetalConfig` stores a single `head_dim` and the quantized
attention kernel is specialized for that dim. When global layers use
`global_head_dim=512`, the mismatch corrupts the KV cache.

**Fix** (from original spec TurboQuant Follow-Up section):
1. Extend `TurboQuantMetalConfig` to per-layer configs
2. Generate shader specializations per unique `(head_dim, num_kv_heads)`
3. Allocate KV cache per-layer with correct dimensions
4. Dispatch correct shader per layer

### 6. Model-Level PLE May Be No-Op — ⚡ Low

Benchmark PPL with model-level PLE enabled vs disabled was identical
(~1.3×10¹⁷). Either:
- PLE contribution is negligible at this PPL scale (likely)
- The model-level PLE computation has a subtle bug

**Verify**: Once PPL is reasonable (after fixing items 1-3), compare PPL
with and without PLE to confirm it helps.

## Implementation Plan

### Task 1: Pre/Post FFN LayerNorms

**Files to modify:**

- `crates/ironmill-inference/src/weight_loading.rs` — Add fields:
  ```rust
  pub pre_ffn_norm: Option<D>,
  pub post_ffn_norm: Option<D>,
  ```
  Load from `model.layers.{i}.pre_feedforward_layernorm.weight` and
  `model.layers.{i}.post_feedforward_layernorm.weight`.

- `crates/ironmill-inference/src/metal/inference.rs` — In the layer loop,
  before `encode_ffn_block`:
  ```rust
  if let Some(ref pre_norm) = lw.pre_ffn_norm {
      ops::encode_rms_norm(&enc, &pipelines.rms_norm, &ops::RmsNormParams {
          input: &bufs.norm_out,
          weight: pre_norm,
          output: &bufs.norm_out, // in-place
          hidden_size: h as u32,
          token_count: token_count as u32,
          eps,
      });
      enc.memory_barrier_buffers();
  }
  ```
  After `encode_ffn_block`:
  ```rust
  if let Some(ref post_norm) = lw.post_ffn_norm {
      ops::encode_rms_norm(&enc, &pipelines.rms_norm, &ops::RmsNormParams {
          input: &bufs.ffn_down,
          weight: post_norm,
          output: &bufs.ffn_down, // in-place
          hidden_size: h as u32,
          token_count: token_count as u32,
          eps,
      });
      enc.memory_barrier_buffers();
  }
  ```

- `crates/ironmill-compile/src/templates/gemma.rs` — Emit pre/post FFN norms
  in `emit_gemma4_transformer_layer` when config has these weights.

### Task 2: Layer Scalar

**Files to modify:**

- `crates/ironmill-inference/src/weight_loading.rs` — Add:
  ```rust
  pub layer_scalar: Option<D>,
  ```
  Load from `model.layers.{i}.layer_scalar`.

- `crates/ironmill-inference/src/metal/inference.rs` — After the FFN block
  (and post-FFN norm if present), before the residual add, apply:
  ```rust
  if let Some(ref scalar) = lw.layer_scalar {
      ops::encode_elementwise_mul_scalar(&enc, ...);
  }
  ```
  This requires a simple element-wise multiply-by-scalar kernel (or reuse
  the existing `fused_softcap` pattern).

### Task 3: QK Norm Verification

Check that `q_norm` and `k_norm` weights are loaded for Gemma 4 layers:
```bash
grep -n "q_norm\|k_norm" crates/ironmill-inference/src/weight_loading.rs
```
The weight names should be `model.layers.{i}.self_attn.q_norm.weight` and
`model.layers.{i}.self_attn.k_norm.weight`. Verify these match the checkpoint
tensor names.

### Task 4: TurboQuant Per-Layer Config

See the TurboQuant Follow-Up section in `gemma4-architecture-support.md`.
This is the largest remaining task and requires Metal shader changes.

## Compile-Side Gaps

The compile-side template (`gemma.rs`) should also emit the pre/post FFN
norms and layer scalar. Currently `emit_gemma4_transformer_layer` calls
`emit_mlp_gelu` directly without pre/post norms. Update to emit:
1. `pre_feedforward_layernorm` RMSNorm before MLP
2. MLP (gate + up + gelu + down)
3. `post_feedforward_layernorm` RMSNorm after MLP
4. `layer_scalar` multiply (if present)
5. Residual add

## Regression Testing

No regressions from the Gemma 4 branch:

| Crate | main | gemma4/base | Cause |
|-------|------|-------------|-------|
| mil-rs | 493 pass / 15 fail | 485 pass / 31 fail | +16 fail from missing fixture files in worktree (onnx, reader, writer tests). Not code regressions. |
| ironmill-compile | 306 pass / 7 fail | 319 pass / 7 fail | +13 pass (new Gemma 4 tests). Same 7 pre-existing failures. |
| ironmill-inference | 251 pass / 0 fail | 251 pass / 0 fail | No change. |

The `LayerContext` struct extension (adding `layer_type`, `effective_head_dim`,
`effective_num_kv_heads`) is backward-compatible — all existing templates pass
default values. The `emit_rotary_embedding` signature change (adding `head_dim`
parameter) was updated in all call sites.

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Pre/post FFN norms interact with PLE ordering | Wrong norm application order | Cross-reference HF `Gemma4TextDecoderLayer.forward` — PLE is after post-FFN norm + residual |
| `layer_scalar` is very small or zero for some layers | Silent degradation | Log scalar values during first load |
| RMSNorm in-place on `norm_out` corrupts data | Wrong MLP input | Use a separate temp buffer if in-place dispatch is unsafe |
| TurboQuant per-layer config changes KV cache layout | Breaks existing models | Gate behind Gemma 4 config; existing models use uniform head_dim |
