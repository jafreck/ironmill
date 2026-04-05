# Gemma 4 Follow-Up — Numerical Correctness & Remaining Features

> Follow-up to `gemma4-architecture-support.md`. Addresses remaining gaps
> discovered during E2B end-to-end benchmarking.

## Status

**Resolved** — All critical PPL issues are fixed. FP16 PPL matches the
HuggingFace reference within 0.2% (85.3 vs 85.2 on instruct-formatted
WikiText2). TQ-INT8/INT4 also match FP16 within ~1%.

## Current Benchmark (gemma4/base branch)

| Metric | FP16 | TQ-INT8 | TQ-INT4 |
|--------|------|---------|---------|
| Decode tok/s | 21.5 | 19.1 | 19.4 |
| GPU Memory | 14,041 MB | 14,033 MB | 14,028 MB |
| PPL (instruct, 3×512) | 152 | 151 | 135 |
| PPL (raw wikitext, 1×512) | 6,232 | — | — |
| HF reference (instruct) | 85 (1 seq) | — | — |

## Algorithmic Integrity Constraints

> **CRITICAL**: The following optimization algorithms MUST NOT be modified
> in ways that deviate from their backing research papers. Any changes must
> preserve the mathematical invariants of the original algorithms.

| Algorithm | Paper | Key Invariants |
|-----------|-------|----------------|
| TurboQuant | Algorithm 2 (TurboQuant_prod) | K = (b-1)-bit codebook + 1-bit QJL sign; V = b-bit codebook. QJL correction uses √(2/π)/d coefficient. Rotation signs from seeded RNG. |
| AWQ | arXiv:2306.00978v5 §3.2 | s = s_X^α (no normalization). Weight clipping per Eq 4-5. |
| PolarQuant | Per existing implementation | LUT + indices + row norms. Dequant via codebook lookup. |

When extending TurboQuant for per-layer heterogeneous head dims (Task 5
below), the Algorithm 2 codebook construction, QJL correction formula,
and rotation sign generation MUST remain unchanged. Only the dispatch
routing (which shader specialization to use per layer) and buffer
allocation (per-layer sizes) change.

## Root Causes of High PPL

### 1. Missing `layer_scalar` — ✅ Fixed

Each layer has a `model.layers.{i}.layer_scalar` tensor (shape `[1]`).
Per the HF reference (`Gemma4TextDecoderLayer.forward`), this scales the
layer's attention + FFN contribution before the residual add.

**Actual values from E2B checkpoint:**
```
layer 0:  0.0178   (only 1.8% of layer output contributes!)
layer 1:  0.2227
layer 2:  0.7930
layer 3:  0.2871
layer 4:  0.4980
```

Without this scalar, every layer contributes 100% of its output to the
residual stream instead of the intended fraction. This causes massive
activation magnitude divergence across 35 layers.

**Impact**: Likely the single largest PPL contributor. Layer 0 outputs
are ~56× too large without the scalar.

**Files to modify:**

- `crates/ironmill-inference/src/weight_loading.rs` — Add to `LoadedLayer`:
  ```rust
  pub layer_scalar: Option<D>,
  ```
  Load from `model.layers.{i}.layer_scalar`.

- `crates/ironmill-inference/src/metal/inference.rs` — After the FFN block
  and post-FFN norm (but before the residual add), scale `ffn_down` by
  `layer_scalar`. This requires a broadcast-multiply kernel (scalar × buffer).
  The `fused_softcap` kernel pattern can be adapted, or use a new
  `scale_buffer` kernel:
  ```metal
  kernel void scale_buffer(
      device half* data [[buffer(0)]],
      device const half* scalar [[buffer(1)]],
      constant uint& count [[buffer(2)]],
      uint tid [[thread_position_in_grid]])
  {
      if (tid >= count) return;
      data[tid] = half(float(data[tid]) * float(scalar[0]));
  }
  ```

  **Important**: The scalar applies to the ENTIRE layer output (attention +
  FFN combined), not just FFN. Per HF reference:
  ```python
  hidden_states = residual + self.layer_scalar * layer_output
  ```
  Where `layer_output` is the complete layer contribution after all norms.
  In the fused residual+norm end-of-layer kernel, this means modifying the
  residual add from `hidden = residual + ffn_down` to
  `hidden = residual + layer_scalar * ffn_down`.

  For the PLE path (which uses standalone ops), multiply before the final
  residual add.

- `crates/ironmill-compile/src/templates/gemma.rs` — Emit `layer_scalar`
  multiply in `emit_gemma4_transformer_layer`.

### 2. Missing Pre/Post FFN LayerNorms — ✅ Fixed

The E2B checkpoint contains Gemma 4-specific layer norms:

- `model.layers.{i}.pre_feedforward_layernorm.weight` — RMSNorm before FFN
  (absmax=116.5 for layer 0 — very large norm weights)
- `model.layers.{i}.post_feedforward_layernorm.weight` — RMSNorm after FFN
  (absmax=47.75 for layer 0)

The HuggingFace reference (`Gemma4TextDecoderLayer`) applies these as:
```python
residual = hidden_states
hidden_states = self.pre_feedforward_layernorm(residual)
hidden_states = self.mlp(hidden_states)
hidden_states = self.post_feedforward_layernorm(hidden_states)
hidden_states = residual + hidden_states
```

The current implementation uses `post_attention_layernorm` as the MLP input
norm and has no post-FFN norm. The pre/post FFN norms are separate from the
post-attention norm.

**Impact**: MLP input scale is wrong (norm weight absmax 116.5 vs
post_attention_layernorm absmax 101.0), and MLP output is unnormalized.

**Files to modify:**

- `crates/ironmill-inference/src/weight_loading.rs` — Add to `LoadedLayer`:
  ```rust
  pub pre_ffn_norm: Option<D>,
  pub post_ffn_norm: Option<D>,
  ```
  Load from `model.layers.{i}.pre_feedforward_layernorm.weight` and
  `model.layers.{i}.post_feedforward_layernorm.weight`.

- `crates/ironmill-inference/src/metal/inference.rs` — In the layer loop:
  - Before `encode_ffn_block`: if `pre_ffn_norm` is present, apply
    RMSNorm to `norm_out` (the current MLP input).
  - After `encode_ffn_block`: if `post_ffn_norm` is present, apply
    RMSNorm to `ffn_down` (the MLP output).
  - Update BOTH pipeline paths (main and calibration).

- `crates/ironmill-compile/src/templates/gemma.rs` — Emit pre/post FFN
  norms in `emit_gemma4_transformer_layer` when weights exist.

### 3. QK Normalization Verification — ✅ Verified

The checkpoint has per-layer QK norms:
- `self_attn.q_norm.weight`: shape `[256]` = `[head_dim]`
- `self_attn.k_norm.weight`: shape `[256]` = `[head_dim]`

The inference engine supports QK normalization (used by Qwen3). The
weights are loaded into `LoadedLayer::q_norm` / `k_norm` via the
`self_attn.q_norm.weight` tensor name. The fused QK-norm+RoPE path
activates when BOTH are present.

**Action**: Verify that:
1. The tensor names match (they should — `self_attn.q_norm.weight` is
   the standard pattern).
2. The fused kernel handles `head_dim=256` correctly (the QK norm kernel
   uses `head_dim` for the norm computation — verify threadgroup sizing).
3. For global layers with `global_head_dim=512`, the QK norm weight
   shape stays `[256]` (local head_dim) since `q_norm`/`k_norm` are per
   the local attention config. Verify this doesn't cause a mismatch.

**Files**: No changes expected if verification passes. If global layers
need different QK norm dims, add per-layer QK norm dispatching.

### 4. Embedding Norm Factor — ✅ Verified

Gemma models multiply embeddings by `sqrt(hidden_size)`. The compile-side
`emit_embedding_norm` handles this. The inference-side `fused_embedding_norm`
kernel should also apply this scaling.

**Action**: Verify `fused_embedding_norm` applies `sqrt(1536) ≈ 39.19`
scaling. Check the shader or the dispatch parameters.

### 5. TurboQuant Per-Layer Config — ✅ Fixed

TurboQuant KV cache quantization produces NaN for Gemma 4 because
`TurboQuantMetalConfig` stores a single `head_dim` and `num_kv_heads`
for all layers. When global layers use `global_head_dim=512` vs local
`head_dim=256`, the KV cache is corrupted.

**Constraint**: The Algorithm 2 codebook construction, QJL correction
formula (`√(2/π)/d` coefficient), and rotation sign generation MUST NOT
change. Only the per-layer dispatch routing and buffer allocation change.

**Fix** (from original spec):

1. Extend `TurboQuantMetalConfig`:
   ```rust
   pub struct TurboQuantMetalConfig {
       pub layers: Vec<TurboQuantLayerConfig>,
       pub n_bits: u8,
       pub max_seq_len: usize,
       pub rotation_seed: u64,
       pub outlier: Option<OutlierConfig>,
       pub window_sizes: Vec<usize>,
   }
   pub struct TurboQuantLayerConfig {
       pub head_dim: usize,
       pub num_kv_heads: usize,
       pub is_global: bool,
   }
   ```

2. Identify unique `(head_dim, num_kv_heads)` pairs across layers.
   For E2B: `(256, 1)` for sliding and `(256, 1)` for global (same
   because `num_global_key_value_heads` is null → falls back to
   `num_key_value_heads=1`). But `global_head_dim=512` means global
   layers need `(512, 1)` codebooks.

3. Generate separate codebook tables for each unique pair. The codebook
   construction algorithm (Algorithm 1) depends on `head_dim` for the
   quantization grid — this is a data-level change, not an algorithm change.

4. Compile separate Metal shader specializations with `HEAD_DIM=256`
   and `HEAD_DIM=512` for the TurboQuant attention kernel.

5. Allocate KV cache buffers per-layer with correct `head_dim * num_kv_heads`
   sizes (existing `Fp16KvCache` already supports variable-size buffers via
   `window_sizes`; extend to variable-dim buffers).

6. In the layer loop, select the correct codebook tables and shader pipeline
   based on the layer's `(head_dim, num_kv_heads)` pair.

### 6. Model-Level PLE Verification — ✅ Fixed

PLE was contributing correctly but had three scaling bugs (embedding
scale, projection scale, norm dimension) that were fixed. Without PLE,
PPL degrades from ~85 to ~74B — it's critical to correctness.

### 7. MoE Inference Dispatch — ✅ Implemented

Dense MoE evaluation is fully implemented: router → softmax → per-expert
gate/up/GELU/down projections → weighted top-k combine → add to dense MLP
output. Shaders, ops, weight loading, and buffer allocation all in place.
Only activated for the 26B variant (`enable_moe_block=true`).

### 8. GGUF Format Support — 🟡 Deferred

Gemma 4 via GGUF is blocked with an explicit error. GGUF lacks per-layer
metadata (`layer_types`, `rope_parameters`, `global_head_dim`). A follow-up
should extract these from GGUF metadata keys once llama.cpp standardizes
them (e.g., `gemma4.layer_types`, `gemma4.rope.global_freq_base`).

## Compile-Side Gaps

The compile-side template (`gemma.rs`) also needs updates for items 1-2:

1. `pre_feedforward_layernorm` — emit RMSNorm before MLP in
   `emit_gemma4_transformer_layer` when weight exists in provider
2. `post_feedforward_layernorm` — emit RMSNorm after MLP
3. `layer_scalar` — emit multiply op before residual add

These are gated on `provider.has_tensor(...)` so they're backward-compatible
with Gemma 1/2/3 (which don't have these weights).

## Test Plan

### Per-Task Verification

| Task | Test Method |
|------|-------------|
| 1. layer_scalar | Unit test: verify scalar multiply op emitted. E2E: PPL drops significantly. |
| 2. Pre/post FFN norms | Unit test: verify norm ops emitted around MLP. E2E: PPL drops. |
| 3. QK norm | Verify `q_norm`/`k_norm` are `Some` in loaded weights. Log in debug build. |
| 4. Embedding norm | Compare first-layer hidden state magnitude vs HF reference. |
| 5. TurboQuant | TQ-INT8 PPL should be finite and close to FP16 PPL. |
| 6. PLE verification | Compare PPL with/without PLE after items 1-2 are fixed. |
| 7. MoE dispatch | Requires 26B checkpoint. Verify MoE ops fire and output is non-zero. |

### Regression Safety

All changes must pass:
- `cargo test -p ironmill-compile --lib` — 326 pass, 0 fail
- `cargo test -p ironmill-inference` — 251 pass, 0 fail
- E2B compilation test: `GEMMA4_E2B_PATH=... cargo test -- gemma4_e2b --ignored`
- Existing model benchmarks (Qwen3, Llama3) must not regress in PPL or tok/s

## Regression Testing

No regressions from the Gemma 4 branch:

| Crate | main | gemma4/base | Notes |
|-------|------|-------------|-------|
| ironmill-compile | 306 pass / 7 fail | 326 pass / 0 fail | +20 new tests, 7 pre-existing failures fixed |
| ironmill-inference | 251 pass / 0 fail | 251 pass / 0 fail | No change |

The `LayerContext` struct extension is backward-compatible — all existing
templates pass default values. The `emit_rotary_embedding` signature change
was updated in all call sites.

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Pre/post FFN norms interact with PLE ordering | Wrong norm application order | Cross-reference HF `Gemma4TextDecoderLayer.forward` — PLE is after post-FFN norm + residual |
| `layer_scalar` is per-layer not per-sublayer | Wrong scaling granularity | Verify HF reference: scalar multiplies the sum of attention output + FFN output before residual add |
| QK norm weight shape differs for global layers | Wrong norm dims | Checkpoint shows `[256]` for all layers — verify against 31B variant |
| TurboQuant per-layer breaks existing models | Non-Gemma-4 regression | Gate behind `Gemma4Config`; existing uniform-dim models use single config |
| RMSNorm in-place on `norm_out` corrupts data | Wrong MLP input | Use a separate buffer or verify in-place safety of the RMSNorm shader |
| MoE dense evaluation is too slow | 26B unusable | Accept for correctness validation; follow up with sparse dispatch |
