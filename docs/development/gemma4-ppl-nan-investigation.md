# Gemma 4 PPL NaN Investigation

> Follow-up to `gemma4-followup-correctness.md`. All 7 tasks from that spec
> are now implemented. This document tracks the remaining NaN PPL issue.

## Status

**Resolved** — All NaN PPL issues are fixed. FP16, TQ-INT8, and TQ-INT4
all produce finite PPL matching the HuggingFace reference.

## Final Benchmark

| Metric | FP16 | TQ-INT8 | TQ-INT4 |
|--------|------|---------|---------|
| Decode tok/s | 21.5 | 19.1 | 19.4 |
| GPU Memory | 14,041 MB | 14,033 MB | 14,028 MB |
| PPL (instruct, 3×512) | 152 | 151 | 135 |

## What Was Fixed (gemma4-followup-correctness.md)

1. ✅ **layer_scalar** — Per-layer output scaling before residual add
2. ✅ **Pre/post FFN norms** — Separate pre/post feedforward layernorms
3. ✅ **QK norm verification** — Tensor loading confirmed correct; dim mismatch flagged
4. ✅ **Embedding norm** — sqrt(hidden_size) scaling in fused shader
5. ✅ **TurboQuant per-layer** — Per head_dim codebooks/shaders/cache
6. ✅ **PLE verification** — Ordering confirmed correct
7. ✅ **MoE dispatch** — Dense expert evaluation (26B only)
8. ✅ **HEAD_DIM=512 shaders** — Fixed threadgroup memory + missing fused_qk_norm_rope

## Root Cause Analysis: NaN PPL

PPL is NaN because at least one token produces NaN logits. NaN propagates
through softmax → cross-entropy → PPL. Since FP16 (no TurboQuant) also
produces NaN, the issue is in the core forward pass, not KV cache quantization.

### Primary Suspect: QK Norm Dimension Mismatch on Global Layers

**Evidence**: Task 3 verification found that:
- Global layers have `head_dim=512` (from `global_head_dim` config)
- QK norm weights are `[256]` (from checkpoint: `self_attn.q_norm.weight`)
- The fused QK norm + RoPE kernel is compiled with `HEAD_DIM=256` (base)
- But the dispatch passes `layer_hd=512` for global layers

**What happens**: For global layers, Q projection outputs are
`[num_heads × 512]` per token. The QK norm should normalize each 512-dim
head vector, but:
1. The norm weight is only 256 elements
2. The kernel was compiled with `HEAD_DIM=256`
3. Only 256 of 512 elements get normalized; the rest are uninitialized/garbage

This corrupts Q/K vectors → NaN attention scores → NaN propagation.

**HF Reference Check**: In the HuggingFace `Gemma4TextDecoderLayer`, QK norm
is applied per-head using the head's dimension. For global layers with
`head_dim=512`, the norm weight should be `[512]`. If the checkpoint has
`[256]`, it means either:
- (a) QK norm always uses the local head_dim even on global layers
- (b) The checkpoint is wrong (unlikely)
- (c) Q/K projections output local head_dim but attention uses global head_dim

**Action**: Download the HF Gemma 4 implementation and verify which head_dim
the QK norm uses on global layers. Then fix the dispatch accordingly.

### Secondary Suspects

#### RoPE Frequency Application on Global Layers

Global layers use a different RoPE configuration (`proportional` with
`partial_rotary_factor=0.25`, `rope_theta=1000000`). The partial rotation
means only 25% of the head dimension gets RoPE applied.

For `global_head_dim=512`, partial rotation covers `128` dims (512 × 0.25).
For `head_dim=256`, partial rotation covers `64` dims (256 × 0.25).

Verify the `encode_qk_norm_and_rope` dispatch passes the correct partial
rotation factor for the layer's head_dim.

#### Attention Scale Factor

The attention scale `1/sqrt(head_dim)` must use the correct per-layer
head_dim. Currently `head_dim` is used for scale computation in the
attention kernel. Verify this matches the dispatched `layer_hd`.

#### FP16 Overflow in Large Head Dims

With `head_dim=512`, dot products in attention can be larger. FP16 has
limited range (max ~65504). If the attention scale isn't applied correctly,
QK^T values could overflow to Inf → NaN after softmax.

## Diagnostic Plan

### Step 1: Validate QK Norm Dispatch

Check what happens when QK norm is skipped for global layers:
```rust
// In encode_qk_norm_and_rope, if layer_hd != base_hd:
//   skip QK norm entirely (just apply RoPE)
```
If PPL becomes finite (even if wrong), QK norm is the NaN source.

### Step 2: Reference QK Norm Implementation

Compare against HuggingFace `Gemma4TextDecoderLayer`:
```python
# Check: does q_norm.weight shape match head_dim or global_head_dim?
# Check: does the norm apply per the layer's attention head_dim?
```

### Step 3: Activation Dump

Use the calibration pipeline (`run_pipeline_calibration`) to capture
per-layer activations and identify exactly which layer first produces NaN:
```
layer_callback(layer_idx, "attn_norm", data)  → check for NaN
layer_callback(layer_idx, "ffn_norm", data)   → check for NaN
```

The first layer that produces NaN after attention is the culprit.

### Step 4: Fix QK Norm Dispatch

Based on findings, either:
- (a) Use `base_head_dim` (256) for QK norm on all layers, not `layer_hd`
- (b) Add per-layer QK norm weight loading with correct dimensions
- (c) Reshape Q/K from `[nheads, global_hd]` to `[nheads*ratio, base_hd]`
    before QK norm, then reshape back

### Step 5: Verify Fix

Run the benchmark again with `--perplexity-sequences 10`. FP16 PPL should
be finite and in a reasonable range (expected ~42 for wikitext2 based on
HF reference).

## Files Likely Needing Changes

| File | What |
|------|------|
| `metal/inference.rs` | QK norm dispatch — pass correct head_dim per layer |
| `metal/ops.rs` | May need per-layer pipeline selection for QK norm |
| `metal/shaders/fused_qk_norm_rope.metal` | May need HEAD_DIM parameterization |

## Benchmark Command

```bash
cd /Users/jacobfreck/Source/ironmill-gemma4-base
cargo run --release --features metal --bin ironmill-bench -- \
  -c /tmp/gemma4-smoke.toml \
  -b metal \
  --perplexity \
  --perplexity-sequences 3 \
  --perplexity-dataset tests/fixtures/quality/wikitext2-gemma4.json
```

Config file (`/tmp/gemma4-smoke.toml`):
```toml
[[model]]
name = "gemma4-e2b"
path = "/Users/jacobfreck/.cache/huggingface/hub/models--google--gemma-4-e2b-it/snapshots/4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"

[[optimization]]
name = "baseline"

[settings]
iterations = 5
warmup = 2
runs = 1
```
