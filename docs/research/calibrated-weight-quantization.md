# Calibrated Weight Quantization

## Summary

This document covers ironmill's strategy for advanced post-training weight
quantization, spanning five methods across three generations:

1. **Baseline:** INT4 MinMax (simple, no calibration)
2. **Calibration-aware:** AWQ and GPTQ (use activation data from calibration samples)
3. **Rotation-optimized:** SpinQuant and QuIP# (learned/Hadamard rotations + advanced codebooks)

All five methods quantize **weights only** — activations remain in FP16 at
inference time. The calibration data informs quantization decisions but does
not retrain the model.

Burn (Rust) has basic INT4 PTQ with MinMax calibration — the simplest form
of weight quantization that only looks at weight ranges. GPTQ and AWQ use
*activation data* to make smarter decisions (~0.3 pp perplexity loss vs
~1-3 pp for MinMax). SpinQuant and QuIP# add pre-quantization rotation to
further reduce outlier sensitivity, achieving state-of-the-art quality at
≤4 bits.

No Rust framework implements any of these five methods. Adding native support
to ironmill would make it the first Rust tool capable of activation-aware and
rotation-optimized weight quantization — and the first fully Python-free LLM
quantization-to-deployment pipeline.

ironmill already has a head start: PolarQuant implements random Hadamard
rotation + beta-scalar quantization. SpinQuant and QuIP# are natural
evolutions of this existing infrastructure.

## What Exactly Gets Quantized

**Weights only.** Both GPTQ and AWQ:

- Take a pretrained FP16 model
- Run a small calibration dataset (~128–512 samples) through it
- Use the resulting activation statistics to make smarter quantization decisions
- Output a model with INT4 weights + per-group scale/zero-point metadata
- At inference, weights are dequantized on-the-fly (INT4 → FP16) before matmul

The calibration data is used to *inform* the quantization, not to retrain or
fine-tune. No gradients are computed. No backpropagation.

## How They Work

### GPTQ (Hessian-Guided)

1. Feed calibration data through each layer, capture input activations X
2. Compute the Hessian: `H = 2 * Xᵀ * X` (sensitivity of output to weight changes)
3. Quantize weights sequentially within each row/group
4. After each weight is quantized, use H to compensate: adjust remaining
   unquantized weights to absorb the error
5. Result: INT4 weights with per-group scales, dramatically less error than
   naive round-to-nearest

**Key property:** Second-order error compensation. Quantizes one column at a time,
redistributing error across remaining columns via the inverse Hessian.

### AWQ (Activation-Aware Scaling)

1. Feed calibration data, collect per-channel activation magnitudes
2. Identify salient channels (top ~1% by activation magnitude)
3. Compute per-channel scaling factors via grid search to minimize output MSE
4. Scale weights down by these factors before quantization (scale activations up
   to compensate at runtime)
5. Quantize the scaled weights to INT4

**Key property:** Simpler than GPTQ. No Hessian inversion. Faster to run.
Protects important weights by scaling rather than error compensation.

### Comparison

| Aspect | GPTQ | AWQ |
|--------|------|-----|
| Calibration data | ~128 samples | ~128 samples |
| Core technique | Hessian error compensation | Activation-aware scaling |
| Math complexity | Matrix inversion per block | Grid search per channel |
| Quantization speed (70B) | Hours | Minutes |
| Peak memory during quant | High (Hessian storage) | Lower |
| Output quality (INT4) | Excellent | Excellent (within ~1% of each other) |
| Output format | INT4 weights + scales + zeros | INT4 weights + scales + zeros |

Both produce the same storage format: packed INT4 weights with per-group
scale and zero-point tensors. The difference is *how* the quantization
parameters are chosen.

### SpinQuant (Learned Rotation + Quantization)

SpinQuant (Meta, ICLR 2025) extends the rotation-before-quantization idea
that ironmill's PolarQuant already implements — but replaces random Hadamard
matrices with *learned* rotation matrices optimized to minimize quantization
error.

1. Initialize rotation matrices at key points: residual stream, attention
   blocks, KV cache projections
2. Optimize rotations via Cayley parameterization on a small validation set
   (~same calibration data as GPTQ/AWQ)
3. Rotations are "absorbed" into adjacent weights — zero runtime overhead
4. Quantize the rotated weights using standard affine quantization (INT4)

**Key property:** The rotations have no effect at full precision — they only
help when weights are subsequently quantized. They "spread" outlier values
across dimensions, making the weight distribution more uniform and
quantization-friendly.

**Results:** On LLaMA-3 8B at W4A4KV4, SpinQuant narrows the accuracy gap
to full precision by 45% vs random rotations (QuaRot). Only 2.9pp gap vs
25pp for SmoothQuant.

**ironmill connection:** PolarQuant already does random Hadamard rotation.
SpinQuant is the learned upgrade — same infrastructure, better rotations.
The calibration runner built for AWQ/GPTQ is reused here to optimize
rotation matrices.

### QuIP# (Hadamard + Lattice Codebooks)

QuIP# (Cornell, ICML 2024) combines two techniques:

1. **Randomized Hadamard transform** — same as PolarQuant's rotation step.
   Increases weight "incoherence" by spreading outliers across dimensions.
2. **E8 lattice vector quantization** — instead of scalar INT4 quantization,
   quantizes groups of 8 weights to the nearest point on the E8 lattice
   (a mathematically optimal packing in 8D space). Uses codebook lookup.
3. **Optional fine-tuning** — small LDLQ-based refinement pass

**Key property:** Lattice codebooks exploit the post-rotation weight
distribution (approximately spherical) far better than scalar quantization.
Achieves state-of-the-art quality at ≤4 bits per weight.

**ironmill connection:** PolarQuant already does step 1 (Hadamard rotation)
and palettization does LUT codebook quantization. QuIP# is the combination
of both with a mathematically optimal codebook (E8 lattice instead of
k-means). The existing `constexpr_lut_to_dense` op can represent the
codebook output.

### D2Quant (Dual-Scale + Deviation-Aware Correction)

D2Quant (2026) tackles the sub-4-bit frontier (2-bit, 3-bit):

1. **Dual-Scale Quantizer (DSQ):** Uses two separate quantization scales
   within each weight group — one for normal weights, one for outliers.
   The outlier scale captures extreme values without distorting the main
   distribution.
2. **Deviation-Aware Correction (DAC):** Post-quantization, corrects the
   activation distribution shift in LayerNorm layers caused by weight
   quantization. Adjusts LayerNorm parameters to compensate.

**Key property:** Achieves usable 2-bit and 3-bit quantization without
requiring low-bit hardware operators. Outperforms other sub-4-bit methods.

**ironmill connection:** Extends the quantization pass with dual-scale
support and adds a post-quantization LayerNorm correction pass. Builds on
the same calibration infrastructure as AWQ/GPTQ.

### All Methods Compared

| Method | Technique | Bits | Calibration | Quality | Complexity |
|--------|-----------|------|-------------|---------|------------|
| MinMax | Weight range | 4-8 | None | Baseline | Trivial |
| AWQ | Activation-aware scaling | 4 | ~128 samples | Excellent | Low |
| GPTQ | Hessian error compensation | 4 | ~128 samples | Excellent | Medium |
| SpinQuant | Learned rotation + affine | 4 | ~128 samples | Best-in-class | Medium |
| QuIP# | Hadamard + E8 lattice | 2-4 | ~128 samples | Best-in-class | High |
| D2Quant | Dual-scale + LayerNorm fix | 2-3 | ~128 samples | Best sub-4-bit | Medium |

The field is converging on a three-step pattern: **rotate → quantize → compensate**.

| Step | ironmill today | Target |
|------|---------------|--------|
| Rotate | Random Hadamard (PolarQuant) ✅ | Learned rotations (SpinQuant) |
| Quantize | Beta-scalar / LUT (PolarQuant) ✅ | Affine INT4, lattice codebook (QuIP#) |
| Compensate | None | Hessian (GPTQ), scaling (AWQ), dual-scale (D2Quant) |

## Landscape: What Exists in Rust Today

### Burn

Burn has real INT4 PTQ (post-training quantization):
- **MinMax calibration** — computes quantization range from weight min/max values
- **Per-block quantization** — block sizes like 32, configurable
- **Q4F / Q4S** — 4-bit full-range and symmetric modes
- **Packed storage** — packed uint32 format

What Burn does NOT have:
- No activation-aware calibration (only looks at weight values, not model behavior)
- No Hessian computation (GPTQ)
- No per-channel importance scaling (AWQ)
- No error compensation across weight groups

Burn's MinMax is the simplest PTQ baseline. At INT4, this typically degrades
perplexity by 1-3 points. GPTQ/AWQ close this to ~0.3-0.5 points by using
calibration data to understand *how the model uses those weights*.

### Candle

No quantization passes. Can load pre-quantized GGUF models for inference.

### llama.cpp / GGML

Has its own quantization formats (Q4_0, Q4_K_M, etc.) with importance-matrix
support. Written in C/C++, not Rust. Different algorithm family from GPTQ/AWQ.

## Current State in ironmill

### What we already have

1. **Pre-quantized GPTQ/AWQ import** (`onnx_graph.rs`)
   - Detects `*.qweight` / `*.scales` / `*.qzeros` patterns
   - Preserves quantization metadata as const attributes
   - Models quantized elsewhere can be imported and run

2. **Weight-only INT8** (`Int8QuantizePass`)
   - Min/max affine quantization of FP32 consts
   - `calibration_dir` field exists but is unused
   - Emits `constexpr_affine_dequantize` ops

3. **PolarQuant** (`PolarQuantPass`)
   - Hadamard rotation + beta-scalar quantization
   - 2-bit and 4-bit support
   - No calibration needed (statistics-free)

4. **Palettization** (`PalettizePass`)
   - K-means LUT quantization, 1/2/4/6/8-bit
   - No calibration

5. **TurboQuant** (inference runtime)
   - INT8 KV-cache compression with QJL correction
   - Runtime quantize/dequantize, not weight compression

### What's missing

- **No calibration pipeline** — cannot run inference on calibration data
- **No Hessian computation** — needed for GPTQ
- **No activation statistics collection** — needed for AWQ
- **No INT4 weight encoding** — current INT8 path doesn't support 4-bit packing
- **No grouped quantization** — current INT8 is per-tensor or per-channel,
  not per-group (typical GPTQ/AWQ group size is 128)

## Proposed Architecture

### Phase 0: INT4 MinMax Weight Quantization (Baseline)

Before calibration-aware methods, ironmill should support basic INT4 MinMax —
the same approach Burn already offers. This is useful on its own and builds the
INT4 infrastructure that GPTQ/AWQ depend on.

**What exists today:**
- `Int8QuantizePass` does MinMax affine quantization, but hard-coded to UINT8
- `constexpr_affine_dequantize` has no bit-width metadata
- `PalettizePass` already handles 4-bit packing (LUT-based, different path)

**What's needed:**
1. Generalize `Int8QuantizePass` → `AffineQuantizePass` with configurable bit width
2. Quant math: `scale = (max - min) / 15.0` for UINT4 (or `/7.0` for signed INT4)
3. Pack two 4-bit values per byte in `quantized_data`
4. Add `bit_width` metadata to `QuantizationInfo::AffineDequantize`
5. Add `ScalarType::UInt4` (or store as UInt8-packed with explicit bit_width)
6. Update `constexpr_affine_dequantize` consumers to unpack nibbles
7. Per-group quantization support (group_size: 32, 64, 128, 256)

**Per-group is important:** Per-tensor INT4 is too coarse for LLMs. Per-group
(e.g., 128 weights sharing one scale/zero) is standard practice and needed by
all three methods (MinMax, AWQ, GPTQ).

This phase delivers immediate value and is prerequisite infrastructure for
phases 1-3.

### Phase 1: Calibration Infrastructure

Add a calibration runner that can execute a MIL program on sample inputs and
capture intermediate activations.

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Calibration │───▶│ MIL Interpreter  │───▶│ Activation Store │
│ Dataset     │    │ (forward only)   │    │ (per-layer)      │
└─────────────┘    └──────────────────┘    └─────────────────┘
```

**Requirements:**
- Minimal MIL interpreter that can execute the graph forward-only
- Hook mechanism to capture activations at linear/matmul op inputs
- Memory-efficient: stream batches, accumulate statistics only
- No autograd, no backward pass

**Calibration data sources:**

ironmill operates on token IDs, not raw text. It has no built-in tokenizer.
The existing perplexity benchmark already uses pre-tokenized JSON files
(`Vec<Vec<u32>>`) such as `wikitext2-qwen3.json`. Calibration should follow
the same pattern.

Options (in order of recommendation):

1. **Pre-tokenized JSON / safetensors (recommended default)**
   Matches the existing `PerplexityDataset` format. Ship a small set of
   pre-tokenized calibration files for common tokenizers (Llama, Qwen, Mistral).
   ~128 sequences × 2048 tokens is sufficient for both GPTQ and AWQ.
   Zero additional dependencies.

2. **HuggingFace `tokenizers` crate (optional feature `--features tokenize`)**
   The `tokenizers` crate is pure Rust — HuggingFace wrote it in Rust first,
   the Python version is just bindings. It's the de facto standard (~3.5M
   downloads/month), supports BPE/WordPiece/Unigram (Llama, GPT, BERT, Qwen,
   Mistral, etc.), and can load any `tokenizer.json` from the HF Hub.

   With this feature enabled, users can point at raw text + a tokenizer and
   calibrate in one command:
   ```
   ironmill compile model.safetensors \
     --quantize awq \
     --tokenizer meta-llama/Llama-3-8B \
     --calibration-text wikitext.txt
   ```
   Internally:
   ```rust
   use tokenizers::Tokenizer;
   let tok = Tokenizer::from_pretrained("meta-llama/Llama-3-8B", None)?;
   let enc = tok.encode(text, false)?;
   // enc.get_ids() → &[u32] — feed directly to calibration runner
   ```
   This completes the fully Python-free story: raw text → tokenize → calibrate
   → quantize → deploy, all in one Rust binary.

3. **Random token IDs (zero-effort fallback)**
   Surprisingly viable — research shows random calibration inputs degrade
   GPTQ/AWQ quality by only ~0.3-0.5 perplexity points vs real data.
   Useful for quick iteration and CI. No dependencies, no data files.

4. **User-provided tensors**
   Accept a directory of safetensors/numpy files containing raw activation
   tensors. Maximum flexibility, but puts the burden on the user. Compatible
   with the existing `calibration_dir` field on `Int8QuantizePass`.

### Phase 2: AWQ Pass (Simpler, Ship First)

AWQ is the better first target because:
- No matrix inversion (simpler math)
- Lower memory usage during quantization
- Faster to run
- Comparable output quality to GPTQ

```rust
pub struct AwqQuantizePass {
    pub bits: u8,              // 4 or 8
    pub group_size: usize,     // typically 128
    pub calibration: ActivationStore,
}
```

**Algorithm steps implemented as a MIL pass:**

1. For each linear op's weight const:
   a. Retrieve cached input activations from calibration
   b. Compute per-channel activation magnitudes
   c. Grid search for optimal per-channel scaling factors
   d. Apply scaling to weights
   e. Quantize to INT4 with per-group scale/zero-point
   f. Rewrite op: replace const with `constexpr_affine_dequantize`
      (or new `constexpr_int4_dequantize`)

### Phase 3: GPTQ Pass

```rust
pub struct GptqQuantizePass {
    pub bits: u8,              // 4 or 8
    pub group_size: usize,     // typically 128
    pub block_size: usize,     // columns processed together (typically 128)
    pub dampening: f64,        // Hessian diagonal dampening (typically 0.01)
    pub calibration: ActivationStore,
}
```

**Algorithm steps:**

1. For each linear op's weight const:
   a. Retrieve cached input activations from calibration
   b. Compute Hessian: `H = 2 * Xᵀ * X`
   c. Add dampening: `H += dampening * diag(H) * I`
   d. Compute Cholesky decomposition of H
   e. For each column block:
      - Quantize weights in the block
      - Compute quantization error
      - Compensate remaining columns using inverse Hessian
   f. Rewrite op with quantized weights + scales + zeros

### Phase 4: SpinQuant Pass (Learned Rotations)

Upgrades PolarQuant's random Hadamard rotations to learned rotations.

```rust
pub struct SpinQuantPass {
    pub bits: u8,                    // 4
    pub group_size: usize,           // 128
    pub rotation_epochs: usize,      // optimization iterations (typically 100)
    pub calibration: ActivationStore,
}
```

**Algorithm steps:**

1. Initialize rotation matrices R_i at each insertion point
   (residual connections, attention Q/K/V projections) using Hadamard
   matrices (reuse PolarQuant's existing rotation infrastructure)
2. For each optimization step:
   a. Apply current rotations to weights
   b. Quantize the rotated weights (INT4 affine)
   c. Run calibration data through the quantized model
   d. Compute loss (MSE or cross-entropy vs FP16 reference)
   e. Update rotations via Cayley gradient descent on the
      orthogonal manifold (rotations must stay orthogonal)
3. Absorb final learned rotations into adjacent weight matrices
   (zero runtime cost — rotations become part of the weights)
4. Quantize the rotation-absorbed weights using the Phase 0
   INT4 affine path (or AWQ/GPTQ for even better quality)

**Key reuse:** PolarQuant's `rotation.rs` already has Hadamard matrix
generation and weight rotation application. SpinQuant replaces the
fixed Hadamard with a learned matrix while keeping the same rotation
insertion points and absorption logic.

**Note:** This phase requires the calibration runner to support
full forward passes with loss computation (slightly more than the
activation-capture needed for AWQ/GPTQ). The MLX backend is well-suited
for this since it supports autograd for rotation optimization.

### Phase 5: QuIP# Pass (Lattice Codebook Quantization)

Combines Hadamard rotation with E8 lattice vector quantization for
state-of-the-art quality at ≤4 bits.

```rust
pub struct QuipSharpPass {
    pub bits: u8,                // 2, 3, or 4
    pub calibration: ActivationStore,
}
```

**Algorithm steps:**

1. Apply randomized Hadamard rotation to weights
   (reuse PolarQuant's rotation infrastructure)
2. For each weight matrix, process in groups of 8 values:
   a. Find nearest E8 lattice point for each 8-element vector
   b. Encode as lattice codebook index
   c. Store as packed codebook indices + scale
3. Optionally apply LDLQ fine-tuning (Hessian-based refinement,
   reuses GPTQ's Cholesky infrastructure)
4. Emit as `constexpr_lut_to_dense` ops (reuse existing palettization
   MIL op with E8 codebook instead of k-means centroids)

**E8 lattice codebook:** The E8 lattice is a fixed, mathematically
optimal sphere packing in 8 dimensions. The codebook is a constant
lookup table (not learned per-model), making it simple to implement:
- 2-bit: 256 entries (8 values × 2 bits = 16 bits per vector)
- 3-bit: 16M entries (impractical) → use hierarchical encoding
- 4-bit: standard E8 with scaling

**Key reuse:**
- PolarQuant's Hadamard rotation (step 1)
- Palettization's `constexpr_lut_to_dense` op (step 4)
- GPTQ's Cholesky/Hessian code (step 3, optional refinement)

### Phase 6: D2Quant Pass (Sub-4-bit with Dual-Scale)

Extends quantization to 2-bit and 3-bit with dual-scale handling.

```rust
pub struct D2QuantPass {
    pub bits: u8,                // 2 or 3
    pub group_size: usize,       // typically 128
    pub outlier_threshold: f32,  // percentile for outlier detection
    pub calibration: ActivationStore,
}
```

**Algorithm steps:**

1. For each linear op's weight const:
   a. Retrieve cached input activations from calibration
   b. Partition weights in each group into normal vs outlier
      (based on magnitude percentile threshold)
   c. Compute separate scale/zero for each partition
   d. Quantize normal weights with normal scale (2/3-bit)
   e. Quantize outlier weights with outlier scale (higher precision)
   f. Store both partitions with a bitmask indicating which is which
2. Post-quantization LayerNorm correction (DAC):
   a. For each LayerNorm following a quantized linear:
      - Run calibration data through quantized model
      - Measure activation distribution shift vs FP16
      - Adjust LayerNorm weight/bias to compensate
   b. This is a separate MIL pass that runs after weight quantization

**New MIL representation:** Needs a dual-scale variant of
`constexpr_affine_dequantize` that stores two scale/zero pairs
plus an outlier bitmask per group.

### INT4/INT2 Encoding & Inference Support

Shared infrastructure used by all phases above.

New MIL op or extension of existing ops:

```
constexpr_int4_dequantize(
    quantized_data: uint8,   // packed: 2 weights per byte
    scales: fp16,            // per-group
    zero_points: int4/fp16,  // per-group
    group_size: i32,
    axis: i32
) -> fp16
```

Metal shader for INT4 dequantization during inference:
- Unpack 2×INT4 from each byte
- Apply `(int4_val - zero_point) * scale` per group
- Fuse with subsequent matmul where possible

## Integration with Existing Pipeline

```
SafeTensors/GGUF/ONNX
        │
        ▼
   MIL IR (FP16)
        │
        ▼
┌─────────────────────┐
│  Standard Passes     │  DCE, const fold, fusion
└─────────┬───────────┘
          │
          ├──── Phase 0 (no calibration):
          │     ┌──────────────────────┐
          │     │  INT4 MinMax Pass     │  Per-group affine quantization
          │     └──────────┬───────────┘
          │                │
          ├──── Phases 1-3 (calibration-aware):
          │     ┌──────────────────────┐
          │     │  Calibration Run      │  Forward pass on sample data
          │     └──────────┬───────────┘
          │                │
          │     ┌──────────────────────┐
          │     │  AWQ / GPTQ Pass      │  Activation-informed quantization
          │     └──────────┬───────────┘
          │                │
          ├──── Phases 4-5 (rotation-optimized):
          │     ┌──────────────────────┐
          │     │  Calibration Run      │
          │     └──────────┬───────────┘
          │                │
          │     ┌──────────────────────┐
          │     │  SpinQuant / QuIP#    │  Learned rotation + quantize
          │     └──────────┬───────────┘
          │                │
          ├──── Phase 6 (sub-4-bit):
          │     ┌──────────────────────┐
          │     │  D2Quant Pass         │  Dual-scale 2/3-bit + LayerNorm fix
          │     └──────────┬───────────┘
          │                │
          ├────────────────┘
          ▼
┌─────────────────────┐
│  Post-Quant Passes   │  Type repropagation, layout opt
└─────────┬───────────┘
          │
          ▼
    Metal / CoreML / ANE
```

The calibration + quantization step slots in after fusion passes (so we
quantize the optimized graph) and before backend-specific lowering.

## Key Design Decisions

### 1. Calibration requires forward passes with activation capture

The calibration runner needs to execute the model on sample inputs and
capture intermediate activations at each linear layer. Options:

| Option | Pros | Cons |
|--------|------|------|
| **Metal inference engine + hooks** | No external deps, already runs models, fastest | Must add activation hook API |
| **MLX backend** | Already integrated, has autograd (needed for SpinQuant) | Requires mlx-c install |
| **New MIL interpreter** | Pure, generic | Massive effort, reinventing a runtime |

**Recommendation:** Use ironmill's own Metal inference engine. It already
runs LLM forward passes via `prefill()`. The only addition needed is an
activation hook mechanism to tap intermediate GPU buffers:

```rust
pub trait ActivationHook {
    fn on_linear_input(&mut self, layer: usize, name: &str, activation: &[f16]);
}

fn prefill_with_hooks(
    &mut self,
    tokens: &[u32],
    hooks: &mut dyn ActivationHook,
) -> Result<Logits, InferenceError>;
```

The Metal backend already computes these activations — they flow through
GPU buffers between layers. Adding hooks means reading them back to CPU
at tap points (`MTLBuffer.contents()` per layer). This is a one-time
calibration cost, not a hot path.

**Exception — SpinQuant (Phase 4):** Rotation optimization requires
gradients (loss → rotation matrix updates). This is the one case where
MLX's autograd capability is genuinely useful. All other methods (MinMax,
AWQ, GPTQ, QuIP#, D2Quant) only need forward passes and can use the
Metal engine directly.

### 2. INT4 packing format

Match the GPTQ/AWQ community standard:
- 8 INT4 values packed into one `uint32`
- Little-endian bit ordering
- Per-group scales stored as FP16
- Compatible with HuggingFace `safetensors` INT4 layout

This ensures models quantized by ironmill can be loaded by other tools and
vice versa.

### 3. Group size

Default 128 (industry standard). Support 32, 64, 128, 256.
Smaller groups = better accuracy, more metadata overhead.

### 4. Which layers to quantize

By default: all `linear` ops with weight dimensions ≥ a threshold (e.g., 256).
Skip embedding layers and LM head (standard practice).
User-configurable include/exclude patterns.

## Effort Estimate

| Phase | Scope | Complexity | Reuses |
|-------|-------|------------|--------|
| 0. INT4 MinMax | Generalize INT8 pass + INT4 packing + per-group | Low-Medium | Int8QuantizePass |
| 1. Calibration infra | Activation hooks on Metal inference engine | Low-Medium | MetalInference prefill |
| 2. AWQ pass | Per-channel scaling + INT4 quantize + MIL rewrite | Medium | Phase 0 INT4 |
| 3. GPTQ pass | Hessian + Cholesky + error compensation | High | Phase 0 INT4 |
| 4. SpinQuant pass | Cayley rotation optimization + absorption | Medium-High | PolarQuant rotation.rs, MLX autograd |
| 5. QuIP# pass | E8 lattice codebook + Hadamard | High | PolarQuant + palettize LUT |
| 6. D2Quant pass | Dual-scale + LayerNorm correction | Medium | Phase 0 + calibration |

**Delivery milestones:**
- **M1:** Phase 0 — immediate value, Burn parity (INT4 MinMax)
- **M2:** Phases 0+1+2 — AWQ, first calibration-aware method in Rust
- **M3:** Phase 3 — GPTQ, the industry workhorse
- **M4:** Phase 4 — SpinQuant, evolved PolarQuant with best-in-class quality
- **M5:** Phase 5 — QuIP#, state-of-the-art ≤4-bit
- **M6:** Phase 6 — D2Quant, sub-4-bit frontier (2-bit, 3-bit)

## Success Criteria

1. Quantize Llama-3-8B from FP16 SafeTensors → INT4 AWQ in pure Rust
2. Output matches HuggingFace AWQ reference within ±0.5 perplexity
3. Quantized model runs on Metal GPU via existing inference path
4. Zero Python dependencies in the entire pipeline
5. Performance: quantization completes in <30 min on M-series Mac
6. SpinQuant achieves ≤3pp perplexity gap at W4 on LLaMA-3 8B
7. QuIP# achieves usable 2-bit quantization (≤5pp perplexity gap)

## References

### Calibration-Aware Methods
- [GPTQ paper](https://arxiv.org/abs/2210.17323) — Frantar et al., 2022
- [AWQ paper](https://arxiv.org/abs/2306.00978) — Lin et al., 2023
- [GPTQ as Babai's nearest plane](https://openreview.net/forum?id=NFB4QGGS65) — geometric interpretation, 2026
- [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) — reference Python implementation
- [llm-awq](https://github.com/mit-han-lab/llm-awq) — reference AWQ implementation

### Rotation-Optimized Methods
- [SpinQuant paper](https://arxiv.org/abs/2405.16406) — Liu et al., ICLR 2025
- [SpinQuant code](https://github.com/facebookresearch/SpinQuant) — Meta reference implementation
- [QuIP# paper](https://arxiv.org/abs/2402.04396) — Tseng et al., ICML 2024
- [QuaRot](https://arxiv.org/abs/2404.00456) — random rotation baseline, 2024

### Sub-4-bit Methods
- [D2Quant paper](https://arxiv.org/abs/2602.02546) — Yan et al., 2026
- [ParetoQ](https://arxiv.org/abs/2502.02631) — scaling laws for 1-4 bit, 2025
- [KurTail](https://aclanthology.org/2025.findings-emnlp.943.pdf) — kurtosis-guided rotation, EMNLP 2025

### Surveys
- [JCST 2026 Quantization Survey](https://link.springer.com/content/pdf/10.1007/s11390-026-5979-1.pdf) — Tsinghua comprehensive review
- [Awesome-LLM-Quantization](https://github.com/pprp/Awesome-LLM-Quantization) — curated paper list
