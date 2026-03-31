# PolarQuant Weight Quantization

> Design doc for static model weight compression using PolarQuant.
>
> **Status**: Partially implemented — `PolarQuantPass` exists, needs ANE
> integration testing and pipeline wiring for the FP16 baseline path.

## Problem

Qwen3-0.6B in FP16 requires 1.5 GB for weights. Larger models (7B+) need
12–14 GB, exceeding ANE's practical memory budget. Weight quantization
reduces memory and improves throughput by fitting more of the model into
ANE's ~32 MB on-chip SRAM.

| Model | FP16 | 4-bit PolarQuant | Reduction |
|-------|------|-------------------|-----------|
| Qwen3-0.6B | 1.5 GB | ~400 MB | ~3.7× |
| Qwen3-4B | 8.2 GB | ~2.2 GB | ~3.7× |
| Llama-3-8B | 16 GB | ~4.3 GB | ~3.7× |

The reduction is ~3.7× rather than the theoretical 4× (16÷4 bits) because
small tensors below the `min_elements` threshold remain in FP16, and
per-row norms add a small overhead.

## How PolarQuant Works

PolarQuant is a rotation-based vector quantization method from Google
Research ([arXiv:2502.02617](https://arxiv.org/abs/2502.02617)). It
achieves high-quality quantization without per-group normalization
constants, reducing storage overhead compared to GPTQ or AWQ.

### Algorithm (per weight matrix)

```
Input:  W ∈ ℝ^{rows × cols}  (FP16 weight matrix)
Output: LUT + indices + row_norms  (compressed representation)

1. Row-normalize:     W_hat[i] = W[i] / ‖W[i]‖,  norms[i] = ‖W[i]‖
2. Pad to power-of-2: W_hat → W_padded ∈ ℝ^{rows × 2^k}
3. Rotate:            W_rot = W_padded × H(seed)
                      (seeded randomized Hadamard transform)
4. Quantize:          idx[i,j] = quantize(W_rot[i,j], boundaries)
                      (Beta-optimal scalar quantization)
5. Pack:              indices = pack(idx, n_bits)
6. Store:             constexpr_lut_to_dense(lut, indices, shape)
                      + mul(dequantized, norms)  [row-norm rescaling]
```

### Why rotation matters

Raw weight distributions have outliers that degrade uniform quantization.
The Hadamard rotation spreads information across all dimensions, producing
a distribution that the Beta-optimal quantizer handles well. The rotation
matrix is never stored — it's regenerated from the seed.

### Beta-optimal quantizer

Instead of uniform bin spacing, PolarQuant uses Lloyd-Max optimal levels
derived from the Beta(1/2, (dim-1)/2) distribution — the marginal
distribution of a squared coordinate on the unit sphere S^{d-1}. After
row-normalization and Hadamard rotation, each scalar in the weight
matrix approximately follows this distribution. This is implemented
in `beta_quantizer.rs`:

- `beta_optimal_levels(dim, n_bits)` → reconstruction levels
- `beta_optimal_boundaries(dim, n_bits)` → quantization thresholds
- `quantize_to_index(value, boundaries)` → codebook index

## Existing Implementation

### Core passes (`crates/mil-rs/src/ir/passes/`)

| File | Purpose |
|------|---------|
| `polar_quantize.rs` | `PolarQuantPass` — the main quantization pass |
| `polar_rotation_fusion.rs` | `PolarRotationFusionPass` — cancels rotations between adjacent layers |
| `beta_quantizer.rs` | Beta-optimal level/boundary computation |
| `rotation.rs` | Hadamard rotation utilities |

### How `PolarQuantPass` works

For each `const` op with ≥ `min_elements` FP16/FP32 values:

1. Extracts weight data, reshapes to 2D `[rows, cols]`
2. Computes per-row L2 norms, normalizes rows
3. Pads columns to next power of 2
4. Applies seeded randomized Hadamard rotation
5. Quantizes each scalar using Beta-optimal boundaries
6. Packs indices (n_bits per value)
7. Replaces the `const` op with:
   - `constexpr_lut_to_dense(lut=levels, indices=packed, shape=original)`
   - `mul(dequantized_weight, row_norms)` to restore scale

### Rotation fusion

When two adjacent linear layers both use PolarQuant with the same seed
and are connected only through safe elementwise ops (relu, add, mul,
sub), the Hadamard rotation at the output of layer A and the inverse
rotation at the input of layer B cancel out.
`PolarRotationFusionPass` detects this pattern and eliminates the
redundant rotation, saving compute. For unpaired boundary layers (e.g.,
broken by softmax or layernorm), it inserts an explicit inverse-rotation
matmul.

Note: Q/K/V projections and gate/up projections in standard transformers
are parallel branches, not sequential — rotation fusion does not apply
to them. It does apply to genuinely sequential linear chains, such as
down-projection following an activation.

## Backend-Specific Impact

PolarQuant's benefits vary significantly by inference backend:

| Benefit | ANE (private API) | CoreML GPU (public API) | Metal GPU (direct) |
|---------|-------------------|-------------------------|---------------------|
| Disk / distribution size | ✅ 3.7× smaller | ✅ 3.7× smaller | ✅ 3.7× smaller |
| In-memory weight size | ❌ FP16 after dequant | ✅ compressed in VRAM | ✅ compressed in VRAM |
| Throughput | ❌ no gain | ⚠️ marginal | ✅ bandwidth reduction |

**Metal GPU (direct)** is the strongest fit. GPU inference is typically
memory-bandwidth-bound; reading 4-bit weights instead of 16-bit reduces
bandwidth pressure ~4× per matmul. The dequantization cost (LUT lookup
per index) is trivially cheap in a Metal shader — a few ALU ops per
element, easily hidden by the memory latency it saves. Weights stay
compressed in `MTLBuffer`s, so a 7B model fits in ~4 GB VRAM instead
of 14 GB.

**CoreML GPU** handles `constexpr_lut_to_dense` natively and likely
keeps weights compressed in VRAM, but Apple's documentation is vague
on the runtime representation. Dequantization overhead is opaque.

**ANE** cannot use compressed weights at runtime — the private API
rejects `constexpr_lut_to_dense` and requires FP16 `const` ops.
PolarQuant is useful only for smaller model files on disk.

## ANE Constraints — Verified Limitations

### Supported MIL ops

The `constexpr_lut_to_dense` op is the CoreML mechanism for palettized
weights. ANE supports it with LUT sizes in {2, 4, 16, 64, 256},
corresponding to n_bits ∈ {1, 2, 4, 6, 8}.

### INT4 weight rejection — compile-time only

ANE rejects INT4/UINT4 data types for runtime tensors. The
`constexpr_lut_to_dense` op is a **compile-time** expansion in CoreML's
public API — the CoreML compiler dequantizes weights and passes FP16
to the ANE backend.

**Verified (2026-03):** ANE's private API (used by ironmill) **rejects
`constexpr_lut_to_dense` outright** — `ANECCompile() FAILED`. The
private API only accepts ops in the ANE native op set (conv, add, mul,
etc.). The `constexpr_lut_to_dense` dequantization is performed by
CoreML's compiler layer, not by ANE itself.

**This is a critical limitation.** PolarQuant's memory savings apply to:
- ✅ **Model file size on disk** (4-bit indices + small LUT)
- ✅ **Model loading time** (less data to read from storage)
- ✅ **Distribution size** (smaller downloads)

But NOT to:
- ❌ **ANE on-chip weight storage** (always FP16 after dequantization)
- ❌ **Runtime memory** (FP16-sized after dequantization)
- ❌ **Inference speed** (no reduced-precision arithmetic)

For the ANE private API path, PolarQuant must **dequantize to FP16 on
the CPU** before passing weights to the ANE compiler. The quantized
format is used only for storage and distribution.

For true on-chip compression, ANE would need runtime palettization
support, which is not available through either the private or public API.
The GPU backend (via CoreML public API) DOES support
`constexpr_lut_to_dense` and would benefit from smaller model files.

## Integration Plan

### Phase 1: Static weight compression (model file size)

Since ANE's private API rejects `constexpr_lut_to_dense`, PolarQuant
integration must use a **load-time dequantization** approach:

```
Compressed file (4-bit PolarQuant)
  → CPU dequantize to FP16 (on model load)
  → MIL IR with FP16 const ops
  → ANE compile pipeline
  → ANE blob (FP16 weights)
```

This saves disk/distribution size but not runtime memory. The
dequantization step runs once at model load time.

**Current status**: `PolarQuantPass` is wired into the MIL `PassPipeline`
and accessible via `--polar-quantize <BITS>` in the CLI and bench tools
for offline compression. A corresponding **dequantization pass** is
needed to expand `constexpr_lut_to_dense` → FP16 `const` ops before
ANE compilation.

**Configuration**: `--polar-quantize 4` CLI flag exists in `ironmill-cli`
and `ironmill-bench`.

**Expected results**:
- ONNX file → ~400 MB compressed (vs 1.5 GB FP16)
- Compilation time may increase (dequantization during compile)
- PPL impact: minimal (PolarQuant at 4-bit typically <0.5 PPL increase)

### Phase 2: Rotation fusion

Run `PolarRotationFusionPass` after `PolarQuantPass` to cancel
Hadamard rotations between sequential linear layers connected through
safe elementwise ops. This is already wired via `pipeline.with_polar_quant()`,
which schedules both passes together.

### Phase 3: Combined with TurboQuant

PolarQuant (static weights) + TurboQuant (dynamic KV cache) gives
comprehensive compression:

| Component | Method | Precision | Reduction |
|-----------|--------|-----------|-----------|
| Weights | PolarQuant | 4-bit LUT | 3.7× |
| KV cache | TurboQuant | INT8 | 2× |
| KV cache | TurboQuant + QJL | 1-bit signs | 6× |

### Phase 4: Calibration-aware quantization

Current PolarQuantPass is data-free (no calibration set). Future work:
- GPTQ-style layer-wise calibration using a small dataset
- Mixed-precision: 4-bit for most layers, 8-bit for sensitive ones
  (first/last layers, attention projections)
- Per-expert quantization for MoE models (already supported by
  `PerExpertQuantPass`)

## Comparison with Other Methods

| Method | Bits | Normalization | Calibration | PPL (7B, WikiText-2) |
|--------|------|---------------|-------------|---------------------|
| FP16 baseline | 16 | — | — | ~6 |
| GPTQ | 4 | per-group | yes | ~6.5 |
| AWQ | 4 | per-group | yes | ~6.4 |
| PolarQuant | 4 | none (rotation) | no | ~6.3 |
| PolarQuant | 2 | none (rotation) | no | ~8 |

PolarQuant's advantage: no per-group scales/zeros stored, and no
calibration data needed. The Beta-optimal quantizer is derived
analytically from the weight distribution after rotation.

## References

- [PolarQuant paper](https://arxiv.org/abs/2502.02617) — Han, Kacham, Karbasi, Mirrokni, Zandieh, 2025
- [TurboQuant paper](https://arxiv.org/abs/2504.19874) — Zandieh, Daliri, Hadian, Mirrokni, 2025
- [AISTATS 2026 poster](https://virtual.aistats.org/virtual/2026/poster/11018) — Vector quantization generalization
- `crates/mil-rs/src/ir/passes/polar_quantize.rs` — Implementation
- `crates/mil-rs/src/ir/passes/beta_quantizer.rs` — Quantizer math
- `docs/design/turboquant.md` — KV cache compression design
