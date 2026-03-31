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
| Qwen3-0.6B | 1.5 GB | ~400 MB | 3.7× |
| Qwen3-4B | 8.2 GB | ~2.2 GB | 3.7× |
| Llama-3-8B | 16 GB | ~4.3 GB | 3.7× |

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
a near-Gaussian distribution that the Beta-optimal quantizer handles well.
The rotation matrix is never stored — it's regenerated from the seed.

### Beta-optimal quantizer

Instead of uniform bin spacing, PolarQuant uses Lloyd-Max optimal levels
derived from the Beta(dim/2, dim/2) distribution (the marginal
distribution of a random unit vector's projection). This is implemented
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

When two adjacent linear layers both use PolarQuant with the same seed,
the Hadamard rotation at the output of layer A and the inverse rotation
at the input of layer B cancel out. `PolarRotationFusionPass` detects
this pattern and eliminates the redundant rotation, saving compute.

## ANE Constraints — Important Caveat

### Supported MIL ops

The `constexpr_lut_to_dense` op is the CoreML mechanism for palettized
weights. ANE supports it with LUT sizes in {2, 4, 16, 64, 256},
corresponding to n_bits ∈ {1, 2, 4, 6, 8}.

### INT4 weight rejection — compile-time only

ANE rejects INT4/UINT4 data types for runtime tensors. However,
`constexpr_lut_to_dense` is a **compile-time** expansion — the ANE
compiler dequantizes the weights during compilation and stores them
in the `.espresso.net` blob in FP16 format. The quantization saves
storage/transfer size, but the on-chip representation is still FP16.

**This is a critical limitation.** PolarQuant's memory savings apply to:
- ✅ **Model file size on disk** (4-bit indices + small LUT)
- ✅ **Model loading time** (less data to read from storage)
- ✅ **Distribution size** (smaller downloads)

But NOT to:
- ❌ **ANE on-chip weight storage** (always FP16 after compilation)
- ❌ **Runtime memory** (compiled blob is FP16-sized)
- ❌ **Inference speed** (no reduced-precision arithmetic)

For true on-chip compression, ANE would need runtime palettization
support, which is not currently available through the private API.
The GPU backend (via CoreML public API) DOES support runtime
palettization and would benefit fully from PolarQuant.

## Integration Plan

### Phase 1: Static weight compression (model file size)

Wire `PolarQuantPass` into the ANE compilation pipeline:

```
ONNX → MIL IR → split → [PolarQuantPass(4-bit)] → MatmulToConv
     → AneLayoutPass → compile → ANE blob
```

**Where it runs**: After splitting, before `compile_and_load_sub`.
Each sub-program's weight `const` ops are quantized independently.

**Configuration**: Add `--quantize polar:4` CLI flag to `ironmill-bench`.

**Expected results**:
- ONNX file → ~400 MB compressed (vs 1.5 GB FP16)
- Compilation time may increase (dequantization during compile)
- PPL impact: minimal (PolarQuant at 4-bit typically <0.5 PPL increase)

### Phase 2: Rotation fusion

Run `PolarRotationFusionPass` after `PolarQuantPass` to cancel
Hadamard rotations between consecutive linear layers (Q→K→V
projections, gate→up→down FFN projections).

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

- [PolarQuant paper](https://arxiv.org/abs/2502.02617) — Shwu et al., 2025
- [TurboQuant](https://research.google/pubs/polarquant-quantizing-kv-caches-with-polar-transformation/) — Google Research
- [AISTATS 2026 poster](https://virtual.aistats.org/virtual/2026/poster/11018) — Vector quantization generalization
- `crates/mil-rs/src/ir/passes/polar_quantize.rs` — Implementation
- `crates/mil-rs/src/ir/passes/beta_quantizer.rs` — Quantizer math
- `docs/design/turboquant.md` — KV cache compression design
