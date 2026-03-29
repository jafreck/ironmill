# PolarQuant Weight Quantization — Implementation Plan

## 1. Overview

PolarQuant (Google, AISTATS 2026) enables high-quality weight compression at
3–4 bits per coordinate — below INT8 but above the quality floor of naive
low-bit quantization — by exploiting a mathematical property: after a random
orthogonal rotation, the coordinates of a high-dimensional vector follow a
concentrated Beta distribution with an analytically known form. This eliminates
the need for per-block normalization (scale/zero-point) and allows the use of a
precomputed, distribution-optimal scalar quantizer.

This plan describes how to add PolarQuant as a new static weight quantization
pass in ironmill, targeting 2–4 bit weight compression with minimal quality
loss. The pass fits alongside the existing `fp16-quantization`,
`int8-quantization`, and `palettization` passes.

### Goals

- Compress weight tensors to 2–4 bits per coordinate with better quality than
  naive low-bit quantization and comparable quality to INT8 at ~2× smaller size.
- Require no calibration data (the quantizer is data-oblivious).
- Emit standard CoreML ops so compiled models run on ANE/GPU/CPU without
  custom runtime changes.

### Non-goals

- Runtime KV cache compression (that requires inference engine changes).
- QJL residual correction (the second stage of full TurboQuant — designed for
  online inner-product estimation, not static weights).

---

## 2. Algorithm (adapted for static weights)

For each weight tensor W with shape `[M, N]` where `N ≥ 64`:

```
1.  Generate a random rotation matrix R ∈ ℝ^(N×N).
    Use a randomized Hadamard transform for O(N log N) cost:
      R = (1/√N) · D · H_N
    where H_N is the Walsh-Hadamard matrix and D is a diagonal ±1 matrix
    drawn from a fixed seed.

2.  Rotate the input dimension of W:
      W' = W · Rᵀ                    # shape [M, N]
    Each row of W' now has coordinates ≈ Beta(½, (N-1)/2) after normalization.

3.  For target bit-width b (e.g., 4), look up the precomputed optimal
    quantization levels {q_1, …, q_{2^b}} for Beta(½, (N-1)/2).
    These levels are the Lloyd-Max quantizer boundaries for the Beta CDF.

4.  Quantize each element of W' independently:
      idx[i,j] = argmin_k |W'[i,j] − q_k|
    Pack indices at b bits each.

5.  Store:
    - Packed indices   (b bits per element)
    - LUT of levels    (2^b float16 values)
    - Rotation seed    (single u64, enough to regenerate R)
    - Norm per row     (float16, one per row — the Beta distribution
                        describes the *direction*; magnitude is separate)
    - Row norms are stored as a float16 const tensor with shape [M, 1]
      and multiplied back during dequantization:
        W'_dequant[i,j] = LUT[indices[i,j]] * row_norms[i]
```

At inference time, CoreML reconstructs the weight via:
```
W_approx = LUT[indices] · row_norms  →  un-rotate  →  W_approx · R
```

### Why this beats existing options

| Method | Bits | Needs calibration? | Per-block overhead |
|--------|------|--------------------|--------------------|
| INT8 affine | 8 | Optional | scale + zero_point per tensor/channel |
| Palettization (k-means) | 1–8 | No (but data-dependent clustering) | LUT per tensor |
| **PolarQuant** | **2–4** | **No** | **LUT (fixed, precomputed) + seed** |

PolarQuant's LUT is derived from the distribution, not learned from the data.
Two models with the same dimension and bit-width share the exact same LUT.

---

## 3. Architecture & file layout

```
crates/mil-rs/src/ir/passes/
├── mod.rs                       # add: pub mod polar_quantize; pub mod rotation;
│                                #       pub mod beta_quantizer;
│                                #       pub mod polar_rotation_fusion;
├── rotation.rs                  # NEW — Hadamard transform + rotation utilities
├── polar_quantize.rs            # NEW — PolarQuantPass implementation
├── polar_rotation_fusion.rs     # NEW — PolarRotationFusionPass (inter-layer fusion)
├── beta_quantizer.rs            # NEW — precomputed Beta-optimal quantizer levels
└── tensor_utils.rs              # extend: add f16 decode helper if needed

crates/mil-rs/src/ir/
├── pipeline.rs                  # register "polar-quantization" pass; add has_polar_quant
├── tensor.rs                    # no changes expected
└── types.rs                     # no changes expected

crates/mil-rs/src/convert/
└── ir_to_proto.rs               # verify constexpr_lut_to_dense handles our payloads

crates/ironmill-cli/src/
└── main.rs                      # add --polar-quantize <BITS> flag

crates/mil-rs/tests/
├── optimization_passes.rs       # add PolarQuant unit tests
└── cross_feature.rs             # add PolarQuant + pipeline composition tests
```

---

## 4. Implementation tasks

### Task 1 — Rotation utilities (`rotation.rs`)

Create `crates/mil-rs/src/ir/passes/rotation.rs`.

**Public API:**

```rust
/// Applies a seeded randomized Hadamard transform to a matrix in-place.
///
/// `data` is a row-major [rows, cols] matrix stored as &mut [f32].
/// Each row is rotated independently. `cols` must be a power of two;
/// pad with zeros if necessary.
pub fn rotate_rows_hadamard(data: &mut [f32], rows: usize, cols: usize, seed: u64);

/// Applies the inverse (transpose) of the same rotation.
pub fn unrotate_rows_hadamard(data: &mut [f32], rows: usize, cols: usize, seed: u64);

/// Pads `cols` up to the next power of two. Returns (padded_data, padded_cols).
pub fn pad_to_power_of_two(data: &[f32], rows: usize, cols: usize) -> (Vec<f32>, usize);
```

**Implementation notes:**

- The Walsh-Hadamard transform is applied in-place with O(N log N) per row
  using the standard butterfly algorithm.
- The diagonal sign matrix D is generated from a seeded PRNG (use `rand`
  crate with `StdRng::seed_from_u64`). The seed is stored in the model so
  the same rotation can be reproduced at load time.
- Inverse transform = apply D then H again (Hadamard is self-inverse; D is
  self-inverse). So `unrotate = rotate` with the same seed.

**Dependencies:** `rand` crate — must be added as a new workspace dependency
in the root `Cargo.toml` (`[workspace.dependencies]` section) and then
referenced in `crates/mil-rs/Cargo.toml` as `rand = { workspace = true }`.
`rand` is not currently a direct or transitive dependency of any workspace
crate (`kmeans.rs` is explicitly deterministic — max-distance init, no
randomness).

---

### Task 2 — Beta-optimal scalar quantizer (`beta_quantizer.rs`)

Create `crates/mil-rs/src/ir/passes/beta_quantizer.rs`.

**Public API:**

```rust
/// Returns the optimal reconstruction levels for quantizing a
/// Beta(0.5, (dim-1)/2) random variable to `n_bits` bits.
///
/// Returns `2^n_bits` levels as f32, sorted ascending.
/// These are the Lloyd-Max centroids for the Beta distribution.
pub fn beta_optimal_levels(dim: usize, n_bits: u8) -> Vec<f32>;

/// Returns the decision boundaries (midpoints between adjacent levels).
/// Length: 2^n_bits - 1.
pub fn beta_optimal_boundaries(dim: usize, n_bits: u8) -> Vec<f32>;

/// Quantize a single value to the nearest level index.
pub fn quantize_to_index(value: f32, boundaries: &[f32]) -> u8;
```

**Implementation notes:**

- Precompute levels for common (dim, n_bits) pairs at compile time using
  `const` tables or `lazy_static`. Practical dims: 64, 128, 256, 512, 1024,
  2048, 4096. Practical bits: 2, 3, 4.
- For uncommon dims, compute on the fly using the Beta CDF inverse
  (`betaincinv`) to place initial levels at quantiles, then run a few
  iterations of the Lloyd-Max algorithm on the analytical distribution.
- The Beta CDF/inverse CDF can be computed via the regularized incomplete
  beta function. Use the `statrs` crate, which is already a workspace
  dependency (used by `ironmill-bench`).

**Dependencies:** Add `statrs = { workspace = true }` to
`crates/mil-rs/Cargo.toml`. The crate is already declared in the workspace
root (`statrs = "0.18"`) and used by `ironmill-bench`, so this adds no new
external dependency to the lockfile.

---

### Task 3 — PolarQuant pass (`polar_quantize.rs`)

Create `crates/mil-rs/src/ir/passes/polar_quantize.rs`.

**Pass struct:**

```rust
/// PolarQuant weight quantization pass.
///
/// Applies a random rotation (Hadamard transform) to weight tensors,
/// then quantizes each coordinate using a distribution-optimal scalar
/// quantizer at the specified bit-width.
pub struct PolarQuantPass {
    /// Target bit-width per coordinate (2, 3, or 4).
    pub n_bits: u8,
    /// Fixed seed for the rotation matrix (reproducibility).
    pub seed: u64,
    /// Minimum tensor size (in elements) to quantize.
    /// Small tensors are left unquantized.
    pub min_elements: usize,
}
```

**Pass logic (following INT8 pass pattern):**

```
for each function in program:
  for each op in function.body:
    if op.op_type != "const":
      continue
    if op value is not Tensor { dtype: Float32 } with numel >= min_elements:
      continue

    1. Extract tensor data as Vec<f32>, shape
    2. Interpret as [rows, cols] matrix:
       - For rank ≥ 2: rows = product(shape[..rank-1]), cols = shape[rank-1]
       - For rank 1: rows = 1, cols = shape[0]
    3. Pad cols to next power of two
    4. Compute row norms; normalize each row to unit length
    5. Apply rotate_rows_hadamard(data, rows, padded_cols, seed)
    6. Get Beta-optimal levels and boundaries for (padded_cols, n_bits)
    7. Quantize each element → index; pack indices at n_bits
    8. If cols was padded, truncate indices back to original cols
       (discard the padded columns — they are not part of the original
       weight and are only needed during rotation)
    9. Build LUT from the Beta-optimal levels (cast to f16 for storage)
   10. Store row norms as a separate float16 const tensor [rows, 1]
   11. Mutate op:
       - op.op_type = "constexpr_lut_to_dense"
       - attrs["lut"] = Value::Tensor { data: lut_bytes, shape: [2^n_bits], dtype: Float16 }
       - attrs["indices"] = Value::Tensor { data: packed_indices, shape: original_shape, dtype: UInt8 }
       - attrs["shape"] = original shape
       - Store rotation seed as op metadata for the fusion pass
   12. Insert a const op for row_norms and a mul op to scale the
       dequantized output: dequant * row_norms
```

**Note:** The PolarQuant pass itself only handles per-tensor rotation and
quantization. The inter-layer rotation fusion (pairing adjacent linears so
rotations cancel) is handled by `PolarRotationFusionPass` (Task 7), which
runs as a mandatory follow-up pass in the pipeline.

**Un-rotation strategy — Layered approach:**

The rotated, quantized weight needs to be un-rotated at inference time.
We use a two-tier strategy:

- **Primary: Fused inter-layer rotation** — for paired consecutive linear
  ops, the rotation applied to one layer's output dimension cancels with
  the rotation applied to the next layer's input dimension. Zero storage
  overhead, zero load-time cost.
- **Fallback: Explicit matmul** — for unpaired layers (boundary layers,
  layers adjacent to non-fusible ops), emit an explicit `matmul` with the
  inverse rotation matrix. This still gives full PolarQuant compression on
  those layers instead of leaving them at FP16/INT8.

**Tier 1 — Fused rotation (paired layers):**

For consecutive linear layers `y = W₂ · act(W₁ · x)`:
- Rotate W₁'s output dimension (rows) with R: `W₁_rot = R · W₁`
- Rotate W₂'s input dimension (columns) with the same R: `W₂_rot = W₂ · R^T`
- The rotations cancel through the connecting activation:
  `W₂_rot · act(W₁_rot · x) = W₂ · R^T · act(R · W₁ · x)`
  For ReLU-family activations, `act(R · z) ≈ R · act(z)` is a good
  empirical approximation when R is a Hadamard rotation, because Hadamard
  rotations distribute energy evenly across coordinates (near-isometry),
  minimizing the number of sign flips. The product therefore collapses to
  approximately `W₂ · W₁ · x`.
- No extra matmul or rotation matrix storage needed.

**Tier 2 — Explicit matmul (unpaired layers):**

For layers that cannot be paired (first/last in a chain, adjacent to
non-fusible ops):
```
constexpr_lut_to_dense → dequantized W'  (rotated domain)
const(R_inv)           → rotation matrix  (N×N float16)
matmul(W', R_inv)      → W_approx         (unrotated)
```
- R_inv is stored as a float16 const. For typical transformer head_dim=128,
  this is 128×128×2 = 32 KiB per unique dimension — negligible.
- CoreML materializes `constexpr_*` ops at model load, so the matmul runs
  once at load time, not per-inference.

**Pairing algorithm (`PolarRotationFusionPass`):**

The pass walks the op graph and pairs adjacent linear ops (matmul, linear,
conv) that share a connecting dimension. Each pair shares a rotation seed:

```
1.  Build a dependency graph of linear-family ops.
2.  Greedily pair consecutive ops: for each linear op L₁, if its sole
    consumer is another linear op L₂ (possibly through an element-wise
    activation), pair (L₁, L₂) with a shared seed.
3.  Unpaired ops at chain boundaries (first input layer, final output
    layer, or layers adjacent to non-fusible ops like softmax or
    layer-norm) use the explicit matmul fallback: emit
    constexpr_lut_to_dense + const(R_inv) + matmul.
4.  Assign a unique rotation seed per pair.
```

**Edge cases:**
- **Unpaired boundary layers:** Use explicit matmul fallback. These are
  typically the embedding layer and the final projection head — the R_inv
  overhead is small, and they still get full PolarQuant compression.
- **Non-fusible ops between linears (softmax, layernorm, etc.):** Break the
  chain. Each side of the break uses the matmul fallback.
- **Single isolated linear ops:** Use the matmul fallback.

---

### Task 4 — Pipeline and CLI integration

#### 4a. Pass registration (`pipeline.rs`)

```rust
// In KNOWN_PASSES:
"polar-quantization",

// In pass_from_name:
"polar-quantization" => {
    let n_bits = params.get("bits")
        .and_then(|v| v.as_integer())
        .unwrap_or(4) as u8;
    let seed = params.get("seed")
        .and_then(|v| v.as_integer())
        .unwrap_or(42) as u64;
    let min_elements = params.get("min_elements")
        .and_then(|v| v.as_integer())
        .unwrap_or(1024) as usize;
    Ok(Box::new(PolarQuantPass { n_bits, seed, min_elements }))
}
```

Add builder method:

```rust
pub fn with_polar_quant(mut self, n_bits: u8) -> Result<Self> {
    if self.has_fp16 || self.has_int8 || self.has_palettize {
        return Err(MilError::Validation(
            "polar-quantization is mutually exclusive with fp16/int8/palettization".into(),
        ));
    }
    if !matches!(n_bits, 2 | 3 | 4) {
        return Err(MilError::Validation(format!(
            "polar-quantize n_bits must be 2, 3, or 4, got {n_bits}"
        )));
    }
    if n_bits == 2 {
        log::warn!(
            "2-bit PolarQuant may produce significant quality loss on some \
             architectures; consider 3- or 4-bit for production use"
        );
    }
    self.has_polar_quant = true;
    self.passes.push(Box::new(PolarQuantPass::new(n_bits)));
    self.passes.push(Box::new(PolarRotationFusionPass::new()));
    Ok(self)
}
```

#### 4b. CLI integration (`main.rs`)

Add a dedicated `--polar-quantize <BITS>` flag, following the same pattern as
the existing `--palettize <BITS>` flag (separate from `--quantize`):

```rust
/// PolarQuant weight quantization bit-width (2, 3, or 4).
#[arg(long = "polar-quantize", value_name = "BITS")]
polar_quantize: Option<u8>,
```

Dispatch logic (after the existing `--palettize` block):

```rust
if let Some(bits) = opts.polar_quantize {
    pipeline = pipeline.with_polar_quant(bits)?;
}
```

The `with_polar_quant()` builder method handles mutual exclusion with
`fp16`/`int8`/`palettize` (see §4a), matching how `with_palettize()` rejects
conflicts with INT8.

#### 4c. TOML config support

```toml
[[passes]]
name = "polar-quantization"
enabled = true

[passes.params]
bits = 4
seed = 42
min_elements = 1024
```

---

### Task 5 — Serialization verification

PolarQuant emits `constexpr_lut_to_dense` (already supported) plus `const`
and `matmul` ops (both already supported). No changes needed to
`ir_to_proto.rs`, but verify:

1. `constexpr_lut_to_dense` with our LUT shape/dtype serializes correctly.
2. The rotation matrix const serializes as Float16 tensor.
3. The inserted matmul op has correct input/output wiring.

Add a round-trip test: IR → protobuf → IR → verify op structure.

---

### Task 6 — Tests

#### Unit tests (`optimization_passes.rs`)

| Test | What it verifies |
|------|-----------------|
| `polar_quant_converts_const_to_lut` | `const` → `constexpr_lut_to_dense` with rotation metadata |
| `polar_quant_preserves_output_shape` | Output tensor shape matches original weight shape |
| `polar_quant_round_trip_quality` | Dequantized weight MSE is within theoretical bound for the bit-width |
| `polar_quant_skips_small_tensors` | Tensors below `min_elements` are left as-is |
| `polar_quant_skips_non_float32` | Non-FP32 tensors are not touched |
| `polar_quant_handles_non_power_of_two` | Tensors with non-power-of-two inner dim are padded/handled correctly |
| `polar_quant_deterministic_with_seed` | Same seed produces identical quantized output |
| `polar_quant_different_seeds_differ` | Different seeds produce different (but equally valid) output |
| `polar_quant_rejects_invalid_bits` | `n_bits` outside {2, 3, 4} returns an error |
| `polar_quant_rejects_zero_bits` | `n_bits = 0` is rejected |
| `polar_quant_handles_empty_program` | Pass is a no-op on programs with no const ops |

#### Quality tests

| Test | What it verifies |
|------|-----------------|
| `polar_4bit_better_than_naive_4bit` | PolarQuant-4 MSE < naive uniform-4-bit MSE on random Gaussian weights |
| `polar_4bit_comparable_to_int8` | PolarQuant-4 MSE is within 2× of INT8 MSE (half the bits) |
| `polar_3bit_acceptable_quality` | PolarQuant-3 MSE stays below a defined threshold |

#### Rotation fusion tests (`polar_rotation_fusion.rs`)

| Test | What it verifies |
|------|-----------------|
| `fusion_pairs_consecutive_linears` | Two adjacent matmul ops are paired with a shared seed |
| `fusion_pairs_through_relu` | Linear → ReLU → linear chain is paired |
| `fusion_breaks_at_softmax` | Softmax between linears breaks the chain |
| `fusion_breaks_at_layernorm` | LayerNorm between linears breaks the chain |
| `fusion_unpaired_gets_matmul_fallback` | Unpaired boundary layers emit constexpr + const(R_inv) + matmul |
| `fusion_handles_branching_graph` | Ops with multiple consumers are not incorrectly paired |
| `fused_pair_quality_vs_unfused` | Fused pair MSE ≈ unfused pair MSE on random weights |
| `fallback_matmul_round_trip_correct` | Explicit matmul un-rotation recovers original weight within tolerance |

#### Pipeline composition tests (`cross_feature.rs`)

| Test | What it verifies |
|------|-----------------|
| `polar_quant_mutually_exclusive_with_int8` | Pipeline rejects combining both |
| `polar_quant_mutually_exclusive_with_fp16` | Pipeline rejects combining both |
| `polar_quant_mutually_exclusive_with_palettize` | Pipeline rejects combining both |
| `polar_quant_works_with_attention_fusion` | Fusion passes + PolarQuant compose correctly |
| `polar_quant_works_with_kv_cache_pass` | KV cache pass + PolarQuant compose correctly |

#### Rotation utility tests (`rotation.rs`)

| Test | What it verifies |
|------|-----------------|
| `hadamard_is_orthogonal` | R · R^T ≈ I (within floating point tolerance) |
| `hadamard_is_self_inverse` | rotate then rotate again ≈ identity |
| `hadamard_preserves_norm` | ‖Rx‖ ≈ ‖x‖ |
| `pad_to_power_of_two_correct` | Padding works for various input sizes |

#### Beta quantizer tests (`beta_quantizer.rs`)

| Test | What it verifies |
|------|-----------------|
| `levels_sorted_ascending` | Quantization levels are monotonically increasing |
| `levels_count_matches_bits` | 2^n_bits levels returned |
| `levels_within_unit_interval` | All levels in [0, 1] (Beta distribution support) |
| `quantize_round_trip_bounded_error` | Quantize → dequantize error bounded by half the max bin width |

---

### Task 8 — Benchmarks (`ironmill-bench`)

Add PolarQuant to the existing benchmark harness in `crates/ironmill-bench/`.

#### 8a. Benchmark config (`config.rs`)

Add PolarQuant entries to the default benchmark matrix alongside the existing
`int8` and `palettize-4` configs:

```rust
OptConfig {
    name: "polar-4".to_string(),
    quantize: None,
    palettize: None,
    polar_quantize: Some(4),
    no_fusion: false,
    disabled_passes: vec![],
},
OptConfig {
    name: "polar-3".to_string(),
    quantize: None,
    palettize: None,
    polar_quantize: Some(3),
    no_fusion: false,
    disabled_passes: vec![],
},
```

Add the `polar_quantize: Option<u8>` field to `OptConfig` (defaulting to
`None` for existing configs).

#### 8b. Compiler integration (`compiler.rs`)

Wire `polar_quantize` into the pipeline construction, following the
existing `palettize` pattern:

```rust
if let Some(bits) = opt.polar_quantize {
    pipeline = pipeline.with_polar_quant(bits)?;
}
```

#### 8c. Quality benchmarks (new: `quality.rs`)

Add a quality benchmark module that measures **quantization fidelity**, not
just inference latency. This enables automated tracking of the quality
claims in §6 (Tasks 6's quality tests).

```rust
/// For each (model, bit-width) pair:
/// 1. Load the original FP32 weights.
/// 2. Apply PolarQuant at the target bit-width.
/// 3. Dequantize and compute per-tensor MSE vs. original.
/// 4. Compare against INT8 and palettize-4 MSE on the same weights.
/// 5. Report: MSE, PSNR, compression ratio, and relative quality
///    (polar-4 MSE / int8 MSE).
pub fn run_quality_benchmarks(matrix: &BenchMatrix, settings: &Settings) -> Vec<QualityResult>;
```

Extend `report.rs` to include a quality summary table in the benchmark
output:

| Model | Method | Bits | MSE | PSNR (dB) | vs INT8 | Size ratio |
|-------|--------|------|-----|-----------|---------|------------|

#### 8d. CLI flag

Add `--quality` flag to `ironmill-bench` to run quality benchmarks in
addition to (or instead of) latency benchmarks.

---

## 5. CoreML compatibility

### Emission format

PolarQuant emits different op patterns depending on whether a layer is
paired (fused) or unpaired (fallback):

**Paired layers (fused rotation — no extra ops):**
```
┌─────────────────────────────────────┐
│ constexpr_lut_to_dense (W₁)        │  W₁ quantized in rotated domain
│   The rotation is baked into the    │  (rows rotated by R before quantization)
│   quantized representation itself.  │
├─────────────────────────────────────┤
│ const(row_norms₁) [M₁, 1] f16     │  row magnitudes (separated from direction)
├─────────────────────────────────────┤
│ mul(W₁_dequant, row_norms₁)        │  restore per-row scale
├─────────────────────────────────────┤
│ matmul(x, W₁_scaled)               │  output is in rotated activation space
├─────────────────────────────────────┤
│ activation (relu, gelu, etc.)       │  applied in rotated space
├─────────────────────────────────────┤
│ constexpr_lut_to_dense (W₂)        │  W₂ quantized with matching input rotation
│   Input columns rotated by same R.  │  (R^T absorbed into W₂ before quantization)
├─────────────────────────────────────┤
│ const(row_norms₂) [M₂, 1] f16     │  row magnitudes for W₂
├─────────────────────────────────────┤
│ mul(W₂_dequant, row_norms₂)        │  restore per-row scale
├─────────────────────────────────────┤
│ matmul(act_out, W₂_scaled)          │  R^T · R cancels → unrotated output
└─────────────────────────────────────┘
```

**Unpaired layers (explicit matmul fallback):**
```
┌─────────────────────────────────────┐
│ constexpr_lut_to_dense              │  ← quantized + LUT (existing CoreML op)
│   lut: [2^b] float16 levels        │
│   indices: packed b-bit indices     │
│   shape: original weight shape      │
│   output: W_rotated (float16)       │
├─────────────────────────────────────┤
│ const(row_norms) [M, 1] f16        │  ← per-row magnitudes
├─────────────────────────────────────┤
│ mul(W_rotated, row_norms)           │  ← restore per-row scale
├─────────────────────────────────────┤
│ const                               │  ← inverse rotation matrix
│   val: R_inv [N, N] float16        │
├─────────────────────────────────────┤
│ matmul                              │  ← un-rotation (one-time at model load)
│   x: W_scaled                       │
│   y: R_inv                          │
│   output: W_approx (float16)        │
└─────────────────────────────────────┘
```

All emitted ops are standard MIL/CoreML operations. No custom ops needed.

### Memory at load time

CoreML materializes `constexpr_*` ops at model load. For paired layers,
this is plain LUT dequantization — no extra cost. For unpaired layers,
the rotation matmul also runs at load time (its inputs are all constants,
so CoreML constant-folds it). At inference time, both paths see plain
float16 weight tensors — zero per-inference overhead.

### ANE compatibility

The materialized float16 weights are ANE-compatible for both paths. The
fallback matmul runs at model load time on CPU/GPU (one-time constant
folding). No ANE concerns.

### Storage savings

| Bit-width | Bytes per element | vs FP16 | vs INT8 |
|-----------|-------------------|---------|---------|
| 4-bit | 0.5 + LUT amortized | 4× smaller | 2× smaller |
| 3-bit | 0.375 + LUT amortized | 5.3× smaller | 2.7× smaller |
| 2-bit | 0.25 + LUT amortized | 8× smaller | 4× smaller |

With fused rotation, paired layers have zero storage overhead beyond the
packed indices and shared LUT. Unpaired layers add one R_inv matrix per
unique (dimension, seed) pair — typically 32 KiB for head_dim=128.

---

## 6. Risks and open questions

### R1 — Rotation approximation through non-linearities

Fused rotation assumes `act(R · z) ≈ R · act(z)` for element-wise
activations. This holds well for Hadamard rotations (which are
near-isometries that distribute energy evenly) but is an approximation.
The quality of the approximation varies by activation function: ReLU is
exact for positive inputs, GELU and SiLU introduce larger errors.

**Mitigation:** The fusion pass should whitelist activation types that are
safe to fuse through. Start with `relu` and `leaky_relu` (piecewise linear,
best approximation). Gate `gelu`, `silu`, and `swish` behind an `--aggressive-fusion` flag
or quality threshold. Empirically validate quality on representative
transformer architectures (LLaMA-style, GPT-style). If quality degrades,
restrict fusion to truly linear pairs (no activation between) and leave
non-fusible layers using the explicit matmul fallback.

### R2 — Rotation matrix storage for unpaired layers

Unpaired layers store an R_inv matrix (N×N float16). For large dimensions
(N > 1024), this is > 2 MiB per unique dimension.

**Mitigation:** In practice, transformer models reuse a small number of
unique inner dimensions (e.g., head_dim=128, hidden_dim=4096). R_inv
matrices are shared across all layers with the same dimension and seed,
so only one copy per unique (dim, seed) pair is stored. For head_dim=128,
R_inv is 32 KiB — negligible. For hidden_dim=4096, R_inv is 32 MiB but
these layers are typically paired (fused), so the fallback rarely applies
to large dimensions.

### R3 — Non-power-of-two inner dimensions

The Hadamard transform requires power-of-two dimensions. Real models often
have dims like 768, 1024, 2048 (usually powers of two), but not always.

**Mitigation:** Pad to next power of two, apply rotation, then truncate back.
The truncated coordinates are discarded after rotation — this is
mathematically valid as long as we truncate consistently in both rotation and
un-rotation.

### R4 — Quality at very low bit-widths (2-bit)

PolarQuant's theoretical guarantees are strongest at moderate bit-widths
(3–4). At 2 bits, the quantization error may be significant for some
architectures.

**Mitigation:** Default to 4-bit. Make 2-bit opt-in with a warning.

### R5 — `statrs` dependency

**Decision: use `statrs`.** The crate is already a workspace dependency
(`statrs = "0.18"` in root `Cargo.toml`) used by `ironmill-bench`. Adding
`statrs = { workspace = true }` to `crates/mil-rs/Cargo.toml` introduces no
new external dependency to the lockfile and provides a well-tested Beta CDF
and inverse CDF implementation.

### R6 — CoreML constant-folding of fallback matmul

The fallback path emits `constexpr_lut_to_dense → matmul(W', R_inv)` and
assumes CoreML will constant-fold this chain at model load time (since both
inputs are constants). While this is the expected behavior for
`constexpr_*`-derived values, it has not been verified empirically for the
specific case of a matmul between two `constexpr` outputs.

**Mitigation:** Add a serialization + CoreML validation test (Task 5) that
loads a model with a fallback matmul layer and verifies the weight tensor is
fully materialized before first inference. If CoreML does not constant-fold,
the fallback is to pre-multiply in the pass and emit the unrotated weight
directly (losing no compression — the packed indices + LUT are still smaller
than the dense result, and the dense result is only needed at runtime).

---

## 7. Task summary

| # | Task | Files | Depends on |
|---|------|-------|------------|
| 1 | Rotation utilities | `passes/rotation.rs` | — |
| 2 | Beta-optimal quantizer | `passes/beta_quantizer.rs` | — |
| 3 | PolarQuant pass (per-tensor) | `passes/polar_quantize.rs` | 1, 2 |
| 4 | Pipeline + CLI integration | `pipeline.rs`, `main.rs`, `passes/mod.rs` | 3, 7 (stub) |
| 5 | Serialization verification | `ir_to_proto.rs` (verify only) | 3 |
| 6 | Unit + quality tests | `tests/optimization_passes.rs`, `tests/cross_feature.rs` | 3, 4, 7 |
| 7 | Inter-layer rotation fusion | `passes/polar_rotation_fusion.rs` | 3 |
| 8 | Benchmarks | `crates/ironmill-bench/` | 3, 7 |
