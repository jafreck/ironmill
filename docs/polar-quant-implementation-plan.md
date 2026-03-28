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
├── rotation.rs                  # NEW — Hadamard transform + rotation utilities
├── polar_quantize.rs            # NEW — PolarQuantPass implementation
├── beta_quantizer.rs            # NEW — precomputed Beta-optimal quantizer levels
└── tensor_utils.rs              # extend: add f16 decode helper if needed

crates/mil-rs/src/ir/
├── pipeline.rs                  # register "polar-quantization" pass
├── tensor.rs                    # no changes expected
└── types.rs                     # no changes expected

crates/mil-rs/src/convert/
└── ir_to_proto.rs               # verify constexpr_lut_to_dense handles our payloads

crates/ironmill-cli/src/
└── main.rs                      # add --quantize polar-4 / polar-3 / polar-2

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

**Dependencies:** `rand` crate (already a transitive dependency via `kmeans.rs`
which uses `rand::Rng`).

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
  beta function. Use the `statrs` crate or implement a standalone
  approximation (the latter avoids a new dependency).

**Dependencies:** Consider adding `statrs` crate for Beta distribution
functions, or implement a minimal incomplete beta function (~50 lines).

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
    8. Build LUT from the Beta-optimal levels (cast to f16 for storage)
    9. Mutate op:
       - op.op_type = "constexpr_lut_to_dense"
       - attrs["lut"] = Value::Tensor { data: lut_bytes, shape: [2^n_bits], dtype: Float16 }
       - attrs["indices"] = Value::Tensor { data: packed_indices, shape: original_shape, dtype: UInt8 }
       - attrs["shape"] = original shape
       - Store rotation metadata for un-rotation at inference
   10. Insert un-rotation op after dequantization (see below)
```

**Un-rotation strategy:**

The rotated, quantized weight needs to be un-rotated at inference time.
Two emission strategies, selectable via a config flag:

**Strategy A — Explicit matmul (simple, general):**
```
constexpr_lut_to_dense → dequantized W'
const(R_inv) → rotation matrix
matmul(dequantized_W', R_inv) → W_approx
```
- Replace downstream references to the original const with the matmul output.
- R_inv is stored as a float16 const (N×N matrix).
- **Cost**: one extra matmul per weight tensor at model load time (CoreML
  materializes constexpr ops at load, not per-inference).

**Strategy B — Fused rotation (optimized, limited scope):**
For consecutive linear layers `y = W₂ · W₁ · x`:
- Rotate W₁'s output dimension and W₂'s input dimension with the same R.
- The rotations cancel: `W₂·R^T · R·W₁ = W₂·W₁`.
- No extra matmul needed.
- Only applicable when adjacent ops are both linear/matmul.
- Implement as a follow-up optimization pass (`PolarRotationFusionPass`).

**Recommendation:** Implement Strategy A first. CoreML evaluates `constexpr_*`
ops at model load time, so the matmul overhead is a one-time cost, not
per-inference. Strategy B can be added later as a separate pass.

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
    self.passes.push(Box::new(PolarQuantPass::new(n_bits)));
    Ok(self)
}
```

#### 4b. CLI integration (`main.rs`)

Extend the `--quantize` flag to accept `polar-2`, `polar-3`, `polar-4`:

```rust
"polar-2" | "polar-3" | "polar-4" => {
    let bits: u8 = opts.quantize.strip_prefix("polar-").unwrap().parse().unwrap();
    pipeline = pipeline.with_polar_quant(bits)?;
}
```

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
| `polar_quant_converts_const_to_lut` | `const` → `constexpr_lut_to_dense` + rotation matmul chain |
| `polar_quant_preserves_output_shape` | Output tensor shape matches original weight shape |
| `polar_quant_round_trip_quality` | Dequantized weight MSE is within theoretical bound for the bit-width |
| `polar_quant_skips_small_tensors` | Tensors below `min_elements` are left as-is |
| `polar_quant_skips_non_float32` | Non-FP32 tensors are not touched |
| `polar_quant_handles_non_power_of_two` | Tensors with non-power-of-two inner dim are padded/handled correctly |
| `polar_quant_deterministic_with_seed` | Same seed produces identical quantized output |
| `polar_quant_different_seeds_differ` | Different seeds produce different (but equally valid) output |

#### Quality tests

| Test | What it verifies |
|------|-----------------|
| `polar_4bit_better_than_naive_4bit` | PolarQuant-4 MSE < naive uniform-4-bit MSE on random Gaussian weights |
| `polar_4bit_comparable_to_int8` | PolarQuant-4 MSE is within 2× of INT8 MSE (half the bits) |
| `polar_3bit_acceptable_quality` | PolarQuant-3 MSE stays below a defined threshold |

#### Pipeline composition tests (`cross_feature.rs`)

| Test | What it verifies |
|------|-----------------|
| `polar_quant_mutually_exclusive_with_int8` | Pipeline rejects combining both |
| `polar_quant_mutually_exclusive_with_fp16` | Pipeline rejects combining both |
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

## 5. CoreML compatibility

### Emission format

PolarQuant emits three ops per quantized weight:

```
┌─────────────────────────────────────┐
│ constexpr_lut_to_dense              │  ← quantized + LUT (existing CoreML op)
│   lut: [2^b] float16 levels        │
│   indices: packed b-bit indices     │
│   shape: original weight shape      │
│   output: W_rotated (float16)       │
├─────────────────────────────────────┤
│ const                               │  ← rotation matrix (existing CoreML op)
│   val: R_inv [N, N] float16        │
├─────────────────────────────────────┤
│ matmul                              │  ← un-rotation (existing CoreML op)
│   x: W_rotated                      │
│   y: R_inv                          │
│   output: W_approx (float16)        │
└─────────────────────────────────────┘
```

All three ops are standard MIL/CoreML operations. No custom ops needed.

### Memory at load time

CoreML materializes `constexpr_*` ops at model load. The rotation matmul also
runs at load time if its inputs are all constants. So at inference time, the
model sees a plain float16 weight tensor — zero runtime overhead.

### ANE compatibility

The materialized float16 weights are ANE-compatible. The matmul during load
runs on CPU/GPU (it's a one-time const-folding step). No ANE concerns.

### Storage savings

| Bit-width | Bytes per element | vs FP16 | vs INT8 |
|-----------|-------------------|---------|---------|
| 4-bit | 0.5 + LUT amortized | 4× smaller | 2× smaller |
| 3-bit | 0.375 + LUT amortized | 5.3× smaller | 2.7× smaller |
| 2-bit | 0.25 + LUT amortized | 8× smaller | 4× smaller |

The rotation matrix R adds N² × 2 bytes per unique inner dimension. For a
typical transformer with head_dim=128 used across dozens of layers, R is
shared (same seed → same matrix), so it's 128 × 128 × 2 = 32 KiB — negligible.

---

## 6. Risks and open questions

### R1 — Rotation matrix storage for large dimensions

For inner dimensions > 1024, R is > 2 MiB (stored as dense float16).
**Mitigation:** Store only the seed and regenerate R at model load time. This
requires the runtime to implement the Hadamard transform, which breaks the
"emit only standard CoreML ops" goal. Alternative: use a structured rotation
(block-diagonal Hadamard) to keep R small.

**Decision needed:** Set a dimension threshold above which we switch from
storing R explicitly to using a structured/block-diagonal approach.

### R2 — Interaction with CoreML's constexpr materialization

CoreML materializes `constexpr_*` ops at model load. If the matmul following
`constexpr_lut_to_dense` is also folded at load time (constant propagation),
the runtime cost is truly zero. Need to verify this behavior empirically.

**Mitigation:** If CoreML doesn't constant-fold through the matmul, add an
option to pre-materialize: run the dequant + matmul at conversion time and
store the result as a plain float16 const. This loses the storage benefit
but confirms correctness.

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

Adding a new crate dependency for the Beta distribution functions. The
alternative is implementing the incomplete beta function from scratch (~50–100
lines, well-documented algorithm).

**Decision needed:** New dependency vs. inline implementation. Inline is
preferred to keep the dependency tree small.

---

## 7. Task summary

| # | Task | Files | Depends on |
|---|------|-------|------------|
| 1 | Rotation utilities | `passes/rotation.rs` | — |
| 2 | Beta-optimal quantizer | `passes/beta_quantizer.rs` | — |
| 3 | PolarQuant pass | `passes/polar_quantize.rs` | 1, 2 |
| 4 | Pipeline + CLI integration | `pipeline.rs`, `main.rs`, `passes/mod.rs` | 3 |
| 5 | Serialization verification | `ir_to_proto.rs` (verify only) | 3 |
| 6 | Unit + quality tests | `tests/optimization_passes.rs`, `tests/cross_feature.rs` | 3, 4 |
| 7 | Inter-layer rotation fusion (follow-up) | `passes/polar_rotation_fusion.rs` | 3 |
