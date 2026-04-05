# Calibrated Weight Quantization — Implementation Plan

Implementation plan for the research doc at `docs/research/calibrated-weight-quantization.md`.
Adds activation-aware and rotation-optimized weight quantization to ironmill,
making it the first fully Python-free LLM quantization-to-deployment pipeline
in Rust.

## Current State

### What exists

| Component | Location | Capabilities |
|-----------|----------|-------------|
| `Int8QuantizePass` | `mil-rs/src/ir/passes/int8_quantize.rs` | UINT8 MinMax affine quant, per-tensor and per-channel via `Granularity` enum. `calibration_dir: Option<PathBuf>` field exists but is unused by `run()`. Emits `constexpr_affine_dequantize` ops with `quantized_data` (raw bytes), `scale`, `zero_point`, `axis`. |
| `PolarQuantPass` | `mil-rs/src/ir/passes/polar_quantize.rs` | Seeded randomized Hadamard rotation + symmetric absmax scalar quantization. Supports 2/4/6/8-bit via `n_bits: u8`. Emits `constexpr_lut_to_dense` with LUT, packed indices, row norms, and `polar_quant_seed`. |
| `rotation.rs` | `mil-rs/src/ir/passes/rotation.rs` | `rotate_rows_hadamard` (in-place Walsh–Hadamard butterfly with seeded random signs), `pad_to_power_of_two`. Note: the Hadamard transform is self-inverse, so `unrotate_rows_hadamard` applies the same transform. Also: `ironmill-inference/src/turboquant/rotation.rs` has runtime-side rotation/sign/QJL generation. |
| `PalettizePass` | `mil-rs/src/ir/passes/palettize.rs` | K-means LUT quantization (1/2/4/6/8-bit). `GroupedPalettizePass` variant clusters per output-channel group. Emits `constexpr_lut_to_dense`. |
| `QuantizationInfo` | `mil-rs/src/weights.rs:74-102` | `None`, `LutToDense { lut, lut_dtype, indices, original_shape, n_bits, row_norms, norms_dtype, polar_quant_seed }`, `AffineDequantize { scale, zero_point, scale_dtype, zero_point_dtype, axis }` — no bit_width or group_size fields. Scale/zero_point are raw `Vec<u8>` bytes with separate dtype fields. |
| GPTQ/AWQ import | `mil-rs/src/convert/onnx_graph.rs:1162-1342` | `detect_prequantized_weights()` detects `*.qweight`/`*.scales`/`*.qzeros`/`*.g_idx` patterns → `PreQuantFormat::{QLoRA, Gptq, Awq}`. Import only via `prequantized_to_op()`, no quantization. |
| Metal inference | `ironmill-inference/src/metal/inference.rs` | `prefill()` → `run_pipeline()` forward pass. **Single command buffer** for entire forward pass, committed once after all layers + final norm + LM head. MPS matmul for dense, custom shaders for RMSNorm/RoPE/SiLU/attention. No activation hooks. No mid-pipeline GPU→CPU readback. |
| TurboQuant | `ironmill-inference/src/metal/turboquant/` | KV-cache INT8/INT4 compression. Fully GPU-resident — no mid-pipeline CPU readback. Owns `qjl_matrix`, `rotation_signs`; `MetalKvCache` has `k_qjl_signs`, `k_r_norms`. Optional outlier-channel split with separate QJL matrices. |
| `PerplexityDataset` | `ironmill-bench/src/perplexity.rs:10-26` | `{ name, model, vocab_size, seq_len, num_sequences, eos_token_id, sequences: Vec<Vec<u32>> }`. Loads pre-tokenized JSON via serde. Default fixture: `tests/fixtures/quality/wikitext2-qwen3.json`. |
| `ScalarType` | `mil-rs/src/ir/tensor.rs` | F16/F32/F64, I8/I16/I32/I64, U8/U16/U32/U64, Bool. No sub-8-bit types. `byte_size()` returns 1/2/4/8 only. Palettization/PolarQuant use packed `UInt8` storage. |
| `PassPipeline` | `mil-rs/src/ir/pipeline.rs` | Sequential `Vec<Box<dyn Pass>>` runner. Builders: `with_fp16()`, `with_int8()`, `with_palettize()`, `with_polar_quant()`. TOML config via `pass_from_name()`. Currently **mutually exclusive** quantization — no composing rotation + quantization passes. |

### Key gaps

- No INT4 weight encoding or per-group quantization
- No calibration pipeline (cannot run inference on calibration data)
- No activation statistics collection or Hessian computation
- `constexpr_affine_dequantize` has no bit_width or group_size metadata
- `QuantizationInfo::AffineDequantize` has no bit_width or group_size
- No INT4 dequantization Metal shader
- No `ScalarType` for sub-8-bit values (INT4 uses packed UInt8 storage)
- No linear algebra dependency for Cholesky (needed by GPTQ)
- `PassPipeline` cannot compose rotation + quantization passes (blocks Phases 4–6)
- No small test model fixture for quantization integration tests

## Risks and Mitigations

### Risk 1: Metal command buffer structure blocks calibration hooks (HIGH)

**Problem:** `run_pipeline()` encodes ALL layers into a single Metal command
buffer, committed once at the end. Mid-pipeline `MTLBuffer.contents()` reads
on uncommitted buffers are undefined behavior.

**Mitigation:** Task 1.0 (spike) validates the approach early. The calibration
path needs a separate execution mode that commits per-layer:
- End current command buffer at each layer boundary
- `commit()` + `waitUntilCompleted()`
- Read intermediate buffers to CPU
- Create new command buffer for next layer

This is slow (~10-50× vs single-CB) but acceptable for one-time calibration.
The normal inference path stays unchanged.

### Risk 2: No linear algebra crate for GPTQ Hessian (MEDIUM)

**Problem:** GPTQ requires Cholesky decomposition on ~4096×4096 matrices.
No linear algebra dependency exists in the workspace.

**Mitigation:** Use `faer` (pure Rust, no LAPACK/C dependency, excellent
performance). Add as optional dependency: `faer = { version = "0.20", optional = true }`.
Gate GPTQ pass behind `#[cfg(feature = "gptq")]`.

### Risk 3: E8 lattice codebook underspecified (MEDIUM)

**Problem:** At 4-bit, the E8 codebook has ~65K entries. Nearest-neighbor
search design, hierarchical encoding for 3-bit, and Metal shader integration
need design work beyond what the research doc covers.

**Mitigation:** Task 5.0 adds a research spike gate before committing to
full implementation. Prototype the codebook + nearest-neighbor search,
benchmark feasibility, decide on 2-bit-only vs 2-4-bit scope.

### Risk 4: PassPipeline composability (MEDIUM)

**Problem:** SpinQuant needs to compose rotation + quantization (e.g., rotate
then AWQ/GPTQ). The current pipeline treats quantization passes as mutually
exclusive.

**Mitigation:** Task 4.0 extends `PassPipeline` to support pass composition
before SpinQuant implementation. SpinQuant inserts rotation as a pre-pass,
followed by any quantization pass. This is a small change — the pipeline
already supports arbitrary `add_pass()` ordering; the limitation is only in
the builder convenience methods.

---

## Implementation Phases

### Phase 0: INT4 MinMax Weight Quantization

Generalize the existing INT8 path to support configurable bit width and
per-group quantization. Delivers immediate value (Burn parity) and builds
shared infrastructure for all subsequent phases.

**Prerequisites:** Task 0.0 (test fixture).

#### Task 0.0 — Small test model fixture

**File:** New `tests/fixtures/tiny-llama/` (or similar)

Create a minimal test model (e.g., 2-layer, 128-dim LLaMA-like architecture)
in safetensors format that can be used across all quantization tests. This
avoids depending on large model downloads for CI and integration tests.

- Generate deterministic weights (seeded RNG) so tests are reproducible.
- Include a matching pre-tokenized calibration file (128 sequences × 64 tokens).
- Small enough to run in CI (<10MB total).

**Acceptance criteria:**
- Model loads through existing safetensors → MIL pipeline.
- Model runs inference through Metal engine (produces logits).
- Calibration data loads through `PerplexityDataset` format.

#### Task 0.1 — Add per-group quantization math

**File:** `crates/mil-rs/src/ir/passes/int8_quantize.rs`

Extend `Int8QuantizePass` → `AffineQuantizePass` (or add alongside it):

- Add `bits: u8` field (default 8, support 4/8) and `group_size: Option<usize>`
  field (None = per-tensor/per-channel, Some(128) = per-group).
- Implement per-group quantization: partition weight rows into groups of
  `group_size`, compute independent `scale`/`zero_point` per group.
- INT4 math: `scale = (max - min) / 15.0`, `zero_point = round(-min / scale)`,
  `q = clamp(round(x / scale) + zero_point, 0, 15)`.
- Pack two INT4 values per byte (low nibble first, matching GPTQ/AWQ community
  standard: 8 INT4 values per `uint32`, little-endian).

**Acceptance criteria:**
- Unit tests: quantize a known tensor at 4-bit per-group-128, verify packed
  output, verify round-trip dequantize error is within expected bounds.
- Existing INT8 per-tensor and per-channel tests still pass (backward compat).

#### Task 0.2 — Extend `constexpr_affine_dequantize` metadata

**Files:** `crates/mil-rs/src/ir/passes/int8_quantize.rs`,
`crates/mil-rs/src/convert/ir_to_proto.rs`, `crates/mil-rs/src/convert/proto_to_ir.rs`,
`crates/mil-rs/src/weights.rs`

- Add `bit_width` (u8) and `group_size` (optional i32) attributes to the
  `constexpr_affine_dequantize` op emission.
- Update `QuantizationInfo::AffineDequantize` to include `bit_width: u8` and
  `group_size: Option<usize>`. Note: existing `scale`/`zero_point` are raw
  `Vec<u8>` bytes with separate `scale_dtype`/`zero_point_dtype` fields —
  new fields follow this pattern.
- Update proto serialization/deserialization to persist these fields.
- Ensure readers handle missing fields gracefully (default `bit_width=8`,
  `group_size=None` for legacy models).

**Acceptance criteria:**
- Round-trip: write an INT4 per-group quantized model to proto, read it back,
  verify metadata is preserved.
- Legacy INT8 models load without error (backward compat).

#### Task 0.3 — INT4 packing utilities

**File:** New `crates/mil-rs/src/ir/passes/int4_pack.rs` (or extend `tensor_utils.rs`)

- `pack_int4(values: &[u8]) -> Vec<u8>` — pack pairs of 4-bit values into bytes.
- `unpack_int4(packed: &[u8], count: usize) -> Vec<u8>` — reverse.
- `pack_int4_u32(values: &[u8]) -> Vec<u32>` — 8 values per u32 (GPTQ format).
- `unpack_int4_u32(packed: &[u32], count: usize) -> Vec<u8>`.

**Acceptance criteria:**
- Round-trip tests for all pack/unpack variants with edge cases (odd counts,
  boundary values 0 and 15).

#### Task 0.4 — Pipeline integration

**File:** `crates/mil-rs/src/ir/pipeline.rs`

- Add `with_int4(self, group_size: usize) -> Result<Self>` builder method.
- Register the new pass in `pass_from_name()` for TOML config: `"int4-quantize"`.
- Default group_size: 128.

**Acceptance criteria:**
- `PassPipeline::new().with_int4(128)` builds and runs on a small test program.
- TOML config `quantize = "int4"` works through `from_config_str()`.

#### Task 0.5 — INT4 Metal dequantization shader

**File:** New `crates/ironmill-inference/src/metal/shaders/int4_dequant.metal`
(or extend existing shader infrastructure)

- Unpack 2×INT4 from each byte.
- Apply `(int4_val - zero_point) * scale` per group.
- Support fused dequant+matmul path for inference.

**Also:** Update `MetalInference::run_pipeline()` to dispatch through INT4
dequant path when weight metadata indicates `bit_width=4`.

**Acceptance criteria:**
- Metal shader compiles.
- Inference on a small INT4-quantized model produces correct output
  (verified against CPU dequant reference).

#### Task 0.6 — Perplexity evaluation baseline

Run the existing `PerplexityDataset` benchmark on the test fixture model at
FP16, then at INT4 MinMax, to establish baseline perplexity numbers. This
becomes the reference for all subsequent quantization methods.

**Acceptance criteria:**
- FP16 perplexity recorded.
- INT4 MinMax perplexity recorded, gap is ≤3pp (expected range for MinMax).

---

### Phase 1: Calibration Infrastructure

Add a calibration runner that executes the model on sample inputs and captures
intermediate activations. This is the foundation for AWQ, GPTQ, SpinQuant,
QuIP#, and D2Quant.

**⚠️ Task 1.0 is the highest-risk item in the entire plan.** Prototype early.

#### Task 1.0 — Metal calibration dispatch spike (RISK MITIGATION)

**File:** `crates/ironmill-inference/src/metal/inference.rs`

**Problem:** `run_pipeline()` uses a single Metal command buffer for the
entire forward pass, committed once after all layers + LM head (lines
~489–1407). Mid-pipeline `MTLBuffer.contents()` reads on uncommitted buffers
are undefined behavior. There are no existing mid-pipeline GPU→CPU readbacks
anywhere in the inference code.

**Spike goal:** Validate that per-layer command buffer commits work correctly
for calibration:

1. Create `run_pipeline_calibration()` — a variant that commits the command
   buffer after each transformer layer, reads the linear input buffers back
   to CPU, then creates a new command buffer for the next layer.
2. Verify: logits from calibration mode match normal `run_pipeline()` output
   exactly (bit-for-bit or within FP16 rounding).
3. Measure overhead: calibration mode will be ~10-50× slower than single-CB
   mode. Quantify the actual cost on a real model.

**If the spike fails** (e.g., per-layer commits cause incorrect results due to
buffer aliasing, or overhead is >100×), fallback options are:
- Use MTLSharedEvent to signal between encoding and CPU readback
- Insert blit commands to copy intermediate buffers before commit
- Use `MTLCaptureManager` for programmatic GPU capture

**Acceptance criteria:**
- Calibration dispatch produces identical logits to normal dispatch.
- Per-layer readback returns valid FP16 activation data.
- Overhead measured and documented (target: <50× slowdown).

#### Task 1.1 — Activation hook trait and split stores

**File:** New `crates/ironmill-inference/src/calibration/mod.rs`
(new module under `ironmill-inference`)

```rust
pub trait ActivationHook {
    fn on_linear_input(&mut self, layer: usize, name: &str, activation: &[f16]);
}

/// Lightweight store for AWQ: only per-channel magnitude statistics.
pub struct AwqActivationStore {
    /// layer_name -> per-channel mean absolute activation magnitudes
    pub magnitudes: HashMap<String, ChannelMagnitudes>,
}

pub struct ChannelMagnitudes {
    pub mean_abs: Vec<f32>,      // running mean |x| per channel
    pub sample_count: usize,
}

/// Heavier store for GPTQ: accumulates X^T X Hessian per layer.
pub struct GptqActivationStore {
    /// layer_name -> accumulated Hessian (X^T X) in row-major f32
    pub hessians: HashMap<String, HessianAccumulator>,
}

pub struct HessianAccumulator {
    pub xtx: Vec<f32>,          // n_features × n_features, row-major
    pub n_features: usize,
    pub sample_count: usize,
}
```

Split into two stores because:
- AWQ needs only per-channel magnitudes: O(n_features) per layer, ~32KB for 4096-dim.
- GPTQ needs full `X^T X`: O(n_features²) per layer, ~64MB for 4096-dim.
- Running both simultaneously would double the memory cost unnecessarily.
- Both implement `ActivationHook` and accumulate streaming statistics.

**Acceptance criteria:**
- Unit test: feed synthetic activations to both store types, verify statistics.
- `AwqActivationStore`: mean magnitudes match hand-computed values.
- `GptqActivationStore`: accumulated `X^T X` matches `2 * X^T * X` for known X.

#### Task 1.2 — Metal inference activation hooks

**File:** `crates/ironmill-inference/src/metal/inference.rs`

Depends on Task 1.0 (spike) succeeding.

- Add `run_pipeline_with_hooks(&mut self, token_ids: &[u32], hooks: &mut dyn ActivationHook)`
  method that uses the per-layer-commit calibration dispatch from Task 1.0.
- At each linear projection boundary, read the input buffer back to CPU and
  call `hooks.on_linear_input(layer, name, &activation_data)`.
- `prefill_with_hooks()` wraps this, chunking like normal `prefill()`.
- Normal inference path (`run_pipeline()`) is completely unchanged.

**Acceptance criteria:**
- `prefill_with_hooks` with a no-op hook produces same logits as `prefill`.
- With `AwqActivationStore` hook: activations captured for every linear layer.
- With `GptqActivationStore` hook: Hessians accumulated correctly.

#### Task 1.3 — Calibration dataset loader

**File:** New `crates/ironmill-inference/src/calibration/dataset.rs`

- Reuse the `PerplexityDataset` JSON format from `ironmill-bench`.
- `CalibrationDataset::load(path: &Path) -> Result<Self>` — loads pre-tokenized
  JSON (`Vec<Vec<u32>>`).
- `CalibrationDataset::random(vocab_size: u32, seq_len: usize, n_sequences: usize, seed: u64)` —
  deterministic random token fallback for CI.
- `CalibrationDataset::iter_batches(batch_size: usize) -> impl Iterator` —
  yields batches of token sequences.

**Acceptance criteria:**
- Loads existing `wikitext2-qwen3.json` fixture.
- Random dataset generates valid, deterministic token IDs.

#### Task 1.4 — Calibration runner (end-to-end)

**File:** New `crates/ironmill-inference/src/calibration/runner.rs`

- `CalibrationRunner` orchestrates: load model → load calibration data →
  run prefill with hooks → return populated activation store.
- Generic over `ActivationHook` — works with both `AwqActivationStore` and
  `GptqActivationStore`.
- Streams batches to limit peak memory.
- Progress reporting (batch count, layer count).

**Acceptance criteria:**
- Integration test with test fixture model (Task 0.0): run calibration, get
  non-empty activation store with stats for all linear layers.

---

### Phase 2: AWQ Pass

First calibration-aware quantization method. Simpler than GPTQ (no matrix
inversion), faster, comparable quality.

#### Task 2.1 — AWQ core algorithm

**File:** New `crates/mil-rs/src/ir/passes/awq_quantize.rs`

```rust
pub struct AwqQuantizePass {
    pub bits: u8,                       // 4 or 8
    pub group_size: usize,              // typically 128
    pub activation_stats: AwqActivationStore,
    pub grid_search_steps: usize,       // default 20
    pub salient_percentile: f32,        // default 0.99
}
```

Algorithm per linear op:
1. Retrieve per-channel activation magnitudes from `AwqActivationStore`.
2. Identify salient channels (top 1% by activation magnitude).
3. Grid search for optimal per-channel scaling factors to minimize output MSE.
   Search range: `[1.0, max_scale]` in `grid_search_steps` steps per channel.
4. Scale weights by inverse of scaling factors.
5. Quantize scaled weights to INT4 per-group (reuse Phase 0 infrastructure).
6. Rewrite op: replace const with `constexpr_affine_dequantize` carrying
   bit_width=4, group_size, and the scaling factors as metadata.

**Acceptance criteria:**
- Unit test with synthetic weights + activations: AWQ-quantized weights have
  lower reconstruction MSE than naive MinMax on the same weights.
- Grid search converges (MSE decreases monotonically).

#### Task 2.2 — AWQ scaling factor persistence

The per-channel scaling factors computed by AWQ must be applied at runtime
(activations are scaled up to compensate for weight scaling down).

Options (decide during implementation):
- **Fuse into adjacent LayerNorm weights** (preferred — zero runtime cost).
  The scale factors can be absorbed into the preceding LayerNorm's weight
  parameter since LayerNorm computes `gamma * normalized + beta`.
- **Store as metadata** and apply in the dequant shader (fallback if fusion
  is not always possible, e.g., no preceding LayerNorm).
- **Emit explicit `mul` ops** in the MIL graph (most general, small runtime cost).

**Acceptance criteria:**
- Quantized model produces correct output (within tolerance) on Metal inference.
- Perplexity gap vs FP16 is ≤1pp on test fixture (AWQ target).

#### Task 2.3 — Pipeline integration and CLI

**Files:** `crates/mil-rs/src/ir/pipeline.rs`, `crates/ironmill-cli/`

- Add `with_awq(self, activation_stats: AwqActivationStore, group_size: usize)`
  builder method to `PassPipeline`.
- CLI flag: `--quantize awq` (triggers calibration run + AWQ pass).
- `--calibration-data <path>` is required; omitting it is a compile error.
- Register `"awq-quantize"` in `pass_from_name()` for TOML config.

**Acceptance criteria:**
- End-to-end: `ironmill compile model --quantize awq --calibration-data data.json`
  produces a quantized model that runs on Metal.
- Perplexity evaluation passes (Task 0.6 framework).

---

### Phase 3: GPTQ Pass

The industry-standard Hessian-guided quantization method.

#### Task 3.0 — Add `faer` dependency

**File:** `crates/mil-rs/Cargo.toml`

Add `faer` as an optional dependency gated behind a `gptq` feature flag:

```toml
[features]
gptq = ["dep:faer"]

[dependencies]
faer = { version = "0.20", optional = true }
```

`faer` is pure Rust (no C/LAPACK dependency), provides optimized Cholesky
decomposition, and handles the 4096×4096 scale needed for GPTQ Hessians.

**Acceptance criteria:**
- `cargo check -p mil-rs --features gptq` succeeds.
- Simple Cholesky test passes using `faer`.

#### Task 3.1 — Hessian computation

**File:** New `crates/mil-rs/src/ir/passes/gptq/hessian.rs`

- `accumulate_hessian(xtx: &mut [f32], batch: &[f32], n_features: usize)` —
  incremental `X^T X` accumulation (called by `GptqActivationStore`).
- `finalize_hessian(xtx: &mut [f32], sample_count: usize, dampening: f64)` —
  scale by `2/n`, add dampening: `H += dampening * mean(diag(H)) * I`.
- `cholesky_decompose(h: &[f32], n: usize) -> Result<Vec<f32>>` — via `faer`.
- `cholesky_inverse_row(l: &[f32], n: usize, row: usize) -> Vec<f32>` —
  compute single row of H⁻¹ via forward/back substitution (GPTQ only needs
  one row at a time during the quantization loop).

**Acceptance criteria:**
- Unit test: known SPD matrix → Cholesky matches expected L.
- Dampened Hessian is always positive definite (Cholesky succeeds).
- Inverse row matches `numpy.linalg.inv` reference for small matrices.

#### Task 3.2 — GPTQ quantization loop

**File:** New `crates/mil-rs/src/ir/passes/gptq/mod.rs`

```rust
#[cfg(feature = "gptq")]
pub struct GptqQuantizePass {
    pub bits: u8,
    pub group_size: usize,
    pub block_size: usize,     // columns processed together (128)
    pub dampening: f64,        // Hessian diagonal dampening (0.01)
    pub activation_stats: GptqActivationStore,
}
```

Algorithm per linear op:
1. Retrieve accumulated `X^T X` from `GptqActivationStore`, finalize Hessian.
2. Cholesky decompose the dampened Hessian via `faer`.
3. For each column block (size `block_size`):
   - Quantize weights in the block to INT4.
   - Compute quantization error: `err = w_float - dequant(w_quant)`.
   - Compensate remaining columns: `W[:, remaining] -= err * H_inv_row / H_inv_diag`.
4. Rewrite op with quantized weights + per-group scales + zeros.

**Acceptance criteria:**
- GPTQ-quantized weights have lower reconstruction error than both MinMax and
  AWQ on the same test case.
- Output matches reference Python AutoGPTQ implementation within tolerance on
  a small weight matrix (provide reference values in test).

#### Task 3.3 — Pipeline integration

- Add `with_gptq(...)` to `PassPipeline` (gated behind `#[cfg(feature = "gptq")]`).
- CLI flag: `--quantize gptq`. `--calibration-data` is required; omitting it
  is a compile error.
- TOML config: `quantize = "gptq"`.
- Perplexity evaluation gate.

---

### Phase 4: SpinQuant Pass

Upgrades PolarQuant's random Hadamard rotations to learned rotations optimized
on calibration data.

#### Task 4.0 — PassPipeline composability for rotation + quantization

**File:** `crates/mil-rs/src/ir/pipeline.rs`

The current `PassPipeline` builder methods (`with_int8`, `with_palettize`,
`with_polar_quant`) each insert a single quantization pass. SpinQuant needs
to compose: rotation pass → quantization pass (AWQ or GPTQ or INT4).

- Add `with_spinquant(self, ...) -> Result<Self>` that inserts the rotation
  optimization pass followed by a configurable quantization pass.
- Alternatively, add a general `with_rotation_then_quantize(rotation, quantize)`
  builder that composes any rotation + quantization pass pair.
- Ensure the pipeline ordering is: standard passes → rotation → quantization
  → post-quant passes (type repropagation, layout opt).

**Acceptance criteria:**
- Can build a pipeline with SpinQuant rotation + AWQ quantization.
- Can build a pipeline with SpinQuant rotation + GPTQ quantization.
- Pass ordering is validated.

#### Task 4.1 — Cayley parameterization for orthogonal optimization

**File:** New `crates/mil-rs/src/ir/passes/spinquant/cayley.rs`

- Represent rotation as `R = (I - A)(I + A)^{-1}` where A is skew-symmetric.
  Uses `faer` for the matrix inversion (reuse Phase 3 dependency).
- `CayleyOptimizer` struct with configurable strategy:
  - CMA-ES (evolutionary, best for high-dim) — recommended default
  - Random search with shrinking radius (simplest fallback)
- Loss evaluation: forward-pass MSE through Metal engine (receives a callback).
- Convergence criterion: loss improvement < threshold for N consecutive steps.

**Acceptance criteria:**
- Generated R is orthogonal: `‖R * R^T - I‖_F < 1e-5`.
- Optimization reduces forward-pass loss over iterations on test fixture.

#### Task 4.2 — SpinQuant pass

**File:** New `crates/mil-rs/src/ir/passes/spinquant/mod.rs`

```rust
pub struct SpinQuantPass {
    pub bits: u8,
    pub group_size: usize,
    pub rotation_epochs: usize,
    pub quantize_method: QuantizeMethod, // MinMax, AWQ, or GPTQ
    pub activation_stats: AwqActivationStore, // or GptqActivationStore
}

pub enum QuantizeMethod { MinMax, Awq, Gptq }
```

Algorithm:
1. Initialize rotations at residual connections, Q/K/V projections using
   Hadamard matrices (reuse `rotation.rs` — specifically `rotate_rows_hadamard`
   which is self-inverse, so the same function handles both directions).
2. Optimize rotations via Cayley parameterization on calibration loss.
3. Absorb final learned rotations into adjacent weight matrices
   (zero runtime cost — rotations become part of the weights).
4. Quantize rotation-absorbed weights using the selected `QuantizeMethod`.

**Key reuse:** `rotation.rs` for Hadamard init and rotation application,
`PolarQuantPass` for the rewrite scaffolding. Note: PolarQuant's rotation is
the same self-inverse Walsh–Hadamard butterfly; SpinQuant replaces the fixed
Hadamard with a learned matrix but reuses the insertion points and absorption
logic.

**Acceptance criteria:**
- Learned rotations produce lower quantization error than random Hadamard
  (PolarQuant baseline) on test fixture.
- Perplexity gap ≤3pp at W4 (SpinQuant target from research doc).

---

### Phase 5: QuIP# Pass

Hadamard rotation + E8 lattice vector quantization for state-of-the-art
quality at ≤4 bits.

#### Task 5.0 — E8 lattice research spike (RISK MITIGATION)

**Goal:** Validate feasibility before committing to full implementation.

Prototype and answer:
1. **Codebook generation:** Generate the E8 lattice points. At 2-bit (256
   entries for 8-dim vectors), this is straightforward. At 4-bit (~65K entries),
   is brute-force nearest-neighbor search fast enough, or do we need a KD-tree
   or hashing scheme?
2. **Metal shader:** Can the codebook lookup be implemented efficiently in a
   Metal compute shader? What's the memory footprint for the codebook texture?
3. **Integration with `constexpr_lut_to_dense`:** Does the existing op handle
   8-element vector groups, or does it need extension?
4. **Benchmark:** Compare E8 2-bit vs PolarQuant 2-bit on reconstruction error.

**Gate:** If 4-bit E8 nearest-neighbor search exceeds 10ms per weight matrix,
scope QuIP# to 2-bit only and use scalar INT4 for 4-bit.

**Acceptance criteria:**
- Working E8 codebook generation + nearest-neighbor search prototype.
- Benchmark results documented.
- Go/no-go decision for full QuIP# scope.

#### Task 5.1 — E8 lattice codebook

**File:** New `crates/mil-rs/src/ir/passes/quip_sharp/e8_lattice.rs`

- Precompute E8 lattice codebook (fixed, mathematically defined — not learned).
  The E8 lattice is the set of points in ℝ⁸ where all coordinates are integers
  or all are half-integers, and their sum is even.
- `nearest_e8(vector: &[f32; 8]) -> (u16, [f32; 8])` — find nearest lattice
  point, return index and quantized vector.
- 2-bit: 256 entries (8 values × 2 bits = 16 bits per vector).
  Use the D8 half-lattice (a subset of E8) for efficient enumeration.
- 4-bit: standard E8 with per-vector scaling. Scope depends on Task 5.0 spike.
- Codebook stored as a compile-time constant array.

**Acceptance criteria:**
- Nearest-neighbor search returns correct lattice points for known test vectors.
- Codebook is deterministic (same input → same output).
- Search performance: <1ms for a 4096×4096 weight matrix at 2-bit.

#### Task 5.2 — QuIP# pass

**File:** New `crates/mil-rs/src/ir/passes/quip_sharp/mod.rs`

Algorithm:
1. Apply randomized Hadamard rotation (reuse `rotation.rs`).
2. For each weight matrix, process in groups of 8: find nearest E8 point,
   encode as codebook index + per-vector scale.
3. Optional LDLQ refinement (reuse GPTQ's `faer`-based Cholesky from Phase 3).
4. Emit as `constexpr_lut_to_dense` (reuse palettization op with E8 codebook
   instead of k-means centroids).

**Acceptance criteria:**
- QuIP# at 2-bit achieves better reconstruction error than naive 2-bit scalar
  quantization and PolarQuant 2-bit.
- Perplexity gap ≤5pp at 2-bit (QuIP# target from research doc).

---

### Phase 6: D2Quant Pass

Sub-4-bit quantization (2-bit, 3-bit) with dual-scale handling and LayerNorm
correction.

#### Task 6.1 — Dual-scale quantizer

**File:** New `crates/mil-rs/src/ir/passes/d2quant/mod.rs`

```rust
pub struct D2QuantPass {
    pub bits: u8,                // 2 or 3
    pub group_size: usize,       // typically 128
    pub outlier_threshold: f32,  // percentile for outlier detection (default 0.99)
    pub calibration: AwqActivationStore,
}
```

- Partition weights per group into normal vs outlier (by magnitude percentile).
- Compute separate scale/zero for each partition.
- Store both partitions with a bitmask indicating which weights are outliers.
- New MIL op: `constexpr_dual_scale_dequantize` with attributes:
  `quantized_data`, `normal_scale`, `normal_zero`, `outlier_scale`,
  `outlier_zero`, `outlier_mask`, `group_size`, `bit_width`.
- Corresponding Metal shader for dual-scale dequantization.

**Acceptance criteria:**
- Dual-scale 2-bit achieves lower reconstruction error than single-scale 2-bit.
- Round-trip through proto serialization preserves all metadata.

#### Task 6.2 — Deviation-Aware Correction (DAC)

**File:** New `crates/mil-rs/src/ir/passes/d2quant/dac.rs`

- Post-quantization MIL pass (separate from the quantization pass).
- For each LayerNorm following a quantized linear:
  1. Run calibration data through the quantized model (reuse calibration runner).
  2. Measure activation distribution shift vs FP16 reference: compute
     per-channel mean and variance delta.
  3. Adjust LayerNorm `weight` (gamma) and `bias` (beta) to compensate:
     `gamma_new = gamma * (sigma_fp16 / sigma_quant)`,
     `beta_new = beta + gamma * (mu_fp16 - mu_quant)`.
- Requires two calibration runs: one on FP16 model, one on quantized model.

**Acceptance criteria:**
- DAC reduces perplexity gap by ≥0.5pp on quantized test fixture.

#### Task 6.3 — Pipeline integration

- Add `with_d2quant(...)` to `PassPipeline`.
- CLI flag: `--quantize d2quant --bits 2`.
- TOML config: `quantize = "d2quant"`.
- Perplexity evaluation gate.

---

## Dependency Graph

```
Phase 0 (INT4 MinMax)
  ├─── Task 0.0 (test fixture)         ← prerequisite for all integration tests
  ├─── Task 0.1 (per-group quant math)
  ├─── Task 0.2 (metadata extension)   ← depends on 0.1
  ├─── Task 0.3 (INT4 packing)
  ├─── Task 0.4 (pipeline integration) ← depends on 0.1, 0.2
  ├─── Task 0.5 (Metal shader)         ← depends on 0.2, 0.3
  └─── Task 0.6 (perplexity baseline)  ← depends on 0.0, 0.5

Phase 1 (Calibration)                  ← independent of Phase 0 (except 0.0)
  ├─── Task 1.0 (Metal CB spike) ⚠️    ← HIGHEST RISK — do first
  ├─── Task 1.1 (hook trait + stores)
  ├─── Task 1.2 (Metal hooks)          ← depends on 1.0, 1.1
  ├─── Task 1.3 (dataset loader)
  └─── Task 1.4 (runner)               ← depends on 1.1, 1.2, 1.3

Phase 2 (AWQ)                          ← depends on Phase 0 + Phase 1
  ├─── Task 2.1 (AWQ algorithm)        ← depends on 0.1, 0.3, 1.1
  ├─── Task 2.2 (scaling persistence)  ← depends on 2.1
  └─── Task 2.3 (pipeline + CLI)       ← depends on 2.1, 2.2

Phase 3 (GPTQ)                         ← depends on Phase 0 + Phase 1
  ├─── Task 3.0 (add faer dep)
  ├─── Task 3.1 (Hessian)              ← depends on 1.1, 3.0
  ├─── Task 3.2 (GPTQ loop)            ← depends on 0.1, 0.3, 3.1
  └─── Task 3.3 (pipeline)             ← depends on 3.2

Phase 4 (SpinQuant)                    ← depends on Phase 0 + Phase 1 + Phase 2 or 3
  ├─── Task 4.0 (pipeline composability) ← depends on Phase 2 or 3 existing
  ├─── Task 4.1 (Cayley optimizer)      ← depends on 3.0 (faer)
  └─── Task 4.2 (SpinQuant pass)        ← depends on 4.0, 4.1

Phase 5 (QuIP#)                        ← depends on Phase 0 + Phase 1
  ├─── Task 5.0 (E8 research spike) ⚠️  ← GATE: go/no-go for full scope
  ├─── Task 5.1 (E8 codebook)          ← depends on 5.0
  └─── Task 5.2 (QuIP# pass)           ← depends on 5.1, Phase 0, Phase 1

Phase 6 (D2Quant)                      ← depends on Phase 0 + Phase 1
  ├─── Task 6.1 (dual-scale quant)
  ├─── Task 6.2 (DAC)                  ← depends on 6.1, Phase 1
  └─── Task 6.3 (pipeline)             ← depends on 6.1, 6.2
```

**Parallelism opportunities:**
- Phase 0 and Phase 1 are independent (except shared test fixture 0.0) and
  should be developed concurrently. **Start Task 1.0 (spike) immediately.**
- Phase 2 and Phase 3 can proceed in parallel once prerequisites are done.
- Task 5.0 (E8 spike) can start anytime — it's pure research with no deps.
- Task 4.1 (Cayley) can start once `faer` is added (Task 3.0), independent
  of the rest of Phase 3.

## Delivery Milestones

| Milestone | Phases | Outcome | Quality Gate |
|-----------|--------|---------|-------------|
| **M1** | 0 | INT4 MinMax per-group quantization — Burn parity | Perplexity gap ≤3pp vs FP16 |
| **M2** | 0 + 1 + 2 | AWQ — first calibration-aware method in Rust | Perplexity gap ≤1pp vs FP16 |
| **M3** | 3 | GPTQ — industry-standard Hessian method | Perplexity gap ≤0.5pp vs FP16; matches AutoGPTQ ±0.5pp |
| **M4** | 4 | SpinQuant — evolved PolarQuant with learned rotations | Perplexity gap ≤3pp at W4 |
| **M5** | 5 | QuIP# — state-of-the-art ≤4-bit with lattice codebooks | 2-bit perplexity gap ≤5pp |
| **M6** | 6 | D2Quant — sub-4-bit frontier (2-bit, 3-bit) | 2-bit better than MinMax 4-bit |

Each milestone includes a perplexity evaluation on the test fixture model and,
where feasible, on Llama-3-8B.

## Design Decisions

### 1. Calibration engine: Metal inference with per-layer commit mode

The Metal inference engine already runs LLM forward passes. The calibration
path adds a per-layer command buffer commit mode to `run_pipeline()` that
enables GPU→CPU readback of intermediate activations.

**Key constraint discovered during review:** The normal inference path uses a
single command buffer for the entire forward pass. The calibration path must
commit per-layer and create new command buffers, which is ~10-50× slower but
acceptable for one-time calibration. The normal path is unchanged.

### 2. INT4 packing: GPTQ/AWQ community standard

8 INT4 values per `uint32`, little-endian. Per-group FP16 scales. Compatible
with HuggingFace safetensors INT4 layout for interop.

### 3. Activation statistics: streaming, split by use case

Two separate activation store types:
- `AwqActivationStore`: O(n_features) per layer — lightweight, fast.
- `GptqActivationStore`: O(n_features²) per layer — heavier, needed for Hessian.

Both accumulate streaming statistics (not full tensors), keeping memory bounded
regardless of calibration dataset size. The stores are separate because running
AWQ doesn't need the Hessian overhead, and GPTQ doesn't need per-channel
magnitudes.

### 4. SpinQuant optimization: derivative-free (no autograd)

Rotation matrices are small (~4096×4096, ~6 per model). CMA-ES or random
search over Cayley parameters works by evaluating forward-pass loss through
the Metal engine. Avoids autograd framework dependency. Uses `faer` for the
matrix inversion in the Cayley map (shared with GPTQ).

### 5. New passes live in `mil-rs` (not `ironmill-compile`)

Following the existing pattern: `Int8QuantizePass`, `PalettizePass`, and
`PolarQuantPass` all live in `mil-rs/src/ir/passes/`. The new passes
(`AwqQuantizePass`, `GptqQuantizePass`, `SpinQuantPass`, `QuipSharpPass`,
`D2QuantPass`) follow the same `trait Pass` pattern.

### 6. Calibration infrastructure lives in `ironmill-inference`

The activation hooks require the Metal inference engine. The calibration
runner, dataset loader, and activation stores live in `ironmill-inference`
under a new `calibration` module. The MIL passes in `mil-rs` receive the
collected activation stores as input.

### 7. Linear algebra: `faer` (pure Rust)

GPTQ Cholesky and SpinQuant Cayley inversion use `faer` — a pure Rust linear
algebra library with no C/LAPACK/BLAS dependency. Gated behind
`features = ["gptq"]` on `mil-rs`. Handles 4096×4096 matrices efficiently.

### 8. QuIP# gated behind research spike

The E8 lattice codebook at 4-bit has ~65K entries with unclear nearest-neighbor
search performance. Task 5.0 is a mandatory research spike that produces a
go/no-go decision before full implementation. If 4-bit is infeasible, scope
to 2-bit only.

## Crate Dependency Changes

| Crate | New Dependency | Feature Gate | Purpose |
|-------|---------------|-------------|---------|
| `mil-rs` | `faer = "0.20"` | `gptq` | Cholesky decomposition (GPTQ), Cayley inversion (SpinQuant) |
| `ironmill-inference` | (none) | `calibration` | Gates calibration-mode code paths |

No other new external dependencies. The `tokenizers` crate (HuggingFace, pure
Rust) is a future optional dependency for raw-text calibration but is not
needed for the initial implementation which uses pre-tokenized JSON.

## Success Criteria

1. Quantize Llama-3-8B FP16 → INT4 AWQ in pure Rust
2. Output matches HuggingFace AWQ reference within ±0.5 perplexity
3. Quantized model runs on Metal GPU via existing inference path
4. Zero Python dependencies in the entire pipeline
5. Quantization completes in <30 min on M-series Mac
6. SpinQuant achieves ≤3pp perplexity gap at W4 on LLaMA-3 8B
7. QuIP# achieves usable 2-bit quantization (≤5pp perplexity gap)

## References

See `docs/research/calibrated-weight-quantization.md` for full paper
references, algorithm details, and landscape analysis.
