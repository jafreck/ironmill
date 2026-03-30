# TurboQuant on ANE-Direct — Implementation Plan

> **Status:** Implemented (see notes below)
>
> **Implementation notes:**
> - All components implemented in `turboquant.rs`, `turboquant_mil.rs`, CLI, and benchmarks.
> - GQA head expansion via `tile` is implemented. `tile`, `reshape`, and `slice_by_index`
>   are eval-verified as intermediate ops within the TurboQuant INT8 cache pipeline
>   (30/30 checks pass, max_err=0.014). They cannot compile as standalone ANE programs.
>
> **Prerequisites:**
> - [TurboQuant Research Analysis](../research/turboquant-analysis.md)
> - [TurboQuant ANE/Orion Feasibility](../research/turboquant-ane-orion.md)
> - [ANE Op Support Matrix](../research/ane-op-support-matrix.md) — empirically verified op set
>
> **Scope:** Full TurboQuant (PolarQuant + QJL + runtime KV cache compression)
> on the ANE-direct backend.

## Overview

TurboQuant compresses KV caches via rotation + optimal scalar quantization.
This plan implements it as a runtime feature of the `ironmill-ane` backend
using **INT8 cache storage** (2× bandwidth reduction), which is the maximum
compression empirically verified to work end-to-end on ANE.

> **Why INT8, not INT4?** The ANE compiler comprehensively rejects INT4/UINT4
> across all paths: inputs, casts, arithmetic, matmul, conv, dequantize.
> Verified by `ane_dtype_probe`. INT8 is the lowest bit-width the ANE accepts
> as a storage format. TurboQuant's rotation trick still provides better quality
> at INT8 than naive affine quantization (no per-block overhead, near-optimal
> distortion from Beta-optimal levels).
>
> **Bandwidth impact:** INT8 cache = 1 byte/element vs FP16 = 2 bytes/element.
> For a 7B model at seq=4096, KV cache drops from ~2GB to ~1GB. On M4 Pro
> (273 GB/s), attention KV read time drops from ~7ms to ~3.5ms per token.

### What exists today

| Component | Status | Location |
|---|---|---|
| Randomized Hadamard rotation | ✅ Implemented | `rotation.rs` |
| Beta-optimal quantization levels | ✅ Implemented | `beta_quantizer.rs` |
| Static PolarQuant weight pass | ✅ Implemented (1/2/4/6/8-bit) | `polar_quantize.rs` |
| Rotation fusion pass | ✅ Implemented | `polar_rotation_fusion.rs` |
| ANE-direct compile + eval (`AneModel::compile_and_load` / `predict`) | ✅ Working | `ironmill-ane/src/lib.rs` |
| Sub-program splitting | ✅ Working | `split.rs` |
| IOSurface tensor I/O | ✅ Working | `tensor.rs` |
| KV cache IR pass | ✅ Implemented | `kv_cache.rs` (`KvCachePass`, not the runtime `KvCacheManager` proposed below) |
| ANE op support probing | ✅ 69 ops verified | `ane_op_probe`, `ane_op_eval`, `ane_op_fuzz` |
| ANE dtype probing | ✅ INT8 verified, INT4 rejected | `ane_dtype_probe` |
| INT8 cache pipeline | ✅ 30/30 eval checks pass | `ane_op_eval` (includes `test_turboquant_int8_cache_pipeline`; run via `cargo run -p ironmill-ane --example ane_op_eval`) |

### What needs to be built

| Component | Phase | Effort |
|---|---|---|
| `AneTensor` partial byte write | 1 | Small |
| `TurboQuantConfig` + cache manager | 1 | Medium |
| Cache-write ANE sub-program | 2 | Medium |
| Cache-read + attention ANE sub-program | 2 | Medium |
| QJL correction sub-program (optional) | 3 | Small |
| Inference loop orchestrator | 2 | Large |
| CLI integration + benchmarks | 3 | Medium |

---

## Phase 1 — Runtime Infrastructure

### 1.1 Partial tensor writes

**Why:** Cache mutation via CPU interception requires writing new INT8 K/V tokens
into specific positions of a persistent IOSurface buffer. Today `AneTensor`
only supports full-buffer writes.

**File:** `crates/ironmill-ane/src/tensor.rs` (new methods on existing `AneTensor`)

```rust
/// Write raw bytes at a byte offset within the surface.
/// Used for INT8 cache writes where each element is 1 byte.
pub fn write_bytes_at(&mut self, byte_offset: usize, data: &[u8]) -> Result<()> {
    let end = byte_offset + data.len();
    if end > self.alloc_size {
        return Err(AneError::SurfaceError(format!(
            "write_bytes_at: offset {} + len {} exceeds alloc {}",
            byte_offset, data.len(), self.alloc_size
        )));
    }
    self.with_locked_base(0, |base| unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            (base as *mut u8).add(byte_offset),
            data.len(),
        );
    })
}

/// Read raw bytes from a specific offset within the surface.
pub fn read_bytes_at(&self, byte_offset: usize, len: usize) -> Result<Vec<u8>> {
    let end = byte_offset + len;
    if end > self.alloc_size {
        return Err(AneError::SurfaceError(format!(
            "read_bytes_at: offset {} + len {} exceeds alloc {}",
            byte_offset, len, self.alloc_size
        )));
    }
    // Lock surface, copy bytes, unlock
    // (implementation follows existing read_bytes pattern with offset)
}
```

Uses the existing `with_locked_base` IOSurface locking mechanism. Also add
f16-typed partial writes for QJL residual storage:

```rust
pub fn write_f16_at(&mut self, offset_elements: usize, data: &[f16]) -> Result<()>;
pub fn read_f16_at(&self, offset_elements: usize, len: usize) -> Result<Vec<f16>>;
```

**Verification:** Add eval test that writes partial data, reads back full tensor,
confirms only the target region changed.

### 1.2 TurboQuant configuration

**File:** `crates/ironmill-ane/src/turboquant.rs` (new)

```rust
pub struct TurboQuantConfig {
    /// Number of distinct quantization bits. Supported values: 1, 2, 4, 6, 8
    /// (corresponding to 2, 4, 16, 64, 256 LUT entries).
    /// Controls quality, not storage format — storage is always INT8.
    /// Beta-optimal levels are computed for 2^n_bits distinct values
    /// within the [-128, 127] INT8 range.
    /// Validated at construction time; unsupported values are rejected.
    pub n_bits: u8,
    /// Maximum sequence length for cache allocation.
    pub max_seq_len: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of KV heads (may differ from num_heads for GQA).
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Hadamard rotation seed (deterministic, shared with dequant).
    pub rotation_seed: u64,
    /// Enable QJL 1-bit bias correction (adds ~1 bit overhead).
    pub enable_qjl: bool,
}
```

### 1.3 KV cache manager

**File:** `crates/ironmill-ane/src/turboquant.rs`

Manages persistent IOSurface-backed cache tensors across inference steps.

```rust
pub struct KvCacheManager {
    config: TurboQuantConfig,
    /// Per-layer K caches: [num_kv_heads, max_seq_len, head_dim] as INT8.
    /// Stored as rotated, quantized INT8 values. Dequantization and
    /// un-rotation happen on ANE in the attention sub-program.
    k_caches: Vec<AneTensor>,
    /// Per-layer V caches (same format as K).
    v_caches: Vec<AneTensor>,
    /// Current sequence position (next write index).
    seq_pos: usize,
    /// Precomputed Beta-optimal quantization levels [2^n_bits].
    quant_levels: Vec<f32>,
    /// Precomputed quantization boundaries [2^n_bits - 1].
    quant_boundaries: Vec<f32>,
    /// Precomputed Hadamard rotation signs for the seed.
    rotation_signs: Vec<f32>,
    /// Optional: per-layer QJL residual sign caches (fp16 ±1).
    qjl_sign_caches: Option<Vec<AneTensor>>,
}

impl KvCacheManager {
    pub fn new(config: TurboQuantConfig) -> Result<Self>;

    /// Write new INT8 K/V tokens into the cache at the current position.
    /// Called between ANE sub-programs in the inference loop.
    ///
    /// 1. Read INT8 K_quant / V_quant from ANE output tensor
    /// 2. Write INT8 bytes into persistent cache at seq_pos via write_bytes_at
    /// 3. Optionally compute QJL residual signs on CPU
    /// 4. Advance seq_pos
    pub fn update_cache(
        &mut self,
        layer: usize,
        k_quantized: &[u8],
        v_quantized: &[u8],
    ) -> Result<()>;

    /// Get cache tensor refs for ANE sub-program input binding.
    pub fn cache_tensors(&self, layer: usize) -> (&AneTensor, &AneTensor);

    /// Current valid sequence length.
    pub fn seq_len(&self) -> usize;

    /// Reset cache (new conversation).
    pub fn reset(&mut self);
}
```

**Cache update strategy:** CPU interception at sub-program boundary. The CPU
work is minimal — just `write_bytes_at` to splice INT8 bytes into the IOSurface
at O(1) cost. All dequantization and un-rotation happens on ANE during the
subsequent cache-read sub-program.

---

## Phase 2 — ANE Sub-Programs

TurboQuant splits each attention layer into sub-programs connected by the Rust
inference loop. All ops below have been verified against the ANE compiler
(see [`ane-op-support-matrix.md`](../research/ane-op-support-matrix.md)).

### 2.1 Cache-write sub-program

Runs once per new token per layer. Rotates and quantizes K/V projections to INT8.

```
Inputs:  K_proj  [1, num_kv_heads, 1, head_dim]  (fp16)
         V_proj  [1, num_kv_heads, 1, head_dim]  (fp16)

Ops:                                                       Verified?
──────────────────────────────────────────────────────────────────────
  matmul(K_proj, R_const)       → K_rotated   (fp16)      ✅ eval
  mul(K_rotated, inv_scale)     → K_scaled    (fp16)      ✅ eval
  add(K_scaled, zero_point)     → K_shifted   (fp16)      ✅ eval
  round(K_shifted)              → K_rounded   (fp16)      ✅ eval
  clip(K_rounded, -128, 127)    → K_clamped   (fp16)      ✅ eval
  cast(K_clamped, int8)         → K_quant     (int8)      ✅ eval
  [same chain for V]

Outputs: K_quant, V_quant  [1, num_kv_heads, 1, head_dim]  (int8)
```

All 6 ops are eval-verified, including `cast fp16→int8` (verified in
`test_int8_round_trip`). The full quantization chain is verified by
`test_int8_quantize_dequantize`.

**MIL generation:** New function `emit_cache_write_mil(config) -> String` that
generates MIL text with the rotation matrix and quantization constants baked as
`const` ops.

### 2.2 CPU cache interception (between sub-programs)

After the cache-write sub-program returns, Rust code:

1. Reads `K_quant` / `V_quant` INT8 bytes from the output `AneTensor`
2. Writes INT8 bytes into persistent cache via `write_bytes_at` at `seq_pos × head_dim`
3. Optionally computes QJL residual signs on CPU: `sign(K_original - K_dequantized)`
4. Advances `seq_pos`

This is O(head_dim × num_kv_heads) bytes copied per token — microseconds.
No dequantization or un-rotation on CPU; those happen on ANE at read time.

### 2.3 Cache-read + attention sub-program

Runs once per new token per layer. Dequantizes cached INT8 K/V to fp16 and
computes attention. **This is where the 2× bandwidth win happens** — the ANE
reads 1 byte/element from the cache IOSurface instead of 2.

```
Inputs:  Q         [1, num_heads, 1, head_dim]         (fp16)
         K_cache   [1, num_kv_heads, max_seq, head_dim] (int8, persistent)
         V_cache   [1, num_kv_heads, max_seq, head_dim] (int8, persistent)

Ops:                                                        Verified?
───────────────────────────────────────────────────────────────────────
  slice_by_index(K_cache, 0..seq_len)  → K_int8 (int8)    ⚠️ compile
  cast(K_int8, fp16)                   → K_fp16            ✅ eval
  mul(K_fp16, scale)                   → K_scaled          ✅ eval
  sub(K_scaled, offset)                → K_dequant         ✅ eval
  matmul(K_dequant, R_inv)             → K_unrotated       ✅ eval
  [same dequant chain for V]
  [optional: GQA head expansion via tile]                  ⚠️ compile
  matmul(Q, K_unrotated^T)             → QK                ✅ eval
  mul(QK, scale_factor)                → QK_s              ✅ eval
  softmax(QK_s, axis=-1)              → attn_weights       ✅ eval
  matmul(attn_weights, V_unrotated)   → attn_out           ✅ eval

Outputs: attn_out  [1, num_heads, 1, head_dim]  (fp16)
```

**INT8 function inputs are supported** when the first op converts to fp16.
The dtype probe showed `tensor<int8, ...>` as a function input to `identity`
fails (because identity preserves dtype), but `cast(x=tensor<int8>, dtype=fp16)`
succeeds — the MIL parser accepts INT8 function parameters when they flow into
a cast/dequantize op. This means the cache IOSurface can be bound directly as
an INT8-typed input to the attention sub-program.

> **Composite verification:** `test_turboquant_int8_cache_pipeline` verifies
> the full path: RMSNorm → INT8 quantize → `cast int8→fp16` → dequant → dot
> product in a single ANE sub-program. Observed max_err=0.014 (the test
> asserts correctness within a tolerance of 1.5; 0.014 is the empirically
> measured worst case).

### 2.4 Inference loop orchestrator

**File:** `crates/ironmill-ane/src/turboquant.rs`

```rust
pub struct TurboQuantModel {
    config: TurboQuantConfig,
    cache: KvCacheManager,
    /// Per-layer compiled sub-programs (shared across layers with same arch).
    cache_write_program: LoadedProgram,
    attention_program: LoadedProgram,
    qjl_program: Option<LoadedProgram>,
    /// FFN programs (one per unique layer architecture).
    ffn_programs: Vec<LoadedProgram>,
    runtime: AneRuntime,
}

impl TurboQuantModel {
    /// Compile all sub-programs from a model's IR.
    pub fn compile(
        program: &Program,
        config: TurboQuantConfig,
        ane_config: AneConfig,
    ) -> Result<Self>;

    /// Run one token through the full model.
    pub fn step(&mut self, token_embedding: &AneTensor) -> Result<AneTensor> {
        for layer in 0..self.config.num_layers {
            // 1. Compute Q/K/V projections (ANE sub-program)
            // 2. Run cache-write sub-program → INT8 quantized K/V
            // 3. CPU: write INT8 bytes to persistent cache IOSurface
            // 4. Run attention sub-program: read INT8 cache → dequant → un-rotate → attention
            // 5. Optional: QJL correction sub-program
            // 6. Run FFN sub-program
        }
        // Final: output projection
    }
}
```

**Program compilation budget:** ANE caps at ~119 compilations per process.
TurboQuant needs ~4 unique sub-programs (cache-write, attention, FFN, optional QJL).
Since all layers share the same architecture, programs are compiled once and
reused across layers with different weight tensors passed via IOSurface inputs.
Total: ~4–6 programs, well within budget.

---

## Phase 3 — QJL Correction + Polish

### 3.1 QJL correction sub-program (optional)

Adds 1 bit per element for unbiased attention score estimation.

```
Inputs:  Q_signs        [1, num_heads, 1, head_dim]         (fp16, ±1)
         residual_signs  [1, num_kv_heads, seq_len, head_dim] (fp16, ±1)

Ops:                                                          Verified?
─────────────────────────────────────────────────────────────────────────
  greater(Q, zero_const)         → Q_pos     (bool)          ✅ eval
  select(Q_pos, +1, -1)          → Q_sign    (fp16)          ✅ eval
  matmul(Q_sign, residual^T)     → correction (fp16)         ✅ eval
  mul(correction, scale_const)   → scaled     (fp16)         ✅ eval

Outputs: scaled  (fp16, added to attention logits before softmax)
```

All ops are eval-verified. The composite pattern is verified by
`test_qjl_sign_extraction`.

**Residual sign storage:** During cache-write CPU interception, compute
`sign(K_original - K_dequantized)` and store as ±1 fp16 in a separate
IOSurface cache. Cost: 2 bytes per element (could be packed to 1 bit with
custom packing, but fp16 ±1 is simpler and the ANE reads fp16 natively).

### 3.2 CLI integration

New flags to be added to the `Compile` subcommand:

```sh
# Compile model with TurboQuant INT8 KV cache (4-bit precision, INT8 storage)
ironmill compile model.onnx \
  --backend ane-direct \
  --kv-quant turbo-int8 \
  --max-seq-len 4096

# With QJL correction (better quality, +1 bit overhead)
ironmill compile model.onnx \
  --backend ane-direct \
  --kv-quant turbo-int8 \
  --kv-quant-qjl \
  --max-seq-len 4096
```

### 3.3 Benchmarks

Extend `ironmill-bench` to compare:

| Configuration | Bytes/elem | Memory (7B, seq=4096) | Expected speedup |
|---|---|---|---|
| FP16 baseline (CoreML) | 2.0 | ~2.0 GB | 1× |
| FP16 baseline (ANE-direct) | 2.0 | ~2.0 GB | 1× |
| TurboQuant INT8 | 1.0 | ~1.0 GB | ~2× |
| TurboQuant INT8 + QJL | 1.0 + 0.125 | ~1.1 GB | ~2× |

---

## Reuse from existing codebase

| Existing code | Reuse in TurboQuant |
|---|---|
| `rotation.rs::rotate_rows_hadamard` | Generate rotation matrix const for cache-write sub-program |
| `rotation.rs::unrotate_rows_hadamard` | CPU-side un-rotation during cache interception |
| `beta_quantizer.rs::beta_optimal_levels` | Precompute quantization LUT for given dim and n_bits |
| `beta_quantizer.rs::beta_optimal_boundaries` | CPU-side quantize during cache interception |
| `beta_quantizer.rs::quantize_to_index` | CPU-side index lookup |
| `AneCompiler::compile_mil_text` | Compile sub-programs from generated MIL text |
| `AneRuntime::eval` | Execute sub-programs |
| `AneTensor` | IOSurface-backed tensor I/O |
| `split.rs` sub-program patterns | Model for building standalone sub-programs |
| `kv_cache.rs` cache detection | Identify cache inputs in source IR |
| `ir_to_mil_text.rs` emitter | Reference for MIL text generation |

---

## Risks and mitigations

### R1 — Private API stability
ANE-direct uses undocumented Apple APIs that may change between macOS versions.

**Mitigation:** Feature-gated behind `ane-direct`. The `ane_op_probe` / `ane_op_eval`
/ `ane_op_fuzz` / `ane_dtype_probe` examples serve as a regression suite — re-run
after macOS updates to detect changes. All 30 eval tests must pass before shipping.

### R2 — ANE program compilation limit (~119)
Exceeding the limit crashes the process.

**Mitigation:** TurboQuant needs ~4–6 unique programs total (shared across layers).
Well within budget. Monitor via `AneCompiler::compile_count()` (in `mil-rs::ffi::ane`).

### R3 — INT4 not available; limited to 2× bandwidth reduction
The ANE compiler comprehensively rejects INT4/UINT4 via the MIL text path.
INT8 gives 2× bandwidth reduction, not the 4× that TurboQuant achieves on
GPU with custom kernels.

**Mitigation:** 2× is still significant — on M4 Pro it cuts KV read time from
~7ms to ~3.5ms at seq=4096 for a 7B model. TurboQuant's rotation trick at
INT8 also gives better quality than naive INT8 affine quantization. If a
sub-MIL INT4 path (like maderix's weight descriptor approach) becomes viable,
the architecture supports it — only the cache manager and IOSurface dtype
would change.

### R4 — CPU interception latency
Cache update runs on CPU between ANE sub-programs.

**Mitigation:** With INT8 cache, CPU interception per token per layer is minimal — just a byte copy:
- 128 × 8 = 1024 bytes for GQA models (head_dim × num_kv_heads, e.g. Llama-3 7B with 8 KV heads); 4096 bytes for MHA (32 KV heads)
- `write_bytes_at`: ~1 μs per IOSurface write
- No dequantization or un-rotation on CPU (that's ANE's job at read time)
- Total: < 5 μs per layer, ~160 μs for 32 layers (GQA)
- vs. attention compute: ~1–20 ms per layer

Negligible compared to attention latency.

### R5 — Quantization correctness
TurboQuant's theory guarantees near-optimal distortion, but practical impact
depends on the model.

**Mitigation:** Validate on standard benchmarks (perplexity, LongBench).
The `ane_op_eval` suite already verifies the arithmetic pipeline produces
correct results (RMSNorm observed max_err=0.002 with test tolerance 0.1,
QJL sign extraction exact, affine quantize exact). End-to-end quality
validation is part of Phase 3.

---

## References

1. [ANE Op Support Matrix](../research/ane-op-support-matrix.md) — 69 verified ops with eval status
2. [TurboQuant Research Analysis](../research/turboquant-analysis.md) — paper summary and ironmill relevance
3. [TurboQuant ANE/Orion Feasibility](../research/turboquant-ane-orion.md) — op decomposition and program structure
4. Static PolarQuant pass — implemented in `crates/mil-rs/src/ir/passes/polar_quantize.rs`
5. TurboQuant paper: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
6. PolarQuant paper: [arXiv:2502.02617](https://arxiv.org/abs/2502.02617)
7. maderix/ANE: [github.com/maderix/ANE](https://github.com/maderix/ANE) — independent ANE reverse engineering
