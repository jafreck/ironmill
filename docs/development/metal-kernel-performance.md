# Metal Kernel Performance: Closing the llama.cpp Gap

## Problem

Qwen 3.5-4B INT4+TQ-INT8 on M2 Max 64 GB:

| Metric | ironmill | llama.cpp | Gap |
|--------|----------|-----------|-----|
| Decode throughput | 20 tok/s | 90 tok/s | 4.5× |
| Prefill throughput | 220 tok/s | 1100 tok/s | 5× |
| Theoretical bandwidth floor | 6.6 ms/tok | 6.6 ms/tok | — |
| Measured decode latency | 48 ms/tok | 11 ms/tok | 4.5× |

Both systems read the same 2.63 GB of INT4 weights per token. The theoretical
bandwidth floor on M2 Max (400 GB/s) is ~6.6 ms. The entire performance gap is
dispatch overhead, kernel inefficiency, and barrier stalls.

## Root Cause Analysis

Two structural problems affect **all** weight formats simultaneously.

### Problem 1: Decode matvec uses scalar dequant, no simdgroup MMA

All quantized decode kernels use the same inefficient pattern:
- **32 threads per threadgroup** (one SIMD group)
- **One output row per threadgroup** — wastes GPU parallelism
- Each lane independently reads packed weights, dequants through float,
  and accumulates in scalar registers
- **No `simdgroup_multiply_accumulate`** hardware utilization

The Dense FP16 `matvec` kernel already uses the correct pattern (256 threads,
8 simdgroups, 64 output rows/TG, simdgroup MMA). An AMX-capable INT4 kernel
(`affine_matvec_int4_amx`) exists in the codebase and proves this approach works
for quantized weights, but it is **not wired into any dispatch path**.

The AMX pattern works by cooperatively dequantizing a weight tile into FP16 in
threadgroup memory, then running `simdgroup_multiply_accumulate` on those FP16
tiles. This intermediate FP16 step is required because Apple's simdgroup matrix
hardware only accepts `half` inputs — there is no integer-input MMA on Apple
Silicon.

llama.cpp uses the same approach: cooperative dequant to shared memory →
simdgroup matrix multiply, processing 64+ output rows per threadgroup.

**Estimated cost: ~15–20 ms of the 48 ms/tok decode latency.**

### Problem 2: Prefill matmul uses K-tile of 8

All prefill matmul kernels use `MATMUL_K_TILE = 8`. For Qwen 3.5's K=2560:
- **320 main-loop iterations**, each requiring a `threadgroup_barrier`
- Each iteration performs a single 8×8 simdgroup MMA — terrible arithmetic
  intensity
- Shared memory tiles are 1 KB (64×8 FP16) — severely underutilized

Increasing K-tile from 8 to 32 reduces loop iterations from 320 to 80, performs
4 MMA operations per barrier (instead of 1), and grows shared memory to a modest
4 KB per buffer — well within Metal's 32 KB threadgroup memory limit.

llama.cpp uses K-tile of 32–64.

**Estimated cost: ~3–4× of the prefill latency.**

### Secondary: Memory barrier overhead

Per-token barrier count in the hot path:

| File | Barriers | Notes |
|------|----------|-------|
| `pipeline.rs` | 21 | Resource-specific `memory_barrier_with_resources()` |
| `ffn.rs` | 11 | Resource-specific `memory_barrier_with_resources()` |
| `gdn.rs` | 7 | Resource-specific `memory_barrier_with_resources()` |
| `ple.rs` | 15 | **All use full-scope `memory_barrier_buffers()`** |
| **Total** | **54** | |

PLE's 15 full-scope barriers flush ALL GPU buffer caches instead of only the
buffer being written. The other files already use resource-specific barriers,
but some may be redundant (placed between independent dispatches). llama.cpp
uses ~30 resource-specific barriers for ~250 dispatches.

**Estimated cost: ~8–12 ms.**

---

## Current Kernel Inventory

### Weight Projection Kernels

Every linear projection dispatches through `encode_projection_q8()` in
`projection.rs`, which selects a kernel based on the `WeightBuffer` enum
variant. The formats relevant to LLM inference and their current kernels:

| Format | Decode kernel (matvec) | TG | MMA? | Prefill kernel (matmul) | TG | K-tile |
|--------|----------------------|-----|------|------------------------|----|--------|
| Dense FP16 | `matvec` | 256 | ✅ | `matmul` | 256 | 8 |
| Affine INT4 | `affine_matvec_int4` | 32 | ❌ | `affine_matmul_int4` | 256 | 8 |
| Affine INT8 | `affine_matvec_int8` | 32 | ❌ | `affine_matmul_int8` | 256 | 8 |
| D2Quant 3-bit | `d2quant_matvec_3bit` | 32 | ❌ | `d2quant_matmul_3bit` | 256 | 8 |

Affine INT4 covers both AWQ and GPTQ models (same runtime format after
quantization). Affine INT8 is used for sensitive-layer mixed precision.
D2Quant 3-bit is the dual-scale quantization format.

> **Dead code:** Two additional kernels have PSOs loaded in `ops.rs` but are
> never dispatched: `affine_matvec_int4_rowmajor` (row-major INT4 variant)
> and `affine_matvec_int4_amx` (the AMX prototype discussed below). Both
> should be wired in or removed as part of this work.

> **Note:** The codebase also contains `WeightBuffer::Quantized` (a LUT-based
> palettization format from the CoreML pipeline, labeled "PolarQuant" in code).
> This is **not** related to the PolarQuant KV-cache quantization research
> (arXiv 2502.02617). It is a legacy CoreML format with no backing LLM
> quantization research and no active model configs use it for LLM inference.
> It is excluded from this performance plan. A separate cleanup task should
> evaluate removing it from the inference path entirely.

### Decode-Only Fused/Batched Specializations

Exist only for **Affine INT4** and **Dense FP16**.

| Kernel | Format | Purpose | TG | MMA? |
|--------|--------|---------|-----|------|
| `gdn_batched_matvec` | Dense FP16 | 4-way GDN projections | 256 | ✅ |
| `gdn_batched_affine_matvec_int4` | Affine INT4 | 4-way GDN projections | 32 | ❌ |
| `batched_affine_matvec_int4` | Affine INT4 | FFN gate+up | 32 | ❌ |
| `fused_ffn_gate_up_act_int4` | Affine INT4 | FFN gate+up+SiLU | 32 | ❌ |
| `fused_residual_norm_matvec` | Dense FP16 | End-of-layer P1 fusion | 256 | ✅ |
| `fused_residual_norm_affine_matvec_int4` | Affine INT4 | End-of-layer P1 fusion | 32 | ❌ |
| `affine_matvec_int4xq8` | Affine INT4 | Q8-input integer dot | 32 | ❌ |
| `affine_matvec_int4_amx` | Affine INT4 | **UNUSED** AMX decode | 256 | ✅ |
| `affine_matvec_int4_rowmajor` | Affine INT4 | **UNUSED** row-major decode | 32 | ❌ |

### Non-Projection Kernels (Out of Scope)

- `d2quant_embedding_lookup_3bit` — active, dispatched from PLE for D2Quant
  embedding gather. Not a weight projection; no changes needed.

### Systems Not Affected

- **TurboQuant**: KV cache compression only (`turboquant_cache_write`,
  `turboquant_attention`). Orthogonal to weight projections. No changes needed.
- **Attention**: `fused_sdpa`, `prefill_attention_v2`, `flash_decode` split/reduce.
  Already FlashAttention-style tiled. Not a bottleneck.
- **Elementwise**: `rms_norm`, `residual_add`, `silu_gate`, etc. Not on the
  critical path.

---

## Architecture: Shared Kernel Pattern + Format-Specific Dequant

### Key Insight

The matmul/matvec algorithm is identical across all weight formats. Only the
**weight tile load and dequantization** step differs. Each format decodes packed
weights into FP16 in threadgroup memory so that `simdgroup_multiply_accumulate`
can operate on them. The MMA accumulation, double-buffering, and output store
are the same across every format.

### Shared Kernel Pattern

Each shader file is independently compiled by `build.rs` via `xcrun metal`
into its own `.metallib`. The shared structure is expressed as a consistent
kernel pattern that each format implements, with identical constants:

```metal
// ── Shared constants (replicated in each .metal file) ──
constant constexpr uint MATMUL_K_TILE  = 32;       // up from 8
constant constexpr uint TM_TILE        = 64;
constant constexpr uint TN_TILE        = 64;
constant constexpr uint TN_STRIDE      = TN_TILE + 1;  // bank-conflict-free
constant constexpr uint N_SIMDGROUPS   = 8;
constant constexpr uint TG_SIZE        = N_SIMDGROUPS * 32;  // 256
constant constexpr uint TN_BLOCKS      = TN_TILE / 8;
constant constexpr uint K_BLOCKS       = MATMUL_K_TILE / 8;  // 4 MMA ops per K-tile
```

The build system (`build.rs`) already prepends `#define` headers for head-dim
dependent shaders. If constant synchronization across files becomes a concern,
the same pattern can prepend shared constants via `build.rs` string
concatenation — no header file needed.

### Kernel Structure (Prefill Matmul)

Every prefill matmul kernel follows this template. Only the B-tile load
function differs:

```metal
kernel void affine_matmul_int4(/* format-specific buffers */) {
    // 1. Setup: accumulators, threadgroup memory, coordinates
    simdgroup_matrix<float,8,8> acc[TN_BLOCKS];
    threadgroup half tg_a[2][TM_TILE * MATMUL_K_TILE];    // 8 KB
    threadgroup half tg_bt[2][MATMUL_K_TILE * TN_STRIDE];  // 8 KB

    // 2. Main double-buffered loop (SHARED structure)
    for (uint t = 0; t < num_k_steps; t++) {
        // Load next A tile (SHARED — identical for all formats)
        load_a_tile(A, tg_a[nxt], tg_m, k_base, M, K, tid);

        // Load next B tile (FORMAT-SPECIFIC — only this differs)
        dequant_b_tile_affine_int4(B, scales, zeros, tg_bt[nxt], ...);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MMA accumulate (SHARED — identical for all formats)
        for (uint kb = 0; kb < K_BLOCKS; kb++) {
            simdgroup_load(a_mat, tg_a[cur] + sgid*8*MATMUL_K_TILE + kb*8, ...);
            for (uint j = 0; j < TN_BLOCKS; j++) {
                simdgroup_load(bt_mat, tg_bt[cur] + kb*8*TN_STRIDE + j*8, ...);
                simdgroup_multiply_accumulate(acc[j], a_mat, bt_mat, acc[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 3. Output store (SHARED — identical for all formats)
    store_results(acc, C, tg_m, tg_n, M, N, tid, sgid);
}
```

### Kernel Structure (Decode AMX Matvec)

Same separation. The decode matvec uses cooperative dequant into threadgroup
FP16, then `simdgroup_multiply_accumulate` — matching the existing
`affine_matvec_int4_amx` pattern (256 threads, 64 output rows per TG):

```metal
kernel void affine_matvec_int4(/* format-specific buffers */) {
    simdgroup_matrix<float,8,8> acc(0);
    threadgroup half tg_w[MV_ROWS_PER_TG * MV_TILE_K];  // weight tile
    threadgroup half tg_x[MV_TILE_K];                     // input tile

    for (uint kt = 0; kt < K; kt += MV_TILE_K) {
        // Cooperative input load (SHARED)
        load_x_tile(A, tg_x, kt, K, tid);

        // Cooperative weight dequant (FORMAT-SPECIFIC)
        dequant_w_tile_affine_int4(B, scales, zeros, tg_w, ...);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Simdgroup MMA (SHARED)
        for (uint kb = 0; kb < tile_k/8; kb++) {
            simdgroup_load(w_T, tg_w + sg_row*MV_TILE_K + kb*8, ...);
            simdgroup_load(x_mat, tg_x + kb*8, 0);
            simdgroup_multiply_accumulate(acc, x_mat, w_T, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Extract and store (SHARED)
}
```

### Format-Specific Dequant Functions

Each format provides an inline B-tile load function. This is the only code
that varies between formats:

| Format | Dequant function | How it differs |
|--------|-----------------|----------------|
| Affine INT4 | `dequant_b_tile_affine_int4()` | Nibble unpack, per-group `(q-zero)*scale`, blocked layout |
| Affine INT8 | `dequant_b_tile_affine_int8()` | Byte unpack, per-group `(q-zero)*scale`, blocked layout |
| D2Quant 3-bit | `dequant_b_tile_d2quant_3bit()` | 3-bit unpack, outlier mask, dual scale+zero, row-major |

Affine INT4 and INT8 share the same dequant formula (`(q - zero) * scale`),
differing only in unpacking (nibble vs byte) and blocked-layout stride. These
can be two variants of the same function or two ~20-line inline functions.

### Can Affine INT4 and D2Quant Share a Kernel?

**No.** The formats differ at every level:

| | Affine INT4 | D2Quant 3-bit |
|---|---|---|
| Formula | `(nibble - zero) * scale` | `(q - zero) * scale` with outlier select |
| Params | Per-group `scale` + `zero` | Dual `normal`/`outlier` scale+zero + mask |
| Packing | 4-bit nibbles (2 per byte) | 3-bit (8 vals → 3 bytes) |
| Layout | Blocked `[N/64, K/8, 64, 4]` | Row-major `[N, ceil(K/8)*3]` |

They share the MMA framework but cannot share dequant code.

---

## Implementation Plan

### Phase 1: Prefill K-tile Fix

**Scope:** All weight formats. Prefill path only.

1. Increase `MATMUL_K_TILE` from 8 to 32 in all prefill matmul kernels:
   - `matvec.metal` → `matmul` (FP16)
   - `affine_matmul.metal` → `affine_matmul_int4`, `affine_matmul_int8`
   - `d2quant_matmul.metal` → `d2quant_matmul_3bit`
2. Restructure each kernel's inner loop to follow the shared pattern
   (double-buffered, K_BLOCKS MMA ops per iteration)
3. Shared memory per kernel: `tg_a[2][64×32]` + `tg_bt[2][32×65]` = 16 KB
   total (within Metal's 32 KB threadgroup memory limit)
4. No Rust dispatch changes needed — same kernel names, same buffer bindings,
   same threadgroup counts. Only the inner loop changes.

**Expected impact:** ~3–4× prefill speedup (220 → ~700–900 tok/s).

### Phase 2: AMX Decode Matvec

**Scope:** All quantized formats. Decode path only.

1. Wire the existing `affine_matvec_int4_amx` kernel into the Affine INT4
   decode dispatch path in `projection.rs`
2. Create AMX-style matvec for remaining formats with format-specific
   cooperative dequant:
   - Affine INT8: new `affine_matvec_int8_amx`
   - D2Quant 3-bit: new `d2quant_matvec_3bit_amx`
3. Update Rust dispatch in `projection.rs`:
   - `encode_affine_projection` → 256 threads/TG, `ceil(N/64)` threadgroups
   - `encode_d2quant_projection` → same
4. Rebuild decode batched/fused INT4 kernels on AMX framework:
   - `batched_affine_matvec_int4` (FFN gate+up)
   - `gdn_batched_affine_matvec_int4` (4-way GDN)
   - `fused_ffn_gate_up_act_int4`
   - `fused_residual_norm_affine_matvec_int4`

**Expected impact:** ~2× decode speedup (20 → ~40–50 tok/s).

### Phase 3: Barrier Cleanup

**Scope:** All formats. Shared infrastructure.

1. Convert all 15 `memory_barrier_buffers()` calls in `ple.rs` to
   resource-specific `memory_barrier_with_resources()`
2. Audit `pipeline.rs` (21 barriers), `ffn.rs` (11), `gdn.rs` (7) — these
   already use resource-specific `memory_barrier_with_resources()`, so the
   audit is for **removing redundant barriers** between independent dispatches,
   not converting their barrier type
3. Remove or narrow identified redundant barriers

**Expected impact:** ~5–10 ms saved per token.

### Phase 4: Q8 Integer Dot Product

**Scope:** Affine INT4 only. Decode path.

1. Rebuild `affine_matvec_int4xq8` on the Phase 2 AMX framework with true
   integer accumulation (`int8 × int4 → int32`, float scale per-group only)
2. Specific to affine quantization where the scale structure allows factoring
   out the float multiply from the inner loop

**Expected impact:** Additional ~2–3 ms decode for affine INT4.

---

## File Organization After Refactoring

### Metal Shaders

```
shaders/
├── matvec.metal                 ← FP16 matvec/matmul (K-tile=32 prefill)
├── affine_matmul.metal          ← Affine INT4/INT8 (AMX decode + K-tile=32 prefill)
├── d2quant_matmul.metal         ← D2Quant 3-bit (AMX decode + K-tile=32 prefill)
├── fused_residual_norm.metal    ← Update INT4 fused variant to AMX
├── gdn_recurrent.metal          ← GDN recurrent (fused decode kernels, no changes)
├── attention.metal              ← No changes
├── flash_decode.metal           ← No changes
├── fused_sdpa.metal             ← No changes
├── turboquant.metal             ← No changes (KV cache, orthogonal)
├── quantized_matmul.metal       ← Legacy LUT palettization (no changes, out of scope)
├── ...                          ← Other unchanged shaders
```

Each `.metal` file is independently compiled to `.metallib` by `build.rs`. The
shared kernel pattern (constants, MMA loop structure, output store) is replicated
in each file — the actual shared code is ~20 lines of constants and the MMA loop
structure. If synchronization becomes a maintenance burden, `build.rs` can prepend
shared constants via string concatenation (the same pattern already used for
turboquant helpers and head-dim defines).

### Rust Dispatch

- `projection.rs` — update threadgroup size from 32 to 256 for quantized matvec
  dispatch, threadgroup count from `(N,1,1)` to `(ceil(N/64),1,1)`
- `ops.rs` — update encode functions for new dispatch sizing
- `ffn.rs`, `gdn.rs` — no structural changes
- `ple.rs` — barrier-only changes (Phase 3)

The `WeightBuffer` enum and `encode_projection_q8` dispatch logic remain
unchanged. Only the PSOs and dispatch parameters change.

---

## Expected Impact

| Phase | Decode (tok/s) | Prefill (tok/s) | Key metric |
|-------|---------------|-----------------|------------|
| Baseline | 20 | 220 | |
| Phase 1: K-tile=32 (all prefill) | 20 | ~700–900 | 3–4× prefill |
| Phase 2: AMX decode (all quantized) | ~40–50 | ~700–900 | 2× decode |
| Phase 3: Barrier cleanup | ~50–55 | ~750–950 | −5–10 ms |
| Phase 4: Integer Q8 (INT4 only) | ~55–65 | ~750–950 | −2–3 ms |
| **Total** | **~55–65** | **~750–950** | |
| llama.cpp reference | 90 | 1100 | |

Remaining gap after all phases:
- Decode: ~1.4× — attributable to Q4_K_M super-block format advantage (inline
  scales, integer dot products at the format level)
- Prefill: ~1.1–1.4× — near parity

## Validation

Each phase has three levels of validation: kernel correctness, numerical
equivalence, and performance measurement.

### Kernel-Level Correctness Tests

The codebase has one kernel correctness test
(`crates/ironmill-bench/tests/kernel_correctness.rs`). Each phase must add or
update correctness tests that compare the new kernel output against a CPU
reference implementation.

**Phase 1 (prefill K-tile=32):** For each updated matmul kernel, write a test
that:
1. Creates a small matrix (e.g., M=16, N=64, K=256 — at least one full K-tile)
2. Packs weights in the format's layout (blocked INT4, blocked INT8, D2Quant
   3-bit row-major, FP16 blocked)
3. Runs the GPU kernel
4. Computes the same matmul on CPU with FP32 reference weights
5. Asserts max element-wise error < 0.05 (FP16 precision bound)
6. Tests edge cases: K not divisible by 32, N not divisible by 64, M=1

Required tests (all in `kernel_correctness.rs`):
- `affine_matmul_int4_k32_correctness`
- `affine_matmul_int8_k32_correctness`
- `d2quant_matmul_3bit_k32_correctness`
- `matmul_fp16_k32_correctness`

**Phase 2 (AMX decode):** For each new AMX matvec kernel, write a test that:
1. Creates a vector (M=1) and weight matrix (e.g., N=256, K=2560)
2. Packs weights in the format's layout
3. Runs the GPU kernel
4. Compares to CPU reference
5. Asserts max error < 0.02 (matvec is tighter — single output row)
6. Tests N not divisible by 64 (partial last threadgroup)

Required tests:
- `affine_matvec_int4_amx_correctness` (verifying the existing kernel)
- `affine_matvec_int8_amx_correctness`
- `d2quant_matvec_3bit_amx_correctness`
- `batched_affine_matvec_int4_amx_correctness` (gate+up batched)
- `fused_residual_norm_affine_matvec_int4_amx_correctness`

**Phase 3 (barriers):** No new kernel tests — barriers don't affect numerical
output. Validated by the end-to-end PPL regression.

**Phase 4 (integer Q8):** Test that:
1. Quantizes input vector to Q8 (INT8 + per-group scales)
2. Runs INT4×Q8 AMX kernel
3. Compares to float reference (tolerance: max error < 0.1 due to double
   quantization)
- `affine_matvec_int4xq8_amx_correctness`

### Numerical Equivalence (End-to-End)

After each phase, the full model must produce identical quality:

```bash
# PPL must not regress beyond ±0.05 of baseline (9.34 for INT4+TQ-INT8)
cargo run --release -p ironmill-bench --features metal -- \
  --config configs/qwen35-4b-kernel-perf.toml -b metal \
  --perplexity --perplexity-sequences 1 \
  --perplexity-dataset tests/fixtures/quality/wikitext2-qwen35.json
```

PPL thresholds:
- **Phase 1–3:** PPL must match baseline exactly (same math, different tiling)
- **Phase 4:** PPL may increase by ≤0.05 (Q8 input quantization introduces
  minor additional rounding)

### Performance Measurement

After each phase, measure throughput and compare to baseline:

```bash
# Decode throughput (target: see phase-specific thresholds below)
cargo run --release -p ironmill-bench --features metal -- \
  --config configs/qwen35-4b-kernel-perf.toml -b metal -i 20 -w 5 -r 3

# Prefill throughput
cargo run --release -p ironmill-bench --features metal -- \
  --config configs/qwen35-4b-kernel-perf.toml -b metal -i 20 -w 5 -r 3 \
  --prefill-bench
```

Phase-specific performance acceptance criteria:

| Phase | Decode threshold | Prefill threshold | PPL threshold |
|-------|-----------------|-------------------|---------------|
| Phase 1 | ≥ 18 tok/s (no regression) | ≥ 500 tok/s | = baseline |
| Phase 2 | ≥ 35 tok/s | ≥ 500 tok/s (no regression) | = baseline |
| Phase 3 | ≥ 40 tok/s | ≥ 550 tok/s | = baseline |
| Phase 4 | ≥ 45 tok/s | ≥ 550 tok/s (no regression) | ≤ baseline + 0.05 |

These are minimum thresholds, not targets. Expected values are higher (see
Expected Impact table above). If a phase fails to meet its threshold, the
kernel change has a bug or the bottleneck analysis was wrong — do not proceed
to the next phase.

### GPU Error Check

Every test run must verify `cmd_buf.status() != Error` after command buffer
completion. The existing `run_pipeline_inner` already checks this
(`pipeline.rs:1015–1019`). Any GPU command buffer error is a hard failure.

### Build Verification

After modifying any `.metal` shader:
```bash
# Must compile without warnings (--features metal not needed; build.rs
# compiles shaders unconditionally on macOS)
cargo check -p ironmill-inference

# Run the existing unit tests
cargo test -p ironmill-inference --lib
```

## Hardware

Apple M2 Max, 64 GB unified memory, macOS 15.x.
Metal GPU Family: Apple 8 (supports `memoryBarrierWithResources:count:`).
