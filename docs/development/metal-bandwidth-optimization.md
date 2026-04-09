# Metal Bandwidth Optimization: Aligning with MLX Performance

## Problem

Qwen 3.5-4B INT4 gs=128 decode on M2 Max 64 GB:

| Framework | Quantization | Decode | Prefill | BW Util |
|-----------|-------------|--------|---------|---------|
| **ironmill** | **INT4 superblock gs=128** | **38 tok/s** | **242 tok/s** | **~10%** |
| MLX | INT4 gs=128 | 97 tok/s | — | ~50% |
| llama.cpp | Q4_K_M gs=64 | 67 tok/s | 1066 tok/s | ~35% |
| Theoretical floor | — | 152 tok/s | — | 100% (400 GB/s) |

ironmill is **2.5× behind MLX** on identical quantization and **1.8× behind
llama.cpp**. The problem is purely on-GPU memory bandwidth utilization.
No CPU overhead (GPU time ≈ wall time, 98% GPU-bound).

## Progress Summary

### Original Bottlenecks (from 19 tok/s baseline)

The original plan (written at 19 tok/s) identified three structural problems.
Here is their current status:

| Original Problem | Status | What Was Done | Impact |
|-----------------|--------|---------------|--------|
| Byte-by-byte nibble unpacking | ✅ **Fixed** | Superblock migration + uint16/uint32 vectorized loads | 19 → 31 tok/s |
| Scattered scale/zero metadata | ✅ **Fixed** | Inline superblock headers (scale+bias in 4B prefix) | Included above |
| Tiny 1-row TGs (2560 TGs/proj) | ✅ **Fixed** | 8 rows/TG (2 SGs × 4 rows/SG), 320 TGs/proj | 31 → 36 tok/s |

Additional optimizations landed:

| Optimization | Impact |
|-------------|--------|
| Compile-time `#define GS` specialization (÷ → >>) | 36 → 38 tok/s |
| MLX-style pre-scaled input trick (eliminates 8 shifts+subs) | No change (memory-bound) |
| Kernel fusion: residual+RMSNorm+projection (3-way) | In place for decode |
| Kernel fusion: gate+up+activation (FFN) | In place for decode |

**Net result: 19 → 38 tok/s (2× improvement). But MLX is at 97 tok/s.**

### Previous Approach (Failed)

The prior plan (`metal-kernel-performance.md`) misdiagnosed the bottleneck
as "kernel inefficiency" and "lack of simdgroup MMA." Its 4 phases all failed
or had no effect. MMA is wrong for M=1 decode — both MLX and llama.cpp use
scalar dot products with SIMD reduction, the same algorithm ironmill uses.

---

## Current Root Cause Analysis

### Profiled Data

```
Weights per token (INT4 gs=128):  1.02 GB
M2 Max peak bandwidth:            400 GB/s
Theoretical minimum latency:      2.56 ms
Measured decode latency:          ~26 ms
Achieved bandwidth:               ~40 GB/s
Bandwidth utilization:            ~10%
```

### Why 2.5× Behind MLX

Two structural problems remain after the superblock migration. Both were
identified by comparing ironmill's `superblock_matvec_int4` kernel against
MLX's `qmv_fast_impl` (in `mlx/backend/metal/kernels/quantized.h`).

#### 1. Superblock header cache waste (~33% bandwidth loss)

ironmill's superblock format places a 4-byte header (2B scale + 2B bias)
before every group's packed data. For INT4 gs=128, each superblock is
68 bytes: 4 header + 64 data.

In the decode kernel, 32 SIMD lanes read from 2 consecutive groups per
K-iteration. The memory span per row is:

```
Group 0: [sb+4 .. sb+67]     64 bytes data (lanes 0–15)
          [sb+68 .. sb+71]    4 bytes header (GROUP 1 — GAP)
Group 1: [sb+72 .. sb+135]   64 bytes data (lanes 16–31)
```

Total span: 132 bytes across 3 cache lines (192 bytes fetched).
Useful data: 128 bytes → **67% cache utilization**.

MLX uses **separate contiguous arrays** for weights, scales, and biases.
32 lanes × 8 packed bytes = 256 contiguous bytes per row → 4 cache lines,
**100% utilization**. No gaps.

```
ironmill:  128 useful / 192 fetched = 67% utilization
MLX:       256 useful / 256 fetched = 100% utilization
Ratio:     0.67×
```

#### 2. Half the values per thread (~2× loop overhead)

ironmill loads **8 elements per thread** per K-iteration (2 × uint16 = 4
packed bytes). MLX loads **16 elements per thread** (4 × uint16 = 8 packed
bytes). With 32 lanes per simdgroup:

```
ironmill:  8 × 32 = 256 elements per K block
MLX:       16 × 32 = 512 elements per K block
```

For K=2560, ironmill needs 10 K-iterations vs MLX's 5. Each iteration
incurs pointer arithmetic (group index, superblock offset), branch checks,
and per-row metadata reads. **2× more iterations = 2× more overhead per
useful byte.**

#### Combined Impact

```
Cache efficiency:   0.67× (superblock headers)
Values per thread:  0.50× (half the elements)
Combined:           0.67 × 0.50 = 0.33×
Predicted:          97 × 0.33 = 32 tok/s
Measured:           38 tok/s
```

The model predicts ~33% of MLX throughput; measured is ~39%. The small
overestimate is explained by ironmill's inline scale reads being slightly
cheaper than MLX's separate-array lookups (better spatial locality per group).

#### Additional MLX Advantages (Secondary)

- **Split-K dispatch**: MLX splits the K dimension across multiple TGs
  (split_k=8 for K≤8192), launching 8× more TGs for better occupancy
  and memory-latency hiding. ironmill processes full K per TG.
- **Contiguous row-major weights**: MLX's weight array has zero gaps,
  enabling maximum memory streaming bandwidth.

---

## Remaining Optimization Plan

### Strategy: Close the 2.5× MLX gap via cache efficiency + per-thread throughput

The original 4-phase plan is largely completed. The remaining gap requires
two targeted changes that address the specific root causes above, plus
the occupancy sweep from `decode-perf-investigation.md`.

### Phase A: Double Values Per Thread (Target: ~55 tok/s)

**Highest-confidence change.** Load 16 elements per thread per K-iteration
instead of 8, matching MLX's `qmv_fast_impl`. This halves the number of
K loop iterations and doubles the useful bytes per memory request.

#### Current code (8 elements/thread):
```metal
// 2 × uint16 = 4 packed bytes = 8 nibbles per lane per iteration
device const uint16_t *ws = (device const uint16_t *)(sb + SB_HEADER_BYTES) + u16_in_data;
float accum = xp0 * float(ws[0] & 0x000f) + xp1 * float(ws[0] & 0x00f0)
            + xp2 * float(ws[0] & 0x0f00) + xp3 * float(ws[0] & 0xf000)
            + xp4 * float(ws[1] & 0x000f) + xp5 * float(ws[1] & 0x00f0)
            + xp6 * float(ws[1] & 0x0f00) + xp7 * float(ws[1] & 0xf000);
```

#### Proposed (16 elements/thread):
```metal
// 4 × uint16 = 8 packed bytes = 16 nibbles per lane per iteration
uint k_words = K / 16;  // half as many iterations
for (uint w = lane; w < k_words; w += 32) {
    uint k_elem = w * 16;
    uint g = k_elem / GS;
    uint u16_in_data = (k_elem % GS) / 4;

    // Pre-scale 16 input values (4 × 4-element pattern)
    // ... xp0..xp15 with ÷1, ÷16, ÷256, ÷4096 pattern ...

    for (uint r = 0; r < SB_ROWS_PER_SG; r++) {
        device const uchar *sb = W + row * sb_stride + g * SB_BYTES_INT4;
        float scale = float(*(device const half *)(sb));
        float bias  = float(*(device const half *)(sb + 2));

        device const uint16_t *ws =
            (device const uint16_t *)(sb + SB_HEADER_BYTES) + u16_in_data;

        float accum = xp0 * float(ws[0] & 0x000f) + xp1 * float(ws[0] & 0x00f0)
                    + xp2 * float(ws[0] & 0x0f00) + xp3 * float(ws[0] & 0xf000)
                    + xp4 * float(ws[1] & 0x000f) + xp5 * float(ws[1] & 0x00f0)
                    + xp6 * float(ws[1] & 0x0f00) + xp7 * float(ws[1] & 0xf000)
                    + xp8  * float(ws[2] & 0x000f) + xp9  * float(ws[2] & 0x00f0)
                    + xp10 * float(ws[2] & 0x0f00) + xp11 * float(ws[2] & 0xf000)
                    + xp12 * float(ws[3] & 0x000f) + xp13 * float(ws[3] & 0x00f0)
                    + xp14 * float(ws[3] & 0x0f00) + xp15 * float(ws[3] & 0xf000);

        result[r] += scale * accum + bias * x_sum;
    }
}
```

**Constraint:** 16 elements spans either 1 or 2 groups depending on alignment.
For GS=128, 16 elements is always within a single group (16 < 128), so the
scale/bias lookup stays simple. For GS=32, 16 elements spans exactly 1 group
(16 < 32), also safe.

**Note on cross-group spanning:** Each lane processes 16 consecutive K elements
starting at `k_elem = lane * 16`. With 32 lanes × 16 = 512 elements per
K block, and GS=128, this spans 4 groups per iteration. Each lane stays
within its own group. The lane→group mapping is `g = (lane * 16) / GS`.
Groups change at lane boundaries (lane 0-7 → g, lane 8-15 → g+1, etc.),
so scale/bias must be looked up per-lane, which is already the case.

**Expected impact:** ~1.5–2× decode improvement (38 → 55–70 tok/s).
Halving K iterations reduces per-iteration overhead and doubles useful
bytes per memory transaction.

**Files to change:**
- `shaders/quantized/affine_matvec.metal` — `superblock_matvec_int4`
- `shaders/quantized/affine_batched.metal` — `batched_affine_matvec_int4`
- `shaders/quantized/affine_fused.metal` — all INT4 kernels
- No Rust dispatch changes (same TG count, same buffer bindings).

### Phase B: Separate Scale/Bias Arrays (Target: ~75 tok/s)

Replace superblock inline headers with MLX-style separate contiguous arrays.
This eliminates the 4-byte gaps that waste ~33% of cache fetches.

#### Layout change:
```
BEFORE (superblock):
  W[row][group] = [scale:2B][bias:2B][data:GS/2 bytes]
  68 bytes per superblock for INT4 gs=128

AFTER (separate arrays):
  W_data[row][K/2]     — contiguous packed nibbles, no gaps
  W_scales[row][K/GS]  — separate half array
  W_biases[row][K/GS]  — separate half array
```

This changes the weight packing format and adds 2 buffer bindings per kernel.

#### Kernel change:
```metal
// BEFORE: inline header read
device const uchar *sb = W + row * sb_stride + g * SB_BYTES_INT4;
float scale = float(*(device const half *)(sb));
float bias  = float(*(device const half *)(sb + 2));
device const uint16_t *ws = (device const uint16_t *)(sb + SB_HEADER_BYTES) + ...;

// AFTER: separate array read
float scale = float(scales[row * num_groups + g]);
float bias  = float(biases[row * num_groups + g]);
device const uint16_t *ws = (device const uint16_t *)(W_data + row * k_half) + ...;
```

**Trade-off:** Separate scale/bias reads lose superblock spatial locality
(scale is no longer in the same cache line as its data). However, with
K/GS = 20 groups per row, the scale/bias arrays are only 80 bytes per row
— easily cached in L2 after the first group iteration. The net effect is
positive because the weight data array becomes gap-free.

**Expected impact:** ~1.3–1.5× on top of Phase A (55 → 70–80 tok/s).
Eliminates 33% cache waste from superblock headers.

**Files to change:**
- `metal/weights.rs` — add `scales`/`biases` buffer fields, update packing
- `shaders/quantized/affine_matvec.metal` — read from separate arrays
- `shaders/quantized/affine_batched.metal` — same
- `shaders/quantized/affine_fused.metal` — same
- `shaders/quantized/affine_matmul.metal` — prefill kernels too
- `metal/projection.rs` — add buffer bindings for scales/biases
- `metal/ops/quantized.rs` — update encode helpers

### Phase C: Occupancy Sweep (Target: find optimal TG config)

Run the 6-config occupancy experiment from `decode-perf-investigation.md`
to determine if the current 2 SG × 4 rows/SG structure is optimal.

| Config | SGs/TG | Rows/SG | Threads/TG | Rows/TG | Expected TGs |
|--------|--------|---------|------------|---------|-------------|
| A | 1 | 2 | 32 | 2 | 1280 |
| B | 1 | 4 | 32 | 4 | 640 |
| C | 1 | 8 | 32 | 8 | 320 |
| D | 2 | 2 | 64 | 4 | 640 |
| **E (current)** | **2** | **4** | **64** | **8** | **320** |
| F | 2 | 8 | 64 | 16 | 160 |

**Decision rule:** If any config wins by ≥ 5% over current, adopt it.
Run this AFTER Phase A to test with the improved kernel.

**Files to change:**
- `shaders/quantized/superblock_header.metal` — `SB_ROWS_PER_SG`, `SB_NUM_SIMDGROUPS`
- `shaders/quantized/affine_common.metal` — same constants
- `metal/projection.rs` — dispatch grid/thread counts

### Phase D: INT8 + 3-bit Prefill Vectorization (Target: 400+ tok/s prefill)

INT4 prefill is already vectorized (uint32 loads in B-tile load). But INT8
and 3-bit paths still use byte-by-byte reads:

```metal
// INT8 prefill — still byte-by-byte:
uchar q = sb[SB_HEADER_BYTES + sb_offset];

// 3-bit prefill — 3 byte reads per 8 values:
uint bits = uint(B_packed[off]) | (uint(B_packed[off+1]) << 8)
          | (uint(B_packed[off+2]) << 16);
```

**INT8 fix:** Replace with uint32 loads (4 bytes = 4 INT8 values per load).
**3-bit fix:** Replace 3-byte concatenation with a single uint32 load
(read 4 bytes, mask to 24 useful bits, unpack 8 3-bit values).

**Files to change:**
- `shaders/quantized/affine_matmul.metal` — `superblock_matmul_int8` B-tile load
- `shaders/quantized/d2quant_matmul.metal` — `d2quant_matmul_3bit` B-tile load

---

## Expected Impact

| Phase | Decode (tok/s) | Prefill (tok/s) | Key Change |
|-------|---------------|-----------------|------------|
| Baseline (current) | 38 | 242 | Superblock + pre-scaled input |
| Phase A: 16 vals/thread | ~55 | ~300 | Half K iterations, 2× useful bytes/request |
| Phase B: Separate arrays | ~75 | ~400 | Eliminate 33% cache waste |
| Phase C: Occupancy sweep | ~80 | ~400 | Optimal TG configuration |
| Phase D: INT8/3-bit prefill | ~80 | ~500 | Vectorize remaining prefill paths |
| **Projected total** | **~80** | **~500** | |
| MLX reference | 97 | — | |
| llama.cpp reference | 67 | 1066 | |

The remaining gap to MLX after all phases (~17%) would require:
- **Split-K dispatch** (multiple TGs per output row, final reduction)
- **Auto-specialized kernel variants** per group-size/dim/dtype
- **Lazy evaluation / compute graph batching** (framework-level change)

These are larger architectural investments beyond kernel optimization.

---

## Completed Work (Historical)

The following phases from the original plan have been implemented:

### ✅ Phase 1: Vectorized Wide Loads → DONE (19 → 31 tok/s)

Migrated from byte-by-byte blocked layout to row-major superblock format
with uint16/uint32 vectorized loads. Scale/bias read inline from 4-byte
superblock headers. All decode, batched, and fused kernels updated.

Commits: `617280a`, `50cb934`, `312683b`, `970b81e`.

### ✅ Phase 2: Kernel Fusion → DONE (decode path)

- 3-way residual+RMSNorm+projection fusion via
  `encode_fused_residual_norm_affine_matvec_int4` (used at end of each
  layer to fuse with next layer's first projection).
- Gate+up+activation fusion via `superblock_fused_ffn_gate_up_act_int4`
  (eliminates separate activation dispatch).
- Down projection remains separate (intermediate vector too large for TG
  memory).
- Prefill still uses separate dispatches (room for future improvement).

### ✅ Phase 3: Multi-Row Threadgroups → DONE (8 rows/TG)

Implemented 2 simdgroups × 4 rows/SG = 8 rows/TG, dispatching
`ceil(N/8)` TGs × 64 threads. Optimal configuration TBD via Phase C
occupancy sweep.

### ⚠️ Phase 4: Prefill Dequant → PARTIAL

INT4 prefill matmul (`superblock_matmul_int4`) uses uint32 vectorized
B-tile loads. INT8 and 3-bit paths still use byte-by-byte reads (see
Phase D above).

---

## Validation

### Per-Phase Correctness

Each phase must pass:
```bash
cargo check -p ironmill-inference
cargo test -p ironmill-inference --lib    # 241 tests
```

### Numerical Equivalence

PPL must not regress beyond ±0.05 of baseline (9.39):
```bash
cargo run --release -p ironmill-bench --features metal -- \
  --config configs/qwen35-4b-kernel-perf.toml --suite perplexity
```

### Performance Thresholds

| Phase | Decode minimum | Prefill minimum | PPL |
|-------|---------------|-----------------|-----|
| Phase A: 16 vals/thread | ≥ 48 tok/s | ≥ 250 tok/s | = baseline |
| Phase B: Separate arrays | ≥ 65 tok/s | ≥ 350 tok/s | = baseline |
| Phase C: Occupancy sweep | ≥ 65 tok/s | ≥ 350 tok/s | = baseline |
| Phase D: INT8/3-bit | ≥ 65 tok/s | ≥ 400 tok/s | = baseline |

### Decode Benchmark

Use the AWQ-INT4 decode config (this exercises `superblock_matvec_int4`):
```bash
cargo run --release -p ironmill-bench --features metal -- \
  --config configs/qwen35-4b-decode-perf.toml
```

### GPU Profiling

Use the `gpu_start_time()`/`gpu_end_time()` API to verify GPU time tracks
wall time:
```rust
cmd_buf.commit();
cmd_buf.wait_until_completed();
let gpu_ms = (cmd_buf.gpu_end_time() - cmd_buf.gpu_start_time()) * 1000.0;
```

After each phase, GPU time should decrease proportionally to tok/s gains.
If wall time decreases but GPU time doesn't, the improvement is CPU-side
(not bandwidth) and the phase hasn't addressed the real bottleneck.

---

## Related Documents

- `decode-perf-investigation.md` — Detailed investigation brief with 5
  concrete experiments (occupancy sweep, Xcode profiling, separate arrays
  test, dispatch overhead, MLX source analysis).
- `metal-kernel-performance.md` — Original (failed) kernel optimization
  plan that misdiagnosed the bottleneck as MMA/ALU.

## Hardware

Apple M2 Max, 64 GB unified memory, macOS 15.x.
Peak memory bandwidth: 400 GB/s.
Metal GPU Family: Apple 8.
