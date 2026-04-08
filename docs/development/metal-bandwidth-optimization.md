# Metal Bandwidth Optimization: Aligning with MLX Performance

## Problem

Qwen 3.5-4B INT4+TQ-INT8 decode on M2 Max 64 GB:

| Framework | Decode | Prefill | Bandwidth Utilization |
|-----------|--------|---------|----------------------|
| **ironmill** | **19 tok/s (52 ms)** | **228 tok/s** | **5% (20 GB/s)** |
| MLX | 40–60 tok/s | ~800 tok/s | ~75–85% |
| llama.cpp | 50–90 tok/s | ~1100 tok/s | ~80–90% |
| Theoretical floor | 152 tok/s (6.6 ms) | — | 100% (400 GB/s) |

ironmill reads 1.02 GB of INT4 weights per token. The M2 Max delivers
400 GB/s peak bandwidth, giving a 6.6 ms theoretical floor. The entire
52 ms decode latency is GPU-side (confirmed: GPU time = 51.0 ms, wall
time = 51.9 ms — 98% GPU-bound). No CPU overhead, no dispatch gap. The
problem is purely **on-GPU memory bandwidth utilization at 5%**.

## Previous Approach (Failed)

The prior plan (`metal-kernel-performance.md`) misdiagnosed the bottleneck
as "kernel inefficiency" and "lack of simdgroup MMA." It proposed 4 phases:

| Phase | What it did | Result |
|-------|-------------|--------|
| P1: K-tile=32 prefill | Increased MMA ops per barrier | ✅ Landed, **no effect** on prefill (228 tok/s unchanged) |
| P2: AMX decode matvec | Replaced scalar kernels with MMA | ❌ Slower (9→13 tok/s), reverted |
| P3: Barrier cleanup | Converted PLE barriers to resource-specific | ✅ Landed, **no measurable effect** |
| P4: Q8 integer dot | AMX-based integer accumulation | ❌ Never attempted |

**Why it failed:** MMA is wrong for M=1 decode. Both MLX and llama.cpp use
scalar dot products with SIMD reduction for decode — the same pattern
ironmill already uses. The speed difference is not the algorithm but
**how memory is accessed**: coalesced wide loads vs scattered byte reads,
inline metadata vs separate arrays, fused dispatches vs individual kernels.

## Root Cause Analysis

### Profiled Data

```
Total weight reads per token:     1.02 GB
M2 Max peak bandwidth:            400 GB/s
Theoretical minimum latency:      2.56 ms
Measured GPU time:                 51.0 ms
Achieved bandwidth:               20.1 GB/s
Bandwidth utilization:            5.0%
```

### Why 5% Bandwidth

Three structural problems in the current kernel design:

#### 1. Byte-by-byte nibble unpacking

Current code reads one `uchar` at a time:
```metal
uchar packed = B_packed[byte_idx];     // 1-byte load
uchar lo = packed & 0x0F;
uchar hi = (packed >> 4) & 0x0F;
```

This generates 1-byte device memory transactions. Metal's memory subsystem
coalesces adjacent accesses within a simdgroup into 32-byte or 128-byte
cache-line transactions. With 32 lanes each reading 1 byte at stride > 1,
every transaction wastes 97% of the loaded cache line.

**MLX approach:** Reads `uint32_t` or `uint64_t` (4–8 bytes per load),
unpacking 8–16 nibbles in registers. This fills cache lines efficiently.

#### 2. Scattered scale/zero metadata reads

Current code looks up scale/zero from separate arrays per element:
```metal
float s0 = float(scales[scale_row + g0]);   // random access into [N, groups]
float z0 = float(zeros[scale_row + g0]);    // random access into [N, groups]
```

For group_size=128, each output row reads K/128 = 20 scale+zero pairs from
arrays laid out as `[N, num_groups]`. Adjacent threadgroups read adjacent rows,
but the scale array has stride `num_groups` = 20 — so adjacent TGs access
addresses 40 bytes apart. With `N=2560` threadgroups, this scatters across
51 KB of scale data per projection.

**llama.cpp approach:** Q4_K_M packs scales inline with quantized data in
144-byte super-blocks. Scale reads are sequential with weight reads — zero
extra cache lines.

**MLX approach:** Still uses separate scale arrays but processes multiple
groups per simdgroup with vectorized loads, amortizing the cost.

#### 3. Tiny threadgroups with no data reuse

Each threadgroup processes 1 output row with 32 threads. Per-TG weight
data read is only ~1.3 KB (K/2 = 1280 bytes for K=2560). The input vector
x (5 KB) is read by every TG but cached in L2 after the first read.

With 2560 TGs per projection and 7 projections per layer, there are
~18,000 TGs per layer × 28 layers = ~500,000 TGs per token. Each TG
reads a tiny amount of data before terminating, causing constant TG
scheduling overhead and preventing the memory subsystem from building
sustained streaming bandwidth.

**MLX approach:** Also uses ~1 row per TG for QMV, but compensates with
wide vectorized loads and kernel fusion to reduce total dispatch count.

---

## Optimization Plan

### Strategy: Match MLX's bandwidth patterns, not llama.cpp's data format

Adopting llama.cpp's Q4_K_M super-block format would require rewriting
the quantization pipeline, weight packing, and all kernels. Instead, we
adopt MLX's approach: keep the current group-quantized layout but fix
how the GPU reads it.

### Phase 1: Vectorized Wide Loads (Target: 15–20% bandwidth → 40+ tok/s)

**The single highest-impact change.** Replace byte-by-byte weight reads
with vectorized loads that fill cache lines efficiently.

#### 1a. Pack 8 nibbles per uint32 load

```metal
// BEFORE: 1 byte per load, 2 nibbles
uchar packed = B_packed[byte_idx];

// AFTER: 4 bytes per load, 8 nibbles
uint packed4 = ((device const uint*)B_packed)[word_idx];
// Unpack 8 nibbles from the 32-bit word
float w0 = (float(packed4 & 0xF) - z) * s;
float w1 = (float((packed4 >> 4) & 0xF) - z) * s;
float w2 = (float((packed4 >> 8) & 0xF) - z) * s;
// ... 8 values from one 4-byte load
```

This increases the effective memory transaction size by 4× and improves
cache-line utilization from ~3% to ~12% per load.

#### 1b. Pre-fetch scale/zero per group

Instead of looking up scale/zero per element pair, load them once per
group and reuse across all elements in that group:

```metal
// Load scale/zero once per group (every 128 elements)
float s = float(scales[scale_row + grp]);
float z = float(zeros[scale_row + grp]);

// Process all 64 element pairs in this group with cached s/z
for (uint k = lane; k < group_half; k += 32) {
    uint packed4 = ((device const uint*)B_group)[k];
    // Unpack and accumulate using cached s, z
}
```

#### 1c. Align loads to blocked layout boundaries

The current blocked layout `[N/64, K/8, 64, 4]` stores 4 packed bytes
(8 nibbles) per innermost dimension — exactly one `uint32_t`. The byte
indexing arithmetic should be replaced with word-aligned indexing:

```metal
// BEFORE: complex byte indexing with division
uint byte_idx = (n_block * k_blocks + kb) * block_bytes
              + n_local * local_k_bytes + b;

// AFTER: word-aligned block indexing
uint word_idx = (n_block * k_blocks + kb) * (BLK_N)
              + n_local;  // one uint32 per (n_local, k_block)
uint packed4 = ((device const uint*)B_packed)[word_idx];
```

**Expected impact:** 3–4× bandwidth improvement (20 → 60–80 GB/s).
This alone should reach 35–45 tok/s.

**Files changed:**
- `shaders/quantized/affine_matvec.metal` — `affine_matvec_int4`
- `shaders/quantized/affine_batched.metal` — `batched_affine_matvec_int4`, `gdn_batched_affine_matvec_int4`
- `shaders/quantized/affine_fused.metal` — `fused_ffn_gate_up_act_int4`, `affine_matvec_int4xq8`
- No Rust dispatch changes needed — same TG counts, same buffer bindings.

### Phase 2: Kernel Fusion (Target: 25–35% bandwidth → 50–60 tok/s)

Reduce Metal dispatch count by fusing adjacent kernels that share buffers.
Each dispatch boundary costs ~2–5 µs of scheduling overhead plus a full
L2 cache flush via the inter-dispatch memory barrier. With ~500,000 TGs
per token, even small per-TG overhead accumulates.

#### 2a. Fuse RMSNorm + First Projection

Currently: `encode_rms_norm` → barrier → `encode_projection` (Q or QKV).

Fused: Single kernel reads input, computes RMSNorm inline, then does the
matvec — eliminating the intermediate buffer write and the barrier.

This pattern already partially exists in `fused_residual_norm_affine_matvec_int4`
for end-of-layer P1 fusion. Extend it to cover layer-start norm + first
projection as well.

**Impact:** Eliminates 28 dispatches + 28 barriers (one per layer).
Saves the norm_out intermediate buffer write (28 × 2560 × 2 = 140 KB/token).

#### 2b. Fuse Residual + RMSNorm + Projection (3-way)

Currently: `encode_residual_add` → barrier → `encode_rms_norm` → barrier →
`encode_projection`.

Fused: Single kernel that reads hidden_state + residual, computes
`hidden = hidden + residual`, normalizes, and feeds directly into the
matvec. Eliminates 2 barriers and 2 intermediate buffer round-trips.

**Impact:** Eliminates 56 dispatches + 56 barriers (two per layer).

#### 2c. Fuse Gate+Up+Activation+Down (FFN mega-kernel)

Currently the FFN block dispatches:
1. `batched_affine_matvec_int4` or `fused_ffn_gate_up_act_int4` (gate+up)
2. barrier
3. `affine_matvec_int4` (down projection)

The fused FFN kernel would do gate+up+activation+down in a single dispatch
by writing the intermediate `silu(gate) * up` result to threadgroup memory
and immediately reading it back for the down projection. This only works
if `intermediate_size` (6912) fits in threadgroup memory — it does not
(6912 × 2 = 13.8 KB for one row, but we need the full intermediate
vector for the down projection).

**Alternative:** Fuse gate+up+activation (already done) and keep down
as a separate dispatch, but eliminate the barrier between them by using
`threadgroup_barrier` only within the fused kernel.

**Impact:** Already partially realized by `fused_ffn_gate_up_act_int4`.
Focus fusion effort on 2a and 2b instead.

**Files changed:**
- `shaders/quantized/affine_fused.metal` — new fused norm+matvec kernel
- `metal/pipeline.rs` — rewire encode calls to use fused paths
- `metal/ops/quantized.rs` — new encode function for fused kernel

### Phase 3: Multi-Row Threadgroups (Target: 40–50% bandwidth → 60–75 tok/s)

Process 2–4 output rows per threadgroup to amortize TG scheduling overhead
and improve memory streaming.

#### Design

```metal
// 2 rows per TG: 64 threads (2 simdgroups)
// Each simdgroup handles 1 output row
kernel void affine_matvec_int4_2row(
    ...,
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    uint row = tgid * 2 + sgid;
    if (row >= N) return;

    // Both simdgroups read the same input x (shared in L1)
    // Each reads different weight rows (adjacent in blocked layout)
    float acc = 0.0f;
    for (uint k = lane; k < half_K; k += 32) {
        uint packed4 = load_weight_word(row, k);
        // ... vectorized dequant + dot product
    }
    acc = simd_sum(acc);
    if (lane == 0) C[row] = half(acc);
}
```

- Dispatch: `(ceil(N/2), 1, 1)` TGs × `(64, 1, 1)` threads
- Halves TG count → halves scheduling overhead
- Adjacent rows in the same N-block read adjacent memory → better
  coalescing within the blocked layout
- Input vector `x` is shared across simdgroups in L1 cache

**Why not 64 rows (AMX style)?** The failed Phase 2 showed that
cooperative dequant into threadgroup memory + MMA adds more overhead
than it saves for M=1. The sweet spot is 2–4 rows with independent
scalar dot products sharing only the input vector through cache.

**Files changed:**
- `shaders/quantized/affine_matvec.metal` — new `_2row` variant
- `shaders/quantized/affine_batched.metal` — update batched kernels
- `shaders/quantized/affine_fused.metal` — update fused kernels
- `metal/ops/quantized.rs` — update dispatch to (ceil(N/2), 1, 1) × (64, 1, 1)
- `metal/projection.rs` — update decode dispatch

### Phase 4: Prefill Dequant Optimization (Target: 500+ tok/s prefill)

The prefill matmul kernels already have correct structure (K-tile=32,
double-buffered, simdgroup MMA) but still achieve only 228 tok/s. The
bottleneck is the same: the B-tile load phase uses byte-by-byte dequant.

Apply the same vectorized load pattern from Phase 1 to the B-tile load
functions in all prefill matmul kernels:

- `affine_matmul_int4` — vectorize the B-tile dequant loop
- `affine_matmul_int8` — same pattern, byte loads → uint32 loads
- `d2quant_matmul_3bit` — vectorize 3-bit unpacking (3 bytes → 8 values)

**Expected impact:** 2–3× prefill improvement (228 → 500–700 tok/s).

**Files changed:**
- `shaders/quantized/affine_matmul.metal`
- `shaders/quantized/d2quant_matmul.metal`

---

## Expected Impact

| Phase | Decode (tok/s) | Prefill (tok/s) | BW Util | Key Change |
|-------|---------------|-----------------|---------|------------|
| Baseline | 19 | 228 | 5% | — |
| Phase 1: Wide loads | ~40 | ~300 | ~20% | uint32 reads, cached scale/zero |
| Phase 2: Kernel fusion | ~55 | ~400 | ~28% | −56 barriers, −56 dispatches |
| Phase 3: Multi-row TG | ~65 | ~400 | ~33% | 2 rows/TG, halved TG count |
| Phase 4: Prefill dequant | ~65 | ~600 | ~33% | Vectorized B-tile loads |
| **Total** | **~65** | **~600** | **~33%** | |
| MLX reference | 40–60 | ~800 | ~75–85% | |
| llama.cpp reference | 50–90 | ~1100 | ~80–90% | |

The remaining gap to MLX after all phases (~0–35%) would require:
- **Inline scale packing** (format change, not just kernel change)
- **Auto-specialized kernel variants** per group-size/dim/dtype
- **Lazy evaluation / compute graph batching** (framework-level change)

These are larger architectural investments beyond kernel optimization.

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
| Phase 1 | ≥ 30 tok/s | ≥ 250 tok/s | = baseline |
| Phase 2 | ≥ 45 tok/s | ≥ 350 tok/s | = baseline |
| Phase 3 | ≥ 55 tok/s | ≥ 350 tok/s | = baseline |
| Phase 4 | ≥ 55 tok/s | ≥ 500 tok/s | = baseline |

### GPU Profiling

Use the `gpu_start_time()`/`gpu_end_time()` API (added to
`ironmill-metal-sys::CommandBuffer`) to verify GPU time tracks wall time:
```rust
cmd_buf.commit();
cmd_buf.wait_until_completed();
let gpu_ms = (cmd_buf.gpu_end_time() - cmd_buf.gpu_start_time()) * 1000.0;
```

After each phase, GPU time should decrease proportionally to tok/s gains.
If wall time decreases but GPU time doesn't, the improvement is CPU-side
(not bandwidth) and the phase hasn't addressed the real bottleneck.

---

## Hardware

Apple M2 Max, 64 GB unified memory, macOS 15.x.
Peak memory bandwidth: 400 GB/s.
Metal GPU Family: Apple 8.
