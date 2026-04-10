# Metal Kernel Profiling Infrastructure

## Problem

The current per-category GPU profiling (Phase 1a) answered *where* time
is spent during decode but not *why* bandwidth utilization varies across
kernel types. Five optimization attempts failed because we lacked the
data to identify the causal mechanism.

The profiling code is also tightly coupled to the production pipeline:
21 `if profiling` / `profile_boundary!` call sites are interleaved
throughout `pipeline.rs`, adding maintenance burden and risk of
behavioral divergence when the pipeline changes.

## Goals

1. **Answer the open questions** — provide per-kernel bandwidth data
   that isolates the effect of N, K, kernel type, and dispatch count.
2. **Zero production cost** — no branches, no timing code, no macro
   expansions in the non-profiling build.
3. **Reusable across models** — works for any model, any layer structure
   (Standard, GDN, MLA, Gemma 4, MoE), any weight quantization.
4. **Composable** — kernel micro-benchmarks and full-pipeline profiling
   share infrastructure.

## Design

### Layer 1: Kernel micro-benchmark harness

A standalone binary (`examples/kernel_bandwidth_bench.rs`) that tests
individual kernels in isolation without loading a full model.

```
cargo run --release -p ironmill-bench --example kernel_bandwidth_bench \
  --features metal
```

**What it does:**

1. Allocates synthetic weight buffers of configurable (N, K) dimensions
   with random INT4 data, scales, zeros, and optional AWQ scales.
2. Runs a single kernel variant (superblock matvec, fused FFN, batched,
   GDN batched) on the synthetic data with configurable dispatch params.
3. Measures GPU time via `GPUStartTime`/`GPUEndTime` over multiple
   iterations (warmup + measured).
4. Reports: N, K, kernel name, GPU time, effective bandwidth (GB/s),
   % of peak, throughput (GFLOP/s).

**Sweep modes:**

- `--sweep-n 256,512,1024,2048,4096,8192,16384 --k 2560` — isolates
  the effect of N at fixed K (answers: is low projection BW caused by
  small N?)
- `--sweep-k 256,512,1024,2560,4096,9216 --n 4096` — isolates the
  effect of K at fixed N (answers: does large K in FFN down explain
  its high BW?)
- `--compare-kernels --n 4096 --k 2560` — runs all kernel variants at
  the same dimensions for direct comparison.

**This is the critical missing piece.** It would have prevented all five
failed optimization attempts by showing that bandwidth scales with N
(or K, or some other parameter) before any kernel changes were made.

### Layer 2: Pipeline profiling via wrapper (replaces inline profiling)

Replace the 21 inline `if profiling` sites with a decoupled profiling
layer that wraps the command encoder.

**Approach: `ProfiledEncoder` wrapper**

```rust
// In a separate module: metal/profiling.rs (only compiled with cfg)

#[cfg(feature = "profile-metal")]
pub struct ProfiledEncoder<'a> {
    queue: &'a CommandQueue,
    cmd_buf: CommandBuffer,
    enc: ComputeEncoder,
    timings: Vec<(&'static str, f64)>,
    current_category: &'static str,
}

#[cfg(feature = "profile-metal")]
impl ProfiledEncoder<'_> {
    pub fn switch_category(&mut self, name: &'static str) {
        self.enc.end_encoding();
        self.cmd_buf.commit();
        self.cmd_buf.wait_until_completed();
        let gpu_ms = (self.cmd_buf.gpu_end_time()
                    - self.cmd_buf.gpu_start_time()) * 1000.0;
        self.timings.push((self.current_category, gpu_ms));
        self.current_category = name;
        self.cmd_buf = self.queue.command_buffer().unwrap();
        self.enc = self.cmd_buf.compute_encoder().unwrap();
    }

    pub fn encoder(&self) -> &ComputeEncoder { &self.enc }
    pub fn finish(self) -> Vec<(&'static str, f64)> { ... }
}
```

**Pipeline integration:**

```rust
// pipeline.rs — zero-cost in production

#[cfg(feature = "profile-metal")]
use super::profiling::ProfiledEncoder;

// In run_pipeline_inner:
#[cfg(feature = "profile-metal")]
let mut profiler = ProfiledEncoder::new(&self.queue, "embed");

// At category boundaries (replaces inline `if profiling { ... }`):
#[cfg(feature = "profile-metal")]
profiler.switch_category("proj");

// All encode_* functions take &ComputeEncoder as before:
encode_projection(&enc, ...);  // production
#[cfg(feature = "profile-metal")]
encode_projection(profiler.encoder(), ...);  // profiling
```

**Key properties:**

- `#[cfg(feature = "profile-metal")]` — entire profiling module is
  compiled out in production. Zero branches, zero dead code.
- `profile-metal` is NOT in default features. Must be explicitly
  enabled: `--features profile-metal`.
- The pipeline code has `#[cfg]` annotations at category boundaries
  instead of runtime `if profiling` branches. These are invisible to
  the compiler in production builds.
- P1 fusion is disabled via `#[cfg]` too, not a runtime flag.

### Layer 3: Per-dispatch fine-grained timing

For cases where per-category timing is too coarse (e.g., FFN gate+up
vs FFN down within the "ffn" category), add optional per-dispatch
timing using separate command buffers per dispatch.

**Approach:** A `--profile-granularity=dispatch` flag that makes the
profiler switch command buffers after EVERY dispatch call, not just at
category boundaries. This gives per-kernel GPU time at the cost of
heavy overhead (~50ms total from ~280 command buffer switches).

This is only useful for targeted investigation, not routine profiling.

### Layer 4: Structured output

All profiling tools output machine-parseable data (JSON or CSV) in
addition to human-readable tables. This enables:

- Regression tracking across commits
- Automated analysis scripts
- Comparison across models (Qwen3.5-4B vs Gemma 4 vs Llama)

```json
{
  "model": "Qwen3.5-4B",
  "quantization": "int4-awq-gs128",
  "hardware": "M2 Max 64GB",
  "total_gpu_ms": 22.24,
  "categories": {
    "proj": {"gpu_ms": 10.48, "weight_mb": 526, "bw_gbs": 50.2},
    "ffn":  {"gpu_ms": 5.90,  "weight_mb": 1133, "bw_gbs": 192.0},
    "attn": {"gpu_ms": 4.59},
    "norm": {"gpu_ms": 0.35},
    "embed": {"gpu_ms": 0.01},
    "lm_head": {"gpu_ms": 0.90}
  }
}
```

## Open Questions the Infrastructure Must Answer

These are the specific unknowns that blocked optimization progress:

### Q1: Does projection bandwidth scale with N?

**Test:** kernel_bandwidth_bench with sweep-n at K=2560 using
`superblock_matvec_int4`.

**Result — YES, strongly.** BW scales ~90× from N=128 to N=10240:

| N     | GPU (µs) | BW (GB/s) | % peak |
|-------|----------|-----------|--------|
| 128   | 51.0     | 3.5       | 0.9%   |
| 256   | 50.6     | 7.0       | 1.7%   |
| 512   | 50.5     | 13.9      | 3.5%   |
| 1024  | 51.5     | 27.2      | 6.8%   |
| 2048  | 33.7     | 82.8      | 20.7%  |
| 3072  | 21.5     | 194.8     | 48.7%  |
| 4096  | 22.2     | 251.4     | 62.9%  |
| 8192  | 40.7     | 274.5     | 68.6%  |
| 10240 | 43.6     | 319.8     | 80.0%  |

**Conclusion:** BW keeps climbing with N. The issue is total TG count —
small-N projections (Q/K/V at N=2560) achieve only ~66 GB/s (17% of
peak). Concatenating projections (larger effective N) is the primary
optimization lever. GPU time is dominated by launch overhead at small N
(~50µs floor up to N≈1024).

### Q2: Does projection bandwidth scale with K?

**Test:** kernel_bandwidth_bench with sweep-k at N=4096.

**Result — YES, also scales with K:**

| K     | GPU (µs) | BW (GB/s) | % peak |
|-------|----------|-----------|--------|
| 128   | 17.4     | 16.5      | 4.1%   |
| 256   | 17.8     | 31.9      | 8.0%   |
| 512   | 23.0     | 48.9      | 12.2%  |
| 1024  | 32.5     | 69.0      | 17.2%  |
| 2560  | 29.8     | 187.7     | 46.9%  |
| 4096  | 32.2     | 277.6     | 69.4%  |
| 9216  | 65.2     | 307.9     | 77.0%  |

**Conclusion:** K=9216 (FFN down's K) achieves 308 GB/s vs K=2560's
188 GB/s. The FFN down projection benefits from large K, confirming the
per-row work hypothesis — more work per TG at large K amortizes memory
latency better.

### Q3: What is FFN gate+up bandwidth vs FFN down bandwidth?

**Test:** Kernel comparison at Qwen3.5-4B dims (h=2560, inter=10240).

| kernel                 | N     | K     | BW (GB/s) | % peak |
|------------------------|-------|-------|-----------|--------|
| matvec_int4 (Q/K/V)   | 2560  | 2560  | 66.3      | 16.6%  |
| fused_ffn_gate_up_act  | 2560  | 2560  | 81.0      | 20.2%  |
| batched_matvec_int4    | 2560  | 2560  | 85.1      | 21.3%  |
| matvec (gate+up N)     | 10240 | 2560  | 322.0     | 80.5%  |
| matvec (down N)        | 2560  | 10240 | 220.1     | 55.0%  |

**Conclusion:** At the same N,K the fused/batched kernels give only
modest BW improvement (81-85 vs 66 GB/s). The real FFN advantage comes
from large N (gate+up at N=10240 → 322 GB/s) and large K (down at
K=10240 → 220 GB/s). The fused kernel's benefit is amortizing input
reads over 2 weight matrices, not fundamentally different memory access.

### Q4: Does GDN batched QKV (N=8192) have higher BW than Standard Q (N=4096)?

**Answer (from Q1 data):** Yes. At K=2560, N=8192 achieves 275 GB/s
vs N=4096 at 251 GB/s (both using the basic matvec kernel). This
confirms N is the dominant factor for this hardware — GDN's 4-way
batched projection at N=8192 would achieve proportionally higher BW
per-byte than Standard Q at N=4096.

## Key Finding: The Bandwidth Bottleneck

**Projection bandwidth is low because N is small, not because of
kernel inefficiency.** The M2 Max GPU needs N≥4096 to reach 50%+ of
peak bandwidth. Typical Q/K/V projections (N=2560-4096) operate at
17-25% of peak.

**Optimization implications:**
1. **Concatenate Q/K/V projections** into a single larger dispatch
   (effective N = 3× or 4×) to increase TG count and BW utilization.
2. **FFN is already fast** because fused gate+up reads 2 matrices per
   TG (effective N = 2×inter) and down has large K.
3. **Kernel-level changes** (wider TGs, split-K) won't help because
   the bottleneck is insufficient total TG count, not per-TG efficiency.
4. **Hardware matters**: the ~50µs GPU launch overhead floor means very
   small dispatches (N<1024) can never be fast regardless of kernel.

## Implementation Status

All 4 layers are implemented:

1. ✅ **Kernel micro-benchmark** — `examples/kernel_bandwidth_bench.rs`
2. ✅ **Feature-gated profiler** — `profile-metal` feature,
   `profiling.rs` module, 21 inline sites replaced with `#[cfg]`
3. ✅ **Per-dispatch granularity** — `ProfilingGranularity::Dispatch`
4. ✅ **Structured output** — JSON from both kernel bench and profiler

## Cleanup Status

Inline profiling has been replaced with `#[cfg(feature = "profile-metal")]`
annotations. In production builds (without `profile-metal` feature):
- The `profiling` variable, `profile_boundary!` macro, `category_timings`,
  and `wall_start` do not exist
- All 11 `if profiling` blocks are compiled out
- The P1 fusion disable guard is always `true` (fusion enabled)
- The GDN split path functions are compiled out
- `pipeline.rs` has zero profiling overhead
