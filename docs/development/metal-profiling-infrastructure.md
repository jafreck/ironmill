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

If BW plateaus above N=4096, the issue is per-TG and kernel changes
could help. If BW keeps climbing with N, the issue is total TG count
and only dispatch-level changes (concatenating projections) would help.

### Q2: Does projection bandwidth scale with K?

**Test:** kernel_bandwidth_bench with sweep-k at N=4096.

If BW is higher at K=9216 (FFN down's K), it explains why FFN is
faster: the down projection benefits from large K, not just the fused
gate+up. This would mean the per-row work hypothesis was correct.

### Q3: What is FFN gate+up bandwidth vs FFN down bandwidth?

**Test:** Per-dispatch profiling within the FFN category.

If fused gate+up (N=9216, K=2560, 32 threads) achieves 300 GB/s and
down (N=2560, K=9216, 64 threads) achieves 100 GB/s, then the fused
kernel genuinely helps for large N. If both achieve ~192 GB/s, the
factor is K, not fusion.

### Q4: Does GDN batched QKV (N=8192) have higher BW than Standard Q (N=4096)?

**Test:** Per-dispatch profiling comparing GDN vs Standard layers.

If GDN batched at N=8192 is proportionally faster per-byte, N is the
dominant factor for this hardware.

## Implementation Order

1. **Kernel micro-benchmark** (Layer 1) — highest value, answers
   Q1-Q2 directly, no pipeline changes needed.
2. **Feature-gated profiler** (Layer 2) — replaces 21 inline sites,
   decouples profiling from production.
3. **Per-dispatch granularity** (Layer 3) — answers Q3-Q4, builds on
   Layer 2.
4. **Structured output** (Layer 4) — enables regression tracking,
   builds on all layers.

## Cleanup: Remove Inline Profiling

Once Layer 2 is implemented, remove from `pipeline.rs`:
- The `profiling` variable and `profile_boundary!` macro
- All 11 `if profiling` blocks
- The `wall_start` / `category_timings` variables
- The P1 fusion disable (`!profiling &&` guard)
- The GDN split path (`encode_gdn_projections` / `encode_gdn_core_and_output`)
- The `kernel_timing`-gated eprintln block

This reduces `pipeline.rs` by ~100 lines and eliminates all profiling
coupling from the production decode path.
