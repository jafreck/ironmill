# ANE Inference Improvements — Implementation Spec

> **Status:** Proposed
> **Depends on:** [ANE Runtime Behavior](../research/ane-runtime-behavior.md), [Metal→ANE Adapter](metal-ane-adapter.md)
> **Crate:** `ironmill-ane-sys`, `ironmill-inference/src/ane/`

## Overview

Seven improvements to ANE inference, ordered by risk and impact. All are grounded in empirical findings from the ANE runtime probes — no speculative API usage.

The ANE backend currently suffers from three structural problems:
1. **Algorithmic divergence** from Metal's TurboQuant (different quantizer, different QJL formula)
2. **Per-layer CPU roundtrips** between every ANE eval
3. **Suboptimal program granularity** — many small MIL programs where fewer fused ones would reduce dispatch overhead

---

## 1. Unify Codebook Quantizer Math

**Risk:** Low — all required ops confirmed compilable on ANE
**Impact:** High — eliminates algorithmic divergence with Metal
**Compile budget cost:** 0 (same number of programs, different ops inside)

### Problem
ANE uses beta-optimal quantization levels (`mil_rs::ir::passes::beta_quantizer`). Metal uses Lloyd-Max Gaussian codebooks (`ironmill-inference/src/turboquant/codebook.rs`). They produce different reconstruction quality for the same bit-width.

### Change
Rewrite `ane/turboquant/mil_emitter.rs` cache-write program to use Lloyd-Max codebooks expressed as MIL `greater`/`select` chains instead of scalar `round`/`clip`.

### MIL pattern for codebook lookup (b-bit, 2^b levels)
```
// boundaries: [b0, b1, ..., b_{2^b - 2}] (precomputed on CPU)
// levels:     [l0, l1, ..., l_{2^b - 1}] (precomputed on CPU)

// For each value x, find which bucket it falls into:
c0 = greater(x, boundary_0)          // x > b0?
c1 = greater(x, boundary_1)          // x > b1?
...
// Select corresponding level:
q = select(c0, level_1, level_0)
q = select(c1, level_2, q)
...
```

For INT8 (256 levels) this is 255 `greater` + 255 `select` = 510 ops. This is **untested** — the probe only confirmed 4-op chains (2 `greater` + 2 `select`). A dedicated probe is needed before attempting 256-level codebooks on ANE. For practical use with 2-bit or 4-bit codebooks (4 or 16 levels), the chain is 3-15 ops — well within proven limits.

### Files to modify
- `crates/ironmill-inference/src/ane/turboquant/mil_emitter.rs` — rewrite `build_cache_write_program()` quantization section
- `crates/ironmill-inference/src/ane/turboquant/model.rs` — use `src/turboquant/codebook.rs` (shared module) for level/boundary generation instead of `beta_quantizer`

### Validation
- Compile the rewritten cache-write program on ANE (probe test)
- Compare quantization error vs old beta-optimal (unit test with known input distribution)
- End-to-end perplexity test on a reference model

---

## 2. Chaining API for Pipelined Execution

**Risk:** Medium — API exists and constructs, but runtime behavior is undocumented
**Impact:** Very high — could eliminate all per-layer CPU roundtrips
**Compile budget cost:** 0 (same programs, different dispatch)

### Problem
The current ANE decode loop calls `eval()` once per layer with CPU orchestration between. For a 32-layer model, that's 32 CPU↔ANE roundtrips per token. Metal does the entire forward pass in a single command buffer.

### Hypothesis
`_ANEChainingRequest` with `loopbackInputSymbolIndex`/`loopbackOutputSymbolIndex` can feed one program's output directly as the next program's input without returning to CPU. Combined with `_ANESharedEvents` for completion signaling, this could pipeline all layer programs.

### Investigation steps

#### Step 1: Two-program chain
Compile two trivial MIL programs (e.g., `add(x, 1)` and `mul(x, 2)`). Chain them so program 1's output feeds program 2's input via loopback. Verify the result is `(x + 1) * 2`, not `x * 2` or `x + 1`.

```rust
// Pseudo-code for the experiment
let prog_a = compile("add(x, 1)");
let prog_b = compile("mul(x, 2)");

let chain = ChainingRequest::new(
    inputs: [input_surface],
    output_sets: [output_surface],
    lb_input_symbol_id: [0],   // prog_b's input 0
    lb_output_symbol_id: [0],  // prog_a's output 0
    procedure_index: 0,
    signal_events: None,
    ...
);

client.prepare_chaining(model, options, chain, qos)?;
// Then evaluate — does it work?
```

#### Step 2: Multi-program chain
If step 1 works, chain 4 programs to simulate a simplified layer pipeline:
1. RMSNorm (matmul + reduce + mul)
2. Linear projection (matmul)
3. Quantized cache write (round + clip + cast)
4. Attention (matmul + softmax + matmul)

#### Step 3: Full layer chain
If step 2 works, chain all subprograms for a complete transformer layer and compare latency vs the current per-eval approach.

### Files to modify
- `crates/ironmill-inference/src/ane/decode.rs` — add chained execution path alongside current per-layer eval
- `crates/ironmill-ane-sys/tests/ane_probe.rs` — add chaining experiment probes

### Validation
- Correctness: compare output of chained vs sequential execution
- Latency: measure wall-clock time for both paths
- Budget: verify chaining doesn't consume extra compile slots

---

## 3. Fuse MIL Programs

**Risk:** Low-Medium — the probe confirmed 16-op chains compile; larger fusions (FFN) may exceed tested limits and need their own probe
**Impact:** Medium — reduces per-eval dispatch overhead
**Compile budget cost:** Reduces budget usage (fewer programs per model)

### Problem
The current ANE backend compiles many small subprograms:
- Decode bundle: `pre_attn`, `fp16_attn`, `post_attn`
- TurboQuant: `cache_write_program`, `attention_program`, `qjl_program`

Note: some fusion already exists — the `cache_write_fused` flag indicates cache-write ops can be folded into `pre_attn`. Each `eval()` call has fixed overhead (~10-50μs per dispatch). Further fusing reduces dispatches.

### Fusion opportunities

| Current | Fused | Dispatch savings |
|---------|-------|-----------------|
| RMSNorm program + Q/K/V proj program | Single `pre_attn` program | 1 eval saved |
| Cache write K + Cache write V | Single cache write program | 1 eval saved |
| Output proj + residual add + RMSNorm | Single `post_attn` program | 1-2 evals saved |
| Gate proj + SiLU + Up proj + Down proj | Single FFN program | 2-3 evals saved |

Per layer savings: 5-7 fewer evals. For 32 layers: **160-224 fewer evals per token.**

### Constraints
- Fused programs have more I/O tensors → larger IOSurface allocations
- ANE may reject programs above a certain op count — probe confirmed 16 ops; FFN fusion will likely exceed this and needs a dedicated compile test
- Compile budget is consumed per-program — fewer programs is better
- Some fusion may already exist via `cache_write_fused` flag — check before duplicating

### Files to modify
- `crates/ironmill-inference/src/ane/turboquant/mil_emitter.rs` — add fused program builders
- `crates/ironmill-inference/src/ane/decode.rs` — use fused programs when available
- `crates/ironmill-core/src/ane/mil_text.rs` — may need to support larger program serialization

### Validation
- Correctness: compare fused output vs unfused
- Compile: verify fused programs don't exceed ANE's internal limits
- Latency: benchmark fused vs unfused decode

---

## 4. QoS Optimization

**Risk:** None — trivial configuration change
**Impact:** Low-medium — may reduce scheduling latency
**Compile budget cost:** 0

### Problem
ironmill hardcodes `ANE_QOS = 21` (default priority 5). The probe revealed:

| QoS | Value | Priority | Use case |
|-----|-------|----------|----------|
| Real-time | 0 | 2 (highest) | Latency-critical decode |
| User-interactive | 33 | 3 | Interactive UI-driven inference |
| User-initiated | 25 | 4 | User-triggered batch |
| **Default** | **21** | **5** | **Current ironmill setting** |
| Utility | 17 | 5 | Background processing (same priority as Default, different queue index) |
| Background | 9 | 6 (lowest) | Non-urgent |

Lower priority number = higher priority. For interactive inference, QoS 33 (user-interactive, priority 3) or QoS 0 (real-time, priority 2) would reduce ANE scheduling latency.

### Change
- Add a `qos` field to `TurboQuantConfig` or `AneConfig`
- Default to `QoSMapper::ane_user_interactive_task_qos()` (33) for decode
- Allow override to real-time (0) for latency-critical applications
- Use `QoSMapper::ane_utility_task_qos()` (17) for prefill/batch operations

### Files to modify
- `crates/ironmill-ane-sys/src/model.rs` — parameterize QoS in `compile_mil_text()`, `eval()`
- `crates/ironmill-inference/src/ane/turboquant/model.rs` — pass QoS through
- `crates/ironmill-inference/src/ane/decode.rs` — pass QoS from config

### Validation
- Benchmark decode latency at QoS 21 vs 33 vs 0
- Verify no ObjC exceptions from QoS changes

---

## 5. Performance Profiling via perf_stats_mask

**Risk:** Low — all 32 bits accepted, just undocumented
**Impact:** Enables data-driven optimization (meta-improvement)
**Compile budget cost:** 0

### Problem
No visibility into ANE hardware execution time. Metal has GPU timestamps via command buffer profiling. ANE has `perf_stats_mask` and `PerformanceStats` but the mask bit meanings are undocumented.

### Investigation plan

#### Step 1: Systematic bit probing
For each of the 32 mask bits, compile a model with that bit set, eval it, and read the resulting `PerformanceStats`. Record which bits produce non-zero `hw_execution_time` and `perf_counter_data`.

```rust
for bit in 0..32 {
    model.set_perf_stats_mask(1 << bit);
    eval(&model, &inputs, &outputs)?;
    let stats = model.perf_stats(); // read stats after eval
    eprintln!("bit {bit}: hw_time={}, counters={:?}", 
              stats.hw_execution_time(), stats.perf_counter_data());
}
```

#### Step 2: Build profiling wrapper
Once useful bits are identified, create a `profile_eval()` helper that wraps `eval()` with stats collection and returns structured timing data.

#### Step 3: Integrate with benchmarks
Add ANE profiling to the benchmark harness so we can compare ANE vs Metal per-layer timing.

### Files to modify
- `crates/ironmill-ane-sys/tests/ane_probe.rs` — add perf stats probes
- `crates/ironmill-ane-sys/src/model.rs` — add `eval_with_stats()` convenience
- `crates/ironmill-inference/src/ane/decode.rs` — optional profiling mode

### Validation
- Probe results document which bits produce useful data
- Profiling overhead measurement (stats-enabled vs disabled)

---

## 6. ANE↔GPU Shared Events

**Risk:** High — completely untested, may require entitlements
**Impact:** Very high — enables heterogeneous compute pipelining
**Compile budget cost:** 0

### Problem
ANE and Metal GPU are separate compute units that currently can't coordinate without CPU mediation. If `SharedSignalEvent`/`SharedWaitEvent` work with `IOSurfaceSharedEvent` (the same mechanism Metal uses for cross-queue sync), the CPU could be removed from the critical path entirely.

### Hypothesis
Metal's `MTLSharedEvent` and ANE's `_ANESharedSignalEvent` both reference `IOSurfaceSharedEvent`. If they share the same kernel-level event object, a Metal command buffer could signal an event that an ANE program waits on (or vice versa), enabling:

- GPU computes Q/K/V projections → signals event → ANE runs attention
- ANE writes KV cache → signals event → GPU reads for next layer
- True pipelined execution across both accelerators

### Investigation steps

#### Step 1: Create a shared IOSurfaceSharedEvent
Use Metal to create an `MTLSharedEvent`, extract its `IOSurfaceSharedEvent` handle, and pass it to `SharedSignalEvent::new()`. Does the constructor succeed?

#### Step 2: Signal from Metal, wait on ANE
Encode a Metal command that signals the shared event at value N. Create an ANE request with a `SharedWaitEvent` that waits for value N. Does the ANE eval block until the Metal signal fires?

#### Step 3: Bidirectional sync
If step 2 works, test the reverse: ANE signals, Metal waits.

### Files to modify
- `crates/ironmill-ane-sys/tests/ane_probe.rs` — shared event experiment
- `crates/ironmill-inference/src/ane/decode.rs` — hybrid ANE+GPU execution path (if viable)
- Possibly new `crates/ironmill-inference/src/hybrid/` module for cross-accelerator scheduling

### Validation
- Correctness: verify signal/wait semantics match expectations
- Latency: measure sync overhead vs CPU-mediated handoff
- Stability: stress-test with rapid signal/wait cycles

---

## 7. K/V Asymmetric Quantization

**Risk:** Low — separate MIL programs for K and V are known to compile
**Impact:** Medium — matches Metal's precision allocation
**Compile budget cost:** +1 program per layer (doubles cache-write programs)

### Problem
Metal uses (b−1)-bit K + QJL correction and b-bit V. ANE uses the same bit-width for both. This wastes bits on V (which doesn't benefit from QJL) and under-allocates bits for K (where QJL correction helps).

### Change
Emit separate cache-write MIL programs for K and V:
- **K program:** (b−1)-bit codebook quantization + QJL sign storage
- **V program:** b-bit codebook quantization, no QJL

### Compile budget impact
Currently: 1 cache-write program per layer → L programs total
After: 2 cache-write programs per layer → 2L programs total

For a 32-layer model: 32 → 64 cache-write programs. With the 119 budget and other programs (attention, pre_attn, post_attn, lm_head chunks), this is tight. May require the `patch_weights()` optimization (reuses donor `net.plist`, doesn't consume budget) for all-but-first-layer K programs.

### Files to modify
- `crates/ironmill-inference/src/ane/turboquant/mil_emitter.rs` — separate K/V cache-write builders
- `crates/ironmill-inference/src/ane/turboquant/model.rs` — manage K and V programs independently
- `crates/ironmill-inference/src/ane/decode.rs` — call K and V cache-writes separately

### Validation
- Verify K and V programs compile within budget
- Compare perplexity with symmetric vs asymmetric quantization
- Measure cache memory savings (V uses fewer bits → smaller V cache)

---

## Dependency Graph

```
                    ┌──────────────┐
                    │ 4. QoS Tuning│ (independent, trivial)
                    └──────────────┘

┌──────────────────┐    ┌─────────────────────┐
│ 1. Unify Codebook│───▶│ 7. K/V Asymmetric   │
│    Math          │    │    Quantization      │
└──────────────────┘    └─────────────────────┘

┌──────────────────┐    ┌─────────────────────┐
│ 5. Perf Stats    │───▶│ Benchmarking all     │
│    Probing       │    │ improvements         │
└──────────────────┘    └─────────────────────┘

┌──────────────────┐
│ 3. Fuse MIL      │ (independent)
│    Programs      │
└──────────────────┘

┌──────────────────┐    ┌─────────────────────┐
│ 2. Chaining API  │───▶│ 6. ANE↔GPU Shared   │
│    Investigation │    │    Events            │
└──────────────────┘    └─────────────────────┘
```

Items 1, 3, 4, and 5 can proceed in parallel.
Item 2 must precede item 6.
Item 1 must precede item 7.

---

## Implementation Order

| Phase | Item | Effort | Risk |
|-------|------|--------|------|
| A | 4. QoS Tuning | Small | None |
| A | 1. Unify Codebook Math | Medium | Low |
| A | 5. Perf Stats Probing | Small | Low |
| B | 3. Fuse MIL Programs | Medium | Low |
| B | 2. Chaining API Investigation | Medium | Medium |
| C | 7. K/V Asymmetric Quantization | Medium | Low |
| C | 6. ANE↔GPU Shared Events | Large | High |

Phase A items are independent and can be parallelized.
Phase B items are independent of each other but benefit from Phase A profiling data.
Phase C items depend on Phases A/B results.
