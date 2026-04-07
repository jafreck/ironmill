# Metal Dispatch Overhead Reduction Plan

## Problem

Qwen 3.5-4B INT4+TQ-INT8 decode on M2 Max 64 GB runs at **20.9 tok/s
(47.9 ms/tok)**. llama.cpp achieves **90 tok/s (11.1 ms/tok)** on the same
hardware with a comparable model (Qwen3-4B Q4_K_M). The 4.3× gap breaks
down as:

| Source | Estimated cost | Notes |
|--------|---------------|-------|
| Excess memory barriers | ~20–25 ms | 77 full-scope barriers vs ~30 needed |
| INT4 affine dequant format | ~5–8 ms | 3 buffer reads + float math vs Q4_K inline scales + int dot |
| Dispatch encoding overhead | ~3–5 ms | ObjC selector lookups, buffer binding churn |
| Bandwidth floor (irreducible) | ~6.6 ms | 2.63 GB @ 400 GB/s |
| **Total** | **~37–45 ms** | measured: 41.3 ms overhead above bandwidth floor |

The bandwidth floor for both systems is similar (~6–7 ms). The entire gap
is overhead.

## Root Cause Analysis

### 1. Memory barrier abuse (primary: ~20–25 ms)

Every logical step in the decode pipeline ends with
`enc.memory_barrier_buffers()`, which calls:

```objc
[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
```

This is a **full GPU cache flush** — it forces all pending writes across
ALL buffers to become visible before any subsequent dispatch can read ANY
buffer. On M2 Max, each costs ~0.2–0.5 ms of GPU stall time.

The current decode path has **77 barriers for 180 dispatches** (0.43
barriers per dispatch). Many are between dispatches with no data dependency:

- Q, K, V projections write to `q_proj`, `k_proj`, `v_proj` (independent
  buffers) — barrier between them is unnecessary
- FFN block → MoE block (MoE reads different inputs)
- Post-FFN norm → copy_buffer (sequential but different buffers)

llama.cpp's ggml scheduler inserts barriers only at computed data dependency
edges (~30 barriers for ~250 dispatches = 0.12 per dispatch).

### 2. INT4 affine quantization format (secondary: ~5–8 ms)

The current INT4 dequant inner loop does 12 FLOPs per 2 elements:

```metal
// Current: 3 buffer reads + 6 float ops per element pair
uchar packed = B_packed[byte_idx];       // read 1: weight data
float s0 = float(scales[scale_row+g0]);  // read 2: scale buffer
float z0 = float(zeros[scale_row+g0]);   // read 3: zero buffer
float w0 = (float(packed & 0x0F) - z0) * s0;  // 3 float ops
acc += float(A[k2]) * w0;               // 2 float ops (read input + FMA)
```

llama.cpp's Q4_K format stores scales inline with the data (one contiguous
super-block per 256 elements) and pre-quantizes the input vector to Q8,
enabling integer dot products:

```metal
// Q4_K: 1 buffer read + integer math
int4 w = unpack_q4(block.data[k]);      // read 1: data+scales together
int8 x = quantized_input[k];            // read from threadgroup mem
acc += int(w) * int(x);                 // integer multiply-add
```

### 3. ObjC dispatch overhead (minor: ~1–2 ms)

Every Metal API call (`set_buffer`, `set_pipeline`, `dispatch_threadgroups`,
`memory_barrier_buffers`) performs a fresh `sel_registerName` lookup. With
~10 calls per dispatch × 180 dispatches = ~1800 selector lookups per decode
step. Each lookup is ~50–100 ns, totaling ~0.1–0.2 ms. Small but fixable.

## Implementation Plan

### P0: Barrier audit and removal (target: −20 ms, ~40 tok/s)

**Approach:** Audit every `memory_barrier_buffers()` call in the decode
path. Categorize each as REQUIRED, REMOVABLE, or REPLACEABLE. Remove
unnecessary barriers and switch required ones to resource-specific barriers.

**Step 1: Add `memory_barrier_with_resources` to metal-sys**

```rust
// crates/ironmill-metal-sys/src/command.rs
pub fn memory_barrier_with_resources(&self, buffers: &[&MetalBuffer]) {
    // [encoder memoryBarrierWithResources:resources count:count]
    let ptrs: Vec<*mut c_void> = buffers.iter()
        .map(|b| b.as_raw_ptr())
        .collect();
    let sel = sel!("memoryBarrierWithResources:count:");
    // ... objc_msgSend
}
```

**Step 2: Remove barriers between independent dispatches**

These barriers have no data dependency and can be deleted outright:

| Location | Before | After | Why safe |
|----------|--------|-------|----------|
| Post-embedding (line ~364) | fused_embedding_norm | PLE embed | Different output buffers |
| Post-V-norm (line ~524) | v_norm on v_proj | qk_norm_rope on q_proj,k_proj | Different buffers |
| Post-FFN (line ~706) | FFN down_proj | MoE block | MoE reads different inputs |
| Post-FFN-norm (line ~739) | post_ffn_norm | copy_buffer | Sequential on same buffer — keep |

**Step 3: Replace full-scope barriers with resource-specific barriers**

For remaining required barriers, switch from `memoryBarrierWithScope:` (all
buffers) to `memoryBarrierWithResources:` (specific buffers):

```rust
// Before: flushes ALL buffer caches
enc.memory_barrier_buffers();

// After: only flushes the specific buffer that was just written
enc.memory_barrier_with_resources(&[&bufs.norm_out]);
```

Key replacements in the per-layer decode path:

| After dispatch | Barrier resource(s) |
|----------------|-------------------|
| Q/K/V projections | `q_proj, k_proj, v_proj` |
| QK-norm + RoPE | `q_proj, k_proj` (in-place) |
| Attention compute | `attn_out` |
| O projection | `ffn_down` |
| Fused residual+norm | `hidden_state, norm_out` |
| FFN fused gate+up+act | `ffn_gate` |
| FFN down-proj | `ffn_down` |
| End-of-layer P1 | `hidden_state, norm_out, [proj_output]` |

**Expected impact:** Removing ~20 unnecessary barriers and narrowing ~50
remaining barriers should reduce GPU stall time by ~20 ms.

### P1: Q8 input quantization (target: −5 ms, ~50 tok/s)

**Approach:** Before each INT4 matvec dispatch, quantize the FP16 input
vector to INT8 with a per-group scale factor. The inner loop then uses
integer multiply-add instead of float dequant.

**Step 1: CPU-side Q8 quantization kernel**

A lightweight Metal kernel that reads `norm_out` (FP16, [1, K]) and writes:
- `q8_data`: [K] int8 values (quantized input)
- `q8_scales`: [K/group_size] float scales

```metal
kernel void quantize_input_q8(
    device const half *input   [[buffer(0)]],
    device char *q8_data       [[buffer(1)]],
    device float *q8_scales    [[buffer(2)]],
    constant uint &K           [[buffer(3)]],
    constant uint &group_size  [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint group = tid;
    uint start = group * group_size;
    // Find max abs in group
    float max_abs = 0;
    for (uint i = start; i < start + group_size && i < K; i++)
        max_abs = fmax(max_abs, fabs(float(input[i])));
    float scale = max_abs / 127.0f;
    q8_scales[group] = scale;
    float inv_scale = (scale > 0) ? 127.0f / max_abs : 0;
    for (uint i = start; i < start + group_size && i < K; i++)
        q8_data[i] = char(round(float(input[i]) * inv_scale));
}
```

**Step 2: INT4×Q8 integer dot product kernel**

```metal
kernel void affine_matvec_int4xq8(
    device const char *A_q8         [[buffer(0)]],   // [K] int8
    device const float *A_scales    [[buffer(1)]],   // [K/gs] float
    device const uchar *B_packed    [[buffer(2)]],   // INT4 weights
    device const half *w_scales     [[buffer(3)]],   // weight scales
    device const half *w_zeros      [[buffer(4)]],   // weight zeros
    device half *C                  [[buffer(5)]],   // output
    ...)
{
    int acc = 0;
    for (uint k = lane; k < half_K; k += 32) {
        uchar packed = B_packed[...];
        int w0 = int(packed & 0x0F) - int_zero;  // integer subtract
        int w1 = int(packed >> 4)   - int_zero;
        acc += int(A_q8[k*2])   * w0;  // integer multiply
        acc += int(A_q8[k*2+1]) * w1;
    }
    // Final: float(acc) * w_scale * a_scale
    acc_sum = simd_sum(acc);
    C[tid] = half(float(acc_sum) * combined_scale);
}
```

This eliminates the `float(nibble) * float(scale)` chain and replaces it
with integer `int8 × int4 → int32`. Apple GPU integer ALUs are as fast
as float ALUs, but the reduced operation count (no float conversion, no
separate zero subtraction per scale group) cuts per-element cost by ~2×.

**Amortization:** The Q8 quantization dispatch runs once per norm_out
vector. The cost is shared across all projections that read norm_out (Q, K,
V, gate, up for standard layers; QKV/Z/A/B for GDN layers). Net overhead
per layer: 1 lightweight dispatch for Q8 quantization, saving ~6× float
ops across ~4–7 projection dispatches.

**Expected impact:** ~2× faster inner loop → ~5 ms saved.

### P2: Selector caching (target: −0.5 ms)

Cache all ObjC selectors at init time instead of calling
`sel_registerName` on every Metal API call:

```rust
struct CachedSelectors {
    set_pipeline: *mut c_void,
    set_buffer: *mut c_void,
    set_bytes: *mut c_void,
    dispatch_threadgroups: *mut c_void,
    memory_barrier_scope: *mut c_void,
    memory_barrier_resources: *mut c_void,
}

lazy_static! {
    static ref SELS: CachedSelectors = CachedSelectors {
        set_pipeline: unsafe { sel_registerName(sel!("setComputePipelineState:")) },
        // ...
    };
}
```

**Expected impact:** ~0.5 ms (minor, but trivial to implement).

## Expected Results

| Optimization | ms saved | Cumulative tok/s |
|-------------|----------|-----------------|
| Current baseline | — | 20.9 |
| P0: Barrier audit | ~20 ms | ~36 |
| P1: Q8 input quant | ~5 ms | ~44 |
| P2: Selector caching | ~0.5 ms | ~45 |
| **Total** | **~25 ms** | **~45 tok/s** |

Remaining gap to llama.cpp (90 tok/s) after P0–P2 would be ~2×, attributable
to:
- Q4_K_M super-block format vs INT4 affine (inherent format difference)
- llama.cpp kernel maturity (years of community optimization)
- Qwen3.5 GDN layer overhead (not present in Qwen3)

## Non-goals

- Changing the INT4 affine quantization format to Q4_K_M (would require
  rewriting the entire weight loading + matmul pipeline)
- Speculative decoding (orthogonal optimization, covered in
  `qwen35-inference-perf.md` as P4)
- Prefill optimization (this document focuses on decode throughput)

## Validation

After each phase, verify:
1. PPL on wikitext2-qwen35 is unchanged (9.34 ± 0.05 for INT4+TQ-INT8)
2. Decode throughput measured with 20 iterations, 5 warmup, 3 runs
3. No GPU command buffer errors (`cmd_buf.status() != Error`)

```bash
cargo test --release -p ironmill-bench --features metal --test qwen35_bench \
  -- qwen35_int4_tq_int8_ppl_regression --ignored --nocapture

cargo run --release -p ironmill-bench --features metal -- \
  --config configs/qwen35-bench.toml -b metal \
  -i 20 -w 5 -r 3
```

## Hardware

Apple M2 Max, 64 GB unified memory, macOS 15.x.
Metal GPU Family: Apple 8 (supports `memoryBarrierWithResources:count:`).
