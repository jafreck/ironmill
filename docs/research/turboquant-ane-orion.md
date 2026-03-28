# TurboQuant on ANE via Orion — Feasibility Analysis

> **Prerequisite reading:** [TurboQuant Research Analysis](turboquant-analysis.md)
>
> **Prerequisite implementation:** [ANE Direct Runtime Backend](../ane-direct-runtime-plan.md)
> — the ANE runtime must be built before this plan can be executed.

## Motivation

TurboQuant's runtime KV cache compression is blocked in ironmill today because
CoreML's `MLModel` API is a black box — ironmill cannot intercept the inference
loop, manage KV cache memory, or insert quantize/dequantize steps between
attention stages.

[Orion](https://github.com/mechramc/Orion) removes this barrier. By reverse-
engineering Apple's private ANE APIs (`_ANEClient`, `_ANECompiler`), Orion
exposes direct control over program compilation, execution, and device memory
via IOSurface-backed zero-copy tensors. This gives ironmill the missing
ingredient: **ownership of the inference loop**.

This document analyzes whether TurboQuant's full pipeline (PolarQuant + QJL)
can run **entirely on ANE** without CPU/GPU fallback, and what ironmill would
need to make that work.

---

## Can TurboQuant run entirely on ANE?

### ANE's native op set

ironmill's ANE validator (`validate.rs`) recognizes these ops as ANE-compatible:

```
conv, matmul, linear, relu, sigmoid, tanh, softmax, batch_norm,
add, mul, sub, real_div, concat, reshape, transpose, max_pool,
avg_pool, reduce_mean, layer_norm, squeeze, expand_dims, pad,
cast, const, clip, gather, split, slice_by_index, pow, sqrt,
select, scaled_dot_product_attention
```

ANE-supported data types per ironmill's validator: `Float16`, `Float32`, `Int8`.
**Note:** This list reflects CoreML's public API restrictions, not ANE hardware
capabilities. See the data type section below for the full picture.

### TurboQuant op decomposition

Every TurboQuant operation can be expressed using ANE-native ops:

| TurboQuant step | ANE decomposition | Ops used |
|----------------|-------------------|----------|
| **Hadamard rotation** | Matrix multiply with precomputed R | `matmul` or `conv` (1×1) |
| **Row norm extraction** | Normalize rows before quantization | `mul`, `sqrt`, `reduce_mean`, `real_div` |
| **Scalar quantize** (float → int) | Scale + offset + clamp + cast | `mul` → `add` → `clip` → `cast` |
| **Scalar dequantize** (int → float) | Cast + scale + offset | `cast` → `mul` → `add` |
| **QJL sign extraction** | Threshold at zero → ±1 | `select` (x > 0 → +1, else -1) |
| **QJL inner product correction** | Dot product of sign vectors | `mul` → `reduce_mean` → `mul` |
| **LUT dequantize** | Index into 2^b reconstruction levels | `gather` (from 16-entry const table) |
| **KV cache slice** | Read a range from the cache tensor | `slice_by_index` |

**Result: no op-level blockers.** Every step maps to native ANE operations.

### The data type situation: CoreML limitation, not hardware

ironmill's ANE validator (`is_ane_dtype`) lists only `Float16`, `Float32`, and
`Int8`. This reflects **CoreML's public API restrictions**, not ANE hardware
capabilities.

Reverse engineering by the Orion project and related efforts (maderix/ANE,
mdaiter/ane) has confirmed that **ANE hardware natively supports INT4/UINT4**:

- Custom MIL programs can define tensors with 4-bit data types via bitfield
  manipulation of weight descriptors, bypassing CoreML's compilation step.
- ANE executes 4-bit convolutions and matmuls directly, achieving **nearly 2×
  throughput compared to INT8**.
- ANE upconverts INT4 to FP16 internally for arithmetic, but storage and
  memory transfer use true 4-bit — this is where the bandwidth win comes from.
- The main constraint is **memory alignment**: 4-bit tensors require specific
  alignment for ANE's DMA engines.

Apple hides these modes in the public API for stability and battery-life
predictability. Orion bypasses this entirely.

**Impact on TurboQuant:** No Int8-container workaround is needed. With Orion,
the KV cache can be stored and computed as **native 4-bit** on ANE, getting
the full 4× memory bandwidth reduction over FP16 and 2× throughput over INT8
without any pack/unpack overhead.

---

## Storage strategy: native 4-bit on ANE via Orion

With Orion exposing ANE's native INT4 support, the storage strategy simplifies
dramatically. No pack/unpack step is needed — ANE handles 4-bit tensors
directly.

```
┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE STEP t                         │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────┐  │
│  │ Attention │    │ Rotate + │    │ Store to KV cache    │  │
│  │ projection│───▶│ quantize │───▶│ (INT4 native on ANE) │  │
│  │ (ANE)     │    │ (ANE)    │    │                      │  │
│  └──────────┘    └──────────┘    └──────────┬───────────┘  │
│                                              │              │
│                                   ┌──────────▼──────────┐   │
│                                   │ 4-bit KV cache      │   │
│                                   │ (unified memory,    │   │
│                                   │  ANE-aligned)       │   │
│                                   └──────────┬──────────┘   │
│                                              │              │
│  ┌──────────┐    ┌──────────┐    ┌───────────▼──────────┐  │
│  │ Attention │◀───│ Un-rotate│◀───│ Dequantize           │  │
│  │ scores   │    │ (ANE)    │    │ (ANE, INT4→FP16)     │  │
│  └──────────┘    └──────────┘    └──────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

ANE handles the INT4↔FP16 conversion internally — it stores in 4-bit and
upconverts to FP16 for arithmetic automatically. This means:

- **No pack/unpack programs** needed (eliminates Risk R3 entirely)
- **No ANE↔CPU round-trips** for format conversion
- **True 4× bandwidth reduction** vs FP16, not just a storage trick

### Where the memory win happens

| Cache format | Bytes per element | 7B model, seq=4096 | Throughput vs FP16 |
|-------------|-------------------|--------------------|--------------------|
| Float16 | 2.0 | ~2.0 GB | 1× (baseline) |
| Int8 | 1.0 | ~1.0 GB | ~2× |
| INT4 native | 0.5 | ~0.5 GB | ~4× (2× over Int8) |
| 3-bit packed | 0.375 | ~0.4 GB | ~5× |

With native INT4 on ANE, both the memory savings and the compute throughput
improvement are real — ANE's internal datapath processes 4-bit values with
nearly 2× the throughput of INT8.

---

## ANE program structure

With Orion, each phase of inference is compiled as a separate ANE program.
TurboQuant splits attention into three programs:

### Program 1 — Cache write (runs once per new token)

```
Inputs:  K_proj [batch, heads, 1, head_dim]  (Float16, from projection)
         V_proj [batch, heads, 1, head_dim]  (Float16, from projection)
         R      [head_dim, head_dim]          (Float16, const rotation matrix)

Ops:     matmul(K_proj, R^T)                  → K_rotated   (Float16)
         mul + add + clip + cast              → K_quant     (INT4, 16 levels)
         matmul(V_proj, R^T)                  → V_rotated   (Float16)
         mul + add + clip + cast              → V_quant     (INT4, 16 levels)

Outputs: K_quant, V_quant  (INT4)
```

### Program 2 — Cache read + attention (runs once per new token)

```
Inputs:  Q         [batch, heads, 1, head_dim]     (Float16)
         K_cache   [batch, heads, seq_len, head_dim] (INT4, native)
         V_cache   [batch, heads, seq_len, head_dim] (INT4, native)
         LUT       [16]                               (Float16, const levels)
         R         [head_dim, head_dim]               (Float16, const rotation)

Ops:     gather(LUT, K_cache)                   → K_dequant  (Float16)
         matmul(K_dequant, R)                   → K_approx   (Float16)
         gather(LUT, V_cache)                   → V_dequant  (Float16)
         matmul(V_dequant, R)                   → V_approx   (Float16)
         scaled_dot_product_attention(Q, K, V)  → attn_out   (Float16)

Outputs: attn_out (Float16)
```

### Program 3 — QJL correction (optional, runs once per new token)

```
Inputs:  residual_signs  [batch, heads, seq_len, head_dim]  (Int8, ±1)
         Q               [batch, heads, 1, head_dim]         (Float16)

Ops:     cast(Q → sign)                        → Q_sign     (Int8)
         matmul(Q_sign, residual_signs^T)       → correction (Float16)
         mul(correction, scale_factor)          → scaled     (Float16)

Outputs: scaled  (Float16, added to attention logits)
```

Programs 1 and 2 are the core path. Program 3 (QJL) adds 1 bit of overhead
for unbiased attention scores and can be omitted if slight bias is acceptable.

### Program compilation budget

Orion discovered ANE caps at ~119 program compilations per process. A typical
transformer layer needs ~3–5 programs (FFN, attention variants, cache ops).
With TurboQuant adding 2–3 programs per layer:

- 32-layer model × 5 programs = 160 — **over budget**
- Mitigation: share programs across layers (same architecture = same compiled
  program, different weight pointers). This reduces to ~5–8 unique programs
  total, well within budget.

---

## Performance analysis

### Where time is spent in autoregressive attention

During token generation (decode phase), attention is **memory-bandwidth bound**:
the model reads the entire KV cache for every new token. On Apple Silicon:

| Chip | Memory bandwidth | KV read time (7B, seq=4096, FP16) |
|------|-----------------|-----------------------------------|
| M2 | 100 GB/s | ~20 ms |
| M3 Pro | 150 GB/s | ~13 ms |
| M4 Pro | 273 GB/s | ~7 ms |

With 4-bit TurboQuant, the KV cache is 4× smaller. Read times scale linearly:

| Chip | FP16 KV read | 4-bit KV read | Speedup |
|------|-------------|---------------|---------|
| M2 | ~20 ms | ~5 ms | 4× |
| M3 Pro | ~13 ms | ~3.3 ms | 4× |
| M4 Pro | ~7 ms | ~1.8 ms | 4× |

### Overhead from TurboQuant ops

The rotation matmul is the dominant overhead:

```
head_dim = 128
matmul(K, R^T): 128 × 128 = 16,384 FLOPs per head per token
```

At 32 heads × 2 (K+V) × 2 (rotate + un-rotate) = 128 matmuls per token:

```
128 × 16,384 = ~2M FLOPs per token
```

ANE delivers ~11 TFLOPS FP16 (M4). Time: **~0.2 μs per token** — negligible
compared to the millisecond-scale attention read time.

### Net effect

The bandwidth savings (milliseconds) dwarf the compute overhead (microseconds).
TurboQuant on ANE is a clear net win for autoregressive decoding, especially at
long context lengths where KV cache size dominates.

---

## What ironmill needs to build

### Phase 1 — Orion backend integration

| Component | Description |
|-----------|-------------|
| `Backend::Orion` | New compiler backend alongside `Xcrun` and `AneDirect`. Compiles MIL IR sub-programs to ANE-native binaries via Orion's compiler API. |
| `OrionRuntime` | Rust wrapper around Orion's execution API. Loads compiled programs, manages IOSurface tensors, executes inference steps. |
| `KvCacheManager` | Manages the packed 4-bit KV cache in IOSurface memory. Handles pack/unpack, cache growth, and eviction. |
| Program splitter | Given a full model graph, split into sub-programs for ANE execution (FFN, attention-write, attention-read, etc.). |

### Phase 2 — TurboQuant runtime pass

| Component | Description |
|-----------|-------------|
| `TurboQuantRuntimePass` | Transform the attention sub-graph: insert rotation matmuls, quantize/dequantize chains, LUT consts, and QJL correction. Emits the three ANE programs described above. |
| Rotation matrix generation | Reuse from the static PolarQuant pass (`rotation.rs`). Generate seeded Hadamard matrix, embed as Float16 const. |
| Beta-optimal LUT | Reuse from static pass (`beta_quantizer.rs`). Precomputed 16-entry LUT for 4-bit, 8-entry for 3-bit. |
| Cache format codec | Pack/unpack between Int8 (ANE) and 4-bit packed (memory). Runs as a small ANE program or via NEON on host. |

### Phase 3 — End-to-end pipeline

| Component | Description |
|-----------|-------------|
| CLI command | `ironmill run --runtime orion --kv-quant turbo-4 model.onnx` |
| Benchmark integration | Extend `ironmill-bench` to compare CoreML vs Orion+TurboQuant across context lengths. |
| Correctness validation | Compare attention outputs between FP16 baseline and TurboQuant at various bit-widths. Measure perplexity delta on standardized prompts. |

---

## Risks

### R1 — Orion stability and API surface

Orion is a research project using private Apple APIs. These APIs can change
between macOS versions without notice. ironmill would take a dependency on an
unstable, reverse-engineered interface.

**Mitigation:** Isolate behind a feature flag (`--features orion-runtime`).
Keep CoreML as the default backend. Pin to specific macOS versions in CI.

### R2 — ANE program compilation limit (~119 per process)

Exceeding the limit crashes the ANE runtime.

**Mitigation:** Share compiled programs across layers (same architecture =
same binary, different weight pointers via IOSurface rebinding). A 32-layer
model should need ~5–8 unique programs, not 32×5.

### R3 — INT4 memory alignment requirements

ANE's DMA engines require specific memory alignment for 4-bit tensors. Tensor
shapes and strides must be configured to match ANE's internal tiling. Orion's
reverse engineering has documented some of these constraints, but not all
combinations have been tested.

**Mitigation:** Follow Orion's documented alignment rules for 4-bit tensors.
Add alignment validation to ironmill's ANE program emitter. Fall back to Int8
containers if a specific tensor shape doesn't meet alignment requirements.

### R4 — Correctness of attention with quantized K/V

Quantization introduces approximation error in attention scores. While
TurboQuant's theory guarantees near-optimal distortion, practical impact on
generation quality depends on the model.

**Mitigation:** Validate on standard benchmarks (perplexity, LongBench) before
shipping. Offer 4-bit (safe) and 3-bit (aggressive) presets. Always allow
fallback to FP16.

### R5 — Scope of effort

This is not a pass — it's a new runtime backend. The Orion integration alone
is a substantial project before TurboQuant enters the picture.

**Mitigation:** Phase the work. Phase 1 (Orion backend) is independently
useful for inference benchmarking. Phase 2 (TurboQuant) builds on it. Phase 1
can ship without Phase 2.

---

## Comparison: TurboQuant paths with and without Orion

| Capability | Static pass (Path A) | Orion + TurboQuant |
|-----------|---------------------|--------------------|
| Weight compression | ✅ 2–4 bit | ✅ 2–4 bit |
| KV cache compression | ❌ | ✅ 2–4 bit |
| Runtime overhead | Zero (materialized at load) | Minimal (~0.2 μs/token for rotation) |
| Memory savings at inference | Weights only | Weights + KV cache |
| Max context length increase | None | 4× at 4-bit |
| Requires Orion | No | Yes |
| Requires custom runtime | No | Yes |
| CoreML compatible | Yes | No (Orion only) |
| Implementation effort | Medium | High |

**Recommendation:** Build both. The static PolarQuant pass (Path A) ships
first with zero runtime dependencies. The Orion+TurboQuant runtime ships
later for users who need long-context inference on constrained devices.

---

## References

1. Orion: [github.com/mechramc/Orion](https://github.com/mechramc/Orion)
2. Orion paper: [arXiv:2603.06728](https://arxiv.org/abs/2603.06728)
3. TurboQuant: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
4. PolarQuant: [arXiv:2502.02617](https://arxiv.org/abs/2502.02617)
5. ANE reverse engineering: [NYU Shanghai writeup](https://rits.shanghai.nyu.edu/ai/reverse-engineering-apples-neural-engine-to-train-transformers-on-m4/)
