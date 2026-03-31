# ANE Op Discovery–Driven Optimizations

Concrete optimizations unlocked by ironmill's 52 novel ANE op discoveries.
These capabilities are unique to ironmill — no other open-source project
(maderix/ANE, Orion, Espresso, ANEgpt, hollance/neural-engine) has verified
the ops that enable them.

See `docs/design/ane-op-support-matrix.md` for the full cross-reference.

---

## 1. Full-ANE Transformer Layer (Zero CPU Normalization)

**Enabled by:** `layer_norm` ✅ eval · `rsqrt(epsilon)` ✅ eval · `erf` ✅ eval

### Problem

Every ANE project bounces data to CPU for LayerNorm/RMSNorm between
transformer layers. This ANE→CPU→ANE roundtrip dominates decode latency:

```
ANE: QKV + attention + FFN
          ↕ CPU: LayerNorm       ← data copy overhead
ANE: QKV + attention + FFN
          ↕ CPU: LayerNorm       ← data copy overhead
...
```

Espresso works around this with "fused 3-layer kernels" — cramming 3 layers
into 1 ANE dispatch to reduce the number of CPU roundtrips. maderix and Orion
run normalization entirely on CPU via vDSP.

### What ironmill can do

ironmill verified that `layer_norm` runs directly on ANE (max_err=0.001).
Combined with `rsqrt(epsilon)` for RMSNorm, the full transformer block stays
on ANE:

```
ANE: embed → [LayerNorm → QKV → attention → LayerNorm → FFN] × N → head
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     Zero CPU compute ops in the loop
```

### Implementation

- Update ANE MIL emitter (`turboquant_mil.rs`) to emit `layer_norm` and
  `rsqrt(x, epsilon=...)` instead of CPU fallback or `pow(x, -0.5)`
  decomposition
- Update attention split to keep normalization in pre_attn/post_attn
  subprograms rather than extracting it to CPU
- Target: full transformer layer with zero CPU compute in the critical path

### Expected impact

Eliminating CPU normalization removes the primary latency bottleneck in
every competing ANE project. The CPU roundtrip involves IOSurface reads,
fp16→fp32 conversion, vDSP computation, fp32→fp16 conversion, and IOSurface
writes — typically 0.5-1.0ms per layer on M4.

---

## 2. Native GELU via `erf`

**Enabled by:** `erf` ✅ eval (max_err=0.001)

### Problem

GELU is the dominant activation in modern transformers (GPT, Llama, etc.).
Without `erf`, it must be decomposed to the tanh approximation:

```
gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
```

This requires 7+ MIL ops: `pow`, `mul` (×3), `add` (×2), `tanh`.

### What ironmill can do

With `erf`, GELU is exact in 5 ops:

```
gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

Ops: `mul`, `erf`, `add`, `mul`, `mul` — and the ANE's `erf` has 10× better
precision than the tanh approximation (0.001 vs 0.01 max error).

### Implementation

- Add `GELUExactFusionPass` for ANE-direct backend that pattern-matches the
  tanh decomposition and replaces it with the `erf`-based version
- Or emit `erf`-based GELU directly in `OpSubstitutionPass` when targeting
  ANE-direct

---

## 3. INT8 Activation Quantization Pipeline

**Enabled by:** `quantize` ✅ eval · `dequantize` ✅ eval · `cast fp16↔int8`
✅ eval · `round` ✅ eval · `clip` ✅ eval

### Problem

ANE SRAM bandwidth is the throughput bottleneck for large models. FP16
activations between layers consume 2 bytes per element through L2 SRAM.

### What ironmill can do

maderix demonstrated 1.88× throughput via INT8 W8A8 activation caching on
M4. ironmill has now independently verified the full
`quantize → dequantize` pipeline works on ANE, including in multi-op contexts
(add→quantize→dequantize passes with max_err=0.0).

Insert quantize/dequantize pairs between layers to halve SRAM bandwidth:

```
Layer N output (fp16) → quantize (int8) → [stored in SRAM at 1 byte/elem]
                                         → dequantize (fp16) → Layer N+1 input
```

### Implementation

- `ActivationQuantizationPass`: insert `quantize`/`dequantize` pairs between
  transformer layers in the ANE-direct emitter
- Extend TurboQuant from KV-cache-only to full activation INT8 caching
- CLI flag: `--activation-quant int8`
- Quality gate: measure perplexity impact from INT8 activation rounding

### Expected impact

1.5-2× throughput for memory-bandwidth-bound models on ANE, based on
maderix's empirical results (18.6 TOPS FP16 → 35.1 TOPS INT8 W8A8).

---

## 4. On-ANE Causal Attention Masking

**Enabled by:** `greater` ✅ · `greater_equal` ✅ · `less` ✅ · `select` ✅ ·
`cast fp16↔bool` ✅ · `logical_not` ✅

### Problem

ANE's `scaled_dot_product_attention` ignores the causal mask parameter
(Orion constraint #6, maderix confirmed). Every project decomposes attention
manually and runs masking on CPU:

```
ANE: Q @ K^T
  ↕ CPU: apply causal mask + softmax     ← roundtrip
ANE: scores @ V
```

### What ironmill can do

ironmill verified that comparison ops (`greater`, `less`, `select`) and
boolean casts all work on ANE. The full masking sequence can run on-device:

```
ANE: Q @ K^T → greater(positions, positions_T) → select(mask, scores, -inf)
           → softmax → @ V
```

### Implementation

- Update attention split MIL emitter to emit comparison + select ops for
  causal masking instead of falling back to CPU
- Eliminates the ANE→CPU→ANE roundtrip that all other projects are forced into

Note: the `status=0x1d` error documented in
`ane-attention-split-investigation.md` was caused by the ANE minimum I/O
tensor size constraint (C > ~768, S < 32), not by masking. Causal masking
is a separate concern — it enables keeping the full attention block on
ANE rather than decomposing attention to do masking on CPU.

### Expected impact

Removes the single biggest reason projects decompose attention manually.
If the full attention block stays on ANE, the CPU roundtrip for masking
(~0.2-0.5ms per layer) is eliminated.

---

## 5. `log` for On-ANE Loss & Log-Softmax

**Enabled by:** `log(epsilon)` ✅ eval (max_err=0.005)

### Discovery

No project in the open-source ANE ecosystem knew logarithm was possible on
ANE. Espresso doesn't list it. Orion doesn't have it. maderix never tested
it. The key was the mandatory `epsilon` parameter — without it, the ANE
compiler rejects `log`; with it, `log` compiles and produces correct results.

### What this unlocks

- **Log-softmax:** `log(softmax(x))` entirely on ANE — used in loss
  computation during training and in some sampling strategies
- **Cross-entropy loss:** `sum(-target * log(pred))` on ANE
- **Entropy / KL divergence:** information-theoretic computations on-device
- **Probabilistic sampling:** log-probability computations without CPU

### Implementation

- Add `log(x, epsilon=...)` to the ANE-direct MIL emitter op vocabulary
- Use in training backward pass (benefits maderix/Orion/Espresso training
  loops if they adopt the epsilon pattern)
- Emit log-softmax as `log(softmax(x), epsilon)` instead of decomposing

---

## 6. Single-Op Normalization via `reduce_l2_norm`

**Enabled by:** `reduce_l2_norm` ✅ compile · `reduce_log_sum_exp` ✅ eval

### What this simplifies

| Pattern | Before (multi-op) | After (single-op) |
|---|---|---|
| L2 normalize | `x / sqrt(reduce_sum(square(x)))` (4 ops) | `x / reduce_l2_norm(x)` (2 ops) |
| Log-sum-exp | `log(reduce_sum(exp(x)))` (3 ops) | `reduce_log_sum_exp(x)` (1 op) |

### Implementation

- Update relevant passes to emit single-op reductions when targeting ANE
- `reduce_log_sum_exp` is particularly valuable for numerically stable
  attention computation

---

## 7. `pad` for Causal Convolution & Shape Alignment

**Enabled by:** `pad` ✅ eval (max_err=0.025)

### What this enables

- Asymmetric pre-padding for causal convolutions (left-pad only) — critical
  for WaveNet-style architectures and causal conv in speech models
- Shape alignment before conv/matmul to satisfy ANE's channel alignment
  requirements (multiples of 32) without reshape workarounds
- The `pad` op was not listed in any ANE project's op support set

---

## 8. `inverse` for Direct Reciprocal

**Enabled by:** `inverse(epsilon)` ✅ eval (max_err=0.001)

### What this simplifies

Replace `real_div(1, x)` with `inverse(x, epsilon)` — single op, slightly
better numerical stability due to epsilon guarding against division by zero.
Useful in attention scaling (`1/sqrt(d_k)`), normalization denominators, and
any reciprocal computation.

---

## Summary: Capability Matrix

| Capability | Required ops | Other projects | ironmill |
|---|---|---|---|
| Full-ANE transformer (no CPU norm) | `layer_norm`, `rsqrt(eps)` | ❌ CPU fallback | ✅ |
| Native exact GELU | `erf` | ❌ 7-op tanh decomp | ✅ 5-op exact |
| INT8 activation pipeline | `quantize`, `dequantize` | ❌ unverified | ✅ 1.88× potential |
| On-ANE causal masking | `greater`, `select`, bool cast | ❌ CPU masking | ✅ |
| On-ANE logarithm | `log(eps)` | ❌ not possible | ✅ |
| Single-op L2 norm | `reduce_l2_norm` | ❌ 4-op decomp | ✅ 2-op |
| Single-op log-sum-exp | `reduce_log_sum_exp` | ❌ 3-op decomp | ✅ 1-op |
| Causal conv padding | `pad` | ❌ not tested | ✅ |
| Direct reciprocal | `inverse(eps)` | ❌ `real_div(1,x)` | ✅ single op |

These 52 op discoveries collectively enable ironmill to keep **significantly
more of the inference and training graph on ANE** than any competing project,
reducing the CPU↔ANE data transfers that dominate on-device LLM latency.
