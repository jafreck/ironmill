# TurboQuant End-to-End Inference Integration

> **Status:** In Progress
>
> **Prerequisites:**
> - [TurboQuant Implementation](turboquant-implementation.md) — completed
> - [ANE BLOBFILE Investigation](ane-blobfile-investigation.md) — resolved (root cause: `emit_const_op` attribute lookup)
> - [ANE MIL Emitter Compatibility](ane-mil-emitter-compat.md) — open (blocks layer ANE compilation)
>
> **Goal:** Run Qwen3-0.6B on ANE-direct with and without TurboQuant INT8
> KV cache compression. Measure tokens/sec, memory, and perplexity delta.

## Design

TurboQuant integrates into a new `AneInference` engine as an optional
mode. When enabled, the per-layer execution loop replaces the model's
internal attention and KV cache with TurboQuant's INT8 pipeline.
Everything else (embedding, projections, FFN, lm_head) runs unchanged
on ANE sub-programs.

Each `layer_N` sub-program is split at the attention boundary during
compilation, producing two sub-programs per layer: `pre_attn` and
`post_attn`. The attention ops between them are excluded from the split
and handled at runtime. This split is required for both paths — FP16
baseline also needs it for stateful KV cache management. In TurboQuant
mode, TurboQuant's cache-write + attention programs replace the
attention step; in FP16 mode, an FP16 attention sub-program or
CPU-managed KV cache fills the gap.

### Execution flow (one token, decode phase)

```
Token ID → Embedding sub-program (ANE)
         → For each layer:
             pre_attn sub-program (ANE): norm → Q/K/V projection
             ┌─ if TurboQuant ──────────────────────────────────────┐
             │  cache-write (ANE): rotate + quantize K/V → INT8     │
             │  CPU: write INT8 bytes into persistent cache          │
             │  attention (ANE): dequant INT8 cache → attend → out  │
             ├─ else (FP16 baseline) ───────────────────────────────┤
             │  fp16_attn sub-program (ANE): attention with FP16 KV │
             │  CPU: write FP16 K/V into persistent cache            │
             └──────────────────────────────────────────────────────┘
             post_attn sub-program (ANE): O proj → residual → FFN → residual
         → lm_head sub-program (ANE) → logits → sample
```

### Comparison benchmark

```
┌──────────────────────────┬──────────────────────────┐
│ Baseline (FP16)          │ TurboQuant (INT8)        │
├──────────────────────────┼──────────────────────────┤
│ AneInference::decode()   │ AneInference::decode()   │  ← same code path
│ FP16 attention sub-prog  │ TurboQuant attention     │  ← only divergence
│ FP16 KV cache (2 B/elem) │ INT8 KV cache (1 B/elem) │
└──────────────────────────┴──────────────────────────┘

Metrics: tokens/sec, KV cache MB, perplexity (WikiText-2 / LongBench)
```

### Shared vs divergent code

`AneInference` is a **single struct** that serves both the FP16 baseline
and TurboQuant paths. The `turboquant: Option<TurboQuantModel>` field is
the only branching point.

**Shared (identical for both paths):**
- ONNX → MIL IR conversion
- Model architecture extraction
- Attention-boundary layer splitting (both paths need pre_attn/post_attn)
- ANE compilation of embedding, pre_attn, post_attn, lm_head sub-programs
- `decode()` control flow (per-layer loop, tensor threading)
- `generate()` loop, sampling, EOS detection
- Tensor I/O (token ID → embedding, logits → f32)
- Benchmarking harness and quality measurement
- CLI integration

**Divergent (~20 lines in the per-layer loop):**
```rust
// Inside decode(), for each layer:
let (q, k_proj, v_proj) = self.run_pre_attn(layer)?;

let attn_out = if let Some(tq) = &mut self.turboquant {
    // TurboQuant: rotate+quantize K/V to INT8, write to cache,
    // dequant+attend from INT8 cache
    tq.step_attention(layer, &q, &k_proj, &v_proj)?
} else {
    // Baseline: run FP16 attention sub-program, copy FP16 KV
    // to persistent cache via write_f16_at
    self.run_fp16_attention(layer, &q, &k_proj, &v_proj)?
};

self.run_post_attn(layer, &attn_out)?;
```

**Why both paths need the layer split:** Even the FP16 baseline requires
stateful KV cache management — the ONNX model's `past_key_values` /
`present` must be threaded across decode steps. This means both paths
intercept at the attention boundary to manage cache state. The split
is not TurboQuant-specific infrastructure; it's a prerequisite for any
autoregressive ANE inference.

---

## Implementation Plan

### Phase 1 — Model architecture extraction

Extract `num_heads`, `num_kv_heads`, `head_dim`, `num_layers` from a
loaded ONNX/MIL Program. The ONNX converter already reads these from
`GroupQueryAttention` attributes — surface them as a `ModelArch` struct.

**File:** `crates/mil-rs/src/analysis/arch.rs` (new)

```rust
pub struct ModelArch {
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
}

/// Detect architecture parameters by walking the program's ops.
pub fn detect_model_arch(program: &Program) -> Option<ModelArch>;
```

Walk the program's ops: count layer groups, find attention ops, extract
head counts from op attributes or tensor shapes.

Also add a convenience constructor on `TurboQuantConfig`:

```rust
impl TurboQuantConfig {
    pub fn from_arch(arch: &ModelArch, max_seq_len: usize) -> Result<Self> {
        Self::new(8, max_seq_len, arch.num_heads, arch.num_kv_heads,
                  arch.head_dim, arch.num_layers)
    }
}
```

### Phase 2 — Attention-boundary layer splitting

Extend `split_for_ane` to split each `layer_N` into two sub-programs
at the attention boundary (the attention ops are excluded and handled
at runtime):

```
layer_N_pre_attn:   norm → Q/K/V linear projections
                    outputs: Q, K_proj, V_proj, residual_hidden
layer_N_post_attn:  O projection → residual add → FFN → residual add
                    inputs: attn_output, residual_hidden
```

**File:** `crates/ironmill-ane/src/split.rs`

The split point is detectable: find the attention op cluster (the ops
between Q/K/V matmuls and the output projection matmul). In the MIL IR
from Qwen3-0.6B this is the `GroupQueryAttention` lowered pattern.

Add a `SplitConfig` flag:

```rust
pub struct SplitConfig {
    pub max_weight_size: usize,
    pub split_attention: bool,  // new: split at attention boundary
}
```

When `split_attention` is false, behavior is unchanged.

### Phase 3 — Autoregressive `AneInference` engine

New struct that manages stateful autoregressive inference. Replaces the
stateless `AneModel::predict()` for decode workloads.

**File:** `crates/ironmill-ane/src/inference.rs` (new)

```rust
pub struct AneInference {
    /// Embedding sub-program.
    embedding: LoadedSubProgram,
    /// Per-layer: (pre_attn, post_attn) sub-programs.
    layers: Vec<LayerPrograms>,
    /// lm_head sub-program.
    lm_head: LoadedSubProgram,
    /// Optional TurboQuant model (replaces layer attention when enabled).
    turboquant: Option<TurboQuantModel>,
    /// FP16 KV caches (used when TurboQuant is disabled).
    /// Per-layer (K, V) tensors. None when TurboQuant manages the cache.
    fp16_kv_caches: Option<Vec<(AneTensor, AneTensor)>>,
    /// Runtime handle.
    runtime: AneRuntime,
    /// Current sequence position.
    seq_pos: usize,
}

/// A loaded sub-program with pre-allocated I/O tensors.
struct LoadedSubProgram {
    loaded: LoadedProgram,
    input_tensors: Vec<AneTensor>,
    output_tensors: Vec<AneTensor>,
}

struct LayerPrograms {
    pre_attn: LoadedSubProgram,
    /// FP16 attention sub-program. Only compiled in baseline mode;
    /// in TurboQuant mode this is None (TurboQuantModel handles attention).
    fp16_attn: Option<LoadedSubProgram>,
    post_attn: LoadedSubProgram,
}

impl AneInference {
    /// Build from a Program. When `turbo_config` is Some, TurboQuant
    /// replaces the FP16 attention sub-programs (they are not compiled,
    /// saving ANE compile budget).
    pub fn compile(
        program: &Program,
        turbo_config: Option<TurboQuantConfig>,
    ) -> Result<Self>;

    /// Process one token, return logits.
    pub fn decode(&mut self, token_id: u32) -> Result<Vec<f32>>;

    /// Reset all state for a new conversation.
    pub fn reset(&mut self);
}
```

For the **baseline (no TurboQuant)**: each layer runs pre_attn → FP16
attention sub-program → post_attn, with FP16 KV cache managed via
IOSurface partial writes.

For **TurboQuant mode**: each layer runs pre_attn → TurboQuant
cache-write + attention → post_attn, with INT8 KV cache.

Same code path, same model, only the attention + cache differs.

### Phase 4 — Generation loop

**File:** `crates/ironmill-ane/src/inference.rs`

```rust
impl AneInference {
    /// Prefill: process all prompt tokens sequentially, populating the
    /// KV cache. Returns logits for the last prompt token.
    ///
    /// Batch prefill (processing multiple tokens in one ANE eval) is not
    /// supported initially — the attention sub-programs are compiled for
    /// single-token decode shapes. Sequential prefill calls decode()
    /// for each prompt token.
    pub fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<Vec<f32>>;

    /// Generate tokens autoregressively.
    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        temperature: f32,
    ) -> Result<Vec<u32>>;
}
```

`generate()` calls `prefill()` then loops `decode()` with sampling.
Sampling strategies: greedy (temperature=0), temperature scaling,
top-k/top-p. EOS token detection stops generation.

> **Future optimization — Metal batch prefill:** Sequential prefill is
> the biggest performance bottleneck. A 2K-token prompt takes ~22 s at
> 11 ms/token. Batched Metal prefill could reduce this to ~200 ms by
> dispatching one large GPU matmul per layer. The architecture supports
> this cleanly: Metal prefill fills the KV cache, then the per-token
> decode loop switches to ANE. This is deferred because it requires a
> separate Metal compute pipeline and is independent of the ANE decode
> path being built here.

### Phase 5 — Benchmark + quality comparison

**File:** `crates/ironmill-ane/examples/turboquant_e2e_bench.rs`

Run Qwen3-0.6B with both paths:

```rust
let onnx = mil_rs::reader::onnx::read_onnx("tests/fixtures/qwen3-0.6b.onnx")?;
let program = mil_rs::convert::onnx_graph::onnx_to_program(&onnx)?.program;
let arch = mil_rs::analysis::arch::detect_model_arch(&program)
    .expect("failed to detect model architecture");

// Baseline: FP16 KV cache
let mut baseline = AneInference::compile(&program, None)?;
let baseline_output = baseline.generate(&prompt, 128, 0.0)?;

// TurboQuant: INT8 KV cache
let tq_config = TurboQuantConfig::from_arch(&arch, 512)?;
let mut turbo = AneInference::compile(&program, Some(tq_config))?;
let turbo_output = turbo.generate(&prompt, 128, 0.0)?;
```

Measure and compare:
- **Throughput**: tokens/sec (decode latency)
- **Memory**: KV cache size (INT8 vs FP16)
- **Quality**: Token-level agreement, perplexity on test corpus

---

## ANE Constraints

**Compile budget:** ANE caps at ~119 compiled programs per process.
Budget for Qwen3-0.6B with attention splitting:
- Baseline: 1 embedding + 29 pre_attn + 29 post_attn + 1 lm_head = 60
  (FP16 attention is handled externally, not as a compiled sub-program)
- TurboQuant: 1 embedding + 29 pre_attn + 29 post_attn + 1 lm_head + 2–3 TQ programs = 63
- Both well within budget

**Uniform allocation sizes:** All tensors passed to a single `eval()`
call must have the same IOSurface allocation size. `AneInference` must
compute uniform alloc sizes per sub-program group (embedding, pre_attn,
attention, post_attn, lm_head) and allocate all I/O tensors accordingly.
Use `uniform_alloc_size()` and `AneTensor::new_with_min_alloc()`.

---

## Files Changed

| File | Change |
|---|---|
| `crates/mil-rs/src/analysis/arch.rs` | New: `detect_model_arch()` |
| `crates/ironmill-ane/src/split.rs` | Extend: attention-boundary splitting |
| `crates/ironmill-ane/src/inference.rs` | New: `AneInference` engine |
| `crates/ironmill-ane/src/lib.rs` | Register `inference` module |
| `crates/ironmill-ane/examples/turboquant_e2e_bench.rs` | New: comparison benchmark |
| `crates/ironmill-cli/src/main.rs` | Wire `--kv-quant` into inference path (future) |

## Test Model

**Qwen3-0.6B** (`tests/fixtures/qwen3-0.6b.onnx`, ~315 MB):
- 29 layers, 16 attention heads, 8 KV heads (GQA 2:1)
- head_dim = 64, hidden_size = 1024
- Autoregressive ONNX with `past_key_values` / `present`
- Already in test fixtures

## References

- [TurboQuant Implementation](turboquant-implementation.md)
- [ANE BLOBFILE Investigation](ane-blobfile-investigation.md)
- [ANE MIL Emitter Compatibility](ane-mil-emitter-compat.md)
- [ANE Op Support Matrix](../research/ane-op-support-matrix.md)
