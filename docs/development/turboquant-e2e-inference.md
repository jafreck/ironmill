# TurboQuant End-to-End Inference Integration

> **Status:** Planning
>
> **Prerequisites:**
> - [TurboQuant Implementation](turboquant-implementation.md) — completed
> - [ANE BLOBFILE Investigation](ane-blobfile-investigation.md) — open
>
> **Goal:** Run Qwen3-0.6B on ANE-direct with and without TurboQuant INT8
> KV cache compression. Measure tokens/sec, memory, and perplexity delta.

## Design

TurboQuant integrates into `AneModel` as an optional mode. When enabled,
the per-layer execution loop replaces the model's internal attention and
KV cache with TurboQuant's INT8 pipeline. Everything else (embedding,
projections, FFN, lm_head) runs unchanged on ANE sub-programs.

The key insight: we don't need to restructure sub-program splitting.
Instead, we modify how each `layer_N` sub-program is **compiled** — we
split the attention out of each layer at the IR level before compilation,
producing three sub-programs per layer. The non-attention portions
compile and run normally; the attention portion is replaced by
TurboQuant's cache-write + attention programs.

### Execution flow (one token)

```
Token ID → Embedding sub-program (ANE)
         → For each layer:
             pre_attn sub-program (ANE): norm → Q/K/V projection
             TurboQuant cache-write (ANE): rotate + quantize K/V → INT8
             CPU: write INT8 bytes into persistent KV cache
             TurboQuant attention (ANE): dequant INT8 cache → attention
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

**File:** `crates/mil-rs/src/ir/program.rs` (or new `crates/mil-rs/src/analysis/arch.rs`)

```rust
pub struct ModelArch {
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
}

pub fn detect_model_arch(program: &Program) -> Option<ModelArch>;
```

Walk the program's ops: count layer groups, find attention ops, extract
head counts from op attributes or tensor shapes.

### Phase 2 — Attention-boundary layer splitting

Extend `split_for_ane` to split each `layer_N` into three sub-programs
at the attention boundary:

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
    embedding: (LoadedProgram, Vec<AneTensor>, Vec<AneTensor>),
    /// Per-layer: (pre_attn, post_attn) sub-programs with I/O tensors.
    layers: Vec<LayerPrograms>,
    /// lm_head sub-program.
    lm_head: (LoadedProgram, Vec<AneTensor>, Vec<AneTensor>),
    /// Optional TurboQuant model (replaces layer attention when enabled).
    turboquant: Option<TurboQuantModel>,
    /// Runtime handle.
    runtime: AneRuntime,
    /// Current sequence position.
    seq_pos: usize,
}

struct LayerPrograms {
    pre_attn: (LoadedProgram, Vec<AneTensor>, Vec<AneTensor>),
    post_attn: (LoadedProgram, Vec<AneTensor>, Vec<AneTensor>),
    /// FP16 KV cache (used when TurboQuant is disabled).
    kv_cache: Option<(AneTensor, AneTensor)>,
}

impl AneInference {
    /// Build from a Program. When `turbo_config` is Some, TurboQuant
    /// replaces the attention sub-programs.
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
    /// Generate tokens autoregressively.
    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        temperature: f32,
    ) -> Result<Vec<u32>>;
}
```

Implements: prefill (batch or sequential) → decode loop → greedy/temp
sampling → EOS detection.

### Phase 5 — Benchmark + quality comparison

**File:** `crates/ironmill-ane/examples/turboquant_e2e_bench.rs`

Run Qwen3-0.6B with both paths:

```rust
// Baseline: FP16 KV cache
let mut baseline = AneInference::compile(&program, None)?;
let baseline_output = baseline.generate(&prompt, 128, 0.0)?;

// TurboQuant: INT8 KV cache
let tq_config = TurboQuantConfig::from_arch(&arch)?;
let mut turbo = AneInference::compile(&program, Some(tq_config))?;
let turbo_output = turbo.generate(&prompt, 128, 0.0)?;
```

Measure and compare:
- **Throughput**: tokens/sec (decode latency)
- **Memory**: KV cache size (INT8 vs FP16)
- **Quality**: Token-level agreement, perplexity on test corpus

---

## Files Changed

| File | Change |
|---|---|
| `crates/mil-rs/src/analysis/arch.rs` | New: `detect_model_arch()` |
| `crates/ironmill-ane/src/split.rs` | Extend: attention-boundary splitting |
| `crates/ironmill-ane/src/inference.rs` | New: `AneInference` engine |
| `crates/ironmill-ane/src/lib.rs` | Register `inference` module |
| `crates/ironmill-ane/examples/turboquant_e2e_bench.rs` | New: comparison benchmark |
| `crates/ironmill-cli/src/main.rs` | Wire `--kv-quant` into inference path |

## Test Model

**Qwen3-0.6B** (`tests/fixtures/qwen3-0.6b.onnx`, 300 MB):
- 28 layers, 14 attention heads, 2 KV heads (GQA 7:1)
- head_dim = 64, hidden_size = 896
- Autoregressive ONNX with `past_key_values` / `present`
- Already in test fixtures

## References

- [TurboQuant Implementation](turboquant-implementation.md)
- [ANE BLOBFILE Investigation](ane-blobfile-investigation.md)
- [ANE Op Support Matrix](../research/ane-op-support-matrix.md)
