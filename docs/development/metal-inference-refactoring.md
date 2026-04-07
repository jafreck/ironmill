# Metal Inference Refactoring — Implementation Plan

> Refactor the `ironmill-inference` Metal backend from a 6,600-line monolith
> into focused, single-concern modules while preserving every optimization and
> the public API surface.

## Status

**Proposed**

## Problem

`metal/inference.rs` has grown to 6,629 lines. It contains weight loading,
buffer allocation, RoPE cache construction, MPS matmul caching, the full decode
pipeline, a near-duplicate calibration pipeline, GDN linear-attention helpers,
MoE dispatch, PLE dispatch, MLA weight absorption, CPU GDN fallback, and
projection dispatch. Architecture-specific branching for Gemma 4, Qwen 3.5/GDN,
DeepSeek/MLA, and standard LLaMA-like models is interleaved throughout.

Key pain points:

- **Readability:** the `run_pipeline_inner` function alone is ~1,000 lines with
  nested architecture branches.
- **Duplication:** `run_pipeline_calibration` duplicates most of the pipeline
  logic with per-layer GPU sync inserted. `load()` and `load_weights()` share
  significant initialization code.
- **Coupling:** every new architecture feature (PLE, MoE, GDN, MLA, DAC)
  requires edits deep inside the pipeline loop rather than in a self-contained
  module.
- **Onboarding:** understanding any single concern requires reading thousands
  of unrelated lines.

## Approach

Split `metal/inference.rs` into 12 focused modules, none exceeding 500 lines.
Introduce a `ModelPlan` struct that captures all architecture-specific decisions
at load time so the pipeline loop is a flat sequence of data-driven dispatches
with no runtime architecture inspection.

No new traits or dynamic dispatch. Architecture differences are data in the
plan, not vtable calls. All existing optimizations are preserved.

## Target Module Layout

```
src/metal/
├── mod.rs                  re-exports
├── config.rs               MetalConfig, Gemma4Config, GdnModelConfig (unchanged)
├── error.rs                MetalError (unchanged)
├── ops.rs                  MetalPipelines + encode_* wrappers (unchanged)
├── weights.rs              WeightBuffer, MetalWeights (unchanged)
├── dequant.rs              CPU dequant helpers (unchanged)
├── bundle.rs               .ironml-gpu bundle loader (unchanged)
├── mla.rs                  MLA config + absorb_mla_weights (moved from inference.rs)
│
├── engine.rs        NEW    MetalInference struct + InferenceEngine impl
├── plan.rs          NEW    ModelPlan, LayerPlan, AttentionKind, ResidualStrategy
├── loading.rs       NEW    load(), load_weights(), resource initialization
├── pipeline.rs      NEW    run_pipeline_inner, prefill_all_logits
├── calibration.rs   NEW    run_pipeline_calibration, prefill_with_hooks, calibrate_dac
├── projection.rs    NEW    encode_projection + per-format dispatchers
├── attention.rs     NEW    QK-norm + RoPE + KV cache + attention encoding
├── ffn.rs           NEW    encode_ffn_block, encode_moe_block
├── gdn.rs           NEW    GdnState + encode_gdn_prefill/decode + CPU fallback
├── ple.rs           NEW    PLE model-level + per-layer dispatch
├── kv_cache.rs      NEW    Fp16KvCache
├── buffers.rs       NEW    IntermediateBuffers, MpsMatmulCache, RoPE cache, helpers
│
├── turboquant/             (unchanged)
```

## Detailed Design

### ModelPlan (plan.rs)

A top-level struct resolved once at load time. It replaces all scattered
`gemma4_config`, `model_config`, and feature-flag reads during the pipeline
loop.

```rust
pub(crate) struct ModelPlan {
    pub hidden_size: usize,
    pub num_attention_heads: u32,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub embed_scale: f32,           // 1.0 or sqrt(hidden_size) for Gemma
    pub layers: Vec<LayerPlan>,
    pub final_logit_softcapping: Option<f32>,
    pub enable_turboquant: bool,
    pub use_fa2_prefill: bool,
    pub max_seq_len: usize,
    pub tq_n_bits: usize,
    pub ple: Option<PlePlan>,
    pub has_dac: bool,
}
```

`LayerPlan` gains a `ResidualStrategy` enum to replace the current nested
if/else for post-layer residual handling:

```rust
pub(crate) enum ResidualStrategy {
    /// Fused residual + RMSNorm kernel (standard LLaMA-like)
    Fused,
    /// Separate residual, then scale, then norm (layer_scalar present)
    SplitWithScale,
    /// Gemma post-attn norm applied before residual add
    GemmaPostAttnNorm,
}
```

### Pipeline Simplification (pipeline.rs)

`run_pipeline_inner` becomes a ~100-line function of sequential dispatch calls.
Each call delegates to a single-concern module:

```
embedding_and_first_norm → [PLE model-level] → per-layer {
    attention or GDN → [DAC bias] → FFN → [MoE] → [post-FFN norm]
    → [PLE per-layer] → residual + next-layer norm
} → final norm → LM head → [softcap]
```

Each `→` is one function call into a sibling module. Optional steps (brackets)
are gated by `LayerPlan` fields, not architecture config inspection.

### Calibration Deduplication (calibration.rs)

`run_pipeline_calibration` currently duplicates `run_pipeline_inner` to insert
per-layer command-buffer commits and activation readbacks. After the refactoring,
it reuses the same encode functions from `attention.rs`, `ffn.rs`, etc., but
wraps each layer's dispatches in its own command-buffer cycle:

```rust
for (layer_idx, plan) in model_plan.layers.iter().enumerate() {
    let mut cmd = queue.command_buffer()?;
    let enc = cmd.compute_encoder()?;

    attention::encode_layer(&enc, ...)?;
    ffn::encode_block(&enc, ...)?;
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Read back activations for calibration hooks
    readback_activations(bufs, layer_idx, callback)?;
}
```

### Loading Consolidation (loading.rs)

`load()` and `load_weights()` are unified into a single initialization flow
with clear sub-steps:

1. Load weights via `MetalWeights::load()`
2. Detect architecture configs (Gemma4, GDN, MLA, CLA)
3. Compile shader pipelines
4. Allocate intermediate buffers
5. Build RoPE caches
6. Initialize KV cache (TurboQuant or FP16)
7. Initialize GDN state
8. Absorb MLA weights
9. Build `ModelPlan`
10. Build MPS matmul cache

Both public entry points (`load()` and `load_weights()`) call into this shared
flow, differing only in how they obtain the `WeightProvider` and `MetalConfig`.

### MLA Absorption (mla.rs)

`absorb_mla_weights` and `read_f16_buffer` move from inference.rs into `mla.rs`,
which already owns `MlaConfig`, `MlaKvCache`, and `absorb_weights`. This
colocates the load-time weight transform with the MLA math it depends on.

## Phases

### Phase 1 — Extract pure-function modules

Move code out of `inference.rs` into new modules. Each step is a single commit
that compiles and passes tests.

| Step | New file | Lines moved | What moves |
|------|----------|-------------|------------|
| 1.1 | `plan.rs` | ~200 | `LayerPlan`, `AttentionKind`, `RopeTable`, `MoeLayerConfig`, `LayerPlan::build` |
| 1.2 | `projection.rs` | ~250 | `ProjectionMatmul`, `encode_projection`, `encode_polarquant_projection`, `encode_affine_projection`, `encode_d2quant_projection` |
| 1.3 | `kv_cache.rs` | ~200 | `Fp16KvCache` struct + impl |
| 1.4 | `buffers.rs` | ~350 | `IntermediateBuffers`, `MpsMatmulCache`, `LayerMatmuls`, `bytes_as_f16`, `read_buffer_f32`, `write_buffer_f32`, `read_weight_f32`, `ModelConfigExt`, `build_rope_cache`, `build_matmul_cache` |
| 1.5 | `gdn.rs` | ~500 | `GdnState`, `GdnLayerState`, `encode_gdn_prefill`, `encode_gdn_decode`, `run_gdn_layer_cpu`, `matvec`, `softplus` |
| 1.6 | `ffn.rs` | ~250 | `encode_ffn_block`, `encode_moe_block` |
| 1.7 | `attention.rs` | ~450 | `encode_qk_norm_and_rope`, `encode_kv_cache_and_attention`, `encode_end_of_layer_residual` |
| 1.8 | `ple.rs` | ~200 | PLE model-level computation, PLE per-layer block |

After Phase 1, `inference.rs` drops to ~2,500 lines: the struct definition,
`load()`, `load_weights()`, `run_pipeline_inner`, `run_pipeline_calibration`,
MLA absorption, and trait impls.

**Validation:** `cargo build --release -p ironmill-inference --features metal`
and `cargo test --release -p ironmill-bench --features metal` after each step.

### Phase 2 — Introduce ModelPlan and simplify pipeline

| Step | Change |
|------|--------|
| 2.1 | Add `ModelPlan`, `PlePlan`, `ResidualStrategy` to `plan.rs` |
| 2.2 | Build `ModelPlan` in loading code, store as `engine.plan` |
| 2.3 | Rewrite `run_pipeline_inner` to read `ModelPlan` exclusively (no more `self.gemma4_config`, `self.model_config`, `self.gdn_state` reads in the loop) |
| 2.4 | Create `engine.rs` — move `MetalInference` struct + `MetalArtifacts` + `new()` + `InferenceEngine` impl + `CalibratingEngine` impl |

After Phase 2, `inference.rs` no longer exists. The pipeline loop is ~100 lines.

**Validation:** full benchmark suite to confirm identical output (PPL match).

### Phase 3 — Consolidate loading and calibration

| Step | Change |
|------|--------|
| 3.1 | Create `loading.rs` — merge `load()` and `load_weights()` into shared init flow |
| 3.2 | Create `calibration.rs` — refactor `run_pipeline_calibration` to reuse encode functions from Phase 1 modules |
| 3.3 | Move `absorb_mla_weights` + `read_f16_buffer` into `mla.rs` |

**Validation:** calibration tests + benchmark PPL match.

### Phase 4 — Cleanup

| Step | Change |
|------|--------|
| 4.1 | Remove `ModelConfigExt` trait (`.num_kv_heads()` duplicates a struct field) |
| 4.2 | Gate `IRONMILL_SAVE_LOGITS` / `IRONMILL_DEBUG_LOGITS` debug blocks behind `#[cfg(debug_assertions)]` |
| 4.3 | Audit and remove any remaining `pub` that should be `pub(crate)` |
| 4.4 | Update module-level doc comments for each new file |

## File Size Targets

| File | Est. lines | Responsibility |
|------|------------|----------------|
| `engine.rs` | ~200 | Struct definition, trait impls, constructor |
| `plan.rs` | ~300 | ModelPlan + LayerPlan + builders |
| `loading.rs` | ~500 | Load flow, resource init, matmul/RoPE cache |
| `pipeline.rs` | ~400 | Pipeline loop, prefill, decode helpers |
| `calibration.rs` | ~400 | Calibration pipeline, DAC, activation hooks |
| `projection.rs` | ~250 | Linear projection dispatch by weight format |
| `attention.rs` | ~450 | Standard attention block encoding |
| `ffn.rs` | ~250 | FFN + MoE block encoding |
| `gdn.rs` | ~500 | GDN state, GPU/CPU encode |
| `ple.rs` | ~200 | PLE computation |
| `kv_cache.rs` | ~200 | FP16 KV cache |
| `buffers.rs` | ~350 | Intermediate buffers, helpers |

Total: ~4,000 lines (down from 6,629 — reduced by eliminating calibration
pipeline duplication and debug instrumentation).

## What Does Not Change

- All Metal shader files (`.metal`)
- `build.rs` (shader precompilation)
- `ops.rs` (pipeline compilation + encode wrappers)
- `weights.rs` (weight buffer types + loading)
- `config.rs` (architecture config parsing)
- `bundle.rs` (`.ironml-gpu` format)
- `dequant.rs` (CPU dequantization)
- `turboquant/` (TurboQuant KV cache compression)
- Public API surface (`MetalInference::new`, `load`, `load_weights`, all
  `InferenceEngine` / `CalibratingEngine` methods)
- All optimizations: TurboQuant, FA2 prefill, CLA, DAC, MLA absorption,
  fused kernels, layer_scalar, PLE, MoE, GDN, sliding window

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Borrow checker friction with `&mut self` across module boundaries | Make `MetalInference` fields `pub(crate)`. Module functions take `&MetalInference` or individual field references, not `&mut self`. |
| Behavioral regression | PPL validation after each phase. Diff logits against known-good baseline for Gemma 4 E2B + Qwen 3.5. |
| Merge conflicts with concurrent work | Phase 1 steps are pure moves (no logic changes). Rebase-friendly. |
| Calibration refactor introduces subtle ordering bugs | Keep `run_pipeline_calibration` as-is through Phase 2; only refactor in Phase 3 after the encode functions are stable. |
