# Implementation Plan: SafeTensors + GGUF Input for Ironmill

## Problem Statement

Ironmill currently accepts only ONNX as a model input format. The LLM ecosystem
primarily distributes models as HuggingFace SafeTensors (training/fine-tuning) and
GGUF (local inference). Supporting both unlocks Python-free CoreML conversion for
the most common LLM distribution formats.

**Key insight**: SafeTensors and GGUF are weight-only formats (GGUF includes some
architecture metadata, SafeTensors does not). Neither contains a computation graph.
A new **architecture template** layer must construct the MIL IR graph, then slot
in weights from either format. This is fundamentally different from ONNX conversion,
which reads a complete graph.

## Approach

```
                    ┌─────────────────────┐
                    │  Weight Providers    │
                    │  (format readers)    │
                    ├──────────┬──────────┤
                    │SafeTensors│  GGUF   │
                    │ + config  │(self-    │
                    │  .json    │describing│
                    └─────┬────┴────┬─────┘
                          │         │
                          ▼         ▼
                    ┌─────────────────────┐
                    │   WeightProvider     │
                    │   trait              │
                    │   + ModelConfig      │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ Architecture        │
                    │ Templates           │
                    │ (LLaMA, Qwen, etc.) │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  MIL IR Program     │
                    │  (existing IR API)  │
                    └─────────────────────┘
```

## Module Layout

New code lives in `crates/mil-rs/src/convert/` alongside the existing ONNX path:

```
crates/mil-rs/src/convert/
├── mod.rs                      # MODIFY: add module declarations + re-exports
├── ir_to_mil_text.rs           # existing (unchanged, feature-gated: ane-direct)
├── ir_to_proto.rs              # existing (unchanged)
├── lora.rs                     # existing — LoRA adapter merging (see interaction notes below)
├── moe.rs                      # existing (unchanged)
├── onnx_graph.rs               # existing (unchanged)
├── onnx_to_mil.rs              # existing (unchanged)
├── pipeline.rs                 # existing (MODIFY: add safetensors/gguf stage sources)
├── proto_to_ir.rs              # existing (unchanged)
├── weights/                    # NEW: weight provider layer
│   ├── mod.rs                  #   trait + ModelConfig + format detection
│   ├── safetensors.rs          #   SafeTensors reader (including LoRA adapter dirs)
│   └── gguf.rs                 #   GGUF reader
└── templates/                  # NEW: architecture templates
    ├── mod.rs                  #   registry + dispatch
    ├── config.rs               #   HF config.json / GGUF metadata → ModelConfig
    ├── llama.rs                #   LLaMA family template
    ├── qwen.rs                 #   Qwen family template
    └── gemma.rs                #   Gemma family template

crates/mil-rs/src/reader/
├── mod.rs                      # MODIFY: add read_safetensors, read_gguf
├── mlmodel.rs                  # existing (unchanged)
├── mlpackage.rs                # existing (unchanged)
├── onnx.rs                     # existing (unchanged)
├── safetensors.rs              # NEW: file-level reader
└── gguf.rs                     # NEW: file-level reader

crates/ironmill-cli/src/main.rs # MODIFY: detect input format, route accordingly
```

## Dependencies to Add

In `crates/mil-rs/Cargo.toml`:
```toml
[dependencies]
safetensors = "0.7"             # HuggingFace SafeTensors format (0.7.0 on crates.io)
gguf-rs-lib = "0.2"             # GGUF format reader (0.2.5 on crates.io)

# safetensors: pure Rust, zero-copy, maintained by HuggingFace
# gguf-rs-lib: type-safe API, serde integration, mmap/zero-copy,
#   well-maintained, strong documentation. Preferred over gguf-rs
#   (more minimal, lower-level) and woolly-gguf (unpublished: 0.0.0).
#
# Both crates verified on crates.io as of 2026-03.
```

## Detailed Design

### Design A: WeightProvider Trait + ModelConfig

**Goal**: Abstract over weight formats so templates don't care where tensors come from.

```rust
// crates/mil-rs/src/convert/weights/mod.rs

/// Architecture-agnostic model configuration extracted from
/// config.json (SafeTensors) or GGUF metadata.
pub struct ModelConfig {
    pub architecture: Architecture,     // LLaMA, Qwen, Gemma, etc.
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,     // for GQA
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub tie_word_embeddings: bool,
    /// Architecture-specific parameters that don't belong in the
    /// common struct (e.g. Gemma sliding_window_size, Qwen attention
    /// bias flags). Templates downcast via architecture match.
    pub extra: HashMap<String, serde_json::Value>,
}

pub enum Architecture {
    Llama,
    Qwen,
    Gemma,
    // future: Mistral, Phi, etc.
}

/// Trait abstracting over weight storage formats.
/// Templates call this to get named weight tensors.
pub trait WeightProvider {
    /// Get a tensor by its canonical name (e.g. "model.layers.0.self_attn.q_proj.weight")
    /// Returns a borrowed view to avoid copying multi-GB tensors.
    fn tensor(&self, name: &str) -> Result<WeightTensor<'_>>;

    /// List all available tensor names
    fn tensor_names(&self) -> Vec<&str>;

    /// Get model configuration
    fn config(&self) -> &ModelConfig;
}

/// Borrowed view of a weight tensor. Uses Cow to allow zero-copy
/// mmap access from SafeTensors while still supporting owned data
/// from GGUF dequantization.
pub struct WeightTensor<'a> {
    pub data: Cow<'a, [u8]>,
    pub shape: Vec<usize>,
    pub dtype: ScalarType,
}
```

**Why a trait?** The template code calls `provider.tensor("model.layers.0.self_attn.q_proj.weight")`
and doesn't know or care whether that came from SafeTensors or GGUF. This also allows
future formats (GGML, pickle, etc.) without touching template code.

### Design B: SafeTensors Reader

**Goal**: Load `.safetensors` files + `config.json` into a `WeightProvider`.

```rust
// crates/mil-rs/src/convert/weights/safetensors.rs

pub struct SafeTensorsProvider {
    config: ModelConfig,
    /// Memory-mapped safetensors file(s). Tensors are borrowed
    /// directly from mmap, avoiding multi-GB copies.
    mmaps: Vec<Mmap>,
    tensor_index: HashMap<String, TensorLocation>,
}

impl SafeTensorsProvider {
    /// Load from a HuggingFace model directory containing:
    /// - config.json
    /// - model.safetensors (or model-00001-of-00002.safetensors, etc.)
    /// - tokenizer.json (ignored at this layer)
    pub fn load(model_dir: &Path) -> Result<Self> { ... }
}

impl WeightProvider for SafeTensorsProvider { ... }
```

Key implementation details:
- Parse `config.json` → `ModelConfig` via architecture-specific parsers:
  - Deserialize into a loose `serde_json::Value` first
  - Read `model_type` to determine architecture
  - Extract common fields with `#[serde(default)]` for optional ones
  - Populate `ModelConfig::extra` with architecture-specific fields
    (e.g., Gemma `sliding_window`, Qwen `use_sliding_window`)
- Handle sharded models: `model.safetensors.index.json` maps tensor names to shard files
- Use the `safetensors` crate's zero-copy mmap deserialization — `WeightTensor` borrows
  directly from the mmap via `Cow::Borrowed`
- Convert tensor dtypes: `safetensors::Dtype` → ironmill `ScalarType`
- Map HF naming convention to canonical names (usually identity — HF names are the standard)
- **LoRA adapter directories**: When the SafeTensors directory contains LoRA adapter
  weights (detected via `adapter_config.json` or `lora_A`/`lora_B` naming patterns),
  the provider should merge them into base weights using the existing `merge_lora`
  logic from `convert/lora.rs`. The existing LoRA module operates on ONNX initializers
  (`(name, &[u8], shape)` tuples), so extract a shared merge kernel that both the
  ONNX path and `SafeTensorsProvider` can call. The merge formula is:
  `W_new = W + (alpha / rank) * B @ A`

### Design C: GGUF Reader

**Goal**: Load `.gguf` files into a `WeightProvider`.

```rust
// crates/mil-rs/src/convert/weights/gguf.rs

pub struct GgufProvider {
    config: ModelConfig,
    /// Owned dequantized tensors — GGUF quantized data must be
    /// converted to FP16, so zero-copy is not possible here.
    tensors: HashMap<String, OwnedWeightTensor>,
}

impl GgufProvider {
    /// Load from a single GGUF file or a split-shard set.
    /// For split shards (e.g., model-00001-of-00003.gguf), pass any
    /// shard path — sibling shards are discovered automatically.
    pub fn load(gguf_path: &Path) -> Result<Self> { ... }
}

impl WeightProvider for GgufProvider { ... }
```

Key implementation details:
- GGUF metadata → `ModelConfig`:
  - `general.architecture` → `Architecture` enum
  - `llama.embedding_length` → `hidden_size`
  - `llama.feed_forward_length` → `intermediate_size`
  - `llama.block_count` → `num_hidden_layers`
  - `llama.attention.head_count` → `num_attention_heads`
  - `llama.attention.head_count_kv` → `num_key_value_heads`
  - etc. (GGUF metadata keys are standardized per architecture)
  - Architecture-specific metadata → `ModelConfig::extra`
- **Split-shard GGUF**: Large models may be split across multiple files
  (e.g., `model-00001-of-00003.gguf`). The provider auto-discovers sibling
  shards by filename pattern and merges their tensor maps.
- GGUF tensor name mapping: GGUF uses different names than HuggingFace
  (e.g., `blk.0.attn_q.weight` vs `model.layers.0.self_attn.q_proj.weight`).
  The provider must remap to canonical HF-style names or the templates need
  to accept either convention. Recommend: remap in the provider.
- **Dequantization**: GGUF tensors are stored in llama.cpp-specific block
  quantization formats (Q4_0, Q4_K_M, Q5_K_S, Q8_0, etc.) that have no
  direct CoreML equivalent. Dequantization bridges these two worlds:
  - GGUF quant formats → FP16 → ironmill's CoreML-compatible quant passes
  - This lets ironmill's existing pipeline (FP16, INT8, palettization,
    mixed precision) handle the final quantization for CoreML/ANE.
  - **v1 strategy**: Dequantize all tensors to FP16 on load. Simple,
    correct, and reuses the full pass pipeline.
  - **Memory note**: Dequantizing Q4 → FP16 roughly quadruples memory
    (a 4GB Q4 model becomes ~16GB in FP16). For v1 this is acceptable
    for typical model sizes (≤13B). For larger models, future work can
    add per-layer streaming dequantization.
  - **v2 optimization (future)**: Map GGUF quant formats directly to
    CoreML `constexpr_lut_to_dense` ops, avoiding the dequant round-trip.

### Design D: MIL Ops Used by Templates

The MIL IR uses **string-typed ops** — `Operation` has an `op_type: String` field, not
an enum. Adding a new op means emitting `Operation::new("rms_norm", ...)` — no IR schema
changes required. Several ops needed by templates are already recognized by existing passes:

**Ops already recognized by passes** (no changes needed):
- `rms_norm` — Recognized by `LayerSchedulePass` (as `NORM_OPS`) and `ModelSplitPass`.
  Templates emit it directly; the ANE decomposition pass can lower it to
  `concat([x,-x]) → layer_norm → slice` when targeting the Neural Engine.
- `silu` — Recognized by `LayerSchedulePass` (as `ACTIVATION_OPS`). Templates emit it
  for LLaMA/Qwen/Gemma MLP gate activations. Passes can fuse `linear → silu` patterns.

**Ops that can be decomposed from existing primitives** (no new op needed):
- Rotary embeddings: precompute cos/sin tables as `const` tensors, apply with
  `gather`, `mul`, `add`, `concat`, `slice` — all already exist.
- GQA (grouped query attention): expressible with `reshape`, `transpose`, `matmul`,
  `softmax`, existing ops.

Templates should emit high-level ops (`rms_norm`, `silu`) where possible and let
passes handle decomposition for target-specific lowering (ANE, GPU, etc.).

### Design E: Architecture Templates

**Goal**: Construct a MIL IR `Program` for a given architecture using weight tensors.

```rust
// crates/mil-rs/src/convert/templates/mod.rs

pub fn weights_to_program(provider: &dyn WeightProvider) -> Result<ConversionResult> {
    match provider.config().architecture {
        Architecture::Llama => llama::build_program(provider),
        Architecture::Qwen  => qwen::build_program(provider),
        Architecture::Gemma => gemma::build_program(provider),
    }
}
```

Each template builds the MIL IR using the existing builder API:

```rust
// crates/mil-rs/src/convert/templates/llama.rs (sketch)

pub fn build_program(provider: &dyn WeightProvider) -> Result<ConversionResult> {
    let config = provider.config();
    let mut program = Program::new("1.0");
    let mut function = Function::new("main");

    // Define inputs
    function.with_input("input_ids", TensorType::new(ScalarType::Int32, vec![1, config.max_position_embeddings]));
    function.with_input("position_ids", TensorType::new(ScalarType::Int32, vec![config.max_position_embeddings]));
    function.with_input("causal_mask", TensorType::new(
        ScalarType::Float16,
        vec![1, 1, config.max_position_embeddings, config.max_position_embeddings],
    ));

    let mut block = Block::new();

    // Emit embedding lookup
    let embed_weight = provider.tensor("model.embed_tokens.weight")?;
    emit_const(&mut block, "embed_tokens_weight", &embed_weight);
    emit_gather(&mut block, "embeddings", "embed_tokens_weight", "input_ids");

    // Emit transformer layers
    for layer_idx in 0..config.num_hidden_layers {
        emit_transformer_layer(&mut block, provider, config, layer_idx)?;
    }

    // Emit final norm + LM head
    emit_rms_norm(&mut block, provider, config, "model.norm")?;
    emit_lm_head(&mut block, provider, config)?;

    function.body = block;
    program.add_function(function);
    Ok(ConversionResult { program, warnings: vec![] })
}

fn emit_transformer_layer(
    block: &mut Block,
    provider: &dyn WeightProvider,
    config: &ModelConfig,
    layer: usize,
) -> Result<()> {
    let prefix = format!("model.layers.{layer}");

    // Input norm (RMSNorm)
    emit_rms_norm(block, provider, config, &format!("{prefix}.input_layernorm"))?;

    // Self-attention (Q/K/V projections as conv or linear)
    emit_attention(block, provider, config, layer)?;

    // Residual add
    emit_residual_add(block)?;

    // Post-attention norm
    emit_rms_norm(block, provider, config, &format!("{prefix}.post_attention_layernorm"))?;

    // MLP (gate/up/down projections)
    emit_mlp(block, provider, config, layer)?;

    // Residual add
    emit_residual_add(block)?;

    Ok(())
}
```

**Key design decisions in templates**:

1. **Emit ANE-friendly ops by default** — The template can emit `conv` (1x1) instead
   of `linear` for projections when an `--ane` flag is set. This is where ANEMLL's
   patterns get baked in at the source rather than as post-hoc passes.

2. **RMSNorm emission** — The template emits the new `rms_norm` op (see Design D).
   A later ANE decomposition pass converts it to
   `concat([x,-x]) → layer_norm → slice` when targeting the Neural Engine.

3. **MLP gate activation** — Templates emit `silu` (see Design D) for LLaMA/Qwen
   gate activations. Passes can fuse `linear → silu` patterns.

4. **KV-cache emission** — Templates emit static-shape cache tensors as program state
   inputs, with fixed-size slicing for reads/writes.

5. **Rotary embeddings** — Precompute cos/sin tables as `const` tensors in the graph
   rather than emitting the computation dynamically. Uses existing `gather`, `mul`,
   `add`, `concat`, `slice` ops — no new op needed.

6. **Reuse existing passes** — Templates emit a clean graph; the existing PassPipeline
   handles fusion, quantization, FP16 conversion, op splitting, etc. Templates should
   not duplicate pass logic.

### Design F: CLI Integration

**Goal**: Auto-detect input format and route to the right converter.

```rust
// In crates/ironmill-cli/src/main.rs, modify the compile subcommand:

fn detect_input_format(path: &Path) -> InputFormat {
    if path.extension() == Some("onnx") {
        InputFormat::Onnx
    } else if path.extension() == Some("gguf") {
        InputFormat::Gguf
    } else if path.is_dir() {
        // Check for config.json + *.safetensors
        if path.join("config.json").exists() {
            InputFormat::SafeTensors
        } else {
            InputFormat::Unknown
        }
    } else if path.extension() == Some("safetensors") {
        InputFormat::SafeTensors  // single file, look for config.json alongside
    } else {
        InputFormat::Unknown
    }
}
```

CLI usage would be:
```bash
# ONNX (existing)
ironmill compile model.onnx -o model.mlpackage

# SafeTensors (directory containing config.json + *.safetensors)
ironmill compile ./Qwen3-0.6B/ -o model.mlpackage

# GGUF (single file, self-describing)
ironmill compile model.gguf -o model.mlpackage

# All three go through the same PassPipeline and output the same format
```

### Design G: Pipeline Manifest Support

**Goal**: Allow multi-stage pipelines from SafeTensors/GGUF, not just ONNX.

Extend `pipeline.rs` manifest format:
```toml
[pipeline]
name = "llama-3.2-1b-ane"

[[stages]]
name = "embeddings"
safetensors = "./Llama-3.2-1B/"
component = "embeddings"        # NEW: which component to extract
quantize = "fp16"

[[stages]]
name = "transformer"
safetensors = "./Llama-3.2-1B/"
component = "transformer"
quantize = "int8"
depends_on = "embeddings"

[[stages]]
name = "lm_head"
safetensors = "./Llama-3.2-1B/"
component = "lm_head"
quantize = "fp16"
depends_on = "transformer"
```

This maps directly to ANEMLL's multi-component split pattern but stays within
ironmill's existing pipeline infrastructure.

## Implementation Order

### Milestone 1: Foundation (WeightProvider + ModelConfig)
- Define `WeightProvider` trait and `ModelConfig` struct (Design A)
- Define `Architecture` enum
- Add `convert/weights/mod.rs`
- Verify `rms_norm` and `silu` op strings are handled by all relevant passes
  (they are already recognized by `LayerSchedulePass` and `ModelSplitPass` —
  see Design D — but audit remaining passes for completeness)
- No external dependencies yet, just types

### Milestone 2: SafeTensors Reader
- Add `safetensors = "0.7"` dependency
- Implement `SafeTensorsProvider` with mmap zero-copy (Design B)
- Parse `config.json` → `ModelConfig` with architecture-specific parsers
- Handle sharded models
- Refactor `convert/lora.rs` to extract a format-agnostic merge kernel, then wire
  it into `SafeTensorsProvider` for LoRA adapter directories
- Add `reader/safetensors.rs` for file-level entry point
- Test: load a real HF model directory, verify tensor shapes/dtypes
- Test: load a LoRA adapter directory, verify merged weights match expected output

### Milestone 3: LLaMA Template (MVP)
- Implement `templates/llama.rs` (Design E)
- Start with inference-only (no KV-cache state, no prefill/decode split)
- Emit: embedding → N×(norm→attention→residual→norm→MLP→residual) → norm → lm_head
- Use `linear` ops initially (not Conv2d — let passes handle ANE conversion later)
- Wire into CLI: `ironmill compile ./Llama-3.2-1B/ -o llama.mlpackage` (Design F)
- Test: compile, validate with `ironmill validate`, run with `ironmill-coreml`

### Milestone 4: GGUF Reader
- Add `gguf-rs-lib = "0.2"` dependency
- Implement `GgufProvider` with metadata → `ModelConfig` mapping (Design C)
- Implement GGUF tensor name → HF canonical name remapping
- Handle split-shard GGUF files (auto-discover siblings)
- Dequantize to FP16 on read (v1 strategy — see Design C memory note)
- Test: load a GGUF model, verify same `Program` output as SafeTensors for same model

### Milestone 5: ANE-Aware Templates
- Add `--ane` flag to templates that emits Conv2d instead of Linear
- Add RMSNorm decomposition in template (or rely on the pass from the ANE patterns doc)
- Add static KV-cache state inputs
- Add prefill/decode function split (multi-function CoreML model)
- Precomputed rotary embedding constants
- Coordinate with `crates/ironmill-ane/` — ANE-specific lowering passes
  (e.g., `AttentionDecomposePass`, `AneLayoutPass`) live there behind the
  `ane-direct` feature flag. Ensure template-emitted ops are compatible
  with these passes and that the `ane-direct` pipeline works end-to-end
  with the new template-generated programs.

### Milestone 6: Additional Architectures + Pipeline
- `templates/qwen.rs` — extends LLaMA with Qwen-specific differences
- `templates/gemma.rs` — sliding window attention, split KV-cache
- Pipeline manifest support for `safetensors =` and `gguf =` stage sources (Design G)
- Multi-component output (embed/transformer/lm_head as separate .mlpackage files)

## Testing Strategy

### Unit Tests
- `ModelConfig` parsing from sample `config.json` fixtures
- `ModelConfig` extraction from sample GGUF metadata
- Tensor name remapping (GGUF → canonical)
- Individual template helper functions (emit_attention, emit_mlp, etc.)
- LoRA merge kernel: verify merged weights match expected output for both
  ONNX-sourced and SafeTensors-sourced adapter weights

### Integration Tests
- Round-trip: SafeTensors dir → Program → .mlpackage → validate
- Round-trip: GGUF file → Program → .mlpackage → validate
- Same model from SafeTensors vs GGUF produces equivalent Programs
- Template output passes ANE validation for known-good models
- Pipeline manifest with SafeTensors source produces valid multi-component output
- LoRA adapter SafeTensors dir → merged Program matches base+adapter merge from ONNX path
- ANE-direct pipeline (`ane-direct` feature) works end-to-end with template-generated programs

### Fixture Models
- Use small models for CI: `Qwen/Qwen3-0.6B` (SafeTensors) or quantized 0.5B GGUF
- Download script in `scripts/download-fixtures.sh` (already exists, extend it)

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Architecture templates are tedious per-model work | Start with LLaMA (covers many finetunes). Qwen/Gemma share 90% of the template. |
| GGUF dequantization loses quality | v1 accepts this. v2 can pass through quantized weights directly. |
| Template IR differs subtly from ONNX-converted IR | Validate both paths produce equivalent output for the same model. Share passes. |
| GGUF tensor name mapping is fragile | Use llama.cpp's canonical mapping. Test against multiple GGUF producers. |
| HF config.json schema varies across models | Parse only the fields ModelConfig needs. Use `serde(default)` for optional fields. |
| Template output doesn't run correctly on ANE | Validate against ANEMLL's known-good output for the same model. Use ironmill-bench. |
| LoRA merge logic is duplicated across ONNX and SafeTensors paths | Extract a shared format-agnostic merge kernel from `convert/lora.rs` in Milestone 2. |
| ANE passes in `ironmill-ane` are incompatible with template-emitted ops | Audit ANE passes (behind `ane-direct` feature flag) against template op patterns in Milestone 5. |
