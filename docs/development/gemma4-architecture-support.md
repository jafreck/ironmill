# Gemma 4 Architecture Support — Implementation Spec

> Adds support for Google's Gemma 4 model family (released April 2, 2026).
> Dense 31B, MoE 26B, and on-device E2B/E4B variants.

## Status

**Under development** — Phase 1 (E2B end-to-end) first.

## Background

Gemma 4 is Google's latest open-weight model family. The text decoder extends
the Gemma 3 architecture with several new features. ironmill already supports
Gemma 1/2/3 via `Architecture::Gemma`.

### Gemma 4 Variants

| Variant | Params | Context | Architecture | Key Differences |
|---------|--------|---------|--------------|-----------------|
| E2B | 2.3B eff. (5.1B total) | 128K | Dense + PLE | Per-Layer Embeddings, on-device |
| E4B | 4.5B eff. (8B total) | 128K | Dense + PLE | Per-Layer Embeddings, on-device |
| 26B A4B | 26B (3.8B active) | 256K | MoE | Mixture-of-Experts FFN |
| 31B | 31B | 256K | Dense | Largest dense variant |

### HuggingFace Config

The HuggingFace `model_type` values are:
- `"gemma4"` — top-level multimodal config (wraps text + vision + audio)
- `"gemma4_text"` — text decoder config (what we care about)

The text decoder config introduces these new fields compared to Gemma 3:

```json
{
  "model_type": "gemma4_text",
  "hidden_size": 2304,
  "intermediate_size": 9216,
  "num_hidden_layers": 30,
  "num_attention_heads": 8,
  "num_key_value_heads": 4,
  "head_dim": 256,
  "hidden_activation": "gelu_pytorch_tanh",
  "sliding_window": 512,
  "layer_types": ["sliding_attention", "sliding_attention", ..., "full_attention"],
  "rope_parameters": {
    "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
    "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1000000.0}
  },
  "num_global_key_value_heads": null,
  "global_head_dim": 512,
  "attention_k_eq_v": false,
  "num_kv_shared_layers": 0,
  "vocab_size_per_layer_input": 262144,
  "hidden_size_per_layer_input": 256,
  "enable_moe_block": false,
  "num_experts": null,
  "top_k_experts": null,
  "moe_intermediate_size": null,
  "use_double_wide_mlp": false
}
```

> **Note**: The config above is a representative template. Actual E2B/E4B configs
> are expected to set `attention_k_eq_v: true` and `num_kv_shared_layers > 0`.
> The exact values must be validated against a real E2B checkpoint download before
> implementation begins. The 31B config is expected to use `attention_k_eq_v: false`,
> `num_kv_shared_layers: 0`, and `hidden_size_per_layer_input: 0`.

Key architectural innovations:
- **5:1 sliding/global attention pattern**: `layer_types` is an explicit list per
  layer (`"sliding_attention"` or `"full_attention"`), with the last layer always
  forced to `"full_attention"`.
- **Per-layer-type RoPE**: Sliding layers use standard RoPE (θ=10K), global
  layers use proportional RoPE (θ=1M, `partial_rotary_factor=0.25`).
- **Different head dims per layer type**: Global layers can use `global_head_dim`
  (default 512) which may differ from local `head_dim` (default 256).
- **Different KV head counts per layer type**: `num_global_key_value_heads` may
  differ from `num_key_value_heads`.

## Quantization Compatibility

### Weight Quantization (INT4/INT8/AWQ/PolarQuant/D2Quant) — ✅ Compatible

All weight quantization passes operate on MIL IR const tensors and are
architecture-agnostic. Once the model passes through the template into MIL IR,
all existing weight quantization methods work with zero changes. This includes
MoE expert weights, PLE embeddings, and any new const tensors.

### TurboQuant (KV Cache Quantization) — ⚠️ Partially Compatible

`TurboQuantMetalConfig` stores a **single** `head_dim`, `num_kv_heads`, and
`num_layers` for the entire model. This breaks when:

| Assumption | Gemma 4 Reality | Breaks? |
|---|---|---|
| Uniform `head_dim` across layers | Global layers may use `global_head_dim=512` vs local `head_dim=256` | **Yes** if they differ |
| Uniform `num_kv_heads` across layers | `num_global_key_value_heads` may differ | **Yes** if they differ |
| `head_dim` is power-of-2 | 256 ✓, 512 ✓ | No |
| Causal attention (no mask) | Sliding window on 5/6 of layers | OK for decode (seq_len=1) |

**Mitigation**: For the 31B dense variant, if global/local head dims and KV head
counts happen to be uniform, TurboQuant works as-is. A full fix requires making
`TurboQuantMetalConfig` per-layer or supporting two configs (local vs global).
This is deferred to a follow-up.

## Implementation Phases

### Phase 1 — E2B End-to-End (Compile + Inference)

Support loading, compiling, and running the Gemma 4 E2B model end-to-end. E2B is
the smallest variant (5.1B total, ~2.5 GB in FP16), making it ideal for fast
iteration. It exercises the full Gemma 4 feature set: per-layer attention types,
per-layer RoPE, Per-Layer Embeddings, K=V sharing, and KV shared layers.

### Phase 2 — Remaining Dense Variants (31B, E4B)

Validate 31B and E4B with Phase 1 infrastructure. 31B is a strict subset (no
PLE/K=V/KV sharing). E4B is the same architecture as E2B. Primarily validation,
not new code.

### Phase 3 — MoE (26B)

Add Mixture-of-Experts feed-forward dispatch. Largest scope change.

---

## Phase 1 — E2B End-to-End

### Files to Modify

#### 1. `crates/mil-rs/src/weights.rs`

**Add `"gemma4"` and `"gemma4_text"` aliases to `Architecture::from_str`:**

```rust
fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s.to_lowercase().as_str() {
        "llama" | "llama2" | "llama3" | "codellama" | "mistral" => Ok(Architecture::Llama),
        "qwen" | "qwen2" | "qwen3" => Ok(Architecture::Qwen),
        "gemma" | "gemma2" | "gemma3" | "gemma4" | "gemma4_text" => Ok(Architecture::Gemma),
        _ => Err(MilError::Validation(format!(
            "unsupported architecture: {s}"
        ))),
    }
}
```

No new `Architecture` enum variant needed — Gemma 4's text decoder uses the same
basic structure (embedding → transformer layers → norm → LM head). Differentiation
happens via config fields in `extra`.

**Add helper method for Gemma 4 layer types:**

```rust
/// Per-layer-type RoPE configuration for Gemma 4.
pub struct RopeLayerConfig {
    pub theta: f64,
    pub partial_rotary_factor: f64,
}

impl ModelConfig {
    /// Extract per-layer attention types from Gemma 4 config.
    ///
    /// Returns `None` for non-Gemma-4 models. When present, each entry
    /// is `"sliding_attention"` or `"full_attention"`.
    pub fn layer_types(&self) -> Option<Vec<String>> {
        let val = self.extra.get("layer_types")?;
        let arr = val.as_array()?;
        let types: Option<Vec<String>> = arr.iter().map(|v| v.as_str().map(String::from)).collect();
        types
    }

    /// Get per-layer-type RoPE parameters from Gemma 4 config.
    ///
    /// Returns a map from layer type name to its RoPE configuration.
    /// `partial_rotary_factor` of 1.0 means full rotation (standard RoPE).
    pub fn rope_parameters(&self) -> Option<HashMap<String, RopeLayerConfig>> {
        let val = self.extra.get("rope_parameters")?;
        let obj = val.as_object()?;
        let mut result = HashMap::new();
        for (key, params) in obj {
            let theta = params.get("rope_theta").and_then(|v| v.as_f64()).unwrap_or(10000.0);
            let partial_rotary_factor = params.get("partial_rotary_factor").and_then(|v| v.as_f64()).unwrap_or(1.0);
            result.insert(key.clone(), RopeLayerConfig { theta, partial_rotary_factor });
        }
        Some(result)
    }

    /// Get the global head dim for Gemma 4 full-attention layers.
    /// Falls back to `self.head_dim` if not specified.
    pub fn global_head_dim(&self) -> usize {
        self.extra.get("global_head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(self.head_dim)
    }

    /// Get the number of KV heads for global attention layers.
    /// Falls back to `self.num_key_value_heads` if not specified.
    pub fn num_global_key_value_heads(&self) -> usize {
        self.extra.get("num_global_key_value_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(self.num_key_value_heads)
    }

    /// Get the number of Q heads for global attention layers.
    ///
    /// When `global_head_dim != head_dim`, the Q head count changes to
    /// maintain `hidden_size = num_attention_heads * head_dim`. Falls back to
    /// `self.num_attention_heads` when `global_head_dim` is not set or equals
    /// `head_dim`.
    pub fn num_global_attention_heads(&self) -> usize {
        let global_hd = self.global_head_dim();
        if global_hd == self.head_dim {
            self.num_attention_heads
        } else {
            self.hidden_size / global_hd
        }
    }
}
```

#### 2. `crates/ironmill-compile/src/weights/safetensors.rs`

**Handle nested `text_config` in `parse_hf_config`:**

Gemma 4 multimodal configs have `model_type: "gemma4"` at the top level, with
the text decoder config nested under `text_config`. When we detect
`model_type == "gemma4"`, we should extract and parse from `text_config` instead,
since ironmill only handles the text decoder.

```rust
// In parse_hf_config, after parsing model_type:
let (model_type, json_root) = if model_type == "gemma4" {
    // Multimodal wrapper — drill into text_config
    let text_config = json.get("text_config").ok_or_else(|| {
        MilError::Validation("gemma4 config.json missing 'text_config'".into())
    })?;
    let inner_type = text_config.get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("gemma4_text");
    (inner_type, text_config)
} else {
    (model_type, &json)
};
// Then parse all fields from json_root instead of json
```

This ensures that when loading from a full multimodal checkpoint directory,
we correctly read `hidden_size`, `num_hidden_layers`, etc. from the text config.

The `extra` field collection already preserves all unknown keys (including
`layer_types`, `rope_parameters`, `global_head_dim`, etc.), so no changes
needed there — they'll flow through automatically.

#### 3. `crates/ironmill-compile/src/templates/shared.rs`

**Add per-layer RoPE table emission:**

The current `emit_rope_tables` precomputes a single set of cos/sin tables using
`config.rope_theta` and `config.head_dim`. Gemma 4 needs different RoPE tables
per layer type:
- Sliding layers: θ=10,000, full `head_dim`
- Global layers: θ=1,000,000, partial rotation (only first `head_dim * 0.25` dims rotated)

```rust
/// Precompute cos/sin frequency tables for a specific RoPE configuration.
/// Returns `(cos_table_name, sin_table_name)`.
pub(super) fn emit_rope_tables_with_params(
    block: &mut Block,
    head_dim: usize,
    max_pos: usize,
    theta: f64,
    partial_rotary_factor: f64,
    name_prefix: &str,
) -> (String, String) {
    let rotary_dim = ((head_dim as f64 * partial_rotary_factor) as usize / 2) * 2;
    let half_dim = rotary_dim / 2;
    // ... same logic as emit_rope_tables but using rotary_dim, half_dim, theta
    // Non-rotated dimensions are left as identity (cos=1, sin=0)
}
```

**Add partial-rotation-aware RoPE application:**

When `partial_rotary_factor < 1.0`, only the first `rotary_dim` dimensions of
Q/K get rotated. The remaining dimensions pass through unchanged.

```rust
/// Apply RoPE with partial rotation support.
/// When rotary_dim < head_dim, only the first rotary_dim dimensions are rotated.
pub(super) fn emit_rotary_embedding_partial(
    block: &mut Block,
    q_name: &str,
    k_name: &str,
    head_dim: usize,
    rotary_dim: usize,
    layer_idx: usize,
    cos_table: &str,
    sin_table: &str,
) -> (String, String) {
    if rotary_dim == head_dim {
        // Full rotation — delegate to existing emit_rotary_embedding logic
    } else {
        // Split Q/K into rotated and pass-through parts
        // Apply RoPE to rotated part
        // Concatenate rotated + pass-through
    }
}
```

**Add per-layer attention config to `LayerContext`:**

`emit_attention_core` reads `config.head_dim`, `config.num_attention_heads`, and
`config.num_key_value_heads` directly to reshape Q/K/V projections. Since global
layers may use different dimensions, `LayerContext` must carry per-layer effective
values so `emit_attention_core` can use them instead of the global config.

```rust
pub(super) struct LayerContext<'a> {
    pub provider: &'a dyn WeightProvider,
    pub config: &'a ModelConfig,
    pub layer_idx: usize,
    pub rope_cos: &'a str,
    pub rope_sin: &'a str,
    /// Per-layer attention type for Gemma 4 ("sliding_attention" or "full_attention").
    /// None for non-Gemma-4 models.
    pub layer_type: Option<&'a str>,
    /// Effective head dimension for this layer's attention.
    /// Global layers may use `global_head_dim` (e.g. 512) instead of `head_dim` (e.g. 256).
    pub effective_head_dim: usize,
    /// Effective number of Q heads for this layer's attention.
    /// Must satisfy `hidden_size = effective_num_attention_heads * effective_head_dim`.
    pub effective_num_attention_heads: usize,
    /// Effective number of KV heads for this layer's attention.
    /// Global layers may use `num_global_key_value_heads`.
    pub effective_num_kv_heads: usize,
}
```

This is a backwards-compatible extension — existing templates set `layer_type: None`,
`effective_head_dim: config.head_dim`, `effective_num_attention_heads: config.num_attention_heads`,
and `effective_num_kv_heads: config.num_key_value_heads`.

**Update `emit_attention_core` to use `LayerContext` overrides:**

`emit_attention_core` currently reads `config.head_dim`, `config.num_attention_heads`,
and `config.num_key_value_heads` directly for Q/K/V reshape. Update it to read
`ctx.effective_head_dim`, `ctx.effective_num_attention_heads`, and
`ctx.effective_num_kv_heads` instead. The weight tensors loaded from the provider
already have the correct shapes per layer, so only the reshape dimensions need
overriding.

#### 4. `crates/ironmill-compile/src/templates/gemma.rs`

**Update `build_program` to handle Gemma 4 per-layer attention:**

```rust
pub fn build_program(provider: &dyn WeightProvider) -> Result<ConversionResult, MilError> {
    let config = provider.config().clone();
    // ...existing setup...

    // Gemma 4: check for per-layer attention types
    let layer_types = config.layer_types();
    let rope_params = config.rope_parameters();

    // Emit RoPE tables — Gemma 4 needs separate tables per attention type
    let (default_cos, default_sin, global_cos, global_sin) = if let Some(ref rp) = rope_params {
        // Emit sliding-attention RoPE (default)
        let (sc, ss) = if let Some(cfg) = rp.get("sliding_attention") {
            emit_rope_tables_with_params(block, config.head_dim, max_pos, cfg.theta, cfg.partial_rotary_factor, "rope_sliding")
        } else {
            emit_rope_tables(block, &config)
        };
        // Emit full-attention RoPE
        let global_head_dim = config.global_head_dim();
        let (gc, gs) = if let Some(cfg) = rp.get("full_attention") {
            emit_rope_tables_with_params(block, global_head_dim, max_pos, cfg.theta, cfg.partial_rotary_factor, "rope_global")
        } else {
            (sc.clone(), ss.clone())
        };
        (sc, ss, Some(gc), Some(gs))
    } else {
        let (c, s) = emit_rope_tables(block, &config);
        (c, s, None, None)
    };

    // Transformer layers
    for layer_idx in 0..config.num_hidden_layers {
        let lt = layer_types.as_ref().map(|lts| lts[layer_idx].as_str());
        let is_global = lt == Some("full_attention");

        // Select effective per-layer dimensions and RoPE tables
        let (eff_cos, eff_sin) = if is_global {
            (global_cos.as_deref().unwrap_or(&default_cos),
             global_sin.as_deref().unwrap_or(&default_sin))
        } else {
            (default_cos.as_str(), default_sin.as_str())
        };
        let eff_head_dim = if is_global { config.global_head_dim() } else { config.head_dim };
        let eff_num_attention_heads = if is_global {
            config.num_global_attention_heads()
        } else {
            config.num_attention_heads
        };
        let eff_num_kv_heads = if is_global {
            config.num_global_key_value_heads()
        } else {
            config.num_key_value_heads
        };

        let ctx = LayerContext {
            provider,
            config: &config,
            layer_idx,
            rope_cos: eff_cos,
            rope_sin: eff_sin,
            layer_type: lt,
            effective_head_dim: eff_head_dim,
            effective_num_attention_heads: eff_num_attention_heads,
            effective_num_kv_heads: eff_num_kv_heads,
        };
        hidden = emit_gemma_transformer_layer(block, &ctx, &hidden, &mut warnings)?;
    }
    // ...rest unchanged...
}
```

**Update `emit_gemma_transformer_layer` to use per-layer attention config:**

When `ctx.layer_type == Some("full_attention")`, the attention block uses different
dimensions. Since `LayerContext` now carries the effective values, `emit_attention_core`
reads `ctx.effective_head_dim`, `ctx.effective_num_attention_heads`, and
`ctx.effective_num_kv_heads` instead of `config.head_dim`, `config.num_attention_heads`,
and `config.num_key_value_heads`. The RoPE tables are already correct via
`ctx.rope_cos`/`ctx.rope_sin`. No separate `AttentionParams` struct is needed —
`LayerContext` is the single source of per-layer config.

Note: `emit_attention_core` currently reads `config.head_dim` (at `shared.rs:589-623`),
`config.num_attention_heads`, and `config.num_key_value_heads` for Q/K/V reshape
operations. These reads must be replaced with `ctx.effective_head_dim`,
`ctx.effective_num_attention_heads`, and `ctx.effective_num_kv_heads`. The weight
tensor shapes from the provider are already correct per-layer, so only the reshape
and RoPE dimensions need the override.

#### 5. `crates/ironmill-compile/src/weights/gguf.rs`

**Add Gemma 4 guard to `extract_model_config`.**

The `arch_str` parsed from `general.architecture` is lowercased and passed to
`Architecture::from_str`, so the alias addition in `weights.rs` covers architecture
detection automatically. However, `extract_model_config` returns
`extra: HashMap::new()` — it does not populate the `extra` field with unknown
GGUF metadata keys. This means `config.layer_types()`, `config.rope_parameters()`,
`config.global_head_dim()`, and `config.num_global_key_value_heads()` will all
return default/fallback values, silently producing a model with **wrong RoPE on
every global layer**.

**Mitigation**: Add a guard that rejects Gemma 4 from GGUF until proper metadata
extraction is implemented:

```rust
// In extract_model_config, after architecture parsing:
if arch_str == "gemma4" || arch_str == "gemma4_text" {
    return Err(MilError::Validation(
        "Gemma 4 is not yet supported via GGUF — use safetensors format. \
         GGUF lacks the per-layer attention metadata (layer_types, rope_parameters, \
         global_head_dim) required for correct compilation.".into()
    ));
}
```

A follow-up can add proper GGUF metadata extraction by reading Gemma 4-specific
keys (e.g., `gemma4.layer_types`, `gemma4.rope.global_freq_base`) into `extra`
once the GGUF key format is standardized by llama.cpp.

### Trivial Updates Required

- `crates/ironmill-compile/src/templates/llama.rs` — Update `LayerContext`
  construction to add new fields with defaults:
  `layer_type: None, effective_head_dim: config.head_dim, effective_num_attention_heads: config.num_attention_heads, effective_num_kv_heads: config.num_key_value_heads`
- `crates/ironmill-compile/src/templates/qwen.rs` — Same `LayerContext` default
  field additions as llama.rs.

### Files NOT Changed (Phase 1)

- `crates/ironmill-compile/src/templates/mod.rs` — No changes. Dispatch already
  routes `Architecture::Gemma` to `gemma::build_program`.
- `crates/mil-rs/src/ir/passes/` — Weight quantization passes are IR-level,
  architecture-agnostic. No changes needed.
- `crates/ironmill-inference/src/metal/turboquant/` — Deferred to follow-up.

### MoE Config Guard

Phase 1 should error if the config enables MoE, which is deferred to Phase 3.

Add to `build_program` in `gemma.rs`:

```rust
// Guard against Phase 3 MoE (not yet implemented)
if config.extra.get("enable_moe_block").and_then(|v| v.as_bool()).unwrap_or(false) {
    return Err(MilError::Validation(
        "Gemma 4 MoE (enable_moe_block=true) is not yet supported".into()
    ));
}
```

### `use_double_wide_mlp` Handling

The HF config includes `"use_double_wide_mlp": false`. When `true`, this doubles
the MLP intermediate size (using `2 * intermediate_size` for the gate/up projection).

### `hidden_activation` Variant

The config specifies `"hidden_activation": "gelu_pytorch_tanh"`, which is the
GELU approximation `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.
The existing `emit_mlp_gelu` helper (`shared.rs:839`) must be validated to
confirm it uses this tanh approximation, not the exact GELU
(`x * Φ(x)` via `erf`). If it uses exact GELU, update it or add a variant.

### Logit Softcapping

Gemma 3 supported `attn_logit_softcapping` and `final_logit_softcapping`. The
existing Gemma template (`gemma.rs`) does not currently apply softcapping.
Validate against the Gemma 4 config whether these fields are present:
- If absent or `null`: no action needed.
- If present with non-null values: add softcapping support in Phase 1.

**Important**: The actual E2B checkpoint may use `use_double_wide_mlp: true`. This
must be checked against the real config before implementation begins:

- If E2B uses `false`: no changes needed for Phase 1.
- If E2B uses `true`: implement the double-wide MLP in Phase 1 by passing
  `2 * intermediate_size` to `emit_mlp_gelu` for the gate/up projections. The
  weight tensors will already have the correct doubled shape.

For the MoE guard (Phase 3), add a temporary guard only for MoE + double-wide
combinations if needed.

### Per-Layer Sliding Window During Prefill

For decode (seq_len=1), the sliding window distinction is irrelevant — all layers
see only the current token. For prefill (seq_len > 1), sliding and global layers
need different attention windows.

The Metal attention shaders already enforce sliding window via the `window_size`
parameter (`attention.metal:240-249`): when `window_size > 0`, the shader computes
`kv_start = max(0, causal_len - window_size)` to restrict the attention window.
When `window_size == 0`, no restriction is applied (full causal attention).

**Phase 1 approach**: Emit per-layer `layer_types` as a program attribute so the
inference engine can pass the correct `window_size` per layer:

```rust
if let Some(ref lts) = layer_types {
    let types_str = lts.join(",");
    program.set_attribute("layer_types", types_str);
}
```

At inference time, the layer loop passes:
- Sliding layers: `window_size = config.sliding_window()` (e.g. 512)
- Global layers: `window_size = 0` (full causal, no restriction)

No shader changes or CPU-side mask tensors are needed — the existing shader
`window_size` parameter handles this directly.

### Per-Layer Embeddings (PLE)

E2B/E4B use `vocab_size_per_layer_input` and `hidden_size_per_layer_input` to
add a small per-layer embedding lookup at each decoder layer. This is a residual
embedding added to the hidden state before each layer.

**Files to modify (compile):**
- `crates/ironmill-compile/src/templates/gemma.rs` — Add PLE emission in the
  layer loop when `hidden_size_per_layer_input > 0`.
- `crates/ironmill-compile/src/templates/shared.rs` — Add `emit_per_layer_embedding`
  and `emit_gather` helpers. `emit_gather` is a new function (does not exist yet)
  that emits a MIL `gather` op to index an embedding table by token IDs.

**`input_ids` availability in the layer loop:**

PLE requires access to the original `input_ids` at every layer for the gather
operation. Currently, `input_ids` is consumed by the initial `emit_embedding` call
and is not referenced again. To make it available throughout the layer loop:

1. In `build_program`, the `input_ids` block argument name (returned by
   `block.add_argument(...)`) must be preserved as a variable and passed into
   the PLE helper.
2. At inference time, the `input_ids_buf` Metal buffer is already available in
   `run_pipeline_inner()` for the entire forward pass — no additional threading
   is needed on the inference side.

**Compile-time emission:**

In the layer loop of `build_program`, before the transformer layer:

```rust
// Per-Layer Embedding (Gemma 4 E2B/E4B)
let ple_hidden_size = config.extra.get("hidden_size_per_layer_input")
    .and_then(|v| v.as_u64()).unwrap_or(0) as usize;
if ple_hidden_size > 0 {
    hidden = emit_per_layer_embedding(block, provider, layer_idx, &hidden, &input_ids_name, ple_hidden_size)?;
}
```

```rust
/// Emit a MIL `gather` operation to index into an embedding table.
///
/// Loads the embedding weight from the provider and gathers rows
/// corresponding to the given indices tensor.
pub(super) fn emit_gather(
    block: &mut Block,
    provider: &dyn WeightProvider,
    weight_prefix: &str,
    indices: &str,
    out_name: &str,
) -> Result<String, MilError> {
    let table = emit_const(block, provider, &format!("{weight_prefix}.weight"), out_name)?;
    let gathered = block.add_op("gather", &format!("{out_name}_gather"), &[&table, indices]);
    Ok(gathered)
}

/// Emit a per-layer embedding lookup + projection + residual add.
///
/// Looks up `input_ids` in a per-layer embedding table, projects to hidden_size,
/// and adds the result to the current hidden state.
pub(super) fn emit_per_layer_embedding(
    block: &mut Block,
    provider: &dyn WeightProvider,
    layer_idx: usize,
    hidden: &str,
    input_ids_name: &str,
    ple_hidden_size: usize,
) -> Result<String, MilError> {
    let prefix = format!("model.layers.{layer_idx}");
    // 1. Gather from per-layer embedding table using input_ids
    let embed = emit_gather(block, provider, &format!("{prefix}.per_layer_input"),
                            input_ids_name, &format!("l{layer_idx}_ple_embed"))?;
    // 2. Project from ple_hidden_size to hidden_size
    let proj = emit_linear(block, provider, &format!("{prefix}.per_layer_proj"),
                           &embed, &format!("l{layer_idx}_ple_proj"), &mut vec![])?;
    // 3. Residual add
    let out = emit_residual_add(block, hidden, &proj,
                                &format!("l{layer_idx}_ple_residual"))?;
    Ok(out)
}
```

Weight tensor names (expected pattern):
```
model.layers.{i}.per_layer_input.weight   # [vocab_size_per_layer_input, hidden_size_per_layer_input]
model.layers.{i}.per_layer_proj.weight     # [hidden_size_per_layer_input, hidden_size]
```

**Files to modify (inference):**
- `crates/ironmill-inference/src/metal/inference.rs` — In the layer loop, encode
  PLE gather + projection + residual add before the attention block when
  PLE weights are present.

### K=V Sharing

When `attention_k_eq_v` is true, the key projection output is reused as the
value projection. This saves one linear projection per layer.

**Files to modify (compile):**
- `crates/ironmill-compile/src/templates/shared.rs` — In `emit_attention`
  (not `emit_attention_core`, which receives pre-projected Q/K/V tensor names),
  when `attention_k_eq_v` is set, skip the V linear projection and alias K
  output as V:

```rust
// In emit_attention (shared.rs:538), before calling emit_attention_core:
let v = if ctx.config.extra.get("attention_k_eq_v")
    .and_then(|v| v.as_bool()).unwrap_or(false) {
    k.clone() // Reuse K projection output as V
} else {
    emit_linear(block, ctx.provider, &format!("{prefix}.v_proj"),
                input, &format!("l{}_v_proj", ctx.layer_idx), warnings)?
};
```

**Files to modify (inference):**
- `crates/ironmill-inference/src/metal/inference.rs` — In
  `encode_kv_cache_and_attention`, when K=V is enabled, skip V projection
  dispatch and copy K buffer to V slot (or alias the same buffer).

### KV Shared Layers

When `num_kv_shared_layers > 0`, consecutive decoder layers share the same KV
projections. Layer `i` reuses KV from layer `i - (i % num_kv_shared_layers)`.

> **Validation required**: Confirm this grouping logic against HuggingFace's
> `modeling_gemma4.py`. Some implementations use
> `(layer_idx // num_kv_shared_layers) * num_kv_shared_layers` as the anchor,
> which has subtly different behavior at boundaries. The anchor detection formula
> below assumes layer 0 is always an anchor and groups are `[0,1], [2,3], ...`.

**Files to modify (compile):**
- `crates/ironmill-compile/src/templates/gemma.rs` — Track KV projection outputs
  across layers and reuse when sharing is active. Only emit K/V projections for
  "anchor" layers (where `layer_idx % num_kv_shared_layers == 0`):

```rust
let num_kv_shared = config.extra.get("num_kv_shared_layers")
    .and_then(|v| v.as_u64()).unwrap_or(0) as usize;

// In layer loop:
let is_kv_anchor = num_kv_shared == 0 || (layer_idx % num_kv_shared == 0);
// Pass is_kv_anchor to LayerContext; emit_attention skips K/V
// projections and references the anchor layer's outputs when !is_kv_anchor.
```

**Files to modify (inference):**
- `crates/ironmill-inference/src/metal/inference.rs` — Similar to CLA anchor
  layers, skip KV projection for non-anchor layers and reference the anchor
  layer's KV cache entries. The existing `kv_buffer_for_layer` pattern in
  `Fp16KvCache` provides the right abstraction.

### Inference Runtime Changes

The inference engine (`ironmill-inference`) must be updated to support Gemma 4
features. The engine uses an explicit layer loop in `run_pipeline_inner()`
(at `inference.rs:755-1029`) and reads model dimensions from `ModelConfig`.

#### Per-Layer Head Dimensions

The engine currently uses global `mc.head_dim` and `mc.num_key_value_heads` for
all layers. Gemma 4 global layers may use different values.

**Files to modify:**
- `crates/ironmill-inference/src/metal/inference.rs`
- `crates/ironmill-inference/src/metal/config.rs`

**Changes:**

1. Add per-layer config to the inference config:

```rust
/// Per-layer attention configuration for models with heterogeneous layers.
pub struct LayerAttentionConfig {
    pub head_dim: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub window_size: usize,
    pub is_global: bool,
    /// Index into the compiled RoPE frequency tables.
    /// 0 = sliding (default), 1 = global (if different from sliding).
    pub rope_table_index: usize,
}
```

2. In `run_pipeline_inner()`, select per-layer dimensions in the layer loop:

```rust
for layer_idx in 0..mc.num_hidden_layers {
    let layer_cfg = &layer_configs[layer_idx];
    // Use layer_cfg.head_dim, layer_cfg.num_kv_heads for attention dispatch
    // Use layer_cfg.window_size for KV cache window
    // Use layer_cfg.rope_table_index to select correct RoPE cos/sin buffers
    ...
}
```

3. Populate `layer_configs` from `ModelConfig` at model load time, using
   `layer_types()`, `global_head_dim()`, `num_global_attention_heads()`,
   `num_global_key_value_heads()`, and `sliding_window()`.

#### Per-Layer RoPE at Inference Time

The inference engine must apply the correct RoPE frequency tables per layer. The
compile stage emits separate cos/sin constant tensors for sliding vs global RoPE.
At inference, the engine must:

1. Load both sets of RoPE tables (sliding and global) into Metal buffers at
   model init time.
2. In the layer loop, use `layer_cfg.rope_table_index` to select the correct
   cos/sin buffers when encoding the RoPE application.
3. For partial rotation (global layers with `partial_rotary_factor < 1.0`),
   the RoPE shader must apply rotation to only the first `rotary_dim` dimensions
   and pass the remaining dimensions through unchanged. This may require a
   shader variant or an additional `rotary_dim` parameter in `LayerAttentionConfig`.

#### KV Cache Allocation

The KV cache must be allocated per-layer with potentially different buffer sizes
when `global_head_dim != head_dim` or `num_global_key_value_heads != num_key_value_heads`.

**Changes to `Fp16KvCache`:**
- Accept per-layer `(head_dim, num_kv_heads)` pairs instead of global values
- Allocate buffers with `num_kv_heads * head_dim` per layer
- The existing per-layer `window_sizes` infrastructure already handles
  variable-size allocations; extend it to variable dimensions

#### Metal Pipeline Compilation

`MetalPipelines::compile` currently takes a single `head_dim`. If global layers
use `global_head_dim=512`, a second set of attention kernels must be compiled
for that head dimension.

```rust
// In MetalPipelines::compile:
let pipelines_local = compile_attention_pipelines(&device, head_dim)?;
let pipelines_global = if global_head_dim != head_dim {
    Some(compile_attention_pipelines(&device, global_head_dim)?)
} else {
    None
};
```

#### Per-Layer Embedding (PLE) in Inference

Add PLE dispatch in the layer loop before attention:

```rust
if ple_hidden_size > 0 {
    // 1. Gather from per-layer embedding table
    encode_gather(&encoder, &ple_weights[layer_idx], &input_ids_buf, &ple_buf);
    // 2. Linear projection
    encode_linear(&encoder, &ple_proj_weights[layer_idx], &ple_buf, &ple_proj_buf);
    // 3. Residual add to hidden state
    encode_add(&encoder, &hidden_buf, &ple_proj_buf, &hidden_buf);
    encoder.memory_barrier_with_resources(&[&hidden_buf]);
}
```

#### Sliding Window in Metal Shaders

Per-layer window sizes are passed to the existing shader `window_size` parameter
as described in the "Per-Layer Sliding Window During Prefill" section above.
No shader changes needed.

### Validation

**E2B end-to-end test:**

```rust
#[test]
#[ignore] // Requires E2B checkpoint download
fn gemma4_e2b_forward_pass() {
    let model_path = std::env::var("GEMMA4_E2B_PATH")
        .expect("Set GEMMA4_E2B_PATH to Gemma 4 E2B checkpoint directory");
    // 1. Load and compile
    let compiled = compile_model(&model_path).expect("E2B compilation should succeed");
    // 2. Initialize inference engine
    let engine = MetalInference::new(&compiled).expect("E2B inference setup should succeed");
    // 3. Run a short forward pass
    let input_ids = vec![2u32, 1234, 5678]; // BOS + 2 tokens
    let logits = engine.forward(&input_ids).expect("E2B forward pass should succeed");
    // 4. Verify output shape
    assert_eq!(logits.len(), 262144); // vocab_size
    // 5. Verify logits are not NaN/Inf
    assert!(logits.iter().all(|x| x.is_finite()), "logits contain NaN/Inf");
}
```

### Tests

#### Unit Tests in `crates/mil-rs/src/weights.rs`

```rust
#[test]
fn architecture_from_str_gemma4() {
    assert_eq!("gemma4".parse::<Architecture>().unwrap(), Architecture::Gemma);
    assert_eq!("gemma4_text".parse::<Architecture>().unwrap(), Architecture::Gemma);
}

#[test]
fn layer_types_extraction() {
    let mut config = /* minimal config */;
    config.extra.insert("layer_types".into(), serde_json::json!([
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention"
    ]));
    let lt = config.layer_types().unwrap();
    assert_eq!(lt.len(), 6);
    assert_eq!(lt[5], "full_attention");
}

#[test]
fn rope_parameters_extraction() {
    let mut config = /* minimal config */;
    config.extra.insert("rope_parameters".into(), serde_json::json!({
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1000000.0}
    }));
    let rp = config.rope_parameters().unwrap();
    let sliding = &rp["sliding_attention"];
    assert_eq!(sliding.theta, 10000.0);
    assert_eq!(sliding.partial_rotary_factor, 1.0);
    let global = &rp["full_attention"];
    assert_eq!(global.theta, 1000000.0);
    assert_eq!(global.partial_rotary_factor, 0.25);
}
```

#### Unit Tests in `crates/ironmill-compile/src/templates/gemma.rs`

```rust
#[test]
fn build_program_gemma4_with_layer_types() {
    let mut config = tiny_gemma_config();
    config.extra.insert("layer_types".into(), serde_json::json!([
        "sliding_attention", "full_attention"
    ]));
    config.extra.insert("rope_parameters".into(), serde_json::json!({
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1000000.0}
    }));
    config.num_hidden_layers = 2;
    let provider = StubProvider::new(config)
        .with_gemma_weights()
        .with_global_attention_weights(512, 4); // global_head_dim=512, global_kv_heads=4
    let result = build_program(&provider).expect("Gemma 4 build should succeed");
    assert_eq!(result.program.functions.len(), 1);
}

#[test]
fn build_program_gemma4_model_type_accepted() {
    // Verify "gemma4" model_type is accepted
    assert!(Architecture::from_str("gemma4").is_ok());
    assert!(Architecture::from_str("gemma4_text").is_ok());
}

#[test]
fn build_program_gemma4_with_ple() {
    let mut config = tiny_gemma_config();
    config.extra.insert("hidden_size_per_layer_input".into(), serde_json::json!(256));
    config.extra.insert("vocab_size_per_layer_input".into(), serde_json::json!(262144));
    config.num_hidden_layers = 2;
    let provider = StubProvider::new(config)
        .with_llama_weights()
        .with_ple_weights(256, 262144); // per-layer embedding + projection weights
    let result = build_program(&provider).expect("Gemma 4 PLE build should succeed");
    assert_eq!(result.program.functions.len(), 1);
}

#[test]
fn build_program_gemma4_with_kv_sharing() {
    let mut config = tiny_gemma_config();
    config.extra.insert("attention_k_eq_v".into(), serde_json::json!(true));
    config.num_hidden_layers = 2;
    let provider = StubProvider::new(config).with_llama_weights();
    let result = build_program(&provider).expect("Gemma 4 K=V build should succeed");
    // Verify no V projection ops were emitted
    let main = &result.program.functions[0].body;
    let v_proj_ops: Vec<_> = main.ops.iter()
        .filter(|op| op.name.contains("v_proj"))
        .collect();
    assert!(v_proj_ops.is_empty(), "K=V sharing should skip V projections");
}

#[test]
fn build_program_gemma4_with_kv_shared_layers() {
    let mut config = tiny_gemma_config();
    config.extra.insert("num_kv_shared_layers".into(), serde_json::json!(2));
    config.num_hidden_layers = 4;
    let provider = StubProvider::new(config).with_llama_weights();
    let result = build_program(&provider).expect("Gemma 4 KV shared layers build should succeed");
    // Layers 0,2 are anchors; layers 1,3 reuse their KV projections
    assert_eq!(result.program.functions.len(), 1);
}

#[test]
fn build_program_gemma4_rejects_moe() {
    let mut config = tiny_gemma_config();
    config.extra.insert("enable_moe_block".into(), serde_json::json!(true));
    let provider = StubProvider::new(config).with_llama_weights();
    let err = build_program(&provider).unwrap_err();
    assert!(err.to_string().contains("MoE"), "should reject MoE config");
}
```

#### Integration Test in `crates/ironmill-compile/src/weights/safetensors.rs`

```rust
#[test]
fn test_parse_hf_config_gemma4_nested() {
    // Test that gemma4 multimodal config drills into text_config
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("config.json");
    std::fs::write(&config_path, r#"{
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 2304,
            "intermediate_size": 9216,
            "num_hidden_layers": 30,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 262144,
            "layer_types": ["sliding_attention", "full_attention"],
            "rope_parameters": {
                "sliding_attention": {"rope_theta": 10000.0},
                "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25}
            }
        }
    }"#).unwrap();
    let config = parse_hf_config(&config_path).unwrap();
    assert_eq!(config.architecture, Architecture::Gemma);
    assert_eq!(config.hidden_size, 2304);
    assert!(config.layer_types().is_some());
}
```

---

## Phase 2 — Remaining Dense Variants (31B, E4B)

Phase 1 implements all features needed for the E2B on-device model. The remaining
dense variants should work with the same code:

- **31B**: Structurally simpler — no PLE, no K=V sharing, no KV shared layers.
  Uses `hidden_size_per_layer_input: 0`, `attention_k_eq_v: false`,
  `num_kv_shared_layers: 0`. These features are gated by config values, so 31B
  compilation and inference should work with zero additional code.

- **E4B**: Same architecture as E2B (Dense + PLE). Different model dimensions
  but identical code paths.

### Validation

1. Download 31B and E4B checkpoints
2. Compile each variant — verify no errors
3. Run forward pass — verify finite logits with correct `vocab_size`
4. Cross-reference a short generation against HuggingFace transformers output

### TurboQuant Heterogeneous Head Dims

If 31B uses `global_head_dim != head_dim`, TurboQuant KV cache quantization
needs updating (see TurboQuant Follow-Up section). If they happen to be equal
for 31B, TurboQuant works as-is. Validate against the actual checkpoint config.

---

## Phase 3 — MoE (26B)

The 26B variant uses `enable_moe_block = true` with `num_experts` experts and
`top_k_experts` active per token.

### MoE Architecture

Each MoE layer replaces the standard dense MLP with:
1. **Router**: linear projection `[hidden_size, num_experts]` → softmax → top-k
2. **Expert FFN**: `num_experts` independent MLPs, each with
   `moe_intermediate_size` hidden dim
3. **Combine**: weighted sum of top-k expert outputs

### Files to Create

- `crates/ironmill-compile/src/templates/moe.rs` — MoE MLP emission helpers.

### Files to Modify

- `crates/ironmill-compile/src/templates/mod.rs` — Add `pub mod moe;`
- `crates/ironmill-compile/src/templates/gemma.rs` — Dispatch to MoE MLP when
  `enable_moe_block` is set in config.
- `crates/ironmill-compile/src/templates/shared.rs` — Add `emit_router` and
  `emit_expert_dispatch` helpers.

### Weight Tensor Names (expected)

```
model.layers.{i}.mlp.router.weight                    # [hidden_size, num_experts]
model.layers.{i}.mlp.experts.{j}.gate_proj.weight      # [moe_intermediate_size, hidden_size]
model.layers.{i}.mlp.experts.{j}.up_proj.weight        # [moe_intermediate_size, hidden_size]
model.layers.{i}.mlp.experts.{j}.down_proj.weight      # [hidden_size, moe_intermediate_size]
```

### Runtime Considerations

MoE inference on Metal requires either:
- **Dense evaluation**: run all experts, mask unused (simpler, no dynamic dispatch)
- **Sparse evaluation**: dynamic expert selection with scatter/gather (efficient
  but requires new Metal kernels)

The MIL IR representation should use dense evaluation initially, as it maps
cleanly to existing matmul primitives.

### Inference Runtime (MoE)

**Files to modify:**
- `crates/ironmill-inference/src/metal/inference.rs` — In `encode_ffn_block`,
  when MoE is enabled for the current layer, dispatch router + expert evaluation
  instead of the standard dense MLP.

**Dense evaluation approach** (Phase 3 initial):

```rust
// 1. Router: linear projection → softmax → top-k selection
encode_linear(&encoder, &router_weight, &hidden_buf, &router_logits_buf);
encode_softmax(&encoder, &router_logits_buf, &router_probs_buf);
// 2. Run ALL expert MLPs (dense — no dynamic dispatch)
for expert_idx in 0..num_experts {
    encode_expert_ffn(&encoder, &expert_weights[expert_idx],
                      &hidden_buf, &expert_out_bufs[expert_idx]);
}
// 3. Weighted combine using router probs (top-k masking)
encode_moe_combine(&encoder, &expert_out_bufs, &router_probs_buf,
                    top_k, &hidden_buf);
```

**Metal kernel needs:**
- `encode_moe_combine`: weighted sum of top-k expert outputs, zeroing non-selected
  experts. This is the only new kernel — router/softmax/expert-FFN reuse existing
  linear/activation/matmul kernels.

**Future optimization**: Replace dense evaluation with sparse dispatch (only run
selected experts) via a scatter/gather kernel. This reduces compute from
`O(num_experts)` to `O(top_k)` per token but requires a new Metal kernel for
dynamic expert routing.

> **Performance note**: Dense evaluation is a significant compute multiplier.
> For 26B with e.g. 64 experts and `top_k=4`, dense evaluation runs 16× more
> expert FFN compute than necessary. Sparse dispatch is critical for usable
> inference latency and should be prioritized as a fast follow-up after initial
> correctness validation.

---

## TurboQuant Follow-Up (Future)

To fully support TurboQuant with heterogeneous attention layers:

1. Extend `TurboQuantMetalConfig` to support per-layer configs:
   ```rust
   pub struct TurboQuantMetalConfig {
       pub layers: Vec<TurboQuantLayerConfig>,
       // ...shared fields...
   }
   pub struct TurboQuantLayerConfig {
       pub head_dim: usize,
       pub num_kv_heads: usize,
       pub is_global: bool,
   }
   ```
2. Generate separate Metal shader specializations for each unique
   `(head_dim, num_kv_heads)` pair.
3. Allocate KV cache buffers per-layer with correct dimensions.
4. Dispatch to the correct shader per layer during decode.

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Gemma 4 config.json format differs from HF transformers code | Config parsing fails | Validate against actual checkpoint downloads |
| Weight tensor naming differs from expected pattern | Missing weights at load time | Test with real Gemma 4 safetensors; HF naming is standardized |
| `partial_rotary_factor` interacts unexpectedly with GQA | Incorrect attention output | Cross-reference HF transformers `modeling_gemma4.py` implementation |
| MoE routing requires dynamic shapes in MIL IR | IR emission fails | Phase 3 can use dense (all-experts) evaluation to avoid dynamic shapes |
| `vocab_size=262144` exceeds typical sizes | Metal buffer allocation failures or OOM during inference for the embedding table (`[262144, hidden_size]`) | Validate max buffer size against Metal device limits; test on target hardware |
| PLE per-layer embeddings consume ~3.8 GB for E2B | OOM on memory-constrained devices (30 layers × `[262144, 256]` × FP16 ≈ 3.8 GB for PLE tables alone, on top of model weights) | Validate total memory budget on target hardware; consider FP16→INT8 PLE quantization if needed |
| GGUF format lacks Gemma 4 metadata keys | Silent miscompilation with wrong RoPE | Phase 1 adds an explicit error for Gemma 4 via GGUF; follow-up adds proper extraction |
| Prefill masking differs per layer type | Incorrect attention for sliding vs global layers during prefill | Emit `layer_types` as program attribute; runtime constructs per-layer masks |
| `hidden_activation` variant mismatch | Subtly incorrect MLP output (exact GELU vs tanh-approximated GELU) | Validate `emit_mlp_gelu` uses the `gelu_pytorch_tanh` approximation |
