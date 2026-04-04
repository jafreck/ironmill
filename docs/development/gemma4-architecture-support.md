# Gemma 4 Architecture Support — Implementation Spec

> Adds support for Google's Gemma 4 model family (released April 2, 2026).
> Dense 31B, MoE 26B, and on-device E2B/E4B variants.

## Status

**Under development** — Phase 1 (dense text decoder) first.

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

### Phase 1 — Dense Text Decoder (31B)

Support loading and compiling the Gemma 4 31B dense model. This is the highest
priority since it's the most popular variant and structurally closest to the
existing Gemma template.

### Phase 2 — On-Device Variants (E2B/E4B)

Add Per-Layer Embeddings and K=V sharing for the on-device models.

### Phase 3 — MoE (26B)

Add Mixture-of-Experts feed-forward dispatch. Largest scope change.

---

## Phase 1 — Dense Text Decoder

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
    /// Returns a map from layer type name to `(rope_theta, partial_rotary_factor)`.
    /// `partial_rotary_factor` of 1.0 means full rotation (standard RoPE).
    pub fn rope_parameters(&self) -> Option<HashMap<String, (f64, f64)>> {
        let val = self.extra.get("rope_parameters")?;
        let obj = val.as_object()?;
        let mut result = HashMap::new();
        for (key, params) in obj {
            let theta = params.get("rope_theta").and_then(|v| v.as_f64()).unwrap_or(10000.0);
            let partial = params.get("partial_rotary_factor").and_then(|v| v.as_f64()).unwrap_or(1.0);
            result.insert(key.clone(), (theta, partial));
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
    /// Override RoPE tables for this specific layer type.
    /// None means use the default rope_cos/rope_sin.
    pub rope_cos_override: Option<&'a str>,
    pub rope_sin_override: Option<&'a str>,
}
```

This is a backwards-compatible extension — existing templates set `layer_type: None`
and `rope_cos_override: None`.

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
        let (sc, ss) = if let Some(&(theta, prf)) = rp.get("sliding_attention") {
            emit_rope_tables_with_params(block, config.head_dim, max_pos, theta, prf, "rope_sliding")
        } else {
            emit_rope_tables(block, &config)
        };
        // Emit full-attention RoPE
        let global_head_dim = config.global_head_dim();
        let (gc, gs) = if let Some(&(theta, prf)) = rp.get("full_attention") {
            emit_rope_tables_with_params(block, global_head_dim, max_pos, theta, prf, "rope_global")
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

        let ctx = LayerContext {
            provider,
            config: &config,
            layer_idx,
            rope_cos: &default_cos,
            rope_sin: &default_sin,
            layer_type: lt,
            rope_cos_override: if is_global { global_cos.as_deref() } else { None },
            rope_sin_override: if is_global { global_sin.as_deref() } else { None },
        };
        hidden = emit_gemma_transformer_layer(block, &ctx, &hidden, &mut warnings)?;
    }
    // ...rest unchanged...
}
```

**Update `emit_gemma_transformer_layer` to use per-layer attention config:**

When `ctx.layer_type == Some("full_attention")`, the attention block should use:
- `config.global_head_dim()` instead of `config.head_dim`
- `config.num_global_key_value_heads()` instead of `config.num_key_value_heads`
- The overridden RoPE tables (`ctx.rope_cos_override` / `ctx.rope_sin_override`)

This requires a new `emit_gemma4_attention` helper that wraps `emit_attention_core`
with per-layer dimension overrides, or extending `emit_attention_core` to accept
override head dim / KV head counts.

The cleanest approach is to add an `AttentionParams` struct:

```rust
/// Per-layer attention parameters that may differ from the global ModelConfig.
pub(super) struct AttentionParams {
    pub head_dim: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rope_cos: String,
    pub rope_sin: String,
}
```

And add an `emit_attention_core_with_params` that uses these overrides instead of
reading from `config`.

#### 5. `crates/ironmill-compile/src/weights/gguf.rs`

**Add `"gemma4"` to GGUF architecture parsing in `extract_model_config`.**

The `arch_str` parsed from `general.architecture` is lowercased and passed to
`Architecture::from_str`, so the alias addition in `weights.rs` covers this
automatically. No changes needed in `gguf.rs` itself unless Gemma 4 GGUF files
use different metadata keys (unlikely — GGUF format is standardized).

### Files NOT Changed (Phase 1)

- `crates/ironmill-compile/src/templates/mod.rs` — No changes. Dispatch already
  routes `Architecture::Gemma` to `gemma::build_program`.
- `crates/ironmill-compile/src/templates/llama.rs` — Not affected.
- `crates/ironmill-compile/src/templates/qwen.rs` — Not affected.
- `crates/mil-rs/src/ir/passes/` — Weight quantization passes are IR-level,
  architecture-agnostic. No changes needed.
- `crates/ironmill-inference/src/metal/turboquant/` — Deferred to follow-up.

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
    assert_eq!(rp["sliding_attention"], (10000.0, 1.0));
    assert_eq!(rp["full_attention"], (1000000.0, 0.25));
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
    let provider = StubProvider::new(config).with_llama_weights();
    let result = build_program(&provider).expect("Gemma 4 build should succeed");
    assert_eq!(result.program.functions.len(), 1);
}

#[test]
fn build_program_gemma4_model_type_accepted() {
    // Verify "gemma4" model_type is accepted
    assert!(Architecture::from_str("gemma4").is_ok());
    assert!(Architecture::from_str("gemma4_text").is_ok());
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

## Phase 2 — On-Device Variants (E2B/E4B)

### Per-Layer Embeddings (PLE)

E2B/E4B use `vocab_size_per_layer_input` and `hidden_size_per_layer_input` to
add a small per-layer embedding lookup at each decoder layer. This is a residual
embedding added to the hidden state before each layer.

**Files to modify:**
- `crates/ironmill-compile/src/templates/gemma.rs` — Add PLE emission in the
  layer loop when `hidden_size_per_layer_input > 0`.
- `crates/ironmill-compile/src/templates/shared.rs` — Add `emit_per_layer_embedding`
  helper.

Weight tensor names (expected pattern):
```
model.layers.{i}.per_layer_input.weight   # [vocab_size_per_layer_input, hidden_size_per_layer_input]
model.layers.{i}.per_layer_proj.weight     # [hidden_size_per_layer_input, hidden_size]
```

### K=V Sharing

When `attention_k_eq_v` is true, the key projection output is reused as the
value projection. This saves one linear projection per layer.

**Files to modify:**
- `crates/ironmill-compile/src/templates/gemma.rs` — In attention emission,
  when `attention_k_eq_v` is set, skip V projection and pass K output as V.

### KV Shared Layers

When `num_kv_shared_layers > 0`, consecutive decoder layers share the same KV
projections. Layer `i` reuses KV from layer `i - (i % num_kv_shared_layers)`.

**Files to modify:**
- `crates/ironmill-compile/src/templates/gemma.rs` — Track KV projection outputs
  across layers and reuse when sharing is active.

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
