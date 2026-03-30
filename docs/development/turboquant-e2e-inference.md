# TurboQuant End-to-End Inference Integration

> **Status:** Planning
>
> **Prerequisites:**
> - [TurboQuant Implementation](turboquant-implementation.md) — completed
> - [ANE BLOBFILE Investigation](ane-blobfile-investigation.md) — open
>
> **Goal:** Run real autoregressive LLM inference on ANE with TurboQuant
> INT8 KV cache compression, starting from token IDs and producing logits.

## Current State

TurboQuant's attention pipeline is implemented and eval-verified (34/34
ANE checks pass), but it operates in isolation — the caller must provide
pre-computed Q/K/V projections. Real inference requires the full
transformer: embedding → norm → projections → attention → FFN → lm_head.

### What works today

| Component | Status | Notes |
|---|---|---|
| ONNX → MIL IR conversion | ✅ | Qwen3-0.6B loads successfully |
| Sub-program splitting | ✅ | embedding / layer_N / lm_head |
| ANE compile + eval | ✅ | `AneModel::compile_and_load` / `predict` |
| TurboQuant cache-write | ✅ | 34/34 eval, compiles on ANE |
| TurboQuant attention | ✅ | 34/34 eval, GQA tile support |
| TurboQuant QJL correction | ✅ | Compiles on ANE, sign computation implemented |
| INT8 KV cache manager | ✅ | Per-layer IOSurface caches, seq_pos tracking |
| TurboQuant benchmark | ✅ | 11.1ms/token (Qwen3-0.6B arch, dummy weights) |

### What's missing

| Component | Effort | Blocking? |
|---|---|---|
| Hybrid layer splitting | Large | Yes |
| Autoregressive KV cache in AneModel | Large | Yes |
| Model arch extraction from ONNX | Small | Yes |
| Decode loop / generation | Medium | Yes |
| Embedding + lm_head tensor I/O | Small | Yes |

---

## Gap Analysis

### 1. Hybrid layer splitting

**Problem:** `split_for_ane` splits the model into coarse sub-programs
(`embedding`, `layer_0`, `layer_1`, ..., `lm_head`). Each `layer_N`
contains the full transformer block: norm → Q/K/V projections → attention
→ output projection → residual → FFN. TurboQuant needs to intercept at
the attention boundary.

**What's needed:** Split each `layer_N` into three sub-programs:

```
layer_N_pre_attn:   RMSNorm → Q/K/V linear projections
layer_N_attn:       [replaced by TurboQuantModel::step_attention]
layer_N_post_attn:  output projection → residual → FFN → residual
```

**Approach options:**

A. **Op-level split** — Extend `split_for_ane` to detect attention
   boundaries within each layer (look for `GroupQueryAttention` or the
   decomposed attention pattern) and split there. Requires understanding
   the MIL IR structure of each layer.

B. **Graph surgery** — After splitting into `layer_N`, find the attention
   ops, remove them, and insert placeholder inputs/outputs. Run the
   pre-attention and post-attention portions as separate ANE sub-programs
   with TurboQuant handling the middle.

C. **Dual execution** — Run the full `layer_N` sub-program for Q/K/V
   projections + FFN on ANE, but separately run TurboQuant attention.
   This requires extracting Q/K/V projection outputs from the layer
   sub-program (add them as additional outputs) and feeding TurboQuant's
   attention output back in.

Option C is likely fastest to implement since it doesn't require
restructuring the sub-program splitter.

### 2. Autoregressive KV cache management

**Problem:** `AneModel::predict()` is stateless — it runs a single
forward pass with no state between calls. The ONNX model has
`past_key_values.N.key/value` inputs and `present.N.key/value` outputs,
but the ANE runtime doesn't thread them.

**What's needed:**

```rust
pub struct AneAutoregressive {
    model: AneModel,  // or sub-program collection
    kv_cache: KvCacheManager,  // from TurboQuant
    seq_pos: usize,
}

impl AneAutoregressive {
    /// Prefill: process prompt tokens (can batch)
    pub fn prefill(&mut self, token_ids: &[u32]) -> Result<Vec<f32>>;
    
    /// Decode: generate one token
    pub fn decode(&mut self, token_id: u32) -> Result<Vec<f32>>;
    
    /// Reset for new conversation
    pub fn reset(&mut self);
}
```

Two approaches:

A. **ONNX-level**: Pass `past_key_values` as function inputs, receive
   `present` as outputs, copy between calls. The `KvCachePass` already
   detects these inputs. This is the standard approach but uses FP16
   caches (no TurboQuant benefit).

B. **TurboQuant-integrated**: Replace the model's internal attention +
   KV cache with TurboQuant's INT8 pipeline. The model's Q/K/V projections
   feed into `TurboQuantModel::step_attention`, which manages the INT8
   cache internally. This is the approach that delivers bandwidth savings.

### 3. Model architecture extraction from ONNX

**Problem:** `TurboQuantConfig::new` requires `num_heads`, `num_kv_heads`,
`head_dim`, `num_layers`. These exist in the ONNX graph as attributes on
`GroupQueryAttention` nodes but aren't extracted.

**Solution:** After ONNX → MIL conversion, walk the Program's ops and
extract architecture parameters:

```rust
pub fn extract_model_arch(program: &Program) -> Option<ModelArch> {
    // Count layer groups (num_layers)
    // Find GroupQueryAttention or attention pattern ops
    // Extract num_heads, kv_num_heads from op attributes
    // Infer head_dim from tensor shapes
}
```

Alternatively, parse the ONNX model's metadata or config.json if
available (SafeTensors path already does this).

**Effort:** Small. The ONNX converter already reads `num_heads` and
`kv_num_heads` from `GroupQueryAttention` attributes — just need to
surface them.

### 4. Decode loop / generation

**Problem:** No token-by-token generation loop exists.

**What's needed:**

```rust
pub fn generate(
    model: &mut AneAutoregressive,
    prompt_tokens: &[u32],
    max_tokens: usize,
    temperature: f32,
) -> Vec<u32> {
    // 1. Prefill prompt
    let mut logits = model.prefill(prompt_tokens)?;
    let mut output = Vec::new();
    
    for _ in 0..max_tokens {
        // 2. Sample next token
        let token = sample(&logits, temperature);
        if token == eos_token { break; }
        output.push(token);
        
        // 3. Decode next token
        logits = model.decode(token)?;
    }
    output
}
```

**Dependencies:** Requires autoregressive state management (#2) and
embedding/lm_head I/O (#5). Sampling itself is straightforward.

### 5. Embedding + lm_head tensor I/O

**Problem:** Real inference starts with token IDs (u32) and ends with
logits (f32). The current pipeline works with fp16 tensors.

**What's needed:**
- **Embedding**: Token ID → gather from embedding weight matrix → fp16 tensor.
  The embedding sub-program from `split_for_ane` already handles this.
- **lm_head**: The final sub-program outputs logits. Need to read fp16
  from the output AneTensor and convert to f32 for sampling.
- **Tokenizer**: Not in scope for ironmill. Assume token IDs are provided.

**Effort:** Small. The tensor I/O primitives exist (`AneTensor::write_f16`,
`read_f16`). Just need the conversion layer.

---

## Recommended Implementation Order

### Phase 1 — Architecture extraction + end-to-end plumbing
1. Extract model arch from ONNX/MIL IR (`num_heads`, `head_dim`, etc.)
2. Build `AneAutoregressive` struct with stateful KV cache
3. Implement prefill (batch forward pass with cache population)
4. Implement single-token decode

### Phase 2 — TurboQuant attention swap
5. Split layer sub-programs to expose Q/K/V projection outputs
6. Wire TurboQuant cache-write + attention into the layer execution loop
7. Thread INT8 KV cache state across decode steps

### Phase 3 — Generation loop + benchmarking
8. Implement sampling (greedy, temperature, top-k/top-p)
9. Build generate() loop
10. Benchmark against FP16 baseline: tokens/sec, memory, perplexity

---

## Architecture Diagram

```
Token IDs
    │
    ▼
┌──────────┐
│ Embedding │  (ANE sub-program)
└────┬─────┘
     │ fp16 hidden states
     ▼
┌──────────────────────────────────────────────┐
│ Layer N                                       │
│                                               │
│  ┌─────────────┐                              │
│  │ RMSNorm     │  (ANE sub-program)           │
│  │ Q/K/V proj  │                              │
│  └──┬──┬──┬────┘                              │
│     Q  K  V                                   │
│     │  │  │                                    │
│     ▼  ▼  ▼                                   │
│  ┌──────────────────┐                         │
│  │ TurboQuant        │                        │
│  │  cache-write (ANE)│ K,V → INT8 → cache     │
│  │  attention (ANE)  │ Q + INT8 cache → output │
│  └────────┬─────────┘                         │
│           │ attn_out                           │
│           ▼                                    │
│  ┌─────────────┐                              │
│  │ O proj      │                              │
│  │ Residual    │  (ANE sub-program)           │
│  │ FFN / MLP   │                              │
│  │ Residual    │                              │
│  └──────┬──────┘                              │
│         │                                      │
└─────────┼──────────────────────────────────────┘
          │ (repeat for all layers)
          ▼
┌──────────┐
│ lm_head  │  (ANE sub-program)
└────┬─────┘
     │ logits
     ▼
  Sampling → next token
```

---

## Test Model

**Qwen3-0.6B** (`tests/fixtures/qwen3-0.6b.onnx`, 300 MB):
- 28 layers
- 14 attention heads, 2 KV heads (GQA 7:1)
- head_dim = 64, hidden_size = 896
- Autoregressive ONNX with past_key_values

This is the right model to target — small enough for fast iteration,
uses GQA (exercises the tile path), and is already in the test fixtures.

## References

- [TurboQuant Implementation](turboquant-implementation.md) — completed INT8 cache pipeline
- [ANE BLOBFILE Investigation](ane-blobfile-investigation.md) — weight delivery workaround
- [ANE Op Support Matrix](../research/ane-op-support-matrix.md) — verified ops
