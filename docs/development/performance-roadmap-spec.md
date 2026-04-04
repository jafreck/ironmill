# Performance Roadmap — Implementation Spec

Implementation spec derived from [performance-roadmap.md](../design/performance-roadmap.md).
Each task is self-contained with file paths, type signatures, acceptance criteria,
and explicit dependencies on other tasks.

Baseline (Qwen3-0.6B, Metal FP16, Apple Silicon):
- Decode: ~35 tok/s
- Prefill 1024 tok: 1287 tok/s
- Prefill 8192 tok: 214 tok/s
- PPL (1024-tok wikitext2): 12.86

---

## Task 1 — Fused SDPA with simdgroup matrix tiling

**Goal:** Replace the current per-(head, query) attention dispatch with a tiled
FlashAttention-style kernel that keeps all intermediate data in SRAM. This is the
single largest bottleneck — at 8192 tokens the GPU is 73% slower than CPU.

See [ADR-0001](../adr/0001-fused-sdpa-metal-kernel.md) for the full design decision.

### Files to create

```
crates/ironmill-inference/src/metal/shaders/fused_sdpa.metal
```

### Files to modify

```
crates/ironmill-inference/src/metal/inference.rs  — replace attention dispatch in encode_kv_cache_and_attention()
crates/ironmill-inference/src/metal/ops.rs         — add pipeline for fused_sdpa, remove encode_prefill_attention()
crates/ironmill-inference/src/metal/config.rs      — replace use_fa2_prefill with fused_sdpa tile config
```

### Kernel signature (Metal Shading Language)

```metal
kernel void fused_sdpa(
    device const half*   Q           [[buffer(0)]],   // [token_count × num_q_heads × head_dim]
    device const half*   K           [[buffer(1)]],   // [seq_len × num_kv_heads × head_dim]
    device const half*   V           [[buffer(2)]],   // [seq_len × num_kv_heads × head_dim]
    device half*         O           [[buffer(3)]],   // [token_count × num_q_heads × head_dim]
    constant uint&       seq_len     [[buffer(4)]],
    constant uint&       token_count [[buffer(5)]],
    constant uint&       head_dim    [[buffer(6)]],
    constant uint&       num_q_heads [[buffer(7)]],
    constant uint&       num_kv_heads[[buffer(8)]],
    constant float&      scale       [[buffer(9)]],   // 1/sqrt(head_dim)
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_id   [[simdgroup_index_in_threadgroup]]
);
```

### Implementation notes

- Tile over Q blocks (Br tokens) × KV tiles (Bc tokens) using 8×8 simdgroup
  matrix operations (`simd_matrix_multiply`).
- Online softmax (Milakov & Gimelshein): track running max and denominator per Q
  row, rescale O accumulator only when the running max changes.
- Adopt FA4's conditional rescaling: skip rescale when new KV tile's max does not
  exceed the running max (~90% of tiles).
- GQA-aware: multiple Q heads within the same KV head group share KV tile loads.
  Dispatch: one threadgroup per (kv_head_group, q_block). Within the threadgroup,
  each simdgroup handles one Q head from the group.
- Causal mask: skip entire KV tiles where all positions are masked.
- Both prefill (token_count > 1) and decode (token_count == 1) paths use the same
  kernel — the decode case is a single Q-row tile.

### Acceptance criteria

- `cargo test -p ironmill-inference` passes (existing correctness tests).
- PPL on wikitext2 1024-tok is within ±0.05 of baseline (12.86).
- Prefill 8192 tok ≥ 800 tok/s (matching CPU baseline, ~3.7× improvement).
- Prefill 2048 tok ≥ 2000 tok/s (≥2.5× improvement).
- Decode throughput does not regress below 33 tok/s.
- No increase in peak memory usage (all intermediates in SRAM/threadgroup memory).

### Dependencies

None.

---

## Task 2 — EAGLE-3 / P-EAGLE speculative decoding

**Goal:** Implement speculative decoding using an EAGLE-3 draft head with
P-EAGLE parallel candidate generation. Target 3–6× decode throughput.

### Files to create

```
crates/ironmill-inference/src/speculative/
├── mod.rs             — SpeculativeEngine wrapper
├── draft_head.rs      — EAGLE-3 draft MLP (multi-layer feature fusion)
├── tree.rs            — tree-structured candidate generation + verification
├── config.rs          — SpecConfig (depth, width, acceptance threshold)
└── specbundle.rs      — SpecBundle checkpoint loader

crates/ironmill-inference/src/metal/shaders/draft_head.metal  — draft head matmul kernel
```

### Files to modify

```
crates/ironmill-inference/src/engine.rs            — add speculative_decode() to InferenceEngine trait
crates/ironmill-inference/src/metal/inference.rs   — implement speculative_decode(), KV cache rollback
crates/ironmill-inference/src/metal/turboquant/cache.rs — use truncate_to(pos) from Task 5 for KV cache rollback
crates/ironmill-inference/src/lib.rs               — re-export speculative module
```

### Key types

```rust
/// Configuration for speculative decoding.
pub struct SpecConfig {
    pub max_draft_depth: usize,      // max tokens per speculation round
    pub tree_width: usize,           // candidates per position
    pub acceptance_threshold: f32,   // min probability for acceptance
}

/// EAGLE-3 draft head: small MLP fusing features from target model layers.
pub struct DraftHead {
    layer_weights: Vec<MetalBuffer>,  // projection weights per fused layer
    hidden_dim: usize,
    vocab_size: usize,
}

impl DraftHead {
    /// Load from a SpecBundle checkpoint.
    pub fn from_specbundle(path: &Path, device: &MetalDevice) -> Result<Self>;

    /// Generate K candidate tokens in a single forward pass (P-EAGLE).
    pub fn draft_parallel(
        &self,
        encoder: &ComputeEncoder,
        hidden_states: &[MetalBuffer],  // features from target model layers
        config: &SpecConfig,
    ) -> Vec<DraftCandidate>;
}

/// Tree-structured candidate with parent pointers.
pub struct DraftCandidate {
    pub token_id: u32,
    pub log_prob: f32,
    pub parent_idx: Option<usize>,
    pub depth: usize,
}
```

### Verification loop (Leviathan et al. rejection sampling)

1. Draft head produces tree of K candidates from fused target-model features.
2. Pack candidate token IDs into a batch and prefill through the target model.
3. For each draft token `x_i` along the best tree path, compare target model
   probability `p_target(x_i)` with draft probability `p_draft(x_i)`:
   - Sample `u ~ Uniform(0, 1)`.
   - **Accept** if `u < p_target(x_i) / p_draft(x_i)` (i.e., with probability
     `min(1, p_target(x_i) / p_draft(x_i))`).
   - **Reject** otherwise — discard `x_i` and all subsequent draft tokens.
4. Sample a **correction token** from the adjusted distribution:
   - If rejected at position `i`: sample from
     `normalize(max(0, p_target(x) - p_draft(x)))` for all tokens `x`.
   - If all K tokens accepted: sample a bonus token from `p_target` directly.
5. Rollback KV cache (via `truncate_to()` from Task 5) to the last accepted
   position + 1 (the correction token position).

This stochastic acceptance preserves the target model's output distribution
exactly, guaranteeing mathematically lossless speculation.

### Acceptance criteria

- Decode throughput ≥ 100 tok/s for Qwen3-0.6B with a matching SpecBundle.
- PPL does not change (speculative decoding is mathematically lossless).
- Graceful fallback: if no SpecBundle is available, falls back to standard decode.
- KV cache rollback is correct: generating N tokens with speculation produces
  identical output to generating N tokens without speculation.
- `cargo test -p ironmill-inference -- speculative` passes.

### Dependencies

- **Task 3** (min-p sampling) — verification requires full sampler chain for
  probability comparison.
- **Task 5** (KV cache reuse) — rollback requires position-aware cache management.

---

## Task 3 — Min-p sampling + full sampler chain

**Goal:** Replace the current temperature-only sampler with a full sampler chain
matching llama.cpp's `llama_sampler` capabilities.

### Files to modify

```
crates/ironmill-inference/src/sampling.rs — rewrite with sampler chain
```

### Key types

```rust
/// Sampler configuration.
pub struct SamplerConfig {
    pub temperature: f32,              // 0.0 = greedy
    pub min_p: f32,                    // 0.0 = disabled (default: 0.05)
    pub top_k: usize,                 // 0 = disabled
    pub top_p: f32,                   // 1.0 = disabled
    pub repeat_penalty: f32,          // 1.0 = disabled
    pub repeat_window: usize,         // tokens to look back for repeat penalty
    pub frequency_penalty: f32,       // 0.0 = disabled
    pub presence_penalty: f32,        // 0.0 = disabled
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig {
            temperature: 0.7,
            min_p: 0.05,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            repeat_window: 64,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }
}

/// Stateful sampler that tracks token history for repetition penalties.
pub struct Sampler {
    config: SamplerConfig,
    recent_tokens: VecDeque<u32>,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self;

    /// Apply the full sampler chain to logits and return a sampled token ID.
    /// Chain order: repetition penalty → top-k → top-p → min-p → temperature → sample.
    pub fn sample(&mut self, logits: &mut [f32]) -> u32;

    /// Reset token history (e.g., on conversation clear).
    pub fn reset(&mut self);
}
```

### Sampler chain (applied in order, matching llama.cpp default)

1. **Repetition/frequency/presence penalty** — scan `recent_tokens` window,
   apply multiplicative penalty for repeat, additive for frequency/presence.
2. **Top-k filtering** — if enabled, keep only top-k logits by value.
3. **Top-p (nucleus) filtering** — if enabled, sort logits descending, compute
   cumulative softmax probabilities, keep the smallest set of tokens whose
   cumulative probability exceeds `top_p`.
4. **Min-p filtering** — compute `threshold = min_p × max(softmax(logits))`,
   zero out all tokens with `softmax(logit) < threshold`.
5. **Temperature scaling** — divide logits by temperature. If temperature ≤ 0,
   short-circuit to argmax.
6. **Categorical sampling** — softmax the surviving logits, sample from the
   resulting distribution.

Note: this order matches llama.cpp's default sampler chain, where truncation
filters (top-k, top-p, min-p) operate on the raw logit distribution before
temperature scaling. This prevents temperature from inflating low-probability
tokens past the min-p threshold.

### Acceptance criteria

- `temperature: 0.0` produces identical output to current greedy path.
- `min_p: 0.0, top_k: 0, top_p: 1.0` with temperature > 0 matches current
  temperature-only sampling behavior.
- Unit tests for each sampler stage in isolation.
- Unit test: min-p with high-confidence logits ([10.0, 1.0, 1.0, ...]) produces
  near-deterministic output (only the top token survives).
- Unit test: min-p with low-confidence logits ([1.0, 1.0, 1.0, ...]) preserves
  most tokens (permissive).
- No GPU kernel changes — all CPU-side logit post-processing.

### Dependencies

None.

---

## Task 4 — RadixAttention prompt caching

**Goal:** Cache computed KV activations in a radix tree so that requests sharing
prompt prefixes skip redundant prefill computation.

### Files to create

```
crates/ironmill-inference/src/cache/
├── mod.rs          — re-exports
├── radix_tree.rs   — RadixTree<Token, KvSlice> implementation
├── prefix_cache.rs — PrefixCache wrapping RadixTree with LRU eviction
└── policy.rs       — eviction policies (LRU per node, memory budget)
```

### Files to modify

```
crates/ironmill-inference/src/engine.rs            — add prefill_with_cache() method
crates/ironmill-inference/src/metal/inference.rs   — implement prefix-aware prefill
crates/ironmill-inference/src/metal/turboquant/cache.rs — add copy_from_slice() for cache restoration
crates/ironmill-inference/src/lib.rs               — re-export cache module
```

### Key types

```rust
/// A token-keyed radix tree node storing a KV cache slice reference.
pub struct RadixNode {
    children: HashMap<u32, Box<RadixNode>>,
    /// KV cache data for this node's token span.
    /// None for the root node.
    kv_slice: Option<KvCacheSlice>,
    last_access: Instant,
    token_span: Vec<u32>,
}

/// Reference to a contiguous range of KV cache positions.
pub struct KvCacheSlice {
    /// Per-layer K and V buffers (CPU copies for cache restoration).
    layer_data: Vec<KvLayerSlice>,
    start_pos: usize,
    len: usize,
}

pub struct PrefixCache {
    root: RadixNode,
    memory_budget: usize,
    current_memory: usize,
}

impl PrefixCache {
    pub fn new(memory_budget: usize) -> Self;

    /// Find the longest cached prefix. Returns the number of tokens matched
    /// and a reference to the cached KV data.
    pub fn lookup(&mut self, tokens: &[u32]) -> (usize, Vec<&KvCacheSlice>);

    /// Insert a new KV cache entry for a token sequence.
    pub fn insert(&mut self, tokens: &[u32], kv_data: KvCacheSlice);

    /// Evict least-recently-used entries until under budget.
    pub fn evict_to_budget(&mut self);
}
```

### Simplified path for single-user local inference

For ironmill's primary use case (local single-user), a simplified linear prefix
cache (without the full radix tree) captures most of the benefit:
- Store the last N conversation prefixes as flat `(Vec<u32>, KvCacheSlice)` entries.
- On new request, find the longest matching prefix by linear scan.
- Evict oldest entries when memory budget is exceeded.

The full radix tree should still be implemented for serving scenarios, but the
simplified path should be the default for `MetalInference`.

### Acceptance criteria

- Repeated prefill of the same 1024-token prompt hits the cache (0 new tokens
  computed on second call).
- Prefill of a prompt sharing an 800-token prefix with a cached prompt computes
  only the 200+ new tokens.
- LRU eviction correctly frees memory when budget is exceeded.
- PPL is unchanged — cached KV data must be bit-identical to recomputed data.
- `cargo test -p ironmill-inference -- cache` passes.

### Dependencies

- **Task 5** (KV cache reuse) — prefix caching requires position-aware cache
  management and `copy_from_slice()`.

---

## Task 5 — KV cache reuse across turns

**Goal:** Preserve KV cache state between conversation turns so that only new
tokens require computation. Currently `MetalInference::reset()` clears all state.

### Files to modify

```
crates/ironmill-inference/src/engine.rs            — add truncate_to(pos) and seq_pos() to InferenceEngine
crates/ironmill-inference/src/metal/inference.rs   — implement truncate_to(), stop clearing cache on new turn
crates/ironmill-inference/src/metal/turboquant/cache.rs — add truncate_to(pos: usize)
```

### Key interface changes

```rust
pub trait InferenceEngine {
    // Existing methods:
    fn load(&mut self, ...) -> Result<()>;
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>>;
    fn decode_step(&mut self, token: u32) -> Result<Vec<f32>>;
    fn reset(&mut self);

    // New methods:
    /// Current sequence position (number of tokens in KV cache).
    fn seq_pos(&self) -> usize;

    /// Truncate KV cache to the given position, discarding tokens after `pos`.
    /// Used to remove old turns when the context window is full.
    fn truncate_to(&mut self, pos: usize);
}
```

```rust
impl MetalKvCache {
    /// Truncate cache to `pos` tokens. Does not deallocate — just moves
    /// the write pointer back. Subsequent writes overwrite old data.
    pub fn truncate_to(&mut self, pos: usize) {
        assert!(pos <= self.seq_pos);
        self.seq_pos = pos;
    }
}
```

### Chat turn reuse flow

1. Turn 1: user sends `[system_prompt | user_msg_1]` → prefill all tokens.
   `seq_pos = len(system_prompt) + len(user_msg_1) + len(assistant_response_1)`.
2. Turn 2: user sends `user_msg_2`. Engine checks `seq_pos` — all prior tokens
   are still in the KV cache. Only prefill `user_msg_2` starting at current
   `seq_pos`. No `reset()` call.
3. Context window full: `truncate_to(len(system_prompt))` to keep the system
   prompt but discard old turns, then re-prefill from the most recent turns
   that fit.

### Acceptance criteria

- After prefill + decode of turn 1, a second `prefill()` call with new tokens
  appends to the existing KV cache (no recomputation of prior tokens).
- `seq_pos()` returns the correct count after prefill, decode, and truncate.
- `truncate_to(0)` is equivalent to `reset()`.
- PPL is unchanged — appending to the cache produces identical results to
  a full re-prefill of the entire sequence.
- Existing tests still pass (they call `reset()` which continues to work).

### Dependencies

None.

---

## Task 6 — Sliding window attention with ring-buffer KV cache

**Goal:** Enforce sliding window attention for models that specify it (Qwen3,
Mistral) and use a ring-buffer KV cache to bound memory for infinite-length
generation.

### Files to modify

```
crates/ironmill-inference/src/metal/inference.rs   — clamp attention mask, select SWA layers
crates/ironmill-inference/src/metal/shaders/fused_sdpa.metal — add window_size uniform, clamp KV range
crates/ironmill-inference/src/metal/turboquant/cache.rs — ring buffer write logic
crates/ironmill-compile/src/weights/safetensors.rs    — parse sliding_window and max_window_layers from HuggingFace config
crates/ironmill-inference/src/metal/config.rs      — add sliding_window field
```

### Key changes

```rust
// In ModelConfig or a new AttentionConfig:
pub struct AttentionLayerConfig {
    pub use_sliding_window: bool,  // true for layers < max_window_layers
    pub window_size: usize,        // e.g. 4096
}
```

```metal
// In fused_sdpa.metal — add window clamping:
constant uint& window_size [[buffer(10)]];  // 0 = full attention (no window)

uint kv_start = (window_size > 0) ? max(0, (int)seq_pos - (int)window_size) : 0;
uint kv_end = seq_pos;
// Only iterate KV tiles in [kv_start, kv_end)
```

```rust
impl MetalKvCache {
    /// Write position for ring buffer. Wraps around at window_size.
    pub fn ring_pos(&self, layer: usize) -> usize {
        if self.window_sizes[layer] > 0 {
            self.seq_pos % self.window_sizes[layer]
        } else {
            self.seq_pos
        }
    }
}
```

### Acceptance criteria

- Qwen3 model config with `sliding_window: 4096` and `max_window_layers: 21`
  correctly applies SWA to layers 0–20 and full attention to layers 21–27.
- Memory usage for SWA layers is bounded at `window_size × head_dim × 2 × num_kv_heads`
  regardless of sequence length.
- Generating 16384 tokens does not OOM on SWA layers.
- PPL on 1024-tok wikitext2 is unchanged (1024 < window_size, so no effect).
- `cargo test -p ironmill-inference` passes.

### Dependencies

None (but composes with Task 1 if fused SDPA is landed first).

---

## Task 7 — Continuous batching with vAttention-style memory

**Goal:** Enable concurrent request processing with dynamic sequence
add/remove and contiguous KV cache memory allocation (no paging indirection
in attention kernels).

### Files to create

```
crates/ironmill-inference/src/serving/
├── mod.rs            — re-exports
├── scheduler.rs      — continuous batching scheduler
├── sequence.rs       — per-sequence state (SequenceState)
├── pool.rs           — vAttention-style contiguous sub-allocator
└── batch.rs          — batch assembly for kernel dispatch

crates/ironmill-inference/src/metal/pool.rs  — Metal buffer pool with virtual contiguity
```

### Files to modify

```
crates/ironmill-inference/src/engine.rs          — add BatchInferenceEngine trait
crates/ironmill-inference/src/metal/inference.rs — implement batched prefill/decode
```

### Key types

```rust
pub trait BatchInferenceEngine: InferenceEngine {
    /// Add a new sequence to the batch. Returns a sequence handle.
    fn add_sequence(&mut self, tokens: &[u32]) -> Result<SequenceId>;

    /// Remove a completed sequence, freeing its KV memory.
    fn remove_sequence(&mut self, id: SequenceId) -> Result<()>;

    /// Run one batched decode step for all active sequences.
    fn batch_decode_step(&mut self) -> Result<Vec<(SequenceId, Vec<f32>)>>;
}

/// Contiguous sub-allocator for KV cache memory.
/// Inspired by vAttention (Patel et al. 2024), adapted for Metal where
/// OS-level virtual memory demand paging (CUDA VMM) is unavailable.
/// Allocates contiguous slabs from a pre-allocated Metal buffer.
/// No page tables — attention kernels see flat contiguous memory.
pub struct KvPool {
    backing_buffer: MetalBuffer,
    allocations: BTreeMap<SequenceId, KvAllocation>,
    free_list: Vec<(usize, usize)>,  // (offset, length) sorted by offset
}

pub struct KvAllocation {
    offset: usize,
    capacity: usize,  // max tokens (grows via reallocation)
    used: usize,      // current tokens
}

impl KvPool {
    /// Grow a sequence's allocation when it exceeds capacity.
    /// Strategy: attempt to extend in-place if adjacent free space exists.
    /// Otherwise, allocate a new larger slab (2× capacity), copy KV data,
    /// and free the old slab.
    pub fn grow(&mut self, id: SequenceId) -> Result<()>;

    /// Defragment: compact live allocations toward the start of the
    /// backing buffer, coalescing free regions. Called when allocation
    /// fails despite sufficient total free memory. Requires GPU copies
    /// for relocated slabs.
    pub fn defragment(&mut self, encoder: &ComputeEncoder) -> Result<()>;
}
```

### Acceptance criteria

- 4 concurrent sequences decode simultaneously with correct independent output.
- Adding/removing sequences during generation does not corrupt other sequences.
- Memory is reclaimed when sequences complete (visible via pool stats).
- Single-sequence throughput does not regress more than 5% vs unbatched path.
- `cargo test -p ironmill-inference -- serving` passes.

### Dependencies

- **Task 1** (fused SDPA) — batched attention dispatch requires the tiled kernel.
- **Task 5** (KV cache reuse) — per-sequence cache management.

---

## Task 8 — Cross-layer KV cache sharing (CLA)

**Goal:** Implement cross-layer attention where anchor layers share their KV
cache with subsequent layers, reducing KV memory by up to 2×.

### Files to modify

```
crates/ironmill-inference/src/metal/inference.rs        — skip KV cache write for non-anchor layers
crates/ironmill-inference/src/metal/turboquant/cache.rs — shared buffer references between layers
crates/mil-rs/src/weights.rs                            — parse CLA config (anchor_layers list)
crates/ironmill-inference/src/metal/config.rs           — add cla_anchor_layers field
```

### Key changes

```rust
// In MetalConfig or model config:
pub struct ClaConfig {
    /// Which layers are "anchor" layers that store their own KV cache.
    /// Non-anchor layers reuse the KV cache of the nearest preceding anchor.
    pub anchor_layers: Vec<usize>,
}
```

```rust
impl MetalKvCache {
    /// For non-anchor layer `layer`, return the buffer index of its anchor.
    pub fn kv_buffer_for_layer(&self, layer: usize) -> usize {
        // Binary search anchor_layers for nearest anchor <= layer
    }
}
```

### Acceptance criteria

- A model with 28 layers and 14 anchor layers (every other layer) allocates
  KV cache buffers for only 14 layers.
- Non-anchor layers read from their anchor's KV cache.
- PPL on CLA-trained models matches reference (not applicable to standard models
  without CLA training — this is a no-op for non-CLA models).
- Non-CLA models are unaffected (all layers are anchors by default).
- Memory usage is ~50% of baseline for a 14/28 anchor configuration.
- `cargo test -p ironmill-inference` passes.

### Dependencies

None.

---

## Task 9 — Multi-Head Latent Attention (MLA) support

**Goal:** Support DeepSeek-V2/V3's MLA architecture, storing compressed latent
KV cache and using weight absorption to eliminate runtime up-projection.

### Files to create

```
crates/ironmill-inference/src/metal/mla/
├── mod.rs          — MLA engine integration
├── cache.rs        — latent KV cache (compressed)
└── absorption.rs   — weight absorption (fuse W_uk into W_q, W_uv into W_o)
```

### Files to modify

```
crates/ironmill-inference/src/metal/inference.rs — detect MLA config, use absorbed attention path
crates/ironmill-compile/src/weights/safetensors.rs — parse MLA config fields (latent_dim, etc.)
```

### Key types

```rust
pub struct MlaConfig {
    pub kv_latent_dim: usize,       // compressed KV latent dimension (e.g., 512)
    pub q_latent_dim: usize,        // query latent dimension
    pub num_heads: usize,
    pub qk_nope_head_dim: usize,    // non-RoPE portion of Q/K head dimension
    pub qk_rope_head_dim: usize,    // RoPE-applied portion of Q/K head dimension
    pub v_head_dim: usize,          // per-head value dimension (may differ from qk dims)
}

/// Compressed KV cache storing latents instead of full K/V.
/// MLA stores a single joint KV latent (not separate K and V) plus RoPE keys.
pub struct MlaKvCache {
    /// Per-layer joint KV latent buffers: [max_seq_len × kv_latent_dim]
    latent_caches: Vec<MetalBuffer>,
    /// Per-layer RoPE key buffers: [max_seq_len × qk_rope_head_dim]
    rope_k_caches: Vec<MetalBuffer>,
    seq_pos: usize,
    max_seq_len: usize,
}
```

### Implementation notes

- **Weight absorption** (required for efficient inference): at model load time,
  fuse the KV up-projection matrices into the Q and O weight matrices:
  - `W_q_absorbed = W_q · W_uk^T` — query projection absorbs K up-projection.
  - `W_o_absorbed = W_uv · W_o` — output projection absorbs V up-projection.
  This eliminates runtime up-projection entirely. Attention operates directly
  on the compressed latent using absorbed weights.
- **Down-projection**: after computing the hidden state in each layer, project
  to latent space via learned matrix `W_dkv` and store the latent.
- **RoPE split**: MLA splits the key into a latent-compressed part (handled via
  absorption) and a RoPE-applied part. Only the RoPE part is stored separately
  and must be concatenated with the absorbed-Q dot latent during attention.
- **No separate MLA attention kernel**: with absorption, standard attention
  kernels (fused SDPA from Task 1) work directly on the absorbed Q and latent
  K/V. The only kernel change is handling the RoPE key concatenation.

### Acceptance criteria

- DeepSeek-V2-Lite (or similar small MLA model) runs end-to-end.
- KV cache memory per token is `(kv_latent_dim + qk_rope_head_dim) × 2 bytes`
  (one joint KV latent + RoPE key, in half precision) instead of
  `num_heads × head_dim × 2 × 2 bytes` (separate full K and V per head).
- PPL matches the HuggingFace reference for the same model.
- Non-MLA models are unaffected.
- `cargo test -p ironmill-inference -- mla` passes.

### Dependencies

None.

---

## Task 10 — Speculative Streaming (no auxiliary model)

**Goal:** Implement speculative streaming — multi-stream attention heads that
produce both the next token and speculative future tokens in a single forward
pass, without a separate draft model.

### Files to create

```
crates/ironmill-inference/src/speculative/streaming.rs — SpeculativeStreaming engine
```

### Files to modify

```
crates/ironmill-inference/src/metal/inference.rs — add multi-stream head dispatch
crates/ironmill-inference/src/speculative/mod.rs — integrate streaming path
```

### Implementation notes

- Requires a model that has been **fine-tuned with multi-stream attention (MSA)**
  (Bhendawade et al. 2024, arXiv:2402.11131). Standard attention layers are
  replaced with MSA layers during training, using an n-gram prediction objective
  that teaches the model to predict tokens at positions +2, +3, ..., +K
  alongside the standard next-token prediction.
- This is a **fundamental architecture change**, not a plug-in feature — models
  must be specifically trained or fine-tuned with the MSA objective. A matching
  MSA-trained checkpoint is required.
- The inference engine detects MSA-trained heads in the model weights (additional
  projection matrices per MSA layer) and dispatches them alongside the primary head.
- Verification is implicit: the primary head's output at position N+1 confirms
  or rejects the speculative token produced at position N.

### Acceptance criteria

- Models with streaming heads produce speculative tokens (2–3× throughput).
- Models without streaming heads fall back to standard decode (no regression).
- Output is identical to non-speculative decode (lossless).
- `cargo test -p ironmill-inference -- streaming` passes.

### Dependencies

- **Task 2** (EAGLE-3 speculative decoding) — shares verification infrastructure
  and KV cache rollback.

---

## Task 11 — Prefill/decode phase separation

**Goal:** Route prefill (compute-bound) and decode (memory-bound) to optimal
compute units. Prefill uses batched matmul kernels; decode uses matvec kernels.
Optionally route through MLX for ANE/GPU scheduling.

### Files to modify

```
crates/ironmill-inference/src/metal/inference.rs — separate prefill/decode kernel selection
crates/ironmill-inference/src/metal/ops.rs       — add matvec pipeline variants
```

### Files to create (optional MLX path)

```
crates/ironmill-inference/src/mlx/scheduler.rs — MLX-based ANE/GPU phase routing (mlx/ dir already exists)
```

### Implementation notes

- **Metal-only path** (primary): select kernel variant based on token count.
  Prefill (token_count > 1): use batched matmul (MPS or custom tiled kernel).
  Decode (token_count == 1): use matvec kernel optimized for memory bandwidth
  (wider threadgroups, coalesced reads, no unnecessary shared memory).
- **MLX path** (optional): if MLX integration is available, route prefill to
  Neural Engine via MLX's internal scheduling. This is behind a feature flag
  and not a hard dependency.

### Acceptance criteria

- Decode kernel selection is automatic based on token_count.
- Prefill throughput does not regress.
- Decode throughput improves by ≥20% from matvec specialization.
- `cargo test -p ironmill-inference` passes.

### Dependencies

- **Task 1** (fused SDPA) — phase separation builds on the tiled kernel.

---

## Task 12 — Structured generation (grammar/JSON-constrained sampling)

**Goal:** Implement XGrammar-style constrained generation that forces model
output to conform to a schema (JSON, function signatures) by masking invalid
tokens at each decode step.

### Files to create

```
crates/ironmill-inference/src/grammar/
├── mod.rs           — re-exports
├── grammar.rs       — Grammar type (BNF/JSON schema → token constraints)
├── compiler.rs      — JIT mask compilation (precompute context-independent masks)
├── automaton.rs     — persistent pushdown automaton for runtime state
├── mask.rs          — token mask (bitset over vocab)
└── json_schema.rs   — JSON Schema → BNF grammar converter
```

### Files to modify

```
crates/ironmill-inference/src/sampling.rs — apply grammar mask before sampling
crates/ironmill-inference/src/engine.rs   — accept optional Grammar in decode config
```

### Key types

```rust
/// Compiled grammar ready for constrained generation.
pub struct CompiledGrammar {
    /// Pre-computed masks for context-independent token classes.
    /// These are valid regardless of the automaton state.
    precomputed_masks: HashMap<TokenClass, TokenMask>,
    /// Pushdown automaton transitions.
    transitions: Vec<Transition>,
}

/// Runtime state for grammar-constrained generation.
pub struct GrammarState {
    grammar: Arc<CompiledGrammar>,
    stack: Vec<StackSymbol>,       // pushdown automaton stack
    current_state: usize,
}

impl GrammarState {
    /// Compute the token mask for the current state.
    /// Uses precomputed masks for context-independent tokens (~90% of vocab),
    /// evaluates context-dependent tokens against the automaton.
    pub fn token_mask(&self) -> TokenMask;

    /// Advance the automaton state after a token is accepted.
    pub fn advance(&mut self, token_id: u32);

    /// Check if the grammar is in an accepting state.
    pub fn is_complete(&self) -> bool;
}

/// Efficient bitset over vocabulary.
pub struct TokenMask {
    bits: Vec<u64>,  // ceil(vocab_size / 64) u64s
}
```

### Acceptance criteria

- JSON schema constraint: output is always valid JSON matching the schema.
- BNF grammar constraint: output conforms to the grammar.
- Unconstrained generation is unaffected (no mask applied).
- JIT compilation of a large JSON schema completes in < 10ms.
- Runtime mask computation adds < 0.1ms per token (amortized).
- Composes with all sampler chain stages (mask applied before temperature/min-p).
- `cargo test -p ironmill-inference -- grammar` passes.

### Dependencies

None.

---

## Task 13 — TurboSpec adaptive speculation

**Goal:** Implement a closed-loop runtime controller that dynamically tunes
speculation depth, tree width, and acceptance thresholds based on observed
acceptance rates.

### Files to create

```
crates/ironmill-inference/src/speculative/turbospec.rs — adaptive controller
```

### Files to modify

```
crates/ironmill-inference/src/speculative/mod.rs   — integrate TurboSpec controller
crates/ironmill-inference/src/speculative/config.rs — add TurboSpecConfig
```

### Key types

```rust
pub struct TurboSpecConfig {
    pub initial_depth: usize,
    pub min_depth: usize,
    pub max_depth: usize,
    pub ema_alpha: f32,           // exponential moving average decay for acceptance rate
    pub depth_up_threshold: f32,  // increase depth if acceptance rate > this
    pub depth_down_threshold: f32,// decrease depth if acceptance rate < this
}

pub struct TurboSpecController {
    config: TurboSpecConfig,
    acceptance_ema: f32,          // running acceptance rate
    current_depth: usize,
    current_width: usize,
    total_accepted: u64,
    total_proposed: u64,
}

impl TurboSpecController {
    /// Update the controller with the result of a speculation round.
    pub fn observe(&mut self, proposed: usize, accepted: usize);

    /// Get the current recommended speculation config.
    pub fn current_config(&self) -> SpecConfig;

    /// Goodput metric: accepted_tokens / wall_time.
    pub fn goodput(&self) -> f64;
}
```

### Acceptance criteria

- Controller increases depth when acceptance rate is high (> 80%).
- Controller decreases depth when acceptance rate is low (< 40%).
- Goodput (accepted tokens per second) is ≥ 20% higher than static depth.
- Controller converges within 50 speculation rounds on stable input.
- `cargo test -p ironmill-inference -- turbospec` passes.

### Dependencies

- **Task 2** (EAGLE-3 speculative decoding) — TurboSpec wraps the speculation loop.

---

## Task 14 — QuIP# end-to-end weight quantization

**Goal:** Wire the existing QuIP# mil-rs pass (Hadamard rotation + E8 lattice
quantization) through the full compile → bundle → Metal inference pipeline,
enabling 2-bit weight quantization with near-lossless quality.

The core algorithm is already implemented: `QuipSharpPass` performs randomized
Hadamard incoherence processing, quantizes weight vectors to 8-bit E8 lattice
indices, extracts per-row norms, and rewrites ops to `constexpr_lut_to_dense`.
The current pass uses nearest-neighbor E8 search without Hessian guidance.
To match paper-quality results, the pass must be upgraded with Hessian-guided
LDLQ (Lattice-based Data-Less Quantization) adaptive rounding.

### Calibration requirements

QuIP# requires a calibration dataset to estimate per-layer Hessians — the same
pattern as AWQ and GPTQ. The Hessians capture second-order sensitivity
(how much each weight perturbation affects layer output) and guide the adaptive
rounding step in LDLQ.

**Calibration data:** A few hundred to a few thousand text samples (e.g.,
wikitext2 or a representative corpus). Reuse the existing `CalibrationRunner`
infrastructure from the AWQ/GPTQ calibration pipeline.

**Hessian estimation:** Per-layer, computed via forward pass on calibration data.
For each linear layer, the Hessian is approximated as H ≈ X^T X / n where X is
the matrix of input activations across calibration samples. This is the same
approximation used by GPTQ. Extend the existing `HessianStore` (or add a new
activation hook) to collect these per-layer Hessians during the calibration run.

**LDLQ rounding:** Instead of nearest-neighbor E8 codebook lookup, LDLQ uses
the Cholesky decomposition of the layer Hessian to perform adaptive rounding
that accounts for inter-weight correlations. Weights are quantized column by
column (like GPTQ), with the rounding error for each column propagated to
subsequent columns via the Hessian. This is the step that makes QuIP# achieve
<1 PPL loss at 2-bit — without it, naive E8 nearest-neighbor gives substantially
worse results.

**Calibration is mandatory.** `--quip-sharp` without `--calibration-data` is a
compile error. Without Hessian-guided rounding, 2-bit E8 quantization produces
unacceptably high PPL — shipping that path would give users a false impression
of QuIP# quality.

### Files to modify

```
crates/mil-rs/src/ir/passes/quip_sharp/mod.rs       — add LDLQ rounding path that accepts per-layer Hessian
crates/mil-rs/src/ir/pipeline.rs                    — add quip_sharp to KNOWN_PASSES array and pass_from_name()
crates/ironmill-cli/src/main.rs                     — add --quip-sharp CLI flag; require --calibration-data
crates/ironmill-compile/src/convert/pipeline.rs     — add quip_sharp to TOML pipeline stage
crates/ironmill-compile/src/gpu/bundle.rs           — write QuIP# bundle metadata (method: "quip_sharp", seed)
crates/ironmill-core/src/gpu/bundle.rs              — add quip_sharp_seed field to LutToDense manifest
crates/ironmill-inference/src/metal/dequant.rs      — read quip_sharp_seed from manifest; dispatch to QuIP# dequant
crates/ironmill-inference/src/metal/ops.rs          — add pipeline for quip_sharp_matvec / quip_sharp_matmul
crates/ironmill-inference/src/metal/inference.rs    — select QuIP# kernel for QuIP#-quantized layers
crates/ironmill-compile/src/weights/mil_provider.rs — read quip_sharp_seed from mil model metadata
crates/ironmill-inference/src/calibration/runner.rs — extend CalibrationRunner with HessianHook for QuIP#
```

### Files to create

```
crates/ironmill-inference/src/metal/shaders/quip_sharp.metal
```

### Bundle metadata changes

The `LutToDense` manifest variant gains an optional `quip_sharp_seed` field:

```json
{
  "format": "lut_to_dense",
  "n_bits": 8,
  "method": "quip_sharp",
  "quip_sharp_seed": 42,
  "files": {
    "indices": "layer.attn.q_proj.bin",
    "lut": "layer.attn.q_proj.lut",
    "norms": "layer.attn.q_proj.nrm"
  }
}
```

The `method` field distinguishes QuIP# from PolarQuant bundles. The seed is
required for reconstructing the deterministic Hadamard rotation matrix at
inference time.

### Metal kernel signature

```metal
// E8 codebook in constant memory (256 entries × 8 half values = 4 KB)
constant half e8_codebook[256][8] = { /* embedded at compile time */ };

kernel void quip_sharp_matvec(
    device const uint8_t* indices     [[buffer(0)]],  // [rows × cols/8] E8 lattice indices
    device const half*    norms       [[buffer(1)]],  // [rows] per-row norm factors
    device const half*    x           [[buffer(2)]],  // [cols] input vector
    device half*          y           [[buffer(3)]],  // [rows] output vector
    constant uint&        rows        [[buffer(4)]],
    constant uint&        cols        [[buffer(5)]],
    constant uint&        seed        [[buffer(6)]],  // Hadamard rotation seed
    uint tid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
);
```

### Dequant logic (per row)

```
1. Load E8 indices for the row (cols/8 indices, each selecting one of 256 E8 vectors)
2. Look up codebook vectors: w_quantized[i*8..(i+1)*8] = e8_codebook[index[i]]
3. Apply inverse randomized Hadamard transform: w_rotated = H_seed^{-1} · w_quantized
   - Fast Walsh–Hadamard transform (log2(cols) stages, in-register)
   - Randomized: multiply by diagonal sign matrix derived from seed before/after WHT
4. Multiply by per-row norm: w_final = norm[row] · w_rotated
5. Dot product with input vector x
```

For the matvec kernel, steps 3–5 are fused: the Hadamard-rotated input vector
is precomputed once (H_seed · x), then each row is a simple dot product of
codebook-reconstructed values with the rotated input, scaled by the row norm.
This avoids per-row inverse Hadamard transforms entirely.

### Fused matvec optimization

The key insight: instead of dequantizing W and computing W·x, compute
W_q · (H·x) · norms, where W_q is the quantized (codebook-reconstructed)
weight and H·x is the Hadamard-rotated input. The Hadamard rotation of x is
O(n log n) and done once per layer, amortized across all rows.

```
1. Precompute x_rot = H_seed · x  (one WHT of the input vector, O(n log n))
2. For each row:
   a. Accumulate dot product of E8 codebook vectors with corresponding
      8-element chunks of x_rot
   b. Scale by row norm
```

This makes the inner loop a pure codebook-lookup + dot-product — the same
pattern as PolarQuant matvec but with the E8 codebook and 8-element vector
granularity.

### Matmul kernel

```metal
kernel void quip_sharp_matmul(
    device const uint8_t* indices     [[buffer(0)]],  // [rows × cols/8]
    device const half*    norms       [[buffer(1)]],  // [rows]
    device const half*    X           [[buffer(2)]],  // [batch × cols]
    device half*          Y           [[buffer(3)]],  // [batch × rows]
    constant uint&        rows        [[buffer(4)]],
    constant uint&        cols        [[buffer(5)]],
    constant uint&        batch       [[buffer(6)]],
    constant uint&        seed        [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_id   [[simdgroup_index_in_threadgroup]]
);
```

Tiles over (row_block, batch_block) using simdgroup operations. Each
threadgroup precomputes the Hadamard rotation of its X tile columns once,
then performs codebook-lookup dot products across row tiles.

### CLI integration

```
ironmill compile --quip-sharp --calibration-data wikitext2.json model.safetensors -o model.gpu
ironmill compile --quip-sharp --calibration-data wikitext2.json --quip-sharp-seed 42 model.safetensors -o model.gpu
```

`--calibration-data` is required when `--quip-sharp` is specified. Omitting it
is a compile error. Default seed: 0 (deterministic). The seed must match
between compile and inference (stored in bundle manifest).

### TOML pipeline config

```toml
[quantize]
method = "quip_sharp"
seed = 42
calibration_data = "wikitext2.json"  # required
```

Added to `pass_from_name()` in pipeline.rs alongside existing methods.

### CPU fallback dequant

Update `crates/ironmill-inference/src/metal/dequant.rs` to handle
`quip_sharp_seed`:

```rust
if let Some(seed) = quip_sharp_seed {
    // 1. Reconstruct from LUT indices using E8 codebook
    // 2. Apply inverse randomized Hadamard (using seed)
    // 3. Multiply by row norms
} else {
    // Existing PolarQuant path
}
```

### Acceptance criteria

- `cargo test -p mil-rs -- quip_sharp` passes (existing + new round-trip tests).
- `cargo test -p ironmill-inference` passes (no regression).
- PPL on wikitext2 1024-tok with QuIP# 2-bit Qwen3-0.6B ≤ 14.0
  (within ~1.14 of FP16 baseline 12.86).
- Bundle size for Qwen3-0.6B is ≤ 110 MB (vs ~1.2 GB FP16, ~6× reduction;
  2-bit weights + codebook + norms overhead).
- Decode throughput ≥ 30 tok/s (≥85% of FP16 baseline; memory-bandwidth
  savings from smaller weights should partially offset dequant compute cost).
- `ironmill compile --quip-sharp --calibration-data data.json` produces a
  valid `.gpu` bundle that loads and runs inference correctly.
- CLI rejects `--quip-sharp` without `--calibration-data` with a clear error.
- Round-trip test: quantize → bundle → load → dequant → compare against
  mil-rs dequant reference with max error < 1e-3.
- Hessian estimation completes in < 5 minutes for Qwen3-0.6B on a single
  Apple Silicon GPU with 512 calibration samples.

### Dependencies

- None (independent — all prerequisite algorithmic code exists in mil-rs).

---

## Dependency graph

```
Independent (can start immediately):
  ├── Task 1   Fused SDPA kernel
  ├── Task 3   Min-p sampling + full sampler chain
  ├── Task 5   KV cache reuse across turns
  ├── Task 6   Sliding window attention
  ├── Task 8   Cross-layer KV sharing (CLA)
  ├── Task 9   MLA support
  ├── Task 12  Structured generation (JSON/grammar)
  └── Task 14  QuIP# end-to-end weight quantization

Task 1 (Fused SDPA) ──→ Task 11  Prefill/decode phase separation
                    └──→ Task 7   Continuous batching + vAttention (also needs Task 5)

Task 5 (KV reuse)  ──→ Task 4   RadixAttention prompt caching
                   └──→ Task 2   EAGLE-3 speculative decoding (also needs Task 3)

Task 3 (Min-p)     ──→ Task 2   EAGLE-3 speculative decoding (also needs Task 5)

Task 2 (EAGLE-3)   ──→ Task 10  Speculative Streaming
                   └──→ Task 13  TurboSpec adaptive speculation
```
