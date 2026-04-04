# Performance Improvement Roadmap

Current baseline (Qwen3-0.6B, Metal FP16, Apple Silicon):

| Metric | Value |
|--------|-------|
| Decode throughput | ~35 tok/s (single-token autoregressive) |
| Prefill 1024 tok | 758 tok/s |
| Prefill 8192 tok | 214 tok/s |
| PPL (1024-tok wikitext2) | 12.86 (matches HuggingFace reference) |

Reference comparison (HuggingFace transformers, CPU FP32, same hardware):

| Sequence length | ironmill GPU | HF CPU | Gap |
|-----------------|-------------|--------|-----|
| 1024 | 1287 tok/s | 1104 tok/s | **+17%** |
| 2048 | 778 tok/s | 1073 tok/s | −28% |
| 4096 | 418 tok/s | 995 tok/s | −58% |
| 8192 | 214 tok/s | 793 tok/s | −73% |

---

## Tier 1 — Critical performance gaps

### 1. Fused Scaled Dot-Product Attention (SDPA)

**Status:** Proposed ([ADR-0001](adr/0001-fused-sdpa-metal-kernel.md))  
**Expected impact:** 3–5× prefill throughput at ≥2048 tokens  
**Complexity:** High

The single largest bottleneck. The current attention kernel dispatches one threadgroup per (head, query), giving O(n²) threadgroup count with severe causal-mask load imbalance. At 8192 tokens the GPU is 73% slower than CPU.

A fused SDPA kernel tiles over both Q and KV dimensions using `simdgroup_matrix_multiply_accumulate`, keeping all intermediate data (scores, softmax state, weighted-V accumulator) in SRAM. Each KV tile is loaded once from global memory and reused across all Q tokens in the block. This matches the FlashAttention-2 algorithm that PyTorch and llama.cpp use.

**Target:** ≥800 tok/s at 8192 tokens (matching CPU baseline).

### 2. Speculative decoding

**Status:** Not implemented  
**Expected impact:** 2–3× decode throughput  
**Complexity:** Medium

Use a small draft model (e.g., 0.1B) to propose N candidate tokens, then verify them in a single forward pass of the full model. Accepted tokens skip individual decode steps. Requires:

- Draft model integration (load two models simultaneously)
- Batched verification pass (prefill N candidate tokens)
- Rejection sampling for distribution-correct generation
- KV cache rollback for rejected tokens

llama.cpp implements this as "speculative sampling" with configurable draft length. PyTorch has `torch.compile` + speculative decoding in torchtune.

**Target:** 70–100 tok/s decode for Qwen3-0.6B (vs ~35 tok/s today).

### 3. Top-k / top-p / repetition penalty sampling

**Status:** Temperature-only (greedy/temperature scaling exist, no nucleus sampling)  
**Expected impact:** Table stakes for generation quality  
**Complexity:** Low

Every production LLM interface requires at minimum: top-k filtering, nucleus (top-p) sampling, repetition penalty, and frequency penalty. These operate on the logits vector after the forward pass — no kernel changes needed, just CPU-side post-processing.

- Top-k: keep only the k highest logits, zero the rest
- Top-p (nucleus): keep logits whose cumulative softmax probability ≤ p
- Repetition penalty: scale down logits for tokens that appeared in context
- Min-p: newer alternative to top-p with better tail behavior

**Target:** Feature-complete sampling API matching llama.cpp's `llama_sampler`.

---

## Tier 2 — Important for production use

### 4. Prompt caching / KV cache persistence

**Status:** Not implemented  
**Expected impact:** Skip re-prefill for shared prompt prefixes  
**Complexity:** Medium

For multi-turn chat, the system prompt + conversation history is re-prefilled on every turn. With KV cache persistence:

- Cache the KV state after prefilling a prompt prefix
- On subsequent turns, restore the cached state and only prefill new tokens
- Hash-based cache key (prompt token sequence → cached KV state)

llama.cpp implements this as "prompt caching" with `llama_kv_cache_seq_cp`. This is especially impactful for long system prompts (1K+ tokens).

**Target:** Zero re-prefill cost for unchanged prompt prefixes.

### 5. KV cache reuse across turns

**Status:** Not implemented (engine.reset() clears all state)  
**Expected impact:** Eliminate redundant computation in multi-turn chat  
**Complexity:** Low–Medium

Currently `MetalInference::reset()` clears the entire KV cache. For multi-turn chat, the engine should:

- Track which KV cache positions correspond to which conversation turns
- On a new user message, keep existing KV entries and only prefill the new tokens
- Support cache truncation (remove old turns when context window is full)

This is simpler than full prompt caching — it just means not discarding valid KV state between calls.

### 6. Sliding window attention

**Status:** Partial (config field exists, mask not enforced in Metal kernels)  
**Expected impact:** Enables infinite-length generation with bounded memory  
**Complexity:** Medium

Qwen3 and Mistral use sliding window attention where each token attends to at most W preceding tokens (e.g., W=4096). This bounds KV cache memory to W entries regardless of generation length.

The `config.json` field `sliding_window` is already parsed but the Metal attention kernels don't enforce the window — they attend to all positions up to the causal mask. Implementing this requires:

- Modifying the attention kernel's causal mask to `max(0, pos - W) .. pos`
- Ring-buffer KV cache that overwrites old entries
- Per-layer window selection (Qwen3 `max_window_layers` controls which layers use sliding vs full attention)

---

## Tier 3 — Serving and scale

### 7. Continuous batching

**Status:** Not implemented  
**Expected impact:** N× throughput for concurrent requests  
**Complexity:** High

Process multiple independent sequences simultaneously, dynamically adding/removing sequences as they complete. The matmul batch dimension absorbs multiple sequences with negligible overhead since GPU ALUs are underutilized on single-sequence inference.

Requires:
- Multi-sequence KV cache management
- Dynamic batch assembly (different sequences at different positions)
- Request queue with preemption
- Paged attention or pre-allocated per-sequence cache slots

This is the core of vLLM and llama.cpp's server mode. Critical for any production serving deployment but less important for local single-user inference.

### 8. Paged attention

**Status:** Not implemented  
**Expected impact:** Eliminates KV cache memory fragmentation  
**Complexity:** High

Allocate KV cache in fixed-size pages (e.g., 16 tokens per page) instead of contiguous pre-allocated blocks. Benefits:

- No wasted memory from pre-allocating max_seq_len per sequence
- Dynamic memory sharing across batched sequences
- Enables longer contexts without OOM
- Foundation for continuous batching

The attention kernel reads KV data through a page table (indirect lookup). This adds one level of indirection but eliminates the memory waste of contiguous allocation.

### 9. Tensor parallelism

**Status:** Not implemented  
**Expected impact:** Run larger models by splitting across GPU cores  
**Complexity:** High

Apple Silicon has unified memory but multiple GPU core clusters. Tensor parallelism splits large matmuls across cores with all-reduce synchronization. More relevant for M2 Ultra / M3 Ultra with 2 GPU dies, or future multi-chip configurations.

Not a priority for single-die Apple Silicon where the GPU is already a single device.

### 10. CPU offloading

**Status:** Not implemented  
**Expected impact:** Fit models larger than GPU VRAM  
**Complexity:** Medium

Keep some layers on CPU (using Accelerate BLAS) while critical layers run on GPU. llama.cpp supports this with `--n-gpu-layers`. On Apple Silicon with unified memory, the "offloading" is mainly about which compute unit runs each layer — data doesn't need physical transfer.

---

## Tier 4 — Quality and efficiency

### 11. Activation-aware quantization improvements

**Status:** Partial (AWQ, GPTQ exist at compile time)  
**Expected impact:** Better quality per bit  
**Complexity:** Medium

- **SmoothQuant:** Migrate quantization difficulty from activations to weights using per-channel scaling. Improves INT8 quality significantly.
- **GGUF IQ (importance-aware quantization):** Non-uniform quantization that assigns more bits to important weights. llama.cpp's IQ2/IQ3 formats achieve better quality than uniform INT4.
- **Runtime dynamic quantization:** Quantize activations on-the-fly based on their actual distribution, rather than using fixed scales from calibration.

### 12. Sparse attention patterns

**Status:** Not implemented  
**Expected impact:** Reduce attention compute for very long sequences  
**Complexity:** High

For sequences >8K tokens, full attention becomes prohibitive even with fused SDPA. Sparse patterns (local + stride, BigBird, Longformer) reduce attention from O(n²) to O(n√n) or O(n·log(n)). Most modern long-context models don't use these (they rely on RoPE scaling + full attention), so this is lower priority unless targeting 100K+ context.

---

## Implementation order

```
Near-term (immediate impact):
  ├── #3  Top-k/top-p sampling (low effort, table stakes)
  ├── #5  KV cache reuse across turns (low effort, big UX win)
  └── #1  Fused SDPA kernel (high effort, fixes scaling wall)

Medium-term (production readiness):
  ├── #2  Speculative decoding (2–3× decode speed)
  ├── #4  Prompt caching
  └── #6  Sliding window attention

Long-term (serving infrastructure):
  ├── #7  Continuous batching
  ├── #8  Paged attention
  └── #9  Tensor parallelism
```
