# Performance Improvement Roadmap

Current baseline (Qwen3-0.6B, Metal FP16, Apple Silicon):

| Metric | Value |
|--------|-------|
| Decode throughput | ~35 tok/s (single-token autoregressive) |
| Prefill 1024 tok | 1287 tok/s |
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

### 1. Fused SDPA with simdgroup matrix tiling

**Status:** Proposed ([ADR-0001](../adr/0001-fused-sdpa-metal-kernel.md))
**Expected impact:** 3–5× prefill throughput at ≥2048 tokens
**Complexity:** High
**SOTA reference:** FlashAttention-3 (Dao et al., NeurIPS 2024)

The single largest bottleneck. The current attention kernel dispatches one threadgroup per (head, query), giving O(n²) threadgroup count with severe causal-mask load imbalance. At 8192 tokens the GPU is 73% slower than CPU.

FlashAttention-3 introduces warp-specialization to overlap matmul compute with async memory loads, plus interleaved block-quantized softmax. While FA3's async TMA features are CUDA/Hopper-specific, the core tiling strategy (Q blocks × KV tiles with online softmax, `simdgroup_matrix_multiply_accumulate` for both QK^T and weighted-V) maps directly to Metal's simdgroup matrix hardware.

The Metal adaptation should tile over both Q and KV dimensions using 8×8 simdgroup tiles, keeping all intermediate data in SRAM. GQA-aware: multiple Q heads share KV tile loads from the same KV head group.

**Target:** ≥800 tok/s at 8192 tokens (matching CPU baseline).

### 2. EAGLE-3 / P-EAGLE speculative decoding

**Status:** Not implemented
**Expected impact:** 3–6× decode throughput
**Complexity:** Medium–High
**SOTA reference:** EAGLE-3 (NeurIPS 2025), P-EAGLE (AWS/vLLM 2025)

EAGLE-3 is the current SOTA for speculative decoding, using multi-layer feature fusion (not just second-top-layer extrapolation like EAGLE-1) to generate richer draft trees. Pre-trained SpecBundle draft heads are available for Qwen and Llama families.

P-EAGLE extends this with parallel draft generation — all K candidate tokens produced in a single forward pass of the draft head instead of K sequential passes, eliminating the sequential bottleneck at high speculation depths.

Implementation requires:
- EAGLE-3 draft head (small MLP that fuses features from target model layers)
- Tree-structured candidate generation with dynamic depth
- Batched verification via prefill of candidate token tree
- KV cache rollback for rejected branches
- SpecBundle checkpoint loading for common model families

The draft head is ~2–5% of target model parameters and runs on the same GPU. Combined with ironmill's existing TurboQuant KV cache, this should compose cleanly.

**Target:** 100–200 tok/s decode for Qwen3-0.6B (vs ~35 tok/s today).

### 3. Min-p sampling + full sampler chain

**Status:** Temperature-only (no nucleus, min-p, or repetition penalty)
**Expected impact:** Table stakes for generation quality
**Complexity:** Low
**SOTA reference:** Min-p (ICLR 2025, Nguyen et al.)

Min-p sampling is the SOTA replacement for top-p (nucleus) sampling. Instead of a fixed cumulative probability threshold, min-p scales the cutoff dynamically: `prob(token) > min_p × prob(top_token)`. This adapts to model confidence — restrictive when confident, permissive when uncertain — giving better coherence at high temperatures than top-p.

Full sampler chain (matching llama.cpp's `llama_sampler`):
1. Repetition/frequency/presence penalty
2. Temperature scaling
3. Min-p filtering (replaces top-p as default)
4. Top-k filtering (optional, compositional with min-p)
5. Categorical sampling from filtered distribution

All of this is CPU-side logit post-processing — no kernel changes needed.

---

## Tier 2 — Important for production use

### 4. RadixAttention prompt caching

**Status:** Not implemented
**Expected impact:** Eliminate re-prefill for shared prompt prefixes
**Complexity:** Medium
**SOTA reference:** RadixAttention (SGLang, LMSYS 2024)

RadixAttention (from SGLang) caches computed KV activations in a radix tree (Patricia trie). When a new request shares any prompt prefix with a cached entry, only the new tokens need computation. This gives up to 5× throughput improvement on workloads with ≥60% prefix overlap (chatbots, RAG, tool-augmented agents).

This is strictly better than naive "save/restore KV cache" because:
- Multiple divergent continuations share the common prefix cache
- Cache-aware scheduling maximizes hit rates
- Eviction is LRU per radix node, not per sequence

For ironmill's local single-user case, a simplified version (linear prefix matching without the full radix tree) would capture most of the benefit for multi-turn chat.

### 5. KV cache reuse across turns

**Status:** Not implemented (engine.reset() clears all state)
**Expected impact:** Eliminate redundant computation in multi-turn chat
**Complexity:** Low–Medium

Currently `MetalInference::reset()` clears the entire KV cache. For multi-turn chat, the engine should:
- Track which KV cache positions correspond to which conversation turns
- On a new user message, keep existing KV entries and only prefill new tokens
- Support cache truncation (remove old turns when context window is full)

This is a prerequisite for RadixAttention and a standalone win for chat UX.

### 6. Sliding window attention with ring-buffer KV cache

**Status:** Partial (config field parsed, not enforced in Metal kernels)
**Expected impact:** Bounded memory for infinite-length generation
**Complexity:** Medium

Qwen3 and Mistral use sliding window attention (W=4096 or similar). The attention kernel should clamp the causal mask to `max(0, pos - W)..pos` and the KV cache should use a ring buffer that overwrites old entries. Qwen3's `max_window_layers` controls which layers use sliding vs full attention.

---

## Tier 3 — Serving infrastructure

### 7. Continuous batching with vAttention-style memory

**Status:** Not implemented
**Expected impact:** N× throughput for concurrent requests
**Complexity:** High
**SOTA reference:** vAttention (Microsoft, ASPLOS 2025)

vAttention improves on PagedAttention by maintaining virtual contiguity of KV cache memory while enabling dynamic physical allocation. This means standard attention kernels (like our fused SDPA) work without modification — no page table indirection in the kernel. Memory is allocated on-demand but appears contiguous to the GPU.

On Metal/Apple Silicon, the equivalent would be using Metal's virtual memory APIs (if available) or a simpler pool allocator with contiguous sub-allocations per sequence. The key insight: avoid rewriting attention kernels for paging.

Continuous batching dynamically adds/removes sequences as they complete, maximizing GPU utilization. Combined with vAttention-style allocation, this enables production serving on Apple Silicon.

### 8. Cross-layer KV cache sharing (CLA)

**Status:** Not implemented
**Expected impact:** 2× KV cache memory reduction
**Complexity:** Medium (requires model support or fine-tuning)
**SOTA reference:** CLA (NeurIPS 2024), FusedKV (ICLR 2026)

Cross-Layer Attention shares KV caches between adjacent layers — subsequent layers reuse the KV cache from "anchor" layers instead of storing their own. This reduces KV memory by 2× or more beyond GQA, with negligible quality loss.

FusedKV (ICLR 2026) extends this by learnably fusing informative K/V from bottom and middle layers, achieving better quality-memory trade-offs than simple sharing.

This requires model architecture support (the model must be trained or fine-tuned with CLA). ironmill's role is to detect CLA-enabled models and share cache buffers between layers accordingly.

### 9. Multi-Head Latent Attention (MLA) support

**Status:** Not implemented
**Expected impact:** 5–10× KV cache compression vs MHA
**Complexity:** Medium
**SOTA reference:** DeepSeek-V2/V3 MLA architecture

MLA projects keys and values into a shared low-dimensional latent space, storing only the compressed latent per token. During attention, the latent is up-projected on-the-fly. Used by DeepSeek-V2/V3 models. ironmill would need:
- Detect MLA config in model metadata
- Store compressed latent KV cache instead of full K/V
- On-the-fly up-projection in the attention kernel

---

## Tier 4 — Advanced optimizations

### 10. Speculative Streaming (no auxiliary model)

**Status:** Not implemented
**Expected impact:** 2–3× decode throughput without draft model
**Complexity:** High
**SOTA reference:** Speculative Streaming (OpenReview 2025)

Eliminates the need for a separate draft model by integrating speculated token planning within the target model using multi-stream attention heads. Each forward pass produces both the "correct" next token and speculative future tokens. Ideal for edge/resource-constrained deployment where loading a separate draft model is impractical.

### 11. Activation-aware quantization (XQuant)

**Status:** Partial (AWQ, GPTQ exist at compile time)
**Expected impact:** Better quality per bit at ultra-low precision
**Complexity:** Medium
**SOTA reference:** XQuant (EMNLP 2025)

XQuant achieves ultra-low-bit KV cache quantization (2–4 bit) with cross-layer compression, using data-free calibration. This composes with ironmill's existing TurboQuant — XQuant could replace or supplement the codebook-based quantization with learned cross-layer compression for even better quality/memory trade-offs.

### 12. TurboSpec adaptive speculation

**Status:** Not implemented
**Expected impact:** +20–50% over static speculative decoding
**Complexity:** Medium
**SOTA reference:** TurboSpec (Berkeley 2025)

Closed-loop feedback system that dynamically tunes speculation depth, draft tree width, and acceptance thresholds at runtime based on observed acceptance rates. Maximizes "goodput" (accepted tokens per second) rather than raw proposals. Would be implemented as a runtime controller wrapping the EAGLE-3 draft head.

---

## Implementation order

```
Near-term (immediate impact):
  ├── #3  Min-p sampling + full sampler chain
  ├── #5  KV cache reuse across turns
  └── #1  Fused SDPA kernel (ADR-0001)

Medium-term (production readiness):
  ├── #2  EAGLE-3 / P-EAGLE speculative decoding
  ├── #4  RadixAttention prompt caching
  └── #6  Sliding window attention

Long-term (serving + architectural):
  ├── #7  Continuous batching + vAttention
  ├── #8  Cross-layer KV sharing (CLA)
  ├── #9  MLA support
  └── #10-12 Advanced optimizations
```

## References

- FlashAttention-3: https://arxiv.org/abs/2407.08608
- EAGLE-3: https://github.com/SafeAILab/EAGLE
- P-EAGLE: https://vllm.ai/blog/p-eagle
- SpecBundle: https://www.lmsys.org/blog/2025-12-23-spec-bundle-phase-1/
- Min-p sampling: https://arxiv.org/abs/2407.01082
- RadixAttention: https://www.lmsys.org/blog/2024-01-17-sglang/
- vAttention: https://arxiv.org/abs/2405.04437
- CLA: https://arxiv.org/abs/2405.12981
- FusedKV: https://openreview.net/forum?id=4pivvEJiCl
- XQuant: https://github.com/brinenick511/XQuant
- TurboSpec: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html
- Speculative Streaming: https://openreview.net/forum?id=jt8wI3ZzXG
