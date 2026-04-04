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

## Attention & compute

### 1. Fused SDPA with simdgroup matrix tiling

**Status:** Proposed ([ADR-0001](../adr/0001-fused-sdpa-metal-kernel.md))
**Expected impact:** 3–5× prefill throughput at ≥2048 tokens
**Complexity:** High
**SOTA reference:** FlashAttention-4 (Dao et al., arXiv 2026); FA3 (NeurIPS 2024) for Metal-applicable tiling

The single largest bottleneck. The current attention kernel dispatches one threadgroup per (head, query), giving O(n²) threadgroup count with severe causal-mask load imbalance. At 8192 tokens the GPU is 73% slower than CPU.

FlashAttention-4 is the latest in the FA lineage, co-designing algorithm and kernel pipelining for Blackwell's asymmetric hardware scaling. Key hardware-agnostic insights from FA4: conditional rescaling (only ~10% of softmax output blocks need numerically-stabilizing rescale, reducing work vs always rescaling) and polynomial SFU-free exponential approximation. FA3's core tiling strategy (Q blocks × KV tiles with online softmax) remains the most applicable to Metal's simdgroup matrix hardware, since FA4's async TMA and 2-CTA MMA features are CUDA/Blackwell-specific.

The Metal adaptation should tile over both Q and KV dimensions using 8×8 simdgroup tiles, keeping all intermediate data in SRAM. GQA-aware: multiple Q heads share KV tile loads from the same KV head group. Adopt FA4's conditional rescaling to skip unnecessary numerical stabilization passes.

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

## Caching & memory

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

## Serving & architecture

### 7. Continuous batching with vAttention-style memory

**Status:** Not implemented
**Expected impact:** N× throughput for concurrent requests
**Complexity:** High
**SOTA reference:** vAttention (Microsoft, ASPLOS 2025)

vAttention improves on PagedAttention by maintaining virtual contiguity of KV cache memory while enabling dynamic physical allocation. This means standard attention kernels (like our fused SDPA) work without modification — no page table indirection in the kernel. Memory is allocated on-demand but appears contiguous to the GPU.

On Metal/Apple Silicon, the equivalent would be using Metal's virtual memory APIs (if available) or a simpler pool allocator with contiguous sub-allocations per sequence. The key insight: avoid rewriting attention kernels for paging.

Continuous batching dynamically adds/removes sequences as they complete, maximizing GPU utilization. Combined with vAttention-style allocation, this enables production serving on Apple Silicon.

### 8. Cross-layer KV cache sharing (CLA)

**Status:** Not implemented (TurboQuant handles per-vector compression; CLA handles per-layer reduction)
**Expected impact:** 2× KV cache memory reduction (composes with TurboQuant for multiplicative savings)
**Complexity:** Medium (requires model support or fine-tuning)
**SOTA reference:** CLA (NeurIPS 2024), FusedKV (ICLR 2026)

Cross-Layer Attention shares KV caches between adjacent layers — subsequent layers reuse the KV cache from "anchor" layers instead of storing their own. This is orthogonal to TurboQuant's per-vector quantization: CLA reduces the number of KV vectors stored, TurboQuant compresses each vector. Combined: a model with 28 layers using CLA (14 anchor layers) + TurboQuant INT4 would use ~3.5% of the naive FP16 KV memory.

FusedKV (ICLR 2026) extends CLA with learnable cross-layer fusion for better quality-memory trade-offs.

Requires model architecture support (trained with CLA). ironmill detects CLA config and shares cache buffers between layers accordingly.

### 9. Multi-Head Latent Attention (MLA) support

**Status:** Not implemented
**Expected impact:** 5–10× KV cache compression vs MHA
**Complexity:** Medium
**SOTA reference:** DeepSeek-V2/V3 MLA architecture; FlashMLA (DeepSeek 2025); MHA2MLA (arXiv 2025)

MLA projects keys and values into a shared low-dimensional latent space, storing only the compressed latent per token. During attention, the latent is up-projected on-the-fly. Used by DeepSeek-V2/V3 models. FlashMLA provides optimized CUDA kernels with FP8 cache support and matrix absorption (pre-fusing projection matrices to reduce runtime ops). MHA2MLA demonstrates converting existing MHA models (Llama, Qwen) to MLA via joint low-rank SVD approximation and minimal fine-tuning, achieving 92% KV cache reduction with negligible quality loss — meaning ironmill needn't wait for natively MLA-trained models.

ironmill would need:
- Detect MLA config in model metadata
- Store compressed latent KV cache instead of full K/V
- On-the-fly up-projection in the attention kernel (or matrix absorption to eliminate it)
- Metal-optimized MLA kernel analogous to FlashMLA

---

## Speculation & advanced

### 10. Speculative Streaming (no auxiliary model)

**Status:** Not implemented
**Expected impact:** 2–3× decode throughput without draft model
**Complexity:** High
**SOTA reference:** Speculative Streaming (OpenReview 2025)

Eliminates the need for a separate draft model by integrating speculated token planning within the target model using multi-stream attention heads. Each forward pass produces both the "correct" next token and speculative future tokens. Ideal for edge/resource-constrained deployment where loading a separate draft model is impractical.

### 11. Prefill/decode phase separation

**Status:** Not implemented
**Expected impact:** 1.5–2× prefill throughput on M-series chips
**Complexity:** Medium
**SOTA reference:** MLX phase-optimized scheduling (Apple, 2025)

Prefill is compute-bound (large batch matmuls) while decode is memory-bandwidth-bound (single-token matvecs). MLX exploits this by routing prefill to the Neural Engine (which has dedicated matrix units) and decode to the GPU (which has maximum memory bandwidth). On Apple Silicon with unified memory, there's no data copy cost for this split.

ironmill has an experimental ANE inference client (`ane-direct`), but it should not be a hard dependency for the Metal inference path. The preferred approach is to route through MLX's scheduling layer (which handles ANE/GPU dispatch internally) rather than the low-level ANE-direct path. If MLX integration is not available, the Metal GPU path should handle both phases — the optimization becomes selecting the right kernel (batched matmul for prefill, matvec for decode) rather than switching compute units.

This is a "nice to have" behind MLX integration, not a requirement for the Metal pipeline.

### 12. Structured generation (grammar/JSON-constrained sampling)

**Status:** Not implemented
**Expected impact:** Required for agent and tool-use applications
**Complexity:** Medium
**SOTA reference:** XGrammar 2 (MLSys 2025)

Constrained generation forces model output to conform to a schema (JSON, XML, function signatures) by masking invalid tokens at each decode step. XGrammar 2 is the current SOTA: it classifies tokens as context-independent (precomputed and cached) vs context-dependent (checked at runtime with a persistent pushdown automaton stack), achieving up to 100× speedup over prior approaches. JIT mask compilation reduces preprocessing from seconds to milliseconds for large grammars. TagDispatch efficiently switches grammar masks on-the-fly for multi-schema tool-calling.

Essential for tool-use agents, structured data extraction, and API response formatting. Composes with all sampling strategies (min-p, temperature, etc.) as an additional logit mask applied before sampling.

### 13. TurboSpec adaptive speculation

**Status:** Not implemented
**Expected impact:** +20–50% over static speculative decoding
**Complexity:** Medium
**SOTA reference:** TurboSpec (Berkeley 2025)

Closed-loop feedback system that dynamically tunes speculation depth, draft tree width, and acceptance thresholds at runtime based on observed acceptance rates. Maximizes "goodput" (accepted tokens per second) rather than raw proposals. Would be implemented as a runtime controller wrapping the EAGLE-3 draft head.

---

## Implementation order

All items will be implemented. Ordering reflects hard dependencies only — items at the same level can be parallelized across agents.

```
Independent (no dependencies, can start immediately):
  ├── #3  Min-p sampling + full sampler chain
  ├── #5  KV cache reuse across turns
  ├── #1  Fused SDPA kernel (ADR-0001)
  ├── #6  Sliding window attention
  ├── #8  Cross-layer KV sharing (CLA)
  ├── #9  MLA support
  └── #12 Structured generation (JSON/grammar)

Depends on #1 (Fused SDPA):
  └── #11 Prefill/decode phase separation

Depends on #5 (KV cache reuse):
  └── #4  RadixAttention prompt caching

Depends on #3 (sampling) + #5 (KV reuse):
  └── #2  EAGLE-3 / P-EAGLE speculative decoding

Depends on #2 (speculative decoding):
  ├── #10 Speculative Streaming
  └── #13 TurboSpec adaptive speculation

Depends on #1 (Fused SDPA) + #5 (KV reuse):
  └── #7  Continuous batching + vAttention
```

## References

- FlashAttention-4: https://arxiv.org/abs/2603.05451
- FlashAttention-3: https://arxiv.org/abs/2407.08608
- EAGLE-3: https://github.com/SafeAILab/EAGLE
- P-EAGLE: https://vllm.ai/blog/p-eagle
- SpecBundle: https://www.lmsys.org/blog/2025-12-23-spec-bundle-phase-1/
- Min-p sampling: https://arxiv.org/abs/2407.01082
- RadixAttention: https://www.lmsys.org/blog/2024-01-17-sglang/
- vAttention: https://arxiv.org/abs/2405.04437
- CLA: https://arxiv.org/abs/2405.12981
- FusedKV: https://openreview.net/forum?id=4pivvEJiCl
- MLA / DeepSeek-V2: https://arxiv.org/abs/2405.04434
- FlashMLA: https://github.com/deepseek-ai/FlashMLA
- MHA2MLA: https://arxiv.org/abs/2502.14837
- TurboSpec: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html
- Speculative Streaming: https://openreview.net/forum?id=jt8wI3ZzXG
- XGrammar: https://arxiv.org/abs/2411.15100
- MLX phase scheduling: https://yage.ai/share/mlx-apple-silicon-en-20260331.html
