# ANE Inference Optimizations

Current state: Qwen3-0.6B end-to-end on ANE with FP16 baseline (6.1
tok/s) and TurboQuant INT8 (5.5 tok/s, 14.5 MB KV cache). Both paths
run but neither computes real attention - 0% token agreement between
them confirms output is not meaningful yet.

Key constraints discovered (see `ane-attention-split-investigation.md`):
- ANE rejects IOSurface I/O tensors `[1, C, 1, S]` when C > ~768 and
  S < 32 (status 0x1d)
- Multi-input ANE requests can cause 0x1d errors (documented by
  maderix/ANE; ironmill uses multi-input successfully with 2-4 inputs,
  so this may be hardware/macOS-version specific)
- All input IOSurfaces in a single eval must have the same alloc size

---

## Phase 0 - Pad TurboQuant MIL shapes to S ≥ 32

**Problem:** TurboQuant's MIL programs declare S=1 shapes for their
function inputs: cache-write uses `[1, kv_ch, 1, 1]` (kv_ch=1024) and
attention uses `[1, q_ch, 1, 1]` (q_ch=2048). Both exceed the ~768
channel threshold. These currently work only because `MIN_SURFACE_ALLOC`
(48KB) inflates the IOSurface beyond the minimum the ANE requires.
Removing MIN_SURFACE_ALLOC (Phase 5) would cause these to fail.

**Fix:** Apply the same S ≥ 32 padding to TurboQuant's MIL emission in
`turboquant_mil.rs`. Update `emit_cache_write_mil` and
`emit_attention_mil` to declare input shapes with S=32 instead of S=1.
Update the staging buffer writes to use scatter (column-0) layout, and
reads to use gather.

**Files:**
- `crates/ironmill-ane/src/turboquant_mil.rs` - MIL shape declarations
- `crates/ironmill-ane/src/turboquant.rs` - staging I/O

## Phase 1 - Implement FP16 attention sub-programs (correctness)

**Problem:** The FP16 baseline path has no real attention computation.
When `fp16_attn` is `None` (always, currently), `decode()` passes raw Q
through as the attention output. Without attention, output is garbage -
0% token agreement between baseline and TurboQuant confirms this.

**Fix:** The attention split identifies attention cluster ops (softmax,
QK matmul, AV matmul) between Q/K/V projections and O-projection.
Instead of stripping them, compile them as `layer_N_fp16_attn`
sub-programs and populate the `fp16_attn` field.

All required ops are ✅ eval-verified on ANE: `softmax`, `matmul`,
`mul`, `reshape`, `transpose`.

**Implementation:**
1. In `split.rs`, add `SplitConfig` option to emit the attention cluster
   as `layer_N_fp16_attn` sub-programs instead of discarding them
2. Apply matmul→conv and SiLU passes to these sub-programs
3. In `inference.rs` `compile()`, detect `layer_N_fp16_attn` sub-
   programs and populate `fp16_attn`
4. The `decode()` FP16 path already has wiring to call `fp16_attn`

**Impact:** Correctness. Required for meaningful output and quality
benchmarks. Not a throughput optimization.

**Files:**
- `crates/ironmill-ane/src/split.rs` - attention cluster emission
- `crates/ironmill-ane/src/inference.rs` - sub-program classification

## Phase 2 - Fix structural attention split

**Problem:** The structural attention split falls back to the name
heuristic on every run ("structural attention split failed, falling back
to name heuristic (12 ops)"). This produces suboptimal sub-programs:
- pre_attn is just RMSNorm (no Q/K/V projections)
- post_attn contains projections + O projection + FFN

**Root cause (empirically confirmed):** `find_softmax_ops()` returns
empty - no softmax ops found in the 12 ops per layer. The ONNX model
uses `GroupQueryAttention` which the converter decomposes into ops
including `softmax`, so softmax should exist. The 12 ops per layer is
suspiciously few - the issue is likely in how model_split partitions
layers before the attention split runs, or in how the decomposed GQA
ops are structured after passes. Needs further investigation.

**Fix:** Debug with op-type listing for a single layer's ops at split
time. Determine whether softmax is missing (decomposition issue) or
present but not found (detection issue).

**Impact:** Better sub-program boundaries, more ops on ANE
(projections in pre_attn). Prerequisite for Phase 1 (FP16 attention
requires correct attention cluster identification).

**Files:**
- `crates/ironmill-ane/src/split.rs` - structural split logic

## Phase 3 - Pre-allocate TurboQuant output tensors

**Problem:** `step_attention()` allocates `k_quant`, `v_quant`, and
`attn_out` as new IOSurface-backed tensors on every call. That's 3
IOSurface creates/destroys × 29 layers × N tokens. IOSurface allocation
involves kernel calls and is relatively expensive.

**Fix:** Pre-allocate these in the `TurboQuantModel` struct during
`compile()`, alongside the existing staging buffers. Reuse them across
calls.

**Impact:** Reduced allocation overhead. Simple change.

**Files:**
- `crates/ironmill-ane/src/turboquant.rs` - `TurboQuantModel` struct,
  `compile()`, `step_attention()`

## Phase 4 - Remove MIN_SURFACE_ALLOC

**Depends on:** Phase 0 (TurboQuant shape padding).

**Problem:** `tensor.rs` enforces a 48KB minimum IOSurface allocation
(`MIN_SURFACE_ALLOC = 49152`). Both Orion and maderix/ANE use exact-fit
allocations with no minimum. The 48KB minimum wastes memory for small
tensors and was a false assumption.

**Fix:** Remove `MIN_SURFACE_ALLOC` and let IOSurfaces be exactly the
data size (`C * S * bytes_per_element`). Update `uniform_alloc_size` to
remove the floor.

**Warning:** This will NOT eliminate the need for staging buffers.
Tensors with different shapes still have different byte sizes and
therefore different alloc sizes. The ANE uniform-alloc constraint
remains regardless of MIN_SURFACE_ALLOC. Also, removing MIN_SURFACE_
ALLOC without Phase 0 will break TurboQuant (S=1 shapes with C > 768
currently only work because the 48KB floor inflates the IOSurface).

**Risk:** Low if Phase 0 is done first.

**Files:**
- `crates/ironmill-ane/src/tensor.rs` - `MIN_SURFACE_ALLOC`,
  `uniform_alloc_size`, `new_with_min_alloc`

## Phase 5 - Reduce TurboQuant CPU round-trips

**Problem:** `step_attention` does 8 CPU↔IOSurface memcpy operations
per layer (3 `gather_column0` reads + 3 staging `write_f16` + 2
`read_f16` for INT8 interception). At 29 layers, that's 232 memcpy
operations per token.

The CPU interception (reading quantized K/V from ANE, converting to
INT8 bytes, writing to persistent cache) is architecturally necessary -
ANE outputs whole tensors with no mechanism for partial/offset writes,
so the ANE can't write directly to a specific position in the persistent
KV cache IOSurface.

**Fix options:**
- Replace `gather_column0` + `write_f16` staging round-trips with
  direct IOSurface-to-IOSurface byte copies (avoid the CPU Vec
  intermediary). This requires `IOSurfaceLock` on both surfaces and
  a strided memcpy.
- After Phase 0 (S=32 padding for TurboQuant MIL), the staging buffers
  may become unnecessary if TurboQuant alloc sizes can be made to match
  pre-attn alloc sizes.

**Impact:** ~10-20% latency reduction for TurboQuant path.

**Files:**
- `crates/ironmill-ane/src/turboquant.rs` - `step_attention()`

## Phase 6 - Spatial input packing

**Problem:** All sub-program I/O tensors have dim 3 (S) padded from 1
to 32 to satisfy the ANE minimum-S constraint. This means:
- Tensors are 32x larger than needed
- `write_f16_padded` scatters C values across stride-32 positions
- `read_f16_channels` gathers them back
- ANE computes on all 32 columns (31 are zeros)

**Fix (maderix pattern):** Pack useful data into the spatial dimension
instead of padding with zeros. maderix/ANE packs activations + weight
matrices into a single input tensor along dim 3, then uses
`slice_by_size` inside the MIL to extract them.

For ironmill:
1. For each sub-program, pack all function inputs into a single tensor
   along dim 3 (spatial concatenation)
2. Emit `slice_by_size` ops at the start of the MIL to extract each
   logical input
3. Pack all outputs into a single tensor via `concat`
4. `decode()` writes all inputs contiguously into one IOSurface

This also addresses the multi-input robustness concern (Phase 7).

**Impact:** Unverified - directionally reduces wasted compute and
eliminates scatter/gather overhead, but the magnitude depends on whether
ANE latency is compute-bound or dispatch-bound for these small tensors.

**Files:**
- `crates/ironmill-ane/src/inference.rs` - `compile()`, `decode()`,
  `compile_and_load_sub()`
- New pass or post-split transform to pack/unpack I/O

## Phase 7 - Single-input robustness

**Problem:** maderix/ANE documents that multi-input ANE requests cause
0x1d errors. Ironmill currently uses multiple inputs successfully:
- post_attn: 2 inputs (attention output + residual)
- TurboQuant cache-write: 3 inputs (K, V, rotation matrix)
- TurboQuant attention: 4 inputs (Q, K_cache, V_cache, unrotation)

These work on the current hardware/macOS version but may be fragile.

**Fix:** Pack all inputs into a single spatial tensor (same as Phase 6).

**Note:** Subsumed by Phase 6 if spatial packing is implemented.

**Impact:** Robustness across macOS versions and hardware generations.

**Files:**
- `crates/ironmill-ane/src/inference.rs`
- `crates/ironmill-ane/src/turboquant_mil.rs`

## Dependencies

```
Phase 0 (TQ S=32 padding) ──→ Phase 4 (remove MIN_SURFACE_ALLOC)
Phase 2 (fix structural split) ──→ Phase 1 (FP16 attention)
Phase 6 (spatial packing) ──→ Phase 7 (single-input, subsumed)

Independent: Phase 3 (pre-alloc TQ tensors)
Independent: Phase 5 (reduce TQ memcpy)
```

Three parallel tracks:

| Track | Phases | Goal |
|-------|--------|------|
| **Correctness** | 2 → 1 | Real attention output |
| **TQ robustness** | 0 → 4 | Remove fragile MIN_SURFACE_ALLOC dependency |
| **Performance** | 3, 5, 6 | Reduce overhead (all independent of each other) |

Phases 0, 2, 3 can all start immediately in parallel.

## References

- maderix/ANE: https://github.com/maderix/ANE - training on ANE,
  documents single-input constraint and spatial packing pattern
- Orion: https://github.com/mechramc/Orion - ANE inference runtime,
  exact-fit IOSurface allocation, GPT-2 124M (d_model=768)
- `docs/development/ane-attention-split-investigation.md` - root cause
  analysis of the 0x1d eval error and ANE shape constraints
- `docs/research/ane-op-support-matrix.md` - op support verification
