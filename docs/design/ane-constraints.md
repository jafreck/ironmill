# ANE Hardware Constraints & Diagnostics

> Consolidated from `ane-performance-diagnosis.md` and `ane-program-compilation-limit.md`.
> Original docs archived in `docs/archive/`.
>
> Key external reference: [Inside the M4 Apple Neural Engine](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)
> (maderix, 2026) — benchmarks using direct `_ANEClient` API.

## Performance Architecture

The ANE is fundamentally a **convolution engine** optimized for deep operation
graphs. Key findings from maderix benchmarks on M4:

| Principle | Detail |
|---|---|
| **Deep graphs, not wide** | Chain 16–64 ops in one MIL program. Single ops waste ~70% of capacity due to dispatch overhead (~0.095ms per eval). |
| **Conv over matmul** | 1×1 convolutions use the fast datapath. Matmul is ~3× slower for the same FLOPs. |
| **Stay under 32 MB working set** | ANE has ~32 MB on-chip SRAM. Exceeding it causes DRAM spills and ~30% throughput drop. |
| **INT8 saves bandwidth, not compute** | ANE dequantizes INT8 to FP16 before compute. No 2× speedup — only smaller memory transfers. |
| **True peak: 19 TFLOPS FP16** | Apple's "38 TOPS" counts INT8 ops as 2× FP16. Real throughput is 19 TFLOPS regardless of quantization. |
| **94% utilization at depth 32+** | Deep conv graphs reach near-theoretical peak. |
| **6.6 TFLOPS/W efficiency** | ~80× more efficient per FLOP than A100. Hard power gating at idle (0 mW). |

### Implications for ironmill

- **Minimize sub-program count per layer.** Each ANE eval has ~0.095ms dispatch
  overhead. A 3-way split (pre_attn + attention + post_attn) pays 3× overhead
  vs a single deep program. The ideal is one program per layer.
- **Use conv 1×1 instead of matmul** for all weight projections (already done
  via `AneMatmulToConvPass`).
- **Don't assume op count causes compilation failures.** The ANE compiler
  (`_ANECCompile`) accepts 64+ op programs. Failures are more likely caused by
  specific op combinations, invalid MIL (duplicate variable names, unsupported
  ops like `gather`), or shape constraints — not by program size.

## Hardware Limits

### Tensor Constraints

| Constraint | Limit | Notes |
|---|---|---|
| Max tensor dimension | 16,384 | Per dimension |
| Max conv kernel | 16×16 | |
| Max matmul/linear inner dim | 16,384 | |
| Channel alignment | 32 | Required for ANE efficiency |
| Byte alignment | 64 | Memory access alignment |
| Tensor layout | `[1, C, H, W]` (NCHW) | ANE-native format |

### IOSurface Constraints

| Constraint | Limit | Notes |
|---|---|---|
| Minimum allocation | 16,384 bytes (16KB) | Previously assumed 48KB - empirically corrected |
| Shape rejection | `C > ~768` AND `S < 32` | ANE rejects `[1,C,1,S]` I/O tensors matching this |
| Below-minimum rejection | status `0x1d` | IOSurfaces below 16KB are rejected |

### Data Type Constraints

| Constraint | Details | Notes |
|---|---|---|
| Compute precision | FP16 only | ANE dequantizes INT8 to FP16 before compute - no native INT8 arithmetic |
| INT8 support | Storage/transport only | `cast(int8→fp16)` ✅, `cast(fp16→int8)` ✅, but INT8 arithmetic (`add`/`mul` on int8) ❌ |
| INT4/UINT4 support | Comprehensively rejected | All paths rejected: inputs, outputs, casts, consts, `constexpr_lut_to_dense`, `constexpr_blockwise_shift_scale` |
| INT8 function outputs | ❌ Rejected | INT8 function *inputs* work, but INT8 *outputs* are rejected by the ANE compiler |

### INT8 Cache Bandwidth: No Throughput Gain

**Critical finding:** INT8 KV cache inputs provide **no throughput advantage**
over FP16 on ANE, at any model size or sequence length tested. INT8 is
consistently 3–30% *slower* than FP16 for the same attention computation.

The ANE's `cast(int8→fp16)` op adds O(seq_len × kv_channels) work that is
**not compensated** by reduced memory bandwidth. Even at cache sizes well
beyond SRAM (~32 MB), FP16 outperforms INT8.

**Benchmark methodology:** Identical attention MIL programs compiled and
evaluated on ANE, differing only in cache input dtype (INT8 vs FP16).
The "INT8 raw" column uses `cache_int8=true` with no dequant scale and
no rotation - structurally identical to FP16 except for the `cast(int8→fp16)`.

```
Config                        INT8+TQ  INT8raw     FP16 raw/fp16    Cache
                                 (μs)     (μs)     (μs)              (MB)
────────────────────────────────────────────────────────────────────────
Qwen3-0.6B @ 512                  255      245      225    0.92x     1.0
Qwen3-0.6B @ 2048                 787      761      739    0.97x     4.0
8B-style (8 KV) @ 512             634      612      590    0.96x     2.0
8B-style (8 KV) @ 2048           2155     2146     2006    0.93x     8.0
8B-style (8 KV) @ 4096           4187     4156     3901    0.94x    16.0
8B MHA (32 KV) @ 2048            1971     1965     1429    0.73x    32.0
8B MHA (32 KV) @ 4096            3811     3797     2681    0.71x    64.0
8B MHA (32 KV) @ 8192            7450     7412     5172    0.70x   128.0
8B GQA (8 KV) @ 8192             8145     8139     7622    0.94x    32.0
```

Key observations:
- INT8+TQ vs INT8raw are nearly identical - dequant/rotation overhead is negligible
- INT8raw vs FP16 shows **pure cast cost** - 3–30% slower depending on cache size
- The gap *widens* at larger KV dimensions (32 KV heads: 30% slower at 128 MB cache)
- Aligns with maderix's finding: "INT8 and FP16 deliver nearly identical throughput.
  The ANE dequantizes INT8 weights to FP16 before compute."

**Implication for TurboQuant:** INT8 KV cache on ANE provides **memory savings
only** (50% cache reduction), never throughput gains. TQ's value proposition on
ANE is enabling longer contexts within memory budgets, not faster inference.

**Reproduce:**
```sh
cargo run -p ironmill-inference --example cache_bandwidth_bench --release
```

### Compilation Constraints

| Constraint | Limit | Notes |
|---|---|---|
| Compile budget | ~119 per process | Bypassable with separate compile/load lifecycle |
| Simultaneously loaded models | ~55 | Model-size dependent; observed on M-series |
| Sub-program weight budget | ~8 MB fp16 | Per-program, not per-model |

### Per-Program Compilation Failures

The ANE compiler (`_ANECCompile`) rejects programs without providing a
reason code. Known causes of compilation failure include:

- **Unsupported ops** — `gather` is genuinely unsupported at runtime
- **Invalid MIL** — duplicate variable names, dangling references
- **Op combination bugs** — `split` combined with `matmul + softmax + tile`
  fails even though each op compiles individually
- **Shape constraints** — `C > ~768` with `S < 32` on I/O tensors

Op count alone is **not** a reliable predictor of failure. The maderix
benchmarks demonstrate 64-op deep graphs compiling successfully. The
Qwen3-0.6B pre_attn sub-program compiles with ~37 non-const ops and
~8 MB of weights.

When `ANECCompile()` fails, the dumped MIL text (written to `/tmp/`)
should be inspected for the actual cause rather than assumed to be a
size or op-count issue.

## Diagnostic Workflow

### 1. Validate model ANE compatibility

```sh
ironmill validate model.mlpackage
```

Reports ANE compatibility percentage, unsupported ops, compute-unit
annotations, and fallback reasons.

### 2. Benchmark inference

```sh
cargo run -p ironmill-bench -- --model model.safetensors --arch qwen3
```

Measures per-layer latency, total throughput, memory usage.

### 3. Probe individual ops

```sh
cargo run -p ironmill-ane --example ane_op_probe    # compile-time probe (85 ops)
cargo run -p ironmill-ane --example ane_op_eval     # eval-time verification
cargo run -p ironmill-ane --example ane_op_fuzz     # name fuzzing (400+ variants)
cargo run -p ironmill-ane --example ane_dtype_probe  # data type probe
```

### 4. Debug compilation failures

When `ANECCompile()` fails, the MIL text is dumped to `/tmp/ironmill_debug_*.mil`.
Inspect it for:
- Duplicate variable names (e.g., `z_output1` defined twice)
- Unsupported ops (`gather`, `scatter`)
- Problematic op combinations (`split` + `matmul` + `softmax`)
- Dangling references (ops referencing stripped/removed values)
- Shape constraints on I/O tensors

## Common Failure Modes

| Symptom | Likely cause | Investigation |
|---|---|---|
| `ANECCompile() FAILED` | Invalid MIL, unsupported op, or op combination bug | Inspect dumped MIL in `/tmp/`. NOT necessarily op count — 64+ op programs compile fine. |
| `status=0x1d` at eval | IOSurface too small (< 16KB) | Increase `MIN_SURFACE_ALLOC` |
| `status=0x1d` at eval | Multi-input request pattern | Reduce number of inputs per sub-program |
| Wrong output shapes | S≥32 padding applied to attention sub-programs | Use per-sub-program padding |
| Model loads but outputs zeros | Output tensors not extracted from IOSurface | Check tensor read-back code |
| Budget exhaustion at ~55 models | Too many loaded models | Use load/unload lifecycle |
| Compile budget at ~119 | Too many compilations per process | Cache compiled artifacts; restart process |
| `gather` rejection | Runtime gather unsupported | Use CPU for dynamic index lookups |

## References

- [ANE Op Support Matrix](ane-op-support-matrix.md) — 74 verified ops with error bounds
- [ANE Inference Design](ane-inference.md) — inference pipeline status
- [TurboQuant Design](turboquant.md) — INT8 KV cache compression
- [Inside the M4 Apple Neural Engine](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615) — maderix benchmarks (external)
