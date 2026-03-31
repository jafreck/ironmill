# ANE Hardware Constraints & Diagnostics

> Consolidated from `ane-performance-diagnosis.md` and `ane-program-compilation-limit.md`.
> Original docs archived in `docs/archive/`.

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
| Minimum allocation | 16,384 bytes (16KB) | Previously assumed 48KB — empirically corrected |
| Shape rejection | `C > ~768` AND `S < 32` | ANE rejects `[1,C,1,S]` I/O tensors matching this |
| Below-minimum rejection | status `0x1d` | IOSurfaces below 16KB are rejected |

### Data Type Constraints

| Constraint | Details | Notes |
|---|---|---|
| Compute precision | FP16 only | ANE dequantizes INT8 to FP16 before compute — no native INT8 arithmetic |
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
no rotation — structurally identical to FP16 except for the `cast(int8→fp16)`.

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
- INT8+TQ vs INT8raw are nearly identical — dequant/rotation overhead is negligible
- INT8raw vs FP16 shows **pure cast cost** — 3–30% slower depending on cache size
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

### Per-Program Resource Limit

The most consequential constraint discovered during Qwen3-0.6B bringup:

- Pre_attn sub-program with `Q[2048,1024]` + `K[1024,1024]` + `V[1024,1024]`
  projections = ~8 MB fp16 weights, 3 conv outputs, ~37 non-const ops
- Adding **any** compute op (even a single scalar `mul`) causes `ANECCompile() FAILED`
- Reordering outputs can sometimes succeed where adding ops fails
- The constraint is NOT weight size alone — it is a combination of program
  structure, op count, intermediate buffer count, and output-op restrictions

**Practical impact:** Cache-write fusion into pre_attn is infeasible for
models at Qwen3-0.6B scale. The cache-write step must remain a separate
sub-program.

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

### 4. Debug sub-program splitting

If a model compiles but produces wrong output, check sub-program boundaries:
- Verify I/O tensor shapes meet ANE constraints (`C ≤ 768` or `S ≥ 32`)
- Check that S≥32 padding is applied only to appropriate sub-programs
- Confirm weight budget per sub-program is under ~8 MB

## Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| `ANECCompile() FAILED` | Program exceeds resource limit | Split into smaller sub-programs |
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
