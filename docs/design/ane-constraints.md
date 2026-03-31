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
