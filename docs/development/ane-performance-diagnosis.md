# ANE Performance Diagnosis

Guide for measuring and diagnosing ANE performance issues and hardware limit constraints.

## Quick Reference

| Tool | What it measures | Invocation |
|------|-----------------|------------|
| `ironmill validate` | Per-op ANE compatibility, limit violations | `cargo run --release -p ironmill-cli -- validate <model> --format text` |
| `ironmill-bench` | Latency, throughput, memory, power | `cargo run --release -p ironmill-bench -- --model <model> --backend ane` |
| `ane_op_probe` | ANE compiler acceptance per op | `cargo run -p ironmill-ane --example ane_op_probe --release` |
| `ane_op_eval` | Numeric correctness vs CPU reference | `cargo run -p ironmill-ane --example ane_op_eval --release` |
| `ane_dtype_probe` | Data type support (casts, arithmetic, matmul) | `cargo run -p ironmill-ane --example ane_dtype_probe --release` |
| Split debug | Sub-program partitioning details | `IRONMILL_SPLIT_DEBUG=1 cargo run ...` |

## ANE Hardware Limits

From `crates/mil-rs/src/validate.rs`:

| Constraint | Limit | Consequence |
|-----------|-------|-------------|
| Max tensor dimension | 16,384 | Op falls back to CPU/GPU |
| Max conv kernel | 16×16 | Op falls back to CPU/GPU |
| Max matmul/linear inner dim | 16,384 | Op falls back to CPU/GPU |
| Channel alignment | 32 | Reduced throughput if unaligned |
| Byte alignment | 64 | Slower memory access if unaligned |
| Min IOSurface size | 16,384 bytes | Rejection with status 0x1d if smaller |
| Compile budget | ~119 per process | `BudgetExhausted` error |
| Sub-program weight size | 64 MB default | Auto-chunked by splitter |

## Diagnosis Workflow

### Step 1: Validate — identify limit violations

```bash
cargo run --release -p ironmill-cli -- validate <model> --format text
```

The validation report shows:
- **ANE compatibility %** — what fraction of ops can run on ANE
- **ANE compute %** — what fraction of FLOPs run on ANE
- **Per-op fallback reasons** — exactly which limit each op violates
- **Performance annotations** — supported-but-slow patterns (large gather, channel-moving transpose, large reshape)

Use `--format json` for structured output.

### Step 2: Benchmark — measure actual impact

```bash
# Compare all backends
cargo run --release -p ironmill-bench -- --model <model> --backend all --iterations 200

# Direct ANE path (bypasses CoreML dispatch)
cargo run --release -p ironmill-bench --features ane-direct -- --model <model> --ane-direct

# With power/energy metrics (requires sudo for powermetrics)
cargo run --release -p ironmill-bench -- --model <model> --backend ane --power

# Save and compare baselines
cargo run --release -p ironmill-bench -- --model <model> --save-baseline before_change
# ... make changes ...
cargo run --release -p ironmill-bench -- --model <model> --compare-baseline before_change
```

Key metrics to compare:
- **Latency** (mean, p50, p95, p99)
- **Utilization %** — time spent in compute vs dispatch overhead
- **Memory** — RSS growth, model load cost, efficiency ratio
- **Power** — inferences/watt, joules/inference

### Step 3: Probe — test op acceptance

```bash
# Which ops does ANE accept?
cargo run -p ironmill-ane --example ane_op_probe --release

# Are accepted ops numerically correct?
cargo run -p ironmill-ane --example ane_op_eval --release

# What data types are supported?
cargo run -p ironmill-ane --example ane_dtype_probe --release

# Discover undocumented op support
cargo run -p ironmill-ane --example ane_op_fuzz --release
```

### Step 4: Debug splitting

```bash
IRONMILL_SPLIT_DEBUG=1 cargo run --release -p ironmill-cli -- compile <model>
```

This shows how the model is partitioned into sub-programs (embedding, layer_N, lm_head) and whether attention boundaries are split.

## Interpreting Results

### "Is this an ANE limit or a splitting issue?"

1. Run `validate` — if compatibility % is high but performance is poor, the issue is likely splitting/dispatch overhead, not op rejection.
2. Run `ironmill-bench --backend all` — if GPU outperforms ANE significantly, check utilization %. Low utilization suggests dispatch overhead from too many sub-programs.
3. Check split debug output — if layers are being chunked beyond the natural boundaries, the `max_weight_size` budget may need tuning.

### "Is this an op-support gap?"

1. Run `validate` — look for fallback ops with reasons like "unsupported op type".
2. Run `ane_op_probe` — test whether alternative formulations of the same operation are accepted.
3. Check `docs/research/ane-op-support-matrix.md` for known support status.

### Common patterns

| Symptom | Likely cause | Next step |
|---------|-------------|-----------|
| Low ANE compatibility % | Many ops exceed dimension limits | Check if op splitting pass is active |
| High compatibility but slow | Dispatch overhead from many sub-programs | Review split granularity |
| Numeric errors on ANE | Precision loss in quantized path | Run `ane_op_eval`, compare FP16 vs INT8 |
| Status 0x1d rejection | IOSurface tensor too small | Check tensor dimensions, `ANE_MIN_SURFACE_BYTES` |
| `BudgetExhausted` | Too many compiled programs | Reduce `max_programs` or reuse models |
