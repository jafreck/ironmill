# ANE Per-Program Compilation Limit Investigation

## Summary

Appending **any** compute ops to the Qwen3-0.6B pre_attn sub-program
causes `ANECCompile() FAILED`. The specific ANE resource that's exhausted
is unknown — the ANE compiler provides no diagnostic beyond the generic
failure code.

## What Was Tested

All tests use the Qwen3-0.6B ONNX model's `layer_0_pre_attn` sub-program
after the structural attention split. The base program contains:

- **RMSNorm:** mul, reduce_mean, add, pow, mul, mul (6 compute ops)
- **Q projection:** conv with weight [2048, 1024, 1, 1]
- **K projection:** conv with weight [1024, 1024, 1, 1]
- **V projection:** conv with weight [1024, 1024, 1, 1]
- **Total weights:** ~8 MB (fp16), 4 entries (3 conv + 1 norm)
- **Function outputs:** 3 (Q, K_proj, V_proj — all conv outputs)

### Test Results

| # | Change | Result | Notes |
|---|--------|--------|-------|
| 1 | No changes (base program) | ✅ Compiles | 13.9 tok/s (FP16 baseline) |
| 2 | Reorder outputs [Q, K, V] | ✅ Compiles | 13.1 tok/s (TQ non-fused) |
| 3 | Add 1 `mul` per K/V output | ❌ FAILED | Scalar const, ~2 bytes extra weight |
| 4 | Add `mul` + `add` + `round` + `clip` per K/V | ❌ FAILED | Full quantization, no rotation |
| 5 | Add grouped conv (rotation) + quantization | ❌ FAILED | +64 KB rotation weight |
| 6 | Add grouped conv + quantization + int8 cast | ❌ FAILED | Same as 5 + cast round-trip |
| 7 | Add `matmul` (rotation) + quantization | ❌ FAILED | ANE rejects mixed conv+matmul |

### Key Observation

Test 3 is the critical one: adding a single `mul(x=K_proj, y=scalar_const)`
to each K/V output adds only ~2 bytes of scalar const weight. The total
weight size barely changes (~8.000 MB → ~8.000 MB). Yet ANE rejects it.

This means **the constraint is NOT a weight size limit**. The failure
occurs when the function outputs are changed from direct conv outputs
to the outputs of downstream compute ops. Something about the program
structure — not the weight data size — triggers the limit.

## Hypotheses (Unconfirmed)

1. **Output op constraint:** ANE may require function outputs to be
   direct outputs of conv/linear ops, not the outputs of elementwise ops.
   (Weakly supported: reordering conv outputs works, but changing outputs
   to non-conv ops fails.)

2. **Total op count limit:** The base program has ~37 non-const ops.
   Adding 8 more (2× mul, add, round, clip) may exceed a per-program
   limit. (Weakly supported: the base program is already large.)

3. **Intermediate buffer limit:** Each additional op requires an
   intermediate buffer. ANE may have a fixed number of buffer slots.
   (No direct evidence.)

4. **Op type restriction:** ANE may not support `round`, `clip`, or
   `cast` ops in programs that also contain `conv` ops with large weights.
   (Contradicted by: these ops work in standalone cache-write program,
   though that uses `matmul` not `conv`.)

5. **Total program memory budget:** The compiled program's memory
   footprint (weights + buffers + instructions) may exceed a hardware
   limit. The 3 large conv ops (~8 MB weights each requiring their own
   intermediate output buffers) may already saturate this budget.
   (Plausible but unconfirmed.)

## How to Reproduce

### Prerequisites

```bash
# Ensure the test fixture exists
ls tests/fixtures/qwen3-0.6b.onnx
# If missing: ./scripts/download-fixtures.sh
```

### Run the benchmark

```bash
# TQ-only (avoids ANE slot conflicts with FP16 baseline)
cargo run -p ironmill-ane --example turboquant_e2e_bench --release -- tq

# Both modes (may hit ANE slot exhaustion on second compile)
cargo run -p ironmill-ane --example turboquant_e2e_bench --release -- both
```

### Expected output (current code — reorder only, no fusion)

```
cache-write fusion: enabled (28 layers, −1 ANE eval/layer)
  │  Compile time: ~6s
  │  Throughput: ~13 tok/s
  │  KV cache size: 14.5 MB
```

### To reproduce the ANE compilation failure

In `inject_cache_write_ops()` in `crates/ironmill-ane/src/inference.rs`,
add any compute op to the K/V outputs. For example, add a `mul` op:

```rust
// After the Q/K/V identification, before the output reordering:
use mil_rs::ir::{Operation, TensorType, Value};

let mut new_ops = Vec::new();
let mut inv_scale_op = Operation::new("const", "tq_cw_inv_scale")
    .with_input("val", Value::Float(522.0))
    .with_output("tq_cw_inv_scale");
inv_scale_op.output_types = vec![Some(TensorType::new(ScalarType::Float16, vec![]))];
new_ops.push(inv_scale_op);

for (prefix, input_ref) in [("k", &k_name), ("v", &v_name)] {
    let output_name = format!("tq_cw_{prefix}_scaled");
    let flat_shape = vec![1, kv_ch, 1, s]; // s from sub.outputs[k_idx].shape[3]
    let mut mul_op = Operation::new("mul", &output_name)
        .with_input("x", Value::Reference(input_ref.clone()))
        .with_input("y", Value::Reference("tq_cw_inv_scale".into()))
        .with_output(&output_name);
    mul_op.output_types = vec![Some(TensorType::new(ScalarType::Float16, flat_shape))];
    new_ops.push(mul_op);
}
func.body.operations.append(&mut new_ops);

// Then update outputs to reference the mul outputs instead of raw K/V
```

The MIL text is dumped to `/tmp/ironmill_debug_layer_0_pre_attn.mil`
on failure for inspection.

### Debug the split

```bash
IRONMILL_SPLIT_DEBUG=1 cargo run -p ironmill-ane --example turboquant_e2e_bench --release -- tq 2>&1 | grep split
```

## Impact on Cache-Write Fusion

The cache-write fusion (Priority 1 in turboquant-zero-copy.md) cannot
be realized by appending ops to pre_attn for models where the projection
weights already saturate ANE's per-program budget.

### Alternative approaches

1. **Split pre_attn further:** Separate Q projection (4 MB) into its own
   sub-program. K/V projections + cache-write ops would be a second
   sub-program (~4 MB + rotation). This keeps the total eval count at 3
   (same as non-fused), so no throughput gain.

2. **Pre-quantize weights:** Apply rotation to the K/V projection weights
   at compile time (fuse rotation into the projection weight matrix).
   Only scalar quantization (mul, round, clip) would need to be appended.
   This is mathematically equivalent and adds less compute, but may still
   exceed the ANE limit.

3. **CPU-side fusion:** Move the cache-write ops to CPU instead of ANE.
   The rotation matmul + quantization would run on CPU between pre_attn
   and attention evals. This trades ANE eval overhead for CPU compute
   overhead — may not be net positive for small models.

4. **Smaller models:** Models with smaller projection weights (e.g.,
   fewer heads or smaller head_dim) may have enough ANE headroom for
   fusion. Test with smaller architectures to validate the approach.

## References

- `inject_cache_write_ops()` in `crates/ironmill-ane/src/inference.rs`
- `turboquant_e2e_bench` example in `crates/ironmill-ane/examples/`
- `step_attention_fused()` in `crates/ironmill-ane/src/turboquant.rs`
- `docs/development/turboquant-zero-copy.md` — performance context
