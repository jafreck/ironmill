# INT4 Affine Weight Quantization — End-to-End Metal Inference Pipeline

## Problem

The CWQ implementation added:
- `AffineQuantizePass` — produces INT4 `constexpr_affine_dequantize` MIL ops
- INT4 Metal dequant shader — unpacks/dequantizes INT4 weights on GPU
- `AffineQuantizedWeight` + dispatch in `MetalInference::run_pipeline()`

But these pieces aren't connected end-to-end. A user cannot yet:
```bash
ironmill compile model.safetensors --target gpu --quantize int4
ironmill-bench --model model.ironml-gpu --backend metal --perplexity
```

The missing link is the **compile → bundle → load** pipeline: getting INT4
affine-quantized weights from the MIL IR into a `.ironml-gpu` bundle that
the Metal inference engine can load.

## Current State

### What works

```
SafeTensors → SafeTensorsProvider → MetalWeights::load() → FP16 inference ✅
SafeTensors → GpuCompileBuilder → PolarQuant → .ironml-gpu → MetalBundleProvider → inference ✅
MIL Program → AffineQuantizePass → constexpr_affine_dequantize ops ✅ (but no export path)
MetalWeights::load() with AffineDequantize { bit_width=4 } → AffineQuantizedWeight → INT4 dispatch ✅
```

### What's broken

```
SafeTensors → ??? → INT4 affine quantize → .ironml-gpu → MetalBundleProvider → INT4 inference ❌
```

The gap has 3 parts:

1. **`GpuCompileBuilder` only supports PolarQuant** — cannot run
   `AffineQuantizePass` or accept an arbitrary `PassPipeline`
   (`crates/ironmill-compile/src/gpu/mod.rs:39-136`)

2. **Bundle schema has no affine tensor format** — `write_gpu_bundle()`
   serializes `AffineDequantize` weights as plain `Dense` (CPU-dequantized
   FP16), discarding the INT4 quantization
   (`crates/ironmill-compile/src/gpu/bundle.rs:119-132`)

3. **`MetalBundleProvider` can't read affine tensors** — only returns
   `QuantizationInfo::None` or `LutToDense`
   (`crates/ironmill-inference/src/metal/bundle.rs:77-127`)

## Implementation Plan

### Task 1: Extend GPU bundle schema for affine tensors

**Files:**
- `crates/ironmill-core/src/gpu/bundle.rs` — `TensorManifest` schema
- `crates/ironmill-compile/src/gpu/bundle.rs` — `write_gpu_bundle()`

Add an `affine_dequantize` variant to the tensor manifest:

```rust
// In TensorManifest (or equivalent struct):
pub struct AffineTensorManifest {
    pub quantized_data_file: String,   // packed INT4 bytes
    pub scales_file: String,           // per-group FP16 scales
    pub zeros_file: String,            // per-group FP16 zero points
    pub shape: Vec<usize>,             // original tensor shape
    pub bit_width: u8,                 // 4
    pub group_size: usize,             // 128
    pub axis: i64,                     // quantization axis
    pub dtype: String,                 // output dtype ("float16")
}
```

Update `write_gpu_bundle()` to handle `AffineDequantize`:
- Write `quantized_data` to `weights/{name}.qdata`
- Write `scale` to `weights/{name}.scale`
- Write `zero_point` to `weights/{name}.zeros`
- Store metadata in `manifest.json`

### Task 2: Extend `MetalBundleProvider` to read affine tensors

**File:** `crates/ironmill-inference/src/metal/bundle.rs`

Update the `tensor()` method to recognize the `affine_dequantize` manifest
variant and return:
```rust
WeightTensor {
    data: Cow::Owned(packed_int4_bytes),
    shape: original_shape,
    dtype: ScalarType::UInt8,  // packed storage
    quant_info: QuantizationInfo::AffineDequantize {
        scale: scale_bytes,
        zero_point: zero_bytes,
        scale_dtype: ScalarType::Float16,
        zero_point_dtype: ScalarType::Float16,
        axis,
        bit_width: 4,
        group_size: Some(128),
    },
}
```

Once this is returned correctly, the existing `MetalWeights::load()` at
`crates/ironmill-inference/src/metal/weights.rs:301-358` already handles it:
it detects `bit_width == 4`, creates `AffineQuantizedWeight`, and
`run_pipeline()` dispatches through the INT4 dequant shader.

### Task 3: Extend `GpuCompileBuilder` with INT4 affine support

**File:** `crates/ironmill-compile/src/gpu/mod.rs`

Two approaches (choose one):

**Option A: Accept an arbitrary PassPipeline** (more general)
```rust
impl GpuCompileBuilder {
    pub fn with_pass_pipeline(mut self, pipeline: PassPipeline) -> Self {
        self.custom_pipeline = Some(pipeline);
        self
    }
}
```
In `build()`, if `custom_pipeline` is set, use it instead of the
PolarQuant-only default.

**Option B: Add an `int4_quantize` method** (simpler)
```rust
impl GpuCompileBuilder {
    pub fn int4_quantize(mut self, group_size: usize) -> Self {
        self.int4_group_size = Some(group_size);
        self
    }
}
```
In `build()`, if `int4_group_size` is set, run `AffineQuantizePass` instead
of `PolarQuantPass`.

**Recommendation:** Option A — it future-proofs for AWQ, GPTQ, etc. without
further changes to `GpuCompileBuilder`.

### Task 4: Wire CLI `--target gpu --quantize int4`

**File:** `crates/ironmill-cli/src/main.rs`

Currently `compile_for_gpu()` only accepts `polarquant-N`. Update it to:
- Accept `int4` in the GPU path
- Build a `PassPipeline` with `with_int4(128)`
- Pass it to `GpuCompileBuilder::with_pass_pipeline()`

```rust
"int4" => {
    let pipeline = PassPipeline::new().with_int4(128)?;
    builder = builder.with_pass_pipeline(pipeline);
}
```

### Task 5: Benchmark script

Create a benchmark script that:
1. Compiles Qwen3-8B with `--target gpu --quantize int4`
2. Compiles Qwen3-8B with `--target gpu` (FP16 baseline)
3. Runs Metal bench on both with perplexity
4. Reports: tok/s, PPL, GPU memory, file size

```bash
#!/bin/bash
MODEL_DIR="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"

# Compile FP16 baseline (no quantization)
ironmill compile "$MODEL_DIR" --target gpu --output qwen3-8b-fp16.ironml-gpu

# Compile INT4 weight-quantized
ironmill compile "$MODEL_DIR" --target gpu --quantize int4 --output qwen3-8b-int4.ironml-gpu

# Compare file sizes
echo "=== File Sizes ==="
du -sh qwen3-8b-fp16.ironml-gpu qwen3-8b-int4.ironml-gpu

# Benchmark both
ironmill-bench \
  --model qwen3-8b-fp16.ironml-gpu \
  --model qwen3-8b-int4.ironml-gpu \
  --backend metal \
  --iterations 1024 \
  --warmup 10 \
  --runs 1 \
  --perplexity \
  --perplexity-sequences 1 \
  --output markdown
```

**Expected results for Qwen3-8B:**

| Metric | FP16 | INT4 Weights | INT4 + TQ-INT4 KV |
|--------|------|-------------|-------------------|
| File size | ~15 GB | **~4 GB** | ~4 GB |
| GPU memory | ~30 GB | **~8 GB** | **~8 GB** |
| tok/s | 7.8 | ~7-8 | ~8-9 |
| Perplexity | 10.8 | ~11-12 | ~13-14 |

The key value proposition: **4× model size reduction + 4× GPU memory
reduction** with minimal quality loss.

## Dependency Graph

```
Task 1 (bundle schema)
  └── Task 2 (bundle reader) ← depends on 1
Task 3 (GpuCompileBuilder)   ← independent of 1-2
Task 4 (CLI)                 ← depends on 3
Task 5 (benchmark)           ← depends on 1-4
```

Tasks 1+3 can run in parallel. Task 5 requires all others.

## Files Changed

| Task | Files | Scope |
|------|-------|-------|
| 1 | `ironmill-core/src/gpu/bundle.rs`, `ironmill-compile/src/gpu/bundle.rs` | Bundle schema + writer |
| 2 | `ironmill-inference/src/metal/bundle.rs` | Bundle reader |
| 3 | `ironmill-compile/src/gpu/mod.rs` | Compile builder |
| 4 | `ironmill-cli/src/main.rs` | CLI wiring |
| 5 | New `scripts/bench-int4.sh` | Benchmark script |

## Verification

After all tasks, this command should work end-to-end:

```bash
# Compile with INT4 weight quantization
ironmill compile ~/.cache/huggingface/.../Qwen3-8B --target gpu --quantize int4 --output qwen3-int4.ironml-gpu

# Verify bundle is ~4GB (vs ~15GB for FP16)
du -sh qwen3-int4.ironml-gpu

# Run inference
ironmill-bench --model qwen3-int4.ironml-gpu --backend metal --perplexity --perplexity-sequences 5

# Expected: model loads, INT4 dequant shader dispatches, perplexity is reasonable (~11-12)
```
