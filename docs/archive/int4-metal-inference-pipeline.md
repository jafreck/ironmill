# INT4/INT8 Weight Quantization — End-to-End Metal Inference Pipeline

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

Additionally, GGUF models with pre-quantized weights (Q4_0, Q8_0) are
unnecessarily dequantized to FP16 before inference, discarding the
quantization and wasting memory.

## Design Decisions

### Output format

For GPU targets, quantized weights are written to `.ironml-gpu` bundles.
This is the right choice because:

- **ANE requires bundles** (`.ironml`) — compiled ANE programs + weights
  cannot be represented in standard formats.
- **SafeTensors/GGUF have no native affine quant schema** — SafeTensors
  supports only flat dtype tensors (could use U8 + metadata convention but
  no writer exists). GGUF has fixed block types (Q4_0 = group-32 symmetric)
  that don't match arbitrary group-128 affine with zero points.
- **The bundle writer already exists** and this spec completes it.

**Future scope:** Add a SafeTensors writer following the HuggingFace
GPTQ/AWQ convention (packed weights as U8, scales/zeros as FP16, config in
metadata) for interoperability with other frameworks.

### GGUF pre-quantized weights

GGUF models with native quantization (Q4_0, Q8_0, etc.) should be **used
directly** without dequant→requant. The existing `GgufProvider` dequantizes
everything to FP16 — this is lossy and wasteful. Instead, Q4_0/Q8_0 block
data should be returned as `AffineDequantize` with appropriate parameters.

## Current State

### What works

```
SafeTensors → SafeTensorsProvider → MetalWeights::load() → FP16 inference ✅
SafeTensors → GpuCompileBuilder → PolarQuant → .ironml-gpu → MetalBundleProvider → inference ✅
MIL Program → AffineQuantizePass → constexpr_affine_dequantize ops ✅ (but no export path)
MetalWeights::load() with AffineDequantize { bit_width=4 } → AffineQuantizedWeight → INT4 dispatch ✅
PassPipeline::new().with_int4(group_size) → AffineQuantizePass ✅ (used by ANE path)
CLI `build_pass_pipeline()` already handles `--quantize int4` ✅ (but only for ANE target)
INT4 dequant shader accepts group_size as runtime parameter ✅ (not hardcoded to 128)
```

### What's broken

```
SafeTensors → ??? → INT4 affine quantize → .ironml-gpu → MetalBundleProvider → INT4 inference ❌
GGUF (Q4_0) → dequant to FP16 → loses quantization → wastes memory ❌
GPU bundles → TurboQuant KV hardcoded off ❌
```

The gap has 4 parts:

1. **`GpuCompileBuilder` only supports PolarQuant** — hardcodes
   `PolarQuantPass` with no way to inject a custom `PassPipeline`.
   The SafeTensors/GGUF path directly calls `MilWeightProvider::from_weight_provider()`
   with PolarQuant parameters instead of running a pass pipeline.
   (`crates/ironmill-compile/src/gpu/mod.rs:39-136`)

2. **Bundle schema has no affine tensor format** — `write_gpu_bundle()`
   writes `AffineDequantize` tensors as `TensorManifest::Dense`, storing the
   raw packed INT4 bytes but labeling them with the original dtype (e.g.
   `float16`). This produces a **corrupt bundle** — the data is INT4 but the
   manifest claims FP16. Additionally, the global `QuantizationManifest` is
   hardcoded to `method: "polarquant"`.
   (`crates/ironmill-compile/src/gpu/bundle.rs:119-132`,
    `crates/ironmill-core/src/gpu/bundle.rs:31-51`)

3. **`MetalBundleProvider` can't read affine tensors** — only returns
   `QuantizationInfo::None` or `LutToDense`
   (`crates/ironmill-inference/src/metal/bundle.rs:77-127`)

4. **`GgufProvider` dequantizes pre-quantized weights** — Q4_0 and Q8_0
   tensors are converted to FP16 by `dequant_q4_0_to_fp16()` /
   `dequant_q8_0_to_fp16()`, discarding the existing quantization.
   `QuantizationInfo` is always `None`.
   (`crates/ironmill-compile/src/weights/gguf.rs:564-600, 827-843`)

## Implementation Plan

### Task 1: Extend GPU bundle schema for affine tensors

**Files:**
- `crates/ironmill-core/src/gpu/bundle.rs` — `TensorManifest` + `QuantizationManifest`
- `crates/ironmill-compile/src/gpu/bundle.rs` — `write_gpu_bundle()`

**1a. Add `AffineDequantize` variant to `TensorManifest`:**

```rust
// In TensorManifest enum:
#[serde(rename = "affine_dequantize")]
AffineDequantize {
    quantized_data_file: String,   // packed INT4 bytes
    scales_file: String,           // per-group FP16 scales
    zeros_file: String,            // per-group FP16 zero points
    shape: Vec<usize>,             // original tensor shape
    bit_width: u8,                 // 4
    group_size: usize,             // 128
    axis: i64,                     // quantization axis
    dtype: String,                 // output dtype ("float16")
}
```

**1b. Make `QuantizationManifest` method dynamic:**

Currently `write_gpu_bundle()` hardcodes `method: "polarquant"` (line 140).
Change this to accept the method as a parameter or infer it from the tensor
types. For mixed bundles (some PolarQuant, some affine), use `"mixed"` or
make the method per-tensor rather than global.

**1c. Update `write_gpu_bundle()` to handle `AffineDequantize`:**

Replace the current fallback-to-Dense arm (lines 119-131) with proper
serialization:
- Write `quantized_data` to `weights/{name}.qdata`
- Write `scale` to `weights/{name}.scale`
- Write `zero_point` to `weights/{name}.zeros`
- Store metadata in `manifest.json` using the new `AffineDequantize` variant

Also update the bundle layout doc comment at the top of the file to include
the new extensions (`.qdata`, `.scale`, `.zeros`).

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

### Task 3: Unify GPU compile paths through MIL IR + PassPipeline

**File:** `crates/ironmill-compile/src/gpu/mod.rs`

Currently `GpuCompileBuilder::build()` has two divergent code paths:

- **ONNX**: `import_onnx()` → MIL `Program` → hardcoded `PolarQuantPass` →
  `MilWeightProvider::new()` (lines 89–102)
- **SafeTensors/GGUF**: `WeightProvider` → `MilWeightProvider::from_weight_provider()`
  which bypasses MIL IR entirely and applies PolarQuant inline (lines 104–129)

This means SafeTensors and GGUF cannot use `PassPipeline` at all — no INT4
affine, no INT8, no AWQ, nothing beyond PolarQuant.

**Refactor: unify all three formats through a single path:**

```
ONNX        → import_onnx()             → (Program, Config)
SafeTensors → SafeTensorsProvider        → weights_to_program() → (Program, Config)
GGUF        → GgufProvider (→ FP16)      → weights_to_program() → (Program, Config)
                                              ↓
                                        PassPipeline
                                              ↓
                                        MilWeightProvider::new()
```

The template system (`crates/ironmill-compile/src/templates/mod.rs:27-46`)
already provides `weights_to_program()` which converts any `WeightProvider`
into a MIL `Program`. This is the same path the ANE compile uses. The GPU
builder just needs to call it instead of `from_weight_provider()`.

**Implementation:**

```rust
pub struct GpuCompileBuilder {
    input: PathBuf,
    pipeline: Option<PassPipeline>,
}

impl GpuCompileBuilder {
    pub fn with_pass_pipeline(mut self, pipeline: PassPipeline) -> Self {
        self.pipeline = Some(pipeline);
        self
    }

    pub fn build(self) -> Result<MilWeightProvider, CompileError> {
        let (mut program, config) = match detect_format(&self.input) {
            InputFormat::Onnx => import_onnx(&self.input)?,
            InputFormat::SafeTensors | InputFormat::Gguf => {
                let provider = load_provider(&self.input)?;
                weights_to_program(provider.as_ref())?
            }
            InputFormat::Unsupported(ext) => return Err(/* ... */),
        };

        let pipeline = self.pipeline
            .unwrap_or_else(|| PassPipeline::new());
        pipeline.run(&mut program)?;

        MilWeightProvider::new(&program, config)
    }
}
```

When no pipeline is set, the program passes through unquantized (FP16).
The caller decides quantization — this is a clean separation of concerns.

**GGUF note:** When the user explicitly requests quantization (e.g.
`--quantize int4`), GGUF weights are dequantized to FP16 by `GgufProvider`
before entering the template system, then re-quantized by the pass pipeline.
This double-quantization is lossy when the source GGUF is already quantized
(Q4_0, Q8_0). Task 6 addresses this by teaching `GgufProvider` to pass
through pre-quantized weights directly — for the compile path, FP16/FP32
GGUF sources are the expected input.

**Support matrix after this change:**

| Format | INT4 Affine | INT8 | PolarQuant | AWQ | D2Quant |
|--------|-------------|------|------------|-----|---------|
| ONNX | ✅ | ✅ | ✅ | ✅ | ✅ |
| SafeTensors | ✅ | ✅ | ✅ | ✅ | ✅ |
| GGUF (FP16/FP32) | ✅ | ✅ | ✅ | ✅ | ✅ |
| GGUF (Q4_0/Q8_0) | — (already quantized, use directly via Task 6) | | | | |

### Task 4: Wire CLI `--target gpu --quantize int4`

**File:** `crates/ironmill-cli/src/main.rs`

Currently `compile_for_gpu()` (lines 473-500) is a separate code path from
`build_pass_pipeline()`. It only calls `parse_polarquant_bits()`, which
rejects anything that isn't `polarquant-N`. Meanwhile,
`build_pass_pipeline()` (lines 502-628) already handles `int4`, `int8`,
`awq`, `d2quant`, etc. — but is only used by the ANE compile path.

The fix: **make `compile_for_gpu()` use `build_pass_pipeline()`** (or a
GPU-specific subset) and pass the resulting pipeline to
`GpuCompileBuilder::with_pass_pipeline()` from Task 3.

```rust
fn compile_for_gpu(input_path: &Path, opts: &CompileOpts) -> Result<()> {
    let mut builder = GpuCompileBuilder::new(input_path);

    let pipeline = build_pass_pipeline(opts)?;
    builder = builder.with_pass_pipeline(pipeline);

    let provider = builder.build().context("GPU compilation failed")?;
    // ...
}
```

This automatically gives the GPU path access to `int4`, `int8`, `awq`,
`d2quant`, and any future quantization methods added to
`build_pass_pipeline()` — no per-method wiring needed.

### Task 5: Enable TurboQuant KV cache for GPU bundle inference

**File:** `crates/ironmill-bench/src/main.rs`

The GPU bundle path in `ironmill-bench` (lines 481-484) hardcodes
`enable_turboquant: false`. Meanwhile, the direct SafeTensors Metal path
(lines 428-451) already runs multiple `MetalConfig` variants per model
(`fp16`, `tq-int8`, `tq-int4`).

Apply the same multi-config pattern to the GPU bundle path: for each loaded
bundle, run inference with `enable_turboquant: false` (baseline) and
`enable_turboquant: true` (TQ KV). TurboQuant KV cache is a **runtime**
option — independent of weight format — so this requires no bundle schema
changes.

Also update the config label logic (line 477) which currently hardcodes
`PQ-INT{n_bits}` — it should reflect the actual weight method from the
manifest (e.g. `Affine-INT4` vs `PQ-INT4`).

### Task 6: GGUF pre-quantized weight passthrough

**File:** `crates/ironmill-compile/src/weights/gguf.rs`

Currently `GgufProvider::tensor()` dequantizes all non-FP16 tensors to FP16
(lines 564-600), discarding existing quantization. For Q4_0 and Q8_0, the
quantized data should be passed through directly.

**Q4_0 mapping to `AffineDequantize`:**

Q4_0 blocks (lines 910-956) are 32 values/block, 18 bytes/block:
- 2 bytes: FP16 scale (`d`)
- 16 bytes: 32 × 4-bit values packed low-nibble-first
- Formula: `value = d * (nibble - 8)` (symmetric, implicit zero=8)

This maps to `AffineDequantize`:
- `scale` = per-block FP16 `d` values
- `zero_point` = constant 8 (broadcast as FP16)
- `bit_width` = 4
- `group_size` = 32

The INT4 dequant shader already accepts `group_size` as a runtime buffer
parameter (`int4_dequant.metal:16-46`), so group-32 works without shader
changes.

**Q8_0 mapping:**

Q8_0 blocks (lines 958-997) are 32 values/block, 34 bytes/block:
- 2 bytes: FP16 scale (`d`)
- 32 bytes: 32 × INT8 values
- Formula: `value = d * int8`

This maps to `AffineDequantize`:
- `scale` = per-block FP16 `d` values
- `zero_point` = 0
- `bit_width` = 8
- `group_size` = 32

**Implementation:**

Update `GgufProvider::tensor()` to check the GGUF tensor type before
dequantizing:

```rust
match tensor_type {
    GgmlType::Q4_0 => {
        let (packed_data, scales) = repack_q4_0_blocks(raw_data);
        return WeightTensor {
            data: Cow::Owned(packed_data),
            shape: original_shape,
            dtype: ScalarType::UInt8,
            quant_info: QuantizationInfo::AffineDequantize {
                scale: scales,
                zero_point: broadcast_fp16_zero(8.0, num_groups),
                scale_dtype: ScalarType::Float16,
                zero_point_dtype: ScalarType::Float16,
                axis: 1,  // along output channels
                bit_width: 4,
                group_size: Some(32),
            },
        };
    }
    GgmlType::Q8_0 => { /* similar */ }
    _ => { /* fall through to existing dequant-to-FP16 path */ }
}
```

**Note:** Q4_0 blocks interleave scale + packed data. The `repack` step
must separate scales from packed nibbles into contiguous buffers to match
the layout expected by `AffineQuantizedWeight` and the Metal dequant shader.
Other GGUF types (Q4_1, Q2K–Q6K, IQ*) can remain on the dequant-to-FP16
path for now — they have more complex block structures.

### Task 7: Benchmark script

Create a benchmark script that:
1. Compiles Qwen3-8B with `--target gpu --quantize int4`
2. Compiles Qwen3-8B with `--target gpu` (FP16 baseline)
3. Runs Metal bench on both, each with FP16 KV and TQ-INT4 KV
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

# Benchmark both (bench runs FP16-KV and TQ-KV variants automatically)
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
reduction** with minimal quality loss. Adding TQ KV cache on top gives a
further throughput boost at a small perplexity cost.

## Dependency Graph

```
Task 1 (bundle schema)
  └── Task 2 (bundle reader) ← depends on 1
Task 3 (GpuCompileBuilder)   ← independent of 1-2
Task 4 (CLI)                 ← depends on 3
Task 5 (bench TQ for bundles)← independent of 1-4, 6
Task 6 (GGUF passthrough)    ← independent of 1-5
Task 7 (benchmark script)    ← depends on 1-5 (Task 6 is optional)
```

Tasks 1, 3, 5, and 6 can all run in parallel. Task 7 requires 1-5.

## Files Changed

| Task | Files | Scope |
|------|-------|-------|
| 1 | `ironmill-core/src/gpu/bundle.rs`, `ironmill-compile/src/gpu/bundle.rs` | Bundle schema + manifest + writer |
| 2 | `ironmill-inference/src/metal/bundle.rs` | Bundle reader |
| 3 | `ironmill-compile/src/gpu/mod.rs` | Compile builder unification |
| 4 | `ironmill-cli/src/main.rs` | CLI wiring (`compile_for_gpu` → `build_pass_pipeline`) |
| 5 | `ironmill-bench/src/main.rs` | Enable TQ KV for GPU bundle inference |
| 6 | `ironmill-compile/src/weights/gguf.rs` | Q4_0/Q8_0 direct passthrough |
| 7 | New `scripts/bench-int4.sh` | Benchmark script |

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
