# Specification: Lazy Tensor Storage for Large Model Compilation

**Status:** Implemented  
**Crates affected:** `mil-rs`, `ironmill-compile`, `ironmill-core`, `ironmill-inference`

## 1. Problem

ironmill-compile cannot compile models whose weights exceed available system
RAM. A 70B-parameter model in FP16 requires ~140 GB of weight data. The current
pipeline materializes all weight bytes into the MIL IR `Program` as owned
`Vec<u8>` allocations, then copies them again when extracting to a
`MilWeightProvider`. Peak memory reaches roughly **2× model size**.

The root cause is `Value::Tensor { data: Vec<u8>, ... }` — every weight tensor's
raw bytes are stored inline in the IR with no indirection, lazy loading, or
eviction capability.

### Current memory flow

```
SafeTensors/GGUF (mmap)        ← zero-copy ✅
  → WeightProvider.tensor()    ← zero-copy for FP16 ✅, allocates for BF16/dequant ⚠️
    → emit_weight_const()      ← .into_owned() → Vec<u8> in MIL IR ❌ COPY 1
      → MIL Program (all ops)  ← full model in RAM ❌
        → PassPipeline.run()   ← quantization passes process ALL ops ❌
          → MilWeightProvider   ← extracts all tensors into HashMap<String, Vec<u8>> ❌ COPY 2
            → write_gpu_bundle  ← writes per-tensor to disk ✅
```

## 2. Goals

1. Enable GPU bundle compilation of models that exceed available RAM (e.g. 70B+
   parameter models on a 64 GB machine).
2. Reduce peak memory for models that do fit in RAM by eliminating redundant
   copies.
3. Preserve full correctness of all existing compilation paths (ANE, CoreML,
   ONNX round-trip).
4. Require no changes to the `Pass` trait or existing graph-structural passes.
5. Keep the change backward-compatible: code that constructs `Value::Tensor`
   with inline data continues to work without modification.

## 3. Non-Goals

- Streaming ONNX import (ONNX initializers are already loaded eagerly; this is
  an orthogonal concern).
- Distributed/multi-machine compilation.
- Memory-mapping the output bundle (the GPU bundle writer already writes
  per-tensor files and is not a bottleneck).
- Changes to inference runtime *behavior* (`ironmill-inference`). Compile-time
  code in that crate (e.g., `mil_emitter.rs` construction sites) will receive
  mechanical `TensorData` type updates in Phase B, but no runtime logic changes.

## 4. Design

### 4.1 Lazy Tensor Data — `TensorData` enum

Replace `Value::Tensor`'s inline `data: Vec<u8>` with an enum that supports
both inline and externally-backed storage:

```rust
// crates/mil-rs/src/ir/types.rs

/// Storage backing for tensor data in the IR.
///
/// Small tensors (scalars, norms, RoPE tables) remain inline. Large weight
/// tensors can be backed by an external provider, deferring byte-level
/// access until a pass or writer actually needs the data.
///
/// # Clone warning
/// `Clone` is derived for convenience with small/external tensors, but
/// cloning an `Inline` variant deep-copies the byte buffer. Avoid
/// cloning large inline tensors — use `std::mem::take` or destructive
/// extraction instead.
#[derive(Debug, Clone)]
pub enum TensorData {
    /// Tensor bytes are stored inline (owned). This is the default for
    /// small tensors, newly-constructed constants, and any tensor whose
    /// data has been materialized by a pass.
    Inline(Vec<u8>),

    /// Tensor data lives in an external source (e.g., mmap'd SafeTensors
    /// file). The `provider_key` is the canonical weight name used to
    /// retrieve it from a [`WeightProvider`].
    ///
    /// Operations that need the raw bytes must call
    /// `TensorData::materialize_with()` or `TensorData::resolve_with()`
    /// with access to the provider.
    External {
        /// Canonical weight name (e.g., "model.layers.0.self_attn.q_proj.weight").
        provider_key: String,
        /// Byte length of the tensor data. Available without materialization
        /// for size accounting (e.g., ANE weight-limit enforcement).
        byte_len: usize,
    },
}
```

The `Value::Tensor` variant becomes:

```rust
pub enum Value {
    // ... existing variants unchanged ...

    /// Raw tensor data (for weights/constants).
    Tensor {
        /// Tensor data — inline or externally backed.
        data: TensorData,
        /// Dimensions of the tensor.
        shape: Vec<usize>,
        /// Element data type.
        dtype: ScalarType,
    },
}
```

### 4.2 `TensorData` API

```rust
impl TensorData {
    /// Create inline tensor data from owned bytes.
    pub fn inline(data: Vec<u8>) -> Self {
        TensorData::Inline(data)
    }

    /// Create an external tensor reference.
    pub fn external(provider_key: String, byte_len: usize) -> Self {
        TensorData::External { provider_key, byte_len }
    }

    /// Returns the byte length of the tensor data without materializing.
    pub fn byte_len(&self) -> usize {
        match self {
            TensorData::Inline(data) => data.len(),
            TensorData::External { byte_len, .. } => *byte_len,
        }
    }

    /// Returns a reference to the inline bytes, or `None` if external.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            TensorData::Inline(data) => Some(data),
            TensorData::External { .. } => None,
        }
    }

    /// Returns a mutable reference to the inline bytes, or `None` if external.
    pub fn as_bytes_mut(&mut self) -> Option<&mut Vec<u8>> {
        match self {
            TensorData::Inline(data) => Some(data),
            TensorData::External { .. } => None,
        }
    }

    /// Returns `true` if the data is stored inline.
    pub fn is_inline(&self) -> bool {
        matches!(self, TensorData::Inline(_))
    }

    /// Returns `true` if the data is externally backed.
    pub fn is_external(&self) -> bool {
        matches!(self, TensorData::External { .. })
    }

    /// Consume this `TensorData` and return the inline bytes.
    ///
    /// # Panics
    /// Panics if the data is `External`. Callers must resolve first.
    pub fn into_bytes(self) -> Vec<u8> {
        match self {
            TensorData::Inline(data) => data,
            TensorData::External { provider_key, .. } => {
                panic!(
                    "cannot into_bytes() on External tensor '{provider_key}'; \
                     call resolve_with() or materialize_with() first"
                )
            }
        }
    }

    /// Consume this `TensorData` and return the inline bytes, resolving
    /// external data via the provided closure if necessary.
    pub fn resolve_with<F>(self, loader: F) -> Result<Vec<u8>, MilError>
    where
        F: FnOnce(&str) -> Result<Vec<u8>, MilError>,
    {
        match self {
            TensorData::Inline(data) => Ok(data),
            TensorData::External { provider_key, .. } => loader(&provider_key),
        }
    }

    /// Resolve external data in-place using the provided closure.
    /// No-op if already inline.
    pub fn materialize_with<F>(&mut self, loader: F) -> Result<(), MilError>
    where
        F: FnOnce(&str) -> Result<Vec<u8>, MilError>,
    {
        if let TensorData::External { ref provider_key, .. } = *self {
            let key = provider_key.clone();
            let data = loader(&key)?;
            *self = TensorData::Inline(data);
        }
        Ok(())
    }
}

impl From<Vec<u8>> for TensorData {
    fn from(data: Vec<u8>) -> Self {
        TensorData::Inline(data)
    }
}
```

### 4.3 Convenience Constructor on `Value`

To ease migration, add a helper that mirrors the old construction pattern:

```rust
impl Value {
    /// Convenience constructor for inline tensor values.
    ///
    /// Equivalent to the previous `Value::Tensor { data, shape, dtype }`
    /// construction pattern.
    pub fn tensor(data: Vec<u8>, shape: Vec<usize>, dtype: ScalarType) -> Self {
        Value::Tensor {
            data: TensorData::Inline(data),
            shape,
            dtype,
        }
    }
}
```

### 4.4 Template Emission — External References

`emit_weight_const()` currently calls `tensor.data.into_owned()` to copy mmap'd
bytes into the IR. With lazy storage, it emits an `External` reference for large
tensors:

```rust
// crates/ironmill-compile/src/templates/shared.rs

/// Tensors smaller than this are stored inline to avoid indirection overhead.
/// Chosen to match the system page size — tensors this small don't benefit
/// from lazy loading because the mmap page fault cost dominates. This also
/// covers all scalar constants, norm weights, and bias vectors.
const INLINE_THRESHOLD_BYTES: usize = 4096;

pub(super) fn emit_weight_const(
    block: &mut Block,
    provider: &dyn WeightProvider,
    weight_name: &str,
    const_name: &str,
    _warnings: &mut Vec<String>,
) -> Result<(), MilError> {
    match provider.tensor(weight_name) {
        Ok(tensor) => {
            let data = if tensor.data.len() <= INLINE_THRESHOLD_BYTES {
                // Small tensors (norms, biases): inline to avoid indirection.
                TensorData::inline(tensor.data.into_owned())
            } else {
                // Large tensors (projections, embeddings): defer loading.
                TensorData::external(weight_name.to_string(), tensor.data.len())
            };
            let op = Operation::new("const", const_name)
                .with_attr(
                    "val",
                    Value::Tensor {
                        data,
                        shape: tensor.shape.clone(),
                        dtype: tensor.dtype,
                    },
                )
                .with_attr("onnx_name", Value::String(weight_name.to_string()))
                .with_output(const_name);
            block.add_op(op);
            Ok(())
        }
        Err(e) => Err(MilError::Validation(format!(
            "missing required weight '{weight_name}': {e}"
        ))),
    }
}
```

Tensors constructed from computed data (RoPE tables, scalar constants) continue
to use `TensorData::inline()` directly.

### 4.5 `Program` Weight Provider Attachment

The `Pass` trait signature does NOT change. Instead, the `Program` gains an
optional provider attachment for lazy resolution:

```rust
// crates/mil-rs/src/ir/program.rs

use std::sync::Arc;

pub struct Program {
    pub version: String,
    pub functions: IndexMap<String, Function>,
    pub attributes: HashMap<String, String>,

    /// Optional weight provider for resolving `TensorData::External` references.
    /// Set after template emission, consumed by quantization passes.
    /// Accessible within the crate so passes can clone the `Arc` to avoid
    /// borrow conflicts when iterating `functions` (see §5.2).
    pub(crate) weight_provider: Option<Arc<dyn WeightProvider + Send + Sync>>,
}

impl Program {
    /// Attach a weight provider for lazy tensor resolution.
    pub fn set_weight_provider(&mut self, provider: Arc<dyn WeightProvider + Send + Sync>) {
        self.weight_provider = Some(provider);
    }

    /// Resolve an external tensor key to its byte data.
    ///
    /// Returns an error if no provider is attached or the key is not found.
    pub fn resolve_tensor(&self, key: &str) -> Result<Vec<u8>, MilError> {
        let provider = self.weight_provider.as_ref().ok_or_else(|| {
            MilError::Validation(format!(
                "no weight provider attached; cannot resolve external tensor '{key}'"
            ))
        })?;
        let tensor = provider.tensor(key)?;
        Ok(tensor.data.into_owned())
    }

    /// Returns `true` if a weight provider is attached.
    pub fn has_weight_provider(&self) -> bool {
        self.weight_provider.is_some()
    }

    /// Materialize all `TensorData::External` references in the program,
    /// converting them to `TensorData::Inline`.
    ///
    /// Required before serialization paths (CoreML proto, mlpackage writer)
    /// that need all tensor data inline. Also required before ANE compilation
    /// which reads tensor bytes for blob emission.
    pub fn materialize_all(&mut self) -> Result<(), MilError> {
        let provider = self.weight_provider.clone().ok_or_else(|| {
            MilError::Validation("no weight provider for materialization".into())
        })?;

        for function in self.functions.values_mut() {
            Self::materialize_block(&mut function.body, &*provider)?;
        }
        Ok(())
    }

    fn materialize_block(
        block: &mut Block,
        provider: &dyn WeightProvider,
    ) -> Result<(), MilError> {
        for op in &mut block.operations {
            for val in op.inputs.values_mut().chain(op.attributes.values_mut()) {
                if let Value::Tensor { ref mut data, .. } = val {
                    data.materialize_with(|key| {
                        let tensor = provider.tensor(key)?;
                        Ok(tensor.data.into_owned())
                    })?;
                }
            }
            // Recurse into nested blocks (e.g., while_loop, cond sub-blocks).
            for sub_block in &mut op.blocks {
                Self::materialize_block(sub_block, provider)?;
            }
        }
        Ok(())
    }
}
```

### 4.6 `WeightProvider` — `Send + Sync` Bound

For the `Program` to hold a provider reference via `Arc`, the trait needs
`Send + Sync` bounds:

```rust
// crates/mil-rs/src/weights.rs

pub trait WeightProvider: Send + Sync {
    fn tensor(&self, name: &str) -> Result<WeightTensor<'_>, MilError>;
    fn tensor_names(&self) -> Vec<&str>;
    fn config(&self) -> &ModelConfig;
    fn has_tensor(&self, name: &str) -> bool { ... }
}
```

All six existing implementors (four production, two test-only) store only owned
data (`HashMap`, `Vec`, `Mmap` from `memmap2`) which are `Send + Sync`. No
implementation changes required.

**Implementors:**
- `SafeTensorsProvider` (`ironmill-compile/src/weights/safetensors.rs:150`) — `Mmap` + `HashMap` ✅
- `GgufProvider` (`ironmill-compile/src/weights/gguf.rs:565`) — `Mmap` + `HashMap` ✅
- `MilWeightProvider` (`ironmill-compile/src/weights/mil_provider.rs:433`) — `HashMap<String, ExtractedTensor>` ✅
- `MetalBundleProvider` (`ironmill-inference/src/metal/bundle.rs:78`) — owned data ✅
- `StubProvider` (test-only, `ironmill-compile/src/templates/shared.rs:1032`) ✅
- `MockProvider` (test-only, `ironmill-compile/src/gpu/bundle.rs:400`) ✅

### 4.7 Pass Trait — No Signature Change

The `Pass` trait remains unchanged:

```rust
pub trait Pass {
    fn name(&self) -> &str;
    fn run(&self, program: &mut Program) -> Result<()>;
}
```

Passes access the provider indirectly through `program.resolve_tensor()`.

## 5. Pass Compatibility Analysis

### 5.1 Graph-Structural Passes — Zero Changes Required

These passes inspect only op types, names, and `Value::Reference` connections.
They never access `TensorData` bytes and require **no code changes**:

| Pass | What it inspects |
|------|-----------------|
| `AttentionFusionPass` | op types, reference chains |
| `GqaFusionPass` | op types, reference chains |
| `ConvBatchNormFusionPass` | op types, consumer chains |
| `ConvReluFusionPass` | op types, consumer chains |
| `LinearReluFusionPass` | op types, consumer chains |
| `LayerNormLinearFusionPass` | op types, consumer chains |
| `GeluLinearFusionPass` | op types, consumer chains |
| `ResidualAddFusionPass` | op types, reference tracing |
| `DeadCodeEliminationPass` | output names, reference names |
| `IdentityEliminationPass` | op types, references |
| `LayoutOptimizationPass` | transpose permutation attributes |
| `ConstantFoldPass` | scalar `Int`/`Float` constants only |
| `TypeRepropagationPass` | type annotations |
| `ShapeMaterializePass` | shape attributes |
| `AutoregressiveShapeMaterializePass` | shape attributes |
| `AwqScaleFusionPass` | currently a no-op |

### 5.2 Quantization Passes — Materialize Before Processing

These passes access tensor byte data and must materialize before processing.
Each already processes tensors one-at-a-time via `for op in &mut operations`, so
peak memory is bounded to **one unquantized tensor at a time** plus the
(smaller) quantized replacement:

| Pass | Materialization pattern |
|------|------------------------|
| `AffineQuantizePass` | Remove `val` → materialize → quantize → replace with quantized data |
| `PolarQuantPass` | Extract eligible tensor → materialize → normalize/rotate/quantize → rewrite op |
| `Int8QuantizePass` | Same as Affine |
| `Fp16QuantizePass` | Same as Affine (trivial conversion) |
| `PalettizePass` | Same as Affine (k-means clustering) |
| `GroupedPalettizePass` | Same as Affine |
| `SpinQuantPass` | Same as Polar (per-tensor rotation + quantize) |
| `D2QuantPass` | Same as Affine (per-group) |
| `QuipSharpPass` | Same as Polar (E8 lattice quantize) |
| `AwqQuantizePass` | Collects eligible ops → materialize each → scale + quantize |
| `GptqQuantizePass` | Looks up Hessian → materialize tensor → quantize per-column |
| `DacPass` | Same as Affine |

**Common adaptation pattern:**

Passes that iterate `program.functions.values_mut()` cannot simultaneously call
`program.resolve_tensor()` due to the borrow checker. The solution is to clone
the provider `Arc` before iterating:

```rust
fn run(&self, program: &mut Program) -> Result<()> {
    let provider = program.weight_provider.clone(); // Arc clone (cheap)

    let resolve = |key: &str| -> Result<Vec<u8>> {
        let p = provider.as_ref().ok_or_else(|| ...)?;
        let tensor = p.tensor(key)?;
        Ok(tensor.data.into_owned())
    };

    for function in program.functions.values_mut() {
        for op in &mut function.body.operations {
            if let Some(Value::Tensor { ref mut data, .. }) =
                op.attributes.get_mut("val")
            {
                data.materialize_with(|key| resolve(key))?;
            }
            // Then proceed with existing removal + quantization logic unchanged.
            let val = op.attributes.remove("val").ok_or_else(|| ...)?;
        }
    }
    Ok(())
}
```

### 5.3 Evaluation Pass — Materialize Both Programs

`QuantizationEvaluator::evaluate()` takes `&Program` (immutable) for both
original and quantized programs. It reads tensor data from const ops. This path
requires both programs to have been fully materialized before calling evaluate.
Add a precondition check:

```rust
// In QuantizationEvaluator::evaluate():
// Both programs must have all tensors inline before comparison.
// Callers should call program.materialize_all() first if using lazy tensors.
```

## 6. Usage Site Census and Migration

### 6.1 `mil-rs` Crate

**Definition change:**

| File | Lines | Change |
|------|-------|--------|
| `ir/types.rs` | 29-36 | `data: Vec<u8>` → `data: TensorData` |

**Proto conversion — read data for serialization:**

| File | Lines | Current | Change |
|------|-------|---------|--------|
| `convert/ir_to_proto.rs` | 1029, 1036, 1086 | Pattern match `Value::Tensor { data, .. }` | Use `data.as_bytes().expect("tensor not materialized")` |
| `convert/ir_to_proto.rs` | 1342 | `convert_tensor_data` reads bytes | Same — callers must materialize first |
| `convert/ir_to_proto.rs` | 1555 | `convert_value_to_proto` | Same |

**Proto import — construct from proto bytes:**

| File | Lines | Current | Change |
|------|-------|---------|--------|
| `convert/proto_to_ir.rs` | 287, 345 | Construct `Value::Tensor { data: bytes, .. }` | Use `Value::tensor(bytes, shape, dtype)` or `TensorData::inline(bytes)` |

**Quantization passes:** See §5.2 — all need materialize-before-access pattern.

**Evaluation pass:**

| File | Lines | Current | Change |
|------|-------|---------|--------|
| `ir/passes/eval_quantize.rs` | 107-160 | Reads `data` from both programs | Add precondition: both programs materialized |

### 6.2 `ironmill-compile` Crate

**Templates:**

| File | Lines | Current | Change |
|------|-------|---------|--------|
| `templates/shared.rs` | 43-44 | `tensor.data.into_owned()` | `TensorData::external(name, len)` for large; `TensorData::inline()` for small |
| `templates/shared.rs` | 214, 226 | Construct RoPE tables inline | `TensorData::inline(bytes)` |

**MilWeightProvider:**

| File | Lines | Current | Change |
|------|-------|---------|--------|
| `weights/mil_provider.rs` | 36 | `new(&Program, ...)` | `new(&mut Program, ...)` — destructive extraction |
| `weights/mil_provider.rs` | 61, 68 | `data.clone()` from const ops | `std::mem::take(data)` for inline, resolve for external |
| `weights/mil_provider.rs` | 170, 188, 241, 478, 488, 502 | Read/clone tensor data | Handle `TensorData::External` variant |
| `weights/mil_provider.rs` | 278-324 | `from_weight_provider()` clones | Use `into_owned()` / take |
| `weights/mil_provider.rs` | 375-402, 421-426 | Construct new tensors | `TensorData::inline()` |

**GPU compile builder:**

| File | Lines | Current | Change |
|------|-------|---------|--------|
| `gpu/mod.rs` | 92-121 | Build program + extract provider | Wrap provider in `Arc`, attach to program |

**ANE split:**

| File | Lines | Current | Change |
|------|-------|---------|--------|
| `ane/split.rs` | 952-958 | `data.len()` for size | `data.byte_len()` — works for both variants |
| `ane/split.rs` | 1257-1265 | Construct + clone tensors | `TensorData::inline()` |

**ANE bundle / decode compile (require full materialization):**

| File | Lines | Access | Change |
|------|-------|--------|--------|
| `ane/bundle.rs` | 342-358 | READ_DATA + WRITE_DATA | Call `program.materialize_all()` before ANE compilation |
| `ane/decode_compile.rs` | 38-40, 147-148, 542-545, 582-595, 633-649, 701-717 | READ_DATA | Same — materialize before entry |

**ANE passes (all require materialized tensors):**

| File | Lines | Access | Change |
|------|-------|--------|--------|
| `ane/passes/ane_arg_promotion.rs` | 256-335 | READ_DATA + CONSTRUCT | Materialize, use `TensorData::inline()` for new |
| `ane/passes/ane_layout.rs` | 150-160 | CONSTRUCT | `TensorData::inline()` |
| `ane/passes/ane_matmul_to_conv.rs` | 63-64, 147-160, 217-225 | CLONE + READ + CONSTRUCT | Materialize, `TensorData::inline()` |
| `ane/passes/codebook.rs` | 395-443 | READ_DATA + CONSTRUCT | Materialize, `TensorData::inline()` |
| `ane/passes/mixed_precision.rs` | 183-191, 236-303, 331-379 | READ + META + CONSTRUCT | Materialize, `TensorData::inline()` |
| `ane/passes/op_split.rs` | 153-155 | READ_DATA | Materialize |
| `ane/packing.rs` | 93-123 | CONSTRUCT (small int32 constants) | `TensorData::inline()` |
| `ane/validate.rs` | 413-447 | READ_META only | Use `data.byte_len()` — no materialization |

**Pipeline conversion:**

| File | Lines | Current | Change |
|------|-------|---------|--------|
| `convert/pipeline.rs` | 189-243 | Holds all stage programs | No structural change; each stage materializes independently |

### 6.3 `ironmill-core` Crate

| File | Lines | Access | Change |
|------|-------|--------|--------|
| `ane/mil_text.rs` | 28-34, 710-713, 1006-1011 | CONSTRUCT | `TensorData::inline()` |
| `ane/mil_text.rs` | 272-313 | CLONE_DATA + READ_META | Materialize; use `data.as_bytes().unwrap()` |
| `ane/mil_text.rs` | 375-385, 547-550, 608-687 | READ_DATA | Materialize; `data.as_bytes().unwrap()` |
| `ane/mil_text.rs` | 474-475 | READ_META | `data.byte_len()` — no materialization |

### 6.4 `ironmill-inference` Crate

| File | Lines | Access | Change |
|------|-------|--------|--------|
| `ane/turboquant/mil_emitter.rs` | 28-34, 1065-1092 | CONSTRUCT | `TensorData::inline()` |

### 6.5 `ironmill-cli` Crate

No `Value::Tensor` usage sites found. No changes required.

## 7. Compilation Path Analysis

### 7.1 GPU Bundle Path — SafeTensors/GGUF (streaming, Phase A)

```
SafeTensors/GGUF (mmap) ─── WeightProvider ───┐
                                               ▼
         StreamingGpuBundleBuilder.build()
             for each tensor:
                 load from mmap  ← zero-copy borrow
                 quantize        ← standalone math functions
                 write to disk   ← per-tensor .bin/.lut/.qdata files
                 drop            ← memory freed immediately
             write manifest.json
```

**Peak memory: ~1 GB** (one unquantized tensor + quantized output).

### 7.2 GPU Bundle Path — ONNX (also streaming via initializer extraction)

```
ONNX protobuf ──→ parse initializer metadata (names, shapes, dtypes)
                        │
                        ▼
         StreamingGpuBundleBuilder  ← OnnxInitializerProvider
             for each initializer:
                 load bytes (raw_data / typed fields / external sidecar)
                 quantize
                 write to disk
                 drop
```

ONNX separates initializers (weight tensors) from nodes (computation ops).
The streaming builder only needs initializers — it ignores the computation
graph entirely. For ONNX models with external data (`data_location == 1`),
weights are already in separate files with `(offset, length)` descriptors,
making per-tensor access trivial.

For ONNX models with inline weights (all data in the protobuf), the
`OnnxInitializerProvider` must parse the protobuf to locate each initializer's
bytes. The protobuf is loaded once but individual tensor data can be extracted
and dropped one at a time.

**Peak memory:** ~1 GB + protobuf overhead for inline ONNX. ~1 GB for
external-data ONNX (identical to SafeTensors).

### 7.3 GPU Bundle Path — SafeTensors/GGUF via IR (Phase B, with spill-after-quantize)

```
SafeTensors (mmap) ─── Arc<WeightProvider> ───┐
                                               ▼
        weights_to_program()    ← emits External refs (no data copy)
                │
                ▼
        program.set_weight_provider(arc)
                │
                ▼
        PassPipeline.run()      ← each quant pass materializes one tensor at a time
                │                  quantized data replaces original
                │                  spill_inline_tensors() writes to temp dir after each pass
                │                  quantized Inline data → External (file-backed)
                ▼
        MilWeightProvider::new(&mut program)  ← resolves from spill dir + provider
                │
                ▼
        write_gpu_bundle()      ← writes per-tensor files to disk
```

**Peak memory: ~1 GB** (one unquantized tensor + quantized output). Quantized
tensors are spilled to a temp directory between passes via
`Program::spill_inline_tensors()`, keeping only metadata in the IR. The
`make_resolver()` utility checks the spill directory first, then falls back to
the weight provider. This achieves streaming-like memory efficiency while
retaining full pass pipeline capabilities (AWQ fusion, rotation fusion,
composed quantization).

> **Note:** The original Phase A design included a `StreamingGpuBundleBuilder`
> that bypassed the IR entirely. This was removed in favor of
> spill-after-quantize, which achieves the same memory profile without
> sacrificing graph-level pass capabilities.

### 7.4 ANE Bundle Path (smaller models, full materialization acceptable)

```
SafeTensors (mmap) ──→ weights_to_program()  ← External refs (Phase B)
                           │
                           ▼
                    program.materialize_all()  ← loads all tensors from mmap
                           │
                           ▼
                    ANE split / pack / compile  ← needs full tensor data
                           │
                           ▼
                    write ANE bundle
```

ANE targets smaller models (constrained by ANE weight limits, typically < 1 GB
per subprogram). Full materialization is acceptable.

### 7.5 CoreML / mlpackage Path (full materialization before serialization)

```
Program (with External refs) ──→ program.materialize_all()
                                       │
                                       ▼
                                ir_to_proto()  ← converts inline bytes to proto
                                       │
                                       ▼
                                write_mlpackage()  ← protobuf encode + write
```

### 7.6 ONNX Import Path (no change — already inline)

ONNX import reads initializer bytes eagerly and constructs `TensorData::Inline`.
No behavioral change.

## 8. `MilWeightProvider` — Destructive Extraction

`MilWeightProvider::new()` currently clones every tensor from the Program.
Change it to take ownership via `std::mem::take`.

As with quantization passes (§5.2), the extraction loop iterates
`program.functions.values_mut()` while needing to resolve external tensors.
The same Arc-clone pattern applies:

```rust
// Change signature from &Program to &mut Program
pub fn new(program: &mut Program, config: ModelConfig) -> Result<Self, MilError> {
    // Clone the provider Arc before iterating functions (same pattern as §5.2).
    let provider = program.weight_provider.clone();
    let resolve = |key: &str| -> Result<Vec<u8>, MilError> {
        let p = provider.as_ref().ok_or_else(|| /* ... */)?;
        let tensor = p.tensor(key)?;
        Ok(tensor.data.into_owned())
    };

    // ...
    for op in &mut function.body.operations {
        // ...
        if let Some(Value::Tensor { ref mut data, shape, dtype }) = val {
            let bytes = match std::mem::replace(data, TensorData::inline(Vec::new())) {
                TensorData::Inline(bytes) => bytes,
                TensorData::External { provider_key, .. } => resolve(&provider_key)?,
            };
            tensors.insert(name, ExtractedTensor { data: bytes, shape, dtype, quant_info });
        }
    }
}
```

This eliminates the largest single source of memory duplication.

## 9. `GpuCompileBuilder` Integration (Phase B)

> **Note:** This section shows the Phase B version of `GpuCompileBuilder::build()`,
> which uses `Arc<dyn WeightProvider>` attached to the program. The Phase A
> version (Task A3) is simpler — it drops the source provider before extraction
> and uses `collect_supplement_tensors()` to preserve non-program tensors. The
> Phase B version subsumes A3 because the `Arc` is shared: the program holds a
> reference for lazy resolution, and `supplement_from()` uses the same `Arc`
> after extraction. Clearing `program.weight_provider` before extraction releases
> the program's reference, allowing the `Arc` to drop after `supplement_from()`.

```rust
// crates/ironmill-compile/src/gpu/mod.rs

pub fn build(self) -> Result<MilWeightProvider, CompileError> {
    // ...
    let (mut program, config, base_provider) = match format {
        InputFormat::SafeTensors | InputFormat::Gguf => {
            let provider = load_weight_provider(input, &format)?;
            let config = provider.config().clone();

            // Wrap in Arc for shared ownership between program and builder.
            let provider_arc: Arc<dyn WeightProvider + Send + Sync> =
                Arc::from(provider);

            // Template emission creates External refs — no weight copy.
            let result = crate::templates::weights_to_program(provider_arc.as_ref())?;
            let mut program = result.program;

            // Attach provider for lazy resolution by passes.
            program.set_weight_provider(provider_arc.clone());

            (program, config, Some(provider_arc))
        }
        // ONNX path unchanged — already loads eagerly into Inline tensors.
        InputFormat::Onnx => {
            let (program, config) = import_onnx(input)?;
            (program, config, None)
        }
        // ...
    };

    // Run passes — quantization passes resolve tensors on demand.
    let _report = self.pipeline.run(&mut program)?;

    // Extract weights destructively — program tensor data is consumed.
    let mut provider = MilWeightProvider::new(&mut program, config)?;

    // Supplement from base provider for extra tensors (q_norm, k_norm, etc.).
    if let Some(base) = base_provider.as_ref() {
        provider.supplement_from(base.as_ref())?;
    }

    Ok(provider)
}
```

## 10. Memory Budget Analysis

### 10.1 Before — All Formats (70B FP16, ~140 GB weights)

| Stage | Memory |
|-------|--------|
| mmap'd SafeTensors shards | ~0 (kernel-managed) |
| Template emission copies all weights | +140 GB |
| Program holds all weight data inline | 140 GB |
| MilWeightProvider clones all data | +140 GB |
| **Peak** | **~280 GB** |

### 10.2 ~~After — Streaming GPU Bundle (Phase A, Task A1)~~ → Replaced

> **Note:** The streaming GPU bundle builder was replaced by
> spill-after-quantize in the IR path (§10.4), which achieves the same
> ~1 GB peak memory while retaining graph-level pass capabilities.
> The analysis below is preserved for historical reference.

All input formats achieve ~1 GB peak memory. Every quantization method
(Affine INT4/INT8, PolarQuant, SpinQuant, D2Quant, QuIP#, FP16, Palettize,
AWQ, GPTQ) is supported — they all operate on one tensor at a time using
standalone math functions. AWQ and GPTQ carry pre-computed calibration
HashMaps (channel magnitudes / Hessians) which are small metadata, not weight
data.

| Format | Stage | Memory |
|--------|-------|--------|
| SafeTensors/GGUF | mmap'd shards | ~0 |
| | Per-tensor: load → quantize → write → drop | ~1 GB |
| | **Peak** | **~1 GB** |
| ONNX (external data) | Parse proto metadata | ~10 MB |
| | Per-tensor: read sidecar → quantize → write → drop | ~1 GB |
| | **Peak** | **~1 GB** |
| ONNX (inline, with `Bytes`) | Parse proto (zero-copy raw_data from mmap) | ~50 MB metadata |
| | Per-tensor: slice from mmap → quantize → write → drop | ~1 GB |
| | **Peak** | **~1 GB** |

### 10.3 After — Destructive Extraction Only (Phase A, Tasks A2+A3)

| Stage | Memory |
|-------|--------|
| mmap'd SafeTensors shards | ~0 (kernel-managed) |
| Template emission copies all weights | +140 GB |
| Program holds all weight data inline | 140 GB |
| MilWeightProvider takes ownership (no clone) | ~140 GB |
| Source provider dropped before extraction | −0 (mmap released) |
| **Peak** | **~140 GB** |

### 10.4 Lazy + Spill-After-Quantize via IR (Phase B, same model, INT4)

| Stage | Memory |
|-------|--------|
| mmap'd SafeTensors shards | ~0 (kernel-managed) |
| Template emission creates External refs | ~10 MB (metadata) |
| Quantization: one FP16 tensor materialized at a time | ~1 GB (largest tensor) |
| Spill-after-quantize writes quantized data to temp dir | ~0 (immediately evicted) |
| MilWeightProvider resolves from spill dir + provider | ~1 GB (one tensor at a time) |
| **Peak** | **~1 GB** |

> **Note:** The original design estimated ~36 GB peak for this path because
> quantized tensors accumulated inline in the IR. With spill-after-quantize,
> `PassPipeline::run()` calls `program.spill_inline_tensors(4096)` after each
> pass, evicting large tensors to a temp directory. The resolver
> (`make_resolver`) checks the spill directory first, then falls back to the
> weight provider. This reduces peak memory to ~1 GB — the same as the
> original streaming path — while retaining full graph-level pass
> capabilities.

## 11. Implementation Plan

The plan is structured in two phases. **Phase A** delivers the core capability
(streaming GPU bundle compilation of arbitrarily large models) with minimal
blast radius. **Phase B** is an optional follow-up that brings lazy tensor
benefits to the MIL IR itself, enabling memory-efficient compilation through
the existing pass pipeline and non-GPU paths.

### Phase A: Streaming GPU Bundle + Quick Memory Wins

These tasks are independent and can be parallelized.

#### ~~Task A1: Streaming GPU Bundle Builder~~ → Replaced by Spill-After-Quantize

**Status:** Removed. The streaming builder was replaced by
`Program::spill_inline_tensors()` in the pass pipeline, which achieves the
same ~1 GB peak memory without bypassing the IR. See Phase B tasks below.

The spill-after-quantize mechanism:
- After each pass in `PassPipeline::run()`, large inline tensors (≥ 4096 bytes)
  are written to a `tempfile::TempDir` and replaced with `TensorData::External`
  references.
- The `make_resolver()` utility (used by all 12 quantization passes) checks
  the spill directory first, then falls back to the weight provider.
- `MilWeightProvider::new()` resolves spilled tensors from disk during
  extraction.
- The temp directory is auto-cleaned when the `Program` is dropped.

This approach retains full graph-level pass capabilities (AWQ fusion, rotation
fusion, composed quantization) while achieving streaming-like memory
efficiency.

#### Task A2: `MilWeightProvider` Destructive Extraction

**Crate:** `ironmill-compile`  
**Depends on:** nothing

This is a small, high-value change for the existing (non-streaming) path.
`MilWeightProvider::new()` currently clones every tensor's `Vec<u8>` from the
Program. Change it to take ownership:

- Change signature: `new(&Program, ...)` → `new(&mut Program, ...)`
- Replace `data.clone()` with `std::mem::take(data)` at extraction sites
  (`mil_provider.rs:61-68, 241-248, 478-502`)
- Update all call sites of `MilWeightProvider::new()`

**Impact:** Eliminates ~140 GB of redundant allocation for a 70B model.
Benefits ALL models, not just large ones. Safe, mechanical change.

**Phase B note:** When Task B1 changes `data` from `Vec<u8>` to `TensorData`,
the `std::mem::take` calls become `std::mem::replace(data, TensorData::inline(Vec::new()))`
and must also handle the `TensorData::External` variant (see §8). This is a
mechanical update applied in Task B5.

#### Task A3: Drop Source Provider Before Building MilWeightProvider

**Crate:** `ironmill-compile`  
**Depends on:** nothing

In `GpuCompileBuilder::build()` (`gpu/mod.rs:92-121`), the base
`SafeTensorsProvider`/`GgufProvider` stays alive alongside the
`MilWeightProvider`. Restructure to drop it before extraction:

```rust
/// Collect tensors from the source provider that are NOT emitted into the MIL
/// program (e.g., `q_norm`, `k_norm` weight tensors used by some architectures
/// but referenced only at bundle-writing time, not as MIL const ops).
/// Returns a map of name → (bytes, shape, dtype) for later injection into the
/// MilWeightProvider.
fn collect_supplement_tensors(
    provider: &dyn WeightProvider,
    program: &Program,
) -> HashMap<String, (Vec<u8>, Vec<usize>, ScalarType)> {
    let program_tensor_names: HashSet<&str> = /* names of all const ops in program */;
    provider.tensor_names().iter()
        .filter(|name| !program_tensor_names.contains(name.as_str()))
        .filter_map(|name| {
            let t = provider.tensor(name).ok()?;
            Some((name.to_string(), (t.data.into_owned(), t.shape.clone(), t.dtype)))
        })
        .collect()
}

// After template emission, collect any supplement tensors (q_norm, k_norm)
// that won't be in the MIL program, then drop the source provider.
let supplement_tensors = collect_supplement_tensors(&*provider, &program);
drop(provider); // release mmap pages

let mut mil_provider = MilWeightProvider::new(&mut program, config)?;

/// Inject supplementary tensors that were not part of the MIL program.
mil_provider.apply_supplements(supplement_tensors);
```

**Impact:** Releases the mmap file handles and associated kernel page cache
pressure before the MilWeightProvider allocation.

#### ~~Task A4: Testing for Phase A~~ → Superseded

Testing for the spill-after-quantize mechanism is covered in Task B7.
The `MilWeightProvider` destructive extraction tests remain valid.

### Phase B: Lazy `TensorData` in MIL IR (Optional Follow-Up)

Phase B brings memory efficiency to the full pass pipeline and non-GPU
compilation paths. It is valuable for:
- Compiling 13-30B models through the full pass pipeline (e.g., AWQ + rotation
  fusion + GPU bundle)
- Future CoreML compilation of larger models
- Code cleanliness — making the IR honest about where data lives

Phase B is NOT required for the primary goal (70B+ GPU bundles), which Phase A
fully solves.

#### Task B1: `TensorData` Enum and `Value::Tensor` Refactor

**Crate:** `mil-rs`  
**Depends on:** nothing (can be done in parallel with Phase A)

- Add `TensorData` enum to `ir/types.rs`
- Change `Value::Tensor { data: Vec<u8> }` → `Value::Tensor { data: TensorData }`
- Add `TensorData` API methods (`byte_len`, `as_bytes`, `as_bytes_mut`,
  `is_inline`, `is_external`, `into_bytes`, `resolve_with`, `materialize_with`)
- Add `From<Vec<u8>> for TensorData`
- Add `Value::tensor()` convenience constructor
- Fix all pattern matches in `mil-rs` that destructure `Value::Tensor`:
  - `convert/ir_to_proto.rs` (5 sites)
  - `convert/proto_to_ir.rs` (2 sites)
- Fix all construction sites to use `TensorData::inline()` or `Value::tensor()`

#### Task B2: `Program` Weight Provider Attachment

**Crate:** `mil-rs`  
**Depends on:** Task B1

- Add `weight_provider` field to `Program`
- Add `set_weight_provider()`, `resolve_tensor()`, `has_weight_provider()` methods
- Add `materialize_all()` method with block-level helper
- Add `Send + Sync` bound to `WeightProvider` trait
- Verify all `WeightProvider` implementors satisfy `Send + Sync`
- Update `Program::new()` to initialize `weight_provider: None`
- Update `Program`'s `Clone` impl (clone the `Arc`)

#### Task B3: Update Quantization Passes

**Crate:** `mil-rs`  
**Depends on:** Task B2

- Add shared helper for the materialize-before-extraction pattern
- Update all 12 quantization passes to materialize before accessing tensor bytes:
  `AffineQuantizePass`, `PolarQuantPass`, `SpinQuantPass`, `D2QuantPass`,
  `QuipSharpPass`, `AwqQuantizePass`, `GptqQuantizePass`, `Int8QuantizePass`,
  `Fp16QuantizePass`, `PalettizePass`, `GroupedPalettizePass`, `DacPass`
- Update `PolarRotationFusionPass` — only constructs small new tensors
  (use `TensorData::inline()`), reads attributes (no data bytes)
- Update `QuantizationEvaluator` to document materialization precondition

#### Task B4: Template System — Lazy Emission

**Crate:** `ironmill-compile`  
**Depends on:** Task B1

- Update `emit_weight_const()` to emit `TensorData::External` for large tensors
  and `TensorData::Inline` for small tensors below `INLINE_THRESHOLD_BYTES`
- Update RoPE table construction to use `TensorData::inline()`

#### Task B5: Integration — Lazy `GpuCompileBuilder`

**Crate:** `ironmill-compile`  
**Depends on:** Tasks B1-B4, A2

- Update `GpuCompileBuilder::build()` to wrap provider in `Arc`
- Attach provider to program after template emission
- Manage provider lifetime for minimal peak memory
- Update `MilWeightProvider` to handle `TensorData::External` variant

#### Task B6: Fix Remaining Usage Sites

**Crates:** `ironmill-compile`, `ironmill-core`, `ironmill-inference`  
**Depends on:** Task B1

- Update `ane/split.rs` `weight_data_size()` to use `byte_len()`
- Update `ane/validate.rs` to use `byte_len()` for metadata-only access
- Update `ane/bundle.rs`, `ane/decode_compile.rs` to call
  `program.materialize_all()` before ANE compilation
- Update all ANE passes: `ane_arg_promotion`, `ane_layout`,
  `ane_matmul_to_conv`, `codebook`, `mixed_precision`, `op_split`
- Update `ane/packing.rs` to use `TensorData::inline()` for small constants
- Update `ironmill-core/ane/mil_text.rs` (6 sites)
- Update `ironmill-inference/ane/turboquant/mil_emitter.rs` (4 sites)
- Update `convert/pipeline.rs` if needed

#### Task B7: Testing for Phase B

**Depends on:** Tasks B1-B6

- Add unit tests for `TensorData` API
- Add integration test: build a program with external refs, run a quantization
  pass, verify output matches the eager (baseline) path
- Run full existing test suite
- Benchmark: compare peak RSS for lazy vs eager pass pipeline

## 12. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ~~Streaming quantization diverges from pass-based quantization~~ | — | — | ~~Removed: streaming builder replaced by spill-after-quantize.~~ |
| ~~Eligibility logic duplicated between passes and streaming builder~~ | — | — | ~~Removed: no separate streaming builder.~~ |
| Spill directory I/O bottleneck on slow storage | Low | Medium | Spill goes to OS temp dir (typically SSD/tmpfs). Threshold of 4096 bytes avoids spilling small tensors. Sequential I/O is fast. |
| `MilWeightProvider` destructive extraction breaks callers that re-read the program | Low | Medium | Audit all call sites; `MilWeightProvider::new()` is the last step before bundle writing — the program isn't reused after. |
| Phase B `Send + Sync` bound on `WeightProvider` breaks a downstream implementor | Low | High | All known implementors use `Send + Sync` types. Verify at compile time. |
| Phase B missed usage site panics at runtime | Medium | Medium | `into_bytes()` on External panics with clear message. The census covers all ~60 known sites. |

## 13. Future Work

- **Parallel per-tensor quantization**: the streaming builder iterates tensors
  sequentially; a thread pool could quantize multiple tensors in parallel since
  they are independent. Bounded by disk I/O for writes.
- **External blob support in CoreML writer**: emit weight data as separate blob
  files in the `.mlpackage` instead of inline protobuf fields, enabling
  arbitrarily large CoreML models.
- **ONNX lazy import**: for ONNX models with external data, emit
  `TensorData::External` references instead of eagerly loading initializers
  (requires Phase B).
- **Automatic fallback**: detect available RAM and automatically choose
  streaming vs eager compilation.
