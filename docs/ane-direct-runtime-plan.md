# ANE Direct Runtime Backend — Implementation Plan

## 1. Overview

This plan describes how to build a **Rust-native ANE runtime backend** for
ironmill, providing an alternative to the CoreML `MLModel` path. The backend
uses Apple's private ANE APIs (`_ANEClient`, `_ANECompiler`) — the same APIs
reverse-engineered by [Orion](https://github.com/mechramc/Orion) and
[maderix/ANE](https://github.com/maderix/ANE) — to compile and execute
models directly on the Neural Engine.

### Why

- **Control of the inference loop** — enables runtime optimizations impossible
  through CoreML's opaque `MLModel` API (e.g., TurboQuant KV cache compression,
  custom attention kernels, dynamic batching).
- **Access to hidden hardware features** — ANE natively supports INT4/UINT4,
  nearly doubling throughput over INT8, but CoreML doesn't expose this.
- **No CoreML overhead** — bypasses CoreML's compilation, dispatch, and format
  restrictions.
- **Training capability** — ANE can execute backward passes (demonstrated by
  Orion), opening future on-device fine-tuning.

### What this unlocks

Once the ANE direct backend exists, the
[TurboQuant runtime plan](research/turboquant-ane-orion.md) can be implemented on top
of it for KV cache compression at 3–4 bits with zero CPU/GPU fallback.

### Relationship to existing code

ironmill already has scaffolding for this:

| Component | Status | Location |
|-----------|--------|----------|
| `Backend::AneDirect` enum variant | ✅ Exists | `mil-rs/src/compiler.rs` |
| `ane-direct` feature flag | ✅ Exists | `mil-rs/Cargo.toml`, `ironmill-cli/Cargo.toml` |
| `AneCompiler` skeleton | ⚠️ Stub (returns "not yet implemented") | `mil-rs/src/ffi/ane.rs` |
| `ironmill-coreml` runtime crate | ✅ Production (reference pattern) | `crates/ironmill-coreml/` |
| MIL text emitter | ❌ Missing | — |
| IOSurface tensor management | ❌ Missing | — |
| ANE program execution | ❌ Missing | — |
| Program cache | ❌ Missing | — |
| Weight blob (BLOBFILE) writer | ❌ Missing | — |

---

## 2. Architecture

### Crate structure

Following the existing pattern where `ironmill-coreml` is the CoreML runtime
crate, the ANE runtime lives in a new `ironmill-ane` crate:

```
crates/
├── mil-rs/                        # Core library (exists)
│   ├── src/
│   │   ├── ffi/
│   │   │   ├── mod.rs             # exists
│   │   │   └── ane.rs             # exists (compiler skeleton → complete)
│   │   ├── compiler.rs            # exists (add AneDirect compile path)
│   │   └── convert/
│   │       ├── ir_to_proto.rs     # exists (CoreML protobuf)
│   │       └── ir_to_mil_text.rs  # NEW — MIL text emitter for ANE
│   └── Cargo.toml                 # add IOSurface deps under ane-direct
│
├── ironmill-ane/                  # NEW — ANE runtime crate
│   ├── src/
│   │   ├── lib.rs                 # public API: AneModel, AneTensor, AneRuntime
│   │   ├── runtime.rs             # _ANEClient lifecycle
│   │   ├── program.rs             # compiled program wrapper
│   │   ├── tensor.rs              # IOSurface-backed tensor I/O
│   │   ├── cache.rs               # program cache with eviction
│   │   ├── blobfile.rs            # BLOBFILE weight format writer
│   │   └── split.rs               # model → sub-program splitter
│   └── Cargo.toml
│
├── ironmill-coreml/               # exists (reference pattern)
├── ironmill-cli/                  # exists (add --runtime ane-direct)
└── ironmill-bench/                # exists (add ANE benchmarks)
```

### Dependency flow

```
ironmill-cli
  ├── mil-rs              (IR, conversion, compilation)
  ├── ironmill-coreml     (CoreML runtime, macOS-only)
  └── ironmill-ane        (ANE runtime, macOS-only, feature-gated)
        └── mil-rs        (for IR types + MIL text emitter)
```

### API design (mirroring `ironmill-coreml`)

`ironmill-coreml` exposes:

```rust
pub struct Model { ... }
impl Model {
    pub fn load(path: &Path, compute_units: ComputeUnits) -> Result<Self>;
    pub fn input_description(&self) -> Result<InputDescription>;
    pub fn predict(&self, input: &PredictionInput) -> Result<PredictionOutput>;
}
```

`ironmill-ane` will expose a parallel API:

```rust
pub struct AneModel { ... }
impl AneModel {
    pub fn compile(program: &Program, config: AneConfig) -> Result<Self>;
    pub fn load(compiled_dir: &Path) -> Result<Self>;
    pub fn input_description(&self) -> Vec<TensorDescriptor>;
    pub fn predict(&self, inputs: &[AneTensor]) -> Result<Vec<AneTensor>>;
}

pub struct AneTensor { ... }
impl AneTensor {
    pub fn new(shape: &[usize], dtype: ScalarType) -> Result<Self>;
    pub fn write_f16(&mut self, data: &[f16]);
    pub fn read_f16(&self) -> Vec<f16>;
    pub fn as_iosurface(&self) -> *mut c_void;
}

pub struct AneConfig {
    pub max_programs: usize,         // default: 100 (under ~119 limit)
    pub cache_dir: Option<PathBuf>,  // compiled program cache
    pub enable_int4: bool,           // use INT4 when available
}
```

---

## 3. ANE constraints that shape the design

These constraints, discovered by Orion and maderix, are load-bearing for the
implementation. Violating them causes silent wrong data or crashes with no
useful error messages.

### Critical constraints

| # | Constraint | Impact on design |
|---|-----------|-----------------|
| 3 | Output IOSurfaces ordered **alphabetically** by MIL variable name | MIL text emitter must name outputs to match expected order. Runtime must sort surfaces alphabetically. |
| 5 | **~119 compile limit** per process | Program cache with disk persistence is mandatory. Each layer with distinct weights consumes one compilation slot. |
| 6 | **SDPA causal masks ignored** | Cannot use ANE's SDPA op for autoregressive generation. Must decompose attention into matmul + mask + softmax + matmul. |
| 7 | **Weights baked at compile time** | BLOBFILE must be written before compilation. Weight updates require recompilation. Layers with different weights cannot share compiled programs. |
| 8 | **BLOBFILE offset starts at byte 64** | MIL text weight references use `offset=uint64(64)` pointing to the chunk header, not byte 0. |
| 9 | **milText must be NSData\*** | MIL text must be UTF-8 `NSData`, not `NSString`. |
| 11 | **Empty weight dict at eval time** | `ANEProgramProcessRequestDirect` requires an `NSDictionary` for the weight parameter; always pass `@{}` (empty) since weights are baked at compile time. |
| 13 | Input IOSurfaces ordered **alphabetically** by MIL parameter name | Same as #3 but for inputs. |
| 14 | **Flat buffer = packed shape data** | IOSurface data must be packed contiguously, no stride padding. |

### Memory constraints

| # | Constraint | Impact on design |
|---|-----------|-----------------|
| 2 | Uniform output buffer sizes | All output IOSurfaces for a program must have the same allocation size (pad to max). |
| 4 | Minimum ~49KB IOSurface | Minimum decode sequence length is 16, not 1. Pad small tensors. |
| 12 | Uniform input buffer sizes | Same as #2 but for inputs. |

### Op constraints

| # | Constraint | Impact on design |
|---|-----------|-----------------|
| 1 | No `concat` op | Use multi-output programs instead. |
| 10 | No `gelu` op | Decompose to tanh approximation (ironmill already does this in `OpSubstitutionPass`). |

---

## 4. Implementation tasks

### Task 1 — MIL text emitter (`ir_to_mil_text.rs`)

Create `crates/mil-rs/src/convert/ir_to_mil_text.rs`.

This is the most critical new module. It converts ironmill's MIL IR (`Program`)
into the text-based MIL format that `_ANECompiler` consumes.

**MIL text format** (derived from Orion's generated programs):

```
program(1.0)
func main(
    %x: tensor<fp16, [1, 768, 1, 32]>,
    %mask: tensor<fp16, [1, 1, 1, 32]>
) -> (tensor<fp16, [1, 768, 1, 32]>) {
    %w = const(name="layer0_w", val=blob(file="weights.blob", offset=uint64(64)))
    %bias = const(name="layer0_b", val=blob(file="weights.blob", offset=uint64(128)))
    %y = conv(x=%x, weight=%w, pad_type="valid")
    %z = add(x=%y, y=%bias)
    %out = relu(x=%z)
} -> (%out)
```

Key differences from CoreML protobuf MIL:

| Aspect | CoreML protobuf | ANE MIL text |
|--------|----------------|-------------|
| Format | Binary protobuf | UTF-8 text |
| Tensor layout | Flexible | Must be `[1, C, 1, S]` (NHWC-like) |
| Weights | Inline or external blob | BLOBFILE with byte offset |
| Types | Full type system | `fp16`, `fp32`, `int8`, `int4`, `bool` |
| Const bools | Inline `true`/`false` | Named const refs (constraint from Orion) |

**Public API:**

```rust
/// Convert an ironmill Program to ANE MIL text format.
///
/// Returns the MIL text and a list of weight blob entries that must be
/// written to the BLOBFILE before compilation.
pub fn program_to_mil_text(
    program: &Program,
    config: &MilTextConfig,
) -> Result<(String, Vec<WeightBlobEntry>)>;

pub struct MilTextConfig {
    /// Target tensor layout. Default: [1, C, 1, S].
    pub layout: AneLayout,
    /// Whether to use INT4 for quantized tensors.
    pub enable_int4: bool,
}

pub struct WeightBlobEntry {
    pub name: String,
    pub data: Vec<u8>,
    pub offset: u64,
    pub dtype: ScalarType,
    pub shape: Vec<usize>,
}
```

**Implementation concerns:**

- Must transpose tensors from ironmill's default layout to `[1, C, 1, S]`.
- Must name all variables so that alphabetical ordering matches intended I/O
  order (constraint #3, #13). Use a naming scheme like `a_input0`, `a_input1`
  for inputs and `z_output0`, `z_output1` for outputs.
- Must decompose unsupported ops (concat, gelu) before emission. Run
  `OpSubstitutionPass` first in the pipeline.
- Must emit const bools as named const refs, not inline literals.
- Weight references use `blob(file="...", offset=uint64(N))` syntax. Offsets
  are computed during BLOBFILE construction.

---

### Task 2 — BLOBFILE writer (`blobfile.rs`)

Create `crates/ironmill-ane/src/blobfile.rs`.

The BLOBFILE format stores weight data for ANE programs. Structure:

```
Bytes 0-63:    File header (magic, version, metadata)
Bytes 64-127:  Chunk header (data offset, size)
Bytes 128+:    Weight data (packed, fp16 by default)
```

MIL text references weights at `offset=uint64(64)` (pointing to the chunk
header, not byte 0 — constraint #8).

**Public API:**

```rust
pub struct BlobFileWriter {
    buffer: Vec<u8>,
    entries: Vec<BlobEntry>,
}

impl BlobFileWriter {
    pub fn new() -> Self;
    /// Add a weight tensor. Returns the offset for MIL text reference.
    pub fn add_weight(&mut self, name: &str, data: &[u8], dtype: ScalarType) -> u64;
    /// Write the complete BLOBFILE to disk.
    pub fn write(&self, path: &Path) -> Result<()>;
}
```

---

### Task 3 — IOSurface tensor manager (`tensor.rs`)

Create `crates/ironmill-ane/src/tensor.rs`.

All ANE I/O uses IOSurface-backed memory. This module manages tensor
creation, data transfer, and lifecycle.

**FFI required:**

```rust
#[link(name = "IOSurface", kind = "framework")]
unsafe extern "C" {
    fn IOSurfaceCreate(properties: *const c_void) -> *mut c_void;
    fn IOSurfaceGetBaseAddress(surface: *mut c_void) -> *mut c_void;
    fn IOSurfaceLock(surface: *mut c_void, options: u32, seed: *mut u32) -> i32;
    fn IOSurfaceUnlock(surface: *mut c_void, options: u32, seed: *mut u32) -> i32;
    fn IOSurfaceGetAllocSize(surface: *mut c_void) -> usize;
}
```

Alternatively, use `objc2-io-surface` if available in the `objc2` ecosystem.

**Key design rules (from ANE constraints):**

- All surfaces for a program's inputs must have the same allocation size
  (constraint #12). Pad to the maximum.
- All surfaces for a program's outputs must have the same allocation size
  (constraint #2). Pad to the maximum.
- Minimum allocation ~49KB (constraint #4). Pad small tensors.
- Data must be written **packed** (no stride padding) at the start of the
  surface (constraint #14).
- Layout is always `[1, C, 1, S]` in fp16.

**Public API:**

```rust
pub struct AneTensor {
    surface: *mut c_void,  // IOSurfaceRef
    shape: [usize; 4],     // [1, C, 1, S]
    dtype: ScalarType,
    alloc_size: usize,
}

impl AneTensor {
    /// Create a new IOSurface-backed tensor.
    pub fn new(channels: usize, seq_len: usize, dtype: ScalarType) -> Result<Self>;

    /// Create with a specific minimum allocation size (for uniform sizing).
    pub fn new_with_min_alloc(
        channels: usize,
        seq_len: usize,
        dtype: ScalarType,
        min_alloc: usize,
    ) -> Result<Self>;

    /// Write packed f16 data into the surface.
    pub fn write_f16(&mut self, data: &[f16]);

    /// Read packed f16 data from the surface.
    pub fn read_f16(&self) -> Vec<f16>;

    /// Write packed f32 data (converted to f16 internally).
    pub fn write_f32(&mut self, data: &[f32]);

    /// Raw IOSurface pointer for ANE API calls.
    pub fn as_ptr(&self) -> *mut c_void;

    /// Allocation size in bytes.
    pub fn alloc_size(&self) -> usize;
}

impl Drop for AneTensor {
    fn drop(&mut self) { /* CFRelease the IOSurface */ }
}

/// Compute the uniform allocation size for a set of tensors.
/// All tensors in a program's input (or output) set must use this size.
pub fn uniform_alloc_size(tensors: &[&AneTensor]) -> usize;
```

---

### Task 4 — ANE compiler completion (`ffi/ane.rs`)

Complete the existing `AneCompiler` skeleton in `crates/mil-rs/src/ffi/ane.rs`.

The skeleton already has `objc_getClass`, `sel_registerName`, and `objc_msgSend`
declarations. The `compile()` method needs to be completed.

**What needs to happen:**

```rust
impl AneCompiler {
    pub fn compile(mil_text: &str, weight_path: &Path) -> Result<CompiledProgram> {
        // 1. Create _ANEInMemoryModelDescriptor
        let desc_class = objc_getClass(sel!("_ANEInMemoryModelDescriptor"));
        let desc = objc_msgSend(desc_class, sel_registerName(sel!("alloc")));
        let desc = objc_msgSend(desc, sel_registerName(sel!("init")));

        // 2. Set milText as NSData (NOT NSString — constraint #9)
        let mil_data = NSData::from_bytes(mil_text.as_bytes());
        objc_msgSend(desc, sel_registerName(sel!("setMilText:")), mil_data);

        // 3. Set weight path
        let weight_url = NSURL::from_path(weight_path);
        objc_msgSend(desc, sel_registerName(sel!("setWeightPath:")), weight_url);

        // 4. Compile
        let compiler_class = objc_getClass(sel!("_ANECompiler"));
        let result = objc_msgSend(
            compiler_class,
            sel_registerName(sel!("compileDescriptor:error:")),
            desc,
            &mut error,
        );

        // 5. Return compiled program handle
        Ok(CompiledProgram { inner: result })
    }
}
```

**Note:** The exact Objective-C selector names and argument conventions must be
verified against Orion's source code and maderix's API documentation. The
selectors above are approximations — the actual private API may differ.

> ⚠️ **Incorrect selectors will segfault at runtime with no useful diagnostics.**
> Before porting to Rust FFI, validate all selectors in a minimal Objective-C
> test harness (see Risk R2).

**Compile count tracking:**

```rust
static COMPILE_COUNT: AtomicUsize = AtomicUsize::new(0);

pub fn compile_count() -> usize {
    COMPILE_COUNT.load(Ordering::Relaxed)
}

pub fn remaining_compile_budget() -> usize {
    119_usize.saturating_sub(compile_count())
}
```

---

### Task 5 — ANE runtime (`runtime.rs`)

Create `crates/ironmill-ane/src/runtime.rs`.

This wraps `_ANEClient` for program loading and execution.

**Key operations:**

```rust
pub struct AneRuntime {
    client: *mut c_void,  // _ANEClient instance
}

impl AneRuntime {
    /// Initialize the ANE client.
    pub fn new() -> Result<Self>;

    /// Load a compiled program for execution.
    pub fn load_program(&self, program: &CompiledProgram) -> Result<LoadedProgram>;

    /// Execute a loaded program with input/output tensors.
    ///
    /// Inputs and outputs must be sorted alphabetically by their MIL
    /// parameter/variable names (constraints #3, #13).
    /// All inputs must have uniform allocation size (constraint #12).
    /// All outputs must have uniform allocation size (constraint #2).
    pub fn eval(
        &self,
        program: &LoadedProgram,
        inputs: &[&AneTensor],
        outputs: &mut [&mut AneTensor],
    ) -> Result<()>;

    /// Unload a program (frees ANE resources).
    pub fn unload_program(&self, program: LoadedProgram);
}
```

**FFI calls inside `eval`:**

```rust
// Core eval call — this is the hot path
ANEProgramProcessRequestDirect(program_handle, request, weight_dict)
```

**Weight dict:** Always pass `@{}` (empty NSDictionary) for the weight
parameter in the underlying `ANEProgramProcessRequestDirect` call
(constraint #11). Weights are baked at compile time; the weight dict
parameter exists in the private API but is not used for runtime injection.

**Thread safety:** `_ANEClient` is assumed **not thread-safe**. `AneRuntime`
should be `Send` but not `Sync` — callers must not share a reference across
threads. Use one `AneRuntime` per thread, or wrap in a `Mutex` if shared
access is needed. Verify thread safety empirically once the runtime is
functional.

**Error handling:** All methods return `Result<T, AneError>`. `AneError`
should capture the ANE status code (e.g., `0x1d`) and the operation that
failed:

```rust
#[derive(Debug, thiserror::Error)]
pub enum AneError {
    #[error("ANE compilation failed (status {status:#x}): {context}")]
    CompileFailed { status: u32, context: String },
    #[error("ANE eval failed (status {status:#x}): {context}")]
    EvalFailed { status: u32, context: String },
    #[error("IOSurface creation failed: {0}")]
    SurfaceError(String),
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}
```

---

### Task 6 — Program cache (`cache.rs`)

Create `crates/ironmill-ane/src/cache.rs`.

The ~119 compile limit (constraint #5) makes caching essential. Because
weights are baked at compile time (constraint #7), each layer with different
weights requires its own compilation — a 32-layer transformer consumes ~34
slots (32 layers + embedding + LM head). The cache serves two purposes:

1. **Disk persistence** — compiled programs are serialized to disk and reloaded
   on subsequent runs, bypassing the per-process compile limit entirely.
2. **In-process dedup** — if the same model is loaded twice, or a model reuses
   the exact same weights (e.g., tied embeddings), the cache avoids redundant
   compilation.

**Design:**

```rust
pub struct ProgramCache {
    cache: HashMap<ProgramKey, CompiledProgram>,
    disk_dir: Option<PathBuf>,
    max_entries: usize,
}

/// Cache key: the MIL text hash + weight blob hash.
/// Each unique (architecture, weights) pair requires its own compiled program
/// because weights are baked at compile time (constraint #7).
#[derive(Hash, Eq, PartialEq)]
pub struct ProgramKey {
    mil_text_hash: u64,
    weight_hash: u64,
}

impl ProgramCache {
    pub fn new(disk_dir: Option<PathBuf>, max_entries: usize) -> Self;

    /// Get or compile a program. Returns cached version if available.
    pub fn get_or_compile(
        &mut self,
        mil_text: &str,
        weights: &Path,
    ) -> Result<&CompiledProgram>;

    /// Evict least-recently-used entries.
    pub fn evict(&mut self, count: usize);

    /// Number of compilations performed this session.
    pub fn compile_count(&self) -> usize;

    /// Remaining budget before hitting the ~119 limit.
    pub fn remaining_budget(&self) -> usize;
}
```

**Disk caching:** Compiled ANE programs can be serialized to disk. On
subsequent runs, load from disk instead of recompiling. This bypasses the
per-process compile limit entirely for repeated runs — the primary
mitigation for the ~119 budget.

---

### Task 7 — Model splitter (`split.rs`)

Create `crates/ironmill-ane/src/split.rs`.

A full transformer model cannot be compiled as a single ANE program (too many
ops, weight size limits, compile budget). Split the model into sub-programs
that execute sequentially, all targeting ANE.

> **Note:** The existing `model_split.rs` pass in `crates/mil-rs/src/ir/passes/`
> handles draft/verifier splitting for speculative decoding. This splitter is
> different — it partitions a single model into ANE-sized sub-programs by layer.

**Splitting strategy:**

```
Full model graph
    │
    ├── Embedding program       (ANE — may need SRAM-aware chunking for large vocabs)
    │
    ├── Layer 0 program         (ANE — attention + FFN)
    ├── Layer 1 program         (ANE — same architecture, different weights, separate compilation)
    │   ...
    ├── Layer N program         (ANE)
    │
    ├── Final layernorm program (ANE)
    │
    └── LM head program         (ANE — may need SRAM-aware chunking for large vocabs)
```

**Public API:**

```rust
pub struct ModelSplit {
    pub programs: Vec<SubProgram>,
}

pub struct SubProgram {
    pub name: String,
    pub mil_text: String,
    pub weight_entries: Vec<WeightBlobEntry>,
    pub inputs: Vec<TensorDescriptor>,
    pub outputs: Vec<TensorDescriptor>,
}

/// Split a Program into sub-programs suitable for ANE execution.
pub fn split_for_ane(program: &Program, config: &SplitConfig) -> Result<ModelSplit>;

pub struct SplitConfig {
    /// Maximum weight size per sub-program (bytes). Sub-programs exceeding
    /// this are further chunked to fit ANE SRAM.
    pub max_weight_size: usize,
}
```

**Attention decomposition:** Because ANE's SDPA ignores causal masks
(constraint #6), the splitter must decompose `scaled_dot_product_attention`
into explicit matmul → mask → softmax → matmul chains. This can be done as a
pass before splitting, or during MIL text emission.

---

### Task 8 — `AneModel` facade (`lib.rs`)

Create `crates/ironmill-ane/src/lib.rs`.

This is the high-level API that ties everything together, mirroring
`ironmill-coreml::Model`.

```rust
pub struct AneModel {
    runtime: AneRuntime,
    programs: Vec<LoadedSubProgram>,
    cache: ProgramCache,
    config: AneConfig,
}

struct LoadedSubProgram {
    program: LoadedProgram,
    meta: SubProgram,
    input_tensors: Vec<AneTensor>,
    output_tensors: Vec<AneTensor>,
}

impl AneModel {
    /// Compile a model from ironmill IR and load it for execution.
    pub fn compile_and_load(program: &Program, config: AneConfig) -> Result<Self> {
        // 1. Run ANE-specific passes (attention decomposition, layout, op substitution)
        // 2. Split into sub-programs
        // 3. Emit MIL text + BLOBFILE for each sub-program
        // 4. Compile each sub-program (via cache)
        // 5. Load compiled programs into ANE
        // 6. Pre-allocate IOSurface tensors
        Ok(...)
    }

    /// Run inference.
    pub fn predict(&mut self, inputs: &[AneTensor]) -> Result<Vec<AneTensor>> {
        // Execute sub-programs sequentially, piping outputs → inputs
        for sub in &mut self.programs {
            self.runtime.eval(
                &sub.program,
                &sub.input_tensors.iter().collect::<Vec<_>>(),
                &mut sub.output_tensors.iter_mut().collect::<Vec<_>>(),
            )?;
            // Wire outputs of this sub-program to inputs of the next
        }
        Ok(final_outputs)
    }

    /// Input description (for building dummy inputs, benchmarking).
    pub fn input_description(&self) -> Vec<TensorDescriptor>;
}
```

---

### Task 9 — CLI integration

#### 9a. Add `--runtime` flag to `ironmill-cli`

```rust
/// Runtime backend for inference and benchmarking.
#[arg(long, default_value = "coreml")]
runtime: String,  // "coreml" | "ane-direct"
```

#### 9b. Wire to pipeline

```rust
match opts.runtime.as_str() {
    "coreml" => {
        // existing path: compile → .mlmodelc → ironmill-coreml::Model
    }
    "ane-direct" => {
        #[cfg(feature = "ane-direct")]
        {
            let model = ironmill_ane::AneModel::compile_and_load(&program, config)?;
            // ...
        }
        #[cfg(not(feature = "ane-direct"))]
        bail!("ANE direct backend requires --features ane-direct");
    }
    other => bail!("Unknown runtime: '{other}'"),
}
```

#### 9c. Add workspace member

In root `Cargo.toml`:

```toml
[workspace]
members = [
    "crates/mil-rs",
    "crates/ironmill-cli",
    "crates/ironmill-coreml",
    "crates/ironmill-ane",      # NEW
    "crates/ironmill-bench",
]

[workspace.dependencies]
ironmill-ane = { path = "crates/ironmill-ane", version = "0.1.0" }
```

#### 9d. Feature flag threading

```toml
# crates/ironmill-ane/Cargo.toml
[features]
default = []
int4 = []  # enable INT4 data type support

# crates/ironmill-cli/Cargo.toml
[features]
ane-direct = ["mil-rs/ane-direct", "dep:ironmill-ane"]
```

---

### Task 10 — Benchmark integration

Extend `crates/ironmill-bench/` to support the ANE direct backend:

```rust
enum RuntimeBackend {
    CoreMl(ironmill_coreml::Model),
    #[cfg(feature = "ane-direct")]
    AneDirect(ironmill_ane::AneModel),
}
```

This allows A/B comparison: CoreML vs ANE direct on the same model,
measuring latency, throughput, and memory.

---

## 5. ANE-specific passes

Before MIL text emission, the IR must be prepared for ANE's constraints.
These passes run after the standard pipeline but before the MIL text emitter.

| Pass | What it does | Reason |
|------|-------------|--------|
| `AneLayoutPass` | Reshape all tensors to `[1, C, 1, S]` | ANE requires this layout |
| `AttentionDecomposePass` | Replace `scaled_dot_product_attention` with explicit matmul+mask+softmax+matmul | ANE ignores causal masks in SDPA (constraint #6) |
| `AneConcatEliminationPass` | Replace `concat` with multi-output programs | ANE rejects concat (constraint #1) |
| `AneVariableNamingPass` | Rename I/O variables for alphabetical ordering | IOSurface ordering is alphabetical (constraints #3, #13) |

These live in `crates/mil-rs/src/ir/passes/` alongside existing passes, gated
behind the `ane-direct` feature.

---

## 6. Testing strategy

### Unit tests

| Test | Module | What it verifies |
|------|--------|-----------------|
| `mil_text_round_trip` | `ir_to_mil_text` | IR → MIL text → parse back → verify structure |
| `mil_text_variable_naming` | `ir_to_mil_text` | I/O variables are alphabetically ordered |
| `mil_text_weight_offsets` | `ir_to_mil_text` | BLOBFILE offsets in MIL text match writer output |
| `mil_text_layout_1c1s` | `ir_to_mil_text` | All tensors emitted as [1,C,1,S] |
| `blobfile_header_format` | `blobfile` | Header matches expected 128-byte layout |
| `blobfile_offset_64` | `blobfile` | Weight offsets point to chunk header at byte 64 |
| `tensor_packed_write` | `tensor` | Data is written packed (no stride padding) |
| `tensor_uniform_alloc` | `tensor` | `uniform_alloc_size` returns correct max |
| `tensor_min_49kb` | `tensor` | Small tensors are padded to minimum 49KB |
| `cache_dedup` | `cache` | Identical MIL text returns cached program |
| `cache_budget_tracking` | `cache` | Compile count tracks correctly |
| `split_layers` | `split` | Transformer is split into per-layer sub-programs |
| `attention_decompose` | passes | SDPA replaced with explicit matmul chain |

### Integration tests (require macOS + Apple Silicon)

| Test | What it verifies |
|------|-----------------|
| `compile_simple_program` | MIL text compiles without error via `_ANECompiler` |
| `eval_add_program` | `z = add(x, y)` produces correct output |
| `eval_matmul_program` | Matrix multiply produces correct output |
| `eval_multi_output` | Multi-output program returns surfaces in alphabetical order |
| `eval_weight_blobfile` | Weights loaded from BLOBFILE produce correct output |
| `full_model_inference` | Small model (e.g., 2-layer MLP) runs end-to-end |

Integration tests must be gated behind `#[cfg(all(target_os = "macos", feature = "ane-direct"))]`
and run on CI with Apple Silicon runners.

---

## 7. Task dependency graph

```
Task 1 (MIL text emitter)  ──┐
                              ├──▶ Task 4 (compiler completion) ──┐
Task 2 (BLOBFILE writer)   ──┘                                    │
                                                                   ├──▶ Task 8 (AneModel facade)
Task 3 (IOSurface tensors) ──┐                                    │         │
                              ├──▶ Task 5 (ANE runtime)     ──────┘         │
                              │                                              │
                              └──▶ Task 6 (program cache)   ────────────────┘
                                                                   │
Task 7 (model splitter)    ────────────────────────────────────────┘
                                                                   │
ANE-specific passes        ────────────────────────────────────────┘
                                                                   │
Task 9 (CLI integration)   ◀───────────────────────────────────────┘
Task 10 (benchmarks)       ◀───────────────────────────────────────┘
```

**Parallel tracks:**
- Tasks 1+2 (MIL text + BLOBFILE) can be developed together
- Task 3 (IOSurface) is independent of 1+2
- Tasks 4+5 (compiler + runtime) depend on all of 1, 2, 3
- Task 6 (cache) depends on 4
- Task 7 (splitter) depends on 1
- Task 8 (facade) ties everything together
- Tasks 9+10 (CLI + bench) are last

**Recommended implementation order:**

1. Task 3 — IOSurface tensors (standalone, testable without ANE)
2. Tasks 1+2 — MIL text emitter + BLOBFILE writer (testable as text output)
3. Task 4 — Compiler completion (first real ANE interaction)
4. Task 5 — Runtime (eval a simple program)
5. Task 6 — Program cache
6. ANE-specific passes
7. Task 7 — Model splitter
8. Task 8 — AneModel facade
9. Tasks 9+10 — CLI + benchmarks

---

## 8. Risks

### R1 — Private API instability

`_ANEClient` and `_ANECompiler` are undocumented private APIs. Any macOS
update can change selectors, argument conventions, or behavior.

**Mitigation:** Feature-gate behind `ane-direct`. Pin tested macOS versions.
Wrap every private API call in a version-checked safety layer. Ship CoreML
as the stable default.

### R2 — Selector and argument discovery

The exact Objective-C selectors used by Orion are in Objective-C source code.
Translating them to Rust raw FFI requires careful verification of argument
types, return types, and calling conventions.

**Mitigation:** Cross-reference Orion's source, maderix's API docs, and
mdaiter's reverse engineering. Write a minimal Objective-C test harness that
can be validated before porting to Rust.

### R3 — ANE SRAM limits

ANE has limited on-chip SRAM. Large weight tensors or activations may not
fit, causing silent fallback or compilation failure.

**Mitigation:** The model splitter (Task 7) must respect SRAM limits when
sizing sub-programs. Large layers (embeddings, LM head) should be chunked
into smaller sub-programs that fit in ANE SRAM rather than falling back to
CPU.

### R4 — INT4 memory alignment

ANE's native INT4 support requires specific tensor alignment that is not
fully documented.

**Mitigation:** Start with FP16 and INT8 (well-understood). Add INT4 behind
a separate `int4` feature flag once alignment rules are validated empirically.

### R5 — Debugging difficulty

ANE errors are opaque — `status=0x1d` with no further detail. Debugging
requires binary search over MIL programs and I/O configurations.

**Mitigation:** Comprehensive logging in debug builds. A `--ane-debug` CLI
flag that dumps MIL text, BLOBFILE hex, IOSurface metadata, and eval status
codes. Test with progressively complex programs (add → matmul → conv → full
layer).

### R6 — Scope of effort

This is a significant project — a new runtime backend, not a pass. The MIL
text emitter alone is comparable in scope to the existing `ir_to_proto.rs`.

**Mitigation:** Task 3 → 1+2 → 4 → 5 is the critical path to a "hello world"
eval. Once that works, the remaining tasks are incremental. Ship the backend
as experimental/unstable initially.

---

## 9. References

1. Orion: [github.com/mechramc/Orion](https://github.com/mechramc/Orion)
2. Orion paper: [arXiv:2603.06728](https://arxiv.org/abs/2603.06728)
3. Orion ANE constraints: [docs/ane_constraints.md](https://github.com/mechramc/Orion/blob/main/docs/ane_constraints.md)
4. maderix/ANE: [github.com/maderix/ANE](https://github.com/maderix/ANE)
5. maderix blog: [Inside the M4 Apple Neural Engine](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)
6. mdaiter/ane: [github.com/mdaiter/ane](https://github.com/mdaiter/ane) (Espresso + ANE RE)
7. hollance/neural-engine: [github.com/hollance/neural-engine](https://github.com/hollance/neural-engine)
8. TurboQuant on ANE via Orion: [docs/research/turboquant-ane-orion.md](research/turboquant-ane-orion.md)
