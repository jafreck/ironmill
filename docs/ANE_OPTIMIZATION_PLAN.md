# ANE Optimization Passes — Implementation Plan

## Overview

Add six advanced ANE-targeted optimization passes to `mil-rs`. These fall into two
categories: **always-on** passes that are safe to apply unconditionally, and **opt-in**
passes that trade accuracy or require user-provided data.

### Pass Pipeline (proposed order)

```
1. DeadCodeEliminationPass          (existing, always-on)
2. IdentityEliminationPass          (existing, always-on)
3. ConstantFoldPass                 (existing, always-on)
4. ConvBatchNormWeightFoldPass      (NEW, always-on)
5. ConvBatchNormFusionPass          (existing, always-on)
6. ConvReluFusionPass               (existing, always-on)
7. LinearReluFusionPass             (existing, always-on)
8. AttentionFusionPass              (NEW, always-on, auto-detects pattern)
9. OpSubstitutionPass               (NEW, always-on, driven by ANE support list)
10. LayoutOptimizationPass          (NEW, always-on, auto-detects current layout)
11. ShapeMaterializePass            (existing, opt-in via --input-shape)
12. Fp16QuantizePass                (existing, opt-in via --quantize fp16)
13. Int8QuantizePass                (NEW, opt-in via --quantize int8 --calibration-data)
14. PalettizePass                   (NEW, opt-in via --palettize <bits>)
```

### Mutual Exclusivity Rules

- `Fp16QuantizePass` and `Int8QuantizePass` are **mutually exclusive** — pick one
- `Int8QuantizePass` and `PalettizePass` are **mutually exclusive** — both compress
  weights, applying both degrades quality for no benefit
- `Fp16QuantizePass` and `PalettizePass` can stack (FP16 activations + palettized weights)
- All always-on passes are composable and order-independent (but the listed order is optimal)

### CLI Changes

```
coreml-kit compile model.onnx                                    # always-on passes only
coreml-kit compile model.onnx --quantize fp16                    # + FP16
coreml-kit compile model.onnx --quantize int8 --cal-data imgs/   # + INT8 (mutual excl. w/ fp16)
coreml-kit compile model.onnx --palettize 4                      # 4-bit weight palettization
coreml-kit compile model.onnx --quantize fp16 --palettize 6      # FP16 activations + 6-bit palettes
coreml-kit compile model.onnx --no-fusion                        # disable always-on fusion passes
```

---

## Task 1: Conv+BatchNorm Weight Folding

**Goal**: Mathematically fold BatchNorm parameters into Conv weights, eliminating
BN entirely (not just graph-level marking like the current `ConvBatchNormFusionPass`).

**File**: `crates/mil-rs/src/ir/passes/bn_weight_fold.rs`

### Math

Given conv output `y = W * x + b` and batch_norm `z = γ * (y - μ) / √(σ² + ε) + β`:

```
W_folded = W * γ / √(σ² + ε)
b_folded = (b - μ) * γ / √(σ² + ε) + β
```

### Implementation

```rust
pub struct ConvBatchNormWeightFoldPass;

impl Pass for ConvBatchNormWeightFoldPass {
    fn name(&self) -> &str { "conv-bn-weight-fold" }
    fn run(&self, program: &mut Program) -> Result<()> {
        // For each conv→batch_norm pair (single consumer):
        // 1. Extract conv weight tensor (from the "weight" const op feeding conv)
        // 2. Extract BN parameters: gamma (scale), beta (bias), mean, variance, epsilon
        // 3. Compute folded weights and bias using the formulas above
        // 4. Replace the conv's weight const with the folded weights
        // 5. Replace the conv's bias const with the folded bias (create one if absent)
        // 6. Remove the batch_norm op, rewire outputs
    }
}
```

### Requirements

- Must find the `const` ops that feed into the conv's "weight" and "bias" inputs
- Must find the `const` ops for BN's gamma, beta, mean, variance
- Operates on `Value::Tensor` data — needs FP32 arithmetic on raw byte buffers
- Add helper: `fn tensor_as_f32_slice(data: &[u8]) -> &[f32]`
- Add helper: `fn f32_slice_to_tensor(data: &[f32]) -> Vec<u8>`
- **Ordering**: Must run BEFORE `ConvBatchNormFusionPass` (this does the real work,
  then fusion cleans up the graph structure)
- **Safety**: Only fold when both conv weights and all BN params are available as
  `const` ops (skip if any are dynamic)

### Tests

- Fold a conv+BN pair with known weights → verify folded weights match hand-calculated values
- Skip folding when BN params are not const
- Skip folding when conv output has multiple consumers
- Verify the batch_norm op is removed after folding

---

## Task 2: Attention Pattern Fusion

**Goal**: Detect and fuse the Q·K^T·V scaled dot-product attention pattern into a
single `scaled_dot_product_attention` op that ANE handles natively.

**File**: `crates/mil-rs/src/ir/passes/attention_fusion.rs`

### Pattern to detect

The standard attention pattern in MIL looks like:

```
Q = matmul(input, W_q)        # or linear
K = matmul(input, W_k)
V = matmul(input, W_v)
scores = matmul(Q, transpose(K))
scores_scaled = real_div(scores, sqrt(d_k))  # or mul by 1/√d_k
attn_weights = softmax(scores_scaled)
output = matmul(attn_weights, V)
```

### Implementation

```rust
pub struct AttentionFusionPass;

impl Pass for AttentionFusionPass {
    fn name(&self) -> &str { "attention-fusion" }
    fn run(&self, program: &mut Program) -> Result<()> {
        // 1. Find softmax ops in the block
        // 2. For each softmax, trace backward:
        //    - softmax input should be real_div or mul (scaling)
        //    - scaling input should be matmul (Q·K^T)
        // 3. Trace forward from softmax output:
        //    - Consumer should be matmul (attn_weights · V)
        // 4. If pattern matches:
        //    a. Create a "scaled_dot_product_attention" op with Q, K, V inputs
        //    b. Replace the entire chain with the single fused op
        //    c. Remove the original ops
    }
}
```

### Design

- **Pattern matching**: Use a backward-trace helper that walks `Value::Reference` chains
- **Fuzzy matching**: The pattern may have slight variations (e.g., scaling via `mul`
  instead of `real_div`, optional mask/dropout). Start with the strict pattern, add
  variants later.
- **Auto-detect**: If no attention pattern is found, the pass is a no-op — safe as always-on
- Add helper: `fn trace_input(block: &Block, op: &Operation, input_name: &str) -> Option<&Operation>`
  that follows a Reference to find the producing op

### Tests

- Build a full attention pattern → verify fused into single op
- Partial pattern (missing softmax) → no fusion
- Multiple attention heads → each fused independently
- Non-attention softmax → not fused

---

## Task 3: Op Substitution Pass

**Goal**: Replace ANE-unsupported ops with equivalent supported alternatives.

**File**: `crates/mil-rs/src/ir/passes/op_substitute.rs`

### Substitution Table

| Unsupported Op | Replacement | Notes |
|---|---|---|
| `erf` | `tanh`-based GELU approx | `erf(x) ≈ tanh(√(2/π) * (x + 0.044715 * x³))` — expands to mul/add/tanh |
| `upsample_bilinear` (some modes) | `resize_bilinear` | ANE prefers specific resize variants |
| `conv_transpose` | `upsample` + `conv` | Decompose transposed conv for ANE compatibility |
| `shape` (dynamic) | Materialized const | Replace with const if shape is known from context |

### Implementation

```rust
pub struct OpSubstitutionPass;

impl Pass for OpSubstitutionPass {
    fn name(&self) -> &str { "op-substitution" }
    fn run(&self, program: &mut Program) -> Result<()> {
        // For each op in each block:
        // 1. Check if op_type is in the substitution table
        // 2. If yes, generate replacement op(s)
        // 3. Splice replacement into the block, rewire references
        // Auto-detects: only substitutes ops that are in the unsupported set
    }
}
```

### Design

- Each substitution is a function: `fn substitute_erf(op: &Operation) -> Vec<Operation>`
- The substitution may expand one op into multiple (e.g., erf → 5 ops for tanh approx)
- Use unique name generation for intermediate ops (e.g., `"{original_name}_sub_0"`)
- **Auto-detect**: Check against `is_ane_supported()` from validate.rs — only substitute
  ops that would fall back. If an op is already supported, skip it.

### Tests

- Substitute `erf` → verify expansion produces correct op chain
- Substitute `conv_transpose` → verify decomposition
- Already-supported op → no substitution
- Verify substituted graph still validates (all references resolve)

---

## Task 4: Memory Layout Optimization (NCHW → NHWC)

**Goal**: Reorder tensor layouts to ANE's preferred channel-last (NHWC) format,
minimizing transpose ops at boundaries.

**File**: `crates/mil-rs/src/ir/passes/layout_optimize.rs`

### Background

- ONNX models typically use NCHW (batch, channels, height, width)
- ANE prefers NHWC (batch, height, width, channels) for most ops
- Naïve approach: insert transpose before/after every op. Better: propagate the layout
  and only insert transposes at format boundaries.

### Implementation

```rust
pub struct LayoutOptimizationPass;

impl Pass for LayoutOptimizationPass {
    fn name(&self) -> &str { "layout-optimization" }
    fn run(&self, program: &mut Program) -> Result<()> {
        // 1. Identify 4D tensors in the graph (conv, pool, batch_norm ops)
        // 2. For ops that benefit from NHWC (conv, pool, batch_norm):
        //    a. Insert transpose [0,2,3,1] before the op's input (NCHW→NHWC)
        //    b. Insert transpose [0,3,1,2] after the op's output (NHWC→NCHW)
        // 3. Cancel adjacent transpose pairs (transpose_a · transpose_b = identity)
        // 4. Propagate: if an op's output feeds only NHWC consumers, skip the
        //    back-transpose and let the next op consume NHWC directly
    }
}
```

### Design

- **Auto-detect**: Only applies to 4D tensor ops. If model has no 4D ops, pass is a no-op.
- **Transpose cancellation**: After initial insertion, run a simplification that removes
  `transpose[0,2,3,1]` immediately followed by `transpose[0,3,1,2]` (and vice versa).
- **Conservative**: Start by only handling conv/pool/batch_norm. Expand to other ops later.
- Track layout per-value using a `HashMap<String, Layout>` where `Layout` is `NCHW | NHWC`.
- **Ordering**: Run after fusion passes (fused ops may change layout requirements).

### Tests

- Single conv: verify transposes inserted
- Conv→relu→conv chain: verify intermediate transposes cancelled
- Non-4D ops: verify no transposes inserted
- Model with no spatial ops: pass is no-op

---

## Task 5: INT8 Post-Training Quantization

**Goal**: Quantize weights and activations to INT8 using calibration data for
maximum ANE performance.

**File**: `crates/mil-rs/src/ir/passes/int8_quantize.rs`

### Background

INT8 gives ~2x speedup and ~4x memory reduction over FP32 on ANE, but requires
**calibration data** to compute per-tensor or per-channel scale and zero-point values.

### Implementation

```rust
/// INT8 post-training quantization with calibration data.
///
/// Requires a calibration dataset to compute scale/zero-point values.
/// Mutually exclusive with `Fp16QuantizePass`.
pub struct Int8QuantizePass {
    /// Directory containing calibration input tensors (as .npy or raw binary).
    calibration_dir: PathBuf,
    /// Per-channel (more accurate) or per-tensor (simpler) quantization.
    granularity: Granularity,
}

pub enum Granularity {
    PerTensor,
    PerChannel,
}

impl Pass for Int8QuantizePass {
    fn name(&self) -> &str { "int8-quantization" }
    fn run(&self, program: &mut Program) -> Result<()> {
        // 1. For each const op with FP32 tensor data (weights):
        //    a. Compute min/max values
        //    b. Calculate scale = (max - min) / 255, zero_point = round(-min / scale)
        //    c. Quantize: q = clamp(round(x / scale) + zero_point, 0, 255)
        //    d. Replace tensor data with INT8 bytes
        //    e. Store scale/zero_point as op attributes
        // 2. Add quantize/dequantize ops around non-const inputs:
        //    a. Insert "quantize" op before consumer
        //    b. Insert "dequantize" op after producer
        //    c. (Calibration data determines activation ranges)
    }
}
```

### Design

- **Weight-only quantization first** (simpler, no calibration needed for weights):
  Just quantize const tensors based on their min/max values.
- **Activation quantization** (needs calibration): Read calibration tensors from
  `--cal-data` directory, run them through the graph to collect activation ranges,
  then insert quantize/dequantize nodes.
- **Phase approach**: Implement weight-only first. Activation quantization as a follow-up.
- **Mutual exclusivity**: Error if both `--quantize fp16` and `--quantize int8` are specified.

### CLI

```
coreml-kit compile model.onnx --quantize int8                    # weight-only INT8
coreml-kit compile model.onnx --quantize int8 --cal-data imgs/   # full INT8 with calibration
```

### Tests

- Quantize FP32 weights to INT8 → verify scale/zero_point attributes added
- Verify quantized values are within [0, 255] range
- Verify FP16+INT8 mutual exclusivity error
- Verify INT8+palettize mutual exclusivity error
- Round-trip: quantize then dequantize → verify within tolerance of original

---

## Task 6: Weight Palettization

**Goal**: Compress weights using k-means clustering into lookup tables,
dramatically reducing model size.

**File**: `crates/mil-rs/src/ir/passes/palettize.rs`

### Background

Palettization clusters weight values into 2^n centroids (e.g., 16 for 4-bit,
64 for 6-bit). Each weight is replaced by an index into the palette. CoreML
supports this natively via `constexpr_lut_to_dense` ops.

### Implementation

```rust
/// Compress weights using k-means palettization.
///
/// Each weight tensor is clustered into 2^n_bits centroids. The tensor is
/// then stored as indices + a palette lookup table, reducing size by
/// 32/n_bits × (roughly 8× for 4-bit, ~5× for 6-bit).
///
/// Mutually exclusive with `Int8QuantizePass`.
pub struct PalettizePass {
    /// Number of bits per weight (2, 4, 6, or 8).
    n_bits: u8,
    /// Maximum k-means iterations.
    max_iter: usize,
}

impl PalettizePass {
    pub fn new(n_bits: u8) -> Self {
        assert!(matches!(n_bits, 2 | 4 | 6 | 8));
        Self { n_bits, max_iter: 100 }
    }
}

impl Pass for PalettizePass {
    fn name(&self) -> &str { "palettization" }
    fn run(&self, program: &mut Program) -> Result<()> {
        // For each const op with FP32/FP16 tensor data:
        // 1. Extract weight values as f32
        // 2. Run k-means with k = 2^n_bits centroids
        // 3. Replace weight data with:
        //    a. Palette: Vec<f32> of k centroids
        //    b. Indices: Vec<u8> of packed n-bit indices
        // 4. Change op from "const" to "constexpr_lut_to_dense"
        //    with attributes: lut (palette), indices, shape
    }
}
```

### K-means implementation

- Implement simple k-means in `crates/mil-rs/src/ir/passes/kmeans.rs`:
  - Lloyd's algorithm: init → assign → update → repeat
  - Use kmeans++ initialization for better convergence
  - Convergence: stop when assignments don't change or max_iter reached
- Keep it self-contained (no external ML deps)

### Tests

- Palettize a small tensor with 4-bit → verify 16 centroids, correct indices
- Palettize with 2-bit → verify 4 centroids
- Verify compression ratio (output size < input size)
- Verify INT8+palettize mutual exclusivity error
- Verify FP16+palettize is allowed (they compress different things)
- K-means unit tests: convergence, edge cases (all-same values, k > n_values)

---

## Task 7: Pipeline Manager and CLI Integration

**Goal**: Unify pass management with a `PassPipeline` that handles ordering,
mutual exclusivity, and auto-detection.

**File**: `crates/mil-rs/src/ir/pipeline.rs`

### Implementation

```rust
/// A configured optimization pipeline.
///
/// Manages pass ordering, mutual exclusivity checks, and pass selection
/// based on model characteristics and user flags.
pub struct PassPipeline {
    passes: Vec<Box<dyn Pass>>,
}

impl PassPipeline {
    /// Create the default pipeline with all always-on passes.
    pub fn default() -> Self { ... }

    /// Add FP16 quantization.
    pub fn with_fp16(mut self) -> Self { ... }

    /// Add INT8 quantization. Errors if FP16 is already added.
    pub fn with_int8(mut self, cal_dir: Option<PathBuf>) -> Result<Self> { ... }

    /// Add weight palettization. Errors if INT8 is already added.
    pub fn with_palettize(mut self, n_bits: u8) -> Result<Self> { ... }

    /// Add shape materialization with user-provided shapes.
    pub fn with_shapes(mut self, shapes: HashMap<String, Vec<usize>>) -> Self { ... }

    /// Disable fusion passes (for debugging).
    pub fn without_fusion(mut self) -> Self { ... }

    /// Run the full pipeline, returning a report of what each pass did.
    pub fn run(self, program: &mut Program) -> Result<PipelineReport> { ... }
}

pub struct PipelineReport {
    pub pass_results: Vec<PassResult>,
}

pub struct PassResult {
    pub name: String,
    pub ops_before: usize,
    pub ops_after: usize,
}
```

### CLI Updates

```
--quantize <fp16|int8>     Quantization mode (mutually exclusive)
--cal-data <dir>           Calibration data directory (for int8)
--palettize <bits>         Weight palettization (2, 4, 6, or 8)
--no-fusion                Disable fusion passes
--input-shape <spec>       Shape materialization (existing)
```

### Tests

- Default pipeline includes all always-on passes in correct order
- FP16 + INT8 returns error
- INT8 + palettize returns error
- FP16 + palettize is allowed
- Pipeline report counts ops correctly

---

## Dependency Graph

```
Task 1 (BN Weight Fold)     ──┐
Task 2 (Attention Fusion)   ──┤
Task 3 (Op Substitution)    ──┼──▶ Task 7 (Pipeline Manager + CLI)
Task 4 (Layout Optimization)──┤
Task 5 (INT8 Quantization)  ──┤
Task 6 (Palettization)      ──┘
```

Tasks 1–6 are independent and can be built in any order (or in parallel).
Task 7 integrates them all and must come last.

## Suggested Order

1. **Task 1** — Conv+BN weight fold (builds on existing fusion infra, highest perf impact)
2. **Task 3** — Op substitution (relatively simple, immediate ANE compatibility improvement)
3. **Task 2** — Attention fusion (high value for transformer models)
4. **Task 4** — Layout optimization (complex but important for CNN models)
5. **Task 5** — INT8 quantization (weight-only first, then calibration)
6. **Task 6** — Palettization (most complex — k-means impl)
7. **Task 7** — Pipeline manager + CLI integration
