# MIL Op Parity Plan — coremltools Coverage

ironmill currently supports ~55 unique MIL op types via ~62 ONNX→MIL conversion
handlers. Apple's `coremltools` defines ~170 unique MIL ops across the
iOS15–iOS18 opsets. This document tracks the ~108 missing ops and the
implementation plan to reach full parity.

For each op the work is:

1. **IR support** — `ir_to_proto` serializes it; `proto_to_ir` deserializes it
2. **ONNX converter** — add a handler in `onnx_to_mil.rs` (when an ONNX
   equivalent exists)
3. **Validation** — type / shape inference rules in the validator
4. **Tests** — round-trip serialization + ONNX conversion correctness

---

## Current ironmill MIL ops (~55 unique)

```
add, sub, mul, real_div, pow, sqrt, abs, ceil, floor, erf, exp, clip, neg,
reciprocal, log, sin, cos, relu, sigmoid, tanh, silu, gelu, softmax,
matmul, linear, layer_norm, batch_norm, conv, conv_transpose,
reshape, transpose, concat, split, squeeze, expand_dims, slice_by_index,
tile, reverse, identity, pad, shape, upsample_bilinear,
gather, scatter, select, cast, const, reduce_mean, reduce_sum, cumsum,
avg_pool, max_pool, equal, less, greater, logical_not,
codebook_gather, constexpr_affine_dequantize, constexpr_lut_to_dense,
rms_norm, scaled_dot_product_attention, grouped_query_attention,
kv_cache_read, kv_cache_update, repeat_interleave, dequantize
```

---

## Phase 1 — Elementwise Arithmetic & Comparison Gaps (23 ops) · Critical

These are building blocks used in nearly every model. Many are thin wrappers
with straightforward semantics.

### Elementwise Binary (10 ops)

| Op | Opset | ONNX | Notes |
|----|-------|------|-------|
| `floor_div` | iOS15 | `Div` + `Floor` | integer division |
| `maximum` | iOS15 | `Max` | elementwise max |
| `minimum` | iOS15 | `Min` | elementwise min |
| `mod` | iOS15 | `Mod` | modulo |
| `not_equal` | iOS15 | `Not` + `Equal` | comparison |
| `greater_equal` | iOS15 | `GreaterOrEqual` | comparison |
| `less_equal` | iOS15 | `LessOrEqual` | comparison |
| `logical_and` | iOS15 | `And` | boolean logic |
| `logical_or` | iOS15 | `Or` | boolean logic |
| `logical_xor` | iOS15 | `Xor` | boolean logic |

### Elementwise Unary (13 ops)

| Op | Opset | ONNX | Notes |
|----|-------|------|-------|
| `acos` | iOS15 | `Acos` | trig |
| `asin` | iOS15 | `Asin` | trig |
| `atanh` | iOS15 | `Atanh` | hyperbolic |
| `cosh` | iOS15 | `Cosh` | hyperbolic |
| `exp2` | iOS15 | — | decompose: `exp(x * ln2)` |
| `inverse` | iOS15 | `Reciprocal` | 1/x (alias of reciprocal) |
| `round` | iOS15 | `Round` | round to nearest |
| `rsqrt` | iOS15 | — | decompose: `pow(x, -0.5)` |
| `sign` | iOS15 | `Sign` | signum |
| `sinh` | iOS15 | `Sinh` | hyperbolic |
| `square` | iOS15 | — | decompose: `mul(x, x)` |
| `tan` | iOS15 | `Tan` | trig |
| `threshold` | iOS15 | — | `x if x > α else α` |

---

## Phase 2 — Activation Functions (12 ops) · Critical

Many legacy and specialized CNN / GAN architectures depend on these.

| Op | Opset | ONNX | Notes |
|----|-------|------|-------|
| `clamped_relu` | iOS15 | `Clip` + `Relu` | relu capped at α, β |
| `elu` | iOS15 | `Elu` | exponential linear unit |
| `leaky_relu` | iOS15 | `LeakyRelu` | standard |
| `linear_activation` | iOS15 | — | `α * x + β` |
| `prelu` | iOS15 | `PRelu` | parametric relu |
| `relu6` | iOS15 | — | decompose: `clip(relu(x), 0, 6)` |
| `scaled_tanh` | iOS15 | — | `α * tanh(β * x)` |
| `sigmoid_hard` | iOS15 | `HardSigmoid` | piecewise linear approx |
| `softplus` | iOS15 | `Softplus` | `log(1 + exp(x))` |
| `softplus_parametric` | iOS15 | — | `α * log(1 + exp(β * x))` |
| `softsign` | iOS15 | `Softsign` | `x / (1 + |x|)` |
| `thresholded_relu` | iOS15 | `ThresholdedRelu` | `x if x > α else 0` |

---

## Phase 3 — Reduction Ops (10 ops) · Critical

Used in loss functions, normalization, and attention layers.

| Op | Opset | ONNX | Notes |
|----|-------|------|-------|
| `reduce_argmax` | iOS15 | `ArgMax` | index of max |
| `reduce_argmin` | iOS15 | `ArgMin` | index of min |
| `reduce_max` | iOS15 | `ReduceMax` | max reduction |
| `reduce_min` | iOS15 | `ReduceMin` | min reduction |
| `reduce_prod` | iOS15 | `ReduceProd` | product reduction |
| `reduce_l1_norm` | iOS15 | — | L1 norm reduction |
| `reduce_l2_norm` | iOS15 | — | L2 norm reduction |
| `reduce_log_sum` | iOS15 | `ReduceLogSum` | log of sum |
| `reduce_log_sum_exp` | iOS15 | `ReduceLogSumExp` | log-sum-exp |
| `reduce_sum_square` | iOS15 | `ReduceSumSquare` | sum of squares |

---

## Phase 4 — Scatter / Gather Variants (4 ops) · High

Critical for embedding lookups and advanced indexing patterns.

| Op | Opset | ONNX | Notes |
|----|-------|------|-------|
| `gather_along_axis` | iOS15 | `GatherElements` | per-element gather |
| `gather_nd` | iOS15 | `GatherND` | N-dimensional gather |
| `scatter_along_axis` | iOS15 | `ScatterElements` | per-element scatter |
| `scatter_nd` | iOS15 | `ScatterND` | N-dimensional scatter |

---

## Phase 5 — Tensor Manipulation (11 ops) · High

Used in model reshaping, data preparation, and dynamic shape handling.

| Op | Opset | ONNX | Notes |
|----|-------|------|-------|
| `slice_by_size` | iOS15 | — | slice with size (not end index) |
| `stack` | iOS15 | — | stack tensors along new axis |
| `argsort` | iOS15 | — | sort indices |
| `topk` | iOS15 | `TopK` | top-k values + indices |
| `fill` | iOS15 | `ConstantOfShape` | fill tensor with scalar |
| `fill_like` | iOS16 | — | fill with matching shape |
| `flatten2d` | iOS15 | `Flatten` | flatten to 2D |
| `non_zero` | iOS15 | `NonZero` | indices of non-zero elems |
| `one_hot` | iOS15 | `OneHot` | one-hot encoding |
| `range_1d` | iOS15 | `Range` | arange-style sequence |
| `band_part` | iOS15 | — | triangular mask |

---

## Phase 6 — Tensor Transformation (9 ops) · Medium-High

Spatial rearrangement ops for vision and diffusion models.

| Op | Opset | ONNX | Notes |
|----|-------|------|-------|
| `depth_to_space` | iOS15 | `DepthToSpace` | depth → spatial |
| `space_to_depth` | iOS15 | `SpaceToDepth` | spatial → depth |
| `pixel_shuffle` | iOS15 | — | sub-pixel convolution |
| `pixel_unshuffle` | iOS16 | — | inverse pixel shuffle |
| `reshape_like` | iOS16 | — | reshape to match another tensor |
| `reverse_sequence` | iOS15 | `ReverseSequence` | reverse along seq axis |
| `sliding_windows` | iOS15 | — | extract sliding windows |
| `space_to_batch` | iOS15 | — | spatial → batch |
| `slice_update` | iOS18 | — | in-place slice update |

---

## Phase 7 — Image Resizing (8 ops) · Medium

Required for vision and image-generation models.

| Op | Opset | ONNX | Notes |
|----|-------|------|-------|
| `affine` | iOS15 | — | affine spatial transform |
| `crop` | iOS15 | — | spatial cropping |
| `crop_resize` | iOS15 | — | crop + resize (RoIAlign-like) |
| `resample` | iOS15 | — | grid-based resampling |
| `resize` | iOS17 | `Resize` | unified resize (replaces bilinear / nearest) |
| `resize_bilinear` | iOS15 | — | bilinear interpolation |
| `resize_nearest_neighbor` | iOS15 | — | nearest-neighbor interpolation |
| `upsample_nearest_neighbor` | iOS15 | — | nearest-neighbor upsample |

---

## Phase 8 — Normalization, Pooling & Linear Algebra (5 ops) · Medium

Completes the normalization and pooling families.

| Op | Opset | ONNX | Notes |
|----|-------|------|-------|
| `instance_norm` | iOS15 | `InstanceNormalization` | per-instance normalization |
| `l2_norm` | iOS15 | `LpNormalization` | L2 normalization |
| `local_response_norm` | iOS15 | `LRN` | LRN (legacy CNNs) |
| `l2_pool` | iOS15 | `LpPool` | L2 pooling |
| `einsum` | iOS15 | `Einsum` | einstein summation |

---

## Phase 9 — Recurrent & Random (7 ops) · Medium

RNNs are less common in modern transformer models but still needed for
sequence-to-sequence and speech workloads.

| Op | Opset | ONNX | Notes |
|----|-------|------|-------|
| `lstm` | iOS15 | `LSTM` | long short-term memory |
| `gru` | iOS15 | `GRU` | gated recurrent unit |
| `rnn` | iOS15 | `RNN` | simple recurrent |
| `random_bernoulli` | iOS15 | `Bernoulli` | Bernoulli sampling |
| `random_categorical` | iOS15 | `Multinomial` | categorical sampling |
| `random_normal` | iOS15 | `RandomNormal` | normal distribution |
| `random_uniform` | iOS15 | `RandomUniform` | uniform distribution |

---

## Phase 10 — Control Flow (8 ops) · Medium-Low

Needed for dynamic models with loops and conditionals (e.g. beam search,
autoregressive decoding with early stopping).

| Op | Opset | ONNX | Notes |
|----|-------|------|-------|
| `cond` | iOS15 | `If` | conditional execution |
| `while_loop` | iOS15 | `Loop` | loop construct |
| `make_list` | iOS15 | — | create dynamic list |
| `list_read` | iOS15 | — | read from list |
| `list_write` | iOS15 | — | write to list |
| `list_gather` | iOS15 | — | gather from list |
| `list_scatter` | iOS15 | — | scatter into list |
| `list_length` | iOS15 | — | list length |

---

## Phase 11 — Constexpr & Quantization (6 ops) · Medium

Weight compression ops for iOS 16–18. ironmill already supports
`constexpr_affine_dequantize`, `constexpr_lut_to_dense`, and `dequantize`.

| Op | Opset | ONNX | Notes |
|----|-------|------|-------|
| `quantize` | iOS17 | `QuantizeLinear` | activation quantization |
| `constexpr_cast` | iOS16 | — | compile-time type cast |
| `constexpr_sparse_to_dense` | iOS16 | — | sparse weight decompression |
| `constexpr_blockwise_shift_scale` | iOS18 | — | blockwise quantization |
| `constexpr_lut_to_sparse` | iOS18 | — | LUT + sparsity |
| `constexpr_sparse_blockwise_shift_scale` | iOS18 | — | sparse blockwise quant |

---

## Phase 12 — Domain-Specific & Misc (4 ops) · Low

Specialized ops for classification, object detection, and stateful models.

| Op | Opset | ONNX | Notes |
|----|-------|------|-------|
| `classify` | iOS15 | — | classification label assignment |
| `non_maximum_suppression` | iOS15 | `NonMaxSuppression` | object detection NMS |
| `conv_quantized` | iOS15 | — | quantized convolution variant |
| `read_state` | iOS18 | — | stateful model support |

---

## Summary

| Phase | Category | Ops | Priority |
|-------|----------|-----|----------|
| 1 | Elementwise gaps | 23 | Critical |
| 2 | Activations | 12 | Critical |
| 3 | Reductions | 10 | Critical |
| 4 | Scatter / gather variants | 4 | High |
| 5 | Tensor manipulation | 11 | High |
| 6 | Tensor transformation | 9 | Medium-High |
| 7 | Image resizing | 8 | Medium |
| 8 | Norm / pool / linear algebra | 5 | Medium |
| 9 | Recurrent & random | 7 | Medium |
| 10 | Control flow | 8 | Medium-Low |
| 11 | Constexpr & quantization | 6 | Medium |
| 12 | Domain-specific | 4 | Low |
| **Total** | | **107** | |

---

## Key Files

| File | Role |
|------|------|
| `crates/mil-rs/src/convert/onnx_to_mil.rs` | ONNX op handlers |
| `crates/mil-rs/src/convert/ir_to_proto.rs` | MIL → protobuf serialization |
| `crates/mil-rs/src/convert/proto_to_ir.rs` | protobuf → MIL deserialization |
| `crates/mil-rs/src/validate.rs` | type / shape inference rules |
| `crates/mil-rs/src/ir/operation.rs` | `Operation` struct and attribute handling |

---

## Decomposition Strategy

Some ops can be decomposed into existing primitives rather than adding native
support. Whether to decompose or implement natively is a case-by-case decision —
native ops may receive better ANE scheduling.

| Op | Decomposition |
|----|---------------|
| `inverse(x)` | `real_div(1, x)` |
| `rsqrt(x)` | `pow(x, -0.5)` |
| `square(x)` | `mul(x, x)` |
| `exp2(x)` | `exp(mul(x, ln2))` |
| `relu6(x)` | `clip(relu(x), 0, 6)` |
| `clamped_relu(x)` | `clip(relu(x), α, β)` |
| `linear_activation(x)` | `add(mul(x, α), β)` |
| `softsign(x)` | `real_div(x, add(1, abs(x)))` |
| `neg(x)` | `sub(0, x)` or `mul(x, -1)` |
