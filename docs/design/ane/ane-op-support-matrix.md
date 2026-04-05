# ANE Op Support Matrix

MIL ops verified on Apple's Neural Engine via the private `_ANEInMemoryModel`
compiler. Each ✅ eval op has a test in
[`ane_op_eval.rs`](../../crates/ironmill-inference/examples/ane_op_eval.rs)
that compiles a minimal MIL program, runs it on real ANE hardware, and compares
the output against a CPU reference.

Run with: `cargo run -p ironmill-inference --example ane_op_eval`

> **Methodology:** Tests use `[1,32,1,32]` tensors in ANE-native `[1,C,1,S]`
> NCHW layout with the `program(1.3)` / `func main<ios18>` MIL text format.
> Results may vary by macOS version and chip generation.

## Novel Contributions

This project builds on [prior art](#related-projects) by discovering 38 ops not
verified by any other open-source ANE project (marked 🆕 below), including 33
with full eval verification. Key findings:

- **`rsqrt`, `log`, `inverse`** - all require an undocumented `epsilon` parameter;
  without it the compiler silently rejects them. Previously believed unsupported.
- **`layer_norm`** - all other projects perform normalization on CPU.
- **`erf`** - enables on-ANE GELU without tanh decomposition.
- **All 6 comparison ops** and **`select`/`logical_not`** - enables conditional
  logic entirely on ANE.
- **`quantize`/`dequantize`** - enables full INT8 KV cache pipelines on ANE.
- **8 additional activation/reduction/elementwise ops** discovered via name fuzzing.

Every discovery has a reproducible eval test in
[`ane_op_eval.rs`](../../crates/ironmill-inference/examples/ane_op_eval.rs).

## Ops

Status legend: **✅ eval** = compiled + executed + numerically verified on ANE, **⚠️ compile** = compiles but not eval-verified, **❌** = compiler rejects.

### Arithmetic & Elementwise

| Op | Status | max_err | 🆕 | Eval | Notes |
|---|---|---|---|---|---|
| `add` | ✅ eval | 0.038 | | [test_add](../../crates/ironmill-inference/examples/ane_op_eval.rs#L196) | Binary elementwise |
| `sub` | ✅ eval | 0.10 | | [test_sub](../../crates/ironmill-inference/examples/ane_op_eval.rs#L212) | fp16 rounding at large values |
| `mul` | ✅ eval | 0.050 | | [test_mul](../../crates/ironmill-inference/examples/ane_op_eval.rs#L228) | Binary elementwise |
| `real_div` | ✅ eval | 0.063 | | [test_real_div](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1480) | Binary elementwise |
| `maximum` | ✅ eval | 0.025 | | [test_maximum](../../crates/ironmill-inference/examples/ane_op_eval.rs#L555) | Binary elementwise |
| `minimum` | ✅ eval | 0.013 | | [test_minimum](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1496) | Binary elementwise |
| `floor_div` | ✅ eval | exact | | [test_floor_div](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1511) | Binary elementwise |
| `abs` | ✅ eval | exact | | [test_abs](../../crates/ironmill-inference/examples/ane_op_eval.rs#L274) | Unary |
| `sign` | ✅ eval | exact | 🆕 | [test_sign](../../crates/ironmill-inference/examples/ane_op_eval.rs#L289) | `sign(0) = 0` (mathematically correct) |
| `sqrt` | ✅ eval | 0.008 | | [test_sqrt](../../crates/ironmill-inference/examples/ane_op_eval.rs#L326) | Unary |
| `square` | ✅ eval | 0.025 | 🆕 | [test_square](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1263) | Unary |
| `exp` | ✅ eval | 0.43 | | [test_exp](../../crates/ironmill-inference/examples/ane_op_eval.rs#L341) | fp16 range limits |
| `exp2` | ✅ eval | 0.047 | 🆕 | [test_exp2](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1172) | Unary |
| `erf` | ✅ eval | 0.001 | 🆕 | [test_erf](../../crates/ironmill-inference/examples/ane_op_eval.rs#L571) | Useful for GELU |
| `ceil` | ✅ eval | exact | 🆕 | [test_ceil](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1278) | Unary |
| `floor` | ✅ eval | exact | 🆕 | [test_floor](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1295) | Unary |
| `round` | ✅ eval | exact | 🆕 | [test_round](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1312) | Used in INT8 quantization |
| `atan` | ✅ eval | 0.033 | 🆕 | [test_atan](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1405) | Only trig function supported |
| `pow` | ✅ eval | 0.0004 | | [test_pow](../../crates/ironmill-inference/examples/ane_op_eval.rs#L484) | Scalar const `y` arg |
| `clip` | ✅ eval | 0.0002 | | [test_clip](../../crates/ironmill-inference/examples/ane_op_eval.rs#L536) | Scalar const `alpha`/`beta` args |
| `rsqrt` | ✅ eval | 0.0004 | 🆕 | [test_rsqrt](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1205) | **Requires `epsilon` parameter** |
| `log` | ✅ eval | 0.005 | 🆕 | [test_log](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1187) | **Requires `epsilon` parameter** |
| `inverse` | ✅ eval | 0.001 | 🆕 | [test_inverse](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1222) | **Requires `epsilon` parameter** |
| `neg` | ❌ | - | | [test_neg](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1249) | Workaround: `mul(x, -1)` or `sub(0, x)` |
| `mod` | ❌ | - | | [test_mod](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1530) | Workaround: `sub(x, mul(floor_div(x, y), y))` |
| `sin` | ❌ | - | | [test_sin](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1329) | CPU fallback or polynomial approx |
| `cos` | ❌ | - | | [test_cos](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1343) | CPU fallback or polynomial approx |
| `tan` | ❌ | - | | [test_tan](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1357) | CPU fallback |
| `asin` | ❌ | - | | [test_asin](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1372) | CPU fallback |
| `acos` | ❌ | - | | [test_acos](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1389) | CPU fallback |
| `sinh` | ❌ | - | | [test_sinh](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1419) | Workaround: `(exp(x) - exp(-x)) / 2` |
| `cosh` | ❌ | - | | [test_cosh](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1434) | Workaround: `(exp(x) + exp(-x)) / 2` |

### Activations

| Op | Status | max_err | 🆕 | Eval | Notes |
|---|---|---|---|---|---|
| `relu` | ✅ eval | exact | | [test_relu](../../crates/ironmill-inference/examples/ane_op_eval.rs#L259) | |
| `relu6` | ⚠️ compile | - | 🆕 | - | Discovered via fuzzing |
| `sigmoid` | ✅ eval | 0.003 | | [test_sigmoid](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1144) | |
| `tanh` | ✅ eval | 0.002 | | [test_tanh](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1158) | |
| `softmax` | ✅ eval | 0.0002 | | [test_softmax](../../crates/ironmill-inference/examples/ane_op_eval.rs#L503) | Scalar const `axis` arg |
| `silu` | ✅ eval | 0.015 | 🆕 | [test_silu](../../crates/ironmill-inference/examples/ane_op_eval.rs#L814) | SiLU/Swish |
| `softsign` | ✅ eval | 0.001 | 🆕 | [test_softsign](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1448) | |
| `softplus` | ✅ eval | 0.006 | 🆕 | [test_softplus](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1463) | |

### Comparison

| Op | Status | max_err | 🆕 | Eval | Notes |
|---|---|---|---|---|---|
| `greater` | ✅ eval | exact | 🆕 | [test_greater](../../crates/ironmill-inference/examples/ane_op_eval.rs#L357) | Returns bool tensor |
| `greater_equal` | ✅ eval | exact | 🆕 | [test_greater_equal](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1552) | |
| `less` | ✅ eval | exact | 🆕 | [test_less](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1575) | |
| `less_equal` | ✅ eval | exact | 🆕 | [test_less_equal](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1595) | |
| `equal` | ✅ eval | exact | 🆕 | [test_equal](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1615) | |
| `not_equal` | ✅ eval | exact | 🆕 | [test_not_equal](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1635) | |

### Reductions

| Op | Status | max_err | 🆕 | Eval | Notes |
|---|---|---|---|---|---|
| `reduce_sum` | ✅ eval | 0.013 | | [test_reduce_sum](../../crates/ironmill-inference/examples/ane_op_eval.rs#L400) | Axis -1 and axis 1 ([axis1](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1719)) |
| `reduce_mean` | ✅ eval | 0.0004 | | [test_reduce_mean](../../crates/ironmill-inference/examples/ane_op_eval.rs#L429) | Axis -1 and axis 1 ([axis1](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1746)) |
| `reduce_max` | ✅ eval | exact | | [test_reduce_max](../../crates/ironmill-inference/examples/ane_op_eval.rs#L455) | Axis -1 and axis 1 ([axis1](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1773)) |
| `reduce_min` | ✅ eval | exact | 🆕 | [test_reduce_min](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1657) | Axis -1 and axis 1 ([axis1](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1804)) |
| `reduce_l2_norm` | ✅ eval | 0.0002 | 🆕 | [test_reduce_l2_norm](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1685) | Axis -1 |
| `reduce_log_sum_exp` | ✅ eval | 0.001 | 🆕 | [test_reduce_log_sum_exp](../../crates/ironmill-inference/examples/ane_op_eval.rs#L830) | Useful for log-softmax |
| `reduce_l1_norm` | ⚠️ compile | - | 🆕 | - | Discovered via fuzzing |
| `reduce_log_sum` | ⚠️ compile | - | 🆕 | - | Discovered via fuzzing |
| `reduce_sum_square` | ⚠️ compile | - | 🆕 | - | Discovered via fuzzing |
| `reduce_prod` | ❌ | - | | - | Workaround: `exp(reduce_sum(log(x)))` or iterative mul |

### Linear Algebra

| Op | Status | max_err | 🆕 | Eval | Notes |
|---|---|---|---|---|---|
| `matmul` | ✅ eval | 0.004 | | [test_matmul](../../crates/ironmill-inference/examples/ane_op_eval.rs#L700) | With `transpose_x`/`transpose_y` bool consts |
| `layer_norm` | ✅ eval | 0.001 | 🆕 | [test_layer_norm](../../crates/ironmill-inference/examples/ane_op_eval.rs#L762) | With const `axes` and `epsilon` |
| `batch_norm` | ❌ | - | | - | May need weight blob args; or decompose |

### Shape / Tensor Manipulation

| Op | Status | max_err | 🆕 | Eval | Notes |
|---|---|---|---|---|---|
| `reshape` | ✅ eval | 0.014† | | [test_turboquant_int8_cache_pipeline](../../crates/ironmill-inference/examples/ane_op_eval.rs#L939) | †Cannot compile standalone; verified as intermediate op |
| `concat` | ✅ eval | 0.025 | | [test_concat](../../crates/ironmill-inference/examples/ane_op_eval.rs#L793) | With const `axis` and `interleave` |
| `slice_by_index` | ✅ eval | 0.014† | | [test_turboquant_int8_cache_pipeline](../../crates/ironmill-inference/examples/ane_op_eval.rs#L939) | †Cannot compile standalone; verified as intermediate op |
| `tile` | ✅ eval | 0.014† | | [test_turboquant_int8_cache_pipeline](../../crates/ironmill-inference/examples/ane_op_eval.rs#L939) | †Cannot compile standalone; verified as intermediate op |
| `identity` | ✅ eval | exact | | [test_identity](../../crates/ironmill-inference/examples/ane_op_eval.rs#L244) | Pass-through |
| `transpose` | ⚠️ compile | - | | - | With const `perm` |
| `slice_by_size` | ⚠️ compile | - | | - | With const `begin`/`size` |
| `split` | ⚠️ compile | - | | - | With const `num_splits` and `axis` |
| `expand_dims` | ⚠️ compile | - | | - | With const `axes` |
| `squeeze` | ⚠️ compile | - | | - | With const `axes` |
| `reverse` | ⚠️ compile | - | 🆕 | - | Discovered via fuzzing |

### Conditional

| Op | Status | max_err | 🆕 | Eval | Notes |
|---|---|---|---|---|---|
| `select` | ✅ eval | exact | 🆕 | [test_select](../../crates/ironmill-inference/examples/ane_op_eval.rs#L378) | `select(cond, true_val, false_val)` |
| `logical_not` | ✅ eval | exact | 🆕 | [test_logical_not](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1837) | Unary on bool tensor |
| `logical_and` | ❌ | - | | [test_logical_and](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1859) | Workaround: `mul(cast(a, fp16), cast(b, fp16))` |
| `logical_or` | ❌ | - | | [test_logical_or](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1881) | Workaround: `maximum(cast(a, fp16), cast(b, fp16))` |
| `logical_xor` | ❌ | - | | [test_logical_xor](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1903) | Workaround: `not_equal` on cast-to-fp16 |
| `where` | ❌ | - | | - | Use `select` (same semantics) |

### Type Casting

| Op | Status | max_err | 🆕 | Eval | Notes |
|---|---|---|---|---|---|
| `cast fp16→bool` | ✅ eval | exact | 🆕 | [test_cast_fp16_bool](../../crates/ironmill-inference/examples/ane_op_eval.rs#L731) | Nonzero→true, zero→false |
| `cast bool→fp16` | ✅ eval | exact | 🆕 | [test_cast_fp16_bool](../../crates/ironmill-inference/examples/ane_op_eval.rs#L731) | true→1.0, false→0.0 |
| `cast fp16→int8` | ✅ eval | exact | 🆕 | [test_int8_round_trip](../../crates/ironmill-inference/examples/ane_op_eval.rs#L871) | Truncates to [-128, 127] |
| `cast int8→fp16` | ✅ eval | exact | 🆕 | [test_int8_round_trip](../../crates/ironmill-inference/examples/ane_op_eval.rs#L871) | INT8 round-trip verified |
| `dequantize` | ✅ eval | exact | 🆕 | [test_int8_quantize_dequantize](../../crates/ironmill-inference/examples/ane_op_eval.rs#L898) | INT8/UINT8 with scale/zero_point |
| `quantize` | ✅ eval | exact | 🆕 | [test_quantize_standalone](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1955) | Standalone quantize→dequantize verified |
| `cast fp16→fp32` | ⚠️ compile | - | | - | |
| `cast fp32→fp16` | ⚠️ compile | - | | - | |
| `cast fp16→uint8` | ⚠️ compile | - | | - | |
| `cast uint8→fp16` | ⚠️ compile | - | | - | |
| `cast fp16→int16` | ⚠️ compile | - | | - | |
| `cast int16→fp16` | ⚠️ compile | - | | - | |
| `cast fp16→int32` | ❌ | - | | - | Use int8/int16 path |
| `cast int32→fp16` | ❌ | - | | - | Use int8/int16 path |
| INT4/UINT4 (all paths) | ❌ | - | | - | Comprehensively rejected |

### Spatial

| Op | Status | max_err | 🆕 | Eval | Notes |
|---|---|---|---|---|---|
| `pad` | ✅ eval | 0.025 | 🆕 | [test_pad_constant](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1932) | Constant mode |
| `avg_pool` | ❌ | - | | - | Use `reduce_mean` |
| `max_pool` | ❌ | - | | - | Use `reduce_max` |

### Index / Scatter

| Op | Status | max_err | 🆕 | Eval | Notes |
|---|---|---|---|---|---|
| `scatter` | ❌ | - | | - | CPU interception at sub-program boundary |
| `scatter_nd` | ❌ | - | | - | CPU interception at sub-program boundary |
| `scatter_along_axis` | ❌ | - | | - | CPU interception at sub-program boundary |
| `gather` | ❌ | - | | - | Static `constexpr_lut_to_dense` works; only runtime/dynamic gather fails |

## Composite Pattern Tests

These tests verify multi-op pipelines that matter for real model inference:

| Pattern | max_err | Eval |
|---|---|---|
| QJL sign extraction | exact | [test_qjl_sign_extraction](../../crates/ironmill-inference/examples/ane_op_eval.rs#L601) |
| RMSNorm | 0.002 | [test_rmsnorm_pattern](../../crates/ironmill-inference/examples/ane_op_eval.rs#L633) |
| Affine quantize | - | [test_affine_quantize_pattern](../../crates/ironmill-inference/examples/ane_op_eval.rs#L670) |
| INT8 round-trip | exact | [test_int8_round_trip](../../crates/ironmill-inference/examples/ane_op_eval.rs#L871) |
| INT8 quant→dequant | 0.006 | [test_int8_quantize_dequantize](../../crates/ironmill-inference/examples/ane_op_eval.rs#L898) |
| TurboQuant INT8 cache pipeline | 0.014 | [test_turboquant_int8_cache_pipeline](../../crates/ironmill-inference/examples/ane_op_eval.rs#L939) |
| Generated cache-write MIL | - | [test_generated_cache_write_mil](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1033) |
| Generated attention MIL | - | [test_generated_attention_mil](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1056) |
| Generated QJL MIL | - | [test_generated_qjl_mil](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1086) |
| Generated GQA attention MIL | - | [test_generated_attention_gqa_mil](../../crates/ironmill-inference/examples/ane_op_eval.rs#L1109) |
| RoPE (precomputed sin/cos) | - | [test_rope_precomputed_sincos](../../crates/ironmill-inference/examples/ane_op_eval.rs#L2041) |

## Data Type Support

Verified via `ane_dtype_probe`:

- **Inputs:** only `fp16`, `fp32`, and `bool` accepted as function inputs
- **INT8 storage:** `cast fp16→int8` ✅, `cast int8→fp16` ✅, `dequantize(int8)` ✅
- **INT4/UINT4:** ❌ rejected across all paths (inputs, casts, arithmetic, matmul, dequantize)
- **Arithmetic:** only on `fp16`/`fp32` - integer arithmetic (`add`/`mul` on int8) rejected
- **INT8 is a storage format, not a compute format** - quantize to INT8 for bandwidth, cast back to fp16 for arithmetic

## Notes

### The epsilon discovery

`rsqrt`, `log`, and `inverse` all **require an explicit `epsilon` parameter** to
compile - standard MIL treats it as optional but the ANE compiler rejects
programs without it. This was the single most impactful finding from ironmill's
op verification, recovering 3 ops that were previously believed unsupported.

### Re-running probes

```sh
# Eval-time verification (individual ops + composites + INT8 pipeline)
cargo run -p ironmill-inference --example ane_op_eval

# Compile-time probe
cargo run -p ironmill-ane --example ane_op_probe

# Name fuzzing / discovery (400+ variants)
cargo run -p ironmill-ane --example ane_op_fuzz

# Data type probe
cargo run -p ironmill-ane --example ane_dtype_probe
```

## Related Projects

Other open-source projects working with the ANE via private APIs:

- [maderix/ANE](https://github.com/maderix/ANE) - ANE reverse-engineering, hardware characterization, transformer training proof-of-concept
- [mechramc/Orion](https://github.com/mechramc/Orion) - ANE LLM training & inference runtime with graph IR compiler ([paper](https://arxiv.org/abs/2603.06728))
- [vipuldivyanshu92/ANEgpt](https://github.com/vipuldivyanshu92/ANEgpt) - GPT-style transformer training on ANE
- [hollance/neural-engine](https://github.com/hollance/neural-engine) - Community documentation of ANE capabilities (CoreML layer-level, not MIL op-level)
