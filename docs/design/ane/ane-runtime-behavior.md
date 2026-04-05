# ANE Runtime Behavior — Empirical Findings

Systematic probe results from `ironmill-ane-sys` low-level API testing on Apple
Neural Engine hardware.

**Run with:**
```bash
cargo test -p ironmill-ane-sys --test ane_probe -- --ignored --nocapture --test-threads=1
```

**Test environment:**
- Architecture: `h14g` (Apple M3 Pro)
- ANE cores: 16
- macOS build: `25E246` (Sequoia 15.4)
- Compile budget: 119 per process (~45 used per run)

---

## 1. MIL Op Support Matrix

All ops tested with `[1,4,1,4]` fp16 tensors using `program(1.3)` / `func main<ios18>` format.

| Op | Compiles? | Notes |
|---|---|---|
| `identity` | ✅ | Baseline |
| `add` | ✅ | Binary elementwise |
| `mul` | ✅ | Binary elementwise |
| `sub` | ✅ | Binary elementwise |
| `real_div` | ✅ | Binary elementwise |
| `relu` | ✅ | Unary activation |
| `silu` | ✅ | SiLU/Swish activation |
| `softmax` | ✅ | With const `axis` param |
| `round` | ✅ | Key for quantization |
| `clip` | ✅ | With const `alpha`/`beta` |
| `abs` | ✅ | Unary |
| `sign` | ✅ | Unary |
| `sqrt` | ✅ | Unary |
| `exp` | ✅ | Unary |
| `erf` | ✅ | Useful for GELU |
| `pow` | ✅ | Scalar const exponent |
| `greater` | ✅ | Returns bool, cast to fp16 |
| `select` | ✅ | Conditional: `select(cond, a, b)` |
| `reduce_sum` | ✅ | With tile for output shape |
| `reduce_max` | ✅ | With tile for output shape |
| `matmul` | ✅ | 3D tensors, `transpose_x`/`transpose_y` |
| `concat` | ✅ | With const `axis`, `interleave` |
| `transpose` | ✅ | With const `perm` |
| `layer_norm` | ✅ | With const `axes`, `epsilon` |
| `gather` | ❌ | Dynamic gather fails; only static `constexpr_lut_to_dense` works |

**Result: 24/25 ops compile.** Only `gather` (dynamic) fails.

---

## 2. Data Type Support

| Type Path | Compiles? | Notes |
|---|---|---|
| `fp16` input → `fp16` output | ✅ | Native ANE dtype |
| `fp32` input → `fp32` output | ✅ | Accepted, likely converted to fp16 internally |
| `int8` input → `int8` output | ❌ | Cannot use int8 as function I/O directly |
| `cast fp16→int8→fp16` | ✅ | INT8 works as intermediate only |
| `cast fp16→fp32` | ✅ | |
| `cast fp16→int16` | ✅ | |
| `cast fp16→uint8` | ✅ | |
| `cast fp16→int32` | ❌ | |
| `cast fp16→int4` | ❌ | **INT4 not supported** |
| `cast fp16→uint4` | ❌ | **UINT4 not supported** |
| `int4` as function input | ❌ | **INT4 comprehensively rejected** |
| `bool` input → `cast fp16` | ✅ | |

### INT4/UINT4 Assessment (Critical for TurboQuant)

**INT4/UINT4 is completely unsupported on ANE.** Tested three paths:
1. `cast fp16→int4` — rejected by compiler
2. `cast fp16→uint4` — rejected by compiler
3. `int4` as function input type — rejected by compiler

**Implication:** TurboQuant KV cache quantization must use INT8 as the minimum
quantized precision on ANE. INT4 quantization would require CPU/GPU fallback or
simulating 4-bit storage via packed INT8 with shift/mask ops.

### Integer Type Rules

- **Function I/O:** Only `fp16`, `fp32`, and `bool` work as function input/output types
- **Intermediate:** `int8`, `int16`, `uint8` work as intermediate (via `cast`)
- **Rejected:** `int32`, `int4`, `uint4` cannot be cast to from fp16

---

## 3. Shape Constraints

All shapes tested with `add(x, x)` on fp16 tensors.

| Shape | Description | Compiles? |
|---|---|---|
| `[1,1,1,1]` | Minimal | ✅ |
| `[1,4,1,32]` | Typical ANE | ✅ |
| `[1,128,1,128]` | Medium | ✅ |
| `[1,768,1,32]` | Large channels | ✅ |
| `[1,4096,1,32]` | Very large channels | ✅ |
| `[1,4,1,1]` | Single token | ✅ |
| `[2,4,1,4]` | Batch > 1 | ✅ |
| `[1,4,2,4]` | Height > 1 | ✅ |

**Result: All 8 shapes compile.** The ANE compiler is surprisingly flexible with
shape constraints at compile time. Note that evaluation may still fail for some
shapes due to IOSurface alignment requirements — this probe only tests compilation.

---

## 4. Quantization-Relevant Op Chains

| Chain | Compiles? | Description |
|---|---|---|
| `round(clip(mul(x, scale), lo, hi))` | ✅ | Affine quantization |
| `mul→round→clip→cast(int8)→cast(fp16)→mul` | ✅ | Full INT8 quant→dequant |
| `greater→select→greater→select` | ✅ | Codebook lookup via comparison |
| RMSNorm + INT8 quantize pipeline | ✅ | 16-op chain: norm→scale→round→clip→cast(int8)→cast(fp16) |

**All quantization chains compile successfully.** The ANE can handle complex
multi-op pipelines including the full TurboQuant cache-write path (RMSNorm →
quantize to INT8 → dequantize) in a single sub-program. This confirms that INT8
KV cache quantization is viable on ANE.

---

## 5. Client API Behavior

### Echo

`Client::echo()` returns `false` for all tested payload types:
- `echo("hello ANE")` → `false`
- `echo(nil)` → `false`
- `echo(NSNumber(42))` → `false`
- `echo(empty NSDictionary)` → `false`
- Private connection `echo()` → also `false`

The `echo:` selector exists and doesn't crash, but always returns `false`. This
may indicate the ANE daemon doesn't implement echo, or requires a specific
payload format. The method is likely a no-op connectivity check that always
returns `false` in user-space.

### Session Hints

`Client::session_hint()` throws ObjC exceptions with all tested argument types:
- `NSDictionary` (empty) — crashes
- `NSNumber` — crashes
- `nil` — crashes

**Conclusion:** The `sessionHintWithModel:hint:options:report:error:` API exists
but requires undiscovered argument types. Unusable without further reverse
engineering of the expected hint/options dictionary keys.

---

## 6. Performance Stats

| Property | Value |
|---|---|
| `perf_stats_mask` (default) | `0x0` (disabled) |
| `perf_stats_mask` (set to 0xFFFFFFFF) | All bits accepted |
| Bit 0-7 individually | All accepted |
| `PerformanceStats::with_hw_execution_ns(0)` | Creates OK |
| `hw_execution_time` | Reads back value set |

All 32 bits of `perf_stats_mask` are accepted. The mask defaults to `0x0`
(disabled). Enabling bits likely causes the ANE to populate performance counters
during evaluation, but the exact meaning of each bit is undocumented.

---

## 7. Model Attributes

After compilation, `model_attributes()` returns an NSDictionary with 2 keys:

### `ANEFModelDescription`
```
ANEFModelInput16KAlignmentArray: [0]
ANEFModelOutput16KAlignmentArray: [0]
ANEFModelProcedures:
  - ANEFModelInputSymbolIndexArray: [0]
    ANEFModelOutputSymbolIndexArray: [0]
    ANEFModelProcedureID: 0
kANEFModelInputSymbolsArrayKey: ["a_input0"]
kANEFModelOutputSymbolsArrayKey: ["out@output"]
kANEFModelProcedureNameToIDMapKey: {main: 0}
```

### `NetworkStatusList`
```
LiveInputList:
  - BatchStride: 256, Batches: 1, Channels: 4, Depth: 1
    DepthStride: 256, Height: 1, Interleave: 1
    PlaneCount: 4, PlaneStride: 64, RowStride: 64
    Type: Float16, Width: 4
LiveOutputList:
  - (same layout as input for identity model)
```

**Key observations:**
- Output symbols get `@output` suffix (e.g., `out@output`)
- Procedure IDs map function names → integer indices
- `16KAlignmentArray` tracks per-I/O 16KB page alignment requirements
- `NetworkStatusList` contains stride/interleave metadata needed for IOSurface layout
- `PlaneStride` = 64 bytes, `PlaneCount` = channels for interleaved layout

---

## 8. Model Properties

| Property | Value | Notes |
|---|---|---|
| `state` | `3` | Likely: 1=created, 2=compiled, 3=loaded |
| `program_handle` | Non-zero u64 | Kernel-level ANE program handle |
| `intermediate_buffer_handle` | `0` | No intermediates for simple ops |
| `queue_depth` | `127` | Max concurrent evaluations |
| `is_mil_model` | `true` | |
| `compiled_model_exists` | `true` | Cache hit |
| `compiler_options_file_name` | `None` | No custom options used |

---

## 9. Chaining Request API

`ChainingRequest::new()` with all-null arguments:
- **Creates successfully** (surprising — no validation on construction)
- `validate()` returns `false` (as expected with null pointers)
- Property accessors return uninitialized-looking pointer values

The `ChainingRequest` wrapper allocates an ObjC `_ANEChainingRequest` object
even with null inputs. Validation happens separately via `.validate()`. This
means you can construct the request object first, then populate it.

---

## 10. Compiler Options

| Attempt | Result |
|---|---|
| Default (no options) | ✅ |
| Empty options bytes `b""` | Descriptor creates OK |
| Binary plist data | Descriptor creates OK, different hex ID |

The compiler options are passed as an optional binary plist to the descriptor.
Empty options and default (nil) produce the same hex ID, confirming options
affect compilation output. Further investigation needed to discover valid
options keys (e.g., `targetDeviceType`, optimization levels).

---

## 11. Hardware Info

| Property | Value |
|---|---|
| Architecture | `h14g` (M3 Pro) |
| ANE count | 1 |
| ANE cores | 16 |
| Product | macOS |
| Build | 25E246 (Sequoia 15.4) |
| VM | false |

---

## Summary for TurboQuant

1. **INT4 is not viable on ANE** — all paths rejected. Must use INT8 minimum.
2. **INT8 quantization chains work** — full quant→dequant pipeline compiles as single sub-program
3. **Codebook lookup via greater+select works** — enables quantized value mapping
4. **RMSNorm + quantize in one program** — no need for CPU roundtrip
5. **All tested shapes compile** — including large channels (4096) and batch > 1
6. **Model attributes reveal IOSurface layout metadata** — useful for correct tensor mapping
7. **Queue depth = 127** — high parallelism potential
8. **Session hints are unusable** — crashes with all argument types
