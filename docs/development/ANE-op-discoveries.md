# ANE Operation Discoveries

Findings from the Qwen3-0.6B FP16 attention pipeline investigation.
These supplement `ane-op-support-matrix.md` with compiler-level
constraints discovered empirically during sub-program compilation.

## Key Finding: Eval-Verified ≠ Compiler-Accepted

Individual ops passing eval probes does not guarantee the ANE
**compiler** (`_ANECCompile`) will accept a program containing them.
The compiler applies additional constraints on op combinations,
tensor shapes, and program structure that don't surface during
single-op eval testing.

## Specific Discoveries

### `gather` — Runtime Gather Rejected

- **Status**: ❌ compiler-rejected (confirmed by `ane-op-support-matrix.md`)
- **Context**: RoPE cos/sin cache lookup uses `gather(cos_cache, position_ids, axis=0)`
- **Workaround**: CPU-side gather (trivially cheap for single-token decode: 256 bytes memcpy, ~nanoseconds)
- **Note**: Static `constexpr_lut_to_dense` gather works (compile-time expansion). Only runtime/dynamic gather fails.

### `split` — Unreliable in Attention Sub-Programs

- **Support matrix status**: ⚠️ compile-only
- **Finding**: Programs containing `split` (from RoPE half-rotation) fail `_ANECCompile` when combined with matmul + softmax + tile in the same program
- **Individual probing**: `split` compiles standalone
- **In combination**: `split` + `concat` + `matmul` + `softmax` → compiler rejects the full program
- **Workaround**: Strip RoPE ops from ANE sub-programs; apply rotation on CPU

### `concat` — Works on ANE (Previous Validation Was Wrong)

- **Support matrix status**: ✅ eval-verified (`max_err=0.025`)
- **Previous ironmill assumption**: "ANE does not support concat (constraint #1)" — **FALSE**
- **Finding**: `concat` with const `axis` and `interleave` is eval-verified and works
- **Action taken**: Removed the false concat validation from `inference.rs`

### `matmul` — Dynamic×Dynamic May Require Shape Constraints

- **Support matrix status**: ✅ eval-verified
- **Finding**: `matmul(Q, K, transpose_y=true)` where both Q and K are dynamic (not const weights) fails compilation when shapes are `[1, 2048, 1, 32]` (S=32 padded)
- **Hypothesis**: The S≥32 global padding corrupts per-head attention dimensions. When the attention sub-program reshapes `[1, 2048, 1, 32]` → `[1, 16, 128, 32]`, the S=32 dimension is misinterpreted. The matmul shapes become incorrect for the ANE's internal constraints.
- **TurboQuant comparison**: TurboQuant's matmul works because its MIL programs are hand-written with correct shapes (`[1, num_heads, head_dim, 1]` for Q, `[1, num_kv_heads, head_dim, seq_len]` for K cache)
- **Workaround**: Skip S≥32 padding for fp16_attn sub-programs, or emit MIL with pre-padded-correct shapes

### `reduce_mean` + `pow` — Work Individually, Untested in Combination

- **Support matrix status**: ✅ eval-verified (both)
- **Context**: Per-head Q/K norms use `mul → reduce_mean → add → pow(-0.5) → mul → mul`
- **Finding**: These ops were present in fp16_attn sub-programs that failed compilation, but the failure was caused by other ops (split, shape issues). These ops are NOT confirmed as problematic — they may work in isolation within an attention sub-program.

## Program Budget Constraint

### `_ANEInMemoryModel` Memory Budget

- The ANE has a per-process memory budget for compiled models
- `compile_mil_text` both compiles AND loads the model (consumes a budget slot)
- **Limit observed**: ~55 simultaneously-alive compiled models on M-series (model-size dependent)
- **29-layer Qwen3-0.6B**: 29 layers × 3 sub-programs = 87 → exceeds budget
- **Workaround**: Orion-style `loadWithQoS:`/`unloadWithQoS:` (see below)

### Orion's Solution: Separate Compile/Load Lifecycle

Discovered from [Orion](https://github.com/mechramc/Orion) (`core/ane_runtime.m`):

```objc
// Orion uses SEPARATE compile and load APIs:
[model compileWithQoS:21 options:@{} error:&e];   // compile to ANE format
[model loadWithQoS:21 options:@{} error:&e];       // load into ANE engine

// After unload, reload IS possible:
[model unloadWithQoS:21 error:&e];                 // free ANE slot
[model loadWithQoS:21 options:@{} error:&e];       // reload — WORKS!
```

- `unloadWithQoS:` frees the ANE execution slot but the ObjC model object stays alive
- `loadWithQoS:` can reload using the compiled artifact (net.plist on disk)
- This enables lazy load/unload: compile all programs upfront, unload all, then load ≤3 at a time during decode

### `CompiledProgram` Drop — Prevents Cross-Model Leaks

- Neither `CompiledProgram` nor `LoadedProgram` had `Drop` impls
- Model handles leaked when `AneInference` went out of scope
- When the benchmark ran baseline then TurboQuant in the same process, baseline's ~58 leaked models consumed budget before TurboQuant could compile
- **Fix**: Added `Drop for CompiledProgram` that calls `CFRelease`

## IOSurface Constraints (Refined)

### Minimum Allocation Size

- Previous assumption: 48KB (`MIN_SURFACE_ALLOC = 49152`) — overly conservative
- Empirical finding: 16KB (`ANE_MIN_SURFACE_BYTES = 16384`) is sufficient
- Exact-fit allocations (as used by Orion/maderix) fail for very small tensors
- ANE rejects IOSurfaces below 16KB with status `0x1d`

### S≥32 Padding — Necessary but Causes Shape Issues

- ANE rejects I/O tensors `[1, C, 1, S]` when `C > ~768` and `S < 32`
- The global S≥32 padding in `inference.rs` fixes this for pre_attn/post_attn
- **Problem**: The same padding corrupts attention sub-program shapes where S represents per-head dimensions, not sequence positions
- **Fix needed**: Selective padding — apply S≥32 only to sub-programs with high-C tensors, not to attention sub-programs where shapes have specific meaning

## Remaining Blockers for FP16 Attention on ANE

1. **S≥32 shape corruption**: Global padding makes attention matmul shapes invalid
2. **RoPE on CPU**: `gather` unsupported → rotation must happen on CPU (trivially cheap)
3. **Per-head norms on CPU**: Untested on ANE in combination with attention; safest to do on CPU (also cheap)

The TurboQuant path avoids all three by using hand-written MIL programs with correct shapes and CPU-side rotation.
