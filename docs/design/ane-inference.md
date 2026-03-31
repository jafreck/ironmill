# ANE Inference — Design & Status

> Consolidated from `ane-inference-optimizations.md` and `ANE-op-discoveries.md`.
> Original investigation docs archived in `docs/archive/`.

## Current Status

### What Works
- Decode loop with per-layer sub-program execution (pre_attn → attention → post_attn)
- Pre_attn (QKV projections via 1×1 conv) compiles and runs on ANE
- Post_attn (FFN via 1×1 conv + SiLU) compiles and runs on ANE
- TurboQuant INT8 KV cache pipeline (quantize, cache write/read, dequantize)
- CPU embedding lookup
- CPU RoPE rotation (ANE `gather` is unsupported; CPU cost is trivial)
- Token sampling with temperature, greedy, and EOS detection
- Chunked lm_head on ANE (1×1 conv with donor/patch weight reuse)
- Separate compile/load lifecycle (Orion-style `loadWithQoS`/`unloadWithQoS`)
- `Drop for CompiledProgram` prevents cross-model handle leaks

### What Doesn't Work
- No end-to-end correctness tests (perplexity, token agreement) — see
  `docs/development/QUALITY_BENCHMARK_PLAN.md`

### Performance

| Config | Throughput | KV Cache |
|---|---|---|
| Qwen3-0.6B FP16 baseline | ~6.1 tok/s | 29.0 MB |
| Qwen3-0.6B TurboQuant INT8 | ~5.5 tok/s | 14.5 MB |

## Key Discoveries

### Eval-Verified ≠ Compiler-Accepted
Individual ops passing eval probes does not guarantee the ANE compiler
(`_ANECCompile`) will accept a program containing them. The compiler applies
additional constraints on op combinations, tensor shapes, and program structure
that don't surface during single-op testing.

### Program Budget
- ~55 simultaneously-alive compiled models per process (model-size dependent)
- 29-layer Qwen3-0.6B × 3 sub-programs = 87 → exceeds budget
- Fix: separate compile/load lifecycle — compile all upfront, load ≤3 at a time
- `Drop for CompiledProgram` calls `CFRelease` to prevent handle leaks

### IOSurface Constraints
- Minimum allocation: **16KB** (not 48KB as previously assumed)
- ANE rejects `[1, C, 1, S]` I/O tensors when `C > ~768` and `S < 32`

### S≥32 Padding — Must Be Per-Sub-Program
- S≥32 padding fixes pre_attn/post_attn (high-C tensors)
- Padding **breaks** attention sub-programs where S represents per-head
  dimensions, not sequence positions
- With S=32: `[1, 2048, 1, 32]` → reshape `[1, 16, 128, 32]` — wrong
- Without: `[1, 2048, 1, 1]` → reshape `[1, 16, 128, 1]` — correct dimensions
- But `C=2048 > 768, S=1 < 32` → ANE constraint violation regardless

### Op-Level Findings

| Op | Finding | Workaround |
|---|---|---|
| `gather` | Runtime-rejected | CPU RoPE lookup (trivial: 256 bytes memcpy) |
| `split` | Unreliable with matmul+softmax+tile | Strip RoPE from ANE sub-programs |
| `concat` | ✅ Works (previous assumption it didn't was **wrong**) | N/A |
| `matmul` dynamic×dynamic | May fail when S≥32 padding corrupts shapes | Use correct pre-padded shapes |

## Architecture Requirements for Real Attention

FP16 attention on ANE is implemented via hand-written MIL programs
(`c632f05`). The key architectural decisions:

1. **KV cache as matmul inputs** — IOSurface-backed tensors with `S=seq_len ≥ 32`,
   not single-token `S=1` projections
2. **Hand-written MIL programs** with correct shapes per sub-program type
   (TurboQuant's MIL programs work because shapes are explicitly correct)
3. **CPU RoPE rotation** — `gather` is unsupported but rotation is trivially
   cheap on CPU (~nanoseconds for single-token decode)
4. **Per-head norms on CPU** — untested in combination with attention on ANE;
   safest to offload (also cheap)

## References

- [ANE Op Support Matrix](ane-op-support-matrix.md) — 74 verified ops
- [ANE Constraints](ane-constraints.md) — hardware limits and diagnostics
- [TurboQuant Design](turboquant.md) — INT8 KV cache compression
