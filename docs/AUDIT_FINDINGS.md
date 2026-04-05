# Ironmill Codebase Audit — Findings

**Date:** 2026-04-05
**Scope:** Full source inspection across all workspace crates
**Status:** MLX backend removed (commit `79f53d7`). All remaining findings listed below.

---

## 1. Build Breakers

### 1.1 CLI `compile_output()` return type mismatch
- **File:** `crates/ironmill-cli/src/main.rs:1324-1330`
- **Issue:** `compile_output()` uses `?` but its return type is `()`. The CLI binary fails to compile.
- **Severity:** Build-breaking

---

## 2. Correctness Bugs

### 2.1 Prefix caching inserts empty KV slices
- **File:** `crates/ironmill-inference/src/engine.rs:129-157`
- **Issue:** `prefill_with_cache()` ignores the returned KV slices from the engine and inserts an empty `KvCacheSlice` into the cache. Prefix caching is effectively broken.

### 2.2 MLX-removal leftover — `truncate_to()` was only in MLX
- **Resolved** by MLX removal. Metal's `truncate_to()` should be verified independently.

### 2.3 `SequenceStatus::Waiting` hits `unreachable!()`
- **File:** `crates/ironmill-inference/src/serving/batch.rs:59-69`
- **Issue:** `Waiting` is a real enum variant that can be constructed, but the match arm calls `unreachable!()`.

### 2.4 `ProjectionMatmul::Quantized` hits `unreachable!()`
- **File:** `crates/ironmill-inference/src/metal/inference.rs:293-299`
- **Issue:** Quantized projection is a valid variant that may be reachable with certain quantization configs.

### 2.5 `write_f32()` always fails on IOSurface
- **File:** `crates/ironmill-iosurface/src/tensor.rs:211-219`
- **Issue:** `write_f32()` forwards to `write_f16()`, which rejects non-`Float16` tensors. Float32 writes silently fail.

### 2.6 Streamed chat never appends assistant reply
- **File:** `crates/ironmill-torch/src/chat.rs:75-89`
- **Issue:** `send_stream()` returns a `TextStream` but never appends the assistant's reply to the conversation history. Multi-turn streamed chat is broken.

### 2.7 MIL `compute_unit` doesn't round-trip
- **File:** `crates/mil-rs/src/convert/ir_to_proto.rs` vs `proto_to_ir.rs`
- **Issue:** `compute_unit` is serialized as `tensor<string>` but the deserializer expects a plain `Value::String`. Round-trip fails.

### 2.8 `Value::Int(i64)` silently truncated to i32
- **File:** `crates/mil-rs/src/convert/ir_to_proto.rs:1475-1488`
- **Issue:** Int values are serialized as i32. Values exceeding `i32::MAX` are silently truncated or lost.

### 2.9 Dynamic shapes collapsed to static on export
- **File:** `crates/mil-rs/src/convert/ir_to_proto.rs:1668-1678`
- **Issue:** `None` (unknown) dimensions are exported as `1`, losing dynamic shape information.

### 2.10 Tensor shape unknown dims filtered out
- **File:** `crates/mil-rs/src/convert/proto_to_ir.rs:274-295,343-355`
- **Issue:** Tensor shapes with unknown dims are rebuilt by filtering out `None`, which collapses rank and shape information.

---

## 3. Unimplemented Stubs (`todo!()`)

### 3.1 `BatchRunner` entirely stubbed
- **File:** `crates/ironmill-inference/src/batch_runner.rs:63-84`
- **Issue:** `submit`, `step`, `cancel`, `has_pending`, `active_count` are all `todo!()`. Fields `engine`, `config`, `next_handle` are never read.

### 3.2 `load_jit()` stub
- **File:** `crates/ironmill-inference/src/metal/inference.rs:588-593`
- **Issue:** `load_jit()` body is `todo!()`.

### 3.3 `MetalCompileTarget` / `CoremlCompileTarget` stubs
- **File:** `crates/ironmill-compile/src/compile_target.rs:89-124`
- **Issue:** Both return `Err("not yet implemented")`.

### 3.4 CLI adapter flags stub
- **File:** `crates/ironmill-cli/src/main.rs:1219-1227`
- **Issue:** `--emit-adapter` and `--adapter` bail with "not yet implemented".

### 3.5 ANE hybrid/chained execution stubs
- **File:** `crates/ironmill-inference/src/ane/decode.rs:969-1041`
- **Issue:** Hybrid and chained ANE execution paths return `Err` with TODO comments.

### 3.6 JIT transform stubs
- **File:** `crates/ironmill-inference/src/jit.rs:66-76`
- **Issue:** `Int4AffineTransform`, `Fp16CastTransform`, `PolarQuantTransform` are listed as TODO.

---

## 4. Incomplete Features

### 4.1 MIL IR has no nested block support
- **File:** `crates/mil-rs/src/ir/operation.rs:56-84`, `crates/mil-rs/src/convert/proto_to_ir.rs:95-157`
- **Issue:** `Operation` has no nested block/control-flow support. Proto deserialization ignores `proto.blocks` entirely. Any `cond`/`while`-style ops lose their bodies.

### 4.2 Quantization formats parsed but rejected
- **File:** `crates/ironmill-compile/src/convert/pipeline.rs:59-80,434-444,512-528`
- **Issue:** AWQ, GPTQ, D2Quant, Palettize, and PolarQuant are parsed but rejected at runtime. QuipSharp is hardcoded to `(2, 42)`.

### 4.3 `TargetComputeUnit` stored but never applied
- **File:** `crates/ironmill-compile/src/coreml/build_api.rs:84-93,143-146,189-298`
- **Issue:** `target: TargetComputeUnit` is stored and settable via the API, but `build()` never reads or applies it to the emitted model/pipeline.

### 4.4 GGUF dequant only supports 5 of 15+ types
- **File:** `crates/ironmill-compile/src/weights/gguf.rs:891-900`
- **Issue:** `dequantize_to_fp16()` only supports F32, F16, BF16, Q4_0, Q8_0. All other GGML types (Q4_1, Q5_*, Q8_1, IQ*) return errors. Gemma 4 GGUF is rejected.

### 4.5 GPU bundle drops quantization metadata
- **File:** `crates/ironmill-compile/src/gpu/bundle.rs:129-204`
- **Issue:** GPU bundle serialization drops `g_idx` entirely and forces `scale_dtype`/`zero_point_dtype` back to FP16. Quantization metadata doesn't round-trip.

### 4.6 Only LLaMA template supports ANE lowering
- **File:** `crates/ironmill-compile/src/templates/mod.rs:64-67`
- **Issue:** `TemplateOptions.ane` is only honored by the LLaMA template. Qwen and Gemma templates ignore it entirely.

### 4.7 `extract_component()` is heuristic and breaks dataflow
- **File:** `crates/ironmill-compile/src/templates/mod.rs:98-188`
- **Issue:** Component extraction drops `embed_out` for Transformer-only, and `LmHead` fabricates a fixed hidden-state input.

### 4.8 `Device::Auto` doesn't work on non-macOS
- **File:** `crates/ironmill-torch/src/model.rs:229-303`
- **Issue:** `Device::Auto` resolves to `Cpu` on non-macOS, but `load_pretrained()` / `load_compiled()` only accept `Device::Metal`. Non-macOS is dead.

### 4.9 `use_fa2_prefill` accepted but ignored
- **File:** `crates/ironmill-inference/src/metal/config.rs:94-98`, `metal/inference.rs:972-991`
- **Issue:** FA2 prefill config is accepted but never wired into the execution path.

### 4.10 ANE QK norm and shared-event handoff stubs
- **File:** `crates/ironmill-inference/src/ane/decode.rs:792,978`
- **Issue:** QK norm weight loading from bundle and Metal↔ANE shared-event handoff are still TODO.

### 4.11 `tile.reps` fallback returns incorrect type
- **File:** `crates/mil-rs/src/ir/passes/type_repropagate.rs:161-193`
- **Issue:** If `tile.reps` can't be resolved, returns "incorrect but safe fallback" instead of erroring.

### 4.12 ANE shape conversion is heuristic
- **File:** `crates/ironmill-compile/src/ane/passes/ane_layout.rs:206-289`
- **Issue:** 3D/4D/dynamic rank shapes are handled by collapsing dims and guessing layout.

### 4.13 `GenerateEvent::PromptProcessed` silently dropped
- **File:** `crates/ironmill-torch/src/model.rs:72-80`
- **Issue:** `_ => {}` catch-all silently drops events that callers may need.

### 4.14 Validation is shallow
- **File:** `crates/mil-rs/src/ir/graph.rs:49-93`
- **Issue:** Validation checks refs and duplicate op names only. No op arity, output-type count, nested block, or structural invariant checks.

### 4.15 `TensorData::materialize_with()` doesn't verify length
- **File:** `crates/mil-rs/src/ir/types.rs:102-115`
- **Issue:** After materialization, byte length is never verified against expected `byte_len`.

### 4.16 Program/function attributes dropped on import
- **File:** `crates/mil-rs/src/convert/proto_to_ir.rs:40-56,59-105`
- **Issue:** Program, function, and block attributes and doc strings are dropped during proto import.

### 4.17 `BlobFileValue` not round-trippable
- **File:** `crates/mil-rs/src/convert/proto_to_ir.rs:201-207`, `ir_to_proto.rs:1518-1531`
- **Issue:** `BlobFileValue` is reduced to a string placeholder in IR. Shared-weight/blob refs aren't faithfully preserved.

### 4.18 `with_shapes()` doesn't cover all quantization passes
- **File:** `crates/mil-rs/src/ir/pipeline.rs:1065-1086`
- **Issue:** Only inserts before fp16/int8/palettization/polar-quantization. Pipelines using int4/AWQ/GPTQ/QuIP#/D2Quant/SpinQuant can materialize shapes too late.

### 4.19 Weight loader calls Metal-specific code
- **File:** `crates/ironmill-inference/src/weight_loading.rs:229`
- **Issue:** Shared dequant path calls `crate::metal::dequant::dequant_quip_sharp`. If the weight loading module is ever used without Metal, this breaks.

### 4.20 `cal_data` silently ignored for non-Int8
- **File:** `crates/ironmill-compile/src/convert/pipeline.rs:140-141,517-519`
- **Issue:** Calibration data is only consumed for Int8 quantization; silently ignored for all other modes.

---

## 5. Legacy / Backward-Compatibility Code

### 5.1 NeuralNetwork model rejection
- **File:** `crates/mil-rs/src/convert/proto_to_ir.rs:13-24`
- **Issue:** Rejects legacy `NeuralNetwork` models. Can simplify for alpha (only ML Program models).

### 5.2 Legacy INT8 defaulting
- **File:** `crates/mil-rs/src/weights.rs:273-283`, `crates/ironmill-compile/src/weights/mil_provider.rs:191-195`
- **Issue:** Legacy `bit_width = 8` defaulting for INT8. Removable if no legacy quantized models need loading.

### 5.3 Backward-compat `pub mod mil` re-exports
- **File:** `crates/ironmill-compile/src/lib.rs:59-78`
- **Issue:** Explicit backward-compat re-exports with TODO comments to remove. Leaks internal `tensor_utils` and generated protobuf types.

### 5.4 Compat re-export shim in inference
- **File:** `crates/ironmill-inference/src/model_info.rs:1-6`
- **Issue:** Re-export shim for backward compatibility.

### 5.5 Legacy name-based ANE split fallback
- **File:** `crates/ironmill-compile/src/ane/split.rs:209-220`
- **Issue:** Falls back to a single `main` group based on layer naming. Removable if naming is standardized.

### 5.6 Updatable model training-input synthesis
- **File:** `crates/mil-rs/src/convert/ir_to_proto.rs:381-390,452-461`
- **Issue:** Fallback training-input/output synthesis for updatable models. Removable if updatable models aren't shipping in alpha.

### 5.7 Old byte-write KV cache path
- **File:** `crates/ironmill-inference/src/ane/turboquant/model.rs:192-199,364-410,892-905`
- **Issue:** Old `update_cache` byte-write path vs new direct IOSurface path. Runtime uses the direct path. Old path appears removable.

### 5.8 Legacy non-MIL ANE network-description API
- **File:** `crates/ironmill-ane-sys/src/model.rs:38-41,122-129`
- **Issue:** Legacy network-description API. Removable if MIL text is the only supported ANE input.

### 5.9 Dual `.mlmodel` + `.mlpackage` read/write
- **File:** `crates/ironmill-compile-ffi/src/lib.rs:100-152,236-296`
- **Issue:** Parallel support for old `.mlmodel` and new `.mlpackage` formats. Remove `.mlmodel` to standardize on `.mlpackage`.

---

## 6. Dead Code

### 6.1 `int4` feature flag (zero references)
- **File:** `crates/ironmill-inference/Cargo.toml:28`
- **Issue:** Feature `int4 = []` defined but no `cfg(feature = "int4")` exists anywhere.

### 6.2 `Device::Cuda` variant
- **File:** `crates/ironmill-core/src/device.rs:6-13`
- **Issue:** No CUDA support exists in the codebase. Placeholder variant.

### 6.3 Unused `TokenizerError` variants
- **File:** `crates/ironmill-core/src/tokenizer.rs:48-66`
- **Issue:** `TokenizerError::NotLoaded` and `TokenizerError::Template` have no call sites.

### 6.4 `HfTokenizer.chat_template` parsed but never read
- **File:** `crates/ironmill-core/src/tokenizer.rs:126-130`
- **Issue:** Field is parsed from config but never accessed.

### 6.5 `HfTokenizer::from_file` unused
- **File:** `crates/ironmill-core/src/tokenizer.rs:148-152`
- **Issue:** Public method with no call sites in the workspace.

### 6.6 `build_dummy_inputs` unused
- **File:** `crates/ironmill-inference/src/types.rs:143-155`
- **Issue:** Public helper with no call sites.

### 6.7 `ironmill-torch` dependency in CLI (unused)
- **File:** `crates/ironmill-cli/Cargo.toml:15-20`
- **Issue:** `ironmill-torch` is declared as a dependency but never imported in CLI source.

### 6.8 `TorchError::UnknownFormat` unused
- **File:** `crates/ironmill-torch/src/error.rs:34-36`
- **Issue:** Error variant with no construction sites.

### 6.9 `ChatSession::send_stream` unused
- **File:** `crates/ironmill-torch/src/chat.rs:71-82`
- **Issue:** Public method with no call sites (also has bug §2.6).

### 6.10 `ModelBuilder::with_progress` unused
- **File:** `crates/ironmill-torch/src/model.rs:196-199`
- **Issue:** Public builder method with no call sites.

### 6.11 Unused ANE utility types
- **File:** `crates/ironmill-ane-sys/src/util.rs:21-220`, `src/perf.rs:20-220`
- **Issue:** `AneLog`, `AneErrors`, `AneCloneHelper`, `PerformanceStatsIOSurface`, `QoSMapper` appear to have no non-test consumers.

### 6.12 `ironmill-torch` features declared but unused
- **File:** `crates/ironmill-torch/Cargo.toml:13-17`
- **Issue:** `coreml`, `ane`, and `async` features declared but not gated on in source.

---

## 7. Error Handling

### 7.1 `panic!()` / `unwrap()` in library code
- `crates/ironmill-core/src/ane/mil_text.rs:115-694` — `write!().unwrap()` and `panic!("unsupported")` arms
- `crates/mil-rs/src/ir/types.rs:76-84` — `TensorData::into_bytes()` panics on `External` tensors
- `crates/ironmill-compile/src/weights/mil_provider.rs:208-230` — `expect("tensor not materialized")`
- `crates/ironmill-compile/src/ane/validate.rs:444-456` — `panic!("unsupported scalar type")`

### 7.2 Swallowed errors
- `crates/ironmill-ane-sys/src/model.rs:1152-1168` — FS write errors silently ignored (`let _ = ...`); `make_blobfile().unwrap_or_default()` can emit bad output
- `crates/ironmill-inference/src/shader_cache.rs:46-49,80-89` — cache read/directory errors collapsed to `None`/`0`
- `crates/ironmill-torch/src/text_output.rs:77-105` — token decode failures swallowed via `unwrap_or_default()`

### 7.3 Catch-all error variants
- `crates/ironmill-compile/src/error.rs:19-37` — `CompileError::Other(String)`
- `crates/ironmill-inference/src/engine.rs:14-45` — `InferenceError::Other(anyhow::Error)`
- Multiple `error.rs` files use `String` payloads instead of structured variants

### 7.4 `unreachable!()` on reachable paths
- See §2.3 (`SequenceStatus::Waiting`) and §2.4 (`ProjectionMatmul::Quantized`)

### 7.5 Infallible `Result` returns
- `crates/mil-rs/src/convert/onnx_to_mil.rs:273-519` — many converters return `Result<Vec<Operation>>` but are infallible (always `Ok(...)`)

---

## 8. Public API Surface

### 8.1 Raw internal buffers exposed
- `mil-rs/src/weights.rs:252-309` — `QuantizationInfo`/`WeightTensor` expose raw `Vec<u8>`, seeds, compression layout
- `ironmill-inference/src/types.rs:58-80` — `RuntimeTensor` exposes raw `Vec<u8>`, `Vec<usize>`
- `ironmill-inference/src/generate.rs:51-102` — `GenerateRequest` owns `Vec<u32>` for prompt/stop tokens

### 8.2 Untyped string fields
- `ironmill-core/src/tokenizer.rs:13-16` — `ChatMessage.role: String` should be a `Role` enum
- `ironmill-core/src/model_info.rs:12-41` — `ModelInfo.weight_quantization: String` should be a typed enum
- `ironmill-torch/src/text_output.rs:13-42` — `finish_reason: String` should reuse `FinishReason` enum

### 8.3 Leaking implementation details
- `ironmill-torch/src/model.rs:127-139` — `engine()`/`engine_mut()`/`tokenizer()` expose backend trait objects
- `ironmill-compile/src/lib.rs:59-78` — `pub mod mil` leaks internal `tensor_utils` and protobuf types

### 8.4 Public data bags without builders
- `ironmill-torch/src/gen_params.rs:13-26,80-83` — `GenParams` is all-public with `Vec<u32>` stop tokens
- `ironmill-core/src/model_info.rs:12-41` — `ModelInfo` all-public fields, no validation

### 8.5 CLI arguments parsed but not wired
- `crates/ironmill-cli/src/main.rs:131-166,480-489` — `--target` mostly ignored; only `Gpu` changes behavior
- `crates/ironmill-cli/src/main.rs:256-266,491-508` — `--kv-quant`, `--kv-quant-qjl`, `--max-seq-len` validated/printed but not wired into compilation

---

## 9. Documentation

### 9.1 Inaccurate docs
- `README.md:19-24` — Claims CUDA support (none exists); omits Metal/MLX backends
- `ironmill-inference/src/lib.rs:1-5` — Says "ANE + CoreML backends"; Metal also exists
- `ironmill-inference/src/engine.rs:3-4` — Docs mention ANE/CoreML only

### 9.2 Missing doc comments
- ~650+ public items across the workspace lack doc comments
- Worst coverage: `ironmill-inference` (227), `mil-rs` (150), `ironmill-compile` (114)

### 9.3 TODO comments (not yet addressed)
- `ironmill-compile/src/lib.rs:60-78` — deprecate/remove `mil` re-exports
- `ironmill-inference/src/sampling.rs:85-87` — EOS tokens should come from `ModelArchitecture`
- `ironmill-inference/src/ane/decode.rs:792-795` — load per-layer QK norm weights from bundle
- `ironmill-inference/src/ane/decode.rs:978-990` — implement Metal↔ANE shared-event handoff

### 9.4 `docs/` directory needs triage
- Mix of current reference docs, historical archive, and research notes
- No clear separation between "current" and "archive" beyond the `archive/` subdirectory

---

## 10. Test Coverage

### 10.1 Unsafe FFI crates with zero tests
- `crates/ironmill-metal-sys` — no tests
- `crates/ironmill-coreml-sys` — no tests
- (MLX sys crate removed)

### 10.2 No workspace-level integration tests
- `tests/` directory contains only fixtures, no `.rs` integration test files

### 10.3 Heavily gated test suites
- Many `#[ignore]` tests requiring hardware (ANE), Xcode, or large fixture files
- CI coverage is thin as a result

### 10.4 Stale test fixture
- `tests/fixtures/Qwen3-1.7B/` — no code references found; likely stale

---

## 11. Dependency Hygiene

### 11.1 Path deps instead of workspace deps
- `crates/burn-coreml/Cargo.toml:16-17` — `ironmill-inference` should use `workspace = true`
- `crates/candle-coreml/Cargo.toml:16-17` — same
- `crates/ironmill-bench/Cargo.toml:16-17,32` — `ironmill-inference`, `ironmill-iosurface`, `ironmill-metal-sys`

### 11.2 Duplicate dependency declaration
- `crates/mil-rs/Cargo.toml:27,31` — `tempfile` in both `[dependencies]` and `[dev-dependencies]`

### 11.3 Default features make optional deps always-on
- `crates/ironmill-inference/Cargo.toml` — `default = ["coreml", "ane"]` makes optional backend deps always-on for default builds

### 11.4 Stale `c-api` feature gate docs
- `crates/ironmill-compile-ffi/src/lib.rs` — docs reference a `c-api` feature gate that doesn't exist in `Cargo.toml`

### 11.5 Broad `ironmill-core` scope
- `crates/ironmill-core/src/lib.rs:8-19` — bundles ANE/GPU/tokenizer/weights modules; broader than "shared types"
