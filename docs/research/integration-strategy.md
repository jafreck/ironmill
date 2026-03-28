# Integration Strategy: How `mil-rs` Fits the Rust Ecosystem

## Core Principle: Foundation Crate, Not a Framework Feature

`mil-rs` is a **standalone foundation crate** — not a feature of candle, burn, or any
specific ML framework. It follows the established Rust pattern where a focused library
provides building blocks that multiple higher-level projects compose on top of.

## Precedent: How This Pattern Works in Rust

| Foundation crate | Higher-level consumers | Relationship |
|---|---|---|
| `prost` (protobuf types) | `tonic` (gRPC), many others | Separate repos; tonic depends on prost |
| `serde` (serialization traits) | `serde_json`, `axum`, `rocket`, etc. | Separate repos; hundreds of consumers |
| `bytes` (buffer abstraction) | `tokio`, `prost`, `hyper` | Separate repo; core networking infra |
| **`mil-rs` (CoreML IR + protobuf)** | **`candle-coreml`, `burn-coreml`, CLI** | **Same pattern** |

The key insight: `prost` doesn't live inside `tonic`. `serde` doesn't live inside `axum`.
Foundation crates are independent so that multiple consumers can depend on them without
coupling to each other.

## Why NOT Upstream in Candle

1. **`candle-coreml` is already external.** It lives at `mazhewitt/candle-cormel`, not
   in the huggingface/candle workspace. The precedent is set — CoreML is not an in-tree
   candle concern.

2. **Different abstraction layer.** Candle's in-tree backends (Metal, CUDA) implement
   `BackendStorage` and `BackendDevice` traits for general tensor computation. CoreML is
   not a general compute backend — it runs precompiled models. `mil-rs` is even further
   removed: it's a model *format* library, not a compute library.

3. **Multiple consumers.** Burn, tract, standalone CLI tools, and C/C++ projects via FFI
   all benefit from `mil-rs`. Putting it in candle locks it to one ecosystem.

4. **Burn has its own backend system.** A `burn-coreml` backend would implement Burn's
   `Backend` trait, not candle's. It needs `mil-rs` as a dependency, not candle.

## How Candle's Architecture Works (for context)

```
huggingface/candle workspace
├── candle-core              ← Tensor ops, Backend/Device traits
├── candle-nn                ← Neural network layers
├── candle-metal-kernels     ← Metal GPU shaders (IN-TREE: it's a compute backend)
├── candle-kernels           ← CUDA kernels (IN-TREE: it's a compute backend)
└── candle-onnx              ← ONNX model loader (IN-TREE)

External crates (separate repos, depend on candle-core):
├── candle-coreml            ← CoreML inference bridge (EXTERNAL)
└── ort-candle               ← ONNX Runtime bridge (EXTERNAL)
```

Metal is in-tree because it implements the `BackendStorage` trait for general tensor
math. CoreML is external because it's an inference-only runtime bridge — you hand it a
precompiled model, it runs it.

## How Burn's Architecture Works (for context)

Burn uses a trait-based backend system where any crate can implement `Backend`:

```
tracel-ai/burn workspace
├── burn-core                ← Backend trait, tensor API
├── burn-ndarray             ← Backend: ndarray (IN-TREE)
├── burn-tch                 ← Backend: libtorch (IN-TREE)
├── burn-candle              ← Backend: candle (IN-TREE)
└── burn-wgpu                ← Backend: WebGPU via CubeCL (IN-TREE)

Future (separate repo):
└── burn-coreml              ← Backend: CoreML via mil-rs (EXTERNAL)
```

## The Dependency Graph

```
                    ┌─────────────────────┐
                    │      mil-rs         │  ← Foundation: CoreML protobuf +
                    │  (this project)     │     MIL IR + ONNX→CoreML conversion
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
   ┌──────────────────┐ ┌───────────┐ ┌───────────────┐
   │  candle-coreml   │ │ burn-     │ │ ironmill    │
   │  (existing,      │ │ coreml   │ │ CLI           │
   │   adds optional  │ │ (future) │ │ (this project)│
   │   mil-rs dep)    │ │          │ │               │
   └────────┬─────────┘ └─────┬────┘ └───────────────┘
            │                  │
            ▼                  ▼
   ┌──────────────────┐ ┌───────────┐
   │  candle-core     │ │ burn-core │
   │  (huggingface)   │ │ (tracel)  │
   └──────────────────┘ └───────────┘
```

### What each layer does

| Crate | Owns | Depends on |
|---|---|---|
| `mil-rs` | CoreML protobuf types, MIL IR, ONNX→MIL conversion | `prost` (protobuf) |
| `ironmill` CLI | User-facing conversion tool | `mil-rs` |
| `candle-coreml` | Candle↔CoreML tensor bridge, inference | `candle-core`, optionally `mil-rs` |
| `burn-coreml` | Burn `Backend` trait impl for CoreML | `burn-core`, `mil-rs` |

## Integration Path: Step by Step

### Phase 1: Publish `mil-rs` standalone
- CoreML protobuf reader/writer
- MIL IR data structures
- Basic ONNX → MIL conversion (top ops)
- Published to crates.io independently
- **No dependency on candle or burn**

### Phase 2: `ironmill` CLI
- Uses `mil-rs` for conversion
- Calls `xcrun coremlcompiler` for final compilation
- `cargo install ironmill` — a standalone tool

### Phase 3: Ecosystem bridges (PRs to existing projects)
- **PR to `candle-coreml`**: Add optional `mil-rs` dependency, enabling
  `candle-coreml::convert_onnx("model.onnx")` alongside existing inference
- **New crate `burn-coreml`**: Implement Burn's `Backend` trait, using `mil-rs`
  for model export and CoreML runtime for execution
- **`tract` integration**: Optional CoreML acceleration path

### Phase 4: Upstream discussion
- Once `mil-rs` has users and stability, discuss with candle/burn maintainers
  whether deeper integration makes sense
- Possible outcomes:
  - `mil-rs` stays external (most likely, like `prost`)
  - `candle-coreml` adopts `mil-rs` as a required dep
  - Burn adds `burn-coreml` to their workspace (like `burn-candle` is today)

## What We Don't Do

- **We don't fork or modify candle/burn.** We publish a crate they can optionally depend on.
- **We don't implement `BackendStorage` or `BackendDevice`.** That's the job of bridge
  crates like `candle-coreml` and `burn-coreml`.
- **We don't compete with Metal backends.** Metal (via candle/burn) and CoreML (via our
  toolchain) serve different use cases. Metal is general compute. CoreML is optimized
  model execution with ANE access.

## Analogy Summary

> `mil-rs` is to the Rust ML ecosystem what `prost` is to the Rust networking ecosystem.
>
> `prost` doesn't implement gRPC — `tonic` does that. But `tonic` couldn't exist without
> `prost` providing the protobuf foundation.
>
> Similarly, `mil-rs` doesn't implement ML inference — `candle-coreml` and `burn-coreml`
> do that. But they can't create CoreML models without a Rust-native CoreML format library.
