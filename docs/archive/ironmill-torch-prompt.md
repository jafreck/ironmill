# Implementation Prompt: ironmill-torch Crate

## Context

You are implementing a new Rust crate called `ironmill-torch` in the ironmill workspace at `/Users/jacobfreck/Source/ironmill-api-spec`. The full specification is at `docs/design/ironmill-torch.md` — read it first.

**Architecture**: `ironmill-core` is a foundational shared crate at the bottom of the dependency graph. Both `ironmill-compile` and `ironmill-inference` depend on it. `ironmill-torch` is a NEW crate that sits ABOVE both `ironmill-compile` and `ironmill-inference`, providing a PyTorch-level abstraction for model loading, generation, and chat.

```
ironmill-core       (shared types — DO NOT MODIFY except removing moved types)
  ↑          ↑
ironmill-compile   ironmill-inference
  ↑          ↑
  ironmill-torch   (NEW — you are creating this)
       ↑
  ironmill-cli     (update deps to include ironmill-torch)
```

**CRITICAL**: `ironmill-core` must NOT depend on `ironmill-inference` or `ironmill-compile`. The dependency direction is the opposite. `ironmill-torch` depends on both.

**Branch**: You are working in the `api-spec` worktree at `/Users/jacobfreck/Source/ironmill-api-spec`. The `api-spec` branch has the prerequisite API changes (InferenceEngine trait, GenerateRequest, TokenStream, etc.) that this crate depends on. Create a new branch `feat/ironmill-torch` from `api-spec`.

## What to do

### Phase 1: Create the crate

1. Create `crates/ironmill-torch/` with the structure from spec §3.
2. Add `ironmill-torch` to `Cargo.toml` workspace members and `[workspace.dependencies]`:
   ```toml
   ironmill-torch = { path = "crates/ironmill-torch", version = "0.1.0" }
   ```
3. Create `crates/ironmill-torch/Cargo.toml` per spec §3.1. Dependencies:
   - `ironmill-core = { workspace = true }` 
   - `ironmill-compile = { workspace = true }`
   - `ironmill-inference = { workspace = true }`
   - `mil-rs = { workspace = true }`
   - `thiserror = { workspace = true }`
   - Feature flags: `default = ["metal"]`, `metal`, `ane`, `coreml`, `hf-tokenizer`, `async`
4. Create the source files per spec §4:
   - `src/lib.rs` — public API surface (spec §3.2)
   - `src/error.rs` — `TorchError` (spec §4.1)
   - `src/model.rs` — `Model`, `ModelBuilder` (spec §4.2, §4.3)
   - `src/gen_params.rs` — `GenParams` (spec §4.4)
   - `src/chat.rs` — `ChatSession`, `ChatSessionBuilder` (spec §4.5)
   - `src/text_output.rs` — `TextOutput`, `TextChunk`, `TextStream` (spec §4.6)

### Phase 2: Implement the types

The spec §4 has full type definitions. Implement them with these guidelines:

- `Model` stores `Box<dyn InferenceEngine>` (from `ironmill_inference::engine`), NOT `Box<dyn Any>`.
- `ModelBuilder::build()` is the key integration point. For now, implement `load_pretrained` and `load_compiled` for Metal only (`#[cfg(target_os = "macos")]`), with a fallback `UnsupportedDevice` error. Use:
  - `ironmill_compile::weights::SafeTensorsProvider` to load weights
  - `ironmill_inference::metal::{MetalInference, MetalConfig}` to create the engine
  - `ironmill_core::model_info::ModelInfo::from_config()` for metadata
  - Check what load methods actually exist on `MetalInference` on the `api-spec` branch — use what's available.
- `Model::generate()` should use `TokenStream` from `ironmill_inference::generate` with `CancellationToken`.
- `ChatSession::send_with_params()` must append to history AFTER successful generation (not before — the ironmill-core version had this bug).
- `GenParams::to_generate_request()` maps the user-facing params to `SamplerConfig` + `GenerateRequest`.
- `TextStream` wraps `TokenStream` and decodes each token via the `Tokenizer` trait.
- All public types must have `#[non_exhaustive]` where appropriate.
- Add `#![warn(missing_docs)]` to lib.rs.

### Phase 3: Remove moved types from ironmill-core

After ironmill-torch compiles, remove the types that moved (spec §2.1):
- Delete `crates/ironmill-core/src/model.rs`
- Delete `crates/ironmill-core/src/chat.rs`
- Delete `crates/ironmill-core/src/gen_params.rs`
- Delete `crates/ironmill-core/src/text_output.rs`
- Delete `crates/ironmill-core/src/error.rs`
- Remove `ModelError` from `ironmill-core/src/lib.rs` re-exports
- Remove `Model`, `ModelBuilder`, `ChatSession`, `ChatSessionBuilder`, `GenParams`, `TextOutput`, `TextChunk`, `TextStream` from `ironmill-core/src/lib.rs` re-exports
- KEEP: `Device`, `ModelInfo`, `Tokenizer`, `HfTokenizer`, `ChatMessage`, `TokenizerError`, `ane::*`, `gpu::*`, `weights::*`

After removing, check if anything in `ironmill-compile` or `ironmill-inference` imported the removed types. They should not — these types were only used by the CLI (which doesn't use them yet) and ironmill-core's own tests.

### Phase 4: Wire up ironmill-cli

Add `ironmill-torch = { workspace = true }` to `crates/ironmill-cli/Cargo.toml` dependencies. No need to add CLI commands that use it yet — just ensure the dependency is wired so future work can add a `run` / `generate` / `chat` subcommand.

### Phase 5: Verify

Run these commands and ensure they all pass:
```bash
cargo check --workspace                    # everything compiles
cargo test -p ironmill-torch --lib         # new crate tests pass
cargo test -p ironmill-core --lib          # core still passes after removals
cargo test -p ironmill-compile --lib       # compile unaffected
cargo test -p ironmill-inference --lib     # inference unaffected
```

## Important codebase conventions

- Edition 2024, rust-version 1.85
- `#![deny(unsafe_code)]` on inference crate; avoid unsafe everywhere
- Pre-commit hook runs `cargo fmt` + `cargo clippy`; use `--no-verify` on commits if clippy warns on `warn(missing_docs)` for stubs
- `InferenceEngine` trait is object-safe (Send, no associated types, no load()). Each backend has its own typed `load()` as an inherent method.
- `ModelConfig` uses builder pattern: `ModelConfig::new(arch).with_hidden_size(n)`
- All public enums carry `#[non_exhaustive]`
- Public traits never return `anyhow::Result` — use crate-specific errors
- Existing test counts on api-spec: mil-rs 500, compile 315, inference 261, core 12

## Key principle

Ironmill is **unreleased** — there are zero external consumers. Backward compatibility is not a concern. Prioritize clean design and simplicity over compatibility shims. When removing types from ironmill-core, delete them outright — no `#[deprecated]` re-exports, no facade wrappers, no transition period. If something needs to move, move it cleanly.

## What NOT to do

- Do NOT add `ironmill-inference` or `ironmill-compile` as deps of `ironmill-core`
- Do NOT modify `ironmill-inference` or `ironmill-compile` source code
- Do NOT modify `mil-rs` source code
- Do NOT implement actual GPU kernel dispatch in `ModelBuilder::build()` — use whatever load methods exist on the api-spec branch, stub with `todo!()` if the exact method signature doesn't match
- Do NOT add async runtime dependencies to the main crate (only behind `async` feature)
