# Agent Implementation Guide

This document provides everything a sub-agent needs to implement tasks on `ironmill`.

## What This Project Is

`ironmill` is a Rust-native CoreML model converter. It lets you convert ONNX models to
Apple's CoreML format without Python. Think of it as the Rust equivalent of Apple's
`coremltools` Python package, but focused on conversion rather than training.

**Read these docs for full context** (in order of importance):
1. `docs/research/integration-strategy.md` — how this fits the Rust ecosystem
2. `docs/research/value-proposition.md` — why this exists (honest assessment)
3. `docs/research/ane-research.md` — the macOS AI landscape
4. `docs/research/competitive-analysis.md` — every competitor examined
5. `docs/IMPLEMENTATION_PLAN.md` — the full phased task breakdown

## Project Structure

```
ironmill/
├── Cargo.toml                    # Workspace root
├── crates/
│   ├── mil-rs/                   # Foundation crate: CoreML protobuf + MIL IR
│   │   ├── Cargo.toml
│   │   ├── build.rs              # (to be created) prost-build codegen
│   │   ├── proto/                # (to be created) vendored .proto files
│   │   └── src/
│   │       ├── lib.rs            # Crate root, re-exports
│   │       ├── error.rs          # MilError enum + Result type alias
│   │       ├── proto/            # (to be created) generated protobuf module
│   │       │   └── mod.rs
│   │       ├── ir/               # MIL intermediate representation
│   │       │   ├── mod.rs
│   │       │   ├── graph.rs      # Graph (exists, needs expansion)
│   │       │   ├── operation.rs  # Operation (exists)
│   │       │   ├── tensor.rs     # TensorType, ScalarType (exists)
│   │       │   ├── types.rs      # Value enum (exists)
│   │       │   ├── program.rs    # (to be created) Program, Function, Block
│   │       │   └── pass.rs       # (to be created) optimization pass trait
│   │       ├── reader/           # (to be created) model file readers
│   │       │   ├── mod.rs
│   │       │   ├── mlmodel.rs    # .mlmodel reader
│   │       │   └── mlpackage.rs  # .mlpackage reader
│   │       ├── writer/           # (to be created) model file writers
│   │       │   ├── mod.rs
│   │       │   ├── mlmodel.rs
│   │       │   └── mlpackage.rs
│   │       └── convert/          # (to be created) format converters
│   │           ├── mod.rs
│   │           ├── proto_to_ir.rs
│   │           └── ir_to_proto.rs
│   └── ironmill-cli/           # CLI tool
│       ├── Cargo.toml
│       └── src/
│           └── main.rs           # clap-based CLI (exists, scaffold)
├── docs/
│   ├── IMPLEMENTATION_PLAN.md    # Detailed task breakdown
│   └── research/                 # Background research
│       ├── ane-research.md
│       ├── competitive-analysis.md
│       ├── integration-strategy.md
│       └── value-proposition.md
├── tests/                        # (to be created) integration tests
│   └── fixtures/                 # (to be created) sample .mlmodel/.onnx files
└── scripts/                      # Build/dev helper scripts
    └── vendor-protos.sh          # (to be created) downloads Apple proto files
```

## Coding Conventions

### Error handling
- Use `thiserror` for the `MilError` enum in `crates/mil-rs/src/error.rs`
- All public functions return `Result<T, MilError>` (aliased as `crate::error::Result<T>`)
- The CLI uses `anyhow::Result` for top-level error handling
- Never `unwrap()` in library code; `unwrap()` is acceptable only in tests

### Module organization
- One type per file when the type has significant impl blocks
- Re-export key types from `mod.rs` and from `lib.rs`
- Keep `pub use` paths shallow — users should write `mil_rs::Graph`, not
  `mil_rs::ir::graph::Graph`

### Naming
- Rust standard: `snake_case` functions, `PascalCase` types, `SCREAMING_SNAKE` constants
- MIL operation types use the same names as Apple's MIL spec (e.g., `conv`, `matmul`, `relu`)
- Protobuf-generated types live in `mil_rs::proto` and keep their original proto names

### Documentation
- All public types and functions must have doc comments
- Include `# Examples` in doc comments for key APIs
- Use `///` for item docs, `//!` only for module-level docs

### Testing
- Unit tests go in the same file as the code (`#[cfg(test)] mod tests`)
- Integration tests go in `tests/` at the workspace root
- Test fixtures (sample models) go in `tests/fixtures/`
- Use `#[ignore]` for tests that require macOS-only tools (xcrun)

### Dependencies
- Workspace dependencies are declared in root `Cargo.toml` under `[workspace.dependencies]`
- Crate-level `Cargo.toml` uses `dep = { workspace = true }`
- Minimize external dependencies — prefer std where reasonable

## Key External References

### CoreML protobuf schemas (BSD-3 licensed)
- Repository: https://github.com/apple/coremltools/tree/main/mlmodel/format
- Key files to vendor:
  - `Model.proto` — top-level model container
  - `MIL.proto` — Model Intermediate Language representation
  - `NeuralNetwork.proto` — legacy neural network format
  - `FeatureTypes.proto` — input/output feature type definitions
  - `DataStructures.proto` — shared data structures
  - `Parameters.proto` — training/update parameters
  - All other `.proto` files in that directory (they compile for free)

### MIL operations reference
- Operations catalog: https://deepwiki.com/apple/coremltools/5.2-mil-operations
- MIL spec in coremltools: https://apple.github.io/coremltools/docs-guides/source/model-intermediate-language.html

### ONNX protobuf schema
- Repository: https://github.com/onnx/onnx/tree/main/onnx
- Key file: `onnx.proto3` (or `onnx.proto`)

### .mlpackage directory format
- An `.mlpackage` is a directory containing:
  - `Manifest.json` — lists items and their paths
  - `Data/` subdirectory with:
    - Model spec (protobuf, usually named by hash)
    - Weight files (binary blobs)
- Reference: https://developer.apple.com/documentation/coreml/updating-a-model-file-to-a-model-package

### CoreML model compilation
- `xcrun coremlcompiler compile input.mlpackage output_dir/`
- Requires Xcode or Command Line Tools on macOS
- Produces `.mlmodelc` directory (compiled, ready for runtime)

## How to Build and Test

```bash
# Build everything
cargo build

# Run tests
cargo test

# Run the CLI
cargo run -p ironmill-cli -- --help
cargo run -p ironmill-cli -- compile model.onnx
cargo run -p ironmill-cli -- inspect model.mlmodel

# Check docs compile
cargo doc --no-deps

# Lint
cargo clippy -- -D warnings
```

## Task Acceptance Criteria (General)

A task is done when:
1. The code compiles with no warnings (`cargo clippy -- -D warnings`)
2. All new public APIs have doc comments
3. Unit tests exist for the happy path and at least one error case
4. Existing tests still pass (`cargo test`)
5. The implementation matches what the IMPLEMENTATION_PLAN.md describes

## Test Fixtures

Sample models are in `tests/fixtures/`. Large binary files (`.onnx`, `.mlmodel`) are
gitignored — run `./scripts/download-fixtures.sh` to fetch them.

| File | Format | Size | Purpose |
|------|--------|------|---------|
| `mnist.onnx` | ONNX | 26KB | Tiny model for ONNX reader/conversion tests |
| `squeezenet1.1.onnx` | ONNX | 4.7MB | Real CNN for conversion validation |
| `MobileNet.mlmodel` | CoreML | 16MB | Real CoreML model for reader/writer tests |
| `simple.mlpackage/` | CoreML dir | <1KB | Minimal .mlpackage structure for reader tests |

## Agent Dispatch Rules

### Sequencing
Tasks have dependencies tracked externally. Within each phase, tasks must be done in
dependency order. The dependency chain for Phase 1 is:

```
p1-vendor-protos
  └─▶ p1-prost-codegen
        ├─▶ p1-mlmodel-reader
        │     ├─▶ p1-mlpackage-reader
        │     ├─▶ p1-writer
        │     └─▶ p1-proto-ir-conv ◀── p1-ir-types
        └─▶ p1-ir-types
              └─▶ p1-proto-ir-conv
                    └─▶ p1-tests-ci ◀── p1-writer, p1-mlpackage-reader
                          └─▶ p1-publish
```

### Parallelism
- **Safe to parallelize**: Tasks in different phases (if all deps are met). Also:
  `p1-ir-types` and `p1-mlmodel-reader` can run in parallel (different files).
  `p1-writer` and `p1-mlpackage-reader` can run in parallel (different files).
- **NOT safe to parallelize**: Tasks that modify the same files (`lib.rs`, `Cargo.toml`,
  `mod.rs`). Most Phase 1 tasks touch `lib.rs` to add module declarations — run them
  sequentially or have the later task check for/merge the module declarations.

### Context for each agent
When dispatching a task to a sub-agent, provide:
1. The task ID and full description from the task database
2. A pointer to `docs/AGENT_GUIDE.md` for conventions and structure
3. A pointer to `docs/IMPLEMENTATION_PLAN.md` for the specific task section
4. Which files the previous task created (so the agent knows what exists)

### Commit convention
Each task should be a single commit with message format:
```
feat(mil-rs): <short description>

<what was implemented and why>

Task: <task-id>
Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
```
