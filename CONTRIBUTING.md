# Contributing to ironmill

Thank you for considering contributing! This guide covers the development
workflow for both `mil-rs` (library) and `ironmill-cli` (CLI tool).

## Setting up the development environment

1. **Install Rust 1.85+** via [rustup](https://rustup.rs/).
2. **Clone the repo** and build:
   ```bash
   git clone https://github.com/jafreck/ironmill.git
   cd ironmill
   cargo build --workspace
   cargo test --workspace
   ```
3. **(Optional) Install `protoc`** - only needed if you modify `.proto` files.
   The generated code is committed, so most contributors don't need this.
4. **(Optional) Xcode** - required on macOS for `xcrun coremlcompiler`
   integration tests (these are `#[ignore]`d by default).

## Project structure

```
crates/
  mil-rs/          # Core library
    src/
      ir/          # MIL intermediate representation
        passes/    # Optimization passes
      convert/     # ONNX ↔ MIL ↔ Proto conversion
      reader/      # File readers (.mlmodel, .mlpackage, .onnx)
      writer/      # File writers (.mlmodel, .mlpackage)
      validate.rs  # ANE compatibility checking
      compiler.rs  # xcrun coremlcompiler wrapper
      error.rs     # Error types
      proto/       # Generated protobuf code
    examples/      # Runnable examples
  ironmill-cli/  # CLI binary
tests/
  fixtures/        # Test model files (ONNX, mlmodel, mlpackage)
```

## Adding a new ONNX op mapping

ONNX → MIL conversion lives in `crates/mil-rs/src/convert/onnx_to_mil.rs`.

1. Add a new converter function:
   ```rust
   fn convert_my_op(node: &NodeProto) -> Result<Vec<Operation>> {
       // Map ONNX attributes to MIL IR operation fields.
       let op = Operation::new("mil_op_name", &node.output[0])
           .with_input("x", Value::Reference(node.input[0].clone()))
           .with_output(&node.output[0]);
       Ok(vec![op])
   }
   ```
2. Register it in the `convert_node` match statement in the same file.
3. Add a test in the `#[cfg(test)]` module at the bottom of the file.
4. If the op is ANE-compatible, add it to the `is_ane_supported` list in
   `crates/mil-rs/src/validate.rs`.

## Adding a new optimization pass

Passes live in `crates/mil-rs/src/ir/passes/`.

1. Create a new file, e.g. `my_pass.rs`.
2. Define a struct and implement `Pass`:
   ```rust
   use crate::ir::{Pass, Program};
   use crate::error::Result;

   /// Brief description of what the pass does.
   pub struct MyPass;

   impl Pass for MyPass {
       fn name(&self) -> &str { "my-pass" }

       fn run(&self, program: &mut Program) -> Result<()> {
           for func in program.functions.values_mut() {
               // Transform func.body.operations...
           }
           Ok(())
       }
   }
   ```
3. Register the module in `passes/mod.rs` and re-export the struct.
4. Add tests - at minimum, verify the pass is idempotent (running twice
   produces the same result).

## Test conventions

- Unit tests go in `#[cfg(test)] mod tests` at the bottom of each file.
- Integration tests that need fixture files use
  `Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/")`.
- Tests requiring macOS + Xcode should be `#[ignore]`.
- Run the full suite: `cargo test --workspace`.
- Run a single test: `cargo test -p mil-rs test_name`.

## Code style

- Use `cargo fmt` before committing.
- Run `cargo clippy --workspace` and fix warnings.
- Public items should have doc comments (`///`).
- Keep modules focused - prefer many small files over large ones.

## PR process

1. Fork and create a feature branch from `main`.
2. Make your changes with tests.
3. Ensure `cargo test --workspace`, `cargo clippy --workspace`, and
   `cargo doc --no-deps --workspace` all pass.
4. Open a PR with a clear description of what changed and why.
5. One approving review is required before merging.

## License

By contributing, you agree that your contributions will be licensed under the
Apache-2.0 license as the project.
