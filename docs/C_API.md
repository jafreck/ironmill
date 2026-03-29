# C API Guide

`mil-rs` exposes a C-compatible FFI so you can call the conversion pipeline
from C, Swift, C++, Go, or any language that can link to `extern "C"` symbols.

## Building the static library

Enable the `c-api` feature when building:

```bash
cargo build --release -p mil-rs --features c-api
```

This produces a static library in `target/release/`:

| Platform | File |
|----------|------|
| macOS / Linux | `libmil_rs.a` |
| Windows | `mil_rs.lib` |

## Generating the C header

Install [cbindgen](https://github.com/mozilla/cbindgen) and generate the
header from the crate's config:

```bash
# Install cbindgen (one-time)
cargo install cbindgen

# Generate the header
cbindgen --config crates/mil-rs/cbindgen.toml \
         --crate mil-rs \
         --output mil_rs.h
```

The generated `mil_rs.h` declares all opaque types and function prototypes.

## API overview

### Opaque handle types

| Type | Description | Free with |
|------|-------------|-----------|
| `MilModel *` | A loaded ML model (ONNX or CoreML) | `mil_model_free()` |
| `MilProgram *` | A MIL IR program | `mil_program_free()` |
| `MilValidationReport *` | ANE validation report | `mil_validation_report_free()` |

### Functions

**Lifecycle — reading models:**

| Function | Returns |
|----------|---------|
| `mil_read_onnx(path)` | `MilModel *` (ONNX) |
| `mil_read_mlmodel(path)` | `MilModel *` (CoreML) |
| `mil_read_mlpackage(path)` | `MilModel *` (CoreML) |
| `mil_model_free(model)` | void |

**Conversion:**

| Function | Returns |
|----------|---------|
| `mil_onnx_to_program(model)` | `MilProgram *` |
| `mil_program_to_model(program, spec_version)` | `MilModel *` (CoreML) |
| `mil_program_free(program)` | void |

**Writing:**

| Function | Returns |
|----------|---------|
| `mil_write_mlmodel(model, path)` | `int` (0 = success, −1 = error) |
| `mil_write_mlpackage(model, path)` | `int` (0 = success, −1 = error) |

**Compilation:**

| Function | Returns |
|----------|---------|
| `mil_compile_model(input, output_dir)` | `char *` path (free with `mil_string_free`) |
| `mil_is_compiler_available()` | `bool` |

**Validation:**

| Function | Returns |
|----------|---------|
| `mil_validate_ane(program)` | `MilValidationReport *` |
| `mil_validation_report_to_json(report)` | `char *` JSON (free with `mil_string_free`) |
| `mil_validation_report_free(report)` | void |

**Error handling:**

| Function | Returns |
|----------|---------|
| `mil_last_error()` | `const char *` (do **not** free) |
| `mil_string_free(s)` | void |

## Error handling

Every function that can fail returns either **null** (pointer-returning) or
**−1** (`int`-returning). On failure a human-readable message is stored in a
thread-local. Retrieve it with `mil_last_error()`:

```c
MilModel *model = mil_read_onnx("model.onnx");
if (!model) {
    fprintf(stderr, "error: %s\n", mil_last_error());
    return 1;
}
```

The pointer from `mil_last_error()` is **borrowed** — do not free it. It
remains valid until the next failing call on the same thread.

## Memory management rules

1. Every `*_free()` function accepts null safely (no-op).
2. **Never double-free** — once you call `mil_model_free(model)`, do not use
   `model` again.
3. Strings returned by `mil_compile_model()` and
   `mil_validation_report_to_json()` are **heap-allocated** — you must free
   them with `mil_string_free()`.
4. The pointer from `mil_last_error()` is **not** heap-allocated — do **not**
   pass it to `mil_string_free()`.

## Complete C example

```c
#include <stdio.h>
#include <stdlib.h>
#include "mil_rs.h"

int main(void) {
    // 1. Read an ONNX model
    MilModel *onnx = mil_read_onnx("model.onnx");
    if (!onnx) {
        fprintf(stderr, "read failed: %s\n", mil_last_error());
        return 1;
    }

    // 2. Convert ONNX → MIL Program
    MilProgram *program = mil_onnx_to_program(onnx);
    if (!program) {
        fprintf(stderr, "conversion failed: %s\n", mil_last_error());
        mil_model_free(onnx);
        return 1;
    }

    // 3. Convert MIL Program → CoreML Model (spec version 7)
    MilModel *coreml = mil_program_to_model(program, 7);
    if (!coreml) {
        fprintf(stderr, "program_to_model failed: %s\n", mil_last_error());
        mil_program_free(program);
        mil_model_free(onnx);
        return 1;
    }

    // 4. Write as .mlpackage
    if (mil_write_mlpackage(coreml, "output.mlpackage") != 0) {
        fprintf(stderr, "write failed: %s\n", mil_last_error());
        mil_model_free(coreml);
        mil_program_free(program);
        mil_model_free(onnx);
        return 1;
    }
    printf("Wrote output.mlpackage\n");

    // 5. (Optional) Compile to .mlmodelc
    if (mil_is_compiler_available()) {
        char *compiled = mil_compile_model("output.mlpackage", ".");
        if (compiled) {
            printf("Compiled to %s\n", compiled);
            mil_string_free(compiled);
        }
    }

    // 6. (Optional) Validate ANE compatibility
    MilValidationReport *report = mil_validate_ane(program);
    if (report) {
        char *json = mil_validation_report_to_json(report);
        if (json) {
            printf("Validation: %s\n", json);
            mil_string_free(json);
        }
        mil_validation_report_free(report);
    }

    // 7. Clean up — free every handle exactly once
    mil_model_free(coreml);
    mil_program_free(program);
    mil_model_free(onnx);

    return 0;
}
```

### Compiling the C example

```bash
# Generate the header
cbindgen --config crates/mil-rs/cbindgen.toml --crate mil-rs --output mil_rs.h

# Build the static library
cargo build --release -p mil-rs --features c-api

# Compile and link
cc example.c -o example \
   -I. \
   -Ltarget/release \
   -lmil_rs \
   -framework CoreFoundation \
   -framework Security
```

On Linux, replace the `-framework` flags with `-lpthread -ldl -lm`.

## Swift usage

You can call the C API from Swift using a bridging header:

### Bridging header (`mil_rs-Bridging-Header.h`)

```c
#include "mil_rs.h"
```

### Swift code

```swift
import Foundation

func convertModel(onnxPath: String, outputPath: String) throws {
    // Read the ONNX model
    guard let model = mil_read_onnx(onnxPath) else {
        let err = String(cString: mil_last_error())
        throw NSError(domain: "mil-rs", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: err])
    }
    defer { mil_model_free(model) }

    // Convert to MIL Program
    guard let program = mil_onnx_to_program(model) else {
        let err = String(cString: mil_last_error())
        throw NSError(domain: "mil-rs", code: 2,
                      userInfo: [NSLocalizedDescriptionKey: err])
    }
    defer { mil_program_free(program) }

    // Convert to CoreML Model
    guard let coreml = mil_program_to_model(program, 7) else {
        let err = String(cString: mil_last_error())
        throw NSError(domain: "mil-rs", code: 3,
                      userInfo: [NSLocalizedDescriptionKey: err])
    }
    defer { mil_model_free(coreml) }

    // Write .mlpackage
    let rc = mil_write_mlpackage(coreml, outputPath)
    if rc != 0 {
        let err = String(cString: mil_last_error())
        throw NSError(domain: "mil-rs", code: 4,
                      userInfo: [NSLocalizedDescriptionKey: err])
    }
    print("Wrote \(outputPath)")
}
```

The `defer` pattern ensures handles are freed even if an error occurs,
preventing leaks.

## Linking in Xcode

1. Add `libmil_rs.a` to **Build Phases → Link Binary With Libraries**.
2. Add the header search path in **Build Settings → Header Search Paths**.
3. Set the bridging header in **Build Settings → Objective-C Bridging Header**.
