//! Compile CoreML models using `xcrun coremlcompiler`.
//!
//! This module shells out to Apple's `coremlcompiler` tool to compile
//! `.mlpackage` or `.mlmodel` files into `.mlmodelc` bundles that can be
//! loaded at runtime on Apple platforms.
//!
//! # Requirements
//!
//! macOS with Xcode or Command Line Tools installed.  On other platforms the
//! functions return an error rather than failing to compile.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::error::{MilError, Result};

/// Check if `xcrun coremlcompiler` is available on this system.
pub fn is_compiler_available() -> bool {
    Command::new("xcrun")
        .args(["--find", "coremlcompiler"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Compile a `.mlpackage` or `.mlmodel` to `.mlmodelc` using xcrun.
///
/// The compiled model is placed in `output_dir` as `<model_name>.mlmodelc/`.
/// Returns the path to the compiled `.mlmodelc` directory on success.
///
/// # Errors
///
/// Returns an error if:
/// - `xcrun coremlcompiler` is not found (Xcode / Command Line Tools not installed)
/// - `input` does not exist
/// - the compiler exits with a non-zero status
/// - the expected `.mlmodelc` output is not produced
///
/// # Platform
///
/// This function is only available on macOS with Xcode or Command Line Tools
/// installed.  On other platforms it returns a descriptive error instead of
/// using `#[cfg]` gates.
pub fn compile_model(
    input: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
) -> Result<PathBuf> {
    let input = input.as_ref();
    let output_dir = output_dir.as_ref();

    if !input.exists() {
        return Err(MilError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("input path does not exist: {}", input.display()),
        )));
    }

    if !is_compiler_available() {
        return Err(MilError::Validation(
            "xcrun coremlcompiler not found. Install Xcode or Command Line Tools.".into(),
        ));
    }

    std::fs::create_dir_all(output_dir)?;

    let output = Command::new("xcrun")
        .args(["coremlcompiler", "compile"])
        .arg(input)
        .arg(output_dir)
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(MilError::Validation(format!(
            "coremlcompiler failed: {stderr}"
        )));
    }

    // The compiled model is placed at output_dir/<stem>.mlmodelc
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let compiled = output_dir.join(format!("{stem}.mlmodelc"));

    if compiled.exists() {
        Ok(compiled)
    } else {
        Err(MilError::Validation(
            "compiled model not found in output directory".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_compiler_availability() {
        // Verify the function runs without panicking.
        let available = is_compiler_available();
        println!("xcrun coremlcompiler available: {available}");
    }

    #[test]
    fn compile_nonexistent_input_returns_error() {
        let result = compile_model("does_not_exist.mlpackage", "out");
        assert!(result.is_err());
    }

    #[test]
    #[ignore] // Requires macOS with Xcode
    fn compile_mlmodel_roundtrip() {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures");
        let mlmodel = fixtures.join("MobileNet.mlmodel");
        if !mlmodel.exists() {
            eprintln!("skipping: MobileNet.mlmodel fixture not found");
            return;
        }

        let tmp = tempfile::tempdir().expect("failed to create temp dir");
        let compiled = compile_model(&mlmodel, tmp.path())
            .expect("compile_model failed");

        assert!(compiled.exists(), "compiled .mlmodelc should exist");
        assert!(
            compiled.extension().and_then(|e| e.to_str()) == Some("mlmodelc"),
            "output should have .mlmodelc extension"
        );
    }

    #[test]
    #[ignore] // Requires macOS with Xcode
    fn compile_mlpackage_roundtrip() {
        use crate::{read_mlmodel, write_mlpackage};

        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures");
        let mlmodel = fixtures.join("MobileNet.mlmodel");
        if !mlmodel.exists() {
            eprintln!("skipping: MobileNet.mlmodel fixture not found");
            return;
        }

        let tmp = tempfile::tempdir().expect("failed to create temp dir");

        // Read .mlmodel → write as .mlpackage → compile → verify .mlmodelc
        let model = read_mlmodel(&mlmodel).expect("read_mlmodel failed");
        let pkg_path = tmp.path().join("MobileNet.mlpackage");
        write_mlpackage(&model, &pkg_path).expect("write_mlpackage failed");

        let compiled = compile_model(&pkg_path, tmp.path())
            .expect("compile_model failed");

        assert!(compiled.exists(), "compiled .mlmodelc should exist");
        assert!(
            compiled.extension().and_then(|e| e.to_str()) == Some("mlmodelc"),
            "output should have .mlmodelc extension"
        );
    }
}
