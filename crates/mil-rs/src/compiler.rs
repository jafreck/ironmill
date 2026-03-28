//! Compile CoreML models using `xcrun coremlcompiler` or direct ANE FFI.
//!
//! This module shells out to Apple's `coremlcompiler` tool to compile
//! `.mlpackage` or `.mlmodel` files into `.mlmodelc` bundles that can be
//! loaded at runtime on Apple platforms.
//!
//! When the `ane-direct` feature is enabled, an alternative backend is
//! available that calls the private `_ANECompiler` Objective-C class
//! directly, bypassing `xcrun`.  See [`Backend`] for details.
//!
//! # Requirements
//!
//! macOS with Xcode or Command Line Tools installed.  On other platforms the
//! functions return an error rather than failing to compile.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::error::{MilError, Result};

/// Which compilation backend to use.
///
/// The default is [`Xcrun`](Backend::Xcrun), which shells out to
/// `xcrun coremlcompiler`.  When the `ane-direct` feature is enabled,
/// [`AneDirect`](Backend::AneDirect) calls Apple's private `_ANECompiler`
/// class via Objective-C FFI for potentially faster compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// Shell out to `xcrun coremlcompiler` (the default, stable path).
    #[default]
    Xcrun,

    /// Call the private `_ANECompiler` class directly via Objective-C FFI.
    ///
    /// # ⚠️ Experimental
    ///
    /// This backend uses **undocumented Apple private APIs** that can break
    /// across macOS updates.  It is not suitable for App Store distribution.
    /// Requires the `ane-direct` feature flag.
    AneDirect,
}

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
pub fn compile_model(input: impl AsRef<Path>, output_dir: impl AsRef<Path>) -> Result<PathBuf> {
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

/// Compile a model using the specified [`Backend`].
///
/// When `backend` is [`Backend::Xcrun`], this delegates to [`compile_model`].
/// When `backend` is [`Backend::AneDirect`] and the `ane-direct` feature is
/// enabled, it uses the direct FFI path.  If `ane-direct` is not enabled
/// but `AneDirect` is requested, this returns an error.
///
/// # Errors
///
/// Same errors as [`compile_model`], plus feature-gate errors for the
/// `AneDirect` backend.
pub fn compile_model_with_backend(
    input: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
    backend: Backend,
) -> Result<PathBuf> {
    match backend {
        Backend::Xcrun => compile_model(input, output_dir),
        Backend::AneDirect => compile_model_ane_direct(input, output_dir),
    }
}

#[cfg(feature = "ane-direct")]
fn compile_model_ane_direct(
    input: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
) -> Result<PathBuf> {
    use crate::ffi::ane::AneCompiler;
    AneCompiler::compile(input.as_ref(), output_dir.as_ref()).map_err(Into::into)
}

#[cfg(not(feature = "ane-direct"))]
fn compile_model_ane_direct(
    _input: impl AsRef<Path>,
    _output_dir: impl AsRef<Path>,
) -> Result<PathBuf> {
    Err(MilError::Validation(
        "The 'ane-direct' backend requires the `ane-direct` feature flag. \
         Rebuild with `--features ane-direct`."
            .into(),
    ))
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
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
        let mlmodel = fixtures.join("MobileNet.mlmodel");
        if !mlmodel.exists() {
            eprintln!("skipping: MobileNet.mlmodel fixture not found");
            return;
        }

        let tmp = tempfile::tempdir().expect("failed to create temp dir");
        let compiled = compile_model(&mlmodel, tmp.path()).expect("compile_model failed");

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

        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
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

        let compiled = compile_model(&pkg_path, tmp.path()).expect("compile_model failed");

        assert!(compiled.exists(), "compiled .mlmodelc should exist");
        assert!(
            compiled.extension().and_then(|e| e.to_str()) == Some("mlmodelc"),
            "output should have .mlmodelc extension"
        );
    }
}
