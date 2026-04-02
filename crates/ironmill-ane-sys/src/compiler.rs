//! ANE compilation — backward-compatible thin wrapper over [`crate::model`].
//!
//! All compile/load logic now lives in `model.rs`.  This module delegates to
//! it and converts between the new `InMemoryModel` and the legacy
//! `CompiledProgram` handle type.
//!
//! # ⚠️ Private API — see crate-level docs for risks and caveats.

use std::path::{Path, PathBuf};

use crate::CompiledProgram;
use crate::error::AneSysError;
use crate::objc::{ane_framework, get_class};

/// Namespace for ANE compilation via the private `_ANEInMemoryModel` API.
///
/// This struct holds no state — it is a namespace for the static compilation
/// methods.  All interaction with the Objective-C runtime happens inside the
/// individual method implementations.
pub struct AneCompiler {
    _private: (),
}

impl AneCompiler {
    /// Check whether the ANE framework is available on this system.
    pub fn is_available() -> bool {
        if ane_framework().is_err() {
            return false;
        }
        get_class("_ANEInMemoryModel").is_ok()
    }

    /// Compile a `.mlpackage` (or `.mlmodelc`) to an ANE-optimized bundle.
    ///
    /// This legacy entry point is not used by the direct-ANE backend
    /// (which uses `compile_mil_text` instead), but is retained for API
    /// compatibility.
    pub fn compile(mlpackage_path: &Path, output_dir: &Path) -> Result<PathBuf, AneSysError> {
        if !mlpackage_path.exists() {
            return Err(AneSysError::InvalidInput(format!(
                "input path does not exist: {}",
                mlpackage_path.display()
            )));
        }

        std::fs::create_dir_all(output_dir)?;

        Err(AneSysError::CompilationFailed(
            "mlpackage compilation not supported — use compile_mil_text instead".into(),
        ))
    }

    /// Compile with incremental/delta support, reusing cached artifacts.
    pub fn compile_incremental(
        mlpackage_path: &Path,
        output_dir: &Path,
        cache_dir: &Path,
    ) -> Result<PathBuf, AneSysError> {
        if !mlpackage_path.exists() {
            return Err(AneSysError::InvalidInput(format!(
                "input path does not exist: {}",
                mlpackage_path.display()
            )));
        }

        std::fs::create_dir_all(output_dir)?;
        std::fs::create_dir_all(cache_dir)?;

        let _ = cache_dir;

        Self::compile(mlpackage_path, output_dir)
    }

    /// Compile MIL text + weight blob into an ANE-loaded model.
    ///
    /// Returns a [`CompiledProgram`] wrapping the retained
    /// `_ANEInMemoryModel` handle.
    ///
    /// Delegates to [`crate::model::compile_mil_text`].
    pub fn compile_mil_text(
        mil_text: &str,
        weights: &[(&str, &[u8])],
    ) -> Result<CompiledProgram, AneSysError> {
        let model = crate::model::compile_mil_text(mil_text, weights)?;
        Ok(CompiledProgram::from_model(model))
    }

    /// Number of compilations performed in this process.
    pub fn compile_count() -> usize {
        crate::model::compile_count()
    }

    /// Remaining compile budget before hitting the ~119 limit.
    pub fn remaining_budget() -> usize {
        crate::model::remaining_budget()
    }

    /// Create a new ANE program by reusing a donor's compiled artifacts.
    ///
    /// Delegates to [`crate::model::patch_weights`].
    pub fn patch_weights(
        donor: &CompiledProgram,
        mil_text: &str,
        weights: &[(&str, &[u8])],
    ) -> Result<CompiledProgram, AneSysError> {
        let donor_model = donor.as_model();
        let model = crate::model::patch_weights(donor_model, mil_text, weights)?;
        Ok(CompiledProgram::from_model(model))
    }
}

/// Create a BLOBFILE in Orion's format from raw weight data.
///
/// Delegates to [`crate::model::make_blobfile`].
pub fn make_blobfile(data: &[u8]) -> Result<Vec<u8>, AneSysError> {
    crate::model::make_blobfile(data)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use super::*;

    #[test]
    fn ane_compiler_availability_check_does_not_panic() {
        let _available = AneCompiler::is_available();
    }

    #[test]
    fn compile_nonexistent_input_returns_error() {
        let result = AneCompiler::compile(Path::new("does_not_exist.mlpackage"), Path::new("out"));
        assert!(result.is_err());
        match result.unwrap_err() {
            AneSysError::InvalidInput(_) => {}
            other => panic!("expected InvalidInput, got: {other}"),
        }
    }

    #[test]
    fn compile_incremental_nonexistent_input_returns_error() {
        let result = AneCompiler::compile_incremental(
            Path::new("does_not_exist.mlpackage"),
            Path::new("out"),
            Path::new("cache"),
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            AneSysError::InvalidInput(_) => {}
            other => panic!("expected InvalidInput, got: {other}"),
        }
    }

    #[test]
    fn compile_count_starts_at_zero_or_accumulates() {
        let count = AneCompiler::compile_count();
        assert!(count < 200, "count looks unreasonably high: {count}");
    }

    #[test]
    fn compile_mil_text_budget_tracking() {
        let before = AneCompiler::compile_count();
        let result = AneCompiler::compile_mil_text("program test {}", &[]);
        let _ = (result, before);
    }

    #[test]
    fn remaining_budget_decreases() {
        let remaining = AneCompiler::remaining_budget();
        let count = AneCompiler::compile_count();
        assert_eq!(
            remaining,
            119_usize.saturating_sub(count),
            "remaining budget should equal limit minus count"
        );
    }

    #[test]
    fn compile_mil_text_empty_text_returns_error() {
        let result = AneCompiler::compile_mil_text("", &[]);
        assert!(result.is_err());
        match result.unwrap_err() {
            AneSysError::InvalidInput(msg) => {
                assert!(msg.contains("empty"), "expected 'empty' in message: {msg}");
            }
            other => panic!("expected InvalidInput, got: {other}"),
        }
    }

    #[test]
    fn budget_exhausted_error_display() {
        let err = AneSysError::BudgetExhausted { count: 119 };
        let msg = format!("{err}");
        assert!(
            msg.contains("119"),
            "error message should contain count: {msg}"
        );
        assert!(
            msg.contains("budget"),
            "error message should mention budget: {msg}"
        );
    }

    #[test]
    fn patch_weights_null_donor_returns_error() {
        let donor = unsafe { CompiledProgram::from_raw(std::ptr::null_mut()) };
        let result = AneCompiler::patch_weights(&donor, "program test {}", &[]);
        assert!(result.is_err());
        match result.unwrap_err() {
            AneSysError::InvalidInput(msg) => {
                assert!(msg.contains("null"), "expected 'null' in message: {msg}");
            }
            other => panic!("expected InvalidInput, got: {other}"),
        }
        std::mem::forget(donor);
    }

    #[test]
    fn patch_weights_empty_text_returns_error() {
        let dummy = unsafe { CompiledProgram::from_raw(0x1 as *mut c_void) };
        let result = AneCompiler::patch_weights(&dummy, "", &[]);
        assert!(result.is_err());
        match result.unwrap_err() {
            AneSysError::InvalidInput(msg) => {
                assert!(msg.contains("empty"), "expected 'empty' in message: {msg}");
            }
            other => panic!("expected InvalidInput, got: {other}"),
        }
        std::mem::forget(dummy);
    }
}
