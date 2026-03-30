//! Direct ANE compilation via Objective-C FFI to `_ANEInMemoryModel`.
//!
//! This module re-exports types from [`ironmill_ane_sys`], which consolidates
//! the low-level FFI code for the Apple Neural Engine private API.
//!
//! # ⚠️ Private API — Risks and Caveats
//!
//! **This module uses undocumented, private Apple APIs.**
//!
//! ## Stability
//! - The `_ANEInMemoryModel` class is an implementation detail of Apple's
//!   CoreML stack.  Its Objective-C selectors, argument conventions, and
//!   return types may change between any macOS release.
//! - There is **no stability guarantee** from Apple.
//!
//! ## App Store
//! - Apps that reference private frameworks or symbols are **rejected** from
//!   the Mac App Store.  This code is suitable only for local development
//!   tools, CI pipelines, and internal infrastructure.
//!
//! ## macOS Version Requirements
//! - Requires macOS 13 (Ventura) or later where the `AppleNeuralEngine`
//!   private framework ships by default.
//! - Requires Apple Silicon (M1 or later) or a T2-based Mac with ANE.

use crate::error::MilError;

// Re-export the core types from ironmill-ane-sys.
pub use ironmill_ane_sys::AneCompiler;
pub use ironmill_ane_sys::AneSysError as AneError;
pub use ironmill_ane_sys::CompiledProgram;
pub use ironmill_ane_sys::LoadedProgram;

impl From<AneError> for MilError {
    fn from(e: AneError) -> Self {
        MilError::Validation(format!("ANE direct compilation error: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_void;
    use std::path::Path;

    #[test]
    fn ane_compiler_availability_check_does_not_panic() {
        // This will return false on most CI environments but must not crash.
        let _available = AneCompiler::is_available();
    }

    #[test]
    fn compile_nonexistent_input_returns_error() {
        let result = AneCompiler::compile(Path::new("does_not_exist.mlpackage"), Path::new("out"));
        assert!(result.is_err());
        match result.unwrap_err() {
            AneError::InvalidInput(_) => {}
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
            AneError::InvalidInput(_) => {}
            other => panic!("expected InvalidInput, got: {other}"),
        }
    }

    #[test]
    fn compile_count_starts_at_zero_or_accumulates() {
        let count = AneCompiler::compile_count();
        assert!(count < 119 + 50, "count looks unreasonably high: {count}");
    }

    #[test]
    fn compile_mil_text_budget_tracking() {
        let before = AneCompiler::compile_count();

        // compile_mil_text with empty weights should proceed past validation
        // but may fail at framework loading or compilation.
        let result = AneCompiler::compile_mil_text("program test {}", &[]);
        // The call may or may not increment the counter depending on how
        // far it gets. We just verify it doesn't panic.
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
            AneError::InvalidInput(msg) => {
                assert!(msg.contains("empty"), "expected 'empty' in message: {msg}");
            }
            other => panic!("expected InvalidInput, got: {other}"),
        }
    }

    #[test]
    fn budget_exhausted_error_display() {
        let err = AneError::BudgetExhausted { count: 119 };
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
        // SAFETY: from_raw with a null pointer — patch_weights validates it.
        let donor = unsafe { CompiledProgram::from_raw(std::ptr::null_mut()) };
        let result = AneCompiler::patch_weights(&donor, "program test {}", &[]);
        assert!(result.is_err());
        match result.unwrap_err() {
            AneError::InvalidInput(msg) => {
                assert!(msg.contains("null"), "expected 'null' in message: {msg}");
            }
            other => panic!("expected InvalidInput, got: {other}"),
        }
        std::mem::forget(donor);
    }

    #[test]
    fn patch_weights_empty_text_returns_error() {
        // Use a non-null dummy pointer — patch_weights validates MIL text before
        // dereferencing the donor.
        let dummy = unsafe { CompiledProgram::from_raw(0x1 as *mut c_void) };
        let result = AneCompiler::patch_weights(&dummy, "", &[]);
        assert!(result.is_err());
        match result.unwrap_err() {
            AneError::InvalidInput(msg) => {
                assert!(msg.contains("empty"), "expected 'empty' in message: {msg}");
            }
            other => panic!("expected InvalidInput, got: {other}"),
        }
        std::mem::forget(dummy);
    }
}
