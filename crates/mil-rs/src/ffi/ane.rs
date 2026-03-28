//! Direct ANE compilation via Objective-C FFI to `_ANECompiler`.
//!
//! This module wraps Apple's private `_ANECompiler` class from the
//! `AppleNeuralEngine` framework to compile CoreML models directly to ANE
//! bytecode without shelling out to `xcrun coremlcompiler`.
//!
//! # Production Use
//!
//! For production-quality Objective-C interop, prefer the [`objc2`] crate
//! ecosystem over raw FFI.  This module uses raw `extern "C"` declarations
//! because it targets a *private* framework with no public headers — the
//! `objc2` bindings generator cannot help here.  If Apple ever publishes a
//! public `ANECompiler` API, migrate to `objc2` immediately.
//!
//! [`objc2`]: https://crates.io/crates/objc2
//!
//! # ⚠️ Private API — Risks and Caveats
//!
//! **This module uses undocumented, private Apple APIs.**
//!
//! ## Stability
//! - The `_ANECompiler` class is an implementation detail of Apple's CoreML
//!   stack.  Its Objective-C selectors, argument conventions, and return
//!   types may change between any macOS release — including minor updates.
//! - There is **no stability guarantee** from Apple.  A macOS update can
//!   silently break this code.
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
//!
//! ## Security
//! - The Objective-C runtime calls here are inherently `unsafe`.  We
//!   minimize the unsafe surface but cannot guarantee correctness if Apple
//!   changes the ABI.
//!
//! # Architecture
//!
//! The flow mirrors what `xcrun coremlcompiler compile` does internally:
//!
//! 1. Load the `AppleNeuralEngine.framework` (or `ANECompiler.framework`)
//! 2. Obtain the `_ANECompiler` class via `objc_getClass`
//! 3. Send `compileModel:toPath:options:error:` (or similar) to compile
//!    an `.mlmodelc` bundle into ANE-optimized form
//!
//! The exact selector names are placeholders — they must be reverse-engineered
//! or discovered via class-dump on the target macOS version.

use std::ffi::c_void;
use std::path::{Path, PathBuf};

use crate::error::MilError;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors specific to direct ANE compilation.
#[derive(Debug, thiserror::Error)]
pub enum AneError {
    /// The ANE compiler framework could not be loaded.
    #[error("ANE framework not available: {0}")]
    FrameworkNotFound(String),

    /// The `_ANECompiler` class was not found in the loaded framework.
    #[error("_ANECompiler class not found — macOS version may be unsupported")]
    ClassNotFound,

    /// The compiler returned a runtime error.
    #[error("ANE compilation failed: {0}")]
    CompilationFailed(String),

    /// An I/O error occurred while preparing inputs or reading outputs.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The input path does not exist or is not a valid model.
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

impl From<AneError> for MilError {
    fn from(e: AneError) -> Self {
        MilError::Validation(format!("ANE direct compilation error: {e}"))
    }
}

// ---------------------------------------------------------------------------
// Objective-C runtime FFI declarations
// ---------------------------------------------------------------------------

// These functions are used in the skeleton `compile` implementation and will
// be fully exercised once the Objective-C message sends are implemented.
#[allow(dead_code)]
#[link(name = "objc", kind = "dylib")]
unsafe extern "C" {
    /// Retrieve an Objective-C class by name.
    fn objc_getClass(name: *const u8) -> *mut c_void;

    /// Register (or look up) a selector by name.
    fn sel_registerName(name: *const u8) -> *mut c_void;

    // On arm64, objc_msgSend must NOT be declared as variadic.
    // Instead, declare specific function pointer types for each calling convention needed.
    // When implementing actual calls, cast `objc_msgSend` to the appropriate function pointer type.

    /// Send a message to an Objective-C object (non-variadic base declaration).
    ///
    /// On arm64 (Apple Silicon), the variadic and non-variadic calling
    /// conventions differ.  Declaring this as `fn(..., ...)` causes
    /// undefined behaviour.  Instead, we declare only the base two-argument
    /// form here and cast to specific function-pointer types at each call
    /// site via `std::mem::transmute`.
    fn objc_msgSend(receiver: *mut c_void, sel: *mut c_void) -> *mut c_void;
}

// Type aliases for specific objc_msgSend signatures (for future implementation):
// type MsgSendWithPath = unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_char) -> *mut c_void;
// Usage: let send: MsgSendWithPath = std::mem::transmute(objc_msgSend as *const ());

// ---------------------------------------------------------------------------
// Helper: null-terminated C strings from Rust literals
// ---------------------------------------------------------------------------

/// Create a null-terminated byte slice suitable for Objective-C runtime calls.
macro_rules! sel {
    ($s:expr) => {
        concat!($s, "\0").as_ptr()
    };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// A wrapper around Apple's private `_ANECompiler` Objective-C class.
///
/// This struct holds no state — it is a namespace for the static compilation
/// methods.  All interaction with the Objective-C runtime happens inside the
/// individual method implementations.
pub struct AneCompiler {
    _private: (),
}

impl AneCompiler {
    /// Check whether the ANE compiler framework is available on this system.
    ///
    /// Returns `true` if the `_ANECompiler` class can be resolved via the
    /// Objective-C runtime.  This is a cheap check that does not load the
    /// full framework.
    pub fn is_available() -> bool {
        // SAFETY: `objc_getClass` is safe to call with any null-terminated
        // string — it returns null if the class doesn't exist.
        let cls = unsafe { objc_getClass(sel!("_ANECompiler")) };
        !cls.is_null()
    }

    /// Compile a `.mlpackage` (or `.mlmodelc`) to an ANE-optimized bundle.
    ///
    /// This is the primary entry point.  It mirrors the functionality of
    /// `xcrun coremlcompiler compile <input> <output_dir>` but calls the
    /// private `_ANECompiler` class directly.
    ///
    /// # Arguments
    ///
    /// * `mlpackage_path` — Path to the input `.mlpackage` or `.mlmodelc`.
    /// * `output_dir` — Directory where the compiled output will be placed.
    ///
    /// # Returns
    ///
    /// The path to the compiled `.mlmodelc` bundle inside `output_dir`.
    ///
    /// # Errors
    ///
    /// Returns [`AneError`] if the framework is unavailable, compilation
    /// fails, or the expected output is not produced.
    pub fn compile(
        mlpackage_path: &Path,
        output_dir: &Path,
    ) -> std::result::Result<PathBuf, AneError> {
        if !mlpackage_path.exists() {
            return Err(AneError::InvalidInput(format!(
                "input path does not exist: {}",
                mlpackage_path.display()
            )));
        }

        std::fs::create_dir_all(output_dir)?;

        // --- Objective-C runtime interaction ---
        //
        // The following is a structural skeleton.  The exact selectors and
        // argument conventions must be reverse-engineered for the target
        // macOS version.  The pattern shown here is representative of how
        // the call would be made.

        // Step 1: Resolve the _ANECompiler class
        let cls = unsafe { objc_getClass(sel!("_ANECompiler")) };
        if cls.is_null() {
            return Err(AneError::ClassNotFound);
        }

        // Step 2: Build selector
        // The real selector is something like:
        //   compileModel:toPath:options:error:
        // Placeholder — must be updated after reverse-engineering.
        let _sel = unsafe { sel_registerName(sel!("compileModel:toPath:options:error:")) };

        // Step 3: Convert paths to NSString (via objc_msgSend to NSString)
        // Step 4: Call the compile method
        // Step 5: Check the NSError out-parameter
        // Step 6: Locate and return the output path
        //
        // TODO(ane-direct): Implement the actual Objective-C message sends.
        //   This requires:
        //   1. Creating NSString instances from the Rust path strings
        //   2. Creating an NSDictionary for options (if any)
        //   3. Sending the compile message
        //   4. Reading the NSError out-parameter for failure details
        //   5. Releasing any created Objective-C objects (or using autorelease)
        //
        // Until this is implemented, fall through to the error below.
        // When implementing, remove this block and uncomment the real logic.

        let _ = (cls, _sel); // suppress unused warnings

        Err(AneError::CompilationFailed(
            "ANE direct compilation is not yet implemented — \
             the Objective-C FFI calls require reverse-engineering \
             of _ANECompiler selectors for the target macOS version"
                .into(),
        ))
    }

    /// Compile with incremental/delta support, reusing cached artifacts.
    ///
    /// When the same model is compiled repeatedly (e.g. during development
    /// iteration), this method attempts to reuse previously compiled
    /// sub-graphs from `cache_dir` to speed up compilation.
    ///
    /// # Arguments
    ///
    /// * `mlpackage_path` — Path to the input model.
    /// * `output_dir` — Directory for the compiled output.
    /// * `cache_dir` — Directory for cached intermediate artifacts.
    ///
    /// # Cache Strategy
    ///
    /// The cache stores per-subgraph compilation artifacts keyed by a hash
    /// of the sub-graph's MIL representation.  On subsequent compilations,
    /// only changed sub-graphs are recompiled.  The cache can be safely
    /// deleted at any time — it will be rebuilt on the next compilation.
    ///
    /// # Errors
    ///
    /// Same as [`compile`](Self::compile), plus I/O errors related to the
    /// cache directory.
    pub fn compile_incremental(
        mlpackage_path: &Path,
        output_dir: &Path,
        cache_dir: &Path,
    ) -> std::result::Result<PathBuf, AneError> {
        if !mlpackage_path.exists() {
            return Err(AneError::InvalidInput(format!(
                "input path does not exist: {}",
                mlpackage_path.display()
            )));
        }

        std::fs::create_dir_all(output_dir)?;
        std::fs::create_dir_all(cache_dir)?;

        // --- Incremental compilation skeleton ---
        //
        // The intended flow is:
        //
        // 1. Hash the input model (or its sub-graphs) to produce cache keys.
        // 2. Check `cache_dir` for existing artifacts matching those keys.
        // 3. For cache misses, compile only the changed sub-graphs via the
        //    _ANECompiler private API.
        // 4. Assemble the final .mlmodelc from cached + freshly compiled
        //    artifacts.
        // 5. Store newly compiled artifacts in `cache_dir` for future reuse.
        //
        // TODO(ane-direct): Implement incremental compilation.
        //   This requires the same _ANECompiler reverse-engineering as
        //   `compile()`, plus:
        //   - A hashing scheme for MIL sub-graphs
        //   - Cache serialization/deserialization
        //   - Sub-graph extraction and merging logic

        let _ = cache_dir; // suppress unused warning

        // For now, fall back to the non-incremental path.
        Self::compile(mlpackage_path, output_dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
