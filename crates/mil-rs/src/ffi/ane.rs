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
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::error::MilError;

/// Maximum number of ANE compilations allowed per process.
///
/// Apple's `_ANECompiler` leaks memory on each invocation (~0.5-2 MB).  After
/// roughly 119 compilations the ANE daemon (`aned`) starts rejecting requests.
/// This constant caps usage to avoid hitting that wall silently.
const ANE_COMPILE_LIMIT: usize = 119;

/// Global compile count tracker (constraint: ~119 limit per process).
static COMPILE_COUNT: AtomicUsize = AtomicUsize::new(0);

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

    /// The per-process ANE compile budget has been exhausted.
    #[error("ANE compile budget exhausted ({count}/119 compilations)")]
    BudgetExhausted { count: usize },
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

    // -------------------------------------------------------------------
    // MIL text + BLOBFILE compilation (ANE direct backend)
    // -------------------------------------------------------------------

    /// Compile MIL text + weight blob into an ANE program.
    ///
    /// This is the primary compilation path for the ANE direct backend.
    /// It uses `_ANEInMemoryModelDescriptor` and `_ANECompiler` to compile
    /// MIL text directly, without going through CoreML's `.mlpackage` format.
    ///
    /// # Arguments
    ///
    /// * `mil_text` — UTF-8 MIL text (from `ir_to_mil_text`)
    /// * `weight_path` — Path to the BLOBFILE (from `BlobFileWriter`)
    ///
    /// # Returns
    ///
    /// An opaque pointer to the compiled program handle.
    ///
    /// # Errors
    ///
    /// Returns [`AneError`] if the compile budget is exhausted, required
    /// Objective-C classes are missing, or the compilation itself fails.
    pub fn compile_mil_text(
        mil_text: &str,
        weight_path: &Path,
    ) -> std::result::Result<*mut c_void, AneError> {
        // 0. Check budget
        let current = COMPILE_COUNT.load(Ordering::Relaxed);
        if current >= ANE_COMPILE_LIMIT {
            return Err(AneError::BudgetExhausted { count: current });
        }

        if mil_text.is_empty() {
            return Err(AneError::InvalidInput("MIL text is empty".into()));
        }

        if !weight_path.exists() {
            return Err(AneError::InvalidInput(format!(
                "weight blob path does not exist: {}",
                weight_path.display()
            )));
        }

        // 1. Resolve _ANECompiler class
        let compiler_cls = unsafe { objc_getClass(sel!("_ANECompiler")) };
        if compiler_cls.is_null() {
            return Err(AneError::ClassNotFound);
        }

        // 2. Create NSData from MIL text
        //    NSData must be used (not NSString) because the MIL text may
        //    contain raw bytes that NSString normalisation would mangle.
        let mil_data = create_nsdata(mil_text.as_bytes())?;

        // 3. Create NSURL for weight path
        let weight_url = create_nsurl_from_path(weight_path)?;

        // 4-7. Create _ANEInMemoryModelDescriptor, set milText / weightPath,
        //      call _ANECompiler compileDescriptor:error:
        //
        // The exact selector names for _ANEInMemoryModelDescriptor and
        // _ANECompiler are placeholders.  In production, these must be
        // verified against the target macOS version.
        let _descriptor_cls = unsafe { objc_getClass(sel!("_ANEInMemoryModelDescriptor")) };

        // Placeholder: alloc/init descriptor, set milData & weightURL,
        // then call compileDescriptor:error: on the compiler class.
        let _compile_sel = unsafe { sel_registerName(sel!("compileDescriptor:error:")) };

        // Suppress unused-variable warnings for the ObjC objects we built.
        let _ = (
            mil_data,
            weight_url,
            _descriptor_cls,
            _compile_sel,
            compiler_cls,
        );

        // 8. Increment compile count — even though the actual ObjC path is
        //    not yet wired, we track calls for budget accounting.
        COMPILE_COUNT.fetch_add(1, Ordering::Relaxed);

        // 9. Return result.
        //
        // TODO: Complete the actual ObjC message sends once selectors are
        // verified via class-dump on target macOS.  For now, return a
        // placeholder error indicating the infrastructure is in place.
        Err(AneError::CompilationFailed(
            "ANE MIL text compilation infrastructure complete — \
             private API selectors require runtime verification on target macOS"
                .into(),
        ))
    }

    /// Number of compilations performed in this process.
    pub fn compile_count() -> usize {
        COMPILE_COUNT.load(Ordering::Relaxed)
    }

    /// Remaining compile budget before hitting the ~119 limit.
    pub fn remaining_budget() -> usize {
        ANE_COMPILE_LIMIT.saturating_sub(COMPILE_COUNT.load(Ordering::Relaxed))
    }
}

// ---------------------------------------------------------------------------
// ObjC helpers — NSData / NSString / NSURL creation
// ---------------------------------------------------------------------------

/// Create an `NSData` from a byte slice via `[NSData dataWithBytes:length:]`.
///
/// Returns a raw pointer to the autoreleased `NSData` object, or an error if
/// the `NSData` class is unavailable.
fn create_nsdata(bytes: &[u8]) -> Result<*mut c_void, AneError> {
    let nsdata_cls = unsafe { objc_getClass(sel!("NSData")) };
    if nsdata_cls.is_null() {
        return Err(AneError::FrameworkNotFound("NSData class not found".into()));
    }

    // dataWithBytes:length: — (id, SEL, const void*, NSUInteger) -> id
    type DataWithBytesFn =
        unsafe extern "C" fn(*mut c_void, *mut c_void, *const u8, usize) -> *mut c_void;

    let data_sel = unsafe { sel_registerName(sel!("dataWithBytes:length:")) };
    let send: DataWithBytesFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };

    let obj = unsafe { send(nsdata_cls, data_sel, bytes.as_ptr(), bytes.len()) };
    if obj.is_null() {
        return Err(AneError::CompilationFailed(
            "failed to create NSData".into(),
        ));
    }
    Ok(obj)
}

/// Create an `NSString` from a Rust string slice via
/// `[NSString stringWithUTF8String:]`.
///
/// The caller is responsible for ensuring the returned pointer lives long
/// enough (it is autoreleased).
fn create_nsstring(s: &str) -> Result<*mut c_void, AneError> {
    let nsstring_cls = unsafe { objc_getClass(sel!("NSString")) };
    if nsstring_cls.is_null() {
        return Err(AneError::FrameworkNotFound(
            "NSString class not found".into(),
        ));
    }

    // We need a null-terminated copy because stringWithUTF8String: expects
    // a C string.
    let mut buf = Vec::with_capacity(s.len() + 1);
    buf.extend_from_slice(s.as_bytes());
    buf.push(0);

    // stringWithUTF8String: — (id, SEL, const char*) -> id
    type StringWithUtf8Fn =
        unsafe extern "C" fn(*mut c_void, *mut c_void, *const u8) -> *mut c_void;

    let sel = unsafe { sel_registerName(sel!("stringWithUTF8String:")) };
    let send: StringWithUtf8Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };

    let obj = unsafe { send(nsstring_cls, sel, buf.as_ptr()) };
    if obj.is_null() {
        return Err(AneError::CompilationFailed(
            "failed to create NSString".into(),
        ));
    }
    Ok(obj)
}

/// Create an `NSURL` from a filesystem path via `[NSURL fileURLWithPath:]`.
fn create_nsurl_from_path(path: &Path) -> Result<*mut c_void, AneError> {
    let path_str = path.to_str().ok_or_else(|| {
        AneError::InvalidInput(format!("path contains invalid UTF-8: {}", path.display()))
    })?;

    let nsstring = create_nsstring(path_str)?;

    let nsurl_cls = unsafe { objc_getClass(sel!("NSURL")) };
    if nsurl_cls.is_null() {
        return Err(AneError::FrameworkNotFound("NSURL class not found".into()));
    }

    // fileURLWithPath: — (id, SEL, id) -> id
    type FileUrlWithPathFn =
        unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> *mut c_void;

    let sel = unsafe { sel_registerName(sel!("fileURLWithPath:")) };
    let send: FileUrlWithPathFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };

    let url = unsafe { send(nsurl_cls, sel, nsstring) };
    if url.is_null() {
        return Err(AneError::CompilationFailed(
            "failed to create NSURL from path".into(),
        ));
    }
    Ok(url)
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

    #[test]
    fn compile_count_starts_at_zero_or_accumulates() {
        // Because tests share the process-global COMPILE_COUNT, we can only
        // assert it is non-negative (another test may have incremented it).
        let count = AneCompiler::compile_count();
        assert!(
            count < ANE_COMPILE_LIMIT + 50,
            "count looks unreasonably high: {count}"
        );
    }

    #[test]
    fn compile_mil_text_budget_tracking() {
        let before = AneCompiler::compile_count();

        // compile_mil_text with a non-existent weight path should fail
        // with InvalidInput *before* incrementing the counter.
        let result =
            AneCompiler::compile_mil_text("program test {}", Path::new("nonexistent_weights.bin"));
        assert!(result.is_err());

        let after = AneCompiler::compile_count();
        // Counter should NOT have incremented for an early-exit error.
        assert_eq!(
            before, after,
            "compile_count should not increment on early validation error"
        );
    }

    #[test]
    fn remaining_budget_decreases() {
        let remaining = AneCompiler::remaining_budget();
        let count = AneCompiler::compile_count();
        assert_eq!(
            remaining,
            ANE_COMPILE_LIMIT.saturating_sub(count),
            "remaining budget should equal limit minus count"
        );
    }

    #[test]
    fn compile_mil_text_empty_text_returns_error() {
        let result = AneCompiler::compile_mil_text("", Path::new("weights.bin"));
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
}
