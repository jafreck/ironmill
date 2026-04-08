// FFI functions intentionally accept raw pointers without being marked `unsafe`
// because the unsafety boundary is at the C ↔ Rust FFI boundary, not at each
// individual function.
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(deprecated)]
#![deny(unsafe_op_in_unsafe_fn)]

//! C-compatible FFI API for `mil-rs`.
//!
//! This module exposes the core model conversion pipeline to C, Swift, C++,
//! Go, and any other language that can call `extern "C"` functions.
//!
//! # Error handling
//!
//! Functions that can fail return a null pointer (for pointer-returning
//! functions) or `-1` (for `i32`-returning functions). Call
//! [`mil_last_error`] to retrieve a human-readable error message.
//! The error is stored in a thread-local and remains valid until the next
//! failing call on the same thread.
//!
//! # Memory management
//!
//! Every `*mut` pointer returned by this API must be freed by the
//! corresponding `*_free` function:
//!
//! | Returned type            | Free with                     |
//! |--------------------------|-------------------------------|
//! | `*mut MilModel`          | [`mil_model_free`]            |
//! | `*mut MilProgram`        | [`mil_program_free`]          |
//! | `*mut MilValidationReport` | [`mil_validation_report_free`] |
//! | `*mut c_char`            | [`mil_string_free`]           |

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};

// ---------------------------------------------------------------------------
// Opaque handle types
// ---------------------------------------------------------------------------

/// Internal discriminant for [`MilModel`].
enum MilModelInner {
    CoreMl(Box<ironmill_compile::mil::proto::specification::Model>),
    Onnx(Box<ironmill_compile::mil::proto::onnx::ModelProto>),
}

/// Opaque handle representing a loaded ML model.
///
/// A `MilModel` may wrap either a CoreML protobuf `Model` or an ONNX
/// `ModelProto`. Callers interact with it exclusively through the C API
/// functions and must free it with [`mil_model_free`].
pub struct MilModel(MilModelInner);

/// Opaque handle representing a MIL IR program.
///
/// Created by [`mil_onnx_to_program`]. Must be freed with
/// [`mil_program_free`].
pub struct MilProgram(ironmill_compile::mil::Program);

/// Opaque handle to an ANE validation report.
///
/// Created by [`mil_validate_ane`]. Must be freed with
/// [`mil_validation_report_free`].
pub struct MilValidationReport(ironmill_compile::ane::validate::ValidationReport);

// ---------------------------------------------------------------------------
// Thread-local error state
// ---------------------------------------------------------------------------

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Store a human-readable error message in thread-local state.
fn set_last_error(msg: &str) {
    LAST_ERROR.with(|e| {
        let sanitized = msg.replace('\0', "\\0");
        *e.borrow_mut() = CString::new(sanitized).ok();
    });
}

/// Convert a `*const c_char` to `&str`, returning `None` for null or
/// invalid UTF-8.
fn cstr_to_str<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: Caller ensures `ptr` points to a valid NUL-terminated C string
    // that outlives `'a`. All public C API callers pass CString-owned or
    // string-literal pointers.
    unsafe { CStr::from_ptr(ptr) }.to_str().ok()
}

// ---------------------------------------------------------------------------
// Lifecycle — reading models
// ---------------------------------------------------------------------------

/// Read an ONNX `.onnx` file and return an opaque [`MilModel`] handle.
///
/// Returns null on error; call [`mil_last_error`] for the message.
#[unsafe(no_mangle)]
pub extern "C" fn mil_read_onnx(path: *const c_char) -> *mut MilModel {
    let Some(path_str) = cstr_to_str(path) else {
        set_last_error("mil_read_onnx: path is null or invalid UTF-8");
        return std::ptr::null_mut();
    };
    match ironmill_compile::mil::read_onnx(path_str) {
        Ok(proto) => Box::into_raw(Box::new(MilModel(MilModelInner::Onnx(Box::new(proto))))),
        Err(e) => {
            set_last_error(&format!("mil_read_onnx: {e}"));
            std::ptr::null_mut()
        }
    }
}

/// Read a `.mlmodel` file and return an opaque [`MilModel`] handle.
///
/// Returns null on error; call [`mil_last_error`] for the message.
#[deprecated(note = "Use mlpackage format instead")]
#[unsafe(no_mangle)]
pub extern "C" fn mil_read_mlmodel(path: *const c_char) -> *mut MilModel {
    let Some(path_str) = cstr_to_str(path) else {
        set_last_error("mil_read_mlmodel: path is null or invalid UTF-8");
        return std::ptr::null_mut();
    };
    match ironmill_compile::mil::read_mlmodel(path_str) {
        Ok(model) => Box::into_raw(Box::new(MilModel(MilModelInner::CoreMl(Box::new(model))))),
        Err(e) => {
            set_last_error(&format!("mil_read_mlmodel: {e}"));
            std::ptr::null_mut()
        }
    }
}

/// Read a `.mlpackage` directory and return an opaque [`MilModel`] handle.
///
/// Returns null on error; call [`mil_last_error`] for the message.
#[unsafe(no_mangle)]
pub extern "C" fn mil_read_mlpackage(path: *const c_char) -> *mut MilModel {
    let Some(path_str) = cstr_to_str(path) else {
        set_last_error("mil_read_mlpackage: path is null or invalid UTF-8");
        return std::ptr::null_mut();
    };
    match ironmill_compile::mil::read_mlpackage(path_str) {
        Ok(model) => Box::into_raw(Box::new(MilModel(MilModelInner::CoreMl(Box::new(model))))),
        Err(e) => {
            set_last_error(&format!("mil_read_mlpackage: {e}"));
            std::ptr::null_mut()
        }
    }
}

/// Free a [`MilModel`] previously returned by one of the `mil_read_*`
/// functions.
///
/// Passing a null pointer is a safe no-op.
#[unsafe(no_mangle)]
pub extern "C" fn mil_model_free(model: *mut MilModel) {
    if !model.is_null() {
        // SAFETY: `model` was allocated by `Box::into_raw` in one of the
        // `mil_read_*` functions. The C API contract guarantees it is only
        // freed once.
        drop(unsafe { Box::from_raw(model) });
    }
}

// ---------------------------------------------------------------------------
// Conversion
// ---------------------------------------------------------------------------

/// Convert an ONNX [`MilModel`] to a MIL IR [`MilProgram`].
///
/// The model must have been loaded with [`mil_read_onnx`]. If the model is a
/// CoreML model, an error is set and null is returned.
///
/// Returns null on error; call [`mil_last_error`] for the message.
#[unsafe(no_mangle)]
pub extern "C" fn mil_onnx_to_program(model: *mut MilModel) -> *mut MilProgram {
    if model.is_null() {
        set_last_error("mil_onnx_to_program: model is null");
        return std::ptr::null_mut();
    }
    // SAFETY: `model` is non-null and was allocated by `Box::into_raw` in a
    // `mil_read_*` function. We take `&mut` to access the inner proto; no
    // aliasing is possible because only one call at a time holds this pointer.
    let model = unsafe { &mut *model };
    let onnx = match &mut model.0 {
        MilModelInner::Onnx(proto) => proto,
        MilModelInner::CoreMl(_) => {
            set_last_error("mil_onnx_to_program: model is CoreML, not ONNX");
            return std::ptr::null_mut();
        }
    };
    match ironmill_compile::mil::onnx_to_program(onnx) {
        Ok(result) => Box::into_raw(Box::new(MilProgram(result.program))),
        Err(e) => {
            set_last_error(&format!("mil_onnx_to_program: {e}"));
            std::ptr::null_mut()
        }
    }
}

/// Convert a MIL IR [`MilProgram`] back to a CoreML protobuf [`MilModel`].
///
/// `spec_version` selects the CoreML specification version (e.g. `7`).
///
/// Returns null on error; call [`mil_last_error`] for the message.
#[unsafe(no_mangle)]
pub extern "C" fn mil_program_to_model(
    program: *const MilProgram,
    spec_version: i32,
) -> *mut MilModel {
    if program.is_null() {
        set_last_error("mil_program_to_model: program is null");
        return std::ptr::null_mut();
    }
    // SAFETY: `program` is non-null and was allocated by `Box::into_raw` in
    // `mil_onnx_to_program`. We only take a shared reference.
    let program = unsafe { &*program };
    match ironmill_compile::mil::program_to_model(&program.0, spec_version) {
        Ok(model) => Box::into_raw(Box::new(MilModel(MilModelInner::CoreMl(Box::new(model))))),
        Err(e) => {
            set_last_error(&format!("mil_program_to_model: {e}"));
            std::ptr::null_mut()
        }
    }
}

/// Free a [`MilProgram`] previously returned by [`mil_onnx_to_program`].
///
/// Passing a null pointer is a safe no-op.
#[unsafe(no_mangle)]
pub extern "C" fn mil_program_free(program: *mut MilProgram) {
    if !program.is_null() {
        // SAFETY: `program` was allocated by `Box::into_raw` in
        // `mil_onnx_to_program`. The C API contract guarantees single free.
        drop(unsafe { Box::from_raw(program) });
    }
}

// ---------------------------------------------------------------------------
// I/O — writing models
// ---------------------------------------------------------------------------

/// Write a CoreML [`MilModel`] to a `.mlmodel` file.
///
/// Returns `0` on success, `-1` on error. The model must be a CoreML model
/// (not ONNX). Call [`mil_last_error`] for the message on failure.
#[deprecated(note = "Use mlpackage format instead")]
#[unsafe(no_mangle)]
pub extern "C" fn mil_write_mlmodel(model: *const MilModel, path: *const c_char) -> i32 {
    if model.is_null() {
        set_last_error("mil_write_mlmodel: model is null");
        return -1;
    }
    let Some(path_str) = cstr_to_str(path) else {
        set_last_error("mil_write_mlmodel: path is null or invalid UTF-8");
        return -1;
    };
    // SAFETY: `model` is non-null (checked above) and was allocated by
    // `Box::into_raw` in a `mil_read_*` function.
    let model = unsafe { &*model };
    let coreml = match &model.0 {
        MilModelInner::CoreMl(m) => m,
        MilModelInner::Onnx(_) => {
            set_last_error("mil_write_mlmodel: model is ONNX, not CoreML");
            return -1;
        }
    };
    match ironmill_compile::mil::write_mlmodel(coreml, path_str) {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&format!("mil_write_mlmodel: {e}"));
            -1
        }
    }
}

/// Write a CoreML [`MilModel`] to a `.mlpackage` directory.
///
/// Returns `0` on success, `-1` on error. The model must be a CoreML model
/// (not ONNX). Call [`mil_last_error`] for the message on failure.
#[unsafe(no_mangle)]
pub extern "C" fn mil_write_mlpackage(model: *const MilModel, path: *const c_char) -> i32 {
    if model.is_null() {
        set_last_error("mil_write_mlpackage: model is null");
        return -1;
    }
    let Some(path_str) = cstr_to_str(path) else {
        set_last_error("mil_write_mlpackage: path is null or invalid UTF-8");
        return -1;
    };
    // SAFETY: `model` is non-null (checked above) and was allocated by
    // `Box::into_raw` in a `mil_read_*` function.
    let model = unsafe { &*model };
    let coreml = match &model.0 {
        MilModelInner::CoreMl(m) => m,
        MilModelInner::Onnx(_) => {
            set_last_error("mil_write_mlpackage: model is ONNX, not CoreML");
            return -1;
        }
    };
    match ironmill_compile::mil::write_mlpackage(coreml, path_str) {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&format!("mil_write_mlpackage: {e}"));
            -1
        }
    }
}

// ---------------------------------------------------------------------------
// Compilation
// ---------------------------------------------------------------------------

/// Compile a `.mlpackage` or `.mlmodel` to `.mlmodelc` via `xcrun`.
///
/// Returns a heap-allocated path string on success (free with
/// [`mil_string_free`]) or null on error.
#[unsafe(no_mangle)]
pub extern "C" fn mil_compile_model(
    input_path: *const c_char,
    output_dir: *const c_char,
) -> *mut c_char {
    let Some(input) = cstr_to_str(input_path) else {
        set_last_error("mil_compile_model: input_path is null or invalid UTF-8");
        return std::ptr::null_mut();
    };
    let Some(output) = cstr_to_str(output_dir) else {
        set_last_error("mil_compile_model: output_dir is null or invalid UTF-8");
        return std::ptr::null_mut();
    };
    match ironmill_compile::coreml::compiler::compile_model(input, output) {
        Ok(path) => match CString::new(path.to_string_lossy().into_owned()) {
            Ok(cs) => cs.into_raw(),
            Err(e) => {
                set_last_error(&format!("mil_compile_model: path contains null byte: {e}"));
                std::ptr::null_mut()
            }
        },
        Err(e) => {
            set_last_error(&format!("mil_compile_model: {e}"));
            std::ptr::null_mut()
        }
    }
}

/// Check whether `xcrun coremlcompiler` is available on this system.
#[unsafe(no_mangle)]
pub extern "C" fn mil_is_compiler_available() -> bool {
    ironmill_compile::coreml::compiler::is_compiler_available()
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validate a MIL IR [`MilProgram`] for Apple Neural Engine compatibility.
///
/// Returns an opaque [`MilValidationReport`] handle. Free with
/// [`mil_validation_report_free`]. Returns null if `program` is null.
#[unsafe(no_mangle)]
pub extern "C" fn mil_validate_ane(program: *const MilProgram) -> *mut MilValidationReport {
    if program.is_null() {
        set_last_error("mil_validate_ane: program is null");
        return std::ptr::null_mut();
    }
    // SAFETY: `program` is non-null (checked above) and was allocated by
    // `Box::into_raw` in `mil_onnx_to_program`.
    let program = unsafe { &*program };
    let report = ironmill_compile::ane::validate::validate_ane_compatibility(&program.0);
    Box::into_raw(Box::new(MilValidationReport(report)))
}

/// Serialize a [`MilValidationReport`] to a JSON string.
///
/// Returns a heap-allocated string (free with [`mil_string_free`]) or null
/// on error.
#[unsafe(no_mangle)]
pub extern "C" fn mil_validation_report_to_json(report: *const MilValidationReport) -> *mut c_char {
    if report.is_null() {
        set_last_error("mil_validation_report_to_json: report is null");
        return std::ptr::null_mut();
    }
    let result = std::panic::catch_unwind(|| {
        // SAFETY: `report` is non-null (checked above) and was allocated by
        // `Box::into_raw` in `mil_validate_ane`.
        let report = unsafe { &*report };
        let json = ironmill_compile::ane::validate::validation_report_to_json(&report.0);
        let json = match json {
            Ok(s) => s,
            Err(e) => {
                set_last_error(&format!(
                    "mil_validation_report_to_json: serialization failed: {e}"
                ));
                return std::ptr::null_mut();
            }
        };
        match CString::new(json) {
            Ok(cs) => cs.into_raw(),
            Err(e) => {
                set_last_error(&format!(
                    "mil_validation_report_to_json: JSON contains null byte: {e}"
                ));
                std::ptr::null_mut()
            }
        }
    });
    match result {
        Ok(ptr) => ptr,
        Err(panic) => {
            let msg = if let Some(s) = panic.downcast_ref::<&str>() {
                format!("mil_validation_report_to_json: panic: {s}")
            } else if let Some(s) = panic.downcast_ref::<String>() {
                format!("mil_validation_report_to_json: panic: {s}")
            } else {
                "mil_validation_report_to_json: panic (unknown payload)".to_string()
            };
            set_last_error(&msg);
            std::ptr::null_mut()
        }
    }
}

/// Free a [`MilValidationReport`] previously returned by
/// [`mil_validate_ane`].
///
/// Passing a null pointer is a safe no-op.
#[unsafe(no_mangle)]
pub extern "C" fn mil_validation_report_free(report: *mut MilValidationReport) {
    if !report.is_null() {
        // SAFETY: `report` was allocated by `Box::into_raw` in
        // `mil_validate_ane`. The C API contract guarantees single free.
        drop(unsafe { Box::from_raw(report) });
    }
}

// ---------------------------------------------------------------------------
// Error handling & string management
// ---------------------------------------------------------------------------

/// Return the last error message, or null if no error has been recorded.
///
/// The returned pointer is valid until the next failing C API call on the
/// same thread. Do **not** free the returned pointer.
#[unsafe(no_mangle)]
pub extern "C" fn mil_last_error() -> *const c_char {
    LAST_ERROR.with(|e| {
        e.borrow()
            .as_ref()
            .map_or(std::ptr::null(), |cs| cs.as_ptr())
    })
}

/// Free a string previously returned by the C API (e.g., from
/// [`mil_compile_model`] or [`mil_validation_report_to_json`]).
///
/// Passing a null pointer is a safe no-op.
#[unsafe(no_mangle)]
pub extern "C" fn mil_string_free(s: *mut c_char) {
    if !s.is_null() {
        // SAFETY: `s` was allocated by `CString::into_raw` in
        // `mil_compile_model` or `mil_validation_report_to_json`.
        // The C API contract guarantees single free.
        drop(unsafe { CString::from_raw(s) });
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    /// Read the thread-local error string set by the last failing C API call.
    ///
    /// # Panics
    ///
    /// Panics if no error has been recorded or if the string is not valid UTF-8.
    fn last_error_str() -> String {
        let ptr = mil_last_error();
        assert!(
            !ptr.is_null(),
            "expected an error but mil_last_error() is null"
        );
        // SAFETY: `ptr` is the `as_ptr()` of a thread-local `CString` owned
        // by `LAST_ERROR` — it is NUL-terminated and valid for the duration
        // of this borrow (no intervening failing calls).
        let cstr = unsafe { CStr::from_ptr(ptr) };
        cstr.to_str()
            .expect("error message is not valid UTF-8")
            .to_owned()
    }

    // -- Error lifecycle ------------------------------------------------------

    #[test]
    fn error_lifecycle_set_read_overwrite() {
        // Initially no error.
        let err = mil_last_error();
        assert!(err.is_null());

        // Trigger an error.
        let result = mil_read_onnx(std::ptr::null());
        assert!(result.is_null());

        // Error should be readable.
        let err = mil_last_error();
        assert!(!err.is_null());
        let msg = last_error_str();
        assert!(msg.contains("null"), "unexpected error: {msg}");

        // A second failing call overwrites the previous error.
        let bad_path = CString::new("/nonexistent/path.onnx").unwrap();
        let result = mil_read_onnx(bad_path.as_ptr());
        assert!(result.is_null());

        let msg = last_error_str();
        assert!(msg.contains("mil_read_onnx"), "unexpected error: {msg}");
    }

    // -- Null-pointer handling -----------------------------------------------

    #[test]
    fn null_model_free_is_noop() {
        mil_model_free(std::ptr::null_mut());
    }

    #[test]
    fn null_program_free_is_noop() {
        mil_program_free(std::ptr::null_mut());
    }

    #[test]
    fn null_report_free_is_noop() {
        mil_validation_report_free(std::ptr::null_mut());
    }

    #[test]
    fn null_string_free_is_noop() {
        mil_string_free(std::ptr::null_mut());
    }

    #[test]
    fn read_onnx_null_returns_null() {
        let p = mil_read_onnx(std::ptr::null());
        assert!(p.is_null());
        assert!(!mil_last_error().is_null());
    }

    #[test]
    fn read_mlmodel_null_returns_null() {
        let p = mil_read_mlmodel(std::ptr::null());
        assert!(p.is_null());
        assert!(!mil_last_error().is_null());
    }

    #[test]
    fn read_mlpackage_null_returns_null() {
        let p = mil_read_mlpackage(std::ptr::null());
        assert!(p.is_null());
        assert!(!mil_last_error().is_null());
    }

    #[test]
    fn onnx_to_program_null_returns_null() {
        let p = mil_onnx_to_program(std::ptr::null_mut());
        assert!(p.is_null());
        assert!(!mil_last_error().is_null());
    }

    #[test]
    fn program_to_model_null_returns_null() {
        let p = mil_program_to_model(std::ptr::null(), 7);
        assert!(p.is_null());
        assert!(!mil_last_error().is_null());
    }

    #[test]
    fn write_mlmodel_null_model_returns_error() {
        let path = CString::new("out.mlmodel").unwrap();
        let rc = mil_write_mlmodel(std::ptr::null(), path.as_ptr());
        assert_eq!(rc, -1);
        assert!(!mil_last_error().is_null());
    }

    #[test]
    fn write_mlmodel_null_path_returns_error() {
        // We need a non-null but valid MilModel to test the path check.
        // Construct a dummy CoreML model.
        let model = Box::into_raw(Box::new(MilModel(MilModelInner::CoreMl(Box::new(
            ironmill_compile::mil::proto::specification::Model::default(),
        )))));
        let rc = mil_write_mlmodel(model, std::ptr::null());
        assert_eq!(rc, -1);
        assert!(!mil_last_error().is_null());
        // Clean up
        mil_model_free(model);
    }

    #[test]
    fn write_mlpackage_null_model_returns_error() {
        let path = CString::new("out.mlpackage").unwrap();
        let rc = mil_write_mlpackage(std::ptr::null(), path.as_ptr());
        assert_eq!(rc, -1);
    }

    #[test]
    fn compile_model_null_input_returns_null() {
        let out = CString::new("outdir").unwrap();
        let p = mil_compile_model(std::ptr::null(), out.as_ptr());
        assert!(p.is_null());
        assert!(!mil_last_error().is_null());
    }

    #[test]
    fn compile_model_null_output_returns_null() {
        let inp = CString::new("model.mlpackage").unwrap();
        let p = mil_compile_model(inp.as_ptr(), std::ptr::null());
        assert!(p.is_null());
        assert!(!mil_last_error().is_null());
    }

    #[test]
    fn validate_ane_null_returns_null() {
        let p = mil_validate_ane(std::ptr::null());
        assert!(p.is_null());
        assert!(!mil_last_error().is_null());
    }

    #[test]
    fn validation_report_to_json_null_returns_null() {
        let p = mil_validation_report_to_json(std::ptr::null());
        assert!(p.is_null());
        assert!(!mil_last_error().is_null());
    }

    // -- Nonexistent file handling -------------------------------------------

    #[test]
    fn read_onnx_nonexistent_returns_null_with_error() {
        let path = CString::new("/nonexistent/model.onnx").unwrap();
        let p = mil_read_onnx(path.as_ptr());
        assert!(p.is_null());

        let msg = last_error_str();
        assert!(msg.contains("mil_read_onnx"), "unexpected error: {msg}");
    }

    #[test]
    fn read_mlmodel_nonexistent_returns_null_with_error() {
        let path = CString::new("/nonexistent/model.mlmodel").unwrap();
        let p = mil_read_mlmodel(path.as_ptr());
        assert!(p.is_null());

        let msg = last_error_str();
        assert!(msg.contains("mil_read_mlmodel"), "unexpected error: {msg}");
    }

    #[test]
    fn read_mlpackage_nonexistent_returns_null_with_error() {
        let path = CString::new("/nonexistent/model.mlpackage").unwrap();
        let p = mil_read_mlpackage(path.as_ptr());
        assert!(p.is_null());

        let msg = last_error_str();
        assert!(
            msg.contains("mil_read_mlpackage"),
            "unexpected error: {msg}"
        );
    }

    // -- Compiler availability -----------------------------------------------

    #[test]
    fn is_compiler_available_does_not_panic() {
        let _ = mil_is_compiler_available();
    }

    // -- CoreML / ONNX mismatch errors ----------------------------------------

    #[test]
    fn onnx_to_program_rejects_coreml_model() {
        let model = Box::into_raw(Box::new(MilModel(MilModelInner::CoreMl(Box::new(
            ironmill_compile::mil::proto::specification::Model::default(),
        )))));
        let p = mil_onnx_to_program(model);
        assert!(p.is_null());

        let msg = last_error_str();
        assert!(msg.contains("not ONNX"), "unexpected error: {msg}");

        mil_model_free(model);
    }

    #[test]
    fn write_mlmodel_rejects_onnx_model() {
        let model = Box::into_raw(Box::new(MilModel(MilModelInner::Onnx(Box::new(
            ironmill_compile::mil::proto::onnx::ModelProto::default(),
        )))));
        let path = CString::new("out.mlmodel").unwrap();
        let rc = mil_write_mlmodel(model, path.as_ptr());
        assert_eq!(rc, -1);

        let msg = last_error_str();
        assert!(msg.contains("not CoreML"), "unexpected error: {msg}");

        mil_model_free(model);
    }

    #[test]
    fn write_mlpackage_rejects_onnx_model() {
        let model = Box::into_raw(Box::new(MilModel(MilModelInner::Onnx(Box::new(
            ironmill_compile::mil::proto::onnx::ModelProto::default(),
        )))));
        let path = CString::new("out.mlpackage").unwrap();
        let rc = mil_write_mlpackage(model, path.as_ptr());
        assert_eq!(rc, -1);

        let msg = last_error_str();
        assert!(msg.contains("not CoreML"), "unexpected error: {msg}");

        mil_model_free(model);
    }
}
