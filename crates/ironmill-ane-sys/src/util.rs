//! Utility class wrappers: logging, error factories, clone helpers.
//!
//! Wraps Apple's private `_ANELog`, `_ANEErrors`, and `_ANECloneHelper`
//! classes from the `AppleNeuralEngine` framework.
//!
//! # ⚠️ Private API — see crate-level docs for risks and caveats.

use std::ffi::c_void;

use crate::error::AneSysError;
use crate::objc::{CFRelease, create_nsstring, get_class, objc_msgSend, sel, sel_registerName};

// ---------------------------------------------------------------------------
// AneLog — wraps _ANELog (all class methods, returns os_log handles)
// ---------------------------------------------------------------------------

/// Access to ANE os_log subsystem handles.
///
/// All methods are class methods on `_ANELog` — no instance is needed.
/// Each returns an opaque `os_log_t` pointer.
///
/// **Note:** Currently unused outside tests. Retained for potential future use
/// in ANE diagnostics and debugging.
#[doc(hidden)]
pub struct AneLog;

impl AneLog {
    /// os_log handle for ANE daemon messages.
    pub fn daemon() -> Result<*mut c_void, AneSysError> {
        Self::log_handle("daemon")
    }

    /// os_log handle for ANE compiler messages.
    pub fn compiler() -> Result<*mut c_void, AneSysError> {
        Self::log_handle("compiler")
    }

    /// os_log handle for ANE tool messages.
    pub fn tool() -> Result<*mut c_void, AneSysError> {
        Self::log_handle("tool")
    }

    /// os_log handle for ANE common messages.
    pub fn common() -> Result<*mut c_void, AneSysError> {
        Self::log_handle("common")
    }

    /// os_log handle for ANE test messages.
    pub fn tests() -> Result<*mut c_void, AneSysError> {
        Self::log_handle("tests")
    }

    /// os_log handle for ANE maintenance messages.
    pub fn maintenance() -> Result<*mut c_void, AneSysError> {
        Self::log_handle("maintenance")
    }

    /// os_log handle for ANE framework messages.
    pub fn framework() -> Result<*mut c_void, AneSysError> {
        Self::log_handle("framework")
    }

    fn log_handle(name: &str) -> Result<*mut c_void, AneSysError> {
        let cls = get_class("_ANELog")?;
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let mut sel_buf = Vec::with_capacity(name.len() + 1);
        sel_buf.extend_from_slice(name.as_bytes());
        sel_buf.push(0);
        let sel = unsafe { sel_registerName(sel_buf.as_ptr()) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, sel) };
        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: format!("_ANELog +{name} returned nil"),
            });
        }
        Ok(obj)
    }
}

// ---------------------------------------------------------------------------
// AneErrors — wraps _ANEErrors (error factory class methods)
// ---------------------------------------------------------------------------

/// Factory for creating `NSError` objects in the ANE error domain.
///
/// All methods are class methods on `_ANEErrors` — no instance is needed.
/// Returned `*mut c_void` pointers are autoreleased `NSError` objects.
///
/// **Note:** Currently unused outside tests. Retained for potential future use
/// in ANE error reporting.
#[doc(hidden)]
pub struct AneErrors;

impl AneErrors {
    /// Create an error with an explicit code and description string.
    ///
    /// Calls `+[_ANEErrors createErrorWithCode:description:]`.
    pub fn create_error(code: i64, desc: &str) -> Result<*mut c_void, AneSysError> {
        let cls = get_class("_ANEErrors")?;
        let ns_desc = create_nsstring(desc)?;
        type FactoryFn =
            unsafe extern "C" fn(*mut c_void, *mut c_void, i64, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("createErrorWithCode:description:")) };
        let f: FactoryFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, sel, code, ns_desc) };
        unsafe { CFRelease(ns_desc) };
        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "createErrorWithCode:description: returned nil".into(),
            });
        }
        Ok(obj)
    }

    /// `+[_ANEErrors badArgumentForMethod:]`
    pub fn bad_argument(method: &str) -> Result<*mut c_void, AneSysError> {
        Self::error_for_method(method, "badArgumentForMethod:")
    }

    /// `+[_ANEErrors fileNotFoundErrorForMethod:]`
    pub fn file_not_found(method: &str) -> Result<*mut c_void, AneSysError> {
        Self::error_for_method(method, "fileNotFoundErrorForMethod:")
    }

    /// `+[_ANEErrors invalidModelErrorForMethod:]`
    pub fn invalid_model(method: &str) -> Result<*mut c_void, AneSysError> {
        Self::error_for_method(method, "invalidModelErrorForMethod:")
    }

    /// `+[_ANEErrors programCreationErrorForMethod:]`
    pub fn program_creation_error(method: &str) -> Result<*mut c_void, AneSysError> {
        Self::error_for_method(method, "programCreationErrorForMethod:")
    }

    /// `+[_ANEErrors programLoadErrorForMethod:]`
    pub fn program_load_error(method: &str) -> Result<*mut c_void, AneSysError> {
        Self::error_for_method(method, "programLoadErrorForMethod:")
    }

    /// `+[_ANEErrors timeoutErrorForMethod:]`
    pub fn timeout_error(method: &str) -> Result<*mut c_void, AneSysError> {
        Self::error_for_method(method, "timeoutErrorForMethod:")
    }

    /// Generic helper: calls any `+[_ANEErrors <selector>:]` that takes a
    /// single `NSString` method-name argument and returns an `NSError`.
    pub fn error_for_method(
        method_name: &str,
        factory_selector: &str,
    ) -> Result<*mut c_void, AneSysError> {
        let cls = get_class("_ANEErrors")?;
        let ns_method = create_nsstring(method_name)?;
        type MsgFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> *mut c_void;
        let mut sel_buf = Vec::with_capacity(factory_selector.len() + 1);
        sel_buf.extend_from_slice(factory_selector.as_bytes());
        sel_buf.push(0);
        let sel = unsafe { sel_registerName(sel_buf.as_ptr()) };
        let f: MsgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, sel, ns_method) };
        unsafe { CFRelease(ns_method) };
        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: format!("_ANEErrors +{factory_selector} returned nil"),
            });
        }
        Ok(obj)
    }

    /// `+[_ANEErrors programLoadErrorForMethod:code:]`
    pub fn program_load_error_with_code(
        method: &str,
        code: i64,
    ) -> Result<*mut c_void, AneSysError> {
        let cls = get_class("_ANEErrors")?;
        let ns_method = create_nsstring(method)?;
        type MsgFn =
            unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, i64) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("programLoadErrorForMethod:code:")) };
        let f: MsgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, sel, ns_method, code) };
        unsafe { CFRelease(ns_method) };
        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "programLoadErrorForMethod:code: returned nil".into(),
            });
        }
        Ok(obj)
    }
}

// ---------------------------------------------------------------------------
// AneCloneHelper — wraps _ANECloneHelper
// ---------------------------------------------------------------------------

/// Helpers for cloning ANE model files.
///
/// All methods are class methods on `_ANECloneHelper` — no instance is needed.
///
/// **Note:** Currently unused outside tests. Retained for potential future use
/// in ANE model file management.
#[doc(hidden)]
pub struct AneCloneHelper;

impl AneCloneHelper {
    /// Clone a model file if the source is writable.
    ///
    /// Calls `+[_ANECloneHelper cloneIfWritable:isEncryptedModel:cloneDirectory:]`.
    /// Returns the result `NSString` path (autoreleased), or `nil` on failure.
    pub fn clone_if_writable(
        path: &str,
        is_encrypted: bool,
        clone_dir: &str,
    ) -> Result<*mut c_void, AneSysError> {
        let cls = get_class("_ANECloneHelper")?;
        let ns_path = create_nsstring(path)?;
        let ns_clone_dir = create_nsstring(clone_dir)?;
        type CloneFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            i8,
            *mut c_void,
        ) -> *mut c_void;
        let sel =
            unsafe { sel_registerName(sel!("cloneIfWritable:isEncryptedModel:cloneDirectory:")) };
        let f: CloneFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, sel, ns_path, is_encrypted as i8, ns_clone_dir) };
        unsafe { CFRelease(ns_path) };
        unsafe { CFRelease(ns_clone_dir) };
        // nil is a valid return (clone not needed), so no null check.
        Ok(obj)
    }

    /// Check whether cloning should be skipped for a given model.
    ///
    /// Calls `+[_ANECloneHelper shouldSkipCloneFor:isEncryptedModel:]`.
    pub fn should_skip_clone(path: &str, is_encrypted: bool) -> Result<bool, AneSysError> {
        let cls = get_class("_ANECloneHelper")?;
        let ns_path = create_nsstring(path)?;
        type BoolFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, i8) -> i8;
        let sel = unsafe { sel_registerName(sel!("shouldSkipCloneFor:isEncryptedModel:")) };
        let f: BoolFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let result = unsafe { f(cls, sel, ns_path, is_encrypted as i8) };
        unsafe { CFRelease(ns_path) };
        Ok(result != 0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ane_log_handles() {
        // May fail without the framework; should not panic.
        let _ = AneLog::daemon();
        let _ = AneLog::compiler();
        let _ = AneLog::framework();
    }

    #[test]
    fn ane_errors_create() {
        let _ = AneErrors::create_error(-1, "test error");
    }

    #[test]
    fn ane_errors_factory() {
        let _ = AneErrors::bad_argument("testMethod");
        let _ = AneErrors::file_not_found("testMethod");
    }

    #[test]
    fn ane_clone_helper_skip() {
        // NOTE: _ANECloneHelper methods may throw ObjC exceptions for
        // invalid paths. We only verify the class is loadable, not the
        // method behavior — calling with bogus paths aborts the process.
        let cls = crate::objc::get_class("_ANECloneHelper");
        assert!(cls.is_ok(), "CloneHelper class should be loadable");
    }
}
