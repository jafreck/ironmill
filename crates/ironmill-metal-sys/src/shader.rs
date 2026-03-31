//! Safe wrappers for `MTLLibrary` and `MTLFunction`.

use std::ffi::c_void;

use crate::error::MetalSysError;
use crate::objc::{self, sel};

// ---------------------------------------------------------------------------
// ShaderLibrary
// ---------------------------------------------------------------------------

/// Safe wrapper around an `MTLLibrary` (id<MTLLibrary>).
pub struct ShaderLibrary {
    raw: *mut c_void,
}

// SAFETY: MTLLibrary is thread-safe and immutable once created.
unsafe impl Send for ShaderLibrary {}
unsafe impl Sync for ShaderLibrary {}

impl ShaderLibrary {
    /// Create from a raw retained `id<MTLLibrary>`.
    pub(crate) fn from_raw(raw: *mut c_void) -> Self {
        Self { raw }
    }

    /// Look up a function by name in this library.
    pub fn get_function(&self, name: &str) -> Result<ShaderFunction, MetalSysError> {
        let ns_name = objc::create_nsstring(name)?;
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { objc::sel_registerName(sel!("newFunctionWithName:")) };
        let f: GetFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        let raw = unsafe { f(self.raw, sel, ns_name) };
        // Release the NSString.
        unsafe { objc::CFRelease(ns_name) };

        if raw.is_null() {
            return Err(MetalSysError::ShaderCompilation(format!(
                "function '{name}' not found in library"
            )));
        }
        Ok(ShaderFunction { raw })
    }

    /// Returns the raw `id<MTLLibrary>` pointer.
    pub fn as_raw_ptr(&self) -> *mut c_void {
        self.raw
    }
}

impl Drop for ShaderLibrary {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { objc::CFRelease(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// ShaderFunction
// ---------------------------------------------------------------------------

/// Safe wrapper around an `MTLFunction` (id<MTLFunction>).
pub struct ShaderFunction {
    raw: *mut c_void,
}

// SAFETY: MTLFunction is thread-safe and immutable once created.
unsafe impl Send for ShaderFunction {}
unsafe impl Sync for ShaderFunction {}

impl ShaderFunction {
    /// Returns the raw `id<MTLFunction>` pointer.
    pub fn as_raw_ptr(&self) -> *mut c_void {
        self.raw
    }
}

impl Drop for ShaderFunction {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { objc::CFRelease(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}
