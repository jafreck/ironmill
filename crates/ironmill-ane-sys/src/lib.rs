//! Low-level safe FFI bindings for Apple Neural Engine private APIs.
//!
//! `ironmill-ane-sys` consolidates the Objective-C runtime FFI code that was
//! previously duplicated across `mil-rs` (compiler) and `ironmill-ane`
//! (runtime) into a single crate with a safe public API over unsafe
//! internals.
//!
//! # ⚠️ Private API Warning
//!
//! This crate uses **undocumented Apple private APIs** that may change between
//! macOS releases.  It should not be used in Mac App Store submissions.
//!
//! # macOS Only
//!
//! This crate only compiles on macOS — a `compile_error!` is emitted on
//! other platforms.

#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(not(target_os = "macos"))]
compile_error!("ironmill-ane-sys only supports macOS");

pub mod compiler;
pub mod error;
pub(crate) mod objc;
pub mod process;
pub mod runtime;

pub use compiler::AneCompiler;
pub use error::AneSysError;
pub use runtime::AneRuntime;

use std::ffi::c_void;

// ---------------------------------------------------------------------------
// Program handle types
// ---------------------------------------------------------------------------

/// A compiled ANE program, ready to be loaded into the runtime.
///
/// Produced by [`AneCompiler::compile_mil_text`] or
/// [`AneCompiler::patch_weights`].  Contains the `_ANEInMemoryModel`
/// handle which is already compiled and loaded.
pub struct CompiledProgram {
    /// Opaque handle to the `_ANEInMemoryModel` (compiled + loaded).
    pub(crate) model: *mut c_void,
}

impl std::fmt::Debug for CompiledProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledProgram")
            .field("model", &self.model)
            .finish()
    }
}

impl CompiledProgram {
    /// Create from a raw model pointer.
    ///
    /// # Safety
    /// The caller must ensure `ptr` is a valid, retained `_ANEInMemoryModel`
    /// handle that has been compiled and loaded.
    pub unsafe fn from_raw(ptr: *mut c_void) -> Self {
        Self { model: ptr }
    }

    /// Get the raw model pointer.
    ///
    /// Returns the underlying `_ANEInMemoryModel` handle. The pointer is
    /// valid for the lifetime of this `CompiledProgram`.
    pub fn as_raw_ptr(&self) -> *mut c_void {
        self.model
    }
}

// SAFETY: CompiledProgram can be sent between threads — the handle
// is an opaque pointer that is not accessed until eval.
unsafe impl Send for CompiledProgram {}

impl Drop for CompiledProgram {
    fn drop(&mut self) {
        if !self.model.is_null() {
            // SAFETY: CFRelease on a retained ObjC object.
            unsafe {
                objc::CFRelease(self.model);
            }
            self.model = std::ptr::null_mut();
        }
    }
}

/// A program that has been loaded into the ANE runtime for execution.
///
/// In the Orion-based API, this wraps the same `_ANEInMemoryModel`
/// handle as `CompiledProgram` — loading is done at compile time.
pub struct LoadedProgram {
    /// Opaque handle to the `_ANEInMemoryModel` (compiled + loaded).
    pub(crate) model: *mut c_void,
}

impl LoadedProgram {
    /// Get the raw model pointer.
    pub fn as_raw_ptr(&self) -> *mut c_void {
        self.model
    }
}

// SAFETY: LoadedProgram is Send (can move between threads) but NOT Sync
// (cannot be shared — ANE is not thread-safe).
unsafe impl Send for LoadedProgram {}

impl Drop for LoadedProgram {
    fn drop(&mut self) {
        if !self.model.is_null() {
            // SAFETY: CFRelease on a retained ObjC object.
            unsafe {
                objc::CFRelease(self.model);
            }
            self.model = std::ptr::null_mut();
        }
    }
}
