//! Compiled ANE program handle types.
//!
//! These types bridge the compiler (in `mil-rs`) and the runtime
//! (in this crate), representing compiled and loaded program states.
//!
//! In the Orion-based API, the `_ANEInMemoryModel` instance IS both
//! the compiled and loaded program — the same object is used for
//! compile, load, and eval.

use std::ffi::c_void;

/// A compiled ANE program, ready to be loaded into the runtime.
///
/// Produced by [`mil_rs::ffi::ane::AneCompiler::compile_mil_text`].
/// Contains the `_ANEInMemoryModel` handle which is already compiled
/// and loaded after `compile_mil_text` returns.
#[allow(dead_code)]
pub struct CompiledProgram {
    /// Opaque handle to the `_ANEInMemoryModel` (compiled + loaded).
    pub(crate) model: *mut c_void,
}

impl CompiledProgram {
    /// Create from a raw model pointer returned by
    /// [`mil_rs::ffi::ane::AneCompiler::compile_mil_text`].
    ///
    /// # Safety
    /// The caller must ensure `ptr` is a valid, retained `_ANEInMemoryModel`
    /// handle that has been compiled and loaded.
    pub unsafe fn from_raw(ptr: *mut c_void) -> Self {
        Self { model: ptr }
    }
}

// SAFETY: CompiledProgram can be sent between threads — the handle
// is an opaque pointer that is not accessed until eval.
unsafe impl Send for CompiledProgram {}

/// A program that has been loaded into the ANE runtime for execution.
///
/// In the Orion-based API, this wraps the same `_ANEInMemoryModel`
/// handle as `CompiledProgram` — loading is done at compile time.
#[allow(dead_code)]
pub struct LoadedProgram {
    /// Opaque handle to the `_ANEInMemoryModel` (compiled + loaded).
    pub(crate) model: *mut c_void,
}

// SAFETY: LoadedProgram is Send (can move between threads) but NOT Sync
// (cannot be shared — ANE is not thread-safe).
unsafe impl Send for LoadedProgram {}
