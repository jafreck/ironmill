//! Compiled ANE program handle types.
//!
//! These types bridge the compiler (in `mil-rs`) and the runtime
//! (in this crate), representing compiled and loaded program states.

use std::ffi::c_void;

/// A compiled ANE program, ready to be loaded into the runtime.
///
/// Produced by `_ANECompiler` via [`mil_rs::ffi::ane::AneCompiler`].
/// Contains the opaque compiled program handle from the ANE framework.
#[allow(dead_code)]
pub struct CompiledProgram {
    /// Opaque handle to the compiled ANE program object.
    pub(crate) inner: *mut c_void,
}

// SAFETY: CompiledProgram can be sent between threads — the handle
// is an opaque pointer that is not accessed until loaded.
unsafe impl Send for CompiledProgram {}

/// A program that has been loaded into the ANE runtime for execution.
///
/// Created by [`AneRuntime::load_program`](crate::runtime::AneRuntime).
#[allow(dead_code)]
pub struct LoadedProgram {
    /// Opaque handle to the loaded program in the ANE runtime.
    pub(crate) inner: *mut c_void,
}

// SAFETY: LoadedProgram is Send (can move between threads) but NOT Sync
// (cannot be shared — ANE client is not thread-safe).
unsafe impl Send for LoadedProgram {}
