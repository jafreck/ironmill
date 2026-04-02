//! ANE runtime — backward-compatible thin wrapper over [`crate::model`].
//!
//! All eval/load/unload logic now lives in `model.rs`.  This module
//! delegates to it and converts between `InMemoryModel` and the legacy
//! `CompiledProgram`/`LoadedProgram` handle types.
//!
//! `AneRuntime` is `Send` but **not** `Sync` — the underlying ANE objects
//! are assumed thread-unsafe.

use std::ffi::c_void;

use crate::error::AneSysError;
use crate::objc::{ane_framework, get_class, objc_retain};
use crate::{CompiledProgram, LoadedProgram};

/// ANE runtime wrapping the private ANE framework classes.
///
/// Resolves `_ANEIOSurfaceObject` and `_ANERequest` classes on init.
pub struct AneRuntime {
    /// `_ANEIOSurfaceObject` class pointer (kept for backward compat).
    _aio_cls: *mut c_void,
    /// `_ANERequest` class pointer (kept for backward compat).
    _req_cls: *mut c_void,
}

// SAFETY: The class pointers are only accessed through &self methods.
// Not Sync because the underlying ANE framework is not thread-safe.
unsafe impl Send for AneRuntime {}

impl AneRuntime {
    /// Initialize the ANE runtime.
    ///
    /// Loads the AppleNeuralEngine framework (via the shared lazy handle) and
    /// resolves the `_ANEIOSurfaceObject` and `_ANERequest` classes.
    pub fn new() -> Result<Self, AneSysError> {
        ane_framework().map_err(|_| AneSysError::EvalFailed {
            status: 0,
            context: "failed to dlopen AppleNeuralEngine.framework".into(),
        })?;

        let aio_cls = get_class("_ANEIOSurfaceObject")?;
        let req_cls = get_class("_ANERequest")?;

        Ok(Self {
            _aio_cls: aio_cls,
            _req_cls: req_cls,
        })
    }

    /// Check if the ANE runtime is available on this system.
    pub fn is_available() -> bool {
        if ane_framework().is_err() {
            return false;
        }
        get_class("_ANEInMemoryModel").is_ok()
    }

    /// Load a compiled program for execution.
    ///
    /// In the Orion-based API, the model is already compiled + loaded after
    /// `compile_mil_text`. This retains the model handle into a `LoadedProgram`.
    pub fn load_program(&self, program: &CompiledProgram) -> Result<LoadedProgram, AneSysError> {
        if program.as_raw_ptr().is_null() {
            return Err(AneSysError::EvalFailed {
                status: 0,
                context: "CompiledProgram has null model handle".into(),
            });
        }

        // Retain the model for the LoadedProgram handle.
        objc_retain(program.as_raw_ptr());

        Ok(LoadedProgram {
            model: program.as_raw_ptr(),
        })
    }

    /// Execute a loaded program with IOSurface-backed I/O buffers.
    ///
    /// Delegates to [`crate::model::eval`].
    pub fn eval(
        &self,
        program: &LoadedProgram,
        input_surfaces: &[*mut c_void],
        output_surfaces: &[*mut c_void],
    ) -> Result<(), AneSysError> {
        let model = unsafe { crate::model::InMemoryModel::from_raw(program.model) };
        let result = crate::model::eval(&model, input_surfaces, output_surfaces);
        // Don't let the temporary InMemoryModel drop release the pointer —
        // it's still owned by LoadedProgram.
        std::mem::forget(model);
        result
    }

    /// Core eval implementation taking a raw model pointer.
    ///
    /// # Safety
    ///
    /// `model` must be a valid, non-null `_ANEInMemoryModel` pointer.
    /// All pointers in `input_surfaces` and `output_surfaces` must be valid IOSurface pointers.
    pub unsafe fn eval_raw(
        &self,
        model: *mut c_void,
        input_surfaces: &[*mut c_void],
        output_surfaces: &[*mut c_void],
    ) -> Result<(), AneSysError> {
        let m = unsafe { crate::model::InMemoryModel::from_raw(model) };
        let result = crate::model::eval(&m, input_surfaces, output_surfaces);
        std::mem::forget(m);
        result
    }

    /// Load a previously compiled program into the ANE.
    ///
    /// Delegates to [`InMemoryModel::load`].
    pub fn load_compiled(&self, program: &CompiledProgram) -> Result<(), AneSysError> {
        let model = program.as_model();
        model.load(21)
    }

    /// Unload a compiled program from the ANE, freeing the execution slot.
    pub fn unload_compiled(&self, program: &CompiledProgram) {
        let model = program.as_model();
        let _ = model.unload(21);
    }

    /// Load, evaluate, and unload a compiled program in one call.
    pub fn eval_compiled(
        &self,
        program: &CompiledProgram,
        input_surfaces: &[*mut c_void],
        output_surfaces: &[*mut c_void],
    ) -> Result<(), AneSysError> {
        self.load_compiled(program)?;
        let m = unsafe { crate::model::InMemoryModel::from_raw(program.as_raw_ptr()) };
        let result = crate::model::eval(&m, input_surfaces, output_surfaces);
        std::mem::forget(m);
        self.unload_compiled(program);
        result
    }

    /// Unload a program, freeing ANE resources.
    pub fn unload_program(&self, mut program: LoadedProgram) {
        if !program.model.is_null() {
            let m = unsafe { crate::model::InMemoryModel::from_raw(program.model) };
            let _ = m.unload(21);
            // Let InMemoryModel::drop release the pointer.
            drop(m);
            program.model = std::ptr::null_mut();
        }
    }
}

impl Drop for AneRuntime {
    fn drop(&mut self) {
        // Class pointers are not owned — no cleanup needed.
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_is_available() {
        let _ = AneRuntime::is_available();
    }

    #[test]
    fn runtime_new_on_non_ane() {
        let result = AneRuntime::new();
        match result {
            Ok(_) => {}
            Err(e) => {
                let msg = format!("{e}");
                assert!(
                    msg.contains("ANE")
                        || msg.contains("dlopen")
                        || msg.contains("class not found"),
                    "unexpected error: {msg}"
                );
            }
        }
    }
}
