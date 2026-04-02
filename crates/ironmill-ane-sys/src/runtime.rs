//! ANE runtime — `_ANEInMemoryModel` lifecycle and program execution.
//!
//! Wraps the private ANE classes for loading compiled programs and
//! executing inference on the Apple Neural Engine.
//!
//! `AneRuntime` is `Send` but **not** `Sync` — the underlying ANE objects
//! are assumed thread-unsafe.

use std::ffi::c_void;

use crate::error::AneSysError;
use crate::objc::{
    CFRelease, ane_framework, extract_nserror_description, get_class, ns_array_add,
    ns_empty_dict_unchecked, ns_mutable_array, ns_number_autoreleased, objc_msgSend, objc_retain,
    sel, sel_registerName,
};
use crate::{CompiledProgram, LoadedProgram};

/// QoS value used for load/eval (matches Orion's constant of 21).
const ANE_QOS: i64 = 21;

/// ANE runtime wrapping the private ANE framework classes.
///
/// Resolves `_ANEIOSurfaceObject` and `_ANERequest` classes on init.
pub struct AneRuntime {
    /// `_ANEIOSurfaceObject` class pointer.
    aio_cls: *mut c_void,
    /// `_ANERequest` class pointer.
    req_cls: *mut c_void,
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

        Ok(Self { aio_cls, req_cls })
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
        if program.model.is_null() {
            return Err(AneSysError::EvalFailed {
                status: 0,
                context: "CompiledProgram has null model handle".into(),
            });
        }

        // Retain the model for the LoadedProgram handle.
        objc_retain(program.model);

        Ok(LoadedProgram {
            model: program.model,
        })
    }

    /// Execute a loaded program with IOSurface-backed I/O buffers.
    ///
    /// `input_surfaces` and `output_surfaces` are raw IOSurface pointers
    /// (from `AneTensor::as_ptr()` or equivalent).
    pub fn eval(
        &self,
        program: &LoadedProgram,
        input_surfaces: &[*mut c_void],
        output_surfaces: &[*mut c_void],
    ) -> Result<(), AneSysError> {
        // SAFETY: program.model is valid (held by LoadedProgram), surfaces are caller-provided.
        unsafe { self.eval_raw(program.model, input_surfaces, output_surfaces) }
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
        if input_surfaces.is_empty() {
            return Err(AneSysError::EvalFailed {
                status: 0,
                context: "at least one input surface is required".into(),
            });
        }
        if output_surfaces.is_empty() {
            return Err(AneSysError::EvalFailed {
                status: 0,
                context: "at least one output surface is required".into(),
            });
        }

        // Wrap inputs in _ANEIOSurfaceObject and build index arrays
        let in_arr = ns_mutable_array()?;
        let in_idx = ns_mutable_array()?;
        for (i, &surface) in input_surfaces.iter().enumerate() {
            let wrapped = self.wrap_iosurface(surface)?;
            ns_array_add(in_arr, wrapped);
            ns_array_add(in_idx, ns_number_autoreleased(i as i64)?);
        }

        // Wrap outputs
        let out_arr = ns_mutable_array()?;
        let out_idx = ns_mutable_array()?;
        for (i, &surface) in output_surfaces.iter().enumerate() {
            let wrapped = self.wrap_iosurface(surface)?;
            ns_array_add(out_arr, wrapped);
            ns_array_add(out_idx, ns_number_autoreleased(i as i64)?);
        }

        // Build _ANERequest
        let zero = ns_number_autoreleased(0)?;
        type RequestFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
        ) -> *mut c_void;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let req_sel = unsafe {
            sel_registerName(sel!(
                "requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"
            ))
        };
        // SAFETY: transmute objc_msgSend to the correct 9-arg signature.
        let req_fn: RequestFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: all arguments are valid ObjC objects or nil pointers.
        let request = unsafe {
            req_fn(
                self.req_cls,
                req_sel,
                in_arr,
                in_idx,
                out_arr,
                out_idx,
                std::ptr::null_mut(), // weightsBuffer: nil
                std::ptr::null_mut(), // perfStats: nil
                zero,                 // procedureIndex: @0
            )
        };

        if request.is_null() {
            // SAFETY: CFRelease on ObjC collection objects.
            unsafe {
                CFRelease(in_arr);
                CFRelease(in_idx);
                CFRelease(out_arr);
                CFRelease(out_idx);
            }
            return Err(AneSysError::EvalFailed {
                status: 0,
                context: "failed to create _ANERequest".into(),
            });
        }

        // Evaluate: [model evaluateWithQoS:21 options:@{} request:req error:&e]
        let empty_dict = ns_empty_dict_unchecked()?;
        let mut error: *mut c_void = std::ptr::null_mut();

        type EvalFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            i64,
            *mut c_void,
            *mut c_void,
            *mut *mut c_void,
        ) -> i8;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let eval_sel = unsafe { sel_registerName(sel!("evaluateWithQoS:options:request:error:")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let eval_fn: EvalFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: model/eval_sel/empty_dict/request are valid; error is a valid out-pointer.
        let ok = unsafe { eval_fn(model, eval_sel, ANE_QOS, empty_dict, request, &mut error) };

        // Release temporary ObjC collections
        // SAFETY: CFRelease on ObjC collection objects.
        unsafe {
            CFRelease(in_arr);
            CFRelease(in_idx);
            CFRelease(out_arr);
            CFRelease(out_idx);
            CFRelease(empty_dict);
        }

        if ok == 0 {
            let err_msg = if !error.is_null() {
                extract_nserror_description(error)
            } else {
                "evaluateWithQoS:options:request:error: returned NO".into()
            };
            return Err(AneSysError::EvalFailed {
                status: 1,
                context: err_msg,
            });
        }

        Ok(())
    }

    /// Load a previously compiled program into the ANE.
    pub fn load_compiled(&self, program: &CompiledProgram) -> Result<(), AneSysError> {
        if program.model.is_null() {
            return Err(AneSysError::EvalFailed {
                status: 0,
                context: "load_compiled: null model pointer".into(),
            });
        }

        let mut error: *mut c_void = std::ptr::null_mut();
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let sel = unsafe { sel_registerName(sel!("loadWithQoS:options:error:")) };
        type LoadFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            i64,
            *mut c_void,
            *mut *mut c_void,
        ) -> i8;
        // SAFETY: transmute objc_msgSend to the correct signature.
        let load_fn: LoadFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let empty_dict = ns_empty_dict_unchecked()?;
        let ok = unsafe { load_fn(program.model, sel, ANE_QOS, empty_dict, &mut error) };
        // SAFETY: CFRelease on the dictionary.
        unsafe { CFRelease(empty_dict) };

        if ok == 0 {
            let desc = if !error.is_null() {
                extract_nserror_description(error)
            } else {
                "unknown load error".into()
            };
            return Err(AneSysError::EvalFailed {
                status: 0,
                context: format!("loadWithQoS failed: {desc}"),
            });
        }
        Ok(())
    }

    /// Unload a compiled program from the ANE, freeing the execution slot.
    pub fn unload_compiled(&self, program: &CompiledProgram) {
        if !program.model.is_null() {
            let mut error: *mut c_void = std::ptr::null_mut();
            type UnloadFn =
                unsafe extern "C" fn(*mut c_void, *mut c_void, i64, *mut *mut c_void) -> i8;
            // SAFETY: sel_registerName with a valid null-terminated selector.
            let sel = unsafe { sel_registerName(sel!("unloadWithQoS:error:")) };
            // SAFETY: transmute objc_msgSend to the correct signature.
            let unload_fn: UnloadFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
            // SAFETY: program.model/sel are valid; error is a valid out-pointer.
            unsafe { unload_fn(program.model, sel, ANE_QOS, &mut error) };
        }
    }

    /// Load, evaluate, and unload a compiled program in one call.
    pub fn eval_compiled(
        &self,
        program: &CompiledProgram,
        input_surfaces: &[*mut c_void],
        output_surfaces: &[*mut c_void],
    ) -> Result<(), AneSysError> {
        self.load_compiled(program)?;
        // SAFETY: program.model is valid after load_compiled, surfaces are caller-provided.
        let result = unsafe { self.eval_raw(program.model, input_surfaces, output_surfaces) };
        self.unload_compiled(program);
        result
    }

    /// Unload a program, freeing ANE resources.
    pub fn unload_program(&self, mut program: LoadedProgram) {
        if !program.model.is_null() {
            let mut error: *mut c_void = std::ptr::null_mut();
            type UnloadFn =
                unsafe extern "C" fn(*mut c_void, *mut c_void, i64, *mut *mut c_void) -> i8;
            // SAFETY: sel_registerName with a valid null-terminated selector.
            let sel = unsafe { sel_registerName(sel!("unloadWithQoS:error:")) };
            // SAFETY: transmute objc_msgSend to the correct signature.
            let unload_fn: UnloadFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
            // SAFETY: program.model/sel are valid; error is a valid out-pointer.
            unsafe { unload_fn(program.model, sel, ANE_QOS, &mut error) };

            // SAFETY: CFRelease on the retained model handle.
            unsafe { CFRelease(program.model) };
            program.model = std::ptr::null_mut();
        }
    }

    /// Wrap an IOSurface pointer in `_ANEIOSurfaceObject`.
    fn wrap_iosurface(&self, surface: *mut c_void) -> Result<*mut c_void, AneSysError> {
        type WrapFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let sel = unsafe { sel_registerName(sel!("objectWithIOSurface:")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: WrapFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: aio_cls is a valid class; surface is a valid IOSurface pointer.
        let wrapped = unsafe { f(self.aio_cls, sel, surface) };
        if wrapped.is_null() {
            return Err(AneSysError::EvalFailed {
                status: 0,
                context: "objectWithIOSurface: returned nil".into(),
            });
        }
        Ok(wrapped)
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
