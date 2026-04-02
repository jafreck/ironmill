//! ANE program-for-evaluation wrapper.
//!
//! Wraps Apple's private `_ANEProgramForEvaluation` class from the
//! `AppleNeuralEngine` framework.
//!
//! # ⚠️ Private API — see crate-level docs for risks and caveats.

use std::ffi::c_void;

use crate::error::AneSysError;
use crate::objc::{CFRelease, get_class, objc_msgSend, objc_retain, sel, sel_registerName};

// ---------------------------------------------------------------------------
// ProgramForEvaluation — wraps _ANEProgramForEvaluation
// ---------------------------------------------------------------------------

/// Wrapper around `_ANEProgramForEvaluation`, which pairs a compiled program
/// handle with an intermediate buffer and queue depth for ANE evaluation.
///
/// Owns a retained ObjC object; released on drop.
pub struct ProgramForEvaluation {
    raw: *mut c_void,
}

// SAFETY: The raw handle is only accessed through &self methods.
unsafe impl Send for ProgramForEvaluation {}

impl ProgramForEvaluation {
    /// Raw ObjC object pointer.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    /// Create a program from a device controller, intermediate buffer handle,
    /// and queue depth.
    ///
    /// Calls `+[_ANEProgramForEvaluation programWithController:intermediateBufferHandle:queueDepth:]`.
    ///
    /// # Safety
    ///
    /// `controller` must be a valid, retained `_ANEDeviceController` ObjC
    /// object pointer (e.g. from [`DeviceController::as_raw`]).
    pub unsafe fn with_controller(
        controller: *mut c_void,
        intermediate_buffer_handle: u64,
        queue_depth: i8,
    ) -> Result<Self, AneSysError> {
        let cls = get_class("_ANEProgramForEvaluation")?;
        type FactoryFn =
            unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, u64, i8) -> *mut c_void;
        let sel = unsafe {
            sel_registerName(sel!(
                "programWithController:intermediateBufferHandle:queueDepth:"
            ))
        };
        let f: FactoryFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe {
            f(
                cls,
                sel,
                controller,
                intermediate_buffer_handle,
                queue_depth,
            )
        };
        if obj.is_null() {
            return Err(AneSysError::ProgramCreationFailed(
                "programWithController:intermediateBufferHandle:queueDepth: returned nil".into(),
            ));
        }
        objc_retain(obj);
        Ok(Self { raw: obj })
    }

    /// Create a program from raw handles and queue depth.
    ///
    /// Calls `+[_ANEProgramForEvaluation programWithHandle:intermediateBufferHandle:queueDepth:]`.
    pub fn with_handle(
        handle: u64,
        intermediate_buffer_handle: u64,
        queue_depth: i8,
    ) -> Result<Self, AneSysError> {
        let cls = get_class("_ANEProgramForEvaluation")?;
        type FactoryFn =
            unsafe extern "C" fn(*mut c_void, *mut c_void, u64, u64, i8) -> *mut c_void;
        let sel = unsafe {
            sel_registerName(sel!(
                "programWithHandle:intermediateBufferHandle:queueDepth:"
            ))
        };
        let f: FactoryFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, sel, handle, intermediate_buffer_handle, queue_depth) };
        if obj.is_null() {
            return Err(AneSysError::ProgramCreationFailed(
                "programWithHandle:intermediateBufferHandle:queueDepth: returned nil".into(),
            ));
        }
        objc_retain(obj);
        Ok(Self { raw: obj })
    }

    // -- Getters / Setters ---------------------------------------------------

    /// The program handle (`u64`).
    pub fn program_handle(&self) -> u64 {
        type U64Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64;
        let sel = unsafe { sel_registerName(sel!("programHandle")) };
        let f: U64Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Set the program handle.
    pub fn set_program_handle(&self, handle: u64) {
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, u64);
        let sel = unsafe { sel_registerName(sel!("setProgramHandle:")) };
        let f: SetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, handle) };
    }

    /// The intermediate buffer handle (`u64`).
    pub fn intermediate_buffer_handle(&self) -> u64 {
        type U64Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64;
        let sel = unsafe { sel_registerName(sel!("intermediateBufferHandle")) };
        let f: U64Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Set the intermediate buffer handle.
    pub fn set_intermediate_buffer_handle(&self, handle: u64) {
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, u64);
        let sel = unsafe { sel_registerName(sel!("setIntermediateBufferHandle:")) };
        let f: SetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, handle) };
    }

    /// Queue depth (`i8`).
    pub fn queue_depth(&self) -> i8 {
        type I8Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("queueDepth")) };
        let f: I8Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Number of async requests currently in flight (`i64`).
    pub fn current_async_requests_in_flight(&self) -> i64 {
        type I64Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i64;
        let sel = unsafe { sel_registerName(sel!("currentAsyncRequestsInFlight")) };
        let f: I64Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Set the number of async requests currently in flight.
    pub fn set_current_async_requests_in_flight(&self, count: i64) {
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, i64);
        let sel = unsafe { sel_registerName(sel!("setCurrentAsyncRequestsInFlight:")) };
        let f: SetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, count) };
    }

    /// The dispatch semaphore guarding requests in flight.
    pub fn requests_in_flight(&self) -> *mut c_void {
        type PtrFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("requestsInFlight")) };
        let f: PtrFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// The associated `_ANEDeviceController` object (raw pointer).
    pub fn controller(&self) -> *mut c_void {
        type PtrFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("controller")) };
        let f: PtrFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }
}

impl Drop for ProgramForEvaluation {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { CFRelease(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_handle_returns_err_or_ok() {
        // On machines without ANE the class lookup may fail; that's fine.
        let _ = ProgramForEvaluation::with_handle(0, 0, 1);
    }
}
