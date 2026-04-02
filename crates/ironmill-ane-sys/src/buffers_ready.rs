//! Safe wrappers for ANE buffer-readiness and output-set primitives.
//!
//! Wraps `_ANEInputBuffersReady` and `_ANEOutputSetEnqueue` — the ObjC
//! classes that describe input buffer state and output set configuration
//! for ANE evaluation requests.

#[cfg(target_os = "macos")]
use std::ffi::c_void;

#[cfg(target_os = "macos")]
use crate::error::AneSysError;
#[cfg(target_os = "macos")]
use crate::objc::{CFRelease, get_class, objc_msgSend, objc_retain, sel, sel_registerName};

// ---------------------------------------------------------------------------
// InputBuffersReady
// ---------------------------------------------------------------------------

/// Describes input buffer readiness for an ANE procedure (`_ANEInputBuffersReady`).
#[cfg(target_os = "macos")]
pub struct InputBuffersReady {
    raw: *mut c_void,
}

#[cfg(target_os = "macos")]
unsafe impl Send for InputBuffersReady {}

#[cfg(target_os = "macos")]
impl InputBuffersReady {
    /// Create via `+[_ANEInputBuffersReady inputBuffersWithProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:]`.
    ///
    /// # Safety
    ///
    /// `input_buffer_info_index` and `input_free_value` must be valid `NSArray` pointers (or null).
    pub unsafe fn new(
        procedure_index: u32,
        input_buffer_info_index: *mut c_void,
        input_free_value: *mut c_void,
        execution_delay: u64,
    ) -> Result<Self, AneSysError> {
        let cls = get_class("_ANEInputBuffersReady")?;

        type FactoryFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            u32,
            *mut c_void,
            *mut c_void,
            u64,
        ) -> *mut c_void;

        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe {
            sel_registerName(sel!(
                "inputBuffersWithProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:"
            ))
        };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: FactoryFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: cls is valid; arguments match the ObjC type encoding.
        let obj = unsafe {
            f(
                cls,
                s,
                procedure_index,
                input_buffer_info_index,
                input_free_value,
                execution_delay,
            )
        };

        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "inputBuffersWithProcedureIndex:... returned nil".into(),
            });
        }

        // Retain — factory returns autoreleased.
        objc_retain(obj);
        Ok(Self { raw: obj })
    }

    /// Raw `_ANEInputBuffersReady` pointer.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    /// Validate the input buffers (`-validate`).
    pub fn validate(&self) -> bool {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u8;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("validate")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANEInputBuffersReady.
        unsafe { f(self.raw, s) != 0 }
    }

    /// Get the procedure index (`-procedureIndex`).
    pub fn procedure_index(&self) -> u32 {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("procedureIndex")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANEInputBuffersReady.
        unsafe { f(self.raw, s) }
    }

    /// Get the execution delay (`-executionDelay`).
    pub fn execution_delay(&self) -> u64 {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("executionDelay")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANEInputBuffersReady.
        unsafe { f(self.raw, s) }
    }

    /// Get the input buffer info index NSArray (`-inputBufferInfoIndex`).
    pub fn input_buffer_info_index(&self) -> *mut c_void {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("inputBufferInfoIndex")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANEInputBuffersReady.
        unsafe { f(self.raw, s) }
    }

    /// Get the input free value NSArray (`-inputFreeValue`).
    pub fn input_free_value(&self) -> *mut c_void {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("inputFreeValue")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANEInputBuffersReady.
        unsafe { f(self.raw, s) }
    }
}

#[cfg(target_os = "macos")]
impl Drop for InputBuffersReady {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: CFRelease on a retained ObjC object.
            unsafe { CFRelease(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// OutputSetEnqueue
// ---------------------------------------------------------------------------

/// Describes an output set for ANE enqueue (`_ANEOutputSetEnqueue`).
#[cfg(target_os = "macos")]
pub struct OutputSetEnqueue {
    raw: *mut c_void,
}

#[cfg(target_os = "macos")]
unsafe impl Send for OutputSetEnqueue {}

#[cfg(target_os = "macos")]
impl OutputSetEnqueue {
    /// Create via `+[_ANEOutputSetEnqueue outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:]`.
    pub fn new(
        procedure_index: u32,
        set_index: u32,
        signal_value: u64,
        signal_not_required: bool,
        is_open_loop: bool,
    ) -> Result<Self, AneSysError> {
        let cls = get_class("_ANEOutputSetEnqueue")?;

        type FactoryFn =
            unsafe extern "C" fn(*mut c_void, *mut c_void, u32, u32, u64, u8, u8) -> *mut c_void;

        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe {
            sel_registerName(sel!(
                "outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:"
            ))
        };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: FactoryFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: cls is valid; BOOL args passed as u8 (0/1).
        let obj = unsafe {
            f(
                cls,
                s,
                procedure_index,
                set_index,
                signal_value,
                signal_not_required as u8,
                is_open_loop as u8,
            )
        };

        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "outputSetWithProcedureIndex:... returned nil".into(),
            });
        }

        // Retain — factory returns autoreleased.
        objc_retain(obj);
        Ok(Self { raw: obj })
    }

    /// Raw `_ANEOutputSetEnqueue` pointer.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    /// Get the procedure index (`-procedureIndex`).
    pub fn procedure_index(&self) -> u32 {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("procedureIndex")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANEOutputSetEnqueue.
        unsafe { f(self.raw, s) }
    }

    /// Get the set index (`-setIndex`).
    pub fn set_index(&self) -> u32 {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("setIndex")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANEOutputSetEnqueue.
        unsafe { f(self.raw, s) }
    }

    /// Get the signal value (`-signalValue`).
    pub fn signal_value(&self) -> u64 {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("signalValue")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANEOutputSetEnqueue.
        unsafe { f(self.raw, s) }
    }

    /// Get the signal-not-required flag (`-signalNotRequired`).
    pub fn signal_not_required(&self) -> bool {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u8;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("signalNotRequired")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANEOutputSetEnqueue; B returns 1-byte _Bool.
        unsafe { f(self.raw, s) != 0 }
    }

    /// Get the open-loop flag (`-isOpenLoop`).
    pub fn is_open_loop(&self) -> bool {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u8;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("isOpenLoop")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANEOutputSetEnqueue; B returns 1-byte _Bool.
        unsafe { f(self.raw, s) != 0 }
    }
}

#[cfg(target_os = "macos")]
impl Drop for OutputSetEnqueue {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: CFRelease on a retained ObjC object.
            unsafe { CFRelease(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}
