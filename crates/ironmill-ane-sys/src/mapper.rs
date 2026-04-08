//! Safe wrapper for `_ANEProgramIOSurfacesMapper`.
//!
//! This type manages IOSurface mapping/unmapping for ANE programs,
//! bridging between compiled models, requests, and the ANE device
//! controller.

use std::ffi::c_void;

use crate::error::AneSysError;
use crate::objc::{
    ane_safe_cfrelease, extract_nserror_description, get_class, objc_msgSend, objc_retain, sel,
    sel_registerName,
};

// ───────────────────────────────────────────────────────────────────
// ProgramIOSurfacesMapper
// ───────────────────────────────────────────────────────────────────

/// Safe wrapper around `_ANEProgramIOSurfacesMapper`.
///
/// Each instance owns a retained ObjC object handle that is released
/// on drop via `CFRelease`.
pub struct ProgramIOSurfacesMapper {
    raw: *mut c_void,
}

// SAFETY: The raw ObjC handle is only accessed through &self methods and
// ownership is exclusive (no aliasing).
unsafe impl Send for ProgramIOSurfacesMapper {}

impl Drop for ProgramIOSurfacesMapper {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                ane_safe_cfrelease(self.raw as *const c_void);
            }
            self.raw = std::ptr::null_mut();
        }
    }
}

#[cfg(target_os = "macos")]
impl ProgramIOSurfacesMapper {
    /// Return the raw ObjC object pointer.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    /// `+[_ANEProgramIOSurfacesMapper mapperWithController:]`
    ///
    /// Factory — creates a mapper from a device controller.
    ///
    /// # Safety
    ///
    /// `controller` must be a valid `_ANEDeviceController` pointer.
    pub unsafe fn mapper_with_controller(controller: *mut c_void) -> Result<Self, AneSysError> {
        let cls = get_class("_ANEProgramIOSurfacesMapper")?;

        type MsgFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("mapperWithController:")) };
        let f: MsgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, sel, controller) };
        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "mapperWithController: returned null".into(),
            });
        }
        objc_retain(obj);

        Ok(Self { raw: obj })
    }

    /// `+[_ANEProgramIOSurfacesMapper mapperWithProgramHandle:]`
    ///
    /// Factory — creates a mapper from a program handle.
    pub fn mapper_with_program_handle(program_handle: u64) -> Result<Self, AneSysError> {
        let cls = get_class("_ANEProgramIOSurfacesMapper")?;

        type MsgFn = unsafe extern "C" fn(*mut c_void, *mut c_void, u64) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("mapperWithProgramHandle:")) };
        let f: MsgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, sel, program_handle) };
        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "mapperWithProgramHandle: returned null".into(),
            });
        }
        objc_retain(obj);

        Ok(Self { raw: obj })
    }

    /// `-[_ANEProgramIOSurfacesMapper mapIOSurfacesWithModel:request:cacheInference:error:]`
    ///
    /// Maps IOSurfaces for the given model and request.  Returns `Ok(())`
    /// on success, or an error with the `NSError` description on failure.
    ///
    /// # Safety
    ///
    /// `model` must be a valid `_ANEInMemoryModel` pointer and `request` a
    /// valid `_ANERequest` pointer.
    pub unsafe fn map_iosurfaces(
        &self,
        model: *mut c_void,
        request: *mut c_void,
        cache_inference: bool,
    ) -> Result<(), AneSysError> {
        let mut error: *mut c_void = std::ptr::null_mut();
        type MsgFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            i8,
            *mut *mut c_void,
        ) -> i8;
        let sel = unsafe {
            sel_registerName(sel!("mapIOSurfacesWithModel:request:cacheInference:error:"))
        };
        let f: MsgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let cache_flag: i8 = if cache_inference { 1 } else { 0 };
        let ok = unsafe { f(self.raw, sel, model, request, cache_flag, &mut error) };

        if ok == 0 {
            let err_msg = if !error.is_null() {
                extract_nserror_description(error)
            } else {
                "mapIOSurfacesWithModel:request:cacheInference:error: returned NO".into()
            };
            return Err(AneSysError::IOSurfaceMappingFailed(err_msg));
        }
        Ok(())
    }

    /// `-[_ANEProgramIOSurfacesMapper unmapIOSurfacesWithModel:request:error:]`
    ///
    /// Unmaps IOSurfaces for the given model and request.
    ///
    /// # Safety
    ///
    /// `model` must be a valid `_ANEInMemoryModel` pointer and `request` a
    /// valid `_ANERequest` pointer.
    pub unsafe fn unmap_iosurfaces(
        &self,
        model: *mut c_void,
        request: *mut c_void,
    ) -> Result<(), AneSysError> {
        let mut error: *mut c_void = std::ptr::null_mut();
        type MsgFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut *mut c_void,
        ) -> i8;
        let sel = unsafe { sel_registerName(sel!("unmapIOSurfacesWithModel:request:error:")) };
        let f: MsgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ok = unsafe { f(self.raw, sel, model, request, &mut error) };

        if ok == 0 {
            let err_msg = if !error.is_null() {
                extract_nserror_description(error)
            } else {
                "unmapIOSurfacesWithModel:request:error: returned NO".into()
            };
            return Err(AneSysError::IOSurfaceMappingFailed(err_msg));
        }
        Ok(())
    }

    /// `-[_ANEProgramIOSurfacesMapper validateRequest:model:]`
    ///
    /// Validates a request against a model.  Returns `true` if valid.
    ///
    /// # Safety
    ///
    /// `request` must be a valid `_ANERequest` pointer and `model` a valid
    /// `_ANEInMemoryModel` pointer.
    pub unsafe fn validate_request(&self, request: *mut c_void, model: *mut c_void) -> bool {
        type MsgFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("validateRequest:model:")) };
        let f: MsgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let result = unsafe { f(self.raw, sel, request, model) };
        result != 0
    }

    /// `-[_ANEProgramIOSurfacesMapper controller]`
    ///
    /// Returns the `_ANEDeviceController` pointer (raw ObjC object).
    pub fn controller(&self) -> *mut c_void {
        type MsgFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("controller")) };
        let f: MsgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// `-[_ANEProgramIOSurfacesMapper deviceController]`
    ///
    /// Returns the `_ANEDeviceController` pointer (raw ObjC object).
    pub fn device_controller(&self) -> *mut c_void {
        type MsgFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("deviceController")) };
        let f: MsgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// `-[_ANEProgramIOSurfacesMapper programHandle]`
    ///
    /// Returns the program handle as `u64` (type encoding `Q`).
    pub fn program_handle(&self) -> u64 {
        type MsgFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64;
        let sel = unsafe { sel_registerName(sel!("programHandle")) };
        let f: MsgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }
}

impl std::fmt::Debug for ProgramIOSurfacesMapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProgramIOSurfacesMapper")
            .field("raw", &self.raw)
            .finish()
    }
}
