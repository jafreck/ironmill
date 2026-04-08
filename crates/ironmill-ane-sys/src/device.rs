//! ANE device info and controller wrappers.
//!
//! Wraps Apple's private `_ANEDeviceInfo` and `_ANEDeviceController` classes
//! from the `AppleNeuralEngine` framework.
//!
//! # ⚠️ Private API — see crate-level docs for risks and caveats.

use std::ffi::c_void;

use crate::error::AneSysError;
use crate::objc::{
    CFRelease, create_nsstring, get_class, nsstring_to_string, objc_msgSend, safe_release, sel,
    sel_registerName,
};

// ---------------------------------------------------------------------------
// DeviceInfo — all class methods, no instance state
// ---------------------------------------------------------------------------

/// Queries about ANE hardware presence and capabilities.
///
/// All methods are class methods on `_ANEDeviceInfo` — no instance is needed.
pub struct DeviceInfo;

impl std::fmt::Debug for DeviceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceInfo").finish()
    }
}

impl DeviceInfo {
    /// Returns `true` if ANE hardware is present on this machine.
    pub fn has_ane() -> Result<bool, AneSysError> {
        let cls = get_class("_ANEDeviceInfo")?;
        type BoolFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("hasANE")) };
        let f: BoolFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        Ok(unsafe { f(cls, sel) } != 0)
    }

    /// Number of ANE units on this machine.
    pub fn num_anes() -> Result<u32, AneSysError> {
        let cls = get_class("_ANEDeviceInfo")?;
        type UintFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32;
        let sel = unsafe { sel_registerName(sel!("numANEs")) };
        let f: UintFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        Ok(unsafe { f(cls, sel) })
    }

    /// Number of ANE cores across all units.
    pub fn num_ane_cores() -> Result<u32, AneSysError> {
        let cls = get_class("_ANEDeviceInfo")?;
        type UintFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32;
        let sel = unsafe { sel_registerName(sel!("numANECores")) };
        let f: UintFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        Ok(unsafe { f(cls, sel) })
    }

    /// ANE architecture type (e.g. `"ane"`, `"ane2"`).
    pub fn architecture_type() -> Result<Option<String>, AneSysError> {
        let cls = get_class("_ANEDeviceInfo")?;
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("aneArchitectureType")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ns = unsafe { f(cls, sel) };
        Ok(unsafe { nsstring_to_string(ns) })
    }

    /// ANE sub-type string.
    pub fn sub_type() -> Result<Option<String>, AneSysError> {
        let cls = get_class("_ANEDeviceInfo")?;
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("aneSubType")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ns = unsafe { f(cls, sel) };
        Ok(unsafe { nsstring_to_string(ns) })
    }

    /// ANE sub-type variant string.
    pub fn sub_type_variant() -> Result<Option<String>, AneSysError> {
        let cls = get_class("_ANEDeviceInfo")?;
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("aneSubTypeVariant")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ns = unsafe { f(cls, sel) };
        Ok(unsafe { nsstring_to_string(ns) })
    }

    /// ANE sub-type and variant combined string.
    pub fn sub_type_and_variant() -> Result<Option<String>, AneSysError> {
        let cls = get_class("_ANEDeviceInfo")?;
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("aneSubTypeAndVariant")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ns = unsafe { f(cls, sel) };
        Ok(unsafe { nsstring_to_string(ns) })
    }

    /// ANE sub-type product variant string.
    pub fn sub_type_product_variant() -> Result<Option<String>, AneSysError> {
        let cls = get_class("_ANEDeviceInfo")?;
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("aneSubTypeProductVariant")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ns = unsafe { f(cls, sel) };
        Ok(unsafe { nsstring_to_string(ns) })
    }

    /// ANE board type identifier.
    pub fn board_type() -> Result<i64, AneSysError> {
        let cls = get_class("_ANEDeviceInfo")?;
        type I64Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i64;
        let sel = unsafe { sel_registerName(sel!("aneBoardType")) };
        let f: I64Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        Ok(unsafe { f(cls, sel) })
    }

    /// Product name string (e.g. `"MacBookPro18,3"`).
    pub fn product_name() -> Result<Option<String>, AneSysError> {
        let cls = get_class("_ANEDeviceInfo")?;
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("productName")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ns = unsafe { f(cls, sel) };
        Ok(unsafe { nsstring_to_string(ns) })
    }

    /// macOS build version string.
    pub fn build_version() -> Result<Option<String>, AneSysError> {
        let cls = get_class("_ANEDeviceInfo")?;
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("buildVersion")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ns = unsafe { f(cls, sel) };
        Ok(unsafe { nsstring_to_string(ns) })
    }

    /// Kernel boot arguments string.
    pub fn boot_args() -> Result<Option<String>, AneSysError> {
        let cls = get_class("_ANEDeviceInfo")?;
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("bootArgs")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ns = unsafe { f(cls, sel) };
        Ok(unsafe { nsstring_to_string(ns) })
    }

    /// Whether this is an Apple-internal build.
    pub fn is_internal_build() -> Result<bool, AneSysError> {
        let cls = get_class("_ANEDeviceInfo")?;
        type BoolFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("isInternalBuild")) };
        let f: BoolFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        Ok(unsafe { f(cls, sel) } != 0)
    }

    /// Whether this machine is a virtual machine.
    pub fn is_virtual_machine() -> Result<bool, AneSysError> {
        let cls = get_class("_ANEDeviceInfo")?;
        type BoolFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("isVirtualMachine")) };
        let f: BoolFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        Ok(unsafe { f(cls, sel) } != 0)
    }

    /// Check if a specific boot argument is present.
    pub fn is_boot_arg_present(arg: &str) -> Result<bool, AneSysError> {
        let cls = get_class("_ANEDeviceInfo")?;
        let ns_arg = create_nsstring(arg)?;
        type BoolArgFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("isBootArgPresent:")) };
        let f: BoolArgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let result = unsafe { f(cls, sel, ns_arg) } != 0;
        unsafe { CFRelease(ns_arg) };
        Ok(result)
    }

    /// Check if a boolean boot argument is set to true.
    pub fn is_bool_boot_arg_set_true(arg: &str) -> Result<bool, AneSysError> {
        let cls = get_class("_ANEDeviceInfo")?;
        let ns_arg = create_nsstring(arg)?;
        type BoolArgFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("isBoolBootArgSetTrue:")) };
        let f: BoolArgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let result = unsafe { f(cls, sel, ns_arg) } != 0;
        unsafe { CFRelease(ns_arg) };
        Ok(result)
    }

    /// Whether pre-compiled model checks are disabled.
    pub fn precompiled_model_checks_disabled() -> Result<bool, AneSysError> {
        let cls = get_class("_ANEDeviceInfo")?;
        type BoolFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("precompiledModelChecksDisabled")) };
        let f: BoolFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        Ok(unsafe { f(cls, sel) } != 0)
    }

    /// Whether excessive power drain when idle is detected.
    pub fn is_excessive_power_drain_when_idle() -> Result<bool, AneSysError> {
        let cls = get_class("_ANEDeviceInfo")?;
        type BoolFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("isExcessivePowerDrainWhenIdle")) };
        let f: BoolFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        Ok(unsafe { f(cls, sel) } != 0)
    }
}

// ---------------------------------------------------------------------------
// DeviceController — wraps _ANEDeviceController
// ---------------------------------------------------------------------------

/// Wrapper around `_ANEDeviceController`, which manages an ANE device handle.
///
/// Owns a retained ObjC object; released on drop.
pub struct DeviceController {
    raw: *mut c_void,
}

// SAFETY: The raw handle is only accessed through &self methods.
unsafe impl Send for DeviceController {}

impl std::fmt::Debug for DeviceController {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceController")
            .field("raw", &self.raw)
            .finish()
    }
}

impl DeviceController {
    /// Raw ObjC object pointer.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    /// Create a controller for the given program handle.
    ///
    /// Calls `+[_ANEDeviceController controllerWithProgramHandle:]`.
    pub fn with_program_handle(handle: u64) -> Result<Self, AneSysError> {
        let cls = get_class("_ANEDeviceController")?;
        type FactoryFn = unsafe extern "C" fn(*mut c_void, *mut c_void, u64) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("controllerWithProgramHandle:")) };
        let f: FactoryFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, sel, handle) };
        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "controllerWithProgramHandle: returned nil".into(),
            });
        }
        // Retain the autoreleased object.
        crate::objc::objc_retain(obj);
        Ok(Self { raw: obj })
    }

    /// Create a controller for privileged VM access.
    ///
    /// Calls `+[_ANEDeviceController controllerWithPrivilegedVM:]`.
    pub fn with_privileged_vm(privileged: bool) -> Result<Self, AneSysError> {
        let cls = get_class("_ANEDeviceController")?;
        type FactoryFn = unsafe extern "C" fn(*mut c_void, *mut c_void, i8) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("controllerWithPrivilegedVM:")) };
        let f: FactoryFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, sel, privileged as i8) };
        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "controllerWithPrivilegedVM: returned nil".into(),
            });
        }
        crate::objc::objc_retain(obj);
        Ok(Self { raw: obj })
    }

    /// Start the device controller.
    pub fn start(&self) {
        type VoidFn = unsafe extern "C" fn(*mut c_void, *mut c_void);
        let sel = unsafe { sel_registerName(sel!("start")) };
        let f: VoidFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) };
    }

    /// Stop the device controller.
    pub fn stop(&self) {
        type VoidFn = unsafe extern "C" fn(*mut c_void, *mut c_void);
        let sel = unsafe { sel_registerName(sel!("stop")) };
        let f: VoidFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) };
    }

    /// The program handle this controller was created with.
    pub fn program_handle(&self) -> u64 {
        type U64Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64;
        let sel = unsafe { sel_registerName(sel!("programHandle")) };
        let f: U64Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Whether this controller has privileged access.
    pub fn is_privileged(&self) -> bool {
        type BoolFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("isPrivileged")) };
        let f: BoolFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        (unsafe { f(self.raw, sel) }) != 0
    }

    /// Current use count.
    pub fn usecount(&self) -> i64 {
        type I64Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i64;
        let sel = unsafe { sel_registerName(sel!("usecount")) };
        let f: I64Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Set the use count.
    pub fn set_usecount(&self, count: i64) {
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, i64);
        let sel = unsafe { sel_registerName(sel!("setUsecount:")) };
        let f: SetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, count) };
    }

    /// Raw pointer to the underlying `ANEDeviceStruct`.
    pub fn device(&self) -> *mut c_void {
        type PtrFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("device")) };
        let f: PtrFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Set the raw device pointer.
    ///
    /// # Safety
    ///
    /// `device` must be a valid `ANEDeviceStruct` pointer or null.
    pub unsafe fn set_device(&self, device: *mut c_void) {
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void);
        let sel = unsafe { sel_registerName(sel!("setDevice:")) };
        let f: SetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, device) };
    }
}

impl Drop for DeviceController {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            safe_release(self.raw);
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
    fn device_info_has_ane() {
        // May return Ok(false) on non-ANE machines; should not panic.
        let _ = DeviceInfo::has_ane();
    }

    #[test]
    fn device_info_num_anes() {
        let _ = DeviceInfo::num_anes();
    }

    #[test]
    fn device_info_architecture_type() {
        let _ = DeviceInfo::architecture_type();
    }
}
