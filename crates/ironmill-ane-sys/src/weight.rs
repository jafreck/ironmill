//! ANE weight, procedure data, and model instance parameter wrappers.
//!
//! Wraps Apple's private `_ANEWeight`, `_ANEProcedureData`, and
//! `_ANEModelInstanceParameters` classes from the `AppleNeuralEngine`
//! framework.
//!
//! # ⚠️ Private API — see crate-level docs for risks and caveats.

use std::ffi::c_void;

use crate::error::AneSysError;
use crate::objc::{
    CFRelease, create_nsstring, get_class, nsstring_to_string, objc_msgSend, objc_retain,
    safe_release, sel, sel_registerName,
};

// ===========================================================================
// Weight — wraps _ANEWeight
// ===========================================================================

/// Safe wrapper around `_ANEWeight`.
///
/// Represents a single weight file referenced by symbol name and URL.
/// Owns a retained ObjC object; released on drop.
pub struct Weight {
    raw: *mut c_void,
}

// SAFETY: The raw handle is only accessed through &self methods.
unsafe impl Send for Weight {}

impl std::fmt::Debug for Weight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Weight").field("raw", &self.raw).finish()
    }
}

impl Weight {
    /// Raw ObjC object pointer.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    /// Create a weight from a symbol name and URL.
    ///
    /// Calls `+[_ANEWeight weightWithSymbolAndURL:weightURL:]`.
    ///
    /// # Safety
    ///
    /// `weight_url` must be a valid `NSURL` object pointer.
    pub unsafe fn with_symbol_and_url(
        symbol: &str,
        weight_url: *mut c_void,
    ) -> Result<Self, AneSysError> {
        let cls = get_class("_ANEWeight")?;
        let ns_sym = create_nsstring(symbol)?;
        type FactoryFn =
            unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void) -> *mut c_void;
        let s = unsafe { sel_registerName(sel!("weightWithSymbolAndURL:weightURL:")) };
        let f: FactoryFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, s, ns_sym, weight_url) };
        unsafe { CFRelease(ns_sym) };
        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "weightWithSymbolAndURL:weightURL: returned nil".into(),
            });
        }
        objc_retain(obj);
        Ok(Self { raw: obj })
    }

    /// Create a weight from a symbol name, URL, and SHA code.
    ///
    /// Calls `+[_ANEWeight weightWithSymbolAndURLSHA:weightURL:SHACode:]`.
    ///
    /// # Safety
    ///
    /// `weight_url` must be a valid `NSURL` and `sha_code` a valid `NSData`.
    pub unsafe fn with_symbol_url_sha(
        symbol: &str,
        weight_url: *mut c_void,
        sha_code: *mut c_void,
    ) -> Result<Self, AneSysError> {
        let cls = get_class("_ANEWeight")?;
        let ns_sym = create_nsstring(symbol)?;
        type FactoryFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
        ) -> *mut c_void;
        let s = unsafe { sel_registerName(sel!("weightWithSymbolAndURLSHA:weightURL:SHACode:")) };
        let f: FactoryFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, s, ns_sym, weight_url, sha_code) };
        unsafe { CFRelease(ns_sym) };
        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "weightWithSymbolAndURLSHA:weightURL:SHACode: returned nil".into(),
            });
        }
        objc_retain(obj);
        Ok(Self { raw: obj })
    }

    /// The weight symbol name.
    pub fn weight_symbol(&self) -> Option<String> {
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let s = unsafe { sel_registerName(sel!("weightSymbol")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ns = unsafe { f(self.raw, s) };
        unsafe { nsstring_to_string(ns) }
    }

    /// The weight URL (`NSURL`).
    pub fn weight_url(&self) -> *mut c_void {
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let s = unsafe { sel_registerName(sel!("weightURL")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, s) }
    }

    /// The SHA code (`NSData`).
    pub fn sha_code(&self) -> *mut c_void {
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let s = unsafe { sel_registerName(sel!("SHACode")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, s) }
    }

    /// Set the weight URL.
    ///
    /// # Safety
    ///
    /// `url` must be a valid `NSURL` object pointer.
    pub unsafe fn set_weight_url(&self, url: *mut c_void) {
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void);
        let s = unsafe { sel_registerName(sel!("setWeightURL:")) };
        let f: SetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, s, url) };
    }

    /// Update the weight URL.
    ///
    /// # Safety
    ///
    /// `url` must be a valid `NSURL` object pointer.
    pub unsafe fn update_weight_url(&self, url: *mut c_void) {
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void);
        let s = unsafe { sel_registerName(sel!("updateWeightURL:")) };
        let f: SetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, s, url) };
    }

    /// The sandbox extension string.
    pub fn sandbox_extension(&self) -> Option<String> {
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let s = unsafe { sel_registerName(sel!("sandboxExtension")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ns = unsafe { f(self.raw, s) };
        unsafe { nsstring_to_string(ns) }
    }

    /// Set the sandbox extension.
    pub fn set_sandbox_extension(&self, ext: &str) -> Result<(), AneSysError> {
        let ns_ext = create_nsstring(ext)?;
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void);
        let s = unsafe { sel_registerName(sel!("setSandboxExtension:")) };
        let f: SetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, s, ns_ext) };
        unsafe { CFRelease(ns_ext) };
        Ok(())
    }
}

impl Drop for Weight {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            safe_release(self.raw);
            self.raw = std::ptr::null_mut();
        }
    }
}

// ===========================================================================
// ProcedureData — wraps _ANEProcedureData
// ===========================================================================

/// Safe wrapper around `_ANEProcedureData`.
///
/// Groups a procedure symbol with its array of [`Weight`] objects.
/// Owns a retained ObjC object; released on drop.
pub struct ProcedureData {
    raw: *mut c_void,
}

// SAFETY: The raw handle is only accessed through &self methods.
unsafe impl Send for ProcedureData {}

impl std::fmt::Debug for ProcedureData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProcedureData")
            .field("raw", &self.raw)
            .finish()
    }
}

impl ProcedureData {
    /// Raw ObjC object pointer.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    /// Create procedure data from a symbol and weight array.
    ///
    /// Calls `+[_ANEProcedureData procedureDataWithSymbol:weightArray:]`.
    ///
    /// # Safety
    ///
    /// `weight_array` must be a valid `NSArray` of `_ANEWeight` objects.
    pub unsafe fn with_symbol_and_weights(
        symbol: &str,
        weight_array: *mut c_void,
    ) -> Result<Self, AneSysError> {
        let cls = get_class("_ANEProcedureData")?;
        let ns_sym = create_nsstring(symbol)?;
        type FactoryFn =
            unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void) -> *mut c_void;
        let s = unsafe { sel_registerName(sel!("procedureDataWithSymbol:weightArray:")) };
        let f: FactoryFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, s, ns_sym, weight_array) };
        unsafe { CFRelease(ns_sym) };
        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "procedureDataWithSymbol:weightArray: returned nil".into(),
            });
        }
        objc_retain(obj);
        Ok(Self { raw: obj })
    }

    /// The procedure symbol name.
    pub fn procedure_symbol(&self) -> Option<String> {
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let s = unsafe { sel_registerName(sel!("procedureSymbol")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ns = unsafe { f(self.raw, s) };
        unsafe { nsstring_to_string(ns) }
    }

    /// The weight array (`NSArray`).
    pub fn weight_array(&self) -> *mut c_void {
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let s = unsafe { sel_registerName(sel!("weightArray")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, s) }
    }
}

impl Drop for ProcedureData {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            safe_release(self.raw);
            self.raw = std::ptr::null_mut();
        }
    }
}

// ===========================================================================
// ModelInstanceParameters — wraps _ANEModelInstanceParameters
// ===========================================================================

/// Safe wrapper around `_ANEModelInstanceParameters`.
///
/// Bundles an instance name with an array of [`ProcedureData`] objects.
/// Owns a retained ObjC object; released on drop.
pub struct ModelInstanceParameters {
    raw: *mut c_void,
}

// SAFETY: The raw handle is only accessed through &self methods.
unsafe impl Send for ModelInstanceParameters {}

impl std::fmt::Debug for ModelInstanceParameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelInstanceParameters")
            .field("raw", &self.raw)
            .finish()
    }
}

impl ModelInstanceParameters {
    /// Raw ObjC object pointer.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    /// Create model instance parameters from procedure data and procedure array.
    ///
    /// Calls `+[_ANEModelInstanceParameters withProcedureData:procedureArray:]`.
    ///
    /// # Safety
    ///
    /// `procedure_array` must be a valid `NSArray` of `_ANEProcedureData` objects.
    pub unsafe fn with_procedure_data(
        name: &str,
        procedure_array: *mut c_void,
    ) -> Result<Self, AneSysError> {
        let cls = get_class("_ANEModelInstanceParameters")?;
        let ns_name = create_nsstring(name)?;
        type FactoryFn =
            unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void) -> *mut c_void;
        let s = unsafe { sel_registerName(sel!("withProcedureData:procedureArray:")) };
        let f: FactoryFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, s, ns_name, procedure_array) };
        unsafe { CFRelease(ns_name) };
        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "withProcedureData:procedureArray: returned nil".into(),
            });
        }
        objc_retain(obj);
        Ok(Self { raw: obj })
    }

    /// The instance name.
    pub fn instance_name(&self) -> Option<String> {
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let s = unsafe { sel_registerName(sel!("instanceName")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ns = unsafe { f(self.raw, s) };
        unsafe { nsstring_to_string(ns) }
    }

    /// The procedure array (`NSArray`).
    pub fn procedure_array(&self) -> *mut c_void {
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let s = unsafe { sel_registerName(sel!("procedureArray")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, s) }
    }
}

impl Drop for ModelInstanceParameters {
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
    fn weight_class_exists() {
        // Should succeed if the ANE framework is available.
        let _ = get_class("_ANEWeight");
    }

    #[test]
    fn procedure_data_class_exists() {
        let _ = get_class("_ANEProcedureData");
    }

    #[test]
    fn model_instance_parameters_class_exists() {
        let _ = get_class("_ANEModelInstanceParameters");
    }
}
