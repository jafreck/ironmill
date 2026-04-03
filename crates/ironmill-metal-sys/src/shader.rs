//! Safe wrappers for `MTLLibrary` and `MTLFunction`.

use std::ffi::c_void;

use crate::error::MetalSysError;
use crate::objc::{self, sel};

// ---------------------------------------------------------------------------
// ShaderLibrary
// ---------------------------------------------------------------------------

/// Safe wrapper around an `MTLLibrary` (id<MTLLibrary>).
pub struct ShaderLibrary {
    raw: *mut c_void,
}

// SAFETY: MTLLibrary is thread-safe and immutable once created.
unsafe impl Send for ShaderLibrary {}
unsafe impl Sync for ShaderLibrary {}

impl ShaderLibrary {
    /// Create from a raw retained `id<MTLLibrary>`.
    pub(crate) fn from_raw(raw: *mut c_void) -> Self {
        Self { raw }
    }

    /// Look up a function by name in this library.
    pub fn get_function(&self, name: &str) -> Result<ShaderFunction, MetalSysError> {
        let ns_name = objc::create_nsstring(name)?;
        // SAFETY: `self.raw` is a valid, retained id<MTLLibrary>, `ns_name`
        // is a valid retained NSString. "newFunctionWithName:" returns a +1
        // retained id<MTLFunction> or nil. ns_name is released after the call.
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { objc::sel_registerName(sel!("newFunctionWithName:")) };
        let f: GetFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        let raw = unsafe { f(self.raw, sel, ns_name) };
        // Release the NSString.
        // SAFETY: ns_name is a retained NSString from create_nsstring.
        unsafe { objc::CFRelease(ns_name) };

        if raw.is_null() {
            return Err(MetalSysError::ShaderCompilation(format!(
                "function '{name}' not found in library"
            )));
        }
        Ok(ShaderFunction { raw })
    }

    /// Look up a function by name with function constant values.
    ///
    /// Function constants allow specializing shaders at pipeline creation time
    /// (e.g., array sizes, loop bounds) without recompiling source.
    pub fn get_function_with_constants(
        &self,
        name: &str,
        constants: &FunctionConstantValues,
    ) -> Result<ShaderFunction, MetalSysError> {
        let ns_name = objc::create_nsstring(name)?;
        let mut error: *mut c_void = std::ptr::null_mut();
        // SAFETY: `self.raw` is a valid id<MTLLibrary>. `ns_name` is a valid
        // NSString. `constants.raw` is a valid MTLFunctionConstantValues.
        // "newFunctionWithName:constantValues:error:" returns +1 retained
        // id<MTLFunction> or nil with error.
        type GetFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut *mut c_void,
        ) -> *mut c_void;
        let sel =
            unsafe { objc::sel_registerName(sel!("newFunctionWithName:constantValues:error:")) };
        let f: GetFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        let raw = unsafe { f(self.raw, sel, ns_name, constants.raw, &mut error) };
        unsafe { objc::CFRelease(ns_name) };

        if raw.is_null() {
            let desc = if !error.is_null() {
                let d = objc::extract_nserror_description(error);
                unsafe { objc::CFRelease(error) };
                d
            } else {
                "unknown error".into()
            };
            return Err(MetalSysError::ShaderCompilation(format!(
                "function '{name}' with constants: {desc}"
            )));
        }
        Ok(ShaderFunction { raw })
    }

    /// Returns the raw `id<MTLLibrary>` pointer.
    pub fn as_raw_ptr(&self) -> *mut c_void {
        self.raw
    }
}

impl Drop for ShaderLibrary {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: self.raw is a retained ObjC object; null-checked above.
            unsafe { objc::CFRelease(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// ShaderFunction
// ---------------------------------------------------------------------------

/// Safe wrapper around an `MTLFunction` (id<MTLFunction>).
pub struct ShaderFunction {
    raw: *mut c_void,
}

// SAFETY: MTLFunction is thread-safe and immutable once created.
unsafe impl Send for ShaderFunction {}
unsafe impl Sync for ShaderFunction {}

impl ShaderFunction {
    /// Returns the raw `id<MTLFunction>` pointer.
    pub fn as_raw_ptr(&self) -> *mut c_void {
        self.raw
    }
}

impl Drop for ShaderFunction {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: self.raw is a retained ObjC object; null-checked above.
            unsafe { objc::CFRelease(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// FunctionConstantValues
// ---------------------------------------------------------------------------

/// Safe wrapper around `MTLFunctionConstantValues`.
///
/// Allows specializing shader functions at pipeline creation time with
/// compile-time constant values (e.g., array sizes, feature flags).
pub struct FunctionConstantValues {
    raw: *mut c_void,
}

// SAFETY: MTLFunctionConstantValues is thread-safe once populated.
unsafe impl Send for FunctionConstantValues {}
unsafe impl Sync for FunctionConstantValues {}

impl FunctionConstantValues {
    /// Create a new empty set of function constant values.
    pub fn new() -> Result<Self, MetalSysError> {
        // SAFETY: Standard ObjC alloc/init pattern on MTLFunctionConstantValues.
        let cls = unsafe { objc::objc_getClass(sel!("MTLFunctionConstantValues")) };
        if cls.is_null() {
            return Err(MetalSysError::FrameworkNotFound);
        }

        type AllocFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let alloc_sel = unsafe { objc::sel_registerName(sel!("alloc")) };
        let alloc_fn: AllocFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        let raw = unsafe { alloc_fn(cls, alloc_sel) };
        if raw.is_null() {
            return Err(MetalSysError::InvalidArgument(
                "MTLFunctionConstantValues alloc failed".into(),
            ));
        }

        type InitFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let init_sel = unsafe { objc::sel_registerName(sel!("init")) };
        let init_fn: InitFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        let obj = unsafe { init_fn(raw, init_sel) };
        if obj.is_null() {
            return Err(MetalSysError::InvalidArgument(
                "MTLFunctionConstantValues init failed".into(),
            ));
        }
        Ok(Self { raw: obj })
    }

    /// Set a `uint` (32-bit unsigned) constant at the given index.
    ///
    /// The index corresponds to `[[function_constant(index)]]` in the shader.
    pub fn set_u32(&self, value: u32, index: usize) {
        // MTLDataTypeUInt = 13
        const MTL_DATA_TYPE_UINT: usize = 13;
        // SAFETY: `self.raw` is a valid MTLFunctionConstantValues.
        // "setConstantValue:type:atIndex:" copies the value.
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *const u32, usize, usize);
        let sel = unsafe { objc::sel_registerName(sel!("setConstantValue:type:atIndex:")) };
        let f: SetFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, &value, MTL_DATA_TYPE_UINT, index) };
    }
}

impl Drop for FunctionConstantValues {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: self.raw is a retained ObjC object; null-checked above.
            unsafe { objc::CFRelease(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}
