//! Safe wrapper for `_ANEVirtualClient` — the IOUserClient-level ANE handle.
//!
//! `_ANEVirtualClient` is the largest class in the ANE private framework.
//! It provides direct access to compile, load, evaluate, and manage models
//! through the kernel IOUserClient interface.
//!
//! # ⚠️ Private API — see crate-level docs for risks and caveats.

use std::ffi::c_void;

use crate::error::AneSysError;
use crate::objc::{
    extract_nserror_description, get_class, nsstring_to_string, objc_msgSend, objc_retain,
    safe_release, sel, sel_registerName,
};

// ---------------------------------------------------------------------------
// Opaque C structs
// ---------------------------------------------------------------------------

/// Opaque representation of the `VirtANEModel` C struct.
///
/// The real layout is `{VirtANEModel=IqIIIIQQQQ[32I][32Q]...}` — too complex
/// to decode field-by-field. We size it generously as an opaque blob.
#[repr(C, align(8))]
pub struct VirtANEModel {
    pub data: [u8; 2048],
}

impl Default for VirtANEModel {
    fn default() -> Self {
        Self { data: [0u8; 2048] }
    }
}

impl std::fmt::Debug for VirtANEModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VirtANEModel")
            .field("data_len", &self.data.len())
            .finish()
    }
}

/// Device information returned by `-getDeviceInfo`.
///
/// Layout: `{DeviceExtendedInfo={DeviceInfo=IqqB}BII[32c][8c]}`
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DeviceExtendedInfo {
    /// `{DeviceInfo=IqqB}` — nested struct.
    pub device_info: DeviceInfoInner,
    pub flag: u8,
    pub field_a: u32,
    pub field_b: u32,
    pub name: [i8; 32],
    pub version: [i8; 8],
}

/// Inner `DeviceInfo` from the type encoding `{DeviceInfo=IqqB}`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DeviceInfoInner {
    pub id: u32,
    pub field_a: i64,
    pub field_b: i64,
    pub present: u8,
}

/// Build version information returned by `-exchangeBuildVersionInfo`.
///
/// Layout: `{BuildVersionInfo=IIQ[32C]CQQ[15Q]}`
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BuildVersionInfo {
    pub major: u32,
    pub minor: u32,
    pub patch: u64,
    pub version_string: [u8; 32],
    pub flag: u8,
    pub field_a: u64,
    pub field_b: u64,
    pub reserved: [u64; 15],
}

// ---------------------------------------------------------------------------
// VirtualClient
// ---------------------------------------------------------------------------

/// Safe wrapper around `_ANEVirtualClient`.
///
/// Owns a retained ObjC object; released on drop.
pub struct VirtualClient {
    raw: *mut c_void,
}

// SAFETY: The raw handle is only accessed through &self methods.
unsafe impl Send for VirtualClient {}

impl std::fmt::Debug for VirtualClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VirtualClient")
            .field("raw", &self.raw)
            .finish()
    }
}

impl VirtualClient {
    /// Raw ObjC object pointer.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    // =======================================================================
    // Construction
    // =======================================================================

    /// Create a new `_ANEVirtualClient` via `+[_ANEVirtualClient new]`.
    pub fn new() -> Result<Self, AneSysError> {
        let cls = get_class("_ANEVirtualClient")?;
        type NewFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("new")) };
        let f: NewFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, sel) };
        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "_ANEVirtualClient +new returned nil".into(),
            });
        }
        Ok(Self { raw: obj })
    }

    /// Get the shared singleton via `+[_ANEVirtualClient sharedConnection]`.
    pub fn shared_connection() -> Result<Self, AneSysError> {
        let cls = get_class("_ANEVirtualClient")?;
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("sharedConnection")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, sel) };
        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "_ANEVirtualClient +sharedConnection returned nil".into(),
            });
        }
        objc_retain(obj);
        Ok(Self { raw: obj })
    }

    /// Wrap an existing raw `_ANEVirtualClient` pointer.
    ///
    /// # Safety
    ///
    /// `ptr` must be a valid, retained `_ANEVirtualClient` object.
    pub unsafe fn from_raw(ptr: *mut c_void) -> Self {
        Self { raw: ptr }
    }

    // =======================================================================
    // Lifecycle / connectivity
    // =======================================================================

    /// IOKit connect handle.
    ///
    /// Wraps `-[_ANEVirtualClient connect]` → `u32`.
    pub fn connect(&self) -> u32 {
        type U32Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32;
        let sel = unsafe { sel_registerName(sel!("connect")) };
        let f: U32Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Whether ANE hardware is present.
    pub fn has_ane(&self) -> bool {
        type BoolFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("hasANE")) };
        let f: BoolFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) != 0 }
    }

    /// Number of ANE units.
    pub fn num_anes(&self) -> u32 {
        type U32Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32;
        let sel = unsafe { sel_registerName(sel!("numANEs")) };
        let f: U32Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Number of ANE cores.
    pub fn num_ane_cores(&self) -> u32 {
        type U32Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32;
        let sel = unsafe { sel_registerName(sel!("numANECores")) };
        let f: U32Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Whether this is an internal build.
    pub fn is_internal_build(&self) -> bool {
        type BoolFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("isInternalBuild")) };
        let f: BoolFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) != 0 }
    }

    // =======================================================================
    // Model operations
    // =======================================================================

    /// Compile a model.
    ///
    /// Wraps `-[_ANEVirtualClient compileModel:options:qos:error:]`.
    ///
    /// # Safety
    ///
    /// `model` and `options` must be valid ObjC object pointers (or nil for options).
    pub unsafe fn compile_model(
        &self,
        model: *mut c_void,
        options: *mut c_void,
        qos: u32,
    ) -> Result<(), AneSysError> {
        let mut error: *mut c_void = std::ptr::null_mut();
        type Fn4 = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            u32,
            *mut *mut c_void,
        ) -> i8;
        let sel = unsafe { sel_registerName(sel!("compileModel:options:qos:error:")) };
        let f: Fn4 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ok = unsafe { f(self.raw, sel, model, options, qos, &mut error) };
        if ok == 0 {
            let msg = if !error.is_null() {
                extract_nserror_description(error)
            } else {
                "compileModel:options:qos:error: returned NO".into()
            };
            return Err(AneSysError::CompilationFailed(msg));
        }
        Ok(())
    }

    /// Load a model.
    ///
    /// Wraps `-[_ANEVirtualClient loadModel:options:qos:error:]`.
    ///
    /// # Safety
    ///
    /// `model` and `options` must be valid ObjC object pointers.
    pub unsafe fn load_model(
        &self,
        model: *mut c_void,
        options: *mut c_void,
        qos: u32,
    ) -> Result<(), AneSysError> {
        let mut error: *mut c_void = std::ptr::null_mut();
        type Fn4 = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            u32,
            *mut *mut c_void,
        ) -> i8;
        let sel = unsafe { sel_registerName(sel!("loadModel:options:qos:error:")) };
        let f: Fn4 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ok = unsafe { f(self.raw, sel, model, options, qos, &mut error) };
        if ok == 0 {
            let msg = if !error.is_null() {
                extract_nserror_description(error)
            } else {
                "loadModel:options:qos:error: returned NO".into()
            };
            return Err(AneSysError::LoadFailed(msg));
        }
        Ok(())
    }

    /// Load a new model instance with instance parameters.
    ///
    /// Wraps `-[_ANEVirtualClient loadModelNewInstance:options:modelInstParams:qos:error:]`.
    ///
    /// # Safety
    ///
    /// `model`, `options`, and `model_inst_params` must be valid ObjC object pointers.
    pub unsafe fn load_model_new_instance(
        &self,
        model: *mut c_void,
        options: *mut c_void,
        model_inst_params: *mut c_void,
        qos: u32,
    ) -> Result<(), AneSysError> {
        let mut error: *mut c_void = std::ptr::null_mut();
        type Fn5 = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            u32,
            *mut *mut c_void,
        ) -> i8;
        let sel = unsafe {
            sel_registerName(sel!(
                "loadModelNewInstance:options:modelInstParams:qos:error:"
            ))
        };
        let f: Fn5 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ok = unsafe {
            f(
                self.raw,
                sel,
                model,
                options,
                model_inst_params,
                qos,
                &mut error,
            )
        };
        if ok == 0 {
            let msg = if !error.is_null() {
                extract_nserror_description(error)
            } else {
                "loadModelNewInstance:options:modelInstParams:qos:error: returned NO".into()
            };
            return Err(AneSysError::LoadFailed(msg));
        }
        Ok(())
    }

    /// Unload a model.
    ///
    /// Wraps `-[_ANEVirtualClient unloadModel:options:qos:error:]`.
    ///
    /// # Safety
    ///
    /// `model` and `options` must be valid ObjC object pointers.
    pub unsafe fn unload_model(
        &self,
        model: *mut c_void,
        options: *mut c_void,
        qos: u32,
    ) -> Result<(), AneSysError> {
        let mut error: *mut c_void = std::ptr::null_mut();
        type Fn4 = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            u32,
            *mut *mut c_void,
        ) -> i8;
        let sel = unsafe { sel_registerName(sel!("unloadModel:options:qos:error:")) };
        let f: Fn4 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ok = unsafe { f(self.raw, sel, model, options, qos, &mut error) };
        if ok == 0 {
            let msg = if !error.is_null() {
                extract_nserror_description(error)
            } else {
                "unloadModel:options:qos:error: returned NO".into()
            };
            return Err(AneSysError::UnloadFailed(msg));
        }
        Ok(())
    }

    /// Evaluate with a model.
    ///
    /// Wraps `-[_ANEVirtualClient evaluateWithModel:options:request:qos:error:]`.
    ///
    /// # Safety
    ///
    /// `model`, `options`, and `request` must be valid ObjC object pointers.
    pub unsafe fn evaluate_with_model(
        &self,
        model: *mut c_void,
        options: *mut c_void,
        request: *mut c_void,
        qos: u32,
    ) -> Result<(), AneSysError> {
        let mut error: *mut c_void = std::ptr::null_mut();
        type Fn5 = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            u32,
            *mut *mut c_void,
        ) -> i8;
        let sel = unsafe { sel_registerName(sel!("evaluateWithModel:options:request:qos:error:")) };
        let f: Fn5 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ok = unsafe { f(self.raw, sel, model, options, request, qos, &mut error) };
        if ok == 0 {
            let msg = if !error.is_null() {
                extract_nserror_description(error)
            } else {
                "evaluateWithModel:options:request:qos:error: returned NO".into()
            };
            return Err(AneSysError::EvalFailed {
                status: 1,
                context: msg,
            });
        }
        Ok(())
    }

    /// Async evaluate with completion event.
    ///
    /// Wraps `-[_ANEVirtualClient doEvaluateWithModel:options:request:qos:completionEvent:error:]`.
    ///
    /// # Safety
    ///
    /// All object parameters must be valid ObjC pointers.
    pub unsafe fn do_evaluate_with_model(
        &self,
        model: *mut c_void,
        options: *mut c_void,
        request: *mut c_void,
        qos: u32,
        completion_event: *mut c_void,
    ) -> Result<(), AneSysError> {
        let mut error: *mut c_void = std::ptr::null_mut();
        type Fn6 = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            u32,
            *mut c_void,
            *mut *mut c_void,
        ) -> i8;
        let sel = unsafe {
            sel_registerName(sel!(
                "doEvaluateWithModel:options:request:qos:completionEvent:error:"
            ))
        };
        let f: Fn6 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ok = unsafe {
            f(
                self.raw,
                sel,
                model,
                options,
                request,
                qos,
                completion_event,
                &mut error,
            )
        };
        if ok == 0 {
            let msg = if !error.is_null() {
                extract_nserror_description(error)
            } else {
                "doEvaluateWithModel:...completionEvent:error: returned NO".into()
            };
            return Err(AneSysError::EvalFailed {
                status: 1,
                context: msg,
            });
        }
        Ok(())
    }

    /// Check if a compiled model exists for the given model object.
    ///
    /// Wraps `-[_ANEVirtualClient compiledModelExistsFor:]`.
    ///
    /// # Safety
    ///
    /// `model` must be a valid ObjC object pointer.
    pub unsafe fn compiled_model_exists_for(&self, model: *mut c_void) -> bool {
        type BoolFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("compiledModelExistsFor:")) };
        let f: BoolFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, model) != 0 }
    }

    /// Check if a compiled model exists matching a hash.
    ///
    /// Wraps `-[_ANEVirtualClient compiledModelExistsMatchingHash:]`.
    ///
    /// # Safety
    ///
    /// `hash` must be a valid ObjC object pointer (NSString or similar).
    pub unsafe fn compiled_model_exists_matching_hash(&self, hash: *mut c_void) -> bool {
        type BoolFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("compiledModelExistsMatchingHash:")) };
        let f: BoolFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, hash) != 0 }
    }

    // =======================================================================
    // IOSurface operations
    // =======================================================================

    /// Map IOSurfaces for a model and request.
    ///
    /// Wraps `-[_ANEVirtualClient mapIOSurfacesWithModel:request:cacheInference:error:]`.
    ///
    /// # Safety
    ///
    /// `model` and `request` must be valid ObjC object pointers.
    pub unsafe fn map_iosurfaces_with_model(
        &self,
        model: *mut c_void,
        request: *mut c_void,
        cache_inference: bool,
    ) -> Result<(), AneSysError> {
        let mut error: *mut c_void = std::ptr::null_mut();
        type MapFn = unsafe extern "C" fn(
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
        let f: MapFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ok = unsafe {
            f(
                self.raw,
                sel,
                model,
                request,
                cache_inference as i8,
                &mut error,
            )
        };
        if ok == 0 {
            let msg = if !error.is_null() {
                extract_nserror_description(error)
            } else {
                "mapIOSurfacesWithModel:request:cacheInference:error: returned NO".into()
            };
            return Err(AneSysError::IOSurfaceMappingFailed(msg));
        }
        Ok(())
    }

    /// Internal map variant (doMapIOSurfacesWithModel).
    ///
    /// Wraps `-[_ANEVirtualClient doMapIOSurfacesWithModel:request:cacheInference:error:]`.
    ///
    /// # Safety
    ///
    /// `model` and `request` must be valid ObjC object pointers.
    pub unsafe fn do_map_iosurfaces_with_model(
        &self,
        model: *mut c_void,
        request: *mut c_void,
        cache_inference: bool,
    ) -> Result<(), AneSysError> {
        let mut error: *mut c_void = std::ptr::null_mut();
        type MapFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            i8,
            *mut *mut c_void,
        ) -> i8;
        let sel = unsafe {
            sel_registerName(sel!(
                "doMapIOSurfacesWithModel:request:cacheInference:error:"
            ))
        };
        let f: MapFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ok = unsafe {
            f(
                self.raw,
                sel,
                model,
                request,
                cache_inference as i8,
                &mut error,
            )
        };
        if ok == 0 {
            let msg = if !error.is_null() {
                extract_nserror_description(error)
            } else {
                "doMapIOSurfacesWithModel:...error: returned NO".into()
            };
            return Err(AneSysError::IOSurfaceMappingFailed(msg));
        }
        Ok(())
    }

    /// Copy data into an IOSurface (NSData variant).
    ///
    /// Wraps `-[_ANEVirtualClient copyToIOSurface:length:ioSID:]`.
    /// Returns the IOSurface pointer (caller must manage lifetime).
    ///
    /// # Safety
    ///
    /// `data` must be a valid ObjC `NSData` pointer. `io_sid` is written on success.
    pub unsafe fn copy_to_iosurface_with_length(
        &self,
        data: *mut c_void,
        length: u64,
        io_sid: &mut u32,
    ) -> *mut c_void {
        type CopyFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            u64,
            *mut u32,
        ) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("copyToIOSurface:length:ioSID:")) };
        let f: CopyFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, data, length, io_sid) }
    }

    /// Copy data into an IOSurface (C-string variant).
    ///
    /// Wraps `-[_ANEVirtualClient copyToIOSurface:size:ioSID:]`.
    /// Returns the IOSurface pointer.
    ///
    /// # Safety
    ///
    /// `data` must be a valid C string pointer. `io_sid` is written on success.
    pub unsafe fn copy_to_iosurface_with_size(
        &self,
        data: *const i8,
        size: u64,
        io_sid: &mut u32,
    ) -> *mut c_void {
        type CopyFn =
            unsafe extern "C" fn(*mut c_void, *mut c_void, *const i8, u64, *mut u32) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("copyToIOSurface:size:ioSID:")) };
        let f: CopyFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, data, size, io_sid) }
    }

    // =======================================================================
    // Diagnostics
    // =======================================================================

    /// Get extended device info.
    ///
    /// Wraps `-[_ANEVirtualClient getDeviceInfo]`.
    ///
    /// On arm64, structs up to a certain size are returned in registers via
    /// the standard `objc_msgSend` ABI (no `objc_msgSend_stret` on arm64).
    pub fn get_device_info(&self) -> DeviceExtendedInfo {
        type InfoFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> DeviceExtendedInfo;
        let sel = unsafe { sel_registerName(sel!("getDeviceInfo")) };
        let f: InfoFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Exchange build version info with the ANE daemon.
    ///
    /// Wraps `-[_ANEVirtualClient exchangeBuildVersionInfo]`.
    pub fn exchange_build_version_info(&self) -> BuildVersionInfo {
        type BviFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> BuildVersionInfo;
        let sel = unsafe { sel_registerName(sel!("exchangeBuildVersionInfo")) };
        let f: BviFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// ANE architecture type string (e.g. `"ane"`, `"ane2"`).
    pub fn ane_architecture_type_str(&self) -> Option<String> {
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("aneArchitectureTypeStr")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ns = unsafe { f(self.raw, sel) };
        unsafe { nsstring_to_string(ns) }
    }

    /// ANE board type identifier.
    pub fn ane_boardtype(&self) -> i64 {
        type I64Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i64;
        let sel = unsafe { sel_registerName(sel!("aneBoardtype")) };
        let f: I64Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// ANE sub-type and variant combined string.
    pub fn ane_sub_type_and_variant(&self) -> Option<String> {
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("aneSubTypeAndVariant")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ns = unsafe { f(self.raw, sel) };
        unsafe { nsstring_to_string(ns) }
    }

    /// Host build version string.
    pub fn host_build_version_str(&self) -> Option<String> {
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("hostBuildVersionStr")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ns = unsafe { f(self.raw, sel) };
        unsafe { nsstring_to_string(ns) }
    }

    /// Negotiated capability mask.
    pub fn negotiated_capability_mask(&self) -> u64 {
        type U64Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64;
        let sel = unsafe { sel_registerName(sel!("negotiatedCapabilityMask")) };
        let f: U64Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Negotiated data interface version.
    pub fn negotiated_data_interface_version(&self) -> u32 {
        type U32Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32;
        let sel = unsafe { sel_registerName(sel!("negotiatedDataInterfaceVersion")) };
        let f: U32Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Output dictionary IOSurface size.
    pub fn output_dict_iosurface_size(&self) -> u64 {
        type U64Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64;
        let sel = unsafe { sel_registerName(sel!("outputDictIOSurfaceSize")) };
        let f: U64Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    // =======================================================================
    // Validation
    // =======================================================================

    /// Create a network validation dictionary.
    ///
    /// Wraps `-[_ANEVirtualClient validateNetworkCreate:uuid:function:directoryPath:scratchPadPath:milTextData:]`.
    /// Returns a raw `CFDictionaryRef` (caller owns).
    ///
    /// # Safety
    ///
    /// All ObjC parameters must be valid object pointers.
    pub unsafe fn validate_network_create(
        &self,
        network_id: u64,
        uuid: *mut c_void,
        function: *mut c_void,
        directory_path: *mut c_void,
        scratch_pad_path: *mut c_void,
        mil_text_data: *mut c_void,
    ) -> *mut c_void {
        type ValFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            u64,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
        ) -> *mut c_void;
        let sel = unsafe {
            sel_registerName(sel!(
                "validateNetworkCreate:uuid:function:directoryPath:scratchPadPath:milTextData:"
            ))
        };
        let f: ValFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe {
            f(
                self.raw,
                sel,
                network_id,
                uuid,
                function,
                directory_path,
                scratch_pad_path,
                mil_text_data,
            )
        }
    }

    /// Get the validate network version.
    ///
    /// Wraps `-[_ANEVirtualClient getValidateNetworkVersion]` → `u64`.
    pub fn get_validate_network_version(&self) -> u64 {
        type U64Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64;
        let sel = unsafe { sel_registerName(sel!("getValidateNetworkVersion")) };
        let f: U64Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Validate environment for pre-compiled binary support.
    pub fn validate_environment_for_precompiled_binary_support(&self) -> bool {
        type BoolFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i8;
        let sel =
            unsafe { sel_registerName(sel!("validateEnvironmentForPrecompiledBinarySupport")) };
        let f: BoolFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) != 0 }
    }

    // =======================================================================
    // Real-time task management
    // =======================================================================

    /// Begin a real-time task.
    pub fn begin_real_time_task(&self) -> bool {
        type BoolFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("beginRealTimeTask")) };
        let f: BoolFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) != 0 }
    }

    /// End a real-time task.
    pub fn end_real_time_task(&self) -> bool {
        type BoolFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("endRealTimeTask")) };
        let f: BoolFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) != 0 }
    }

    // =======================================================================
    // Session hints
    // =======================================================================

    /// Set a session hint for a model.
    ///
    /// Wraps `-[_ANEVirtualClient sessionHintWithModel:hint:options:report:error:]`.
    ///
    /// # Safety
    ///
    /// All ObjC parameters must be valid object pointers.
    pub unsafe fn session_hint_with_model(
        &self,
        model: *mut c_void,
        hint: *mut c_void,
        options: *mut c_void,
        report: *mut c_void,
    ) -> Result<(), AneSysError> {
        let mut error: *mut c_void = std::ptr::null_mut();
        type HintFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut *mut c_void,
        ) -> i8;
        let sel =
            unsafe { sel_registerName(sel!("sessionHintWithModel:hint:options:report:error:")) };
        let f: HintFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ok = unsafe { f(self.raw, sel, model, hint, options, report, &mut error) };
        if ok == 0 {
            let msg = if !error.is_null() {
                extract_nserror_description(error)
            } else {
                "sessionHintWithModel:hint:options:report:error: returned NO".into()
            };
            return Err(AneSysError::SessionHintFailed(msg));
        }
        Ok(())
    }

    // =======================================================================
    // Low-level IOUserClient calls
    // =======================================================================

    /// Direct IOUserClient call with `VirtANEModel` structs.
    ///
    /// Wraps `-[_ANEVirtualClient callIOUserClient:inParams:outParams:]`.
    ///
    /// # Safety
    ///
    /// `in_params` and `out_params` must point to valid `VirtANEModel` structs.
    pub unsafe fn call_io_user_client(
        &self,
        selector: u32,
        in_params: *mut VirtANEModel,
        out_params: *mut VirtANEModel,
    ) -> bool {
        type CallFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            u32,
            *mut VirtANEModel,
            *mut VirtANEModel,
        ) -> i8;
        let sel = unsafe { sel_registerName(sel!("callIOUserClient:inParams:outParams:")) };
        let f: CallFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, selector, in_params, out_params) != 0 }
    }

    /// IOUserClient call with a CFDictionary.
    ///
    /// Wraps `-[_ANEVirtualClient callIOUserClientWithDictionary:inDictionary:error:]`.
    /// Returns a `CFDictionaryRef` (caller owns).
    ///
    /// # Safety
    ///
    /// `in_dict` must be a valid `CFDictionaryRef` or null.
    pub unsafe fn call_io_user_client_with_dictionary(
        &self,
        selector: u32,
        in_dict: *mut c_void,
    ) -> Result<*mut c_void, AneSysError> {
        let mut error: *mut c_void = std::ptr::null_mut();
        type DictFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            u32,
            *mut c_void,
            *mut *mut c_void,
        ) -> *mut c_void;
        let sel =
            unsafe { sel_registerName(sel!("callIOUserClientWithDictionary:inDictionary:error:")) };
        let f: DictFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let result = unsafe { f(self.raw, sel, selector, in_dict, &mut error) };
        if result.is_null() && !error.is_null() {
            let msg = extract_nserror_description(error);
            return Err(AneSysError::EvalFailed {
                status: selector,
                context: msg,
            });
        }
        Ok(result)
    }

    // =======================================================================
    // Misc
    // =======================================================================

    /// Echo test — sends an object and returns success.
    ///
    /// # Safety
    ///
    /// `obj` must be a valid ObjC object pointer.
    pub unsafe fn echo(&self, obj: *mut c_void) -> bool {
        type BoolFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("echo:")) };
        let f: BoolFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, obj) != 0 }
    }

    /// Purge cached compiled model by model object.
    ///
    /// # Safety
    ///
    /// `model` must be a valid ObjC object pointer.
    pub unsafe fn purge_compiled_model(&self, model: *mut c_void) {
        type VoidFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void);
        let sel = unsafe { sel_registerName(sel!("purgeCompiledModel:")) };
        let f: VoidFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, model) };
    }

    /// Purge cached compiled model by hash string.
    ///
    /// # Safety
    ///
    /// `hash` must be a valid ObjC object pointer (NSString).
    pub unsafe fn purge_compiled_model_matching_hash(&self, hash: *mut c_void) {
        type VoidFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void);
        let sel = unsafe { sel_registerName(sel!("purgeCompiledModelMatchingHash:")) };
        let f: VoidFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, hash) };
    }

    /// Send guest build version info to the ANE daemon.
    pub fn send_guest_build_version(&self) {
        type VoidFn = unsafe extern "C" fn(*mut c_void, *mut c_void);
        let sel = unsafe { sel_registerName(sel!("sendGuestBuildVersion")) };
        let f: VoidFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) };
    }

    /// The dispatch queue used by this client.
    pub fn queue(&self) -> *mut c_void {
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("queue")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    // =======================================================================
    // Class-method helpers (convenience wrappers)
    // =======================================================================

    /// Create an IOSurface of the given size.
    ///
    /// Wraps `+[_ANEVirtualClient createIOSurface:ioSID:]`.
    /// Returns the IOSurface pointer. `io_sid` is written on success.
    pub fn create_iosurface(size: u64, io_sid: &mut u32) -> Result<*mut c_void, AneSysError> {
        let cls = get_class("_ANEVirtualClient")?;
        type CreateFn =
            unsafe extern "C" fn(*mut c_void, *mut c_void, u64, *mut u32) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("createIOSurface:ioSID:")) };
        let f: CreateFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let surface = unsafe { f(cls, sel, size, io_sid) };
        if surface.is_null() {
            return Err(AneSysError::IOSurfaceMappingFailed(
                "createIOSurface:ioSID: returned nil".into(),
            ));
        }
        Ok(surface)
    }

    /// Get the code signing identity.
    ///
    /// Wraps `+[_ANEVirtualClient getCodeSigningIdentity]`.
    pub fn get_code_signing_identity() -> Result<Option<String>, AneSysError> {
        let cls = get_class("_ANEVirtualClient")?;
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("getCodeSigningIdentity")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ns = unsafe { f(cls, sel) };
        Ok(unsafe { nsstring_to_string(ns) })
    }
}

impl Drop for VirtualClient {
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
    fn virt_ane_model_default_zeroed() {
        let m = VirtANEModel::default();
        assert!(m.data.iter().all(|&b| b == 0));
    }

    #[test]
    fn virtual_client_new_smoke() {
        // May fail on non-ANE machines; should not panic.
        let _ = VirtualClient::new();
    }

    #[test]
    fn virtual_client_shared_connection_smoke() {
        let _ = VirtualClient::shared_connection();
    }
}
