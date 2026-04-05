//! Comprehensive wrappers for `_ANEInMemoryModel` and
//! `_ANEInMemoryModelDescriptor`.
//!
//! All compile/load/eval logic lives here.

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::error::AneSysError;
use crate::objc::{
    CFRelease, ane_framework, create_nsdata, create_nsnumber, create_nsstring,
    extract_nserror_description, get_class, ns_array_add, ns_dict_set, ns_empty_dict,
    ns_empty_dict_unchecked, ns_mutable_array, ns_mutable_dict, ns_number_autoreleased,
    nsstring_to_string, objc_autoreleasePoolPop, objc_autoreleasePoolPush, objc_msgSend,
    objc_retain, responds_to_selector, sel, sel_registerName,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of ANE compilations allowed per process.
///
/// Apple's ANE stack leaks memory on each invocation (~0.5-2 MB).  After
/// roughly 119 compilations the ANE daemon (`aned`) starts rejecting requests.
const ANE_COMPILE_LIMIT: usize = 119;

/// Global compile count tracker (constraint: ~119 limit per process).
static COMPILE_COUNT: AtomicUsize = AtomicUsize::new(0);

/// QoS value used for compile/load/eval (matches Orion's constant of 21).
pub const ANE_QOS: u32 = 21;

// =========================================================================
// InMemoryModelDescriptor
// =========================================================================

/// Safe wrapper around `_ANEInMemoryModelDescriptor`.
///
/// Describes the network text (MIL or legacy), weights, and compiler options
/// used to create an [`InMemoryModel`].
pub struct InMemoryModelDescriptor {
    raw: *mut c_void,
}

// SAFETY: The raw handle is only accessed through &self methods.
unsafe impl Send for InMemoryModelDescriptor {}

impl InMemoryModelDescriptor {
    /// Create a descriptor from MIL text, weights, and optional compiler
    /// options plist data.
    ///
    /// Wraps `+[_ANEInMemoryModelDescriptor modelWithMILText:weights:optionsPlist:]`.
    ///
    /// Each entry in `weights` is `(path_key, raw_data)` where the key
    /// matches the `@model_path/...` reference in the MIL text.  The data
    /// is automatically wrapped in blobfile format.
    pub fn from_mil_text(
        mil_text: &str,
        weights: &[(&str, &[u8])],
        options: Option<&[u8]>,
    ) -> Result<Self, AneSysError> {
        ane_framework()?;
        let desc_cls = get_class("_ANEInMemoryModelDescriptor")?;

        let mil_data = create_nsdata(mil_text.as_bytes())?;
        let weight_dict = if weights.is_empty() {
            ns_empty_dict()?
        } else {
            build_multi_weight_dict(weights)?
        };
        let options_ptr = match options {
            Some(data) => create_nsdata(data)?,
            None => std::ptr::null_mut(),
        };

        let desc_sel = unsafe { sel_registerName(sel!("modelWithMILText:weights:optionsPlist:")) };
        if !responds_to_selector(desc_cls, desc_sel) {
            unsafe {
                CFRelease(mil_data);
                CFRelease(weight_dict);
                if !options_ptr.is_null() {
                    CFRelease(options_ptr);
                }
            }
            return Err(AneSysError::CompilationFailed(
                "_ANEInMemoryModelDescriptor does not respond to \
                 modelWithMILText:weights:optionsPlist:"
                    .into(),
            ));
        }

        type DescFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
        ) -> *mut c_void;
        let desc_fn: DescFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let descriptor = unsafe { desc_fn(desc_cls, desc_sel, mil_data, weight_dict, options_ptr) };

        // Clean up input ObjC objects regardless of outcome.
        unsafe {
            CFRelease(mil_data);
            CFRelease(weight_dict);
            if !options_ptr.is_null() {
                CFRelease(options_ptr);
            }
        }

        if descriptor.is_null() {
            return Err(AneSysError::CompilationFailed(
                "modelWithMILText:weights:optionsPlist: returned nil".into(),
            ));
        }

        objc_retain(descriptor);
        Ok(Self { raw: descriptor })
    }

    /// Create a descriptor from a network description (legacy non-MIL format).
    ///
    /// Wraps `+[_ANEInMemoryModelDescriptor modelWithNetworkDescription:weights:optionsPlist:]`.
    ///
    /// # Deprecation
    ///
    /// This method uses the legacy network-description format. New code should
    /// use [`InMemoryModelDescriptor::from_mil_text`] instead, which accepts
    /// MIL text — the only ANE input format going forward.
    ///
    /// # Safety
    ///
    /// `weights_dict` must be a valid `NSDictionary` pointer or null.
    #[deprecated(
        since = "0.1.0",
        note = "Legacy non-MIL API. Use `InMemoryModelDescriptor::from_mil_text` instead."
    )]
    pub unsafe fn from_network_description(
        desc: &[u8],
        weights_dict: *mut c_void,
        options: Option<&[u8]>,
    ) -> Result<Self, AneSysError> {
        ane_framework()?;
        let desc_cls = get_class("_ANEInMemoryModelDescriptor")?;

        let desc_data = create_nsdata(desc)?;
        let options_ptr = match options {
            Some(data) => create_nsdata(data)?,
            None => std::ptr::null_mut(),
        };

        let sel =
            unsafe { sel_registerName(sel!("modelWithNetworkDescription:weights:optionsPlist:")) };

        type DescFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
        ) -> *mut c_void;
        let desc_fn: DescFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let descriptor = unsafe { desc_fn(desc_cls, sel, desc_data, weights_dict, options_ptr) };

        unsafe {
            CFRelease(desc_data);
            if !options_ptr.is_null() {
                CFRelease(options_ptr);
            }
        }

        if descriptor.is_null() {
            return Err(AneSysError::CompilationFailed(
                "modelWithNetworkDescription:weights:optionsPlist: returned nil".into(),
            ));
        }

        objc_retain(descriptor);
        Ok(Self { raw: descriptor })
    }

    /// Whether this descriptor describes a MIL model.
    pub fn is_mil_model(&self) -> bool {
        type BoolFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("isMILModel")) };
        let f: BoolFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) != 0 }
    }

    /// The hex-string identifier computed from the descriptor's hashes.
    pub fn hex_string_identifier(&self) -> Option<String> {
        read_nsstring_property(self.raw, "hexStringIdentifier")
    }

    /// Hash of the network text data.
    pub fn network_text_hash(&self) -> Option<String> {
        read_nsstring_property(self.raw, "networkTextHash")
    }

    /// Hash of the weights dictionary.
    pub fn weights_hash(&self) -> Option<String> {
        read_nsstring_property(self.raw, "weightsHash")
    }

    /// Raw ObjC pointer to the `_ANEInMemoryModelDescriptor`.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }
}

impl std::fmt::Debug for InMemoryModelDescriptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InMemoryModelDescriptor")
            .field("raw", &self.raw)
            .finish()
    }
}

impl Drop for InMemoryModelDescriptor {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { CFRelease(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

// =========================================================================
// InMemoryModel
// =========================================================================

/// Safe wrapper around `_ANEInMemoryModel`.
///
/// Represents a model that can be compiled, loaded, evaluated, and unloaded
/// on the Apple Neural Engine.
pub struct InMemoryModel {
    raw: *mut c_void,
}

// SAFETY: The raw handle is only accessed through &self methods.
unsafe impl Send for InMemoryModel {}

impl InMemoryModel {
    /// Create from a descriptor.
    ///
    /// Wraps `+[_ANEInMemoryModel inMemoryModelWithDescriptor:]`.
    pub fn from_descriptor(desc: &InMemoryModelDescriptor) -> Result<Self, AneSysError> {
        ane_framework()?;
        let imm_cls = get_class("_ANEInMemoryModel")?;

        let imm_sel = unsafe { sel_registerName(sel!("inMemoryModelWithDescriptor:")) };
        if !responds_to_selector(imm_cls, imm_sel) {
            return Err(AneSysError::CompilationFailed(
                "_ANEInMemoryModel does not respond to inMemoryModelWithDescriptor:".into(),
            ));
        }

        type ImmFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> *mut c_void;
        let imm_fn: ImmFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let model = unsafe { imm_fn(imm_cls, imm_sel, desc.raw) };
        if model.is_null() {
            return Err(AneSysError::CompilationFailed(
                "inMemoryModelWithDescriptor: returned nil".into(),
            ));
        }

        objc_retain(model);
        Ok(Self { raw: model })
    }

    /// Wrap an already-retained raw `_ANEInMemoryModel` pointer.
    ///
    /// # Safety
    ///
    /// `ptr` must be a valid, retained `_ANEInMemoryModel` handle.
    /// The caller transfers ownership — `Drop` will call `CFRelease`.
    pub unsafe fn from_raw(ptr: *mut c_void) -> Self {
        Self { raw: ptr }
    }

    // -- Lifecycle -----------------------------------------------------------

    /// Compile the model.
    ///
    /// Wraps `-[_ANEInMemoryModel compileWithQoS:options:error:]`.
    pub fn compile(&self, qos: u32) -> Result<(), AneSysError> {
        let empty_dict = ns_empty_dict()?;
        let compile_sel = unsafe { sel_registerName(sel!("compileWithQoS:options:error:")) };
        if !responds_to_selector(self.raw, compile_sel) {
            unsafe { CFRelease(empty_dict) };
            return Err(AneSysError::CompilationFailed(
                "_ANEInMemoryModel does not respond to compileWithQoS:options:error:".into(),
            ));
        }

        let mut error: *mut c_void = std::ptr::null_mut();
        type CompileFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            u32,
            *mut c_void,
            *mut *mut c_void,
        ) -> i8;
        let compile_fn: CompileFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ok = unsafe { compile_fn(self.raw, compile_sel, qos, empty_dict, &mut error) };
        unsafe { CFRelease(empty_dict) };

        if ok == 0 {
            let err_msg = if !error.is_null() {
                extract_nserror_description(error)
            } else {
                "compileWithQoS:options:error: returned NO".into()
            };
            return Err(AneSysError::CompilationFailed(err_msg));
        }
        Ok(())
    }

    /// Load the compiled model into the ANE.
    ///
    /// Wraps `-[_ANEInMemoryModel loadWithQoS:options:error:]`.
    pub fn load(&self, qos: u32) -> Result<(), AneSysError> {
        let empty_dict = ns_empty_dict()?;
        let mut error: *mut c_void = std::ptr::null_mut();
        type LoadFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            u32,
            *mut c_void,
            *mut *mut c_void,
        ) -> i8;
        let load_sel = unsafe { sel_registerName(sel!("loadWithQoS:options:error:")) };
        let load_fn: LoadFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ok = unsafe { load_fn(self.raw, load_sel, qos, empty_dict, &mut error) };
        unsafe { CFRelease(empty_dict) };

        if ok == 0 {
            let err_msg = if !error.is_null() {
                extract_nserror_description(error)
            } else {
                "loadWithQoS:options:error: returned NO".into()
            };
            return Err(AneSysError::LoadFailed(err_msg));
        }
        Ok(())
    }

    /// Unload the model from the ANE.
    ///
    /// Wraps `-[_ANEInMemoryModel unloadWithQoS:error:]`.
    pub fn unload(&self, qos: u32) -> Result<(), AneSysError> {
        let mut error: *mut c_void = std::ptr::null_mut();
        type UnloadFn = unsafe extern "C" fn(*mut c_void, *mut c_void, u32, *mut *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("unloadWithQoS:error:")) };
        let unload_fn: UnloadFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ok = unsafe { unload_fn(self.raw, sel, qos, &mut error) };

        if ok == 0 {
            let err_msg = if !error.is_null() {
                extract_nserror_description(error)
            } else {
                "unloadWithQoS:error: returned NO".into()
            };
            return Err(AneSysError::UnloadFailed(err_msg));
        }
        Ok(())
    }

    /// Evaluate the model with a request object.
    ///
    /// Wraps `-[_ANEInMemoryModel evaluateWithQoS:options:request:error:]`.
    ///
    /// # Safety
    ///
    /// `request` must be a valid `_ANERequest` pointer.
    pub unsafe fn evaluate(&self, qos: u32, request: *mut c_void) -> Result<(), AneSysError> {
        let empty_dict = ns_empty_dict()?;
        let mut error: *mut c_void = std::ptr::null_mut();

        type EvalFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            u32,
            *mut c_void,
            *mut c_void,
            *mut *mut c_void,
        ) -> i8;
        let eval_sel = unsafe { sel_registerName(sel!("evaluateWithQoS:options:request:error:")) };
        let eval_fn: EvalFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ok = unsafe { eval_fn(self.raw, eval_sel, qos, empty_dict, request, &mut error) };
        unsafe { CFRelease(empty_dict) };

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

    /// Evaluate with performance statistics collection.
    ///
    /// Sets `perf_stats_mask` on this model, attaches a fresh
    /// [`PerformanceStats`](crate::perf::PerformanceStats) to the request,
    /// evaluates, and returns the populated stats.  The original mask is
    /// restored afterward.
    ///
    /// # Safety
    ///
    /// `request` must contain valid `_ANEIOSurfaceObject` inputs/outputs.
    pub unsafe fn eval_with_stats(
        &self,
        qos: u32,
        request: &crate::request::AneRequest,
        perf_stats_mask: u32,
    ) -> Result<crate::perf::PerformanceStats, AneSysError> {
        let stats = crate::perf::PerformanceStats::with_hw_execution_ns(0)?;

        let old_mask = self.perf_stats_mask();
        self.set_perf_stats_mask(perf_stats_mask);
        // SAFETY: caller guarantees the request is valid; stats.as_raw() is
        // a retained _ANEPerformanceStats pointer.
        unsafe { request.set_perf_stats(stats.as_raw()) };

        let result = unsafe { self.evaluate(qos, request.as_raw()) };

        self.set_perf_stats_mask(old_mask);
        result?;
        Ok(stats)
    }

    // -- Compiled-model queries ----------------------------------------------

    /// Whether a compiled model already exists on disk for this model.
    pub fn compiled_model_exists(&self) -> bool {
        type BoolFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("compiledModelExists")) };
        let f: BoolFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) != 0 }
    }

    /// Purge any cached compiled model files.
    pub fn purge_compiled_model(&self) {
        type VoidFn = unsafe extern "C" fn(*mut c_void, *mut c_void);
        let sel = unsafe { sel_registerName(sel!("purgeCompiledModel")) };
        let f: VoidFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) };
    }

    // -- Properties (string) -------------------------------------------------

    /// The hex-string identifier for this model.
    pub fn hex_string_identifier(&self) -> Option<String> {
        read_nsstring_property(self.raw, "hexStringIdentifier")
    }

    /// Whether this is a MIL model.
    pub fn is_mil_model(&self) -> bool {
        type BoolFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("isMILModel")) };
        let f: BoolFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) != 0 }
    }

    // -- Properties (state / handles) ----------------------------------------

    /// Model state (`u64`).
    pub fn state(&self) -> u64 {
        type U64Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64;
        let sel = unsafe { sel_registerName(sel!("state")) };
        let f: U64Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Set the model state.
    pub fn set_state(&self, state: u64) {
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, u64);
        let sel = unsafe { sel_registerName(sel!("setState:")) };
        let f: SetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, state) };
    }

    /// Queue depth (`i8`).
    pub fn queue_depth(&self) -> i8 {
        type I8Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("queueDepth")) };
        let f: I8Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Set queue depth.
    pub fn set_queue_depth(&self, depth: i8) {
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, i8);
        let sel = unsafe { sel_registerName(sel!("setQueueDepth:")) };
        let f: SetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, depth) };
    }

    /// Program handle (`u64`).
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

    /// Intermediate buffer handle (`u64`).
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

    /// Performance stats mask (`u32`).
    pub fn perf_stats_mask(&self) -> u32 {
        type U32Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32;
        let sel = unsafe { sel_registerName(sel!("perfStatsMask")) };
        let f: U32Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Set the performance stats mask.
    pub fn set_perf_stats_mask(&self, mask: u32) {
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, u32);
        let sel = unsafe { sel_registerName(sel!("setPerfStatsMask:")) };
        let f: SetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, mask) };
    }

    /// Model attributes dictionary (raw `NSDictionary` pointer).
    pub fn model_attributes(&self) -> *mut c_void {
        type PtrFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("modelAttributes")) };
        let f: PtrFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Set model attributes.
    ///
    /// # Safety
    ///
    /// `attrs` must be a valid `NSDictionary` pointer or null.
    pub unsafe fn set_model_attributes(&self, attrs: *mut c_void) {
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void);
        let sel = unsafe { sel_registerName(sel!("setModelAttributes:")) };
        let f: SetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, attrs) };
    }

    // -- IOSurface mapping ---------------------------------------------------

    /// Map IOSurfaces for a request.
    ///
    /// Wraps `-[_ANEInMemoryModel mapIOSurfacesWithRequest:cacheInference:error:]`.
    ///
    /// # Safety
    ///
    /// `request` must be a valid `_ANERequest` pointer.
    pub unsafe fn map_iosurfaces(
        &self,
        request: *mut c_void,
        cache_inference: bool,
    ) -> Result<(), AneSysError> {
        let mut error: *mut c_void = std::ptr::null_mut();
        type MapFn =
            unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, i8, *mut *mut c_void) -> i8;
        let sel =
            unsafe { sel_registerName(sel!("mapIOSurfacesWithRequest:cacheInference:error:")) };
        let f: MapFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ok = unsafe { f(self.raw, sel, request, cache_inference as i8, &mut error) };

        if ok == 0 {
            let err_msg = if !error.is_null() {
                extract_nserror_description(error)
            } else {
                "mapIOSurfacesWithRequest:cacheInference:error: returned NO".into()
            };
            return Err(AneSysError::IOSurfaceMappingFailed(err_msg));
        }
        Ok(())
    }

    /// Unmap IOSurfaces for a request.
    ///
    /// Wraps `-[_ANEInMemoryModel unmapIOSurfacesWithRequest:]`.
    ///
    /// # Safety
    ///
    /// `request` must be a valid `_ANERequest` pointer.
    pub unsafe fn unmap_iosurfaces(&self, request: *mut c_void) {
        type VoidFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void);
        let sel = unsafe { sel_registerName(sel!("unmapIOSurfacesWithRequest:")) };
        let f: VoidFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, request) };
    }

    // -- File / path helpers -------------------------------------------------

    /// Save model files and return the path, if any.
    pub fn save_model_files(&self) -> Option<String> {
        read_nsstring_property(self.raw, "saveModelFiles")
    }

    /// Local model path on disk, if available.
    pub fn local_model_path(&self) -> Option<String> {
        read_nsstring_property(self.raw, "localModelPath")
    }

    /// The model's `NSURL` (raw pointer, may be null).
    pub fn model_url(&self) -> *mut c_void {
        type PtrFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("modelURL")) };
        let f: PtrFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Compiler options file name.
    pub fn compiler_options_file_name(&self) -> Option<String> {
        read_nsstring_property(self.raw, "compilerOptionsFileName")
    }

    /// Set the compiler options file name.
    pub fn set_compiler_options_file_name(&self, name: &str) {
        let ns = match create_nsstring(name) {
            Ok(s) => s,
            Err(_) => return,
        };
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void);
        let sel = unsafe { sel_registerName(sel!("setCompilerOptionsFileName:")) };
        let f: SetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, ns) };
        unsafe { CFRelease(ns) };
    }

    /// Raw ObjC pointer to the `_ANEInMemoryModel`.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }
}

impl std::fmt::Debug for InMemoryModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InMemoryModel")
            .field("raw", &self.raw)
            .finish()
    }
}

impl Drop for InMemoryModel {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { CFRelease(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

// =========================================================================
// Convenience functions (preserve the simple existing API)
// =========================================================================

/// Compile MIL text into a ready-to-evaluate model (compile + load in one
/// step).
pub fn compile_mil_text(
    mil_text: &str,
    weights: &[(&str, &[u8])],
    qos: u32,
) -> Result<InMemoryModel, AneSysError> {
    // 0. Budget check
    let prev = COMPILE_COUNT.fetch_add(1, Ordering::SeqCst);
    if prev >= ANE_COMPILE_LIMIT {
        COMPILE_COUNT.fetch_sub(1, Ordering::SeqCst);
        return Err(AneSysError::BudgetExhausted { count: prev });
    }

    let result = (|| -> Result<InMemoryModel, AneSysError> {
        if mil_text.is_empty() {
            return Err(AneSysError::InvalidInput("MIL text is empty".into()));
        }

        let pool = unsafe { objc_autoreleasePoolPush() };

        let inner = (|| -> Result<InMemoryModel, AneSysError> {
            // 1. Create descriptor
            #[cfg(debug_assertions)]
            eprintln!(
                "[ane] creating descriptor from MIL text ({} bytes), {} weight(s)...",
                mil_text.len(),
                weights.len()
            );
            let desc = InMemoryModelDescriptor::from_mil_text(mil_text, weights, None)?;

            // 2. Create model
            #[cfg(debug_assertions)]
            eprintln!("[ane] descriptor created, creating in-memory model...");
            let model = InMemoryModel::from_descriptor(&desc)?;

            // 3. Pre-populate temp directory
            if let Some(ref hex) = model.hex_string_identifier() {
                populate_tmp_dir(hex, mil_text, weights)?;
            }

            // 4. Compile
            #[cfg(debug_assertions)]
            eprintln!("[ane] compiling with QoS={qos}...");
            model.compile(qos)?;

            // 5. Load
            model.load(qos)?;

            Ok(model)
        })();

        unsafe { objc_autoreleasePoolPop(pool) };
        inner
    })();

    if result.is_err() {
        COMPILE_COUNT.fetch_sub(1, Ordering::SeqCst);
    }

    result
}

/// Patch weights using a donor model's compiled artifacts.
///
/// Creates a new model with different weights, copies the donor's compiled
/// `net.plist`, then loads — **skipping compilation entirely**.
///
/// Does **not** consume a compile budget slot.
pub fn patch_weights(
    donor: &InMemoryModel,
    mil_text: &str,
    weights: &[(&str, &[u8])],
    qos: u32,
) -> Result<InMemoryModel, AneSysError> {
    if donor.raw.is_null() {
        return Err(AneSysError::InvalidInput(
            "donor model pointer is null".into(),
        ));
    }
    if mil_text.is_empty() {
        return Err(AneSysError::InvalidInput("MIL text is empty".into()));
    }

    let pool = unsafe { objc_autoreleasePoolPush() };

    let result = (|| -> Result<InMemoryModel, AneSysError> {
        // 1. Get donor's hex ID → find its temp dir with net.plist
        let donor_hex = donor.hex_string_identifier().ok_or_else(|| {
            AneSysError::CompilationFailed("failed to get donor model hexStringIdentifier".into())
        })?;
        let donor_tmp = std::env::temp_dir().join(&donor_hex);
        let donor_net_plist = donor_tmp.join("net.plist");
        if !donor_net_plist.exists() {
            return Err(AneSysError::CompilationFailed(format!(
                "donor net.plist not found at {}",
                donor_net_plist.display()
            )));
        }

        // 2. Create descriptor + model with new weights
        let desc = InMemoryModelDescriptor::from_mil_text(mil_text, weights, None)?;
        let model = InMemoryModel::from_descriptor(&desc)?;

        // 3. Get new model's hex ID → set up its temp dir
        let new_hex = model.hex_string_identifier().ok_or_else(|| {
            AneSysError::CompilationFailed("failed to get new model hexStringIdentifier".into())
        })?;
        populate_tmp_dir(&new_hex, mil_text, weights)?;

        // 4. Copy donor's net.plist → new model's temp dir (the key trick)
        let new_tmp = std::env::temp_dir().join(&new_hex);
        let new_net_plist = new_tmp.join("net.plist");
        std::fs::copy(&donor_net_plist, &new_net_plist)?;

        #[cfg(debug_assertions)]
        eprintln!("[ane] patch_weights: copied net.plist from {donor_hex} → {new_hex}");

        // 5. Load (NO compile!)
        model.load(qos)?;

        Ok(model)
    })();

    unsafe { objc_autoreleasePoolPop(pool) };
    result
}

/// Evaluate a model with IOSurface pointers.
///
/// Builds an `_ANERequest` from the surface arrays and calls
/// `evaluateWithQoS:options:request:error:` on the model.
pub fn eval(
    model: &InMemoryModel,
    input_surfaces: &[*mut c_void],
    output_surfaces: &[*mut c_void],
    qos: u32,
) -> Result<(), AneSysError> {
    eval_inner(
        model,
        input_surfaces,
        output_surfaces,
        qos,
        std::ptr::null_mut(),
    )
}

/// Evaluate a model and collect hardware performance stats.
///
/// Like [`eval`] but sets `perf_stats_mask` on the model before evaluation,
/// passes a [`PerformanceStats`](crate::perf::PerformanceStats) collector to
/// the request, and returns the populated stats.  The original mask is
/// restored afterward.
pub fn eval_with_stats(
    model: &InMemoryModel,
    input_surfaces: &[*mut c_void],
    output_surfaces: &[*mut c_void],
    qos: u32,
    perf_stats_mask: u32,
) -> Result<crate::perf::PerformanceStats, AneSysError> {
    let stats = crate::perf::PerformanceStats::with_hw_execution_ns(0)?;

    let old_mask = model.perf_stats_mask();
    model.set_perf_stats_mask(perf_stats_mask);
    let result = eval_inner(model, input_surfaces, output_surfaces, qos, stats.as_raw());

    model.set_perf_stats_mask(old_mask);
    result?;
    Ok(stats)
}

/// Shared eval implementation.  `perf_stats` is passed to the request
/// factory; null disables stats collection.
fn eval_inner(
    model: &InMemoryModel,
    input_surfaces: &[*mut c_void],
    output_surfaces: &[*mut c_void],
    qos: u32,
    perf_stats: *mut c_void,
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

    unsafe {
        let pool = objc_autoreleasePoolPush();

        if let Err(e) = ane_framework() {
            objc_autoreleasePoolPop(pool);
            return Err(e);
        }

        let aio_cls = match get_class("_ANEIOSurfaceObject") {
            Ok(c) => c,
            Err(e) => {
                objc_autoreleasePoolPop(pool);
                return Err(e);
            }
        };
        let req_cls = match get_class("_ANERequest") {
            Ok(c) => c,
            Err(e) => {
                objc_autoreleasePoolPop(pool);
                return Err(e);
            }
        };

        // Allocate retained arrays
        let in_arr = match ns_mutable_array() {
            Ok(a) => a,
            Err(e) => {
                objc_autoreleasePoolPop(pool);
                return Err(e);
            }
        };
        let in_idx = match ns_mutable_array() {
            Ok(a) => a,
            Err(e) => {
                CFRelease(in_arr);
                objc_autoreleasePoolPop(pool);
                return Err(e);
            }
        };
        let out_arr = match ns_mutable_array() {
            Ok(a) => a,
            Err(e) => {
                CFRelease(in_arr);
                CFRelease(in_idx);
                objc_autoreleasePoolPop(pool);
                return Err(e);
            }
        };
        let out_idx = match ns_mutable_array() {
            Ok(a) => a,
            Err(e) => {
                CFRelease(in_arr);
                CFRelease(in_idx);
                CFRelease(out_arr);
                objc_autoreleasePoolPop(pool);
                return Err(e);
            }
        };

        // Use a closure so we always clean up the four arrays
        let result = (|| -> Result<(), AneSysError> {
            // Wrap inputs
            for (i, &surface) in input_surfaces.iter().enumerate() {
                let wrapped = wrap_iosurface(aio_cls, surface)?;
                ns_array_add(in_arr, wrapped);
                ns_array_add(in_idx, ns_number_autoreleased(i as i64)?);
            }

            // Wrap outputs
            for (i, &surface) in output_surfaces.iter().enumerate() {
                let wrapped = wrap_iosurface(aio_cls, surface)?;
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
            let req_sel = sel_registerName(sel!(
                "requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"
            ));
            let req_fn: RequestFn = std::mem::transmute(objc_msgSend as *const ());
            let request = req_fn(
                req_cls,
                req_sel,
                in_arr,
                in_idx,
                out_arr,
                out_idx,
                std::ptr::null_mut(),
                perf_stats,
                zero,
            );

            if request.is_null() {
                return Err(AneSysError::EvalFailed {
                    status: 0,
                    context: "failed to create _ANERequest".into(),
                });
            }

            // Evaluate
            let empty_dict = ns_empty_dict_unchecked()?;

            let mut error: *mut c_void = std::ptr::null_mut();
            type EvalFn = unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                u32,
                *mut c_void,
                *mut c_void,
                *mut *mut c_void,
            ) -> i8;
            let eval_sel = sel_registerName(sel!("evaluateWithQoS:options:request:error:"));
            let eval_fn: EvalFn = std::mem::transmute(objc_msgSend as *const ());
            let ok = eval_fn(model.raw, eval_sel, qos, empty_dict, request, &mut error);

            CFRelease(empty_dict);

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
        })();

        // Always release retained arrays
        CFRelease(in_arr);
        CFRelease(in_idx);
        CFRelease(out_arr);
        CFRelease(out_idx);

        objc_autoreleasePoolPop(pool);

        result
    }
}

/// Check whether the ANE is available on this system.
pub fn is_available() -> bool {
    if ane_framework().is_err() {
        return false;
    }
    get_class("_ANEInMemoryModel").is_ok()
}

/// Number of compilations performed in this process.
pub fn compile_count() -> usize {
    COMPILE_COUNT.load(Ordering::Relaxed)
}

/// Remaining compile budget before hitting the ~119 limit.
pub fn remaining_budget() -> usize {
    ANE_COMPILE_LIMIT.saturating_sub(COMPILE_COUNT.load(Ordering::Relaxed))
}

/// Create a BLOBFILE in Orion's format from raw weight data.
///
/// Format: 128-byte header (file header + chunk descriptor) + data.
pub fn make_blobfile(data: &[u8]) -> Result<Vec<u8>, AneSysError> {
    let data_size = data.len();
    if data_size > u32::MAX as usize {
        return Err(AneSysError::InvalidInput(format!(
            "BLOBFILE data size {} exceeds u32::MAX",
            data_size
        )));
    }
    let total = 128 + data_size;
    let mut buf = vec![0u8; total];

    // File header (bytes 0-63)
    buf[0] = 1;
    buf[4] = 2;

    // Chunk descriptor (bytes 64-127)
    buf[64] = 0xEF; // 0xDEADBEEF magic
    buf[65] = 0xBE;
    buf[66] = 0xAD;
    buf[67] = 0xDE;
    buf[68] = 1;
    // Data size (bytes 72-75, u32 LE)
    buf[72..76].copy_from_slice(&(data_size as u32).to_le_bytes());
    // Data offset = 128 (bytes 80-83, u32 LE)
    buf[80..84].copy_from_slice(&128u32.to_le_bytes());

    // Weight data (bytes 128+)
    buf[128..].copy_from_slice(data);

    Ok(buf)
}

// =========================================================================
// Internal helpers
// =========================================================================

/// Read an NSString property by selector name from an ObjC object.
fn read_nsstring_property(obj: *mut c_void, selector: &str) -> Option<String> {
    let mut buf = Vec::with_capacity(selector.len() + 1);
    buf.extend_from_slice(selector.as_bytes());
    buf.push(0);

    type StrFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
    let sel = unsafe { sel_registerName(buf.as_ptr()) };
    let f: StrFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    let ns = unsafe { f(obj, sel) };
    unsafe { nsstring_to_string(ns) }
}

/// Wrap an IOSurface pointer in `_ANEIOSurfaceObject`.
fn wrap_iosurface(aio_cls: *mut c_void, surface: *mut c_void) -> Result<*mut c_void, AneSysError> {
    type WrapFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> *mut c_void;
    let sel = unsafe { sel_registerName(sel!("objectWithIOSurface:")) };
    let f: WrapFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    let wrapped = unsafe { f(aio_cls, sel, surface) };
    if wrapped.is_null() {
        return Err(AneSysError::EvalFailed {
            status: 0,
            context: "objectWithIOSurface: returned nil".into(),
        });
    }
    Ok(wrapped)
}

/// Build the weight dictionary in Orion's format with multiple entries.
fn build_multi_weight_dict(weights: &[(&str, &[u8])]) -> Result<*mut c_void, AneSysError> {
    let outer_dict = ns_mutable_dict()?;

    for (path_key, data) in weights {
        let blob = make_blobfile(data)?;
        let data_nsdata = create_nsdata(&blob)?;
        let offset_num = create_nsnumber(64)?;

        let inner_dict = ns_mutable_dict()?;
        let data_key = create_nsstring("data")?;
        let offset_key = create_nsstring("offset")?;
        ns_dict_set(inner_dict, data_key, data_nsdata);
        ns_dict_set(inner_dict, offset_key, offset_num);

        unsafe {
            CFRelease(data_key);
            CFRelease(offset_key);
            CFRelease(data_nsdata);
            CFRelease(offset_num);
        }

        let outer_key = create_nsstring(path_key)?;
        ns_dict_set(outer_dict, outer_key, inner_dict);

        unsafe {
            CFRelease(outer_key);
            CFRelease(inner_dict);
        }
    }

    Ok(outer_dict)
}

/// Pre-populate the model's temp directory with MIL text and weight blobs.
fn populate_tmp_dir(
    hex_id: &str,
    mil_text: &str,
    weights: &[(&str, &[u8])],
) -> Result<(), AneSysError> {
    let tmp_dir = std::env::temp_dir().join(hex_id);
    let weights_dir = tmp_dir.join("weights");

    #[cfg(debug_assertions)]
    eprintln!("[ane] hexId={hex_id}, tmp_dir={}", tmp_dir.display());

    std::fs::create_dir_all(&weights_dir)?;
    std::fs::write(tmp_dir.join("model.mil"), mil_text.as_bytes())?;
    for (path_key, data) in weights {
        let rel_path = path_key.strip_prefix("@model_path/").unwrap_or(path_key);
        let full_path = tmp_dir.join(rel_path);
        if let Some(parent) = full_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        #[cfg(debug_assertions)]
        eprintln!(
            "[ane] writing weight {} ({} bytes) → {}",
            path_key,
            data.len(),
            full_path.display()
        );
        let blob = make_blobfile(data)?;
        std::fs::write(&full_path, &blob)?;
    }
    Ok(())
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_count_starts_at_zero_or_accumulates() {
        let count = compile_count();
        assert!(
            count < ANE_COMPILE_LIMIT + 50,
            "count looks unreasonably high: {count}"
        );
    }

    #[test]
    fn remaining_budget_consistent() {
        let remaining = remaining_budget();
        let count = compile_count();
        assert_eq!(
            remaining,
            ANE_COMPILE_LIMIT.saturating_sub(count),
            "remaining budget should equal limit minus count"
        );
    }

    #[test]
    fn compile_mil_text_empty_returns_error() {
        let result = compile_mil_text("", &[], ANE_QOS);
        assert!(result.is_err());
        match result.unwrap_err() {
            AneSysError::InvalidInput(msg) => {
                assert!(msg.contains("empty"), "expected 'empty' in message: {msg}");
            }
            other => panic!("expected InvalidInput, got: {other}"),
        }
    }

    #[test]
    fn patch_weights_null_donor_returns_error() {
        let donor = unsafe { InMemoryModel::from_raw(std::ptr::null_mut()) };
        let result = patch_weights(&donor, "program test {}", &[], ANE_QOS);
        assert!(result.is_err());
        match result.unwrap_err() {
            AneSysError::InvalidInput(msg) => {
                assert!(msg.contains("null"), "expected 'null' in message: {msg}");
            }
            other => panic!("expected InvalidInput, got: {other}"),
        }
        std::mem::forget(donor);
    }

    #[test]
    fn patch_weights_empty_text_returns_error() {
        let dummy = unsafe { InMemoryModel::from_raw(0x1 as *mut c_void) };
        let result = patch_weights(&dummy, "", &[], ANE_QOS);
        assert!(result.is_err());
        match result.unwrap_err() {
            AneSysError::InvalidInput(msg) => {
                assert!(msg.contains("empty"), "expected 'empty' in message: {msg}");
            }
            other => panic!("expected InvalidInput, got: {other}"),
        }
        std::mem::forget(dummy);
    }

    #[test]
    fn make_blobfile_roundtrip() {
        let data = b"hello weights";
        let blob = make_blobfile(data).unwrap();
        assert_eq!(blob.len(), 128 + data.len());
        assert_eq!(&blob[128..], data);
        // Check magic
        assert_eq!(blob[64], 0xEF);
        assert_eq!(blob[65], 0xBE);
        assert_eq!(blob[66], 0xAD);
        assert_eq!(blob[67], 0xDE);
    }

    #[test]
    fn eval_empty_inputs_returns_error() {
        let model = unsafe { InMemoryModel::from_raw(0x1 as *mut c_void) };
        let result = eval(&model, &[], &[std::ptr::null_mut()], ANE_QOS);
        assert!(result.is_err());
        std::mem::forget(model);
    }

    #[test]
    fn eval_empty_outputs_returns_error() {
        let model = unsafe { InMemoryModel::from_raw(0x1 as *mut c_void) };
        let result = eval(&model, &[std::ptr::null_mut()], &[], ANE_QOS);
        assert!(result.is_err());
        std::mem::forget(model);
    }
}
