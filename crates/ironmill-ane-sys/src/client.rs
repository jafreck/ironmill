//! Safe wrappers for `_ANEClient` and `_ANEDaemonConnection`.
//!
//! `_ANEClient` is the high-level ObjC client for the ANE daemon — it exposes
//! compile, load, unload, evaluate, IOSurface mapping, real-time scheduling,
//! chaining, and cache management operations.
//!
//! `_ANEDaemonConnection` is the lower-level XPC transport layer that mirrors
//! most `_ANEClient` methods but with asynchronous `withReply:` blocks.
//!
//! # ⚠️ Private API — see crate-level docs for risks and caveats.

use std::ffi::c_void;

use crate::error::AneSysError;
use crate::objc::{CFRelease, get_class, objc_msgSend, objc_retain, sel, sel_registerName};

// ---------------------------------------------------------------------------
// _ANEClient
// ---------------------------------------------------------------------------

/// Wrapper around `_ANEClient`, the high-level ANE daemon client.
///
/// Owns a retained ObjC object; released on drop.
pub struct Client {
    raw: *mut c_void,
}

// SAFETY: The raw handle is only accessed through &self methods.
unsafe impl Send for Client {}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client").field("raw", &self.raw).finish()
    }
}

impl Client {
    /// Raw ObjC object pointer.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    // -- class-method factories ---------------------------------------------

    /// Singleton shared connection (`+[_ANEClient sharedConnection]`).
    pub fn shared_connection() -> Result<Self, AneSysError> {
        let cls = get_class("_ANEClient")?;
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("sharedConnection")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, sel) };
        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "sharedConnection returned nil".into(),
            });
        }
        objc_retain(obj);
        Ok(Self { raw: obj })
    }

    /// Singleton private connection (`+[_ANEClient sharedPrivateConnection]`).
    pub fn shared_private_connection() -> Result<Self, AneSysError> {
        let cls = get_class("_ANEClient")?;
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("sharedPrivateConnection")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, sel) };
        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "sharedPrivateConnection returned nil".into(),
            });
        }
        objc_retain(obj);
        Ok(Self { raw: obj })
    }

    // -- compile / load / unload -------------------------------------------

    /// `-compileModel:options:qos:error:` → `BOOL`
    ///
    /// # Safety
    ///
    /// `model`, `options`, and `error` must be valid ObjC object pointers (or null).
    pub unsafe fn compile_model(
        &self,
        model: *mut c_void,
        options: *mut c_void,
        qos: u32,
        error: *mut *mut c_void,
    ) -> bool {
        type Fn5 = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            u32,
            *mut *mut c_void,
        ) -> i8;
        let sel = unsafe { sel_registerName(sel!("compileModel:options:qos:error:")) };
        let f: Fn5 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, model, options, qos, error) != 0 }
    }

    /// `-loadModel:options:qos:error:` → `BOOL`
    ///
    /// # Safety
    ///
    /// `model`, `options`, and `error` must be valid ObjC object pointers (or null).
    pub unsafe fn load_model(
        &self,
        model: *mut c_void,
        options: *mut c_void,
        qos: u32,
        error: *mut *mut c_void,
    ) -> bool {
        type Fn5 = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            u32,
            *mut *mut c_void,
        ) -> i8;
        let sel = unsafe { sel_registerName(sel!("loadModel:options:qos:error:")) };
        let f: Fn5 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, model, options, qos, error) != 0 }
    }

    /// `-loadModelNewInstance:options:modelInstParams:qos:error:` → `BOOL`
    ///
    /// # Safety
    ///
    /// All object pointer arguments must be valid ObjC objects (or null).
    pub unsafe fn load_model_new_instance(
        &self,
        model: *mut c_void,
        options: *mut c_void,
        model_inst_params: *mut c_void,
        qos: u32,
        error: *mut *mut c_void,
    ) -> bool {
        type Fn6 = unsafe extern "C" fn(
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
        let f: Fn6 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, model, options, model_inst_params, qos, error) != 0 }
    }

    /// `-unloadModel:options:qos:error:` → `BOOL`
    ///
    /// # Safety
    ///
    /// `model`, `options`, and `error` must be valid ObjC object pointers (or null).
    pub unsafe fn unload_model(
        &self,
        model: *mut c_void,
        options: *mut c_void,
        qos: u32,
        error: *mut *mut c_void,
    ) -> bool {
        type Fn5 = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            u32,
            *mut *mut c_void,
        ) -> i8;
        let sel = unsafe { sel_registerName(sel!("unloadModel:options:qos:error:")) };
        let f: Fn5 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, model, options, qos, error) != 0 }
    }

    // -- evaluate ----------------------------------------------------------

    /// `-evaluateWithModel:options:request:qos:error:` → `BOOL`
    ///
    /// # Safety
    ///
    /// All object pointer arguments must be valid ObjC objects (or null).
    pub unsafe fn evaluate(
        &self,
        model: *mut c_void,
        options: *mut c_void,
        request: *mut c_void,
        qos: u32,
        error: *mut *mut c_void,
    ) -> bool {
        type Fn6 = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            u32,
            *mut *mut c_void,
        ) -> i8;
        let sel = unsafe { sel_registerName(sel!("evaluateWithModel:options:request:qos:error:")) };
        let f: Fn6 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, model, options, request, qos, error) != 0 }
    }

    // -- cache checks / purge ----------------------------------------------

    /// `-compiledModelExistsFor:` → `BOOL`
    ///
    /// # Safety
    ///
    /// `model` must be a valid ObjC object pointer.
    pub unsafe fn compiled_model_exists_for(&self, model: *mut c_void) -> bool {
        type BoolArgFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("compiledModelExistsFor:")) };
        let f: BoolArgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, model) != 0 }
    }

    /// `-compiledModelExistsMatchingHash:` → `BOOL`
    ///
    /// # Safety
    ///
    /// `hash` must be a valid ObjC object pointer.
    pub unsafe fn compiled_model_exists_matching_hash(&self, hash: *mut c_void) -> bool {
        type BoolArgFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("compiledModelExistsMatchingHash:")) };
        let f: BoolArgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, hash) != 0 }
    }

    /// `-purgeCompiledModel:` (void)
    ///
    /// # Safety
    ///
    /// `model` must be a valid ObjC object pointer.
    pub unsafe fn purge_compiled_model(&self, model: *mut c_void) {
        type VoidArgFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void);
        let sel = unsafe { sel_registerName(sel!("purgeCompiledModel:")) };
        let f: VoidArgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, model) };
    }

    /// `-purgeCompiledModelMatchingHash:` (void)
    ///
    /// # Safety
    ///
    /// `hash` must be a valid ObjC object pointer.
    pub unsafe fn purge_compiled_model_matching_hash(&self, hash: *mut c_void) {
        type VoidArgFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void);
        let sel = unsafe { sel_registerName(sel!("purgeCompiledModelMatchingHash:")) };
        let f: VoidArgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, hash) };
    }

    // -- IOSurface mapping -------------------------------------------------

    /// `-mapIOSurfacesWithModel:request:cacheInference:error:` → `BOOL`
    ///
    /// # Safety
    ///
    /// `model`, `request`, and `error` must be valid ObjC object pointers (or null).
    pub unsafe fn map_io_surfaces(
        &self,
        model: *mut c_void,
        request: *mut c_void,
        cache_inference: bool,
        error: *mut *mut c_void,
    ) -> bool {
        type Fn5 = unsafe extern "C" fn(
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
        let f: Fn5 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, model, request, cache_inference as i8, error) != 0 }
    }

    /// `-unmapIOSurfacesWithModel:request:` (void)
    ///
    /// # Safety
    ///
    /// `model` and `request` must be valid ObjC object pointers.
    pub unsafe fn unmap_io_surfaces(&self, model: *mut c_void, request: *mut c_void) {
        type VoidFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void);
        let sel = unsafe { sel_registerName(sel!("unmapIOSurfacesWithModel:request:")) };
        let f: VoidFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, model, request) };
    }

    // -- real-time scheduling ----------------------------------------------

    /// `-beginRealTimeTask` → `BOOL`
    pub fn begin_real_time_task(&self) -> bool {
        type BoolFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("beginRealTimeTask")) };
        let f: BoolFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        (unsafe { f(self.raw, sel) }) != 0
    }

    /// `-endRealTimeTask` → `BOOL`
    pub fn end_real_time_task(&self) -> bool {
        type BoolFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("endRealTimeTask")) };
        let f: BoolFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        (unsafe { f(self.raw, sel) }) != 0
    }

    /// `-evaluateRealTimeWithModel:options:request:error:` → `BOOL`
    ///
    /// # Safety
    ///
    /// All object pointer arguments must be valid ObjC objects (or null).
    pub unsafe fn evaluate_real_time(
        &self,
        model: *mut c_void,
        options: *mut c_void,
        request: *mut c_void,
        error: *mut *mut c_void,
    ) -> bool {
        type Fn5 = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut *mut c_void,
        ) -> i8;
        let sel =
            unsafe { sel_registerName(sel!("evaluateRealTimeWithModel:options:request:error:")) };
        let f: Fn5 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, model, options, request, error) != 0 }
    }

    /// `-loadRealTimeModel:options:qos:error:` → `BOOL`
    ///
    /// # Safety
    ///
    /// `model`, `options`, and `error` must be valid ObjC object pointers (or null).
    pub unsafe fn load_real_time_model(
        &self,
        model: *mut c_void,
        options: *mut c_void,
        qos: u32,
        error: *mut *mut c_void,
    ) -> bool {
        type Fn5 = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            u32,
            *mut *mut c_void,
        ) -> i8;
        let sel = unsafe { sel_registerName(sel!("loadRealTimeModel:options:qos:error:")) };
        let f: Fn5 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, model, options, qos, error) != 0 }
    }

    /// `-unloadRealTimeModel:options:qos:error:` → `BOOL`
    ///
    /// # Safety
    ///
    /// `model`, `options`, and `error` must be valid ObjC object pointers (or null).
    pub unsafe fn unload_real_time_model(
        &self,
        model: *mut c_void,
        options: *mut c_void,
        qos: u32,
        error: *mut *mut c_void,
    ) -> bool {
        type Fn5 = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            u32,
            *mut *mut c_void,
        ) -> i8;
        let sel = unsafe { sel_registerName(sel!("unloadRealTimeModel:options:qos:error:")) };
        let f: Fn5 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, model, options, qos, error) != 0 }
    }

    // -- chaining / async buffers ------------------------------------------

    /// `-prepareChainingWithModel:options:chainingReq:qos:error:` → `BOOL`
    ///
    /// # Safety
    ///
    /// All object pointer arguments must be valid ObjC objects (or null).
    pub unsafe fn prepare_chaining(
        &self,
        model: *mut c_void,
        options: *mut c_void,
        chaining_req: *mut c_void,
        qos: u32,
        error: *mut *mut c_void,
    ) -> bool {
        type Fn6 = unsafe extern "C" fn(
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
                "prepareChainingWithModel:options:chainingReq:qos:error:"
            ))
        };
        let f: Fn6 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, model, options, chaining_req, qos, error) != 0 }
    }

    /// `-buffersReadyWithModel:inputBuffers:options:qos:error:` → `BOOL`
    ///
    /// # Safety
    ///
    /// All object pointer arguments must be valid ObjC objects (or null).
    pub unsafe fn buffers_ready(
        &self,
        model: *mut c_void,
        input_buffers: *mut c_void,
        options: *mut c_void,
        qos: u32,
        error: *mut *mut c_void,
    ) -> bool {
        type Fn6 = unsafe extern "C" fn(
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
                "buffersReadyWithModel:inputBuffers:options:qos:error:"
            ))
        };
        let f: Fn6 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, model, input_buffers, options, qos, error) != 0 }
    }

    // -- echo / introspection ----------------------------------------------

    /// `-echo:` → `BOOL` — connectivity test.
    ///
    /// # Safety
    ///
    /// `payload` must be a valid ObjC object pointer.
    pub unsafe fn echo(&self, payload: *mut c_void) -> bool {
        type BoolArgFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("echo:")) };
        let f: BoolArgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, payload) != 0 }
    }

    /// `-isVirtualClient` → `BOOL`
    pub fn is_virtual_client(&self) -> bool {
        type BoolFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("isVirtualClient")) };
        let f: BoolFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        (unsafe { f(self.raw, sel) }) != 0
    }

    /// `-allowRestrictedAccess` → `BOOL`
    pub fn allow_restricted_access(&self) -> bool {
        type BoolFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("allowRestrictedAccess")) };
        let f: BoolFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        (unsafe { f(self.raw, sel) }) != 0
    }

    // -- session hint ------------------------------------------------------

    /// `-sessionHintWithModel:hint:options:report:error:` → `BOOL`
    ///
    /// # Safety
    ///
    /// All object pointer arguments must be valid ObjC objects (or null).
    pub unsafe fn session_hint(
        &self,
        model: *mut c_void,
        hint: *mut c_void,
        options: *mut c_void,
        report: *mut c_void,
        error: *mut *mut c_void,
    ) -> bool {
        type Fn6 = unsafe extern "C" fn(
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
        let f: Fn6 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, model, hint, options, report, error) != 0 }
    }

    // -- property accessors ------------------------------------------------

    /// `-conn` → raw `_ANEDaemonConnection` pointer.
    pub fn conn(&self) -> *mut c_void {
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("conn")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// `-fastConn` → raw `_ANEDaemonConnection` pointer.
    pub fn fast_conn(&self) -> *mut c_void {
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("fastConn")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { CFRelease(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// _ANEDaemonConnection
// ---------------------------------------------------------------------------

/// Wrapper around `_ANEDaemonConnection`, the XPC transport for the ANE daemon.
///
/// Most methods mirror `_ANEClient` but accept `withReply:` completion blocks.
/// The reply-block parameters are exposed as raw `*mut c_void` for now.
///
/// Owns a retained ObjC object; released on drop.
pub struct DaemonConnection {
    raw: *mut c_void,
}

// SAFETY: The raw handle is only accessed through &self methods.
unsafe impl Send for DaemonConnection {}

impl std::fmt::Debug for DaemonConnection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DaemonConnection")
            .field("raw", &self.raw)
            .finish()
    }
}

impl DaemonConnection {
    /// Raw ObjC object pointer.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    // -- class-method factories ---------------------------------------------

    /// `+[_ANEDaemonConnection daemonConnection]`
    pub fn new() -> Result<Self, AneSysError> {
        let cls = get_class("_ANEDaemonConnection")?;
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("daemonConnection")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, sel) };
        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "daemonConnection returned nil".into(),
            });
        }
        objc_retain(obj);
        Ok(Self { raw: obj })
    }

    /// `+[_ANEDaemonConnection daemonConnectionRestricted]`
    pub fn daemon_connection_restricted() -> Result<Self, AneSysError> {
        let cls = get_class("_ANEDaemonConnection")?;
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("daemonConnectionRestricted")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, sel) };
        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "daemonConnectionRestricted returned nil".into(),
            });
        }
        objc_retain(obj);
        Ok(Self { raw: obj })
    }

    /// `+[_ANEDaemonConnection userDaemonConnection]`
    pub fn user_daemon_connection() -> Result<Self, AneSysError> {
        let cls = get_class("_ANEDaemonConnection")?;
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("userDaemonConnection")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, sel) };
        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "userDaemonConnection returned nil".into(),
            });
        }
        objc_retain(obj);
        Ok(Self { raw: obj })
    }

    // -- property accessors ------------------------------------------------

    /// `-restricted` → `BOOL`
    pub fn restricted(&self) -> bool {
        type BoolFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i8;
        let sel = unsafe { sel_registerName(sel!("restricted")) };
        let f: BoolFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        (unsafe { f(self.raw, sel) }) != 0
    }

    /// `-daemonConnection` → raw `NSXPCConnection` pointer.
    pub fn xpc_connection(&self) -> *mut c_void {
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("daemonConnection")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    // -- async (withReply:) methods ----------------------------------------
    // Reply blocks are passed as raw `*mut c_void` — callers are responsible
    // for constructing valid ObjC block objects.

    /// `-echo:withReply:`
    ///
    /// # Safety
    ///
    /// `payload` and `reply` must be valid ObjC object / block pointers.
    pub unsafe fn echo_with_reply(&self, payload: *mut c_void, reply: *mut c_void) {
        type Fn3 = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void);
        let sel = unsafe { sel_registerName(sel!("echo:withReply:")) };
        let f: Fn3 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, payload, reply) };
    }

    /// `-beginRealTimeTaskWithReply:`
    ///
    /// # Safety
    ///
    /// `reply` must be a valid ObjC block pointer.
    pub unsafe fn begin_real_time_task_with_reply(&self, reply: *mut c_void) {
        type Fn2 = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void);
        let sel = unsafe { sel_registerName(sel!("beginRealTimeTaskWithReply:")) };
        let f: Fn2 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, reply) };
    }

    /// `-endRealTimeTaskWithReply:`
    ///
    /// # Safety
    ///
    /// `reply` must be a valid ObjC block pointer.
    pub unsafe fn end_real_time_task_with_reply(&self, reply: *mut c_void) {
        type Fn2 = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void);
        let sel = unsafe { sel_registerName(sel!("endRealTimeTaskWithReply:")) };
        let f: Fn2 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, reply) };
    }

    /// `-compileModel:sandboxExtension:options:qos:withReply:`
    ///
    /// # Safety
    ///
    /// All object pointer / block arguments must be valid.
    pub unsafe fn compile_model_with_reply(
        &self,
        model: *mut c_void,
        sandbox_ext: *mut c_void,
        options: *mut c_void,
        qos: u32,
        reply: *mut c_void,
    ) {
        type Fn6 = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            u32,
            *mut c_void,
        );
        let sel = unsafe {
            sel_registerName(sel!("compileModel:sandboxExtension:options:qos:withReply:"))
        };
        let f: Fn6 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, model, sandbox_ext, options, qos, reply) };
    }

    /// `-loadModel:sandboxExtension:options:qos:withReply:`
    ///
    /// # Safety
    ///
    /// All object pointer / block arguments must be valid.
    pub unsafe fn load_model_with_reply(
        &self,
        model: *mut c_void,
        sandbox_ext: *mut c_void,
        options: *mut c_void,
        qos: u32,
        reply: *mut c_void,
    ) {
        type Fn6 = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            u32,
            *mut c_void,
        );
        let sel =
            unsafe { sel_registerName(sel!("loadModel:sandboxExtension:options:qos:withReply:")) };
        let f: Fn6 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, model, sandbox_ext, options, qos, reply) };
    }

    /// `-loadModelNewInstance:options:modelInstParams:qos:withReply:`
    ///
    /// # Safety
    ///
    /// All object pointer / block arguments must be valid.
    pub unsafe fn load_model_new_instance_with_reply(
        &self,
        model: *mut c_void,
        options: *mut c_void,
        model_inst_params: *mut c_void,
        qos: u32,
        reply: *mut c_void,
    ) {
        type Fn6 = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            u32,
            *mut c_void,
        );
        let sel = unsafe {
            sel_registerName(sel!(
                "loadModelNewInstance:options:modelInstParams:qos:withReply:"
            ))
        };
        let f: Fn6 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, model, options, model_inst_params, qos, reply) };
    }

    /// `-unloadModel:options:qos:withReply:`
    ///
    /// # Safety
    ///
    /// All object pointer / block arguments must be valid.
    pub unsafe fn unload_model_with_reply(
        &self,
        model: *mut c_void,
        options: *mut c_void,
        qos: u32,
        reply: *mut c_void,
    ) {
        type Fn5 = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            u32,
            *mut c_void,
        );
        let sel = unsafe { sel_registerName(sel!("unloadModel:options:qos:withReply:")) };
        let f: Fn5 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, model, options, qos, reply) };
    }

    /// `-compiledModelExistsFor:withReply:`
    ///
    /// # Safety
    ///
    /// `model` and `reply` must be valid ObjC object / block pointers.
    pub unsafe fn compiled_model_exists_for_with_reply(
        &self,
        model: *mut c_void,
        reply: *mut c_void,
    ) {
        type Fn3 = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void);
        let sel = unsafe { sel_registerName(sel!("compiledModelExistsFor:withReply:")) };
        let f: Fn3 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, model, reply) };
    }

    /// `-compiledModelExistsMatchingHash:withReply:`
    ///
    /// # Safety
    ///
    /// `hash` and `reply` must be valid ObjC object / block pointers.
    pub unsafe fn compiled_model_exists_matching_hash_with_reply(
        &self,
        hash: *mut c_void,
        reply: *mut c_void,
    ) {
        type Fn3 = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void);
        let sel = unsafe { sel_registerName(sel!("compiledModelExistsMatchingHash:withReply:")) };
        let f: Fn3 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, hash, reply) };
    }

    /// `-purgeCompiledModel:withReply:`
    ///
    /// # Safety
    ///
    /// `model` and `reply` must be valid ObjC object / block pointers.
    pub unsafe fn purge_compiled_model_with_reply(&self, model: *mut c_void, reply: *mut c_void) {
        type Fn3 = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void);
        let sel = unsafe { sel_registerName(sel!("purgeCompiledModel:withReply:")) };
        let f: Fn3 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, model, reply) };
    }

    /// `-purgeCompiledModelMatchingHash:withReply:`
    ///
    /// # Safety
    ///
    /// `hash` and `reply` must be valid ObjC object / block pointers.
    pub unsafe fn purge_compiled_model_matching_hash_with_reply(
        &self,
        hash: *mut c_void,
        reply: *mut c_void,
    ) {
        type Fn3 = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void);
        let sel = unsafe { sel_registerName(sel!("purgeCompiledModelMatchingHash:withReply:")) };
        let f: Fn3 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, hash, reply) };
    }

    /// `-prepareChainingWithModel:options:chainingReq:qos:withReply:`
    ///
    /// # Safety
    ///
    /// All object pointer / block arguments must be valid.
    pub unsafe fn prepare_chaining_with_reply(
        &self,
        model: *mut c_void,
        options: *mut c_void,
        chaining_req: *mut c_void,
        qos: u32,
        reply: *mut c_void,
    ) {
        type Fn6 = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            u32,
            *mut c_void,
        );
        let sel = unsafe {
            sel_registerName(sel!(
                "prepareChainingWithModel:options:chainingReq:qos:withReply:"
            ))
        };
        let f: Fn6 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, model, options, chaining_req, qos, reply) };
    }

    /// `-reportTelemetryToPPS:playload:` (void, note: Apple typo in selector)
    ///
    /// # Safety
    ///
    /// `pps` and `payload` must be valid ObjC object pointers.
    pub unsafe fn report_telemetry(&self, pps: *mut c_void, payload: *mut c_void) {
        type Fn3 = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void);
        let sel = unsafe { sel_registerName(sel!("reportTelemetryToPPS:playload:")) };
        let f: Fn3 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, pps, payload) };
    }
}

impl Drop for DaemonConnection {
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
    fn client_shared_connection_does_not_panic() {
        // May fail if framework not present; should not panic.
        let _ = Client::shared_connection();
    }

    #[test]
    fn daemon_connection_does_not_panic() {
        let _ = DaemonConnection::new();
    }
}
