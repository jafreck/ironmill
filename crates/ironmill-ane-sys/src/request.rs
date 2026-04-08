//! Safe wrappers for ANE request objects.
//!
//! Wraps `_ANERequest` (standard evaluation requests with I/O surfaces)
//! and `_ANEChainingRequest` (pipelined loopback requests for chained
//! inference with signal events and loopback buffers).

#[cfg(target_os = "macos")]
use std::ffi::c_void;

#[cfg(target_os = "macos")]
use crate::error::AneSysError;
#[cfg(target_os = "macos")]
use crate::objc::{
    create_nsarray, create_nsnumber_i64, get_class, objc_msgSend, objc_retain, safe_release, sel,
    sel_registerName,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build an `NSArray` of `NSNumber` from a slice of `i64` values.
///
/// # Safety
///
/// Must be called from a thread with an active autorelease pool.
#[cfg(target_os = "macos")]
unsafe fn build_nsnumber_array(values: &[i64]) -> Result<*mut c_void, AneSysError> {
    let mut nums = Vec::with_capacity(values.len());
    for &v in values {
        // SAFETY: create_nsnumber_i64 returns an autoreleased NSNumber.
        let n = unsafe { create_nsnumber_i64(v) };
        if n.is_null() {
            return Err(AneSysError::NullPointer {
                context: "failed to create NSNumber for index".into(),
            });
        }
        nums.push(n);
    }
    // SAFETY: all pointers in nums are valid, non-null NSNumber objects.
    let arr = unsafe { create_nsarray(&nums) };
    if arr.is_null() {
        return Err(AneSysError::NullPointer {
            context: "failed to create NSArray from index slice".into(),
        });
    }
    Ok(arr)
}

// ---------------------------------------------------------------------------
// AneRequest
// ---------------------------------------------------------------------------

/// Wrapper for `_ANERequest` — describes inputs, outputs, and metadata
/// for a single ANE evaluation.
#[cfg(target_os = "macos")]
pub struct AneRequest {
    raw: *mut c_void,
}

#[cfg(target_os = "macos")]
unsafe impl Send for AneRequest {}

#[cfg(target_os = "macos")]
impl std::fmt::Debug for AneRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AneRequest")
            .field("raw", &self.raw)
            .finish()
    }
}

#[cfg(target_os = "macos")]
impl AneRequest {
    /// Create via `+[_ANERequest requestWithInputs:inputIndices:outputs:
    /// outputIndices:weightsBuffer:perfStats:procedureIndex:]`.
    ///
    /// `inputs` / `outputs` are slices of `_ANEIOSurfaceObject` pointers;
    /// `input_indices` / `output_indices` are converted to `NSNumber` arrays
    /// internally.
    ///
    /// # Safety
    ///
    /// Every pointer in `inputs` and `outputs` must be a valid
    /// `_ANEIOSurfaceObject`.  `weights_buffer` and `perf_stats` must be
    /// valid ObjC object pointers or null.
    pub unsafe fn new(
        inputs: &[*mut c_void],
        input_indices: &[i64],
        outputs: &[*mut c_void],
        output_indices: &[i64],
        weights_buffer: *mut c_void,
        perf_stats: *mut c_void,
        procedure_index: i64,
    ) -> Result<Self, AneSysError> {
        let cls = get_class("_ANERequest")?;

        // SAFETY: all pointers in inputs are valid per caller contract.
        let in_arr = unsafe { create_nsarray(inputs) };
        // SAFETY: build_nsnumber_array creates autoreleased NSNumbers.
        let in_idx = unsafe { build_nsnumber_array(input_indices)? };
        // SAFETY: all pointers in outputs are valid per caller contract.
        let out_arr = unsafe { create_nsarray(outputs) };
        // SAFETY: build_nsnumber_array creates autoreleased NSNumbers.
        let out_idx = unsafe { build_nsnumber_array(output_indices)? };
        // SAFETY: create_nsnumber_i64 returns an autoreleased NSNumber.
        let proc_idx = unsafe { create_nsnumber_i64(procedure_index) };

        type FactoryFn = unsafe extern "C" fn(
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
        let s = unsafe {
            sel_registerName(sel!(
                "requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"
            ))
        };
        // SAFETY: transmute objc_msgSend to the correct 9-arg signature.
        let f: FactoryFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: cls is a valid class; all args are valid ObjC objects or nil.
        let obj = unsafe {
            f(
                cls,
                s,
                in_arr,
                in_idx,
                out_arr,
                out_idx,
                weights_buffer,
                perf_stats,
                proc_idx,
            )
        };

        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "requestWithInputs:...:procedureIndex: returned nil".into(),
            });
        }

        // Retain — factory returns autoreleased.
        objc_retain(obj);
        Ok(Self { raw: obj })
    }

    /// Create via `+[_ANERequest requestWithInputs:inputIndices:outputs:
    /// outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:]`.
    ///
    /// # Safety
    ///
    /// Every pointer in `inputs` and `outputs` must be a valid
    /// `_ANEIOSurfaceObject`.  `weights_buffer`, `perf_stats`, and
    /// `shared_events` must be valid ObjC object pointers or null.
    pub unsafe fn with_shared_events(
        inputs: &[*mut c_void],
        input_indices: &[i64],
        outputs: &[*mut c_void],
        output_indices: &[i64],
        weights_buffer: *mut c_void,
        perf_stats: *mut c_void,
        procedure_index: i64,
        shared_events: *mut c_void,
    ) -> Result<Self, AneSysError> {
        let cls = get_class("_ANERequest")?;

        // SAFETY: all pointers in inputs are valid per caller contract.
        let in_arr = unsafe { create_nsarray(inputs) };
        // SAFETY: build_nsnumber_array creates autoreleased NSNumbers.
        let in_idx = unsafe { build_nsnumber_array(input_indices)? };
        // SAFETY: all pointers in outputs are valid per caller contract.
        let out_arr = unsafe { create_nsarray(outputs) };
        // SAFETY: build_nsnumber_array creates autoreleased NSNumbers.
        let out_idx = unsafe { build_nsnumber_array(output_indices)? };
        // SAFETY: create_nsnumber_i64 returns an autoreleased NSNumber.
        let proc_idx = unsafe { create_nsnumber_i64(procedure_index) };

        type FactoryFn = unsafe extern "C" fn(
            *mut c_void,
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
        let s = unsafe {
            sel_registerName(sel!(
                "requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:"
            ))
        };
        // SAFETY: transmute objc_msgSend to the correct 10-arg signature.
        let f: FactoryFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: cls is a valid class; all args are valid ObjC objects or nil.
        let obj = unsafe {
            f(
                cls,
                s,
                in_arr,
                in_idx,
                out_arr,
                out_idx,
                weights_buffer,
                perf_stats,
                proc_idx,
                shared_events,
            )
        };

        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "requestWithInputs:...:sharedEvents: returned nil".into(),
            });
        }

        // Retain — factory returns autoreleased.
        objc_retain(obj);
        Ok(Self { raw: obj })
    }

    /// Create via `+[_ANERequest requestWithInputs:inputIndices:outputs:
    /// outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:
    /// transactionHandle:]`.
    ///
    /// # Safety
    ///
    /// Every pointer in `inputs` and `outputs` must be a valid
    /// `_ANEIOSurfaceObject`.  `weights_buffer`, `perf_stats`,
    /// `shared_events`, and `transaction_handle` must be valid ObjC object
    /// pointers or null.
    pub unsafe fn with_transaction(
        inputs: &[*mut c_void],
        input_indices: &[i64],
        outputs: &[*mut c_void],
        output_indices: &[i64],
        weights_buffer: *mut c_void,
        perf_stats: *mut c_void,
        procedure_index: i64,
        shared_events: *mut c_void,
        transaction_handle: *mut c_void,
    ) -> Result<Self, AneSysError> {
        let cls = get_class("_ANERequest")?;

        // SAFETY: all pointers in inputs are valid per caller contract.
        let in_arr = unsafe { create_nsarray(inputs) };
        // SAFETY: build_nsnumber_array creates autoreleased NSNumbers.
        let in_idx = unsafe { build_nsnumber_array(input_indices)? };
        // SAFETY: all pointers in outputs are valid per caller contract.
        let out_arr = unsafe { create_nsarray(outputs) };
        // SAFETY: build_nsnumber_array creates autoreleased NSNumbers.
        let out_idx = unsafe { build_nsnumber_array(output_indices)? };
        // SAFETY: create_nsnumber_i64 returns an autoreleased NSNumber.
        let proc_idx = unsafe { create_nsnumber_i64(procedure_index) };

        type FactoryFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
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
        let s = unsafe {
            sel_registerName(sel!(
                "requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:"
            ))
        };
        // SAFETY: transmute objc_msgSend to the correct 11-arg signature.
        let f: FactoryFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: cls is a valid class; all args are valid ObjC objects or nil.
        let obj = unsafe {
            f(
                cls,
                s,
                in_arr,
                in_idx,
                out_arr,
                out_idx,
                weights_buffer,
                perf_stats,
                proc_idx,
                shared_events,
                transaction_handle,
            )
        };

        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "requestWithInputs:...:transactionHandle: returned nil".into(),
            });
        }

        // Retain — factory returns autoreleased.
        objc_retain(obj);
        Ok(Self { raw: obj })
    }

    /// Raw `_ANERequest` pointer.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    /// Validate the request (`-validate`).
    pub fn validate(&self) -> bool {
        type BoolFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i8;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("validate")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: BoolFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANERequest object.
        unsafe { f(self.raw, s) != 0 }
    }

    /// Number of IOSurfaces in the request (`-ioSurfacesCount`).
    pub fn io_surfaces_count(&self) -> u64 {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("ioSurfacesCount")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANERequest object.
        unsafe { f(self.raw, s) }
    }

    /// Get the completion handler block (`-completionHandler`).
    pub fn completion_handler(&self) -> *mut c_void {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("completionHandler")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANERequest object.
        unsafe { f(self.raw, s) }
    }

    /// Set the completion handler block (`-setCompletionHandler:`).
    ///
    /// # Safety
    ///
    /// `handler` must be a valid ObjC block pointer or null.
    pub unsafe fn set_completion_handler(&self, handler: *mut c_void) {
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void);
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("setCompletionHandler:")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: SetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANERequest; handler is valid per caller.
        unsafe { f(self.raw, s, handler) };
    }

    /// Get the performance stats object (`-perfStats`).
    pub fn perf_stats(&self) -> *mut c_void {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("perfStats")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANERequest object.
        unsafe { f(self.raw, s) }
    }

    /// Set the performance stats object (`-setPerfStats:`).
    ///
    /// # Safety
    ///
    /// `stats` must be a valid `_ANEPerformanceStats` pointer or null.
    pub unsafe fn set_perf_stats(&self, stats: *mut c_void) {
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void);
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("setPerfStats:")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: SetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANERequest; stats is valid per caller.
        unsafe { f(self.raw, s, stats) };
    }

    /// Get the shared events object (`-sharedEvents`).
    pub fn shared_events(&self) -> *mut c_void {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("sharedEvents")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANERequest object.
        unsafe { f(self.raw, s) }
    }

    /// Set the shared events object (`-setSharedEvents:`).
    ///
    /// # Safety
    ///
    /// `events` must be a valid `_ANESharedEvents` pointer or null.
    pub unsafe fn set_shared_events(&self, events: *mut c_void) {
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void);
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("setSharedEvents:")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: SetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANERequest; events is valid per caller.
        unsafe { f(self.raw, s, events) };
    }

    /// Get the transaction handle (`-transactionHandle`).
    pub fn transaction_handle(&self) -> *mut c_void {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("transactionHandle")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANERequest object.
        unsafe { f(self.raw, s) }
    }

    /// Set the transaction handle (`-setTransactionHandle:`).
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `NSNumber` pointer or null.
    pub unsafe fn set_transaction_handle(&self, handle: *mut c_void) {
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void);
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("setTransactionHandle:")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: SetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANERequest; handle is valid per caller.
        unsafe { f(self.raw, s, handle) };
    }

    /// Get the procedure index as an `NSNumber` pointer (`-procedureIndex`).
    pub fn procedure_index(&self) -> *mut c_void {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("procedureIndex")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANERequest object.
        unsafe { f(self.raw, s) }
    }

    /// Get the input array (`-inputArray`), returns `NSArray` pointer.
    pub fn input_array(&self) -> *mut c_void {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("inputArray")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANERequest object.
        unsafe { f(self.raw, s) }
    }

    /// Get the output array (`-outputArray`), returns `NSArray` pointer.
    pub fn output_array(&self) -> *mut c_void {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("outputArray")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANERequest object.
        unsafe { f(self.raw, s) }
    }

    /// Get the input index array (`-inputIndexArray`), returns `NSArray` pointer.
    pub fn input_index_array(&self) -> *mut c_void {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("inputIndexArray")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANERequest object.
        unsafe { f(self.raw, s) }
    }

    /// Get the output index array (`-outputIndexArray`), returns `NSArray` pointer.
    pub fn output_index_array(&self) -> *mut c_void {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("outputIndexArray")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANERequest object.
        unsafe { f(self.raw, s) }
    }

    /// Get the weights buffer (`-weightsBuffer`).
    pub fn weights_buffer(&self) -> *mut c_void {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("weightsBuffer")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANERequest object.
        unsafe { f(self.raw, s) }
    }
}

#[cfg(target_os = "macos")]
impl Drop for AneRequest {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // Exception-safe release to avoid ObjC exception aborts.
            safe_release(self.raw);
            self.raw = std::ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// ChainingRequest
// ---------------------------------------------------------------------------

/// Wrapper for `_ANEChainingRequest` — pipelined loopback requests for
/// chained inference with signal events and firmware enqueue delays.
#[cfg(target_os = "macos")]
pub struct ChainingRequest {
    raw: *mut c_void,
}

#[cfg(target_os = "macos")]
unsafe impl Send for ChainingRequest {}

impl std::fmt::Debug for ChainingRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChainingRequest")
            .field("raw", &self.raw)
            .finish()
    }
}

#[cfg(target_os = "macos")]
impl ChainingRequest {
    /// Create via `+[_ANEChainingRequest chainingRequestWithInputs:outputSets:
    /// lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:
    /// transactionHandle:fwEnqueueDelay:memoryPoolId:]`.
    ///
    /// NSArray arguments are passed through directly; numeric arguments are
    /// converted to `NSNumber` internally.
    ///
    /// # Safety
    ///
    /// `inputs`, `output_sets`, `lb_input_symbol_id`, `lb_output_symbol_id`,
    /// and `signal_events` must be valid `NSArray` pointers or null.
    pub unsafe fn new(
        inputs: *mut c_void,
        output_sets: *mut c_void,
        lb_input_symbol_id: *mut c_void,
        lb_output_symbol_id: *mut c_void,
        procedure_index: i64,
        signal_events: *mut c_void,
        transaction_handle: i64,
        fw_enqueue_delay: i64,
        memory_pool_id: i64,
    ) -> Result<Self, AneSysError> {
        let cls = get_class("_ANEChainingRequest")?;

        // SAFETY: create_nsnumber_i64 returns autoreleased NSNumber objects.
        let proc_idx = unsafe { create_nsnumber_i64(procedure_index) };
        let tx_handle = unsafe { create_nsnumber_i64(transaction_handle) };
        let delay = unsafe { create_nsnumber_i64(fw_enqueue_delay) };
        let pool_id = unsafe { create_nsnumber_i64(memory_pool_id) };

        type FactoryFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
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
        let s = unsafe {
            sel_registerName(sel!(
                "chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:"
            ))
        };
        // SAFETY: transmute objc_msgSend to the correct 11-arg signature.
        let f: FactoryFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: cls is a valid class; all args are valid ObjC objects or nil.
        let obj = unsafe {
            f(
                cls,
                s,
                inputs,
                output_sets,
                lb_input_symbol_id,
                lb_output_symbol_id,
                proc_idx,
                signal_events,
                tx_handle,
                delay,
                pool_id,
            )
        };

        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "chainingRequestWithInputs:...:memoryPoolId: returned nil".into(),
            });
        }

        // Retain — factory returns autoreleased.
        objc_retain(obj);
        Ok(Self { raw: obj })
    }

    /// Raw `_ANEChainingRequest` pointer.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    /// Validate the request (`-validate`).
    pub fn validate(&self) -> bool {
        type BoolFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i8;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("validate")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: BoolFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANEChainingRequest object.
        unsafe { f(self.raw, s) != 0 }
    }

    /// Get the input buffer array (`-inputBuffer`), returns `NSArray` pointer.
    pub fn input_buffer(&self) -> *mut c_void {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("inputBuffer")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANEChainingRequest object.
        unsafe { f(self.raw, s) }
    }

    /// Get the output sets array (`-outputSets`), returns `NSArray` pointer.
    pub fn output_sets(&self) -> *mut c_void {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("outputSets")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANEChainingRequest object.
        unsafe { f(self.raw, s) }
    }

    /// Get loopback input symbol indices (`-loopbackInputSymbolIndex`),
    /// returns `NSArray` pointer.
    pub fn loopback_input_symbol_index(&self) -> *mut c_void {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("loopbackInputSymbolIndex")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANEChainingRequest object.
        unsafe { f(self.raw, s) }
    }

    /// Get loopback output symbol indices (`-loopbackOutputSymbolIndex`),
    /// returns `NSArray` pointer.
    pub fn loopback_output_symbol_index(&self) -> *mut c_void {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("loopbackOutputSymbolIndex")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANEChainingRequest object.
        unsafe { f(self.raw, s) }
    }

    /// Get the signal events array (`-signalEvents`), returns `NSArray` pointer.
    pub fn signal_events(&self) -> *mut c_void {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("signalEvents")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANEChainingRequest object.
        unsafe { f(self.raw, s) }
    }

    /// Get the transaction handle as an `NSNumber` pointer (`-transactionHandle`).
    pub fn transaction_handle(&self) -> *mut c_void {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("transactionHandle")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANEChainingRequest object.
        unsafe { f(self.raw, s) }
    }

    /// Get the procedure index as an `NSNumber` pointer (`-procedureIndex`).
    pub fn procedure_index(&self) -> *mut c_void {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("procedureIndex")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANEChainingRequest object.
        unsafe { f(self.raw, s) }
    }

    /// Get the firmware enqueue delay as an `NSNumber` pointer (`-fwEnqueueDelay`).
    pub fn fw_enqueue_delay(&self) -> *mut c_void {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("fwEnqueueDelay")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANEChainingRequest object.
        unsafe { f(self.raw, s) }
    }

    /// Get the memory pool ID as an `NSNumber` pointer (`-memoryPoolId`).
    pub fn memory_pool_id(&self) -> *mut c_void {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("memoryPoolId")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANEChainingRequest object.
        unsafe { f(self.raw, s) }
    }
}

#[cfg(target_os = "macos")]
impl Drop for ChainingRequest {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // Exception-safe release to avoid ObjC exception aborts.
            safe_release(self.raw);
            self.raw = std::ptr::null_mut();
        }
    }
}
