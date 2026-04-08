//! Safe wrappers for ANE shared event synchronization primitives.
//!
//! Wraps `_ANESharedEvents`, `_ANESharedSignalEvent`, and
//! `_ANESharedWaitEvent` — the ObjC classes that describe signal/wait
//! fences submitted alongside ANE evaluation requests.

#[cfg(target_os = "macos")]
use std::ffi::c_void;

#[cfg(target_os = "macos")]
use crate::error::AneSysError;
#[cfg(target_os = "macos")]
use crate::objc::{get_class, objc_msgSend, objc_retain, safe_release, sel, sel_registerName};

// ---------------------------------------------------------------------------
// SharedEvents
// ---------------------------------------------------------------------------

/// Container for signal + wait event arrays (`_ANESharedEvents`).
#[cfg(target_os = "macos")]
pub struct SharedEvents {
    raw: *mut c_void,
}

// SAFETY: The raw ObjC handle is only accessed through &self methods and
// ownership is exclusive (no aliasing).
#[cfg(target_os = "macos")]
unsafe impl Send for SharedEvents {}

#[cfg(target_os = "macos")]
impl std::fmt::Debug for SharedEvents {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedEvents")
            .field("raw", &self.raw)
            .finish()
    }
}

#[cfg(target_os = "macos")]
impl SharedEvents {
    /// Create via `+[_ANESharedEvents sharedEventsWithSignalEvents:waitEvents:]`.
    ///
    /// Both arguments are `NSArray` pointers of signal/wait event objects.
    ///
    /// # Safety
    ///
    /// `signal_events` and `wait_events` must be valid `NSArray` pointers (or null).
    pub unsafe fn new(
        signal_events: *mut c_void,
        wait_events: *mut c_void,
    ) -> Result<Self, AneSysError> {
        let cls = get_class("_ANESharedEvents")?;

        type FactoryFn =
            unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void) -> *mut c_void;

        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("sharedEventsWithSignalEvents:waitEvents:")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: FactoryFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: cls is a valid class; signal_events and wait_events are NSArrays.
        let obj = unsafe { f(cls, s, signal_events, wait_events) };

        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "sharedEventsWithSignalEvents:waitEvents: returned nil".into(),
            });
        }

        // Retain — factory returns autoreleased.
        objc_retain(obj);
        Ok(Self { raw: obj })
    }

    /// Raw `_ANESharedEvents` pointer.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    /// Get the signal events array (`-signalEvents`).
    pub fn signal_events(&self) -> *mut c_void {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("signalEvents")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANESharedEvents object.
        unsafe { f(self.raw, s) }
    }

    /// Set the signal events array (`-setSignalEvents:`).
    ///
    /// # Safety
    ///
    /// `events` must be a valid `NSArray` pointer (or null).
    pub unsafe fn set_signal_events(&self, events: *mut c_void) {
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void);
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("setSignalEvents:")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: SetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANESharedEvents; events is a valid NSArray.
        unsafe { f(self.raw, s, events) };
    }

    /// Get the wait events array (`-waitEvents`).
    pub fn wait_events(&self) -> *mut c_void {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("waitEvents")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANESharedEvents object.
        unsafe { f(self.raw, s) }
    }

    /// Set the wait events array (`-setWaitEvents:`).
    ///
    /// # Safety
    ///
    /// `events` must be a valid `NSArray` pointer (or null).
    pub unsafe fn set_wait_events(&self, events: *mut c_void) {
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void);
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("setWaitEvents:")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: SetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANESharedEvents; events is a valid NSArray.
        unsafe { f(self.raw, s, events) };
    }
}

#[cfg(target_os = "macos")]
impl Drop for SharedEvents {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // Exception-safe release to avoid ObjC exception aborts.
            safe_release(self.raw);
            self.raw = std::ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// SharedSignalEvent
// ---------------------------------------------------------------------------

/// A signal event submitted with an ANE request (`_ANESharedSignalEvent`).
#[cfg(target_os = "macos")]
pub struct SharedSignalEvent {
    raw: *mut c_void,
}

// SAFETY: The raw ObjC handle is only accessed through &self methods and
// ownership is exclusive (no aliasing).
#[cfg(target_os = "macos")]
unsafe impl Send for SharedSignalEvent {}

#[cfg(target_os = "macos")]
impl std::fmt::Debug for SharedSignalEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedSignalEvent")
            .field("raw", &self.raw)
            .finish()
    }
}

#[cfg(target_os = "macos")]
impl SharedSignalEvent {
    /// Create via `+[_ANESharedSignalEvent signalEventWithValue:symbolIndex:eventType:sharedEvent:]`.
    ///
    /// # Safety
    ///
    /// `shared_event` must be a valid `IOSurfaceSharedEvent` pointer.
    pub unsafe fn new(
        value: u64,
        symbol_index: u32,
        event_type: i64,
        shared_event: *mut c_void,
    ) -> Result<Self, AneSysError> {
        let cls = get_class("_ANESharedSignalEvent")?;

        type FactoryFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            u64,
            u32,
            i64,
            *mut c_void,
        ) -> *mut c_void;

        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe {
            sel_registerName(sel!(
                "signalEventWithValue:symbolIndex:eventType:sharedEvent:"
            ))
        };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: FactoryFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: cls is valid; arguments match the ObjC type encoding.
        let obj = unsafe { f(cls, s, value, symbol_index, event_type, shared_event) };

        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "signalEventWithValue:symbolIndex:eventType:sharedEvent: returned nil"
                    .into(),
            });
        }

        // Retain — factory returns autoreleased.
        objc_retain(obj);
        Ok(Self { raw: obj })
    }

    /// Raw `_ANESharedSignalEvent` pointer.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    /// Get the signal value (`-value`).
    pub fn value(&self) -> u64 {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("value")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANESharedSignalEvent.
        unsafe { f(self.raw, s) }
    }

    /// Set the signal value (`-setValue:`).
    pub fn set_value(&self, value: u64) {
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, u64);
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("setValue:")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: SetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANESharedSignalEvent.
        unsafe { f(self.raw, s, value) };
    }

    /// Get the symbol index (`-symbolIndex`).
    pub fn symbol_index(&self) -> u32 {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("symbolIndex")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANESharedSignalEvent.
        unsafe { f(self.raw, s) }
    }

    /// Get the event type (`-eventType`).
    pub fn event_type(&self) -> i64 {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i64;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("eventType")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANESharedSignalEvent.
        unsafe { f(self.raw, s) }
    }

    /// Get the IOSurfaceSharedEvent handle (`-sharedEvent`).
    pub fn shared_event(&self) -> *mut c_void {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("sharedEvent")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANESharedSignalEvent.
        unsafe { f(self.raw, s) }
    }

    /// Get the agent mask (`-agentMask`).
    pub fn agent_mask(&self) -> u64 {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("agentMask")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANESharedSignalEvent.
        unsafe { f(self.raw, s) }
    }

    /// Set the agent mask (`-setAgentMask:`).
    pub fn set_agent_mask(&self, mask: u64) {
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, u64);
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("setAgentMask:")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: SetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANESharedSignalEvent.
        unsafe { f(self.raw, s, mask) };
    }

    /// Get the associated wait event (`-waitEvent`).
    pub fn wait_event(&self) -> *mut c_void {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("waitEvent")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANESharedSignalEvent.
        unsafe { f(self.raw, s) }
    }
}

#[cfg(target_os = "macos")]
impl Drop for SharedSignalEvent {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // Exception-safe release to avoid ObjC exception aborts.
            safe_release(self.raw);
            self.raw = std::ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// SharedWaitEvent
// ---------------------------------------------------------------------------

/// A wait event submitted with an ANE request (`_ANESharedWaitEvent`).
#[cfg(target_os = "macos")]
pub struct SharedWaitEvent {
    raw: *mut c_void,
}

// SAFETY: The raw ObjC handle is only accessed through &self methods and
// ownership is exclusive (no aliasing).
#[cfg(target_os = "macos")]
unsafe impl Send for SharedWaitEvent {}

#[cfg(target_os = "macos")]
impl std::fmt::Debug for SharedWaitEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedWaitEvent")
            .field("raw", &self.raw)
            .finish()
    }
}

#[cfg(target_os = "macos")]
impl SharedWaitEvent {
    /// Create via `+[_ANESharedWaitEvent waitEventWithValue:sharedEvent:]`.
    ///
    /// # Safety
    ///
    /// `shared_event` must be a valid `IOSurfaceSharedEvent` pointer.
    pub unsafe fn new(value: u64, shared_event: *mut c_void) -> Result<Self, AneSysError> {
        let cls = get_class("_ANESharedWaitEvent")?;

        type FactoryFn =
            unsafe extern "C" fn(*mut c_void, *mut c_void, u64, *mut c_void) -> *mut c_void;

        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("waitEventWithValue:sharedEvent:")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: FactoryFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: cls is valid; value and shared_event match the type encoding.
        let obj = unsafe { f(cls, s, value, shared_event) };

        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "waitEventWithValue:sharedEvent: returned nil".into(),
            });
        }

        // Retain — factory returns autoreleased.
        objc_retain(obj);
        Ok(Self { raw: obj })
    }

    /// Create via `+[_ANESharedWaitEvent waitEventWithValue:sharedEvent:eventType:]`.
    ///
    /// # Safety
    ///
    /// `shared_event` must be a valid `IOSurfaceSharedEvent` pointer.
    pub unsafe fn with_event_type(
        value: u64,
        shared_event: *mut c_void,
        event_type: u64,
    ) -> Result<Self, AneSysError> {
        let cls = get_class("_ANESharedWaitEvent")?;

        type FactoryFn =
            unsafe extern "C" fn(*mut c_void, *mut c_void, u64, *mut c_void, u64) -> *mut c_void;

        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("waitEventWithValue:sharedEvent:eventType:")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: FactoryFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: cls is valid; arguments match the type encoding.
        let obj = unsafe { f(cls, s, value, shared_event, event_type) };

        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "waitEventWithValue:sharedEvent:eventType: returned nil".into(),
            });
        }

        // Retain — factory returns autoreleased.
        objc_retain(obj);
        Ok(Self { raw: obj })
    }

    /// Raw `_ANESharedWaitEvent` pointer.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    /// Get the wait value (`-value`).
    pub fn value(&self) -> u64 {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("value")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANESharedWaitEvent.
        unsafe { f(self.raw, s) }
    }

    /// Set the wait value (`-setValue:`).
    pub fn set_value(&self, value: u64) {
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, u64);
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("setValue:")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: SetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANESharedWaitEvent.
        unsafe { f(self.raw, s, value) };
    }

    /// Get the IOSurfaceSharedEvent handle (`-sharedEvent`).
    pub fn shared_event(&self) -> *mut c_void {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("sharedEvent")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANESharedWaitEvent.
        unsafe { f(self.raw, s) }
    }

    /// Get the event type (`-eventType`).
    pub fn event_type(&self) -> u64 {
        type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64;
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let s = unsafe { sel_registerName(sel!("eventType")) };
        // SAFETY: transmute objc_msgSend to the correct signature.
        let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: self.raw is a valid _ANESharedWaitEvent.
        unsafe { f(self.raw, s) }
    }
}

#[cfg(target_os = "macos")]
impl Drop for SharedWaitEvent {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // Exception-safe release to avoid ObjC exception aborts.
            safe_release(self.raw);
            self.raw = std::ptr::null_mut();
        }
    }
}
