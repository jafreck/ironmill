//! Safe wrappers for `_ANEIOSurfaceObject` and `_ANEBuffer`.
//!
//! These types bridge IOSurface handles into the ANE runtime's buffer
//! management layer.  Each wrapper owns a retained ObjC object and
//! releases it on drop.

use std::ffi::c_void;

use crate::error::AneSysError;
use crate::objc::{CFRelease, get_class, objc_msgSend, objc_retain, sel, sel_registerName};

// ───────────────────────────────────────────────────────────────────
// AneIOSurfaceObject
// ───────────────────────────────────────────────────────────────────

/// Safe wrapper around `_ANEIOSurfaceObject`.
///
/// Each instance owns a retained ObjC object handle that is released
/// on drop via `CFRelease`.
pub struct AneIOSurfaceObject {
    raw: *mut c_void,
}

unsafe impl Send for AneIOSurfaceObject {}

impl Drop for AneIOSurfaceObject {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: CFRelease on a retained ObjC object.
            unsafe {
                CFRelease(self.raw);
            }
            self.raw = std::ptr::null_mut();
        }
    }
}

#[cfg(target_os = "macos")]
impl AneIOSurfaceObject {
    /// Return the raw ObjC object pointer.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    /// `+[_ANEIOSurfaceObject objectWithIOSurface:]`
    ///
    /// Creates an object from a raw `IOSurfaceRef`.  The surface is retained.
    ///
    /// # Safety
    ///
    /// `iosurface` must be a valid `IOSurfaceRef` pointer.
    pub unsafe fn object_with_iosurface(iosurface: *mut c_void) -> Result<Self, AneSysError> {
        let cls = get_class("_ANEIOSurfaceObject")?;

        type MsgFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("objectWithIOSurface:")) };
        let f: MsgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, sel, iosurface) };
        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "objectWithIOSurface: returned null".into(),
            });
        }
        // Retain the autoreleased object so we own it.
        type RetainFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let retain_sel = unsafe { sel_registerName(sel!("retain")) };
        let retain: RetainFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { retain(obj, retain_sel) };

        Ok(Self { raw: obj })
    }

    /// `+[_ANEIOSurfaceObject objectWithIOSurface:startOffset:]`
    ///
    /// Creates an object from a raw `IOSurfaceRef` with an NSNumber offset.
    ///
    /// # Safety
    ///
    /// `iosurface` must be a valid `IOSurfaceRef` and `start_offset` a valid
    /// `NSNumber` pointer.
    pub unsafe fn object_with_iosurface_offset(
        iosurface: *mut c_void,
        start_offset: *mut c_void,
    ) -> Result<Self, AneSysError> {
        let cls = get_class("_ANEIOSurfaceObject")?;

        type MsgFn =
            unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("objectWithIOSurface:startOffset:")) };
        let f: MsgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, sel, iosurface, start_offset) };
        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "objectWithIOSurface:startOffset: returned null".into(),
            });
        }
        type RetainFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let retain_sel = unsafe { sel_registerName(sel!("retain")) };
        let retain: RetainFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { retain(obj, retain_sel) };

        Ok(Self { raw: obj })
    }

    /// `+[_ANEIOSurfaceObject objectWithIOSurfaceNoRetain:startOffset:]`
    ///
    /// No-retain variant — the surface is NOT retained by the object.
    ///
    /// # Safety
    ///
    /// `iosurface` must be a valid `IOSurfaceRef` and `start_offset` a valid
    /// `NSNumber` pointer.  The caller must ensure the surface outlives this
    /// object since it is not retained.
    pub unsafe fn object_with_iosurface_no_retain(
        iosurface: *mut c_void,
        start_offset: *mut c_void,
    ) -> Result<Self, AneSysError> {
        let cls = get_class("_ANEIOSurfaceObject")?;

        type MsgFn =
            unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("objectWithIOSurfaceNoRetain:startOffset:")) };
        let f: MsgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, sel, iosurface, start_offset) };
        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "objectWithIOSurfaceNoRetain:startOffset: returned null".into(),
            });
        }
        type RetainFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let retain_sel = unsafe { sel_registerName(sel!("retain")) };
        let retain: RetainFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { retain(obj, retain_sel) };

        Ok(Self { raw: obj })
    }

    /// `+[_ANEIOSurfaceObject createIOSurfaceWithWidth:pixel_size:height:]`
    ///
    /// Creates a new IOSurface.  Returns a raw `IOSurfaceRef` pointer, NOT
    /// an `_ANEIOSurfaceObject`.
    pub fn create_iosurface(
        width: i32,
        pixel_size: i32,
        height: i32,
    ) -> Result<*mut c_void, AneSysError> {
        let cls = get_class("_ANEIOSurfaceObject")?;

        type MsgFn = unsafe extern "C" fn(*mut c_void, *mut c_void, i32, i32, i32) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("createIOSurfaceWithWidth:pixel_size:height:")) };
        let f: MsgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let surface = unsafe { f(cls, sel, width, pixel_size, height) };
        if surface.is_null() {
            return Err(AneSysError::IOSurfaceMappingFailed(
                "createIOSurfaceWithWidth:pixel_size:height: returned null".into(),
            ));
        }
        Ok(surface)
    }

    /// `+[_ANEIOSurfaceObject createIOSurfaceWithWidth:pixel_size:height:bytesPerElement:]`
    ///
    /// Creates a new IOSurface with an explicit bytes-per-element.
    /// Returns a raw `IOSurfaceRef` pointer.
    pub fn create_iosurface_bpe(
        width: i32,
        pixel_size: i32,
        height: i32,
        bytes_per_element: i32,
    ) -> Result<*mut c_void, AneSysError> {
        let cls = get_class("_ANEIOSurfaceObject")?;

        type MsgFn =
            unsafe extern "C" fn(*mut c_void, *mut c_void, i32, i32, i32, i32) -> *mut c_void;
        let sel = unsafe {
            sel_registerName(sel!(
                "createIOSurfaceWithWidth:pixel_size:height:bytesPerElement:"
            ))
        };
        let f: MsgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let surface = unsafe { f(cls, sel, width, pixel_size, height, bytes_per_element) };
        if surface.is_null() {
            return Err(AneSysError::IOSurfaceMappingFailed(
                "createIOSurfaceWithWidth:pixel_size:height:bytesPerElement: returned null".into(),
            ));
        }
        Ok(surface)
    }

    /// `-[_ANEIOSurfaceObject ioSurface]`
    ///
    /// Returns the raw `IOSurfaceRef` (`^{__IOSurface=}`) held by this object.
    pub fn iosurface(&self) -> *mut c_void {
        type MsgFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("ioSurface")) };
        let f: MsgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// `-[_ANEIOSurfaceObject startOffset]`
    ///
    /// Returns the `NSNumber` start offset (raw ObjC pointer).
    pub fn start_offset(&self) -> *mut c_void {
        type MsgFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("startOffset")) };
        let f: MsgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// `-[_ANEIOSurfaceObject initWithIOSurface:startOffset:shouldRetain:]`
    ///
    /// Low-level init.  Callers normally use the `objectWith*` factory methods.
    ///
    /// # Safety
    ///
    /// `iosurface` must be a valid `IOSurfaceRef` and `start_offset` a valid
    /// `NSNumber` pointer.
    pub unsafe fn init_with_iosurface(
        iosurface: *mut c_void,
        start_offset: *mut c_void,
        should_retain: bool,
    ) -> Result<Self, AneSysError> {
        let cls = get_class("_ANEIOSurfaceObject")?;

        // +alloc
        type AllocFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let alloc_sel = unsafe { sel_registerName(sel!("alloc")) };
        let alloc_fn: AllocFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let raw = unsafe { alloc_fn(cls, alloc_sel) };
        if raw.is_null() {
            return Err(AneSysError::NullPointer {
                context: "_ANEIOSurfaceObject alloc failed".into(),
            });
        }

        // -initWithIOSurface:startOffset:shouldRetain:
        // Type encoding: (@36@0:8^{__IOSurface=}16@24B32)
        type InitFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            i8,
        ) -> *mut c_void;
        let init_sel =
            unsafe { sel_registerName(sel!("initWithIOSurface:startOffset:shouldRetain:")) };
        let init_fn: InitFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let retain_flag: i8 = if should_retain { 1 } else { 0 };
        let obj = unsafe { init_fn(raw, init_sel, iosurface, start_offset, retain_flag) };
        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "initWithIOSurface:startOffset:shouldRetain: returned null".into(),
            });
        }
        Ok(Self { raw: obj })
    }
}

impl std::fmt::Debug for AneIOSurfaceObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AneIOSurfaceObject")
            .field("raw", &self.raw)
            .finish()
    }
}

// ───────────────────────────────────────────────────────────────────
// AneBuffer
// ───────────────────────────────────────────────────────────────────

/// Safe wrapper around `_ANEBuffer`.
///
/// Each instance owns a retained ObjC object handle that is released
/// on drop via `CFRelease`.
pub struct AneBuffer {
    raw: *mut c_void,
}

unsafe impl Send for AneBuffer {}

impl Drop for AneBuffer {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: CFRelease on a retained ObjC object.
            unsafe {
                CFRelease(self.raw);
            }
            self.raw = std::ptr::null_mut();
        }
    }
}

#[cfg(target_os = "macos")]
impl AneBuffer {
    /// Return the raw ObjC object pointer.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    /// `+[_ANEBuffer bufferWithIOSurfaceObject:symbolIndex:source:]`
    ///
    /// Factory method.  `io_surface_object` and `symbol_index` are ObjC object
    /// pointers (`_ANEIOSurfaceObject` and `NSNumber` respectively).
    ///
    /// # Safety
    ///
    /// `io_surface_object` must be a valid `_ANEIOSurfaceObject` pointer and
    /// `symbol_index` a valid `NSNumber` pointer.
    pub unsafe fn buffer_with_iosurface_object(
        io_surface_object: *mut c_void,
        symbol_index: *mut c_void,
        source: i64,
    ) -> Result<Self, AneSysError> {
        let cls = get_class("_ANEBuffer")?;

        // Type encoding: (@40@0:8@16@24q32)
        type MsgFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            i64,
        ) -> *mut c_void;
        let sel =
            unsafe { sel_registerName(sel!("bufferWithIOSurfaceObject:symbolIndex:source:")) };
        let f: MsgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, sel, io_surface_object, symbol_index, source) };
        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "bufferWithIOSurfaceObject:symbolIndex:source: returned null".into(),
            });
        }
        // Retain the autoreleased object so we own it.
        type RetainFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let retain_sel = unsafe { sel_registerName(sel!("retain")) };
        let retain: RetainFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { retain(obj, retain_sel) };

        Ok(Self { raw: obj })
    }

    /// `-[_ANEBuffer ioSurfaceObject]`
    ///
    /// Returns the `_ANEIOSurfaceObject` pointer (raw ObjC object).
    pub fn iosurface_object(&self) -> *mut c_void {
        type MsgFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("ioSurfaceObject")) };
        let f: MsgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// `-[_ANEBuffer symbolIndex]`
    ///
    /// Returns the `NSNumber` symbol index (raw ObjC pointer).
    pub fn symbol_index(&self) -> *mut c_void {
        type MsgFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("symbolIndex")) };
        let f: MsgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// `-[_ANEBuffer source]`
    ///
    /// Returns the source value as `i64` (type encoding `q`).
    pub fn source(&self) -> i64 {
        type MsgFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i64;
        let sel = unsafe { sel_registerName(sel!("source")) };
        let f: MsgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }
}

impl std::fmt::Debug for AneBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AneBuffer").field("raw", &self.raw).finish()
    }
}

// ───────────────────────────────────────────────────────────────────
// IOSurfaceOutputSets
// ───────────────────────────────────────────────────────────────────

/// Safe wrapper around `_ANEIOSurfaceOutputSets`.
///
/// Pairs a stats `IOSurfaceRef` with an `NSArray` of output buffers.
/// Each instance owns a retained ObjC object handle that is released
/// on drop via `CFRelease`.
pub struct IOSurfaceOutputSets {
    raw: *mut c_void,
}

unsafe impl Send for IOSurfaceOutputSets {}

impl Drop for IOSurfaceOutputSets {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                CFRelease(self.raw);
            }
            self.raw = std::ptr::null_mut();
        }
    }
}

#[cfg(target_os = "macos")]
impl IOSurfaceOutputSets {
    /// Return the raw ObjC object pointer.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    /// `+[_ANEIOSurfaceOutputSets objectWithstatsSurRef:outputBuffer:]`
    ///
    /// Factory — creates an output-sets object from a raw `IOSurfaceRef` and
    /// an `NSArray` of output buffers.
    ///
    /// # Safety
    ///
    /// `stats_sur_ref` must be a valid `IOSurfaceRef` and `output_buffer` a
    /// valid `NSArray` pointer.
    pub unsafe fn object_with_stats_sur_ref(
        stats_sur_ref: *mut c_void,
        output_buffer: *mut c_void,
    ) -> Result<Self, AneSysError> {
        let cls = get_class("_ANEIOSurfaceOutputSets")?;

        type MsgFn =
            unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("objectWithstatsSurRef:outputBuffer:")) };
        let f: MsgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, sel, stats_sur_ref, output_buffer) };
        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "objectWithstatsSurRef:outputBuffer: returned null".into(),
            });
        }
        objc_retain(obj);

        Ok(Self { raw: obj })
    }

    /// `-[_ANEIOSurfaceOutputSets statsSurRef]`
    ///
    /// Returns the raw `IOSurfaceRef` (`^{__IOSurface=}`).
    pub fn stats_sur_ref(&self) -> *mut c_void {
        type MsgFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("statsSurRef")) };
        let f: MsgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// `-[_ANEIOSurfaceOutputSets outputBuffer]`
    ///
    /// Returns the `NSArray` of output buffers (raw ObjC pointer).
    pub fn output_buffer(&self) -> *mut c_void {
        type MsgFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("outputBuffer")) };
        let f: MsgFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }
}

impl std::fmt::Debug for IOSurfaceOutputSets {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IOSurfaceOutputSets")
            .field("raw", &self.raw)
            .finish()
    }
}
