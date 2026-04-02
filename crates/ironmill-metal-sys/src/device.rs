//! Safe wrapper for `MTLDevice`.

use std::ffi::c_void;

use crate::buffer::{MetalBuffer, StorageMode};
use crate::command::CommandQueue;
use crate::error::MetalSysError;
use crate::objc::{self, sel};
use crate::pipeline::ComputePipeline;
use crate::shader::{ShaderFunction, ShaderLibrary};

// ---------------------------------------------------------------------------
// Metal framework link + MTLCreateSystemDefaultDevice
// ---------------------------------------------------------------------------

// SAFETY: These are well-known C FFI declarations from the Metal framework.
#[link(name = "Metal", kind = "framework")]
unsafe extern "C" {
    fn MTLCreateSystemDefaultDevice() -> *mut c_void;
}

// ---------------------------------------------------------------------------
// GpuFamily
// ---------------------------------------------------------------------------

/// GPU family identifiers for capability queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuFamily {
    /// Apple GPU family 7 (M1).
    Apple7,
    /// Apple GPU family 8 (M2).
    Apple8,
    /// Apple GPU family 9 (M3+).
    Apple9,
}

impl GpuFamily {
    /// Returns the Metal `MTLGPUFamily` enum value for this family.
    fn metal_value(self) -> usize {
        match self {
            // MTLGPUFamilyApple7 = 1007
            GpuFamily::Apple7 => 1007,
            // MTLGPUFamilyApple8 = 1008
            GpuFamily::Apple8 => 1008,
            // MTLGPUFamilyApple9 = 1009
            GpuFamily::Apple9 => 1009,
        }
    }
}

// ---------------------------------------------------------------------------
// MetalDevice
// ---------------------------------------------------------------------------

/// Safe wrapper around an `MTLDevice` (id<MTLDevice>).
///
/// Obtained via [`MetalDevice::system_default()`] which calls
/// `MTLCreateSystemDefaultDevice()`.
pub struct MetalDevice {
    raw: *mut c_void,
}

// SAFETY: MTLDevice is thread-safe — Apple documents it as safe to use from
// multiple threads.
unsafe impl Send for MetalDevice {}
unsafe impl Sync for MetalDevice {}

impl MetalDevice {
    /// Create a `MetalDevice` wrapping the system default Metal GPU.
    pub fn system_default() -> Result<Self, MetalSysError> {
        // SAFETY: MTLCreateSystemDefaultDevice is a well-defined C function
        // that returns a retained id<MTLDevice> or nil.
        let raw = unsafe { MTLCreateSystemDefaultDevice() };
        if raw.is_null() {
            return Err(MetalSysError::NoDevice);
        }
        Ok(Self { raw })
    }

    /// Returns the name of this GPU device.
    pub fn name(&self) -> String {
        // SAFETY: `self.raw` is a valid, retained id<MTLDevice> (non-null
        // guaranteed by system_default). "name" returns an autoreleased
        // NSString. transmute casts objc_msgSend to the correct signature.
        type NameFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { objc::sel_registerName(sel!("name")) };
        let f: NameFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        let nsstring = unsafe { f(self.raw, sel) };
        objc::nsstring_to_string(nsstring)
    }

    /// Returns the maximum buffer length in bytes this device supports.
    pub fn max_buffer_length(&self) -> usize {
        // SAFETY: `self.raw` is a valid id<MTLDevice>. "maxBufferLength"
        // returns an NSUInteger property value.
        type LenFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> usize;
        let sel = unsafe { objc::sel_registerName(sel!("maxBufferLength")) };
        let f: LenFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Returns the maximum threadgroup memory length in bytes.
    pub fn max_threadgroup_memory_length(&self) -> usize {
        // SAFETY: `self.raw` is a valid id<MTLDevice>.
        // "maxThreadgroupMemoryLength" returns an NSUInteger property value.
        type LenFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> usize;
        let sel = unsafe { objc::sel_registerName(sel!("maxThreadgroupMemoryLength")) };
        let f: LenFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Returns the current total allocation size (bytes) across all Metal
    /// resources created by this device.
    pub fn current_allocated_size(&self) -> usize {
        // SAFETY: `self.raw` is a valid id<MTLDevice>.
        // "currentAllocatedSize" returns an NSUInteger property value.
        type LenFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> usize;
        let sel = unsafe { objc::sel_registerName(sel!("currentAllocatedSize")) };
        let f: LenFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Returns the maximum threads per threadgroup as (width, height, depth).
    pub fn max_threads_per_threadgroup(&self) -> (usize, usize, usize) {
        // MTLSize is a struct { NSUInteger width, height, depth }.
        // On arm64, small structs are returned in registers, so we can use
        // objc_msgSend (not objc_msgSend_stret which is x86_64 only).
        #[repr(C)]
        struct MtlSize {
            width: usize,
            height: usize,
            depth: usize,
        }
        // SAFETY: `self.raw` is a valid id<MTLDevice>. MtlSize is #[repr(C)]
        // matching MTLSize. On arm64, 3-word structs are returned in registers,
        // so objc_msgSend (not _stret) is correct.
        type SizeFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> MtlSize;
        let sel = unsafe { objc::sel_registerName(sel!("maxThreadsPerThreadgroup")) };
        let f: SizeFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        let s = unsafe { f(self.raw, sel) };
        (s.width, s.height, s.depth)
    }

    /// Check whether this device supports a given GPU family.
    pub fn supports_family(&self, family: GpuFamily) -> bool {
        // SAFETY: `self.raw` is a valid id<MTLDevice>. "supportsFamily:" takes
        // an MTLGPUFamily enum value (usize) and returns a BOOL.
        type BoolFn = unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> bool;
        let sel = unsafe { objc::sel_registerName(sel!("supportsFamily:")) };
        let f: BoolFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, family.metal_value()) }
    }

    /// Returns the raw `id<MTLDevice>` pointer.
    pub fn as_raw_ptr(&self) -> *mut c_void {
        self.raw
    }

    // -- Factory methods for other Metal objects --

    /// Create a Metal buffer with the given size and storage mode.
    pub fn create_buffer(
        &self,
        size: usize,
        mode: StorageMode,
    ) -> Result<MetalBuffer, MetalSysError> {
        if size == 0 {
            return Err(MetalSysError::BufferAllocation(
                "buffer size must be > 0".into(),
            ));
        }
        // -[MTLDevice newBufferWithLength:options:]
        // SAFETY: `self.raw` is a valid id<MTLDevice>. Size is validated > 0
        // above. "newBufferWithLength:options:" returns a +1 retained buffer
        // or nil. transmute casts objc_msgSend to the correct signature.
        type NewBufFn = unsafe extern "C" fn(*mut c_void, *mut c_void, usize, usize) -> *mut c_void;
        let sel = unsafe { objc::sel_registerName(sel!("newBufferWithLength:options:")) };
        let f: NewBufFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        let options = mode.resource_options();
        let raw = unsafe { f(self.raw, sel, size, options) };
        if raw.is_null() {
            return Err(MetalSysError::BufferAllocation(format!(
                "failed to allocate {size} bytes with mode {mode:?}"
            )));
        }
        Ok(MetalBuffer::from_raw(raw, mode))
    }

    /// Create a Metal buffer initialized with the given data.
    pub fn create_buffer_with_data(
        &self,
        data: &[u8],
        mode: StorageMode,
    ) -> Result<MetalBuffer, MetalSysError> {
        if data.is_empty() {
            return Err(MetalSysError::BufferAllocation(
                "buffer data must not be empty".into(),
            ));
        }
        // -[MTLDevice newBufferWithBytes:length:options:]
        // SAFETY: `self.raw` is a valid id<MTLDevice>. `data.as_ptr()` and
        // `data.len()` describe a valid byte slice (non-empty, verified above).
        // "newBufferWithBytes:length:options:" copies the data and returns a
        // +1 retained buffer or nil.
        type NewBufFn =
            unsafe extern "C" fn(*mut c_void, *mut c_void, *const u8, usize, usize) -> *mut c_void;
        let sel = unsafe { objc::sel_registerName(sel!("newBufferWithBytes:length:options:")) };
        let f: NewBufFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        let options = mode.resource_options();
        let raw = unsafe { f(self.raw, sel, data.as_ptr(), data.len(), options) };
        if raw.is_null() {
            return Err(MetalSysError::BufferAllocation(format!(
                "failed to allocate {} bytes with data",
                data.len()
            )));
        }
        Ok(MetalBuffer::from_raw(raw, mode))
    }

    /// Create a command queue for this device.
    pub fn create_command_queue(&self) -> Result<CommandQueue, MetalSysError> {
        // SAFETY: `self.raw` is a valid id<MTLDevice>. "newCommandQueue"
        // returns a +1 retained id<MTLCommandQueue> or nil.
        type NewQueueFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { objc::sel_registerName(sel!("newCommandQueue")) };
        let f: NewQueueFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        let raw = unsafe { f(self.raw, sel) };
        if raw.is_null() {
            return Err(MetalSysError::CommandBuffer(
                "failed to create command queue".into(),
            ));
        }
        Ok(CommandQueue::from_raw(raw))
    }

    /// Compile Metal Shading Language source code into a shader library.
    pub fn compile_shader_source(&self, source: &str) -> Result<ShaderLibrary, MetalSysError> {
        let ns_source = objc::create_nsstring(source)?;
        // Use a nil options dictionary for default compile options.
        // We pass a pointer to an error out-param.
        let mut error: *mut c_void = std::ptr::null_mut();
        // SAFETY: `self.raw` is a valid id<MTLDevice>, `ns_source` is a valid
        // retained NSString. "newLibraryWithSource:options:error:" compiles
        // MSL source and returns a +1 retained library or nil with error.
        // The NSString is released after the call. Error is released if set.
        type NewLibFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut *mut c_void,
        ) -> *mut c_void;
        let sel = unsafe { objc::sel_registerName(sel!("newLibraryWithSource:options:error:")) };
        let f: NewLibFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        let raw = unsafe {
            f(
                self.raw,
                sel,
                ns_source,
                std::ptr::null_mut(), // nil options
                &mut error,
            )
        };
        // Release the NSString source.
        // SAFETY: ns_source is a retained NSString from create_nsstring.
        unsafe { objc::CFRelease(ns_source) };

        if raw.is_null() {
            let desc = if !error.is_null() {
                let d = objc::extract_nserror_description(error);
                // SAFETY: error is a non-null ObjC NSError from the out-param.
                unsafe { objc::CFRelease(error) };
                d
            } else {
                "unknown error".into()
            };
            return Err(MetalSysError::ShaderCompilation(desc));
        }
        Ok(ShaderLibrary::from_raw(raw))
    }

    /// Create a compute pipeline state from a shader function.
    pub fn create_compute_pipeline(
        &self,
        function: &ShaderFunction,
    ) -> Result<ComputePipeline, MetalSysError> {
        let mut error: *mut c_void = std::ptr::null_mut();
        // SAFETY: `self.raw` is a valid id<MTLDevice>, `function` is a valid
        // retained id<MTLFunction>. "newComputePipelineStateWithFunction:error:"
        // returns a +1 retained pipeline or nil with error.
        type NewPipeFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut *mut c_void,
        ) -> *mut c_void;
        let sel =
            unsafe { objc::sel_registerName(sel!("newComputePipelineStateWithFunction:error:")) };
        let f: NewPipeFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        let raw = unsafe { f(self.raw, sel, function.as_raw_ptr(), &mut error) };
        if raw.is_null() {
            let desc = if !error.is_null() {
                let d = objc::extract_nserror_description(error);
                // SAFETY: error is a non-null ObjC NSError from the out-param.
                unsafe { objc::CFRelease(error) };
                d
            } else {
                "unknown error".into()
            };
            return Err(MetalSysError::PipelineCreation(desc));
        }
        Ok(ComputePipeline::from_raw(raw))
    }
}

impl Drop for MetalDevice {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: CFRelease on a retained ObjC object.
            unsafe { objc::CFRelease(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}
