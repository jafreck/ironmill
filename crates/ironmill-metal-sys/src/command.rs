//! Safe wrappers for Metal command queue, command buffer, and compute encoder.

use std::ffi::c_void;

use crate::buffer::MetalBuffer;
use crate::error::MetalSysError;
use crate::objc::{self, sel};
use crate::pipeline::ComputePipeline;

// ---------------------------------------------------------------------------
// CommandBufferStatus
// ---------------------------------------------------------------------------

/// Status of a Metal command buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandBufferStatus {
    /// The command buffer has not been enqueued.
    NotEnqueued,
    /// The command buffer is enqueued.
    Enqueued,
    /// The command buffer has been committed for execution.
    Committed,
    /// The command buffer has been scheduled for execution.
    Scheduled,
    /// The command buffer has completed execution successfully.
    Completed,
    /// The command buffer execution resulted in an error.
    Error,
}

impl CommandBufferStatus {
    fn from_raw(value: usize) -> Self {
        match value {
            0 => Self::NotEnqueued,
            1 => Self::Enqueued,
            2 => Self::Committed,
            3 => Self::Scheduled,
            4 => Self::Completed,
            5 => Self::Error,
            _ => Self::Error,
        }
    }
}

// ---------------------------------------------------------------------------
// CommandQueue
// ---------------------------------------------------------------------------

/// Safe wrapper around an `MTLCommandQueue` (id<MTLCommandQueue>).
pub struct CommandQueue {
    raw: *mut c_void,
}

// SAFETY: MTLCommandQueue is thread-safe.
unsafe impl Send for CommandQueue {}
unsafe impl Sync for CommandQueue {}

impl CommandQueue {
    /// Create from a raw retained `id<MTLCommandQueue>`.
    pub(crate) fn from_raw(raw: *mut c_void) -> Self {
        Self { raw }
    }

    /// Create a new command buffer from this queue.
    pub fn command_buffer(&self) -> Result<CommandBuffer, MetalSysError> {
        // SAFETY: All four unsafe calls below form a single ObjC message send
        // pattern. `self.raw` is a valid, retained id<MTLCommandQueue> (non-null
        // enforced by `from_raw` and null-check at construction). The selector
        // "commandBuffer" returns an autoreleased id<MTLCommandBuffer>.
        // `transmute` casts `objc_msgSend` to the correct function signature.
        type CmdBufFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { objc::sel_registerName(sel!("commandBuffer")) };
        let f: CmdBufFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        let raw = unsafe { f(self.raw, sel) };
        if raw.is_null() {
            return Err(MetalSysError::CommandBuffer(
                "failed to create command buffer".into(),
            ));
        }
        // commandBuffer returns an autoreleased object — retain it.
        objc::objc_retain(raw);
        Ok(CommandBuffer { raw })
    }

    /// Returns the raw `id<MTLCommandQueue>` pointer.
    pub fn as_raw_ptr(&self) -> *mut c_void {
        self.raw
    }
}

impl Drop for CommandQueue {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: self.raw is a retained ObjC object; null-checked above.
            unsafe { objc::CFRelease(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// CommandBuffer
// ---------------------------------------------------------------------------

/// Safe wrapper around an `MTLCommandBuffer` (id<MTLCommandBuffer>).
pub struct CommandBuffer {
    raw: *mut c_void,
}

// SAFETY: MTLCommandBuffer is thread-safe for commit/wait.
unsafe impl Send for CommandBuffer {}
unsafe impl Sync for CommandBuffer {}

impl CommandBuffer {
    /// Create a compute command encoder.
    pub fn compute_encoder(&self) -> Result<ComputeEncoder, MetalSysError> {
        // SAFETY: `self.raw` is a valid, retained id<MTLCommandBuffer>. The
        // selector "computeCommandEncoder" returns an autoreleased encoder.
        // transmute casts objc_msgSend to the matching signature.
        type EncFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { objc::sel_registerName(sel!("computeCommandEncoder")) };
        let f: EncFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        let raw = unsafe { f(self.raw, sel) };
        if raw.is_null() {
            return Err(MetalSysError::CommandBuffer(
                "failed to create compute encoder".into(),
            ));
        }
        // computeCommandEncoder returns an autoreleased object — retain it.
        objc::objc_retain(raw);
        Ok(ComputeEncoder { raw })
    }

    /// Commit the command buffer for execution.
    pub fn commit(&self) {
        // SAFETY: `self.raw` is a valid, retained id<MTLCommandBuffer>.
        // "commit" enqueues the buffer for GPU execution.
        type CommitFn = unsafe extern "C" fn(*mut c_void, *mut c_void);
        let sel = unsafe { objc::sel_registerName(sel!("commit")) };
        let f: CommitFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) };
    }

    /// Block the current thread until the command buffer has completed.
    pub fn wait_until_completed(&self) {
        // SAFETY: `self.raw` is a valid, retained id<MTLCommandBuffer>.
        // "waitUntilCompleted" blocks until GPU execution finishes.
        type WaitFn = unsafe extern "C" fn(*mut c_void, *mut c_void);
        let sel = unsafe { objc::sel_registerName(sel!("waitUntilCompleted")) };
        let f: WaitFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) };
    }

    /// Returns the current status of the command buffer.
    pub fn status(&self) -> CommandBufferStatus {
        // SAFETY: `self.raw` is a valid, retained id<MTLCommandBuffer>.
        // "status" returns an NSUInteger (MTLCommandBufferStatus enum).
        type StatusFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> usize;
        let sel = unsafe { objc::sel_registerName(sel!("status")) };
        let f: StatusFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        let raw_status = unsafe { f(self.raw, sel) };
        CommandBufferStatus::from_raw(raw_status)
    }

    /// Returns the raw `id<MTLCommandBuffer>` pointer.
    pub fn as_raw_ptr(&self) -> *mut c_void {
        self.raw
    }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: self.raw is a retained ObjC object; null-checked above.
            unsafe { objc::CFRelease(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// ComputeEncoder
// ---------------------------------------------------------------------------

/// Safe wrapper around an `MTLComputeCommandEncoder`.
pub struct ComputeEncoder {
    raw: *mut c_void,
}

// SAFETY: Encoder is used single-threaded but can be moved between threads.
unsafe impl Send for ComputeEncoder {}

impl ComputeEncoder {
    /// Set the compute pipeline state for subsequent dispatch calls.
    pub fn set_pipeline(&self, pipeline: &ComputePipeline) {
        // SAFETY: `self.raw` is a valid encoder and `pipeline` is a valid
        // retained pipeline state. "setComputePipelineState:" binds the
        // pipeline for subsequent dispatch calls.
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void);
        let sel = unsafe { objc::sel_registerName(sel!("setComputePipelineState:")) };
        let f: SetFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, pipeline.as_raw_ptr()) };
    }

    /// Bind a buffer at the given index with an offset.
    pub fn set_buffer(&self, buffer: &MetalBuffer, offset: usize, index: usize) {
        // SAFETY: `self.raw` is a valid encoder and `buffer` is a valid
        // retained Metal buffer. "setBuffer:offset:atIndex:" binds the buffer
        // at the specified index for compute kernel access.
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, usize, usize);
        let sel = unsafe { objc::sel_registerName(sel!("setBuffer:offset:atIndex:")) };
        let f: SetFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, buffer.as_raw_ptr(), offset, index) };
    }

    /// Set inline bytes at the given index.
    pub fn set_bytes(&self, data: &[u8], index: usize) {
        // SAFETY: `self.raw` is a valid encoder. `data.as_ptr()` and
        // `data.len()` describe a valid byte slice. "setBytes:length:atIndex:"
        // copies the bytes into the encoder's argument table.
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *const u8, usize, usize);
        let sel = unsafe { objc::sel_registerName(sel!("setBytes:length:atIndex:")) };
        let f: SetFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, data.as_ptr(), data.len(), index) };
    }

    /// Set threadgroup memory length at the given index.
    pub fn set_threadgroup_memory_length(&self, length: usize, index: usize) {
        // SAFETY: `self.raw` is a valid encoder.
        // "setThreadgroupMemoryLength:atIndex:" reserves threadgroup memory.
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, usize, usize);
        let sel = unsafe { objc::sel_registerName(sel!("setThreadgroupMemoryLength:atIndex:")) };
        let f: SetFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, length, index) };
    }

    /// Dispatch threads with a non-uniform grid (requires Apple GPU family 4+).
    pub fn dispatch_threads(
        &self,
        grid_size: (usize, usize, usize),
        threadgroup_size: (usize, usize, usize),
    ) {
        #[repr(C)]
        struct MtlSize {
            width: usize,
            height: usize,
            depth: usize,
        }
        // SAFETY: `self.raw` is a valid encoder. MtlSize is #[repr(C)] matching
        // MTLSize layout. "dispatchThreads:threadsPerThreadgroup:" dispatches a
        // non-uniform grid. On arm64, small structs are returned in registers
        // so objc_msgSend (not _stret) is correct.
        type DispatchFn = unsafe extern "C" fn(*mut c_void, *mut c_void, MtlSize, MtlSize);
        let sel = unsafe { objc::sel_registerName(sel!("dispatchThreads:threadsPerThreadgroup:")) };
        let f: DispatchFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        let grid = MtlSize {
            width: grid_size.0,
            height: grid_size.1,
            depth: grid_size.2,
        };
        let tg = MtlSize {
            width: threadgroup_size.0,
            height: threadgroup_size.1,
            depth: threadgroup_size.2,
        };
        unsafe { f(self.raw, sel, grid, tg) };
    }

    /// Dispatch threadgroups with explicit counts.
    pub fn dispatch_threadgroups(
        &self,
        threadgroup_count: (usize, usize, usize),
        threads_per_threadgroup: (usize, usize, usize),
    ) {
        #[repr(C)]
        struct MtlSize {
            width: usize,
            height: usize,
            depth: usize,
        }
        // SAFETY: `self.raw` is a valid encoder. MtlSize is #[repr(C)] matching
        // MTLSize layout. "dispatchThreadgroups:threadsPerThreadgroup:" dispatches
        // the specified number of threadgroups.
        type DispatchFn = unsafe extern "C" fn(*mut c_void, *mut c_void, MtlSize, MtlSize);
        let sel =
            unsafe { objc::sel_registerName(sel!("dispatchThreadgroups:threadsPerThreadgroup:")) };
        let f: DispatchFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        let groups = MtlSize {
            width: threadgroup_count.0,
            height: threadgroup_count.1,
            depth: threadgroup_count.2,
        };
        let tg = MtlSize {
            width: threads_per_threadgroup.0,
            height: threads_per_threadgroup.1,
            depth: threads_per_threadgroup.2,
        };
        unsafe { f(self.raw, sel, groups, tg) };
    }

    /// End encoding. Must be called before committing the command buffer.
    pub fn end_encoding(&self) {
        // SAFETY: `self.raw` is a valid encoder. "endEncoding" finalizes the
        // encoding; no further encoding calls are valid after this.
        type EndFn = unsafe extern "C" fn(*mut c_void, *mut c_void);
        let sel = unsafe { objc::sel_registerName(sel!("endEncoding")) };
        let f: EndFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) };
    }

    /// Returns the raw encoder pointer.
    pub fn as_raw_ptr(&self) -> *mut c_void {
        self.raw
    }
}

impl Drop for ComputeEncoder {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: self.raw is a retained ObjC object; null-checked above.
            unsafe { objc::CFRelease(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}
