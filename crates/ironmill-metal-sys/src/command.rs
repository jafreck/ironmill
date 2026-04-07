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

    /// Register a block to be called when the command buffer has completed.
    ///
    /// The handler is called on an unspecified dispatch queue. The `sender`
    /// argument to `closure` is the raw command buffer pointer.
    ///
    /// Uses the ObjC block ABI to create a stack block that invokes the
    /// Rust closure, then copies it to the heap via `_Block_copy`.
    pub fn add_completed_handler<F>(&self, closure: F)
    where
        F: Fn() + Send + 'static,
    {
        // Layout of an ObjC stack block that captures a boxed Fn closure.
        #[repr(C)]
        struct BlockLiteral {
            isa: *const c_void,
            flags: i32,
            reserved: i32,
            invoke: unsafe extern "C" fn(*mut BlockLiteral, *mut c_void),
            descriptor: *const BlockDescriptor,
            closure: *mut dyn Fn(),
        }

        #[repr(C)]
        struct BlockDescriptor {
            reserved: u64,
            size: u64,
        }

        static DESCRIPTOR: BlockDescriptor = BlockDescriptor {
            reserved: 0,
            size: std::mem::size_of::<BlockLiteral>() as u64,
        };

        unsafe extern "C" fn invoke_block(block: *mut BlockLiteral, _cmd_buf: *mut c_void) {
            // SAFETY: `block.closure` is a valid leaked Box<dyn Fn()>.
            let closure = unsafe { &*(*block).closure };
            closure();
        }

        // SAFETY: We create a heap block via _Block_copy so it survives
        // past this stack frame. The closure is leaked into a raw pointer
        // and never freed (the handler runs exactly once per command buffer).
        #[link(name = "System")]
        unsafe extern "C" {
            static _NSConcreteStackBlock: *const c_void;
            fn _Block_copy(block: *const c_void) -> *mut c_void;
        }

        let boxed: Box<dyn Fn()> = Box::new(closure);
        let raw_closure = Box::into_raw(boxed);

        let stack_block = BlockLiteral {
            isa: std::ptr::addr_of!(_NSConcreteStackBlock) as *const c_void,
            flags: 1 << 25, // BLOCK_HAS_COPY_DISPOSE not needed for simple case
            reserved: 0,
            invoke: invoke_block,
            descriptor: &DESCRIPTOR,
            closure: raw_closure,
        };

        // Copy to heap so Metal can call it asynchronously.
        let heap_block =
            unsafe { _Block_copy(&stack_block as *const BlockLiteral as *const c_void) };

        // SAFETY: `self.raw` is a valid command buffer. `heap_block` is
        // a valid heap-allocated ObjC block.
        type AddHandlerFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void);
        let sel = unsafe { objc::sel_registerName(sel!("addCompletedHandler:")) };
        let f: AddHandlerFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel, heap_block) };
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

/// Cached ObjC selectors for ComputeEncoder methods.
///
/// Each selector is resolved once via `sel_registerName` and then reused
/// for all subsequent calls, avoiding ~50-100 ns per lookup × ~1800
/// lookups per decode step.
struct CachedSelectors {
    set_pipeline: *mut c_void,
    set_buffer: *mut c_void,
    set_bytes: *mut c_void,
    set_threadgroup_memory_length: *mut c_void,
    dispatch_threads: *mut c_void,
    dispatch_threadgroups: *mut c_void,
    end_encoding: *mut c_void,
    memory_barrier_scope: *mut c_void,
    memory_barrier_resources: *mut c_void,
}

// SAFETY: ObjC selector pointers are globally stable and thread-safe.
unsafe impl Send for CachedSelectors {}
unsafe impl Sync for CachedSelectors {}

static CACHED_SELS: std::sync::OnceLock<CachedSelectors> = std::sync::OnceLock::new();

fn cached_sels() -> &'static CachedSelectors {
    CACHED_SELS.get_or_init(|| unsafe {
        CachedSelectors {
            set_pipeline: objc::sel_registerName(sel!("setComputePipelineState:")),
            set_buffer: objc::sel_registerName(sel!("setBuffer:offset:atIndex:")),
            set_bytes: objc::sel_registerName(sel!("setBytes:length:atIndex:")),
            set_threadgroup_memory_length: objc::sel_registerName(sel!(
                "setThreadgroupMemoryLength:atIndex:"
            )),
            dispatch_threads: objc::sel_registerName(sel!(
                "dispatchThreads:threadsPerThreadgroup:"
            )),
            dispatch_threadgroups: objc::sel_registerName(sel!(
                "dispatchThreadgroups:threadsPerThreadgroup:"
            )),
            end_encoding: objc::sel_registerName(sel!("endEncoding")),
            memory_barrier_scope: objc::sel_registerName(sel!("memoryBarrierWithScope:")),
            memory_barrier_resources: objc::sel_registerName(sel!(
                "memoryBarrierWithResources:count:"
            )),
        }
    })
}

/// Safe wrapper around an `MTLComputeCommandEncoder`.
pub struct ComputeEncoder {
    raw: *mut c_void,
}

// SAFETY: Encoder is used single-threaded but can be moved between threads.
unsafe impl Send for ComputeEncoder {}

impl ComputeEncoder {
    /// Set the compute pipeline state for subsequent dispatch calls.
    pub fn set_pipeline(&self, pipeline: &ComputePipeline) {
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void);
        let f: SetFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        unsafe { f(self.raw, cached_sels().set_pipeline, pipeline.as_raw_ptr()) };
    }

    /// Bind a buffer at the given index with an offset.
    pub fn set_buffer(&self, buffer: &MetalBuffer, offset: usize, index: usize) {
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, usize, usize);
        let f: SetFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        unsafe {
            f(
                self.raw,
                cached_sels().set_buffer,
                buffer.as_raw_ptr(),
                offset,
                index,
            )
        };
    }

    /// Set inline bytes at the given index.
    pub fn set_bytes(&self, data: &[u8], index: usize) {
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *const u8, usize, usize);
        let f: SetFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        unsafe {
            f(
                self.raw,
                cached_sels().set_bytes,
                data.as_ptr(),
                data.len(),
                index,
            )
        };
    }

    /// Set threadgroup memory length at the given index.
    pub fn set_threadgroup_memory_length(&self, length: usize, index: usize) {
        type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, usize, usize);
        let f: SetFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        unsafe {
            f(
                self.raw,
                cached_sels().set_threadgroup_memory_length,
                length,
                index,
            )
        };
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
        type DispatchFn = unsafe extern "C" fn(*mut c_void, *mut c_void, MtlSize, MtlSize);
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
        unsafe { f(self.raw, cached_sels().dispatch_threads, grid, tg) };
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
        type DispatchFn = unsafe extern "C" fn(*mut c_void, *mut c_void, MtlSize, MtlSize);
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
        unsafe { f(self.raw, cached_sels().dispatch_threadgroups, groups, tg) };
    }

    /// End encoding. Must be called before committing the command buffer.
    pub fn end_encoding(&self) {
        type EndFn = unsafe extern "C" fn(*mut c_void, *mut c_void);
        let f: EndFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        unsafe { f(self.raw, cached_sels().end_encoding) };
    }

    /// Insert a memory barrier for buffer writes.
    ///
    /// Ensures all prior dispatches' buffer writes are visible to
    /// subsequent dispatches within the same compute encoder.
    pub fn memory_barrier_buffers(&self) {
        // MTLBarrierScope::Buffers = 1
        type BarrierFn = unsafe extern "C" fn(*mut c_void, *mut c_void, usize);
        let f: BarrierFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        unsafe { f(self.raw, cached_sels().memory_barrier_scope, 1) };
    }

    /// Insert a resource-specific memory barrier.
    ///
    /// Only flushes GPU caches for the specified buffers, unlike
    /// [`memory_barrier_buffers`] which flushes ALL buffer caches.
    /// This reduces GPU stall time when only a subset of buffers
    /// were written by prior dispatches.
    ///
    /// Requires Apple GPU Family 4+ (Apple 8 / M1+).
    pub fn memory_barrier_with_resources(&self, buffers: &[&MetalBuffer]) {
        let ptrs: Vec<*mut c_void> = buffers.iter().map(|b| b.as_raw_ptr()).collect();
        type BarrierResFn =
            unsafe extern "C" fn(*mut c_void, *mut c_void, *const *mut c_void, usize);
        let f: BarrierResFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        unsafe {
            f(
                self.raw,
                cached_sels().memory_barrier_resources,
                ptrs.as_ptr(),
                ptrs.len(),
            )
        };
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
