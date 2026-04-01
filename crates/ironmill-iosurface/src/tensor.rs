//! IOSurface-backed tensor for ANE I/O.
//!
//! [`AneTensor`] provides typed read/write access over IOSurface memory with
//! correct padding and alignment per ANE constraints.

use std::ffi::c_void;

use half::f16;
use mil_rs::ir::ScalarType;

#[cfg(target_os = "macos")]
use crate::surface::ffi;
use crate::surface::{ANE_MIN_SURFACE_BYTES, IOSurfaceError};

// ── Helpers ──────────────────────────────────────────────────────

fn scalar_byte_size(dtype: ScalarType) -> usize {
    match dtype {
        ScalarType::Float16 => 2,
        ScalarType::Float32 => 4,
        ScalarType::Float64 => 8,
        ScalarType::Int8 | ScalarType::UInt8 | ScalarType::Bool => 1,
        ScalarType::Int16 | ScalarType::UInt16 => 2,
        ScalarType::Int32 | ScalarType::UInt32 => 4,
        ScalarType::Int64 | ScalarType::UInt64 => 8,
    }
}

// ── AneTensor ────────────────────────────────────────────────────

/// An IOSurface-backed tensor for ANE I/O.
pub struct AneTensor {
    #[cfg(target_os = "macos")]
    surface: *mut c_void,
    #[cfg(not(target_os = "macos"))]
    buffer: Vec<u8>,
    shape: [usize; 4], // [1, C, 1, S]
    dtype: ScalarType,
    alloc_size: usize,
}

// SAFETY: IOSurface is a kernel-managed object safe to send between threads.
// Not Sync because concurrent access requires explicit locking.
unsafe impl Send for AneTensor {}

impl AneTensor {
    /// Create a new tensor with shape `[1, channels, 1, seq_len]`.
    pub fn new(channels: usize, seq_len: usize, dtype: ScalarType) -> crate::Result<Self> {
        Self::new_with_min_alloc(channels, seq_len, dtype, ANE_MIN_SURFACE_BYTES)
    }

    /// Create with a specific minimum allocation size (for uniform sizing).
    pub fn new_with_min_alloc(
        channels: usize,
        seq_len: usize,
        dtype: ScalarType,
        min_alloc: usize,
    ) -> crate::Result<Self> {
        let data_size = channels * seq_len * scalar_byte_size(dtype);
        let alloc_size = data_size.max(min_alloc).max(ANE_MIN_SURFACE_BYTES);

        #[cfg(target_os = "macos")]
        {
            let surface = crate::surface::create_iosurface(alloc_size)?;
            Ok(Self {
                surface,
                shape: [1, channels, 1, seq_len],
                dtype,
                alloc_size,
            })
        }

        #[cfg(not(target_os = "macos"))]
        {
            Ok(Self {
                buffer: vec![0u8; alloc_size],
                shape: [1, channels, 1, seq_len],
                dtype,
                alloc_size,
            })
        }
    }

    /// Write packed f16 data into the surface.
    pub fn write_f16(&mut self, data: &[f16]) -> crate::Result<()> {
        let expected = self.num_elements();
        if data.len() != expected {
            return Err(IOSurfaceError::CopyFailed(format!(
                "expected {} f16 elements, got {}",
                expected,
                data.len()
            )));
        }
        let byte_len = data.len() * 2;
        // SAFETY: `data` is a valid f16 slice; reinterpreting as bytes is sound
        // because f16 has no alignment requirement stricter than u8 for reads.
        let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };
        self.write_bytes(src)
    }

    /// Read packed f16 data from the surface.
    pub fn read_f16(&self) -> crate::Result<Vec<f16>> {
        let byte_len = self.num_elements() * 2;
        let bytes = self.read_bytes(byte_len)?;
        let mut out = vec![f16::ZERO; self.num_elements()];
        // SAFETY: `bytes` has exactly `byte_len` bytes and `out` has exactly
        // `num_elements` f16 values (byte_len = num_elements * 2). Both
        // pointers are valid and non-overlapping.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, byte_len);
        }
        Ok(out)
    }

    /// Write raw bytes at a byte offset within the surface.
    pub fn write_bytes_at(&mut self, byte_offset: usize, data: &[u8]) -> crate::Result<()> {
        let end = byte_offset.checked_add(data.len()).ok_or_else(|| {
            IOSurfaceError::CopyFailed("write_bytes_at: offset + length overflows".into())
        })?;
        if end > self.alloc_size {
            return Err(IOSurfaceError::CopyFailed(format!(
                "write_bytes_at: offset {} + len {} = {} exceeds allocation ({} bytes)",
                byte_offset,
                data.len(),
                end,
                self.alloc_size
            )));
        }

        #[cfg(target_os = "macos")]
        {
            self.with_locked_base(0, |base| {
                // SAFETY: `base` is a valid locked IOSurface pointer; write
                // target is within the allocation (bounds checked above).
                unsafe {
                    let dst = (base as *mut u8).add(byte_offset);
                    std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
                }
            })
        }

        #[cfg(not(target_os = "macos"))]
        {
            self.buffer[byte_offset..end].copy_from_slice(data);
            Ok(())
        }
    }

    /// Read raw bytes from a specific byte offset within the surface.
    pub fn read_bytes_at(&self, byte_offset: usize, len: usize) -> crate::Result<Vec<u8>> {
        let end = byte_offset.checked_add(len).ok_or_else(|| {
            IOSurfaceError::CopyFailed("read_bytes_at: offset + length overflows".into())
        })?;
        if end > self.alloc_size {
            return Err(IOSurfaceError::CopyFailed(format!(
                "read_bytes_at: offset {} + len {} = {} exceeds allocation ({} bytes)",
                byte_offset, len, end, self.alloc_size
            )));
        }

        #[cfg(target_os = "macos")]
        {
            let mut out = vec![0u8; len];
            self.with_locked_base(ffi::IOSURFACE_LOCK_READ_ONLY, |base| {
                // SAFETY: `base` is a valid locked IOSurface pointer; read
                // source is within the allocation (bounds checked above).
                unsafe {
                    let src = (base as *const u8).add(byte_offset);
                    std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), len);
                }
            })?;
            Ok(out)
        }

        #[cfg(not(target_os = "macos"))]
        {
            Ok(self.buffer[byte_offset..end].to_vec())
        }
    }

    /// Write f16 values at an element offset within the surface.
    pub fn write_f16_at(&mut self, offset_elements: usize, data: &[f16]) -> crate::Result<()> {
        let byte_offset = offset_elements.checked_mul(2).ok_or_else(|| {
            IOSurfaceError::CopyFailed("write_f16_at: offset_elements too large".into())
        })?;
        let byte_len = data.len().checked_mul(2).ok_or_else(|| {
            IOSurfaceError::CopyFailed("write_f16_at: data length too large".into())
        })?;
        // SAFETY: `data` is a valid f16 slice; reinterpreting as bytes is sound.
        let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };
        self.write_bytes_at(byte_offset, src)
    }

    /// Read f16 values from an element offset within the surface.
    pub fn read_f16_at(&self, offset_elements: usize, len: usize) -> crate::Result<Vec<f16>> {
        let byte_offset = offset_elements.checked_mul(2).ok_or_else(|| {
            IOSurfaceError::CopyFailed("read_f16_at: offset_elements too large".into())
        })?;
        let byte_len = len
            .checked_mul(2)
            .ok_or_else(|| IOSurfaceError::CopyFailed("read_f16_at: length too large".into()))?;
        let bytes = self.read_bytes_at(byte_offset, byte_len)?;
        let mut out = vec![f16::ZERO; len];
        // SAFETY: `bytes` has exactly `byte_len` bytes and `out` has `len`
        // f16 values. Both pointers are valid and non-overlapping.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, byte_len);
        }
        Ok(out)
    }

    /// Write packed f32 data (converted to f16 internally).
    pub fn write_f32(&mut self, data: &[f32]) -> crate::Result<()> {
        let f16_data: Vec<f16> = data.iter().map(|&v| f16::from_f32(v)).collect();
        self.write_f16(&f16_data)
    }

    /// Tensor shape `[1, C, 1, S]`.
    pub fn shape(&self) -> [usize; 4] {
        self.shape
    }

    /// Element data type.
    pub fn dtype(&self) -> ScalarType {
        self.dtype
    }

    /// Allocation size in bytes.
    pub fn alloc_size(&self) -> usize {
        self.alloc_size
    }

    /// Number of elements (`C * S` for `[1, C, 1, S]`).
    pub fn num_elements(&self) -> usize {
        self.shape[1] * self.shape[3]
    }

    /// Copy column 0 from `src` into column 0 of `self`.
    ///
    /// Both tensors must have the same channel count (`shape[1]`) and dtype.
    /// Copies one element per channel from `src[c * src_S + 0]` to
    /// `self[c * self_S + 0]`. The rest of each destination row is left
    /// unchanged (staging buffers are pre-zeroed).
    pub fn copy_column0_from(&mut self, src: &AneTensor) -> crate::Result<()> {
        let dst_c = self.shape[1];
        let src_c = src.shape[1];
        if src_c < dst_c {
            return Err(IOSurfaceError::CopyFailed(format!(
                "copy_column0_from: src has {src_c} channels, need at least {dst_c}"
            )));
        }

        let channels = dst_c;
        let bpe = scalar_byte_size(self.dtype);
        let src_s = src.shape[3];
        let dst_s = self.shape[3];
        let src_stride = src_s * bpe;
        let dst_stride = dst_s * bpe;

        #[cfg(target_os = "macos")]
        {
            // SAFETY: Both surfaces are locked before access, base addresses
            // are verified non-null, and the strided copy stays within each
            // surface's allocation bounds (channels * stride ≤ alloc_size).
            let rc = unsafe {
                ffi::IOSurfaceLock(
                    src.surface,
                    ffi::IOSURFACE_LOCK_READ_ONLY,
                    std::ptr::null_mut(),
                )
            };
            if rc != 0 {
                return Err(IOSurfaceError::LockFailed(format!(
                    "copy_column0_from: IOSurfaceLock(src) failed (status {rc})"
                )));
            }
            let src_base = unsafe { ffi::IOSurfaceGetBaseAddress(src.surface) };
            if src_base.is_null() {
                unsafe {
                    ffi::IOSurfaceUnlock(
                        src.surface,
                        ffi::IOSURFACE_LOCK_READ_ONLY,
                        std::ptr::null_mut(),
                    );
                }
                return Err(IOSurfaceError::LockFailed(
                    "copy_column0_from: IOSurfaceGetBaseAddress(src) returned null".into(),
                ));
            }

            let rc = unsafe { ffi::IOSurfaceLock(self.surface, 0, std::ptr::null_mut()) };
            if rc != 0 {
                unsafe {
                    ffi::IOSurfaceUnlock(
                        src.surface,
                        ffi::IOSURFACE_LOCK_READ_ONLY,
                        std::ptr::null_mut(),
                    );
                }
                return Err(IOSurfaceError::LockFailed(format!(
                    "copy_column0_from: IOSurfaceLock(dst) failed (status {rc})"
                )));
            }
            let dst_base = unsafe { ffi::IOSurfaceGetBaseAddress(self.surface) };
            if dst_base.is_null() {
                unsafe { ffi::IOSurfaceUnlock(self.surface, 0, std::ptr::null_mut()) };
                unsafe {
                    ffi::IOSurfaceUnlock(
                        src.surface,
                        ffi::IOSURFACE_LOCK_READ_ONLY,
                        std::ptr::null_mut(),
                    );
                }
                return Err(IOSurfaceError::LockFailed(
                    "copy_column0_from: IOSurfaceGetBaseAddress(dst) returned null".into(),
                ));
            }

            let src_ptr = src_base as *const u8;
            let dst_ptr = dst_base as *mut u8;
            for c in 0..channels {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src_ptr.add(c * src_stride),
                        dst_ptr.add(c * dst_stride),
                        bpe,
                    );
                }
            }

            unsafe { ffi::IOSurfaceUnlock(self.surface, 0, std::ptr::null_mut()) };
            unsafe {
                ffi::IOSurfaceUnlock(
                    src.surface,
                    ffi::IOSURFACE_LOCK_READ_ONLY,
                    std::ptr::null_mut(),
                );
            }
            Ok(())
        }

        #[cfg(not(target_os = "macos"))]
        {
            for c in 0..channels {
                let src_off = c * src_stride;
                let dst_off = c * dst_stride;
                self.buffer[dst_off..dst_off + bpe]
                    .copy_from_slice(&src.buffer[src_off..src_off + bpe]);
            }
            Ok(())
        }
    }

    /// Read column 0 values from an S-padded tensor as f16.
    ///
    /// Returns `C` elements (one per channel), reading only the necessary
    /// bytes at strided positions `c * S * 2` rather than the entire surface.
    /// When `S == 1`, equivalent to [`read_f16`].
    pub fn read_column0_f16(&self) -> crate::Result<Vec<f16>> {
        let channels = self.shape[1];
        let seq_len = self.shape[3];

        if seq_len == 1 {
            return self.read_f16();
        }

        let stride_bytes = seq_len * 2; // 2 bytes per f16
        let mut out = vec![f16::ZERO; channels];

        #[cfg(target_os = "macos")]
        {
            self.with_locked_base(ffi::IOSURFACE_LOCK_READ_ONLY, |base| {
                // SAFETY: Surface is locked and base is verified non-null.
                // Each strided read of 2 bytes at c * stride_bytes is within
                // the allocation (channels * stride_bytes ≤ alloc_size).
                unsafe {
                    let src = base as *const u8;
                    for c in 0..channels {
                        std::ptr::copy_nonoverlapping(
                            src.add(c * stride_bytes),
                            (out.as_mut_ptr() as *mut u8).add(c * 2),
                            2,
                        );
                    }
                }
            })?;
        }

        #[cfg(not(target_os = "macos"))]
        {
            for c in 0..channels {
                let off = c * stride_bytes;
                let bytes = [self.buffer[off], self.buffer[off + 1]];
                out[c] = f16::from_le_bytes(bytes);
            }
        }

        Ok(out)
    }

    /// Copy column 0 from this FP16 tensor into a contiguous region of an
    /// INT8 cache tensor, converting FP16→INT8 in a single locked pass
    /// with zero intermediate allocations.
    ///
    /// Reads `C` FP16 values at strided positions (column 0 of each channel
    /// row in this `[1, C, 1, S]` tensor), converts each to INT8, and writes
    /// `C` contiguous bytes into `dst` at the given offset.
    pub fn copy_column0_fp16_as_int8_to(
        &self,
        dst: &mut AneTensor,
        dst_byte_offset: usize,
    ) -> crate::Result<()> {
        let channels = self.shape[1];
        let src_s = self.shape[3];
        let src_stride_bytes = src_s * 2; // FP16 = 2 bytes per element

        let end = dst_byte_offset.checked_add(channels).ok_or_else(|| {
            IOSurfaceError::CopyFailed(
                "copy_column0_fp16_as_int8_to: offset + channels overflows".into(),
            )
        })?;
        if end > dst.alloc_size {
            return Err(IOSurfaceError::CopyFailed(format!(
                "copy_column0_fp16_as_int8_to: offset {} + channels {} = {} exceeds dst alloc ({})",
                dst_byte_offset, channels, end, dst.alloc_size
            )));
        }

        #[cfg(target_os = "macos")]
        {
            // Lock src read-only.
            let rc = unsafe {
                ffi::IOSurfaceLock(
                    self.surface,
                    ffi::IOSURFACE_LOCK_READ_ONLY,
                    std::ptr::null_mut(),
                )
            };
            if rc != 0 {
                return Err(IOSurfaceError::LockFailed(format!(
                    "copy_column0_fp16_as_int8_to: IOSurfaceLock(src) failed (status {rc})"
                )));
            }
            let src_base = unsafe { ffi::IOSurfaceGetBaseAddress(self.surface) };
            if src_base.is_null() {
                unsafe {
                    ffi::IOSurfaceUnlock(
                        self.surface,
                        ffi::IOSURFACE_LOCK_READ_ONLY,
                        std::ptr::null_mut(),
                    );
                }
                return Err(IOSurfaceError::LockFailed(
                    "copy_column0_fp16_as_int8_to: src base address null".into(),
                ));
            }

            // Lock dst read-write.
            let rc = unsafe { ffi::IOSurfaceLock(dst.surface, 0, std::ptr::null_mut()) };
            if rc != 0 {
                unsafe {
                    ffi::IOSurfaceUnlock(
                        self.surface,
                        ffi::IOSURFACE_LOCK_READ_ONLY,
                        std::ptr::null_mut(),
                    );
                }
                return Err(IOSurfaceError::LockFailed(format!(
                    "copy_column0_fp16_as_int8_to: IOSurfaceLock(dst) failed (status {rc})"
                )));
            }
            let dst_base = unsafe { ffi::IOSurfaceGetBaseAddress(dst.surface) };
            if dst_base.is_null() {
                unsafe { ffi::IOSurfaceUnlock(dst.surface, 0, std::ptr::null_mut()) };
                unsafe {
                    ffi::IOSurfaceUnlock(
                        self.surface,
                        ffi::IOSURFACE_LOCK_READ_ONLY,
                        std::ptr::null_mut(),
                    );
                }
                return Err(IOSurfaceError::LockFailed(
                    "copy_column0_fp16_as_int8_to: dst base address null".into(),
                ));
            }

            let src_ptr = src_base as *const u8;
            let dst_ptr = unsafe { (dst_base as *mut u8).add(dst_byte_offset) };

            for c in 0..channels {
                let mut fp16_bytes = [0u8; 2];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src_ptr.add(c * src_stride_bytes),
                        fp16_bytes.as_mut_ptr(),
                        2,
                    );
                }
                let val = f16::from_le_bytes(fp16_bytes);
                // Values are already rounded/clamped to [-128, 127] by MIL program.
                unsafe { *dst_ptr.add(c) = val.to_f32() as i8 as u8 };
            }

            unsafe { ffi::IOSurfaceUnlock(dst.surface, 0, std::ptr::null_mut()) };
            unsafe {
                ffi::IOSurfaceUnlock(
                    self.surface,
                    ffi::IOSURFACE_LOCK_READ_ONLY,
                    std::ptr::null_mut(),
                );
            }
            Ok(())
        }

        #[cfg(not(target_os = "macos"))]
        {
            for c in 0..channels {
                let src_off = c * src_stride_bytes;
                let bytes = [self.buffer[src_off], self.buffer[src_off + 1]];
                let val = f16::from_le_bytes(bytes);
                dst.buffer[dst_byte_offset + c] = val.to_f32() as i8 as u8;
            }
            Ok(())
        }
    }

    /// Raw IOSurface pointer for ANE API calls.
    #[cfg(target_os = "macos")]
    pub fn as_ptr(&self) -> *mut c_void {
        self.surface
    }

    // ── Private I/O helpers ──────────────────────────────────────

    fn write_bytes(&mut self, src: &[u8]) -> crate::Result<()> {
        if src.len() > self.alloc_size {
            return Err(IOSurfaceError::CopyFailed(format!(
                "data ({} bytes) exceeds allocation ({} bytes)",
                src.len(),
                self.alloc_size
            )));
        }

        #[cfg(target_os = "macos")]
        {
            self.with_locked_base(0, |base| {
                // SAFETY: Surface is locked and base is verified non-null.
                // Write length is verified ≤ alloc_size above.
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), base as *mut u8, src.len());
                }
            })
        }

        #[cfg(not(target_os = "macos"))]
        {
            self.buffer[..src.len()].copy_from_slice(src);
            Ok(())
        }
    }

    fn read_bytes(&self, len: usize) -> crate::Result<Vec<u8>> {
        if len > self.alloc_size {
            return Err(IOSurfaceError::CopyFailed(format!(
                "read ({} bytes) exceeds allocation ({} bytes)",
                len, self.alloc_size
            )));
        }

        #[cfg(target_os = "macos")]
        {
            let mut out = vec![0u8; len];
            self.with_locked_base(ffi::IOSURFACE_LOCK_READ_ONLY, |base| {
                // SAFETY: Surface is locked read-only and base is verified
                // non-null. Read length is verified ≤ alloc_size above.
                unsafe {
                    std::ptr::copy_nonoverlapping(base as *const u8, out.as_mut_ptr(), len);
                }
            })?;
            Ok(out)
        }

        #[cfg(not(target_os = "macos"))]
        {
            Ok(self.buffer[..len].to_vec())
        }
    }

    #[cfg(target_os = "macos")]
    fn with_locked_base<F>(&self, options: u32, f: F) -> crate::Result<()>
    where
        F: FnOnce(*mut c_void),
    {
        // SAFETY: `self.surface` is a valid IOSurface pointer obtained from
        // IOSurfaceCreate and not yet released. Lock/unlock pairs are
        // correctly matched, and base address is verified non-null.
        let rc = unsafe { ffi::IOSurfaceLock(self.surface, options, std::ptr::null_mut()) };
        if rc != 0 {
            return Err(IOSurfaceError::LockFailed(format!(
                "IOSurfaceLock failed (status {rc})"
            )));
        }
        let base = unsafe { ffi::IOSurfaceGetBaseAddress(self.surface) };
        if base.is_null() {
            unsafe { ffi::IOSurfaceUnlock(self.surface, options, std::ptr::null_mut()) };
            return Err(IOSurfaceError::LockFailed(
                "IOSurfaceGetBaseAddress returned null".into(),
            ));
        }
        f(base);
        unsafe { ffi::IOSurfaceUnlock(self.surface, options, std::ptr::null_mut()) };
        Ok(())
    }
}

impl Drop for AneTensor {
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        {
            if !self.surface.is_null() {
                // SAFETY: `self.surface` was obtained from IOSurfaceCreate
                // and has not been released yet. CFRelease is the correct
                // way to release IOSurface objects.
                unsafe {
                    ffi::CFRelease(self.surface);
                }
            }
        }
    }
}

// ── Uniform allocation ──────────────────────────────────────────

/// Compute the uniform allocation size for a set of tensors.
///
/// All tensors in a program's input (or output) set must use this size
/// (ANE constraints #2 & #12).
pub fn uniform_alloc_size(shapes: &[([usize; 4], ScalarType)]) -> usize {
    shapes
        .iter()
        .map(|(shape, dtype)| {
            let data_size = shape[1] * shape[3] * scalar_byte_size(*dtype);
            data_size.max(ANE_MIN_SURFACE_BYTES)
        })
        .max()
        .unwrap_or(ANE_MIN_SURFACE_BYTES)
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::ANE_MIN_SURFACE_BYTES;

    #[test]
    fn tensor_new_shape() {
        let t = AneTensor::new(64, 128, ScalarType::Float16).unwrap();
        assert_eq!(t.shape(), [1, 64, 1, 128]);
    }

    #[test]
    fn tensor_small_uses_ane_minimum() {
        // 4 * 8 * 2 = 64 bytes data, but ANE requires at least 16KB.
        let t = AneTensor::new(4, 8, ScalarType::Float16).unwrap();
        assert_eq!(t.alloc_size(), ANE_MIN_SURFACE_BYTES);
    }

    #[test]
    fn tensor_min_alloc_size() {
        // Small tensors get inflated to ANE minimum.
        let t = AneTensor::new(2, 4, ScalarType::Float16).unwrap();
        assert_eq!(t.alloc_size(), ANE_MIN_SURFACE_BYTES);
        assert_eq!(t.alloc_size(), 16384);
    }

    #[test]
    fn tensor_packed_write_read_f16() {
        let mut t = AneTensor::new(2, 4, ScalarType::Float16).unwrap();
        let data: Vec<f16> = (0..8).map(|i| f16::from_f32(i as f32)).collect();
        t.write_f16(&data).unwrap();
        let out = t.read_f16().unwrap();
        assert_eq!(data, out);
    }

    #[test]
    fn tensor_packed_write_read_f32() {
        let mut t = AneTensor::new(2, 4, ScalarType::Float16).unwrap();
        let data: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
        t.write_f32(&data).unwrap();
        let out = t.read_f16().unwrap();
        let expected: Vec<f16> = data.iter().map(|&v| f16::from_f32(v)).collect();
        assert_eq!(expected, out);
    }

    #[test]
    fn tensor_uniform_alloc() {
        let shapes = vec![
            ([1, 32, 1, 128], ScalarType::Float16), // 32*128*2 = 8192 → 16384 (ANE min)
            ([1, 64, 1, 512], ScalarType::Float16), // 64*512*2 = 65536
            ([1, 16, 1, 64], ScalarType::Float16),  // 16*64*2  = 2048 → 16384 (ANE min)
        ];
        assert_eq!(uniform_alloc_size(&shapes), 65536);
    }

    #[test]
    fn tensor_alloc_size_exact_fit() {
        let t = AneTensor::new(64, 512, ScalarType::Float16).unwrap();
        let data_size = 64 * 512 * 2; // 65536
        assert_eq!(t.alloc_size(), data_size);
    }

    #[test]
    fn partial_write_only_changes_target_region() {
        let channels = 8;
        let seq_len = 16;
        let total = channels * seq_len; // 128 bytes
        let mut t = AneTensor::new(channels, seq_len, ScalarType::Int8).unwrap();

        // Fill entire tensor with 0xAA.
        let fill = vec![0xAAu8; total];
        t.write_bytes_at(0, &fill).unwrap();

        // Partial write: overwrite bytes [16..24] with 0x55.
        let patch = vec![0x55u8; 8];
        t.write_bytes_at(16, &patch).unwrap();

        // Read back full tensor and verify.
        let readback = t.read_bytes_at(0, total).unwrap();
        for (i, &b) in readback.iter().enumerate() {
            if (16..24).contains(&i) {
                assert_eq!(b, 0x55, "byte {i} should be 0x55 (patched region)");
            } else {
                assert_eq!(b, 0xAA, "byte {i} should be 0xAA (untouched region)");
            }
        }
    }

    #[test]
    fn partial_f16_write_read_roundtrip() {
        let channels = 4;
        let seq_len = 8;
        let total_elements = channels * seq_len; // 32 f16 elements
        let mut t = AneTensor::new(channels, seq_len, ScalarType::Float16).unwrap();

        // Fill with zeros.
        let zeros = vec![f16::ZERO; total_elements];
        t.write_f16(&zeros).unwrap();

        // Partial write at element offset 8, length 4.
        let patch = [
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        t.write_f16_at(8, &patch).unwrap();

        // Read back just the patched region.
        let readback = t.read_f16_at(8, 4).unwrap();
        assert_eq!(readback.len(), 4);
        assert_eq!(readback[0], f16::from_f32(1.0));
        assert_eq!(readback[3], f16::from_f32(4.0));

        // Verify untouched region is still zero.
        let before = t.read_f16_at(0, 8).unwrap();
        assert!(before.iter().all(|&v| v == f16::ZERO));
    }

    #[test]
    fn copy_column0_same_seq_len() {
        let mut src = AneTensor::new(4, 8, ScalarType::Float16).unwrap();
        let mut src_data = vec![f16::ZERO; 4 * 8];
        src_data[0] = f16::from_f32(1.0);
        src_data[8] = f16::from_f32(2.0);
        src_data[16] = f16::from_f32(3.0);
        src_data[24] = f16::from_f32(4.0);
        src.write_f16(&src_data).unwrap();

        let mut dst = AneTensor::new(4, 8, ScalarType::Float16).unwrap();
        dst.copy_column0_from(&src).unwrap();

        let out = dst.read_f16().unwrap();
        assert_eq!(out[0], f16::from_f32(1.0));
        assert_eq!(out[8], f16::from_f32(2.0));
        assert_eq!(out[16], f16::from_f32(3.0));
        assert_eq!(out[24], f16::from_f32(4.0));
        assert_eq!(out[1], f16::ZERO);
        assert_eq!(out[9], f16::ZERO);
    }

    #[test]
    fn copy_column0_different_seq_len() {
        let mut src = AneTensor::new(3, 16, ScalarType::Float16).unwrap();
        let mut src_data = vec![f16::ZERO; 3 * 16];
        src_data[0] = f16::from_f32(10.0);
        src_data[16] = f16::from_f32(20.0);
        src_data[32] = f16::from_f32(30.0);
        src.write_f16(&src_data).unwrap();

        let mut dst = AneTensor::new(3, 32, ScalarType::Float16).unwrap();
        dst.copy_column0_from(&src).unwrap();

        let out = dst.read_f16().unwrap();
        assert_eq!(out[0], f16::from_f32(10.0));
        assert_eq!(out[32], f16::from_f32(20.0));
        assert_eq!(out[64], f16::from_f32(30.0));
    }

    #[test]
    fn copy_column0_src_seq1() {
        let mut src = AneTensor::new(2, 1, ScalarType::Float16).unwrap();
        src.write_f16(&[f16::from_f32(5.0), f16::from_f32(6.0)])
            .unwrap();

        let mut dst = AneTensor::new(2, 8, ScalarType::Float16).unwrap();
        dst.copy_column0_from(&src).unwrap();

        let out = dst.read_f16().unwrap();
        assert_eq!(out[0], f16::from_f32(5.0));
        assert_eq!(out[8], f16::from_f32(6.0));
    }

    #[test]
    fn copy_column0_channel_mismatch_errors() {
        let src = AneTensor::new(4, 8, ScalarType::Float16).unwrap();
        let mut dst = AneTensor::new(8, 8, ScalarType::Float16).unwrap();
        assert!(dst.copy_column0_from(&src).is_err());
    }

    #[test]
    fn read_column0_f16_strided() {
        let mut t = AneTensor::new(4, 8, ScalarType::Float16).unwrap();
        let mut data = vec![f16::ZERO; 4 * 8];
        data[0] = f16::from_f32(1.0);
        data[1] = f16::from_f32(99.0); // not column 0
        data[8] = f16::from_f32(2.0);
        data[16] = f16::from_f32(3.0);
        data[24] = f16::from_f32(4.0);
        t.write_f16(&data).unwrap();

        let col0 = t.read_column0_f16().unwrap();
        assert_eq!(col0.len(), 4);
        assert_eq!(col0[0], f16::from_f32(1.0));
        assert_eq!(col0[1], f16::from_f32(2.0));
        assert_eq!(col0[2], f16::from_f32(3.0));
        assert_eq!(col0[3], f16::from_f32(4.0));
    }

    #[test]
    fn read_column0_f16_seq1() {
        let mut t = AneTensor::new(3, 1, ScalarType::Float16).unwrap();
        let data = [f16::from_f32(7.0), f16::from_f32(8.0), f16::from_f32(9.0)];
        t.write_f16(&data).unwrap();

        let col0 = t.read_column0_f16().unwrap();
        assert_eq!(col0, data.to_vec());
    }

    #[test]
    fn copy_column0_fp16_as_int8_basic() {
        // Source: FP16 tensor [1, 4, 1, 32] with INT8-range values in col 0.
        let mut src = AneTensor::new(4, 32, ScalarType::Float16).unwrap();
        let mut src_data = vec![f16::ZERO; 4 * 32];
        // Set column 0 values: indices 0, 32, 64, 96 (stride = 32).
        src_data[0] = f16::from_f32(-128.0);
        src_data[32] = f16::from_f32(0.0);
        src_data[64] = f16::from_f32(42.0);
        src_data[96] = f16::from_f32(127.0);
        src.write_f16(&src_data).unwrap();

        // Dest: INT8 tensor [1, 4, 1, 64].
        let mut dst = AneTensor::new(4, 64, ScalarType::Int8).unwrap();

        // Copy to byte offset 12 (simulating seq_pos=3, C=4 → offset=3*4=12).
        src.copy_column0_fp16_as_int8_to(&mut dst, 12).unwrap();

        let result = dst.read_bytes_at(12, 4).unwrap();
        assert_eq!(result[0], (-128i8) as u8);
        assert_eq!(result[1], 0u8);
        assert_eq!(result[2], 42u8);
        assert_eq!(result[3], 127u8);
    }

    #[test]
    fn copy_column0_fp16_as_int8_negative_values() {
        let mut src = AneTensor::new(3, 32, ScalarType::Float16).unwrap();
        let mut src_data = vec![f16::ZERO; 3 * 32];
        src_data[0] = f16::from_f32(-1.0);
        src_data[32] = f16::from_f32(-50.0);
        src_data[64] = f16::from_f32(100.0);
        src.write_f16(&src_data).unwrap();

        let mut dst = AneTensor::new(3, 32, ScalarType::Int8).unwrap();
        src.copy_column0_fp16_as_int8_to(&mut dst, 0).unwrap();

        let result = dst.read_bytes_at(0, 3).unwrap();
        assert_eq!(result[0], (-1i8) as u8);
        assert_eq!(result[1], (-50i8) as u8);
        assert_eq!(result[2], 100u8);
    }

    #[test]
    fn copy_column0_fp16_as_int8_matches_manual_path() {
        let channels = 8;
        let src_s = 32;
        let dst_s = 64;
        let mut src = AneTensor::new(channels, src_s, ScalarType::Float16).unwrap();

        let test_values: Vec<f16> = [-128.0, -64.0, -1.0, 0.0, 1.0, 42.0, 100.0, 127.0]
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();

        let mut src_data = vec![f16::ZERO; channels * src_s];
        for (c, val) in test_values.iter().enumerate() {
            src_data[c * src_s] = *val;
        }
        src.write_f16(&src_data).unwrap();

        // Old path: read → convert → write.
        let col0 = src.read_column0_f16().unwrap();
        let old_bytes: Vec<u8> = col0.iter().map(|v| (v.to_f32() as i8) as u8).collect();
        let mut dst_old = AneTensor::new(channels, dst_s, ScalarType::Int8).unwrap();
        dst_old.write_bytes_at(16, &old_bytes).unwrap();

        // New path: direct copy.
        let mut dst_new = AneTensor::new(channels, dst_s, ScalarType::Int8).unwrap();
        src.copy_column0_fp16_as_int8_to(&mut dst_new, 16).unwrap();

        // Compare.
        let old_result = dst_old.read_bytes_at(16, channels).unwrap();
        let new_result = dst_new.read_bytes_at(16, channels).unwrap();
        assert_eq!(old_result, new_result, "direct copy must match manual path");
    }

    #[test]
    fn copy_column0_fp16_as_int8_overflow_errors() {
        let src = AneTensor::new(4, 32, ScalarType::Float16).unwrap();
        let mut dst = AneTensor::new(4, 8, ScalarType::Int8).unwrap();
        // dst alloc = max(4*8, 16384) = 16384. Writing 4 bytes at offset 16384 overflows.
        assert!(src.copy_column0_fp16_as_int8_to(&mut dst, 16384).is_err());
    }
}
