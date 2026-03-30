//! IOSurface-backed tensor management for ANE I/O.
//!
//! All ANE inputs and outputs use IOSurface memory. This module manages
//! tensor creation, data transfer, and lifecycle with correct padding
//! and alignment per ANE constraints.

use std::ffi::c_void;

use half::f16;
use mil_rs::ir::ScalarType;

use crate::{AneError, Result};

// ── FFI bindings (macOS) ─────────────────────────────────────────

#[cfg(target_os = "macos")]
#[allow(dead_code)]
mod ffi {
    use std::ffi::c_void;

    #[link(name = "IOSurface", kind = "framework")]
    unsafe extern "C" {
        pub fn IOSurfaceCreate(properties: *const c_void) -> *mut c_void;
        pub fn IOSurfaceGetBaseAddress(surface: *mut c_void) -> *mut c_void;
        pub fn IOSurfaceLock(surface: *mut c_void, options: u32, seed: *mut u32) -> i32;
        pub fn IOSurfaceUnlock(surface: *mut c_void, options: u32, seed: *mut u32) -> i32;
        pub fn IOSurfaceGetAllocSize(surface: *mut c_void) -> usize;
    }

    #[link(name = "IOSurface", kind = "framework")]
    unsafe extern "C" {
        pub static kIOSurfaceAllocSize: *const c_void;
        pub static kIOSurfaceWidth: *const c_void;
        pub static kIOSurfaceHeight: *const c_void;
        pub static kIOSurfaceBytesPerElement: *const c_void;
        pub static kIOSurfaceBytesPerRow: *const c_void;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        pub fn CFRelease(cf: *mut c_void);
        pub fn CFDictionaryCreateMutable(
            allocator: *const c_void,
            capacity: isize,
            key_callbacks: *const c_void,
            value_callbacks: *const c_void,
        ) -> *mut c_void;
        pub fn CFDictionarySetValue(dict: *mut c_void, key: *const c_void, value: *const c_void);
        pub fn CFNumberCreate(
            allocator: *const c_void,
            number_type: i64,
            value_ptr: *const c_void,
        ) -> *mut c_void;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        pub static kCFAllocatorDefault: *const c_void;
        // Opaque structs — we only need their addresses.
        pub static kCFTypeDictionaryKeyCallBacks: u8;
        pub static kCFTypeDictionaryValueCallBacks: u8;
    }

    /// `kCFNumberSInt64Type` = 4.
    pub const CF_NUMBER_SINT64_TYPE: i64 = 4;

    /// `kIOSurfaceLockReadOnly` = 1.
    pub const IOSURFACE_LOCK_READ_ONLY: u32 = 1;
}

/// Minimum IOSurface allocation size required by the ANE hardware.
///
/// The ANE rejects IOSurface tensors below this size with status 0x1d.
/// Both Orion and maderix/ANE use larger tensor sizes in practice
/// (d_model ≥ 768), so this constraint is rarely hit. The previous
/// value of 48KB (49152) was overly conservative; the actual hardware
/// minimum is smaller but undetermined. We use 16KB as a safe lower
/// bound — still a significant reduction from 48KB for small tensors
/// used in testing.
pub(crate) const ANE_MIN_SURFACE_BYTES: usize = 16384; // 16 KB

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

// ── IOSurface creation ───────────────────────────────────────────

#[cfg(target_os = "macos")]
fn create_iosurface(alloc_size: usize) -> Result<*mut c_void> {
    unsafe {
        let dict = ffi::CFDictionaryCreateMutable(
            ffi::kCFAllocatorDefault,
            5,
            std::ptr::addr_of!(ffi::kCFTypeDictionaryKeyCallBacks) as *const c_void,
            std::ptr::addr_of!(ffi::kCFTypeDictionaryValueCallBacks) as *const c_void,
        );
        if dict.is_null() {
            return Err(AneError::SurfaceError(
                "CFDictionaryCreateMutable returned null".into(),
            ));
        }

        let alloc_i64 = alloc_size as i64;
        let one_i64: i64 = 1;
        let props: [(*const c_void, i64); 5] = [
            (ffi::kIOSurfaceAllocSize, alloc_i64),
            (ffi::kIOSurfaceWidth, alloc_i64),
            (ffi::kIOSurfaceHeight, one_i64),
            (ffi::kIOSurfaceBytesPerElement, one_i64),
            (ffi::kIOSurfaceBytesPerRow, alloc_i64),
        ];

        let mut cf_numbers = Vec::with_capacity(5);
        for &(key, value) in &props {
            let num = ffi::CFNumberCreate(
                ffi::kCFAllocatorDefault,
                ffi::CF_NUMBER_SINT64_TYPE,
                &value as *const i64 as *const c_void,
            );
            ffi::CFDictionarySetValue(dict, key, num);
            cf_numbers.push(num);
        }

        let surface = ffi::IOSurfaceCreate(dict);

        for num in cf_numbers {
            ffi::CFRelease(num);
        }
        ffi::CFRelease(dict);

        if surface.is_null() {
            return Err(AneError::SurfaceError(
                "IOSurfaceCreate returned null".into(),
            ));
        }

        Ok(surface)
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
    pub fn new(channels: usize, seq_len: usize, dtype: ScalarType) -> Result<Self> {
        Self::new_with_min_alloc(channels, seq_len, dtype, ANE_MIN_SURFACE_BYTES)
    }

    /// Create with a specific minimum allocation size (for uniform sizing).
    pub fn new_with_min_alloc(
        channels: usize,
        seq_len: usize,
        dtype: ScalarType,
        min_alloc: usize,
    ) -> Result<Self> {
        let data_size = channels * seq_len * scalar_byte_size(dtype);
        let alloc_size = data_size.max(min_alloc).max(ANE_MIN_SURFACE_BYTES);

        #[cfg(target_os = "macos")]
        {
            let surface = create_iosurface(alloc_size)?;
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
    pub fn write_f16(&mut self, data: &[f16]) -> Result<()> {
        let expected = self.num_elements();
        if data.len() != expected {
            return Err(AneError::SurfaceError(format!(
                "expected {} f16 elements, got {}",
                expected,
                data.len()
            )));
        }
        let byte_len = data.len() * 2;
        let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };
        self.write_bytes(src)
    }

    /// Read packed f16 data from the surface.
    pub fn read_f16(&self) -> Result<Vec<f16>> {
        let byte_len = self.num_elements() * 2;
        let bytes = self.read_bytes(byte_len)?;
        let mut out = vec![f16::ZERO; self.num_elements()];
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, byte_len);
        }
        Ok(out)
    }

    /// Write raw bytes at a byte offset within the surface.
    pub fn write_bytes_at(&mut self, byte_offset: usize, data: &[u8]) -> Result<()> {
        let end = byte_offset.checked_add(data.len()).ok_or_else(|| {
            AneError::SurfaceError("write_bytes_at: offset + length overflows".into())
        })?;
        if end > self.alloc_size {
            return Err(AneError::SurfaceError(format!(
                "write_bytes_at: offset {} + len {} = {} exceeds allocation ({} bytes)",
                byte_offset,
                data.len(),
                end,
                self.alloc_size
            )));
        }

        #[cfg(target_os = "macos")]
        {
            self.with_locked_base(0, |base| unsafe {
                let dst = (base as *mut u8).add(byte_offset);
                std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
            })
        }

        #[cfg(not(target_os = "macos"))]
        {
            self.buffer[byte_offset..end].copy_from_slice(data);
            Ok(())
        }
    }

    /// Read raw bytes from a specific byte offset within the surface.
    pub fn read_bytes_at(&self, byte_offset: usize, len: usize) -> Result<Vec<u8>> {
        let end = byte_offset.checked_add(len).ok_or_else(|| {
            AneError::SurfaceError("read_bytes_at: offset + length overflows".into())
        })?;
        if end > self.alloc_size {
            return Err(AneError::SurfaceError(format!(
                "read_bytes_at: offset {} + len {} = {} exceeds allocation ({} bytes)",
                byte_offset, len, end, self.alloc_size
            )));
        }

        #[cfg(target_os = "macos")]
        {
            let mut out = vec![0u8; len];
            self.with_locked_base(ffi::IOSURFACE_LOCK_READ_ONLY, |base| unsafe {
                let src = (base as *const u8).add(byte_offset);
                std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), len);
            })?;
            Ok(out)
        }

        #[cfg(not(target_os = "macos"))]
        {
            Ok(self.buffer[byte_offset..end].to_vec())
        }
    }

    /// Write f16 values at an element offset within the surface.
    pub fn write_f16_at(&mut self, offset_elements: usize, data: &[f16]) -> Result<()> {
        let byte_offset = offset_elements.checked_mul(2).ok_or_else(|| {
            AneError::SurfaceError("write_f16_at: offset_elements too large".into())
        })?;
        let byte_len = data
            .len()
            .checked_mul(2)
            .ok_or_else(|| AneError::SurfaceError("write_f16_at: data length too large".into()))?;
        let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };
        self.write_bytes_at(byte_offset, src)
    }

    /// Read f16 values from an element offset within the surface.
    pub fn read_f16_at(&self, offset_elements: usize, len: usize) -> Result<Vec<f16>> {
        let byte_offset = offset_elements.checked_mul(2).ok_or_else(|| {
            AneError::SurfaceError("read_f16_at: offset_elements too large".into())
        })?;
        let byte_len = len
            .checked_mul(2)
            .ok_or_else(|| AneError::SurfaceError("read_f16_at: length too large".into()))?;
        let bytes = self.read_bytes_at(byte_offset, byte_len)?;
        let mut out = vec![f16::ZERO; len];
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, byte_len);
        }
        Ok(out)
    }

    /// Write packed f32 data (converted to f16 internally).
    pub fn write_f32(&mut self, data: &[f32]) -> Result<()> {
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
    pub fn copy_column0_from(&mut self, src: &AneTensor) -> Result<()> {
        let dst_c = self.shape[1];
        let src_c = src.shape[1];
        if dst_c != src_c {
            return Err(AneError::SurfaceError(format!(
                "copy_column0_from: channel mismatch: dst has {dst_c}, src has {src_c}"
            )));
        }
        if self.dtype != src.dtype {
            return Err(AneError::SurfaceError(
                "copy_column0_from: dtype mismatch".into(),
            ));
        }

        let channels = dst_c;
        let bpe = scalar_byte_size(self.dtype);
        let src_s = src.shape[3];
        let dst_s = self.shape[3];
        let src_stride = src_s * bpe;
        let dst_stride = dst_s * bpe;

        #[cfg(target_os = "macos")]
        {
            unsafe {
                // Lock src read-only.
                let rc = ffi::IOSurfaceLock(
                    src.surface,
                    ffi::IOSURFACE_LOCK_READ_ONLY,
                    std::ptr::null_mut(),
                );
                if rc != 0 {
                    return Err(AneError::SurfaceError(format!(
                        "copy_column0_from: IOSurfaceLock(src) failed (status {rc})"
                    )));
                }
                let src_base = ffi::IOSurfaceGetBaseAddress(src.surface);
                if src_base.is_null() {
                    ffi::IOSurfaceUnlock(
                        src.surface,
                        ffi::IOSURFACE_LOCK_READ_ONLY,
                        std::ptr::null_mut(),
                    );
                    return Err(AneError::SurfaceError(
                        "copy_column0_from: IOSurfaceGetBaseAddress(src) returned null".into(),
                    ));
                }

                // Lock dst read-write.
                let rc = ffi::IOSurfaceLock(self.surface, 0, std::ptr::null_mut());
                if rc != 0 {
                    ffi::IOSurfaceUnlock(
                        src.surface,
                        ffi::IOSURFACE_LOCK_READ_ONLY,
                        std::ptr::null_mut(),
                    );
                    return Err(AneError::SurfaceError(format!(
                        "copy_column0_from: IOSurfaceLock(dst) failed (status {rc})"
                    )));
                }
                let dst_base = ffi::IOSurfaceGetBaseAddress(self.surface);
                if dst_base.is_null() {
                    ffi::IOSurfaceUnlock(self.surface, 0, std::ptr::null_mut());
                    ffi::IOSurfaceUnlock(
                        src.surface,
                        ffi::IOSURFACE_LOCK_READ_ONLY,
                        std::ptr::null_mut(),
                    );
                    return Err(AneError::SurfaceError(
                        "copy_column0_from: IOSurfaceGetBaseAddress(dst) returned null".into(),
                    ));
                }

                let src_ptr = src_base as *const u8;
                let dst_ptr = dst_base as *mut u8;
                for c in 0..channels {
                    std::ptr::copy_nonoverlapping(
                        src_ptr.add(c * src_stride),
                        dst_ptr.add(c * dst_stride),
                        bpe,
                    );
                }

                ffi::IOSurfaceUnlock(self.surface, 0, std::ptr::null_mut());
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
    pub fn read_column0_f16(&self) -> Result<Vec<f16>> {
        let channels = self.shape[1];
        let seq_len = self.shape[3];

        if seq_len == 1 {
            return self.read_f16();
        }

        let stride_bytes = seq_len * 2; // 2 bytes per f16
        let mut out = vec![f16::ZERO; channels];

        #[cfg(target_os = "macos")]
        {
            self.with_locked_base(ffi::IOSURFACE_LOCK_READ_ONLY, |base| unsafe {
                let src = base as *const u8;
                for c in 0..channels {
                    std::ptr::copy_nonoverlapping(
                        src.add(c * stride_bytes),
                        (out.as_mut_ptr() as *mut u8).add(c * 2),
                        2,
                    );
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

    /// Raw IOSurface pointer for ANE API calls.
    #[cfg(target_os = "macos")]
    pub fn as_ptr(&self) -> *mut c_void {
        self.surface
    }

    // ── Private I/O helpers ──────────────────────────────────────

    fn write_bytes(&mut self, src: &[u8]) -> Result<()> {
        if src.len() > self.alloc_size {
            return Err(AneError::SurfaceError(format!(
                "data ({} bytes) exceeds allocation ({} bytes)",
                src.len(),
                self.alloc_size
            )));
        }

        #[cfg(target_os = "macos")]
        {
            self.with_locked_base(0, |base| unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), base as *mut u8, src.len());
            })
        }

        #[cfg(not(target_os = "macos"))]
        {
            self.buffer[..src.len()].copy_from_slice(src);
            Ok(())
        }
    }

    fn read_bytes(&self, len: usize) -> Result<Vec<u8>> {
        if len > self.alloc_size {
            return Err(AneError::SurfaceError(format!(
                "read ({} bytes) exceeds allocation ({} bytes)",
                len, self.alloc_size
            )));
        }

        #[cfg(target_os = "macos")]
        {
            let mut out = vec![0u8; len];
            self.with_locked_base(ffi::IOSURFACE_LOCK_READ_ONLY, |base| unsafe {
                std::ptr::copy_nonoverlapping(base as *const u8, out.as_mut_ptr(), len);
            })?;
            Ok(out)
        }

        #[cfg(not(target_os = "macos"))]
        {
            Ok(self.buffer[..len].to_vec())
        }
    }

    #[cfg(target_os = "macos")]
    fn with_locked_base<F>(&self, options: u32, f: F) -> Result<()>
    where
        F: FnOnce(*mut c_void),
    {
        unsafe {
            let rc = ffi::IOSurfaceLock(self.surface, options, std::ptr::null_mut());
            if rc != 0 {
                return Err(AneError::SurfaceError(format!(
                    "IOSurfaceLock failed (status {rc})"
                )));
            }
            let base = ffi::IOSurfaceGetBaseAddress(self.surface);
            if base.is_null() {
                ffi::IOSurfaceUnlock(self.surface, options, std::ptr::null_mut());
                return Err(AneError::SurfaceError(
                    "IOSurfaceGetBaseAddress returned null".into(),
                ));
            }
            f(base);
            ffi::IOSurfaceUnlock(self.surface, options, std::ptr::null_mut());
        }
        Ok(())
    }
}

impl Drop for AneTensor {
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        {
            if !self.surface.is_null() {
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
        // Allocate an INT8 tensor and write known data everywhere.
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
        // src and dst both [1, 4, 1, 8], copy column 0 across.
        let mut src = AneTensor::new(4, 8, ScalarType::Float16).unwrap();
        let mut src_data = vec![f16::ZERO; 4 * 8];
        // Set column 0 of each channel: indices 0, 8, 16, 24.
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
        // Non-column-0 positions should still be zero.
        assert_eq!(out[1], f16::ZERO);
        assert_eq!(out[9], f16::ZERO);
    }

    #[test]
    fn copy_column0_different_seq_len() {
        // src [1, 3, 1, 16], dst [1, 3, 1, 32].
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
        // src [1, 2, 1, 1] (flat), dst [1, 2, 1, 8].
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
}
