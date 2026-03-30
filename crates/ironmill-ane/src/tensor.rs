//! IOSurface-backed tensor management for ANE I/O.
//!
//! All ANE inputs and outputs use IOSurface memory. This module manages
//! tensor creation, data transfer, and lifecycle with correct padding
//! and alignment per ANE constraints.

use std::ffi::c_void;

use half::f16;
use mil_rs::ir::ScalarType;

use crate::{AneError, Result};

/// Minimum IOSurface allocation size (~49KB, ANE constraint #4).
pub const MIN_SURFACE_ALLOC: usize = 49152; // 48 * 1024

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
        Self::new_with_min_alloc(channels, seq_len, dtype, MIN_SURFACE_ALLOC)
    }

    /// Create with a specific minimum allocation size (for uniform sizing).
    pub fn new_with_min_alloc(
        channels: usize,
        seq_len: usize,
        dtype: ScalarType,
        min_alloc: usize,
    ) -> Result<Self> {
        let data_size = channels * seq_len * scalar_byte_size(dtype);
        let alloc_size = data_size.max(min_alloc).max(MIN_SURFACE_ALLOC);

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
            data_size.max(MIN_SURFACE_ALLOC)
        })
        .max()
        .unwrap_or(MIN_SURFACE_ALLOC)
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
    fn tensor_min_49kb() {
        // 4 * 8 * 2 = 64 bytes, well under the 49KB minimum.
        let t = AneTensor::new(4, 8, ScalarType::Float16).unwrap();
        assert_eq!(t.alloc_size(), MIN_SURFACE_ALLOC);
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
            ([1, 32, 1, 128], ScalarType::Float16), // 32*128*2 = 8192 → 49152
            ([1, 64, 1, 512], ScalarType::Float16), // 64*512*2 = 65536
            ([1, 16, 1, 64], ScalarType::Float16),  // 16*64*2  = 2048 → 49152
        ];
        assert_eq!(uniform_alloc_size(&shapes), 65536);
    }

    #[test]
    fn tensor_alloc_size_includes_padding() {
        let t = AneTensor::new(64, 512, ScalarType::Float16).unwrap();
        let data_size = 64 * 512 * 2;
        assert!(t.alloc_size() >= data_size);
        assert!(t.alloc_size() >= MIN_SURFACE_ALLOC);
    }
}
