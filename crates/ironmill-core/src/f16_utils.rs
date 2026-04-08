//! Shared f16 ↔ byte conversion utilities.
//!
//! Provides safe, zero-copy casts between `&[f16]` and `&[u8]` slices,
//! centralising the `unsafe` pointer casts and `bytemuck` calls that were
//! previously duplicated across `ironmill-iosurface`, `ironmill-inference`,
//! and `ironmill-compile`.

use half::f16;

/// Reinterpret a raw byte slice as `&[f16]` (zero-copy).
///
/// # Panics
///
/// Panics if `bytes.len()` is not a multiple of 2 or if the pointer is not
/// 2-byte aligned.
pub fn bytes_as_f16(bytes: &[u8]) -> &[f16] {
    bytemuck::cast_slice::<u8, f16>(bytes)
}

/// Reinterpret an `&[f16]` slice as raw bytes (zero-copy).
pub fn f16_as_bytes(data: &[f16]) -> &[u8] {
    bytemuck::cast_slice::<f16, u8>(data)
}

/// Copy raw bytes into a pre-allocated `&mut [f16]` buffer.
///
/// # Panics
///
/// Panics if `bytes.len() != out.len() * 2`.
pub fn copy_bytes_to_f16(bytes: &[u8], out: &mut [f16]) {
    assert_eq!(
        bytes.len(),
        out.len() * 2,
        "byte length must be exactly 2× element count"
    );
    bytemuck::cast_slice_mut::<f16, u8>(out).copy_from_slice(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_bytes_as_f16() {
        let values = [f16::from_f32(1.0), f16::from_f32(-2.5), f16::ZERO];
        let bytes = f16_as_bytes(&values);
        let back = bytes_as_f16(bytes);
        assert_eq!(back, &values);
    }

    #[test]
    fn copy_bytes_roundtrip() {
        let original = [f16::from_f32(3.14), f16::from_f32(-1.0)];
        let bytes = f16_as_bytes(&original);
        let mut out = [f16::ZERO; 2];
        copy_bytes_to_f16(bytes, &mut out);
        assert_eq!(out, original);
    }

    #[test]
    #[should_panic(expected = "byte length must be exactly")]
    fn copy_bytes_rejects_mismatched_length() {
        let mut out = [f16::ZERO; 2];
        copy_bytes_to_f16(&[0u8; 3], &mut out);
    }
}
