//! Shared tensor arithmetic helpers.
//!
//! These utilities consolidate common patterns for working with raw tensor
//! byte buffers as typed slices, used across multiple optimization passes.
//!
//! All conversions use safe byte-by-byte reads/writes so that alignment of
//! the underlying `Vec<u8>` does not matter.

/// Decode raw little-endian tensor bytes into a `Vec<f32>`.
///
/// # Panics
///
/// Panics if `data.len()` is not a multiple of 4.
pub fn tensor_as_f32_slice(data: &[u8]) -> Vec<f32> {
    assert!(
        data.len() % 4 == 0,
        "tensor data length must be a multiple of 4 for f32 reinterpretation"
    );
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Convert an `f32` slice back to raw tensor bytes (little-endian).
pub fn f32_slice_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Decode raw little-endian FP16 tensor bytes into a `Vec<f32>`.
///
/// # Panics
///
/// Panics if `data.len()` is not a multiple of 2.
pub fn tensor_f16_as_f32_slice(data: &[u8]) -> Vec<f32> {
    assert!(
        data.len() % 2 == 0,
        "tensor data length must be a multiple of 2 for f16 reinterpretation"
    );
    data.chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_f32_slice() {
        let values: Vec<f32> = vec![1.0, -2.5, 3.14, 0.0, f32::INFINITY];
        let bytes = f32_slice_to_bytes(&values);
        let recovered = tensor_as_f32_slice(&bytes);
        assert_eq!(recovered, values);
    }

    #[test]
    fn round_trip_known_bytes() {
        // 1.0f32 little-endian = 0x00_00_80_3F
        let bytes: Vec<u8> = vec![0x00, 0x00, 0x80, 0x3F];
        let floats = tensor_as_f32_slice(&bytes);
        assert_eq!(floats.len(), 1);
        assert_eq!(floats[0], 1.0f32);

        let back = f32_slice_to_bytes(&floats);
        assert_eq!(back, bytes);
    }

    #[test]
    fn read_modify_write() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0];
        let bytes = f32_slice_to_bytes(&values);

        // Read → modify → write-back (safe replacement for the old _mut API)
        let mut floats = tensor_as_f32_slice(&bytes);
        floats[1] = 42.0;
        let updated_bytes = f32_slice_to_bytes(&floats);

        let recovered = tensor_as_f32_slice(&updated_bytes);
        assert_eq!(recovered, vec![1.0, 42.0, 3.0]);
    }

    #[test]
    fn empty_slice() {
        let bytes: Vec<u8> = vec![];
        let floats = tensor_as_f32_slice(&bytes);
        assert!(floats.is_empty());
    }

    #[test]
    #[should_panic(expected = "multiple of 4")]
    fn misaligned_length_panics() {
        let bytes: Vec<u8> = vec![0x00, 0x00, 0x80];
        let _ = tensor_as_f32_slice(&bytes);
    }
}
