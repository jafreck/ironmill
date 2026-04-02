//! INT4 packing utilities for quantized weight storage.
//!
//! These helpers pack and unpack 4-bit integer values into compact byte and
//! u32 representations used by quantization passes and on-disk formats.
//!
//! Two layouts are provided:
//!
//! - **Byte packing** — pairs of 4-bit values are stored low-nibble-first in
//!   each byte.  `[a, b]` becomes `(b << 4) | (a & 0x0F)`.
//!
//! - **u32 packing** — eight 4-bit values per `u32`, little-endian nibble
//!   order (GPTQ / AWQ community standard).  `value[0]` occupies bits 0-3,
//!   `value[1]` bits 4-7, and so on.

/// Pack pairs of 4-bit values into bytes (low nibble first).
///
/// For values `[a, b]`, the packed byte is `(b << 4) | (a & 0x0F)`.
///
/// Values above 15 are silently masked to their low 4 bits.
pub fn pack_int4(values: &[u8]) -> Vec<u8> {
    let len = values.len();
    let out_len = len.div_ceil(2);
    let mut out = Vec::with_capacity(out_len);

    let mut i = 0;
    while i < len {
        let lo = values[i] & 0x0F;
        let hi = if i + 1 < len { values[i + 1] & 0x0F } else { 0 };
        out.push((hi << 4) | lo);
        i += 2;
    }

    out
}

/// Unpack packed bytes into individual 4-bit values.
///
/// Returns exactly `count` values.  Each byte yields two nibbles
/// (low nibble first).
pub fn unpack_int4(packed: &[u8], count: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(count);
    for &byte in packed {
        if out.len() >= count {
            break;
        }
        out.push(byte & 0x0F);
        if out.len() < count {
            out.push((byte >> 4) & 0x0F);
        }
    }
    out
}

/// Pack 8 INT4 values into one `u32` (GPTQ/AWQ community standard,
/// little-endian nibble order).
///
/// `value[0]` occupies bits 0-3, `value[1]` bits 4-7, …, `value[7]` bits
/// 28-31.  If the input length is not a multiple of 8 the final group is
/// zero-padded.  Values above 15 are silently masked to their low 4 bits.
pub fn pack_int4_u32(values: &[u8]) -> Vec<u32> {
    let out_len = values.len().div_ceil(8);
    let mut out = Vec::with_capacity(out_len);

    for chunk in values.chunks(8) {
        let mut word: u32 = 0;
        for (j, &v) in chunk.iter().enumerate() {
            word |= ((v & 0x0F) as u32) << (j * 4);
        }
        out.push(word);
    }

    out
}

/// Unpack u32-packed INT4 values.
///
/// Returns exactly `count` values extracted from the GPTQ/AWQ little-endian
/// nibble layout.
pub fn unpack_int4_u32(packed: &[u32], count: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(count);
    for &word in packed {
        for j in 0..8 {
            if out.len() >= count {
                break;
            }
            out.push(((word >> (j * 4)) & 0x0F) as u8);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // Byte packing
    // ---------------------------------------------------------------

    #[test]
    fn pack_unpack_round_trip() {
        let values: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let packed = pack_int4(&values);
        let recovered = unpack_int4(&packed, values.len());
        assert_eq!(recovered, values);
    }

    #[test]
    fn pack_known_pair() {
        // [a=3, b=12] → (12 << 4) | 3 = 0xC3
        let packed = pack_int4(&[3, 12]);
        assert_eq!(packed, vec![0xC3]);
    }

    #[test]
    fn pack_odd_count_pads_zero() {
        // [5] → hi=0, lo=5 → 0x05
        let packed = pack_int4(&[5]);
        assert_eq!(packed, vec![0x05]);
        let recovered = unpack_int4(&packed, 1);
        assert_eq!(recovered, vec![5]);
    }

    #[test]
    fn unpack_respects_count() {
        let packed = pack_int4(&[1, 2, 3, 4]);
        // Ask for only 3 of the 4 values.
        let recovered = unpack_int4(&packed, 3);
        assert_eq!(recovered, vec![1, 2, 3]);
    }

    #[test]
    fn pack_empty() {
        let packed = pack_int4(&[]);
        assert!(packed.is_empty());
        let recovered = unpack_int4(&packed, 0);
        assert!(recovered.is_empty());
    }

    #[test]
    fn pack_masks_high_bits() {
        // 0xFF masked to 0x0F → 15
        let packed = pack_int4(&[0xFF, 0xFF]);
        assert_eq!(packed, vec![0xFF]);
        let recovered = unpack_int4(&packed, 2);
        assert_eq!(recovered, vec![15, 15]);
    }

    // ---------------------------------------------------------------
    // u32 packing
    // ---------------------------------------------------------------

    #[test]
    fn u32_pack_unpack_round_trip() {
        let values: Vec<u8> = (0..16).collect();
        let packed = pack_int4_u32(&values);
        assert_eq!(packed.len(), 2);
        let recovered = unpack_int4_u32(&packed, values.len());
        assert_eq!(recovered, values);
    }

    #[test]
    fn u32_pack_known_word() {
        // 8 values: 0,1,2,…,7
        // word = 0 | (1<<4) | (2<<8) | (3<<12) | (4<<16) | (5<<20) | (6<<24) | (7<<28)
        //      = 0x76543210
        let values: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let packed = pack_int4_u32(&values);
        assert_eq!(packed, vec![0x76543210]);
    }

    #[test]
    fn u32_pack_partial_group() {
        // 3 values → 1 u32, remaining positions zero-padded.
        let values: Vec<u8> = vec![0xA, 0xB, 0xC];
        let packed = pack_int4_u32(&values);
        assert_eq!(packed.len(), 1);
        let expected: u32 = 0xA | (0xB << 4) | (0xC << 8);
        assert_eq!(packed[0], expected);
        let recovered = unpack_int4_u32(&packed, 3);
        assert_eq!(recovered, vec![0xA, 0xB, 0xC]);
    }

    #[test]
    fn u32_pack_empty() {
        let packed = pack_int4_u32(&[]);
        assert!(packed.is_empty());
        let recovered = unpack_int4_u32(&packed, 0);
        assert!(recovered.is_empty());
    }

    #[test]
    fn u32_unpack_respects_count() {
        let values: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let packed = pack_int4_u32(&values);
        let recovered = unpack_int4_u32(&packed, 5);
        assert_eq!(recovered, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn u32_pack_all_fifteens() {
        let values: Vec<u8> = vec![15; 8];
        let packed = pack_int4_u32(&values);
        assert_eq!(packed, vec![0xFFFFFFFF]);
        let recovered = unpack_int4_u32(&packed, 8);
        assert_eq!(recovered, values);
    }

    #[test]
    fn u32_pack_single_value() {
        let packed = pack_int4_u32(&[7]);
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0], 7);
        let recovered = unpack_int4_u32(&packed, 1);
        assert_eq!(recovered, vec![7]);
    }

    // ---------------------------------------------------------------
    // Cross-format consistency
    // ---------------------------------------------------------------

    #[test]
    fn byte_and_u32_agree_on_values() {
        let values: Vec<u8> = vec![3, 14, 1, 5, 9, 0, 15, 7, 2, 11];
        let byte_packed = pack_int4(&values);
        let byte_recovered = unpack_int4(&byte_packed, values.len());
        let u32_packed = pack_int4_u32(&values);
        let u32_recovered = unpack_int4_u32(&u32_packed, values.len());
        assert_eq!(byte_recovered, u32_recovered);
        assert_eq!(byte_recovered, values);
    }
}
