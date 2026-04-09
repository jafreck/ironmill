//! BLOBFILE weight format writer for ANE programs.
//!
//! The BLOBFILE stores weight data consumed by ANE at compile time.
//! Weights are baked into compiled programs and cannot be changed at
//! eval time (ANE constraint #7).
//!
//! ## Format layout
//!
//! ```text
//! Bytes 0-63:    File header (magic, version, entry count, reserved)
//! Bytes 64+:     Chunk headers (64 bytes each) followed by weight data
//! ```
//!
//! MIL text weight references use `offset=uint64(64)` pointing to the
//! **chunk header** at byte 64, NOT byte 0. This satisfies critical
//! constraint #8.

use mil_rs::ir::ScalarType;

/// Offset where weight data begins (after header + chunk descriptor).
const DATA_START: usize = 128;

/// Builds a BLOBFILE matching the format used by Orion/ANE.
///
/// Format (from Orion's `orion_make_causal_mask_blob`):
/// - Bytes 0-63: File header (byte 0=1, byte 4=2)
/// - Bytes 64-67: Chunk magic `0xDEADBEEF`
/// - Byte 68: `1`
/// - Bytes 72-75: Data size (u32 LE)
/// - Bytes 80-83: Data offset = 128 (u32 LE)
/// - Bytes 128+: Weight data
pub struct BlobFileWriter {
    data: Vec<u8>,
    has_entry: bool,
}

impl BlobFileWriter {
    /// Create a new writer.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            has_entry: false,
        }
    }

    /// Add a weight tensor. Returns the chunk descriptor offset (always 64).
    ///
    /// MIL text references use `offset=uint64(64)` pointing to the chunk
    /// descriptor at byte 64, matching Orion's BLOBFILE format.
    ///
    /// # Panics
    ///
    /// Panics if called more than once — the BLOBFILE format supports only a
    /// single weight per file (each weight gets its own BLOBFILE in practice).
    pub fn add_weight(&mut self, name: &str, data: &[u8], _dtype: ScalarType) -> u64 {
        assert!(
            !self.has_entry,
            "BlobFileWriter supports only one weight per file; \
             attempted to add '{name}' but a weight was already added"
        );
        self.has_entry = true;
        self.data.extend_from_slice(data);

        64
    }

    /// Get the raw BLOBFILE bytes.
    pub fn as_bytes(&self) -> Vec<u8> {
        self.build()
    }

    /// Build the complete BLOBFILE in Orion's format.
    fn build(&self) -> Vec<u8> {
        let data_size = self.data.len();
        let total = DATA_START + data_size;
        let mut buf = vec![0u8; total];

        // File header (bytes 0-63)
        buf[0] = 1;
        buf[4] = 2;

        // Chunk descriptor (bytes 64-127)
        // Magic: 0xDEADBEEF at bytes 64-67
        buf[64] = 0xEF;
        buf[65] = 0xBE;
        buf[66] = 0xAD;
        buf[67] = 0xDE;
        // Byte 68: 1
        buf[68] = 1;
        // Bytes 72-75: data size (u32 LE)
        assert!(
            data_size <= u32::MAX as usize,
            "BLOBFILE data size {data_size} exceeds u32::MAX; \
             weight data too large for BLOBFILE format"
        );
        buf[72..76].copy_from_slice(&(data_size as u32).to_le_bytes());
        // Bytes 80-83: data offset = 128 (u32 LE)
        buf[80..84].copy_from_slice(&(DATA_START as u32).to_le_bytes());

        // Weight data (bytes 128+)
        buf[DATA_START..].copy_from_slice(&self.data);

        buf
    }
}

impl Default for BlobFileWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blobfile_header_format() {
        let writer = BlobFileWriter::new();
        let bytes = writer.as_bytes();

        // Empty BLOBFILE has header + chunk descriptor = 128 bytes
        assert_eq!(bytes.len(), DATA_START);
        // File header: byte 0 = 1, byte 4 = 2
        assert_eq!(bytes[0], 1);
        assert_eq!(bytes[4], 2);
        // Chunk magic: 0xDEADBEEF at bytes 64-67
        assert_eq!(&bytes[64..68], &[0xEF, 0xBE, 0xAD, 0xDE]);
    }

    #[test]
    fn blobfile_offset_64() {
        let mut writer = BlobFileWriter::new();
        let offset = writer.add_weight("w1", &[1, 2, 3, 4], ScalarType::Float16);
        assert_eq!(offset, 64);
    }

    #[test]
    #[should_panic(expected = "BlobFileWriter supports only one weight per file")]
    fn blobfile_multiple_weights_panics() {
        let mut writer = BlobFileWriter::new();
        writer.add_weight("w1", &[0u8; 16], ScalarType::Float16);
        writer.add_weight("w2", &[0u8; 32], ScalarType::Float32);
    }

    #[test]
    fn blobfile_write_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("weights.blob");

        let mut writer = BlobFileWriter::new();
        let data = vec![0xAB; 64];
        writer.add_weight("test_weight", &data, ScalarType::Float16);
        std::fs::write(&path, writer.as_bytes()).unwrap();

        let read_back = std::fs::read(&path).unwrap();
        assert_eq!(read_back, writer.as_bytes());
    }

    #[test]
    fn blobfile_empty() {
        let writer = BlobFileWriter::new();
        let bytes = writer.as_bytes();

        assert_eq!(bytes.len(), DATA_START);
    }

    #[test]
    fn blobfile_data_integrity() {
        let mut writer = BlobFileWriter::new();

        let data: Vec<u8> = (0..128).collect();
        writer.add_weight("w1", &data, ScalarType::Float16);

        let bytes = writer.as_bytes();
        // Data starts at byte 128
        assert_eq!(&bytes[DATA_START..DATA_START + data.len()], &data[..]);
        // Data size recorded in chunk descriptor at bytes 72-75
        let size = u32::from_le_bytes(bytes[72..76].try_into().unwrap()) as usize;
        assert_eq!(size, data.len());
    }
}
