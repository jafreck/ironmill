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

use std::path::Path;

use mil_rs::ir::ScalarType;

/// Size of the file header in bytes.
#[allow(dead_code)]
const FILE_HEADER_SIZE: usize = 64;

/// Size of the chunk descriptor in bytes.
#[allow(dead_code)]
const CHUNK_DESC_SIZE: usize = 64;

/// Offset where weight data begins (after header + chunk descriptor).
const DATA_START: usize = 128;

/// Metadata for a weight entry in the BLOBFILE.
#[derive(Debug, Clone)]
pub struct BlobEntry {
    pub name: String,
    pub offset: u64,
    pub size: u64,
    pub dtype: ScalarType,
}

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
    entries: Vec<BlobEntry>,
}

impl BlobFileWriter {
    /// Create a new writer.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            entries: Vec::new(),
        }
    }

    /// Add a weight tensor. Returns the chunk descriptor offset (always 64).
    ///
    /// MIL text references use `offset=uint64(64)` pointing to the chunk
    /// descriptor at byte 64, matching Orion's BLOBFILE format.
    pub fn add_weight(&mut self, name: &str, data: &[u8], dtype: ScalarType) -> u64 {
        self.entries.push(BlobEntry {
            name: name.to_string(),
            offset: 64,
            size: data.len() as u64,
            dtype,
        });
        self.data.extend_from_slice(data);

        64
    }

    /// Get the list of blob entries (for validation/debugging).
    pub fn entries(&self) -> &[BlobEntry] {
        &self.entries
    }

    /// Write the complete BLOBFILE to disk.
    pub fn write(&self, path: &Path) -> std::io::Result<()> {
        std::fs::write(path, self.build())
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
    fn blobfile_multiple_weights() {
        let mut writer = BlobFileWriter::new();

        let off1 = writer.add_weight("w1", &[0u8; 16], ScalarType::Float16);
        let off2 = writer.add_weight("w2", &[0u8; 32], ScalarType::Float32);

        // All offsets are 64 (each weight gets its own BLOBFILE in practice)
        assert_eq!(off1, 64);
        assert_eq!(off2, 64);
        assert_eq!(writer.entries().len(), 2);

        // Combined BLOBFILE has all data at byte 128
        let bytes = writer.as_bytes();
        assert_eq!(bytes.len(), DATA_START + 16 + 32);
    }

    #[test]
    fn blobfile_write_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("weights.blob");

        let mut writer = BlobFileWriter::new();
        let data = vec![0xAB; 64];
        writer.add_weight("test_weight", &data, ScalarType::Float16);
        writer.write(&path).unwrap();

        let read_back = std::fs::read(&path).unwrap();
        assert_eq!(read_back, writer.as_bytes());
    }

    #[test]
    fn blobfile_empty() {
        let writer = BlobFileWriter::new();
        let bytes = writer.as_bytes();

        assert_eq!(bytes.len(), DATA_START);
        assert!(writer.entries().is_empty());
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
