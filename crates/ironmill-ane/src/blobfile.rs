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
const FILE_HEADER_SIZE: usize = 64;

/// Size of each chunk header in bytes.
const CHUNK_HEADER_SIZE: usize = 64;

/// Magic bytes identifying a BLOBFILE.
const MAGIC: &[u8; 4] = b"BLOB";

/// Current format version.
const VERSION: u32 = 1;

/// Metadata for a weight entry in the BLOBFILE.
#[derive(Debug, Clone)]
pub struct BlobEntry {
    pub name: String,
    pub offset: u64,
    pub size: u64,
    pub dtype: ScalarType,
}

/// Builds a BLOBFILE incrementally by adding weight tensors.
///
/// Uses a two-pass approach: `add_weight` accumulates entries and raw data,
/// while `write` / `as_bytes` computes the final layout with file header,
/// chunk headers, and packed weight data.
pub struct BlobFileWriter {
    buffer: Vec<u8>,
    entries: Vec<BlobEntry>,
}

impl BlobFileWriter {
    /// Create a new writer with the 64-byte file header pre-written.
    ///
    /// The file header contains magic bytes, version, and a placeholder
    /// entry count that is updated when the final bytes are produced.
    pub fn new() -> Self {
        let mut buffer = vec![0u8; FILE_HEADER_SIZE];

        // Magic: bytes 0-3
        buffer[0..4].copy_from_slice(MAGIC);
        // Version: bytes 4-7 (u32 LE)
        buffer[4..8].copy_from_slice(&VERSION.to_le_bytes());
        // Entry count at bytes 8-11 starts at 0, updated in as_bytes().

        Self {
            buffer,
            entries: Vec::new(),
        }
    }

    /// Add a weight tensor. Returns the offset where the chunk header will
    /// be placed — this is the value for MIL text `blob(offset=uint64(N))`.
    ///
    /// For the first weight the returned offset is 64 (constraint #8).
    pub fn add_weight(&mut self, name: &str, data: &[u8], dtype: ScalarType) -> u64 {
        let chunk_header_offset = self.buffer.len() as u64;

        // Write the 64-byte chunk header.
        // Data offset and size are placeholders — we patch them below once
        // we know where the data will land.
        let data_offset = chunk_header_offset + CHUNK_HEADER_SIZE as u64;
        let data_size = data.len() as u64;

        // Bytes 0-7: absolute data offset (u64 LE)
        self.buffer.extend_from_slice(&data_offset.to_le_bytes());
        // Bytes 8-15: data size (u64 LE)
        self.buffer.extend_from_slice(&data_size.to_le_bytes());
        // Bytes 16-63: reserved (zeroes)
        self.buffer
            .extend_from_slice(&[0u8; CHUNK_HEADER_SIZE - 16]);

        // Append weight data immediately after the chunk header.
        self.buffer.extend_from_slice(data);

        self.entries.push(BlobEntry {
            name: name.to_string(),
            offset: chunk_header_offset,
            size: data_size,
            dtype,
        });

        // Update entry count in the file header.
        let count = self.entries.len() as u32;
        self.buffer[8..12].copy_from_slice(&count.to_le_bytes());

        chunk_header_offset
    }

    /// Get the list of blob entries (for validation/debugging).
    pub fn entries(&self) -> &[BlobEntry] {
        &self.entries
    }

    /// Write the complete BLOBFILE to disk.
    pub fn write(&self, path: &Path) -> std::io::Result<()> {
        std::fs::write(path, self.as_bytes())
    }

    /// Get the raw buffer (for in-memory use).
    pub fn as_bytes(&self) -> &[u8] {
        &self.buffer
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

        assert!(bytes.len() >= FILE_HEADER_SIZE);
        assert_eq!(&bytes[0..4], b"BLOB");
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 1);
        // Zero entries initially.
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 0);
        // Remaining header bytes are zero.
        assert!(bytes[12..FILE_HEADER_SIZE].iter().all(|&b| b == 0));
    }

    #[test]
    fn blobfile_offset_64() {
        let mut writer = BlobFileWriter::new();
        let offset = writer.add_weight("w1", &[1, 2, 3, 4], ScalarType::Float16);
        // Constraint #8: first chunk header at byte 64.
        assert_eq!(offset, 64);
    }

    #[test]
    fn blobfile_multiple_weights() {
        let mut writer = BlobFileWriter::new();

        let off1 = writer.add_weight("w1", &[0u8; 16], ScalarType::Float16);
        let off2 = writer.add_weight("w2", &[0u8; 32], ScalarType::Float32);
        let off3 = writer.add_weight("w3", &[0u8; 8], ScalarType::Int8);

        assert_eq!(off1, 64);
        // off2 = 64 (chunk1) + 64 (chunk1 header) + 16 (data1)
        assert_eq!(off2, 64 + CHUNK_HEADER_SIZE as u64 + 16);
        // off3 = off2 + 64 (chunk2 header) + 32 (data2)
        assert_eq!(off3, off2 + CHUNK_HEADER_SIZE as u64 + 32);

        assert_eq!(writer.entries().len(), 3);

        // Entry count in header is 3.
        let bytes = writer.as_bytes();
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 3);
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

        assert_eq!(bytes.len(), FILE_HEADER_SIZE);
        assert_eq!(&bytes[0..4], b"BLOB");
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 0);
        assert!(writer.entries().is_empty());
    }

    #[test]
    fn blobfile_data_integrity() {
        let mut writer = BlobFileWriter::new();

        let data1: Vec<u8> = (0..128).collect();
        let data2: Vec<u8> = (128..255).collect();

        let off1 = writer.add_weight("w1", &data1, ScalarType::Float16);
        let off2 = writer.add_weight("w2", &data2, ScalarType::Float32);

        let bytes = writer.as_bytes();

        // Read back chunk header 1 and verify data offset / size.
        let hdr1 = off1 as usize;
        let data1_offset = u64::from_le_bytes(bytes[hdr1..hdr1 + 8].try_into().unwrap()) as usize;
        let data1_size =
            u64::from_le_bytes(bytes[hdr1 + 8..hdr1 + 16].try_into().unwrap()) as usize;
        assert_eq!(data1_size, data1.len());
        assert_eq!(&bytes[data1_offset..data1_offset + data1_size], &data1[..]);

        // Read back chunk header 2 and verify data offset / size.
        let hdr2 = off2 as usize;
        let data2_offset = u64::from_le_bytes(bytes[hdr2..hdr2 + 8].try_into().unwrap()) as usize;
        let data2_size =
            u64::from_le_bytes(bytes[hdr2 + 8..hdr2 + 16].try_into().unwrap()) as usize;
        assert_eq!(data2_size, data2.len());
        assert_eq!(&bytes[data2_offset..data2_offset + data2_size], &data2[..]);
    }
}
