//! GGUF weight provider.
//!
//! Parses GGUF model files and provides dequantized weights as FP16 tensors.
//! GGUF is a binary format used by llama.cpp and related tooling to distribute
//! quantized LLM weights. This module implements a minimal parser that:
//!
//! 1. Memory-maps the GGUF file
//! 2. Parses the header, metadata key-value pairs, and tensor descriptors
//! 3. Extracts [`ModelConfig`] from GGUF metadata keys
//! 4. Dequantizes tensor data (Q4_0, Q8_0, F16, F32, BF16) into FP16
//! 5. Remaps GGUF tensor names to HuggingFace-canonical names
//!
//! Split-shard files are auto-discovered by filename pattern.

use std::collections::HashMap;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use half::f16;
use memmap2::Mmap;

use crate::weights::{Architecture, ModelConfig, QuantizationInfo, WeightProvider, WeightTensor};
use mil_rs::MilError;
use mil_rs::ir::ScalarType;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// GGUF magic number: bytes "GGUF" read as a little-endian u32.
const GGUF_MAGIC: u32 = 0x4655_4747;
/// Only version 3 is supported.
const GGUF_VERSION: u32 = 3;
/// Tensor data is aligned to this boundary within the file.
const ALIGNMENT: usize = 32;

// ---------------------------------------------------------------------------
// GGML quantization types
// ---------------------------------------------------------------------------

/// GGML quantization type tags (matches the GGUF spec).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    /// 32-bit IEEE 754 float.
    F32 = 0,
    /// 16-bit IEEE 754 half-precision float.
    F16 = 1,
    /// 4-bit quantization (round-to-nearest, group size 32).
    Q4_0 = 2,
    /// 4-bit quantization with non-zero offset (group size 32).
    Q4_1 = 3,
    /// 5-bit quantization (round-to-nearest, group size 32).
    Q5_0 = 6,
    /// 5-bit quantization with non-zero offset (group size 32).
    Q5_1 = 7,
    /// 8-bit quantization (round-to-nearest, group size 32).
    Q8_0 = 8,
    /// 8-bit quantization with non-zero offset (group size 32).
    Q8_1 = 9,
    /// K-quant 2-bit (super-block quantization).
    Q2K = 10,
    /// K-quant 3-bit (super-block quantization).
    Q3K = 11,
    /// K-quant 4-bit (super-block quantization).
    Q4K = 12,
    /// K-quant 5-bit (super-block quantization).
    Q5K = 13,
    /// K-quant 6-bit (super-block quantization).
    Q6K = 14,
    /// Importance-matrix 2-bit (extra-extra-small variant).
    IQ2XXS = 16,
    /// Importance-matrix 2-bit (extra-small variant).
    IQ2XS = 17,
    /// Importance-matrix 3-bit (extra-extra-small variant).
    IQ3XXS = 18,
    /// Importance-matrix 1-bit (small variant).
    IQ1S = 19,
    /// Importance-matrix 4-bit (non-linear quantization).
    IQ4NL = 20,
    /// Importance-matrix 3-bit (small variant).
    IQ3S = 21,
    /// Importance-matrix 2-bit (small variant).
    IQ2S = 22,
    /// Importance-matrix 4-bit (extra-small variant).
    IQ4XS = 23,
    /// 8-bit signed integer.
    I8 = 24,
    /// 16-bit signed integer.
    I16 = 25,
    /// 32-bit signed integer.
    I32 = 26,
    /// 64-bit signed integer.
    I64 = 27,
    /// 64-bit IEEE 754 double-precision float.
    F64 = 28,
    /// Importance-matrix 1-bit (medium variant).
    IQ1M = 29,
    /// 16-bit brain float (bfloat16).
    BF16 = 30,
}

impl GgmlType {
    fn from_u32(v: u32) -> Result<Self, MilError> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            16 => Ok(Self::IQ2XXS),
            17 => Ok(Self::IQ2XS),
            18 => Ok(Self::IQ3XXS),
            19 => Ok(Self::IQ1S),
            20 => Ok(Self::IQ4NL),
            21 => Ok(Self::IQ3S),
            22 => Ok(Self::IQ2S),
            23 => Ok(Self::IQ4XS),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            29 => Ok(Self::IQ1M),
            30 => Ok(Self::BF16),
            _ => Err(MilError::Validation(format!("unknown GGML type tag: {v}"))),
        }
    }

    /// Number of elements per quantization block.
    fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::F64 => 1,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K => 256,
            _ => 1, // fallback for exotic IQ types
        }
    }

    /// Byte size of one quantization block.
    fn type_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            _ => 0, // exotic IQ types — will error during dequant
        }
    }
}

// ---------------------------------------------------------------------------
// GGUF metadata value types
// ---------------------------------------------------------------------------

/// A value read from the GGUF metadata key-value store.
#[derive(Debug, Clone)]
enum MetadataValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    Str(String),
    Array(Vec<MetadataValue>),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
}

impl MetadataValue {
    fn as_u32(&self) -> Option<u32> {
        match self {
            Self::UInt8(v) => Some(*v as u32),
            Self::UInt16(v) => Some(*v as u32),
            Self::UInt32(v) => Some(*v),
            Self::UInt64(v) => u32::try_from(*v).ok(),
            Self::Int8(v) => u32::try_from(*v).ok(),
            Self::Int16(v) => u32::try_from(*v).ok(),
            Self::Int32(v) => u32::try_from(*v).ok(),
            Self::Int64(v) => u32::try_from(*v).ok(),
            _ => None,
        }
    }

    fn as_usize(&self) -> Option<usize> {
        self.as_u32().map(|v| v as usize)
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float32(v) => Some(*v as f64),
            Self::Float64(v) => Some(*v),
            Self::UInt32(v) => Some(*v as f64),
            Self::Int32(v) => Some(*v as f64),
            Self::UInt64(v) => Some(*v as f64),
            Self::Int64(v) => Some(*v as f64),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    fn as_array(&self) -> Option<&[MetadataValue]> {
        match self {
            Self::Array(a) => Some(a.as_slice()),
            _ => None,
        }
    }

    fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            Self::UInt8(v) => Some(*v != 0),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Binary reader helpers
// ---------------------------------------------------------------------------

/// Minimal cursor-based reader for little-endian binary data.
struct BinReader<'a> {
    cursor: Cursor<&'a [u8]>,
}

impl<'a> BinReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            cursor: Cursor::new(data),
        }
    }

    fn position(&self) -> usize {
        self.cursor.position() as usize
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], MilError> {
        let pos = self.cursor.position() as usize;
        let len = self.cursor.get_ref().len();
        if pos + n > len {
            return Err(MilError::Validation(format!(
                "GGUF: unexpected EOF at offset {pos}, need {n} more bytes"
            )));
        }
        let data = *self.cursor.get_ref();
        self.cursor.set_position((pos + n) as u64);
        Ok(&data[pos..pos + n])
    }

    fn read_u8(&mut self) -> Result<u8, MilError> {
        let b = self.read_bytes(1)?;
        Ok(b[0])
    }

    fn read_i8(&mut self) -> Result<i8, MilError> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16, MilError> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_i16(&mut self) -> Result<i16, MilError> {
        let b = self.read_bytes(2)?;
        Ok(i16::from_le_bytes([b[0], b[1]]))
    }

    fn read_u32(&mut self) -> Result<u32, MilError> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_i32(&mut self) -> Result<i32, MilError> {
        let b = self.read_bytes(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64(&mut self) -> Result<u64, MilError> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_i64(&mut self) -> Result<i64, MilError> {
        let b = self.read_bytes(8)?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_f32(&mut self) -> Result<f32, MilError> {
        let b = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_f64(&mut self) -> Result<f64, MilError> {
        let b = self.read_bytes(8)?;
        Ok(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// Read a GGUF string: u64 length followed by UTF-8 bytes (not null-terminated).
    fn read_gguf_string(&mut self) -> Result<String, MilError> {
        let len = usize::try_from(self.read_u64()?).map_err(|_| {
            MilError::Validation("GGUF: string length exceeds platform limit".into())
        })?;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| MilError::Validation(format!("GGUF: invalid UTF-8 in string: {e}")))
    }

    /// Read a typed metadata value.
    fn read_metadata_value(&mut self, type_tag: u32) -> Result<MetadataValue, MilError> {
        match type_tag {
            0 => Ok(MetadataValue::UInt8(self.read_u8()?)),
            1 => Ok(MetadataValue::Int8(self.read_i8()?)),
            2 => Ok(MetadataValue::UInt16(self.read_u16()?)),
            3 => Ok(MetadataValue::Int16(self.read_i16()?)),
            4 => Ok(MetadataValue::UInt32(self.read_u32()?)),
            5 => Ok(MetadataValue::Int32(self.read_i32()?)),
            6 => Ok(MetadataValue::Float32(self.read_f32()?)),
            7 => {
                let v = self.read_u8()?;
                Ok(MetadataValue::Bool(v != 0))
            }
            8 => Ok(MetadataValue::Str(self.read_gguf_string()?)),
            9 => {
                // Array: element type (u32) + count (u64) + elements
                let elem_type = self.read_u32()?;
                let count = usize::try_from(self.read_u64()?).map_err(|_| {
                    MilError::Validation("GGUF: array count exceeds platform limit".into())
                })?;
                if count > 1_000_000 {
                    return Err(MilError::Validation(format!(
                        "GGUF metadata array count {count} exceeds limit"
                    )));
                }
                let mut elems = Vec::with_capacity(count.min(1 << 20));
                for _ in 0..count {
                    elems.push(self.read_metadata_value(elem_type)?);
                }
                Ok(MetadataValue::Array(elems))
            }
            10 => Ok(MetadataValue::UInt64(self.read_u64()?)),
            11 => Ok(MetadataValue::Int64(self.read_i64()?)),
            12 => Ok(MetadataValue::Float64(self.read_f64()?)),
            _ => Err(MilError::Validation(format!(
                "GGUF: unknown metadata value type: {type_tag}"
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor descriptor (parsed from GGUF header, before data is loaded)
// ---------------------------------------------------------------------------

/// Parsed GGUF tensor info entry.
struct GgufTensorInfo {
    name: String,
    dimensions: Vec<usize>,
    ggml_type: GgmlType,
    /// Offset relative to the start of the tensor data section.
    offset: u64,
}

// ---------------------------------------------------------------------------
// Owned dequantized tensor
// ---------------------------------------------------------------------------

/// Location of a tensor within the mmapped GGUF shards.
///
/// Stores enough metadata to lazily dequantize without re-parsing headers.
struct GgufTensorLocation {
    /// Index into `GgufProvider::mmaps`.
    shard_index: usize,
    /// Absolute byte offset within the mmap.
    abs_offset: usize,
    /// Raw byte length on disk.
    byte_len: usize,
    /// Number of elements in the tensor.
    num_elements: usize,
    /// GGML quantization type.
    ggml_type: GgmlType,
    /// Tensor shape.
    dimensions: Vec<usize>,
}

// ---------------------------------------------------------------------------
// GgufProvider
// ---------------------------------------------------------------------------

/// Weight provider backed by one or more GGUF files.
///
/// Tensor data is memory-mapped and dequantized on each `tensor()` call.
/// F16 tensors are returned as zero-copy borrows from the mmap; other
/// types are dequantized into a fresh `Vec<u8>` and returned as owned data.
pub struct GgufProvider {
    config: ModelConfig,
    /// Memory-mapped shard files, kept alive for the provider lifetime.
    mmaps: Vec<Mmap>,
    /// Tensor metadata indexed by HuggingFace-canonical name.
    tensor_index: HashMap<String, GgufTensorLocation>,
}

impl GgufProvider {
    /// Load a GGUF model from `path`.
    ///
    /// For split-shard models (e.g. `model-00001-of-00003.gguf`), pass any
    /// shard path and siblings will be auto-discovered.
    #[allow(unsafe_code)]
    pub fn load(path: &Path) -> Result<Self, MilError> {
        let shard_paths = discover_shards(path)?;
        let mut all_metadata: HashMap<String, MetadataValue> = HashMap::new();
        let mut tensor_index: HashMap<String, GgufTensorLocation> = HashMap::new();
        let mut mmaps = Vec::with_capacity(shard_paths.len());

        for (shard_idx, shard_path) in shard_paths.iter().enumerate() {
            let file = std::fs::File::open(shard_path)?;
            // SAFETY: we treat the mmap as read-only and do not mutate it.
            let mmap = unsafe { Mmap::map(&file)? };
            let data: &[u8] = &mmap;

            let mut reader = BinReader::new(data);

            // --- Header ---
            let magic = reader.read_u32()?;
            if magic != GGUF_MAGIC {
                return Err(MilError::Validation(format!(
                    "GGUF: invalid magic 0x{magic:08X} in {}",
                    shard_path.display()
                )));
            }

            let version = reader.read_u32()?;
            if version != GGUF_VERSION {
                return Err(MilError::Validation(format!(
                    "GGUF: unsupported version {version} (expected {GGUF_VERSION}) in {}",
                    shard_path.display()
                )));
            }

            let tensor_count = usize::try_from(reader.read_u64()?).map_err(|_| {
                MilError::Validation("GGUF: tensor count exceeds platform limit".into())
            })?;
            let metadata_kv_count = usize::try_from(reader.read_u64()?).map_err(|_| {
                MilError::Validation("GGUF: metadata KV count exceeds platform limit".into())
            })?;

            // --- Metadata KV pairs ---
            for _ in 0..metadata_kv_count {
                let key = reader.read_gguf_string()?;
                let value_type = reader.read_u32()?;
                let value = reader.read_metadata_value(value_type)?;
                all_metadata.insert(key, value);
            }

            // --- Tensor info entries ---
            let mut tensor_infos = Vec::with_capacity(tensor_count);
            for _ in 0..tensor_count {
                let name = reader.read_gguf_string()?;
                let n_dims = usize::try_from(reader.read_u32()?).map_err(|_| {
                    MilError::Validation("GGUF: dimension count exceeds platform limit".into())
                })?;
                let mut dimensions = Vec::with_capacity(n_dims);
                for _ in 0..n_dims {
                    dimensions.push(usize::try_from(reader.read_u64()?).map_err(|_| {
                        MilError::Validation("GGUF: dimension value exceeds platform limit".into())
                    })?);
                }
                let ggml_type = GgmlType::from_u32(reader.read_u32()?)?;
                let offset = reader.read_u64()?;
                tensor_infos.push(GgufTensorInfo {
                    name,
                    dimensions,
                    ggml_type,
                    offset,
                });
            }

            // --- Compute tensor data section start ---
            // The tensor data section starts at the next ALIGNMENT boundary
            // after all header/metadata/tensor-info bytes.
            let header_end = reader.position();
            let data_section_start = align_offset(header_end, ALIGNMENT)?;

            // --- Index each tensor's location for lazy dequantization ---
            for info in &tensor_infos {
                let abs_offset = data_section_start
                    .checked_add(info.offset as usize)
                    .ok_or_else(|| {
                        MilError::Validation(format!(
                            "GGUF: tensor '{}' offset overflow",
                            info.name
                        ))
                    })?;
                let num_elements: usize = info
                    .dimensions
                    .iter()
                    .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                    .ok_or_else(|| {
                        MilError::Validation(format!(
                            "GGUF: tensor '{}' dimensions overflow",
                            info.name
                        ))
                    })?;
                if num_elements == 0 {
                    continue;
                }
                let byte_len = tensor_byte_size(num_elements, info.ggml_type)?;
                let end = abs_offset.checked_add(byte_len).ok_or_else(|| {
                    MilError::Validation(format!(
                        "GGUF: tensor '{}' data range overflow",
                        info.name
                    ))
                })?;
                if end > data.len() {
                    return Err(MilError::Validation(format!(
                        "GGUF: tensor '{}' data extends beyond file (offset {abs_offset}, \
                         need {byte_len} bytes, file is {} bytes)",
                        info.name,
                        data.len()
                    )));
                }

                let hf_name = remap_tensor_name(&info.name);
                tensor_index.insert(
                    hf_name,
                    GgufTensorLocation {
                        shard_index: shard_idx,
                        abs_offset,
                        byte_len,
                        num_elements,
                        ggml_type: info.ggml_type,
                        dimensions: info.dimensions.clone(),
                    },
                );
            }

            mmaps.push(mmap);
        }

        let config = extract_model_config(&all_metadata)?;

        Ok(Self {
            config,
            mmaps,
            tensor_index,
        })
    }
}

impl WeightProvider for GgufProvider {
    fn tensor(&self, name: &str) -> Result<WeightTensor<'_>, MilError> {
        let loc = self
            .tensor_index
            .get(name)
            .ok_or_else(|| MilError::Validation(format!("GGUF: tensor not found: {name}")))?;

        // F16 data can be borrowed directly from the mmap — no
        // dequantization or caching needed.
        if loc.ggml_type == GgmlType::F16 {
            let mmap = &self.mmaps[loc.shard_index];
            let expected = loc
                .num_elements
                .checked_mul(2)
                .ok_or_else(|| MilError::Validation("GGUF: F16 size overflow".into()))?;
            if loc.byte_len < expected {
                return Err(MilError::Validation(
                    "GGUF: F16 tensor data too short".into(),
                ));
            }
            let raw = &mmap[loc.abs_offset..loc.abs_offset + expected];
            return Ok(WeightTensor::borrowed(
                raw,
                loc.dimensions.clone(),
                ScalarType::Float16,
            ));
        }

        // Q4_0: repack blocks into separate packed-nibble and scale buffers.
        // Only for 2D+ tensors — 1D tensors fall through to the FP16 dequant path
        // since per-group quantization requires an axis to group along.
        if loc.ggml_type == GgmlType::Q4_0 && loc.dimensions.len() >= 2 {
            let mmap = &self.mmaps[loc.shard_index];
            let raw = &mmap[loc.abs_offset..loc.abs_offset + loc.byte_len];
            let (packed_data, scales, zero_point) = repack_q4_0(raw, loc.num_elements)?;
            return Ok(
                WeightTensor::owned(packed_data, loc.dimensions.clone(), ScalarType::UInt8)
                    .with_quant_info(QuantizationInfo::AffineDequantize {
                        scale: scales,
                        zero_point,
                        scale_dtype: ScalarType::Float16,
                        zero_point_dtype: ScalarType::Float16,
                        axis: Some(1),
                        bit_width: 4,
                        group_size: Some(32),
                        awq_scales: None,
                        g_idx: None,
                    }),
            );
        }

        // Q8_0: repack blocks into separate int8 and scale buffers.
        // Only for 2D+ tensors — same rationale as Q4_0 above.
        if loc.ggml_type == GgmlType::Q8_0 && loc.dimensions.len() >= 2 {
            let mmap = &self.mmaps[loc.shard_index];
            let raw = &mmap[loc.abs_offset..loc.abs_offset + loc.byte_len];
            let (quant_data, scales, zero_point) = repack_q8_0(raw, loc.num_elements)?;
            return Ok(
                WeightTensor::owned(quant_data, loc.dimensions.clone(), ScalarType::Int8)
                    .with_quant_info(QuantizationInfo::AffineDequantize {
                        scale: scales,
                        zero_point,
                        scale_dtype: ScalarType::Float16,
                        zero_point_dtype: ScalarType::Float16,
                        axis: Some(1),
                        bit_width: 8,
                        group_size: Some(32),
                        awq_scales: None,
                        g_idx: None,
                    }),
            );
        }

        // All other types: dequantize to FP16 (existing path)
        let mmap = &self.mmaps[loc.shard_index];
        let raw = &mmap[loc.abs_offset..loc.abs_offset + loc.byte_len];
        let fp16_data = dequantize_to_fp16(raw, loc.num_elements, loc.ggml_type)?;

        Ok(WeightTensor::owned(
            fp16_data,
            loc.dimensions.clone(),
            ScalarType::Float16,
        ))
    }

    fn tensor_names(&self) -> Vec<&str> {
        self.tensor_index.keys().map(|s| s.as_str()).collect()
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn has_tensor(&self, name: &str) -> bool {
        self.tensor_index.contains_key(name)
    }
}

// ---------------------------------------------------------------------------
// Split-shard discovery
// ---------------------------------------------------------------------------

/// Discover all shard files for a split GGUF model.
///
/// Recognises the pattern `<stem>-NNNNN-of-NNNNN.gguf`. If the path does
/// not match this pattern it is treated as a single file.
fn discover_shards(path: &Path) -> Result<Vec<PathBuf>, MilError> {
    let path = std::fs::canonicalize(path)?;
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| MilError::Validation("GGUF: invalid file path".into()))?;

    // Pattern: <prefix>-NNNNN-of-NNNNN.gguf
    if let Some(prefix) = detect_shard_prefix(file_name) {
        let parent = path.parent().ok_or_else(|| {
            MilError::Validation("GGUF: cannot determine parent directory".into())
        })?;
        let mut shards: Vec<PathBuf> = Vec::new();
        for entry in std::fs::read_dir(parent)? {
            let entry = entry?;
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with(&prefix)
                && name_str.ends_with(".gguf")
                && detect_shard_prefix(&name_str)
                    .map(|p| p == prefix)
                    .unwrap_or(false)
            {
                shards.push(entry.path());
            }
        }
        shards.sort();
        if shards.is_empty() {
            shards.push(path);
        }
        Ok(shards)
    } else {
        Ok(vec![path])
    }
}

/// If `name` matches `<prefix>-NNNNN-of-NNNNN.gguf`, returns `<prefix>`.
fn detect_shard_prefix(name: &str) -> Option<String> {
    let stem = name.strip_suffix(".gguf")?;
    // Expect: ...-NNNNN-of-NNNNN
    let parts: Vec<&str> = stem.rsplitn(4, '-').collect();
    // parts = [NNNNN, "of", NNNNN, rest...]
    if parts.len() >= 3 && parts[1] == "of" {
        let _total: usize = parts[0].parse().ok()?;
        let _index: usize = parts[2].parse().ok()?;
        // Reconstruct prefix: everything before the shard numbering
        let shard_suffix_len = parts[0].len() + parts[1].len() + parts[2].len() + 3; // 3 dashes
        let prefix = &stem[..stem.len() - shard_suffix_len];
        Some(prefix.to_string())
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Metadata → ModelConfig
// ---------------------------------------------------------------------------

fn get_meta_str<'a>(
    meta: &'a HashMap<String, MetadataValue>,
    key: &str,
) -> Result<&'a str, MilError> {
    meta.get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| MilError::Validation(format!("GGUF: missing metadata key '{key}'")))
}

fn get_meta_usize(meta: &HashMap<String, MetadataValue>, key: &str) -> Result<usize, MilError> {
    meta.get(key)
        .and_then(|v| v.as_usize())
        .ok_or_else(|| MilError::Validation(format!("GGUF: missing metadata key '{key}'")))
}

fn get_meta_f64(meta: &HashMap<String, MetadataValue>, key: &str) -> Result<f64, MilError> {
    meta.get(key)
        .and_then(|v| v.as_f64())
        .ok_or_else(|| MilError::Validation(format!("GGUF: missing metadata key '{key}'")))
}

fn extract_model_config(meta: &HashMap<String, MetadataValue>) -> Result<ModelConfig, MilError> {
    let arch_str = get_meta_str(meta, "general.architecture")?;
    let arch = arch_str.to_lowercase();

    // Gemma 4 requires per-layer metadata (layer_types, rope_parameters,
    // global_head_dim) that GGUF does not encode. Reject early with a clear
    // message rather than silently producing a broken model.
    if arch == "gemma4" || arch == "gemma4_text" {
        return Err(MilError::Validation(
            "Gemma 4 is not supported via GGUF — use SafeTensors format. \
             GGUF lacks the per-layer attention metadata (layer_types, rope_parameters, \
             global_head_dim) required for correct compilation."
                .into(),
        ));
    }

    let architecture: Architecture = arch_str.parse()?;

    let hidden_size = get_meta_usize(meta, &format!("{arch}.embedding_length"))?;
    let intermediate_size = get_meta_usize(meta, &format!("{arch}.feed_forward_length"))?;
    let num_hidden_layers = get_meta_usize(meta, &format!("{arch}.block_count"))?;
    let num_attention_heads = get_meta_usize(meta, &format!("{arch}.attention.head_count"))?;
    let num_key_value_heads = get_meta_usize(meta, &format!("{arch}.attention.head_count_kv"))
        .unwrap_or(num_attention_heads);

    let rms_norm_eps =
        get_meta_f64(meta, &format!("{arch}.attention.layer_norm_rms_epsilon")).unwrap_or(1e-5);

    let rope_theta = get_meta_f64(meta, &format!("{arch}.rope.freq_base")).unwrap_or(10000.0);

    let max_position_embeddings =
        get_meta_usize(meta, &format!("{arch}.context_length")).unwrap_or(4096);

    // Vocab size: prefer metadata key, fall back to tokenizer token array length.
    let vocab_size = get_meta_usize(meta, &format!("{arch}.vocab_size")).or_else(|_| {
        meta.get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .ok_or_else(|| MilError::Validation("GGUF: cannot determine vocab_size".into()))
    })?;

    let head_dim = ModelConfig::default_head_dim(hidden_size, num_attention_heads);

    let tie_word_embeddings = meta
        .get(&format!("{arch}.tie_word_embeddings"))
        .or_else(|| meta.get("general.tie_word_embeddings"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    Ok(ModelConfig::new(architecture)
        .with_hidden_size(hidden_size)
        .with_intermediate_size(intermediate_size)
        .with_num_hidden_layers(num_hidden_layers)
        .with_num_attention_heads(num_attention_heads)
        .with_num_key_value_heads(num_key_value_heads)
        .with_head_dim(head_dim)
        .with_vocab_size(vocab_size)
        .with_max_position_embeddings(max_position_embeddings)
        .with_rms_norm_eps(rms_norm_eps)
        .with_rope_theta(rope_theta)
        .with_tie_word_embeddings(tie_word_embeddings)
        .with_extra(HashMap::new()))
}

// ---------------------------------------------------------------------------
// GGUF → HuggingFace tensor name remapping
// ---------------------------------------------------------------------------

/// Map a GGUF tensor name to the HuggingFace-canonical name used by
/// architecture templates.
fn remap_tensor_name(gguf_name: &str) -> String {
    // Static 1:1 mappings (no layer index)
    match gguf_name {
        "token_embd.weight" => return "model.embed_tokens.weight".into(),
        "output_norm.weight" => return "model.norm.weight".into(),
        "output.weight" => return "lm_head.weight".into(),
        _ => {}
    }

    // Layer-indexed mappings: blk.{n}.<suffix> → model.layers.{n}.<hf_suffix>
    if let Some(rest) = gguf_name.strip_prefix("blk.") {
        if let Some(dot_pos) = rest.find('.') {
            let layer_str = &rest[..dot_pos];
            let suffix = &rest[dot_pos + 1..];
            let hf_suffix = match suffix {
                "attn_q.weight" => "self_attn.q_proj.weight",
                "attn_k.weight" => "self_attn.k_proj.weight",
                "attn_v.weight" => "self_attn.v_proj.weight",
                "attn_output.weight" => "self_attn.o_proj.weight",
                "ffn_gate.weight" => "mlp.gate_proj.weight",
                "ffn_up.weight" => "mlp.up_proj.weight",
                "ffn_down.weight" => "mlp.down_proj.weight",
                "attn_norm.weight" => "input_layernorm.weight",
                "ffn_norm.weight" => "post_attention_layernorm.weight",
                // Pass through unknown suffixes
                other => other,
            };
            return format!("model.layers.{layer_str}.{hf_suffix}");
        }
    }

    // Fallback: pass through unrecognised names unchanged
    gguf_name.to_string()
}

// ---------------------------------------------------------------------------
// Dequantization
// ---------------------------------------------------------------------------

/// Compute total byte size for `num_elements` stored in the given GGML type.
fn tensor_byte_size(num_elements: usize, ggml_type: GgmlType) -> Result<usize, MilError> {
    let bs = ggml_type.block_size();
    let ts = ggml_type.type_size();
    if bs == 0 || ts == 0 {
        return Err(MilError::Validation(format!(
            "GGUF: cannot compute byte size for unsupported type {:?}",
            ggml_type
        )));
    }
    // num_elements must be a multiple of block_size
    if num_elements % bs != 0 {
        return Err(MilError::Validation(format!(
            "GGUF: element count {num_elements} is not a multiple of block size {bs} \
             for type {:?}",
            ggml_type
        )));
    }
    (num_elements / bs).checked_mul(ts).ok_or_else(|| {
        MilError::Validation(format!(
            "GGUF: tensor byte size overflow ({num_elements} elements, type {:?})",
            ggml_type
        ))
    })
}

/// Dequantize raw tensor bytes to FP16, returning the FP16 byte buffer.
fn dequantize_to_fp16(
    raw: &[u8],
    num_elements: usize,
    ggml_type: GgmlType,
) -> Result<Vec<u8>, MilError> {
    match ggml_type {
        GgmlType::F32 => dequant_f32_to_fp16(raw, num_elements),
        GgmlType::F16 => dequant_f16_passthrough(raw, num_elements),
        GgmlType::BF16 => dequant_bf16_to_fp16(raw, num_elements),
        GgmlType::Q4_0 => dequant_q4_0_to_fp16(raw, num_elements),
        GgmlType::Q4_1 => dequant_q4_1_to_fp16(raw, num_elements),
        GgmlType::Q5_0 => dequant_q5_0_to_fp16(raw, num_elements),
        GgmlType::Q5_1 => dequant_q5_1_to_fp16(raw, num_elements),
        GgmlType::Q6K => dequant_q6_k_to_fp16(raw, num_elements),
        GgmlType::Q8_0 => dequant_q8_0_to_fp16(raw, num_elements),
        other => Err(MilError::Validation(format!(
            "GGUF: dequantization not yet implemented for {other:?}. \
             Supported types: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q6_K, Q8_0"
        ))),
    }
}

/// F32 → FP16: read f32 values, convert each to f16.
fn dequant_f32_to_fp16(raw: &[u8], num_elements: usize) -> Result<Vec<u8>, MilError> {
    let expected_bytes = num_elements
        .checked_mul(4)
        .ok_or_else(|| MilError::Validation("GGUF: F32 dequant size overflow".into()))?;
    if raw.len() < expected_bytes {
        return Err(MilError::Validation(
            "GGUF: F32 tensor data too short".into(),
        ));
    }
    let out_capacity = num_elements
        .checked_mul(2)
        .ok_or_else(|| MilError::Validation("GGUF: F32 dequant output size overflow".into()))?;
    let mut out = Vec::with_capacity(out_capacity);
    for i in 0..num_elements {
        let off = i * 4;
        let val = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
        let h = f16::from_f32(val);
        out.extend_from_slice(&h.to_le_bytes());
    }
    Ok(out)
}

/// F16 → FP16: passthrough (just copy the bytes).
fn dequant_f16_passthrough(raw: &[u8], num_elements: usize) -> Result<Vec<u8>, MilError> {
    let expected = num_elements
        .checked_mul(2)
        .ok_or_else(|| MilError::Validation("GGUF: F16 dequant size overflow".into()))?;
    if raw.len() < expected {
        return Err(MilError::Validation(
            "GGUF: F16 tensor data too short".into(),
        ));
    }
    Ok(raw[..expected].to_vec())
}

/// BF16 → FP16: convert each bf16 value to f16.
///
/// bf16 has the same exponent range as f32 but only 8 mantissa bits.
/// We go through f32 as an intermediate: bf16 → f32 → f16.
fn dequant_bf16_to_fp16(raw: &[u8], num_elements: usize) -> Result<Vec<u8>, MilError> {
    let expected_bytes = num_elements
        .checked_mul(2)
        .ok_or_else(|| MilError::Validation("GGUF: BF16 dequant size overflow".into()))?;
    if raw.len() < expected_bytes {
        return Err(MilError::Validation(
            "GGUF: BF16 tensor data too short".into(),
        ));
    }
    let out_capacity = num_elements
        .checked_mul(2)
        .ok_or_else(|| MilError::Validation("GGUF: BF16 dequant output size overflow".into()))?;
    let mut out = Vec::with_capacity(out_capacity);
    for i in 0..num_elements {
        let off = i * 2;
        let bf16_bits = u16::from_le_bytes([raw[off], raw[off + 1]]);
        // bf16 → f32: place bf16 bits in the upper 16 bits of a 32-bit float
        let f32_val = f32::from_bits((bf16_bits as u32) << 16);
        let h = f16::from_f32(f32_val);
        out.extend_from_slice(&h.to_le_bytes());
    }
    Ok(out)
}

/// Q4_0 → FP16: each block is 32 values.
///
/// Block layout (18 bytes):
///   - 2 bytes: f16 scale factor (`d`)
///   - 16 bytes: 32 × 4-bit unsigned nibbles packed 2-per-byte (low nibble first)
///
/// Dequantization: `value = d * (nibble - 8)`
fn dequant_q4_0_to_fp16(raw: &[u8], num_elements: usize) -> Result<Vec<u8>, MilError> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 18;

    let num_blocks = num_elements / BLOCK_SIZE;
    let expected_bytes = num_blocks
        .checked_mul(BLOCK_BYTES)
        .ok_or_else(|| MilError::Validation("GGUF: Q4_0 dequant size overflow".into()))?;
    if raw.len() < expected_bytes {
        return Err(MilError::Validation(
            "GGUF: Q4_0 tensor data too short".into(),
        ));
    }

    let out_capacity = num_elements
        .checked_mul(2)
        .ok_or_else(|| MilError::Validation("GGUF: Q4_0 dequant output size overflow".into()))?;
    let mut out = Vec::with_capacity(out_capacity);

    for block_idx in 0..num_blocks {
        let block = &raw[block_idx * BLOCK_BYTES..];
        // Scale factor stored as f16
        let d = f16::from_le_bytes([block[0], block[1]]);
        let d_f32 = d.to_f32();
        let quant_data = &block[2..BLOCK_BYTES];

        for &byte in quant_data.iter().take(16) {
            // Low nibble → even element, high nibble → odd element
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;

            let v_lo = f16::from_f32(d_f32 * lo as f32);
            let v_hi = f16::from_f32(d_f32 * hi as f32);
            out.extend_from_slice(&v_lo.to_le_bytes());
            out.extend_from_slice(&v_hi.to_le_bytes());
        }
    }

    Ok(out)
}

/// Q8_0 → FP16: each block is 32 values.
///
/// Block layout (34 bytes):
///   - 2 bytes: f16 scale factor (`d`)
///   - 32 bytes: 32 × int8 values
///
/// Dequantization: `value = d * int8_value`
fn dequant_q8_0_to_fp16(raw: &[u8], num_elements: usize) -> Result<Vec<u8>, MilError> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 34;

    let num_blocks = num_elements / BLOCK_SIZE;
    let expected_bytes = num_blocks
        .checked_mul(BLOCK_BYTES)
        .ok_or_else(|| MilError::Validation("GGUF: Q8_0 dequant size overflow".into()))?;
    if raw.len() < expected_bytes {
        return Err(MilError::Validation(
            "GGUF: Q8_0 tensor data too short".into(),
        ));
    }

    let out_capacity = num_elements
        .checked_mul(2)
        .ok_or_else(|| MilError::Validation("GGUF: Q8_0 dequant output size overflow".into()))?;
    let mut out = Vec::with_capacity(out_capacity);

    for block_idx in 0..num_blocks {
        let block = &raw[block_idx * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]);
        let d_f32 = d.to_f32();
        let quant_data = &block[2..BLOCK_BYTES];

        for &q in quant_data {
            let val = d_f32 * (q as i8) as f32;
            let h = f16::from_f32(val);
            out.extend_from_slice(&h.to_le_bytes());
        }
    }

    Ok(out)
}

/// Q4_1 → FP16: each block is 32 values.
///
/// Block layout (20 bytes):
///   - 2 bytes: f16 scale factor (`d`)
///   - 2 bytes: f16 minimum value (`m`)
///   - 16 bytes: 32 × 4-bit unsigned nibbles packed 2-per-byte (low nibble first)
///
/// Dequantization: `value = d * nibble + m`
fn dequant_q4_1_to_fp16(raw: &[u8], num_elements: usize) -> Result<Vec<u8>, MilError> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 20;

    let num_blocks = num_elements / BLOCK_SIZE;
    let expected_bytes = num_blocks
        .checked_mul(BLOCK_BYTES)
        .ok_or_else(|| MilError::Validation("GGUF: Q4_1 dequant size overflow".into()))?;
    if raw.len() < expected_bytes {
        return Err(MilError::Validation(
            "GGUF: Q4_1 tensor data too short".into(),
        ));
    }

    let out_capacity = num_elements
        .checked_mul(2)
        .ok_or_else(|| MilError::Validation("GGUF: Q4_1 dequant output size overflow".into()))?;
    let mut out = Vec::with_capacity(out_capacity);

    for block_idx in 0..num_blocks {
        let block = &raw[block_idx * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let m = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let quant_data = &block[4..BLOCK_BYTES];

        for &byte in quant_data.iter().take(16) {
            let lo = (byte & 0x0F) as f32;
            let hi = ((byte >> 4) & 0x0F) as f32;

            let v_lo = f16::from_f32(d * lo + m);
            let v_hi = f16::from_f32(d * hi + m);
            out.extend_from_slice(&v_lo.to_le_bytes());
            out.extend_from_slice(&v_hi.to_le_bytes());
        }
    }

    Ok(out)
}

/// Q5_0 → FP16: each block is 32 values.
///
/// Block layout (22 bytes):
///   - 2 bytes: f16 scale factor (`d`)
///   - 4 bytes: 32 high bits (one per element, packed into 32-bit integer)
///   - 16 bytes: 32 × 4-bit low nibbles packed 2-per-byte (low nibble first)
///
/// Dequantization: `value = d * (((high_bit << 4) | low_nibble) - 16)`
fn dequant_q5_0_to_fp16(raw: &[u8], num_elements: usize) -> Result<Vec<u8>, MilError> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 22;

    let num_blocks = num_elements / BLOCK_SIZE;
    let expected_bytes = num_blocks
        .checked_mul(BLOCK_BYTES)
        .ok_or_else(|| MilError::Validation("GGUF: Q5_0 dequant size overflow".into()))?;
    if raw.len() < expected_bytes {
        return Err(MilError::Validation(
            "GGUF: Q5_0 tensor data too short".into(),
        ));
    }

    let out_capacity = num_elements
        .checked_mul(2)
        .ok_or_else(|| MilError::Validation("GGUF: Q5_0 dequant output size overflow".into()))?;
    let mut out = Vec::with_capacity(out_capacity);

    for block_idx in 0..num_blocks {
        let block = &raw[block_idx * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let quant_data = &block[6..BLOCK_BYTES];

        for j in 0..16 {
            let byte = quant_data[j];
            let lo_nibble = (byte & 0x0F) as i32;
            let hi_nibble = ((byte >> 4) & 0x0F) as i32;

            let lo_high = ((qh >> j) & 1) as i32;
            let hi_high = ((qh >> (j + 16)) & 1) as i32;

            let lo_val = d * ((lo_nibble | (lo_high << 4)) - 16) as f32;
            let hi_val = d * ((hi_nibble | (hi_high << 4)) - 16) as f32;

            out.extend_from_slice(&f16::from_f32(lo_val).to_le_bytes());
            out.extend_from_slice(&f16::from_f32(hi_val).to_le_bytes());
        }
    }

    Ok(out)
}

/// Q5_1 → FP16: each block is 32 values.
///
/// Block layout (24 bytes):
///   - 2 bytes: f16 scale factor (`d`)
///   - 2 bytes: f16 minimum value (`m`)
///   - 4 bytes: 32 high bits packed into a u32
///   - 16 bytes: 32 × 4-bit low nibbles packed 2-per-byte (low nibble first)
///
/// Dequantization: `value = d * ((high_bit << 4) | low_nibble) + m`
fn dequant_q5_1_to_fp16(raw: &[u8], num_elements: usize) -> Result<Vec<u8>, MilError> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 24;

    let num_blocks = num_elements / BLOCK_SIZE;
    let expected_bytes = num_blocks
        .checked_mul(BLOCK_BYTES)
        .ok_or_else(|| MilError::Validation("GGUF: Q5_1 dequant size overflow".into()))?;
    if raw.len() < expected_bytes {
        return Err(MilError::Validation(
            "GGUF: Q5_1 tensor data too short".into(),
        ));
    }

    let out_capacity = num_elements
        .checked_mul(2)
        .ok_or_else(|| MilError::Validation("GGUF: Q5_1 dequant output size overflow".into()))?;
    let mut out = Vec::with_capacity(out_capacity);

    for block_idx in 0..num_blocks {
        let block = &raw[block_idx * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let m = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let quant_data = &block[8..BLOCK_BYTES];

        for j in 0..16 {
            let byte = quant_data[j];
            let lo_nibble = (byte & 0x0F) as i32;
            let hi_nibble = ((byte >> 4) & 0x0F) as i32;

            let lo_high = ((qh >> j) & 1) as i32;
            let hi_high = ((qh >> (j + 16)) & 1) as i32;

            let lo_val = d * (lo_nibble | (lo_high << 4)) as f32 + m;
            let hi_val = d * (hi_nibble | (hi_high << 4)) as f32 + m;

            out.extend_from_slice(&f16::from_f32(lo_val).to_le_bytes());
            out.extend_from_slice(&f16::from_f32(hi_val).to_le_bytes());
        }
    }

    Ok(out)
}

/// Q6_K → FP16: K-quant 6-bit (256 elements per super-block, 210 bytes/block).
///
/// Super-block layout (210 bytes):
///   - 128 bytes: packed 6-bit quantized values (ql, lower 4 bits)
///   - 64 bytes:  packed high 2-bit values (qh)
///   - 16 bytes:  per-sub-block scales (int8, 16 sub-blocks of 16 elements)
///   - 2 bytes:   f16 super-block scale factor (`d`)
///
/// Dequantization: `value = d * scale_i * (q - 32)`
fn dequant_q6_k_to_fp16(raw: &[u8], num_elements: usize) -> Result<Vec<u8>, MilError> {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 210;

    let num_blocks = num_elements / BLOCK_SIZE;
    let expected_bytes = num_blocks
        .checked_mul(BLOCK_BYTES)
        .ok_or_else(|| MilError::Validation("GGUF: Q6_K dequant size overflow".into()))?;
    if raw.len() < expected_bytes {
        return Err(MilError::Validation(
            "GGUF: Q6_K tensor data too short".into(),
        ));
    }

    let out_capacity = num_elements
        .checked_mul(2)
        .ok_or_else(|| MilError::Validation("GGUF: Q6_K dequant output size overflow".into()))?;
    let mut out = Vec::with_capacity(out_capacity);

    for block_idx in 0..num_blocks {
        let block = &raw[block_idx * BLOCK_BYTES..];
        let ql = &block[0..128]; // lower 4 bits
        let qh = &block[128..192]; // upper 2 bits
        let scales = &block[192..208]; // 16 int8 scales
        let d = f16::from_le_bytes([block[208], block[209]]).to_f32();

        let mut fp16_buf = vec![0u8; BLOCK_SIZE * 2];

        for h in 0..2u32 {
            for l in 0..32u32 {
                let q1 = (ql[(h * 64 + l) as usize] & 0x0F) as i32
                    | ((((qh[(h * 32 + l) as usize] >> 0) & 3) as i32) << 4);
                let q2 = (ql[(h * 64 + l + 32) as usize] & 0x0F) as i32
                    | ((((qh[(h * 32 + l) as usize] >> 2) & 3) as i32) << 4);
                let q3 = ((ql[(h * 64 + l) as usize] >> 4) & 0x0F) as i32
                    | ((((qh[(h * 32 + l) as usize] >> 4) & 3) as i32) << 4);
                let q4 = ((ql[(h * 64 + l + 32) as usize] >> 4) & 0x0F) as i32
                    | ((((qh[(h * 32 + l) as usize] >> 6) & 3) as i32) << 4);

                let sc1 = scales[(h * 8 + l / 16) as usize] as i8;
                let sc2 = scales[(h * 8 + l / 16 + 2) as usize] as i8;
                let sc3 = scales[(h * 8 + l / 16 + 4) as usize] as i8;
                let sc4 = scales[(h * 8 + l / 16 + 6) as usize] as i8;

                let idx1 = (h * 128 + l) as usize;
                let idx2 = (h * 128 + l + 32) as usize;
                let idx3 = (h * 128 + l + 64) as usize;
                let idx4 = (h * 128 + l + 96) as usize;

                let v1 = d * (sc1 as f32) * (q1 - 32) as f32;
                let v2 = d * (sc2 as f32) * (q2 - 32) as f32;
                let v3 = d * (sc3 as f32) * (q3 - 32) as f32;
                let v4 = d * (sc4 as f32) * (q4 - 32) as f32;

                let base = idx1 * 2;
                fp16_buf[base..base + 2].copy_from_slice(&f16::from_f32(v1).to_le_bytes());
                let base = idx2 * 2;
                fp16_buf[base..base + 2].copy_from_slice(&f16::from_f32(v2).to_le_bytes());
                let base = idx3 * 2;
                fp16_buf[base..base + 2].copy_from_slice(&f16::from_f32(v3).to_le_bytes());
                let base = idx4 * 2;
                fp16_buf[base..base + 2].copy_from_slice(&f16::from_f32(v4).to_le_bytes());
            }
        }

        out.extend_from_slice(&fp16_buf);
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Q4_0 / Q8_0 repack (quantized passthrough)
// ---------------------------------------------------------------------------

/// (quant_data, scales, zero_point) triple returned by repack helpers.
type RepackResult = Result<(Vec<u8>, Vec<u8>, Vec<u8>), MilError>;

/// Repack Q4_0 blocks: separate packed nibbles from per-block scales.
///
/// Q4_0 blocks interleave scale + packed data. This function separates them
/// into contiguous buffers for the `AffineDequantize` layout.
///
/// Returns: (packed_nibble_data, scales_fp16, zero_point_fp16)
fn repack_q4_0(raw: &[u8], num_elements: usize) -> RepackResult {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 18;

    let num_blocks = num_elements / BLOCK_SIZE;
    let expected_bytes = num_blocks
        .checked_mul(BLOCK_BYTES)
        .ok_or_else(|| MilError::Validation("GGUF: Q4_0 repack size overflow".into()))?;
    if raw.len() < expected_bytes {
        return Err(MilError::Validation(
            "GGUF: Q4_0 tensor data too short for repack".into(),
        ));
    }

    // Each block has 16 bytes of packed nibbles (32 values, 2 per byte)
    let packed_capacity = num_blocks
        .checked_mul(16)
        .ok_or_else(|| MilError::Validation("GGUF: Q4_0 packed data size overflow".into()))?;
    let scales_capacity = num_blocks
        .checked_mul(2)
        .ok_or_else(|| MilError::Validation("GGUF: Q4_0 scales size overflow".into()))?;

    let mut packed_data = Vec::with_capacity(packed_capacity);
    let mut scales = Vec::with_capacity(scales_capacity);

    for block_idx in 0..num_blocks {
        let block = &raw[block_idx * BLOCK_BYTES..];
        // 2-byte FP16 scale
        scales.extend_from_slice(&block[0..2]);
        // 16 bytes of packed nibbles
        packed_data.extend_from_slice(&block[2..BLOCK_BYTES]);
    }

    // Q4_0 formula: value = d * (nibble - 8)
    // Symmetric quantization with implicit zero_point = 8.
    let zero_point = broadcast_fp16_constant(f16::from_f32(8.0), num_blocks);

    Ok((packed_data, scales, zero_point))
}

/// Repack Q8_0 blocks: separate int8 values from per-block scales.
///
/// Returns: (int8_data, scales_fp16, zero_point_fp16)
fn repack_q8_0(raw: &[u8], num_elements: usize) -> RepackResult {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 34;

    let num_blocks = num_elements / BLOCK_SIZE;
    let expected_bytes = num_blocks
        .checked_mul(BLOCK_BYTES)
        .ok_or_else(|| MilError::Validation("GGUF: Q8_0 repack size overflow".into()))?;
    if raw.len() < expected_bytes {
        return Err(MilError::Validation(
            "GGUF: Q8_0 tensor data too short for repack".into(),
        ));
    }

    let data_capacity = num_blocks
        .checked_mul(32)
        .ok_or_else(|| MilError::Validation("GGUF: Q8_0 data size overflow".into()))?;
    let scales_capacity = num_blocks
        .checked_mul(2)
        .ok_or_else(|| MilError::Validation("GGUF: Q8_0 scales size overflow".into()))?;

    let mut quant_data = Vec::with_capacity(data_capacity);
    let mut scales = Vec::with_capacity(scales_capacity);

    for block_idx in 0..num_blocks {
        let block = &raw[block_idx * BLOCK_BYTES..];
        // 2-byte FP16 scale
        scales.extend_from_slice(&block[0..2]);
        // 32 bytes of int8 values
        quant_data.extend_from_slice(&block[2..BLOCK_BYTES]);
    }

    // Q8_0 formula: value = d * int8_value (zero_point = 0)
    let zero_point = broadcast_fp16_constant(f16::from_f32(0.0), num_blocks);

    Ok((quant_data, scales, zero_point))
}

/// Create a contiguous buffer of `count` copies of an FP16 value.
fn broadcast_fp16_constant(value: f16, count: usize) -> Vec<u8> {
    let bytes = value.to_le_bytes();
    let mut buf = Vec::with_capacity(count * 2);
    for _ in 0..count {
        buf.extend_from_slice(&bytes);
    }
    buf
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Round `offset` up to the next multiple of `alignment`.
fn align_offset(offset: usize, alignment: usize) -> Result<usize, MilError> {
    let aligned = offset
        .checked_add(alignment - 1)
        .ok_or_else(|| MilError::Validation("GGUF: alignment offset overflow".into()))?;
    Ok(aligned & !(alignment - 1))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_offset() {
        assert_eq!(align_offset(0, 32).unwrap(), 0);
        assert_eq!(align_offset(1, 32).unwrap(), 32);
        assert_eq!(align_offset(31, 32).unwrap(), 32);
        assert_eq!(align_offset(32, 32).unwrap(), 32);
        assert_eq!(align_offset(33, 32).unwrap(), 64);
    }

    #[test]
    fn test_ggml_type_from_u32() {
        assert_eq!(GgmlType::from_u32(0).unwrap(), GgmlType::F32);
        assert_eq!(GgmlType::from_u32(1).unwrap(), GgmlType::F16);
        assert_eq!(GgmlType::from_u32(2).unwrap(), GgmlType::Q4_0);
        assert_eq!(GgmlType::from_u32(8).unwrap(), GgmlType::Q8_0);
        assert_eq!(GgmlType::from_u32(30).unwrap(), GgmlType::BF16);
        assert!(GgmlType::from_u32(999).is_err());
    }

    #[test]
    fn test_remap_tensor_name() {
        assert_eq!(
            remap_tensor_name("token_embd.weight"),
            "model.embed_tokens.weight"
        );
        assert_eq!(remap_tensor_name("output_norm.weight"), "model.norm.weight");
        assert_eq!(remap_tensor_name("output.weight"), "lm_head.weight");
        assert_eq!(
            remap_tensor_name("blk.0.attn_q.weight"),
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            remap_tensor_name("blk.15.ffn_gate.weight"),
            "model.layers.15.mlp.gate_proj.weight"
        );
        assert_eq!(
            remap_tensor_name("blk.3.attn_norm.weight"),
            "model.layers.3.input_layernorm.weight"
        );
        assert_eq!(
            remap_tensor_name("blk.7.ffn_norm.weight"),
            "model.layers.7.post_attention_layernorm.weight"
        );
        // Unknown names pass through
        assert_eq!(remap_tensor_name("custom.tensor"), "custom.tensor");
    }

    #[test]
    fn test_detect_shard_prefix() {
        assert_eq!(
            detect_shard_prefix("model-00001-of-00003.gguf"),
            Some("model".into())
        );
        assert_eq!(
            detect_shard_prefix("my-model-00002-of-00005.gguf"),
            Some("my-model".into())
        );
        assert_eq!(detect_shard_prefix("single.gguf"), None);
        assert_eq!(detect_shard_prefix("model.gguf"), None);
    }

    #[test]
    fn test_dequant_f32_to_fp16() {
        let values: Vec<f32> = vec![1.0, -2.5, 0.0, 3.125];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result = dequant_f32_to_fp16(&raw, 4).unwrap();
        assert_eq!(result.len(), 8); // 4 elements × 2 bytes
        // Verify first value
        let h = f16::from_le_bytes([result[0], result[1]]);
        assert!((h.to_f32() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_dequant_f16_passthrough() {
        let values: Vec<f16> = vec![f16::from_f32(1.0), f16::from_f32(-0.5), f16::from_f32(0.0)];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result = dequant_f16_passthrough(&raw, 3).unwrap();
        assert_eq!(result, raw);
    }

    #[test]
    fn test_dequant_bf16_to_fp16() {
        // BF16 for 1.0: same as f32 upper 16 bits → 0x3F80
        let bf16_one: u16 = 0x3F80;
        let raw: Vec<u8> = bf16_one.to_le_bytes().to_vec();
        let result = dequant_bf16_to_fp16(&raw, 1).unwrap();
        let h = f16::from_le_bytes([result[0], result[1]]);
        assert!((h.to_f32() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_dequant_q4_0_to_fp16() {
        // Build a single Q4_0 block (32 elements, 18 bytes)
        let d = f16::from_f32(0.5);
        let mut block = Vec::new();
        block.extend_from_slice(&d.to_le_bytes());
        // 16 bytes of nibbles: each byte = (hi << 4) | lo
        // Set all nibbles to 8 → dequant value = 0.5 * (8-8) = 0
        for _ in 0..16 {
            block.push(0x88); // lo=8, hi=8
        }
        let result = dequant_q4_0_to_fp16(&block, 32).unwrap();
        assert_eq!(result.len(), 64); // 32 × 2 bytes
        // All values should be 0
        for i in 0..32 {
            let h = f16::from_le_bytes([result[i * 2], result[i * 2 + 1]]);
            assert!(h.to_f32().abs() < 1e-6);
        }

        // Test with nibble = 12 → dequant = 0.5 * (12-8) = 2.0
        let mut block2 = Vec::new();
        block2.extend_from_slice(&d.to_le_bytes());
        for _ in 0..16 {
            block2.push(0xCC); // lo=12, hi=12
        }
        let result2 = dequant_q4_0_to_fp16(&block2, 32).unwrap();
        for i in 0..32 {
            let h = f16::from_le_bytes([result2[i * 2], result2[i * 2 + 1]]);
            assert!((h.to_f32() - 2.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_dequant_q8_0_to_fp16() {
        // Build a single Q8_0 block (32 elements, 34 bytes)
        let d = f16::from_f32(0.25);
        let mut block = Vec::new();
        block.extend_from_slice(&d.to_le_bytes());
        // 32 int8 values: all = 4 → dequant = 0.25 * 4 = 1.0
        for _ in 0..32 {
            block.push(4u8);
        }
        let result = dequant_q8_0_to_fp16(&block, 32).unwrap();
        assert_eq!(result.len(), 64);
        for i in 0..32 {
            let h = f16::from_le_bytes([result[i * 2], result[i * 2 + 1]]);
            assert!((h.to_f32() - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_tensor_byte_size() {
        assert_eq!(tensor_byte_size(32, GgmlType::Q4_0).unwrap(), 18);
        assert_eq!(tensor_byte_size(64, GgmlType::Q4_0).unwrap(), 36);
        assert_eq!(tensor_byte_size(32, GgmlType::Q8_0).unwrap(), 34);
        assert_eq!(tensor_byte_size(4, GgmlType::F32).unwrap(), 16);
        assert_eq!(tensor_byte_size(4, GgmlType::F16).unwrap(), 8);
        // Non-multiple of block size should error
        assert!(tensor_byte_size(33, GgmlType::Q4_0).is_err());
    }

    /// Build a minimal valid GGUF v3 file in memory and parse it.
    #[test]
    fn test_parse_minimal_gguf() {
        let gguf_bytes = build_test_gguf();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.gguf");
        std::fs::write(&path, &gguf_bytes).unwrap();

        let provider = GgufProvider::load(&path).unwrap();
        let config = provider.config();
        assert_eq!(config.architecture, Architecture::Llama);
        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.num_hidden_layers, 2);
        assert_eq!(config.num_attention_heads, 4);
        assert_eq!(config.vocab_size, 3);

        // Verify the test tensor was loaded and remapped
        assert!(provider.has_tensor("model.embed_tokens.weight"));
        let t = provider.tensor("model.embed_tokens.weight").unwrap();
        assert_eq!(t.shape, vec![3, 64]);
        assert_eq!(t.dtype, ScalarType::Float16);
    }

    /// Helper: build a minimal GGUF v3 binary with one F32 tensor.
    fn build_test_gguf() -> Vec<u8> {
        let mut buf = Vec::new();

        // --- Header ---
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes()); // magic
        buf.extend_from_slice(&GGUF_VERSION.to_le_bytes()); // version
        buf.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&8u64.to_le_bytes()); // metadata_kv_count

        // --- Metadata KV pairs ---
        // Helper closures
        let write_kv_string = |buf: &mut Vec<u8>, key: &str, val: &str| {
            // key
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            // type = 8 (string)
            buf.extend_from_slice(&8u32.to_le_bytes());
            // value
            buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
            buf.extend_from_slice(val.as_bytes());
        };

        let write_kv_u32 = |buf: &mut Vec<u8>, key: &str, val: u32| {
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            buf.extend_from_slice(&4u32.to_le_bytes()); // type = uint32
            buf.extend_from_slice(&val.to_le_bytes());
        };

        let write_kv_f32 = |buf: &mut Vec<u8>, key: &str, val: f32| {
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            buf.extend_from_slice(&6u32.to_le_bytes()); // type = float32
            buf.extend_from_slice(&val.to_le_bytes());
        };

        write_kv_string(&mut buf, "general.architecture", "llama");
        write_kv_u32(&mut buf, "llama.embedding_length", 64);
        write_kv_u32(&mut buf, "llama.feed_forward_length", 256);
        write_kv_u32(&mut buf, "llama.block_count", 2);
        write_kv_u32(&mut buf, "llama.attention.head_count", 4);
        write_kv_u32(&mut buf, "llama.attention.head_count_kv", 4);
        write_kv_f32(&mut buf, "llama.attention.layer_norm_rms_epsilon", 1e-5);

        // tokenizer.ggml.tokens as string array with 3 elements → vocab_size=3
        {
            let key = "tokenizer.ggml.tokens";
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            buf.extend_from_slice(&9u32.to_le_bytes()); // type = array
            buf.extend_from_slice(&8u32.to_le_bytes()); // element type = string
            buf.extend_from_slice(&3u64.to_le_bytes()); // count
            for tok in &["<s>", "</s>", "hello"] {
                buf.extend_from_slice(&(tok.len() as u64).to_le_bytes());
                buf.extend_from_slice(tok.as_bytes());
            }
        }

        // --- Tensor info ---
        // One tensor: "token_embd.weight", shape [3, 64], type F32, offset 0
        {
            let name = "token_embd.weight";
            buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
            buf.extend_from_slice(&3u64.to_le_bytes()); // dim 0
            buf.extend_from_slice(&64u64.to_le_bytes()); // dim 1
            buf.extend_from_slice(&0u32.to_le_bytes()); // type = F32
            buf.extend_from_slice(&0u64.to_le_bytes()); // offset
        }

        // --- Pad to alignment ---
        let header_end = buf.len();
        let data_start = align_offset(header_end, ALIGNMENT).unwrap();
        buf.resize(data_start, 0u8);

        // --- Tensor data: 3×64 F32 values ---
        let num_elements = 3 * 64;
        for i in 0..num_elements {
            let val = (i as f32) * 0.01;
            buf.extend_from_slice(&val.to_le_bytes());
        }

        buf
    }

    #[test]
    fn test_bin_reader_read_gguf_string() {
        let s = "hello";
        let mut data = Vec::new();
        data.extend_from_slice(&(s.len() as u64).to_le_bytes());
        data.extend_from_slice(s.as_bytes());
        let mut reader = BinReader::new(&data);
        assert_eq!(reader.read_gguf_string().unwrap(), "hello");
    }

    #[test]
    fn test_bin_reader_eof_error() {
        let data = [0u8; 2];
        let mut reader = BinReader::new(&data);
        assert!(reader.read_u32().is_err());
    }

    #[test]
    fn test_metadata_value_conversions() {
        let v = MetadataValue::UInt32(42);
        assert_eq!(v.as_u32(), Some(42));
        assert_eq!(v.as_usize(), Some(42));
        assert_eq!(v.as_f64(), Some(42.0));
        assert_eq!(v.as_str(), None);

        let v = MetadataValue::Str("test".into());
        assert_eq!(v.as_str(), Some("test"));
        assert_eq!(v.as_u32(), None);

        let v = MetadataValue::Float32(3.125);
        assert!((v.as_f64().unwrap() - 3.125).abs() < 0.001);

        let v = MetadataValue::Bool(true);
        assert_eq!(v.as_bool(), Some(true));

        let v = MetadataValue::Array(vec![MetadataValue::UInt8(1), MetadataValue::UInt8(2)]);
        assert_eq!(v.as_array().unwrap().len(), 2);
    }

    #[test]
    fn repack_q4_0_separates_scales_and_data() {
        // One Q4_0 block: 18 bytes = 2 (scale) + 16 (packed nibbles)
        let scale_bytes = f16::from_f32(0.5).to_le_bytes();
        let mut block = Vec::new();
        block.extend_from_slice(&scale_bytes);
        block.extend_from_slice(&[0x98; 16]); // nibbles: low=8, high=9

        let (packed, scales, zeros) = repack_q4_0(&block, 32).unwrap();
        assert_eq!(packed.len(), 16);
        assert_eq!(scales.len(), 2);
        assert_eq!(zeros.len(), 2); // one FP16 value for zero_point
        assert_eq!(&scales[..], &scale_bytes);
        assert_eq!(&packed[..], &[0x98; 16]);
    }

    #[test]
    fn repack_q8_0_separates_scales_and_data() {
        // One Q8_0 block: 34 bytes = 2 (scale) + 32 (int8)
        let scale_bytes = f16::from_f32(1.0).to_le_bytes();
        let mut block = Vec::new();
        block.extend_from_slice(&scale_bytes);
        block.extend_from_slice(&[42i8 as u8; 32]);

        let (data, scales, zeros) = repack_q8_0(&block, 32).unwrap();
        assert_eq!(data.len(), 32);
        assert_eq!(scales.len(), 2);
        assert_eq!(zeros.len(), 2);
        assert_eq!(&scales[..], &scale_bytes);
        // Verify zero_point is FP16(0.0)
        let zero_fp16 = f16::from_f32(0.0).to_le_bytes();
        assert_eq!(&zeros[..], &zero_fp16);
    }

    #[test]
    fn gemma4_gguf_rejected_with_clear_error() {
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            MetadataValue::Str("gemma4".into()),
        );
        let err = extract_model_config(&meta).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("not supported via GGUF"),
            "expected Gemma 4 rejection message, got: {msg}"
        );
    }

    #[test]
    fn gemma4_text_gguf_rejected_with_clear_error() {
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            MetadataValue::Str("Gemma4_Text".into()),
        );
        let err = extract_model_config(&meta).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("not supported via GGUF"),
            "expected Gemma 4 rejection message, got: {msg}"
        );
    }
}
