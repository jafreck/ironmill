//! Inference-side bundle manifest types and helpers.
//!
//! These types mirror the compile-crate's manifest types but are owned by
//! inference, so loading a `.ironml` bundle does not require pulling in the
//! full compile pipeline.

use half::f16;
use ironmill_core::ane::bundle::LmHeadKind;
use ironmill_iosurface::AneTensor;
use mil_rs::ir::ScalarType;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Manifest types (deserialized from manifest.json)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct BundleManifest {
    pub format_version: u32,
    pub model_type: BundleModelType,
    #[serde(default)]
    pub sub_programs: Vec<SubProgramManifest>,
    pub decode: Option<DecodeManifest>,
}

#[non_exhaustive]
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BundleModelType {
    Simple,
    Decode,
}

#[derive(Debug, Deserialize)]
pub struct SubProgramManifest {
    pub name: String,
    pub inputs: Vec<TensorDescriptorManifest>,
    pub outputs: Vec<TensorDescriptorManifest>,
    pub input_packing: Option<InputPackingManifest>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TensorDescriptorManifest {
    pub name: String,
    pub shape: [usize; 4],
    pub dtype: ScalarType,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InputPackingManifest {
    pub offsets: Vec<usize>,
    pub sizes: Vec<usize>,
}

#[derive(Debug, Deserialize)]
pub struct DecodeManifest {
    pub architecture: ArchitectureManifest,
    pub lm_head: LmHeadManifest,
    pub layers: Vec<LayerManifest>,
}

#[derive(Debug, Deserialize)]
pub struct ArchitectureManifest {
    pub vocab_size: usize,
    pub eos_tokens: Vec<u32>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub rope_theta: f64,
    pub max_seq_len: usize,
    pub qk_norm: bool,
}

#[derive(Debug, Deserialize)]
pub struct LmHeadManifest {
    pub kind: LmHeadKind,
    pub vocab_size: usize,
    pub hidden_size: usize,
    #[serde(default)]
    pub chunks: Vec<SubProgramManifest>,
}

#[derive(Debug, Deserialize)]
pub struct LayerManifest {
    pub index: usize,
    pub pre_attn: SubProgramManifest,
    pub post_attn: Option<SubProgramManifest>,
    pub fp16_attn: Option<SubProgramManifest>,
    pub cache_write_fused: bool,
    pub donor_compatible: bool,
}

impl TensorDescriptorManifest {
    /// Returns the scalar type of this tensor.
    pub fn scalar_type(&self) -> mil_rs::ir::ScalarType {
        self.dtype
    }
}

// ---------------------------------------------------------------------------
// Inference-side input packing
// ---------------------------------------------------------------------------

/// Inference-side input packing metadata (mirrors compile's InputPacking).
#[derive(Debug, Clone)]
pub struct InputPacking {
    pub offsets: Vec<usize>,
    pub sizes: Vec<usize>,
}

impl From<InputPackingManifest> for InputPacking {
    fn from(m: InputPackingManifest) -> Self {
        Self {
            offsets: m.offsets,
            sizes: m.sizes,
        }
    }
}

/// Convert core-crate InputPacking to inference-side InputPacking.
impl From<ironmill_core::ane::packing::InputPacking> for InputPacking {
    fn from(ip: ironmill_core::ane::packing::InputPacking) -> Self {
        Self {
            offsets: ip.offsets,
            sizes: ip.sizes,
        }
    }
}

// ---------------------------------------------------------------------------
// Packed input writer
// ---------------------------------------------------------------------------

/// Write multiple logical inputs into a single spatially-packed tensor.
///
/// Each input's data is written at its packing offset within the tensor's
/// spatial dimension. The tensor layout is NCHW: element (ch, s) is at
/// flat index `ch * total_s + s`.
pub fn write_packed_inputs(
    tensor: &mut AneTensor,
    inputs: &[&[f16]],
    packing: &InputPacking,
) -> anyhow::Result<()> {
    let [_, channels, _, total_s] = tensor.shape();
    let mut packed = vec![f16::ZERO; channels * total_s];
    for (i, data) in inputs.iter().enumerate() {
        if i >= packing.offsets.len() {
            break;
        }
        let offset = packing.offsets[i];
        let c = data.len().min(channels);
        for ch in 0..c {
            packed[ch * total_s + offset] = data[ch];
        }
    }
    Ok(tensor.write_f16(&packed)?)
}

// ---------------------------------------------------------------------------
// Bundle weight extraction
// ---------------------------------------------------------------------------

/// Size in bytes per element for a given scalar type string.
fn bytes_per_element(dtype: &str) -> usize {
    match dtype {
        "fp16" => 2,
        "fp32" | "int32" => 4,
        "int8" | "uint8" | "bool" => 1,
        other => {
            eprintln!("warning: unknown dtype '{other}', defaulting to fp16 (2 bytes)");
            2
        }
    }
}

/// Extract individual weight entries from a combined BLOBFILE blob by
/// parsing the MIL text for BLOBFILE references.
///
/// The bundle stores all weights for a sub-program in a single BLOBFILE.
/// The MIL text references individual weight paths with tensor shapes.
/// This function parses those references, computes sizes, and splits the
/// combined blob's data section into individual `(path, raw_data)` pairs
/// suitable for `device.compile()`.
pub fn extract_weight_entries_from_bundle(
    mil_text: &str,
    weight_blob: &[u8],
) -> Vec<(String, Vec<u8>)> {
    // BLOBFILE data starts at byte 128 in the blob format.
    const DATA_START: usize = 128;

    let raw_data = if weight_blob.len() > DATA_START {
        &weight_blob[DATA_START..]
    } else {
        return Vec::new();
    };

    // Parse MIL text for BLOBFILE references.
    // Pattern: val=tensor<DTYPE, [D0,D1,...,DN]>(BLOBFILE(path=string("PATH"), offset=uint64(64)))
    let mut entries = Vec::new();
    let mut offset = 0usize;

    // Find all BLOBFILE references in order of appearance.
    let mut search_start = 0;
    while let Some(blob_pos) = mil_text[search_start..].find("BLOBFILE(path=string(\"") {
        let abs_pos = search_start + blob_pos;

        // Extract path: BLOBFILE(path=string("PATH"), ...)
        let path_start = abs_pos + "BLOBFILE(path=string(\"".len();
        let path_end = match mil_text[path_start..].find("\")") {
            Some(e) => path_start + e,
            None => {
                search_start = path_start;
                continue;
            }
        };
        let path = mil_text[path_start..path_end].to_string();

        // Find the tensor type annotation before this BLOBFILE reference.
        // Look backwards from BLOBFILE for val=tensor<DTYPE, [DIMS]>(BLOBFILE
        // Pattern: tensor<DTYPE, [D0,D1,...,DN]>(BLOBFILE
        let before = &mil_text[..abs_pos];
        let tensor_size = if let Some(tensor_start) = before.rfind("val=tensor<") {
            parse_tensor_byte_size(&mil_text[tensor_start..abs_pos])
        } else if let Some(tensor_start) = before.rfind("tensor<") {
            parse_tensor_byte_size(&mil_text[tensor_start..abs_pos])
        } else {
            // Fallback: try parsing just "tensor<...>(" before BLOBFILE
            0
        };

        if tensor_size > 0 && offset + tensor_size <= raw_data.len() {
            entries.push((path, raw_data[offset..offset + tensor_size].to_vec()));
            offset += tensor_size;
        } else if tensor_size == 0 && !raw_data.is_empty() {
            // Single weight or couldn't parse size — use remaining data.
            entries.push((path, raw_data[offset..].to_vec()));
            break;
        }

        search_start = path_end;
    }

    // If no entries were found but we have data, this might be a single-weight
    // sub-program. Try to find any BLOBFILE path and use all data.
    if entries.is_empty() && !raw_data.is_empty() {
        if let Some(blob_pos) = mil_text.find("BLOBFILE(path=string(\"") {
            let path_start = blob_pos + "BLOBFILE(path=string(\"".len();
            if let Some(path_end_rel) = mil_text[path_start..].find("\")") {
                let path = mil_text[path_start..path_start + path_end_rel].to_string();
                entries.push((path, raw_data.to_vec()));
            }
        }
    }

    entries
}

/// Parse tensor byte size from a MIL type annotation fragment like
/// `val=tensor<fp16, [128,64,1,1]>(`.
fn parse_tensor_byte_size(fragment: &str) -> usize {
    // Extract dtype: tensor<DTYPE, [...]>
    let dtype_start = match fragment.find("tensor<") {
        Some(p) => p + "tensor<".len(),
        None => return 0,
    };
    let rest = &fragment[dtype_start..];
    let dtype_end = match rest.find(',') {
        Some(p) => p,
        None => return 0,
    };
    let dtype = rest[..dtype_end].trim();
    let bpe = bytes_per_element(dtype);

    // Extract shape: [D0,D1,...,DN]>
    let bracket_start = match rest.find('[') {
        Some(p) => p + 1,
        None => return 0,
    };
    let bracket_end = match rest[bracket_start..].find(']') {
        Some(p) => bracket_start + p,
        None => return 0,
    };
    let dims_str = &rest[bracket_start..bracket_end];
    let num_elements: usize = dims_str
        .split(',')
        .filter_map(|d| d.trim().parse::<usize>().ok())
        .product();

    num_elements * bpe
}
