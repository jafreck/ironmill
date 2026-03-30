//! End-to-end integration test: GGUF → GgufProvider → template → MIL IR Program.
//!
//! Builds a synthetic GGUF v3 file with all required metadata and F16
//! tensors, loads it through `GgufProvider`, and converts to a MIL IR program.
//! No network access required.

use std::fs;

use ironmill_compile::templates::weights_to_program;
use ironmill_compile::weights::gguf::GgufProvider;
use ironmill_compile::weights::{Architecture, WeightProvider};

// ---------------------------------------------------------------------------
// Config constants (must match GGUF metadata)
// ---------------------------------------------------------------------------

const HIDDEN: usize = 32;
const INTERMEDIATE: usize = 64;
const NUM_LAYERS: usize = 2;
const NUM_HEADS: usize = 4;
const NUM_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const VOCAB: usize = 100;
const MAX_POS: usize = 64;

// ---------------------------------------------------------------------------
// GGUF binary builder
// ---------------------------------------------------------------------------

const GGUF_MAGIC: u32 = 0x4655_4747;
const GGUF_VERSION: u32 = 3;
const ALIGNMENT: usize = 32;
const GGML_TYPE_F16: u32 = 1;

struct GgufBuilder {
    buf: Vec<u8>,
}

impl GgufBuilder {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }

    fn write_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_gguf_string(&mut self, s: &str) {
        self.write_u64(s.len() as u64);
        self.buf.extend_from_slice(s.as_bytes());
    }

    /// Write a metadata KV pair: string value (type tag 8).
    fn write_kv_string(&mut self, key: &str, value: &str) {
        self.write_gguf_string(key);
        self.write_u32(8); // type = string
        self.write_gguf_string(value);
    }

    /// Write a metadata KV pair: uint32 value (type tag 4).
    fn write_kv_u32(&mut self, key: &str, value: u32) {
        self.write_gguf_string(key);
        self.write_u32(4); // type = uint32
        self.write_u32(value);
    }

    /// Write a metadata KV pair: float32 value (type tag 6).
    fn write_kv_f32(&mut self, key: &str, value: f32) {
        self.write_gguf_string(key);
        self.write_u32(6); // type = float32
        self.write_f32(value);
    }

    /// Write a tensor info entry. Returns the byte size for this tensor's data.
    fn write_tensor_info(
        &mut self,
        name: &str,
        dims: &[usize],
        ggml_type: u32,
        offset: u64,
    ) -> usize {
        self.write_gguf_string(name);
        self.write_u32(dims.len() as u32);
        for &d in dims {
            self.write_u64(d as u64);
        }
        self.write_u32(ggml_type);
        self.write_u64(offset);

        // F16: 2 bytes per element
        let n_elements: usize = dims.iter().product();
        n_elements * 2
    }

    /// Pad to the given alignment.
    fn align_to(&mut self, alignment: usize) {
        let rem = self.buf.len() % alignment;
        if rem != 0 {
            let pad = alignment - rem;
            self.buf.extend(std::iter::repeat(0u8).take(pad));
        }
    }

    fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

/// All tensors needed for a 2-layer LLaMA model, using GGUF naming.
fn gguf_tensor_specs() -> Vec<(&'static str, Vec<usize>)> {
    let mut tensors = Vec::new();

    tensors.push(("token_embd.weight", vec![VOCAB, HIDDEN]));

    for i in 0..NUM_LAYERS {
        let layer_tensors: Vec<(&str, Vec<usize>)> = vec![
            ("attn_q.weight", vec![NUM_HEADS * HEAD_DIM, HIDDEN]),
            ("attn_k.weight", vec![NUM_KV_HEADS * HEAD_DIM, HIDDEN]),
            ("attn_v.weight", vec![NUM_KV_HEADS * HEAD_DIM, HIDDEN]),
            ("attn_output.weight", vec![HIDDEN, NUM_HEADS * HEAD_DIM]),
            ("ffn_gate.weight", vec![INTERMEDIATE, HIDDEN]),
            ("ffn_up.weight", vec![INTERMEDIATE, HIDDEN]),
            ("ffn_down.weight", vec![HIDDEN, INTERMEDIATE]),
            ("attn_norm.weight", vec![HIDDEN]),
            ("ffn_norm.weight", vec![HIDDEN]),
        ];
        for (suffix, shape) in layer_tensors {
            tensors.push((
                // Leak the string so we get &'static str — fine for tests
                Box::leak(format!("blk.{i}.{suffix}").into_boxed_str()) as &'static str,
                shape,
            ));
        }
    }

    tensors.push(("output_norm.weight", vec![HIDDEN]));
    tensors.push(("output.weight", vec![VOCAB, HIDDEN]));

    tensors
}

/// Build a complete synthetic GGUF file as a byte vector.
fn build_gguf_bytes() -> Vec<u8> {
    let tensor_specs = gguf_tensor_specs();
    let metadata_kv_count: u64 = 10; // all required metadata keys

    let mut b = GgufBuilder::new();

    // --- Header ---
    b.write_u32(GGUF_MAGIC);
    b.write_u32(GGUF_VERSION);
    b.write_u64(tensor_specs.len() as u64);
    b.write_u64(metadata_kv_count);

    // --- Metadata KV pairs ---
    b.write_kv_string("general.architecture", "llama");
    b.write_kv_u32("llama.embedding_length", HIDDEN as u32);
    b.write_kv_u32("llama.feed_forward_length", INTERMEDIATE as u32);
    b.write_kv_u32("llama.block_count", NUM_LAYERS as u32);
    b.write_kv_u32("llama.attention.head_count", NUM_HEADS as u32);
    b.write_kv_u32("llama.attention.head_count_kv", NUM_KV_HEADS as u32);
    b.write_kv_f32("llama.attention.layer_norm_rms_epsilon", 1e-6);
    b.write_kv_f32("llama.rope.freq_base", 10000.0);
    b.write_kv_u32("llama.context_length", MAX_POS as u32);
    b.write_kv_u32("llama.vocab_size", VOCAB as u32);

    // --- Tensor info entries ---
    // Compute offsets: each tensor is placed sequentially (F16, 2 bytes/element)
    let mut data_offset: u64 = 0;
    let mut tensor_sizes = Vec::new();
    for (name, dims) in &tensor_specs {
        let byte_size = b.write_tensor_info(name, dims, GGML_TYPE_F16, data_offset);
        tensor_sizes.push(byte_size);
        data_offset += byte_size as u64;
    }

    // --- Alignment padding before tensor data section ---
    b.align_to(ALIGNMENT);

    // --- Tensor data (all zeros, F16) ---
    for size in &tensor_sizes {
        b.buf.extend(std::iter::repeat(0u8).take(*size));
    }

    b.into_bytes()
}

/// Write a synthetic GGUF file into `dir` and return the file path.
fn write_gguf_file(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("model.gguf");
    let bytes = build_gguf_bytes();
    fs::write(&path, bytes).unwrap();
    path
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn gguf_provider_loads_config_correctly() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_gguf_file(dir.path());

    let provider = GgufProvider::load(&path).expect("load should succeed");
    let config = provider.config();

    assert_eq!(config.architecture, Architecture::Llama);
    assert_eq!(config.hidden_size, HIDDEN);
    assert_eq!(config.intermediate_size, INTERMEDIATE);
    assert_eq!(config.num_hidden_layers, NUM_LAYERS);
    assert_eq!(config.num_attention_heads, NUM_HEADS);
    assert_eq!(config.num_key_value_heads, NUM_KV_HEADS);
}

#[test]
fn gguf_provider_exposes_remapped_tensor_names() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_gguf_file(dir.path());

    let provider = GgufProvider::load(&path).unwrap();

    // GGUF names should be remapped to HuggingFace-canonical names.
    assert!(
        provider.has_tensor("model.embed_tokens.weight"),
        "token_embd.weight should map to model.embed_tokens.weight"
    );
    assert!(
        provider.has_tensor("model.layers.0.self_attn.q_proj.weight"),
        "blk.0.attn_q.weight should map to model.layers.0.self_attn.q_proj.weight"
    );
    assert!(
        provider.has_tensor("model.layers.1.mlp.down_proj.weight"),
        "blk.1.ffn_down.weight should map to model.layers.1.mlp.down_proj.weight"
    );
    assert!(
        provider.has_tensor("model.norm.weight"),
        "output_norm.weight should map to model.norm.weight"
    );
    assert!(
        provider.has_tensor("lm_head.weight"),
        "output.weight should map to lm_head.weight"
    );
}

#[test]
fn gguf_weights_to_program_produces_valid_ir() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_gguf_file(dir.path());

    let provider = GgufProvider::load(&path).unwrap();

    let result = weights_to_program(&provider).expect("weights_to_program should succeed");

    let main = result
        .program
        .main()
        .expect("program should have a main function");

    assert!(
        !main.body.operations.is_empty(),
        "main function body should have operations"
    );
    assert!(
        !main.body.outputs.is_empty(),
        "main function should have outputs"
    );

    let op_types: Vec<&str> = main
        .body
        .operations
        .iter()
        .map(|op| op.op_type.as_str())
        .collect();

    assert!(
        op_types.contains(&"const"),
        "program should contain const ops, got: {op_types:?}"
    );
    assert!(
        op_types.contains(&"linear"),
        "program should contain linear ops, got: {op_types:?}"
    );
    assert!(
        op_types.contains(&"rms_norm"),
        "program should contain rms_norm ops, got: {op_types:?}"
    );
    assert!(
        op_types.contains(&"add"),
        "program should contain add ops, got: {op_types:?}"
    );
}

#[test]
fn gguf_program_is_marked_autoregressive() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_gguf_file(dir.path());

    let provider = GgufProvider::load(&path).unwrap();
    let result = weights_to_program(&provider).unwrap();

    assert!(
        result.program.is_autoregressive(),
        "LLaMA program should be marked autoregressive"
    );
}

#[test]
fn gguf_config_has_correct_vocab_size() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_gguf_file(dir.path());

    let provider = GgufProvider::load(&path).expect("provider should load");
    assert_eq!(
        provider.config().vocab_size,
        VOCAB,
        "vocab_size should match the metadata value"
    );
}
