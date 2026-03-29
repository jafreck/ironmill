//! Criterion benchmarks for SafeTensors and GGUF model loading + template conversion.
//!
//! Creates synthetic model data on disk, then benchmarks the load → template → pipeline
//! path for both formats. Dimensions are larger than the e2e tests to yield
//! meaningful timings.

use std::fs;

use criterion::{Criterion, criterion_group, criterion_main};
use safetensors::Dtype;
use safetensors::tensor::TensorView;
use tempfile::TempDir;

use mil_rs::PassPipeline;
use mil_rs::convert::templates::weights_to_program;
use mil_rs::convert::weights::gguf::GgufProvider;
use mil_rs::convert::weights::safetensors::SafeTensorsProvider;

// ---------------------------------------------------------------------------
// Model dimensions — bigger than the e2e tests for meaningful timings
// ---------------------------------------------------------------------------

const HIDDEN: usize = 64;
const INTERMEDIATE: usize = 128;
const NUM_LAYERS: usize = 4;
const NUM_HEADS: usize = 8;
const NUM_KV_HEADS: usize = 4;
const HEAD_DIM: usize = 8; // HIDDEN / NUM_HEADS
const VOCAB: usize = 256;
const MAX_POS: usize = 128;

// ---------------------------------------------------------------------------
// SafeTensors fixture helpers
// ---------------------------------------------------------------------------

fn config_json() -> String {
    serde_json::json!({
        "model_type": "llama",
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": NUM_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "vocab_size": VOCAB,
        "max_position_embeddings": MAX_POS,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "head_dim": HEAD_DIM,
        "tie_word_embeddings": false
    })
    .to_string()
}

fn zeros_f16(shape: &[usize]) -> Vec<u8> {
    let n: usize = shape.iter().product();
    vec![0u8; n * 2]
}

fn build_llama_tensors() -> Vec<(String, Vec<u8>, Vec<usize>)> {
    let mut tensors = Vec::new();

    let mut add = |name: &str, shape: &[usize]| {
        tensors.push((name.to_string(), zeros_f16(shape), shape.to_vec()));
    };

    add("model.embed_tokens.weight", &[VOCAB, HIDDEN]);

    for i in 0..NUM_LAYERS {
        let p = format!("model.layers.{i}");
        add(
            &format!("{p}.self_attn.q_proj.weight"),
            &[NUM_HEADS * HEAD_DIM, HIDDEN],
        );
        add(
            &format!("{p}.self_attn.k_proj.weight"),
            &[NUM_KV_HEADS * HEAD_DIM, HIDDEN],
        );
        add(
            &format!("{p}.self_attn.v_proj.weight"),
            &[NUM_KV_HEADS * HEAD_DIM, HIDDEN],
        );
        add(
            &format!("{p}.self_attn.o_proj.weight"),
            &[HIDDEN, NUM_HEADS * HEAD_DIM],
        );
        add(
            &format!("{p}.mlp.gate_proj.weight"),
            &[INTERMEDIATE, HIDDEN],
        );
        add(&format!("{p}.mlp.up_proj.weight"), &[INTERMEDIATE, HIDDEN]);
        add(
            &format!("{p}.mlp.down_proj.weight"),
            &[HIDDEN, INTERMEDIATE],
        );
        add(&format!("{p}.input_layernorm.weight"), &[HIDDEN]);
        add(&format!("{p}.post_attention_layernorm.weight"), &[HIDDEN]);
    }

    add("model.norm.weight", &[HIDDEN]);
    add("lm_head.weight", &[VOCAB, HIDDEN]);

    tensors
}

fn create_safetensors_fixture() -> TempDir {
    let dir = TempDir::new().expect("create temp dir");

    fs::write(dir.path().join("config.json"), config_json()).unwrap();

    let raw = build_llama_tensors();
    let views: Vec<(String, TensorView<'_>)> = raw
        .iter()
        .map(|(name, data, shape)| {
            let tv = TensorView::new(Dtype::F16, shape.clone(), data).unwrap();
            (name.clone(), tv)
        })
        .collect();

    let bytes = safetensors::serialize(views, None).unwrap();
    fs::write(dir.path().join("model.safetensors"), bytes).unwrap();

    dir
}

// ---------------------------------------------------------------------------
// GGUF fixture helpers
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

    fn write_kv_string(&mut self, key: &str, value: &str) {
        self.write_gguf_string(key);
        self.write_u32(8);
        self.write_gguf_string(value);
    }

    fn write_kv_u32(&mut self, key: &str, value: u32) {
        self.write_gguf_string(key);
        self.write_u32(4);
        self.write_u32(value);
    }

    fn write_kv_f32(&mut self, key: &str, value: f32) {
        self.write_gguf_string(key);
        self.write_u32(6);
        self.write_f32(value);
    }

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

        let n_elements: usize = dims.iter().product();
        n_elements * 2 // F16: 2 bytes per element
    }

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

fn gguf_tensor_specs() -> Vec<(String, Vec<usize>)> {
    let mut tensors = Vec::new();

    tensors.push(("token_embd.weight".to_string(), vec![VOCAB, HIDDEN]));

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
            tensors.push((format!("blk.{i}.{suffix}"), shape));
        }
    }

    tensors.push(("output_norm.weight".to_string(), vec![HIDDEN]));
    tensors.push(("output.weight".to_string(), vec![VOCAB, HIDDEN]));

    tensors
}

fn build_gguf_bytes() -> Vec<u8> {
    let tensor_specs = gguf_tensor_specs();
    let metadata_kv_count: u64 = 10;

    let mut b = GgufBuilder::new();

    // Header
    b.write_u32(GGUF_MAGIC);
    b.write_u32(GGUF_VERSION);
    b.write_u64(tensor_specs.len() as u64);
    b.write_u64(metadata_kv_count);

    // Metadata
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

    // Tensor info entries
    let mut data_offset: u64 = 0;
    let mut tensor_sizes = Vec::new();
    for (name, dims) in &tensor_specs {
        let byte_size = b.write_tensor_info(name, dims, GGML_TYPE_F16, data_offset);
        tensor_sizes.push(byte_size);
        data_offset += byte_size as u64;
    }

    // Alignment padding before tensor data
    b.align_to(ALIGNMENT);

    // Tensor data (all zeros)
    for size in &tensor_sizes {
        b.buf.extend(std::iter::repeat(0u8).take(*size));
    }

    b.into_bytes()
}

fn create_gguf_fixture() -> TempDir {
    let dir = TempDir::new().expect("create temp dir");
    let bytes = build_gguf_bytes();
    fs::write(dir.path().join("model.gguf"), bytes).unwrap();
    dir
}

// ---------------------------------------------------------------------------
// SafeTensors benchmarks
// ---------------------------------------------------------------------------

fn bench_safetensors_load(c: &mut Criterion) {
    let dir = create_safetensors_fixture();
    c.bench_function("safetensors/load", |b| {
        b.iter(|| {
            SafeTensorsProvider::load(dir.path()).expect("load");
        });
    });
}

fn bench_safetensors_to_program(c: &mut Criterion) {
    let dir = create_safetensors_fixture();
    let provider = SafeTensorsProvider::load(dir.path()).expect("load");
    c.bench_function("safetensors/to_program", |b| {
        b.iter(|| {
            weights_to_program(&provider).expect("template");
        });
    });
}

fn bench_safetensors_pipeline(c: &mut Criterion) {
    let dir = create_safetensors_fixture();
    let provider = SafeTensorsProvider::load(dir.path()).expect("load");
    c.bench_function("safetensors/full_pipeline_fp16", |b| {
        b.iter(|| {
            let result = weights_to_program(&provider).expect("template");
            let mut prog = result.program;
            PassPipeline::new()
                .with_fp16()
                .expect("fp16")
                .run(&mut prog)
                .expect("pipeline");
        });
    });
}

// ---------------------------------------------------------------------------
// GGUF benchmarks
// ---------------------------------------------------------------------------

fn bench_gguf_load(c: &mut Criterion) {
    let dir = create_gguf_fixture();
    let gguf_path = dir.path().join("model.gguf");
    c.bench_function("gguf/load", |b| {
        b.iter(|| {
            GgufProvider::load(&gguf_path).expect("load");
        });
    });
}

fn bench_gguf_to_program(c: &mut Criterion) {
    let dir = create_gguf_fixture();
    let gguf_path = dir.path().join("model.gguf");
    let provider = GgufProvider::load(&gguf_path).expect("load");
    c.bench_function("gguf/to_program", |b| {
        b.iter(|| {
            weights_to_program(&provider).expect("template");
        });
    });
}

fn bench_gguf_pipeline(c: &mut Criterion) {
    let dir = create_gguf_fixture();
    let gguf_path = dir.path().join("model.gguf");
    let provider = GgufProvider::load(&gguf_path).expect("load");
    c.bench_function("gguf/full_pipeline_fp16", |b| {
        b.iter(|| {
            let result = weights_to_program(&provider).expect("template");
            let mut prog = result.program;
            PassPipeline::new()
                .with_fp16()
                .expect("fp16")
                .run(&mut prog)
                .expect("pipeline");
        });
    });
}

// ---------------------------------------------------------------------------
// Cross-format comparison
// ---------------------------------------------------------------------------

fn bench_format_comparison(c: &mut Criterion) {
    let st_dir = create_safetensors_fixture();
    let gguf_dir = create_gguf_fixture();
    let gguf_path = gguf_dir.path().join("model.gguf");

    let mut group = c.benchmark_group("format_load");
    group.bench_function("safetensors", |b| {
        b.iter(|| SafeTensorsProvider::load(st_dir.path()).expect("load"));
    });
    group.bench_function("gguf", |b| {
        b.iter(|| GgufProvider::load(&gguf_path).expect("load"));
    });
    group.finish();

    let st_provider = SafeTensorsProvider::load(st_dir.path()).expect("load");
    let gguf_provider = GgufProvider::load(&gguf_path).expect("load");

    let mut group = c.benchmark_group("format_to_program");
    group.bench_function("safetensors", |b| {
        b.iter(|| weights_to_program(&st_provider).expect("template"));
    });
    group.bench_function("gguf", |b| {
        b.iter(|| weights_to_program(&gguf_provider).expect("template"));
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_safetensors_load,
    bench_safetensors_to_program,
    bench_safetensors_pipeline,
    bench_gguf_load,
    bench_gguf_to_program,
    bench_gguf_pipeline,
    bench_format_comparison,
);
criterion_main!(benches);
