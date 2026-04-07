use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::{Deserialize, Serialize};

/// KV cache quantization strategy for TurboQuant benchmarks.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum KvQuantMode {
    #[default]
    None,
    TurboInt4,
    TurboInt8,
    TurboInt8Qjl,
}

impl std::fmt::Display for KvQuantMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvQuantMode::None => write!(f, "none"),
            KvQuantMode::TurboInt4 => write!(f, "turbo-int4"),
            KvQuantMode::TurboInt8 => write!(f, "turbo-int8"),
            KvQuantMode::TurboInt8Qjl => write!(f, "turbo-int8-qjl"),
        }
    }
}

/// A benchmark run is the cartesian product of models × optimizations × backends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchMatrix {
    pub models: Vec<ModelConfig>,
    pub optimizations: Vec<OptConfig>,
    pub backends: Vec<String>,
    pub settings: Settings,
    #[serde(default)]
    pub benchmarks: BenchmarkSelection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub path: PathBuf,
    #[serde(default)]
    pub input_shapes: Vec<(String, Vec<usize>)>,
    /// Optional HuggingFace model directory for template-based compilation.
    /// When set, the benchmark loads weights from safetensors files in this
    /// directory and builds MIL IR using the architecture template instead
    /// of parsing an ONNX file.
    #[serde(default)]
    pub model_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptConfig {
    pub name: String,
    #[serde(default)]
    pub quantize: Option<String>,
    #[serde(default)]
    pub palettize: Option<u8>,
    #[serde(default)]
    pub polar_quantize: Option<u8>,
    /// D2Quant dual-scale quantization bit-width (2 or 3).
    #[serde(default)]
    pub d2quant: Option<u8>,
    /// JIT INT4 affine per-group weight quantization.
    #[serde(default)]
    pub int4: bool,
    /// AWQ calibration directory containing awq_magnitudes.json.
    /// When set with int4=true, applies activation-aware scaling.
    #[serde(default)]
    pub awq_calib_dir: Option<String>,
    #[serde(default)]
    pub no_fusion: bool,
    #[serde(default)]
    pub disabled_passes: Vec<String>,
    /// KV cache quantization strategy
    #[serde(default)]
    pub kv_quant: KvQuantMode,
    /// Maximum sequence length for KV cache
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    #[serde(default = "default_iterations")]
    pub iterations: usize,
    #[serde(default = "default_warmup")]
    pub warmup: usize,
    #[serde(default = "default_runs")]
    pub runs: usize,
    #[serde(default)]
    pub backends: Vec<String>,
}

fn default_iterations() -> usize {
    1
}
fn default_warmup() -> usize {
    0
}
fn default_runs() -> usize {
    1
}
fn default_max_seq_len() -> usize {
    128
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            iterations: default_iterations(),
            warmup: default_warmup(),
            runs: default_runs(),
            backends: vec!["all".to_string()],
        }
    }
}

/// Which benchmark suites to run. Selectable from TOML config.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BenchmarkSelection {
    /// Suite IDs to run (e.g. ["decode", "prefill", "quality"]).
    /// Empty means "run all applicable suites".
    #[serde(default)]
    pub suites: Vec<String>,
    /// Prefill lengths for the prefill suite.
    #[serde(default)]
    pub prefill_lengths: Vec<usize>,
    /// Context lengths for the context-decode suite.
    #[serde(default)]
    pub context_lengths: Vec<usize>,
    /// Enable perplexity evaluation.
    #[serde(default)]
    pub perplexity: bool,
    /// Number of sequences for perplexity.
    #[serde(default = "default_perplexity_sequences")]
    pub perplexity_sequences: usize,
    /// Perplexity stride.
    #[serde(default = "default_perplexity_stride")]
    pub perplexity_stride: usize,
    /// Path to perplexity dataset.
    #[serde(default = "default_perplexity_dataset")]
    pub perplexity_dataset: String,
    /// Enable weight fidelity quality benchmarks.
    #[serde(default)]
    pub quality: bool,
}

fn default_perplexity_sequences() -> usize {
    50
}
fn default_perplexity_stride() -> usize {
    512
}
fn default_perplexity_dataset() -> String {
    "tests/fixtures/quality/wikitext2-qwen3.json".to_string()
}

/// Intermediate struct matching the TOML array-of-tables layout.
#[derive(Deserialize)]
struct ConfigFile {
    model: Vec<ModelConfig>,
    optimization: Vec<OptConfig>,
    #[serde(default)]
    settings: Settings,
    #[serde(default)]
    benchmarks: BenchmarkSelection,
}

/// Load a benchmark matrix from a TOML config file.
pub fn load_config(path: &Path) -> anyhow::Result<BenchMatrix> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read config: {}", path.display()))?;
    let file: ConfigFile = toml::from_str(&content)
        .with_context(|| format!("failed to parse config: {}", path.display()))?;

    let backends = if file.settings.backends.is_empty() {
        vec!["all".to_string()]
    } else {
        file.settings.backends.clone()
    };

    Ok(BenchMatrix {
        models: file.model,
        optimizations: file.optimization,
        backends,
        settings: file.settings,
        benchmarks: file.benchmarks,
    })
}

/// Build the default benchmark matrix — fast smoke-test config.
///
/// Uses a single small LLM (Qwen3-0.6B) with short sequence length for
/// quick iteration. Pass `--config` with a TOML file for full-suite runs.
pub fn default_matrix() -> BenchMatrix {
    let models = vec![ModelConfig {
        name: "Qwen3-0.6B".to_string(),
        path: PathBuf::from("tests/fixtures/qwen3-0.6b.onnx"),
        input_shapes: vec![],
        model_dir: None,
    }];

    let optimizations = vec![
        OptConfig {
            name: "baseline".to_string(),
            quantize: None,
            palettize: None,
            polar_quantize: None,
            d2quant: None,
            int4: false,
            no_fusion: true,
            disabled_passes: vec![],
            kv_quant: KvQuantMode::None,
            max_seq_len: default_max_seq_len(),
            awq_calib_dir: None,
        },
        OptConfig {
            name: "default".to_string(),
            quantize: None,
            palettize: None,
            polar_quantize: None,
            d2quant: None,
            int4: false,
            no_fusion: false,
            disabled_passes: vec![],
            kv_quant: KvQuantMode::None,
            max_seq_len: default_max_seq_len(),
            awq_calib_dir: None,
        },
        OptConfig {
            name: "fp16".to_string(),
            quantize: Some("fp16".to_string()),
            palettize: None,
            polar_quantize: None,
            d2quant: None,
            int4: false,
            no_fusion: false,
            disabled_passes: vec![],
            kv_quant: KvQuantMode::None,
            max_seq_len: default_max_seq_len(),
            awq_calib_dir: None,
        },
        OptConfig {
            name: "int8".to_string(),
            quantize: Some("int8".to_string()),
            palettize: None,
            polar_quantize: None,
            d2quant: None,
            int4: false,
            no_fusion: false,
            disabled_passes: vec![],
            kv_quant: KvQuantMode::None,
            max_seq_len: default_max_seq_len(),
            awq_calib_dir: None,
        },
        OptConfig {
            name: "palettize-4".to_string(),
            quantize: None,
            palettize: Some(4),
            polar_quantize: None,
            d2quant: None,
            int4: false,
            no_fusion: false,
            disabled_passes: vec![],
            kv_quant: KvQuantMode::None,
            max_seq_len: default_max_seq_len(),
            awq_calib_dir: None,
        },
        OptConfig {
            name: "polar-4".to_string(),
            quantize: None,
            palettize: None,
            polar_quantize: Some(4),
            d2quant: None,
            int4: false,
            no_fusion: false,
            disabled_passes: vec![],
            kv_quant: KvQuantMode::None,
            max_seq_len: default_max_seq_len(),
            awq_calib_dir: None,
        },
        OptConfig {
            name: "polar-3".to_string(),
            quantize: None,
            palettize: None,
            polar_quantize: Some(3),
            d2quant: None,
            int4: false,
            no_fusion: false,
            disabled_passes: vec![],
            kv_quant: KvQuantMode::None,
            max_seq_len: default_max_seq_len(),
            awq_calib_dir: None,
        },
        // TurboQuant INT8
        OptConfig {
            name: "turbo-int8".to_string(),
            quantize: None,
            palettize: None,
            polar_quantize: None,
            d2quant: None,
            int4: false,
            no_fusion: false,
            disabled_passes: vec![],
            kv_quant: KvQuantMode::TurboInt8,
            max_seq_len: default_max_seq_len(),
            awq_calib_dir: None,
        },
        // TurboQuant INT8 + QJL
        OptConfig {
            name: "turbo-int8-qjl".to_string(),
            quantize: None,
            palettize: None,
            polar_quantize: None,
            d2quant: None,
            int4: false,
            no_fusion: false,
            disabled_passes: vec![],
            kv_quant: KvQuantMode::TurboInt8Qjl,
            max_seq_len: default_max_seq_len(),
            awq_calib_dir: None,
        },
    ];

    BenchMatrix {
        models,
        optimizations,
        backends: vec!["all".to_string()],
        settings: Settings::default(),
        benchmarks: BenchmarkSelection::default(),
    }
}

/// Generate a cache key for a (model, optimization) pair.
///
/// Hashes the model path file name, file metadata (size + mtime when available),
/// and the serialized optimization config. Returns a hex-encoded hash.
pub fn cache_key(model: &ModelConfig, opt: &OptConfig) -> String {
    let mut hasher = DefaultHasher::new();

    // Hash model file name
    if let Some(name) = model.path.file_name() {
        name.hash(&mut hasher);
    }
    model.path.hash(&mut hasher);

    // Hash file metadata if the file exists
    if let Ok(meta) = fs::metadata(&model.path) {
        meta.len().hash(&mut hasher);
        if let Ok(mtime) = meta.modified() {
            mtime.hash(&mut hasher);
        }
    }

    // Hash input shapes (different shapes produce different compiled models)
    if let Ok(shapes_json) = serde_json::to_string(&model.input_shapes) {
        shapes_json.hash(&mut hasher);
    }

    // Hash serialized optimization config
    if let Ok(opt_json) = serde_json::to_string(opt) {
        opt_json.hash(&mut hasher);
    }

    let mut key = format!("{:016x}", hasher.finish());

    // Include KV quant mode explicitly for readable cache directory names
    if matches!(
        opt.kv_quant,
        KvQuantMode::TurboInt8 | KvQuantMode::TurboInt8Qjl
    ) {
        key.push_str(&format!("_kv-{}_seq-{}", opt.kv_quant, opt.max_seq_len));
    }

    key
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_matrix() {
        let m = default_matrix();
        assert_eq!(m.models.len(), 1);
        assert_eq!(m.optimizations.len(), 9);
        assert_eq!(m.backends, vec!["all"]);
        assert_eq!(m.settings.iterations, 1);
        assert_eq!(m.settings.warmup, 0);
        assert_eq!(m.settings.runs, 1);
    }

    #[test]
    fn test_default_matrix_model_names() {
        let m = default_matrix();
        assert_eq!(m.models[0].name, "Qwen3-0.6B");
    }

    #[test]
    fn test_default_matrix_opt_names() {
        let m = default_matrix();
        let names: Vec<&str> = m.optimizations.iter().map(|o| o.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "baseline",
                "default",
                "fp16",
                "int8",
                "palettize-4",
                "polar-4",
                "polar-3",
                "turbo-int8",
                "turbo-int8-qjl",
            ]
        );
    }

    #[test]
    fn test_load_config_roundtrip() {
        let toml_content = r#"
[[model]]
name = "TestModel"
path = "test.onnx"

[[optimization]]
name = "baseline"
no_fusion = true

[[optimization]]
name = "fp16"
quantize = "fp16"

[settings]
iterations = 100
warmup = 10
runs = 5
backends = ["cpu", "gpu"]
"#;
        let file: ConfigFile = toml::from_str(toml_content).unwrap();
        assert_eq!(file.model.len(), 1);
        assert_eq!(file.model[0].name, "TestModel");
        assert_eq!(file.optimization.len(), 2);
        assert!(file.optimization[0].no_fusion);
        assert_eq!(file.optimization[1].quantize.as_deref(), Some("fp16"));
        assert_eq!(file.settings.iterations, 100);
        assert_eq!(file.settings.warmup, 10);
        assert_eq!(file.settings.runs, 5);
        assert_eq!(file.settings.backends, vec!["cpu", "gpu"]);
    }

    #[test]
    fn test_cache_key_deterministic() {
        let model = ModelConfig {
            name: "test".to_string(),
            path: PathBuf::from("nonexistent.onnx"),
            input_shapes: vec![],
            model_dir: None,
        };
        let opt = OptConfig {
            name: "baseline".to_string(),
            quantize: None,
            palettize: None,
            polar_quantize: None,
            d2quant: None,
            int4: false,
            no_fusion: true,
            disabled_passes: vec![],
            kv_quant: KvQuantMode::None,
            max_seq_len: default_max_seq_len(),
            awq_calib_dir: None,
        };
        let key1 = cache_key(&model, &opt);
        let key2 = cache_key(&model, &opt);
        assert_eq!(key1, key2);
        assert_eq!(key1.len(), 16); // 16 hex chars
    }

    #[test]
    fn test_cache_key_varies_with_opt() {
        let model = ModelConfig {
            name: "test".to_string(),
            path: PathBuf::from("nonexistent.onnx"),
            input_shapes: vec![],
            model_dir: None,
        };
        let opt1 = OptConfig {
            name: "baseline".to_string(),
            quantize: None,
            palettize: None,
            polar_quantize: None,
            d2quant: None,
            int4: false,
            no_fusion: true,
            disabled_passes: vec![],
            kv_quant: KvQuantMode::None,
            max_seq_len: default_max_seq_len(),
            awq_calib_dir: None,
        };
        let opt2 = OptConfig {
            name: "fp16".to_string(),
            quantize: Some("fp16".to_string()),
            palettize: None,
            polar_quantize: None,
            d2quant: None,
            int4: false,
            no_fusion: false,
            disabled_passes: vec![],
            kv_quant: KvQuantMode::None,
            max_seq_len: default_max_seq_len(),
            awq_calib_dir: None,
        };
        assert_ne!(cache_key(&model, &opt1), cache_key(&model, &opt2));
    }

    #[test]
    fn test_cache_key_turbo_quant_suffix() {
        let model = ModelConfig {
            name: "test".to_string(),
            path: PathBuf::from("nonexistent.onnx"),
            input_shapes: vec![],
            model_dir: None,
        };
        let opt = OptConfig {
            name: "turbo-int8".to_string(),
            quantize: None,
            palettize: None,
            polar_quantize: None,
            d2quant: None,
            int4: false,
            no_fusion: false,
            disabled_passes: vec![],
            kv_quant: KvQuantMode::TurboInt8,
            max_seq_len: 4096,
        };
        let key = cache_key(&model, &opt);
        assert!(key.contains("_kv-turbo-int8_seq-4096"));
    }

    #[test]
    fn test_kv_quant_mode_display() {
        assert_eq!(KvQuantMode::None.to_string(), "none");
        assert_eq!(KvQuantMode::TurboInt8.to_string(), "turbo-int8");
        assert_eq!(KvQuantMode::TurboInt8Qjl.to_string(), "turbo-int8-qjl");
    }

    #[test]
    fn test_kv_quant_mode_default_deserialize() {
        let toml_content = r#"
[[model]]
name = "TestModel"
path = "test.onnx"

[[optimization]]
name = "baseline"
no_fusion = true
"#;
        let file: ConfigFile = toml::from_str(toml_content).unwrap();
        assert_eq!(file.optimization[0].kv_quant, KvQuantMode::None);
        assert_eq!(file.optimization[0].max_seq_len, 128);
    }

    #[test]
    fn test_settings_default() {
        let s = Settings::default();
        assert_eq!(s.iterations, 1);
        assert_eq!(s.warmup, 0);
        assert_eq!(s.runs, 1);
        assert_eq!(s.backends, vec!["all"]);
    }
}
