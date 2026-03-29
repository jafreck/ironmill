use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::{Deserialize, Serialize};

/// A benchmark run is the cartesian product of models × optimizations × backends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchMatrix {
    pub models: Vec<ModelConfig>,
    pub optimizations: Vec<OptConfig>,
    pub backends: Vec<String>,
    pub settings: Settings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub path: PathBuf,
    #[serde(default)]
    pub input_shapes: Vec<(String, Vec<usize>)>,
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
    #[serde(default)]
    pub no_fusion: bool,
    #[serde(default)]
    pub disabled_passes: Vec<String>,
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
    200
}
fn default_warmup() -> usize {
    20
}
fn default_runs() -> usize {
    3
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            iterations: 200,
            warmup: 20,
            runs: 3,
            backends: vec!["all".to_string()],
        }
    }
}

/// Intermediate struct matching the TOML array-of-tables layout.
#[derive(Deserialize)]
struct ConfigFile {
    model: Vec<ModelConfig>,
    optimization: Vec<OptConfig>,
    #[serde(default)]
    settings: Settings,
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
    })
}

/// Build the default benchmark matrix (matches bench-inference.sh).
pub fn default_matrix() -> BenchMatrix {
    let models = vec![
        ModelConfig {
            name: "MobileNetV2".to_string(),
            path: PathBuf::from("tests/fixtures/mobilenetv2.onnx"),
            input_shapes: vec![],
        },
        ModelConfig {
            name: "SqueezeNet".to_string(),
            path: PathBuf::from("tests/fixtures/squeezenet1.1.onnx"),
            input_shapes: vec![],
        },
        ModelConfig {
            name: "Whisper-tiny-encoder".to_string(),
            path: PathBuf::from("tests/fixtures/whisper-tiny-encoder.onnx"),
            input_shapes: vec![],
        },
        ModelConfig {
            name: "DistilBERT".to_string(),
            path: PathBuf::from("tests/fixtures/distilbert.onnx"),
            input_shapes: vec![],
        },
        ModelConfig {
            name: "ViT-base".to_string(),
            path: PathBuf::from("tests/fixtures/vit-base.onnx"),
            input_shapes: vec![],
        },
        ModelConfig {
            name: "Qwen3-0.6B".to_string(),
            path: PathBuf::from("tests/fixtures/qwen3-0.6b.onnx"),
            input_shapes: vec![],
        },
    ];

    let optimizations = vec![
        OptConfig {
            name: "baseline".to_string(),
            quantize: None,
            palettize: None,
            polar_quantize: None,
            no_fusion: true,
            disabled_passes: vec![],
        },
        OptConfig {
            name: "default".to_string(),
            quantize: None,
            palettize: None,
            polar_quantize: None,
            no_fusion: false,
            disabled_passes: vec![],
        },
        OptConfig {
            name: "fp16".to_string(),
            quantize: Some("fp16".to_string()),
            palettize: None,
            polar_quantize: None,
            no_fusion: false,
            disabled_passes: vec![],
        },
        OptConfig {
            name: "int8".to_string(),
            quantize: Some("int8".to_string()),
            palettize: None,
            polar_quantize: None,
            no_fusion: false,
            disabled_passes: vec![],
        },
        OptConfig {
            name: "palettize-4".to_string(),
            quantize: None,
            palettize: Some(4),
            polar_quantize: None,
            no_fusion: false,
            disabled_passes: vec![],
        },
        OptConfig {
            name: "polar-4".to_string(),
            quantize: None,
            palettize: None,
            polar_quantize: Some(4),
            no_fusion: false,
            disabled_passes: vec![],
        },
        OptConfig {
            name: "polar-3".to_string(),
            quantize: None,
            palettize: None,
            polar_quantize: Some(3),
            no_fusion: false,
            disabled_passes: vec![],
        },
    ];

    BenchMatrix {
        models,
        optimizations,
        backends: vec!["all".to_string()],
        settings: Settings::default(),
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

    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_matrix() {
        let m = default_matrix();
        assert_eq!(m.models.len(), 6);
        assert_eq!(m.optimizations.len(), 7);
        assert_eq!(m.backends, vec!["all"]);
        assert_eq!(m.settings.iterations, 200);
        assert_eq!(m.settings.warmup, 20);
        assert_eq!(m.settings.runs, 3);
    }

    #[test]
    fn test_default_matrix_model_names() {
        let m = default_matrix();
        assert_eq!(m.models[0].name, "MobileNetV2");
        assert_eq!(m.models[1].name, "SqueezeNet");
        assert_eq!(m.models[2].name, "Whisper-tiny-encoder");
        assert_eq!(m.models[3].name, "DistilBERT");
        assert_eq!(m.models[4].name, "ViT-base");
        assert_eq!(m.models[5].name, "Qwen3-0.6B");
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
                "polar-3"
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
        };
        let opt = OptConfig {
            name: "baseline".to_string(),
            quantize: None,
            palettize: None,
            polar_quantize: None,
            no_fusion: true,
            disabled_passes: vec![],
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
        };
        let opt1 = OptConfig {
            name: "baseline".to_string(),
            quantize: None,
            palettize: None,
            polar_quantize: None,
            no_fusion: true,
            disabled_passes: vec![],
        };
        let opt2 = OptConfig {
            name: "fp16".to_string(),
            quantize: Some("fp16".to_string()),
            palettize: None,
            polar_quantize: None,
            no_fusion: false,
            disabled_passes: vec![],
        };
        assert_ne!(cache_key(&model, &opt1), cache_key(&model, &opt2));
    }

    #[test]
    fn test_settings_default() {
        let s = Settings::default();
        assert_eq!(s.iterations, 200);
        assert_eq!(s.warmup, 20);
        assert_eq!(s.runs, 3);
        assert_eq!(s.backends, vec!["all"]);
    }
}
