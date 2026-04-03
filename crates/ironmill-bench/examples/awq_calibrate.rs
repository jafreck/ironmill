//! AWQ calibration runner: collects activation magnitudes for AWQ quantization.
//!
//! Usage:
//!   cargo run --release --example awq_calibrate --features metal -- \
//!       <model_dir> <dataset.json> <output_dir>

use ironmill_compile::weights::{SafeTensorsProvider, WeightProvider};
use ironmill_inference::calibration::{CalibrationDataset, CalibrationRunner};
use ironmill_inference::engine::InferenceEngine;
use ironmill_inference::metal::{MetalConfig, MetalInference};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <model_dir> <dataset.json> <output_dir>", args[0]);
        std::process::exit(1);
    }
    let model_dir = &args[1];
    let dataset_path = &args[2];
    let output_dir = &args[3];

    eprintln!("Loading model from {model_dir}...");
    let config = MetalConfig {
        max_seq_len: 2048,
        prefill_chunk_size: Some(128),
        enable_turboquant: false,
        ..MetalConfig::default()
    };
    let mut engine = MetalInference::new(config.clone())?;
    let provider = SafeTensorsProvider::load(std::path::Path::new(model_dir))?;
    engine.load_weights(&provider, config)?;
    eprintln!("Model loaded.");

    let dataset = CalibrationDataset::load(std::path::Path::new(dataset_path))?;
    eprintln!(
        "Dataset: {} seqs × {} tokens",
        dataset.num_sequences, dataset.seq_len
    );

    let mut runner = CalibrationRunner::new();
    runner.max_sequences = Some(dataset.num_sequences);
    let store = runner.collect_awq_stats(&mut engine, &dataset)?;

    let mc = provider.config();
    let n_layers = mc.num_hidden_layers;
    let mut weight_names: Vec<(String, usize, &str)> = Vec::new();
    for i in 0..n_layers {
        for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            weight_names.push((format!("l{i}_{proj}_weight"), i, "attn"));
        }
        for proj in &["gate_proj", "up_proj", "down_proj"] {
            weight_names.push((format!("l{i}_{proj}_weight"), i, "ffn"));
        }
    }

    let magnitudes = store.to_channel_magnitudes(&weight_names);
    eprintln!("Collected magnitudes for {} projections", magnitudes.len());

    let activations = store.to_activations(&weight_names);
    let activation_token_count = activations
        .values()
        .next()
        .map(|a| {
            let n_features = magnitudes.values().next().map(|m| m.len()).unwrap_or(1);
            if n_features > 0 {
                a.len() / n_features
            } else {
                0
            }
        })
        .unwrap_or(0);
    eprintln!(
        "Collected activations for {} projections ({} tokens)",
        activations.len(),
        activation_token_count
    );

    std::fs::create_dir_all(output_dir)?;
    let out_path = std::path::Path::new(output_dir).join("awq_magnitudes.json");
    let json = serde_json::to_string_pretty(&magnitudes)?;
    std::fs::write(&out_path, json)?;
    eprintln!("Saved magnitudes to {}", out_path.display());

    if !activations.is_empty() {
        let act_path = std::path::Path::new(output_dir).join("awq_activations.json");
        let act_json = serde_json::to_string(&activations)?;
        std::fs::write(&act_path, act_json)?;
        eprintln!("Saved activations to {}", act_path.display());

        let tc_path = std::path::Path::new(output_dir).join("awq_token_count.json");
        let tc_json = serde_json::to_string_pretty(&activation_token_count)?;
        std::fs::write(&tc_path, tc_json)?;
        eprintln!("Saved token count to {}", tc_path.display());
    }

    Ok(())
}
