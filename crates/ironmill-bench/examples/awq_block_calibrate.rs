//! AWQ block-level calibration: searches optimal alpha and clip ranges using
//! full transformer block forward evaluation on Metal GPU.
//!
//! For each layer, swaps in quantize-then-dequantized weights and measures
//! block-output MSE against the FP16 reference to pick the best alpha. Then
//! runs per-group clip search on the scaled weights.
//!
//! Usage:
//!   cargo run --release --example awq_block_calibrate --features metal -- \
//!       <model_dir> <dataset.json> <output_dir>

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use half::f16;
use serde_json;

use ironmill_compile::weights::quantized::{
    AwqTensorConfig, quantize_affine_into, search_clip_ranges,
};
use ironmill_compile::weights::{SafeTensorsProvider, WeightProvider};
use ironmill_inference::calibration::{ActivationHook, CalibrationDataset};
use ironmill_inference::engine::InferenceEngine;
use ironmill_inference::metal::{MetalConfig, MetalInference};
use mil_rs::ir::ScalarType;

// ── Custom hook: captures per-layer activation data ────────────

struct BlockCalibrationHook {
    /// Per-layer attn_norm activations as f32: layer_idx → [tokens × hidden_size].
    attn_norm_acts: HashMap<usize, Vec<f32>>,
    /// Per-layer ffn_norm activations as f32: layer_idx → [tokens × hidden_size].
    ffn_norm_acts: HashMap<usize, Vec<f32>>,
    /// Per-layer block output hidden states (FP16 bytes): layer_idx → bytes.
    block_outputs: HashMap<usize, Vec<u8>>,
}

impl BlockCalibrationHook {
    fn new() -> Self {
        Self {
            attn_norm_acts: HashMap::new(),
            ffn_norm_acts: HashMap::new(),
            block_outputs: HashMap::new(),
        }
    }
}

impl ActivationHook for BlockCalibrationHook {
    fn on_linear_input(
        &mut self,
        layer: usize,
        name: &str,
        activation: &[f16],
        _n_features: usize,
    ) {
        match name {
            "attn_norm" => {
                let f32_data: Vec<f32> = activation.iter().map(|v| v.to_f32()).collect();
                self.attn_norm_acts.insert(layer, f32_data);
            }
            "ffn_norm" => {
                let f32_data: Vec<f32> = activation.iter().map(|v| v.to_f32()).collect();
                self.ffn_norm_acts.insert(layer, f32_data);
            }
            "block_output" => {
                let bytes = bytemuck_cast_f16_to_bytes(activation);
                self.block_outputs.insert(layer, bytes);
            }
            _ => {}
        }
    }
}

/// Reinterpret `&[f16]` as raw bytes without `unsafe` in the example crate
/// (which has `#![deny(unsafe_code)]` in the main binary, but examples are
/// separate compilation units). We just do a simple copy.
fn bytemuck_cast_f16_to_bytes(data: &[f16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 2);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

// ── Weight loading helpers ─────────────────────────────────────

/// Projection names that live under `self_attn` in the HuggingFace naming.
const ATTN_PROJS: &[&str] = &["q_proj", "k_proj", "v_proj", "o_proj"];
/// Projection names that live under `mlp`.
const FFN_PROJS: &[&str] = &["gate_proj", "up_proj", "down_proj"];

/// Build the HuggingFace tensor name for a layer projection weight.
fn hf_weight_name(layer: usize, proj: &str) -> String {
    let block = if ATTN_PROJS.contains(&proj) {
        "self_attn"
    } else {
        "mlp"
    };
    format!("model.layers.{layer}.{block}.{proj}.weight")
}

/// Load a weight tensor as f32 and return (data, [out_features, in_features]).
fn load_weight_f32(
    provider: &SafeTensorsProvider,
    layer: usize,
    proj: &str,
) -> Result<(Vec<f32>, [usize; 2]), Box<dyn std::error::Error>> {
    let name = hf_weight_name(layer, proj);
    let tensor = provider.tensor(&name)?;
    let shape = &tensor.shape;
    assert!(
        shape.len() == 2,
        "Expected 2D weight for {name}, got {shape:?}"
    );
    let out_features = shape[0];
    let in_features = shape[1];

    let f32_data = match tensor.dtype {
        ScalarType::Float16 => {
            let f16_slice: &[u8] = &tensor.data;
            f16_slice
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        }
        ScalarType::Float32 => tensor
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        dt => return Err(format!("Unsupported dtype {dt:?} for {name}").into()),
    };

    Ok((f32_data, [out_features, in_features]))
}

// ── Quantize / dequantize round-trip ───────────────────────────

/// Quantize-then-dequantize a weight matrix with AWQ scaling applied.
///
/// Returns the dequantized f32 weights with the inverse scale applied,
/// i.e. the "approximated FP16 weights" that would result from quantization.
fn quantize_dequant_scaled(
    weights: &[f32],
    out_features: usize,
    in_features: usize,
    scales: &[f32],
    group_size: usize,
) -> Vec<f32> {
    let qmax = 15.0_f32; // INT4
    let n_groups = in_features.div_ceil(group_size);
    let mut result = vec![0.0f32; weights.len()];

    let mut quant_buf = vec![0u8; group_size];

    for row in 0..out_features {
        for g in 0..n_groups {
            let g_start = g * group_size;
            let g_end = (g_start + group_size).min(in_features);
            let gsize = g_end - g_start;

            // Scale weights
            let group_vals: Vec<f32> = (g_start..g_end)
                .map(|c| weights[row * in_features + c] * scales[c])
                .collect();

            // Quantize
            let (scale, zp) = quantize_affine_into(&group_vals, qmax, &mut quant_buf[..gsize]);

            // Dequantize and undo scaling
            for j in 0..gsize {
                let dequant = (quant_buf[j] as f32 - zp) * scale;
                let c = g_start + j;
                result[row * in_features + c] = dequant / scales[c];
            }
        }
    }

    result
}

/// Compute MSE between two FP16 byte buffers.
fn mse_f16_bytes(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() / 2;
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    for i in 0..n {
        let va = f16::from_le_bytes([a[i * 2], a[i * 2 + 1]]).to_f64();
        let vb = f16::from_le_bytes([b[i * 2], b[i * 2 + 1]]).to_f64();
        let d = va - vb;
        sum += d * d;
    }
    sum / n as f64
}

// ── Alpha search ───────────────────────────────────────────────

/// Compute AWQ scales from activation magnitudes and an alpha value.
///
/// Matches the reference AWQ normalization:
///   scales[c] = x_max[c]^alpha
///   scales /= sqrt(max(scales) * min(scales))
fn compute_awq_scales(x_max: &[f32], alpha: f32) -> Vec<f32> {
    if alpha == 0.0 {
        return vec![1.0; x_max.len()];
    }
    let mut scales: Vec<f32> = x_max.iter().map(|&m| m.powf(alpha).max(1e-4)).collect();
    let max_s = scales.iter().cloned().fold(0.0_f32, f32::max);
    let min_s = scales.iter().cloned().fold(f32::INFINITY, f32::min);
    let norm = (max_s * min_s).sqrt().max(1e-8);
    for s in &mut scales {
        *s /= norm;
    }
    scales
}

/// Compute per-channel mean absolute activation (x_max) from flat [tokens × features] data.
fn compute_channel_magnitudes(activations: &[f32], n_features: usize) -> Vec<f32> {
    if n_features == 0 || activations.is_empty() {
        return Vec::new();
    }
    let n_tokens = activations.len() / n_features;
    let mut mags = vec![0.0_f32; n_features];
    for t in 0..n_tokens {
        for c in 0..n_features {
            mags[c] += activations[t * n_features + c].abs();
        }
    }
    if n_tokens > 0 {
        for m in &mut mags {
            *m /= n_tokens as f32;
        }
    }
    mags
}

// ── Weight group definitions ──────────────────────────────────

struct WeightGroup {
    proj_names: Vec<&'static str>,
    norm_key: &'static str, // "attn" or "ffn"
}

fn weight_groups() -> Vec<WeightGroup> {
    vec![
        WeightGroup {
            proj_names: vec!["q_proj", "k_proj", "v_proj"],
            norm_key: "attn",
        },
        WeightGroup {
            proj_names: vec!["o_proj"],
            norm_key: "attn",
        },
        WeightGroup {
            proj_names: vec!["gate_proj", "up_proj"],
            norm_key: "ffn",
        },
        WeightGroup {
            proj_names: vec!["down_proj"],
            norm_key: "ffn",
        },
    ]
}

// ── Main ───────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <model_dir> <dataset.json> <output_dir>", args[0]);
        std::process::exit(1);
    }
    let model_dir = &args[1];
    let dataset_path = &args[2];
    let output_dir = &args[3];

    // ── Load model FP16 on Metal ──────────────────────────────
    eprintln!("Loading model from {model_dir}...");
    let config = MetalConfig::default()
        .with_max_seq_len(2048)
        .with_prefill_chunks(128)
        .without_turboquant();
    let mut engine = MetalInference::new(config.clone())?;
    let provider = SafeTensorsProvider::load(Path::new(model_dir))?;
    engine.load_weights(&provider, config)?;
    eprintln!(
        "Model loaded ({:.0} MB GPU)",
        engine.gpu_allocated_bytes() as f64 / 1e6
    );

    let mc = provider.config();
    let n_layers = mc.num_hidden_layers;
    let hidden_size = mc.hidden_size;
    let group_size = 128_usize;

    let dataset = CalibrationDataset::load(Path::new(dataset_path))?;
    eprintln!(
        "Dataset: {} seqs × {} tokens",
        dataset.num_sequences, dataset.seq_len
    );

    // ── Phase 1: FP16 forward pass — capture per-layer I/O ───
    eprintln!("\n=== Phase 1: FP16 calibration forward pass ===");
    let phase1_start = Instant::now();

    let mut hook = BlockCalibrationHook::new();

    // Use the calibration runner loop, running all sequences through the hook.
    for (i, seq) in dataset.sequences.iter().enumerate() {
        InferenceEngine::reset(&mut engine);
        ironmill_inference::calibration::CalibratingEngine::prefill_with_hooks(
            &mut engine,
            seq,
            &mut hook,
        )?;
        if (i + 1) % 4 == 0 || i + 1 == dataset.sequences.len() {
            eprintln!("[phase1] {}/{} sequences", i + 1, dataset.sequences.len());
        }
    }

    let n_tokens = hook
        .attn_norm_acts
        .values()
        .next()
        .map(|a| a.len() / hidden_size)
        .unwrap_or(0);
    eprintln!(
        "Phase 1 complete in {:.1}s — captured {} layers, {} tokens/layer",
        phase1_start.elapsed().as_secs_f64(),
        hook.block_outputs.len(),
        n_tokens,
    );

    // ── Phase 2: Block-level alpha search ─────────────────────
    eprintln!("\n=== Phase 2: Block-level alpha search ===");
    let phase2_start = Instant::now();

    let alpha_candidates: Vec<f32> = (0..=20).map(|i| i as f32 * 0.05).collect();
    let groups = weight_groups();

    let mut block_config: HashMap<String, AwqTensorConfig> = HashMap::new();
    let mut magnitudes_map: HashMap<String, Vec<f32>> = HashMap::new();

    // Layer 0: skip block-level search, use per-tensor fallback alpha=0.5
    for proj in ATTN_PROJS.iter().chain(FFN_PROJS.iter()) {
        let key = format!("l0_{proj}_weight");
        block_config.insert(
            key.clone(),
            AwqTensorConfig {
                alpha: 0.5,
                clip_maxvals: None,
            },
        );
    }

    // Compute and store magnitudes for layer 0
    if let Some(acts) = hook.attn_norm_acts.get(&0) {
        let mags = compute_channel_magnitudes(acts, hidden_size);
        for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            magnitudes_map.insert(format!("l0_{proj}_weight"), mags.clone());
        }
    }
    if let Some(acts) = hook.ffn_norm_acts.get(&0) {
        let mags = compute_channel_magnitudes(acts, hidden_size);
        for proj in &["gate_proj", "up_proj", "down_proj"] {
            magnitudes_map.insert(format!("l0_{proj}_weight"), mags.clone());
        }
    }

    // Layers 1..N: full block-level alpha search
    for layer_idx in 1..n_layers {
        let layer_start = Instant::now();

        // Block input = block_output[layer_idx - 1]
        let block_input = match hook.block_outputs.get(&(layer_idx - 1)) {
            Some(inp) => inp,
            None => {
                eprintln!("[layer {layer_idx}] WARNING: no block input, using fallback alpha=0.5");
                for proj in ATTN_PROJS.iter().chain(FFN_PROJS.iter()) {
                    let key = format!("l{layer_idx}_{proj}_weight");
                    block_config.insert(
                        key,
                        AwqTensorConfig {
                            alpha: 0.5,
                            clip_maxvals: None,
                        },
                    );
                }
                continue;
            }
        };

        // Reference block output
        let ref_output = match hook.block_outputs.get(&layer_idx) {
            Some(out) => out,
            None => {
                eprintln!(
                    "[layer {layer_idx}] WARNING: no reference output, using fallback alpha=0.5"
                );
                for proj in ATTN_PROJS.iter().chain(FFN_PROJS.iter()) {
                    let key = format!("l{layer_idx}_{proj}_weight");
                    block_config.insert(
                        key,
                        AwqTensorConfig {
                            alpha: 0.5,
                            clip_maxvals: None,
                        },
                    );
                }
                continue;
            }
        };

        let token_count = block_input.len() / (hidden_size * 2);

        // Compute magnitudes for this layer
        let attn_mags = hook
            .attn_norm_acts
            .get(&layer_idx)
            .map(|a| compute_channel_magnitudes(a, hidden_size))
            .unwrap_or_else(|| vec![1.0; hidden_size]);

        let ffn_mags = hook
            .ffn_norm_acts
            .get(&layer_idx)
            .map(|a| compute_channel_magnitudes(a, hidden_size))
            .unwrap_or_else(|| vec![1.0; hidden_size]);

        // Store magnitudes for output
        for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            magnitudes_map.insert(format!("l{layer_idx}_{proj}_weight"), attn_mags.clone());
        }
        for proj in &["gate_proj", "up_proj", "down_proj"] {
            magnitudes_map.insert(format!("l{layer_idx}_{proj}_weight"), ffn_mags.clone());
        }

        for group in &groups {
            let mags = if group.norm_key == "attn" {
                &attn_mags
            } else {
                &ffn_mags
            };

            // Pre-load FP16 weights for all projections in this group
            let mut proj_weights: Vec<(&str, Vec<f32>, [usize; 2])> = Vec::new();
            for &proj in &group.proj_names {
                let (w, shape) = load_weight_f32(&provider, layer_idx, proj)?;
                proj_weights.push((proj, w, shape));
            }

            // Determine which in_features dimension to use for scales.
            // All projections in a group share the same activation norm,
            // so their in_features should match the norm dimension.
            let in_features_for_scales = proj_weights[0].2[1];

            // Ensure magnitudes match the weight in_features dimension.
            // For o_proj and down_proj the in_features differ from hidden_size.
            let effective_mags = if mags.len() != in_features_for_scales {
                // Activations don't match weight dim — this projection uses a
                // different norm output size (e.g. o_proj has num_heads×head_dim).
                // Fall back to uniform magnitudes.
                vec![1.0_f32; in_features_for_scales]
            } else {
                mags.clone()
            };

            // Search alpha candidates — each can be evaluated independently.
            // We must use sequential evaluation because `run_single_layer` is
            // &mut self on the engine (not parallelizable across alphas).
            let mut best_alpha = 0.5_f32;
            let mut best_loss = f64::INFINITY;

            for &alpha in &alpha_candidates {
                let scales = compute_awq_scales(&effective_mags, alpha);

                // For each projection: quantize-dequant with these scales
                let mut all_ok = true;
                let mut swap_handles: Vec<(
                    &str,
                    Option<ironmill_inference::metal::weights::WeightBuffer>,
                )> = Vec::new();

                for &(proj, ref w, [out_f, in_f]) in &proj_weights {
                    // Scales may need adapting if in_features != scale length
                    let proj_scales = if scales.len() == in_f {
                        &scales
                    } else {
                        // Shouldn't happen given effective_mags, but guard
                        all_ok = false;
                        break;
                    };
                    let dq = quantize_dequant_scaled(w, out_f, in_f, proj_scales, group_size);
                    let buf = engine.create_dense_f16_buffer(&dq)?;
                    let original = engine.swap_layer_weight(layer_idx, proj, buf);
                    swap_handles.push((proj, original));
                }

                let loss = if all_ok {
                    // Run single layer forward
                    match engine.run_single_layer(layer_idx, block_input, token_count) {
                        Ok(output) => mse_f16_bytes(&output, ref_output),
                        Err(e) => {
                            eprintln!("[layer {layer_idx}] run_single_layer error: {e}");
                            f64::INFINITY
                        }
                    }
                } else {
                    f64::INFINITY
                };

                // Restore original weights
                for (proj, original) in swap_handles.into_iter().rev() {
                    if let Some(orig) = original {
                        engine.swap_layer_weight(layer_idx, proj, orig);
                    }
                }

                if loss < best_loss {
                    best_loss = loss;
                    best_alpha = alpha;
                }
            }

            for &proj in &group.proj_names {
                let key = format!("l{layer_idx}_{proj}_weight");
                block_config.insert(
                    key,
                    AwqTensorConfig {
                        alpha: best_alpha,
                        clip_maxvals: None,
                    },
                );
            }

            eprintln!(
                "  [layer {layer_idx}] {:?} → alpha={best_alpha:.2} (loss={best_loss:.2e})",
                group.proj_names,
            );
        }

        eprintln!(
            "[layer {layer_idx}/{n_layers}] done in {:.1}s",
            layer_start.elapsed().as_secs_f64(),
        );
    }

    eprintln!(
        "Phase 2 complete in {:.1}s",
        phase2_start.elapsed().as_secs_f64(),
    );

    // ── Phase 3: Clip search ──────────────────────────────────
    eprintln!("\n=== Phase 3: Clip search ===");
    let phase3_start = Instant::now();

    // Per reference AWQ: skip Q and K projections for clipping.
    let clip_projs: &[&str] = &["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"];

    for layer_idx in 0..n_layers {
        for &proj in clip_projs {
            let key = format!("l{layer_idx}_{proj}_weight");
            let alpha = block_config.get(&key).map(|c| c.alpha).unwrap_or(0.5);

            // Get activations for this projection's norm
            let norm_key = if ATTN_PROJS.contains(&proj) {
                "attn"
            } else {
                "ffn"
            };
            let activations = if norm_key == "attn" {
                hook.attn_norm_acts.get(&layer_idx)
            } else {
                hook.ffn_norm_acts.get(&layer_idx)
            };
            let activations = match activations {
                Some(a) => a,
                None => continue,
            };

            let (w, [out_features, in_features]) = load_weight_f32(&provider, layer_idx, proj)?;

            // Activations must match in_features; if not, skip clip for this proj
            if activations.len() < n_tokens * in_features {
                continue;
            }

            // Compute magnitudes for this projection's norm
            let mags = compute_channel_magnitudes(activations, in_features);
            let scales = compute_awq_scales(&mags, alpha);

            // Apply AWQ scaling to weights
            let scaled_weights: Vec<f32> = w
                .iter()
                .enumerate()
                .map(|(i, &val)| val * scales[i % in_features])
                .collect();

            // Run clip search
            let clip_maxvals = search_clip_ranges(
                &scaled_weights,
                out_features,
                in_features,
                group_size,
                15.0,
                activations,
                n_tokens,
                20,  // clip_grid
                0.5, // max_shrink
            );

            // Check if any clips are non-trivial (not all infinity)
            let has_clips = clip_maxvals.iter().any(|&v| v < f32::INFINITY);
            if has_clips {
                if let Some(cfg) = block_config.get_mut(&key) {
                    cfg.clip_maxvals = Some(clip_maxvals);
                }
            }
        }
        if (layer_idx + 1) % 4 == 0 || layer_idx + 1 == n_layers {
            eprintln!("[phase3] {}/{n_layers} layers clipped", layer_idx + 1);
        }
    }

    eprintln!(
        "Phase 3 complete in {:.1}s",
        phase3_start.elapsed().as_secs_f64(),
    );

    // ── Phase 4: Write output ─────────────────────────────────
    eprintln!("\n=== Phase 4: Writing output ===");
    std::fs::create_dir_all(output_dir)?;

    // 1. Block config JSON
    let config_path = Path::new(output_dir).join("awq_block_config.json");
    let config_json = serde_json::to_string_pretty(&block_config)?;
    std::fs::write(&config_path, config_json)?;
    eprintln!("Saved block config to {}", config_path.display());

    // 2. Magnitudes JSON (backward-compatible with existing calibration tool)
    let mag_path = Path::new(output_dir).join("awq_magnitudes.json");
    let mag_json = serde_json::to_string_pretty(&magnitudes_map)?;
    std::fs::write(&mag_path, mag_json)?;
    eprintln!("Saved magnitudes to {}", mag_path.display());

    // Summary
    let n_configs = block_config.len();
    let n_clipped = block_config
        .values()
        .filter(|c| c.clip_maxvals.is_some())
        .count();
    eprintln!(
        "\nDone: {n_configs} tensor configs ({n_clipped} with clip ranges) for {n_layers} layers"
    );

    Ok(())
}
