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

use ironmill_compile::weights::calibration::{
    ATTN_PROJS, AwqTensorConfig, FFN_PROJS, compute_awq_scales, compute_channel_magnitudes,
    quantize_dequant_scaled, search_clip_ranges, weight_groups,
};
use ironmill_compile::weights::{SafeTensorsProvider, WeightProvider};
use ironmill_inference::calibration::{ActivationHook, CalibrationDataset};
use ironmill_inference::engine::InferenceEngine;
use ironmill_inference::metal::{MetalConfig, MetalInference};
use mil_rs::ir::ScalarType;

// ── Custom hook: captures per-layer activation data ────────────

struct BlockCalibrationHook {
    /// Per-layer running mean of |activation| per channel (attn_norm).
    /// Accumulated across all calibration sequences per reference AWQ.
    attn_norm_mags: HashMap<usize, ChannelMagAccum>,
    /// Per-layer running mean of |activation| per channel (ffn_norm).
    ffn_norm_mags: HashMap<usize, ChannelMagAccum>,
    /// Per-layer raw activations from the LAST sequence only, for clip search.
    attn_norm_acts_last: HashMap<usize, Vec<f32>>,
    ffn_norm_acts_last: HashMap<usize, Vec<f32>>,
}

/// Running accumulator for per-channel activation statistics.
struct ChannelMagAccum {
    /// Running mean of |x| per channel (for AWQ scale computation).
    mean_abs: Vec<f32>,
    /// Running sum of x² per channel (for activation-weighted reconstruction loss).
    sum_sq: Vec<f32>,
    /// Total tokens accumulated so far.
    sample_count: usize,
}

impl ChannelMagAccum {
    fn new(n_features: usize) -> Self {
        Self {
            mean_abs: vec![0.0; n_features],
            sum_sq: vec![0.0; n_features],
            sample_count: 0,
        }
    }

    /// Accumulate a batch of activations: [n_tokens × n_features].
    fn accumulate(&mut self, data: &[f32], n_features: usize) {
        let n_tokens = data.len() / n_features;
        if n_tokens == 0 {
            return;
        }
        let mut batch_abs_sum = vec![0.0_f32; n_features];
        for t in 0..n_tokens {
            let row = &data[t * n_features..(t + 1) * n_features];
            for (c, &val) in row.iter().enumerate() {
                batch_abs_sum[c] += val.abs();
                self.sum_sq[c] += val * val;
            }
        }
        // Weighted running mean update for mean_abs.
        let old_count = self.sample_count as f32;
        let new_count = (self.sample_count + n_tokens) as f32;
        for c in 0..n_features {
            self.mean_abs[c] = (self.mean_abs[c] * old_count + batch_abs_sum[c]) / new_count;
        }
        self.sample_count += n_tokens;
    }
}

impl BlockCalibrationHook {
    fn new() -> Self {
        Self {
            attn_norm_mags: HashMap::new(),
            ffn_norm_mags: HashMap::new(),
            attn_norm_acts_last: HashMap::new(),
            ffn_norm_acts_last: HashMap::new(),
        }
    }
}

impl ActivationHook for BlockCalibrationHook {
    fn on_linear_input(&mut self, layer: usize, name: &str, activation: &[f16], n_features: usize) {
        match name {
            "attn_norm" => {
                let f32_data: Vec<f32> = activation.iter().map(|v| v.to_f32()).collect();
                self.attn_norm_mags
                    .entry(layer)
                    .or_insert_with(|| ChannelMagAccum::new(n_features))
                    .accumulate(&f32_data, n_features);
                // Keep last sequence's raw activations for clip search.
                self.attn_norm_acts_last.insert(layer, f32_data);
            }
            "ffn_norm" => {
                let f32_data: Vec<f32> = activation.iter().map(|v| v.to_f32()).collect();
                self.ffn_norm_mags
                    .entry(layer)
                    .or_insert_with(|| ChannelMagAccum::new(n_features))
                    .accumulate(&f32_data, n_features);
                self.ffn_norm_acts_last.insert(layer, f32_data);
            }
            _ => {}
        }
    }
}

// ── Weight loading helpers ─────────────────────────────────────

/// Build the HuggingFace tensor name for a layer projection weight.
fn hf_weight_name(layer: usize, proj: &str) -> String {
    let block = if ATTN_PROJS.contains(&proj) {
        "self_attn"
    } else {
        "mlp"
    };
    format!("model.layers.{layer}.{block}.{proj}.weight")
}

/// Check if a layer is GDN (linear attention) by probing for self_attn.q_proj.
fn is_gdn_layer(provider: &SafeTensorsProvider, layer: usize) -> bool {
    !provider.has_tensor(&format!("model.layers.{layer}.self_attn.q_proj.weight"))
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
        .attn_norm_mags
        .values()
        .next()
        .map(|a| a.sample_count)
        .unwrap_or(0);
    let n_tokens_last_seq = hook
        .attn_norm_acts_last
        .values()
        .next()
        .map(|a| a.len() / hidden_size)
        .unwrap_or(0);
    eprintln!(
        "Phase 1 complete in {:.1}s — {} layers, {} total tokens/layer ({} in last seq)",
        phase1_start.elapsed().as_secs_f64(),
        hook.attn_norm_mags.len().max(hook.ffn_norm_mags.len()),
        n_tokens,
        n_tokens_last_seq,
    );

    // ── Phase 2: Activation-weighted alpha search (CPU) ───────
    eprintln!("\n=== Phase 2: Activation-weighted alpha search (reference AWQ) ===");
    let phase2_start = Instant::now();

    let groups = weight_groups();

    let mut block_config: HashMap<String, AwqTensorConfig> = HashMap::new();
    let mut magnitudes_map: HashMap<String, Vec<f32>> = HashMap::new();
    // Cache f32 weights loaded during Phase 2 for reuse in Phase 3.
    let mut weight_cache: HashMap<(usize, String), (Vec<f32>, [usize; 2])> = HashMap::new();

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

    // Compute and store magnitudes for layer 0 (from accumulated stats)
    if let Some(mags) = hook.attn_norm_mags.get(&0) {
        for proj in ATTN_PROJS {
            magnitudes_map.insert(format!("l0_{proj}_weight"), mags.mean_abs.clone());
        }
    }
    if let Some(mags) = hook.ffn_norm_mags.get(&0) {
        for proj in FFN_PROJS {
            magnitudes_map.insert(format!("l0_{proj}_weight"), mags.mean_abs.clone());
        }
    }

    // Layers 1..N: activation-weighted alpha search (CPU, per reference AWQ)
    for layer_idx in 1..n_layers {
        let layer_start = Instant::now();
        let gdn = is_gdn_layer(&provider, layer_idx);

        // Use accumulated magnitudes (running mean across all sequences)
        let attn_mags = hook
            .attn_norm_mags
            .get(&layer_idx)
            .map(|a| a.mean_abs.clone())
            .unwrap_or_else(|| vec![1.0; hidden_size]);

        let ffn_mags = hook
            .ffn_norm_mags
            .get(&layer_idx)
            .map(|a| a.mean_abs.clone())
            .unwrap_or_else(|| vec![1.0; hidden_size]);

        // Store magnitudes for output (only for projections that exist)
        if !gdn {
            for proj in ATTN_PROJS {
                magnitudes_map.insert(format!("l{layer_idx}_{proj}_weight"), attn_mags.clone());
            }
        }
        for proj in FFN_PROJS {
            magnitudes_map.insert(format!("l{layer_idx}_{proj}_weight"), ffn_mags.clone());
        }

        // Skip attention groups for GDN layers
        let layer_groups: Vec<&_> = groups
            .iter()
            .filter(|g| !(gdn && g.norm_key == "attn"))
            .collect();

        for group in &layer_groups {
            let mags = if group.norm_key == "attn" {
                &attn_mags
            } else {
                &ffn_mags
            };

            // Get the per-channel activation power Σ_t X[t,c]² for this group's norm.
            let act_sq = if group.norm_key == "attn" {
                hook.attn_norm_mags
                    .get(&layer_idx)
                    .map(|a| a.sum_sq.clone())
            } else {
                hook.ffn_norm_mags.get(&layer_idx).map(|a| a.sum_sq.clone())
            };

            // Load weights for all projections in this group and cache them.
            let mut proj_weights: Vec<(&str, Vec<f32>, [usize; 2])> = Vec::new();
            for &proj in &group.proj_names {
                let cache_key = (layer_idx, proj.to_string());
                let (w, shape) = if let Some(cached) = weight_cache.get(&cache_key) {
                    cached.clone()
                } else {
                    let loaded = load_weight_f32(&provider, layer_idx, proj)?;
                    weight_cache.insert(cache_key, loaded.clone());
                    loaded
                };
                proj_weights.push((proj, w, shape));
            }

            let in_features_for_scales = proj_weights[0].2[1];

            // For o_proj and down_proj the in_features differ from hidden_size.
            let effective_mags = if mags.len() != in_features_for_scales {
                vec![1.0_f32; in_features_for_scales]
            } else {
                mags.clone()
            };

            // Per-channel activation power for the reconstruction loss.
            // Fall back to uniform if not available or wrong dimension.
            let effective_act_sq = match act_sq {
                Some(ref sq) if sq.len() == in_features_for_scales => sq.clone(),
                _ => vec![1.0_f32; in_features_for_scales],
            };

            // CPU activation-weighted reconstruction loss, matching reference AWQ:
            //   loss = Σ_proj Σ_row Σ_c (W_dq[r,c] - W[r,c])² · act_sq[c]
            //
            // This evaluates weight quality directly using activation importance,
            // rather than running a GPU block forward. The reference AWQ uses the
            // same objective for single-linear groups (o_proj, down_proj) and a
            // submodule forward for multi-linear groups — the activation-weighted
            // reconstruction is a close approximation for both.
            let eval_alpha = |alpha: f32| -> f64 {
                let scales = compute_awq_scales(&effective_mags, alpha);
                let mut total_loss = 0.0_f64;

                for &(_, ref w, [out_f, in_f]) in &proj_weights {
                    if scales.len() != in_f {
                        return f64::INFINITY;
                    }
                    let dq = quantize_dequant_scaled(w, out_f, in_f, &scales, group_size);

                    // Activation-weighted per-column MSE.
                    for row in 0..out_f {
                        for c in 0..in_f {
                            let idx = row * in_f + c;
                            let err = dq[idx] - w[idx];
                            total_loss += (err * err * effective_act_sq[c]) as f64;
                        }
                    }
                }

                total_loss
            };

            // ── Coarse pass (matching reference grid: 20 steps over 0..1) ──
            let n_grid = 20_usize;
            let mut best_alpha = 0.5_f32;
            let mut best_loss = f64::INFINITY;

            for step in 0..n_grid {
                let alpha = step as f32 / n_grid as f32;
                let loss = eval_alpha(alpha);
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
        let gdn = is_gdn_layer(&provider, layer_idx);
        for &proj in clip_projs {
            // Skip attention projections for GDN layers
            if gdn && ATTN_PROJS.contains(&proj) {
                continue;
            }
            let key = format!("l{layer_idx}_{proj}_weight");
            let alpha = block_config.get(&key).map(|c| c.alpha).unwrap_or(0.5);

            // Get raw activations (last sequence) and accumulated magnitudes.
            let norm_key = if ATTN_PROJS.contains(&proj) {
                "attn"
            } else {
                "ffn"
            };
            let activations = if norm_key == "attn" {
                hook.attn_norm_acts_last.get(&layer_idx)
            } else {
                hook.ffn_norm_acts_last.get(&layer_idx)
            };
            let activations = match activations {
                Some(a) => a,
                None => continue,
            };

            // Use cached weights from Phase 2 when available, otherwise load.
            let cache_key = (layer_idx, proj.to_string());
            let (w, [out_features, in_features]) =
                if let Some(cached) = weight_cache.get(&cache_key) {
                    cached.clone()
                } else {
                    load_weight_f32(&provider, layer_idx, proj)?
                };

            let clip_n_tokens = activations.len() / in_features;
            if clip_n_tokens == 0 {
                continue;
            }

            // Use accumulated magnitudes for AWQ scales (all sequences).
            let mags_accum = if norm_key == "attn" {
                hook.attn_norm_mags.get(&layer_idx)
            } else {
                hook.ffn_norm_mags.get(&layer_idx)
            };
            let mags = match mags_accum {
                Some(a) if a.mean_abs.len() == in_features => a.mean_abs.clone(),
                _ => compute_channel_magnitudes(activations, in_features),
            };
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
                clip_n_tokens,
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
