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
    f16_slice_to_bytes, mse_f16_bytes, quantize_dequant_scaled, search_clip_ranges, weight_groups,
};
use ironmill_compile::weights::{SafeTensorsProvider, WeightProvider};
use ironmill_inference::calibration::{ActivationHook, CalibrationDataset};
use ironmill_inference::engine::InferenceEngine;
use ironmill_inference::metal::{
    GpuCalibrationEngine, MetalConfig, MetalInference, update_weight_buffer_f16,
};
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
                let bytes = f16_slice_to_bytes(activation);
                self.block_outputs.insert(layer, bytes);
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
    eprintln!("\n=== Phase 2: Block-level alpha search (coarse→fine) ===");
    let phase2_start = Instant::now();

    // Coarse→fine: 6 coarse candidates, then ±0.1 fine around best (step 0.02).
    let coarse_alphas: Vec<f32> = (0..=5).map(|i| i as f32 * 0.2).collect();
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

    // Compute and store magnitudes for layer 0
    if let Some(acts) = hook.attn_norm_acts.get(&0) {
        let mags = compute_channel_magnitudes(acts, hidden_size);
        for proj in ATTN_PROJS {
            magnitudes_map.insert(format!("l0_{proj}_weight"), mags.clone());
        }
    }
    if let Some(acts) = hook.ffn_norm_acts.get(&0) {
        let mags = compute_channel_magnitudes(acts, hidden_size);
        for proj in FFN_PROJS {
            magnitudes_map.insert(format!("l0_{proj}_weight"), mags.clone());
        }
    }

    // Reset engine once before the alpha search loop.
    InferenceEngine::reset(&mut engine);

    // Layers 1..N: full block-level alpha search
    for layer_idx in 1..n_layers {
        let layer_start = Instant::now();
        let gdn = is_gdn_layer(&provider, layer_idx);

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

            // GDN layers have no attention projections — only FFN groups
            // remain. Skip the expensive GPU alpha search and use a fixed
            // alpha, which is close to optimal for FFN-only groups.
            if gdn {
                let best_alpha = 0.5_f32;
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
                    "  [layer {layer_idx}] {:?} → alpha={best_alpha:.2} (GDN, fixed)",
                    group.proj_names,
                );
                continue;
            }

            // Pre-allocate reusable GPU scratch buffers (one per projection).
            let mut scratch_bufs: Vec<_> = proj_weights
                .iter()
                .map(|&(_, _, [out_f, in_f])| engine.create_dense_f16_buffer_sized(out_f * in_f))
                .collect::<Result<Vec<_>, _>>()?;

            // Helper: evaluate a single alpha candidate.
            let eval_alpha =
                |engine: &mut MetalInference,
                 scratch_bufs: &mut [ironmill_inference::metal::weights::WeightBuffer],
                 alpha: f32|
                 -> Result<f64, Box<dyn std::error::Error>> {
                    let scales = compute_awq_scales(&effective_mags, alpha);
                    let mut swap_handles: Vec<(
                        &str,
                        Option<ironmill_inference::metal::weights::WeightBuffer>,
                    )> = Vec::new();

                    for (i, &(proj, ref w, [out_f, in_f])) in proj_weights.iter().enumerate() {
                        if scales.len() != in_f {
                            return Ok(f64::INFINITY);
                        }
                        let dq = quantize_dequant_scaled(w, out_f, in_f, &scales, group_size);
                        update_weight_buffer_f16(&scratch_bufs[i], &dq)?;
                        let scratch = std::mem::replace(
                            &mut scratch_bufs[i],
                            ironmill_inference::metal::weights::WeightBuffer::empty(),
                        );
                        let original = engine.swap_layer_weight(layer_idx, proj, scratch);
                        swap_handles.push((proj, original));
                    }

                    let loss = match engine.run_single_layer(layer_idx, block_input, token_count) {
                        Ok(output) => mse_f16_bytes(&output, ref_output),
                        Err(e) => {
                            eprintln!("[layer {layer_idx}] run_single_layer error: {e}");
                            f64::INFINITY
                        }
                    };

                    // Restore original weights, recovering scratch buffers.
                    for (i, (proj, original)) in swap_handles.into_iter().enumerate().rev() {
                        if let Some(orig) = original {
                            let returned = engine.swap_layer_weight(layer_idx, proj, orig);
                            if let Some(buf) = returned {
                                scratch_bufs[i] = buf;
                            }
                        }
                    }

                    Ok(loss)
                };

            // ── Coarse pass ──
            let mut best_alpha = 0.5_f32;
            let mut best_loss = f64::INFINITY;

            for &alpha in &coarse_alphas {
                let loss = eval_alpha(&mut engine, &mut scratch_bufs, alpha)?;
                if loss < best_loss {
                    best_loss = loss;
                    best_alpha = alpha;
                }
            }

            // ── Fine pass ── ±0.1 around best, step 0.02.
            // Early termination: stop if no improvement for 3 consecutive evals.
            let fine_lo = (best_alpha - 0.1).max(0.0);
            let fine_hi = (best_alpha + 0.1).min(1.0);
            let mut fine_alpha = fine_lo;
            let mut no_improve_count = 0_u32;
            while fine_alpha <= fine_hi + 1e-6 {
                let already_tested = coarse_alphas
                    .iter()
                    .any(|&c| (c - fine_alpha).abs() < 0.005);
                if !already_tested {
                    let loss = eval_alpha(&mut engine, &mut scratch_bufs, fine_alpha)?;
                    if loss < best_loss {
                        best_loss = loss;
                        best_alpha = fine_alpha;
                        no_improve_count = 0;
                    } else {
                        no_improve_count += 1;
                        if no_improve_count >= 3 {
                            break;
                        }
                    }
                }
                fine_alpha += 0.02;
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

            // Use cached weights from Phase 2 when available, otherwise load.
            let cache_key = (layer_idx, proj.to_string());
            let (w, [out_features, in_features]) =
                if let Some(cached) = weight_cache.get(&cache_key) {
                    cached.clone()
                } else {
                    load_weight_f32(&provider, layer_idx, proj)?
                };

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
