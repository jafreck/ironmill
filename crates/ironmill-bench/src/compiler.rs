use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::config::{ModelConfig, OptConfig, cache_key};

/// Build the optimization pipeline from an [`OptConfig`].
fn build_pipeline(opt: &OptConfig) -> Result<mil_rs::ir::PassPipeline> {
    let mut pipeline = mil_rs::ir::PassPipeline::default();
    if opt.no_fusion {
        pipeline = pipeline.without_fusion();
    }
    match opt.quantize.as_deref() {
        Some("fp16") => {
            pipeline = pipeline.with_fp16()?;
        }
        Some("int8") => {
            pipeline = pipeline.with_int8(None)?;
        }
        _ => {}
    }
    if let Some(bits) = opt.palettize {
        pipeline = pipeline.with_palettize(bits)?;
    }
    if let Some(bits) = opt.polar_quantize {
        pipeline = pipeline.with_polar_quant(bits)?;
    }
    if let Some(bits) = opt.d2quant {
        pipeline = pipeline.with_d2quant(bits, 128, 0.99, None)?;
    }
    Ok(pipeline)
}

/// Parse a model into MIL IR without running any optimization passes.
///
/// When `model.model_dir` is set, loads weights from SafeTensors files and
/// builds MIL IR using the architecture template. Otherwise falls back to
/// ONNX parsing.
pub fn parse_model(model: &ModelConfig) -> Result<mil_rs::ir::Program> {
    if let Some(model_dir) = &model.model_dir {
        parse_model_from_template(model_dir)
    } else {
        parse_model_from_onnx(model)
    }
}

/// Parse an ONNX model into MIL IR.
fn parse_model_from_onnx(model: &ModelConfig) -> Result<mil_rs::ir::Program> {
    let (mut onnx, model_dir) = mil_rs::read_onnx_with_dir(&model.path)?;
    let mut config = mil_rs::ConversionConfig::default();
    config.model_dir = Some(model_dir);
    let result = mil_rs::onnx_to_program_with_config(&mut onnx, &config)?;
    Ok(result.program)
}

/// Build MIL IR from SafeTensors weights using the architecture template.
fn parse_model_from_template(model_dir: &Path) -> Result<mil_rs::ir::Program> {
    use std::sync::Arc;

    let config_path = model_dir.join("config.json");
    let _hf_config = ironmill_compile::weights::safetensors::parse_hf_config(&config_path)
        .with_context(|| format!("failed to parse config.json in {}", model_dir.display()))?;

    let provider = ironmill_compile::weights::SafeTensorsProvider::load(model_dir)
        .with_context(|| format!("failed to load SafeTensors from {}", model_dir.display()))?;

    let result = ironmill_compile::templates::weights_to_program(&provider)
        .with_context(|| "failed to build MIL program from template")?;

    for warning in &result.warnings {
        eprintln!("template warning: {warning}");
    }

    let mut program = result.program;
    // Attach the weight provider so optimization passes can resolve external tensors.
    program.set_weight_provider(Arc::new(provider));
    Ok(program)
}

/// Apply the optimization pipeline to a program in-place.
pub fn optimize_program(
    program: &mut mil_rs::ir::Program,
    opt: &OptConfig,
) -> Result<()> {
    let pipeline = build_pipeline(opt)?;
    pipeline.run(program)?;
    Ok(())
}

/// Compile an ONNX model with the given optimization config.
/// Returns the path to the compiled .mlmodelc directory.
/// Uses caching to avoid recompilation when possible.
pub fn compile_model(
    model: &ModelConfig,
    opt: &OptConfig,
    cache_dir: &Path,
    force: bool,
) -> Result<PathBuf> {
    let key = cache_key(model, opt);
    let entry_dir = cache_dir.join(format!(
        "{}-{}",
        model.name.to_lowercase().replace(' ', "_"),
        key
    ));
    let mlmodelc_path = entry_dir.join("model.mlmodelc");

    if !force && mlmodelc_path.exists() {
        return Ok(mlmodelc_path);
    }

    std::fs::create_dir_all(&entry_dir)?;

    let mut program = parse_model(model)?;
    optimize_program(&mut program, opt)?;

    // Materialize all external tensor refs before serialization.
    if program.has_weight_provider() {
        program.materialize_all()?;
    }

    let model_proto = mil_rs::program_to_model(&program, 7)?;

    let mlpackage_path = entry_dir.join("model.mlpackage");
    mil_rs::write_mlpackage(&model_proto, &mlpackage_path)?;

    let compiled_path =
        ironmill_compile::coreml::compiler::compile_model(&mlpackage_path, &entry_dir)?;

    Ok(compiled_path)
}

/// Compile from an already-parsed program (avoids re-reading ONNX).
///
/// Clones the program internally so the caller's copy is not modified.
pub fn compile_model_from_program(
    program: &mil_rs::ir::Program,
    model: &ModelConfig,
    opt: &OptConfig,
    cache_dir: &Path,
    force: bool,
) -> Result<PathBuf> {
    let key = cache_key(model, opt);
    let entry_dir = cache_dir.join(format!(
        "{}-{}",
        model.name.to_lowercase().replace(' ', "_"),
        key
    ));
    let mlmodelc_path = entry_dir.join("model.mlmodelc");

    if !force && mlmodelc_path.exists() {
        return Ok(mlmodelc_path);
    }

    std::fs::create_dir_all(&entry_dir)?;

    let mut prog = program.clone();
    optimize_program(&mut prog, opt)?;

    if prog.has_weight_provider() {
        prog.materialize_all()?;
    }

    let model_proto = mil_rs::program_to_model(&prog, 7)?;

    let mlpackage_path = entry_dir.join("model.mlpackage");
    mil_rs::write_mlpackage(&model_proto, &mlpackage_path)?;

    let compiled_path =
        ironmill_compile::coreml::compiler::compile_model(&mlpackage_path, &entry_dir)?;

    Ok(compiled_path)
}

/// Build the optimized MIL IR program for a model without compiling to CoreML.
///
/// Returns the program after the optimization pipeline has been applied.
/// Used for quality measurement and the ANE direct runtime path.
pub fn build_optimized_program(
    model: &ModelConfig,
    opt: &OptConfig,
) -> Result<mil_rs::ir::Program> {
    let mut program = parse_model(model)?;
    optimize_program(&mut program, opt)?;
    Ok(program)
}

/// Build from an already-parsed program (avoids re-reading ONNX).
///
/// Clones the program so the caller's copy is not modified.
pub fn build_optimized_program_from(
    program: &mil_rs::ir::Program,
    opt: &OptConfig,
) -> Result<mil_rs::ir::Program> {
    let mut prog = program.clone();
    optimize_program(&mut prog, opt)?;
    Ok(prog)
}

/// Build the optimized MIL IR program for a model without compiling to CoreML.
///
/// This variant is gated behind `ane-direct` and is kept for backward
/// compatibility — the ungated [`build_optimized_program`] is now preferred.
#[cfg(feature = "ane-direct")]
pub fn build_optimized_program_ane(
    model: &ModelConfig,
    opt: &OptConfig,
) -> Result<mil_rs::ir::Program> {
    build_optimized_program(model, opt)
}

/// Remove all cached compilation artifacts.
pub fn clean_cache(cache_dir: &Path) -> Result<()> {
    if cache_dir.exists() {
        std::fs::remove_dir_all(cache_dir)?;
    }
    Ok(())
}
