use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::config::{ModelConfig, OptConfig, cache_key};

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

    let (mut onnx, model_dir) = ironmill_compile::mil::read_onnx_with_dir(&model.path)?;
    let config = ironmill_compile::mil::ConversionConfig {
        model_dir: Some(model_dir),
        ..Default::default()
    };
    let conversion_result = ironmill_compile::mil::onnx_to_program_with_config(&mut onnx, &config)?;
    let mut program = conversion_result.program;

    let mut pipeline = ironmill_compile::mil::PassPipeline::default();
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

    pipeline.run(&mut program)?;

    let model_proto = ironmill_compile::mil::program_to_model(&program, 7)?;

    let mlpackage_path = entry_dir.join("model.mlpackage");
    ironmill_compile::mil::write_mlpackage(&model_proto, &mlpackage_path)?;

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
) -> Result<ironmill_compile::mil::Program> {
    let (mut onnx, model_dir) = ironmill_compile::mil::read_onnx_with_dir(&model.path)?;
    let config = ironmill_compile::mil::ConversionConfig {
        model_dir: Some(model_dir),
        ..Default::default()
    };
    let conversion_result = ironmill_compile::mil::onnx_to_program_with_config(&mut onnx, &config)?;
    let mut program = conversion_result.program;

    let mut pipeline = ironmill_compile::mil::PassPipeline::default();
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

    pipeline.run(&mut program)?;

    Ok(program)
}

/// Build the optimized MIL IR program for a model without compiling to CoreML.
///
/// This variant is gated behind `ane-direct` and is kept for backward
/// compatibility — the ungated [`build_optimized_program`] is now preferred.
#[cfg(feature = "ane-direct")]
pub fn build_optimized_program_ane(
    model: &ModelConfig,
    opt: &OptConfig,
) -> Result<ironmill_compile::mil::Program> {
    build_optimized_program(model, opt)
}

/// Remove all cached compilation artifacts.
pub fn clean_cache(cache_dir: &Path) -> Result<()> {
    if cache_dir.exists() {
        std::fs::remove_dir_all(cache_dir)?;
    }
    Ok(())
}
