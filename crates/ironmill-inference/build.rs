//! Build script for ironmill-inference: precompile Metal shaders into .metallib
//! binaries for common HEAD_DIM values, eliminating ~100-500ms of runtime
//! shader compilation.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// HEAD_DIM values to precompile. Covers the vast majority of LLMs:
///   64  — GPT-2 small, some Qwen variants
///   80  — Llama 3
///   128 — LLaMA 2, Qwen 2/3, Mistral, Gemma
///   256 — large models with wide attention
const HEAD_DIMS: &[usize] = &[64, 80, 128, 256];

fn main() {
    // Only precompile when targeting macOS with the metal feature.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let metal_feature = env::var("CARGO_FEATURE_METAL").is_ok();
    if target_os != "macos" || !metal_feature {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shader_dir = Path::new("src/metal/shaders");
    let helpers_dir = Path::new("src/shaders");

    // ── HEAD_DIM-independent shaders (compiled once) ────────────
    let independent = [
        "normalization",
        "activation",
        "rope",
        "elementwise",
        "embedding",
        "quantized_matmul",
        "kv_scatter",
        "matvec",
        "fused_residual_norm",
        "fused_embedding_norm",
        "int4_dequant",
        "affine_matmul",
        "quip_sharp",
    ];

    for name in &independent {
        let src = shader_dir.join(format!("{name}.metal"));
        let lib = out_dir.join(format!("{name}.metallib"));
        compile_shader(&src, &lib, &[]);
    }

    // ── HEAD_DIM-dependent shaders (per-variant) ────────────────
    // Turboquant needs helpers prepended; we write a merged temp file.
    let tq_helpers_src =
        std::fs::read_to_string(helpers_dir.join("turboquant_helpers.metal")).unwrap();
    let tq_main_src = std::fs::read_to_string(shader_dir.join("turboquant.metal")).unwrap();
    let attn_src = std::fs::read_to_string(shader_dir.join("attention.metal")).unwrap();
    let fused_qk_src =
        std::fs::read_to_string(shader_dir.join("fused_qk_norm_rope.metal")).unwrap();
    let sdpa_src = std::fs::read_to_string(shader_dir.join("fused_sdpa.metal")).unwrap();

    for &hd in HEAD_DIMS {
        let defines = [
            format!("-DHEAD_DIM={hd}"),
            format!("-DHEAD_DIM_PACKED={}", hd / 2),
        ];

        // Attention
        let attn_tmp = out_dir.join(format!("_attn_hd{hd}.metal"));
        let header = format!(
            "#define HEAD_DIM {hd}\n#define HEAD_DIM_PACKED {}\n",
            hd / 2
        );
        std::fs::write(&attn_tmp, format!("{header}{attn_src}\n{fused_qk_src}")).unwrap();
        compile_shader(
            &attn_tmp,
            &out_dir.join(format!("attention_hd{hd}.metallib")),
            &[],
        );

        // Fused SDPA — reduce tile sizes for large head dims to fit in 32KB threadgroup memory.
        let sdpa_tile_defines = if hd >= 256 {
            "#define SDPA_BR 16\n#define SDPA_BC 16\n".to_string()
        } else {
            String::new()
        };
        let sdpa_tmp = out_dir.join(format!("_sdpa_hd{hd}.metal"));
        std::fs::write(&sdpa_tmp, format!("{header}{sdpa_tile_defines}{sdpa_src}")).unwrap();
        compile_shader(
            &sdpa_tmp,
            &out_dir.join(format!("fused_sdpa_hd{hd}.metallib")),
            &[],
        );

        // Turboquant (helpers + main, with defines prepended)
        let tq_tmp = out_dir.join(format!("_tq_hd{hd}.metal"));
        let tq_combined = format!("{header}{tq_helpers_src}\n{tq_main_src}");
        std::fs::write(&tq_tmp, tq_combined).unwrap();
        compile_shader(
            &tq_tmp,
            &out_dir.join(format!("turboquant_hd{hd}.metallib")),
            &defines,
        );
    }

    // Rerun when shader sources change.
    println!("cargo:rerun-if-changed=src/metal/shaders");
    println!("cargo:rerun-if-changed=src/shaders");
}

fn compile_shader(src: &Path, output: &Path, extra_args: &[String]) {
    let air = output.with_extension("air");

    // Step 1: .metal → .air (intermediate representation)
    let mut cmd = Command::new("xcrun");
    cmd.arg("metal")
        .arg("-c")
        .arg("-Wno-unused-variable")
        .arg(src)
        .arg("-o")
        .arg(&air);
    for arg in extra_args {
        cmd.arg(arg);
    }
    let result = cmd
        .output()
        .expect("failed to run `xcrun metal` — is Xcode installed?");
    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        panic!(
            "Metal shader compilation failed for {}: {}",
            src.display(),
            stderr
        );
    }

    // Step 2: .air → .metallib (GPU binary)
    let result = Command::new("xcrun")
        .arg("metallib")
        .arg(&air)
        .arg("-o")
        .arg(output)
        .output()
        .expect("failed to run `xcrun metallib`");
    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        panic!(
            "metallib creation failed for {}: {}",
            output.display(),
            stderr
        );
    }

    // Clean up intermediate .air file.
    let _ = std::fs::remove_file(&air);
}
