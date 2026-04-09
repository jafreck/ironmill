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
///   512 — Gemma 4 global attention layers
const HEAD_DIMS: &[usize] = &[64, 80, 128, 256, 512];

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

    // Absolute include path so #include "common/..." works from temp files in OUT_DIR.
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let shader_include_dir = manifest_dir.join("src/metal/shaders");

    // ── Consolidated domain metallibs ──────────────────────────
    // Group related single-kernel metallibs into domain bundles to reduce
    // the number of include_bytes! calls and simplify ShaderLibraries.
    let norm_shaders: &[(&str, &str)] = &[
        ("normalization", "norm"),
        ("fused_residual_norm", "norm"),
        ("fused_embedding_norm", "norm"),
        ("fused_softcap", "attention"),
    ];
    compile_shader_group(
        norm_shaders,
        "norm",
        shader_dir,
        &out_dir,
        &shader_include_dir,
    );

    let elementwise_shaders: &[(&str, &str)] = &[
        ("elementwise", "elementwise"),
        ("activation", "elementwise"),
        ("embedding", "embedding"),
        ("rope", "rope"),
    ];
    compile_shader_group(
        elementwise_shaders,
        "elementwise",
        shader_dir,
        &out_dir,
        &shader_include_dir,
    );

    let quantized_shaders: &[(&str, &str)] = &[
        ("quantized_matmul", "quantized"),
        ("int4_dequant", "quantized"),
        ("d2quant_matmul", "quantized"),
        ("quip_sharp", "quantized"),
        ("quantize", "quantized"),
    ];
    compile_shader_group(
        quantized_shaders,
        "quantized",
        shader_dir,
        &out_dir,
        &shader_include_dir,
    );

    // ── Individual metallibs (already domain-scoped) ────────────
    let individual: &[(&str, &str)] = &[
        ("kv_scatter", "kv"),
        ("matvec", "linear"),
        ("gdn_recurrent", "gdn"),
        ("ple_kernels", "ple"),
        ("moe", "moe"),
        ("superblock_norm", "norm"),
    ];

    for &(name, subdir) in individual {
        let src = shader_dir.join(subdir).join(format!("{name}.metal"));
        let lib = out_dir.join(format!("{name}.metallib"));
        compile_shader(&src, &lib, &[], &shader_include_dir);
    }

    // ── Affine matmul: AMX kernels (no function constants) ──
    {
        let quantized_dir = shader_dir.join("quantized");
        let affine_common =
            std::fs::read_to_string(quantized_dir.join("affine_common.metal")).unwrap();
        let amx_src = std::fs::read_to_string(quantized_dir.join("affine_amx.metal")).unwrap();
        let combined = format!("{affine_common}\n{amx_src}");
        let tmp = out_dir.join("_affine_combined.metal");
        std::fs::write(&tmp, &combined).unwrap();
        compile_shader(
            &tmp,
            &out_dir.join("affine_matmul.metallib"),
            &[],
            &shader_include_dir,
        );
    }

    // ── Superblock kernels (with GS function constant) ──
    {
        let quantized_dir = shader_dir.join("quantized");
        let sb_common =
            std::fs::read_to_string(quantized_dir.join("superblock_common.metal")).unwrap();
        let sb_parts = [
            "affine_matvec",
            "affine_matmul",
            "affine_batched",
            "affine_fused",
        ];
        let mut combined = sb_common;
        for part in &sb_parts {
            let src = std::fs::read_to_string(quantized_dir.join(format!("{part}.metal"))).unwrap();
            combined.push('\n');
            combined.push_str(&src);
        }
        let tmp = out_dir.join("_superblock_combined.metal");
        std::fs::write(&tmp, &combined).unwrap();
        compile_shader(
            &tmp,
            &out_dir.join("superblock.metallib"),
            &[],
            &shader_include_dir,
        );
    }

    // ── HEAD_DIM-dependent shaders (per-variant) ────────────────
    // Turboquant needs helpers prepended; we write a merged temp file.
    let tq_helpers_src =
        std::fs::read_to_string(helpers_dir.join("turboquant_helpers.metal")).unwrap();
    let tq_main_src =
        std::fs::read_to_string(shader_dir.join("turboquant/turboquant.metal")).unwrap();
    let attn_src = std::fs::read_to_string(shader_dir.join("attention/attention.metal")).unwrap();
    let fused_qk_src =
        std::fs::read_to_string(shader_dir.join("norm/fused_qk_norm_rope.metal")).unwrap();
    let sdpa_src = std::fs::read_to_string(shader_dir.join("attention/fused_sdpa.metal")).unwrap();

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
            &shader_include_dir,
        );

        // Fused SDPA — reduce tile sizes for large head dims to fit in 32KB threadgroup memory.
        let sdpa_tile_defines = if hd >= 256 {
            "#define SDPA_BR 4\n#define SDPA_BC 4\n".to_string()
        } else {
            String::new()
        };
        let sdpa_tmp = out_dir.join(format!("_sdpa_hd{hd}.metal"));
        std::fs::write(&sdpa_tmp, format!("{header}{sdpa_tile_defines}{sdpa_src}")).unwrap();
        compile_shader(
            &sdpa_tmp,
            &out_dir.join(format!("fused_sdpa_hd{hd}.metallib")),
            &[],
            &shader_include_dir,
        );

        // FlashDecoding split+reduce (separate metallib to avoid inflating
        // the original fused_sdpa kernel's threadgroup memory budget).
        let fd_src =
            std::fs::read_to_string(shader_dir.join("attention/flash_decode.metal")).unwrap();
        let fd_tile_defines = if hd >= 256 {
            "#define SPLIT_BC 8\n".to_string()
        } else {
            String::new()
        };
        let fd_tmp = out_dir.join(format!("_fd_hd{hd}.metal"));
        std::fs::write(&fd_tmp, format!("{header}{fd_tile_defines}{fd_src}")).unwrap();
        compile_shader(
            &fd_tmp,
            &out_dir.join(format!("flash_decode_hd{hd}.metallib")),
            &[],
            &shader_include_dir,
        );

        // Turboquant (helpers + main, with defines prepended)
        let tq_tmp = out_dir.join(format!("_tq_hd{hd}.metal"));
        let tq_combined = format!("{header}{tq_helpers_src}\n{tq_main_src}");
        std::fs::write(&tq_tmp, tq_combined).unwrap();
        compile_shader(
            &tq_tmp,
            &out_dir.join(format!("turboquant_hd{hd}.metallib")),
            &defines,
            &shader_include_dir,
        );
    }

    // Rerun when shader sources change.
    println!("cargo:rerun-if-changed=src/metal/shaders");
    println!("cargo:rerun-if-changed=src/metal/shaders/common");
    println!("cargo:rerun-if-changed=src/shaders");
}

/// Compile multiple `.metal` files into individual `.air` files, then link them
/// into a single `.metallib`.  This consolidates related kernels into one GPU
/// binary, reducing `include_bytes!` calls and `ShaderLibraries` fields.
fn compile_shader_group(
    shaders: &[(&str, &str)],
    output_name: &str,
    shader_dir: &Path,
    out_dir: &Path,
    include_dir: &Path,
) {
    let mut air_files: Vec<PathBuf> = Vec::new();

    // Step 1: compile each .metal → .air
    for &(name, subdir) in shaders {
        let src = shader_dir.join(subdir).join(format!("{name}.metal"));
        let air = out_dir.join(format!("{name}.air"));

        let result = Command::new("xcrun")
            .arg("metal")
            .arg("-c")
            .arg("-Wno-unused-variable")
            .arg("-I")
            .arg(include_dir)
            .arg(&src)
            .arg("-o")
            .arg(&air)
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
        air_files.push(air);
    }

    // Step 2: link all .air files → one .metallib
    let output = out_dir.join(format!("{output_name}.metallib"));
    let mut cmd = Command::new("xcrun");
    cmd.arg("metallib");
    for air in &air_files {
        cmd.arg(air);
    }
    cmd.arg("-o").arg(&output);

    let result = cmd.output().expect("failed to run `xcrun metallib`");
    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        panic!(
            "metallib creation failed for {}: {}",
            output.display(),
            stderr
        );
    }

    // Clean up intermediate .air files.
    for air in &air_files {
        let _ = std::fs::remove_file(air);
    }
}

fn compile_shader(src: &Path, output: &Path, extra_args: &[String], include_dir: &Path) {
    let air = output.with_extension("air");

    // Step 1: .metal → .air (intermediate representation)
    let mut cmd = Command::new("xcrun");
    cmd.arg("metal")
        .arg("-c")
        .arg("-Wno-unused-variable")
        .arg("-I")
        .arg(include_dir)
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
