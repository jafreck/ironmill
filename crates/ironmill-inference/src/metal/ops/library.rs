//! Shader library loading — embed precompiled `.metallib` binaries and
//! compile HEAD_DIM-dependent shaders at runtime when necessary.

use ironmill_metal_sys::{MetalDevice, ShaderLibrary};

use crate::metal::error::MetalError;

/// All loaded shader libraries, passed to `compile_*_shaders` helpers.
pub(super) struct ShaderLibraries {
    pub norm: ShaderLibrary,
    pub act: ShaderLibrary,
    pub rope: ShaderLibrary,
    pub elem: ShaderLibrary,
    pub embed: ShaderLibrary,
    pub quantize: ShaderLibrary,
    pub qmm: ShaderLibrary,
    pub kv_scatter: ShaderLibrary,
    pub matvec: ShaderLibrary,
    pub fused_rn: ShaderLibrary,
    pub fused_en: ShaderLibrary,
    pub int4_dequant: ShaderLibrary,
    pub quip_sharp: ShaderLibrary,
    pub fused_softcap: ShaderLibrary,
    pub ple: ShaderLibrary,
    pub affine_mm: ShaderLibrary,
    pub d2quant_mm: ShaderLibrary,
    pub gdn: ShaderLibrary,
    pub attn: ShaderLibrary,
    pub tq: ShaderLibrary,
    pub sdpa: ShaderLibrary,
    pub fd: ShaderLibrary,
}

impl ShaderLibraries {
    /// Load all precompiled shader libraries from embedded metallib binaries.
    pub fn load(
        device: &MetalDevice,
        head_dim: usize,
        rotary_dim: usize,
    ) -> Result<Self, MetalError> {
        let (attn, tq, sdpa, fd) = Self::load_head_dim_shaders(device, head_dim, rotary_dim)?;
        Ok(ShaderLibraries {
            norm: device
                .load_library_from_data(include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/normalization.metallib"
                )))
                .map_err(MetalError::Metal)?,
            act: device
                .load_library_from_data(include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/activation.metallib"
                )))
                .map_err(MetalError::Metal)?,
            rope: device
                .load_library_from_data(include_bytes!(concat!(env!("OUT_DIR"), "/rope.metallib")))
                .map_err(MetalError::Metal)?,
            elem: device
                .load_library_from_data(include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/elementwise.metallib"
                )))
                .map_err(MetalError::Metal)?,
            embed: device
                .load_library_from_data(include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/embedding.metallib"
                )))
                .map_err(MetalError::Metal)?,
            quantize: device
                .load_library_from_data(include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/quantize.metallib"
                )))
                .map_err(MetalError::Metal)?,
            qmm: device
                .load_library_from_data(include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/quantized_matmul.metallib"
                )))
                .map_err(MetalError::Metal)?,
            kv_scatter: device
                .load_library_from_data(include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/kv_scatter.metallib"
                )))
                .map_err(MetalError::Metal)?,
            matvec: device
                .load_library_from_data(include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/matvec.metallib"
                )))
                .map_err(MetalError::Metal)?,
            fused_rn: device
                .load_library_from_data(include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/fused_residual_norm.metallib"
                )))
                .map_err(MetalError::Metal)?,
            fused_en: device
                .load_library_from_data(include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/fused_embedding_norm.metallib"
                )))
                .map_err(MetalError::Metal)?,
            int4_dequant: device
                .load_library_from_data(include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/int4_dequant.metallib"
                )))
                .map_err(MetalError::Metal)?,
            quip_sharp: device
                .load_library_from_data(include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/quip_sharp.metallib"
                )))
                .map_err(MetalError::Metal)?,
            fused_softcap: device
                .load_library_from_data(include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/fused_softcap.metallib"
                )))
                .map_err(MetalError::Metal)?,
            ple: device
                .load_library_from_data(include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/ple_kernels.metallib"
                )))
                .map_err(MetalError::Metal)?,
            affine_mm: device
                .load_library_from_data(include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/affine_matmul.metallib"
                )))
                .map_err(MetalError::Metal)?,
            d2quant_mm: device
                .load_library_from_data(include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/d2quant_matmul.metallib"
                )))
                .map_err(MetalError::Metal)?,
            gdn: device
                .load_library_from_data(include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/gdn_recurrent.metallib"
                )))
                .map_err(MetalError::Metal)?,
            attn,
            tq,
            sdpa,
            fd,
        })
    }

    /// Load attention, turboquant, fused SDPA, and FlashDecoding shaders for a specific HEAD_DIM.
    ///
    /// Uses precompiled `.metallib` for common values (64, 80, 128, 256, 512);
    /// falls back to runtime source compilation for uncommon dimensions.
    fn load_head_dim_shaders(
        device: &MetalDevice,
        head_dim: usize,
        rotary_dim: usize,
    ) -> Result<(ShaderLibrary, ShaderLibrary, ShaderLibrary, ShaderLibrary), MetalError> {
        // Macro to embed a precompiled metallib by head_dim.
        macro_rules! precompiled {
            (attn $hd:literal) => {
                include_bytes!(concat!(env!("OUT_DIR"), "/attention_hd", $hd, ".metallib"))
            };
            (tq $hd:literal) => {
                include_bytes!(concat!(env!("OUT_DIR"), "/turboquant_hd", $hd, ".metallib"))
            };
            (sdpa $hd:literal) => {
                include_bytes!(concat!(env!("OUT_DIR"), "/fused_sdpa_hd", $hd, ".metallib"))
            };
            (fd $hd:literal) => {
                include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/flash_decode_hd",
                    $hd,
                    ".metallib"
                ))
            };
        }

        let try_precompiled = |attn_data: &[u8],
                               tq_data: &[u8],
                               sdpa_data: &[u8],
                               fd_data: &[u8]|
         -> Result<
            (ShaderLibrary, ShaderLibrary, ShaderLibrary, ShaderLibrary),
            MetalError,
        > {
            let attn = device
                .load_library_from_data(attn_data)
                .map_err(MetalError::Metal)?;
            let tq = device
                .load_library_from_data(tq_data)
                .map_err(MetalError::Metal)?;
            let sdpa = device
                .load_library_from_data(sdpa_data)
                .map_err(MetalError::Metal)?;
            let fd = device
                .load_library_from_data(fd_data)
                .map_err(MetalError::Metal)?;
            Ok((attn, tq, sdpa, fd))
        };

        match head_dim {
            64 => try_precompiled(
                precompiled!(attn "64"),
                precompiled!(tq "64"),
                precompiled!(sdpa "64"),
                precompiled!(fd "64"),
            ),
            80 => try_precompiled(
                precompiled!(attn "80"),
                precompiled!(tq "80"),
                precompiled!(sdpa "80"),
                precompiled!(fd "80"),
            ),
            128 => try_precompiled(
                precompiled!(attn "128"),
                precompiled!(tq "128"),
                precompiled!(sdpa "128"),
                precompiled!(fd "128"),
            ),
            256 => {
                // Head dim 256: FA2 prefill kernel exceeds 32KB threadgroup limit
                // with Q_CHUNK=32. Compile at runtime with reduced tile size.
                let header = format!(
                    "#define HEAD_DIM 256\n#define HEAD_DIM_PACKED 128\n#define ROTARY_DIM {rotary_dim}\n"
                );
                let attn_src_raw = include_str!("../shaders/attention/attention.metal");
                let fused_qknr_src = include_str!("../shaders/norm/fused_qk_norm_rope.metal");
                let attn_src_patched = attn_src_raw
                    .replace(
                        "constant constexpr uint FA2_Q_CHUNK = 32;",
                        "constant constexpr uint FA2_Q_CHUNK = 8;",
                    )
                    .replace(
                        "constant constexpr uint FA2_KV_TILE = 32;",
                        "constant constexpr uint FA2_KV_TILE = 16;",
                    );
                let attn_src = format!("{header}{attn_src_patched}\n{fused_qknr_src}");
                let tq_helpers = include_str!("../../shaders/turboquant_helpers.metal");
                let tq_src_raw = include_str!("../shaders/turboquant/turboquant.metal");
                let tq_src = format!("{header}{tq_helpers}\n{tq_src_raw}");
                // fused_sdpa exceeds 32KB threadgroup memory at head_dim=256.
                // Use an empty stub — decode uses FlashDecoding split+reduce instead.
                let sdpa = device
                    .compile_shader_source(&format!(
                        "{header}#include <metal_stdlib>\nusing namespace metal;\n"
                    ))
                    .map_err(MetalError::Metal)?;
                let fd_src_raw = include_str!("../shaders/attention/flash_decode.metal");
                let fd_src = format!("{header}#define SPLIT_BC 8\n{fd_src_raw}");

                let attn = device
                    .compile_shader_source(&attn_src)
                    .map_err(MetalError::Metal)?;
                let tq = device
                    .compile_shader_source(&tq_src)
                    .map_err(MetalError::Metal)?;
                // sdpa already compiled above as an empty stub
                let fd = device
                    .compile_shader_source(&fd_src)
                    .map_err(MetalError::Metal)?;
                Ok((attn, tq, sdpa, fd))
            }
            512 => try_precompiled(
                precompiled!(attn "512"),
                precompiled!(tq "512"),
                precompiled!(sdpa "512"),
                precompiled!(fd "512"),
            ),
            _ => {
                let header = format!(
                    "#define HEAD_DIM {head_dim}\n#define HEAD_DIM_PACKED {}\n#define ROTARY_DIM {rotary_dim}\n",
                    head_dim / 2
                );
                let attn_src_raw = include_str!("../shaders/attention/attention.metal");
                let fused_qk_src = include_str!("../shaders/norm/fused_qk_norm_rope.metal");
                let attn_src = format!("{header}{attn_src_raw}\n{fused_qk_src}");
                let tq_helpers = include_str!("../../shaders/turboquant_helpers.metal");
                let tq_src_raw = include_str!("../shaders/turboquant/turboquant.metal");
                let tq_src = format!("{header}{tq_helpers}\n{tq_src_raw}");
                let sdpa_tile_defines = if head_dim >= 256 {
                    "#define SDPA_BR 4\n#define SDPA_BC 4\n"
                } else {
                    ""
                };
                let sdpa_src_raw = include_str!("../shaders/attention/fused_sdpa.metal");
                let sdpa_src = format!("{header}{sdpa_tile_defines}{sdpa_src_raw}");
                let fd_tile_defines = if head_dim >= 256 {
                    "#define SPLIT_BC 8\n"
                } else {
                    ""
                };
                let fd_src_raw = include_str!("../shaders/attention/flash_decode.metal");
                let fd_src = format!("{header}{fd_tile_defines}{fd_src_raw}");

                let attn = device
                    .compile_shader_source(&attn_src)
                    .map_err(MetalError::Metal)?;
                let tq = device
                    .compile_shader_source(&tq_src)
                    .map_err(MetalError::Metal)?;
                let sdpa = device
                    .compile_shader_source(&sdpa_src)
                    .map_err(MetalError::Metal)?;
                let fd = device
                    .compile_shader_source(&fd_src)
                    .map_err(MetalError::Metal)?;
                Ok((attn, tq, sdpa, fd))
            }
        }
    }
}
