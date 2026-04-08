# Superblock Weight Layout: Unified Decode + Prefill Optimization

## Problem

After vectorizing INT4 matvec kernels (uint32 loads, 2-row TGs), ironmill
achieves **31 tok/s** decode on Qwen 3.5-4B INT4 (M2 Max 64 GB). MLX
achieves ~50 tok/s and llama.cpp ~90 tok/s on equivalent workloads.

Profiling and kernel analysis show the remaining gap is **not** algorithmic
— all three frameworks use the same SIMD-reduction matvec pattern for M=1
decode. The gap is structural: **memory access coalescing**.

### Root Cause: Blocked Layout Prevents Coalescing

ironmill stores INT4 weights in a blocked layout optimized for the prefill
tiled GEMM's simdgroup MMA access pattern:

```
Current layout: [N/64, K/8, 64, 4]
                 ^^^^  ^^^  ^^  ^
                 N-blk K-blk rows bytes_per_row_per_kblk
```

For decode (M=1 matvec), each SIMD lane processes a different K-block for
the same output row. Consecutive K-blocks for the same row are **256 bytes
apart** (interleaved with 63 other rows). This causes:

- **32 separate cache-line fetches** per simdgroup per iteration (one per lane)
- Each 128-byte cache line has only **4 bytes** used (3.1% utilization)
- Effective bandwidth: ~31 GB/s (8% of 400 GB/s peak)

MLX and llama.cpp store K contiguously per row. Consecutive lanes read
consecutive bytes → **one cache-line fetch** per simdgroup per iteration →
100% utilization. This is the 2–3× bandwidth difference.

### Why the Blocked Layout Doesn't Help Prefill Either

The blocked layout was designed for simdgroup MMA, but MMA operates on
data already loaded into **threadgroup memory**. The global memory load
pattern (256 threads loading a [64×32] B-tile) determines actual DRAM
bandwidth. With the current layout, the B-tile load scatters across
blocks; with row-major, it reads contiguous K values from the same
row — coalesced either way, but row-major is strictly better.

---

## Proposed Layout: Row-Major Inline Superblocks

### Design

Pack scale, zero, and weight data into self-contained **superblocks**
laid out contiguously per row. Each superblock covers one quantization
group (typically 128 elements):

```
┌────────────────── one superblock (68 bytes for group_size=128) ──────────────────┐
│  scale (half, 2B)  │  zero (half, 2B)  │  packed INT4 weights (64 bytes)        │
└────────────────────┴───────────────────┴─────────────────────────────────────────┘

Full row:   [SB_0][SB_1][SB_2]...[SB_{G-1}]     G = K / group_size superblocks
Full matrix: row_0 | row_1 | ... | row_{N-1}     contiguous in memory
```

For INT4 with `group_size=128`:
- Superblock size: `2 + 2 + 128/2 = 68 bytes`
- Superblocks per row: `K / 128`
- Total bytes per row: `K/128 × 68 = K × 0.53125` bytes
- INT8 variant: `2 + 2 + 128 = 132 bytes` per superblock

### Memory: Identical to Current

```
Current:   data[N × K/2] + scales[N × G × 2] + zeros[N × G × 2]
         = N × (K/2 + G×4) bytes

Proposed:  superblocks[N × G × (4 + group_size/2)]
         = N × G × (4 + group_size/2)
         = N × (G×4 + K/2) bytes

         ✓ Identical total memory. Metadata moves inline, not duplicated.
```

For Qwen 3.5-4B (1.02 GB weights): **zero additional memory**.

### Comparison with MLX and llama.cpp

| Property | ironmill (current) | MLX | llama.cpp Q4_K_M | **Proposed** |
|---|---|---|---|---|
| K contiguous per row | ❌ 256B stride | ✅ | ✅ | ✅ |
| Inline scale/zero | ❌ separate arrays | ❌ separate | ✅ 144B super-block | ✅ 68B super-block |
| Uniform format | ✅ | ✅ | ❌ mixed-width scales | ✅ |
| Decode coalescing | 3.1% cache util | ~100% | ~100% | ~100% |
| AWQ support | ✅ | ✅ | ❌ | ✅ (separate array) |
| Prefill B-tile load | scattered | coalesced | coalesced | coalesced |
| Memory overhead | baseline | baseline | baseline | **zero** |

---

## Decode Kernel Design

### Architecture: MLX-Style QMV with Inline Scales

```
Dispatch: (ceil(N/8), 1, 1) threadgroups × (64, 1, 1) threads
          2 simdgroups × 4 rows per simdgroup = 8 output rows per TG
```

Each simdgroup processes **4 output rows** using the same input vector,
amortizing the input load cost 4×. Each lane handles `values_per_thread`
weight elements per iteration.

```metal
// Constants for INT4
constexpr int values_per_thread = 16;  // 16 INT4 values = 8 bytes per lane
constexpr int block_size = values_per_thread * SIMD_SIZE;  // 512 elements per iter
constexpr int rows_per_sg = 4;
constexpr int num_simdgroups = 2;
constexpr int bytes_per_superblock = 4 + group_size / 2;  // 68 for gs=128

kernel void superblock_matvec_int4(
    device const uchar *W,           // [N, G, bytes_per_superblock]
    device const half  *x,           // [1, K]
    device half        *y,           // [1, N]
    constant uint      &N,
    constant uint      &K,
    constant uint      &group_size,
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]])
{
    uint base_row = tgid * (num_simdgroups * rows_per_sg)
                  + sgid * rows_per_sg;

    uint num_groups = K / group_size;
    uint sb_stride = num_groups * bytes_per_superblock;  // bytes per row

    // Pre-scale input: divide by positional powers of 2 so that
    // masked (un-shifted) nibble × pre-scaled input = correct product.
    // This eliminates all bit-shift operations from the hot loop.
    thread float x_buf[values_per_thread];
    float result[rows_per_sg] = {0};

    device const uchar *x_ptr = (device const uchar *)x;

    for (uint g = 0; g < num_groups; g++) {
        // Load input chunk for this group iteration
        uint k_base = g * group_size + lane * values_per_thread;
        float x_sum = 0;
        for (int i = 0; i < values_per_thread; i += 4) {
            x_buf[i]     = float(x[k_base + i]);
            x_buf[i + 1] = float(x[k_base + i + 1]) / 16.0f;
            x_buf[i + 2] = float(x[k_base + i + 2]) / 256.0f;
            x_buf[i + 3] = float(x[k_base + i + 3]) / 4096.0f;
            x_sum += float(x[k_base + i]) + float(x[k_base + i + 1])
                   + float(x[k_base + i + 2]) + float(x[k_base + i + 3]);
        }

        for (uint r = 0; r < rows_per_sg; r++) {
            uint row = base_row + r;
            if (row >= N) break;

            // Superblock pointer: scale + zero + weight data
            // All inline, fetched in the same cache line as weights
            device const uchar *sb = W + row * sb_stride
                                       + g * bytes_per_superblock;
            float scale = float(*(device const half *)(sb));
            float zero  = float(*(device const half *)(sb + 2));

            // Weight data starts at sb + 4. Each lane reads 8 bytes
            // (16 nibbles) at lane-strided offset.
            // COALESCED: 32 lanes × 8 bytes = 256 bytes contiguous
            device const uint16_t *wp =
                (device const uint16_t *)(sb + 4 + lane * 8);

            float acc = 0;
            for (int i = 0; i < values_per_thread / 4; i++) {
                acc += x_buf[4*i]     * float(wp[i] & 0x000F)
                     + x_buf[4*i + 1] * float(wp[i] & 0x00F0)
                     + x_buf[4*i + 2] * float(wp[i] & 0x0F00)
                     + x_buf[4*i + 3] * float(wp[i] & 0xF000);
            }

            result[r] += scale * acc + zero * x_sum;
        }
    }

    // Reduce across SIMD lanes
    for (uint r = 0; r < rows_per_sg; r++) {
        result[r] = simd_sum(result[r]);
        if (lane == 0 && base_row + r < N) {
            y[base_row + r] = half(result[r]);
        }
    }
}
```

### Key Optimizations

1. **Perfect coalescing**: 32 lanes read 256 contiguous bytes per
   simdgroup per group iteration. One cache-line transaction per 128
   bytes → 2 transactions for 256 bytes, all useful.

2. **Inline scale/zero**: Scale and zero are at the start of each
   superblock. When the GPU fetches the first cache line of weight
   data, scale/zero come along for free — zero extra memory transactions.

3. **Pre-scaled input trick** (from MLX): Input values are pre-divided
   by positional powers of 2, so the hot loop multiplies the raw masked
   nibble (without bit-shifting) by the pre-scaled input. This
   eliminates `N × K` bit-shift operations from the decode path.

4. **Multi-row amortization**: 4 rows per simdgroup share the same
   input vector load and pre-scaling. Input load cost is amortized 4×.

5. **Zero-sum dequantization**: Instead of `(nibble - zero) * scale`
   per element, accumulate `scale * sum(nibble * x_prescaled)` and
   `zero * sum(x)` separately, applying the zero correction once per
   group. Saves one multiply per element.

---

## Prefill Kernel Adaptation

The prefill tiled GEMM loads B-tiles into threadgroup memory before
MMA. Only the **B-tile load** changes; MMA computation is unchanged.

### Current B-Tile Load (Blocked Layout)

```metal
// Scattered: each thread computes a complex block address
uint n_blk = g_n / BLK_N;
uint n_loc = g_n % BLK_N;
uint k_blk = g_k / BLK_K;
uint k_loc = g_k % BLK_K;
uint byte_idx = (n_blk * total_k_blocks + k_blk) * blk_bytes
              + n_loc * (BLK_K / 2) + k_loc / 2;
uchar packed = B_packed[byte_idx];
```

### Proposed B-Tile Load (Superblock Layout)

```metal
// Sequential: compute superblock position directly
uint sb_idx = g_k / group_size;  // which superblock
uint sb_offset = g_k % group_size;  // position within superblock
uint byte_in_sb = 4 + sb_offset / 2;  // skip inline scale+zero header
uint row_offset = g_n * sb_stride + sb_idx * bytes_per_superblock + byte_in_sb;
uchar packed = W[row_offset];
uchar nibble = (sb_offset % 2 == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);

// Scale/zero: inline in the superblock header
float s = float(*(device const half *)(W + g_n * sb_stride
                                         + sb_idx * bytes_per_superblock));
float z = float(*(device const half *)(W + g_n * sb_stride
                                         + sb_idx * bytes_per_superblock + 2));
half val = half((float(nibble) - z) * s);
```

The key improvement: threads loading the same row read from contiguous
memory addresses (K is contiguous within superblocks). With 256 threads
loading a [64×32] B-tile, threads assigned to the same row access
adjacent bytes → coalesced.

---

## Weight Packing: `pack_superblocks()`

Replace `pack_quantized_blocked()` with a new function that interleaves
scale/zero metadata with weight data:

```rust
/// Pack quantized weights into row-major inline superblocks.
///
/// Input:
///   data:   [N, K/2] row-major packed INT4 bytes
///   scales: [N, G] FP16 scale values (2 bytes each)
///   zeros:  [N, G] FP16 zero-point values (2 bytes each)
///
/// Output:
///   [N, G, 4 + group_size/2] bytes — superblocks with inline metadata
fn pack_superblocks(
    data: &[u8],
    scales: &[u8],
    zeros: &[u8],
    n: usize,
    k: usize,
    group_size: usize,
) -> Vec<u8> {
    let num_groups = k / group_size;
    let data_bytes_per_group = group_size / 2;  // INT4: 2 elements per byte
    let sb_bytes = 4 + data_bytes_per_group;     // 2B scale + 2B zero + data
    let row_bytes = num_groups * sb_bytes;

    let mut out = vec![0u8; n * row_bytes];

    for row in 0..n {
        for g in 0..num_groups {
            let sb_offset = row * row_bytes + g * sb_bytes;

            // Copy inline scale (2 bytes FP16)
            let scale_src = (row * num_groups + g) * 2;
            out[sb_offset..sb_offset + 2]
                .copy_from_slice(&scales[scale_src..scale_src + 2]);

            // Copy inline zero (2 bytes FP16)
            let zero_src = (row * num_groups + g) * 2;
            out[sb_offset + 2..sb_offset + 4]
                .copy_from_slice(&zeros[zero_src..zero_src + 2]);

            // Copy weight data for this group
            let data_src = row * (k / 2) + g * data_bytes_per_group;
            out[sb_offset + 4..sb_offset + sb_bytes]
                .copy_from_slice(&data[data_src..data_src + data_bytes_per_group]);
        }
    }
    out
}
```

### Rust-Side Changes

`AffineQuantizedWeight` simplifies — `scales` and `zeros` are eliminated
as separate buffers:

```rust
pub struct AffineQuantizedWeight {
    /// Superblock data: [N, G, 4 + group_size/2] with inline scale/zero.
    pub data: MetalBuffer,
    /// Number of elements per quantization group.
    pub group_size: u32,
    /// Quantization bit width (4 or 8).
    pub bit_width: u8,
    /// (out_features, in_features) logical dimensions.
    pub shape: (usize, usize),
    /// Optional AWQ per-column scales [in_features] FP16 (kept separate).
    pub awq_scales: Option<MetalBuffer>,
}
```

---

## Implementation Plan

### Phase 1: Superblock Packing + Decode Kernel

**Files changed:**
- `metal/weights.rs` — replace `pack_quantized_blocked()` with `pack_superblocks()`,
  merge scales/zeros into single buffer, simplify `AffineQuantizedWeight`
- `metal/shaders/quantized/affine_matvec.metal` — new `superblock_matvec_int4`
  kernel with coalesced access + pre-scaled input trick
- `metal/ops/quantized.rs` — new pipeline state, updated dispatch
- `metal/projection.rs` — use superblock kernel for decode, update buffer bindings
- `metal/shaders/quantized/affine_fused.metal` — update fused FFN kernel
- `metal/shaders/quantized/affine_batched.metal` — update batched kernels
- `metal/shaders/norm/fused_residual_norm.metal` — update end-of-layer fusion

**Buffer binding change:** All kernels that currently bind `(data, scales, zeros)`
at indices `(1, 2, 3)` switch to `(superblock_data)` at index `(1)`, with scale/zero
read inline from the superblock header.

### Phase 2: Prefill Kernel Adaptation

**Files changed:**
- `metal/shaders/quantized/affine_matmul.metal` — update B-tile load to read
  from superblock layout with inline scale/zero

### Phase 3: INT4×Q8, INT8, and Fused Variants

- `affine_matvec_int4xq8` — superblock layout with Q8 input
- `affine_matvec_int8` — INT8 superblock (132 bytes per block)
- End-of-layer fused kernel — update to read superblocks

### Phase 4: Cleanup

- Remove old `pack_quantized_blocked()` function
- Remove old blocked-layout kernel variants
- Update calibration engine to use new layout
- Update tests

---

## Expected Impact

| Metric | Current | Projected | Source |
|--------|---------|-----------|--------|
| Decode cache-line utilization | 3.1% | ~100% | Coalesced 256B reads |
| Effective weight bandwidth | 31 GB/s | 80–120 GB/s | 3× cache improvement |
| Decode tok/s | 31 | 60–90 | Bandwidth-limited estimate |
| Prefill tok/s | 228 | 300–400 | Better B-tile coalescing |
| PPL | 9.39 | 9.39 | Same weights, same math |
| Memory | 2789 MB | 2789 MB | Zero overhead |

The projected 60–90 tok/s range would match MLX (50–60) and approach
llama.cpp (90) on the same hardware. The remaining gap to llama.cpp's
upper end would be due to Q4_K_M's mixed-precision scales (6-bit super-
block scales vs our uniform FP16) and llama.cpp's compute graph batching.

---

## Validation

### Correctness
```bash
cargo check -p ironmill-inference
cargo test -p ironmill-inference --lib    # 241 tests
```

### Numerical Equivalence
PPL must remain 9.39 ± 0.05:
```bash
cargo run --release -p ironmill-bench --features metal -- \
  --config configs/qwen35-4b-kernel-perf.toml --suite perplexity
```

### Performance Thresholds

| Phase | Decode minimum | Prefill minimum |
|-------|---------------|-----------------|
| Phase 1 (decode kernel) | ≥ 50 tok/s | = baseline (228) |
| Phase 2 (prefill adapt) | ≥ 50 tok/s | ≥ 280 tok/s |
| Phase 3 (all variants) | ≥ 55 tok/s | ≥ 280 tok/s |

---

## Non-Goals

- **Changing the quantization algorithm** (GPTQ, AWQ, etc.) — this is
  purely a memory layout optimization.
- **Mixed-precision scales** (like Q4_K_M's 6-bit super-block scales) —
  FP16 scales are simple and accurate.
- **Speculative decoding** — orthogonal optimization that compounds on
  top of kernel-level gains.
- **Compute graph batching / lazy eval** — architectural change beyond
  kernel scope.

---

## Hardware

Apple M2 Max, 64 GB unified memory, macOS 15.x.
Peak memory bandwidth: 400 GB/s. Metal GPU Family: Apple 8.
