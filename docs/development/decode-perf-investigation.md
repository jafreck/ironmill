# Investigation Brief: Why MLX Achieves 2.5× Higher Decode Throughput Than ironmill

## Measured Performance (M2 Max 64GB, Qwen 3.5-4B INT4)

| Framework | Quantization | Decode tok/s | Prefill tok/s |
|-----------|-------------|-------------|---------------|
| **MLX** | INT4 gs=128 | **97** | — |
| **llama.cpp** | Q4_K_M gs=64 | **67** | **1066** |
| **ironmill** | INT4 superblock gs=128 | **38** | **242** |

ironmill is 2.5× behind MLX on identical quantization (INT4 gs=128) and 1.8× behind llama.cpp.

### Hot Kernel Under Test

The target configuration is **AWQ-INT4** (INT4 with activation-aware weight
quantization). During decode, this takes the following dispatch path in
`projection.rs`:

1. `WeightBuffer::AffineQuantized` with `awq_scales.is_some() == true`
2. Skips the `int4xq8` fast-path (that path requires `awq_scales.is_none()`)
3. Falls through to `encode_affine_projection` → **`superblock_matvec_int4`**

The `superblock_matvec_int4` kernel at line 265 of `projection.rs` is the hot
kernel for ALL experiments below. The `int4xq8` kernel is NOT exercised with
AWQ weights.

## What We Already Ruled Out

### ✅ ALU / Compute Bottleneck — NOT the issue
Implemented MLX's pre-scaled input trick (eliminates 8 shifts + 8 subtracts per uint32). Zero decode improvement. The kernel is **memory-bound**, not compute-bound.

### ✅ Integer Division Overhead — Partially addressed
Implemented compile-time `#define GS` specialization (separate metallibs per group_size). Compiler optimizes `/ GS` → `>> 7`. Gave modest improvement (36 → 38 tok/s) but NOT the 2.5× gap.

### ✅ Memory Layout (Coalescing) — Partially addressed
Migrated from blocked layout (3.1% cache utilization) to row-major superblocks. Gave ~23% improvement (31 → 38 tok/s). Consecutive lanes now read consecutive bytes within superblock data.

### ✅ PPL / Correctness — Verified
ironmill PPL = 9.39, exact match across all kernel variants.

## What We Have NOT Investigated

### 1. GPU Occupancy and Thread Structure
**This is the #1 suspect.** ironmill uses:
- 64 threads/TG (2 simdgroups × 32 threads)
- 8 rows/TG (4 rows/simdgroup)
- 320 TGs for N=2560

MLX uses:
- 32 threads/TG (1 simdgroup) for basic QMV
- Variable rows/TG (12-18 typical, architecture-dependent)
- ~142-213 TGs for N=2560

**Key question**: Does ironmill's 2-simdgroup design cause higher register pressure that reduces the number of concurrent TGs per GPU core? Profile with Xcode Metal System Trace to measure actual occupancy.

### 2. MLX's QMV Quad Variant
MLX has a `affine_qmv_quad` kernel variant using 4-thread "quad groups" for dimensions D=64 and D=128. This is a fundamentally different parallelism strategy — 4 threads cooperate on a group of K elements instead of 32 threads each handling independent K ranges. This may enable better memory access patterns for small group sizes.

**Action**: Read the actual `affine_qmv_quad` kernel source at `mlx/backend/metal/kernels/quantized_nax.h` (search for `qmv_quad` template). Compare its memory access pattern to ironmill's.

### 3. Number of Weight Reads Per Token
ironmill processes 4 rows per simdgroup. Each row reads from a different memory location (separated by `sb_stride` = 1360 bytes for K=2560, gs=128). That's 4 scattered reads per simdgroup per iteration.

MLX with more rows per TG might ALSO scatter, but if its iteration structure is different (e.g., iterating over K in the OUTER loop and rows in the inner loop, vs ironmill which iterates over K in the outer loop with a row inner loop), the memory access pattern changes significantly.

**Action**: Profile both frameworks with Instruments Metal System Trace. Compare:
- Bandwidth utilization %
- Cache hit rate
- Memory stall cycles
- Occupancy (concurrent TGs per core)

### 4. Dispatch Overhead / Graph Batching
ironmill dispatches each projection as a separate Metal compute command. For a transformer layer with Q, K, V, O, gate, up, down projections = 7+ dispatches per layer × 32 layers = 224+ dispatches per token.

MLX may batch multiple operations or have lower per-dispatch overhead. llama.cpp uses a compute graph that batches operations.

**Action**: Measure dispatch overhead. Add timing between `encoder.dispatch_threadgroups` calls. Check if the 38 tok/s is spending significant time in dispatch overhead vs actual GPU compute.

### 5. Superblock Header Gaps vs Separate Arrays
ironmill's superblock layout injects 4-byte headers every 64 bytes (for gs=128). MLX uses separate arrays with zero gaps. For 32 lanes reading from a superblock:
- Lanes 0-15: read bytes [4, 68) of superblock g
- Lanes 16-31: read bytes [72, 136) of superblock g+1
- Gap: 4 bytes (header of superblock g+1)

This means 2 cache lines for 128B useful data = 50% utilization per SIMD group iteration. MLX's contiguous layout achieves 100%.

**Action**: Test by temporarily reverting to separate scale/bias arrays (like MLX) while keeping row-major K-contiguous weights. Compare decode tok/s. This isolates the superblock header impact.

### 6. Fused Operations
MLX community has a fused RMSNorm+QMV kernel (https://github.com/quivent/mlx-fused-qmv) that eliminates the barrier between normalization and the first projection. ironmill already has `fused_residual_norm_affine_matvec_int4` but it may not be used for all applicable layers.

**Action**: Check how many layers actually use the fused kernel vs separate norm+projection. Each missed fusion = 1 extra dispatch + 1 memory round-trip for the normalized hidden state.

### 7. Input Vector Caching
For decode (M=1), the input vector `A[K]` is shared across ALL N output rows. Ideally it's loaded once into L1/L2 cache and broadcast to all TGs. With 320 TGs all reading the same K-element input, the GPU's L2 broadcast efficiency matters.

MLX may structure its TGs to maximize L1 reuse of the input vector (e.g., by having each TG process many rows sequentially, so the same cached input serves more rows before eviction).

**Action**: Experiment with different rows_per_TG values (2, 4, 8, 16, 32) and measure decode tok/s for each. Plot the curve to find the optimum.

## Concrete Experiments to Run

### Experiment A: Occupancy sweep
Change `SB_ROWS_PER_SG` and `SB_NUM_SIMDGROUPS` in the decode kernel:
```
Config 1: 1 SG × 2 rows = 2 rows/TG, 32 threads → ceil(N/2) = 1280 TGs
Config 2: 1 SG × 4 rows = 4 rows/TG, 32 threads → ceil(N/4) = 640 TGs
Config 3: 2 SG × 2 rows = 4 rows/TG, 64 threads → ceil(N/4) = 640 TGs  (current-like)
Config 4: 2 SG × 4 rows = 8 rows/TG, 64 threads → ceil(N/8) = 320 TGs  (current)
Config 5: 2 SG × 8 rows = 16 rows/TG, 64 threads → ceil(N/16) = 160 TGs
Config 6: 4 SG × 4 rows = 16 rows/TG, 128 threads → ceil(N/16) = 160 TGs
```
## Test Process

Each step below is designed to isolate ONE variable, measure its impact,
and decide whether to keep the change before moving on. Run the full
benchmark after each step: decode tok/s is the primary metric, PPL must
remain 9.39 ± 0.05.

```
Benchmark command:
  cargo run --release -p ironmill-bench --features metal -- \
    --config configs/qwen35-4b-decode-perf.toml
```

### Step 1: Occupancy sweep (isolates thread structure)

**What to change**: The constants are defined in TWO shader files — BOTH must
be updated for each config:

1. `crates/ironmill-inference/src/metal/shaders/quantized/superblock_header.metal` (lines 25-26):
```metal
constant constexpr uint SB_ROWS_PER_SG = <vary>;     // line 25
constant constexpr uint SB_NUM_SIMDGROUPS = <vary>;   // line 26
// SB_ROWS_PER_TG is derived (line 27): SB_NUM_SIMDGROUPS * SB_ROWS_PER_SG
```

2. `crates/ironmill-inference/src/metal/shaders/quantized/affine_common.metal` (lines 55-56):
```metal
constant constexpr uint SB_ROWS_PER_SG = <vary>;     // line 55
constant constexpr uint SB_NUM_SIMDGROUPS = <vary>;   // line 56
// SB_ROWS_PER_TG is derived (line 57): SB_NUM_SIMDGROUPS * SB_ROWS_PER_SG
```

3. The dispatch in `crates/ironmill-inference/src/metal/projection.rs` (line 265):
```rust
// Current: ceil(N/8) TGs × 64 threads (8 rows per TG)
encoder.dispatch_threadgroups((n.div_ceil(8), 1, 1), (64, 1, 1));
// Change to match new SB_ROWS_PER_TG and SB_NUM_SIMDGROUPS × 32:
encoder.dispatch_threadgroups(
    (n.div_ceil(NEW_ROWS_PER_TG), 1, 1),
    (NEW_NUM_SIMDGROUPS * 32, 1, 1),
);
```

**Configs to test** (change all THREE locations, build, bench, record — one at a time):

| Config | SGs/TG | Rows/SG | Threads/TG | Rows/TG | Expected TGs (N=2560) |
|--------|--------|---------|------------|---------|----------------------|
| A | 1 | 2 | 32 | 2 | 1280 |
| B | 1 | 4 | 32 | 4 | 640 |
| C | 1 | 8 | 32 | 8 | 320 |
| D | 2 | 2 | 64 | 4 | 640 |
| **E (current)** | **2** | **4** | **64** | **8** | **320** |
| F | 2 | 8 | 64 | 16 | 160 |

**Decision rule**: Pick the config with the highest decode tok/s. If the
winner is ≥ 5% better than current, adopt it. The winner tells us:
- If fewer threads/TG wins → occupancy was the bottleneck
- If more rows/TG wins → amortization was the bottleneck
- If current wins → thread structure is not the issue, move to Step 2

### Step 2: Profile with Xcode (diagnose if unclear)

Only needed if Step 1 doesn't yield a clear winner or the improvement
is small (< 10%). Run Metal System Trace on the best config from Step 1:

```bash
xctrace record --template 'Metal System Trace' --launch -- \
  cargo run --release -p ironmill-bench --features metal -- \
    --config configs/qwen35-4b-decode-perf.toml
```

Open in Instruments. For the `superblock_matvec_int4` shader, record:
- **GPU Bandwidth Utilization** (% of peak)
- **Occupancy** (concurrent TGs per core)
- **Stall cycles** breakdown (memory vs ALU vs barrier)
- **L1/L2 cache hit rates**

This tells you WHERE the bottleneck is: memory latency (need more
occupancy), memory bandwidth (need fewer bytes per token), or something
else.

### Step 3: Separate arrays test (isolates superblock header impact)

**What to change**: Revert to separate `data`, `scales`, `zeros` Metal
buffers (like MLX uses) while KEEPING row-major K-contiguous weights.
This means: `pack_superblocks()` → simple row-major pack, and the kernel
reads scale/zero from separate arrays instead of inline headers.

**Decision rule**: If separate arrays > superblock by ≥ 5%, the 4-byte
headers between groups are causing meaningful cache waste. Consider
reverting to MLX-style layout. If ≤ 5% difference, superblocks are fine
and the issue is elsewhere.

### Step 4: Dispatch overhead measurement

**What to change**: Nothing in the kernel. Add timing around the full
decode pass (all layers) vs just the GPU portion:

```rust
let cpu_start = std::time::Instant::now();
// ... encode all dispatches ...
encoder.end_encoding();
command_buffer.commit();
command_buffer.wait_until_completed();
let gpu_time = command_buffer.gpu_end_time() - command_buffer.gpu_start_time();
let wall_time = cpu_start.elapsed();
eprintln!("GPU: {gpu_time:.3}ms, wall: {wall_time:.3}ms, overhead: {:.1}%",
    (1.0 - gpu_time / wall_time.as_secs_f64()) * 100.0);
```

**Decision rule**: If overhead > 20%, the issue is dispatch/encoding CPU
cost, not GPU kernel speed. Investigate command buffer batching, parallel
encoding, or compute graph compilation.

### Step 5: Read MLX QMV source (if still behind after Steps 1-4)

Fetch and read the actual kernel function (not just helpers):
- `mlx/backend/metal/kernels/quantized_nax.h` — search for `affine_qmv_fast`
- `mlx/backend/metal/quantized.cpp` — search for `get_qmv_batch_limit` and dispatch logic

Document the EXACT differences in:
1. How many rows each thread handles
2. How the K dimension is iterated (inner vs outer loop)
3. Whether threadgroup memory is used for input vector caching
4. The `quad` variant's cooperative access pattern

Then implement the most promising difference as a new kernel variant and
benchmark it.

## Success Criteria

| Milestone | Decode tok/s | What it means |
|-----------|-------------|---------------|
| Current | 38 | Baseline after superblock + pre-scaled |
| Parity with llama.cpp | ≥ 65 | Thread structure + memory tuning sufficient |
| Parity with MLX | ≥ 90 | Need MLX-level kernel design or graph batching |
| Spec target | ≥ 60 | Phase 1 threshold from superblock-weight-layout.md |

## Files to Read

### ironmill
- Current decode kernel: `crates/ironmill-inference/src/metal/shaders/quantized/affine_matvec.metal`
- Dispatch: `crates/ironmill-inference/src/metal/projection.rs` (encode_affine_projection, **line 265** = decode dispatch)
- Superblock header (has SB_ROWS_PER_SG): `crates/ironmill-inference/src/metal/shaders/quantized/superblock_header.metal`
- Affine common (ALSO has SB_ROWS_PER_SG): `crates/ironmill-inference/src/metal/shaders/quantized/affine_common.metal`
- Build system: `crates/ironmill-inference/build.rs` (superblock metallib compilation)
- Benchmark config: `configs/qwen35-4b-decode-perf.toml` (AWQ-INT4, decode-only)

### MLX (GitHub: ml-explore/mlx)
- QMV kernel: `mlx/backend/metal/kernels/quantized_nax.h` (search for `affine_qmv_fast` and `affine_qmv_quad`)
- Helpers: `mlx/backend/metal/kernels/quantized.h` (`load_vector`, `qdot`)
- Dispatch: `mlx/backend/metal/quantized.cpp` (kernel selection, TG sizing)
- Template instantiation: `mlx/backend/metal/kernels/quantized.metal`

### llama.cpp (GitHub: ggml-org/llama.cpp)
- Metal kernels: `ggml/src/ggml-metal/ggml-metal.metal` (search for `kernel_mul_mv_q4_K`)
- Kernel tuning: `ggml/src/ggml-metal/ggml-metal-impl.h` (N_R0_Q4_K, N_SG_Q4_K)
