# Test Design Spec - Optimization Features (Phases 5–8)

Covers integration, CLI, and cross-feature interaction tests for all 22 tasks.
Existing 432 unit tests cover passes in isolation - this spec fills the gaps.

---

## File Layout

```
crates/mil-rs/tests/
├── integration.rs            (existing)
├── onnx_conversion.rs        (existing)
├── common/
│   └── mod.rs                (NEW - shared test helpers)
├── optimization_passes.rs    (NEW - Phase 5 integration tests)
├── llm_features.rs           (NEW - Phase 6 integration tests)
├── pipeline_features.rs      (NEW - Phase 7 integration tests)
├── moe_features.rs           (NEW - Phase 8 integration tests)
└── cross_feature.rs          (NEW - cross-phase composition tests)

crates/ironmill-cli/tests/
└── cli_integration.rs        (NEW - CLI flag and subcommand tests)
```

---

## 1. `optimization_passes.rs` - Phase 5 Integration (15 tests)

These verify that Phase 5 passes compose correctly when run through the
full pipeline, not just in isolation.

### 5.1 NHWC Layout

```
nhwc_pipeline_produces_valid_proto
  Build a program with 3 conv ops → run default PassPipeline → serialize
  via program_to_model → verify transpose ops have correct perm attributes
  and output types with unknown dims (rank preserved).

nhwc_layout_idempotent
  Run LayoutOptimizationPass twice on same program → verify identical output
  (no double-transpose insertion).
```

### 5.2 ANE Validation

```
validate_conv_exceeding_ane_limits
  Build program with conv kernel 32×32 (exceeds ANE 16×16 limit) →
  run validation → verify op flagged as NOT ane_eligible, reported as
  GPU fallback in compute split.

validate_json_report_structure
  Build program with mixed ANE-eligible and ineligible ops → generate
  JSON report → parse JSON → verify: per-op ane_eligible field,
  performance_annotations array, total_estimated_flops > 0,
  ane_compute_pct between 0–100.

validate_flops_proportional_to_shape
  Build two programs: one with [1,64,64,3] conv, one with [1,128,128,3] →
  validate both → verify second has ~4× the estimated flops.
```

### 5.3 Mixed-Precision

```
mixed_precision_pipeline_integration
  Build program with 4 const ops (2 named "attention.*", 2 named "ffn.*")
  → run PassPipeline with mixed-precision preset fp16_int8() →
  verify attention weights are FP16, FFN weights are INT8 (constexpr_affine_dequantize).

mixed_precision_config_from_toml_file
  Write a TOML config to tempfile → load via with_mixed_precision() →
  run pipeline → verify rules applied correctly.

per_channel_int8_produces_correct_shapes
  Build program with [64, 128] const tensor → run Int8QuantizePass with
  PerChannel granularity → verify scale tensor shape is [64] (per output channel),
  not scalar.
```

### 5.4 Fusion Patterns

```
full_transformer_block_fusion
  Build a realistic transformer block: LayerNorm → Linear → GELU → Linear →
  Residual Add → Attention (with GQA broadcast) → run all fusion passes
  via PassPipeline → verify: LayerNorm+Linear fused, GELU+Linear fused,
  residual_add typed correctly, attention fused into grouped_query_attention.
  Count final op count vs initial.

fusion_order_independence
  Build same program → run fusion passes in two different orderings →
  verify identical final op count and types (order shouldn't matter for
  non-overlapping patterns).
```

### 5.5 Compute Unit Annotations

```
compute_unit_annotation_roundtrip
  Build program with mix of conv (ANE-eligible) and large gather (GPU-fallback)
  ops → run ComputeUnitAnnotationPass → serialize via program_to_model →
  deserialize via model_to_program → verify compute_unit attributes preserved.

annotations_match_validation
  Build program → run both validate() and ComputeUnitAnnotationPass →
  verify every op marked Ane by the pass is also flagged ane_eligible
  by validation, and vice versa.
```

---

## 2. `llm_features.rs` - Phase 6 Integration (10 tests)

### 6.1 KV Cache

```
kv_cache_ops_inserted_for_cache_inputs
  Build program with function inputs named "past_key_values.0.key" and
  "past_key_values.0.value" → run KvCachePass → verify kv_cache_read
  and kv_cache_update ops inserted with correct max_seq_length attribute.

kv_cache_shapes_materialized
  Build program with dynamic cache dims → run KvCachePass → verify
  all cache tensor shapes have concrete values (no -1 dims remaining).
```

### 6.2 Autoregressive Support

```
autoregressive_detection_from_cache_tensors
  Build ONNX-like program with past_key_values inputs → run
  detect_autoregressive_pattern → verify program tagged autoregressive.

autoregressive_detection_no_false_positive
  Build ONNX-like program with attention_mask input but NO cache tensors →
  verify program is NOT tagged autoregressive.

stateful_model_export_has_state_descriptors
  Build AR-tagged program with KV cache ops → serialize via
  program_to_model → verify Model proto has state feature descriptors
  for cache tensors.

ar_shape_materialization_fixes_seq_dims
  Build AR program with dynamic sequence length inputs → run
  AutoregressiveShapeMaterializePass → verify sequence dims
  materialized to 1, cache dims materialized to max_seq_length.
```

### 6.3 LoRA Merge

```
lora_merge_weight_math
  Create base weight [4,3] + LoRA A [2,3] + B [4,2] with alpha=1.0 →
  merge_lora → verify result = base + (alpha/rank) * B @ A with
  element-wise comparison (tolerance 1e-5).

lora_merge_disabled_preserves_weights
  Build ONNX model with LoRA weights → convert with merge_lora=false →
  verify original base weights unchanged, LoRA initializers removed.
```

### 6.4 Updatable Export

```
updatable_model_has_training_inputs
  Build program → call program_to_updatable_model with 2 updatable layers →
  verify Model proto has is_updatable=true and training inputs inferred
  from the updatable layers' inputs.

updatable_model_loss_references_output_tensor
  Build program with updatable layers → export → verify loss layer input
  references the layer's output tensor name, not the operation name.
```

---

## 3. `pipeline_features.rs` - Phase 7 Integration (10 tests)

### 7.1 Multi-ONNX Pipeline

```
pipeline_manifest_parse_and_validate
  Write a TOML manifest with 3 stages and inter-stage deps → parse →
  verify topological order, stage configs, and that cycle detection
  works (add a cycle → assert error).

pipeline_output_manifest_json
  Convert a 2-stage pipeline (using in-memory ONNX fixtures) →
  verify output JSON manifest has correct stage names, I/O tensor
  descriptors, and dependency edges.
```

### 7.2 Op Splitting

```
oversized_matmul_split_into_tiles
  Build program with single matmul [1024, 8192] × [8192, 4096] →
  set budget to 64MB → run OpSplittingPass → verify op replaced with
  multiple slice+matmul+concat ops, each tile's estimated memory < budget.

splitting_preserves_output_dimensions
  Build program with known weight shapes → split → verify concat output
  dimensions match original output dimensions.
```

### 7.3 Causal Convolution

```
causal_conv_attribute_survives_pipeline
  Build ONNX-style conv with asymmetric left-only padding [2,0] →
  convert via onnx_to_mil → run full PassPipeline → verify causal
  attribute preserved through fusion (conv+relu should still fuse,
  causal attribute on fused op).
```

### 7.4 Codebook / RVQ

```
rvq_pattern_fused_end_to_end
  Build program simulating 4-codebook RVQ: 4 const codebooks → 4 gathers →
  3 nested adds → run CodebookOptimizationPass → verify fused into single
  codebook_gather op with stacked [4, K, D] weight.
```

### 7.5 Speculative Decoding

```
model_split_produces_valid_draft_and_verifier
  Build program with 6 transformer layers → split at layer 2 →
  verify draft has 2 layers, verifier has 6, both have same
  input names, and draft outputs are a subset of verifier ops.

split_with_different_quantization
  Split program → apply FP16 to draft, INT8 to verifier →
  serialize both → verify different weight dtypes in each.
```

---

## 4. `moe_features.rs` - Phase 8 Integration (12 tests)

### 8.1 MoE Detection & Splitting

```
moe_detection_name_based
  Build program with ops named "expert_0.linear", "expert_1.linear",
  "gate.linear" → detect_moe → verify topology with 2 experts.

moe_split_produces_shared_and_experts
  Build 2-expert MoE program → split_moe → verify shared program
  contains router ops, each expert program contains only its ops,
  and manifest JSON has correct structure.

moe_manifest_describes_execution_flow
  Split MoE → verify manifest has "shared" stage with router outputs
  and "expert-N" stages with correct I/O tensor names.
```

### 8.2 Multi-Function Bundle

```
multi_function_model_has_all_functions
  Split MoE → bundle via program_to_multi_function_model → verify
  Model proto has functions named "main", "expert_0", "expert_1".

multi_function_dedup_only_shared_consts
  Create MoE with shared embedding + per-expert weights → bundle →
  verify shared consts are stubs in expert functions, but each
  expert's unique weights have full data (no cross-expert dedup).
```

### 8.3 Per-Expert Quantization

```
hot_expert_fp16_cold_expert_palettized
  Build 2-expert program → configure expert_0 as "hot" (FP16),
  expert_1 as "cold" (4-bit palettize) → run PerExpertQuantPass →
  verify expert_0 weights are FP16, expert_1 has constexpr_lut_to_dense.
```

### 8.4 Top-K Fusion

```
top_k_fusion_keeps_only_selected_experts
  Build 4-expert program → provide frequency profile
  {0: 0.5, 1: 0.3, 2: 0.15, 3: 0.05} → fuse top-2 →
  verify result has ops from experts 0 and 1 only, router removed.
```

### 8.5 Sub-2-bit Quantization

```
one_bit_palettization_produces_two_centroids
  Build program with const tensor → PalettizePass(1) → verify
  LUT has exactly 2 entries and packed indices use 1-bit width.

grouped_palettization_per_group_codebooks
  Build program with [128, 64] weight → GroupedPalettizePass(group_size=32) →
  verify 4 groups, each with its own codebook entries, n_groups attribute set.

prequantized_gptq_weights_preserved
  Build ONNX model with initializers named "*.qweight", "*.qzeros",
  "*.scales" → convert → verify quantization_format attribute set,
  weights not re-quantized by downstream passes.
```

### 8.6 Configurable Pipeline

```
toml_config_matches_programmatic_pipeline
  Build identical pipeline two ways: via PassPipeline::new() with
  with_fp16(), and via from_config_str() with equivalent TOML →
  run both on same program → verify identical output op count.

pipeline_report_comparison
  Run two configs on same program → call PipelineReport::compare() →
  verify output has per-pass rows with before/after op counts and
  delta columns.
```

### 8.7 Layer-Wise Scheduling

```
layer_schedule_detects_all_types
  Build program with conv+bn+relu cluster, attention block, FFN block,
  and layer_norm → run LayerSchedulePass with config
  {attention: fp16, ffn: int8} → verify attention weights are FP16,
  FFN weights are INT8, conv and norm unchanged.
```

### 8.8 ANE Direct (Feature-Gated)

```
#[cfg(feature = "ane-direct")]
ane_compiler_not_available_returns_error
  Call AneCompiler::compile() with nonexistent path → verify
  error returned (not a panic).

backend_enum_defaults_to_xcrun
  Verify Backend::default() == Backend::Xcrun and that
  compile_model_with_backend with Xcrun backend works the same
  as the existing compile_model function.
```

---

## 5. `cross_feature.rs` - Cross-Phase Composition (8 tests)

These are the highest-value tests - they verify features from different
phases compose without interference.

```
nhwc_plus_mixed_precision_pipeline
  Build conv-heavy program → run pipeline with NHWC layout + mixed-precision
  (attention FP16, conv INT8) → verify: transposes inserted, quantization
  applied AFTER layout change, output serializable.

kv_cache_plus_compute_unit_annotations
  Build AR program with KV cache → run KvCachePass + ComputeUnitAnnotationPass
  → verify cache ops annotated with compute_unit, non-cache ops also annotated.

moe_split_plus_per_expert_quant_plus_bundle
  Build MoE program → split → apply per-expert quantization (hot=FP16,
  cold=4-bit) → bundle into multi-function model → verify: single Model
  proto with multiple functions, each expert function has correct weight
  dtype, shared weights deduplicated.

op_splitting_plus_codebook_optimization
  Build program with oversized matmul AND RVQ codebook pattern → run both
  OpSplittingPass and CodebookOptimizationPass → verify both transformations
  applied without interference.

full_pipeline_with_all_default_passes
  Build a realistic transformer program (conv → attention → FFN → norm) →
  run PassPipeline::new() (all default passes) → verify: passes don't
  error, op count reduced, output serializable via program_to_model.

mixed_precision_plus_layer_schedule
  Build program → configure mixed-precision with type rules AND
  layer-schedule with layer-type rules → verify layer-schedule takes
  precedence for detected layers, mixed-precision covers the rest.

model_split_plus_different_pipelines
  Split program into draft/verifier → run different PassPipeline configs
  on each (draft: aggressive quant, verifier: FP16) → serialize both →
  verify different weight representations.

configurable_pipeline_with_all_new_passes
  Load a TOML config that enables every new pass (kv-cache, op-split,
  compute-unit, codebook, layer-schedule) → run on program → verify
  all passes execute without errors.
```

---

## 6. `cli_integration.rs` - CLI Tests (20 tests)

All CLI tests use `cargo run -p ironmill-cli --quiet --` via `std::process::Command`.
Tests that require model fixtures use `fixture_path()`.

### Compile Subcommand

```
cli_compile_default
  ironmill compile tests/fixtures/mnist.onnx → exit 0, output .mlpackage exists.

cli_compile_with_fp16
  ironmill compile mnist.onnx --quantize fp16 → exit 0.

cli_compile_with_int8
  ironmill compile mnist.onnx --quantize int8 --cal-data <tmpdir> → exit 0.

cli_compile_with_mixed_precision_preset
  ironmill compile mnist.onnx --quantize mixed-fp16-int8 → exit 0.

cli_compile_with_quantize_config
  Write TOML to tmpfile → ironmill compile mnist.onnx --quantize-config <tmp> → exit 0.

cli_compile_no_fusion
  ironmill compile mnist.onnx --no-fusion → exit 0, verify output
  (should have more ops than default).

cli_compile_with_input_shape
  ironmill compile mnist.onnx --input-shape "input:1,1,28,28" → exit 0.

cli_compile_annotate_compute_units
  ironmill compile mnist.onnx --annotate-compute-units → exit 0.

cli_compile_ane_memory_budget
  ironmill compile mnist.onnx --ane-memory-budget 512MB → exit 0.

cli_compile_split_draft_layers
  ironmill compile mnist.onnx --split-draft-layers 2 → exit 0,
  verify *-draft.mlpackage and *-verifier.mlpackage both exist.

cli_compile_target_cpu_and_ne
  ironmill compile mnist.onnx --target cpu-and-ne → exit 0 (no warning).

cli_compile_pipeline_config
  Write pipeline TOML → ironmill compile mnist.onnx --pipeline-config <tmp> → exit 0.
```

### Validate Subcommand

```
cli_validate_text_format
  ironmill validate mnist.onnx → exit 0, stdout contains "ANE".

cli_validate_json_format
  ironmill validate mnist.onnx --format json → exit 0, stdout parses as valid JSON
  with "ane_compute_pct" field.
```

### Other Subcommands

```
cli_inspect
  ironmill inspect mnist.onnx → exit 0, stdout non-empty.

cli_compile_pipeline_subcommand
  Write pipeline TOML manifest referencing mnist.onnx →
  ironmill compile-pipeline <manifest> -o <tmpdir> → exit 0.

cli_pipeline_report
  Write two pipeline TOMLs → ironmill pipeline-report mnist.onnx <a.toml> <b.toml>
  → exit 0, stdout contains comparison table.
```

### Error Cases

```
cli_compile_missing_input
  ironmill compile nonexistent.onnx → exit non-zero, stderr contains error.

cli_compile_conflicting_quantize_flags
  ironmill compile mnist.onnx --quantize fp16 --quantize int8 →
  verify graceful error (not panic).

cli_compile_invalid_epochs
  ironmill compile mnist.onnx --updatable-layers "x" --epochs -5 →
  exit non-zero, stderr contains "positive".
```

---

## Shared Test Utilities - `crates/mil-rs/tests/common/mod.rs`

```rust
/// Path to test fixture files.
pub fn fixture_path(name: &str) -> PathBuf { ... }

/// Build a minimal transformer-like program with N layers.
/// Each layer: LayerNorm → Linear (Q/K/V) → MatMul → Softmax → Linear → Add
pub fn build_transformer_program(n_layers: usize) -> Program { ... }

/// Build a program with N conv ops (for NHWC/fusion testing).
pub fn build_conv_program(n_convs: usize) -> Program { ... }

/// Build a program simulating MoE with N experts.
/// Includes: router (linear → softmax), N expert linears, gate output.
pub fn build_moe_program(n_experts: usize) -> Program { ... }

/// Build a program with KV cache inputs (autoregressive).
/// Includes: past_key_values inputs/outputs, attention ops.
pub fn build_autoregressive_program(max_seq_len: usize) -> Program { ... }

/// Build a program with RVQ codebook pattern (N codebooks).
/// Each codebook: const → gather → chain of adds.
pub fn build_rvq_program(n_codebooks: usize) -> Program { ... }

/// Run default pipeline and return the report.
pub fn run_default_pipeline(program: &mut Program) -> PipelineReport { ... }

/// Serialize program to Model proto and back, verify round-trip.
pub fn assert_serialization_roundtrip(program: &Program) { ... }

/// Run CLI command and return (success, stdout, stderr).
pub fn run_cli(args: &[&str]) -> (bool, String, String) { ... }
```

---

## Summary

| Test File                  | Tests | Coverage Area                       |
|----------------------------|------:|-------------------------------------|
| `optimization_passes.rs`   |    15 | Phase 5 pass composition            |
| `llm_features.rs`          |    10 | Phase 6 AR/KV/LoRA/updatable        |
| `pipeline_features.rs`     |    10 | Phase 7 multi-model/split/codebook  |
| `moe_features.rs`          |    12 | Phase 8 MoE/quant/config            |
| `cross_feature.rs`         |     8 | Cross-phase feature composition     |
| `cli_integration.rs`       |    20 | CLI flags, subcommands, errors      |
| **Total new tests**        |**75** |                                     |

Combined with existing 432 unit tests + 17 integration tests = **524 total tests**.
