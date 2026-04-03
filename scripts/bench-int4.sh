#!/bin/bash
# INT4 weight quantization benchmark for GPU bundles.
#
# Compiles a model with FP16 (baseline) and INT4 affine quantization,
# then benchmarks both on the Metal backend with multiple KV cache configs.
#
# Usage:
#   ./scripts/bench-int4.sh [MODEL_DIR]
#
# If MODEL_DIR is not specified, uses the default Qwen3-8B snapshot path.

set -euo pipefail

MODEL_DIR="${1:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
OUTPUT_DIR="${2:-.}"

FP16_BUNDLE="${OUTPUT_DIR}/qwen3-8b-fp16.ironml-gpu"
INT4_BUNDLE="${OUTPUT_DIR}/qwen3-8b-int4.ironml-gpu"

if [ ! -d "$MODEL_DIR" ] && [ ! -f "$MODEL_DIR" ]; then
    echo "Error: model path not found: $MODEL_DIR"
    echo "Usage: $0 [MODEL_DIR] [OUTPUT_DIR]"
    exit 1
fi

echo "=== INT4 Weight Quantization Benchmark ==="
echo "Model: $MODEL_DIR"
echo ""

# Compile FP16 baseline (no quantization)
echo "--- Compiling FP16 baseline ---"
ironmill compile "$MODEL_DIR" --target gpu --output "$FP16_BUNDLE"
echo ""

# Compile INT4 weight-quantized
echo "--- Compiling INT4 affine quantized ---"
ironmill compile "$MODEL_DIR" --target gpu --quantize int4 --output "$INT4_BUNDLE"
echo ""

# Compare file sizes
echo "=== File Sizes ==="
du -sh "$FP16_BUNDLE" "$INT4_BUNDLE"
echo ""

# Benchmark both (bench runs fp16-kv and tq-int4-kv variants automatically)
echo "=== Benchmark ==="
ironmill-bench \
    --model "$FP16_BUNDLE" \
    --model "$INT4_BUNDLE" \
    --backend metal \
    --iterations 1024 \
    --warmup 10 \
    --runs 1 \
    --perplexity \
    --perplexity-sequences 1 \
    --output markdown
