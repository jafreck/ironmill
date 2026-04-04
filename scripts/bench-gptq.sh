#!/bin/bash
# GPTQ INT4 weight quantization benchmark for GPU bundles.
#
# End-to-end: calibrate Hessians → GPTQ-INT4 quantize → benchmark vs FP16.
# Uses the Rust integration test which handles calibration + compilation
# + perplexity evaluation in a single pass.
#
# Usage:
#   ./scripts/bench-gptq.sh
#
# Requires: Metal GPU, Qwen3-0.6B SafeTensors fixture, wikitext2 dataset.
# Pre-requisite: ./scripts/download-fixtures.sh

set -euo pipefail

echo "=== GPTQ INT4 vs FP16 Benchmark ==="
echo ""
echo "Running: calibrate → GPTQ quantize → perplexity eval"
echo ""

cargo test -p ironmill-bench --release --features metal,gptq -- gptq4_full_comparison --ignored --nocapture
