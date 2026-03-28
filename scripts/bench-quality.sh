#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "==> Building quality_report (release)..."
cargo build --release --example quality_report --quiet

echo ""
./target/release/examples/quality_report tests/fixtures/squeezenet1.1.onnx
./target/release/examples/quality_report tests/fixtures/mnist.onnx
