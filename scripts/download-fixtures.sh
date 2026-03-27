#!/usr/bin/env bash
# Download test fixture model files for integration tests.
# Run from the repository root: ./scripts/download-fixtures.sh

set -euo pipefail

FIXTURE_DIR="tests/fixtures"
mkdir -p "$FIXTURE_DIR"

echo "Downloading test fixtures..."

# MNIST ONNX (tiny, ~26KB)
if [ ! -f "$FIXTURE_DIR/mnist.onnx" ]; then
    echo "  mnist.onnx..."
    curl -sL -o "$FIXTURE_DIR/mnist.onnx" \
        "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx"
fi

# SqueezeNet ONNX (~4.7MB)
if [ ! -f "$FIXTURE_DIR/squeezenet1.1.onnx" ]; then
    echo "  squeezenet1.1.onnx..."
    curl -sL -o "$FIXTURE_DIR/squeezenet1.1.onnx" \
        "https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx"
fi

# MobileNet CoreML (.mlmodel, ~16MB)
if [ ! -f "$FIXTURE_DIR/MobileNet.mlmodel" ]; then
    echo "  MobileNet.mlmodel..."
    curl -sL -o "$FIXTURE_DIR/MobileNet.mlmodel" \
        "https://github.com/hollance/MobileNet-CoreML/raw/master/MobileNet.mlmodel"
fi

echo "Done. Fixtures in $FIXTURE_DIR/:"
ls -lh "$FIXTURE_DIR/"
