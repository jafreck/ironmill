#!/usr/bin/env bash
# Download test fixture model files for integration tests.
# Run from the repository root: ./scripts/download-fixtures.sh

set -euo pipefail

FIXTURE_DIR="tests/fixtures"
mkdir -p "$FIXTURE_DIR"

echo "Downloading test fixtures..."

# MobileNetV2 ONNX (~14MB)
if [ ! -f "$FIXTURE_DIR/mobilenetv2.onnx" ]; then
    echo "  mobilenetv2.onnx..."
    curl -sL -o "$FIXTURE_DIR/mobilenetv2.onnx.tmp" \
        "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
    # Run ONNX shape inference so all intermediate types are populated.
    python3 -c "import onnx; m=onnx.load('$FIXTURE_DIR/mobilenetv2.onnx.tmp'); m=onnx.shape_inference.infer_shapes(m); onnx.save(m, '$FIXTURE_DIR/mobilenetv2.onnx')" 2>/dev/null \
        && rm -f "$FIXTURE_DIR/mobilenetv2.onnx.tmp" \
        || mv "$FIXTURE_DIR/mobilenetv2.onnx.tmp" "$FIXTURE_DIR/mobilenetv2.onnx"
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

# Whisper Medium encoder ONNX (~769M params, ~1.5GB)
# Validation target for ANE optimization — encoder-only (fixed 30s mel input).
# Use `--skip-whisper` to skip this large download.
if [[ "${1:-}" != "--skip-whisper" ]]; then
    if [ ! -f "$FIXTURE_DIR/whisper-medium-encoder.onnx" ]; then
        echo "  whisper-medium-encoder.onnx (~1.5GB, this may take a while)..."
        curl -sL -o "$FIXTURE_DIR/whisper-medium-encoder.onnx" \
            "https://huggingface.co/onnx-community/whisper-medium/resolve/main/encoder_model.onnx"
    fi
else
    echo "  Skipping whisper-medium-encoder.onnx (--skip-whisper)"
fi

echo "Done. Fixtures in $FIXTURE_DIR/:"
ls -lh "$FIXTURE_DIR/"
