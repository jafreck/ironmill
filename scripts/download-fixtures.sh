#!/usr/bin/env bash
# Download test fixture model files for integration tests.
# Run from the repository root: ./scripts/download-fixtures.sh
#
# Flags:
#   --skip-llm      Skip Qwen3-0.6B SafeTensors + GGUF downloads (~2GB)
#   --skip-whisper   Skip whisper-medium-encoder.onnx download (~1.5GB)
#   --skip-all       Skip all large downloads (LLM + whisper)

set -euo pipefail

FIXTURE_DIR="tests/fixtures"
mkdir -p "$FIXTURE_DIR"

# Parse flags
SKIP_LLM=false
SKIP_WHISPER=false
for arg in "$@"; do
    case "$arg" in
        --skip-llm)     SKIP_LLM=true ;;
        --skip-whisper)  SKIP_WHISPER=true ;;
        --skip-all)      SKIP_LLM=true; SKIP_WHISPER=true ;;
    esac
done

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

# Qwen3-0.6B SafeTensors + GGUF for weight-format benchmarks.
# Use `--skip-llm` or `--skip-all` to skip these large downloads.
if [ "$SKIP_LLM" = false ]; then
    # Qwen3-0.6B SafeTensors (~1.4GB)
    QWEN_DIR="$FIXTURE_DIR/Qwen3-0.6B"
    if [ ! -d "$QWEN_DIR" ] || [ ! -f "$QWEN_DIR/model.safetensors" ]; then
        echo "  Qwen3-0.6B SafeTensors (~1.4GB, this may take a while)..."
        mkdir -p "$QWEN_DIR"
        curl -sL -o "$QWEN_DIR/config.json" \
            "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json"
        curl -sL -o "$QWEN_DIR/model.safetensors" \
            "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors"
    fi

    # Qwen3-0.6B GGUF Q8_0 (~639MB)
    if [ ! -f "$FIXTURE_DIR/Qwen3-0.6B-Q8_0.gguf" ]; then
        echo "  Qwen3-0.6B-Q8_0.gguf (~639MB, this may take a while)..."
        curl -sL -o "$FIXTURE_DIR/Qwen3-0.6B-Q8_0.gguf" \
            "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf"
    fi
else
    echo "  Skipping Qwen3-0.6B downloads (--skip-llm)"
fi

# Whisper Medium encoder ONNX (~769M params, ~1.5GB)
# Validation target for ANE optimization — encoder-only (fixed 30s mel input).
# Use `--skip-whisper` or `--skip-all` to skip this large download.
if [ "$SKIP_WHISPER" = false ]; then
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
