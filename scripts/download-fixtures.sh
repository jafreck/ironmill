#!/usr/bin/env bash
# Download test fixture model files for integration tests.
# Run from the repository root: ./scripts/download-fixtures.sh
#
# Flags:
#   --skip-whisper   Skip the large whisper-medium-encoder download
#   --skip-large     Skip all large (>100MB) transformer model downloads
#   --all            Download everything including all transformer models

set -euo pipefail

FIXTURE_DIR="tests/fixtures"
mkdir -p "$FIXTURE_DIR"

SKIP_WHISPER=false
SKIP_LARGE=false
for arg in "$@"; do
    case "$arg" in
        --skip-whisper) SKIP_WHISPER=true ;;
        --skip-large)   SKIP_LARGE=true ;;
        --all)          SKIP_WHISPER=false; SKIP_LARGE=false ;;
    esac
done

echo "Downloading test fixtures..."

# ── Core fixtures (always downloaded) ─────────────────────────────────

# MobileNetV2 ONNX (~14MB) — CNN classification
if [ ! -f "$FIXTURE_DIR/mobilenetv2.onnx" ]; then
    echo "  mobilenetv2.onnx..."
    curl -sL -o "$FIXTURE_DIR/mobilenetv2.onnx.tmp" \
        "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
    # Run ONNX shape inference so all intermediate types are populated.
    python3 -c "import onnx; m=onnx.load('$FIXTURE_DIR/mobilenetv2.onnx.tmp'); m=onnx.shape_inference.infer_shapes(m); onnx.save(m, '$FIXTURE_DIR/mobilenetv2.onnx')" 2>/dev/null \
        && rm -f "$FIXTURE_DIR/mobilenetv2.onnx.tmp" \
        || mv "$FIXTURE_DIR/mobilenetv2.onnx.tmp" "$FIXTURE_DIR/mobilenetv2.onnx"
fi

# SqueezeNet ONNX (~4.7MB) — CNN classification
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

# ── Transformer fixtures (small, <50MB each) ──────────────────────────

# Whisper-tiny encoder (~33MB) — encoder-only audio transformer, d=384
if [ ! -f "$FIXTURE_DIR/whisper-tiny-encoder.onnx" ]; then
    echo "  whisper-tiny-encoder.onnx (~33MB)..."
    curl -sL -o "$FIXTURE_DIR/whisper-tiny-encoder.onnx" \
        "https://huggingface.co/onnx-community/whisper-tiny/resolve/main/onnx/encoder_model.onnx"
fi

# ── Transformer fixtures (large, >100MB each) ─────────────────────────
# Use --skip-large to skip these, or --all to include everything.

if [ "$SKIP_LARGE" = false ]; then

    # Whisper-tiny decoder (~118MB) — decoder with cross-attention, d=384
    if [ ! -f "$FIXTURE_DIR/whisper-tiny-decoder.onnx" ]; then
        echo "  whisper-tiny-decoder.onnx (~118MB)..."
        curl -sL -o "$FIXTURE_DIR/whisper-tiny-decoder.onnx" \
            "https://huggingface.co/onnx-community/whisper-tiny/resolve/main/onnx/decoder_model.onnx"
    fi

    # DistilBERT (~256MB) — encoder-only NLU transformer, d=768
    if [ ! -f "$FIXTURE_DIR/distilbert.onnx" ]; then
        echo "  distilbert.onnx (~256MB)..."
        curl -sL -o "$FIXTURE_DIR/distilbert.onnx" \
            "https://huggingface.co/philschmid/distilbert-onnx/resolve/main/model.onnx"
    fi

    # ViT-base-patch16-224 (~347MB) — vision transformer, d=768
    if [ ! -f "$FIXTURE_DIR/vit-base.onnx" ]; then
        echo "  vit-base.onnx (~347MB)..."
        curl -sL -o "$FIXTURE_DIR/vit-base.onnx" \
            "https://huggingface.co/google/vit-base-patch16-224/resolve/main/model.onnx"
    fi

    # Qwen3-0.6B (~300MB graph + ~1.9GB external data) — decoder-only LLM, d=1024
    if [ ! -f "$FIXTURE_DIR/qwen3-0.6b.onnx" ]; then
        echo "  qwen3-0.6b.onnx (~300MB + ~1.9GB data)..."
        curl -sL -o "$FIXTURE_DIR/qwen3-0.6b.onnx" \
            "https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX/resolve/main/onnx/model.onnx"
        curl -sL -o "$FIXTURE_DIR/model.onnx_data" \
            "https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX/resolve/main/onnx/model.onnx_data"
    fi

else
    echo "  Skipping large transformer models (--skip-large)"
fi

# ── Optional heavyweight fixtures ─────────────────────────────────────

if [ "$SKIP_WHISPER" = false ] && [ "$SKIP_LARGE" = false ]; then
    # Whisper Medium encoder ONNX (~769M params, ~1.5GB)
    if [ ! -f "$FIXTURE_DIR/whisper-medium-encoder.onnx" ]; then
        echo "  whisper-medium-encoder.onnx (~1.5GB, this may take a while)..."
        curl -sL -o "$FIXTURE_DIR/whisper-medium-encoder.onnx" \
            "https://huggingface.co/onnx-community/whisper-medium/resolve/main/encoder_model.onnx"
    fi
else
    echo "  Skipping whisper-medium-encoder.onnx"
fi

echo "Done. Fixtures in $FIXTURE_DIR/:"
ls -lh "$FIXTURE_DIR/"
