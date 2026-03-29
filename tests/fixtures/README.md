# Test Fixtures

Sample model files for integration tests. These are NOT checked into git
(see `.gitignore`). Run `scripts/download-fixtures.sh` to fetch them.

## Files

### Core fixtures (always downloaded)

| File | Format | Architecture | Size | Source | PolarQuant? |
|------|--------|-------------|------|--------|-------------|
| `mnist.onnx` | ONNX | CNN (LeNet) | ~26KB | ONNX Model Zoo | No |
| `squeezenet1.1.onnx` | ONNX | CNN | ~4.7MB | ONNX Model Zoo | No |
| `mobilenetv2.onnx` | ONNX | CNN | ~14MB | ONNX Model Zoo | No |
| `MobileNet.mlmodel` | CoreML | CNN | ~16MB | hollance/MobileNet-CoreML | N/A |
| `simple.mlpackage/` | CoreML | Minimal | ~1KB | Generated | N/A |

### Transformer fixtures (downloaded by default)

| File | Format | Architecture | Size | Inner dim | PolarQuant? |
|------|--------|-------------|------|-----------|-------------|
| `whisper-tiny-encoder.onnx` | ONNX | Encoder-only (audio) | ~33MB | d=384 | Yes |
| `whisper-tiny-decoder.onnx` | ONNX | Decoder + cross-attn | ~118MB | d=384 | Yes |
| `distilbert.onnx` | ONNX | Encoder NLU | ~256MB | d=768 | Yes |
| `vit-base.onnx` | ONNX | Vision Transformer | ~347MB | d=768 | Yes |

### Optional heavyweight fixtures

| File | Format | Architecture | Size | Source |
|------|--------|-------------|------|--------|
| `qwen3-0.6b.onnx` + `model.onnx_data` | ONNX | Decoder-only LLM | ~2.2GB | onnx-community/Qwen3-0.6B-ONNX |
| `whisper-medium-encoder.onnx` | ONNX | Encoder (audio) | ~1.5GB | onnx-community/whisper-medium |

## Downloading

```bash
# Download core + transformer fixtures (recommended)
./scripts/download-fixtures.sh

# Download only core fixtures (fast, <50MB total)
./scripts/download-fixtures.sh --skip-large

# Download everything including whisper-medium
./scripts/download-fixtures.sh --all
```

## Architecture coverage

The fixture matrix covers diverse model architectures to validate
ironmill's optimization passes across different op patterns:

- **CNN** — conv + pooling + dense (MobileNetV2, SqueezeNet)
- **Encoder-only transformer** — self-attention + FFN (DistilBERT, Whisper encoder)
- **Decoder transformer** — causal self-attention + cross-attention (Whisper decoder)
- **Decoder-only LLM** — causal self-attention + RoPE + GQA (Qwen3-0.6B)
- **Vision transformer** — patch embedding + self-attention (ViT)

PolarQuant weight quantization targets transformer architectures where
the inner dimension (hidden_dim, head_dim) is ≥ 64.

## Creating new fixtures

For `.mlpackage` test fixtures, use the one-off Python script:

```bash
python3 scripts/create-mlpackage-fixture.py
```

This requires `coremltools` (`pip install coremltools`) and is only needed once.
The generated fixture should then be committed or added to the download script.
