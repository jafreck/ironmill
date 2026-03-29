# Test Fixtures

Sample model files for integration tests. These are NOT checked into git
(see `.gitignore`). Run `scripts/download-fixtures.sh` to fetch them.

## Files

| File | Format | Size | Source | Used by |
|------|--------|------|--------|---------|
| `mnist.onnx` | ONNX | ~26KB | ONNX Model Zoo (mnist-12) | ONNX reader tests, conversion tests |
| `squeezenet1.1.onnx` | ONNX | ~4.7MB | ONNX Model Zoo (squeezenet1.1-7) | Conversion + validation tests |
| `MobileNet.mlmodel` | CoreML (.mlmodel) | ~16MB | hollance/MobileNet-CoreML | .mlmodel reader/writer round-trip tests |
| `Qwen3-0.6B/` | SafeTensors (dir) | ~1.4GB | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | `weight_formats` benchmark (SafeTensors load/template/pipeline) |
| `Qwen3-0.6B-Q8_0.gguf` | GGUF | ~639MB | [Qwen/Qwen3-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) | `weight_formats` benchmark (GGUF load/template/pipeline) |
| `whisper-medium-encoder.onnx` | ONNX | ~1.5GB | [onnx-community/whisper-medium](https://huggingface.co/onnx-community/whisper-medium) | `pipeline` benchmark (whisper encoder) |

## Downloading

```bash
./scripts/download-fixtures.sh            # all fixtures
./scripts/download-fixtures.sh --skip-llm     # skip Qwen3 SafeTensors + GGUF
./scripts/download-fixtures.sh --skip-whisper  # skip whisper encoder
./scripts/download-fixtures.sh --skip-all      # skip all large downloads
```

## Creating new fixtures

For `.mlpackage` test fixtures, use the one-off Python script:

```bash
python3 scripts/create-mlpackage-fixture.py
```

This requires `coremltools` (`pip install coremltools`) and is only needed once.
The generated fixture should then be committed or added to the download script.
