# Test Fixtures

Sample model files for integration tests. These are NOT checked into git
(see `.gitignore`). Run `scripts/download-fixtures.sh` to fetch them.

## Files

| File | Format | Size | Source | Used by |
|------|--------|------|--------|---------|
| `mnist.onnx` | ONNX | ~26KB | ONNX Model Zoo (mnist-12) | ONNX reader tests, conversion tests |
| `squeezenet1.1.onnx` | ONNX | ~4.7MB | ONNX Model Zoo (squeezenet1.1-7) | Conversion + validation tests |
| `MobileNet.mlmodel` | CoreML (.mlmodel) | ~16MB | hollance/MobileNet-CoreML | .mlmodel reader/writer round-trip tests |

## Downloading

```bash
./scripts/download-fixtures.sh
```

## Creating new fixtures

For `.mlpackage` test fixtures, use the one-off Python script:

```bash
python3 scripts/create-mlpackage-fixture.py
```

This requires `coremltools` (`pip install coremltools`) and is only needed once.
The generated fixture should then be committed or added to the download script.
