#!/bin/bash
cd /Users/jacobfreck/Source/coreml-kit-task7
# Test if CLI prevents int8 + palettize
echo "Testing: --quantize int8 --palettize 4"
cargo run --bin coreml-kit -- compile dummy.onnx --quantize int8 --palettize 4 2>&1 | grep -i "exclusive\|error" | head -3
