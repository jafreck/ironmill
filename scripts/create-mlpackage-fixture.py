#!/usr/bin/env python3
"""
One-off script to create a small .mlpackage test fixture.
Requires: pip install coremltools numpy
Requires: Python 3.11 or earlier (coremltools doesn't support 3.14+)

This only needs to be run once to regenerate the fixture with an actual
protobuf model spec inside. The directory structure in tests/fixtures/simple.mlpackage
is already committed with just the Manifest.json for structural tests.
"""

import coremltools as ct
import numpy as np

# Create a trivial model: y = x * 2 + 1
# This is the smallest possible MIL program model.
from coremltools.converters.mil import Builder as mb

@mb.program(input_specs=[mb.TensorSpec(shape=(1, 4))])
def simple_model(x):
    w = mb.const(val=np.array([[2.0, 2.0, 2.0, 2.0]], dtype=np.float32).T, name="weight")
    y = mb.matmul(x=x, y=w, name="matmul")
    b = mb.const(val=np.array([[1.0]], dtype=np.float32), name="bias")
    z = mb.add(x=y, y=b, name="add")
    return z

model = ct.convert(simple_model, convert_to="mlprogram")
model.save("tests/fixtures/simple.mlpackage")
print("Created tests/fixtures/simple.mlpackage")
