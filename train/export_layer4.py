"""
Extract a submodel up to FC1+ReLU from the existing mnist-cnn.onnx.
Saves public/models/layer4.onnx with output shape (1, 128).

Run: python export_layer4.py
Requires: pip install onnx onnxruntime
"""

import os
import sys
import onnx
import onnxruntime as ort
import numpy as np
from onnx import helper

# Add train directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from device_utils import get_onnx_providers_with_info

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'public', 'models')
FULL_PATH   = os.path.join(MODELS_DIR, 'mnist-cnn.onnx')
LAYER4_PATH = os.path.join(MODELS_DIR, 'layer4.onnx')

model = onnx.load(FULL_PATH)

# Print nodes so we can identify FC1+ReLU output tensor name
relu_outputs = []
for node in model.graph.node:
    if node.op_type == 'Relu':
        relu_outputs.append(node.output[0])

print("Relu output tensor names:", relu_outputs)
# Typical PyTorch export: first Relu = after conv1, second = after conv2,
# third = after fc1. We want index 2.
# If the model has only 2 relus (pool+flatten don't add relu), print to verify.

# Find fc1 relu: last relu whose input feeds from a Gemm (FC) node
fc_relu = None
gemm_outputs = set()
for node in model.graph.node:
    if node.op_type == 'Gemm':
        gemm_outputs.update(node.output)
for node in model.graph.node:
    if node.op_type == 'Relu' and node.input[0] in gemm_outputs:
        # First Gemm-relu is FC1+relu
        if fc_relu is None:
            fc_relu = node.output[0]
            print(f"FC1+ReLU output tensor: {fc_relu}")

if fc_relu is None:
    # Fallback: use third relu
    fc_relu = relu_outputs[2] if len(relu_outputs) >= 3 else relu_outputs[-1]
    print(f"Fallback: using relu output: {fc_relu}")

# Add fc_relu as a graph output so extract_model can work
value_info = helper.make_tensor_value_info(fc_relu, onnx.TensorProto.FLOAT, [1, 128])
model.graph.output.insert(0, value_info)

# Extract submodel up to fc_relu
sub = onnx.utils.extract_model(FULL_PATH, LAYER4_PATH,
                                input_names=['input'],
                                output_names=[fc_relu])
print(f"Saved: {LAYER4_PATH}")

# Validate
providers, provider_str = get_onnx_providers_with_info()
print(f"Using ONNX Runtime: {provider_str}")
sess = ort.InferenceSession(LAYER4_PATH, providers=providers)
dummy = np.zeros((1, 1, 28, 28), dtype=np.float32)
out = sess.run(None, {'input': dummy})
print(f"Validation output shape: {out[0].shape}  (expected (1, 128))")
print("Done.")
