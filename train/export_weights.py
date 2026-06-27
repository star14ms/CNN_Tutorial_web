"""
Export conv layer kernel weights from ONNX models to JSON for browser visualization.

Usage (from repo root, with onnx installed):
  conda activate deep-learning
  python train/export_weights.py

Outputs:
  public/models/v1_weights.json
  public/models/v2_weights.json

JSON format:
  { "convs": [ { "shape": [outC, inC, kH, kW], "data": [...flat float array...] }, ... ] }
  Convs are ordered by their appearance in the ONNX graph (matches layer index order).
"""

import json
import os
import onnx
import onnx.numpy_helper
import numpy as np

MODELS = [
    ('public/models/mnist-cnn.onnx',    'public/models/v1_weights.json'),
    ('public/models/v2_mnist-cnn.onnx', 'public/models/v2_weights.json'),
]


def extract_conv_weights(onnx_path):
    model = onnx.load(onnx_path)
    init_map = {init.name: init for init in model.graph.initializer}

    convs = []
    for node in model.graph.node:
        if node.op_type != 'Conv':
            continue
        weight_init = init_map.get(node.input[1])
        if weight_init is None:
            continue
        arr = onnx.numpy_helper.to_array(weight_init).astype(np.float32)
        convs.append({
            'shape': list(arr.shape),
            'data':  arr.flatten().tolist(),
        })

    return {'convs': convs}


def main():
    root = os.path.join(os.path.dirname(__file__), '..')
    for rel_in, rel_out in MODELS:
        in_path  = os.path.join(root, rel_in)
        out_path = os.path.join(root, rel_out)
        if not os.path.exists(in_path):
            print(f'SKIP (not found): {in_path}')
            continue
        print(f'Extracting {in_path} …')
        weights = extract_conv_weights(in_path)
        print(f'  Found {len(weights["convs"])} conv layers: ' +
              ', '.join(str(w['shape']) for w in weights['convs']))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(weights, f, separators=(',', ':'))
        kb = os.path.getsize(out_path) / 1024
        print(f'  → {out_path} ({kb:.1f} KB)')


if __name__ == '__main__':
    main()
