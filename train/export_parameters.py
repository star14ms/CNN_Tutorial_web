"""
Export conv weights+biases and linear (Gemm) weights+biases from ONNX models.

Usage (from repo root, with onnx installed):
  conda activate deep-learning
  python train/export_parameters.py

Outputs:
  public/models/v1_parameters.json
  public/models/v2_parameters.json

JSON format:
  {
    "convs":   [ { "shape": [outC, inC, kH, kW], "data": [...], "bias": [...outC...] }, ... ],
    "linears": [ { "shape": [outF, inF],          "data": [...], "bias": [...outF...] }, ... ]
  }
  Convs and linears are ordered by their appearance in the ONNX graph.
"""

import json
import os
import onnx
import onnx.numpy_helper
import numpy as np

MODELS = [
    ('public/models/mnist-cnn.onnx',    'public/models/v1_parameters.json'),
    ('public/models/v2_mnist-cnn.onnx', 'public/models/v2_parameters.json'),
]


def extract_parameters(onnx_path):
    model = onnx.load(onnx_path)
    init_map = {init.name: init for init in model.graph.initializer}

    convs, linears = [], []
    for node in model.graph.node:
        if node.op_type == 'Conv':
            if len(node.input) < 2 or not node.input[1]:
                continue
            w = init_map.get(node.input[1])
            if w is None:
                continue
            arr = onnx.numpy_helper.to_array(w).astype(np.float32)
            entry = {'shape': list(arr.shape), 'data': arr.flatten().tolist()}
            if len(node.input) > 2 and node.input[2]:
                b = init_map.get(node.input[2])
                if b is not None:
                    entry['bias'] = onnx.numpy_helper.to_array(b).astype(np.float32).tolist()
            convs.append(entry)

        elif node.op_type == 'Gemm':
            if len(node.input) < 2 or not node.input[1]:
                continue
            w = init_map.get(node.input[1])
            if w is None:
                continue
            # PyTorch nn.Linear exports with transB=1, so shape is [outF, inF]
            arr = onnx.numpy_helper.to_array(w).astype(np.float32)
            entry = {'shape': list(arr.shape), 'data': arr.flatten().tolist()}
            if len(node.input) > 2 and node.input[2]:
                b = init_map.get(node.input[2])
                if b is not None:
                    entry['bias'] = onnx.numpy_helper.to_array(b).astype(np.float32).tolist()
            linears.append(entry)

    return {'convs': convs, 'linears': linears}


def main():
    root = os.path.join(os.path.dirname(__file__), '..')
    for rel_in, rel_out in MODELS:
        in_path  = os.path.join(root, rel_in)
        out_path = os.path.join(root, rel_out)
        if not os.path.exists(in_path):
            print(f'SKIP (not found): {in_path}')
            continue
        print(f'Extracting {in_path} …')
        params = extract_parameters(in_path)
        print(f'  Found {len(params["convs"])} conv layers: ' +
              ', '.join(str(c['shape']) for c in params['convs']))
        print(f'  Found {len(params["linears"])} linear layers: ' +
              ', '.join(str(l['shape']) for l in params['linears']))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(params, f, separators=(',', ':'))
        kb = os.path.getsize(out_path) / 1024
        print(f'  → {out_path} ({kb:.1f} KB)')


if __name__ == '__main__':
    main()
