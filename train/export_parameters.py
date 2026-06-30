"""
Export conv weights+biases and linear (Gemm) weights+biases from ONNX models.
Stores float arrays as base64-encoded binary (much smaller than JSON number arrays).

Usage (from repo root):
  conda activate deep-learning
  python train/export_parameters.py --dataset mnist
  python train/export_parameters.py --dataset fashion_mnist
  python train/export_parameters.py --dataset kuzushiji_mnist
  python train/export_parameters.py  # all datasets

Output per model:
  public/models/{dataset}/{model}/parameters.json

JSON format:
  {
    "convs":   [ { "shape": [outC,inC,kH,kW], "data": "<base64>", "bias": "<base64>" }, ... ],
    "linears": [ { "shape": [outF,inF],        "data": "<base64>", "bias": "<base64>" }, ... ]
  }
  data/bias values are base64-encoded little-endian float32 arrays.
"""

import argparse
import base64
import json
import os
import onnx
import onnx.numpy_helper
import numpy as np

DATASETS  = ['mnist', 'fashion_mnist', 'kuzushiji_mnist']
MODEL_IDS = ['linear', 'v1', 'v1bn', 'v2small', 'v2', 'vit']


def arr_to_b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode('ascii')


def extract_parameters(onnx_path: str) -> dict:
    model    = onnx.load(onnx_path)
    init_map = {init.name: init for init in model.graph.initializer}

    convs, linears = [], []
    for node in model.graph.node:
        if node.op_type == 'Conv':
            if len(node.input) < 2 or not node.input[1]:
                continue
            w = init_map.get(node.input[1])
            if w is None:
                continue
            arr   = onnx.numpy_helper.to_array(w).astype(np.float32)
            entry = {'shape': list(arr.shape), 'data': arr_to_b64(arr)}
            if len(node.input) > 2 and node.input[2]:
                b = init_map.get(node.input[2])
                if b is not None:
                    entry['bias'] = arr_to_b64(onnx.numpy_helper.to_array(b).astype(np.float32))
            convs.append(entry)

        elif node.op_type == 'Gemm':
            if len(node.input) < 2 or not node.input[1]:
                continue
            w = init_map.get(node.input[1])
            if w is None:
                continue
            arr   = onnx.numpy_helper.to_array(w).astype(np.float32)
            entry = {'shape': list(arr.shape), 'data': arr_to_b64(arr)}
            if len(node.input) > 2 and node.input[2]:
                b = init_map.get(node.input[2])
                if b is not None:
                    entry['bias'] = arr_to_b64(onnx.numpy_helper.to_array(b).astype(np.float32))
            linears.append(entry)

    return {'convs': convs, 'linears': linears}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS, help='Process one dataset only')
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else DATASETS

    for ds in datasets:
        print(f'\n=== {ds} ===')
        for model_id in MODEL_IDS:
            onnx_path = os.path.join('public', 'models', ds, model_id, 'full.onnx')
            out_path  = os.path.join('public', 'models', ds, model_id, 'parameters.json')
            if not os.path.exists(onnx_path):
                print(f'  [skip] {model_id}: full.onnx not found')
                continue
            print(f'  {model_id}: extracting…', end=' ', flush=True)
            params = extract_parameters(onnx_path)
            nc, nl = len(params['convs']), len(params['linears'])
            with open(out_path, 'w') as f:
                json.dump(params, f, separators=(',', ':'))
            kb = os.path.getsize(out_path) / 1024
            print(f'{nc} conv, {nl} linear → {kb:.0f} KB')


if __name__ == '__main__':
    main()
