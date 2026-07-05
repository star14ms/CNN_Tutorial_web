"""
Evaluate all ONNX models on the test set and write accuracies.json.

Usage:
    conda activate deep-learning
    python train/eval_accuracy.py --dataset mnist
    python train/eval_accuracy.py --dataset fashion_mnist
    python train/eval_accuracy.py --dataset kuzushiji_mnist
"""

import argparse
import json
import os
import sys
import numpy as np
import onnxruntime as ort
from torchvision import datasets, transforms

# Add train directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from device_utils import get_onnx_providers_with_info

DATASET_CONFIGS = {
    'mnist': {
        'cls': datasets.MNIST,
        'mean': 0.1307,
        'std':  0.3081,
    },
    'fashion_mnist': {
        'cls': datasets.FashionMNIST,
        'mean': 0.2860,
        'std':  0.3530,
    },
    'kuzushiji_mnist': {
        'cls': datasets.KMNIST,
        'mean': 0.1918,
        'std':  0.3483,
    },
}

MODEL_IDS = ['linear', 'v1', 'v1bn', 'v2small', 'v2', 'vit']


def evaluate(onnx_path, images, labels, providers):
    sess = ort.InferenceSession(onnx_path, providers=providers)
    input_name = sess.get_inputs()[0].name

    # Probe with batch=1 to detect fixed-batch-size exports before running all data
    try:
        sess.run(None, {input_name: images[:1]})[0]
        probe_ok = True
    except Exception:
        return None  # model broken even for batch=1

    # Check whether large batches work
    try:
        sess.run(None, {input_name: images[:8]})[0]
        batch = 256
    except Exception:
        batch = 1  # fixed-batch export — use one sample at a time

    correct = 0
    for start in range(0, len(images), batch):
        x = images[start:start+batch]
        out = sess.run(None, {input_name: x})[0]
        preds = np.argmax(out, axis=1)
        correct += (preds == labels[start:start+batch]).sum()
    return 100.0 * correct / len(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--data-root', default='train/data')
    args = parser.parse_args()

    # Get optimal ONNX providers
    providers, provider_str = get_onnx_providers_with_info()
    print(f'Using ONNX Runtime: {provider_str}')

    cfg = DATASET_CONFIGS[args.dataset]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((cfg['mean'],), (cfg['std'],)),
    ])
    ds = cfg['cls'](root=args.data_root, train=False, download=True, transform=transform)

    images = np.stack([np.array(img) for img, _ in ds]).astype(np.float32)
    # ToTensor() already produces (1,28,28) so stack gives (N,1,28,28)
    labels = np.array([lbl for _, lbl in ds])

    models_dir = f'public/models/{args.dataset}'
    accuracies = {}
    for model_id in MODEL_IDS:
        onnx_path = os.path.join(models_dir, model_id, 'full.onnx')
        if not os.path.exists(onnx_path):
            print(f'  [skip] {model_id}: {onnx_path} not found')
            continue
        acc = evaluate(onnx_path, images, labels, providers)
        if acc is None:
            print(f'  [skip] {model_id}: ONNX model incompatible with onnxruntime (try re-exporting)')
            continue
        accuracies[model_id] = round(acc, 2)
        print(f'  {model_id}: {acc:.2f}%')

    out_path = os.path.join(models_dir, 'accuracies.json')
    with open(out_path, 'w') as f:
        json.dump(accuracies, f, indent=2)
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
