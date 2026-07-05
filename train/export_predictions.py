"""
Export test-set predictions for all models to binary files.
Each file is N uint8 values (one predicted class per test image).

Usage (from repo root):
  conda activate deep-learning
  python train/export_predictions.py --dataset mnist
  python train/export_predictions.py  # all datasets

Output per model:
  public/models/{dataset}/{model}/test_preds.bin
"""

import argparse
import os
import sys
import struct
import numpy as np
import onnxruntime as ort

# Add train directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from device_utils import get_onnx_providers

DATASETS  = ['mnist', 'fashion_mnist', 'kuzushiji_mnist']
MODEL_IDS = ['linear', 'v1', 'v1bn', 'v2small', 'v2', 'vit']

NORM = {
    'mnist':           (0.1307, 0.3081),
    'fashion_mnist':   (0.2860, 0.3530),
    'kuzushiji_mnist': (0.1918, 0.3483),
}


def load_test_set(dataset: str):
    base = os.path.join('public', 'data', dataset, 'test')
    imgs = np.frombuffer(open(f'{base}_images.bin', 'rb').read(), dtype=np.uint8)
    lbls = np.frombuffer(open(f'{base}_labels.bin', 'rb').read(), dtype=np.uint8)
    n = len(lbls)
    imgs = imgs.reshape(n, 1, 28, 28).astype(np.float32) / 255.0
    mean, std = NORM[dataset]
    imgs = (imgs - mean) / std
    return imgs, lbls


def predict_all(onnx_path: str, images: np.ndarray, batch: int = 256):
    opts = ort.SessionOptions()
    opts.log_severity_level = 4  # suppress ORT stderr (only FATAL)
    providers = get_onnx_providers()
    sess = ort.InferenceSession(onnx_path, providers=providers, sess_options=opts)
    name = sess.get_inputs()[0].name
    # Check if model supports the batch size
    try:
        sess.run(None, {name: images[:1]})
    except Exception:
        return None
    try:
        sess.run(None, {name: images[:batch]})
    except Exception:
        batch = 1
    preds = []
    for start in range(0, len(images), batch):
        out = sess.run(None, {name: images[start:start + batch]})[0]
        preds.append(np.argmax(out, axis=1).astype(np.uint8))
    return np.concatenate(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS)
    args = parser.parse_args()
    datasets = [args.dataset] if args.dataset else DATASETS

    for ds in datasets:
        print(f'\n=== {ds} ===')
        try:
            images, labels = load_test_set(ds)
        except FileNotFoundError:
            print(f'  [skip] test set not found — run export_dataset.py --dataset {ds}')
            continue

        for model_id in MODEL_IDS:
            onnx_path = os.path.join('public', 'models', ds, model_id, 'full.onnx')
            out_path  = os.path.join('public', 'models', ds, model_id, 'test_preds.bin')
            if not os.path.exists(onnx_path):
                print(f'  [skip] {model_id}: full.onnx not found')
                continue
            print(f'  {model_id}: predicting…', end=' ', flush=True)
            preds = predict_all(onnx_path, images)
            if preds is None:
                print('skipped (inference failed)')
                continue
            acc = 100.0 * (preds == labels).mean()
            with open(out_path, 'wb') as f:
                f.write(preds.tobytes())
            print(f'{acc:.2f}% acc → {len(preds)} bytes')


if __name__ == '__main__':
    main()
