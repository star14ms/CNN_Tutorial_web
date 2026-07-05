"""
Export torchinfo.summary() stats for each model × dataset combination.
Saves torchinfo.json in public/models/{dataset}/{model}/ for use by the LAYERS modal.

Usage (from repo root):
  conda activate deep-learning
  python train/export_torchinfo.py --dataset mnist
  python train/export_torchinfo.py --dataset cifar10
  python train/export_torchinfo.py          # all datasets

Output per model:
  public/models/{dataset}/{model}/torchinfo.json

JSON format:
  {
    "multAddsM":    3.40,
    "paramsSizeMB": 0.29,
    "fwdBwdSizeMB": 1.13,
    "totalSizeMB":  1.42,
    "totalParams":  344330
  }
"""

import argparse
import json
import os
import sys
import torch

# Allow importing train scripts as modules
sys.path.insert(0, os.path.dirname(__file__))

import torchinfo

DATASETS = ['mnist', 'fashion_mnist', 'kuzushiji_mnist', 'cifar10', 'cifar100', 'svhn']

DATASET_SHAPES = {
    'mnist':           (1, 1, 28, 28),
    'fashion_mnist':   (1, 1, 28, 28),
    'kuzushiji_mnist': (1, 1, 28, 28),
    'cifar10':         (1, 3, 32, 32),
    'cifar100':        (1, 3, 32, 32),
    'svhn':            (1, 3, 32, 32),
}

DATASET_PARAMS = {
    'mnist':           {'in_channels': 1, 'img_size': 28, 'num_classes': 10},
    'fashion_mnist':   {'in_channels': 1, 'img_size': 28, 'num_classes': 10},
    'kuzushiji_mnist': {'in_channels': 1, 'img_size': 28, 'num_classes': 10},
    'cifar10':         {'in_channels': 3, 'img_size': 32, 'num_classes': 10},
    'cifar100':        {'in_channels': 3, 'img_size': 32, 'num_classes': 100},
    'svhn':            {'in_channels': 3, 'img_size': 32, 'num_classes': 10},
}

PUBLIC_MODELS = os.path.join(os.path.dirname(__file__), '..', 'public', 'models')


def build_model(model_id, p):
    ic, sz, nc = p['in_channels'], p['img_size'], p['num_classes']
    if model_id == 'linear':
        from train_linear import LinearClassifier
        return LinearClassifier(ic * sz * sz, nc)
    elif model_id == 'v1':
        from train_v1 import SimpleCNN
        return SimpleCNN(ic, sz, nc)
    elif model_id == 'v1bn':
        from train_v1_bn import SimpleCNNBN
        return SimpleCNNBN(ic, sz, nc)
    elif model_id == 'v2small':
        from train_v2_small import DeepCNNSmall
        return DeepCNNSmall(ic, sz, nc)
    elif model_id == 'v2':
        from train_v2 import DeepCNN
        return DeepCNN(ic, sz, nc)
    elif model_id == 'vit':
        from train_vit import ViT
        return ViT(in_channels=ic, img_size=sz, num_classes=nc)
    else:
        raise ValueError(f'Unknown model: {model_id}')


def extract_stats(summary) -> dict:
    param_mb  = summary.total_param_bytes / 1024 / 1024
    fwdbwd_mb = summary.total_output_bytes * 2 / 1024 / 1024
    return {
        'totalParams':  int(summary.total_params),
        'multAddsM':    round(summary.total_mult_adds / 1e6, 4),
        'paramsSizeMB': round(param_mb, 4),
        'fwdBwdSizeMB': round(fwdbwd_mb, 4),
        'totalSizeMB':  round(param_mb + fwdbwd_mb, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS + ['all'], default='all')
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]

    for ds in datasets:
        print(f'\n=== {ds} ===')
        p = DATASET_PARAMS[ds]
        input_shape = DATASET_SHAPES[ds]

        for model_id in ['linear', 'v1', 'v1bn', 'v2small', 'v2', 'vit']:
            out_dir = os.path.join(PUBLIC_MODELS, ds, model_id)
            if not os.path.isdir(out_dir):
                print(f'  [skip] {model_id}: output dir not found')
                continue

            print(f'  {model_id}: ', end='', flush=True)
            try:
                model = build_model(model_id, p)
                model.eval()
                summary = torchinfo.summary(model, input_size=input_shape, verbose=0)
                stats = extract_stats(summary)
                out_path = os.path.join(out_dir, 'torchinfo.json')
                with open(out_path, 'w') as f:
                    json.dump(stats, f, indent=2)
                print(f"Mult-Adds: {stats['multAddsM']:.2f} M, Params: {stats['paramsSizeMB']:.2f} MB → saved")
            except Exception as e:
                print(f'ERROR: {e}')


if __name__ == '__main__':
    main()
