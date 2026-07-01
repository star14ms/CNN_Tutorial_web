"""
Export dataset images + labels as flat binary files for browser loading.

Output files in public/data/{dataset}/:
  test_images.bin   — N × (img_size × img_size × in_channels) uint8,  HWC order
  test_labels.bin   — N uint8
  train_images.bin  — N × (img_size × img_size × in_channels) uint8,  HWC order
  train_labels.bin  — N uint8

Grayscale datasets (MNIST family): 28×28×1 = 784 bytes/image, stored as (H, W).
RGB datasets (CIFAR/SVHN):         32×32×3 = 3072 bytes/image, stored as (H, W, C).

The browser divides by 255 to get [0, 1] floats; ModelInference applies
dataset-specific normalization (per-channel mean/std) before ONNX inference.

Usage:
  conda activate deep-learning
  python train/export_dataset.py --dataset mnist
  python train/export_dataset.py --dataset fashion_mnist
  python train/export_dataset.py --dataset kuzushiji_mnist
  python train/export_dataset.py --dataset cifar10
  python train/export_dataset.py --dataset cifar100
  python train/export_dataset.py --dataset svhn
"""

import argparse
import os
import numpy as np
from torchvision import datasets

DATASET_MAP = {
    'mnist':           {'class': datasets.MNIST,        'in_channels': 1, 'img_size': 28},
    'fashion_mnist':   {'class': datasets.FashionMNIST, 'in_channels': 1, 'img_size': 28},
    'kuzushiji_mnist': {'class': datasets.KMNIST,       'in_channels': 1, 'img_size': 28},
    'cifar10':         {'class': datasets.CIFAR10,      'in_channels': 3, 'img_size': 32},
    'cifar100':        {'class': datasets.CIFAR100,     'in_channels': 3, 'img_size': 32},
    'svhn':            {'class': datasets.SVHN,         'in_channels': 3, 'img_size': 32},
}

DATA_DIR    = os.path.join(os.path.dirname(__file__), 'data')
PUBLIC_DATA = os.path.join(os.path.dirname(__file__), '..', 'public', 'data')


def load_split(dataset_id, cfg, split_name):
    """Load a train or test split, returning (images_list, labels_list)."""
    ds_class = cfg['class']
    if dataset_id == 'svhn':
        ds = ds_class(DATA_DIR, split=split_name, download=True)
        # SVHN labels are stored in .labels attribute
        labels = [int(ds.labels[i]) for i in range(len(ds))]
    else:
        is_train = (split_name == 'train')
        ds = ds_class(DATA_DIR, train=is_train, download=True)
        labels = [int(ds.targets[i]) for i in range(len(ds))]
    return ds, labels


def export_split(dataset_id, cfg, split_name, output_dir):
    C, S = cfg['in_channels'], cfg['img_size']
    pixels_per_image = S * S * C

    ds, labels_list = load_split(dataset_id, cfg, split_name)
    n = len(ds)

    images = np.zeros((n, pixels_per_image), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.uint8)

    for i, (img, _) in enumerate(ds):
        arr = np.array(img, dtype=np.uint8)   # (H, W) for grayscale, (H, W, C) for RGB
        images[i] = arr.flatten()
        labels[i] = labels_list[i]

    img_path = os.path.join(output_dir, f'{split_name}_images.bin')
    lbl_path = os.path.join(output_dir, f'{split_name}_labels.bin')
    images.tofile(img_path)
    labels.tofile(lbl_path)

    print(f'  {split_name}: {n} samples  ({pixels_per_image} bytes/image, {C}ch {S}×{S})')
    print(f'    images → {img_path}  ({os.path.getsize(img_path) // 1024} KB)')
    print(f'    labels → {lbl_path}  ({os.path.getsize(lbl_path) // 1024} KB)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist', choices=list(DATASET_MAP.keys()),
                        help='Dataset to export')
    args = parser.parse_args()

    cfg = DATASET_MAP[args.dataset]
    output_dir = os.path.join(PUBLIC_DATA, args.dataset)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f'Exporting {args.dataset} data for browser...\n')
    export_split(args.dataset, cfg, 'test',  output_dir)
    print()
    export_split(args.dataset, cfg, 'train', output_dir)
    print('\nDone.')


if __name__ == '__main__':
    main()
