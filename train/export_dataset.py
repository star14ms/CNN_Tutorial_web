"""
Export dataset images + labels as flat binary files for browser loading.

Output files in public/data/{dataset}/:
  test_images.bin   — N × 784 uint8
  test_labels.bin   — N uint8
  train_images.bin  — N × 784 uint8
  train_labels.bin  — N uint8

Each image is stored as 784 raw uint8 pixel values (0=black, 255=white).
The browser divides by 255 to get [0,1] floats; ModelInference applies
dataset-specific normalization before ONNX inference.

Usage:
  conda activate deep-learning
  python train/export_dataset.py --dataset mnist
  python train/export_dataset.py --dataset fashion_mnist
  python train/export_dataset.py --dataset kuzushiji_mnist
"""

import argparse
import os
import numpy as np
from torchvision import datasets

DATASET_MAP = {
    'mnist':            datasets.MNIST,
    'fashion_mnist':    datasets.FashionMNIST,
    'kuzushiji_mnist':  datasets.KMNIST,
}

DATA_DIR    = os.path.join(os.path.dirname(__file__), 'data')
PUBLIC_DATA = os.path.join(os.path.dirname(__file__), '..', 'public', 'data')


def export_split(ds_class, split_name, is_train, output_dir):
    ds = ds_class(DATA_DIR, train=is_train, download=True)
    n = len(ds)
    images = np.zeros((n, 784), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.uint8)

    for i, (img, label) in enumerate(ds):
        images[i] = np.array(img, dtype=np.uint8).flatten()
        labels[i] = label

    img_path = os.path.join(output_dir, f'{split_name}_images.bin')
    lbl_path = os.path.join(output_dir, f'{split_name}_labels.bin')
    images.tofile(img_path)
    labels.tofile(lbl_path)

    print(f'  {split_name}: {n} samples')
    print(f'    images → {img_path}  ({os.path.getsize(img_path) // 1024} KB)')
    print(f'    labels → {lbl_path}  ({os.path.getsize(lbl_path) // 1024} KB)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist', choices=list(DATASET_MAP.keys()),
                        help='Dataset to export')
    args = parser.parse_args()

    ds_class = DATASET_MAP[args.dataset]
    output_dir = os.path.join(PUBLIC_DATA, args.dataset)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f'Exporting {args.dataset} data for browser...\n')
    export_split(ds_class, 'test',  is_train=False, output_dir=output_dir)
    print()
    export_split(ds_class, 'train', is_train=True,  output_dir=output_dir)
    print('\nDone.')


if __name__ == '__main__':
    main()
