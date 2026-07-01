"""
Train a Linear classifier and export ONNX layer checkpoints.

Architecture:
  Input (in_channels*img_size*img_size,) flattened
  → Linear(in_features→512) + ReLU + Dropout(0.5)  [layer0]
  → Linear(512→num_classes) + Softmax               [full]

Dataset support:
  1-channel 28×28: mnist, fashion_mnist, kuzushiji_mnist  → in_features=784,  num_classes=10
  3-channel 32×32: cifar10, svhn                          → in_features=3072, num_classes=10
  3-channel 32×32: cifar100                               → in_features=3072, num_classes=100

Usage:
  conda activate deep-learning
  python train/train_linear.py --dataset mnist
  python train/train_linear.py --dataset fashion_mnist
  python train/train_linear.py --dataset kuzushiji_mnist
  python train/train_linear.py --dataset cifar10
  python train/train_linear.py --dataset cifar100
  python train/train_linear.py --dataset svhn
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

DATASET_MAP = {
    'mnist':           {'class': datasets.MNIST,        'mean': (0.1307,),                'std': (0.3081,),                'in_channels': 1, 'img_size': 28, 'num_classes': 10},
    'fashion_mnist':   {'class': datasets.FashionMNIST, 'mean': (0.2860,),                'std': (0.3530,),                'in_channels': 1, 'img_size': 28, 'num_classes': 10},
    'kuzushiji_mnist': {'class': datasets.KMNIST,       'mean': (0.1918,),                'std': (0.3483,),                'in_channels': 1, 'img_size': 28, 'num_classes': 10},
    'cifar10':         {'class': datasets.CIFAR10,      'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2470, 0.2435, 0.2616), 'in_channels': 3, 'img_size': 32, 'num_classes': 10},
    'cifar100':        {'class': datasets.CIFAR100,     'mean': (0.5071, 0.4867, 0.4408), 'std': (0.2675, 0.2565, 0.2761), 'in_channels': 3, 'img_size': 32, 'num_classes': 100},
    'svhn':            {'class': datasets.SVHN,         'mean': (0.4377, 0.4438, 0.4728), 'std': (0.1980, 0.2010, 0.1970), 'in_channels': 3, 'img_size': 32, 'num_classes': 10},
}

PUBLIC_MODELS = os.path.join(os.path.dirname(__file__), '..', 'public', 'models')
DATA_DIR      = os.path.join(os.path.dirname(__file__), 'data')


# ── Model ──────────────────────────────────────────────────────────────────────

class LinearClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.in_features = in_features
        self.fc1     = nn.Linear(in_features, 512)
        self.relu1   = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ── Submodels ──────────────────────────────────────────────────────────────────

class LinUpToLayer0(nn.Module):
    """→ after FC1+ReLU, no dropout  (512,)"""
    def __init__(self, m):
        super().__init__()
        self.fc1, self.relu1 = m.fc1, m.relu1
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.relu1(self.fc1(x))

class LinFullWithSoftmax(nn.Module):
    """Full model → Softmax(num_classes)"""
    def __init__(self, m):
        super().__init__()
        self.fc1, self.relu1, self.fc2 = m.fc1, m.relu1, m.fc2
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu1(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=1)


# ── Dataset loading ─────────────────────────────────────────────────────────────

def get_datasets(dataset_id, transform_train, transform_test):
    cfg = DATASET_MAP[dataset_id]
    if dataset_id == 'svhn':
        train_ds = datasets.SVHN(DATA_DIR, split='train', download=True, transform=transform_train)
        test_ds  = datasets.SVHN(DATA_DIR, split='test',  download=True, transform=transform_test)
    else:
        train_ds = cfg['class'](DATA_DIR, train=True,  download=True, transform=transform_train)
        test_ds  = cfg['class'](DATA_DIR, train=False, download=True, transform=transform_test)
    return train_ds, test_ds


# ── Training ───────────────────────────────────────────────────────────────────

def train(dataset_id='mnist'):
    cfg = DATASET_MAP[dataset_id]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}  dataset: {dataset_id}')

    base_tf = [transforms.ToTensor(), transforms.Normalize(cfg['mean'], cfg['std'])]
    if cfg['in_channels'] == 3:
        train_tf = transforms.Compose([transforms.RandomHorizontalFlip()] + base_tf)
    else:
        train_tf = transforms.Compose(base_tf)
    test_tf = transforms.Compose(base_tf)

    train_ds, test_ds = get_datasets(dataset_id, train_tf, test_tf)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)

    in_features = cfg['in_channels'] * cfg['img_size'] * cfg['img_size']
    model = LinearClassifier(in_features, cfg['num_classes']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, 16):
        model.train()
        total_loss = 0
        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in test_dl:
                images, labels = images.to(device), labels.to(device)
                correct += (model(images).argmax(1) == labels).sum().item()
        acc = correct / len(test_ds) * 100
        print(f'Epoch {epoch:2d} | loss={total_loss/len(train_dl):.4f} | test acc={acc:.2f}%')
        if acc > best_acc:
            best_acc = acc
        scheduler.step()

    print(f'\nBest test accuracy: {best_acc:.2f}%')
    return model


# ── Export ─────────────────────────────────────────────────────────────────────

def export_onnx(submodel, path, input_shape):
    dummy = torch.zeros(*input_shape)
    submodel.cpu().eval()
    torch.onnx.export(
        submodel, dummy, path,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
    )
    print(f'  Exported: {os.path.basename(path)}  ({os.path.getsize(path)//1024} KB)')


def validate_onnx(path, input_shape, expected_shape):
    try:
        import onnxruntime as ort
    except ImportError:
        print(f'  [skip validation — onnxruntime unavailable]')
        return
    try:
        sess = ort.InferenceSession(path)
        dummy = np.zeros(input_shape, dtype=np.float32)
        out = sess.run(None, {'input': dummy})
        shape = out[0].shape
        ok = '✓' if shape == expected_shape else '✗'
        print(f'  {ok} {os.path.basename(path)}: output shape = {shape}  (expected {expected_shape})')
    except Exception as e:
        print(f'  [validation error: {e}]')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist', choices=list(DATASET_MAP.keys()))
    args = parser.parse_args()

    cfg = DATASET_MAP[args.dataset]
    C, S, N = cfg['in_channels'], cfg['img_size'], cfg['num_classes']
    input_shape = (1, C, S, S)

    models_dir = os.path.join(PUBLIC_MODELS, args.dataset, 'linear')
    os.makedirs(models_dir, exist_ok=True)

    model = train(args.dataset)
    model.cpu().eval()

    print('\nModel summary (torchinfo):')
    try:
        from torchinfo import summary
        summary(model, input_size=input_shape, col_names=['output_size', 'num_params'])
    except ImportError:
        print('  torchinfo not installed')

    print(f'\nExporting ONNX models (linear / {args.dataset})...')
    exports = [
        (LinUpToLayer0(model),      'layer0.onnx', (1, 512)),
        (LinFullWithSoftmax(model), 'full.onnx',   (1, N)),
    ]
    for submodel, name, expected in exports:
        path = os.path.join(models_dir, name)
        export_onnx(submodel, path, input_shape)
        validate_onnx(path, input_shape, expected)

    print('\nAll linear models exported successfully.')


if __name__ == '__main__':
    main()
