"""
Train a Linear classifier on MNIST/Fashion-MNIST/Kuzushiji-MNIST and export ONNX layer checkpoints.

Architecture:
  Input (784,) → Linear(784→512) + ReLU + Dropout(0.5)  [layer0]
               → Linear(512→10) + Softmax               [full]

Total params: 784*512+512 + 512*10+10 = 401,920 + 5,130 = 407,050

Usage:
  conda activate deep-learning
  python train/train_linear.py --dataset mnist
  python train/train_linear.py --dataset fashion_mnist
  python train/train_linear.py --dataset kuzushiji_mnist
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
    'mnist':           (datasets.MNIST,        (0.1307,), (0.3081,)),
    'fashion_mnist':   (datasets.FashionMNIST, (0.2860,), (0.3530,)),
    'kuzushiji_mnist': (datasets.KMNIST,       (0.1918,), (0.3483,)),
}

PUBLIC_MODELS = os.path.join(os.path.dirname(__file__), '..', 'public', 'models')
DATA_DIR      = os.path.join(os.path.dirname(__file__), 'data')


# ── Model ──────────────────────────────────────────────────────────────────────

class MnistLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1     = nn.Linear(784, 512)
        self.relu1   = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(512, 10)

    def forward(self, x):
        x = x.to(self.fc1.weight.device).view(x.size(0), -1)
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
        x = x.to(self.fc1.weight.device).view(x.size(0), -1)
        return self.relu1(self.fc1(x))

class LinFullWithSoftmax(nn.Module):
    """Full model → Softmax(10)"""
    def __init__(self, m):
        super().__init__()
        self.fc1, self.relu1, self.fc2 = m.fc1, m.relu1, m.fc2
    def forward(self, x):
        x = x.to(self.fc1.weight.device).view(x.size(0), -1)
        x = self.relu1(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=1)


# ── Training ───────────────────────────────────────────────────────────────────

def train(dataset_id='mnist'):
    ds_class, norm_mean, norm_std = DATASET_MAP[dataset_id]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}  dataset: {dataset_id}')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    train_ds = ds_class(DATA_DIR, train=True,  download=True, transform=transform)
    test_ds  = ds_class(DATA_DIR, train=False, download=True, transform=transform)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)

    model = MnistLinear().to(device)
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

def export_onnx(submodel, path, input_shape=(1, 1, 28, 28)):
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


def validate_onnx(path, expected_shape):
    try:
        import onnxruntime as ort
    except ImportError:
        print(f'  [skip validation — onnxruntime unavailable]')
        return
    try:
        sess = ort.InferenceSession(path)
        dummy = np.zeros((1, 1, 28, 28), dtype=np.float32)
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

    models_dir = os.path.join(PUBLIC_MODELS, args.dataset, 'linear')
    os.makedirs(models_dir, exist_ok=True)

    model = train(args.dataset)
    model.cpu().eval()

    print('\nModel summary (torchinfo):')
    try:
        from torchinfo import summary
        summary(model, input_size=(1, 1, 28, 28), col_names=['output_size', 'num_params'])
    except ImportError:
        print('  torchinfo not installed')

    print(f'\nExporting ONNX models (linear / {args.dataset})...')
    exports = [
        (LinUpToLayer0(model),      'layer0.onnx', (1, 512)),
        (LinFullWithSoftmax(model), 'full.onnx',   (1, 10)),
    ]
    for submodel, name, expected in exports:
        path = os.path.join(models_dir, name)
        export_onnx(submodel, path)
        validate_onnx(path, expected)

    print('\nAll linear models exported successfully.')


if __name__ == '__main__':
    main()
