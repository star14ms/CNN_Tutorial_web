"""
Train Simple CNN + BatchNorm + Dropout on MNIST/Fashion-MNIST/Kuzushiji-MNIST.

Architecture (same spatial structure as v1 but with BN + Dropout):
  Conv1(1→32, 3×3) + BatchNorm2d(32) + ReLU             → (32, 28, 28)  [layer0]
  MaxPool(2×2)                                           → (32, 14, 14)  [layer1]
  Conv2(32→64, 3×3) + BatchNorm2d(64) + ReLU            → (64, 14, 14)  [layer2]
  MaxPool(2×2)                                           → (64,  7,  7)  [layer3]
  Flatten                                                → (3136,)
  FC1(3136→128) + ReLU + Dropout(0.5)                   → (128,)         [layer4]
  FC2(128→10) + Softmax                                 → (10,)          [full]

Usage:
  conda activate deep-learning
  python train/train_v1_bn.py --dataset mnist
  python train/train_v1_bn.py --dataset fashion_mnist
  python train/train_v1_bn.py --dataset kuzushiji_mnist
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

class MnistCNNv1BN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(1,  32, 3, padding=1)
        self.bn1     = nn.BatchNorm2d(32)
        self.relu1   = nn.ReLU()
        self.pool1   = nn.MaxPool2d(2, 2)

        self.conv2   = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2     = nn.BatchNorm2d(64)
        self.relu2   = nn.ReLU()
        self.pool2   = nn.MaxPool2d(2, 2)

        self.fc1     = nn.Linear(64 * 7 * 7, 128)
        self.relu3   = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))   # (32, 14, 14)
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))   # (64,  7,  7)
        x = x.flatten(1)
        x = self.dropout(self.relu3(self.fc1(x)))
        return self.fc2(x)


# ── Submodels ──────────────────────────────────────────────────────────────────

class V1BNUpToLayer0(nn.Module):
    """→ after Conv1+BN+ReLU  (32, 28, 28)"""
    def __init__(self, m):
        super().__init__()
        self.conv1, self.bn1, self.relu1 = m.conv1, m.bn1, m.relu1
    def forward(self, x):
        return self.relu1(self.bn1(self.conv1(x)))

class V1BNUpToLayer1(nn.Module):
    """→ after MaxPool1  (32, 14, 14)"""
    def __init__(self, m):
        super().__init__()
        self.conv1, self.bn1, self.relu1, self.pool1 = m.conv1, m.bn1, m.relu1, m.pool1
    def forward(self, x):
        return self.pool1(self.relu1(self.bn1(self.conv1(x))))

class V1BNUpToLayer2(nn.Module):
    """→ after Conv2+BN+ReLU  (64, 14, 14)"""
    def __init__(self, m):
        super().__init__()
        self.conv1, self.bn1, self.relu1, self.pool1 = m.conv1, m.bn1, m.relu1, m.pool1
        self.conv2, self.bn2, self.relu2 = m.conv2, m.bn2, m.relu2
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        return self.relu2(self.bn2(self.conv2(x)))

class V1BNUpToLayer3(nn.Module):
    """→ after MaxPool2  (64, 7, 7)"""
    def __init__(self, m):
        super().__init__()
        self.conv1, self.bn1, self.relu1, self.pool1 = m.conv1, m.bn1, m.relu1, m.pool1
        self.conv2, self.bn2, self.relu2, self.pool2 = m.conv2, m.bn2, m.relu2, m.pool2
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        return self.pool2(self.relu2(self.bn2(self.conv2(x))))

class V1BNUpToLayer4(nn.Module):
    """→ after FC1+ReLU, no dropout  (128,)"""
    def __init__(self, m):
        super().__init__()
        self.conv1, self.bn1, self.relu1, self.pool1 = m.conv1, m.bn1, m.relu1, m.pool1
        self.conv2, self.bn2, self.relu2, self.pool2 = m.conv2, m.bn2, m.relu2, m.pool2
        self.fc1, self.relu3 = m.fc1, m.relu3
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.flatten(1)
        return self.relu3(self.fc1(x))

class V1BNFullWithSoftmax(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.conv1, self.bn1, self.relu1, self.pool1 = m.conv1, m.bn1, m.relu1, m.pool1
        self.conv2, self.bn2, self.relu2, self.pool2 = m.conv2, m.bn2, m.relu2, m.pool2
        self.fc1, self.relu3, self.fc2 = m.fc1, m.relu3, m.fc2
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.flatten(1)
        x = self.relu3(self.fc1(x))
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

    model = MnistCNNv1BN().to(device)
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
    submodel.eval()
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

    models_dir = os.path.join(PUBLIC_MODELS, args.dataset, 'v1bn')
    os.makedirs(models_dir, exist_ok=True)

    model = train(args.dataset)
    model.cpu().eval()

    print('\nModel summary (torchinfo):')
    try:
        from torchinfo import summary
        summary(model, input_size=(1, 1, 28, 28), col_names=['output_size', 'num_params'])
    except ImportError:
        print('  torchinfo not installed')

    print(f'\nExporting ONNX models (v1bn / {args.dataset})...')
    exports = [
        (V1BNUpToLayer0(model),      'layer0.onnx', (1, 32, 28, 28)),
        (V1BNUpToLayer1(model),      'layer1.onnx', (1, 32, 14, 14)),
        (V1BNUpToLayer2(model),      'layer2.onnx', (1, 64, 14, 14)),
        (V1BNUpToLayer3(model),      'layer3.onnx', (1, 64,  7,  7)),
        (V1BNUpToLayer4(model),      'layer4.onnx', (1, 128)),
        (V1BNFullWithSoftmax(model), 'full.onnx',   (1, 10)),
    ]
    for submodel, name, expected in exports:
        path = os.path.join(models_dir, name)
        export_onnx(submodel, path)
        validate_onnx(path, expected)

    print('\nAll v1bn models exported successfully.')


if __name__ == '__main__':
    main()
