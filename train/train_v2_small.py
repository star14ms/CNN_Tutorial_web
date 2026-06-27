"""
Train Deep CNN (small) + BatchNorm + Dropout on MNIST.

Same 4-conv architecture as v2, but with reduced channels (16/32 instead of 32/64)
and FC(128) instead of FC(512) to hit ~421K total parameters.

Architecture:
  Conv1(1→16,  3×3) + BatchNorm2d(16)  + ReLU           → (16,  28, 28)  [layer0]
  Conv2(16→16, 3×3) + BatchNorm2d(16)  + ReLU           → (16,  28, 28)  [layer1_conv]
  MaxPool(2×2)                                           → (16,  14, 14)  [layer1]
  Conv3(16→32, 3×3) + BatchNorm2d(32)  + ReLU           → (32,  14, 14)  [layer2]
  Conv4(32→32, 3×3) + BatchNorm2d(32)  + ReLU           → (32,  14, 14)  [layer3_conv]
  MaxPool(2×2)                                           → (32,   7,  7)  [layer3]
  Flatten                                                → (1568,)
  FC1(1568→256) + ReLU + Dropout(0.5)                   → (256,)         [layer4]
  FC2(256→10) + Softmax                                 → (10,)          [full]

Param count:
  Conv1: 1*16*3*3 + 16 = 160
  BN1: 16*2 = 32
  Conv2: 16*16*3*3 + 16 = 2320
  BN2: 32
  Conv3: 16*32*3*3 + 32 = 4640
  BN3: 64
  Conv4: 32*32*3*3 + 32 = 9248
  BN4: 64
  FC1: 1568*256 + 256 = 401664
  FC2: 256*10 + 10 = 2570
  Total ≈ 420,794

Usage:
  conda activate deep-learning
  python train/train_v2_small.py --dataset mnist
  python train/train_v2_small.py --dataset fashion_mnist
  python train/train_v2_small.py --dataset kuzushiji_mnist
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

class MnistCNNv2Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(1,  16, 3, padding=1)
        self.bn1     = nn.BatchNorm2d(16)
        self.relu1   = nn.ReLU()

        self.conv2   = nn.Conv2d(16, 16, 3, padding=1)
        self.bn2     = nn.BatchNorm2d(16)
        self.relu2   = nn.ReLU()
        self.pool1   = nn.MaxPool2d(2, 2)

        self.conv3   = nn.Conv2d(16, 32, 3, padding=1)
        self.bn3     = nn.BatchNorm2d(32)
        self.relu3   = nn.ReLU()

        self.conv4   = nn.Conv2d(32, 32, 3, padding=1)
        self.bn4     = nn.BatchNorm2d(32)
        self.relu4   = nn.ReLU()
        self.pool2   = nn.MaxPool2d(2, 2)

        self.fc1     = nn.Linear(32 * 7 * 7, 256)
        self.relu5   = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))              # (16, 28, 28)
        x = self.pool1(self.relu2(self.bn2(self.conv2(x))))  # (16, 14, 14)
        x = self.relu3(self.bn3(self.conv3(x)))              # (32, 14, 14)
        x = self.pool2(self.relu4(self.bn4(self.conv4(x))))  # (32,  7,  7)
        x = x.flatten(1)
        x = self.dropout(self.relu5(self.fc1(x)))
        return self.fc2(x)


# ── Submodels ──────────────────────────────────────────────────────────────────

class V2SUpToLayer0(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.conv1, self.bn1, self.relu1 = m.conv1, m.bn1, m.relu1
    def forward(self, x):
        return self.relu1(self.bn1(self.conv1(x)))

class V2SUpToLayer1Conv(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.conv1, self.bn1, self.relu1 = m.conv1, m.bn1, m.relu1
        self.conv2, self.bn2, self.relu2 = m.conv2, m.bn2, m.relu2
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        return self.relu2(self.bn2(self.conv2(x)))

class V2SUpToLayer1(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.conv1, self.bn1, self.relu1 = m.conv1, m.bn1, m.relu1
        self.conv2, self.bn2, self.relu2, self.pool1 = m.conv2, m.bn2, m.relu2, m.pool1
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        return self.pool1(self.relu2(self.bn2(self.conv2(x))))

class V2SUpToLayer2(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.conv1, self.bn1, self.relu1 = m.conv1, m.bn1, m.relu1
        self.conv2, self.bn2, self.relu2, self.pool1 = m.conv2, m.bn2, m.relu2, m.pool1
        self.conv3, self.bn3, self.relu3 = m.conv3, m.bn3, m.relu3
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(self.relu2(self.bn2(self.conv2(x))))
        return self.relu3(self.bn3(self.conv3(x)))

class V2SUpToLayer3Conv(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.conv1, self.bn1, self.relu1 = m.conv1, m.bn1, m.relu1
        self.conv2, self.bn2, self.relu2, self.pool1 = m.conv2, m.bn2, m.relu2, m.pool1
        self.conv3, self.bn3, self.relu3 = m.conv3, m.bn3, m.relu3
        self.conv4, self.bn4, self.relu4 = m.conv4, m.bn4, m.relu4
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        return self.relu4(self.bn4(self.conv4(x)))

class V2SUpToLayer3(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.conv1, self.bn1, self.relu1 = m.conv1, m.bn1, m.relu1
        self.conv2, self.bn2, self.relu2, self.pool1 = m.conv2, m.bn2, m.relu2, m.pool1
        self.conv3, self.bn3, self.relu3 = m.conv3, m.bn3, m.relu3
        self.conv4, self.bn4, self.relu4, self.pool2 = m.conv4, m.bn4, m.relu4, m.pool2
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        return self.pool2(self.relu4(self.bn4(self.conv4(x))))

class V2SUpToLayer4(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.conv1, self.bn1, self.relu1 = m.conv1, m.bn1, m.relu1
        self.conv2, self.bn2, self.relu2, self.pool1 = m.conv2, m.bn2, m.relu2, m.pool1
        self.conv3, self.bn3, self.relu3 = m.conv3, m.bn3, m.relu3
        self.conv4, self.bn4, self.relu4, self.pool2 = m.conv4, m.bn4, m.relu4, m.pool2
        self.fc1, self.relu5 = m.fc1, m.relu5
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.pool2(self.relu4(self.bn4(self.conv4(x))))
        x = x.flatten(1)
        return self.relu5(self.fc1(x))

class V2SFullWithSoftmax(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.conv1, self.bn1, self.relu1 = m.conv1, m.bn1, m.relu1
        self.conv2, self.bn2, self.relu2, self.pool1 = m.conv2, m.bn2, m.relu2, m.pool1
        self.conv3, self.bn3, self.relu3 = m.conv3, m.bn3, m.relu3
        self.conv4, self.bn4, self.relu4, self.pool2 = m.conv4, m.bn4, m.relu4, m.pool2
        self.fc1, self.relu5, self.fc2 = m.fc1, m.relu5, m.fc2
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.pool2(self.relu4(self.bn4(self.conv4(x))))
        x = x.flatten(1)
        x = self.relu5(self.fc1(x))
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

    model = MnistCNNv2Small().to(device)
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
    submodel.cpu().eval()
    dummy = torch.zeros(*input_shape)
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

    models_dir = os.path.join(PUBLIC_MODELS, args.dataset, 'v2small')
    os.makedirs(models_dir, exist_ok=True)

    model = train(args.dataset)
    model.cpu().eval()

    print('\nModel summary (torchinfo):')
    try:
        from torchinfo import summary
        summary(model, input_size=(1, 1, 28, 28), col_names=['output_size', 'num_params'])
    except ImportError:
        print('  torchinfo not installed')

    print(f'\nExporting ONNX models (v2small / {args.dataset})...')
    exports = [
        (V2SUpToLayer0(model),      'layer0.onnx',      (1, 16, 28, 28)),
        (V2SUpToLayer1Conv(model),  'layer1_conv.onnx', (1, 16, 28, 28)),
        (V2SUpToLayer1(model),      'layer1.onnx',      (1, 16, 14, 14)),
        (V2SUpToLayer2(model),      'layer2.onnx',      (1, 32, 14, 14)),
        (V2SUpToLayer3Conv(model),  'layer3_conv.onnx', (1, 32, 14, 14)),
        (V2SUpToLayer3(model),      'layer3.onnx',      (1, 32,  7,  7)),
        (V2SUpToLayer4(model),      'layer4.onnx',      (1, 256)),
        (V2SFullWithSoftmax(model), 'full.onnx',        (1, 10)),
    ]
    for submodel, name, expected in exports:
        path = os.path.join(models_dir, name)
        export_onnx(submodel, path)
        validate_onnx(path, expected)

    print('\nAll v2small models exported successfully.')


if __name__ == '__main__':
    main()
