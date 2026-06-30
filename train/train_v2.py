"""
Train a deeper CNN (v2) on MNIST with BatchNorm and Dropout.
Export 6 ONNX layer-checkpoint files for browser visualization.

Architecture:
  Conv1(1→32, 3×3) + BatchNorm + ReLU                     → (32, 28, 28)  [layer0]
  Conv2(32→32, 3×3) + BatchNorm + ReLU                     → (32, 28, 28)  [layer1_conv]
  MaxPool(2×2)                                       → (32, 14, 14)  [layer1]
  Conv3(32→64, 3×3) + BatchNorm + ReLU                     → (64, 14, 14)  [layer2]
  Conv4(64→64, 3×3) + BatchNorm + ReLU                     → (64, 14, 14)  [layer3_conv]
  MaxPool(2×2)                                       → (64,  7,  7)  [layer3]
  FC(3136→512) + ReLU  [Dropout(0.5) train only]   → (512,)         [layer4]
  FC(512→10) + Softmax                              → (10,)          [full]

Usage:
  conda activate deep-learning
  python train/train_v2.py --dataset mnist
  python train/train_v2.py --dataset fashion_mnist
  python train/train_v2.py --dataset kuzushiji_mnist
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

class MnistCNNv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(1,  32, 3, padding=1)
        self.bn1     = nn.BatchNorm2d(32)
        self.relu1   = nn.ReLU()

        self.conv2   = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2     = nn.BatchNorm2d(32)
        self.relu2   = nn.ReLU()
        self.pool1   = nn.MaxPool2d(2, 2)

        self.conv3   = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3     = nn.BatchNorm2d(64)
        self.relu3   = nn.ReLU()

        self.conv4   = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.relu4   = nn.ReLU()
        self.pool2   = nn.MaxPool2d(2, 2)

        self.fc1     = nn.Linear(64 * 7 * 7, 512)
        self.relu5   = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(512, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))            # (32, 28, 28)
        x = self.pool1(self.relu2(self.bn2(self.conv2(x)))) # (32, 14, 14)
        x = self.relu3(self.bn3(self.conv3(x)))            # (64, 14, 14)
        x = self.pool2(self.relu4(self.bn4(self.conv4(x)))) # (64,  7,  7)
        x = x.flatten(1)
        x = self.dropout(self.relu5(self.fc1(x)))          # (512,)
        return self.fc2(x)                                  # (10,)  raw logits


# ── Submodels for layer-by-layer export ────────────────────────────────────────

class V2UpToLayer0(nn.Module):
    """→ after Conv1+BN1+ReLU  (32, 28, 28)"""
    def __init__(self, m):
        super().__init__()
        self.conv1, self.bn1, self.relu1 = m.conv1, m.bn1, m.relu1
    def forward(self, x):
        return self.relu1(self.bn1(self.conv1(x)))

class V2UpToLayer1Conv(nn.Module):
    """→ after Conv2+BN2+ReLU, BEFORE Pool1  (32, 28, 28)"""
    def __init__(self, m):
        super().__init__()
        self.conv1, self.bn1, self.relu1 = m.conv1, m.bn1, m.relu1
        self.conv2, self.bn2, self.relu2 = m.conv2, m.bn2, m.relu2
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        return self.relu2(self.bn2(self.conv2(x)))

class V2UpToLayer1(nn.Module):
    """→ after Conv2+BN2+ReLU+Pool1  (32, 14, 14)"""
    def __init__(self, m):
        super().__init__()
        self.conv1, self.bn1, self.relu1 = m.conv1, m.bn1, m.relu1
        self.conv2, self.bn2, self.relu2, self.pool1 = m.conv2, m.bn2, m.relu2, m.pool1
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        return self.pool1(self.relu2(self.bn2(self.conv2(x))))

class V2UpToLayer2(nn.Module):
    """→ after Conv3+BN3+ReLU  (64, 14, 14)"""
    def __init__(self, m):
        super().__init__()
        self.conv1, self.bn1, self.relu1 = m.conv1, m.bn1, m.relu1
        self.conv2, self.bn2, self.relu2, self.pool1 = m.conv2, m.bn2, m.relu2, m.pool1
        self.conv3, self.bn3, self.relu3 = m.conv3, m.bn3, m.relu3
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(self.relu2(self.bn2(self.conv2(x))))
        return self.relu3(self.bn3(self.conv3(x)))

class V2UpToLayer3Conv(nn.Module):
    """→ after Conv4+BN4+ReLU, BEFORE Pool2  (64, 14, 14)"""
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

class V2UpToLayer3(nn.Module):
    """→ after Conv4+BN4+ReLU+Pool2  (64, 7, 7)"""
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

class V2UpToLayer4(nn.Module):
    """→ after FC1+ReLU, NO dropout (eval only)  (512,)"""
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
        return self.relu5(self.fc1(x))   # no dropout in eval

class V2FullWithSoftmax(nn.Module):
    """Full model → Softmax(10)"""
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
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}  dataset: {dataset_id}')

    if device.type == 'cuda':
        # Prevent cuDNN from auto-selecting algorithms that can cause
        # CUDNN_STATUS_INTERNAL_ERROR on some driver/workspace configurations.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    ds_class, norm_mean, norm_std = DATASET_MAP[dataset_id]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    train_ds = ds_class(DATA_DIR, train=True,  download=True, transform=transform)
    test_ds  = ds_class(DATA_DIR, train=False, download=True, transform=transform)
    pin = device.type == 'cuda'
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0, pin_memory=pin)
    test_dl  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0, pin_memory=pin)

    model = MnistCNNv2().to(device)
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
    except ImportError as e:
        print(f'  [skip validation — onnxruntime unavailable: {e}]')
        return
    try:
        sess = ort.InferenceSession(path)
        dummy = np.zeros((1, 1, 28, 28), dtype=np.float32)
        out = sess.run(None, {'input': dummy})
        shape = out[0].shape
        ok = '✓' if shape == expected_shape else '✗'
        print(f'  {ok} {os.path.basename(path)}: output shape = {shape}  (expected {expected_shape})')
    except Exception as e:
        print(f'  [validation error for {os.path.basename(path)}: {e}]')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist', choices=list(DATASET_MAP.keys()))
    args = parser.parse_args()

    models_dir = os.path.join(PUBLIC_MODELS, args.dataset, 'v2')
    os.makedirs(models_dir, exist_ok=True)

    model = train(args.dataset)
    model.cpu().eval()

    print('\nModel summary (torchinfo):')
    try:
        from torchinfo import summary
        summary(model, input_size=(1, 1, 28, 28), col_names=['output_size', 'num_params'])
    except ImportError:
        print('  torchinfo not installed — run: pip install torchinfo')

    print(f'\nExporting ONNX models (v2 / {args.dataset})...')
    exports = [
        (V2UpToLayer0(model),      'layer0.onnx',      (1, 32, 28, 28)),
        (V2UpToLayer1Conv(model),  'layer1_conv.onnx', (1, 32, 28, 28)),
        (V2UpToLayer1(model),      'layer1.onnx',      (1, 32, 14, 14)),
        (V2UpToLayer2(model),      'layer2.onnx',      (1, 64, 14, 14)),
        (V2UpToLayer3Conv(model),  'layer3_conv.onnx', (1, 64, 14, 14)),
        (V2UpToLayer3(model),      'layer3.onnx',      (1, 64,  7,  7)),
        (V2UpToLayer4(model),      'layer4.onnx',      (1, 512)),
        (V2FullWithSoftmax(model), 'full.onnx',        (1, 10)),
    ]
    for submodel, name, expected in exports:
        path = os.path.join(models_dir, name)
        export_onnx(submodel, path)
        validate_onnx(path, expected)

    print('\nAll v2 models exported successfully.')


if __name__ == '__main__':
    main()
