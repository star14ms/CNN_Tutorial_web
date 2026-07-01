"""
Train a deeper CNN (v2) + BatchNorm + Dropout and export ONNX layer checkpoints.

Architecture:
  Conv1(in_channels→32, 3×3) + BN + ReLU      → (32, img_size,   img_size)   [layer0]
  Conv2(32→32,          3×3) + BN + ReLU      → (32, img_size,   img_size)   [layer1_conv]
  MaxPool(2×2)                                 → (32, img_size/2, img_size/2) [layer1]
  Conv3(32→64,          3×3) + BN + ReLU      → (64, img_size/2, img_size/2) [layer2]
  Conv4(64→64,          3×3) + BN + ReLU      → (64, img_size/2, img_size/2) [layer3_conv]
  MaxPool(2×2)                                 → (64, feat,       feat)       [layer3]
  FC1(64*feat*feat→512) + ReLU + Dropout(0.5)  → (512,)                      [layer4]
  FC2(512→num_classes) + Softmax               → (num_classes,)               [full]

  feat = img_size // 4

Dataset support:
  1-channel 28×28: mnist, fashion_mnist, kuzushiji_mnist  → feat=7,  num_classes=10
  3-channel 32×32: cifar10, svhn                          → feat=8,  num_classes=10
  3-channel 32×32: cifar100                               → feat=8,  num_classes=100

Usage:
  conda activate deep-learning
  python train/train_v2.py --dataset mnist
  python train/train_v2.py --dataset fashion_mnist
  python train/train_v2.py --dataset kuzushiji_mnist
  python train/train_v2.py --dataset cifar10
  python train/train_v2.py --dataset cifar100
  python train/train_v2.py --dataset svhn
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

class DeepCNN(nn.Module):
    def __init__(self, in_channels, img_size, num_classes):
        super().__init__()
        feat = img_size // 4
        self.conv1   = nn.Conv2d(in_channels, 32, 3, padding=1)
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

        self.fc1     = nn.Linear(64 * feat * feat, 512)
        self.relu5   = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.pool2(self.relu4(self.bn4(self.conv4(x))))
        x = x.flatten(1)
        x = self.dropout(self.relu5(self.fc1(x)))
        return self.fc2(x)


# ── Submodels ──────────────────────────────────────────────────────────────────

class V2UpToLayer0(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.conv1, self.bn1, self.relu1 = m.conv1, m.bn1, m.relu1
    def forward(self, x):
        return self.relu1(self.bn1(self.conv1(x)))

class V2UpToLayer1Conv(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.conv1, self.bn1, self.relu1 = m.conv1, m.bn1, m.relu1
        self.conv2, self.bn2, self.relu2 = m.conv2, m.bn2, m.relu2
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        return self.relu2(self.bn2(self.conv2(x)))

class V2UpToLayer1(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.conv1, self.bn1, self.relu1 = m.conv1, m.bn1, m.relu1
        self.conv2, self.bn2, self.relu2, self.pool1 = m.conv2, m.bn2, m.relu2, m.pool1
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        return self.pool1(self.relu2(self.bn2(self.conv2(x))))

class V2UpToLayer2(nn.Module):
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

class V2FullWithSoftmax(nn.Module):
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
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}  dataset: {dataset_id}')

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    base_tf = [transforms.ToTensor(), transforms.Normalize(cfg['mean'], cfg['std'])]
    if cfg['in_channels'] == 3:
        train_tf = transforms.Compose([
            transforms.RandomCrop(cfg['img_size'], padding=4),
            transforms.RandomHorizontalFlip(),
        ] + base_tf)
    else:
        train_tf = transforms.Compose(base_tf)
    test_tf = transforms.Compose(base_tf)

    train_ds, test_ds = get_datasets(dataset_id, train_tf, test_tf)
    pin = device.type == 'cuda'
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0, pin_memory=pin)
    test_dl  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0, pin_memory=pin)

    model = DeepCNN(cfg['in_channels'], cfg['img_size'], cfg['num_classes']).to(device)
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
        print(f'  [validation error for {os.path.basename(path)}: {e}]')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist', choices=list(DATASET_MAP.keys()))
    args = parser.parse_args()

    cfg = DATASET_MAP[args.dataset]
    C, S, N = cfg['in_channels'], cfg['img_size'], cfg['num_classes']
    feat = S // 4
    input_shape = (1, C, S, S)

    models_dir = os.path.join(PUBLIC_MODELS, args.dataset, 'v2')
    os.makedirs(models_dir, exist_ok=True)

    model = train(args.dataset)
    model.cpu().eval()

    print('\nModel summary (torchinfo):')
    try:
        from torchinfo import summary
        summary(model, input_size=input_shape, col_names=['output_size', 'num_params'])
    except ImportError:
        print('  torchinfo not installed — run: pip install torchinfo')

    print(f'\nExporting ONNX models (v2 / {args.dataset})...')
    exports = [
        (V2UpToLayer0(model),      'layer0.onnx',      (1, 32, S,    S)),
        (V2UpToLayer1Conv(model),  'layer1_conv.onnx', (1, 32, S,    S)),
        (V2UpToLayer1(model),      'layer1.onnx',      (1, 32, S//2, S//2)),
        (V2UpToLayer2(model),      'layer2.onnx',      (1, 64, S//2, S//2)),
        (V2UpToLayer3Conv(model),  'layer3_conv.onnx', (1, 64, S//2, S//2)),
        (V2UpToLayer3(model),      'layer3.onnx',      (1, 64, feat, feat)),
        (V2UpToLayer4(model),      'layer4.onnx',      (1, 512)),
        (V2FullWithSoftmax(model), 'full.onnx',        (1, N)),
    ]
    for submodel, name, expected in exports:
        path = os.path.join(models_dir, name)
        export_onnx(submodel, path, input_shape)
        validate_onnx(path, input_shape, expected)

    print('\nAll v2 models exported successfully.')


if __name__ == '__main__':
    main()
