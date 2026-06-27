"""
Train a CNN on MNIST and export 5 ONNX files for browser inference.

Exported models (saved to ../public/models/):
  layer0.onnx  - input → after Conv1+ReLU  (32, 28, 28)
  layer1.onnx  - input → after MaxPool1    (32, 14, 14)
  layer2.onnx  - input → after Conv2+ReLU  (64, 14, 14)
  layer3.onnx  - input → after MaxPool2    (64,  7,  7)
  mnist-cnn.onnx - full model → softmax    (10,)

Usage:
  pip install -r requirements.txt
  python train_model.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'public', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Model ──────────────────────────────────────────────────────────────────────

class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)   # (32, 28, 28)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)                # (32, 14, 14)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # (64, 14, 14)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)                # (64,  7,  7)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = x.flatten(1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# Submodels — copy layer references to produce a clean, self-contained ONNX graph
class UpToLayer0(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.conv1 = m.conv1; self.relu1 = m.relu1
    def forward(self, x): return self.relu1(self.conv1(x))

class UpToLayer1(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.conv1 = m.conv1; self.relu1 = m.relu1; self.pool1 = m.pool1
    def forward(self, x): return self.pool1(self.relu1(self.conv1(x)))

class UpToLayer2(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.conv1 = m.conv1; self.relu1 = m.relu1; self.pool1 = m.pool1
        self.conv2 = m.conv2; self.relu2 = m.relu2
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        return self.relu2(self.conv2(x))

class UpToLayer3(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.conv1 = m.conv1; self.relu1 = m.relu1; self.pool1 = m.pool1
        self.conv2 = m.conv2; self.relu2 = m.relu2; self.pool2 = m.pool2
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        return self.pool2(self.relu2(self.conv2(x)))

class FullWithSoftmax(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.conv1 = m.conv1; self.relu1 = m.relu1; self.pool1 = m.pool1
        self.conv2 = m.conv2; self.relu2 = m.relu2; self.pool2 = m.pool2
        self.fc1 = m.fc1; self.relu3 = m.relu3; self.fc2 = m.fc2
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.flatten(1)
        x = self.relu3(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=1)


# ── Training ───────────────────────────────────────────────────────────────────

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)

    model = MnistCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 11):
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


def validate_onnx(path):
    import onnxruntime as ort
    sess = ort.InferenceSession(path)
    dummy = np.zeros((1, 1, 28, 28), dtype=np.float32)
    out = sess.run(None, {'input': dummy})
    print(f'  Validated {os.path.basename(path)}: output shape = {out[0].shape}')


def main():
    model = train()
    model.cpu().eval()

    print('\nModel summary (torchinfo):')
    try:
        from torchinfo import summary
        summary(model, input_size=(1, 1, 28, 28), col_names=['output_size', 'num_params'])
    except ImportError:
        print('  torchinfo not installed — run: pip install torchinfo')

    print('\nExporting ONNX models...')
    exports = [
        (UpToLayer0(model),     'layer0.onnx'),
        (UpToLayer1(model),     'layer1.onnx'),
        (UpToLayer2(model),     'layer2.onnx'),
        (UpToLayer3(model),     'layer3.onnx'),
        (FullWithSoftmax(model),'mnist-cnn.onnx'),
    ]
    for submodel, name in exports:
        path = os.path.join(MODELS_DIR, name)
        export_onnx(submodel, path)
        validate_onnx(path)

    print('\nAll models exported successfully.')


if __name__ == '__main__':
    main()
