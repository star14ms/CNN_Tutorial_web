"""
Train a small Vision Transformer (ViT) and export ONNX layer checkpoints.

Architecture:
  Patch embedding: img_size×img_size → num_patches patches of 4×4 pixels (patch_size=4)
  Each patch flattened to patch_dim = in_channels*4*4, projected to embed_dim=128

  Transformer encoder (4 blocks):
    MultiheadAttention(embed_dim=128, num_heads=8) + LayerNorm + Dropout
    FFN (128 → 64 → 128) + BatchNorm1d + Dropout

  CLS token → FC(128→num_classes) + Softmax

ONNX layer checkpoints exported:
  patch_embed.onnx  — after patch embedding + positional embed  (1, num_patches+1, 128)
  enc0.onnx         — after Transformer block 0                 (1, num_patches+1, 128)
  enc1.onnx         — after Transformer block 1                 (1, num_patches+1, 128)
  enc2.onnx         — after Transformer block 2                 (1, num_patches+1, 128)
  enc3.onnx         — after Transformer block 3                 (1, num_patches+1, 128)
  full.onnx         — CLS token → FC → Softmax(num_classes)    (1, num_classes)

  For 28×28 grayscale: num_patches=49,  patch_dim=16,  seq_len=50
  For 32×32 RGB:       num_patches=64,  patch_dim=48,  seq_len=65

Dataset support:
  1-channel 28×28: mnist, fashion_mnist, kuzushiji_mnist  → num_classes=10
  3-channel 32×32: cifar10, svhn                          → num_classes=10
  3-channel 32×32: cifar100                               → num_classes=100

Usage:
  conda activate deep-learning
  python train/train_vit.py --dataset mnist
  python train/train_vit.py --dataset fashion_mnist
  python train/train_vit.py --dataset kuzushiji_mnist
  python train/train_vit.py --dataset cifar10
  python train/train_vit.py --dataset cifar100
  python train/train_vit.py --dataset svhn
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Add train directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from device_utils import get_device_with_info

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

PATCH_SIZE = 4
EMBED_DIM  = 128
NUM_HEADS  = 8
NUM_LAYERS = 4
FFN_DIM    = 64
DROPOUT    = 0.1


# ── Model ──────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(EMBED_DIM)
        self.attn  = nn.MultiheadAttention(EMBED_DIM, NUM_HEADS, dropout=DROPOUT, batch_first=True)
        self.drop1 = nn.Dropout(DROPOUT)

        self.norm2 = nn.LayerNorm(EMBED_DIM)
        self.ffn   = nn.Sequential(
            nn.Linear(EMBED_DIM, FFN_DIM),
            nn.BatchNorm1d(FFN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FFN_DIM, EMBED_DIM),
            nn.BatchNorm1d(EMBED_DIM),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        B, S, D = x.shape
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop1(attn_out)
        x_norm = self.norm2(x)
        x_flat = x_norm.reshape(-1, D)
        ffn_out = self.ffn(x_flat).reshape(B, S, D)
        return x + ffn_out


class ViT(nn.Module):
    def __init__(self, in_channels, img_size, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_patches = (img_size // PATCH_SIZE) ** 2
        patch_dim = in_channels * PATCH_SIZE * PATCH_SIZE

        self.patch_embed = nn.Linear(patch_dim, EMBED_DIM)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.pos_embed   = nn.Parameter(torch.zeros(1, self.num_patches + 1, EMBED_DIM))
        self.drop_embed  = nn.Dropout(DROPOUT)

        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(NUM_LAYERS)])
        self.norm   = nn.LayerNorm(EMBED_DIM)
        self.head   = nn.Linear(EMBED_DIM, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _extract_patches(self, x):
        B = x.shape[0]
        # (B, C, H, W) → (B, C, H/P, W/P, P, P)
        x = x.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
        # → (B, num_patches, C*P*P)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, self.num_patches, self.in_channels * PATCH_SIZE * PATCH_SIZE)
        return x

    def forward(self, x):
        B = x.shape[0]
        x = self._extract_patches(x)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.drop_embed(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])


# ── Submodels ──────────────────────────────────────────────────────────────────

def _extract_patches(x, in_channels, num_patches):
    B = x.shape[0]
    x = x.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
    x = x.view(B, num_patches, in_channels * PATCH_SIZE * PATCH_SIZE)
    return x


class ViTUpToPatchEmbed(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.in_channels = m.in_channels
        self.num_patches = m.num_patches
        self.patch_embed = m.patch_embed
        self.cls_token   = m.cls_token
        self.pos_embed   = m.pos_embed
        self.drop_embed  = m.drop_embed

    def forward(self, x):
        B = x.shape[0]
        x = _extract_patches(x, self.in_channels, self.num_patches)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return self.drop_embed(x + self.pos_embed)


class ViTUpToBlock(nn.Module):
    def __init__(self, m, n_blocks):
        super().__init__()
        self.in_channels = m.in_channels
        self.num_patches = m.num_patches
        self.patch_embed = m.patch_embed
        self.cls_token   = m.cls_token
        self.pos_embed   = m.pos_embed
        self.drop_embed  = m.drop_embed
        self.blocks      = nn.ModuleList(list(m.blocks)[:n_blocks])

    def forward(self, x):
        B = x.shape[0]
        x = _extract_patches(x, self.in_channels, self.num_patches)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.drop_embed(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        return x


class ViTFullWithSoftmax(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.in_channels = m.in_channels
        self.num_patches = m.num_patches
        self.patch_embed = m.patch_embed
        self.cls_token   = m.cls_token
        self.pos_embed   = m.pos_embed
        self.drop_embed  = m.drop_embed
        self.blocks      = m.blocks
        self.norm        = m.norm
        self.head        = m.head

    def forward(self, x):
        B = x.shape[0]
        x = _extract_patches(x, self.in_channels, self.num_patches)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.drop_embed(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return torch.softmax(self.head(x[:, 0]), dim=1)


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
    device, device_str = get_device_with_info()
    print(f'Using device: {device_str}  dataset: {dataset_id}')

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
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)

    model = ViT(cfg['in_channels'], cfg['img_size'], cfg['num_classes']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, 21):
        model.train()
        total_loss = 0
        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    submodel.cpu().eval()
    dummy = torch.zeros(*input_shape)
    torch.onnx.export(
        submodel, dummy, path,
        opset_version=14,
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
    num_patches = (S // PATCH_SIZE) ** 2
    seq_len     = num_patches + 1
    input_shape = (1, C, S, S)

    models_dir = os.path.join(PUBLIC_MODELS, args.dataset, 'vit')
    os.makedirs(models_dir, exist_ok=True)

    model = train(args.dataset)
    model.cpu().eval()

    print('\nModel summary (torchinfo):')
    try:
        from torchinfo import summary
        summary(model, input_size=input_shape, col_names=['output_size', 'num_params'])
    except ImportError:
        print('  torchinfo not installed')

    print(f'\nExporting ONNX models (vit / {args.dataset})...')
    exports = [
        (ViTUpToPatchEmbed(model),  'patch_embed.onnx', (1, seq_len, EMBED_DIM)),
        (ViTUpToBlock(model, 1),    'enc0.onnx',        (1, seq_len, EMBED_DIM)),
        (ViTUpToBlock(model, 2),    'enc1.onnx',        (1, seq_len, EMBED_DIM)),
        (ViTUpToBlock(model, 3),    'enc2.onnx',        (1, seq_len, EMBED_DIM)),
        (ViTUpToBlock(model, 4),    'enc3.onnx',        (1, seq_len, EMBED_DIM)),
        (ViTFullWithSoftmax(model), 'full.onnx',        (1, N)),
    ]
    for submodel, name, expected in exports:
        path = os.path.join(models_dir, name)
        export_onnx(submodel, path, input_shape)
        validate_onnx(path, input_shape, expected)

    print('\nAll ViT models exported successfully.')


if __name__ == '__main__':
    main()
