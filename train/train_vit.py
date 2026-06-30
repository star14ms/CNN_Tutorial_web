"""
Train a small Vision Transformer (ViT) + BatchNorm + Dropout on MNIST.

Architecture:
  Patch embedding: 28×28 image → 49 patches of 4×4 pixels (patch_size=4)
  Each patch flattened to dim=16, projected to embed_dim=64

  Transformer encoder (4 layers):
    MultiheadAttention(embed_dim=64, num_heads=4) + LayerNorm + Dropout
    FFN (64 → 128 → 64) + BatchNorm1d + Dropout

  CLS token → FC(64→10) + Softmax

Note: This ViT is purpose-built for MNIST visualization, keeping embed_dim small
so each patch token's activations can be rendered as a spatial 7×7 feature grid.

ONNX layer checkpoints exported:
  vit_patch_embed.onnx  — after patch embedding + positional embed  (50, 64)  [layer0]
  vit_enc0.onnx         — after Transformer block 0                 (50, 64)  [layer1]
  vit_enc1.onnx         — after Transformer block 1                 (50, 64)  [layer2]
  vit_enc2.onnx         — after Transformer block 2                 (50, 64)  [layer3]
  vit_enc3.onnx         — after Transformer block 3                 (50, 64)  [layer4]
  vit_mnist-cnn.onnx    — CLS token → FC → Softmax(10)             (10,)     [full]

Usage:
  conda activate deep-learning
  python train/train_vit.py --dataset mnist
  python train/train_vit.py --dataset fashion_mnist
  python train/train_vit.py --dataset kuzushiji_mnist
"""

import argparse
import os
import math
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

PATCH_SIZE  = 4     # 28/4 = 7 patches per side → 49 patches total
NUM_PATCHES = (28 // PATCH_SIZE) ** 2  # 49
EMBED_DIM   = 128
NUM_HEADS   = 8
NUM_LAYERS  = 4
FFN_DIM     = 64
DROPOUT     = 0.1


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
        # x: (B, S, D)  S = num_patches+1
        B, S, D = x.shape
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop1(attn_out)

        x_norm = self.norm2(x)
        # BatchNorm1d expects (B, C) or (B, C, L) — reshape for (B*S, D)
        x_flat = x_norm.reshape(-1, D)
        ffn_out = self.ffn(x_flat).reshape(B, S, D)
        x = x + ffn_out
        return x


class MnistViT(nn.Module):
    def __init__(self):
        super().__init__()
        patch_dim = 1 * PATCH_SIZE * PATCH_SIZE  # 16 (grayscale)
        self.patch_embed = nn.Linear(patch_dim, EMBED_DIM)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.pos_embed   = nn.Parameter(torch.zeros(1, NUM_PATCHES + 1, EMBED_DIM))
        self.drop_embed  = nn.Dropout(DROPOUT)

        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(NUM_LAYERS)])
        self.norm   = nn.LayerNorm(EMBED_DIM)
        self.head   = nn.Linear(EMBED_DIM, 10)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _extract_patches(self, x):
        B = x.shape[0]
        # (B, 1, 28, 28) → (B, 49, 16)
        x = x.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
        # x: (B, 1, 7, 7, 4, 4)
        x = x.contiguous().view(B, 1, NUM_PATCHES, PATCH_SIZE * PATCH_SIZE)
        x = x.squeeze(1)  # (B, 49, 16)
        return x

    def forward(self, x):
        B = x.shape[0]
        x = self._extract_patches(x)           # (B, 49, 16)
        x = self.patch_embed(x)                # (B, 49, 64)
        cls = self.cls_token.expand(B, -1, -1) # (B,  1, 64)
        x = torch.cat([cls, x], dim=1)         # (B, 50, 64)
        x = self.drop_embed(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])              # CLS token → (B, 10)


# ── Submodels ──────────────────────────────────────────────────────────────────

class ViTUpToPatchEmbed(nn.Module):
    """After patch embedding + positional encoding: (B, 50, 64)"""
    def __init__(self, m):
        super().__init__()
        self.patch_embed = m.patch_embed
        self.cls_token   = m.cls_token
        self.pos_embed   = m.pos_embed
        self.drop_embed  = m.drop_embed

    def forward(self, x):
        B = x.shape[0]
        x = x.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
        x = x.contiguous().view(B, 1, NUM_PATCHES, PATCH_SIZE * PATCH_SIZE).squeeze(1)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return self.drop_embed(x + self.pos_embed)


class ViTUpToBlock(nn.Module):
    """After first N transformer blocks: (B, 50, 64)"""
    def __init__(self, m, n_blocks):
        super().__init__()
        self.patch_embed = m.patch_embed
        self.cls_token   = m.cls_token
        self.pos_embed   = m.pos_embed
        self.drop_embed  = m.drop_embed
        self.blocks      = nn.ModuleList(list(m.blocks)[:n_blocks])

    def forward(self, x):
        B = x.shape[0]
        x = x.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
        x = x.contiguous().view(B, 1, NUM_PATCHES, PATCH_SIZE * PATCH_SIZE).squeeze(1)
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
        self.patch_embed = m.patch_embed
        self.cls_token   = m.cls_token
        self.pos_embed   = m.pos_embed
        self.drop_embed  = m.drop_embed
        self.blocks      = m.blocks
        self.norm        = m.norm
        self.head        = m.head

    def forward(self, x):
        B = x.shape[0]
        x = x.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
        x = x.contiguous().view(B, 1, NUM_PATCHES, PATCH_SIZE * PATCH_SIZE).squeeze(1)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.drop_embed(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return torch.softmax(self.head(x[:, 0]), dim=1)


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

    model = MnistViT().to(device)
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

def export_onnx(submodel, path, input_shape=(1, 1, 28, 28)):
    submodel.cpu().eval()
    dummy = torch.zeros(*input_shape)
    torch.onnx.export(
        submodel, dummy, path,
        opset_version=14,  # needed for unfold / dynamic operations
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

    models_dir = os.path.join(PUBLIC_MODELS, args.dataset, 'vit')
    os.makedirs(models_dir, exist_ok=True)

    model = train(args.dataset)
    model.cpu().eval()

    print('\nModel summary (torchinfo):')
    try:
        from torchinfo import summary
        summary(model, input_size=(1, 1, 28, 28), col_names=['output_size', 'num_params'])
    except ImportError:
        print('  torchinfo not installed')

    print(f'\nExporting ONNX models (vit / {args.dataset})...')
    exports = [
        (ViTUpToPatchEmbed(model),  'patch_embed.onnx', (1, 50, EMBED_DIM)),
        (ViTUpToBlock(model, 1),    'enc0.onnx',        (1, 50, EMBED_DIM)),
        (ViTUpToBlock(model, 2),    'enc1.onnx',        (1, 50, EMBED_DIM)),
        (ViTUpToBlock(model, 3),    'enc2.onnx',        (1, 50, EMBED_DIM)),
        (ViTUpToBlock(model, 4),    'enc3.onnx',        (1, 50, EMBED_DIM)),
        (ViTFullWithSoftmax(model), 'full.onnx',        (1, 10)),
    ]
    for submodel, name, expected in exports:
        path = os.path.join(models_dir, name)
        export_onnx(submodel, path)
        validate_onnx(path, expected)

    print('\nAll ViT models exported successfully.')


if __name__ == '__main__':
    main()
