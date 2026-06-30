/**
 * Model configurations for browser visualization.
 * Each config defines layer layout, ONNX model files, and receptive-field connectivity.
 *
 * Inference result keys: 'output' (from 'full' file), all others match modelFile name.
 * Special dataKey '__input__' maps to the raw input pixels (Float32Array 784).
 */

export const MODEL_CONFIGS = {

  // ── Linear ───────────────────────────────────────────────────────────────────

  linear: {
    id:          'linear',
    label:       'Linear',
    description: 'Linear(784→512→10) — 98.59% acc',
    totalParams: 407050,
    parametersFile: 'parameters.json',
    torchinfo: {
      multAddsM:    0.41,
      fwdBwdSizeMB: 0.00,
      paramsSizeMB: 1.63,
      totalSizeMB:  1.64,
    },
    modelFiles: [
      { name: 'layer0', file: 'layer0.onnx' },
      { name: 'full',   file: 'full.onnx' },
    ],
    layerDefs: [
      { name: 'Input',          channels:  1, h: 28, w:  28 },
      { name: 'Flatten',        channels:  1, h:  1, w: 784, dataKey: '__input__' },
      { name: 'FC1 + ReLU',     channels:  1, h:  1, w: 512, dataKey: 'layer0',
        sublayers: [
          { type: 'Linear(784→512)', params: 401920 },
          { type: 'ReLU',            params: 0      },
          { type: 'Dropout(0.5)',    params: 0      },
        ]},
      { name: 'FC2 + Softmax',  channels: 10, h:  1, w:   1, dataKey: 'output',
        channelLabels: '__classes__',
        sublayers: [
          { type: 'Linear(512→10)', params: 5130 },
          { type: 'Softmax',        params: 0    },
        ]},
    ],
    connectivity: [
      null,
      { type: 'flatten', prevLi: 0 },
      { type: 'fc',      prevLi: 1, mode: 'all', sampleStep: 1 },
      { type: 'fc',      prevLi: 2, mode: 'all', sampleStep: 1 },
    ],
  },

  // ── Simple CNN ───────────────────────────────────────────────────────────────

  v1: {
    id:          'v1',
    label:       'Simple CNN',
    description: '2 Conv + MaxPool + FC(128) — 99.20% acc',
    totalParams: 421642,
    parametersFile: 'parameters.json',
    modelFiles: [
      { name: 'layer0', file: 'layer0.onnx' },
      { name: 'layer1', file: 'layer1.onnx' },
      { name: 'layer2', file: 'layer2.onnx' },
      { name: 'layer3', file: 'layer3.onnx' },
      { name: 'layer4', file: 'layer4.onnx' },
      { name: 'full',   file: 'full.onnx' },
    ],
    layerDefs: [
      { name: 'Input',         channels:  1, h: 28, w: 28 },
      { name: 'Conv1 + ReLU', channels: 32, h: 28, w: 28, dataKey: 'layer0',
        sublayers: [
          { type: 'Conv2d(1→32, 3×3, pad=1)', params: 320 },
          { type: 'ReLU',                     params: 0   },
        ]},
      { name: 'MaxPool 1',    channels: 32, h: 14, w: 14, dataKey: 'layer1',
        sublayers: [
          { type: 'MaxPool2d(2×2)', params: 0 },
        ]},
      { name: 'Conv2 + ReLU', channels: 64, h: 14, w: 14, dataKey: 'layer2',
        sublayers: [
          { type: 'Conv2d(32→64, 3×3, pad=1)', params: 18496 },
          { type: 'ReLU',                      params: 0     },
        ]},
      { name: 'MaxPool 2',    channels: 64, h:  7, w:  7, dataKey: 'layer3',
        sublayers: [
          { type: 'MaxPool2d(2×2)', params: 0 },
        ]},
      { name: 'Flatten',       channels:  1, h:  1, w: 3136, dataKey: 'layer3' },
      { name: 'FC1 + ReLU',    channels:  1, h:  1, w: 128, dataKey: 'layer4',
        sublayers: [
          { type: 'Linear(3136→128)', params: 401536 },
          { type: 'ReLU',             params: 0      },
        ]},
      { name: 'FC2 + Softmax', channels: 10, h:  1, w:  1, dataKey: 'output',
        channelLabels: '__classes__',
        sublayers: [
          { type: 'Linear(128→10)', params: 1290 },
          { type: 'Softmax',        params: 0    },
        ]},
    ],
    connectivity: [
      null,
      { type: 'conv',    prevLi: 0, kernel: 3, pad: 1, prevChannels:  1 },
      { type: 'pool',    prevLi: 1, kernel: 2 },
      { type: 'conv',    prevLi: 2, kernel: 3, pad: 1, prevChannels: 32 },
      { type: 'pool',    prevLi: 3, kernel: 2 },
      { type: 'flatten', prevLi: 4 },
      { type: 'fc',      prevLi: 5, mode: 'all', sampleStep: 1 },
      { type: 'fc',      prevLi: 6, mode: 'all', sampleStep: 1 },
    ],
  },

  // ── Simple CNN + BN + Dropout ─────────────────────────────────────────────────

  v1bn: {
    id:          'v1bn',
    label:       'Simple CNN + BN',
    description: '2 Conv + MaxPool + BatchNorm + Dropout + FC(128) — 99.30% acc',
    totalParams: 421834,
    parametersFile: 'parameters.json',
    torchinfo: {
      multAddsM:    4.28,
      fwdBwdSizeMB: 0.60,
      paramsSizeMB: 1.69,
      totalSizeMB:  2.29,
    },
    modelFiles: [
      { name: 'layer0', file: 'layer0.onnx' },
      { name: 'layer1', file: 'layer1.onnx' },
      { name: 'layer2', file: 'layer2.onnx' },
      { name: 'layer3', file: 'layer3.onnx' },
      { name: 'layer4', file: 'layer4.onnx' },
      { name: 'full',   file: 'full.onnx' },
    ],
    layerDefs: [
      { name: 'Input',                        channels:  1, h: 28, w:   28 },
      { name: 'Conv1 + BatchNorm + ReLU',     channels: 32, h: 28, w:   28, dataKey: 'layer0',
        sublayers: [
          { type: 'Conv2d(1→32, 3×3, pad=1)', params: 320 },
          { type: 'BatchNorm2d(32)',           params: 64  },
          { type: 'ReLU',                     params: 0   },
        ]},
      { name: 'MaxPool 1',                    channels: 32, h: 14, w:   14, dataKey: 'layer1',
        sublayers: [
          { type: 'MaxPool2d(2×2)', params: 0 },
        ]},
      { name: 'Conv2 + BatchNorm + ReLU',     channels: 64, h: 14, w:   14, dataKey: 'layer2',
        sublayers: [
          { type: 'Conv2d(32→64, 3×3, pad=1)', params: 18496 },
          { type: 'BatchNorm2d(64)',            params: 128   },
          { type: 'ReLU',                      params: 0     },
        ]},
      { name: 'MaxPool 2',                    channels: 64, h:  7, w:    7, dataKey: 'layer3',
        sublayers: [
          { type: 'MaxPool2d(2×2)', params: 0 },
        ]},
      { name: 'Flatten',                      channels:  1, h:  1, w: 3136, dataKey: 'layer3' },
      { name: 'FC1 + ReLU',                   channels:  1, h:  1, w:  128, dataKey: 'layer4',
        sublayers: [
          { type: 'Linear(3136→128)', params: 401536 },
          { type: 'ReLU',             params: 0      },
          { type: 'Dropout(0.5)',     params: 0      },
        ]},
      { name: 'FC2 + Softmax',                channels: 10, h:  1, w:    1, dataKey: 'output',
        channelLabels: '__classes__',
        sublayers: [
          { type: 'Linear(128→10)', params: 1290 },
          { type: 'Softmax',        params: 0    },
        ]},
    ],
    connectivity: [
      null,
      { type: 'conv',    prevLi: 0, kernel: 3, pad: 1, prevChannels:  1 },
      { type: 'pool',    prevLi: 1, kernel: 2 },
      { type: 'conv',    prevLi: 2, kernel: 3, pad: 1, prevChannels: 32 },
      { type: 'pool',    prevLi: 3, kernel: 2 },
      { type: 'flatten', prevLi: 4 },
      { type: 'fc',      prevLi: 5, mode: 'all', sampleStep: 1 },
      { type: 'fc',      prevLi: 6, mode: 'all', sampleStep: 1 },
    ],
  },

  // ── Deep CNN (small) + BN + Dropout ──────────────────────────────────────────

  v2small: {
    id:          'v2small',
    label:       'Deep CNN (small)',
    description: '4 Conv + MaxPool + BatchNorm + Dropout + FC(256) — 99.56% acc',
    totalParams: 420794,
    parametersFile: 'parameters.json',
    torchinfo: {
      multAddsM:    5.07,
      fwdBwdSizeMB: 0.60,
      paramsSizeMB: 1.68,
      totalSizeMB:  2.29,
    },
    modelFiles: [
      { name: 'layer0',      file: 'layer0.onnx' },
      { name: 'layer1_conv', file: 'layer1_conv.onnx' },
      { name: 'layer1',      file: 'layer1.onnx' },
      { name: 'layer2',      file: 'layer2.onnx' },
      { name: 'layer3_conv', file: 'layer3_conv.onnx' },
      { name: 'layer3',      file: 'layer3.onnx' },
      { name: 'layer4',      file: 'layer4.onnx' },
      { name: 'full',        file: 'full.onnx' },
    ],
    layerDefs: [
      { name: 'Input',                       channels:  1, h: 28, w:   28 },
      { name: 'Conv1 + BatchNorm + ReLU',    channels: 16, h: 28, w:   28, dataKey: 'layer0',
        sublayers: [
          { type: 'Conv2d(1→16, 3×3, pad=1)',  params: 160 },
          { type: 'BatchNorm2d(16)',            params: 32  },
          { type: 'ReLU',                      params: 0   },
        ]},
      { name: 'Conv2 + BatchNorm + ReLU',    channels: 16, h: 28, w:   28, dataKey: 'layer1_conv',
        sublayers: [
          { type: 'Conv2d(16→16, 3×3, pad=1)', params: 2320 },
          { type: 'BatchNorm2d(16)',            params: 32   },
          { type: 'ReLU',                      params: 0    },
        ]},
      { name: 'MaxPool 1',                   channels: 16, h: 14, w:   14, dataKey: 'layer1',
        sublayers: [
          { type: 'MaxPool2d(2×2)', params: 0 },
        ]},
      { name: 'Conv3 + BatchNorm + ReLU',    channels: 32, h: 14, w:   14, dataKey: 'layer2',
        sublayers: [
          { type: 'Conv2d(16→32, 3×3, pad=1)', params: 4640 },
          { type: 'BatchNorm2d(32)',            params: 64   },
          { type: 'ReLU',                      params: 0    },
        ]},
      { name: 'Conv4 + BatchNorm + ReLU',    channels: 32, h: 14, w:   14, dataKey: 'layer3_conv',
        sublayers: [
          { type: 'Conv2d(32→32, 3×3, pad=1)', params: 9248 },
          { type: 'BatchNorm2d(32)',            params: 64   },
          { type: 'ReLU',                      params: 0    },
        ]},
      { name: 'MaxPool 2',                   channels: 32, h:  7, w:    7, dataKey: 'layer3',
        sublayers: [
          { type: 'MaxPool2d(2×2)', params: 0 },
        ]},
      { name: 'Flatten',                     channels:  1, h:  1, w: 1568, dataKey: 'layer3' },
      { name: 'FC1 + ReLU',                  channels:  1, h:  1, w:  256, dataKey: 'layer4',
        sublayers: [
          { type: 'Linear(1568→256)', params: 401664 },
          { type: 'ReLU',             params: 0      },
          { type: 'Dropout(0.5)',     params: 0      },
        ]},
      { name: 'FC2 + Softmax',               channels: 10, h:  1, w:    1, dataKey: 'output',
        channelLabels: '__classes__',
        sublayers: [
          { type: 'Linear(256→10)', params: 2570 },
          { type: 'Softmax',        params: 0    },
        ]},
    ],
    connectivity: [
      null,
      { type: 'conv',    prevLi: 0, kernel: 3, pad: 1, prevChannels:  1 },
      { type: 'conv',    prevLi: 1, kernel: 3, pad: 1, prevChannels: 16 },
      { type: 'pool',    prevLi: 2, kernel: 2 },
      { type: 'conv',    prevLi: 3, kernel: 3, pad: 1, prevChannels: 16 },
      { type: 'conv',    prevLi: 4, kernel: 3, pad: 1, prevChannels: 32 },
      { type: 'pool',    prevLi: 5, kernel: 2 },
      { type: 'flatten', prevLi: 6 },
      { type: 'fc',      prevLi: 7, mode: 'all', sampleStep: 1 },
      { type: 'fc',      prevLi: 8, mode: 'all', sampleStep: 1 },
    ],
  },

  // ── Deep CNN + BN + Dropout ───────────────────────────────────────────────────

  v2: {
    id:          'v2',
    label:       'Deep CNN + BN',
    description: '4 Conv + MaxPool + BatchNorm + Dropout + FC(512) — 99.61% acc',
    totalParams: 1676650,
    parametersFile: 'parameters.json',
    modelFiles: [
      { name: 'layer0',      file: 'layer0.onnx' },
      { name: 'layer1_conv', file: 'layer1_conv.onnx' },
      { name: 'layer1',      file: 'layer1.onnx' },
      { name: 'layer2',      file: 'layer2.onnx' },
      { name: 'layer3_conv', file: 'layer3_conv.onnx' },
      { name: 'layer3',      file: 'layer3.onnx' },
      { name: 'layer4',      file: 'layer4.onnx' },
      { name: 'full',        file: 'full.onnx' },
    ],
    layerDefs: [
      { name: 'Input',               channels:  1, h: 28, w: 28 },
      { name: 'Conv1 + BatchNorm + ReLU',  channels: 32, h: 28, w: 28, dataKey: 'layer0',
        sublayers: [
          { type: 'Conv2d(1→32, 3×3, pad=1)', params: 320 },
          { type: 'BatchNorm2d(32)',           params: 64  },
          { type: 'ReLU',                     params: 0   },
        ]},
      { name: 'Conv2 + BatchNorm + ReLU',  channels: 32, h: 28, w: 28, dataKey: 'layer1_conv',
        sublayers: [
          { type: 'Conv2d(32→32, 3×3, pad=1)', params: 9248 },
          { type: 'BatchNorm2d(32)',            params: 64   },
          { type: 'ReLU',                      params: 0    },
        ]},
      { name: 'MaxPool 1',          channels: 32, h: 14, w: 14, dataKey: 'layer1',
        sublayers: [
          { type: 'MaxPool2d(2×2)', params: 0 },
        ]},
      { name: 'Conv3 + BatchNorm + ReLU',  channels: 64, h: 14, w: 14, dataKey: 'layer2',
        sublayers: [
          { type: 'Conv2d(32→64, 3×3, pad=1)', params: 18496 },
          { type: 'BatchNorm2d(64)',            params: 128   },
          { type: 'ReLU',                      params: 0     },
        ]},
      { name: 'Conv4 + BatchNorm + ReLU',  channels: 64, h: 14, w: 14, dataKey: 'layer3_conv',
        sublayers: [
          { type: 'Conv2d(64→64, 3×3, pad=1)', params: 36928 },
          { type: 'BatchNorm2d(64)',            params: 128   },
          { type: 'ReLU',                      params: 0     },
        ]},
      { name: 'MaxPool 2',          channels: 64, h:  7, w:  7, dataKey: 'layer3',
        sublayers: [
          { type: 'MaxPool2d(2×2)', params: 0 },
        ]},
      { name: 'Flatten',            channels:  1, h:  1, w: 3136, dataKey: 'layer3' },
      { name: 'FC1 + ReLU',         channels:  1, h:  1, w: 512, dataKey: 'layer4',
        sublayers: [
          { type: 'Linear(3136→512)', params: 1606144 },
          { type: 'ReLU',             params: 0       },
          { type: 'Dropout(0.5)',     params: 0       },
        ]},
      { name: 'FC2 + Softmax',      channels: 10, h:  1, w:  1,  dataKey: 'output',
        channelLabels: '__classes__',
        sublayers: [
          { type: 'Linear(512→10)', params: 5130 },
          { type: 'Softmax',        params: 0    },
        ]},
    ],
    connectivity: [
      null,
      { type: 'conv',    prevLi: 0, kernel: 3, pad: 1, prevChannels:  1 },
      { type: 'conv',    prevLi: 1, kernel: 3, pad: 1, prevChannels: 32 },
      { type: 'pool',    prevLi: 2, kernel: 2 },
      { type: 'conv',    prevLi: 3, kernel: 3, pad: 1, prevChannels: 32 },
      { type: 'conv',    prevLi: 4, kernel: 3, pad: 1, prevChannels: 64 },
      { type: 'pool',    prevLi: 5, kernel: 2 },
      { type: 'flatten', prevLi: 6 },
      { type: 'fc',      prevLi: 7, mode: 'all', sampleStep: 1 },
      { type: 'fc',      prevLi: 8, mode: 'all', sampleStep: 1 },
    ],
  },

  // ── Vision Transformer ────────────────────────────────────────────────────────

  vit: {
    id:          'vit',
    label:       'Vision Transformer',
    description: 'ViT (patch=4, embed=128, 4 blocks, 8 heads) — 97.79% acc',
    totalParams: 344330,
    parametersFile: 'parameters.json',
    torchinfo: {
      multAddsM:    3.40,
      fwdBwdSizeMB: 1.13,
      paramsSizeMB: 0.29,
      totalSizeMB:  1.42,
    },
    modelFiles: [
      { name: 'layer0', file: 'patch_embed.onnx' },
      { name: 'layer1', file: 'enc0.onnx' },
      { name: 'layer2', file: 'enc1.onnx' },
      { name: 'layer3', file: 'enc2.onnx' },
      { name: 'layer4', file: 'enc3.onnx' },
      { name: 'full',   file: 'full.onnx' },
    ],
    // ViT outputs (50, 128): 50 tokens (CLS + 49 spatial patches), embed_dim=128.
    // Visualized as channels=50 × (1×128) strips — CHW layout matches ONNX output.
    layerDefs: [
      { name: 'Input',               channels:  1, h: 28, w:  28 },
      { name: 'Patch Embed',         channels: 50, h:  1, w: 128, dataKey: 'layer0',
        sublayers: [
          { type: 'Unfold(4×4 patches → 49 tokens)', params: 0    },
          { type: 'Linear(16→128)',                   params: 2176 },
          { type: 'Pos Embed + CLS token',            params: 6528 },
        ]},
      { name: 'Encoder Block 0',     channels: 50, h:  1, w: 128, dataKey: 'layer1',
        sublayers: [
          { type: 'LayerNorm',                        params: 256   },
          { type: 'MultiheadAttention(128, 8 heads)', params: 66048 },
          { type: 'LayerNorm',                        params: 256   },
          { type: 'FFN(128→64→128) + BN + Dropout',  params: 16960 },
        ]},
      { name: 'Encoder Block 1',     channels: 50, h:  1, w: 128, dataKey: 'layer2',
        sublayers: [
          { type: 'LayerNorm',                        params: 256   },
          { type: 'MultiheadAttention(128, 8 heads)', params: 66048 },
          { type: 'LayerNorm',                        params: 256   },
          { type: 'FFN(128→64→128) + BN + Dropout',  params: 16960 },
        ]},
      { name: 'Encoder Block 2',     channels: 50, h:  1, w: 128, dataKey: 'layer3',
        sublayers: [
          { type: 'LayerNorm',                        params: 256   },
          { type: 'MultiheadAttention(128, 8 heads)', params: 66048 },
          { type: 'LayerNorm',                        params: 256   },
          { type: 'FFN(128→64→128) + BN + Dropout',  params: 16960 },
        ]},
      { name: 'Encoder Block 3',     channels: 50, h:  1, w: 128, dataKey: 'layer4',
        sublayers: [
          { type: 'LayerNorm',                        params: 256   },
          { type: 'MultiheadAttention(128, 8 heads)', params: 66048 },
          { type: 'LayerNorm',                        params: 256   },
          { type: 'FFN(128→64→128) + BN + Dropout',  params: 16960 },
        ]},
      { name: 'CLS → Softmax',       channels: 10, h:  1, w:   1, dataKey: 'output',
        channelLabels: '__classes__',
        sublayers: [
          { type: 'LayerNorm',       params: 256  },
          { type: 'Linear(128→10)', params: 1290 },
          { type: 'Softmax',        params: 0    },
        ]},
    ],
    // ViT uses global attention — no local receptive fields to visualize
    connectivity: [
      null,
      null,
      null,
      null,
      null,
      null,
      null,
    ],
  },

};

export const DEFAULT_MODEL_ID = 'v1';
