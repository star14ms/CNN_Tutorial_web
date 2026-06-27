/**
 * Model configurations for browser visualization.
 * Each config defines layer layout, ONNX model files, and receptive-field connectivity.
 *
 * Both models share the same data key scheme from inference:
 *   inputPixels, layer0, layer1, layer2, layer3, layer4, output
 */

export const MODEL_CONFIGS = {

  v1: {
    id:          'v1',
    label:       'Simple CNN',
    description: '2 Conv + MaxPool + FC(128) — 99.20% acc',
    totalParams: 421642,
    modelFiles: [
      { name: 'layer0', path: 'public/models/layer0.onnx' },
      { name: 'layer1', path: 'public/models/layer1.onnx' },
      { name: 'layer2', path: 'public/models/layer2.onnx' },
      { name: 'layer3', path: 'public/models/layer3.onnx' },
      { name: 'layer4', path: 'public/models/layer4.onnx' },
      { name: 'full',   path: 'public/models/mnist-cnn.onnx' },
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
      { name: 'FC1 + ReLU',    channels:  1, h:  1, w: 128, dataKey: 'layer4',
        sublayers: [
          { type: 'Linear(3136→128)', params: 401536 },
          { type: 'ReLU',             params: 0      },
        ]},
      { name: 'FC2 + Softmax', channels: 10, h:  1, w:  1, dataKey: 'output',
        channelLabels: ['0','1','2','3','4','5','6','7','8','9'],
        sublayers: [
          { type: 'Linear(128→10)', params: 1290 },
          { type: 'Softmax',        params: 0    },
        ]},
    ],
    // Receptive-field connectivity: how each layer's pixels relate to the previous layer
    connectivity: [
      null,
      { type: 'conv', prevLi: 0, kernel: 3, pad: 1, prevChannels:  1 },
      { type: 'pool', prevLi: 1, kernel: 2 },
      { type: 'conv', prevLi: 2, kernel: 3, pad: 1, prevChannels: 32 },
      { type: 'pool', prevLi: 3, kernel: 2 },
      { type: 'fc', prevLi: 4, mode: 'center' },               // FC1 ← MaxPool2 (center pixel per channel)
      { type: 'fc', prevLi: 5, mode: 'all', sampleStep: 1 },   // FC2 ← FC1 (all 128 neurons)
    ],
  },

  v2: {
    id:          'v2',
    label:       'Deep CNN + BatchNorm',
    description: '4 Conv + MaxPool + BatchNorm + Dropout + FC(512) - 99.61% acc',
    totalParams: 1676650,
    modelFiles: [
      { name: 'layer0',      path: 'public/models/v2_layer0.onnx' },
      { name: 'layer1_conv', path: 'public/models/v2_layer1_conv.onnx' },
      { name: 'layer1',      path: 'public/models/v2_layer1.onnx' },
      { name: 'layer2',      path: 'public/models/v2_layer2.onnx' },
      { name: 'layer3_conv', path: 'public/models/v2_layer3_conv.onnx' },
      { name: 'layer3',      path: 'public/models/v2_layer3.onnx' },
      { name: 'layer4',      path: 'public/models/v2_layer4.onnx' },
      { name: 'full',        path: 'public/models/v2_mnist-cnn.onnx' },
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
      { name: 'FC1 + ReLU',         channels:  1, h:  1, w: 512, dataKey: 'layer4',
        sublayers: [
          { type: 'Linear(3136→512)', params: 1606144 },
          { type: 'ReLU',             params: 0       },
          { type: 'Dropout(0.5)',     params: 0       },
        ]},
      { name: 'FC2 + Softmax',      channels: 10, h:  1, w:  1,  dataKey: 'output',
        channelLabels: ['0','1','2','3','4','5','6','7','8','9'],
        sublayers: [
          { type: 'Linear(512→10)', params: 5130 },
          { type: 'Softmax',        params: 0    },
        ]},
    ],
    connectivity: [
      null,
      { type: 'conv', prevLi: 0, kernel: 3, pad: 1, prevChannels:  1 },  // Conv1 ← Input
      { type: 'conv', prevLi: 1, kernel: 3, pad: 1, prevChannels: 32 },  // Conv2 ← Conv1
      { type: 'pool', prevLi: 2, kernel: 2 },                            // MaxPool1 ← Conv2
      { type: 'conv', prevLi: 3, kernel: 3, pad: 1, prevChannels: 32 },  // Conv3 ← MaxPool1
      { type: 'conv', prevLi: 4, kernel: 3, pad: 1, prevChannels: 64 },  // Conv4 ← Conv3
      { type: 'pool', prevLi: 5, kernel: 2 },                            // MaxPool2 ← Conv4
      { type: 'fc', prevLi: 6, mode: 'center' },                         // FC1 ← MaxPool2
      { type: 'fc', prevLi: 7, mode: 'all', sampleStep: 1 },             // FC2 ← FC1
    ],
  },
};

export const DEFAULT_MODEL_ID = 'v1';
