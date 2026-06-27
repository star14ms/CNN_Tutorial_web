export const DATASET_CONFIGS = {
  mnist: {
    id:          'mnist',
    label:       'MNIST',
    normMean:    0.1307,
    normStd:     0.3081,
    modelsPath:  'public/models/mnist',
    dataPath:    'public/data/mnist',
  },
  fashion_mnist: {
    id:          'fashion_mnist',
    label:       'Fashion-MNIST',
    normMean:    0.2860,
    normStd:     0.3530,
    modelsPath:  'public/models/fashion_mnist',
    dataPath:    'public/data/fashion_mnist',
  },
  kuzushiji_mnist: {
    id:          'kuzushiji_mnist',
    label:       'Kuzushiji-MNIST',
    normMean:    0.1918,
    normStd:     0.3483,
    modelsPath:  'public/models/kuzushiji_mnist',
    dataPath:    'public/data/kuzushiji_mnist',
  },
};

export const DEFAULT_DATASET_ID = 'mnist';
