const _IS_LOCAL = ['localhost', '127.0.0.1', ''].includes(window.location.hostname);
const _HF_BASE  = 'https://huggingface.co/star14ms/cnn-tutorial-web/resolve/main';
const _CDN_DATA = id => _IS_LOCAL ? `./public/data/${id}` : `${_HF_BASE}/data/${id}`;

export const DATASET_CONFIGS = {
  mnist: {
    id:          'mnist',
    label:       'MNIST',
    normMean:    0.1307,
    normStd:     0.3081,
    modelsPath:  './public/models/mnist',
    dataPath:    './public/data/mnist',
    classLabels: ['0','1','2','3','4','5','6','7','8','9'],
  },
  fashion_mnist: {
    id:          'fashion_mnist',
    label:       'Fashion-MNIST',
    normMean:    0.2860,
    normStd:     0.3530,
    modelsPath:  './public/models/fashion_mnist',
    dataPath:    './public/data/fashion_mnist',
    classLabels: ['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot'],
  },
  kuzushiji_mnist: {
    id:          'kuzushiji_mnist',
    label:       'Kuzushiji-MNIST',
    normMean:    0.1918,
    normStd:     0.3483,
    modelsPath:  './public/models/kuzushiji_mnist',
    dataPath:    './public/data/kuzushiji_mnist',
    classLabels: ['お','き','す','つ','な','は','ま','や','れ','を'],
  },
};

export const DEFAULT_DATASET_ID = 'mnist';
