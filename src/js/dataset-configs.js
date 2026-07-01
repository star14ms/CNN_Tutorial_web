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
  cifar10: {
    id:          'cifar10',
    label:       'CIFAR-10',
    normMean:    [0.4914, 0.4822, 0.4465],
    normStd:     [0.2470, 0.2435, 0.2616],
    inChannels:  3,
    imgSize:     32,
    modelsPath:  './public/models/cifar10',
    dataPath:    './public/data/cifar10',
    classLabels: ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'],
  },
  cifar100: {
    id:          'cifar100',
    label:       'CIFAR-100',
    normMean:    [0.5071, 0.4867, 0.4408],
    normStd:     [0.2675, 0.2565, 0.2761],
    inChannels:  3,
    imgSize:     32,
    modelsPath:  './public/models/cifar100',
    dataPath:    './public/data/cifar100',
    classLabels: [
      'apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle',
      'bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle',
      'chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur',
      'dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','keyboard',
      'lamp','lawn_mower','leopard','lion','lizard','lobster','man','maple_tree','motorcycle','mountain',
      'mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree','pear','pickup_truck','pine_tree',
      'plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket',
      'rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider',
      'squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor',
      'train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm',
    ],
  },
  svhn: {
    id:          'svhn',
    label:       'SVHN',
    normMean:    [0.4377, 0.4438, 0.4728],
    normStd:     [0.1980, 0.2010, 0.1970],
    inChannels:  3,
    imgSize:     32,
    modelsPath:  './public/models/svhn',
    dataPath:    './public/data/svhn',
    classLabels: ['0','1','2','3','4','5','6','7','8','9'],
  },
};

export const DEFAULT_DATASET_ID = 'mnist';
