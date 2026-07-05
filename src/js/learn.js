/* CNN Learning Center — learn.js */

import { DATASET_CONFIGS } from './dataset-configs.js';
import { t, tf, getLang, onLanguageChange, initI18n } from './i18n.js';

const _datasetLabelsCache = {};
/** Fetch (and cache) the full test-label buffer for a dataset — used to pick a random index. */
async function getDatasetTestLabels(datasetId) {
  if (_datasetLabelsCache[datasetId]) return _datasetLabelsCache[datasetId];
  const cfg = DATASET_CONFIGS[datasetId];
  const res = await fetch(`${cfg.dataPath}/test_labels.bin`);
  const buf = new Uint8Array(await res.arrayBuffer());
  _datasetLabelsCache[datasetId] = buf;
  return buf;
}

/** Fetch one test image (native resolution) via an HTTP Range request. */
async function fetchDatasetImage(datasetId, index) {
  const cfg = DATASET_CONFIGS[datasetId];
  const channels = cfg.inChannels ?? 1;
  const size = cfg.imgSize ?? 28;
  const ppi = channels * size * size;
  const start = index * ppi, end = start + ppi - 1;
  const res = await fetch(`${cfg.dataPath}/test_images.bin`, {
    headers: { Range: `bytes=${start}-${end}` },
  });
  const buf = new Uint8Array(await res.arrayBuffer());
  return { pixels: buf, size, channels };
}

/** Convert a native-resolution (possibly RGB, HWC) image buffer to a size×size grayscale grid in [0,1]. */
function toGrayscaleGrid(pixels, size, channels) {
  const out = Array.from({ length: size }, () => new Array(size).fill(0));
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      if (channels === 1) {
        out[r][c] = pixels[r * size + c] / 255;
      } else {
        const idx = (r * size + c) * channels;
        out[r][c] = (0.299 * pixels[idx] + 0.587 * pixels[idx + 1] + 0.114 * pixels[idx + 2]) / 255;
      }
    }
  }
  return out;
}

/** Render a LaTeX string into el via KaTeX (falling back to plain text if KaTeX isn't loaded). */
function renderFormula(el, latex, note = '') {
  el.innerHTML = '';
  const mathSpan = document.createElement('span');
  el.appendChild(mathSpan);
  if (window.katex) {
    try { window.katex.render(latex, mathSpan, { throwOnError: false, displayMode: false }); }
    catch (e) { mathSpan.textContent = latex; }
  } else {
    mathSpan.textContent = latex;
  }
  if (note) {
    const noteSpan = document.createElement('span');
    noteSpan.style.marginLeft = '10px';
    noteSpan.style.color = 'var(--muted)';
    noteSpan.textContent = note;
    el.appendChild(noteSpan);
  }
}

const TASK_DATA = [
  {
    icon: '🏷️',
    title: { en: 'Classification', ko: '분류' },
    subtitle: { en: 'What is in this image?', ko: '이 이미지에는 무엇이 있나요?' },
    output: { en: 'Single class label + confidence score', ko: '단일 클래스 레이블 + 신뢰도 점수' },
    desc: {
      en: 'The model reads the entire image and outputs one label from a fixed set of categories. It answers the question "what is this?" — not where or how many.',
      ko: "모델은 이미지 전체를 읽고 정해진 범주 집합 중 하나의 레이블을 출력합니다. '이것이 무엇인가?'라는 질문에 답할 뿐, 위치나 개수는 다루지 않습니다.",
    },
    models: [
      { name: 'LeNet (1998)', url: 'http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf' },
      { name: 'AlexNet (2012)', url: 'https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf' },
      { name: 'VGG (2014)', url: 'https://arxiv.org/abs/1409.1556' },
      { name: 'ResNet (2015)', url: 'https://arxiv.org/abs/1512.03385' },
      { name: 'EfficientNet (2019)', url: 'https://arxiv.org/abs/1905.11946' },
      { name: 'ViT (2020)', url: 'https://arxiv.org/abs/2010.11929' },
    ],
  },
  {
    icon: '📦',
    title: { en: 'Object Detection', ko: '객체 탐지' },
    subtitle: { en: 'Where are all the objects?', ko: '객체들은 어디에 있나요?' },
    output: { en: 'Bounding boxes + class labels + confidence', ko: '바운딩 박스 + 클래스 레이블 + 신뢰도' },
    desc: {
      en: 'The model locates every object of interest in the image and draws a bounding box around each one with its class label. Multiple objects of different classes can be detected in a single forward pass.',
      ko: '모델은 이미지 내 관심 객체를 모두 찾아 각각에 클래스 레이블이 달린 바운딩 박스를 그립니다. 한 번의 순전파로 서로 다른 클래스의 여러 객체를 동시에 탐지할 수 있습니다.',
    },
    models: [
      { name: 'YOLO (2015)', url: 'https://arxiv.org/abs/1506.02640' },
      { name: 'Faster R-CNN (2015)', url: 'https://arxiv.org/abs/1506.01497' },
      { name: 'SSD (2016)', url: 'https://arxiv.org/abs/1512.02325' },
      { name: 'DETR (2020)', url: 'https://arxiv.org/abs/2005.12872' },
      { name: 'RT-DETR (2023)', url: 'https://arxiv.org/abs/2304.08069' },
    ],
  },
  {
    icon: '👤',
    title: { en: 'Facial Recognition', ko: '얼굴 인식' },
    subtitle: { en: 'Who is this person?', ko: '이 사람은 누구인가요?' },
    output: { en: 'Identity embedding vector / match score', ko: '신원 임베딩 벡터 / 일치 점수' },
    desc: {
      en: 'The model maps a face image to a compact embedding vector. Two faces are "the same person" if their embeddings are close in vector space. Verification (1:1) and identification (1:N) are the two main tasks.',
      ko: "모델은 얼굴 이미지를 작은 임베딩 벡터로 변환합니다. 두 얼굴의 임베딩이 벡터 공간에서 가까우면 '동일 인물'로 판단합니다. 본인 확인(1:1)과 신원 식별(1:N)이 대표적인 두 가지 과제입니다.",
    },
    models: [
      { name: 'DeepFace (2014)', url: 'https://research.facebook.com/publications/deepface-closing-the-gap-to-human-level-performance-in-face-verification/' },
      { name: 'FaceNet (2015)', url: 'https://arxiv.org/abs/1503.03832' },
      { name: 'ArcFace (2018)', url: 'https://arxiv.org/abs/1801.07698' },
      { name: 'InsightFace (2019)', url: 'https://github.com/deepinsight/insightface' },
      { name: 'AdaFace (2022)', url: 'https://arxiv.org/abs/2204.00964' },
    ],
  },
  {
    icon: '🎨',
    title: { en: 'Segmentation', ko: '세그멘테이션 (분할)' },
    subtitle: { en: 'Label every pixel', ko: '모든 픽셀에 레이블 지정' },
    output: { en: 'Per-pixel class mask (same size as input)', ko: '픽셀 단위 클래스 마스크 (입력과 동일한 크기)' },
    desc: {
      en: 'Instead of a single box, the model assigns a class label to every pixel. Semantic segmentation labels pixels by category; instance segmentation also distinguishes between individual objects of the same class.',
      ko: '단일 박스 대신 모델이 모든 픽셀에 클래스 레이블을 할당합니다. 시맨틱 세그멘테이션은 범주별로 픽셀에 레이블을 붙이고, 인스턴스 세그멘테이션은 같은 클래스라도 개별 객체를 구분합니다.',
    },
    models: [
      { name: 'FCN (2015)', url: 'https://arxiv.org/abs/1411.4038' },
      { name: 'U-Net (2015)', url: 'https://arxiv.org/abs/1505.04597' },
      { name: 'Mask R-CNN (2017)', url: 'https://arxiv.org/abs/1703.06870' },
      { name: 'DeepLab v3+ (2018)', url: 'https://arxiv.org/abs/1802.02611' },
      { name: 'SAM (2023)', url: 'https://arxiv.org/abs/2304.02643' },
    ],
  },
  {
    icon: '🕺',
    title: { en: 'Pose Estimation', ko: '자세 추정' },
    subtitle: { en: 'Where are the body joints?', ko: '신체 관절은 어디에 있나요?' },
    output: { en: 'Keypoint coordinates (joints/skeleton) per person', ko: '사람별 키포인트 좌표 (관절/스켈레톤)' },
    desc: {
      en: 'The model predicts the pixel locations of anatomical keypoints (wrists, elbows, knees, etc.) and connects them into a skeleton. Used for motion capture, fitness apps, and sports analytics.',
      ko: '모델은 손목, 팔꿈치, 무릎 등 신체 키포인트의 픽셀 위치를 예측하고 이를 연결해 스켈레톤을 구성합니다. 모션 캡처, 피트니스 앱, 스포츠 분석 등에 활용됩니다.',
    },
    models: [
      { name: 'OpenPose (2018)', url: 'https://arxiv.org/abs/1812.08008' },
      { name: 'HRNet (2019)', url: 'https://arxiv.org/abs/1902.09212' },
      { name: 'AlphaPose (2016)', url: 'https://arxiv.org/abs/1612.00137' },
    ],
  },
  {
    icon: '🗺️',
    title: { en: 'Depth Estimation', ko: '깊이 추정' },
    subtitle: { en: 'How far away is each pixel?', ko: '각 픽셀은 얼마나 멀리 있나요?' },
    output: { en: 'Per-pixel depth map (distance from camera)', ko: '픽셀 단위 깊이 맵 (카메라로부터의 거리)' },
    desc: {
      en: 'The model predicts a distance value for every pixel from a single 2D image, effectively recovering 3D structure without a depth sensor. Used in AR, robotics, and autonomous driving.',
      ko: '모델은 단일 2D 이미지에서 모든 픽셀의 거리 값을 예측하여, 깊이 센서 없이도 3D 구조를 복원합니다. AR, 로보틱스, 자율주행 등에 사용됩니다.',
    },
    models: [
      { name: 'MiDaS (2019)', url: 'https://arxiv.org/abs/1907.01341' },
      { name: 'DPT (2021)', url: 'https://arxiv.org/abs/2103.13413' },
      { name: 'Depth Anything (2024)', url: 'https://arxiv.org/abs/2401.10891' },
    ],
  },
  {
    icon: '📝',
    title: { en: 'Image Captioning', ko: '이미지 캡셔닝' },
    subtitle: { en: 'Describe this image in words', ko: '이 이미지를 말로 설명하기' },
    output: { en: 'Natural-language sentence describing the image', ko: '이미지를 설명하는 자연어 문장' },
    desc: {
      en: 'The model combines a vision encoder with a language decoder to generate a free-text description of the scene, bridging computer vision and NLP.',
      ko: '모델은 비전 인코더와 언어 디코더를 결합해 장면을 설명하는 자유 형식의 텍스트를 생성하며, 컴퓨터 비전과 자연어 처리를 연결합니다.',
    },
    models: [
      { name: 'Show and Tell (2014)', url: 'https://arxiv.org/abs/1411.4555' },
      { name: 'Show, Attend and Tell (2015)', url: 'https://arxiv.org/abs/1502.03044' },
      { name: 'BLIP (2022)', url: 'https://arxiv.org/abs/2201.12086' },
    ],
  },
  {
    icon: '🔤',
    title: { en: 'OCR / Text Recognition', ko: 'OCR / 문자 인식' },
    subtitle: { en: 'What text is written here?', ko: '여기에 어떤 텍스트가 쓰여 있나요?' },
    output: { en: 'Recognized character/word strings + positions', ko: '인식된 문자/단어 문자열 + 위치' },
    desc: {
      en: 'Optical Character Recognition locates and transcribes printed or handwritten text within an image, converting pixels into machine-readable strings.',
      ko: '광학 문자 인식(OCR)은 이미지 속 인쇄되거나 손으로 쓴 텍스트를 찾아 옮겨 적어, 픽셀을 기계가 읽을 수 있는 문자열로 변환합니다.',
    },
    models: [
      { name: 'CRNN (2015)', url: 'https://arxiv.org/abs/1507.05717' },
      { name: 'Tesseract OCR', url: 'https://github.com/tesseract-ocr/tesseract' },
      { name: 'TrOCR (2021)', url: 'https://arxiv.org/abs/2109.10282' },
    ],
  },
];

const KERNELS = {
  // 8 directional (compass) edge kernels — each responds strongly to an edge/gradient
  // oriented toward that side, unlike a single high center pixel.
  edgeTop:    [[ 1, 1, 1],[ 0, 0, 0],[-1,-1,-1]],
  edgeBottom: [[-1,-1,-1],[ 0, 0, 0],[ 1, 1, 1]],
  edgeLeft:   [[ 1, 0,-1],[ 1, 0,-1],[ 1, 0,-1]],
  edgeRight:  [[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]],
  edgeTL:     [[ 1, 1, 0],[ 1, 0,-1],[ 0,-1,-1]],
  edgeTR:     [[ 0, 1, 1],[-1, 0, 1],[-1,-1, 0]],
  edgeBL:     [[ 0,-1,-1],[ 1, 0,-1],[ 1, 1, 0]],
  edgeBR:     [[-1,-1, 0],[-1, 0, 1],[ 0, 1, 1]],
  blur:       [[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]],
  sharpen:    [[0,-1,0],[-1,5,-1],[0,-1,0]],
  custom:     [[0,0,0],[0,1,0],[0,0,0]],
};

const IMG_SIZES = [28, 32, 64, 128, 224, 256];
const HIDDEN_SIZES = [32, 128, 256, 512, 1024];

/* ── helpers ─────────────────────────────────────────────── */
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function fmt(n) { return n.toLocaleString(); }

/* ── open / close ─────────────────────────────────────────── */
function initOverlay() {
  document.getElementById('learn-open-btn').addEventListener('click', () => {
    document.getElementById('learn-overlay').classList.add('open');
  });
  document.getElementById('learn-close').addEventListener('click', () => {
    document.getElementById('learn-overlay').classList.remove('open');
  });
}

/* ── sidebar active tracking ──────────────────────────────── */
function initSidebar() {
  const content = document.getElementById('learn-content');
  const navItems = document.querySelectorAll('.learn-nav-item');

  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const sec = document.getElementById(item.dataset.section);
      if (sec) sec.scrollIntoView({ behavior: 'smooth' });
    });
  });

  const sections = [...document.querySelectorAll('.learn-section')];
  content.addEventListener('scroll', () => {
    const scrollTop = content.scrollTop;
    let active = sections[0].id;
    for (const s of sections) {
      if (s.offsetTop - content.offsetTop - 60 <= scrollTop) active = s.id;
    }
    navItems.forEach(item => {
      item.classList.toggle('active', item.dataset.section === active);
    });
  });
}

/* ── task cards ───────────────────────────────────────────── */
/** Cards are laid out 2 per row (.task-card-grid uses 2 columns). Expanded cards always get
 *  an explicit max-height set to their real content height (never the generic 700px CSS cap) —
 *  otherwise collapsing from a 700px baseline down to 0 spends most of the transition duration
 *  on invisible change (700→content height) before any visible collapse happens, making it look
 *  like the card doesn't fully close. When two cards in a row are both expanded, they're matched
 *  to the taller one's height. */
function syncTaskCardRowHeights() {
  const grid = document.getElementById('task-cards');
  if (!grid) return;
  const cards = [...grid.children];
  const cols = 2;
  for (let i = 0; i < cards.length; i += cols) {
    const rowCards = cards.slice(i, i + cols);
    // Always clear any stale inline override on collapsed cards in this row —
    // otherwise a leftover max-height from a previous expanded state keeps them stuck open.
    rowCards.filter(c => !c.classList.contains('expanded'))
      .forEach(c => { c.querySelector('.task-card-body').style.maxHeight = ''; });
    const rowExpandedBodies = rowCards
      .filter(c => c.classList.contains('expanded'))
      .map(c => c.querySelector('.task-card-body'));
    if (rowExpandedBodies.length === 0) continue;
    // Measure natural content height with clipping disabled — reading scrollHeight
    // while max-height is still 0 (e.g. right after the class toggle, mid-transition)
    // would otherwise report 0 and permanently bake that in as the new inline max-height.
    const heights = rowExpandedBodies.map(b => {
      const prevInline = b.style.maxHeight;
      b.style.maxHeight = 'none';
      const h = b.scrollHeight;
      b.style.maxHeight = prevInline;
      return h;
    });
    const maxH = Math.max(...heights);
    rowExpandedBodies.forEach(b => { b.style.maxHeight = `${maxH}px`; });
  }
}

function renderTaskCards() {
  const grid = document.getElementById('task-cards');
  const lang = getLang();
  // Remember which cards were expanded so re-rendering on language change doesn't collapse them.
  const expanded = new Set([...grid.children].filter(c => c.classList.contains('expanded')).map(c => c.dataset.idx));
  grid.innerHTML = '';
  TASK_DATA.forEach((task, idx) => {
    const card = document.createElement('div');
    card.className = 'task-card' + (expanded.has(String(idx)) ? ' expanded' : '');
    card.dataset.idx = idx;
    card.innerHTML = `
      <div class="task-card-head">
        <span class="task-card-icon">${task.icon}</span>
        <div><div class="task-card-title">${task.title[lang]}</div><div class="task-card-subtitle">${task.subtitle[lang]}</div></div>
        <span class="task-card-chevron">▼</span>
      </div>
      <div class="task-card-body">
        <p><strong style="color:var(--yellow)">${t('task.outputLabel')}</strong> ${task.output[lang]}</p>
        <p>${task.desc[lang]}</p>
        <div class="task-models">${task.models.map(m => `<a class="task-model-tag" href="${m.url}" target="_blank" rel="noopener noreferrer" title="Open paper / source">${m.name}</a>`).join('')}</div>
      </div>`;
    card.addEventListener('click', e => {
      if (e.target.closest('a')) return;
      card.classList.toggle('expanded');
      syncTaskCardRowHeights();
    });
    grid.appendChild(card);
  });
  syncTaskCardRowHeights();
}

function initTaskCards() {
  renderTaskCards();
  onLanguageChange(renderTaskCards);
}

/* ── idea cards (accordion) ───────────────────────────────── */
function initIdeaCards() {
  document.querySelectorAll('.idea-card-head').forEach(head => {
    head.addEventListener('click', () => {
      const card = head.parentElement;
      const body = card.querySelector('.idea-card-body');
      card.classList.toggle('open');
      if (card.classList.contains('open')) {
        // Set an explicit height matching real content instead of relying on the
        // generic CSS cap — otherwise the collapse transition spends most of its
        // duration on an invisible change before any visible closing happens.
        const prevInline = body.style.maxHeight;
        body.style.maxHeight = 'none';
        const h = body.scrollHeight;
        body.style.maxHeight = prevInline;
        body.style.maxHeight = `${h}px`;
      } else {
        body.style.maxHeight = '';
      }
    });
  });
}

/* ── param counter widget ─────────────────────────────────── */
function initParamCounter() {
  const sizeSlider = document.getElementById('img-size-slider');
  const hiddenSlider = document.getElementById('hidden-slider');
  const sizeLabel = document.getElementById('img-size-label');
  const hiddenLabel = document.getElementById('hidden-label');
  const counter = document.getElementById('param-counter');

  function update() {
    const sz = IMG_SIZES[+sizeSlider.value];
    const h = HIDDEN_SIZES[+hiddenSlider.value];
    const channels = sz >= 64 ? 3 : 1;
    const params = sz * sz * channels * h;
    sizeLabel.textContent = `${sz} × ${sz}` + (channels === 3 ? ' (RGB)' : '');
    hiddenLabel.textContent = h;
    counter.textContent = fmt(params);
  }
  sizeSlider.addEventListener('input', update);
  hiddenSlider.addEventListener('input', update);
  update();
}

/* ── MLP diagram ──────────────────────────────────────────── */
function initMLPWidget() {
  const canvas = document.getElementById('mlp-canvas');
  const ctx = canvas.getContext('2d');
  const layers = [4, 4, 3];
  const W = canvas.width, H = canvas.height;
  const cols = layers.length;
  const nodeR = 14;
  let hovered = null; // {layer, idx}

  function nodePos(layer, idx) {
    const x = (layer + 1) * W / (cols + 1);
    const count = layers[layer];
    const y = (idx + 1) * H / (count + 1);
    return { x, y };
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);

    // draw connections
    for (let l = 0; l < cols - 1; l++) {
      for (let i = 0; i < layers[l]; i++) {
        for (let j = 0; j < layers[l + 1]; j++) {
          const highlight = hovered && (
            (hovered.layer === l && hovered.idx === i) ||
            (hovered.layer === l + 1 && hovered.idx === j)
          );
          const p1 = nodePos(l, i), p2 = nodePos(l + 1, j);
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.strokeStyle = highlight ? 'rgba(240,192,64,0.85)' : 'rgba(255,255,255,0.08)';
          ctx.lineWidth = highlight ? 1.5 : 0.8;
          ctx.stroke();
        }
      }
    }

    // draw nodes
    const layerLabels = ['Input', 'Hidden', 'Output'];
    for (let l = 0; l < cols; l++) {
      for (let i = 0; i < layers[l]; i++) {
        const { x, y } = nodePos(l, i);
        const isHov = hovered && hovered.layer === l && hovered.idx === i;
        ctx.beginPath();
        ctx.arc(x, y, nodeR, 0, Math.PI * 2);
        ctx.fillStyle = isHov ? '#f0c040' : '#2a3a5a';
        ctx.fill();
        ctx.strokeStyle = isHov ? '#f0c040' : '#4a6a9a';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
      ctx.fillStyle = 'rgba(110,118,128,0.8)';
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'center';
      const lx = (l + 1) * W / (cols + 1);
      ctx.fillText(layerLabels[l], lx, H - 4);
    }
  }

  canvas.addEventListener('mousemove', e => {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (W / rect.width);
    const my = (e.clientY - rect.top) * (H / rect.height);
    let found = null;
    outer: for (let l = 0; l < cols; l++) {
      for (let i = 0; i < layers[l]; i++) {
        const { x, y } = nodePos(l, i);
        if ((mx - x) ** 2 + (my - y) ** 2 < nodeR ** 2) { found = { layer: l, idx: i }; break outer; }
      }
    }
    hovered = found;
    draw();
  });
  canvas.addEventListener('mouseleave', () => { hovered = null; draw(); });
  draw();
}

/* ── conv widget ──────────────────────────────────────────── */
const CONV_PATTERNS = {
  checkerboard: GRID => Array.from({ length: GRID }, (_, r) => Array.from({ length: GRID }, (_, c) => (r + c) % 2 === 0 ? 0.9 : 0.1)),
  gradient: GRID => Array.from({ length: GRID }, (_, r) => Array.from({ length: GRID }, (_, c) => (r + c) / (2 * (GRID - 1)))),
  cross: GRID => Array.from({ length: GRID }, (_, r) => Array.from({ length: GRID }, (_, c) => {
    const mid = (GRID - 1) / 2;
    return (Math.abs(r - mid) < 1 || Math.abs(c - mid) < 1) ? 0.95 : 0.08;
  })),
  noise: GRID => Array.from({ length: GRID }, () => Array.from({ length: GRID }, () => Math.random())),
};

const DATASET_LABELS = {
  mnist: 'MNIST', fashion_mnist: 'Fashion-MNIST', kuzushiji_mnist: 'Kuzushiji-MNIST',
  cifar10: 'CIFAR-10', cifar100: 'CIFAR-100', svhn: 'SVHN',
};

function initConvWidget() {
  const DEMO_GRID = 8;
  let GRID = DEMO_GRID;
  const inputCanvas = document.getElementById('conv-input');
  const kernelCanvas = document.getElementById('conv-kernel');
  const outputCanvas = document.getElementById('conv-output');
  const strideEl = document.getElementById('conv-stride');
  const padEl = document.getElementById('conv-pad');
  const formulaEl = document.getElementById('conv-formula');
  const presetEl = document.getElementById('conv-input-preset');
  const statusEl = document.getElementById('conv-input-status');

  const iCtx = inputCanvas.getContext('2d');
  const kCtx = kernelCanvas.getContext('2d');
  const oCtx = outputCanvas.getContext('2d');

  let CELL_I = inputCanvas.width / GRID;
  const CELL_K = kernelCanvas.width / 3;

  let inputGrid = CONV_PATTERNS.checkerboard(GRID);
  let kernel = KERNELS.edgeTop.map(r => [...r]);
  let animPos = null; // {r, c}
  let animTimer = null;
  let playing = false;
  let currentKernelName = 'edgeTop';
  // Tracks which input source is currently active, so the status text (and any
  // re-render triggered by a language switch) always reflects reality instead of
  // being hardcoded back to the initial pattern.
  let currentSource = { type: 'pattern', name: 'checkerboard' };

  function updateStatusText() {
    if (currentSource.type === 'dataset') {
      statusEl.textContent = tf('conv.statusLoaded', {
        name: DATASET_LABELS[currentSource.datasetId], idx: currentSource.idx, label: currentSource.label,
      });
    } else {
      statusEl.textContent = tf('conv.statusPattern', { name: t(`sec3.pattern${currentSource.name[0].toUpperCase()}${currentSource.name.slice(1)}`) });
    }
  }

  function setGrid(newGrid, data) {
    GRID = newGrid;
    CELL_I = inputCanvas.width / GRID;
    inputGrid = data;
    animPos = null;
    redraw();
  }

  function getStride() { return +strideEl.value; }
  function getPad() { return +padEl.value; }

  function computeOutput() {
    const s = getStride(), p = getPad();
    const outSize = Math.floor((GRID + 2 * p - 3) / s) + 1;
    const out = Array.from({ length: outSize }, () => new Array(outSize).fill(0));
    for (let r = 0; r < outSize; r++) {
      for (let c = 0; c < outSize; c++) {
        let val = 0;
        for (let m = 0; m < 3; m++) {
          for (let n = 0; n < 3; n++) {
            const ir = r * s + m - p, ic = c * s + n - p;
            const inp = (ir >= 0 && ir < GRID && ic >= 0 && ic < GRID) ? inputGrid[ir][ic] : 0;
            val += kernel[m][n] * inp;
          }
        }
        out[r][c] = val;
      }
    }
    return out;
  }

  // With padding, the canvas shows GRID + 2*p cells per side: a border of
  // synthetic "0" padding cells around the real input, so the effect of
  // padding is visible instead of purely implicit in the math.
  function getTotalCells() { return GRID + 2 * getPad(); }

  function drawInput() {
    const s = getStride(), p = getPad();
    const total = getTotalCells();
    CELL_I = inputCanvas.width / total;
    // Fill the whole canvas first — with non-integer cell sizes (e.g. 192/28 ≈ 6.86px),
    // rounding leaves thin gaps between cells that would otherwise still show pixels
    // from whatever was previously drawn (a different pattern/image at a different
    // resolution), looking like the two inputs are merged together.
    iCtx.fillStyle = '#0a0a14';
    iCtx.fillRect(0, 0, inputCanvas.width, inputCanvas.height);
    // padding border cells (drawn first, underneath the real cells)
    if (p > 0) {
      iCtx.save();
      iCtx.fillStyle = 'rgba(233,69,96,0.12)';
      iCtx.strokeStyle = 'rgba(233,69,96,0.5)';
      iCtx.setLineDash([3, 2]);
      for (let tr = 0; tr < total; tr++) {
        for (let tc = 0; tc < total; tc++) {
          const isPadCell = tr < p || tr >= GRID + p || tc < p || tc >= GRID + p;
          if (!isPadCell) continue;
          iCtx.fillRect(tc * CELL_I + 1, tr * CELL_I + 1, CELL_I - 2, CELL_I - 2);
          iCtx.strokeRect(tc * CELL_I + 1.5, tr * CELL_I + 1.5, CELL_I - 3, CELL_I - 3);
        }
      }
      iCtx.restore();
    }
    for (let r = 0; r < GRID; r++) {
      for (let c = 0; c < GRID; c++) {
        const bright = Math.round(clamp(inputGrid[r][c], 0, 1) * 255);
        iCtx.fillStyle = `rgb(${bright},${bright},${bright})`;
        iCtx.fillRect((c + p) * CELL_I + 1, (r + p) * CELL_I + 1, CELL_I - 2, CELL_I - 2);
      }
    }
    iCtx.strokeStyle = total > 14 ? 'rgba(42,42,74,0.35)' : '#2a2a4a';
    iCtx.lineWidth = 1;
    for (let i = 0; i <= total; i++) {
      iCtx.beginPath(); iCtx.moveTo(i * CELL_I, 0); iCtx.lineTo(i * CELL_I, inputCanvas.height); iCtx.stroke();
      iCtx.beginPath(); iCtx.moveTo(0, i * CELL_I); iCtx.lineTo(inputCanvas.width, i * CELL_I); iCtx.stroke();
    }
    // A thicker line separates the real input from the zero-padding border.
    if (p > 0) {
      iCtx.strokeStyle = 'rgba(233,69,96,0.6)';
      iCtx.lineWidth = 2;
      iCtx.strokeRect(p * CELL_I, p * CELL_I, GRID * CELL_I, GRID * CELL_I);
    }
    if (animPos) {
      const { r, c } = animPos;
      // The kernel window starts at (r*s, c*s) in this padded coordinate space —
      // no extra "-p" offset needed since padding cells already occupy the border.
      const x = c * s * CELL_I, y = r * s * CELL_I;
      iCtx.strokeStyle = '#f0c040';
      iCtx.lineWidth = 2.5;
      iCtx.strokeRect(x + 1, y + 1, CELL_I * 3 - 2, CELL_I * 3 - 2);
    }
  }

  function drawKernel() {
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        const v = kernel[r][c];
        const norm = clamp((v + 5) / 10, 0, 1);
        const brightness = Math.round(norm * 200);
        kCtx.fillStyle = `rgb(${brightness},${brightness},${brightness})`;
        kCtx.fillRect(c * CELL_K + 1, r * CELL_K + 1, CELL_K - 2, CELL_K - 2);
        kCtx.fillStyle = norm > 0.5 ? '#111' : '#aaa';
        kCtx.font = `bold ${CELL_K * 0.35}px monospace`;
        kCtx.textAlign = 'center';
        kCtx.textBaseline = 'middle';
        kCtx.fillText(v % 1 === 0 ? v : v.toFixed(2), c * CELL_K + CELL_K / 2, r * CELL_K + CELL_K / 2);
      }
    }
    kCtx.strokeStyle = '#2a2a4a';
    kCtx.lineWidth = 1;
    for (let i = 0; i <= 3; i++) {
      kCtx.beginPath(); kCtx.moveTo(i * CELL_K, 0); kCtx.lineTo(i * CELL_K, kernelCanvas.height); kCtx.stroke();
      kCtx.beginPath(); kCtx.moveTo(0, i * CELL_K); kCtx.lineTo(kernelCanvas.width, i * CELL_K); kCtx.stroke();
    }
  }

  function drawOutput() {
    const out = computeOutput();
    const outSize = out.length;
    const CELL_O = outputCanvas.width / outSize;
    let minV = Infinity, maxV = -Infinity;
    out.forEach(row => row.forEach(v => { minV = Math.min(minV, v); maxV = Math.max(maxV, v); }));
    const range = maxV - minV || 1;
    oCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    for (let r = 0; r < outSize; r++) {
      for (let c = 0; c < outSize; c++) {
        const norm = (out[r][c] - minV) / range;
        const bright = Math.round(norm * 255);
        oCtx.fillStyle = `rgb(${bright},${bright},${bright})`;
        oCtx.fillRect(c * CELL_O + 1, r * CELL_O + 1, CELL_O - 2, CELL_O - 2);
        if (animPos && animPos.r === r && animPos.c === c) {
          oCtx.strokeStyle = '#e94560';
          oCtx.lineWidth = 2.5;
          oCtx.strokeRect(c * CELL_O + 1, r * CELL_O + 1, CELL_O - 2, CELL_O - 2);
        }
      }
    }
    oCtx.strokeStyle = '#2a2a4a';
    oCtx.lineWidth = 1;
    for (let i = 0; i <= outSize; i++) {
      oCtx.beginPath(); oCtx.moveTo(i * CELL_O, 0); oCtx.lineTo(i * CELL_O, outputCanvas.height); oCtx.stroke();
      oCtx.beginPath(); oCtx.moveTo(0, i * CELL_O); oCtx.lineTo(outputCanvas.width, i * CELL_O); oCtx.stroke();
    }
  }

  function updateFormula() {
    const s = getStride(), p = getPad();
    const outSize = Math.floor((GRID + 2 * p - 3) / s) + 1;
    const latex = `\\text{output}[i,j] = \\sum_{m,n} \\text{kernel}[m,n] \\cdot \\text{input}[i \\cdot ${s} + m - ${p},\\ j \\cdot ${s} + n - ${p}]`;
    renderFormula(formulaEl, latex, `(output size: ${outSize}×${outSize})`);
  }

  function redraw() { drawInput(); drawKernel(); drawOutput(); updateFormula(); }

  // input click — cycle brightness in steps of 0.2
  inputCanvas.addEventListener('click', e => {
    const rect = inputCanvas.getBoundingClientRect();
    const p = getPad(), total = getTotalCells();
    // Coordinates are in the padded visual grid — subtract p to land on the real
    // (editable) cells; clicks that land on the zero-padding border are ignored.
    const c = Math.floor((e.clientX - rect.left) / rect.width * total) - p;
    const r = Math.floor((e.clientY - rect.top) / rect.height * total) - p;
    if (r >= 0 && r < GRID && c >= 0 && c < GRID) {
      inputGrid[r][c] = ((Math.round(inputGrid[r][c] * 5) + 1) % 6) / 5;
      presetEl.value = '';
      currentSource = { type: 'pattern', name: 'custom' };
      updateStatusText();
      redraw();
    }
  });

  // synthetic pattern selector
  presetEl.addEventListener('change', () => {
    const val = presetEl.value;
    if (CONV_PATTERNS[val]) {
      setGrid(DEMO_GRID, CONV_PATTERNS[val](DEMO_GRID));
      currentSource = { type: 'pattern', name: val };
      updateStatusText();
    }
  });

  // random-sample-from-dataset buttons
  document.querySelectorAll('#conv-dataset-btns [data-dataset]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const datasetId = btn.dataset.dataset;
      btn.disabled = true;
      const prevSource = currentSource;
      statusEl.textContent = tf('conv.statusLoading', { name: DATASET_LABELS[datasetId] });
      try {
        const cfg = DATASET_CONFIGS[datasetId];
        const labels = await getDatasetTestLabels(datasetId);
        const idx = Math.floor(Math.random() * labels.length);
        const { pixels, size, channels } = await fetchDatasetImage(datasetId, idx);
        const grid = toGrayscaleGrid(pixels, size, channels);
        setGrid(size, grid);
        // A dataset image is now the active input — deselect the synthetic-pattern
        // dropdown (back to its placeholder) so it can't be mistaken for the current source.
        presetEl.value = '';
        const className = cfg.classLabels ? cfg.classLabels[labels[idx]] : labels[idx];
        currentSource = { type: 'dataset', datasetId, idx, label: className };
        updateStatusText();
      } catch (err) {
        console.error(`Failed to load ${datasetId} sample:`, err);
        currentSource = prevSource;
        updateStatusText();
      } finally {
        btn.disabled = false;
      }
    });
  });

  // kernel preset buttons
  document.querySelectorAll('[data-kernel]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-kernel]').forEach(b => b.classList.remove('wactive'));
      btn.classList.add('wactive');
      currentKernelName = btn.dataset.kernel;
      kernel = KERNELS[currentKernelName].map(r => [...r]);
      redraw();
    });
  });

  // kernel click to edit
  kernelCanvas.addEventListener('click', e => {
    const rect = kernelCanvas.getBoundingClientRect();
    const c = Math.floor((e.clientX - rect.left) / rect.width * 3);
    const r = Math.floor((e.clientY - rect.top) / rect.height * 3);
    if (r >= 0 && r < 3 && c >= 0 && c < 3) {
      const cur = kernel[r][c];
      const next = cur >= 5 ? -5 : cur + 1;
      kernel[r][c] = next;
      currentKernelName = 'custom';
      document.querySelectorAll('[data-kernel]').forEach(b => b.classList.toggle('wactive', b.dataset.kernel === 'custom'));
      redraw();
    }
  });

  strideEl.addEventListener('change', redraw);
  padEl.addEventListener('change', redraw);

  // animation
  function getPositions() {
    const s = getStride(), p = getPad();
    const outSize = Math.floor((GRID + 2 * p - 3) / s) + 1;
    const positions = [];
    for (let r = 0; r < outSize; r++) for (let c = 0; c < outSize; c++) positions.push({ r, c });
    return positions;
  }
  let animIdx = 0;
  function stepAnim() {
    const positions = getPositions();
    if (animIdx >= positions.length) {
      // Animation reached the end — stop the interval. Forgetting this left the
      // timer running forever in the background (with `playing` already false),
      // so a later "Play" click would spawn a second interval on top of it and
      // "Pause" could only ever cancel the newest one, making the animation look
      // stuck/unpausable.
      clearInterval(animTimer);
      animIdx = 0; animPos = null; redraw(); playing = false;
      document.getElementById('conv-play').textContent = t('sec3.play');
      return;
    }
    animPos = positions[animIdx++];
    redraw();
  }
  document.getElementById('conv-step').addEventListener('click', () => { clearInterval(animTimer); playing = false; document.getElementById('conv-play').textContent = t('sec3.play'); stepAnim(); });
  document.getElementById('conv-play').addEventListener('click', () => {
    if (playing) { clearInterval(animTimer); playing = false; document.getElementById('conv-play').textContent = t('sec3.play'); return; }
    playing = true; document.getElementById('conv-play').textContent = t('sec3.pause');
    animTimer = setInterval(stepAnim, 220);
  });
  document.getElementById('conv-reset').addEventListener('click', () => {
    clearInterval(animTimer); playing = false; animIdx = 0; animPos = null;
    document.getElementById('conv-play').textContent = t('sec3.play'); redraw();
  });

  onLanguageChange(updateStatusText);
  updateStatusText();
  redraw();
}

/* ── activation (ReLU) widget ─────────────────────────────── */
function initReLUWidget() {
  const canvas = document.getElementById('relu-canvas');
  const ctx = canvas.getContext('2d');
  const xValEl = document.getElementById('relu-x-val');
  const yValEl = document.getElementById('relu-y-val');
  const formulaEl = document.getElementById('relu-formula');
  let currentFn = 'relu';
  let dragX = null; // in graph coords [-4, 4]

  const fns = {
    relu:    { fn: x => Math.max(0, x), label: 'f(x) = \\max(0, x)' },
    sigmoid: { fn: x => 1 / (1 + Math.exp(-x)), label: 'f(x) = \\dfrac{1}{1 + e^{-x}}' },
    tanh:    { fn: x => Math.tanh(x), label: 'f(x) = \\tanh(x)' },
  };

  const W = canvas.width, H = canvas.height;
  const PAD = 30;

  function graphToScreen(x, y) {
    return { sx: PAD + (x + 4) / 8 * (W - 2 * PAD), sy: H - PAD - (y + 1.5) / 3 * (H - 2 * PAD) };
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    // axes
    const origin = graphToScreen(0, 0);
    ctx.strokeStyle = '#2a2a4a';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PAD, origin.sy); ctx.lineTo(W - PAD, origin.sy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(origin.sx, PAD); ctx.lineTo(origin.sx, H - PAD); ctx.stroke();
    // axis labels
    ctx.fillStyle = '#6e7680'; ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
    for (let v = -3; v <= 3; v += 1) {
      if (v === 0) continue;
      const { sx } = graphToScreen(v, 0); ctx.fillText(v, sx, origin.sy + 14);
    }
    // curve
    const { fn } = fns[currentFn];
    ctx.beginPath();
    ctx.strokeStyle = '#f0c040'; ctx.lineWidth = 2;
    for (let px = PAD; px <= W - PAD; px++) {
      const x = -4 + (px - PAD) / (W - 2 * PAD) * 8;
      const { sy } = graphToScreen(x, fn(x));
      px === PAD ? ctx.moveTo(px, sy) : ctx.lineTo(px, sy);
    }
    ctx.stroke();
    // drag point
    if (dragX !== null) {
      const y = fn(dragX);
      const { sx, sy } = graphToScreen(dragX, y);
      ctx.beginPath(); ctx.arc(sx, sy, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#e94560'; ctx.fill();
      xValEl.textContent = dragX.toFixed(2);
      yValEl.textContent = y.toFixed(4);
    }
  }

  function handlePointer(e) {
    const rect = canvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (W / rect.width);
    dragX = clamp(-4 + (px - PAD) / (W - 2 * PAD) * 8, -4, 4);
    draw();
  }

  canvas.addEventListener('mousedown', e => { canvas.addEventListener('mousemove', handlePointer); handlePointer(e); });
  window.addEventListener('mouseup', () => canvas.removeEventListener('mousemove', handlePointer));
  canvas.addEventListener('click', handlePointer);

  document.querySelectorAll('[data-fn]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-fn]').forEach(b => b.classList.remove('wactive'));
      btn.classList.add('wactive');
      currentFn = btn.dataset.fn;
      renderFormula(formulaEl, fns[currentFn].label);
      draw();
    });
  });

  renderFormula(formulaEl, fns[currentFn].label);
  dragX = 1.5;
  draw();
}

/* ── pooling widget ───────────────────────────────────────── */
function initPoolWidget() {
  const inputCanvas = document.getElementById('pool-input');
  const outputCanvas = document.getElementById('pool-output');
  const iCtx = inputCanvas.getContext('2d');
  const oCtx = outputCanvas.getContext('2d');
  const SIZE = 4;
  const CELL_I = inputCanvas.width / SIZE;
  const CELL_O = outputCanvas.width / (SIZE / 2);
  let poolMode = 'max';
  let data = Array.from({ length: SIZE }, () => Array.from({ length: SIZE }, () => Math.floor(Math.random() * 10)));

  function randomize() {
    data = Array.from({ length: SIZE }, () => Array.from({ length: SIZE }, () => Math.floor(Math.random() * 10)));
    draw();
  }

  function computePool() {
    const out = [], maxIdx = [];
    for (let r = 0; r < SIZE / 2; r++) {
      out.push([]); maxIdx.push([]);
      for (let c = 0; c < SIZE / 2; c++) {
        const patch = [
          data[r * 2][c * 2], data[r * 2][c * 2 + 1],
          data[r * 2 + 1][c * 2], data[r * 2 + 1][c * 2 + 1],
        ];
        if (poolMode === 'max') {
          const m = Math.max(...patch);
          out[r].push(m);
          const mi = patch.indexOf(m);
          maxIdx[r].push({ dr: Math.floor(mi / 2), dc: mi % 2 });
        } else {
          out[r].push(+(patch.reduce((a, b) => a + b, 0) / 4).toFixed(1));
          maxIdx[r].push(null);
        }
      }
    }
    return { out, maxIdx };
  }

  function draw() {
    const { out, maxIdx } = computePool();
    // input
    for (let r = 0; r < SIZE; r++) {
      for (let c = 0; c < SIZE; c++) {
        const v = data[r][c] / 9;
        iCtx.fillStyle = `rgb(${Math.round(v * 180)},${Math.round(v * 200)},${Math.round(v * 255)})`;
        iCtx.fillRect(c * CELL_I + 1, r * CELL_I + 1, CELL_I - 2, CELL_I - 2);
        iCtx.fillStyle = v > 0.5 ? '#111' : '#ddd';
        iCtx.font = `bold ${CELL_I * 0.45}px monospace`;
        iCtx.textAlign = 'center'; iCtx.textBaseline = 'middle';
        iCtx.fillText(data[r][c], c * CELL_I + CELL_I / 2, r * CELL_I + CELL_I / 2);
      }
    }
    // highlight max cells
    for (let r = 0; r < SIZE / 2; r++) {
      for (let c = 0; c < SIZE / 2; c++) {
        if (maxIdx[r][c] && poolMode === 'max') {
          const { dr, dc } = maxIdx[r][c];
          const ir = r * 2 + dr, ic = c * 2 + dc;
          iCtx.strokeStyle = '#e94560'; iCtx.lineWidth = 2;
          iCtx.strokeRect(ic * CELL_I + 1, ir * CELL_I + 1, CELL_I - 2, CELL_I - 2);
        }
        // pool windows
        iCtx.strokeStyle = '#f0c040'; iCtx.lineWidth = 1.5;
        iCtx.strokeRect(c * 2 * CELL_I + 0.5, r * 2 * CELL_I + 0.5, CELL_I * 2 - 1, CELL_I * 2 - 1);
      }
    }
    // output
    oCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    for (let r = 0; r < SIZE / 2; r++) {
      for (let c = 0; c < SIZE / 2; c++) {
        const v = out[r][c] / 9;
        oCtx.fillStyle = `rgb(${Math.round(v * 180)},${Math.round(v * 200)},${Math.round(v * 255)})`;
        oCtx.fillRect(c * CELL_O + 1, r * CELL_O + 1, CELL_O - 2, CELL_O - 2);
        oCtx.fillStyle = v > 0.5 ? '#111' : '#ddd';
        oCtx.font = `bold ${CELL_O * 0.45}px monospace`;
        oCtx.textAlign = 'center'; oCtx.textBaseline = 'middle';
        oCtx.fillText(out[r][c], c * CELL_O + CELL_O / 2, r * CELL_O + CELL_O / 2);
      }
    }
    oCtx.strokeStyle = '#2a2a4a'; oCtx.lineWidth = 1;
    for (let i = 0; i <= SIZE / 2; i++) {
      oCtx.beginPath(); oCtx.moveTo(i * CELL_O, 0); oCtx.lineTo(i * CELL_O, outputCanvas.height); oCtx.stroke();
      oCtx.beginPath(); oCtx.moveTo(0, i * CELL_O); oCtx.lineTo(outputCanvas.width, i * CELL_O); oCtx.stroke();
    }
  }

  document.querySelectorAll('[data-pool]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-pool]').forEach(b => b.classList.remove('wactive'));
      btn.classList.add('wactive');
      poolMode = btn.dataset.pool;
      draw();
    });
  });
  document.getElementById('pool-randomize').addEventListener('click', randomize);

  draw();
}

/* ── FC & softmax widgets ─────────────────────────────────── */
// Fixed random weights for the FC layer
const FC_W = [
  [0.5, -0.3, 0.8, 0.1],
  [-0.4, 0.7, -0.2, 0.9],
  [0.3, 0.6, -0.5, -0.7],
];
const FC_B = [0.1, -0.2, 0.3];

function initFCWidget() {
  const inputsEl = document.getElementById('fc-inputs');
  const outputsEl = document.getElementById('fc-outputs');
  const LABELS_IN = ['x₁', 'x₂', 'x₃', 'x₄'];
  const LABELS_OUT = ['Class A', 'Class B', 'Class C'];
  const COLORS = ['#4a9eff', '#39d353', '#f0c040'];

  const sliders = LABELS_IN.map((lbl, i) => {
    const wrap = document.createElement('div');
    wrap.className = 'fc-bar-row';
    wrap.innerHTML = `<span class="fc-bar-label">${lbl}</span>
      <input type="range" class="wrange" style="width:100px" min="-2" max="2" step="0.1" value="${i === 0 ? 1 : 0.5}" />
      <span class="fc-bar-val" id="fc-in-${i}">0</span>`;
    inputsEl.appendChild(wrap);
    return wrap.querySelector('input');
  });

  LABELS_OUT.forEach((lbl, j) => {
    const row = document.createElement('div');
    row.className = 'fc-bar-row';
    row.innerHTML = `<span class="fc-bar-label">${lbl}</span>
      <span class="fc-bar-track"><span class="fc-bar-fill" id="fc-out-fill-${j}" style="background:${COLORS[j]}"></span></span>
      <span class="fc-bar-val" id="fc-out-${j}">0</span>`;
    outputsEl.appendChild(row);
  });

  function update() {
    const x = sliders.map((s, i) => { const v = +s.value; document.getElementById(`fc-in-${i}`).textContent = v.toFixed(1); return v; });
    const scores = FC_W.map((row, j) => row.reduce((sum, w, i) => sum + w * x[i], FC_B[j]));
    const maxAbs = Math.max(...scores.map(Math.abs), 1);
    scores.forEach((s, j) => {
      document.getElementById(`fc-out-${j}`).textContent = s.toFixed(2);
      document.getElementById(`fc-out-fill-${j}`).style.width = `${clamp((s / maxAbs) * 50 + 50, 0, 100)}%`;
    });
  }

  sliders.forEach(s => s.addEventListener('input', update));
  update();
}

function initSoftmaxWidget() {
  const inputsEl = document.getElementById('sm-inputs');
  const outputsEl = document.getElementById('sm-outputs');
  const LABELS = ['z₁', 'z₂', 'z₃'];
  const COLORS = ['#e94560', '#4a9eff', '#39d353'];
  const CLASS_LABELS = ['Cat', 'Dog', 'Bird'];

  const sliders = LABELS.map((lbl, i) => {
    const wrap = document.createElement('div');
    wrap.className = 'fc-bar-row';
    wrap.innerHTML = `<span class="fc-bar-label">${lbl}</span>
      <input type="range" class="wrange" style="width:100px" min="-3" max="3" step="0.1" value="${i === 0 ? 2 : i === 1 ? 0.5 : -1}" />
      <span class="fc-bar-val" id="sm-in-${i}">0</span>`;
    inputsEl.appendChild(wrap);
    return wrap.querySelector('input');
  });

  CLASS_LABELS.forEach((lbl, j) => {
    const row = document.createElement('div');
    row.className = 'fc-bar-row';
    row.innerHTML = `<span class="fc-bar-label">${lbl}</span>
      <span class="fc-bar-track"><span class="fc-bar-fill" id="sm-out-fill-${j}" style="background:${COLORS[j]}"></span></span>
      <span class="fc-bar-val" id="sm-out-${j}">0%</span>`;
    outputsEl.appendChild(row);
  });

  function update() {
    const z = sliders.map((s, i) => { const v = +s.value; document.getElementById(`sm-in-${i}`).textContent = v.toFixed(1); return v; });
    const exps = z.map(v => Math.exp(v));
    const sum = exps.reduce((a, b) => a + b, 0);
    exps.forEach((e, j) => {
      const p = e / sum;
      document.getElementById(`sm-out-${j}`).textContent = `${(p * 100).toFixed(1)}%`;
      document.getElementById(`sm-out-fill-${j}`).style.width = `${(p * 100).toFixed(1)}%`;
    });
  }

  sliders.forEach(s => s.addEventListener('input', update));
  update();
}

/* ── diffusion widget ─────────────────────────────────────── */
function initDiffusionWidget() {
  const canvas = document.getElementById('diffusion-canvas');
  const ctx = canvas.getContext('2d');
  const slider = document.getElementById('diffusion-slider');
  const stepValEl = document.getElementById('diffusion-step-val');
  const playBtn = document.getElementById('diffusion-play');
  const resetBtn = document.getElementById('diffusion-reset');
  const N = 32; // pixel grid resolution
  const CELL = canvas.width / N;
  let playing = false;
  let timer = null;

  // deterministic "target image" pattern: a simple radial gradient + ring, values in [0,1]
  const target = Array.from({ length: N }, (_, r) =>
    Array.from({ length: N }, (_, c) => {
      const dx = (c - N / 2) / (N / 2), dy = (r - N / 2) / (N / 2);
      const d = Math.sqrt(dx * dx + dy * dy);
      const ring = Math.exp(-Math.pow((d - 0.55) * 6, 2));
      const core = Math.max(0, 1 - d * 1.4);
      return clamp(ring * 0.9 + core * 0.7, 0, 1);
    }));

  // fixed noise field so the animation is stable, not re-randomized every frame
  const noise = Array.from({ length: N }, () => Array.from({ length: N }, () => Math.random()));

  function draw() {
    const t = +slider.value / 100; // 0 = pure noise, 1 = fully denoised
    stepValEl.textContent = `${slider.value} / 100`;
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const v = clamp(noise[r][c] * (1 - t) + target[r][c] * t, 0, 1);
        const bright = Math.round(v * 255);
        ctx.fillStyle = `rgb(${bright},${bright},${bright})`;
        ctx.fillRect(c * CELL, r * CELL, CELL + 1, CELL + 1);
      }
    }
  }

  slider.addEventListener('input', () => { stopPlaying(); draw(); });

  function stopPlaying() {
    playing = false; clearInterval(timer);
    playBtn.textContent = t('sec3.play');
  }

  playBtn.addEventListener('click', () => {
    if (playing) { stopPlaying(); return; }
    playing = true; playBtn.textContent = t('sec3.pause');
    if (+slider.value >= 100) slider.value = 0;
    timer = setInterval(() => {
      slider.value = Math.min(100, +slider.value + 2);
      draw();
      if (+slider.value >= 100) stopPlaying();
    }, 60);
  });

  resetBtn.addEventListener('click', () => { stopPlaying(); slider.value = 0; draw(); });

  draw();
}

/* ── main init ────────────────────────────────────────────── */
initI18n();
initOverlay();
initSidebar();
initTaskCards();
initIdeaCards();
initParamCounter();
initMLPWidget();
initConvWidget();
initReLUWidget();
initPoolWidget();
initFCWidget();
initSoftmaxWidget();
initDiffusionWidget();
