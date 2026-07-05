/* Lightweight i18n for the header + CNN Learning Center (English / Korean). */

const STORAGE_KEY = 'cnn-tutorial-lang';
let currentLang = localStorage.getItem(STORAGE_KEY) === 'ko' ? 'ko' : 'en';
const listeners = [];

export function getLang() { return currentLang; }

export function t(key) {
  const entry = DICT[key];
  if (!entry) return key;
  return entry[currentLang] ?? entry.en ?? key;
}

/** Simple {placeholder} substitution for dynamically-built strings. */
export function tf(key, params = {}) {
  let s = t(key);
  for (const [k, v] of Object.entries(params)) s = s.replaceAll(`{${k}}`, v);
  return s;
}

export function onLanguageChange(fn) { listeners.push(fn); }

function applyStaticTranslations() {
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    const val = t(key);
    if (el.hasAttribute('data-i18n-html')) el.innerHTML = val;
    else el.textContent = val;
  });
  document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    el.placeholder = t(el.getAttribute('data-i18n-placeholder'));
  });
}

export function setLang(lang) {
  if (lang !== 'en' && lang !== 'ko') return;
  currentLang = lang;
  localStorage.setItem(STORAGE_KEY, lang);
  document.documentElement.lang = lang;
  applyStaticTranslations();
  listeners.forEach(fn => fn(lang));
}

export function initI18n() {
  document.documentElement.lang = currentLang;
  applyStaticTranslations();

  const select = document.getElementById('lang-select');
  if (select) {
    select.value = currentLang;
    select.addEventListener('change', () => setLang(select.value));
  }
}

const DICT = {
  'header.title': { en: 'CNN Visualization', ko: 'CNN 시각화' },
  'header.subtitle': { en: 'Client-side inference & 3D layer visualization', ko: '클라이언트 사이드 추론 및 3D 레이어 시각화' },
  'header.datasetLabel': { en: 'Choose Dataset', ko: '데이터셋 선택' },
  'header.learnBtn': { en: '📖 Learn', ko: '📖 학습' },

  'learn.title': { en: 'CNN Learning Center', ko: 'CNN 학습 센터' },
  'learn.subtitle': { en: 'Interactive guide to Convolutional Neural Networks', ko: '합성곱 신경망(CNN) 대화형 학습 가이드' },
  'learn.close': { en: '✕ Close', ko: '✕ 닫기' },
  'sidebar.contents': { en: 'Contents', ko: '목차' },
  'nav.vision': { en: 'Computer Vision', ko: '컴퓨터 비전' },
  'nav.mlp': { en: 'MLP → CNN', ko: 'MLP → CNN' },
  'nav.conv': { en: 'Conv Layer', ko: '합성곱 레이어' },
  'nav.activation': { en: 'Activation & Pooling', ko: '활성화 함수 & 풀링' },
  'nav.fc': { en: 'FC & Softmax', ko: '완전연결 계층 & 소프트맥스' },
  'nav.beyond': { en: 'Beyond CNNs', ko: 'CNN을 넘어서' },

  'sec1.title': { en: 'Computer Vision Tasks', ko: '컴퓨터 비전 과제' },
  'sec1.intro': {
    en: 'Computer vision teaches machines to understand images. There are several categories of vision tasks, each with a different output format and a different architecture family suited to solving it. Click a card to learn more.',
    ko: '컴퓨터 비전은 기계가 이미지를 이해하도록 가르치는 분야입니다. 비전 과제에는 여러 범주가 있으며, 각각 다른 출력 형식과 이를 해결하는 데 적합한 아키텍처 계열을 가지고 있습니다. 카드를 클릭하면 더 알아볼 수 있습니다.',
  },
  'task.outputLabel': { en: 'Output:', ko: '출력:' },

  'sec2.title': { en: 'From MLP to CNN', ko: 'MLP에서 CNN으로' },
  'sec2.mlp.h3': { en: 'Multi-Layer Perceptron', ko: '다층 퍼셉트론(MLP)' },
  'sec2.mlp.p': {
    en: 'A Multi-Layer Perceptron (MLP) is the simplest deep network: every neuron in one layer connects to every neuron in the next. Hover over a neuron to see its connections.',
    ko: '다층 퍼셉트론(MLP)은 가장 단순한 형태의 심층 신경망으로, 한 층의 모든 뉴런이 다음 층의 모든 뉴런과 연결됩니다. 뉴런에 마우스를 올리면 연결을 확인할 수 있습니다.',
  },
  'sec2.mlp.widgetTitle': { en: 'MLP Diagram (4 → 4 → 3)', ko: 'MLP 다이어그램 (4 → 4 → 3)' },
  'sec2.mlp.note': { en: 'Each connection has an independent learned weight.', ko: '각 연결은 독립적으로 학습된 가중치를 가집니다.' },

  'sec2.hard.h3': { en: 'Why Images Are Hard for MLPs', ko: '이미지가 MLP에게 어려운 이유' },
  'sec2.hard.p': {
    en: "A fully connected layer treats each pixel as independent. For a 28×28 image (784 inputs) and 128 hidden neurons, that's already 100,352 weights — just for one layer. Scale to 256×256 color images and the first layer alone needs over 25 million weights.",
    ko: '완전연결 계층은 각 픽셀을 독립적으로 다룹니다. 28×28 이미지(784개 입력)와 은닉 뉴런 128개만 해도 한 층에서 벌써 100,352개의 가중치가 필요합니다. 256×256 컬러 이미지로 확장하면 첫 번째 층만으로도 2,500만 개가 넘는 가중치가 필요합니다.',
  },
  'sec2.hard.widgetTitle': { en: 'Parameter Explosion', ko: '파라미터 폭증' },
  'sec2.hard.imgSizeLabel': { en: 'Image size:', ko: '이미지 크기:' },
  'sec2.hard.hiddenLabel': { en: 'Hidden neurons:', ko: '은닉 뉴런 수:' },
  'sec2.hard.counterLabel': { en: 'weights in layer 1 alone', ko: '1번째 층만의 가중치 수' },
  'sec2.hard.highlight': {
    en: '<strong>Key problems:</strong> too many parameters → slow training, overfitting. No spatial awareness — the network has no concept that pixel (3,4) is next to pixel (3,5). Not translation invariant — the same feature at a different position looks completely different to the network.',
    ko: '<strong>핵심 문제:</strong> 파라미터가 너무 많아 학습이 느려지고 과적합이 발생합니다. 공간 인식 능력 부재 — 네트워크는 픽셀 (3,4)가 픽셀 (3,5)와 인접해 있다는 개념이 없습니다. 이동 불변성 부재 — 같은 특징이라도 위치가 다르면 네트워크에는 완전히 다르게 보입니다.',
  },

  'sec2.cnn.h3': { en: 'How CNNs Solve This', ko: 'CNN은 이를 어떻게 해결하는가' },
  'sec2.cnn.p': {
    en: 'Convolutional Neural Networks introduce three structural ideas that make image learning tractable:',
    ko: '합성곱 신경망(CNN)은 이미지 학습을 다루기 쉽게 만드는 세 가지 구조적 아이디어를 도입합니다:',
  },
  'idea.local.title': { en: 'Local Connectivity', ko: '지역 연결성 (Local Connectivity)' },
  'idea.local.body': {
    en: 'Each neuron in a conv layer only looks at a small local region of the input (the <em>receptive field</em>), typically 3×3 or 5×5. This dramatically reduces the number of connections and biases the network toward detecting local patterns like edges and corners.',
    ko: '합성곱 계층의 각 뉴런은 입력의 작은 지역(<em>수용 영역</em>, receptive field)만 봅니다. 보통 3×3 또는 5×5 크기입니다. 이는 연결 수를 크게 줄이고, 네트워크가 모서리나 코너 같은 지역적 패턴을 탐지하도록 유도합니다.',
  },
  'idea.weight.title': { en: 'Weight Sharing', ko: '가중치 공유 (Weight Sharing)' },
  'idea.weight.body': {
    en: 'The same kernel (filter) is slid across the entire image. Every position shares the same learned weights. A 3×3 kernel has only 9 weights regardless of image size — compared to millions in a fully connected layer. This also makes the network <em>translation equivariant</em>: the same feature at a different position produces the same response, just shifted.',
    ko: '동일한 커널(필터)이 이미지 전체를 슬라이딩하며 적용됩니다. 모든 위치가 같은 학습된 가중치를 공유합니다. 3×3 커널은 이미지 크기와 상관없이 단 9개의 가중치만 가지며, 이는 완전연결 계층의 수백만 개와 대조적입니다. 이는 네트워크에 <em>이동 등변성(translation equivariance)</em>을 부여합니다: 같은 특징이 다른 위치에 있어도 동일한 반응이 그 위치만큼 이동되어 나타납니다.',
  },
  'idea.hier.title': { en: 'Hierarchical Features', ko: '계층적 특징 (Hierarchical Features)' },
  'idea.hier.body': {
    en: 'Early layers detect simple patterns (edges, colors). Middle layers combine those into textures and shapes. Deep layers represent high-level concepts (eyes, wheels, faces). This hierarchy of features mirrors how the biological visual cortex processes images.',
    ko: '초기 층은 모서리, 색상 같은 단순한 패턴을 탐지합니다. 중간 층은 이를 결합해 질감과 모양을 구성합니다. 깊은 층은 눈, 바퀴, 얼굴 같은 고수준 개념을 표현합니다. 이러한 특징의 계층 구조는 생물학적 시각 피질이 이미지를 처리하는 방식과 유사합니다.',
  },

  'sec3.title': { en: 'Convolutional Layer', ko: '합성곱 계층' },
  'sec3.intro': {
    en: 'A convolution slides a small kernel over the input image. At each position it computes a dot product between the kernel weights and the local input patch, writing the result to the output feature map.',
    ko: '합성곱은 작은 커널을 입력 이미지 위로 슬라이딩합니다. 각 위치에서 커널 가중치와 지역 입력 패치 간의 내적을 계산하여 결과를 출력 특징 맵에 기록합니다.',
  },
  'sec3.sp.h3': { en: 'Stride & Padding', ko: '스트라이드와 패딩' },
  'sec3.sp.p1': {
    en: '<strong>Stride</strong> is how many pixels the kernel moves between applications. Stride 1 slides across every position, producing a dense output the size of the input (minus the kernel\'s edge). Stride 2 skips every other position, halving the output resolution roughly — a cheap way to downsample while convolving.',
    ko: '<strong>스트라이드</strong>는 커널이 한 번에 이동하는 픽셀 수입니다. 스트라이드 1은 모든 위치를 거치며 입력과 거의 같은 크기의 조밀한 출력을 만듭니다. 스트라이드 2는 한 칸씩 건너뛰어 출력 해상도를 대략 절반으로 줄이는, 합성곱과 동시에 다운샘플링하는 저렴한 방법입니다.',
  },
  'sec3.sp.p2': {
    en: '<strong>Padding</strong> adds a border of zero-value pixels around the input before sliding the kernel. With <strong>no padding</strong> ("valid"), the kernel can never be centered on a border pixel, so each side shrinks by half the kernel size — a 3×3 kernel loses 1 pixel per side. <strong>"Same" padding</strong> adds exactly enough zero border (⌊kernel size / 2⌋ — 1 pixel for a 3×3 kernel) so that, at stride 1, the output is the <em>same size</em> as the input. This matters when stacking many conv layers: without it, the feature map keeps shrinking and eventually vanishes.',
    ko: '<strong>패딩</strong>은 커널을 슬라이딩하기 전에 입력 주위에 0 값의 테두리를 추가하는 것입니다. <strong>패딩이 없으면</strong>("valid") 커널이 테두리 픽셀에 중심을 둘 수 없으므로 각 변이 커널 크기의 절반만큼 줄어듭니다 — 3×3 커널은 각 변에서 1픽셀씩 줄어듭니다. <strong>"Same" 패딩</strong>은 정확히 그만큼(3×3 커널 기준 1픽셀, 즉 ⌊커널 크기 / 2⌋)의 0 테두리를 추가하여, 스트라이드 1일 때 출력 크기가 입력과 <em>동일한 크기</em>가 되도록 합니다. 이는 합성곱 층을 여러 겹 쌓을 때 중요합니다 — 패딩이 없으면 특징 맵이 계속 줄어들다 결국 사라지기 때문입니다.',
  },
  'sec3.sp.p3': {
    en: 'Output size formula: <code>output = ⌊(input + 2·padding − kernel) / stride⌋ + 1</code>. Try it below — the pink dashed border on the input shows the padding cells, and the formula under the demo updates live.',
    ko: '출력 크기 공식: <code>output = ⌊(input + 2·padding − kernel) / stride⌋ + 1</code>. 아래에서 직접 확인해보세요 — 입력 이미지의 분홍색 점선 테두리가 패딩 셀을 나타내며, 데모 아래의 공식이 실시간으로 갱신됩니다.',
  },
  'sec3.widgetTitle': { en: 'Interactive Kernel Demo', ko: '대화형 커널 데모' },
  'sec3.inputLabel': { en: 'Input (click cell to cycle brightness)', ko: '입력 (셀을 클릭하면 밝기가 순환합니다)' },
  'sec3.randomFrom': { en: '🎲 Random sample from:', ko: '🎲 무작위 샘플 가져오기:' },
  'sec3.presetPlaceholder': { en: 'Or choose a synthetic pattern…', ko: '또는 합성 패턴을 선택하세요…' },
  'sec3.patternCheckerboard': { en: 'Checkerboard', ko: '체커보드' },
  'sec3.patternGradient': { en: 'Gradient', ko: '그라데이션' },
  'sec3.patternCross': { en: 'Cross', ko: '십자 모양' },
  'sec3.patternNoise': { en: 'Random Noise', ko: '무작위 노이즈' },
  'sec3.patternCustom': { en: 'Custom (hand-edited)', ko: '사용자 지정 (직접 편집)' },
  'sec3.kernelLabel': { en: 'Kernel (click cell to edit)', ko: '커널 (셀을 클릭해 값을 수정하세요)' },
  'sec3.kernelEdgeTop': { en: 'Edge ↑', ko: '에지 ↑' },
  'sec3.kernelEdgeBottom': { en: 'Edge ↓', ko: '에지 ↓' },
  'sec3.kernelEdgeLeft': { en: 'Edge ←', ko: '에지 ←' },
  'sec3.kernelEdgeRight': { en: 'Edge →', ko: '에지 →' },
  'sec3.kernelEdgeTL': { en: 'Corner ↖', ko: '모서리 ↖' },
  'sec3.kernelEdgeTR': { en: 'Corner ↗', ko: '모서리 ↗' },
  'sec3.kernelEdgeBL': { en: 'Corner ↙', ko: '모서리 ↙' },
  'sec3.kernelEdgeBR': { en: 'Corner ↘', ko: '모서리 ↘' },
  'sec3.kernelBlur': { en: 'Blur', ko: '블러' },
  'sec3.kernelSharpen': { en: 'Sharpen', ko: '샤픈' },
  'sec3.kernelCustom': { en: 'Custom', ko: '사용자 지정' },
  'sec3.outputLabel': { en: 'Output', ko: '출력' },
  'sec3.strideLabel': { en: 'Stride:', ko: '스트라이드:' },
  'sec3.paddingLabel': { en: 'Padding:', ko: '패딩:' },
  'sec3.paddingNone': { en: 'None', ko: '없음' },
  'sec3.paddingSame': { en: 'Same', ko: '동일(Same)' },
  'sec3.play': { en: '▶ Play', ko: '▶ 재생' },
  'sec3.pause': { en: '⏸ Pause', ko: '⏸ 일시정지' },
  'sec3.step': { en: 'Step', ko: '한 단계' },
  'sec3.reset': { en: 'Reset', ko: '초기화' },
  'conv.statusPattern': { en: 'Pattern: {name}', ko: '패턴: {name}' },
  'conv.statusLoading': { en: 'Loading random {name} sample…', ko: '{name} 데이터셋에서 무작위 샘플 불러오는 중…' },
  'conv.statusLoaded': { en: '{name} test #{idx} — label: {label}', ko: '{name} 테스트 #{idx} — 레이블: {label}' },

  'sec4.title': { en: 'Activation & Pooling', ko: '활성화 함수 & 풀링' },
  'sec4.act.h3': { en: 'Activation Functions', ko: '활성화 함수' },
  'sec4.act.p': {
    en: "After convolution, an element-wise nonlinearity is applied. Without it, stacking layers would still be a linear function and the network couldn't learn complex patterns. Drag the point on the input axis to explore.",
    ko: '합성곱 이후에는 원소별 비선형 함수가 적용됩니다. 이것이 없다면 층을 아무리 쌓아도 결국 선형 함수에 불과해 복잡한 패턴을 학습할 수 없습니다. 입력 축의 점을 드래그해 살펴보세요.',
  },
  'sec4.act.widgetTitle': { en: 'Activation Function Demo', ko: '활성화 함수 데모' },
  'sec4.act.inputX': { en: 'Input x:', ko: '입력 x:' },
  'sec4.act.outputFx': { en: 'Output f(x):', ko: '출력 f(x):' },

  'sec4.pool.h3': { en: 'Pooling Layer', ko: '풀링 계층' },
  'sec4.pool.p': {
    en: 'Pooling reduces spatial dimensions, making the representation smaller and more robust to small translations. A window slides over the feature map and picks either the <em>maximum</em> value or the <em>average</em>.',
    ko: '풀링은 공간 차원을 줄여 표현을 더 작고 작은 이동에 더 강인하게 만듭니다. 윈도우가 특징 맵 위를 슬라이딩하며 <em>최댓값</em> 또는 <em>평균값</em>을 선택합니다.',
  },
  'sec4.pool.widgetTitle': { en: 'Max / Avg Pooling Demo (4×4 → 2×2, stride 2)', ko: '최대/평균 풀링 데모 (4×4 → 2×2, 스트라이드 2)' },
  'sec4.pool.inputLabel': { en: 'Input', ko: '입력' },
  'sec4.pool.outputLabel': { en: 'Output', ko: '출력' },
  'sec4.pool.randomize': { en: 'Randomize', ko: '무작위로 섞기' },
  'sec4.pool.max': { en: 'Max Pool', ko: '최대 풀링' },
  'sec4.pool.avg': { en: 'Avg Pool', ko: '평균 풀링' },

  'sec5.title': { en: 'Fully Connected Layer & Softmax', ko: '완전연결 계층 & 소프트맥스' },
  'sec5.fc.h3': { en: 'Fully Connected (Linear) Layer', ko: '완전연결(선형) 계층' },
  'sec5.fc.p': {
    en: 'After the convolutional and pooling layers extract spatial features, one or more fully connected layers combine them into class scores. Each output neuron has a weighted connection to every input. Move the sliders to see how input values affect outputs.',
    ko: '합성곱과 풀링 계층이 공간적 특징을 추출한 뒤, 하나 이상의 완전연결 계층이 이를 결합해 클래스 점수를 만듭니다. 각 출력 뉴런은 모든 입력과 가중치로 연결되어 있습니다. 슬라이더를 움직여 입력값이 출력에 미치는 영향을 확인해보세요.',
  },
  'sec5.fc.widgetTitle': { en: 'Matrix Multiply: W·x + b (4 inputs → 3 outputs)', ko: '행렬 곱: W·x + b (입력 4개 → 출력 3개)' },
  'sec5.fc.inputX': { en: 'Input x', ko: '입력 x' },
  'sec5.fc.outputScores': { en: 'Output scores', ko: '출력 점수' },

  'sec5.sm.h3': { en: 'Softmax Output', ko: '소프트맥스 출력' },
  'sec5.sm.p': {
    en: 'Softmax converts raw scores (logits) into probabilities that sum to 1. Large differences in logit values get amplified — the highest logit dominates the output. Adjust the sliders to explore.',
    ko: '소프트맥스는 원시 점수(로짓)를 합이 1이 되는 확률로 변환합니다. 로짓 값의 차이가 클수록 그 차이가 증폭되어 가장 큰 로짓이 출력을 지배하게 됩니다. 슬라이더를 조절해 확인해보세요.',
  },
  'sec5.sm.logitsZ': { en: 'Logits z', ko: '로짓 z' },
  'sec5.sm.probabilities': { en: 'Probabilities', ko: '확률' },

  'sec6.title': { en: 'Beyond CNNs: Diffusion & Generative Models', ko: 'CNN을 넘어서: 디퓨전 & 생성 모델' },
  'sec6.intro': {
    en: 'Everything so far has been a <em>discriminative</em> model — mapping an image to a label, box, or mask. <strong>Generative</strong> models flip the problem: instead of recognizing an image, they create one. Diffusion models are the current state of the art for image, video, and audio generation.',
    ko: '지금까지 다룬 것은 모두 <em>판별(discriminative)</em> 모델로, 이미지를 레이블·박스·마스크에 매핑합니다. <strong>생성(generative)</strong> 모델은 문제를 뒤집습니다 — 이미지를 인식하는 대신 새로 만들어냅니다. 디퓨전 모델은 현재 이미지·영상·오디오 생성 분야의 최신 기술입니다.',
  },
  'sec6.how.h3': { en: 'How Diffusion Models Work', ko: '디퓨전 모델의 작동 원리' },
  'sec6.how.p1': {
    en: 'Diffusion models learn to generate data by reversing a gradual noising process. Training happens in two phases:',
    ko: '디퓨전 모델은 점진적인 노이즈 추가 과정을 역으로 되돌리며 데이터를 생성하는 법을 학습합니다. 학습은 두 단계로 이루어집니다:',
  },
  'sec6.how.p2': {
    en: '<strong style="color:var(--yellow)">Forward process:</strong> a training image is progressively corrupted by adding a small amount of Gaussian noise at each of hundreds of steps, until it becomes indistinguishable from pure static.',
    ko: '<strong style="color:var(--yellow)">순방향 과정(Forward process):</strong> 학습 이미지에 수백 단계에 걸쳐 조금씩 가우시안 노이즈를 더해, 결국 완전한 잡음과 구별할 수 없게 만듭니다.',
  },
  'sec6.how.p3': {
    en: '<strong style="color:var(--yellow)">Reverse process:</strong> a neural network — usually a <strong>U-Net</strong> — is trained to predict and remove the noise added at each step. Once trained, it can start from pure random noise and iteratively "denoise" it into a coherent, realistic image.',
    ko: '<strong style="color:var(--yellow)">역방향 과정(Reverse process):</strong> 신경망(보통 <strong>U-Net</strong>)이 각 단계에서 추가된 노이즈를 예측하고 제거하도록 학습됩니다. 학습이 끝나면 순수한 무작위 노이즈에서 시작해 반복적으로 "잡음을 제거"하며 일관되고 사실적인 이미지를 만들어냅니다.',
  },
  'sec6.how.widgetTitle': { en: 'Denoising Simulation', ko: '노이즈 제거 시뮬레이션' },
  'sec6.how.stepLabel': { en: 'Denoising step:', ko: '노이즈 제거 단계:' },
  'sec6.how.note': {
    en: 'Left: pure noise (step 0). Right: fully denoised image (step 100). Drag the slider or press Play to watch the reverse process.',
    ko: '왼쪽: 순수 노이즈(0단계). 오른쪽: 완전히 노이즈가 제거된 이미지(100단계). 슬라이더를 드래그하거나 재생 버튼을 눌러 역방향 과정을 확인해보세요.',
  },

  'sec6.used.h3': { en: 'Where Diffusion Models Are Used', ko: '디퓨전 모델의 활용 분야' },
  'idea.media.title': { en: 'Media Generation & Synthesis', ko: '미디어 생성 & 합성' },
  'idea.media.body': {
    en: '<strong>Text-to-image:</strong> systems like Stable Diffusion and DALL·E turn a written prompt into a novel, realistic image by guiding the denoising process with text embeddings. <strong>Video generation:</strong> the same idea is extended across time, keeping frames temporally consistent to produce moving clips from a prompt.',
    ko: '<strong>텍스트-투-이미지:</strong> Stable Diffusion, DALL·E 같은 시스템은 텍스트 임베딩으로 노이즈 제거 과정을 유도하여 글로 쓴 프롬프트를 새롭고 사실적인 이미지로 바꿉니다. <strong>영상 생성:</strong> 같은 아이디어를 시간 축으로 확장하여, 프레임 간 시간적 일관성을 유지하며 프롬프트로부터 움직이는 영상을 만들어냅니다.',
  },
  'idea.edit.title': { en: 'Image Editing & Manipulation', ko: '이미지 편집 & 조작' },
  'idea.edit.body': {
    en: '<strong>Inpainting:</strong> erase part of an image and let the model fill the gap based on surrounding context or a text prompt. <strong>Image-to-image:</strong> start from a sketch or photo and prompt the model to restyle it, change the background, or add detail.',
    ko: '<strong>인페인팅:</strong> 이미지 일부를 지우고 주변 맥락이나 텍스트 프롬프트를 바탕으로 모델이 빈 부분을 채우게 합니다. <strong>이미지-투-이미지:</strong> 스케치나 사진에서 시작해 스타일을 바꾸거나, 배경을 변경하거나, 세부 묘사를 추가하도록 프롬프트를 지정합니다.',
  },
  'idea.audio.title': { en: 'Audio & Music', ko: '오디오 & 음악' },
  'idea.audio.body': {
    en: 'Waveforms (or their spectrograms) are treated like image pixels — diffusion progressively refines noise into high-fidelity speech or music.',
    ko: '파형(또는 스펙트로그램)을 이미지 픽셀처럼 다루어, 디퓨전이 노이즈를 점진적으로 정제해 고품질의 음성이나 음악을 만들어냅니다.',
  },
  'idea.industry.title': { en: 'Specialized Industries — incl. Healthcare', ko: '특수 산업 분야 — 의료 포함' },
  'idea.industry.body': {
    en: '<strong>Super-resolution:</strong> medical scans (MRI/CT) and surveillance footage are sharpened by diffusion upscalers, revealing detail invisible in the low-resolution input. <strong>Robotics &amp; science:</strong> diffusion models predict complex physical dynamics for robot planning, and synthesize plausible molecular or material structures for drug and materials discovery.',
    ko: '<strong>초해상도:</strong> 의료 영상(MRI/CT)과 감시 영상을 디퓨전 업스케일러로 선명하게 만들어, 저해상도 입력에서는 보이지 않던 세부 정보를 드러냅니다. <strong>로보틱스 & 과학:</strong> 디퓨전 모델은 로봇 계획을 위한 복잡한 물리적 동역학을 예측하고, 신약 및 재료 발견을 위한 그럴듯한 분자·재료 구조를 합성합니다.',
  },
  'sec6.papers.h3': { en: 'Selected Diffusion Papers', ko: '주요 디퓨전 논문' },
};
