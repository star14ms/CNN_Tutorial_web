/**
 * DatasetLoader — browser-side loader for dataset binary files.
 *
 * Binary format (produced by train/export_dataset.py):
 *   {split}_images.bin  — N × 784 uint8 (raw pixel 0-255)
 *   {split}_labels.bin  — N uint8 (digit 0-9)
 *
 * getImage() returns Float32Array(784) in [0,1]. ModelInference applies
 * dataset-specific normalization internally before ONNX.
 */
export class DatasetLoader {
  constructor() {
    this.reset();
  }

  reset() {
    this._images     = { test: null, train: null };
    this._labels     = { test: null, train: null };
    this._loading    = { test: false, train: false };
    this._predictions = {};
    this._predicting  = false;
    this._datasetId   = null;
  }

  size(split) {
    const labels = this._labels[split];
    return labels ? labels.length : 0;
  }

  isLoaded(split) {
    return this._images[split] !== null && this._labels[split] !== null;
  }

  async load(split, datasetConfig) {
    if (this._datasetId !== datasetConfig.id) this.reset();
    this._datasetId = datasetConfig.id;

    if (this.isLoaded(split) || this._loading[split]) return;
    this._loading[split] = true;
    const base = `${datasetConfig.dataPath}/${split}`;
    const [imgBuf, lblBuf] = await Promise.all([
      fetch(`${base}_images.bin`).then(r => {
        if (!r.ok) throw new Error(`Failed to fetch ${base}_images.bin`);
        return r.arrayBuffer();
      }),
      fetch(`${base}_labels.bin`).then(r => {
        if (!r.ok) throw new Error(`Failed to fetch ${base}_labels.bin`);
        return r.arrayBuffer();
      }),
    ]);
    this._images[split] = new Uint8Array(imgBuf);
    this._labels[split] = new Uint8Array(lblBuf);
    this._loading[split] = false;
  }

  /** Returns Float32Array(784) in [0,1], or null if not loaded. */
  getImage(split, index) {
    const images = this._images[split];
    if (!images) return null;
    const offset = index * 784;
    const raw = images.subarray(offset, offset + 784);
    const out = new Float32Array(784);
    for (let i = 0; i < 784; i++) out[i] = raw[i] / 255;
    return out;
  }

  getLabel(split, index) {
    const labels = this._labels[split];
    return labels ? labels[index] : null;
  }

  hasPredictions(modelId) {
    return !!this._predictions[modelId];
  }

  async computeTestPredictions(modelId, inferFn, onProgress) {
    if (this._predicting) return;
    if (!this.isLoaded('test')) throw new Error('Test set not loaded');

    this._predicting = true;
    const n = this.size('test');
    const preds = new Int32Array(n);

    for (let i = 0; i < n; i++) {
      const pixels = this.getImage('test', i);
      preds[i] = await inferFn(pixels);
      if (onProgress && i % 100 === 0) onProgress(i, n);
    }

    this._predictions[modelId] = preds;
    this._predicting = false;
    onProgress?.(n, n);
  }

  getPrediction(modelId, testIndex) {
    return this._predictions[modelId]?.[testIndex] ?? null;
  }

  randomSample(split, filter = 'all', digit = null, modelId = null) {
    const labels = this._labels[split];
    if (!labels) return null;
    const n = labels.length;

    const candidates = [];
    for (let i = 0; i < n; i++) {
      const lbl = labels[i];
      if (digit !== null && lbl !== digit) continue;

      if (filter === 'correct' || filter === 'incorrect') {
        if (split !== 'test' || !modelId) continue;
        const pred = this._predictions[modelId]?.[i];
        if (pred === undefined || pred === null) continue;
        const isCorrect = pred === lbl;
        if (filter === 'correct' && !isCorrect) continue;
        if (filter === 'incorrect' && isCorrect) continue;
      }

      candidates.push(i);
    }

    if (candidates.length === 0) return null;
    const idx = candidates[Math.floor(Math.random() * candidates.length)];
    return { index: idx, label: labels[idx] };
  }
}
