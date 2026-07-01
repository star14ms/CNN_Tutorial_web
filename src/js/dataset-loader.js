/**
 * DatasetLoader — browser-side loader for dataset binary files.
 *
 * Binary format (produced by train/export_dataset.py):
 *   {split}_images.bin  — N × pixelsPerImage uint8 (raw pixel 0-255, HWC order)
 *   {split}_labels.bin  — N uint8
 *
 * For grayscale 28×28 datasets: pixelsPerImage = 784 (H×W)
 * For RGB 32×32 datasets:       pixelsPerImage = 3072 (H×W×C)
 *
 * Labels are loaded eagerly (small: ~10 KB for test).
 * Images are loaded lazily — single images via HTTP Range requests,
 * or the full buffer on demand for prediction sweeps.
 */
export class DatasetLoader {
  constructor() {
    this.reset();
  }

  reset() {
    this._dataPath        = { test: null, train: null };
    this._images          = { test: null, train: null };   // full buffer, loaded on demand
    this._labels          = { test: null, train: null };
    this._loading         = { test: false, train: false };
    this._imageCache      = { test: {}, train: {} };       // index → Float32Array
    this._predictions     = {};
    this._predicting      = false;
    this._datasetId       = null;
    this._pixelsPerImage  = 784;   // updated on load()
  }

  get pixelsPerImage() { return this._pixelsPerImage; }

  size(split) {
    const labels = this._labels[split];
    return labels ? labels.length : 0;
  }

  /** Returns true when labels (and optionally full images) are loaded. */
  isLoaded(split) {
    return this._labels[split] !== null;
  }

  /** Load only the labels file for this split (fast, ~10–60 KB). */
  async load(split, datasetConfig) {
    if (this._datasetId !== datasetConfig.id) this.reset();
    this._datasetId = datasetConfig.id;
    this._dataPath[split] = `${datasetConfig.dataPath}/${split}`;
    const C = datasetConfig.inChannels ?? 1;
    const S = datasetConfig.imgSize ?? 28;
    this._pixelsPerImage = C * S * S;

    if (this.isLoaded(split) || this._loading[split]) return;
    this._loading[split] = true;
    const res = await fetch(`${this._dataPath[split]}_labels.bin`);
    if (!res.ok) throw new Error(`Failed to fetch labels for ${split}`);
    this._labels[split] = new Uint8Array(await res.arrayBuffer());
    this._loading[split] = false;
  }

  /** Load the full images buffer for this split (needed for prediction sweeps). */
  async _loadFullImages(split) {
    if (this._images[split] !== null) return;
    const res = await fetch(`${this._dataPath[split]}_images.bin`);
    if (!res.ok) throw new Error(`Failed to fetch images for ${split}`);
    this._images[split] = new Uint8Array(await res.arrayBuffer());
  }

  /**
   * Returns Float32Array(pixelsPerImage) in [0,1].
   * Uses HTTP Range request for a single image; falls back to full buffer.
   * Caches the result to avoid duplicate fetches.
   */
  async getImage(split, index) {
    if (this._imageCache[split][index]) return this._imageCache[split][index];

    const ppi = this._pixelsPerImage;
    let raw;
    if (this._images[split]) {
      // Full buffer already loaded (e.g. during prediction sweep)
      const offset = index * ppi;
      raw = this._images[split].subarray(offset, offset + ppi);
    } else {
      const start = index * ppi;
      const end   = start + ppi - 1;
      const res   = await fetch(`${this._dataPath[split]}_images.bin`, {
        headers: { Range: `bytes=${start}-${end}` },
      });
      if (res.status === 206) {
        raw = new Uint8Array(await res.arrayBuffer());
      } else if (res.ok) {
        // Server doesn't support Range — cache the full buffer and extract
        this._images[split] = new Uint8Array(await res.arrayBuffer());
        raw = this._images[split].subarray(start, start + ppi);
      } else {
        throw new Error(`Failed to fetch image ${index} for ${split}`);
      }
    }

    const out = new Float32Array(ppi);
    for (let i = 0; i < ppi; i++) out[i] = raw[i] / 255;
    this._imageCache[split][index] = out;
    return out;
  }

  getLabel(split, index) {
    const labels = this._labels[split];
    return labels ? labels[index] : null;
  }

  hasPredictions(modelId) {
    return !!this._predictions[modelId];
  }

  /** Try loading precomputed predictions from {modelsPath}/test_preds.bin.
   *  Returns true if loaded, false if file not available. */
  async tryLoadPredictions(modelId, modelsPath) {
    if (this._predictions[modelId]) return true;
    try {
      const res = await fetch(`${modelsPath}/test_preds.bin`);
      if (!res.ok) return false;
      const buf = await res.arrayBuffer();
      this._predictions[modelId] = new Int32Array(new Uint8Array(buf));
      return true;
    } catch (_) { return false; }
  }

  async computeTestPredictions(modelId, inferFn, onProgress) {
    if (this._predicting) return;
    if (!this.isLoaded('test')) throw new Error('Test set not loaded');

    this._predicting = true;

    // Load full images buffer once for the sweep (avoids 10 000 Range requests)
    onProgress?.(0, 0, 'Loading images…');
    await this._loadFullImages('test');

    const n = this.size('test');
    const preds = new Int32Array(n);

    for (let i = 0; i < n; i++) {
      const pixels = await this.getImage('test', i);
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
