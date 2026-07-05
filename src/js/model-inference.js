export class ModelInference {
  constructor() {
    this.sessions     = {};
    this.ready        = false;
    this._modelConfig = null;
    this.parameters   = null;
    this._normMean    = [0.1307];
    this._normStd     = [0.3081];
    this._inChannels  = 1;
    this._imgSize     = 28;
  }

  async load(modelConfig, datasetConfig, onProgress) {
    this._modelConfig = modelConfig;
    this.sessions     = {};
    this.ready        = false;
    this._inChannels  = datasetConfig.inChannels ?? 1;
    this._imgSize     = datasetConfig.imgSize ?? 28;
    // Normalise mean/std to arrays (grayscale configs store a scalar)
    const mean = datasetConfig.normMean;
    const std  = datasetConfig.normStd;
    this._normMean = Array.isArray(mean) ? mean : [mean];
    this._normStd  = Array.isArray(std)  ? std  : [std];

    const ort = window.ort;
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/';
    ort.env.wasm.numThreads = 1;

    const modelsBase = `${datasetConfig.modelsPath}/${modelConfig.id}`;
    const files = modelConfig.modelFiles;
    for (let i = 0; i < files.length; i++) {
      const { name, file } = files[i];
      const path = `${modelsBase}/${file}`;
      onProgress?.(i / files.length, `Loading ${modelConfig.label} — ${name}…`);
      try {
        this.sessions[name] = await ort.InferenceSession.create(path, {
          executionProviders: ['wasm'],
        });
      } catch (err) {
        console.error(`Failed to load model ${name}:`, err);
        throw new Error(`Could not load "${name}" for ${modelConfig.label}. Run the training script first.`);
      }
    }

    if (modelConfig.parametersFile) {
      await this._loadParameters(`${modelsBase}/${modelConfig.parametersFile}`);
    } else {
      this.parameters = null;
    }

    await this._loadTorchinfo(`${modelsBase}/torchinfo.json`);

    onProgress?.(1, 'Models ready');
    this.ready = true;
  }

  async _loadParameters(path) {
    try {
      const resp = await fetch(path);
      this.parameters = await resp.json();
    } catch (e) {
      console.warn('Parameters not loaded:', e);
      this.parameters = null;
    }
  }

  async _loadTorchinfo(path) {
    try {
      const resp = await fetch(path);
      if (resp.ok) this.torchinfoStats = await resp.json();
      else this.torchinfoStats = null;
    } catch (e) {
      this.torchinfoStats = null;
    }
  }

  async inferAllLayers(pixelsHWC) {
    if (!this.ready) throw new Error('Models not loaded');
    const ort = window.ort;

    const C   = this._inChannels;
    const S   = this._imgSize;
    const ppi = C * S * S;

    // pixelsHWC is Float32Array in HWC order [0,1]; convert to CHW with per-channel norm
    const normalized = new Float32Array(ppi);
    if (C === 1) {
      for (let i = 0; i < ppi; i++) normalized[i] = (pixelsHWC[i] - this._normMean[0]) / this._normStd[0];
    } else {
      // HWC → CHW: pixel i has channels at offsets i*C, i*C+1, ..., i*C+(C-1)
      const hw = S * S;
      for (let c = 0; c < C; c++) {
        const mean = this._normMean[c];
        const std  = this._normStd[c];
        for (let p = 0; p < hw; p++) {
          normalized[c * hw + p] = (pixelsHWC[p * C + c] - mean) / std;
        }
      }
    }

    const input = new ort.Tensor('float32', normalized, [1, C, S, S]);
    const run   = async (name) => {
      const result = await this.sessions[name].run({ input });
      return result[Object.keys(result)[0]];
    };

    const results = {};
    await Promise.all(
      this._modelConfig.modelFiles.map(async ({ name }) => {
        const t = await run(name);
        const key = name === 'full' ? 'output' : name;
        results[key] = { data: t.data, dims: t.dims };
      })
    );
    return results;
  }
}
