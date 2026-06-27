export class ModelInference {
  constructor() {
    this.sessions   = {};
    this.ready      = false;
    this._modelConfig = null;
    this.parameters  = null;
  }

  async load(modelConfig, onProgress) {
    this._modelConfig = modelConfig;
    this.sessions     = {};
    this.ready        = false;

    const ort = window.ort;
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/';
    ort.env.wasm.numThreads = 1;

    const files = modelConfig.modelFiles;
    for (let i = 0; i < files.length; i++) {
      const { name, path } = files[i];
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
    onProgress?.(1, 'Models ready');
    this.ready = true;
  }

  async loadParameters(path) {
    if (!path) return;
    try {
      const resp = await fetch(path);
      this.parameters = await resp.json();
    } catch (e) {
      console.warn('Parameters not loaded:', e);
      this.parameters = null;
    }
  }

  async inferAllLayers(pixels784) {
    if (!this.ready) throw new Error('Models not loaded');
    const ort = window.ort;

    // MNIST normalization: (pixel − 0.1307) / 0.3081
    const normalized = new Float32Array(784);
    for (let i = 0; i < 784; i++) normalized[i] = (pixels784[i] - 0.1307) / 0.3081;

    const input = new ort.Tensor('float32', normalized, [1, 1, 28, 28]);
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
