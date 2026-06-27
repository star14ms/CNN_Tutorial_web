import { LayerRenderer } from './layer-renderer.js';

const THREE = window.THREE;

export class Visualization {
  constructor(container, modelConfig) {
    this.container     = container;
    this._modelConfig  = modelConfig;
    this.renderer      = null;
    this.scene         = null;
    this.camera        = null;
    this.layerRenderer = null;
    this._animId       = null;

    this._init();
  }

  /** Switch to a new model config — disposes current layers and rebuilds renderer. */
  setModelConfig(config) {
    this._modelConfig = config;
    this.layerRenderer.setModelConfig(config);
  }

  _init() {
    const W = this.container.clientWidth;
    const H = this.container.clientHeight;

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(W, H);
    this.renderer.setClearColor(0x1a1a2e, 1);
    this.container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();

    // Camera looking from above — PIXEL_SIZE refactor makes scene wider
    this.camera = new THREE.PerspectiveCamera(55, W / H, 0.1, 30000);
    this.camera.position.set(0, 1800, 2800);
    this.camera.lookAt(0, -700, 0);

    // Grid on the "floor" below all layers
    const grid = new THREE.GridHelper(4000, 50, 0x222244, 0x1a1a33);
    grid.position.set(0, -1600, 0);
    this.scene.add(grid);

    this.layerRenderer = new LayerRenderer(this.scene, this._modelConfig);

    window.addEventListener('resize', () => this._resize());
    this._animate();
  }

  _resize() {
    const W = this.container.clientWidth;
    const H = this.container.clientHeight;
    this.renderer.setSize(W, H);
    this.camera.aspect = W / H;
    this.camera.updateProjectionMatrix();
  }

  _animate() {
    this._animId = requestAnimationFrame(() => this._animate());
    this.layerRenderer.updateLabelPositions(this.camera);
    this.renderer.render(this.scene, this.camera);
  }

  update(layerData, inputPixels, animated = false, animDelay = 600) {
    if (animated) {
      this.layerRenderer.renderAnimated(layerData, inputPixels, animDelay);
    } else {
      this.layerRenderer.render(layerData, inputPixels);
    }
  }

  reset() {
    this.layerRenderer.dispose();
  }

  getMeshes() {
    return this.layerRenderer.getMeshes();
  }

  handleRaycastHit(mesh, uv) {
    return this.layerRenderer.handleRaycastHit(mesh, uv);
  }

  showReceptiveField(info) {
    return this.layerRenderer.showReceptiveField(
      info.layerIdx, info.channel, info.x, info.y, info.rawData
    );
  }

  /** Start animated RF; returns { contributing, detail } or null. */
  initRFAnimated(info) {
    return this.layerRenderer.initRFAnimated(
      info.layerIdx, info.channel, info.x, info.y, info.rawData
    );
  }

  addRFLine(p, durationMs) {
    this.layerRenderer.addRFLine(p.li, p.c, p.px, p.py, durationMs);
  }

  showLayerConnections(li) {
    this.layerRenderer.showLayerConnections(li);
  }

  clearReceptiveField() {
    this.layerRenderer.clearReceptiveField();
  }

  highlightRFLine(index) {
    this.layerRenderer.highlightRFLine(index);
  }

  setLayerVisible(li, visible) {
    this.layerRenderer.setLayerVisible(li, visible);
  }

  setHoverPixel(mesh, uv) {
    return this.layerRenderer.setHoverPixel(mesh, uv);
  }

  destroy() {
    cancelAnimationFrame(this._animId);
    this.renderer.dispose();
  }
}
