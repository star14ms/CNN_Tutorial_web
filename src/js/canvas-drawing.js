export class DrawingCanvas {
  constructor(canvasEl, onStrokeEnd) {
    this.canvas = canvasEl;
    this.ctx = canvasEl.getContext('2d');
    this.W = 28;
    this.H = 28;
    // Internal pixel buffer (canvas coords, 28×28)
    this.pixels = new Float32Array(this.W * this.H); // 0=black, 1=white (drawing color)
    this.drawing = false;
    this.onStrokeEnd = onStrokeEnd;
    this._lastX = null;
    this._lastY = null;

    this._initOffscreen();
    this._bindEvents();
    this.clear();
  }

  _initOffscreen() {
    // Offscreen 28×28 canvas for pixel data
    this.off = document.createElement('canvas');
    this.off.width  = this.W;
    this.off.height = this.H;
    this.offCtx = this.off.getContext('2d');
    this.offCtx.fillStyle = '#000';
    this.offCtx.fillRect(0, 0, this.W, this.H);
  }

  _bindEvents() {
    const el = this.canvas;
    el.addEventListener('mousedown', e => { if (e.button === 0) { this.drawing = true; this._lastX = null; this._lastY = null; this._draw(e); } });
    el.addEventListener('mousemove', e => { if (this.drawing) this._draw(e); });
    el.addEventListener('mouseup',   () => { if (this.drawing) { this.drawing = false; this._lastX = null; this._lastY = null; this.onStrokeEnd(); } });
    el.addEventListener('mouseleave',() => { if (this.drawing) { this.drawing = false; this._lastX = null; this._lastY = null; this.onStrokeEnd(); } });

    // Touch support
    el.addEventListener('touchstart', e => { e.preventDefault(); this.drawing = true; this._lastX = null; this._lastY = null; this._draw(e.touches[0]); }, {passive:false});
    el.addEventListener('touchmove',  e => { e.preventDefault(); if (this.drawing) this._draw(e.touches[0]); }, {passive:false});
    el.addEventListener('touchend',   e => { e.preventDefault(); this.drawing = false; this._lastX = null; this._lastY = null; this.onStrokeEnd(); }, {passive:false});
  }

  _clientToCanvas(e) {
    const rect = this.canvas.getBoundingClientRect();
    const scaleX = this.W / rect.width;
    const scaleY = this.H / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top)  * scaleY,
    };
  }

  _draw(e) {
    const { x, y } = this._clientToCanvas(e);
    this.offCtx.fillStyle = '#fff';
    if (this._lastX !== null) {
      // Interpolate dots along the stroke to fill gaps when mouse moves fast
      const dx = x - this._lastX;
      const dy = y - this._lastY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const steps = Math.max(1, Math.ceil(dist / 0.8));
      for (let i = 0; i <= steps; i++) {
        const tx = this._lastX + (dx * i) / steps;
        const ty = this._lastY + (dy * i) / steps;
        this.offCtx.beginPath();
        this.offCtx.arc(tx, ty, 1.2, 0, Math.PI * 2);
        this.offCtx.fill();
      }
    } else {
      this.offCtx.beginPath();
      this.offCtx.arc(x, y, 1.2, 0, Math.PI * 2);
      this.offCtx.fill();
    }
    this._lastX = x;
    this._lastY = y;
    this._syncDisplay();
  }

  _syncDisplay() {
    // Scale up from 28×28 to display canvas
    this.ctx.imageSmoothingEnabled = false;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.drawImage(this.off, 0, 0, this.canvas.width, this.canvas.height);
  }

  clear() {
    this.offCtx.fillStyle = '#000';
    this.offCtx.fillRect(0, 0, this.W, this.H);
    this.ctx.fillStyle = '#111';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.pixels.fill(0);
  }

  // Returns Float32Array(784) normalized as MNIST expects: white digit on black → [0,1]
  getPixels() {
    const imgData = this.offCtx.getImageData(0, 0, this.W, this.H);
    const out = new Float32Array(this.W * this.H);
    for (let i = 0; i < out.length; i++) {
      // Red channel of grayscale image; normalize to [0,1]
      out[i] = imgData.data[i * 4] / 255;
    }
    return out;
  }

  // Returns true if the canvas has any non-zero pixels
  hasContent() {
    const pixels = this.getPixels();
    return pixels.some(v => v > 0.05);
  }
}
