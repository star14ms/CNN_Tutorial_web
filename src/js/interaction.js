const THREE = window.THREE;

export class CameraController {
  constructor(camera, domElement, getScene) {
    this.camera    = camera;
    this.el        = domElement;
    this.getScene  = getScene;

    // Spherical coordinates for orbit
    this._theta    = 0;
    this._phi      = Math.PI / 4;
    this._radius   = 1680;          // scene is 8× larger (×2 again)
    this._target   = new THREE.Vector3(0, -540, 0);

    this._mouse    = { x: 0, y: 0, button: -1 };
    this._raycaster = new THREE.Raycaster();

    this._bind();
    this._applySpherical();
  }

  _bind() {
    const el = this.el;
    el.addEventListener('mousedown', e => this._onDown(e));
    el.addEventListener('mousemove', e => this._onMove(e));
    el.addEventListener('mouseup',   e => this._onUp(e));
    el.addEventListener('mouseleave',() => { this._mouse.button = -1; });
    el.addEventListener('wheel',     e => this._onWheel(e), { passive: false });
    el.addEventListener('contextmenu', e => e.preventDefault());
  }

  _onDown(e) {
    this._mouse.button = e.button;
    this._mouse.x = e.clientX;
    this._mouse.y = e.clientY;
  }

  _onMove(e) {
    if (this._mouse.button === -1) return;
    const dx = e.clientX - this._mouse.x;
    const dy = e.clientY - this._mouse.y;
    this._mouse.x = e.clientX;
    this._mouse.y = e.clientY;

    if (this._mouse.button === 0) {
      // Left drag → orbit
      this._theta -= dx * 0.005;
      this._phi   -= dy * 0.005;
      this._phi = Math.max(0.05, Math.min(Math.PI - 0.05, this._phi));
      this._applySpherical();
    } else if (this._mouse.button === 2) {
      // Right drag → pan (translate target in camera's local XY plane)
      const speed = this._radius * 0.001;
      const right = new THREE.Vector3();
      const up    = new THREE.Vector3();
      this.camera.getWorldDirection(new THREE.Vector3()); // ensure matrix is fresh
      right.crossVectors(
        this.camera.getWorldDirection(new THREE.Vector3()),
        this.camera.up
      ).normalize().negate();
      up.copy(this.camera.up);
      this._target.addScaledVector(right, dx * speed);
      this._target.addScaledVector(up,    dy * speed);
      this._applySpherical();
    }
  }

  _onUp(e) {
    if (e.button === this._mouse.button) this._mouse.button = -1;
  }

  _onWheel(e) {
    e.preventDefault();
    this._radius *= 1 + e.deltaY * 0.001;
    this._radius = Math.max(20, Math.min(6000, this._radius));
    this._applySpherical();
  }

  _applySpherical() {
    const sinPhi = Math.sin(this._phi);
    const cosPhi = Math.cos(this._phi);
    this.camera.position.set(
      this._target.x + this._radius * sinPhi * Math.sin(this._theta),
      this._target.y + this._radius * cosPhi,
      this._target.z + this._radius * sinPhi * Math.cos(this._theta)
    );
    this.camera.lookAt(this._target);
  }
}

export class PixelPicker {
  constructor(camera, domElement, getMeshes, onHit, onMiss) {
    this.camera  = camera;
    this.el      = domElement;
    this.getMeshes = getMeshes;
    this.onHit   = onHit;
    this.onMiss  = onMiss;
    this._raycaster = new THREE.Raycaster();
    this._downPos   = null;

    this.el.addEventListener('mousedown', e => {
      this._downPos = { x: e.clientX, y: e.clientY };
    });
    this.el.addEventListener('mouseup', e => {
      if (!this._downPos) return;
      const dx = e.clientX - this._downPos.x;
      const dy = e.clientY - this._downPos.y;
      // Only fire on short clicks (not drags)
      if (Math.sqrt(dx*dx + dy*dy) < 5) this._pick(e);
      this._downPos = null;
    });
  }

  _pick(e) {
    const rect = this.el.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width)  *  2 - 1,
      ((e.clientY - rect.top)  / rect.height) * -2 + 1
    );
    this._raycaster.setFromCamera(ndc, this.camera);

    const meshes = this.getMeshes();
    const hits = this._raycaster.intersectObjects(meshes, false);
    if (hits.length > 0 && hits[0].uv) {
      if (e.button === 0) this.onHit(hits[0].object, hits[0].uv);
    } else if (this.onMiss) {
      // Left-click or right-click on empty space → deselect
      this.onMiss();
    }
  }
}

export class PixelHover {
  constructor(camera, domElement, getMeshes, onHover) {
    this.camera    = camera;
    this.el        = domElement;
    this.getMeshes = getMeshes;
    this.onHover   = onHover;
    this._raycaster = new THREE.Raycaster();

    this.el.addEventListener('mousemove', e => this._onMove(e));
    this.el.addEventListener('mouseleave', () => this.onHover(null, null));
  }

  _onMove(e) {
    const rect = this.el.getBoundingClientRect();
    const ndc  = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width)  *  2 - 1,
      ((e.clientY - rect.top)  / rect.height) * -2 + 1
    );
    this._raycaster.setFromCamera(ndc, this.camera);
    const hits = this._raycaster.intersectObjects(this.getMeshes(), false);
    if (hits.length > 0 && hits[0].uv) {
      this.onHover(hits[0].object, hits[0].uv);
    } else {
      this.onHover(null, null);
    }
  }
}
