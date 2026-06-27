import { normalizeActivations, gridLayout } from './utils.js';

const THREE = window.THREE;

const PIXEL_SIZE        = 64 / 28; // world units per pixel (all layers except output)
const OUTPUT_PIXEL_SIZE = 64;       // each output cell is larger for label visibility
const GAP           = 4.0;    // gap between channel planes
const LAYER_SPACING = 180;    // Y distance between layers

export class LayerRenderer {
  constructor(scene, modelConfig) {
    this.scene         = scene;
    this._layerDefs    = modelConfig.layerDefs;
    this._connectivity = modelConfig.connectivity;

    this.groups        = [];
    this.meshMeta      = [];
    this.meshByLayer   = [];
    this._labelSprites = [];

    this._rfGroup         = null;
    this._rfLines         = [];   // Line objects (static path only)
    this._rfLineSegs      = null; // Single LineSegments (animated path)
    this._rfLineCount     = 0;
    this._rfLineEndpoints = [];
    this._rfSrcDot        = null; // Source pixel dot (for color change on highlight)
    this._rfDstDots       = [];   // Destination pixel dots indexed by line order
    this._highlightTube   = null;
    this._animTimer       = null;
    this._hoverLine     = null;
    this._hoveredKey    = null;

    this._layerData   = null;
    this._inputPixels = null;
    this._layerVisible = Array(this._layerDefs.length).fill(true);
  }

  /** Replace the active model config and rebuild if data is present. */
  setModelConfig(modelConfig) {
    this._layerDefs    = modelConfig.layerDefs;
    this._connectivity = modelConfig.connectivity;
    this._layerVisible = Array(this._layerDefs.length).fill(true);
    this.dispose();
  }

  dispose() {
    this._clearRF();
    this._stopAnim();
    this._clearHover();
    for (const group of this.groups) {
      if (!group) continue;
      group.traverse(obj => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          if (obj.material.map) obj.material.map.dispose();
          obj.material.dispose();
        }
      });
      this.scene.remove(group);
    }
    this.groups        = [];
    this.meshMeta      = [];
    this.meshByLayer   = [];
    this._labelSprites = [];
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  render(layerData, inputPixels) {
    this._layerData   = layerData;
    this._inputPixels = inputPixels;
    this.dispose();
    this._buildAllLayers();
  }

  renderAnimated(layerData, inputPixels, delayMs = 600) {
    this._layerData   = layerData;
    this._inputPixels = inputPixels;
    this.dispose();
    let i = 0;
    const payloads = this._buildLayerPayloads();
    const next = () => {
      if (i >= payloads.length) return;
      this._buildLayer(payloads[i], i);
      i++;
      this._animTimer = setTimeout(next, delayMs);
    };
    next();
  }

  setLayerVisible(li, visible) {
    this._layerVisible[li] = visible;
    if (this.groups[li]) this.groups[li].visible = visible;
  }

  getLayerCount() { return this._layerDefs.length; }

  getLayerName(li) { return this._layerDefs[li]?.name ?? ''; }

  /** Call every frame — keeps labels at the far edge away from camera. */
  updateLabelPositions(camera) {
    const cx  = camera.position.x;
    const cz  = camera.position.z;
    const len = Math.sqrt(cx * cx + cz * cz) || 1;
    const ux  = cx / len;
    const uz  = cz / len;

    for (let li = 0; li < this._labelSprites.length; li++) {
      const sprite = this._labelSprites[li];
      const group  = this.groups[li];
      if (!sprite || !group) continue;
      const { totalW, totalD } = group.userData;
      // Push label outside the layer boundary in the camera direction, with a fixed min
      const halfW = totalW / 2;
      const halfD = totalD / 2;
      const margin = Math.max(Math.abs(ux) * halfW + Math.abs(uz) * halfD + 30, 80);
      sprite.position.set(
        halfW - ux * margin,
        8,
        halfD - uz * margin
      );
    }
  }

  setHoverPixel(mesh, uv) {
    if (!mesh || !uv) {
      if (this._hoverLine) this._hoverLine.visible = false;
      this._hoveredKey = null;
      return null;
    }
    const ud  = mesh.userData;
    if (!ud.layerName) return null;
    const px  = Math.min(Math.floor(uv.x * ud.w), ud.w - 1);
    const py  = Math.min(Math.floor((1 - uv.y) * ud.h), ud.h - 1);
    const key = `${ud.layerIdx},${ud.channelIdx},${px},${py}`;
    if (key !== this._hoveredKey) {
      this._hoveredKey = key;
      this._updateHoverHighlight(mesh, px, py, ud.w, ud.h);
    }
    return {
      layerName: ud.layerName,
      channel:   ud.channelIdx,
      x: px, y: py,
      rawValue:  Number(ud.rawData[ud.offset + py * ud.w + px]),
    };
  }

  showReceptiveField(li, c, px, py, rawData) {
    this._clearRF();
    const srcPos = this._getPixelWorldPos(li, c, px, py);
    if (!srcPos) return null;

    const contributing = this._getContributingPixels(li, c, px, py);

    const group  = new THREE.Group();
    const dotGeo = new THREE.SphereGeometry(0.5, 6, 6);
    const srcDot = new THREE.Mesh(dotGeo, new THREE.MeshBasicMaterial({ color: 0xffdd44 }));
    srcDot.position.copy(srcPos);
    group.add(srcDot);
    this._rfSrcDot = srcDot;

    this._rfLines   = [];
    this._rfDstDots = [];
    if (contributing.length > 0) {
      for (const p of contributing) {
        const dstPos = this._getPixelWorldPos(p.li, p.c, p.px, p.py);
        if (!dstPos) continue;
        const mat  = new THREE.LineBasicMaterial({ color: 0xffdd44, transparent: true, opacity: 0.7 });
        const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([srcPos.clone(), dstPos]), mat);
        this._rfLines.push(line);
        group.add(line);
        const dot = new THREE.Mesh(new THREE.SphereGeometry(0.5, 6, 6), new THREE.MeshBasicMaterial({ color: 0xffdd44 }));
        dot.position.copy(dstPos);
        group.add(dot);
        this._rfDstDots.push(dot);
      }
    }

    this.scene.add(group);
    this._rfGroup = group;
    return this._buildDetailInfo(li, c, px, py, rawData, contributing);
  }

  /** Start animated RF: draw source dot only; returns { contributing, detail }. */
  initRFAnimated(li, c, px, py, rawData) {
    this._clearRF();
    const srcPos = this._getPixelWorldPos(li, c, px, py);
    if (!srcPos) return null;

    const group  = new THREE.Group();
    const srcDot = new THREE.Mesh(new THREE.SphereGeometry(0.5, 6, 6), new THREE.MeshBasicMaterial({ color: 0xffdd44 }));
    srcDot.position.copy(srcPos);
    group.add(srcDot);
    this._rfSrcDot = srcDot;

    // Pre-allocate a single LineSegments buffer for all animated lines (1 draw call).
    const MAX_LINES = 2000;
    const posArr   = new Float32Array(MAX_LINES * 6); // 2 verts × 3 floats each
    const colorArr = new Float32Array(MAX_LINES * 6); // 2 verts × RGB each
    // Default color: yellow (0xffdd44 → 1, 0.867, 0.267)
    for (let i = 0; i < MAX_LINES * 6; i += 3) {
      colorArr[i] = 1.0; colorArr[i + 1] = 0.867; colorArr[i + 2] = 0.267;
    }
    const lsGeo = new THREE.BufferGeometry();
    lsGeo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
    lsGeo.setAttribute('color',    new THREE.BufferAttribute(colorArr, 3));
    lsGeo.setDrawRange(0, 0);
    const lsMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.7 });
    this._rfLineSegs                  = new THREE.LineSegments(lsGeo, lsMat);
    this._rfLineSegs.frustumCulled    = false; // prevent disappearing during camera rotation
    this._rfLineCount     = 0;
    this._rfLineEndpoints = [];
    group.add(this._rfLineSegs);

    this.scene.add(group);
    this._rfGroup  = group;
    this._rfSrcPos = srcPos;
    this._rfLines   = [];
    this._rfDstDots = [];

    const contributing = this._getContributingPixels(li, c, px, py);
    return { contributing, detail: this._buildDetailInfo(li, c, px, py, rawData, contributing) };
  }

  /** Add one RF line with animated drawing from source to destination. */
  addRFLine(prevLi, prevC, prevPx, prevPy, durationMs = 200) {
    if (!this._rfGroup || !this._rfSrcPos || !this._rfLineSegs) return;
    const dstPos = this._getPixelWorldPos(prevLi, prevC, prevPx, prevPy);
    if (!dstPos) return;
    const src     = this._rfSrcPos.clone();
    const lineIdx = this._rfLineCount++;
    this._rfLineEndpoints.push({ src: src.clone(), dst: dstPos.clone() });

    const geo     = this._rfLineSegs.geometry;
    const posAttr = geo.attributes.position;
    const base    = lineIdx * 6;
    // Write src vertex immediately so segment is allocated
    posAttr.array[base]     = src.x;
    posAttr.array[base + 1] = src.y;
    posAttr.array[base + 2] = src.z;

    const startTime = performance.now();
    const tick = () => {
      if (!this._rfGroup) return;
      const t   = Math.min(1, (performance.now() - startTime) / durationMs);
      const cur = src.clone().lerp(dstPos, t);
      posAttr.array[base + 3] = cur.x;
      posAttr.array[base + 4] = cur.y;
      posAttr.array[base + 5] = cur.z;
      posAttr.needsUpdate     = true;
      // Expand draw range to reveal this segment (max of current range and this segment's end)
      const needed = (lineIdx + 1) * 2;
      if (geo.drawRange.count < needed) geo.setDrawRange(0, needed);
      if (t < 1) {
        requestAnimationFrame(tick);
      } else {
        const dot = new THREE.Mesh(new THREE.SphereGeometry(0.5, 6, 6), new THREE.MeshBasicMaterial({ color: 0xffdd44 }));
        dot.position.copy(dstPos);
        if (this._rfGroup) this._rfGroup.add(dot);
        this._rfDstDots[lineIdx] = dot;
      }
    };
    requestAnimationFrame(tick);
  }

  /** Show all representative connections for an entire layer (center pixel per channel). */
  showLayerConnections(li) {
    this._clearRF();
    const conn = this._connectivity[li];
    if (!conn) {
      if (li > 0) this._showDenseConnections(li);
      return;
    }
    const def   = this._layerDefs[li];
    const group = new THREE.Group();
    const dotGeo      = new THREE.SphereGeometry(0.5, 6, 6);
    const cornerMat   = new THREE.LineBasicMaterial({ color: 0x66aaff, transparent: true, opacity: 0.45 });
    const cornerDotMat = new THREE.MeshBasicMaterial({ color: 0x66aaff });
    const yellowMat   = new THREE.LineBasicMaterial({ color: 0xffdd44, transparent: true, opacity: 0.35 });
    const yellowDotMat = new THREE.MeshBasicMaterial({ color: 0xffdd44 });
    // Non-pool layer connection lines use yellow to match individual-pixel selection color
    const lineMat = new THREE.LineBasicMaterial({ color: 0xffdd44, transparent: true, opacity: 0.45 });
    const dotMat  = new THREE.MeshBasicMaterial({ color: 0xffdd44 });

    // Pool layers: draw 4 corner-to-corner lines (blue) + per-pixel lines (yellow)
    if (conn.type === 'pool') {
      const curGroup  = this.groups[li];
      const prevGroup = this.groups[conn.prevLi];
      if (curGroup && prevGroup) {
        const cp = curGroup.position;
        const { totalW: cW, totalD: cD } = curGroup.userData;
        const pp = prevGroup.position;
        const { totalW: pW, totalD: pD } = prevGroup.userData;
        const curCorners  = [
          new THREE.Vector3(cp.x,      cp.y, cp.z),
          new THREE.Vector3(cp.x + cW, cp.y, cp.z),
          new THREE.Vector3(cp.x + cW, cp.y, cp.z + cD),
          new THREE.Vector3(cp.x,      cp.y, cp.z + cD),
        ];
        const prevCorners = [
          new THREE.Vector3(pp.x,      pp.y, pp.z),
          new THREE.Vector3(pp.x + pW, pp.y, pp.z),
          new THREE.Vector3(pp.x + pW, pp.y, pp.z + pD),
          new THREE.Vector3(pp.x,      pp.y, pp.z + pD),
        ];
        // Blue corner lines
        for (let i = 0; i < 4; i++) {
          group.add(new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([curCorners[i], prevCorners[i]]),
            cornerMat
          ));
          const d1 = new THREE.Mesh(dotGeo, cornerDotMat); d1.position.copy(curCorners[i]);
          const d2 = new THREE.Mesh(dotGeo, cornerDotMat); d2.position.copy(prevCorners[i]);
          group.add(d1); group.add(d2);
        }
        // Yellow per-pixel lines: every pixel in pool layer → its source pixels in prev layer
        const poolDef = this._layerDefs[li];
        for (let c = 0; c < poolDef.channels; c++) {
          for (let py = 0; py < poolDef.h; py++) {
            for (let px = 0; px < poolDef.w; px++) {
              const srcPos = this._getPixelWorldPos(li, c, px, py);
              if (!srcPos) continue;
              const srcDot = new THREE.Mesh(dotGeo, yellowDotMat);
              srcDot.position.copy(srcPos);
              group.add(srcDot);
              const contributing = this._getContributingPixels(li, c, px, py);
              for (const p of contributing) {
                const dstPos = this._getPixelWorldPos(p.li, p.c, p.px, p.py);
                if (!dstPos) continue;
                group.add(new THREE.Line(
                  new THREE.BufferGeometry().setFromPoints([srcPos.clone(), dstPos]),
                  yellowMat
                ));
                const dstDot = new THREE.Mesh(dotGeo, yellowDotMat);
                dstDot.position.copy(dstPos);
                group.add(dstDot);
              }
            }
          }
        }
      }
      this.scene.add(group);
      this._rfGroup = group;
      return;
    }

    // For FC layers: show all neurons; for conv: show center pixel per channel
    const isFc = conn.type === 'fc';
    const sources = [];
    if (isFc) {
      for (let c = 0; c < def.channels; c++)
        for (let py = 0; py < def.h; py++)
          for (let px = 0; px < def.w; px++)
            sources.push({ c, px, py });
    } else {
      const cx = Math.floor(def.w / 2);
      const cy = Math.floor(def.h / 2);
      for (let c = 0; c < def.channels; c++)
        sources.push({ c, px: cx, py: cy });
    }

    for (const { c, px, py } of sources) {
      const srcPos = this._getPixelWorldPos(li, c, px, py);
      if (!srcPos) continue;
      const srcDot = new THREE.Mesh(dotGeo, dotMat);
      srcDot.position.copy(srcPos);
      group.add(srcDot);

      const contributing = this._getContributingPixels(li, c, px, py);
      for (const p of contributing) {
        const dstPos = this._getPixelWorldPos(p.li, p.c, p.px, p.py);
        if (!dstPos) continue;
        group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([srcPos.clone(), dstPos]), lineMat));
        const dstDot = new THREE.Mesh(dotGeo, dotMat);
        dstDot.position.copy(dstPos);
        group.add(dstDot);
      }
    }

    this.scene.add(group);
    this._rfGroup = group;
  }

  /** Dense connections for FC / Output layers that have no spatial connectivity. */
  _showDenseConnections(li) {
    const prevLi  = li - 1;
    const def     = this._layerDefs[li];
    const prevDef = this._layerDefs[prevLi];
    const isOutputLayer = li === this._layerDefs.length - 1;

    const group   = new THREE.Group();
    const lineMat = new THREE.LineBasicMaterial({ color: 0x66aaff, transparent: true, opacity: 0.3 });
    const dotMat  = new THREE.MeshBasicMaterial({ color: 0x66aaff });
    const dotGeo  = new THREE.SphereGeometry(0.5, 6, 6);

    // "from" positions in current layer
    const fromPoses = [];
    if (isOutputLayer) {
      for (let c = 0; c < def.channels; c++) {
        const pos = this._getPixelWorldPos(li, c, 0, 0);
        if (pos) fromPoses.push(pos);
      }
    } else {
      // FC: sample every 8th neuron along the strip
      for (let x = 0; x < def.w; x += 8) {
        const pos = this._getPixelWorldPos(li, 0, x, 0);
        if (pos) fromPoses.push(pos);
      }
    }

    // "to" positions in previous layer
    const toPoses = [];
    if (isOutputLayer) {
      // Output → FC: sample FC neurons
      const step = Math.max(1, Math.floor(prevDef.w / 30));
      for (let x = 0; x < prevDef.w && toPoses.length < 30; x += step) {
        const pos = this._getPixelWorldPos(prevLi, 0, x, 0);
        if (pos) toPoses.push(pos);
      }
    } else {
      // FC → Pool/Conv: center of each prev channel
      for (let c = 0; c < prevDef.channels; c++) {
        const pos = this._getPixelWorldPos(prevLi, c,
          Math.floor(prevDef.w / 2), Math.floor(prevDef.h / 2));
        if (pos) toPoses.push(pos);
      }
    }

    let lineCount = 0;
    const MAX_LINES = 2000;
    for (const fromPos of fromPoses) {
      if (lineCount >= MAX_LINES) break;
      const dot = new THREE.Mesh(dotGeo, dotMat);
      dot.position.copy(fromPos);
      group.add(dot);
      for (const toPos of toPoses) {
        if (lineCount >= MAX_LINES) break;
        group.add(new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([fromPos.clone(), toPos.clone()]),
          lineMat
        ));
        lineCount++;
      }
    }
    for (const toPos of toPoses) {
      const dot = new THREE.Mesh(dotGeo, dotMat);
      dot.position.copy(toPos);
      group.add(dot);
    }

    this.scene.add(group);
    this._rfGroup = group;
  }

  clearReceptiveField() { this._clearRF(); }

  /** Highlight one RF line red + thick; pass index=-1 to restore all to yellow. */
  highlightRFLine(index) {
    // Remove previous highlight tube
    if (this._highlightTube) {
      this._highlightTube.geometry.dispose();
      this._highlightTube.material.dispose();
      if (this._rfGroup) this._rfGroup.remove(this._highlightTube);
      this._highlightTube = null;
    }

    // Animated path: vertex colors on single LineSegments
    if (this._rfLineSegs) {
      const colorAttr = this._rfLineSegs.geometry.attributes.color;
      const n = this._rfLineCount;
      for (let i = 0; i < n * 2; i++) {
        colorAttr.setXYZ(i, 1.0, 0.867, 0.267); // yellow
      }
      if (this._rfSrcDot) this._rfSrcDot.material.color.setHex(index >= 0 ? 0xff3333 : 0xffdd44);
      // Reset all dst dots to yellow
      for (const d of this._rfDstDots) if (d) d.material.color.setHex(0xffdd44);
      if (index >= 0 && index < n) {
        colorAttr.setXYZ(index * 2,     1.0, 0.2, 0.2); // red
        colorAttr.setXYZ(index * 2 + 1, 1.0, 0.2, 0.2);
        if (this._rfDstDots[index]) this._rfDstDots[index].material.color.setHex(0xff3333);
        // Tube for thickness on highlighted segment (inset endpoints to avoid penetrating image planes)
        const ep = this._rfLineEndpoints[index];
        if (ep && this._rfGroup) {
          const p1 = ep.src.clone();
          const p2 = ep.dst.clone();
          const dist = p1.distanceTo(p2);
          if (dist > 0.1) {
            const dir = p2.clone().sub(p1).normalize();
            const inset = Math.min(4, dist * 0.1);
            p1.addScaledVector(dir,  inset);
            p2.addScaledVector(dir, -inset);
            const curve   = new THREE.LineCurve3(p1, p2);
            const tubeGeo = new THREE.TubeGeometry(curve, 1, 1, 6, false);
            const tubeMat = new THREE.MeshBasicMaterial({ color: 0xff3333, transparent: true, opacity: 0.9 });
            this._highlightTube = new THREE.Mesh(tubeGeo, tubeMat);
            this._rfGroup.add(this._highlightTube);
          }
        }
      }
      colorAttr.needsUpdate = true;
      return;
    }

    // Static path: individual line materials
    if (this._rfSrcDot) this._rfSrcDot.material.color.setHex(index >= 0 ? 0xff3333 : 0xffdd44);
    for (const d of this._rfDstDots) if (d) d.material.color.setHex(0xffdd44);
    for (const line of this._rfLines) {
      line.material.color.setHex(0xffdd44);
      line.material.opacity = 0.7;
      line.material.needsUpdate = true;
    }
    if (index >= 0 && index < this._rfLines.length) {
      this._rfLines[index].material.color.setHex(0xff3333);
      this._rfLines[index].material.opacity = 1.0;
      this._rfLines[index].material.needsUpdate = true;
      if (this._rfDstDots[index]) this._rfDstDots[index].material.color.setHex(0xff3333);

      const pos = this._rfLines[index].geometry.attributes.position;
      if (pos && pos.count >= 2 && this._rfGroup) {
        const p1 = new THREE.Vector3(pos.getX(0), pos.getY(0), pos.getZ(0));
        const p2 = new THREE.Vector3(pos.getX(1), pos.getY(1), pos.getZ(1));
        const dist = p1.distanceTo(p2);
        if (dist > 0.1) {
          const dir = p2.clone().sub(p1).normalize();
          const inset = Math.min(4, dist * 0.1);
          p1.addScaledVector(dir,  inset);
          p2.addScaledVector(dir, -inset);
          const curve   = new THREE.LineCurve3(p1, p2);
          const tubeGeo = new THREE.TubeGeometry(curve, 1, 1, 6, false);
          const tubeMat = new THREE.MeshBasicMaterial({ color: 0xff3333, transparent: true, opacity: 0.9 });
          this._highlightTube = new THREE.Mesh(tubeGeo, tubeMat);
          this._rfGroup.add(this._highlightTube);
        }
      }
    }
  }

  getMeshes() { return this.meshMeta.map(m => m.mesh); }

  handleRaycastHit(mesh, uv) {
    const ud = mesh.userData;
    if (!ud.layerName) return null;
    const px  = Math.floor(uv.x * ud.w);
    const py  = Math.floor((1 - uv.y) * ud.h);
    const idx = ud.offset + py * ud.w + px;
    return {
      layerIdx:        ud.layerIdx,
      layerName:       ud.layerName,
      channel:         ud.channelIdx,
      x: px, y: py,
      rawValue:        Number(ud.rawData[idx]),
      normalizedValue: Number(ud.normalizedData[idx]),
      rawData:         ud.rawData,
    };
  }

  // ── Build layers ───────────────────────────────────────────────────────────

  _buildLayerPayloads() {
    const ld = this._layerData;
    const ip = this._inputPixels;
    return this._layerDefs.map((def, li) => {
      let data;
      if (li === 0) {
        data = ip;
      } else if (def.dataKey) {
        data = ld?.[def.dataKey]?.data;
      } else {
        // Fallback positional map for configs without dataKey (7-layer legacy)
        const keys = ['layer0','layer1','layer2','layer3','layer4','output'];
        data = ld?.[keys[li - 1]]?.data;
      }
      return { def, data: data ?? new Float32Array(def.channels * def.h * def.w) };
    });
  }

  _buildAllLayers() {
    const payloads = this._buildLayerPayloads();
    for (let li = 0; li < payloads.length; li++) this._buildLayer(payloads[li], li);
  }

  _buildLayer(payload, li) {
    const { def, data } = payload;
    const C = def.channels;
    const H = def.h;
    const W = def.w;

    const isOutput = li === this._layerDefs.length - 1;
    const { cols, rows } = isOutput ? { cols: C, rows: 1 } : gridLayout(C);

    const pxSz  = isOutput ? OUTPUT_PIXEL_SIZE : PIXEL_SIZE;
    const planeW = pxSz * W;
    const planeH = pxSz * H;
    const layerY = -li * LAYER_SPACING;
    const totalW = cols * (planeW + GAP);
    const totalD = rows * (planeH + GAP);

    const group = new THREE.Group();
    group.userData.layerName = def.name;
    group.userData.totalW    = totalW;
    group.userData.totalD    = totalD;
    group.position.set(-totalW / 2, layerY, -totalD / 2);

    const { normalized } = normalizeActivations(Array.from(data));
    if (!this.meshByLayer[li]) this.meshByLayer[li] = {};

    for (let c = 0; c < C; c++) {
      const col    = c % cols;
      const row    = Math.floor(c / cols);
      const cx     = col * (planeW + GAP) + planeW / 2;
      const cz     = row * (planeH + GAP) + planeH / 2;
      const offset = c * H * W;

      const texData = new Uint8Array(W * H * 4);
      for (let py2 = 0; py2 < H; py2++) {
        for (let px2 = 0; px2 < W; px2++) {
          const srcI = offset + py2 * W + px2;
          // Flip Y so row py2=0 appears at visual top (DataTexture V=0 is bottom)
          const dstI = (H - 1 - py2) * W + px2;
          const bv = Math.round(normalized[srcI] * 255);
          texData[dstI * 4]     = bv;
          texData[dstI * 4 + 1] = bv;
          texData[dstI * 4 + 2] = bv;
          texData[dstI * 4 + 3] = 255;
        }
      }
      const tex = new THREE.DataTexture(texData, W, H);
      tex.needsUpdate = true;
      tex.magFilter   = THREE.NearestFilter;
      tex.minFilter   = THREE.NearestFilter;

      const geo  = new THREE.PlaneGeometry(planeW, planeH);
      const mat  = new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.set(cx, 0, cz);

      mesh.add(new THREE.LineSegments(
        new THREE.EdgesGeometry(geo),
        new THREE.LineBasicMaterial({ color: 0x444466 })
      ));

      mesh.userData = {
        layerIdx: li, layerName: def.name, channelIdx: c,
        h: H, w: W, planeW, planeH, rawData: data, normalizedData: normalized, offset,
      };

      group.add(mesh);
      this.meshMeta.push({ mesh, layerIdx: li, channelIdx: c, h: H, w: W });
      this.meshByLayer[li][c] = mesh;

      if (def.channelLabels) {
        this._addCellLabel(group, def.channelLabels[c], data[c], cx, cz, planeW, planeH, isOutput);
      }
    }

    const sprite = this._addLayerLabel(group, def.name, totalW, isOutput);
    while (this._labelSprites.length <= li) this._labelSprites.push(null);
    this._labelSprites[li] = sprite;

    group.visible = this._layerVisible[li] !== false;
    this.scene.add(group);

    while (this.groups.length <= li) this.groups.push(null);
    this.groups[li] = group;
  }

  _addLayerLabel(group, name, totalW, isOutput = false) {
    const canvas  = document.createElement('canvas');
    canvas.width  = 1024;
    canvas.height = 160;
    const ctx     = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.font      = 'bold 104px monospace';
    ctx.fillStyle = '#eeeeee';
    ctx.textAlign = 'center';
    ctx.fillText(name, 512, 112);
    const tex    = new THREE.CanvasTexture(canvas);
    const mat    = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
    const sprite = new THREE.Sprite(mat);
    const aspect = canvas.height / canvas.width;
    // Output layer has a wider content area; scale label proportionally so it appears the same size
    const spriteW = isOutput ? Math.round(totalW * 0.22) : 140;
    sprite.scale.set(spriteW, spriteW * aspect, 1);
    sprite.position.set(totalW / 2, 8, -4);
    group.add(sprite);
    return sprite;
  }

  _addCellLabel(group, digit, prob, cx, cz, planeW, planeH, isOutput = false) {
    const sz     = isOutput ? 512 : 256;
    const canvas = document.createElement('canvas');
    canvas.width  = sz;
    canvas.height = sz;
    const ctx    = canvas.getContext('2d');
    ctx.clearRect(0, 0, sz, sz);
    ctx.font      = isOutput ? `bold ${Math.round(sz * 0.50)}px monospace` : 'bold 120px monospace';
    ctx.fillStyle = '#ffdd44';
    ctx.textAlign = 'center';
    ctx.fillText(digit, sz / 2, Math.round(sz * 0.54));
    const pct = (prob * 100).toFixed(1);
    ctx.font      = isOutput ? `${Math.round(sz * 0.20)}px monospace` : '60px monospace';
    ctx.fillStyle = '#aaaaaa';
    ctx.fillText(pct + '%', sz / 2, Math.round(sz * 0.82));
    const tex    = new THREE.CanvasTexture(canvas);
    const mat    = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(planeW * 0.9, planeW * 0.9, 1);
    // Position above the cell: +Z moves "up" in screen space for this camera angle
    sprite.position.set(cx, 6, cz + planeH);
    group.add(sprite);
  }

  // ── Hover highlight ────────────────────────────────────────────────────────

  _updateHoverHighlight(mesh, px, py, W, H) {
    if (!this._hoverLine) {
      const mat = new THREE.LineBasicMaterial({
        color: 0xffff00, depthTest: false, depthWrite: false, transparent: true, opacity: 0.95,
      });
      this._hoverLine = new THREE.LineLoop(new THREE.BufferGeometry(), mat);
      this._hoverLine.renderOrder = 999;
      this.scene.add(this._hoverLine);
    }
    const { planeW: pw, planeH: ph } = mesh.userData;
    const u1 = px / W,      u2 = (px + 1) / W;
    const v1 = 1 - py / H,  v2 = 1 - (py + 1) / H;
    const lx1 = (u1 - 0.5) * pw, lx2 = (u2 - 0.5) * pw;
    const ly1 = (v1 - 0.5) * ph, ly2 = (v2 - 0.5) * ph;
    const EPS = 3.0; // large enough to stay above the plane at any zoom
    const corners = [
      new THREE.Vector3(lx1, ly1, EPS),
      new THREE.Vector3(lx2, ly1, EPS),
      new THREE.Vector3(lx2, ly2, EPS),
      new THREE.Vector3(lx1, ly2, EPS),
    ].map(c => { mesh.localToWorld(c); return c; });
    this._hoverLine.geometry.setFromPoints(corners);
    this._hoverLine.visible = true;
  }

  // ── Receptive field ────────────────────────────────────────────────────────

  _getContributingPixels(li, c, px, py) {
    const conn = this._connectivity[li];
    if (!conn) return [];
    const prevDef = this._layerDefs[conn.prevLi];
    const pixels  = [];

    if (conn.type === 'conv') {
      const half = Math.floor(conn.kernel / 2);
      // Each output pixel reads from ALL input channels (standard CNN behaviour)
      for (let prevC = 0; prevC < conn.prevChannels; prevC++) {
        for (let dy = -half; dy <= half; dy++) {
          for (let dx = -half; dx <= half; dx++) {
            const ppx = px + dx, ppy = py + dy;
            if (ppx >= 0 && ppx < prevDef.w && ppy >= 0 && ppy < prevDef.h)
              pixels.push({ li: conn.prevLi, c: prevC, px: ppx, py: ppy });
          }
        }
      }
    } else if (conn.type === 'pool') {
      const k = conn.kernel;
      const prevC = c; // pool preserves channels
      for (let dy = 0; dy < k; dy++)
        for (let dx = 0; dx < k; dx++)
          pixels.push({ li: conn.prevLi, c: prevC, px: px * k + dx, py: py * k + dy });
    } else if (conn.type === 'fc') {
      const prevDef = this._layerDefs[conn.prevLi];
      if (conn.mode === 'center') {
        // One representative pixel per channel (spatial center)
        const cpx = Math.floor(prevDef.w / 2);
        const cpy = Math.floor(prevDef.h / 2);
        for (let prevC = 0; prevC < prevDef.channels; prevC++)
          pixels.push({ li: conn.prevLi, c: prevC, px: cpx, py: cpy });
      } else {
        // All pixels with optional stride
        const step = conn.sampleStep ?? 1;
        for (let prevC = 0; prevC < prevDef.channels; prevC++)
          for (let idy = 0; idy < prevDef.h; idy++)
            for (let idx = 0; idx < prevDef.w; idx += step)
              pixels.push({ li: conn.prevLi, c: prevC, px: idx, py: idy });
      }
    }
    return pixels;
  }

  _getPixelWorldPos(li, c, px, py) {
    const mesh = this.meshByLayer[li]?.[c];
    if (!mesh) return null;
    const { h, w, planeW, planeH } = mesh.userData;
    const u  = (px + 0.5) / w;
    const v  = 1 - (py + 0.5) / h;
    const pt = new THREE.Vector3((u - 0.5) * planeW, (v - 0.5) * planeH, 0);
    mesh.localToWorld(pt);
    return pt;
  }

  _buildDetailInfo(li, c, px, py, rawData, contributing) {
    const conn   = this._connectivity[li];
    const def    = this._layerDefs[li];
    const idx    = c * def.h * def.w + py * def.w + px;
    const outVal = rawData ? Number(rawData[idx]) : null;

    const prevValues = contributing.map(({ li: pli, c: pc, px: ppx, py: ppy }) => {
      const prevDef  = this._layerDefs[pli];
      const prevData = this._getRawData(pli);
      const val = prevData
        ? Number(prevData[pc * prevDef.h * prevDef.w + ppy * prevDef.w + ppx])
        : null;
      return { px: ppx, py: ppy, c: pc, val };
    });

    return { layerName: def.name, channel: c, px, py, outVal, conn, prevValues };
  }

  _getRawData(li) {
    const ld = this._layerData, ip = this._inputPixels;
    const map = [ip, ld?.layer0?.data, ld?.layer1?.data, ld?.layer2?.data,
                     ld?.layer3?.data, ld?.layer4?.data, ld?.output?.data];
    return map[li] ?? null;
  }

  // ── Cleanup ────────────────────────────────────────────────────────────────

  _clearRF() {
    this._rfLines         = [];
    this._rfLineSegs      = null;
    this._rfLineCount     = 0;
    this._rfLineEndpoints = [];
    this._rfSrcDot        = null;
    this._rfDstDots       = [];
    if (this._highlightTube) {
      this._highlightTube.geometry.dispose();
      this._highlightTube.material.dispose();
      this._highlightTube = null;
    }
    if (this._rfGroup) {
      this._rfGroup.traverse(o => {
        if (o.geometry) o.geometry.dispose();
        if (o.material) o.material.dispose();
      });
      this.scene.remove(this._rfGroup);
      this._rfGroup = null;
    }
  }

  _clearHover() {
    if (this._hoverLine) {
      this._hoverLine.geometry.dispose();
      this._hoverLine.material.dispose();
      this.scene.remove(this._hoverLine);
      this._hoverLine = null;
    }
    this._hoveredKey = null;
  }

  _stopAnim() {
    if (this._animTimer !== null) { clearTimeout(this._animTimer); this._animTimer = null; }
  }
}
