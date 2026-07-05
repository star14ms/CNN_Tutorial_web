import { normalizeActivations, gridLayout } from './utils.js';

const THREE = window.THREE;

/** Decode a parameter entry — supports base64 binary and plain JS arrays (legacy). */
function _decodeParams(entry) {
  if (!entry) return entry;
  const decode = v => {
    if (typeof v === 'string') {
      const bin = atob(v);
      const buf = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
      return new Float32Array(buf.buffer);
    }
    return v; // already an array (legacy format)
  };
  return { ...entry, data: decode(entry.data), bias: entry.bias ? decode(entry.bias) : undefined };
}

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
    this._rfEpoch         = 0;    // Incremented on each _clearRF; old addRFLine ticks check this
    this._rfLineOpacities = [];   // Per-line opacity stored by showReceptiveField for restore
    this._rfBaseColors    = [];   // Per-line {r,g,b} stored by addRFLine for restore
    this._animTimer       = null;
    this._convLayerWeights = {};  // li → { shape, data, bias? } from setParameters (conv layers)
    this._linearParams     = {};  // li → { shape, data, bias? } from setParameters (fc layers)
    this._hoverLine     = null;
    this._hoveredKey    = null;

    this._layerData   = null;
    this._inputPixels = null;
    this._layerVisible = Array(this._layerDefs.length).fill(true);
  }

  /** Store dataset config for resolving class labels and input shape. */
  setDatasetConfig(datasetConfig) {
    this._classLabels  = datasetConfig.classLabels || ['0','1','2','3','4','5','6','7','8','9'];
    this._inChannels   = datasetConfig.inChannels ?? 1;
    this._inputImgSize = datasetConfig.imgSize ?? 28;
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

  /** Per-layer animated inference: input visible → connection lines 2s → reveal layer → repeat. */
  renderAnimatedPerLayer(layerData, inputPixels, speedMs = 100) {
    this._layerData      = layerData;
    this._inputPixels    = inputPixels;
    this._perLayerSpeedMs = speedMs;
    this._perLayerNextFn  = null;
    this.dispose();
    this._layerAnimEpoch = (this._layerAnimEpoch || 0) + 1;
    const animEpoch = this._layerAnimEpoch;

    // Build all layers; layers 1+ start visible but transparent (will fade in)
    const payloads = this._buildLayerPayloads();
    payloads.forEach((p, li) => {
      this._buildLayer(p, li);
      if (li > 0) {
        if (this.groups[li])        { this.groups[li].visible = true; }
        if (this._labelSprites[li]) { this._labelSprites[li].visible = true; }
        this._setGroupOpacity(li, 0);
      }
    });

    let i = 1;
    const next = () => {
      if (this._layerAnimEpoch !== animEpoch) return;
      if (i >= payloads.length) { this._perLayerNextFn = null; return; }
      const li = i;
      this._perLayerNextFn = null;  // now in rAF phase, not waiting
      // Fade in this layer's images concurrently with line drawing
      this._fadeInGroup(li, animEpoch);
      // Draw connection lines; when fully drawn, fade them out then advance
      this._showLayerConnectionsAnimated(li, () => {
        if (this._layerAnimEpoch !== animEpoch) return;
        this._fadeOutRF(() => {
          if (this._layerAnimEpoch !== animEpoch) return;
          i++;
          // Store next fn so speed slider can reschedule this wait
          const scheduleNext = () => {
            const waitMs = 1000 * (this._perLayerSpeedMs / 100);
            this._perLayerNextFn = next;
            this._animTimer = setTimeout(next, waitMs);
          };
          scheduleNext();
        });
      });
    };
    // Initial wait
    const initWait = 1000 * (this._perLayerSpeedMs / 100);
    this._perLayerNextFn = next;
    this._animTimer = setTimeout(next, initWait);
  }

  /** Update speed mid-animation: rAF ticks read _perLayerSpeedMs dynamically;
   *  if currently in a setTimeout wait, cancel and reschedule with the new speed. */
  setPerLayerSpeed(speedMs) {
    this._perLayerSpeedMs = speedMs;
    if (this._animTimer !== null && this._perLayerNextFn) {
      clearTimeout(this._animTimer);
      this._animTimer = null;
      const fn = this._perLayerNextFn;
      const newWait = 1000 * (speedMs / 100);
      this._animTimer = setTimeout(fn, newWait);
    }
  }

  /** Set opacity on all meshes/sprites in a layer group. */
  _setGroupOpacity(li, opacity) {
    const group = this.groups[li];
    if (group) group.traverse(o => {
      if (o.isMesh || o.isLine || o.isLineSegments) {
        o.material.transparent = true;
        o.material.opacity = opacity;
        o.material.needsUpdate = true;
      }
    });
    const sprite = this._labelSprites?.[li];
    if (sprite?.material) {
      sprite.material.transparent = true;
      sprite.material.opacity = opacity;
      sprite.material.needsUpdate = true;
    }
  }

  /** Animate group opacity 0→1. Duration read dynamically from _perLayerSpeedMs each frame. */
  _fadeInGroup(li, animEpoch) {
    const start = performance.now();
    const tick = () => {
      if (this._layerAnimEpoch !== animEpoch) return;
      const duration = 2000 * (this._perLayerSpeedMs / 100);
      const t = Math.min(1, (performance.now() - start) / duration);
      this._setGroupOpacity(li, t);
      if (t < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  /** Fade out the current RF group's line material. Duration read dynamically each frame. */
  _fadeOutRF(onDone) {
    const group = this._rfGroup;
    if (!group) { onDone?.(); return; }
    let mat = null;
    group.traverse(o => { if (o.material && !mat) mat = o.material; });
    if (!mat) { this._clearRF(); onDone?.(); return; }
    const epoch = this._rfEpoch;
    const startOpacity = mat.opacity;
    const start = performance.now();
    const tick = () => {
      if (this._rfEpoch !== epoch) return;
      const duration = 1000 * (this._perLayerSpeedMs / 100);
      const t = Math.min(1, (performance.now() - start) / duration);
      mat.opacity = startOpacity * (1 - t);
      if (t < 1) {
        requestAnimationFrame(tick);
      } else {
        this._clearRF();
        onDone?.();
      }
    };
    requestAnimationFrame(tick);
  }

  /** Collect all {src, dst} line pairs for a layer's connections (no Three.js objects). */
  _gatherConnectionLines(li) {
    const lines = [];
    const conn  = this._connectivity[li];
    const def   = this._layerDefs[li];
    if (!def) return lines;

    if (conn?.type === 'pool') {
      for (let c = 0; c < def.channels; c++) {
        for (let py = 0; py < def.h; py++) {
          for (let px = 0; px < def.w; px++) {
            const srcPos = this._getPixelWorldPos(li, c, px, py);
            if (!srcPos) continue;
            const winners = this._getMaxPoolPixel(this._getContributingPixels(li, c, px, py), conn);
            for (const w of winners) {
              const dstPos = this._getPixelWorldPos(w.li, w.c, w.px, w.py);
              if (dstPos) lines.push({ src: srcPos.clone(), dst: dstPos.clone() });
            }
          }
        }
      }
      return lines;
    }

    if (conn?.type === 'flatten') {
      const step = Math.max(1, Math.floor(def.w / 128));
      for (let px = 0; px < def.w; px += step) {
        const srcPos = this._getPixelWorldPos(li, 0, px, 0);
        if (!srcPos) continue;
        const contributing = this._getContributingPixels(li, 0, px, 0);
        for (const p of contributing) {
          const dstPos = this._getPixelWorldPos(p.li, p.c, p.px, p.py);
          if (dstPos) lines.push({ src: srcPos.clone(), dst: dstPos.clone() });
        }
      }
      return lines;
    }

    if (conn) {
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
      const srcStep = isFc ? Math.max(1, Math.ceil(sources.length / 16)) : 1;
      for (let si = 0; si < sources.length; si += srcStep) {
        const { c, px, py } = sources[si];
        const srcPos = this._getPixelWorldPos(li, c, px, py);
        if (!srcPos) continue;
        const contributing = this._getContributingPixels(li, c, px, py);
        const dstStep = isFc ? Math.max(1, Math.ceil(contributing.length / 64)) : 1;
        for (let k = 0; k < contributing.length; k += dstStep) {
          const dstPos = this._getPixelWorldPos(contributing[k].li, contributing[k].c, contributing[k].px, contributing[k].py);
          if (dstPos) lines.push({ src: srcPos.clone(), dst: dstPos.clone() });
        }
      }
      return lines;
    }

    // Dense (no connectivity entry) — lines between prev and current layer samples
    const prevLi  = li - 1;
    const prevDef = this._layerDefs[prevLi];
    if (!prevDef) return lines;
    const isOutput = li === this._layerDefs.length - 1;
    const fromStep = Math.max(1, Math.ceil(def.channels / 8));
    const toStep   = Math.max(1, Math.ceil((prevDef.channels || 1) / 8));
    for (let c = 0; c < def.channels; c += fromStep) {
      const srcPos = this._getPixelWorldPos(li, c, 0, 0);
      if (!srcPos) continue;
      const prevChannels = prevDef.channels || 1;
      for (let pc = 0; pc < prevChannels; pc += toStep) {
        const dstPos = this._getPixelWorldPos(prevLi, pc, 0, 0);
        if (dstPos) lines.push({ src: srcPos.clone(), dst: dstPos.clone() });
      }
    }
    return lines;
  }

  /** Draw connection lines for layer li progressively. Duration read dynamically each frame. */
  _showLayerConnectionsAnimated(li, onDone) {
    this._clearRF();
    const lines = this._gatherConnectionLines(li);

    if (!lines.length) {
      onDone?.();
      return;
    }

    // Pre-allocate geometry: start at dst (previous layer), grow toward src (current layer)
    const posArr  = new Float32Array(lines.length * 6);
    for (let i = 0; i < lines.length; i++) {
      const { dst } = lines[i];
      posArr[i * 6]     = dst.x; posArr[i * 6 + 1] = dst.y; posArr[i * 6 + 2] = dst.z;
      posArr[i * 6 + 3] = dst.x; posArr[i * 6 + 4] = dst.y; posArr[i * 6 + 5] = dst.z;
    }
    const geo     = new THREE.BufferGeometry();
    const posAttr = new THREE.BufferAttribute(posArr, 3);
    posAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('position', posAttr);
    const mat = new THREE.LineBasicMaterial({ color: 0xffdd44, transparent: true, opacity: 0.6, depthWrite: false });
    const ls  = new THREE.LineSegments(geo, mat);
    ls.frustumCulled = false;

    const group = new THREE.Group();
    group.add(ls);
    this.scene.add(group);
    this._rfGroup = group;

    const start = performance.now();
    const epoch = this._rfEpoch;
    const tick  = () => {
      if (this._rfEpoch !== epoch) return;
      const duration = 2000 * (this._perLayerSpeedMs / 100);
      const t = Math.min(1, (performance.now() - start) / duration);
      for (let i = 0; i < lines.length; i++) {
        const { src, dst } = lines[i];
        posArr[i * 6 + 3] = dst.x + (src.x - dst.x) * t;
        posArr[i * 6 + 4] = dst.y + (src.y - dst.y) * t;
        posArr[i * 6 + 5] = dst.z + (src.z - dst.z) * t;
      }
      posAttr.needsUpdate = true;
      if (t < 1) {
        requestAnimationFrame(tick);
      } else {
        onDone?.();
      }
    };
    requestAnimationFrame(tick);
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

    const conn        = this._connectivity[li];
    const contributing = this._getContributingPixels(li, c, px, py);
    const visContrib   = (conn?.type === 'pool')
      ? this._getMaxPoolPixel(contributing, conn)
      : contributing;
    const opacities    = conn ? this._computeLineOpacities(visContrib, conn.prevLi) : [];
    this._rfLineOpacities = opacities.slice(); // store for highlight restore

    const group  = new THREE.Group();
    const dotGeo = new THREE.SphereGeometry(0.5, 6, 6);
    const srcDot = new THREE.Mesh(dotGeo, new THREE.MeshBasicMaterial({ color: 0xffdd44 }));
    srcDot.position.copy(srcPos);
    group.add(srcDot);
    this._rfSrcDot = srcDot;

    this._rfLines   = [];
    this._rfDstDots = [];

    const def    = this._layerDefs[li];
    const wObj   = this._convLayerWeights?.[li];
    const fcWObj = this._linearParams?.[li];

    // Opacity for bias → output pixel line: normalized output value
    const outIdx0  = c * def.h * def.w + py * def.w + px;
    const outAbs   = rawData ? Array.from(rawData).map(v => Math.abs(Number(v))) : [1e-6];
    const outMax   = Math.max(...outAbs, 1e-6);
    const biasToOutOpac = rawData ? Math.max(0.05, outAbs[outIdx0] / outMax) * 0.85 : 0.85;

    if (conn?.type === 'conv' && wObj) {
      // Conv with weights: kernel images placed just below each input channel plane;
      // each line splits at the specific kernel weight pixel, then all converge at bias dot.
      const [, inC, kH, kW] = wObj.shape;
      const CELL        = 5;   // must match _buildConvKernelImages
      const WEIGHT_OFF  = 25;  // world units below the input layer
      const BIAS_OFF    = 25;  // world units above the output pixel
      const kHalf = Math.floor(kH / 2);
      const prevDef  = this._layerDefs[conn.prevLi];
      const prevData = this._getRawData(conn.prevLi);
      const centerPx = Math.floor(prevDef.w / 2);
      const centerPy = Math.floor(prevDef.h / 2);

      group.add(this._buildConvKernelImages(c, inC, kH, kW, wObj.data, conn.prevLi, prevDef, WEIGHT_OFF));

      // Bias pixel above the output pixel (if bias data present)
      const pxSz    = PIXEL_SIZE;
      const biasPos = wObj.bias
        ? new THREE.Vector3(srcPos.x, srcPos.y + BIAS_OFF, srcPos.z)
        : null;
      if (biasPos) {
        group.add(this._buildBiasPixel(wObj.bias[c], wObj.bias, biasPos, pxSz));
      }

      // Cache channel-center world positions (called once per channel)
      const chanCenters = {};
      const _getChanCenter = (ic) => {
        if (!chanCenters[ic])
          chanCenters[ic] = this._getPixelWorldPos(conn.prevLi, ic, centerPx, centerPy);
        return chanCenters[ic];
      };

      const beforeData = [], afterData = [];
      const beforeVals = [], afterVals  = [];

      for (const p of visContrib) {
        const dstPos = this._getPixelWorldPos(p.li, p.c, p.px, p.py); // input pixel
        if (!dstPos) continue;
        const dy = p.py - py, dx = p.px - px;
        const ky = dy + kHalf, kx = dx + kHalf;
        if (ky < 0 || ky >= kH || kx < 0 || kx >= kW) continue;
        const wIdx   = (c * inC + p.c) * kH * kW + ky * kW + kx;
        const weight = wObj.data[wIdx] ?? 0;
        const inpVal = prevData
          ? Math.abs(Number(prevData[p.c * prevDef.h * prevDef.w + p.py * prevDef.w + p.px]))
          : 0;
        beforeVals.push(inpVal);
        afterVals.push(Math.abs(inpVal * weight));

        // Kernel weight pixel world position: below the input channel center, offset by kernel (ky,kx)
        const cc = _getChanCenter(p.c);
        if (!cc) continue;
        const splitPt = new THREE.Vector3(
          cc.x + (kx - kHalf) * CELL,
          cc.y - WEIGHT_OFF,
          cc.z + (ky - kHalf) * CELL,
        );
        // input pixel → weight pixel: opacity = input value
        beforeData.push({ src: splitPt.clone(), dst: dstPos });
        // weight pixel → bias dot (or output pixel if no bias): opacity = input×weight
        afterData.push({ src: (biasPos ?? srcPos).clone(), dst: splitPt });

        const dot = new THREE.Mesh(dotGeo, new THREE.MeshBasicMaterial({ color: 0xffdd44 }));
        dot.position.copy(dstPos);
        group.add(dot);
        this._rfDstDots.push(dot);
      }

      const norm = (vals) => {
        const mx = Math.max(...vals, 1e-6);
        return vals.map(v => Math.max(0.05, v / mx) * 0.85);
      };
      const bOpacs = norm(beforeVals), aOpacs = norm(afterVals);
      this._rfLineOpacities = aOpacs.slice();

      if (beforeData.length > 0)
        group.add(this._buildLineSegments(beforeData.map((l, i) => ({ ...l, opacity: bOpacs[i] }))));
      // afterData lines are added individually so highlightRFLine can colour them one by one
      for (let i = 0; i < afterData.length; i++) {
        const { src, dst } = afterData[i];
        const mat  = new THREE.LineBasicMaterial({ color: 0xffdd44, transparent: true, opacity: aOpacs[i] });
        const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([src, dst]), mat);
        this._rfLines.push(line);
        group.add(line);
      }

      // Single bias pixel → output pixel line
      if (biasPos) {
        const biasLine = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([biasPos.clone(), srcPos.clone()]),
          new THREE.LineBasicMaterial({ color: 0xffdd44, transparent: true, opacity: biasToOutOpac })
        );
        group.add(biasLine);
      }

    } else if (conn?.type === 'fc' && fcWObj) {
      // FC with parameters: show weight matrix image (1D input only) + bias convergence dot.
      const [outF, inF] = fcWObj.shape;
      const outIdx      = c * def.h * def.w + py * def.w + px;
      const WEIGHT_OFF  = 25;
      const BIAS_OFF    = 25;
      const prevDef     = this._layerDefs[conn.prevLi];
      const prevData    = this._getRawData(conn.prevLi);

      const pxSz    = PIXEL_SIZE;
      const biasPos = fcWObj.bias
        ? new THREE.Vector3(srcPos.x, srcPos.y + BIAS_OFF, srcPos.z)
        : null;
      if (biasPos) {
        group.add(this._buildBiasPixel(fcWObj.bias[outIdx], fcWObj.bias, biasPos, pxSz));
      }

      const is1D = prevDef.channels === 1 && prevDef.h === 1 && inF === prevDef.w;

      if (is1D) {
        // Weight matrix image: columns = input neurons, rows = output neurons.
        // The target row (outIdx) is positioned at the same Z as the input neurons (refZ).
        const p0 = this._getPixelWorldPos(conn.prevLi, 0, 0, 0);
        const pN = this._getPixelWorldPos(conn.prevLi, 0, inF - 1, 0);
        if (p0 && pN) {
          const cellSize     = inF > 1 ? Math.abs((pN.x - p0.x) / (inF - 1)) : PIXEL_SIZE;
          const imageCenterX = (p0.x + pN.x) / 2;
          const imageCenterY = p0.y - WEIGHT_OFF;
          const refZ         = p0.z;
          // Shift image so row outIdx lands at refZ:
          // world Z of row j = imageCenterZ - (j + 0.5 - outF/2) * cellSize  (local Y → world -Z)
          const imageCenterZ = refZ + (outIdx + 0.5 - outF / 2) * cellSize;

          group.add(this._buildFCWeightImage(
            outIdx, outF, inF, fcWObj.data,
            imageCenterX, imageCenterY, imageCenterZ, cellSize
          ));

          const beforeData = [], afterData = [];
          const beforeVals = [], afterVals  = [];

          for (const p of visContrib) {
            const inputPos = this._getPixelWorldPos(p.li, p.c, p.px, p.py);
            if (!inputPos) continue;
            const wIdx   = outIdx * inF + p.px;
            const weight = fcWObj.data[wIdx] ?? 0;
            const inpVal = prevData
              ? Math.abs(Number(prevData[p.c * prevDef.h * prevDef.w + p.py * prevDef.w + p.px]))
              : 0;
            beforeVals.push(inpVal);
            afterVals.push(Math.abs(inpVal * weight));

            // Split point: directly below input neuron, at the target row's world Z
            const splitPt = new THREE.Vector3(inputPos.x, imageCenterY, refZ);
            beforeData.push({ src: splitPt.clone(), dst: inputPos });
            afterData.push({ src: (biasPos ?? srcPos).clone(), dst: splitPt });

            const dot = new THREE.Mesh(dotGeo, new THREE.MeshBasicMaterial({ color: 0xffdd44 }));
            dot.position.copy(inputPos);
            group.add(dot);
            this._rfDstDots.push(dot);
          }

          const norm = (vals) => {
            const mx = Math.max(...vals, 1e-6);
            return vals.map(v => Math.max(0.05, v / mx) * 0.85);
          };
          const bOpacs = norm(beforeVals), aOpacs = norm(afterVals);
          this._rfLineOpacities = aOpacs.slice();

          if (beforeData.length > 0)
            group.add(this._buildLineSegments(beforeData.map((l, i) => ({ ...l, opacity: bOpacs[i] }))));
          // afterData lines are added individually so highlightRFLine can colour them one by one
          for (let i = 0; i < afterData.length; i++) {
            const { src, dst } = afterData[i];
            const mat  = new THREE.LineBasicMaterial({ color: 0xffdd44, transparent: true, opacity: aOpacs[i] });
            const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([src, dst]), mat);
            this._rfLines.push(line);
            group.add(line);
          }
        }
      } else {
        // Non-1D input (mode=center): lines go directly from inputs to bias/output
        for (let i = 0; i < visContrib.length; i++) {
          const p = visContrib[i];
          const dstPos = this._getPixelWorldPos(p.li, p.c, p.px, p.py);
          if (!dstPos) continue;
          const opacity = opacities[i] ?? 0.7;
          const endPt   = biasPos ?? srcPos;
          const mat  = new THREE.LineBasicMaterial({ color: 0xffdd44, transparent: true, opacity });
          const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([endPt.clone(), dstPos]), mat);
          this._rfLines.push(line);
          group.add(line);
          const dot = new THREE.Mesh(new THREE.SphereGeometry(0.5, 6, 6), new THREE.MeshBasicMaterial({ color: 0xffdd44 }));
          dot.position.copy(dstPos);
          group.add(dot);
          this._rfDstDots.push(dot);
        }
      }

      // Single bias pixel → output pixel line
      if (biasPos) {
        const biasLine = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([biasPos.clone(), srcPos.clone()]),
          new THREE.LineBasicMaterial({ color: 0xffdd44, transparent: true, opacity: biasToOutOpac })
        );
        group.add(biasLine);
      }

    } else {
      // Pool or non-kernel conv: individual lines with per-line opacity
      for (let i = 0; i < visContrib.length; i++) {
        const p = visContrib[i];
        const dstPos = this._getPixelWorldPos(p.li, p.c, p.px, p.py);
        if (!dstPos) continue;
        const opacity = opacities[i] ?? 0.7;
        const mat  = new THREE.LineBasicMaterial({ color: 0xffdd44, transparent: true, opacity });
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
    const posArr   = new Float32Array(MAX_LINES * 6);
    const colorArr = new Float32Array(MAX_LINES * 6);
    // Default color: full yellow (overridden per-line in addRFLine)
    for (let i = 0; i < MAX_LINES * 6; i += 3) {
      colorArr[i] = 1.0; colorArr[i + 1] = 0.867; colorArr[i + 2] = 0.267;
    }
    const lsGeo = new THREE.BufferGeometry();
    lsGeo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
    lsGeo.setAttribute('color',    new THREE.BufferAttribute(colorArr, 3));
    lsGeo.setDrawRange(0, 0);
    const lsMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.75 });
    this._rfLineSegs                  = new THREE.LineSegments(lsGeo, lsMat);
    this._rfLineSegs.frustumCulled    = false; // prevent disappearing during camera rotation
    this._rfLineCount     = 0;
    this._rfLineEndpoints = [];
    this._rfBaseColors    = []; // reset per initRFAnimated
    group.add(this._rfLineSegs);

    this.scene.add(group);
    this._rfGroup  = group;
    this._rfSrcPos = srcPos;
    this._rfLines   = [];
    this._rfDstDots = [];

    const conn        = this._connectivity[li];
    const contributing = this._getContributingPixels(li, c, px, py);
    // For pool: only animate line to the winning pixel
    const visContrib   = (conn?.type === 'pool')
      ? this._getMaxPoolPixel(contributing, conn)
      : contributing;
    const opacities    = conn ? this._computeLineOpacities(visContrib, conn.prevLi) : [];
    // Embed opacity into each contributing pixel so addRFLine can use it
    const visWithOpac  = visContrib.map((p, i) => ({ ...p, opacity: opacities[i] ?? 0.85 }));

    // _buildDetailInfo uses full contributing so formula shows all pool values, not just winner
    return { contributing: visWithOpac, detail: this._buildDetailInfo(li, c, px, py, rawData, contributing) };
  }

  /** Add one RF line with animated drawing from source to destination. */
  addRFLine(prevLi, prevC, prevPx, prevPy, durationMs = 200, opacity = 0.85) {
    if (!this._rfGroup || !this._rfSrcPos || !this._rfLineSegs) return;
    const capturedEpoch = this._rfEpoch; // detect if _clearRF() is called before tick completes
    const dstPos = this._getPixelWorldPos(prevLi, prevC, prevPx, prevPy);
    if (!dstPos) return;
    const src     = this._rfSrcPos.clone();
    const lineIdx = this._rfLineCount++;
    this._rfLineEndpoints.push({ src: src.clone(), dst: dstPos.clone() });

    const geo       = this._rfLineSegs.geometry;
    const posAttr   = geo.attributes.position;
    const colorAttr = geo.attributes.color;
    const base      = lineIdx * 6;
    posAttr.array[base]     = src.x;
    posAttr.array[base + 1] = src.y;
    posAttr.array[base + 2] = src.z;

    // Set opacity-blended yellow color for this line
    const r = opacity, g = 0.867 * opacity, b = 0.267 * opacity;
    colorAttr.array[base]     = r; colorAttr.array[base + 1] = g; colorAttr.array[base + 2] = b;
    colorAttr.array[base + 3] = r; colorAttr.array[base + 4] = g; colorAttr.array[base + 5] = b;
    colorAttr.needsUpdate = true;
    this._rfBaseColors[lineIdx] = { r, g, b }; // store for highlight restore

    const startTime = performance.now();
    const tick = () => {
      // Bail if _clearRF() was called (epoch changed) — prevents corrupt writes to new RF state
      if (!this._rfGroup || this._rfEpoch !== capturedEpoch) return;
      const t   = Math.min(1, (performance.now() - startTime) / durationMs);
      const cur = src.clone().lerp(dstPos, t);
      posAttr.array[base + 3] = cur.x;
      posAttr.array[base + 4] = cur.y;
      posAttr.array[base + 5] = cur.z;
      posAttr.needsUpdate     = true;
      const needed = (lineIdx + 1) * 2;
      if (geo.drawRange.count < needed) geo.setDrawRange(0, needed);
      if (t < 1) {
        requestAnimationFrame(tick);
      } else {
        if (this._rfEpoch !== capturedEpoch) return;
        const dot = new THREE.Mesh(new THREE.SphereGeometry(0.5, 6, 6), new THREE.MeshBasicMaterial({ color: 0xffdd44 }));
        dot.position.copy(dstPos);
        this._rfGroup.add(dot);
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
    const def    = this._layerDefs[li];
    const group  = new THREE.Group();
    const dotGeo = new THREE.SphereGeometry(0.5, 6, 6);

    // Helper: build a Group of LineSegments bucketed by opacity tier so each tier
    // gets its own material with the correct alpha (actual transparency, not color darkening).
    const TIERS = 8;
    const buildLineSegments = (lineData) => {
      const buckets = Array.from({ length: TIERS }, () => []);
      for (const item of lineData) {
        const tier = Math.min(TIERS - 1, Math.floor(item.opacity * TIERS));
        buckets[tier].push(item);
      }
      const g = new THREE.Group();
      for (let t = 0; t < TIERS; t++) {
        if (!buckets[t].length) continue;
        const alpha  = (t + 0.5) / TIERS;
        const posArr = new Float32Array(buckets[t].length * 6);
        let bi = 0;
        for (const { src, dst } of buckets[t]) {
          posArr[bi++] = src.x; posArr[bi++] = src.y; posArr[bi++] = src.z;
          posArr[bi++] = dst.x; posArr[bi++] = dst.y; posArr[bi++] = dst.z;
        }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
        const mat = new THREE.LineBasicMaterial({ color: 0xffdd44, transparent: true, opacity: alpha, depthWrite: false });
        const ls  = new THREE.LineSegments(geo, mat);
        ls.frustumCulled = false;
        g.add(ls);
      }
      return g;
    };

    // Pool layers: 4 blue corner lines + yellow lines only to the max-value source pixel per window
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
        // Blue corner lines (individual — only 4 of them)
        const cornerMat    = new THREE.LineBasicMaterial({ color: 0x66aaff, transparent: true, opacity: 0.45 });
        const cornerDotMat = new THREE.MeshBasicMaterial({ color: 0x66aaff });
        for (let i = 0; i < 4; i++) {
          group.add(new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([curCorners[i], prevCorners[i]]),
            cornerMat
          ));
          const d1 = new THREE.Mesh(dotGeo, cornerDotMat); d1.position.copy(curCorners[i]);
          const d2 = new THREE.Mesh(dotGeo, cornerDotMat); d2.position.copy(prevCorners[i]);
          group.add(d1); group.add(d2);
        }
        // Yellow lines + dots: every pool-output pixel → winning source pixel only (max value)
        const lineData  = [];
        const yellowDotMat = new THREE.MeshBasicMaterial({ color: 0xffdd44 });
        const dstDotSeen   = new Set();
        for (let c = 0; c < def.channels; c++) {
          for (let py = 0; py < def.h; py++) {
            for (let px = 0; px < def.w; px++) {
              const srcPos = this._getPixelWorldPos(li, c, px, py);
              if (!srcPos) continue;
              const contributing = this._getContributingPixels(li, c, px, py);
              const winners      = this._getMaxPoolPixel(contributing, conn);
              const opacities    = this._computeLineOpacities(winners, conn.prevLi);
              // Src dot (pool output pixel)
              const srcDot = new THREE.Mesh(dotGeo, yellowDotMat);
              srcDot.position.copy(srcPos);
              group.add(srcDot);
              for (let i = 0; i < winners.length; i++) {
                const w      = winners[i];
                const dstPos = this._getPixelWorldPos(w.li, w.c, w.px, w.py);
                if (!dstPos) continue;
                lineData.push({ src: srcPos, dst: dstPos, opacity: opacities[i] ?? 0.35 });
                // Dst dot (winning source pixel) — deduplicated
                const key = `${w.li}_${w.c}_${w.px}_${w.py}`;
                if (!dstDotSeen.has(key)) {
                  dstDotSeen.add(key);
                  const dstDot = new THREE.Mesh(dotGeo, yellowDotMat);
                  dstDot.position.copy(dstPos);
                  group.add(dstDot);
                }
              }
            }
          }
        }
        if (lineData.length > 0) group.add(buildLineSegments(lineData));
      }
      this.scene.add(group);
      this._rfGroup = group;
      return;
    }

    // Flatten: sample connections from flatten neurons to their 1:1 source pixels
    if (conn.type === 'flatten') {
      const lineData = [];
      const dotMat2  = new THREE.MeshBasicMaterial({ color: 0xffdd44 });
      const dstSeen2 = new Set();
      const step2    = Math.max(1, Math.floor(def.w / 128));
      for (let px = 0; px < def.w; px += step2) {
        const srcPos = this._getPixelWorldPos(li, 0, px, 0);
        if (!srcPos) continue;
        const srcDot = new THREE.Mesh(dotGeo, dotMat2);
        srcDot.position.copy(srcPos);
        group.add(srcDot);
        const contributing = this._getContributingPixels(li, 0, px, 0);
        const opacities    = this._computeLineOpacities(contributing, conn.prevLi);
        for (let i = 0; i < contributing.length; i++) {
          const p      = contributing[i];
          const dstPos = this._getPixelWorldPos(p.li, p.c, p.px, p.py);
          if (!dstPos) continue;
          lineData.push({ src: srcPos, dst: dstPos, opacity: opacities[i] ?? 0.5 });
          const key = `${p.li}_${p.c}_${p.px}_${p.py}`;
          if (!dstSeen2.has(key)) {
            dstSeen2.add(key);
            const dstDot = new THREE.Mesh(dotGeo, dotMat2);
            dstDot.position.copy(dstPos);
            group.add(dstDot);
          }
        }
      }
      if (lineData.length > 0) group.add(buildLineSegments(lineData));
      this.scene.add(group);
      this._rfGroup = group;
      return;
    }

    // Conv / other: show center pixel per channel; lines to all contributing pixels
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

    const allLineData = [];
    const dotMat = new THREE.MeshBasicMaterial({ color: 0xffdd44 });
    const dstSeen = new Set();
    // For FC: limit sources/destinations to keep line count manageable
    // For Conv: always show all channels (center pixel each) with all their contributing pixels
    const srcStep = isFc ? Math.max(1, Math.ceil(sources.length / 16)) : 1;

    for (let si = 0; si < sources.length; si += srcStep) {
      const { c, px, py } = sources[si];
      const srcPos = this._getPixelWorldPos(li, c, px, py);
      if (!srcPos) continue;
      const srcDot = new THREE.Mesh(dotGeo, dotMat);
      srcDot.position.copy(srcPos);
      group.add(srcDot);

      const contributing = this._getContributingPixels(li, c, px, py);
      const dstStep      = isFc ? Math.max(1, Math.ceil(contributing.length / 64)) : 1;
      const opacities    = this._computeLineOpacities(contributing, conn.prevLi);
      for (let i = 0; i < contributing.length; i += dstStep) {
        const p      = contributing[i];
        const dstPos = this._getPixelWorldPos(p.li, p.c, p.px, p.py);
        if (!dstPos) continue;
        allLineData.push({ src: srcPos, dst: dstPos, opacity: opacities[i] ?? 0.45 });
        const key = `${p.li}_${p.c}_${p.px}_${p.py}`;
        if (!dstSeen.has(key)) {
          dstSeen.add(key);
          const dstDot = new THREE.Mesh(dotGeo, dotMat);
          dstDot.position.copy(dstPos);
          group.add(dstDot);
        }
      }
    }

    if (allLineData.length > 0) group.add(buildLineSegments(allLineData));
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
      // Restore each line to its stored per-line opacity-blended color
      for (let i = 0; i < n; i++) {
        const bc = this._rfBaseColors[i] ?? { r: 1.0, g: 0.867, b: 0.267 };
        colorAttr.setXYZ(i * 2,     bc.r, bc.g, bc.b);
        colorAttr.setXYZ(i * 2 + 1, bc.r, bc.g, bc.b);
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
    for (let i = 0; i < this._rfLines.length; i++) {
      const line = this._rfLines[i];
      line.material.color.setHex(0xffdd44);
      line.material.opacity = this._rfLineOpacities[i] ?? 0.7; // restore per-line opacity
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

  getPixelInfo(layerIdx, channel, x, y) {
    const mesh = this.meshByLayer[layerIdx]?.[channel];
    if (!mesh) return null;
    const ud  = mesh.userData;
    const idx = ud.offset + y * ud.w + x;
    return {
      layerIdx,
      layerName:       ud.layerName,
      channel,
      x, y,
      rawValue:        Number(ud.rawData[idx]),
      normalizedValue: Number(ud.normalizedData[idx]),
      rawData:         ud.rawData,
    };
  }

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
      let data, dims;
      if (li === 0 || def.dataKey === '__input__') {
        data = ip;
      } else if (def.dataKey) {
        const entry = ld?.[def.dataKey];
        data = entry?.data;
        dims = entry?.dims;
      } else {
        // Fallback positional map for configs without dataKey (7-layer legacy)
        const keys = ['layer0','layer1','layer2','layer3','layer4','output'];
        const entry = ld?.[keys[li - 1]];
        data = entry?.data;
        dims = entry?.dims;
      }
      // For output layer with more classes than the hardcoded config (e.g. CIFAR-100 has 100),
      // patch def.channels so connectivity/highlight code uses the real class count.
      let patchedDef = def;
      if (def.channelLabels === '__classes__') {
        const numClasses = dims?.length === 2
          ? Number(dims[1])
          : (this._classLabels?.length ?? def.channels);
        if (numClasses !== def.channels) {
          patchedDef = { ...def, channels: numClasses };
        }
      }
      return { def: patchedDef, data: data ?? new Float32Array(patchedDef.channels * patchedDef.h * patchedDef.w), dims };
    });
  }

  _buildAllLayers() {
    const payloads = this._buildLayerPayloads();
    for (let li = 0; li < payloads.length; li++) this._buildLayer(payloads[li], li);
  }

  _buildLayer(payload, li) {
    const { def, data, dims } = payload;
    // For li===0 (raw input image), use actual dataset dims (model configs hardcode 1×28×28).
    // For intermediate layers, use actual ONNX tensor dims to handle datasets with different
    // spatial sizes (e.g. CIFAR 32×32 gives 16×16 after MaxPool, not 14×14 as in MNIST configs).
    let C, H, W;
    if (li === 0) {
      C = this._inChannels   ?? def.channels;
      H = this._inputImgSize ?? def.h;
      W = this._inputImgSize ?? def.w;
    } else if (dims && dims.length >= 4 && def.channels > 1) {
      // Only override spatial dims for multi-channel layers (Conv/Pool).
      // Flatten reuses the prior layer's 4D tensor but must keep its own 1×W strip layout.
      C = Number(dims[1]); H = Number(dims[2]); W = Number(dims[3]);
    } else if (dims && dims.length === 3 && def.channels > 1) {
      C = Number(dims[0]); H = Number(dims[1]); W = Number(dims[2]);
    } else if (dims && dims.length === 2 && def.channelLabels === '__classes__') {
      // Output layer only: [batch, numClasses] — use actual class count from ONNX
      C = Number(dims[1]); H = 1; W = 1;
    } else if (def.channelLabels === '__classes__') {
      // Fallback: use dataset class count (e.g. 100 for CIFAR-100)
      C = this._classLabels?.length ?? def.channels; H = def.h; W = def.w;
    } else {
      C = def.channels; H = def.h; W = def.w;
    }

    const isOutput = li === this._layerDefs.length - 1;
    // 1D layers (FC activations, h=1 w=1) render as a single linear row.
    // Output layer uses a row for ≤20 classes, square grid for larger counts (e.g. CIFAR-100).
    const is1D = H === 1 && W === 1;
    const { cols, rows } = isOutput
      ? (C <= 20 ? { cols: C, rows: 1 } : gridLayout(C))
      : (is1D ? { cols: C, rows: 1 } : gridLayout(C));

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

    // Input layer: convert HWC → CHW and use raw [0,1] values (no contrast stretch)
    let chw = data;
    if (li === 0 && C > 1) {
      const inC = this._inChannels ?? C;
      const hw  = H * W;
      chw = new Float32Array(inC * hw);
      for (let p = 0; p < hw; p++) {
        for (let c2 = 0; c2 < inC; c2++) {
          chw[c2 * hw + p] = data[p * inC + c2];
        }
      }
    }

    // Skip normalization for the input layer — pixels are already [0,1]
    const normalized = li === 0 ? chw : normalizeActivations(Array.from(chw)).normalized;
    if (!this.meshByLayer[li]) this.meshByLayer[li] = {};

    for (let c = 0; c < C; c++) {
      const col    = c % cols;
      const row    = Math.floor(c / cols);
      const cx     = col * (planeW + GAP) + planeW / 2;
      const cz     = row * (planeH + GAP) + planeH / 2;
      const offset = c * H * W;

      // For RGB input channels: tint channel 0 red, 1 green, 2 blue
      const isRGBInput = li === 0 && C === 3;

      const texData = new Uint8Array(W * H * 4);
      for (let py2 = 0; py2 < H; py2++) {
        for (let px2 = 0; px2 < W; px2++) {
          const srcI = offset + py2 * W + px2;
          // Flip Y so row py2=0 appears at visual top (DataTexture V=0 is bottom)
          const dstI = (H - 1 - py2) * W + px2;
          const bv = Math.round(normalized[srcI] * 255);
          if (isRGBInput) {
            texData[dstI * 4]     = c === 0 ? bv : 0;
            texData[dstI * 4 + 1] = c === 1 ? bv : 0;
            texData[dstI * 4 + 2] = c === 2 ? bv : 0;
          } else {
            texData[dstI * 4]     = bv;
            texData[dstI * 4 + 1] = bv;
            texData[dstI * 4 + 2] = bv;
          }
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
        const labels = def.channelLabels === '__classes__'
          ? (this._classLabels || ['0','1','2','3','4','5','6','7','8','9'])
          : def.channelLabels;
        this._addCellLabel(group, labels[c], data[c], cx, cz, planeW, planeH, isOutput);
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
    canvas.height = 160;
    // Measure text width first so long names are never clipped
    canvas.width  = 256; // temp width for measurement
    const ctx     = canvas.getContext('2d');
    ctx.font      = 'bold 104px monospace';
    const textW   = ctx.measureText(name).width;
    canvas.width  = Math.max(512, Math.ceil(textW) + 80);
    ctx.font      = 'bold 104px monospace'; // re-apply after resize
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#eeeeee';
    ctx.textAlign = 'center';
    ctx.fillText(name, canvas.width / 2, 112);
    const tex    = new THREE.CanvasTexture(canvas);
    const mat    = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
    const sprite = new THREE.Sprite(mat);
    // Keep a fixed world-unit height and scale width proportionally so text isn't distorted
    const spriteH = isOutput ? Math.round(totalW * 0.22 * 160 / 1024) : 22;
    const spriteW = spriteH * (canvas.width / canvas.height);
    sprite.scale.set(spriteW, spriteH, 1);
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
    const maxLabelW = sz * 0.90;
    let labelFontSize = isOutput ? Math.round(sz * 0.50) : 120;
    ctx.font = `bold ${labelFontSize}px sans-serif`;
    while (ctx.measureText(digit).width > maxLabelW && labelFontSize > 16) {
      labelFontSize = Math.round(labelFontSize * 0.85);
      ctx.font = `bold ${labelFontSize}px sans-serif`;
    }
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
    const EPS = 0.1; // depthTest:false + renderOrder:999 handle visibility; keep near 0 to avoid parallax offset
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
    } else if (conn.type === 'flatten') {
      // Each flatten neuron maps 1:1 to one source pixel in the prev layer
      const prevDef2    = this._layerDefs[conn.prevLi];
      const spatialSize = prevDef2.h * prevDef2.w;
      const ch  = Math.floor(px / spatialSize);
      const rem = px % spatialSize;
      const spy = Math.floor(rem / prevDef2.w);
      const spx = rem % prevDef2.w;
      if (ch < prevDef2.channels)
        pixels.push({ li: conn.prevLi, c: ch, px: spx, py: spy });
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

  _getMaxPoolPixel(contributing, conn) {
    const prevData = this._getRawData(conn.prevLi);
    const prevDef  = this._layerDefs[conn.prevLi];
    if (!prevData || !prevDef || contributing.length === 0) return contributing.slice(0, 1);
    let maxIdx = 0, maxVal = -Infinity;
    for (let i = 0; i < contributing.length; i++) {
      const { c, px, py } = contributing[i];
      const val = Number(prevData[c * prevDef.h * prevDef.w + py * prevDef.w + px]);
      if (val > maxVal) { maxVal = val; maxIdx = i; }
    }
    return [contributing[maxIdx]];
  }

  _computeLineOpacities(visContrib, prevLi) {
    const prevData = this._getRawData(prevLi);
    const prevDef  = this._layerDefs[prevLi];
    if (!prevData || !prevDef) return visContrib.map(() => 0.85);
    const vals = visContrib.map(p =>
      Math.abs(Number(prevData[p.c * prevDef.h * prevDef.w + p.py * prevDef.w + p.px] ?? 0))
    );
    const maxAbs = Math.max(...vals, 1e-6);
    return vals.map(v => Math.max(0.05, v / maxAbs) * 0.85);
  }

  /** Load conv/linear parameters and map them to layer indices. */
  setParameters(paramsObj) {
    this._convLayerWeights = {};
    this._linearParams     = {};
    if (!paramsObj) return;
    let convIdx = 0, fcIdx = 0;
    for (let li = 0; li < this._connectivity.length; li++) {
      const t = this._connectivity[li]?.type;
      if (t === 'conv' && paramsObj.convs?.[convIdx] !== undefined) {
        this._convLayerWeights[li] = _decodeParams(paramsObj.convs[convIdx++]);
      } else if (t === 'fc' && paramsObj.linears?.[fcIdx] !== undefined) {
        this._linearParams[li] = _decodeParams(paramsObj.linears[fcIdx++]);
      }
    }
  }

  /** Build grayscale kernel images, one per input channel, placed just below each input channel plane. */
  _buildConvKernelImages(outCh, inC, kH, kW, weightData, prevLi, prevDef, weightOffset) {
    const group  = new THREE.Group();
    const CELL   = 5;  // world units per kernel pixel (must match showReceptiveField)
    const planeW = kW * CELL;
    const planeH = kH * CELL;
    const centerPx = Math.floor(prevDef.w / 2);
    const centerPy = Math.floor(prevDef.h / 2);

    for (let ic = 0; ic < inC; ic++) {
      const cc = this._getPixelWorldPos(prevLi, ic, centerPx, centerPy);
      if (!cc) continue;

      const base   = (outCh * inC + ic) * kH * kW;
      const kSlice = weightData.slice ? weightData.slice(base, base + kH * kW)
                                      : Array.from({ length: kH * kW }, (_, j) => weightData[base + j]);
      let kMin = Infinity, kMax = -Infinity;
      for (const v of kSlice) { if (v < kMin) kMin = v; if (v > kMax) kMax = v; }
      const kRange = Math.max(Math.abs(kMax - kMin), 1e-6);

      const texData = new Uint8Array(kW * kH * 4);
      for (let ky = 0; ky < kH; ky++) {
        for (let kx = 0; kx < kW; kx++) {
          const bv   = Math.round(((kSlice[ky * kW + kx] - kMin) / kRange) * 255);
          const dstI = (kH - 1 - ky) * kW + kx; // flip Y like feature maps
          texData[dstI * 4]     = bv;
          texData[dstI * 4 + 1] = bv;
          texData[dstI * 4 + 2] = bv;
          texData[dstI * 4 + 3] = 220;
        }
      }
      const tex = new THREE.DataTexture(texData, kW, kH);
      tex.needsUpdate = true;
      tex.magFilter   = THREE.NearestFilter;
      tex.minFilter   = THREE.NearestFilter;

      const geo  = new THREE.PlaneGeometry(planeW, planeH);
      const mat  = new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide, transparent: true, opacity: 0.95 });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.rotation.x = -Math.PI / 2;
      // Place kernel image centered below the input channel's feature map
      mesh.position.set(cc.x, cc.y - weightOffset, cc.z);
      mesh.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: 0x888888 })));
      group.add(mesh);
    }
    return group;
  }

  /** Build a Group of LineSegments bucketed by opacity tier (shared helper). */
  _buildLineSegments(lineData) {
    const TIERS   = 8;
    const buckets = Array.from({ length: TIERS }, () => []);
    for (const item of lineData) {
      const tier = Math.min(TIERS - 1, Math.floor(item.opacity * TIERS));
      buckets[tier].push(item);
    }
    const g = new THREE.Group();
    for (let t = 0; t < TIERS; t++) {
      if (!buckets[t].length) continue;
      const alpha  = (t + 0.5) / TIERS;
      const posArr = new Float32Array(buckets[t].length * 6);
      let bi = 0;
      for (const { src, dst } of buckets[t]) {
        posArr[bi++] = src.x; posArr[bi++] = src.y; posArr[bi++] = src.z;
        posArr[bi++] = dst.x; posArr[bi++] = dst.y; posArr[bi++] = dst.z;
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
      const mat = new THREE.LineBasicMaterial({ color: 0xffdd44, transparent: true, opacity: alpha, depthWrite: false });
      const ls  = new THREE.LineSegments(geo, mat);
      ls.frustumCulled = false;
      g.add(ls);
    }
    return g;
  }

  /** Build a horizontal FC weight matrix image mesh. Target row (outIdx) lands at world Z = cz. */
  _buildFCWeightImage(outIdx, outF, inF, weightData, cx, cy, cz, cellSize) {
    const imageW = inF  * cellSize;
    const imageH = outF * cellSize;

    let wMin = Infinity, wMax = -Infinity;
    for (let k = 0; k < weightData.length; k++) {
      const v = weightData[k];
      if (v < wMin) wMin = v;
      if (v > wMax) wMax = v;
    }
    const wRange = Math.max(wMax - wMin, 1e-6);

    // Row j stored at texture row j from bottom (V=0).
    // After rotation.x=-π/2: local Y → world -Z, so row j (from bottom) → world Z = cy - (j+0.5-outF/2)*cellSize.
    // We set mesh.position.z = cz so that row outIdx is at world Z = cz.
    const texData = new Uint8Array(inF * outF * 4);
    for (let j = 0; j < outF; j++) {
      const dim = (j === outIdx) ? 1.0 : 0.15;
      for (let i = 0; i < inF; i++) {
        const w  = weightData[j * inF + i];
        const bv = Math.round(((w - wMin) / wRange) * 255 * dim);
        const dstI = j * inF + i;
        texData[dstI * 4]     = bv;
        texData[dstI * 4 + 1] = bv;
        texData[dstI * 4 + 2] = bv;
        texData[dstI * 4 + 3] = 220;
      }
    }

    const tex = new THREE.DataTexture(texData, inF, outF);
    tex.needsUpdate = true;
    tex.magFilter   = THREE.NearestFilter;
    tex.minFilter   = THREE.NearestFilter;

    const geo  = new THREE.PlaneGeometry(imageW, imageH);
    const mat  = new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide, transparent: true, opacity: 0.95 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.set(cx, cy, cz);
    mesh.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: 0x888888 })));
    return mesh;
  }

  /** Build a 1×1 bias pixel as a PlaneGeometry quad with a grayscale DataTexture. */
  _buildBiasPixel(biasVal, biasArr, pos, pixelSize) {
    const maxAbs = Math.max(...biasArr.map(v => Math.abs(v)), 1e-6);
    const norm   = (biasVal / maxAbs + 1) / 2; // signed: [-1,1] → [0,1]
    const bv     = Math.round(Math.max(0, Math.min(1, norm)) * 255);
    const texData = new Uint8Array([bv, bv, bv, 255]);
    const tex = new THREE.DataTexture(texData, 1, 1);
    tex.magFilter = THREE.NearestFilter;
    tex.minFilter = THREE.NearestFilter;
    tex.needsUpdate = true;
    const geo  = new THREE.PlaneGeometry(pixelSize, pixelSize);
    const mat  = new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.copy(pos);
    const edgesMat = new THREE.LineBasicMaterial({ color: 0x888888, transparent: true, opacity: 0.7 });
    mesh.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), edgesMat));
    return mesh;
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
    if (li === 0) return this._inputPixels;
    const ld  = this._layerData;
    const def = this._layerDefs[li];
    if (def?.dataKey) return ld?.[def.dataKey]?.data ?? null;
    // Legacy fallback for configs without dataKey
    const keys = ['layer0','layer1','layer2','layer3','layer4','output'];
    return ld?.[keys[li - 1]]?.data ?? null;
  }

  // ── Cleanup ────────────────────────────────────────────────────────────────

  _clearRF() {
    this._rfEpoch++;
    this._rfLines         = [];
    this._rfLineSegs      = null;
    this._rfLineCount     = 0;
    this._rfLineEndpoints = [];
    this._rfSrcDot        = null;
    this._rfDstDots       = [];
    this._rfLineOpacities = [];
    this._rfBaseColors    = [];
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
