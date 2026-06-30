import { DrawingCanvas }  from './canvas-drawing.js';
import { ModelInference } from './model-inference.js';
import { Visualization }  from './visualization.js';
import { CameraController, PixelPicker, PixelHover } from './interaction.js';
import { debounce } from './utils.js';
import { MODEL_CONFIGS, DEFAULT_MODEL_ID } from './model-configs.js';
import { DATASET_CONFIGS, DEFAULT_DATASET_ID } from './dataset-configs.js';
import { DatasetLoader } from './dataset-loader.js';

const $ = id => document.getElementById(id);

// Speed steps: index 0 = leftmost (slowest), index 6 = rightmost (fastest)
const ANIM_SPEEDS = [1000, 500, 200, 100, 50, 20, 10];

async function main() {
  const statusEl      = $('status');
  const progressEl    = $('progress-bar');
  const predEl        = $('prediction');
  const confBarsEl    = $('conf-bars');
  const clearBtn      = $('clear-btn');
  const threeDiv      = $('three-container');
  const detailPanel    = $('detail-panel');
  const detailContent  = $('detail-content');
  const detailProgress = $('detail-progress');
  const animToggle    = $('anim-toggle');
  const closeDetail   = $('close-detail');
  const layerModalList = $('layer-modal-list');
  const speedWrap     = $('anim-speed-wrap');
  const speedSlider   = $('anim-speed');

  let animMode        = false;
  let lastResult      = null;
  let lastPixels      = null;
  let currentModelId  = DEFAULT_MODEL_ID;
  let currentDatasetId = DEFAULT_DATASET_ID;
  let _formulaTimer  = null;
  let _animStep      = 0;
  let _animContrib   = null;
  let _animDetail    = null;
  let _animInfo      = null;
  let _activeLayerLi  = null;
  let _selectedPixel  = null; // { layerIdx, channel, x, y } — persists across inference updates
  let _dpViewMode     = 'expansion';
  let _lastDetail     = null;
  let _lastInfo       = null;
  let _lastContrib    = null;
  let _lastShownCount = 0;

  // ── 3D Visualization ───────────────────────────────────────────────────────
  const viz = new Visualization(threeDiv, MODEL_CONFIGS[DEFAULT_MODEL_ID]);
  new CameraController(viz.camera, threeDiv, () => viz.scene);

  // Hover tooltip
  const hoverTip = document.createElement('div');
  hoverTip.id = 'hover-tooltip';
  hoverTip.style.cssText =
    'position:absolute;pointer-events:none;display:none;background:rgba(16,21,48,0.92);' +
    'border:1px solid #2a2a4a;border-radius:6px;padding:5px 9px;font-size:0.75rem;' +
    'color:#c9d1d9;z-index:7;white-space:nowrap;';
  threeDiv.parentElement.appendChild(hoverTip);

  new PixelHover(viz.camera, threeDiv, () => viz.getMeshes(), (mesh, uv) => {
    const info = viz.setHoverPixel(mesh, uv);
    if (info) {
      hoverTip.style.display = 'block';
      hoverTip.innerHTML =
        `<b>${info.layerName}</b> Ch ${info.channel} &nbsp; (${info.x},${info.y})` +
        `<br>val: ${info.rawValue.toFixed(4)}`;
    } else {
      hoverTip.style.display = 'none';
    }
  });

  threeDiv.addEventListener('mousemove', e => {
    const r = threeDiv.getBoundingClientRect();
    hoverTip.style.left = (e.clientX - r.left + 14) + 'px';
    hoverTip.style.top  = (e.clientY - r.top  - 10) + 'px';
  });

  new PixelPicker(viz.camera, threeDiv, () => viz.getMeshes(), (mesh, uv) => {
    const info = viz.handleRaycastHit(mesh, uv);
    if (!info) return;

    // Deselect layer highlight when clicking a pixel
    setActiveLayer(null);

    stopFormulaAnim();
    _selectedPixel = { layerIdx: info.layerIdx, channel: info.channel, x: info.x, y: info.y };
    triggerPixelSelection(info);
  }, () => {
    // Left-click on empty space — cancel selection
    stopFormulaAnim();
    viz.clearReceptiveField();
    detailPanel.style.display = 'none';
    setActiveLayer(null);
    _selectedPixel = null;
  });

  threeDiv.addEventListener('contextmenu', e => {
    e.preventDefault();
  });

  // ── Model loading ──────────────────────────────────────────────────────────
  const model = new ModelInference();

  async function loadModel(modelId) {
    const config = MODEL_CONFIGS[modelId];
    currentModelId = modelId;

    document.querySelectorAll('.model-btn').forEach(btn =>
      btn.classList.toggle('active', btn.dataset.modelId === modelId));

    $('loading-overlay').style.display = 'flex';
    statusEl.textContent = `Loading ${config.label}…`;
    progressEl.style.width = '0%';

    try {
      await model.load(config, DATASET_CONFIGS[currentDatasetId], (ratio, msg) => {
        statusEl.textContent = msg;
        progressEl.style.width = (ratio * 100) + '%';
      });
    } catch (err) {
      statusEl.textContent = '⚠ ' + err.message;
      $('loading-overlay').style.display = 'flex';
      return;
    }

    statusEl.textContent = 'Draw a digit →';
    $('loading-overlay').style.display = 'none';

    rebuildLayerModal(config);
    viz.setDatasetConfig(DATASET_CONFIGS[currentDatasetId]);
    viz.setModelConfig(config);
    viz.setParameters(model.parameters);
    lastResult = null;
    lastPixels = null;
    predEl.textContent   = '—';
    confBarsEl.innerHTML = '';
    detailPanel.style.display = 'none';
    _selectedPixel = null;

    await renderEmpty();
  }

  async function renderEmpty() {
    const blankPixels = new Float32Array(784);
    try {
      const result = await model.inferAllLayers(blankPixels);
      lastResult = result;
      lastPixels = blankPixels;
      viz.update(result, blankPixels, false);
    } catch (_) { /* ignore */ }
  }

  function fmtParams(n) {
    if (n === 0) return '—';
    return n.toLocaleString();
  }

  function rebuildLayerModal(config) {
    // Update header with total param count + torchinfo stats
    const header = $('layer-modal').querySelector('.layer-modal-header');
    if (header) {
      const total = config.totalParams != null
        ? ` <span class="lmi-total">${config.totalParams.toLocaleString()} params</span>`
        : '';
      let statsHtml = '';
      if (config.torchinfo) {
        const t = config.torchinfo;
        statsHtml = `<div class="lmi-torchinfo">`
          + (t.multAddsM    != null ? `<span>Mult-Adds: ${t.multAddsM.toFixed(2)} M</span>` : '')
          + (t.paramsSizeMB != null ? `<span>Params: ${t.paramsSizeMB.toFixed(2)} MB</span>` : '')
          + (t.fwdBwdSizeMB != null ? `<span>Fwd/Bwd: ${t.fwdBwdSizeMB.toFixed(2)} MB</span>` : '')
          + (t.totalSizeMB  != null ? `<span>Total: ${t.totalSizeMB.toFixed(2)} MB</span>` : '')
          + `</div>`;
      }
      header.innerHTML = `Layers${total}${statsHtml}`;
    }

    layerModalList.innerHTML = '';
    _activeLayerLi = null;
    config.layerDefs.forEach((def, li) => {
      if (li === 0) return;
      const item = document.createElement('div');
      item.className  = 'layer-modal-item';
      item.dataset.li = li;

      // Layer params = sum of sublayer params
      const layerParams = def.sublayers
        ? def.sublayers.reduce((s, sl) => s + sl.params, 0)
        : null;

      const mainRow = document.createElement('div');
      mainRow.className = 'lmi-main';
      mainRow.innerHTML = `<span class="lmi-name">${def.name}</span>`
        + (layerParams != null
          ? `<span class="lmi-params">${fmtParams(layerParams)}</span>`
          : '');
      item.appendChild(mainRow);

      if (def.sublayers && def.sublayers.length > 0) {
        const subList = document.createElement('div');
        subList.className = 'lmi-sublayers';
        def.sublayers.forEach(sl => {
          const row = document.createElement('div');
          row.className = 'lmi-sub';
          row.innerHTML = `<span>${sl.type}</span><span>${fmtParams(sl.params)}</span>`;
          subList.appendChild(row);
        });
        item.appendChild(subList);
      }

      item.addEventListener('click', () => {
        stopFormulaAnim();
        detailPanel.style.display = 'none';
            if (_activeLayerLi === li) {
          viz.clearReceptiveField();
          setActiveLayer(null);
        } else {
          setActiveLayer(li);
          viz.showLayerConnections(li);
        }
      });
      layerModalList.appendChild(item);
    });
  }

  function setActiveLayer(li) {
    _activeLayerLi = li;
    layerModalList.querySelectorAll('.layer-modal-item').forEach(el => {
      el.classList.toggle('active', parseInt(el.dataset.li) === li);
    });
  }

  // ── Drawing canvas ─────────────────────────────────────────────────────────
  const runInference = debounce(async () => {
    if (!drawing.hasContent()) return;
    const pixels = drawing.getPixels();
    lastPixels = pixels;
    try {
      const result = await model.inferAllLayers(pixels);
      lastResult = result;
      viz.update(result, pixels, animMode, getSpeed());
      showPrediction(result.output.data);
      // Re-trigger selection with fresh data
      if (_selectedPixel) {
        const { layerIdx, channel, x, y } = _selectedPixel;
        const freshInfo = viz.getPixelInfo(layerIdx, channel, x, y);
        if (freshInfo) {
          stopFormulaAnim();
          triggerPixelSelection(freshInfo);
        }
      } else if (_activeLayerLi !== null) {
        viz.showLayerConnections(_activeLayerLi);
      }
    } catch (err) { console.error('Inference error:', err); }
  }, 300);

  const drawing = new DrawingCanvas($('draw-canvas'), runInference);

  clearBtn.addEventListener('click', () => {
    drawing.clear();
    viz.reset();
    stopFormulaAnim();
    lastResult = null;
    lastPixels = null;
    predEl.textContent    = '—';
    confBarsEl.innerHTML  = '';
    detailPanel.style.display = 'none';
    viz.clearReceptiveField();
    setActiveLayer(null);
    _selectedPixel = null;
    renderEmpty();
  });

  async function loadAccuracies(dsId) {
    try {
      const res = await fetch(`public/models/${dsId}/accuracies.json`);
      if (!res.ok) return;
      const acc = await res.json();
      document.querySelectorAll('.model-btn').forEach(btn => {
        const modelId = btn.dataset.modelId;
        if (acc[modelId] != null) {
          const base = btn.dataset.titleBase || btn.title.replace(/\s*—\s*[\d.]+%.*$/, '');
          btn.dataset.titleBase = base;
          btn.title = `${base} — ${acc[modelId].toFixed(2)}% test acc`;
        }
      });
    } catch (_) { /* accuracies.json not available yet */ }
  }

  // ── Label selector ─────────────────────────────────────────────────────────
  function updateLabelSelector(datasetId) {
    const classLabels = DATASET_CONFIGS[datasetId].classLabels;
    const sel = $('ds-digit');
    while (sel.options.length > 1) sel.remove(1);
    classLabels.forEach((lbl, i) => {
      const opt = document.createElement('option');
      opt.value = i;
      opt.textContent = lbl;
      sel.appendChild(opt);
    });
    // Measure longest label to set adaptive conf-label width
    const probe = document.createElement('span');
    probe.style.cssText = 'position:absolute;visibility:hidden;font-size:0.72rem;white-space:nowrap';
    document.body.appendChild(probe);
    const maxW = Math.max(...classLabels.map(l => { probe.textContent = l; return probe.offsetWidth; }));
    document.body.removeChild(probe);
    document.documentElement.style.setProperty('--conf-label-w', `${maxW + 4}px`);
  }

  // ── Dataset selector ───────────────────────────────────────────────────────
  document.querySelectorAll('.dataset-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const dsId = btn.dataset.dataset;
      if (dsId === currentDatasetId || btn.disabled) return;
      currentDatasetId = dsId;
      document.querySelectorAll('.dataset-btn').forEach(b =>
        b.classList.toggle('active', b.dataset.dataset === dsId));
      updateLabelSelector(dsId);
      dataset.reset();
      drawing.clear();
      await loadAccuracies(dsId);
      await loadModel(currentModelId);
    });
  });

  // ── Model selector ─────────────────────────────────────────────────────────
  document.querySelectorAll('.model-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      if (btn.dataset.modelId === currentModelId) return;
      drawing.clear();
      await loadModel(btn.dataset.modelId);
    });
  });

  // ── Animation toggle ───────────────────────────────────────────────────────
  animToggle.addEventListener('click', () => {
    animMode = !animMode;
    animToggle.textContent = animMode ? 'Anim: ON' : 'Anim: OFF';
    animToggle.classList.toggle('active', animMode);
    speedWrap.style.display = animMode ? 'block' : 'none';
    if (lastResult && lastPixels) viz.update(lastResult, lastPixels, false);
  });

  const SPEED_LABELS = ['1/10×', '1/5×', '1/2×', '1×', '2×', '5×', '10×'];
  // Speed slider — apply immediately to current animation
  function updateSpeedLabel() {
    const idx = parseInt(speedSlider?.value ?? 3, 10);
    $('anim-speed-val').textContent = SPEED_LABELS[Math.max(0, Math.min(6, idx))];
  }
  if (speedSlider) {
    updateSpeedLabel();
    speedSlider.addEventListener('input', () => {
      updateSpeedLabel();
      const speed = getSpeed();
      if (_formulaTimer !== null) {
        // Restart from current step with new speed
        clearTimeout(_formulaTimer);
        _formulaTimer = null;
        startFormulaAnim(speed);
      }
      viz.setAnimationSpeed(speed);
    });
  }

  // ── Detail panel close ─────────────────────────────────────────────────────
  if (closeDetail) {
    closeDetail.addEventListener('click', () => {
      stopFormulaAnim();
      detailPanel.style.display = 'none';
      viz.clearReceptiveField();
      _selectedPixel = null;
    });
  }

  // ── Formula animation helpers ──────────────────────────────────────────────
  function getSpeed() {
    const idx = parseInt(speedSlider?.value ?? 3, 10);
    return ANIM_SPEEDS[Math.max(0, Math.min(6, idx))];
  }

  function stopFormulaAnim() {
    if (_formulaTimer !== null) { clearTimeout(_formulaTimer); _formulaTimer = null; }
    _animStep   = 0;
    _animContrib = null;
    _animDetail  = null;
    _animInfo    = null;
  }

  function startFormulaAnim(speed) {
    if (!_animContrib || !_animDetail || !_animInfo) return;
    const lineDuration = Math.min(speed * 0.7, 400);
    const step = () => {
      if (_formulaTimer === null) return;
      if (_animStep >= _animContrib.length) {
        _formulaTimer = null;
        renderDetailFormula(_animDetail, _animInfo, _animContrib, _animContrib.length);
        return;
      }
      viz.addRFLine(_animContrib[_animStep], lineDuration);
      const shownCount = ++_animStep;
      // Schedule next step immediately so line timing is fixed regardless of KaTeX render cost
      _formulaTimer = setTimeout(step, speed);
      renderDetailFormula(_animDetail, _animInfo, _animContrib, shownCount);
    };
    _formulaTimer = setTimeout(step, speed);
  }

  function triggerPixelSelection(info) {
    if (animMode) {
      const init = viz.initRFAnimated(info);
      if (init) {
        _animContrib = init.contributing;
        _animDetail  = init.detail;
        _animInfo    = info;
        showDetailPanel(init.detail, info, true);
        startFormulaAnim(getSpeed());
      }
    } else {
      const detail = viz.showReceptiveField(info);
      if (detail) showDetailPanel(detail, info, false);
    }
  }

  // ── Detail panel rendering ─────────────────────────────────────────────────
  function showDetailPanel(detail, info, isAnimating) {
    detailPanel.style.display = 'block';
    renderDetailFormula(detail, info, detail.prevValues, isAnimating ? 0 : detail.prevValues.length);
  }

  function renderDetailFormula(detail, info, contributing, shownCount) {
    _lastDetail     = detail;
    _lastInfo       = info;
    _lastContrib    = contributing;
    _lastShownCount = shownCount;

    const { layerName, channel, px, py, outVal, conn, prevValues } = detail;

    if (!conn) {
      detailContent.innerHTML = `<div class="dp-title"><b>${layerName}</b> — input layer</div>`;
      return;
    }

    const isPool = conn.type === 'pool';
    const isFC   = conn.type === 'fc';
    const outStr = outVal != null ? outVal.toFixed(4) : '?';
    const shown  = prevValues.slice(0, shownCount);
    const done   = shownCount >= prevValues.length;

    let html = `<div class="dp-title"><b>${layerName}</b> &nbsp; | &nbsp; Ch ${channel} &nbsp; Pixel: (${px},${py}) &nbsp; Output: ${outStr}</div>`;

    // ── General formula ──
    html += `<div class="dp-formula-general">`;
    if (isPool) {
      html += renderKatex(`y = \\max_{\\Delta y,\\Delta x}\\!\\left(x_{c,\\;ky+\\Delta y,\\;kx+\\Delta x}\\right)`);
    } else if (isFC) {
      html += renderKatex(`y_j = \\mathrm{ReLU}\\!\\left(\\sum_{i} w_{j,i}\\cdot x_i + b_j\\right)`);
    } else {
      html += renderKatex(`y = \\mathrm{ReLU}\\!\\left(\\sum_{c,\\Delta y,\\Delta x} w_{c,\\Delta y,\\Delta x}\\cdot x_{c,\\,y+\\Delta y,\\,x+\\Delta x}+b\\right)`);
    }
    html += `</div>`;

    // ── View mode toggle ──
    html += `<div class="dp-view-toggle">
      <button class="dp-toggle-btn${_dpViewMode === 'expansion' ? ' active' : ''}" data-mode="expansion">Expansion</button>
      <button class="dp-toggle-btn${_dpViewMode === 'matrix' ? ' active' : ''}" data-mode="matrix">Matrix</button>
    </div>`;

    // ── Substituted formula ──
    html += `<div class="dp-formula-sub">`;

    if (_dpViewMode === 'matrix') {
      html += buildMatrixSubHtml(detail, prevValues, shownCount, done, outStr);
    } else {
      // Expansion mode
      if (shown.length === 0) {
        const placeholder = isPool ? 'y = \\max(\\ldots)' : 'y_j = \\mathrm{ReLU}(\\ldots+b_j)';
        html += `<div class="dp-term-placeholder">${renderKatex(placeholder, true)}</div>`;
      } else {
        html += `<div class="dp-terms-list">`;
        if (isPool) {
          for (let i = 0; i < shown.length; i += 3) {
            html += `<div class="dp-term-row">`;
            for (let j = i; j < Math.min(i + 3, shown.length); j++) {
              const p = shown[j];
              const v = p.val != null ? p.val.toFixed(4) : '?';
              html += `<div class="dp-term" data-index="${j}">${renderKatex(`x_{(${p.px},\\,${p.py})} = ${v}`, true)}</div>`;
            }
            html += `</div>`;
          }
          if (done) {
            html += `<div class="dp-term dp-result">${renderKatex(`y = \\max = \\mathbf{${outStr}}`, true)}</div>`;
          }
        } else if (isFC) {
          for (let i = 0; i < shown.length; i += 3) {
            html += `<div class="dp-term-row">`;
            for (let j = i; j < Math.min(i + 3, shown.length); j++) {
              const p = shown[j];
              const v = p.val != null ? p.val.toFixed(4) : '?';
              const sign = j === 0 ? '' : '+\\;';
              html += `<div class="dp-term" data-index="${j}">${renderKatex(`${sign}w_{${channel},${p.px}}\\cdot ${v}`, true)}</div>`;
            }
            html += `</div>`;
          }
          if (done) {
            html += `<div class="dp-term dp-result">${renderKatex(`+b \\;\\Rightarrow\\; y = \\mathrm{ReLU}(\\cdot) = \\mathbf{${outStr}}`, true)}</div>`;
          } else {
            html += `<div class="dp-term dp-muted">${renderKatex('+\\cdots+b)', true)}</div>`;
          }
        } else {
          for (let i = 0; i < shown.length; i += 3) {
            html += `<div class="dp-term-row">`;
            for (let j = i; j < Math.min(i + 3, shown.length); j++) {
              const p = shown[j];
              const v = p.val != null ? p.val.toFixed(4) : '?';
              const dx = p.px - px;
              const dy = p.py - py;
              const sign = j === 0 ? '' : '+\\;';
              html += `<div class="dp-term" data-index="${j}">${renderKatex(`${sign}w_{${p.c},\\,${dx},\\,${dy}}\\cdot ${v}`, true)}</div>`;
            }
            html += `</div>`;
          }
          if (done) {
            html += `<div class="dp-term dp-result">${renderKatex(`+b \\;\\Rightarrow\\; y = \\mathrm{ReLU}(\\cdot) = \\mathbf{${outStr}}`, true)}</div>`;
          } else {
            html += `<div class="dp-term dp-muted">${renderKatex('+\\cdots+b)', true)}</div>`;
          }
        }
        html += `</div>`;
      }
    }

    html += `</div>`;
    detailContent.innerHTML = html;
    const formulaSub = detailContent.querySelector('.dp-formula-sub');
    if (formulaSub) formulaSub.scrollTop = formulaSub.scrollHeight;

    if (!done && prevValues.length > 0) {
      const pct = Math.round((shownCount / prevValues.length) * 100);
      detailProgress.innerHTML = `<div class="dp-progress">
        <div class="dp-progress-bar" style="width:${pct}%"></div>
        <span class="dp-progress-label">${shownCount} / ${prevValues.length} connections</span>
      </div>`;
    } else {
      detailProgress.innerHTML = '';
    }

    // Wire toggle buttons
    detailContent.querySelectorAll('.dp-toggle-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        _dpViewMode = btn.dataset.mode;
        renderDetailFormula(_lastDetail, _lastInfo, _lastContrib, _lastShownCount);
      });
    });

    // Wire term hover/click → highlight 3D line; multi-cell aware
    detailContent.querySelectorAll('.dp-term[data-index]').forEach(el => {
      el.addEventListener('mouseenter', () => {
        const idx = el.dataset.index;
        detailContent.querySelectorAll('.dp-term[data-index]').forEach(t => t.classList.remove('dp-term-active'));
        detailContent.querySelectorAll(`.dp-term[data-index="${idx}"]`).forEach(t => t.classList.add('dp-term-active'));
        viz.highlightRFLine(parseInt(idx));
      });
      el.addEventListener('mouseleave', () => {
        detailContent.querySelectorAll('.dp-term[data-index]').forEach(t => t.classList.remove('dp-term-active'));
        const pinned = detailContent.querySelector('.dp-term-pinned');
        if (pinned) {
          const pidx = pinned.dataset.index;
          detailContent.querySelectorAll(`.dp-term[data-index="${pidx}"]`).forEach(t => t.classList.add('dp-term-active'));
          viz.highlightRFLine(parseInt(pidx));
        } else {
          viz.highlightRFLine(-1);
        }
      });
      el.addEventListener('click', e => {
        e.stopPropagation();
        const idx = el.dataset.index;
        const wasActive = el.classList.contains('dp-term-pinned');
        detailContent.querySelectorAll('.dp-term[data-index]').forEach(t => t.classList.remove('dp-term-pinned'));
        if (!wasActive) {
          detailContent.querySelectorAll(`.dp-term[data-index="${idx}"]`).forEach(t => t.classList.add('dp-term-pinned'));
          viz.highlightRFLine(parseInt(idx));
        } else {
          viz.highlightRFLine(-1);
        }
      });
    });
  }

  function buildMatrixSubHtml(detail, prevValues, shownCount, done, outStr) {
    const { channel, px, py, conn } = detail;
    const isPool = conn.type === 'pool';
    const isFC   = conn.type === 'fc';
    let html = '';

    if (isPool) {
      const k = conn.kernel || 2;
      html += `<div class="dp-mat-wrap"><div class="dp-mat-brace dp-mat-brace-l"></div>`;
      html += `<div class="dp-mat-grid" style="grid-template-columns:repeat(${k},auto)">`;
      for (let j = 0; j < k * k; j++) {
        const revealed = j < shownCount;
        const p = revealed ? prevValues[j] : null;
        const v = revealed ? (p.val != null ? p.val.toFixed(3) : '?') : null;
        const di = revealed ? `data-index="${j}"` : '';
        html += `<span class="dp-term dp-mat-cell${revealed ? '' : ' dp-mat-dim'}" ${di}>${renderKatex(revealed ? v : '\\cdot', true)}</span>`;
      }
      html += `</div><div class="dp-mat-brace dp-mat-brace-r"></div></div>`;
      if (done) html += `<div class="dp-mat-result">${renderKatex(`y = \\max = \\mathbf{${outStr}}`, true)}</div>`;

    } else if (isFC) {
      html += `<div class="dp-mat-fc-pair">`;
      // Weight column vector
      html += `<div class="dp-mat-wrap"><div class="dp-mat-brace dp-mat-brace-l"></div><div class="dp-mat-fc-col">`;
      for (let j = 0; j < shownCount; j++) {
        const p = prevValues[j];
        html += `<span class="dp-term dp-mat-cell" data-index="${j}">${renderKatex(`w_{${channel},${p.px}}`, true)}</span>`;
      }
      if (!done) html += `<span class="dp-mat-dim dp-mat-cell">${renderKatex('\\vdots', true)}</span>`;
      html += `</div><div class="dp-mat-brace dp-mat-brace-r"></div></div>`;
      html += `<span class="dp-mat-op">·</span>`;
      // Input column vector
      html += `<div class="dp-mat-wrap"><div class="dp-mat-brace dp-mat-brace-l"></div><div class="dp-mat-fc-col">`;
      for (let j = 0; j < shownCount; j++) {
        const p = prevValues[j];
        const v = p.val != null ? p.val.toFixed(3) : '?';
        html += `<span class="dp-term dp-mat-cell" data-index="${j}">${renderKatex(v, true)}</span>`;
      }
      if (!done) html += `<span class="dp-mat-dim dp-mat-cell">${renderKatex('\\vdots', true)}</span>`;
      html += `</div><div class="dp-mat-brace dp-mat-brace-r"></div></div>`;
      html += `</div>`;
      if (done) html += `<div class="dp-mat-result">${renderKatex(`+b \\;\\Rightarrow\\; y = \\mathrm{ReLU}(\\cdot) = \\mathbf{${outStr}}`, true)}</div>`;

    } else {
      // Conv: per-channel weight matrix ⊙ input patch
      const k = conn.kernel || 3;
      const half = Math.floor(k / 2);
      const prevCCount = conn.prevChannels || 1;
      for (let c = 0; c < prevCCount; c++) {
        const baseIdx = c * k * k;
        html += `<div class="dp-mat-channel-row">`;
        html += `<span class="dp-mat-ch-label">${renderKatex(`c\\!=\\!${c}:`, true)}</span>`;
        // Weight matrix W_c
        html += `<div class="dp-mat-wrap"><div class="dp-mat-brace dp-mat-brace-l"></div>`;
        html += `<div class="dp-mat-grid" style="grid-template-columns:repeat(${k},auto)">`;
        for (let dyIdx = 0; dyIdx < k; dyIdx++) {
          for (let dxIdx = 0; dxIdx < k; dxIdx++) {
            const j = baseIdx + dyIdx * k + dxIdx;
            const revealed = j < shownCount;
            const dy = dyIdx - half, dx = dxIdx - half;
            const di = revealed ? `data-index="${j}"` : '';
            html += `<span class="dp-term dp-mat-cell${revealed ? '' : ' dp-mat-dim'}" ${di}>${renderKatex(`w_{${c},${dy},${dx}}`, true)}</span>`;
          }
        }
        html += `</div><div class="dp-mat-brace dp-mat-brace-r"></div></div>`;
        html += `<span class="dp-mat-op">⊙</span>`;
        // Input patch X_c
        html += `<div class="dp-mat-wrap"><div class="dp-mat-brace dp-mat-brace-l"></div>`;
        html += `<div class="dp-mat-grid" style="grid-template-columns:repeat(${k},auto)">`;
        for (let dyIdx = 0; dyIdx < k; dyIdx++) {
          for (let dxIdx = 0; dxIdx < k; dxIdx++) {
            const j = baseIdx + dyIdx * k + dxIdx;
            const revealed = j < shownCount;
            const p = revealed ? prevValues[j] : null;
            const v = revealed ? (p.val != null ? p.val.toFixed(3) : '?') : null;
            const di = revealed ? `data-index="${j}"` : '';
            html += `<span class="dp-term dp-mat-cell${revealed ? '' : ' dp-mat-dim'}" ${di}>${renderKatex(revealed ? v : '\\cdot', true)}</span>`;
          }
        }
        html += `</div><div class="dp-mat-brace dp-mat-brace-r"></div></div>`;
        if (c < prevCCount - 1) html += `<span class="dp-mat-plus">+</span>`;
        html += `</div>`;
      }
      if (done) html += `<div class="dp-mat-result">${renderKatex(`+b \\;\\Rightarrow\\; y = \\mathrm{ReLU}(\\cdot) = \\mathbf{${outStr}}`, true)}</div>`;
    }
    return html;
  }

  function renderKatex(latex, inline = false) {
    if (window.katex) {
      try {
        return window.katex.renderToString(latex, { throwOnError: false, displayMode: !inline });
      } catch (_) { /* fall through */ }
    }
    return `<span class="dp-latex-fallback">${latex}</span>`;
  }

  // ── Prediction display ─────────────────────────────────────────────────────
  function showPrediction(softmax) {
    const probs      = Array.from(softmax);
    const maxIdx     = probs.indexOf(Math.max(...probs));
    const classLabels = DATASET_CONFIGS[currentDatasetId].classLabels;
    predEl.textContent = `${classLabels[maxIdx]}  (${(probs[maxIdx]*100).toFixed(1)}%)`;
    confBarsEl.innerHTML = '';
    probs.forEach((p, i) => {
      const row = document.createElement('div');
      row.className = 'conf-row' + (i === maxIdx ? ' top' : '');
      row.innerHTML = `
        <span class="conf-label">${classLabels[i]}</span>
        <div class="conf-bar-wrap">
          <div class="conf-bar" style="width:${p*100}%"></div>
        </div>
        <span class="conf-pct">${(p*100).toFixed(1)}%</span>`;
      confBarsEl.appendChild(row);
    });
  }

  // ── Dataset browser ────────────────────────────────────────────────────────
  const dataset = new DatasetLoader();

  let _dsSplit   = 'test';
  let _dsFilter  = 'all';

  const dsIndexInput = $('ds-index');
  const dsLoadBtn    = $('ds-load');
  const dsRandomBtn  = $('ds-random');
  const dsStatus     = $('ds-status');
  const dsTrueLabel  = $('ds-true-label');
  const dsDigitSel   = $('ds-digit');

  function setDsStatus(msg) { dsStatus.textContent = msg; }

  function showTrueLabel(idx, split) {
    const lbl = dataset.getLabel(split, idx);
    if (lbl !== null) {
      const classLabels = DATASET_CONFIGS[currentDatasetId].classLabels;
      dsTrueLabel.textContent = `True label: ${classLabels[lbl] ?? lbl}`;
      dsTrueLabel.style.display = 'block';
    } else {
      dsTrueLabel.style.display = 'none';
    }
  }

  async function loadAndInferDataset(split, index) {
    if (!dataset.isLoaded(split)) {
      setDsStatus(`Loading ${split} set…`);
      try { await dataset.load(split, DATASET_CONFIGS[currentDatasetId]); }
      catch (e) {
        setDsStatus(`Error: run: python train/export_dataset.py --dataset ${currentDatasetId}`);
        return;
      }
    }
    const n = dataset.size(split);
    if (index < 0 || index >= n) {
      setDsStatus(`Index out of range (0–${n - 1})`);
      return;
    }
    setDsStatus(`Loading image…`);
    const pixels = await dataset.getImage(split, index);
    showTrueLabel(index, split);
    setDsStatus(`${split} #${index}`);

    // Paint the dataset image onto the draw canvas for visual feedback
    const imgData = new ImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const v = Math.round(pixels[i] * 255);
      imgData.data[i * 4]     = v;
      imgData.data[i * 4 + 1] = v;
      imgData.data[i * 4 + 2] = v;
      imgData.data[i * 4 + 3] = 255;
    }
    drawing.loadImageData(imgData);

    lastPixels = pixels;
    try {
      const result = await model.inferAllLayers(pixels);
      lastResult = result;
      viz.update(result, pixels, animMode, getSpeed());
      showPrediction(result.output.data);
      if (_selectedPixel) {
        const { layerIdx, channel, x, y } = _selectedPixel;
        const freshInfo = viz.getPixelInfo(layerIdx, channel, x, y);
        if (freshInfo) { stopFormulaAnim(); triggerPixelSelection(freshInfo); }
      } else if (_activeLayerLi !== null) {
        viz.showLayerConnections(_activeLayerLi);
      }
    } catch (err) { console.error('Inference error:', err); }
  }

  // Split toggle (Test / Train)
  document.querySelectorAll('[data-split]').forEach(btn => {
    btn.addEventListener('click', () => {
      _dsSplit = btn.dataset.split;
      document.querySelectorAll('[data-split]').forEach(b =>
        b.classList.toggle('active', b.dataset.split === _dsSplit));
      dsTrueLabel.style.display = 'none';
    });
  });

  // Filter toggle (Any / Correct / Wrong)
  document.querySelectorAll('[data-filter]').forEach(btn => {
    btn.addEventListener('click', () => {
      _dsFilter = btn.dataset.filter;
      document.querySelectorAll('[data-filter]').forEach(b =>
        b.classList.toggle('active', b.dataset.filter === _dsFilter));
    });
  });

  // Load by index
  dsLoadBtn.addEventListener('click', async () => {
    const idx = parseInt(dsIndexInput.value, 10);
    if (isNaN(idx)) { setDsStatus('Enter a valid index'); return; }
    await loadAndInferDataset(_dsSplit, idx);
  });
  dsIndexInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') dsLoadBtn.click();
  });

  // Random sample
  dsRandomBtn.addEventListener('click', async () => {
    const digit  = dsDigitSel.value === '' ? null : parseInt(dsDigitSel.value, 10);
    const filter = _dsFilter;

    if (!dataset.isLoaded(_dsSplit)) {
      setDsStatus(`Loading ${_dsSplit} set…`);
      try { await dataset.load(_dsSplit, DATASET_CONFIGS[currentDatasetId]); }
      catch (e) { setDsStatus(`Error: run: python train/export_dataset.py --dataset ${currentDatasetId}`); return; }
    }

    // TP/FP filtering needs predictions cached
    if ((filter === 'correct' || filter === 'incorrect') && _dsSplit === 'test') {
      if (!dataset.hasPredictions(currentModelId)) {
        const dsConfig = DATASET_CONFIGS[currentDatasetId];
        const loaded = await dataset.tryLoadPredictions(
          currentModelId, `${dsConfig.modelsPath}/${currentModelId}`
        );
        if (!loaded) {
          setDsStatus('Computing predictions (10k images)…');
          await dataset.computeTestPredictions(currentModelId,
            async (pixels) => {
              const result = await model.inferAllLayers(pixels);
              const probs  = Array.from(result.output.data);
              return probs.indexOf(Math.max(...probs));
            },
            (done, total) => setDsStatus(`Computing… ${done}/${total}`)
          );
        }
      }
    }

    const sample = dataset.randomSample(_dsSplit, filter, digit, currentModelId);
    if (!sample) {
      setDsStatus('No matching samples found');
      return;
    }
    dsIndexInput.value = sample.index;
    await loadAndInferDataset(_dsSplit, sample.index);
  });

  // ── Initial model load ─────────────────────────────────────────────────────
  updateLabelSelector(DEFAULT_DATASET_ID);
  await loadAccuracies(DEFAULT_DATASET_ID);
  await loadModel(DEFAULT_MODEL_ID);
}

main();
