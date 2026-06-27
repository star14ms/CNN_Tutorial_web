import { DrawingCanvas }  from './canvas-drawing.js';
import { ModelInference } from './model-inference.js';
import { Visualization }  from './visualization.js';
import { CameraController, PixelPicker, PixelHover } from './interaction.js';
import { debounce } from './utils.js';
import { MODEL_CONFIGS, DEFAULT_MODEL_ID } from './model-configs.js';

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

  let animMode       = false;
  let lastResult     = null;
  let lastPixels     = null;
  let currentModelId = DEFAULT_MODEL_ID;
  let _formulaTimer  = null;
  let _animStep      = 0;
  let _animContrib   = null;
  let _animDetail    = null;
  let _animInfo      = null;
  let _activeLayerLi = null;

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

    if (animMode) {
      const speed = getSpeed();
      const init  = viz.initRFAnimated(info);
      if (init) {
        _animContrib = init.contributing;
        _animDetail  = init.detail;
        _animInfo    = info;
        showDetailPanel(init.detail, info, true);
        startFormulaAnim(speed);
      }
    } else {
      const detail = viz.showReceptiveField(info);
      if (detail) showDetailPanel(detail, info, false);
    }
  }, () => {
    // Left-click on empty space — cancel selection
    stopFormulaAnim();
    viz.clearReceptiveField();
    detailPanel.style.display = 'none';
    setActiveLayer(null);
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
      await model.load(config, (ratio, msg) => {
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
    viz.setModelConfig(config);
    lastResult = null;
    lastPixels = null;
    predEl.textContent   = '—';
    confBarsEl.innerHTML = '';
    detailPanel.style.display = 'none';

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
    // Update header with total param count
    const header = $('layer-modal').querySelector('.layer-modal-header');
    if (header) {
      const total = config.totalParams != null
        ? ` <span class="lmi-total">${config.totalParams.toLocaleString()} params</span>`
        : '';
      header.innerHTML = `Layers${total}`;
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
      viz.update(result, pixels, false);
      showPrediction(result.output.data);
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
    renderEmpty();
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

  // Speed slider — apply immediately to current animation
  function updateSpeedLabel() {
    $('anim-speed-val').textContent = ANIM_SPEEDS[parseInt(speedSlider.value)] + ' ms';
  }
  if (speedSlider) {
    updateSpeedLabel();
    speedSlider.addEventListener('input', () => {
      updateSpeedLabel();
      if (_formulaTimer !== null) {
        // Restart from current step with new speed
        clearTimeout(_formulaTimer);
        _formulaTimer = null;
        startFormulaAnim(getSpeed());
      }
    });
  }

  // ── Detail panel close ─────────────────────────────────────────────────────
  if (closeDetail) {
    closeDetail.addEventListener('click', () => {
      stopFormulaAnim();
      detailPanel.style.display = 'none';
      viz.clearReceptiveField();
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

  // ── Detail panel rendering ─────────────────────────────────────────────────
  function showDetailPanel(detail, info, isAnimating) {
    detailPanel.style.display = 'block';
    renderDetailFormula(detail, info, detail.prevValues, isAnimating ? 0 : detail.prevValues.length);
  }

  function renderDetailFormula(detail, info, contributing, shownCount) {
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
      html += renderKatex(
        `y_j = \\mathrm{ReLU}\\!\\left(\\sum_{i} w_{j,i}\\cdot x_i + b_j\\right)`
      );
    } else {
      html += renderKatex(
        `y = \\mathrm{ReLU}\\!\\left(\\sum_{c,\\Delta y,\\Delta x} w_{c,\\Delta y,\\Delta x}\\cdot x_{c,\\,y+\\Delta y,\\,x+\\Delta x}+b\\right)`
      );
    }
    html += `</div>`;

    // ── Substituted formula: terms one by one ──
    html += `<div class="dp-formula-sub">`;

    if (shown.length === 0) {
      const placeholder = isPool
        ? 'y = \\max(\\ldots)'
        : 'y_j = \\mathrm{ReLU}(\\ldots+b_j)';
      html += `<div class="dp-term-placeholder">${renderKatex(placeholder, true)}</div>`;
    } else {
      html += `<div class="dp-terms-list">`;
      if (isPool) {
        for (let i = 0; i < shown.length; i += 3) {
          html += `<div class="dp-term-row">`;
          for (let j = i; j < Math.min(i + 3, shown.length); j++) {
            const p = shown[j];
            const v = p.val != null ? p.val.toFixed(4) : '?';
            html += `<div class="dp-term" data-index="${j}">${renderKatex(
              `x_{(${p.px},\\,${p.py})} = ${v}`, true
            )}</div>`;
          }
          html += `</div>`;
        }
        if (done) {
          html += `<div class="dp-term dp-result">${renderKatex(
            `y = \\max = \\mathbf{${outStr}}`, true
          )}</div>`;
        }
      } else if (isFC) {
        for (let i = 0; i < shown.length; i += 3) {
          html += `<div class="dp-term-row">`;
          for (let j = i; j < Math.min(i + 3, shown.length); j++) {
            const p = shown[j];
            const v = p.val != null ? p.val.toFixed(4) : '?';
            const sign = j === 0 ? '' : '+\\;';
            html += `<div class="dp-term" data-index="${j}">${renderKatex(
              `${sign}w_{${channel},${p.px}}\\cdot ${v}`, true
            )}</div>`;
          }
          html += `</div>`;
        }
        if (done) {
          html += `<div class="dp-term dp-result">${renderKatex(
            `+b \\;\\Rightarrow\\; y = \\mathrm{ReLU}(\\cdot) = \\mathbf{${outStr}}`, true
          )}</div>`;
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
            html += `<div class="dp-term" data-index="${j}">${renderKatex(
              `${sign}w_{${p.c},\\,${dx},\\,${dy}}\\cdot ${v}`, true
            )}</div>`;
          }
          html += `</div>`;
        }
        if (done) {
          html += `<div class="dp-term dp-result">${renderKatex(
            `+b \\;\\Rightarrow\\; y = \\mathrm{ReLU}(\\cdot) = \\mathbf{${outStr}}`, true
          )}</div>`;
        } else {
          html += `<div class="dp-term dp-muted">${renderKatex('+\\cdots+b)', true)}</div>`;
        }
      }
      html += `</div>`;
    }

    html += `</div>`;
    detailContent.innerHTML = html;
    const termsList = detailContent.querySelector('.dp-terms-list');
    if (termsList) termsList.scrollTop = termsList.scrollHeight;

    if (!done && prevValues.length > 0) {
      const pct = Math.round((shownCount / prevValues.length) * 100);
      detailProgress.innerHTML = `<div class="dp-progress">
        <div class="dp-progress-bar" style="width:${pct}%"></div>
        <span class="dp-progress-label">${shownCount} / ${prevValues.length} connections</span>
      </div>`;
    } else {
      detailProgress.innerHTML = '';
    }

    // Wire term hover → highlight corresponding 3D line red
    detailContent.querySelectorAll('.dp-term[data-index]').forEach(el => {
      el.addEventListener('mouseenter', () => {
        detailContent.querySelectorAll('.dp-term[data-index]').forEach(t => t.classList.remove('dp-term-active'));
        el.classList.add('dp-term-active');
        viz.highlightRFLine(parseInt(el.dataset.index));
      });
      el.addEventListener('mouseleave', () => {
        el.classList.remove('dp-term-active');
        const pinned = detailContent.querySelector('.dp-term-pinned');
        if (pinned) {
          viz.highlightRFLine(parseInt(pinned.dataset.index));
        } else {
          viz.highlightRFLine(-1);
        }
      });
      el.addEventListener('click', e => {
        e.stopPropagation();
        const idx = parseInt(el.dataset.index);
        const wasActive = el.classList.contains('dp-term-pinned');
        detailContent.querySelectorAll('.dp-term[data-index]').forEach(t => t.classList.remove('dp-term-pinned'));
        if (!wasActive) {
          el.classList.add('dp-term-pinned');
          viz.highlightRFLine(idx);
        } else {
          viz.highlightRFLine(-1);
        }
      });
    });
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
    const probs  = Array.from(softmax);
    const maxIdx = probs.indexOf(Math.max(...probs));
    predEl.textContent = `${maxIdx}  (${(probs[maxIdx]*100).toFixed(1)}%)`;
    confBarsEl.innerHTML = '';
    probs.forEach((p, i) => {
      const row = document.createElement('div');
      row.className = 'conf-row' + (i === maxIdx ? ' top' : '');
      row.innerHTML = `
        <span class="conf-label">${i}</span>
        <div class="conf-bar-wrap">
          <div class="conf-bar" style="width:${p*100}%"></div>
        </div>
        <span class="conf-pct">${(p*100).toFixed(1)}%</span>`;
      confBarsEl.appendChild(row);
    });
  }

  // ── Initial model load ─────────────────────────────────────────────────────
  await loadModel(DEFAULT_MODEL_ID);
}

main();
