# CLAUDE.md — CNN Tutorial Web Project

## Project Overview

Client-side CNN visualization web app. Users draw a digit on a canvas, run browser-side ONNX inference, and explore each layer's feature maps in an interactive Three.js 3D scene. No server required.

**Stack:** Vanilla JS (ES modules), Three.js r160, ONNX Runtime Web 1.14, KaTeX 0.16.9

---

## Key Files

| File | Role |
|------|------|
| `index.html` | Entry point, layout |
| `src/js/main.js` | App orchestration, pixel click handling, animation loop |
| `src/js/layer-renderer.js` | Three.js layer geometry, receptive field lines, highlighting |
| `src/js/model-configs.js` | Layer definitions and connectivity for each model |
| `src/js/model-inference.js` | ONNX session loading and inference |
| `src/css/style.css` | All styling |
| `public/models/` | ONNX model files (committed to git) |
| `train/` | PyTorch training scripts |

---

## Architecture

- **Layers** are rendered as Three.js `Group` objects positioned along the Y-axis (`LAYER_SPACING = 180`).
- Each channel is a flat `PlaneGeometry` quad textured with a `DataTexture` (viridis colormap).
- **Receptive field lines** use a pre-allocated `THREE.LineSegments` with vertex colors (animated path) or individual `THREE.Line` objects (static path).
- `frustumCulled = false` on `_rfLineSegs` prevents lines disappearing during camera rotation (bounding sphere is zero when buffer is pre-allocated with zeros).
- **Label positions** are computed camera-direction-aware: `margin = max(|ux|*halfW + |uz|*halfD + 30, 80)` — pushes the label just outside the layer's bounding box in the camera direction, consistent for all layer sizes.

---

## Visual Conventions

| Color | Meaning |
|-------|---------|
| Yellow `0xffdd44` | Receptive field lines, per-pixel connections (individual pixel selected or layer connections) |
| Blue `0x66aaff` | Pool layer corner lines (cluster bounding box corners) |
| Red `0xff3333` | Highlighted/active line + source dot + destination dot |

---

## Bug/Misimplementation Log

See `BUGS.md` for the full history of reported bugs and misimplementations. **Always read `BUGS.md` before implementing new features** to avoid repeating known mistakes.

### Automatic BUGS.md update rule

When the user sends a request using a numbered list or bullet points (including `-`, `*`, `•`, or any other bullet symbol), follow this process for **each item**:

1. **Classify the item.** Determine whether it is:
   - A **bug / misimplementation** — something that was implemented incorrectly, behaves contrary to intent, or was visually wrong.
   - A **new feature** — something that did not exist before.
   - A **refactor / rename** — structural change with no behavioral bug.

2. **If the item is a bug or misimplementation:**
   - Fix it as normal.
   - After fixing, **append a new entry to `BUGS.md`** (next B-N number) with:
     - `**Reported:**` session/context
     - `**Root cause:**` what the actual mistake was in the code
     - `**Fix:**` what was changed to correct it
     - `**Prevention:**` rule to avoid repeating it
   - If the fix requires multiple attempts or a follow-up request refines the same bug, **update the existing entry** rather than adding a duplicate.

3. **If the item is a new feature or refactor**, do not add a BUGS.md entry — those belong in commit messages and git history only.

This keeps `BUGS.md` current without requiring a manual "update BUGS.md" request after every session.

---

## Adding New Models

When adding a new model:

1. **Train** using the `deep-learning` conda environment:
   ```bash
   conda activate deep-learning
   python train/train_<modelname>.py
   ```

2. **Export** ONNX files for each layer checkpoint (see existing `train/` scripts for pattern).

3. **Add to `model-configs.js`**: Define `layerDefs`, `connectivity`, `modelFiles`, `totalParams`.

4. **Add test accuracy**:
   - In `index.html`: Add to the `title` attribute of the model selection button (e.g., `title="... — 99.20% test acc"`).
   - In `model-configs.js`: Add `description` field with accuracy (shown in LAYERS modal header).

5. **Add `torchinfo.summary()` stats** to the LAYERS modal. Run `torchinfo.summary(model, (1, 1, 28, 28))` and record:
   - Total params
   - Trainable params
   - Total mult-adds (M)
   - Forward/backward pass size (MB)
   - Params size (MB)
   - Estimated Total Size (MB)

   Add these to the model config under a `torchinfo` key and render them in the LAYERS modal list item for the model.

6. **Test** in the browser: draw a digit, verify inference runs, verify all layers render and connectivity visualization works.

---

## Development Notes

- No build step — ES modules served directly. Run `python -m http.server 8080` or any static server.
- `.gitignore` excludes `data/`, `train/data/`, `node_modules/`, `__pycache__/`.
- ONNX model files in `public/models/` **are** committed.
