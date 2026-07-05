# BUGS.md — Reported Bugs & Misimplementations

This file tracks all bugs and deviations from intent reported by the user.
**Claude should read this before implementing features** to avoid repeating known issues.
Claude should **append new entries** whenever the user reports a bug or misimplementation.

---

## Bug / Misimplementation History

### B-01: Animation — Lines become invisible during camera rotation/zoom
**Reported:** Session 1
**Root cause:** `THREE.LineSegments` with a pre-allocated, zero-filled `Float32Array` has a bounding sphere at the origin with radius 0. Three.js frustum-culls the object when that sphere is outside the camera frustum, even though the actually-drawn vertices are in view.
**Fix:** `this._rfLineSegs.frustumCulled = false;`
**Prevention:** Always set `frustumCulled = false` on any `LineSegments` whose geometry is pre-allocated and progressively filled.

---

### B-02: Layer name label — last layer moves too extreme on camera drag
**Reported:** Session 1
**Root cause:** Old margin formula `Math.min(Math.max(totalW, totalD) * 0.5 + 20, 300)` gave `margin=70` for FC1 (narrow strip) and `margin=300` for the Output layer (10 large cells, `totalW≈680`), making the output label orbit wildly compared to other layers.
**Fix:** Camera-direction-aware formula: `margin = Math.max(Math.abs(ux) * halfW + Math.abs(uz) * halfD + 30, 80)`. This places the label exactly 30 units outside the layer's bounding box in the camera direction, consistently for all layers.
**Prevention:** When positioning labels relative to layer size, account for camera direction so large layers don't produce outsized label offsets.

---

### B-03: Progress bar inside scrollable `#detail-content` div
**Reported:** Session 1
**Root cause:** `.dp-progress` was placed inside `#detail-content` which is `overflow-y: auto`, causing the progress bar to scroll away during animation.
**Fix:** Moved `.dp-progress` outside `#detail-content` into a sibling `#detail-progress` div below it, outside the scroll region.
**Prevention:** Progress/status indicators that must stay visible during scrollable content animation should live *outside* the scrollable container.

---

### B-04: Source pixel dot not turning red on line highlight
**Reported:** Session 1
**Root cause:** `highlightRFLine()` only changed line colors; the source dot mesh material was not updated.
**Fix:** Added `if (this._rfSrcDot) this._rfSrcDot.material.color.setHex(index >= 0 ? 0xff3333 : 0xffdd44);` in both the animated and static paths of `highlightRFLine`.
**Prevention:** When highlighting a line, always update *all* associated visual elements: line color, source dot, destination dot.

---

### B-05: Target/destination pixel dot not turning red on line highlight
**Reported:** Session 2
**Root cause:** Only the source dot was tracked (`_rfSrcDot`). Destination dots were added to the scene group but not stored for later color manipulation.
**Fix:** Added `_rfDstDots[]` array. Populated in `showReceptiveField` loop and `addRFLine` tick callback (when `t===1`). In `highlightRFLine`, reset all dst dots to yellow then set `_rfDstDots[index]` to red.
**Prevention:** Track *all* interactive visual objects (both ends of a line) by reference at creation time.

---

### B-06: Pool layer visualization — drawing per-channel center-pixel connections
**Reported:** Session 1 (original pooling visualization)
**Original intent:** Pooling reduces spatial resolution without changing channel content — the visualization should convey "same spatial region, fewer pixels."
**Wrong approach:** Drew lines from center pixel of each pool-layer channel to center pixel of prev-layer channel (same as conv layers).
**Fix (session 1):** Drew 4 blue lines from the pool cluster's 4 bounding box corner vertices to the prev-layer cluster's 4 bounding box corners.
**Fix (session 2):** Added per-pixel yellow lines from every pixel in the pool layer to its 2×2 source region in the prev layer (on top of the 4 blue corner lines).
**Prevention:** Pool layers spatially reduce; corner-to-corner lines convey this. Per-pixel lines show exact mapping. Never use center-pixel-per-channel lines for pool visualization.

---

### B-07: Layer connection lines — using blue for all connection types
**Reported:** Session 2
**Root cause:** `showLayerConnections` used blue (`0x66aaff`) for all lines (conv, FC), inconsistent with individual pixel selection which uses yellow (`0xffdd44`).
**Fix:** Changed `lineMat` for conv/FC branch to yellow `0xffdd44`. Only pool corner lines remain blue `0x66aaff`.
**Prevention:** Layer connection lines (not corner-structural lines) should use the same color as individual-pixel RF lines (yellow). Blue is reserved for structural/bounding-box lines (pool corners).

---

### B-08: `#dp-terms-list` not auto-scrolling during animation
**Reported:** Session 2
**Root cause:** `renderDetailFormula` replaces `detailContent.innerHTML` each tick but did not scroll the terms list to the bottom afterward.
**Fix:** After `detailContent.innerHTML = html`, added: `const termsList = detailContent.querySelector('.dp-terms-list'); if (termsList) termsList.scrollTop = termsList.scrollHeight;`
**Prevention:** When a scrollable list is rebuilt from scratch on each animation tick, always explicitly scroll it to the desired position after rebuilding.

---

### B-09: RF line highlight broken after batching afterData into LineSegments
**Reported:** Session 3 (after adding weight visualization)
**Root cause:** `afterData` lines (weight→bias, bias→output) were built with `_buildLineSegments()` into a single `LineSegments` object. `_rfLines` was left empty because individual `THREE.Line` objects were never pushed. `highlightRFLine(index)` then read from an empty array and did nothing.
**Fix:** `afterData` lines are always built as individual `new THREE.Line(...)` objects and pushed to `_rfLines[]`, one per line. Only `beforeData` (input→weight) uses the batched `_buildLineSegments` path.
**Prevention:** Anything that needs per-line highlighting at runtime must be stored as a separate object reference. Batched `LineSegments` are opaque — individual line color/opacity cannot be changed after construction.

---

### B-10: Bias pixel rendered as sphere instead of flat pixel quad
**Reported:** Session 3
**Root cause:** Bias was visualized with `THREE.SphereGeometry`, inconsistent with how channel pixels and conv kernels are rendered (flat `PlaneGeometry` + `DataTexture`).
**Fix:** Replaced with `_buildBiasPixel()` method: a `PlaneGeometry` quad with a `DataTexture` that maps the bias value through the viridis colormap, same pipeline as all other pixel quads.
**Prevention:** All scalar/tensor values in this visualization are shown as flat `PlaneGeometry` + `DataTexture` quads. Never use 3D geometry (sphere, box) for data pixels.

---

### B-11: Bias pixel on last FC layer inflated to OUTPUT_PIXEL_SIZE
**Reported:** Session 3
**Root cause:** The bias pixel `pxSz` was computed as `(li === this._layerDefs.length - 1) ? OUTPUT_PIXEL_SIZE : PIXEL_SIZE`. `OUTPUT_PIXEL_SIZE = 64` is reserved for the large output label cells of the final layer. The bias pixel above those cells was also inheriting this size, making it 64 world-units — nearly as large as the output cell itself.
**Fix:** Bias pixels always use `PIXEL_SIZE` regardless of layer. Only the output layer's actual channel quad uses `OUTPUT_PIXEL_SIZE`.
**Prevention:** `OUTPUT_PIXEL_SIZE` applies only to the layer's primary channel quads in `_renderLayer`. Auxiliary elements (bias pixels, weight images) always use `PIXEL_SIZE`.

---

### B-12: Opacity of bias→output line hardcoded instead of data-driven
**Reported:** Session 3
**Root cause:** All three line segments (input→weight, weight→bias, bias→output) were hardcoded to `opacity: 0.9`. The intent was: input value drives input→weight opacity; input×weight drives weight→bias opacity; output activation drives bias→output opacity.
**Fix:** Computed `biasToOutOpac = Math.max(0.05, outAbs[outIdx] / outMax) * 0.85` from the layer's raw activation data and used it for the bias→output line's opacity. The input→weight and weight→bias opacities were already factored through `_computeLineOpacities`.
**Prevention:** Every line segment in the RF path should reflect the magnitude of the value flowing through it at that point. Hardcoded opacity ignores the actual data.

---

### B-13: FC1 weight/bias visualization unavailable (is1D check failed)
**Reported:** Session 3 (implicit — clicking FC1 output pixels showed no weight matrix)
**Root cause:** The `is1D` check (`prevDef.channels === 1 && prevDef.h === 1 && inF === prevDef.w`) was designed to detect a 1D-vector input and enable the FC weight matrix path. FC1's previous layer was MaxPool2 (`channels=64, h=7, w=7`), which is 3D — so `is1D` was always false for FC1, and the weight matrix was never shown.
**Fix:** Inserted a `Flatten` layer (`channels=1, h=1, w=3136, dataKey='layer3'`) between MaxPool2 and FC1 in both v1 and v2 configs. Flatten's 1:1 pixel mapping was added to `_getContributingPixels` and `showLayerConnections`. Now `is1D` is true for FC1 and the full weight matrix + bias pixel path activates.
**Prevention:** FC weight matrix visualization requires the previous layer to be a flat 1D vector. If the previous layer is spatial, a Flatten layer must exist in the config to bridge them.

---

### B-14: showLayerConnections for Conv — channel sampling skipped every other channel
**Reported:** Session 3
**Root cause:** `srcStep = Math.max(1, Math.ceil(sources.length / 16))` was applied to all layer types. For a 32-channel conv layer `srcStep = 2`, so only 16 of 32 channels were shown. Some output channels had zero lines drawn.
**Fix:** `srcStep` and `dstStep` throttling is now conditional on `isFc`. Conv layers always use `srcStep = 1` (all channels, one center pixel each) and `dstStep = 1` (all contributing pixels per center pixel). Only FC layers apply the 16/64 throttle.
**Prevention:** Sampling limits for FC layers (which have thousands of neurons) must not be blindly applied to conv layers (which have at most 64 channels and 9 contributing pixels each). Branch the logic on `isFc`.

---

### B-15: Line opacity encoded as color darkening (yellow→black) instead of transparency
**Reported:** Session 3
**Root cause:** `buildLineSegments` and `_buildLineSegments` encoded `opacity` by multiplying the yellow RGB values: `colArr = [opacity, 0.867*opacity, 0.267*opacity]`. Low-opacity lines appeared as dark/black lines rather than faint transparent yellow lines. Visually indistinguishable from invisible.
**Fix:** Both helpers now bucket lines into 8 opacity tiers and create a separate `LineBasicMaterial` per tier with `transparent: true, opacity: alpha, depthWrite: false`. All lines render as yellow with varying alpha.
**Prevention:** `THREE.LineBasicMaterial` with `vertexColors` can only tint by multiplying RGB. For true transparency, use `material.opacity` with `transparent: true`. Since one `LineSegments` shares one material, per-line alpha requires either separate objects or tier-bucketed materials.

---

### B-16: Layer name labels clipped for long names
**Reported:** Session 3
**Root cause:** `_addLayerLabel` created a canvas with fixed `width = 1024`. At `104px monospace`, names like `"Conv1 + BatchNorm + ReLU"` (~1550px rendered) overflowed the canvas and were clipped at both ends.
**Fix:** Text width is now measured with `ctx.measureText(name).width` before finalizing `canvas.width = Math.max(512, Math.ceil(textW) + 80)`. Sprite scale uses a fixed world-unit height (`spriteH = 22`) and derives width proportionally from the canvas aspect ratio, so the rendered label maintains consistent text size regardless of name length.
**Prevention:** Never use a fixed canvas width for text sprites. Always measure the text first and size the canvas to fit. Compute sprite scale from a fixed world height, not a fixed world width, to avoid distortion.

---

### B-17: Grid lines — high division count produced near-invisible dark lines
**Reported:** Session 3
**Root cause:** `GridHelper(6000, 120, 0x222244, 0x1a1a33)` — the grid color `0x1a1a33` is nearly identical to the scene background `0x1a1a2e`. At 120 divisions the grid appeared as a dark haze with no discernible individual lines. The user intended visible bright blue structural lines.
**Fix:** `GridHelper(6000, 20, 0x334488, 0x334488)` — both center-line and grid-line colors set to the same visible blue; divisions reduced to 20 so each line is clearly distinct.
**Prevention:** Grid colors must contrast with the scene background. Both `colorCenterLine` and `colorGrid` must be set intentionally; defaulting one to near-background color defeats the purpose. Use divisions ≤ 20 for legible individual lines.

---

### B-18: Learn page dropdown collapse — instant jump then shrink
**Reported:** Session 4 (Learn Center)
**Root cause:** `.task-card-body` and `.idea-card-body` in `style.css` only declared `transition: max-height ...`, while `padding` (and `border-top-color` for task cards) changed instantly with no transition. On collapse, padding/border snapped to their closed values immediately, then max-height animated separately afterward — visually a jump followed by a shrink instead of one continuous motion.
**Fix:** Added `padding` and `border-color` to the same `transition` declaration as `max-height` on both `.task-card-body` and `.idea-card-body`, so all three animate in sync.
**Prevention:** When an expand/collapse effect changes multiple CSS properties (max-height, padding, border, etc.), all of them must share one `transition` list. Animating only one property while others change instantly always produces a visible jump.

---

### B-19: Conv widget — Pause button stops working after animation plays to completion
**Reported:** Session 4 (Learn Center)
**Root cause:** In `initConvWidget`'s `stepAnim` (`learn.js`), the branch that runs when the sliding-kernel animation reaches its last position set `playing = false` and reset the button text, but never called `clearInterval(animTimer)`. The `setInterval` timer kept firing forever in the background. The next "Play" click saw `playing === false` and started a *second* `setInterval`, overwriting the `animTimer` reference — so "Pause" could only ever cancel the newest interval, leaving the original one running uncontrollably and making the animation look stuck/unpausable.
**Fix:** Added `clearInterval(animTimer)` to the completion branch of `stepAnim` so the timer is always torn down when the animation finishes naturally, matching what the manual pause/reset handlers already did.
**Prevention:** Any code path that sets a "not running" flag (`playing = false`) must also clear the corresponding timer in the same place — never let a flag and its timer's lifetime drift apart. Whenever `setInterval` is reassigned to a variable, everything that changes that variable's target must clear the previous one first.

### B-20: Conv widget — dataset image status reverts to synthetic pattern text on language switch
**Reported:** Session 4 (Learn Center)
**Root cause:** `initConvWidget`'s `onLanguageChange` callback and the initial status line were hardcoded to `tf('conv.statusPattern', { name: t('sec3.patternCheckerboard') })` regardless of what was actually loaded. After loading a random dataset image, the status text was correct until the user switched languages (or the callback ran for any reason) — at which point it snapped back to "Pattern: Checkerboard" even though the input canvas still showed the real dataset image, making it look like the dataset image "wasn't applied."
**Fix:** Added a `currentSource` state object (`{type: 'pattern'|'dataset', ...}`) updated by every input-changing action (pixel edit, pattern dropdown, dataset load), and a single `updateStatusText()` that renders the correct status from `currentSource`. `onLanguageChange` now calls `updateStatusText()` instead of a hardcoded string. Also added a missing `sec3.patternCustom` i18n key for the hand-edited-pixel case.
**Prevention:** Never hardcode a UI-refresh callback (like a language-change listener) to a fixed default string — derive it from actual current state, or any refresh triggered by an unrelated event will silently overwrite genuine state with a stale default.
