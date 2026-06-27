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
