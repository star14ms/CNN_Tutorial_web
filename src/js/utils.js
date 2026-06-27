export function grayscaleColor(value) {
  const v = Math.min(1, Math.max(0, value));
  return { r: v, g: v, b: v };
}

export function normalizeActivations(data) {
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    if (data[i] < min) min = data[i];
    if (data[i] > max) max = data[i];
  }
  const range = max - min || 1;
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = (data[i] - min) / range;
  return { normalized: out, min, max };
}

export function debounce(fn, ms) {
  let timer;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms); };
}

export function gridLayout(n) {
  // Find exact factor pair (rows, cols) with rows*cols = n that minimises rows+cols (closest to square)
  let bestRows = 1, bestCols = n;
  for (let r = 1; r * r <= n; r++) {
    if (n % r === 0) {
      const c = n / r;
      if (r + c < bestRows + bestCols) { bestRows = r; bestCols = c; }
    }
  }
  return { rows: bestRows, cols: bestCols };
}
