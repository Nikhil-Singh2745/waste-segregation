// The same pipeline as wastesort/infer.py, written in JavaScript so it can run in a browser:
//   photo -> letterbox -> DETECTOR -> boxes -> square crop of each box -> CLASSIFIER -> label per box
// Images are plain {data, width, height} objects with RGBA bytes (what a <canvas> gives you).
// `ort` is onnxruntime (web in the browser, node in the test), passed in so this file has no dependencies.

const sigmoid = (v) => 1 / (1 + Math.exp(-v));

// Python's round(): halves go to the even number.
export function pyRound(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

// For every output pixel: which input pixels it is made of, and how much of each.
// Shrinking averages the covered area (like cv2.INTER_AREA), enlarging blends the two neighbours.
function axisWeights(src, dst) {
  const taps = [];
  const scale = src / dst;
  for (let i = 0; i < dst; i++) {
    const t = [];
    if (dst < src) {
      const start = i * scale, end = Math.min(src, (i + 1) * scale);
      for (let j = Math.floor(start); j < end; j++) {
        const w = (Math.min(j + 1, end) - Math.max(j, start)) / scale;
        if (w > 1e-9) t.push(j, w);
      }
    } else {
      const f = (i + 0.5) * scale - 0.5;
      const j = Math.floor(f), frac = f - j;
      t.push(Math.min(src - 1, Math.max(0, j)), 1 - frac, Math.min(src - 1, Math.max(0, j + 1)), frac);
    }
    taps.push(t);
  }
  return taps;
}

// RGBA image -> RGB bytes of size dw x dh.
export function resize(img, dw, dh) {
  const { data, width: sw, height: sh } = img;
  const wx = axisWeights(sw, dw), wy = axisWeights(sh, dh);
  const tmp = new Float32Array(sh * dw * 3);
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < dw; x++) {
      const t = wx[x];
      let r = 0, g = 0, b = 0;
      for (let k = 0; k < t.length; k += 2) {
        const p = (y * sw + t[k]) * 4, w = t[k + 1];
        r += data[p] * w; g += data[p + 1] * w; b += data[p + 2] * w;
      }
      const o = (y * dw + x) * 3;
      tmp[o] = r; tmp[o + 1] = g; tmp[o + 2] = b;
    }
  }
  const out = new Uint8ClampedArray(dw * dh * 3);
  for (let y = 0; y < dh; y++) {
    const t = wy[y];
    for (let x = 0; x < dw; x++) {
      let r = 0, g = 0, b = 0;
      for (let k = 0; k < t.length; k += 2) {
        const p = (t[k] * dw + x) * 3, w = t[k + 1];
        r += tmp[p] * w; g += tmp[p + 1] * w; b += tmp[p + 2] * w;
      }
      const o = (y * dw + x) * 3;
      out[o] = r; out[o + 1] = g; out[o + 2] = b;
    }
  }
  return out;
}

// Cut a square around a box, repeating the edge pixels where the square leaves the photo
// (prepare_data.square_crop). Returns RGB bytes of size x size.
export function squareCrop(img, box, pad, size) {
  const { data, width: w, height: h } = img;
  const [x1, y1, x2, y2] = box;
  const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
  const side = Math.max(Math.max(x2 - x1, y2 - y1) * (1 + 2 * pad), 8);
  const sx1 = pyRound(cx - side / 2), sy1 = pyRound(cy - side / 2);
  const cw = pyRound(cx + side / 2) - sx1, ch = pyRound(cy + side / 2) - sy1;
  const crop = new Uint8ClampedArray(cw * ch * 4);
  for (let y = 0; y < ch; y++) {
    const sy = Math.min(h - 1, Math.max(0, sy1 + y));
    for (let x = 0; x < cw; x++) {
      const sx = Math.min(w - 1, Math.max(0, sx1 + x));
      const s = (sy * w + sx) * 4, o = (y * cw + x) * 4;
      crop[o] = data[s]; crop[o + 1] = data[s + 1]; crop[o + 2] = data[s + 2]; crop[o + 3] = 255;
    }
  }
  return resize({ data: crop, width: cw, height: ch }, size, size);
}

// RGB bytes -> normalised float values in channel-first order, written into `out` at `offset`.
function writeTensor(rgb, n, out, offset, mean, std) {
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < n; i++) out[offset + c * n + i] = (rgb[i * 3 + c] / 255 - mean[c]) / std[c];
  }
}

function iou(a, b) {
  const iw = Math.min(a[2], b[2]) - Math.max(a[0], b[0]), ih = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  const inter = Math.max(0, iw) * Math.max(0, ih);
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter);
}

export function createSorter(ort, detector, classifier, meta) {
  const S = meta.det_size, stride = meta.det_stride, G = S / stride, C = meta.cls_size;
  const nClasses = meta.classes.length;

  // Returns boxes [x1, y1, x2, y2, score] in pixels of the original photo.
  async function detect(img, conf, topk = 60, iouLimit = 0.5) {
    const { width: w, height: h } = img;
    const r = S / Math.max(h, w);
    const nw = Math.max(1, pyRound(w * r)), nh = Math.max(1, pyRound(h * r));
    const ox = Math.floor((S - nw) / 2), oy = Math.floor((S - nh) / 2);
    const boxed = new Uint8ClampedArray(S * S * 3).fill(114);      // gray padding
    const small = resize(img, nw, nh);
    for (let y = 0; y < nh; y++) boxed.set(small.subarray(y * nw * 3, (y + 1) * nw * 3), ((oy + y) * S + ox) * 3);
    const input = new Float32Array(3 * S * S);
    writeTensor(boxed, S * S, input, 0, meta.mean, meta.std);
    const out = (await detector.run({ input: new ort.Tensor("float32", input, [1, 3, S, S]) })).output.data;

    // Channel 0 is the heat-map. Keep cells that are a local maximum and above the threshold.
    const n = G * G;
    const heat = new Float32Array(n);
    for (let i = 0; i < n; i++) heat[i] = sigmoid(out[i]);
    let found = [];
    for (let y = 0; y < G; y++) {
      for (let x = 0; x < G; x++) {
        const i = y * G + x, s = heat[i];
        if (s < conf) continue;
        let peak = true;
        for (let dy = -1; dy <= 1 && peak; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const yy = y + dy, xx = x + dx;
            if (yy >= 0 && yy < G && xx >= 0 && xx < G && heat[yy * G + xx] > s) { peak = false; break; }
          }
        }
        if (!peak) continue;
        const bx = (x + sigmoid(out[n + i])) * stride, by = (y + sigmoid(out[2 * n + i])) * stride;
        const bw = Math.exp(Math.min(6, Math.max(-4, out[3 * n + i]))) * stride;
        const bh = Math.exp(Math.min(6, Math.max(-4, out[4 * n + i]))) * stride;
        found.push([bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2, s]);
      }
    }
    found.sort((a, b) => b[4] - a[4]);
    found = found.slice(0, topk);
    const kept = [];                                               // non-maximum suppression
    for (const b of found) if (!kept.some((k) => iou(k, b) > iouLimit)) kept.push(b);

    const clip = (v, hi) => Math.min(hi, Math.max(0, v));
    return kept
      .map(([x1, y1, x2, y2, s]) => [clip((x1 - ox) / r, w), clip((y1 - oy) / r, h), clip((x2 - ox) / r, w), clip((y2 - oy) / r, h), s])
      .filter((b) => b[2] - b[0] > 4 && b[3] - b[1] > 4);
  }

  // Returns the class probabilities of each box. The 4-rotation average is inside the ONNX file.
  async function classify(img, boxes, batch = 8) {
    const probs = [];
    for (let start = 0; start < boxes.length; start += batch) {
      const part = boxes.slice(start, start + batch);
      const input = new Float32Array(part.length * 3 * C * C);
      part.forEach((b, i) => writeTensor(squareCrop(img, b, 0.12, C), C * C, input, i * 3 * C * C, meta.mean, meta.std));
      const out = (await classifier.run({ input: new ort.Tensor("float32", input, [part.length, 3, C, C]) })).output.data;
      for (let i = 0; i < part.length; i++) probs.push(Array.from(out.slice(i * nClasses, (i + 1) * nClasses)));
    }
    return probs;
  }

  async function predict(img, conf = meta.default_conf) {
    const boxes = await detect(img, conf);
    const probs = await classify(img, boxes);
    const objects = boxes.map((b, i) => {
      const c = probs[i].indexOf(Math.max(...probs[i]));
      return { box: b.slice(0, 4).map(pyRound), detector_score: b[4], label: meta.classes[c], class_id: c, confidence: probs[i][c] };
    });
    objects.sort((a, b) => a.class_id - b.class_id || a.box[0] - b.box[0]);
    const counts = Object.fromEntries(meta.classes.map((name, i) => [name, objects.filter((o) => o.class_id === i).length]));
    return { objects, counts };
  }

  return { detect, classify, predict };
}
