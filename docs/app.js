// The web page: loads the two ONNX models, takes a photo, shows boxes and bins.
// All the model logic lives in pipeline.js. `ort` (ONNX Runtime Web) comes from the <script> tag in index.html.
import { createSorter, squareCrop } from "./pipeline.js";

const MAX_SIDE = 1280;   // shrink huge phone photos first, same as app.py
const $ = (id) => document.getElementById(id);
const stage = $("stage"), canvas = $("result"), video = $("video"), empty = $("empty");
const statusEl = $("status"), fileInput = $("file"), conf = $("conf"), confOut = $("confOut");

let meta, sorter, photo = null, running = false, rerun = false, stream = null;

function say(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

// ------------------------------------------------------------------ loading the models
async function download(urls, onProgress) {
  const responses = await Promise.all(urls.map((u) => fetch(u)));
  if (responses.some((r) => !r.ok)) throw new Error("download failed");
  const total = responses.reduce((s, r) => s + Number(r.headers.get("content-length") || 0), 0);
  let loaded = 0;
  return Promise.all(responses.map(async (r) => {
    const reader = r.body.getReader(), chunks = [];
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value); loaded += value.length;
      onProgress(loaded, total);
    }
    return new Uint8Array(await new Blob(chunks).arrayBuffer());
  }));
}

async function init() {
  buildBins(await (await fetch("models/meta.json")).json());
  conf.value = meta.default_conf;
  confOut.textContent = meta.default_conf.toFixed(2);
  const mb = (n) => (n / 1e6).toFixed(0);
  const [det, cls] = await download(["models/detector.onnx", "models/classifier.onnx"], (loaded, total) => {
    say(`Loading the two networks… ${mb(loaded)}${total ? " of " + mb(total) : ""} MB`);
    if (total) $("bar").style.width = (100 * loaded / total) + "%";
  });
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  const options = { executionProviders: ["wasm"] };
  sorter = createSorter(ort, await ort.InferenceSession.create(det, options), await ort.InferenceSession.create(cls, options), meta);
  $("progress").classList.add("done");
  fileInput.disabled = false;
  $("camera").disabled = !navigator.mediaDevices?.getUserMedia;
  say("Ready. Choose a photo of some waste.");
}

// ------------------------------------------------------------------ bins
function buildBins(m) {
  meta = m;
  $("bins").innerHTML = meta.classes.map((name, i) => `
    <article class="bin" style="--c: rgb(${meta.colors[i].join(",")})">
      <div class="bin-lid"><span class="bin-name">${meta.nice_names[i]}</span><span class="bin-count" id="count-${i}">0</span></div>
      <div class="bin-body" id="bin-${i}"><p class="bin-empty">Nothing sorted here yet.</p></div>
    </article>`).join("");
}

function fillBins(objects) {
  meta.classes.forEach((_, i) => {
    const mine = objects.filter((o) => o.class_id === i);
    const body = $("bin-" + i);
    $("count-" + i).textContent = mine.length;
    body.replaceChildren();
    if (!mine.length) {
      body.innerHTML = `<p class="bin-empty">${photo ? "None in this photo." : "Nothing sorted here yet."}</p>`;
    }
    mine.forEach((o, k) => {
      const size = 160, rgb = squareCrop(photo, o.box, 0.06, size);
      const c = document.createElement("canvas");
      c.width = c.height = size;
      c.title = `${meta.nice_names[i]}, ${Math.round(o.confidence * 100)}% sure`;
      c.style.animationDelay = (k * 60) + "ms";
      const px = new ImageData(size, size);
      for (let p = 0; p < size * size; p++) {
        px.data[p * 4] = rgb[p * 3]; px.data[p * 4 + 1] = rgb[p * 3 + 1]; px.data[p * 4 + 2] = rgb[p * 3 + 2]; px.data[p * 4 + 3] = 255;
      }
      c.getContext("2d").putImageData(px, 0, 0);
      body.append(c);
    });
  });
}

// ------------------------------------------------------------------ drawing the boxes
function draw(objects) {
  const ctx = canvas.getContext("2d");
  ctx.putImageData(photo, 0, 0);
  const big = Math.max(photo.width, photo.height);
  const t = Math.max(2, Math.round(big / 320)), fs = Math.max(13, Math.round(big / 52));
  ctx.font = `600 ${fs}px "Instrument Sans", system-ui, sans-serif`;
  ctx.textBaseline = "middle";
  ctx.lineWidth = t;
  for (const o of objects) {
    const [x1, y1, x2, y2] = o.box, col = `rgb(${meta.colors[o.class_id].join(",")})`;
    ctx.strokeStyle = ctx.fillStyle = col;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const text = `${meta.nice_names[o.class_id]} ${Math.round(o.confidence * 100)}%`;
    const tw = ctx.measureText(text).width + fs * 0.7, th = fs * 1.5;
    const tx = Math.max(0, Math.min(x1 - t / 2, photo.width - tw));
    const ty = y1 - th > 0 ? y1 - th : y1;          // label above the box, or inside it at the top edge
    ctx.fillRect(tx, ty, tw, th);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, tx + fs * 0.35, ty + th / 2);
  }
}

// ------------------------------------------------------------------ running the pipeline
const nextFrame = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));

async function sort() {
  if (!photo || !sorter) return;
  if (running) { rerun = true; return; }       // slider moved while busy: run once more afterwards
  running = true;
  stage.classList.add("busy");
  say("Sorting…");
  await nextFrame();                            // let the browser paint "Sorting…" before the heavy work
  try {
    const t0 = performance.now();
    const { objects } = await sorter.predict(photo, Number(conf.value));
    const ms = Math.round(performance.now() - t0), n = objects.length;
    draw(objects);
    fillBins(objects);
    say(n ? `Found ${n} item${n === 1 ? "" : "s"} in ${ms} ms, on this device.`
          : "No items found. Lower the detector confidence, or try a photo with the items on a plain background.");
  } catch (err) {
    console.error(err);
    say("The networks could not process this photo. Reload the page and try again.", true);
  }
  stage.classList.remove("busy");
  running = false;
  if (rerun) { rerun = false; sort(); }
}

// Anything drawable (file, video frame) -> RGBA pixels, at most MAX_SIDE on the long side.
function setPhoto(source, w, h) {
  const k = Math.min(1, MAX_SIDE / Math.max(w, h));
  canvas.width = Math.max(1, Math.round(w * k));
  canvas.height = Math.max(1, Math.round(h * k));
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
  photo = ctx.getImageData(0, 0, canvas.width, canvas.height);
  canvas.hidden = false; empty.hidden = true;
  stage.classList.add("has-photo");
  sort();
}

async function openFile(file) {
  if (!file || !sorter) return;
  stopCamera();
  try {
    const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
    setPhoto(bmp, bmp.width, bmp.height);
    bmp.close();
  } catch {
    say("This browser cannot open that file as a photo. Try a JPG or PNG.", true);
  }
}

// ------------------------------------------------------------------ camera
function showCameraButtons(on) {
  $("shoot").hidden = $("cancelCamera").hidden = !on;
  $("camera").hidden = $("chooseLabel").hidden = on;
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 } } });
  } catch {
    say("The camera is blocked or missing. Allow camera access for this page, or choose a photo instead.", true);
    return;
  }
  video.srcObject = stream;
  await video.play();
  video.hidden = false; canvas.hidden = true; empty.hidden = true;
  stage.classList.add("has-photo");
  showCameraButtons(true);
  say("Point the camera at the waste, then take the photo.");
}

function stopCamera() {
  if (!stream) return;
  stream.getTracks().forEach((t) => t.stop());
  stream = null;
  video.hidden = true;
  showCameraButtons(false);
  if (photo) canvas.hidden = false;
  else { empty.hidden = false; stage.classList.remove("has-photo"); say("Ready. Choose a photo of some waste."); }
}

// ------------------------------------------------------------------ wiring
fileInput.addEventListener("change", () => { openFile(fileInput.files[0]); fileInput.value = ""; });
$("camera").addEventListener("click", startCamera);
$("cancelCamera").addEventListener("click", stopCamera);
$("shoot").addEventListener("click", () => {
  const w = video.videoWidth, h = video.videoHeight;
  setPhoto(video, w, h);
  stopCamera();
});
conf.addEventListener("input", () => { confOut.textContent = Number(conf.value).toFixed(2); });
conf.addEventListener("change", sort);
addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) openFile(item.getAsFile());
});
for (const type of ["dragenter", "dragover"]) stage.addEventListener(type, (e) => { e.preventDefault(); stage.classList.add("over"); });
for (const type of ["dragleave", "drop"]) stage.addEventListener(type, (e) => { e.preventDefault(); stage.classList.remove("over"); });
stage.addEventListener("drop", (e) => openFile(e.dataTransfer.files[0]));

init().catch((err) => {
  console.error(err);
  say("The networks did not load. Check your connection and reload the page.", true);
});
