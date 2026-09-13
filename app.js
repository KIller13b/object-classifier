import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1/dist/transformers.min.js";

const MODEL = "Xenova/clip-vit-base-patch32";
const ONNX_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0-dev.20250409-89f8206ba4/dist/ort-wasm-simd-threaded";

const statusEl = document.getElementById("status");
const statusText = document.getElementById("status-text");
const spinner = document.getElementById("spinner");
const progressWrap = document.getElementById("progress-wrap");
const progress = document.getElementById("progress");
const preview = document.getElementById("preview");
const noImg = document.getElementById("no-img");
const resultsEl = document.getElementById("results");
const classesEl = document.getElementById("classes");
const topKEl = document.getElementById("topK");
const maxClassesEl = document.getElementById("maxClasses");
const engineEl = document.getElementById("engine");
const photoBtn = document.getElementById("photo-btn");
const chooseBtn = document.getElementById("choose-btn");
const cameraInput = document.getElementById("camera-input");
const galleryInput = document.getElementById("gallery-input");

let pipe = null;
let modelReady = false;
let pendingImage = null;

const usrAgent = navigator.userAgent;
const isIOS = /iPhone|iPad|iPod/i.test(usrAgent);

let effectiveDevice = "wasm";

function resolveDevice() {
  const choice = engineEl.value;
  if (choice === "webgpu") return navigator.gpu ? "webgpu" : "wasm";
  if (choice === "wasm") return "wasm";
  return navigator.gpu ? "webgpu" : "wasm";
}

if (isIOS) {
  // iOS Safari: stable single-threaded WASM settings.
  env.backends.onnx.wasm.numThreads = 1;
  env.backends.onnx.wasm.wasmPaths = {
    mjs: `${ONNX_WASM}.mjs`,
    wasm: `${ONNX_WASM}.wasm`,
  };
}

function setStatus(msg, pct = -1) {
  statusText.textContent = msg;
  if (pct >= 0) {
    progressWrap.style.display = "block";
    progress.style.width = `${pct}%`;
  }
}

function updateClassesLimit() {
  const max = parseInt(maxClassesEl.value, 10);
  const values = classesEl.value.split(",").map((s) => s.trim()).filter(Boolean);
  if (values.length > max) {
    classesEl.value = values.slice(0, max).join(", ");
  }
}
classesEl.addEventListener("input", updateClassesLimit);
maxClassesEl.addEventListener("change", updateClassesLimit);

async function loadModel() {
  effectiveDevice = resolveDevice();
  setStatus(effectiveDevice === "wasm"
    ? "Loading model on CPU (may take a while)..."
    : "Loading model on GPU...", 0);
  try {
    pipe = await pipeline("zero-shot-image-classification", MODEL, {
      device: effectiveDevice,
      progress_callback: (p) => {
        if (p.status === "progress") {
          setStatus(
            `Downloading ${(p.file || "").split("/").pop() || "model"}...`,
            Math.round(p.progress ?? 0)
          );
        }
      },
    });
  } catch (err) {
    if (effectiveDevice === "webgpu") {
      setStatus("WebGPU failed here — falling back to WASM...", 5);
      effectiveDevice = "wasm";
      pipe = await pipeline("zero-shot-image-classification", MODEL, {
        device: "wasm",
      });
    } else {
      setStatus("Model failed to load: " + err.message);
      alert("Model failed to load: " + err.message);
      throw err;
    }
  }
  modelReady = true;
  progressWrap.style.display = "none";
  spinner.classList.remove("go");
  statusEl.classList.add("ready");
  setStatus(effectiveDevice === "wasm"
    ? "Ready & fully offline · CPU (up to ~60 s per photo)"
    : "Ready & fully offline · GPU");
  if (pendingImage) runClassify(pendingImage);
}

function classifyFromFile(file) {
  if (!file) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    const small = downscale(img) || img;
    preview.src = small.src;
    preview.style.display = "block";
    noImg.style.display = "none";
    if (modelReady) runClassify(small);
    else pendingImage = small;
  };
  img.src = url;
}

function downscale(img, maxSide = 1024) {
  const scale = Math.min(1, maxSide / Math.max(img.naturalWidth, img.naturalHeight));
  if (scale >= 1) return null;
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(img.naturalWidth * scale);
  canvas.height = Math.round(img.naturalHeight * scale);
  canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
  const out = new Image();
  out.src = canvas.toDataURL("image/jpeg", 0.9);
  return out;
}

async function runClassify(img) {
  const classes = classesEl.value.split(",").map((s) => s.trim()).filter(Boolean);
  if (classes.length === 0) {
    setStatus("Add at least one class name in the box above");
    return;
  }
  const topK = parseInt(topKEl.value, 10);
  setStatus(effectiveDevice === "wasm"
    ? "Classifying on CPU — this can take up to a minute, please wait..."
    : "Classifying... (GPU)");
  await img.decode();
  const start = performance.now();
  const out = await pipe(img, classes, { topk: topK });
  const ms = Math.round(performance.now() - start);
  renderResults(out, ms);
  setStatus(`Done in ${ms} ms · ${effectiveDevice}`);
  spinner.classList.remove("go");
}

function renderResults(out, ms) {
  resultsEl.innerHTML = "";
  const head = document.createElement("div");
  head.className = "result-head";
  head.innerHTML = `<span>Top ${out.length}</span><span>${ms} ms</span>`;
  resultsEl.appendChild(head);
  out.forEach((r) => {
    const pct = (r.score * 100).toFixed(1);
    const row = document.createElement("div");
    row.className = "pred";
    row.innerHTML = `
      <span class="label">${escapeHtml(r.label)}</span>
      <div class="bar"><div class="bar-fill" style="width:${pct}%"></div></div>
      <span class="conf">${pct}%</span>`;
    resultsEl.appendChild(row);
  });
}

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

photoBtn.addEventListener("click", () => cameraInput.click());
chooseBtn.addEventListener("click", () => galleryInput.click());
cameraInput.addEventListener("change", (e) => classifyFromFile(e.target.files[0]));
galleryInput.addEventListener("change", (e) => classifyFromFile(e.target.files[0]));

engineEl.addEventListener("change", async () => {
  if (!modelReady) return;
  try { pipe?.dispose?.(); } catch {}
  pipe = null;
  modelReady = false;
  statusEl.classList.remove("ready");
  spinner.classList.add("go");
  await loadModel();
});

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => navigator.serviceWorker.register("./sw.js"));
}

loadModel();