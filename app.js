import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1/dist/transformers.min.js";

const MODEL = "Xenova/clip-vit-base-patch32";

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
const photoBtn = document.getElementById("photo-btn");
const chooseBtn = document.getElementById("choose-btn");
const cameraInput = document.getElementById("camera-input");
const galleryInput = document.getElementById("gallery-input");

let pipe = null;
let modelReady = false;
let pendingImage = null;

if (/iPhone|iPad|iPod/i.test(navigator.userAgent)) {
  env.backends.onnx.wasm.numThreads = 1;
}

const device = "gpu" in navigator ? "webgpu" : "wasm";

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
  setStatus("Downloading model (~200 MB, first time only)", 0);
  try {
    pipe = await pipeline("zero-shot-image-classification", MODEL, {
      device,
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
    if (device === "webgpu") {
      setStatus("WebGPU unavailable here, falling back to CPU...", 5);
      pipe = await pipeline("zero-shot-image-classification", MODEL, {
        device: "wasm",
      });
    } else {
      alert("Model failed to load: " + err.message);
      throw err;
    }
  }
  modelReady = true;
  progressWrap.style.display = "none";
  spinner.classList.remove("go");
  statusEl.classList.add("ready");
  setStatus("Ready & fully offline");
  if (pendingImage) runClassify(pendingImage);
}

function classifyFromFile(file) {
  if (!file) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    preview.src = url;
    preview.style.display = "block";
    noImg.style.display = "none";
    if (modelReady) runClassify(img);
    else pendingImage = img;
  };
  img.src = url;
}

async function runClassify(img) {
  const classes = classesEl.value.split(",").map((s) => s.trim()).filter(Boolean);
  if (classes.length === 0) {
    setStatus("Add at least one class name in the box above");
    return;
  }
  const topK = parseInt(topKEl.value, 10);
  setStatus(`Classifying... (${device})`);
  // For images captured as blobs, ensure the browser has decoded it
  if (img.complete === false || img.naturalWidth === 0) await img.decode();
  const start = performance.now();
  if (!img.complete) return;
  const out = await pipe(img, classes, { topk: topK });
  const ms = Math.round(performance.now() - start);
  renderResults(out, ms);
  setStatus(`Done in ${ms} ms · ${device}`);
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

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => navigator.serviceWorker.register("./sw.js"));
}

loadModel();