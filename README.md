# Object Classifier — Fully Offline CLIP for iPhone

Zero-shot object classification that runs **100% on your device** — no server, no API keys, no training data, no Apple Developer account, no Mac required. Point your camera at anything and it tells you what it is in plain English.

🌐 **Live app:** https://killer13b.github.io/object-classifier/

---

## What it does

- **Take a photo or pick one from your library** and get instant labels with confidence scores.
- **Zero-shot:** type the objects you expect (`cat, drone, fire extinguisher, backpack…`) and it recognizes them — no training needed, any class list works.
- **100% offline** after the first model download. Photos never leave your phone.
- **No Mac, no Xcode, no Apple Developer account.** It's an installable web app (PWA) that runs CLIP in the phone's browser.

## Quick start (iPhone)

1. Open **https://killer13b.github.io/object-classifier/** in Safari.
2. The first launch downloads the model (**~160 MB** — allow a few minutes on Wi-Fi, keep the screen on).
3. When it shows **"Ready & fully offline"**:
   - **Take Photo** — opens the camera
   - **Choose Image** — opens your photo library
4. Edit the list of classes to match what you're looking for, then tap.
5. Tap **Share → Add to Home Screen** to install it like a native app.

After that it works in Airplane Mode, forever, with no internet.

## How it works

```
iPhone browser
    ├── Transformers.js (ONNX Runtime)
    │     ├── CLIP ViT-B/32 vision encoder → image features
    │     ├── CLIP text encoder → class-name features
    │     └── cosine similarity → confidence per class
    ├── Service worker → app shell cached offline
    └── Model cache → model files cached after first download
```

CLIP compares your photo's visual features against text features of whatever class names you provide. The most similar class wins. This is the same technique that powers modern search, only it runs entirely inside the browser.

## Configuration

| Setting | What it does |
|---|---|
| Objects list | Comma-separated class names. Only list classes you actually expect — CLIP always picks the *least-wrong* answer from your list. |
| Show top | How many results to display. |
| Max classes | Cap the candidate list (boosts speed and focus). |

## Tech stack

| Piece | Choice |
|---|---|
| Inference | [`@huggingface/transformers`](https://github.com/huggingface/transformers.js) v3 (WebGPU with WASM fallback) |
| Model | [`Xenova/clip-vit-base-patch32`](https://huggingface.co/Xenova/clip-vit-base-patch32) (quantized ONNX, ~160 MB) |
| Framework | None — vanilla HTML/CSS/JS, zero build step |
| Hosting | GitHub Pages (free, static, HTTPS) |
| Offline | Service worker + Cache API |

## Local development

```bash
# serve the app locally (no build step)
python3 -m http.server 8080 --bind 0.0.0.0
# open http://localhost:8080
```

Note: browser camera + service workers require HTTPS (or `localhost`). For a quick public preview, `localhost.run`, `serveo.net`, or GitHub Pages all work.

## Deployment (GitHub Pages)

The app is a static folder — deploy by pushing it:

```bash
git add .
git commit -m "update"
git push          # Pages redeploys automatically in ~1 minute
```

Pages is already configured on repo `KIller13b/object-classifier` and enabled from the `main` branch root.

## Project structure

```
pwa/
├── index.html            # UI (single file)
├── app.js                # CLIP pipeline + classification logic
├── sw.js                 # Service worker (offline app shell)
├── manifest.webmanifest  # PWA install manifest
└── *.png                 # App icons
```

## Known limitations

- **Speed:** ~2–4 s per photo on recent iPhones (WebGPU), ~5–8 s on older iOS (WASM fallback). It classifies still photos, not live video.
- **Semantics:** CLIP recognizes *general concepts*. Brand names and novel-specific classes (e.g. *"defect type 7"*) won't work without fine-tuning; for new objects just reword the class list.
- **Candidate bias:** CLIP always returns an answer from your class list, even if none matches. Keep the list to things you actually expect.
- **iOS version:** needs iOS 17.4+ to load the model reliably; older iOS may crash while downloading large models.

## Extending

Ideas for next steps:
- Swap the model in `app.js` for a larger CLIP (ViT-L/14) or a distilled variant — just change the `MODEL` constant.
- Add a "save class lists" presets feature (persist to `localStorage`).
- Turn it into a native Core ML app and sideload it (see the repo history for the reasoning).
- Fine-tune the classifiers on your own object set, then export to ONNX and drop the weights into the app.

## License

Private project. Model weights are from [Hugging Face](https://huggingface.co/Xenova/clip-vit-base-patch32) (OpenAI CLIP, MIT) and follow their respective licenses.