# AdaIN Studio — Neural Style Transfer

A Flask web app that repaints any photo in the style of a chosen artwork in real time, using **Adaptive Instance Normalization (AdaIN)** — [Huang & Belongie, ICCV 2017](https://arxiv.org/abs/1703.06868).

### Demo

<video src="ADN_Output.mp4" controls width="100%">
  Your browser doesn't support inline video — <a href="ADN_Output.mp4">download/view it directly</a>.
</video>

![AdaIN Studio hero](docs/screenshot-hero.png)

## How it works

1. A frozen, ImageNet-pretrained **VGG-19** encoder extracts feature maps from both the content photo and the style image.
2. **AdaIN** aligns the content features' channel-wise mean/variance to the style features' — a single, parameter-free statistics swap that transfers style without a slow optimization loop.
3. A compact, trained **decoder** network inverts those features back into an image.

The whole pipeline is a single forward pass and is fully convolutional, so it runs at any input resolution (256 / 512px, selectable in the UI).

![Result view](docs/screenshot-result.png)

## Features

- Upload your own content/style images, or pick from a curated preset gallery (filterable by category)
- Adjustable style strength (alpha blending) and output resolution
- Before/after compare slider on the result
- Light/dark theme, responsive layout
- Download or share the generated image
- Self-diagnosing preset gallery — if the sample image folders are missing, the page tells you exactly which path it looked for instead of silently showing "0 styles"

## Tech stack

- **Backend:** Flask, Flask-WTF, PyTorch, torchvision
- **Model:** VGG-19 encoder (frozen) + custom-trained AdaIN decoder
- **Frontend:** Jinja2 templates, vanilla CSS/JS (no frontend framework/build step)

## Project structure

```
app.py                  # Flask app: routes, forms, preset handling, style_transfer()
utils/models.py         # VGGEncoder, Decoder network definitions
utils/utils.py          # AdaIN math (adaptive_instance_normalization, calc_mean_std)
templates/index.html    # Single-page frontend
static/                 # CSS is inline in the template; presets/uploads live here
content_data/            style_data/        # Source images for the preset gallery
examples/                                   # Hero/demo before-after images
experiment/final_exp/   # Trained decoder weights go here (see Setup)
vgg_normalised.pth      # Pretrained VGG-19 weights go here (see Setup)
```

## Setup

```bash
git clone <your-repo-url>
cd NST_code
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### Model weights (required, not included in this repo)

Trained model weights are binary files tens of megabytes in size and are **not committed to git** (see `.gitignore`). You need these two files in place before the app will start:

| File | Location |
|---|---|
| `vgg_normalised.pth` | project root |
| `decoder_final.pth` | `experiment/final_exp/` |

Run this once after cloning to fetch them automatically:
```bash
python download_weights.py
```
(The URLs are already filled in at the top of that file, pointing to this repo's GitHub Release.) If it reports a file as `SKIPPED`, or you'd rather do it by hand, download `vgg_normalised.pth` and `decoder_final.pth` from this repo's **Releases** page and place them at the paths above yourself.

If you're forking this to train your own style transfer model instead, run `train.py` against your own dataset to produce `decoder_final.pth`, then edit the URLs at the top of `download_weights.py` to point at wherever you host your own weights.

### Run

```bash
python app.py
```

By default this starts on `http://127.0.0.1:5000` with debug mode **off**. To develop locally with auto-reload and the interactive debugger:

```bash
# Windows PowerShell
$env:FLASK_DEBUG="1"; python app.py
# macOS/Linux
FLASK_DEBUG=1 python app.py
```

**Do not set `FLASK_DEBUG=1` on a publicly hosted deployment** — the Werkzeug interactive debugger allows arbitrary code execution if it's reachable from the internet.

### Deploying on Render

1. Push this repo to GitHub (see below), then on Render: **New → Web Service**, connect the repo.
2. **Build Command:**
   ```
   pip install -r requirements.txt && python download_weights.py
   ```
   Your `.pth` weight files aren't in git (see above), so they have to be fetched during the build. The URLs are already filled in at the top of `download_weights.py` (pointing at this repo's GitHub Release) — Render will pick them up automatically. If you ever change where the weights are hosted, either edit those URLs directly or set `VGG_WEIGHTS_URL` / `DECODER_WEIGHTS_URL` as environment variables in Render's dashboard, which override the values in the file.
3. **Start Command:**
   ```
   gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120
   ```
   (Render also auto-detects this from the included `Procfile`, so you can usually leave the Start Command blank.) `app.run()`'s built-in dev server is not meant for production — Render runs your app through gunicorn instead.
4. Add an environment variable `SECRET_KEY` set to any long random string (don't reuse a value from this README).
5. **Heads up on free-tier RAM:** Render's free web service plan has 512MB RAM, which can be too little for PyTorch plus a VGG-19 forward pass together — app.py caps input images and output resolution (256/512px only) to reduce peak memory use, but if the service still crashes/restarts under load, the paid "Starter" tier (more RAM) is the reliable fix. See the demo video above for the app running without that constraint.

The app itself reads `HOST`, `PORT`, `SECRET_KEY` and `FLASK_DEBUG` from the environment, so none of this requires code changes — Render just needs to be told to set them.

## Credits

AdaIN algorithm: Huang, X. & Belongie, S. (2017), *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization*, ICCV.
