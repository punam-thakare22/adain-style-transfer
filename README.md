# AdaIN Studio — Neural Style Transfer

A Flask web app that repaints any photo in the style of a chosen artwork in real time, using **Adaptive Instance Normalization (AdaIN)** — [Huang & Belongie, ICCV 2017](https://arxiv.org/abs/1703.06868).

![AdaIN Studio hero](docs/screenshot-hero.png)

## How it works

1. A frozen, ImageNet-pretrained **VGG-19** encoder extracts feature maps from both the content photo and the style image.
2. **AdaIN** aligns the content features' channel-wise mean/variance to the style features' — a single, parameter-free statistics swap that transfers style without a slow optimization loop.
3. A compact, trained **decoder** network inverts those features back into an image.

The whole pipeline is a single forward pass and is fully convolutional, so it runs at any input resolution (256 / 512 / 1024px, selectable in the UI).

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

Trained model weights are binary files tens of megabytes in size and are **not committed to git** (see `.gitignore`). You need to place two files yourself before the app will start:

| File | Location |
|---|---|
| `vgg_normalised.pth` | project root |
| `decoder_final.pth` | `experiment/final_exp/` |

If you trained your own decoder with `train.py`, point it at your dataset and it will produce `decoder_final.pth`. Otherwise, host your existing weight files somewhere (GitHub Releases, Google Drive, Hugging Face Hub) and download them into the paths above.

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

### Deploying

The app reads `HOST`, `PORT`, `SECRET_KEY` and `FLASK_DEBUG` from the environment, so it runs unchanged on hosts like Render/Railway/Fly.io that assign their own port. Set a real `SECRET_KEY` in that environment for any public deployment.

## Credits

AdaIN algorithm: Huang, X. & Belongie, S. (2017), *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization*, ICCV.
