"""
Downloads the two trained model weight files this app needs to run.

These are binary files tens of MB each and are intentionally excluded from
git (see .gitignore) -- GitHub repos shouldn't carry large binaries. Anyone
cloning this repo (including an interviewer trying it locally) needs this
one extra step before `python app.py` will work.

Usage:
    python download_weights.py

Fill in VGG_URL and DECODER_URL below with direct download links (a GitHub
Release asset link works well -- see the README's "Setup" section for how
to create one). Environment variables of the same name override these if
set, which is what Render's build step uses.
"""
import os
import urllib.request

# --- Filled in with this repo's own GitHub Release assets ---
VGG_URL = os.environ.get(
    "VGG_WEIGHTS_URL",
    "https://github.com/punam-thakare22/adain-style-transfer/releases/download/v1.0/vgg_normalised.pth",
)
DECODER_URL = os.environ.get(
    "DECODER_WEIGHTS_URL",
    "https://github.com/punam-thakare22/adain-style-transfer/releases/download/v1.0/decoder_final.pth",
)
# ----------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGETS = [
    (VGG_URL, os.path.join(BASE_DIR, "vgg_normalised.pth")),
    (DECODER_URL, os.path.join(BASE_DIR, "experiment", "final_exp", "decoder_final.pth")),
]


def download(url, dest):
    if not url:
        print(f"SKIPPED: no URL set for {os.path.basename(dest)} "
              f"(edit this script or set the matching environment variable).")
        return
    if os.path.exists(dest):
        print(f"Already present: {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading {os.path.basename(dest)} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")


if __name__ == "__main__":
    for url, dest in TARGETS:
        download(url, dest)
    print("\nDone. If anything was SKIPPED, open this file and fill in VGG_URL / DECODER_URL,")
    print("or place vgg_normalised.pth / experiment/final_exp/decoder_final.pth there yourself.")
