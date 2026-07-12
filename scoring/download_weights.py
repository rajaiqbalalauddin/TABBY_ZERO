"""
Download pretrained NIMA weights (MobileNetV2 backbone, trained on AVA dataset).

Usage:
    python scoring/download_weights.py

This downloads the weights to weights/nima_mobilenetv2.pth which is the default
path expected by NimaScorer in config/settings.yaml.

If automatic download fails, you can train your own weights using the AVA dataset:
    1. Download AVA dataset (images + labels)
    2. Use scoring/train_nima.py (provided separately)
    3. Place resulting .pth file in weights/
"""

import os
import sys
import urllib.request
import hashlib

# Pretrained NIMA MobileNetV2 weights (trained on AVA dataset)
# Source: idealo/image-quality-assessment (converted to pure PyTorch state_dict)
WEIGHTS_URL = "https://github.com/titu1994/neural-image-assessment/releases/download/v0.5/nima_mobilenetv2_ava.pth"
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "nima_mobilenetv2.pth")


def download_weights() -> None:
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    if os.path.exists(WEIGHTS_PATH):
        print(f"Weights already exist at: {WEIGHTS_PATH}")
        return

    print(f"Downloading NIMA weights...")
    print(f"  URL: {WEIGHTS_URL}")
    print(f"  Destination: {WEIGHTS_PATH}")
    print()

    try:
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH, _progress_hook)
        print(f"\nDone! Weights saved to: {WEIGHTS_PATH}")
    except Exception as e:
        print(f"\nAutomatic download failed: {e}")
        print()
        print("Manual alternatives:")
        print("  1. Download NIMA weights from a trusted source")
        print("  2. Place the .pth file at: weights/nima_mobilenetv2.pth")
        print("  3. Or train your own using the AVA dataset")
        print()
        print("The pipeline will still work without weights (using ImageNet baseline)")
        print("but scores will be less accurate for aesthetic assessment.")
        sys.exit(1)


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 // total_size)
        bar = "#" * (percent // 2) + "-" * (50 - percent // 2)
        print(f"\r  [{bar}] {percent}%", end="", flush=True)


if __name__ == "__main__":
    download_weights()
