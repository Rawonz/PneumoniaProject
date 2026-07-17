"""
Exploratory Data Analysis – Chest X-Ray Pneumonia Dataset
==========================================================
Run before training to understand class distribution, image statistics,
and visualise sample images from each class.

Usage:
  python eda.py --data_dir data/chest_xray --output_dir outputs
"""

import argparse
import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from PIL import Image
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",   default="data/chest_xray")
parser.add_argument("--output_dir", default="outputs/eda")
args = parser.parse_args()

DATA_DIR = Path(args.data_dir)
OUT_DIR  = Path(args.output_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]
CLASSES = ["NORMAL", "PNEUMONIA"]


# ─────────────── 1. Count images per split × class ───────────────
counts = defaultdict(dict)
for split in SPLITS:
    for cls in CLASSES:
        p = DATA_DIR / split / cls
        counts[split][cls] = len(list(p.glob("*.jpeg")) + list(p.glob("*.jpg")) + list(p.glob("*.png")))
        print(f"  {split:6s} / {cls:10s}: {counts[split][cls]:5d}")

# ─────────────── 2. Bar chart of distribution ────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(SPLITS))
w = 0.35
bars_n = ax.bar(x - w/2, [counts[s]["NORMAL"]    for s in SPLITS], w, label="Normal",    color="#4A90D9")
bars_p = ax.bar(x + w/2, [counts[s]["PNEUMONIA"] for s in SPLITS], w, label="Pneumonia", color="#E05252")
ax.set_xticks(x); ax.set_xticklabels([s.title() for s in SPLITS])
ax.set_ylabel("Image count"); ax.set_title("Class Distribution per Split", fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=0.3)
for bar in [*bars_n, *bars_p]:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "class_distribution.png", dpi=150)
plt.close()
print(f"\nClass distribution chart → {OUT_DIR / 'class_distribution.png'}")

# ─────────────── 3. Sample images from each class ────────────────
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
fig.suptitle("Sample X-Ray Images per Class (Train Split)", fontsize=14, fontweight="bold")
for row, cls in enumerate(CLASSES):
    img_dir = DATA_DIR / "train" / cls
    imgs    = list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.jpg"))
    samples = random.sample(imgs, min(5, len(imgs)))
    for col, path in enumerate(samples):
        img = Image.open(path).convert("L")
        axes[row][col].imshow(img, cmap="gray")
        axes[row][col].set_title(cls, fontsize=10, color="#333")
        axes[row][col].axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "sample_images.png", dpi=150)
plt.close()
print(f"Sample images        → {OUT_DIR / 'sample_images.png'}")

# ─────────────── 4. Pixel intensity histogram ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Pixel Intensity Distribution (100 Train Images per Class)", fontweight="bold")
for idx, cls in enumerate(CLASSES):
    img_dir = DATA_DIR / "train" / cls
    imgs    = (list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.jpg")))[:100]
    all_px  = []
    for p in imgs:
        arr = np.array(Image.open(p).convert("L").resize((224, 224))) / 255.0
        all_px.extend(arr.ravel())
    axes[idx].hist(all_px, bins=50, color=("#4A90D9" if cls == "NORMAL" else "#E05252"), alpha=0.75, edgecolor="none")
    axes[idx].set_title(cls); axes[idx].set_xlabel("Pixel intensity"); axes[idx].set_ylabel("Count")
    axes[idx].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "pixel_intensity.png", dpi=150)
plt.close()
print(f"Pixel intensity hist → {OUT_DIR / 'pixel_intensity.png'}")

# ─────────────── 5. Image size distribution ──────────────────────
widths, heights, labels_list = [], [], []
for cls in CLASSES:
    img_dir = DATA_DIR / "train" / cls
    imgs    = (list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.jpg")))[:200]
    for p in imgs:
        w, h = Image.open(p).size
        widths.append(w); heights.append(h); labels_list.append(cls)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Image Dimension Distribution (Train – 200 per Class)", fontweight="bold")
colors = {"NORMAL": "#4A90D9", "PNEUMONIA": "#E05252"}
for cls in CLASSES:
    mask = [l == cls for l in labels_list]
    w_cls = [w for w, m in zip(widths,  mask) if m]
    h_cls = [h for h, m in zip(heights, mask) if m]
    axes[0].hist(w_cls, bins=30, label=cls, alpha=0.6, color=colors[cls])
    axes[1].hist(h_cls, bins=30, label=cls, alpha=0.6, color=colors[cls])
axes[0].set_title("Width");  axes[0].set_xlabel("px"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].set_title("Height"); axes[1].set_xlabel("px"); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "image_dimensions.png", dpi=150)
plt.close()
print(f"Image dimensions     → {OUT_DIR / 'image_dimensions.png'}")

print("\nEDA complete.\n")
