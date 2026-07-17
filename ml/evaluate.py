"""
Model Evaluation Script
========================
Loads the saved .h5 model and runs a full evaluation on the test set,
printing metrics and regenerating confusion-matrix / ROC-curve plots.

Usage:
  python evaluate.py --model_path outputs/pneumonia_vgg19_final.h5 \
                     --data_dir   data/chest_xray \
                     --output_dir outputs
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="outputs/pneumonia_vgg19_final.h5")
parser.add_argument("--data_dir",   default="data/chest_xray")
parser.add_argument("--output_dir", default="outputs")
parser.add_argument("--img_size",   type=int, default=224)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

IMG_SIZE = (args.img_size, args.img_size)
OUT_DIR  = Path(args.output_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading model …")
model = tf.keras.models.load_model(args.model_path)

test_gen = ImageDataGenerator(rescale=1.0/255).flow_from_directory(
    Path(args.data_dir) / "test",
    target_size=IMG_SIZE,
    batch_size=args.batch_size,
    class_mode="binary",
    shuffle=False,
)

CLASS_NAMES = list(test_gen.class_indices.keys())
print(f"Classes : {CLASS_NAMES}")
print(f"Test    : {test_gen.samples} samples")

proba = model.predict(test_gen, verbose=1).ravel()
preds = (proba >= 0.5).astype(int)
true  = test_gen.classes

acc  = accuracy_score(true, preds)
prec = precision_score(true, preds)
rec  = recall_score(true, preds)
f1   = f1_score(true, preds)
cm   = confusion_matrix(true, preds)

print(f"\n{'─'*40}")
print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print(f"{'─'*40}\n")
print(classification_report(true, preds, target_names=CLASS_NAMES))

# ── Confusion matrix ──
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_title("Confusion Matrix – Test Set", fontweight="bold")
ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix_eval.png", dpi=150)
plt.close()

# ── ROC curve ──
fpr, tpr, _ = roc_curve(true, proba)
roc_auc     = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
ax.plot([0,1],[0,1],"--", color="gray")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve – Test Set", fontweight="bold")
ax.legend(loc="lower right"); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "roc_curve.png", dpi=150)
plt.close()

print(f"Plots saved to {OUT_DIR}")

metrics = {
    "accuracy":  round(acc,  4),
    "precision": round(prec, 4),
    "recall":    round(rec,  4),
    "f1":        round(f1,   4),
    "roc_auc":   round(roc_auc, 4),
}
with open(OUT_DIR / "eval_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Done.")
