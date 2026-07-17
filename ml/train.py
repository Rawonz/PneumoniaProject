"""
Chest X-Ray Pneumonia Detection - VGG19 Training Script
========================================================
Dataset: Kaggle Chest X-Ray Images (Pneumonia)
  https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Directory structure expected:
  data/chest_xray/
    train/
      NORMAL/
      PNEUMONIA/
    val/
      NORMAL/
      PNEUMONIA/
    test/
      NORMAL/
      PNEUMONIA/

Usage:
  python train.py --data_dir data/chest_xray --epochs 20 --batch_size 32
"""

import argparse
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ─────────────────────────── CLI args ────────────────────────────
parser = argparse.ArgumentParser(description="Train VGG19 for pneumonia detection")
parser.add_argument("--data_dir",   default="data/chest_xray", help="Root of dataset")
parser.add_argument("--output_dir", default="outputs",          help="Where to save model + artifacts")
parser.add_argument("--img_size",   type=int, default=224,       help="Image size (square)")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs",     type=int, default=20)
parser.add_argument("--lr",         type=float, default=1e-4,   help="Learning rate")
parser.add_argument("--fine_tune",  action="store_true",         help="Unfreeze last VGG19 block for fine-tuning")
args = parser.parse_args()

IMG_SIZE   = (args.img_size, args.img_size)
BATCH_SIZE = args.batch_size
EPOCHS     = args.epochs
LR         = args.lr
DATA_DIR   = Path(args.data_dir)
OUT_DIR    = Path(args.output_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR   = DATA_DIR / "val"
TEST_DIR  = DATA_DIR / "test"

print(f"\n{'='*60}")
print(f"  VGG19 Pneumonia Classifier  |  {datetime.now():%Y-%m-%d %H:%M}")
print(f"{'='*60}")
print(f"  Data : {DATA_DIR}")
print(f"  Out  : {OUT_DIR}")
print(f"  IMG  : {IMG_SIZE}  Batch: {BATCH_SIZE}  Epochs: {EPOCHS}  LR: {LR}")
print(f"{'='*60}\n")

# ──────────────────────── Data generators ────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)

val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
)
val_gen = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)
test_gen = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

CLASS_NAMES = list(train_gen.class_indices.keys())
print(f"Classes : {CLASS_NAMES}")
print(f"Train   : {train_gen.samples} samples")
print(f"Val     : {val_gen.samples} samples")
print(f"Test    : {test_gen.samples} samples\n")

# ───────────────────────── Class weights ─────────────────────────
counts    = np.bincount(train_gen.classes)
total     = counts.sum()
class_wt  = {i: total / (len(counts) * c) for i, c in enumerate(counts)}
print(f"Class weights: {class_wt}\n")

# ──────────────────────── Build model ────────────────────────────
base_model = VGG19(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
base_model.trainable = False          # Freeze during initial training

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),
], name="vgg19_pneumonia")

model.compile(
    optimizer=optimizers.Adam(learning_rate=LR),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)
model.summary()

# ──────────────────────── Callbacks ──────────────────────────────
ckpt_path = OUT_DIR / "best_model.h5"
cb_list = [
    callbacks.ModelCheckpoint(
        str(ckpt_path), monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
    ),
    callbacks.CSVLogger(str(OUT_DIR / "training_log.csv")),
]

# ────────────────────── Phase 1: Frozen base ─────────────────────
print("\n── Phase 1: Training head (frozen VGG19 base) ──\n")
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    class_weight=class_wt,
    callbacks=cb_list,
)

# ──────────── Optional Phase 2: Fine-tune last VGG19 block ───────
if args.fine_tune:
    print("\n── Phase 2: Fine-tuning last VGG19 block ──\n")
    for layer in base_model.layers[-4:]:   # last conv block (block5)
        layer.trainable = True

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR / 10),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    cb_list[0] = callbacks.ModelCheckpoint(
        str(ckpt_path), monitor="val_accuracy", save_best_only=True, verbose=1
    )
    history_ft = model.fit(
        train_gen,
        epochs=EPOCHS // 2,
        validation_data=val_gen,
        class_weight=class_wt,
        callbacks=cb_list,
    )
    # Merge histories
    for k in history.history:
        history.history[k].extend(history_ft.history.get(k, []))

# ──────────────────── Evaluate on test set ───────────────────────
print("\n── Evaluating on test set ──\n")
model.load_weights(str(ckpt_path))

test_preds_proba = model.predict(test_gen, verbose=1).ravel()
test_preds = (test_preds_proba >= 0.5).astype(int)
test_true  = test_gen.classes

acc  = accuracy_score(test_true, test_preds)
prec = precision_score(test_true, test_preds)
rec  = recall_score(test_true, test_preds)
f1   = f1_score(test_true, test_preds)
cm   = confusion_matrix(test_true, test_preds)

print(f"\n{'─'*40}")
print(f"  Test Accuracy  : {acc:.4f}")
print(f"  Precision      : {prec:.4f}")
print(f"  Recall         : {rec:.4f}")
print(f"  F1-Score       : {f1:.4f}")
print(f"{'─'*40}\n")
print(classification_report(test_true, test_preds, target_names=CLASS_NAMES))

# ─────────────────────── Save metrics ────────────────────────────
metrics = {
    "accuracy":  round(acc,  4),
    "precision": round(prec, 4),
    "recall":    round(rec,  4),
    "f1":        round(f1,   4),
    "class_names": CLASS_NAMES,
    "confusion_matrix": cm.tolist(),
    "trained_at": datetime.now().isoformat(),
}
with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved → {OUT_DIR / 'metrics.json'}")

# ─────────────────── Plot training curves ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("VGG19 Pneumonia Classifier – Training Curves", fontsize=14, fontweight="bold")

axes[0].plot(history.history["accuracy"],     label="Train Accuracy")
axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history["loss"],     label="Train Loss")
axes[1].plot(history.history["val_loss"], label="Val Loss")
axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch"); axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "training_curves.png", dpi=150)
plt.close()
print(f"Training curves → {OUT_DIR / 'training_curves.png'}")

# ─────────────────── Plot confusion matrix ───────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    ax=ax,
)
ax.set_title("Confusion Matrix – Test Set", fontsize=13, fontweight="bold")
ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
plt.close()
print(f"Confusion matrix → {OUT_DIR / 'confusion_matrix.png'}")

# ────────────────── Export final model ───────────────────────────
final_path = OUT_DIR / "pneumonia_vgg19_final.h5"
model.save(str(final_path))
print(f"\nFinal model saved → {final_path}")

# Save class indices so the API can decode predictions
with open(OUT_DIR / "class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f, indent=2)
print(f"Class indices   → {OUT_DIR / 'class_indices.json'}")

print(f"\n{'='*60}")
print("  Training complete!")
print(f"{'='*60}\n")
