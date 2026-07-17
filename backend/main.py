"""
FastAPI Backend – Pneumonia Detection Service
=============================================
Exposes a single POST /predict endpoint that accepts a chest X-ray image
and returns the predicted class (NORMAL / PNEUMONIA) with confidence.

Environment variables:
  MODEL_PATH        – Path to .h5 model file   (default: ../ml/outputs/pneumonia_vgg19_final.h5)
  CLASS_INDICES_PATH– Path to class_indices.json (default: ../ml/outputs/class_indices.json)
  PORT              – Server port               (default: 8000)

Run locally:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import io
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf

# ──────────────────────── Config ─────────────────────────────────
MODEL_PATH         = os.getenv("MODEL_PATH",         "../ml/outputs/pneumonia_vgg19_final.h5")
CLASS_INDICES_PATH = os.getenv("CLASS_INDICES_PATH", "../ml/outputs/class_indices.json")
IMG_SIZE           = (224, 224)
MAX_FILE_BYTES     = 10 * 1024 * 1024   # 10 MB

# ──────────────────────── FastAPI app ─────────────────────────────
app = FastAPI(
    title="Pneumonia Detection API",
    description="Upload a chest X-ray image and receive a binary classification (NORMAL / PNEUMONIA).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────── Model loading ───────────────────────────
_model: Optional[tf.keras.Model] = None
_class_names: Optional[list]     = None


def get_model() -> tf.keras.Model:
    global _model
    if _model is None:
        model_path = Path(MODEL_PATH)
        if not model_path.exists():
            raise RuntimeError(
                f"Model file not found at '{model_path}'. "
                "Train the model first (see ml/train.py) and update MODEL_PATH."
            )
        print(f"Loading model from {model_path} …")
        _model = tf.keras.models.load_model(str(model_path))
        print("Model loaded.")
    return _model


def get_class_names() -> list:
    global _class_names
    if _class_names is None:
        p = Path(CLASS_INDICES_PATH)
        if p.exists():
            with open(p) as f:
                indices = json.load(f)
            # indices = {"NORMAL": 0, "PNEUMONIA": 1}
            _class_names = [k for k, v in sorted(indices.items(), key=lambda x: x[1])]
        else:
            _class_names = ["NORMAL", "PNEUMONIA"]   # sensible default
    return _class_names


# ──────────────────────── Startup ─────────────────────────────────
@app.on_event("startup")
async def startup_event():
    try:
        get_model()
        get_class_names()
    except RuntimeError as exc:
        print(f"[WARN] {exc}")
        print("[WARN] The /predict endpoint will return 503 until the model is present.")


# ──────────────────────── Helper ──────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert raw bytes → (1, 224, 224, 3) float32 array scaled to [0,1]."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ──────────────────────── Routes ──────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Pneumonia Detection API is running. POST an image to /predict"}


@app.get("/health")
async def health():
    model_ready = Path(MODEL_PATH).exists()
    return {"status": "ok" if model_ready else "model_missing", "model_ready": model_ready}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept a JPEG/PNG chest X-ray image and return:
    - **prediction**: "NORMAL" | "PNEUMONIA"
    - **confidence**: probability (0-1) for the predicted class
    - **probabilities**: {NORMAL: float, PNEUMONIA: float}
    - **inference_ms**: server-side inference time in milliseconds
    """
    # ── Validate ──────────────────────────────────────────────────
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Upload a JPEG or PNG image.",
        )

    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB).")
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file received.")

    # ── Load model ────────────────────────────────────────────────
    try:
        model      = get_model()
        class_names = get_class_names()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # ── Preprocess ────────────────────────────────────────────────
    try:
        img_array = preprocess_image(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not process image: {exc}")

    # ── Inference ─────────────────────────────────────────────────
    t0   = time.perf_counter()
    prob = float(model.predict(img_array, verbose=0)[0][0])   # sigmoid output
    ms   = round((time.perf_counter() - t0) * 1000, 1)

    # prob ≈ P(class_index=1) – which class that maps to depends on the generator ordering
    # By convention keras ImageDataGenerator orders alphabetically: NORMAL=0, PNEUMONIA=1
    pneumonia_prob = prob
    normal_prob    = 1.0 - prob
    predicted_idx  = int(prob >= 0.5)
    prediction     = class_names[predicted_idx]
    confidence     = pneumonia_prob if predicted_idx == 1 else normal_prob

    return JSONResponse({
        "prediction":   prediction,
        "confidence":   round(confidence, 4),
        "probabilities": {
            class_names[0]: round(normal_prob,    4),
            class_names[1]: round(pneumonia_prob, 4),
        },
        "inference_ms": ms,
    })
