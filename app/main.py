from __future__ import annotations

import io
from typing import Any, Dict, List
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import base64
from io import BytesIO
from PIL import Image
import cv2

from . import config
from .model_loader import get_model
from .preprocess import load_image_bytes, preprocess_for_model
from .gradcam import compute_gradcam

app = FastAPI(title="Chest X-ray Detection API", version="1.0.0")

# UI index path
BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_PATH = BASE_DIR / "web" / "index.html"


@app.on_event("startup")
def _warmup():
    # Attempt to load model on startup so first request is fast
    try:
        m, input_size, num_classes = get_model()
        # Optional: warmup with zeros
        w, h = input_size
        dummy = np.zeros((1, h, w, 3), dtype=np.float32)
        _ = m.predict(dummy, verbose=0)
    except Exception as e:  # pragma: no cover
        # Don't crash startup; prediction will raise detailed error
        print(f"[WARN] Model warmup failed: {e}")


@app.get("/", include_in_schema=False)
def index():
    # Serve a simple UI to upload an image
    if INDEX_PATH.exists():
        return FileResponse(INDEX_PATH)
    return JSONResponse({"message": "UI not found. Create web/index.html to use the UI."})


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        _, input_size, num_classes = get_model()
        return {
            "status": "ok",
            "model_loaded": True,
            "input_size": {"width": input_size[0], "height": input_size[1]},
            "num_classes": num_classes,
            "labels": config.CLASS_LABELS,
        }
    except Exception as e:
        return {"status": "error", "model_loaded": False, "error": str(e)}


@app.post("/predict")
def predict(file: UploadFile = File(...), top_k: int = 3) -> JSONResponse:
    try:
        contents = file.file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded file")

    try:
        model, input_size, num_classes = get_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        arr = load_image_bytes(contents, target_size=input_size, force_rgb=config.FORCE_RGB)
        x = preprocess_for_model(arr, normalize=config.NORMALIZE)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {e}")

    try:
        preds = model.predict(x, verbose=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference error: {e}")

    probs = preds[0]
    # If output is a single logit, wrap into 2-class style probabilities using sigmoid
    if np.ndim(probs) == 0 or (isinstance(probs, np.ndarray) and probs.shape == ()):  # scalar
        p1 = float(1.0 / (1.0 + np.exp(-float(probs))))
        probs = np.array([1.0 - p1, p1], dtype=np.float32)
    else:
        probs = np.array(probs).astype(float)

    # Normalize if not already
    if probs.sum() <= 0 or np.any(np.isnan(probs)):
        # Softmax
        e = np.exp(probs - np.max(probs))
        probs = e / e.sum()

    # Derive labels
    if config.CLASS_LABELS and len(config.CLASS_LABELS) == len(probs):
        labels = config.CLASS_LABELS
    else:
        labels = [f"class_{i}" for i in range(len(probs))]

    # Top-k
    top_k = max(1, min(top_k, len(probs)))
    indices = np.argsort(probs)[::-1][:top_k]

    top = [
        {
            "index": int(i),
            "label": labels[i],
            "score": float(probs[i]),
        }
        for i in indices
    ]

    predicted_index = int(indices[0])
    predicted_label = labels[predicted_index]

    # Thresholded boolean (for binary)
    threshold = float(config.SCORE_THRESHOLD)
    above_threshold = float(probs[predicted_index]) >= threshold if threshold > 0 else None

    return JSONResponse(
        content={
            "top": top,
            "predicted": {
                "index": predicted_index,
                "label": predicted_label,
                "score": float(probs[predicted_index]),
                "above_threshold": above_threshold,
            },
            "raw": probs.tolist(),
        }
    )


def _encode_png(arr: np.ndarray) -> str:
    # arr expected uint8 HxWxC (RGB) or HxW (grayscale)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


@app.post("/predict_visualize")
def predict_visualize(file: UploadFile = File(...), top_k: int = 3, overlay_alpha: float = 0.45) -> JSONResponse:
    try:
        contents = file.file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded file")

    try:
        model, input_size, num_classes = get_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        arr = load_image_bytes(contents, target_size=input_size, force_rgb=config.FORCE_RGB)  # float32 0..255
        x = preprocess_for_model(arr, normalize=config.NORMALIZE)  # (1,H,W,C) float32
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {e}")

    try:
        preds = model.predict(x, verbose=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference error: {e}")

    probs = preds[0]
    if np.ndim(probs) == 0 or (isinstance(probs, np.ndarray) and probs.shape == ()):  # scalar
        p1 = float(1.0 / (1.0 + np.exp(-float(probs))))
        probs = np.array([1.0 - p1, p1], dtype=np.float32)
    else:
        probs = np.array(probs).astype(float)

    # Normalize if not already
    if probs.sum() <= 0 or np.any(np.isnan(probs)):
        e = np.exp(probs - np.max(probs))
        probs = e / e.sum()

    # Labels
    if config.CLASS_LABELS and len(config.CLASS_LABELS) == len(probs):
        labels = config.CLASS_LABELS
    else:
        labels = [f"class_{i}" for i in range(len(probs))]

    # Top-k
    top_k = max(1, min(top_k, len(probs)))
    indices = np.argsort(probs)[::-1][:top_k]
    predicted_index = int(indices[0])
    predicted_label = labels[predicted_index]

    # Grad-CAM for predicted class
    try:
        heatmap, used_class = compute_gradcam(model, x, class_index=predicted_index)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM error: {e}")

    # Prepare overlay
    # arr is HxWx3 float32 in 0..255
    base_rgb = np.clip(arr, 0, 255).astype(np.uint8)
    # OpenCV expects BGR
    base_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
    heat_255 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_255, cv2.COLORMAP_JET)
    overlay_bgr = cv2.addWeighted(base_bgr, 1.0 - overlay_alpha, heat_color, overlay_alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    # Encode images
    heat_gray_rgb = cv2.cvtColor(heat_255, cv2.COLOR_GRAY2RGB)
    out = {
        "predicted": {
            "index": predicted_index,
            "label": predicted_label,
            "score": float(probs[predicted_index]),
        },
        "top": [
            {"index": int(i), "label": labels[i], "score": float(probs[i])} for i in indices
        ],
        "raw": probs.tolist(),
        "images": {
            "input": _encode_png(base_rgb),
            "heatmap": _encode_png(heat_gray_rgb),
            "overlay": _encode_png(overlay_rgb),
        },
    }
    return JSONResponse(content=out)
