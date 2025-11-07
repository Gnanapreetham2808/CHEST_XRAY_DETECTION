from __future__ import annotations

import os
from typing import List
from pathlib import Path
import json

import numpy as np
import streamlit as st
import cv2
from PIL import Image

from app.model_loader import get_model
from app.preprocess import preprocess_for_model
from app.gradcam import compute_gradcam
from app import config

st.set_page_config(page_title="Chest X-ray Detection", layout="wide")

st.title("🩺 Chest X-ray Detection")
st.write("Upload an X-ray image to get predicted disease probabilities and Grad-CAM visual explanations.")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    labels_input = st.text_input(
        "Class labels (comma-separated)",
        value=",".join(config.CLASS_LABELS) if config.CLASS_LABELS else "",
        help="Override labels for display. Must match the model's output size.",
    )
    if labels_input.strip():
        labels = [s.strip() for s in labels_input.split(",") if s.strip()]
    else:
        # Will derive later
        labels = None
    top_k = st.slider("Top-K", min_value=1, max_value=10, value=3)
    overlay_alpha = st.slider("Overlay alpha", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
    normalize_mode = config.NORMALIZE
    st.caption(f"Model path: {config.MODEL_PATH}")

# File uploader
uploaded = st.file_uploader("Select an image", type=["png", "jpg", "jpeg"])

@st.cache_resource(show_spinner=True)
def _load_model_cached():
    model, input_size, num_classes = get_model()
    return model, input_size, num_classes

@st.cache_data(show_spinner=False)
def _prepare_image(data: bytes, input_size):
    from app.preprocess import load_image_bytes
    arr = load_image_bytes(data, target_size=input_size, force_rgb=config.FORCE_RGB)
    x = preprocess_for_model(arr, normalize=normalize_mode)
    return arr, x

@st.cache_data(show_spinner=False)
def _predict(model, x):
    preds = model.predict(x, verbose=0)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    probs = preds[0]
    if np.ndim(probs) == 0 or (isinstance(probs, np.ndarray) and probs.shape == ()):  # scalar -> binary
        p1 = float(1.0 / (1.0 + np.exp(-float(probs))))
        probs = np.array([1.0 - p1, p1], dtype=np.float32)
    else:
        probs = np.array(probs).astype(float)
    if probs.sum() <= 0 or np.any(np.isnan(probs)):
        e = np.exp(probs - np.max(probs))
        probs = e / e.sum()
    return probs

@st.cache_data(show_spinner=False)
def _gradcam(model, x, class_index):
    heatmap, used_class = compute_gradcam(model, x, class_index=class_index)
    return heatmap

def _make_overlay(base_rgb: np.ndarray, heatmap: np.ndarray, alpha: float):
    base_rgb_u8 = np.clip(base_rgb, 0, 255).astype(np.uint8)
    base_bgr = cv2.cvtColor(base_rgb_u8, cv2.COLOR_RGB2BGR)
    heat_255 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_255, cv2.COLORMAP_JET)
    overlay_bgr = cv2.addWeighted(base_bgr, 1.0 - alpha, heat_color, alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return heat_255, overlay_rgb

def _try_load_labels_file(expected_len: int | None = None) -> List[str] | None:
    """Load labels from a local file (labels.txt), one label per line.
    Optionally validate length if expected_len is provided.
    """
    for p in [Path("labels.txt"), Path(__file__).resolve().parent / "labels.txt"]:
        if p.exists():
            try:
                lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
                if expected_len is None or len(lines) == expected_len:
                    return lines
            except Exception:
                pass
    return None

@st.cache_data(show_spinner=False)
def _load_class_descriptions() -> dict:
    """Load class descriptions from class_descriptions.json, keyed by lowercase label."""
    for p in [Path("class_descriptions.json"), Path(__file__).resolve().parent / "class_descriptions.json"]:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                # normalize keys to lowercase
                return {str(k).lower(): str(v) for k, v in data.items()}
            except Exception:
                pass
    return {}

if uploaded is not None:
    st.subheader("Prediction Results")
    model, input_size, num_classes = _load_model_cached()

    arr, x = _prepare_image(uploaded.getvalue(), input_size)
    probs = _predict(model, x)

    # Determine labels with precedence: sidebar -> labels.txt -> env config -> smart defaults
    use_labels: List[str]
    if labels and len(labels) == len(probs):
        use_labels = labels
    else:
        file_labels = _try_load_labels_file(expected_len=len(probs))
        if file_labels is not None:
            use_labels = file_labels
        elif config.CLASS_LABELS and len(config.CLASS_LABELS) == len(probs):
            use_labels = config.CLASS_LABELS
        elif len(probs) == 2:
            # Common binary default
            use_labels = ["NORMAL", "PNEUMONIA"]
        else:
            use_labels = [f"class_{i}" for i in range(len(probs))]

    k = min(top_k, len(probs))
    indices = np.argsort(probs)[::-1][:k]
    predicted_index = int(indices[0])
    predicted_label = use_labels[predicted_index]
    predicted_score = float(probs[predicted_index])

    cols = st.columns([1,2])
    with cols[0]:
        st.image(arr.astype(np.uint8), caption="Input", use_column_width=True)
    with cols[1]:
        # Prominent predicted disease name with score
        display_label = predicted_label
        display_pct = predicted_score * 100.0
        # Simple coloring heuristic
        if "normal" in predicted_label.lower():
            st.success(f"Predicted: {display_label} ({display_pct:.1f}%)")
        else:
            st.warning(f"Predicted: {display_label} ({display_pct:.1f}%)")

        # Show description if available
        desc_map = _load_class_descriptions()
        desc = desc_map.get(predicted_label.lower())
        if desc:
            st.info(desc)
        st.markdown("### Top Classes")
        for i in indices:
            pct = probs[i] * 100.0
            bar = st.progress(min(max(pct/100.0, 0.0),1.0), text=f"{use_labels[i]}: {pct:.1f}%")
        st.markdown("\n")

    # Grad-CAM
    try:
        heatmap = _gradcam(model, x, class_index=predicted_index)
        heat_255, overlay_rgb = _make_overlay(arr, heatmap, overlay_alpha)
        hcols = st.columns(2)
        with hcols[0]:
            st.image(heat_255, caption="Heatmap", use_column_width=True)
        with hcols[1]:
            st.image(overlay_rgb, caption="Overlay", use_column_width=True)
    except Exception as e:
        st.warning(f"Grad-CAM failed: {e}")

    st.markdown("### Raw Probabilities")
    prob_table = {"label": use_labels, "probability": [float(p) for p in probs]}
    st.dataframe(prob_table, use_container_width=True)

    with st.expander("Class descriptions"):
        desc_map = _load_class_descriptions()
        if desc_map:
            for lbl in use_labels:
                d = desc_map.get(lbl.lower())
                if d:
                    st.markdown(f"**{lbl}** — {d}")
                else:
                    st.markdown(f"**{lbl}** — (no description)")

    st.caption(f"Model version: {os.path.basename(config.MODEL_PATH)} | Classes: {len(probs)}")
else:
    st.info("Upload an image to begin.")

st.markdown("---")
st.caption("This tool is for research/education only. Not a diagnostic device.")
