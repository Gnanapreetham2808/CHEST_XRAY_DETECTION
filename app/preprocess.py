from __future__ import annotations

from io import BytesIO
from typing import Tuple

import numpy as np
from PIL import Image

try:
    # Optional: use keras utilities if available
    import keras
    from keras.applications.imagenet_utils import preprocess_input as imagenet_preprocess
except Exception:  # pragma: no cover
    keras = None
    imagenet_preprocess = None


def load_image_bytes(data: bytes, target_size: Tuple[int, int], force_rgb: bool = True) -> np.ndarray:
    with Image.open(BytesIO(data)) as img:
        if force_rgb:
            img = img.convert("RGB")
        # target_size is (w, h)
        img = img.resize((target_size[0], target_size[1]), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
        return arr


def preprocess_for_model(arr: np.ndarray, normalize: str = "none") -> np.ndarray:
    # Scale to [0,1]
    x = arr / 255.0
    # Imagenet normalization if requested
    if normalize == "imagenet" and imagenet_preprocess is not None:
        # imagenet_preprocess expects range [0,255] or [-128..] depending on mode; scale back temporarily
        x255 = (x * 255.0).astype(np.float32)
        x255 = imagenet_preprocess(x255, data_format="channels_last")
        # Bring to reasonable scale
        x = x255
    # Add batch dim
    x = np.expand_dims(x, axis=0)
    return x
