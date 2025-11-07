from __future__ import annotations

import os
import threading
from typing import Optional, Tuple

import numpy as np

try:
    import keras
    from keras.models import load_model
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Failed to import keras. Make sure 'keras' and backend (e.g., tensorflow) are installed."
    ) from e

from . import config

_model = None
_model_lock = threading.Lock()
_input_size: Optional[Tuple[int, int]] = None
_num_classes: Optional[int] = None


def _infer_input_size(m) -> Tuple[int, int]:
    try:
        ishape = m.input_shape
        # shapes like (None, H, W, C)
        if isinstance(ishape, (list, tuple)):
            if isinstance(ishape[0], (list, tuple)):
                ishape = ishape[0]
        h, w = ishape[1], ishape[2]
        if h is None or w is None:
            raise ValueError("Unknown input H/W")
        return int(w), int(h)
    except Exception:
        return config.IMG_SIZE


def _infer_num_classes(m) -> int:
    try:
        oshape = m.output_shape
        if isinstance(oshape, (list, tuple)):
            if isinstance(oshape[-1], (list, tuple)):
                oshape = oshape[-1]
        n = int(oshape[-1])
        return n
    except Exception:
        # Try a tiny forward on zeros if input size is known
        try:
            w, h = _infer_input_size(m)
            x = np.zeros((1, h, w, 3), dtype="float32")
            y = m(x, training=False)
            n = int(y.shape[-1])
            return n
        except Exception:
            return 1


def get_model():
    global _model, _input_size, _num_classes
    if _model is not None:
        return _model, _input_size, _num_classes

    with _model_lock:
        if _model is not None:
            return _model, _input_size, _num_classes
        try:
            m = load_model(config.MODEL_PATH)
        except Exception as e:
            raise RuntimeError(
                "Could not load model. If this file contains only weights, you'll need the model architecture code to load weights into."
            ) from e
        # Determine sizes
        inferred_w, inferred_h = _infer_input_size(m)
        cfg_w, cfg_h = config.IMG_SIZE
        # Prefer config override if set by user; otherwise use model inferred
        if os.getenv("IMG_SIZE"):
            _input_size = (cfg_w, cfg_h)
        else:
            _input_size = (inferred_w, inferred_h)
        _num_classes = _infer_num_classes(m)
        _model = m
        return _model, _input_size, _num_classes
