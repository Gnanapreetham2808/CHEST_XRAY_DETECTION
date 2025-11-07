import os
from typing import List, Optional, Tuple


def _parse_img_size(value: str | None) -> Tuple[int, int]:
    if not value:
        return 224, 224
    try:
        if "," in value:
            w, h = value.split(",")
            return int(w.strip()), int(h.strip())
        # single int -> square
        s = int(value.strip())
        return s, s
    except Exception:
        return 224, 224


MODEL_PATH: str = os.getenv("MODEL_PATH", "ConvMixer_WEIGHTS_BIASES.keras")
IMG_SIZE: Tuple[int, int] = _parse_img_size(os.getenv("IMG_SIZE"))
NORMALIZE: str = os.getenv("NORMALIZE", "none").lower()  # options: none, imagenet
SCORE_THRESHOLD: float = float(os.getenv("SCORE_THRESHOLD", "0.0"))

_labels = os.getenv("CLASS_LABELS")
CLASS_LABELS: Optional[List[str]] = (
    [s.strip() for s in _labels.split(",") if s.strip()] if _labels else None
)

# Whether to force RGB conversion (recommended)
FORCE_RGB: bool = os.getenv("FORCE_RGB", "1") not in {"0", "false", "False"}
