from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf


def _auto_find_last_conv_layer(model: tf.keras.Model) -> Optional[str]:
    # Prefer layers with 4D output and 'conv' in name
    for layer in reversed(model.layers):
        try:
            out_shape = layer.output.shape
            if len(out_shape) == 4:
                if hasattr(layer, 'name') and ('conv' in layer.name.lower() or 'mix' in layer.name.lower()):
                    return layer.name
        except Exception:
            continue
    # Fallback: any 4D output layer
    for layer in reversed(model.layers):
        try:
            out_shape = layer.output.shape
            if len(out_shape) == 4:
                return layer.name
        except Exception:
            continue
    return None


def compute_gradcam(
    model: tf.keras.Model,
    img_tensor: np.ndarray,
    class_index: Optional[int] = None,
    last_conv_layer_name: Optional[str] = None,
) -> Tuple[np.ndarray, int]:
    """
    Compute Grad-CAM heatmap for the predicted class (or provided class_index).

    Args:
        model: Keras model
        img_tensor: numpy array with shape (1, H, W, C), float32
        class_index: if None, uses argmax of model prediction
        last_conv_layer_name: if None, auto-detects a suitable conv layer

    Returns:
        heatmap: (H, W) float32 in [0,1]
        class_index: the class used for gradients
    """
    assert img_tensor.ndim == 4 and img_tensor.shape[0] == 1, "img_tensor must be (1,H,W,C)"

    preds = model(img_tensor, training=False)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    preds_np = preds.numpy()

    if class_index is None:
        class_index = int(np.argmax(preds_np[0]))

    # Find last conv layer
    if last_conv_layer_name is None:
        last_conv_layer_name = _auto_find_last_conv_layer(model)
        if last_conv_layer_name is None:
            raise RuntimeError("Could not find a suitable conv layer for Grad-CAM")

    last_conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        # Handle multi-output models where conv_outputs or predictions may be lists/tuples
        if isinstance(conv_outputs, (list, tuple)):
            conv_outputs = conv_outputs[0]
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        # predictions should be shape (1, num_classes, ...) - reduce if needed
        while tf.rank(predictions) > 2:
            # Squeeze trailing singleton dims except batch & classes
            shape = predictions.shape
            if shape[-1] == 1:
                predictions = tf.squeeze(predictions, axis=-1)
            else:
                break
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)

    # pooled grads across H,W
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

    # weight conv outputs by grads
    conv_outputs = conv_outputs[0]  # (Hc, Wc, C)
    pooled_grads = pooled_grads[0]
    conv_outputs_weighted = conv_outputs * pooled_grads
    heatmap = tf.reduce_sum(conv_outputs_weighted, axis=-1)

    # relu and normalize
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy().astype(np.float32)

    # resize to input image size
    H, W = img_tensor.shape[1], img_tensor.shape[2]
    heatmap = tf.image.resize(heatmap[..., None], (H, W), method="bilinear").numpy()[..., 0]
    heatmap = np.clip(heatmap, 0.0, 1.0)
    return heatmap, class_index
