import io
import json
import os
import base64
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.densenet import preprocess_input


MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "pneumonia_model.keras")
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, "class_indices.json")

IMAGE_SIZE: Tuple[int, int] = (224, 224)


app = FastAPI(
    title="Pneumonia Detection API",
    version="1.0.0",
    description="Prototype API for educational use only. Not a medical device.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model: tf.keras.Model | None = None
class_indices: Dict[str, int] | None = None
index_to_label: Dict[int, str] | None = None


def _load_class_indices() -> Dict[str, int]:
    if not os.path.exists(CLASS_INDICES_PATH):
        raise RuntimeError(f"Missing class indices file: {CLASS_INDICES_PATH}")
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise RuntimeError("class_indices.json is empty or invalid.")
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, int):
            raise RuntimeError("class_indices.json must be a mapping of string->int.")
    return data


@app.on_event("startup")
def _startup() -> None:
    global model, class_indices, index_to_label
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Missing model file: {MODEL_PATH}")

    class_indices = _load_class_indices()
    index_to_label = {idx: label for label, idx in class_indices.items()}

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)


def _validate_image_upload(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")


def _preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError as e:
        raise HTTPException(status_code=400, detail="Invalid image file.") from e

    img = img.convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


def _load_rgb_image_array(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError as e:
        raise HTTPException(status_code=400, detail="Invalid image file.") from e

    img = img.convert("RGB").resize(IMAGE_SIZE)
    return np.array(img, dtype=np.float32)


def _preprocess_rgb_array(rgb_array: np.ndarray) -> np.ndarray:
    arr = preprocess_input(rgb_array.copy())
    return np.expand_dims(arr, axis=0)


def _find_last_conv_layer_name(keras_model: tf.keras.Model) -> str:
    for layer in reversed(keras_model.layers):
        try:
            output_shape = layer.output_shape
        except Exception:
            continue
        if isinstance(output_shape, tuple) and len(output_shape) == 4:
            return layer.name
    raise RuntimeError("No 2D convolutional feature layer found for Grad-CAM.")


def _get_connected_output_tensor(layer: tf.keras.layers.Layer) -> tf.Tensor:
    node_list = getattr(layer, "_inbound_nodes", [])
    if node_list:
        # For nested models (e.g. transfer-learning backbones), layer.output may
        # point to the inner graph; inbound node outputs are tied to the caller graph.
        output_tensors = node_list[-1].output_tensors
        if isinstance(output_tensors, (list, tuple)):
            if not output_tensors:
                raise RuntimeError(f"Layer '{layer.name}' has no output tensors.")
            return output_tensors[0]
        return output_tensors

    output_tensor = getattr(layer, "output", None)
    if output_tensor is None:
        raise RuntimeError(f"Layer '{layer.name}' does not expose an output tensor.")
    return output_tensor


def _encode_png_base64(image_arr: np.ndarray) -> str:
    image_uint8 = np.clip(image_arr, 0, 255).astype(np.uint8)
    image = Image.fromarray(image_uint8)
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def _describe_activation_region(bbox_norm: Dict[str, float]) -> str:
    cx = (bbox_norm["x_min"] + bbox_norm["x_max"]) / 2.0
    cy = (bbox_norm["y_min"] + bbox_norm["y_max"]) / 2.0

    if cx < 0.33:
        horizontal = "left"
    elif cx > 0.66:
        horizontal = "right"
    else:
        horizontal = "central"

    if cy < 0.33:
        vertical = "upper"
    elif cy > 0.66:
        vertical = "lower"
    else:
        vertical = "mid"

    if horizontal == "central":
        return f"{vertical} central lung region"
    return f"{vertical} {horizontal} lung region"


def _compute_gradcam(
    keras_model: tf.keras.Model,
    input_tensor: np.ndarray,
    rgb_array: np.ndarray,
) -> Dict[str, Any]:
    last_conv_layer_name = _find_last_conv_layer_name(keras_model)
    last_conv_layer = keras_model.get_layer(last_conv_layer_name)
    conv_output_tensor = _get_connected_output_tensor(last_conv_layer)

    model_inputs = keras_model.inputs
    model_output = keras_model.outputs[0]
    grad_model = tf.keras.models.Model(
        inputs=model_inputs,
        outputs=[conv_output_tensor, model_output],
    )

    with tf.GradientTape() as tape:
        grad_inputs: Any = [input_tensor] if len(model_inputs) == 1 else input_tensor
        conv_outputs, predictions = grad_model(grad_inputs)
        target_score = predictions[:, 0]

    gradients = tape.gradient(target_score, conv_outputs)
    if gradients is None:
        raise RuntimeError("Failed to compute Grad-CAM gradients.")

    pooled_grads = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    cam = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    cam = tf.nn.relu(cam)

    cam = cam.numpy()
    cam_min, cam_max = float(np.min(cam)), float(np.max(cam))
    if cam_max <= cam_min:
        cam = np.zeros_like(cam, dtype=np.float32)
    else:
        cam = (cam - cam_min) / (cam_max - cam_min)

    cam_resized = tf.image.resize(cam[..., np.newaxis], IMAGE_SIZE).numpy().squeeze()
    cam_resized = np.clip(cam_resized, 0.0, 1.0)

    heat_uint8 = (cam_resized * 255.0).astype(np.uint8)
    color_overlay = np.stack(
        [
            heat_uint8,
            (heat_uint8 * 0.45).astype(np.uint8),
            np.zeros_like(heat_uint8, dtype=np.uint8),
        ],
        axis=-1,
    ).astype(np.float32)

    alpha = (cam_resized * 0.6)[..., np.newaxis]
    blended = rgb_array * (1.0 - alpha) + color_overlay * alpha

    threshold = max(0.35, float(np.quantile(cam_resized, 0.85)))
    mask = cam_resized >= threshold
    coords = np.argwhere(mask)

    if coords.size == 0:
        peak_y, peak_x = np.unravel_index(np.argmax(cam_resized), cam_resized.shape)
        y_min = y_max = int(peak_y)
        x_min = x_max = int(peak_x)
    else:
        y_min = int(coords[:, 0].min())
        y_max = int(coords[:, 0].max())
        x_min = int(coords[:, 1].min())
        x_max = int(coords[:, 1].max())

    h, w = cam_resized.shape
    bbox_norm = {
        "x_min": round(x_min / max(1, w - 1), 4),
        "y_min": round(y_min / max(1, h - 1), 4),
        "x_max": round(x_max / max(1, w - 1), 4),
        "y_max": round(y_max / max(1, h - 1), 4),
    }

    return {
        "layer": last_conv_layer_name,
        "location_text": _describe_activation_region(bbox_norm),
        "bbox_normalized": bbox_norm,
        "overlay_image_base64": _encode_png_base64(blended),
    }


def _predict(pneumonia_prob: float) -> Tuple[str, float]:
    # Sigmoid output: probability >= 0.5 => class index 1
    predicted_index = 1 if pneumonia_prob >= 0.5 else 0
    label = (index_to_label or {}).get(predicted_index, "PNEUMONIA" if predicted_index == 1 else "NORMAL")
    confidence = pneumonia_prob if predicted_index == 1 else (1.0 - pneumonia_prob)
    return label, float(confidence)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    _validate_image_upload(file)
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Could not read uploaded file.") from e
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    rgb_array = _load_rgb_image_array(image_bytes)
    x = _preprocess_rgb_array(rgb_array)
    try:
        y = model.predict(x, verbose=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Model inference failed.") from e

    try:
        pneumonia_probability = float(np.ravel(y)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail="Unexpected model output.") from e

    pneumonia_probability = max(0.0, min(1.0, pneumonia_probability))
    label, confidence = _predict(pneumonia_probability)

    try:
        gradcam = _compute_gradcam(model, x, rgb_array)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {str(e)}") from e

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "pneumonia_probability": round(pneumonia_probability, 4),
        "gradcam": gradcam,
    }

