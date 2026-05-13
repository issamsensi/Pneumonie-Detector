import io
import json
import os
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

    x = _preprocess_image_bytes(image_bytes)
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

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "pneumonia_probability": round(pneumonia_probability, 4),
    }

