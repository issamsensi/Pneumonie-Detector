from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MODEL_CACHE: dict[str, tf.keras.Model] = {}
IMAGE_SIZE = 224
MASK_THRESHOLD = 0.5
DEFAULT_VARS = tf.constant([[0.5, 0.0, 0.0]], dtype=tf.float32)


def load_model(model_path: Path) -> tf.keras.Model:
    from keras.layers import Lambda

    model_key = str(model_path.resolve())
    if model_key not in MODEL_CACHE:
        full_model = tf.keras.models.load_model(str(model_path), compile=False)
        img_input = full_model.input[0]
        tiled = Lambda(lambda x: tf.tile(DEFAULT_VARS, [tf.shape(x)[0], 1]), name="tile_vars")(img_input)
        outputs = full_model([img_input, tiled])
        inference_model = tf.keras.Model(inputs=img_input, outputs=outputs, name="unet_inference")
        MODEL_CACHE[model_key] = inference_model
    return MODEL_CACHE[model_key]


def validate_image_file(filename: str) -> None:
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Only JPG, JPEG, and PNG files are supported.")


def image_to_base64(image: np.ndarray) -> str:
    pil_image = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    resized = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE))
    array = np.asarray(resized).astype(np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def count_regions(mask: np.ndarray, threshold: float = MASK_THRESHOLD) -> int:
    binary = (mask > threshold).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(binary)
    return max(0, num_labels - 1)


def make_mask_overlay(
    original_rgb: np.ndarray, mask: np.ndarray, threshold: float = MASK_THRESHOLD
) -> np.ndarray:
    height, width = original_rgb.shape[:2]
    mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    blue_bg = np.dstack(
        [
            np.clip(gray * 0.4, 0, 255),
            np.clip(gray * 0.5, 0, 255),
            np.clip(gray * 0.9, 0, 255),
        ]
    ).astype(np.uint8)
    blue_bg_bgr = cv2.cvtColor(blue_bg, cv2.COLOR_RGB2BGR)

    heatmap_uint8 = np.uint8(np.clip(mask_resized * 255.0, 0, 255))
    jet_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay_bgr = cv2.addWeighted(blue_bg_bgr, 0.35, jet_heatmap, 0.65, 0)

    binary_mask = (mask_resized > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        confidence = float(np.mean(mask_resized[y : y + h, x : x + w]))
        label = f"{confidence * 100:.1f}%"

        cv2.rectangle(overlay_bgr, (x, y), (x + w, y + h), (255, 255, 255), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y1 = max(0, y - th - 8)
        label_y2 = max(0, y)
        label_x2 = min(width - 1, x + tw + 12)
        cv2.rectangle(overlay_bgr, (x, label_y1), (label_x2, label_y2), (15, 18, 30), -1)
        cv2.putText(
            overlay_bgr,
            label,
            (x + 6, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return overlay_bgr


def analyze_uploaded_image(uploaded_file, model_path: Path) -> dict[str, Any]:
    validate_image_file(uploaded_file.filename)

    pil_image = Image.open(uploaded_file.stream)
    pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
    original_rgb = np.asarray(pil_image)

    input_tensor = preprocess_image(pil_image)

    model = load_model(model_path)
    mask_prob, class_prob = model.predict(input_tensor, verbose=0)
    mask = mask_prob[0, :, :, 0]
    raw_confidence = float(class_prob[0, 0])
    verdict = "pneumonia" if raw_confidence > 0.5 else "normal"

    if verdict == "pneumonia":
        verdict_confidence = raw_confidence
    else:
        verdict_confidence = max(0.0, min(0.99, 1.0 - raw_confidence if raw_confidence > 0 else 0.99))

    regions = count_regions(mask)

    overlay_bgr = make_mask_overlay(original_rgb, mask)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    original_b64 = image_to_base64(original_rgb)
    overlay_b64 = image_to_base64(overlay_rgb)

    return {
        "verdict": verdict,
        "confidence": round(verdict_confidence, 4),
        "regions": regions,
        "original_b64": f"data:image/png;base64,{original_b64}",
        "overlay_b64": f"data:image/png;base64,{overlay_b64}",
        "analysis_date": analysis_date,
        "report_text": build_report_text(verdict, confidence=verdict_confidence, regions=regions, analysis_date=analysis_date),
    }


def build_report_text(verdict: str, confidence: float, regions: int, analysis_date: str) -> str:
    verdict_label = "PNEUMONIA DETECTED" if verdict == "pneumonia" else "NORMAL"
    return (
        f"Analysis Result: {verdict_label}\n"
        f"Regions detected: {regions}\n"
        f"Confidence: {confidence * 100:.1f}%\n"
        "Recommendation: Consult a radiologist for confirmation\n"
        f"Date: {analysis_date}"
    )


def create_report_pdf(
    patient_id: str,
    verdict: str,
    confidence: str,
    regions: str,
    analysis_date: str,
):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontSize=18, leading=22, textColor=colors.HexColor("#0f172a")))
    styles.add(ParagraphStyle(name="ReportBody", parent=styles["BodyText"], fontSize=11, leading=15, textColor=colors.HexColor("#111827")))

    verdict_label = "PNEUMONIA DETECTED" if verdict == "pneumonia" else "NORMAL"
    story = [
        Paragraph("Pneumonia Detection Report", styles["ReportTitle"]),
        Spacer(1, 0.2 * inch),
        Paragraph(f"Patient ID: {patient_id or 'Anonymous'}", styles["ReportBody"]),
        Paragraph(f"Analysis Result: {verdict_label}", styles["ReportBody"]),
        Paragraph(f"Regions detected: {regions}", styles["ReportBody"]),
        Paragraph(f"Confidence: {float(confidence) * 100:.1f}%", styles["ReportBody"]),
        Paragraph("Recommendation: Consult a radiologist for confirmation", styles["ReportBody"]),
        Paragraph(f"Date: {analysis_date}", styles["ReportBody"]),
        Spacer(1, 0.3 * inch),
    ]

    data = [["Disclaimer", "This tool is for research purposes only and does not replace professional medical diagnosis"]]
    table = Table(data, colWidths=[1.2 * inch, 5.3 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e0f2fe")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0f172a")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("LEADING", (0, 0), (-1, -1), 13),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#3b82f6")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#93c5fd")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(table)

    document.build(story)
    buffer.seek(0)
    return buffer
