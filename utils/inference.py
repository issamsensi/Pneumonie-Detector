from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from ultralytics import YOLO

from utils.gradcam import YOLOGradCAM, make_blue_tinted_overlay

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MODEL_CACHE: dict[str, YOLO] = {}


def load_model(model_path: Path) -> YOLO:
    model_key = str(model_path.resolve())
    if model_key not in MODEL_CACHE:
        MODEL_CACHE[model_key] = YOLO(model_key)
    return MODEL_CACHE[model_key]


def validate_image_file(filename: str) -> None:
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Only JPG, JPEG, and PNG files are supported.")


def image_to_base64(image: np.ndarray, color_space: str = "rgb") -> str:
    pil_image = Image.fromarray(image if color_space == "rgb" else cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def pil_image_to_tensor(image: Image.Image, size: int = 640) -> torch.Tensor:
    resized = image.resize((size, size))
    array = np.asarray(resized).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor


def extract_boxes(result) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    if result.boxes is None:
        return boxes

    for box in result.boxes:
        confidence = float(box.conf.item())
        xyxy = box.xyxy[0].tolist()
        boxes.append({"xyxy": xyxy, "confidence": confidence})
    return boxes


def analyze_uploaded_image(uploaded_file, model_path: Path) -> dict[str, Any]:
    validate_image_file(uploaded_file.filename)

    pil_image = Image.open(uploaded_file.stream)
    pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
    original_rgb = np.asarray(pil_image)

    model = load_model(model_path)
    detection_results = model.predict(source=pil_image, imgsz=640, conf=0.25, device="cpu", verbose=False)[0]
    boxes = extract_boxes(detection_results)

    confidence = max((box["confidence"] for box in boxes), default=0.0)
    regions = len(boxes)
    verdict = "pneumonia" if regions > 0 else "normal"
    verdict_confidence = confidence if verdict == "pneumonia" else max(0.0, min(0.99, 1.0 - confidence if confidence > 0 else 0.99))

    input_tensor = pil_image_to_tensor(pil_image)
    with torch.enable_grad():
        input_tensor.requires_grad_(True)
        with YOLOGradCAM(model) as gradcam:
            heatmap = gradcam.build_heatmap(input_tensor)

    overlay_bgr = make_blue_tinted_overlay(original_rgb, heatmap, boxes)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    original_b64 = image_to_base64(original_rgb, color_space="rgb")
    overlay_b64 = image_to_base64(overlay_rgb, color_space="rgb")

    return {
        "verdict": verdict,
        "confidence": round(float(verdict_confidence), 4),
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
