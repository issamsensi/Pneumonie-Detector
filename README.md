# Pneumonie Detector

A Flask web app that detects pneumonia from chest X-ray images using a **TensorFlow/Keras U-Net** model. The U-Net segmentation mask is overlaid on the original X-ray with bounding boxes and confidence scores — no Grad-CAM needed.

**Status:** Experimental — for research/demonstration only; not for clinical use.

---

**Overview**

- **Purpose:** Upload a chest X-ray (JPG/PNG) via the web UI or API and receive a JSON analysis with a verdict, confidence, region count, and base64-encoded original + overlay images. A PDF report can be generated for a patient.
- **Model:** A U-Net trained on chest X-rays (`model/model_unet_full.keras`). Inference wraps the full model with tiled conditional variables so only the image tensor is needed at prediction time.
- **Visualisation:** The segmentation mask is rendered as a jet heatmap over a blue-background X-ray with bounding boxes and per-region confidence labels.

**Quick Demo**

1. Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app (development mode):

```bash
python app.py
# App listens on http://0.0.0.0:5000
```

4. Open http://127.0.0.1:5000 in your browser.

---

**API**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| POST | `/analyze` | Submit an image (`multipart/form-data`, field `image`) — returns JSON |
| GET | `/download/report/<patient_id>` | Download PDF report |

**`POST /analyze` response:**

```json
{
  "verdict": "pneumonia",
  "confidence": 0.9876,
  "regions": 3,
  "original_b64": "data:image/png;base64,...",
  "overlay_b64": "data:image/png;base64,...",
  "analysis_date": "2026-06-22 00:05",
  "report_text": "Analysis Result: PNEUMONIA DETECTED\nRegions detected: 3\nConfidence: 98.8%\nRecommendation: Consult a radiologist for confirmation\nDate: 2026-06-22 00:05"
}
```

Example curl:

```bash
curl -X POST -F "image=@xray.jpg" http://127.0.0.1:5000/analyze
```

**`GET /download/report/<patient_id>`** — query params: `verdict`, `confidence`, `regions`, `date`

---

**Inference internals**

- `utils/inference.py` handles image preprocessing, model loading, prediction, mask overlay, and PDF generation.
- The model is a U-Net with an auxiliary classifier branch. Inference strips the conditional variable input so the caller only supplies the image.
- The overlay pipeline: resize mask → generate jet heatmap → blend with a dimmed blue-background X-ray → draw bounding boxes around connected components with confidence labels.
- Input validation enforces a 10 MB upload limit and JPG/JPEG/PNG only.

---

**Development**

- Run with `debug=True` for development; use Gunicorn/uvicorn behind a reverse proxy for production.
- Tests: not included — consider adding unit tests around `utils/inference.py`.

**Security & Privacy**

- This is a demonstration tool. Do not use for clinical decisions. Follow privacy/regulatory requirements when processing medical images.

---

**License**

- No license file included. Add an appropriate license (e.g., MIT) if open-sourcing.
