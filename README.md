# Pneumonie Detector

A small Flask web app and inference pipeline to detect pneumonia from chest X-ray images using a PyTorch model.

**Status:** Experimental — use for research or demonstration only; not for clinical use.

---

**Overview**

- **Purpose:** Upload a chest X‑ray (JPG/PNG) via the web UI or API and receive a JSON analysis with a verdict, confidence, and regions of interest. A PDF report can be generated for a patient.
- **Main files:** [app.py](app.py#L1) (Flask app), [utils/inference.py](utils/inference.py#L1) (inference + report generation), and the trained model at `model/pneumonia_best.pt`.

**Quick Demo**

1. Activate a Python environment (the repo contains a `pn/` venv you can use):

```bash
# use the included virtualenv if you want
source pn/bin/activate

# or create a fresh venv
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

4. Open your browser at http://127.0.0.1:5000 to use the web UI.

---

**API / Usage**

- GET / — web UI ([index.html](templates/index.html#L1))
- POST /analyze — submit an image file (`multipart/form-data`, field name `image`)
  - Returns JSON with fields such as `verdict`, `confidence`, `regions`, and `patient_id`.

Example curl (replace image path):

```bash
curl -X POST -F "image=@/path/to/xray.jpg" http://127.0.0.1:5000/analyze
```

- GET /download/report/<patient_id>?verdict=...&confidence=...&regions=... — download a generated PDF report for the given `patient_id`.

---

**Model**

- The model used for inference is included (or expected) at `model/pneumonia_best.pt`.
- If you replace the model file, keep the same path or update `MODEL_PATH` in [app.py](app.py#L1).

**Inference internals**

- Inference and report generation are implemented in `utils/inference.py`. The Flask endpoint calls `analyze_uploaded_image()` and `create_report_pdf()` to run prediction and return results.
- Input validation: `app.py` enforces a 10 MB upload limit and accepts JPG/JPEG/PNG images.

---

**Development & Notes**

- Run the app with debug enabled (as in [app.py](app.py#L1)) for development; disable `debug=True` for production and use a proper WSGI server (Gunicorn/uvicorn behind a reverse proxy).
- Consider containerizing for reproducible deployments.
- Tests: none included — add unit tests around `utils/inference.py` for model loading and image preprocessing.

**Security & Privacy**

- This project is a demonstration. DO NOT use it to make clinical decisions. Be sure to follow privacy/regulatory requirements when processing medical images.

---

**Contributing**

- Fork, add tests or improvements, and submit a pull request.

**License**

- No license file is included. Add an appropriate license (e.g., MIT) if you plan to open-source this project.

**Acknowledgements**

- Built as a research/demo project — credit model/data sources as appropriate when publishing results.
