from __future__ import annotations

from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from flask import Flask, jsonify, render_template, request, send_file

from utils.inference import analyze_uploaded_image, create_report_pdf

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "pneumonia_best.pt"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["JSON_SORT_KEYS"] = False


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.post("/analyze")
def analyze() -> tuple[dict, int]:
    uploaded_file = request.files.get("image")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "Please upload a JPG, JPEG, or PNG image."}), 400

    try:
        analysis = analyze_uploaded_image(uploaded_file, MODEL_PATH)
        return jsonify(analysis), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Analysis failed")
        return jsonify({"error": f"Analysis failed: {exc}"}), 500


@app.get("/download/report/<patient_id>")
def download_report(patient_id: str):
    verdict = request.args.get("verdict", "normal")
    confidence = request.args.get("confidence", "0")
    regions = request.args.get("regions", "0")
    analysis_date = request.args.get("date") or datetime.now().strftime("%Y-%m-%d %H:%M")

    pdf_buffer = create_report_pdf(
        patient_id=patient_id,
        verdict=verdict,
        confidence=confidence,
        regions=regions,
        analysis_date=analysis_date,
    )

    filename = f"pneumonia_report_{quote(patient_id or 'anonymous')}.pdf"
    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
