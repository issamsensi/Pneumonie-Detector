# Pneumonia Detector (Full‑Stack Prototype)

Clean, modern **educational/prototype** web app that predicts **NORMAL** vs **PNEUMONIA** from a chest X‑ray image.

> **Disclaimer**: This tool is for educational purposes only and must not replace professional medical diagnosis.

## Tech stack

- **Frontend**: React + Vite + Tailwind CSS
- **Backend**: FastAPI (Python)
- **Model**: TensorFlow/Keras model `pneumonia_model.keras` (sigmoid output)

## Project structure

```
frontend/
backend/
  main.py
  model/
    pneumonia_model.keras
    class_indices.json
```

## Backend

### Setup

TensorFlow typically supports **Python 3.10–3.12**. If your system Python is newer, create a venv with a supported version.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
cd backend
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

- `GET /health` → health check
- `POST /predict` → multipart upload (`file`) returns:

```json
{
  "label": "PNEUMONIA",
  "confidence": 0.94,
  "pneumonia_probability": 0.94
}
```

## Frontend

### Setup

```bash
cd frontend
npm install
```

### Run

```bash
cd frontend
npm run dev
```

Open the local URL shown by Vite (usually `http://localhost:5173`).

### Configure API base URL (optional)

By default the UI calls `http://localhost:8000`. To override:

```bash
cd frontend
echo "VITE_API_BASE_URL=http://localhost:8000" > .env.local
```

## Notes on predictions

- The model output is **sigmoid** \(0..1\), interpreted as **pneumonia probability**
- If probability **≥ 0.5**, the backend maps to **class index 1**
- `class_indices.json` is expected like:
  - `{"NORMAL": 0, "PNEUMONIA": 1}`

