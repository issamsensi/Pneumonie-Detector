# Pneumonia Detection

A **Streamlit** web app that detects pneumonia from chest X-ray images using a **TensorFlow/Keras U-Net** model. The U-Net segmentation mask is overlaid on the original X-ray with bounding boxes and confidence scores.

**Dataset:** Trained on the [RSNA Pneumonia Processed Dataset](https://www.kaggle.com/datasets/iamtapendu/rsna-pneumonia-processed-dataset) (~27K chest X-rays with segmentation masks).

**Status:** Experimental — for research/demonstration only; not for clinical use.

---

**Overview**

- **Purpose:** Upload a chest X-ray (JPG/PNG) and receive a verdict, confidence score, region count, side-by-side original + heatmap overlay, and an exportable PDF report.
- **Model:** A U-Net trained on chest X-rays (`model/model_unet_full.keras`). Inference wraps the full model with tiled conditional variables so only the image tensor is needed at prediction time.
- **Visualisation:** The segmentation mask is rendered as a jet heatmap over a blue-background X-ray with bounding boxes and per-region confidence labels.
- **Framework:** Migrated from Flask to Streamlit for simplified deployment and Hugging Face Spaces compatibility.

**Quick Demo (local)**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
# Opens http://localhost:8501
```

**Deploy on Hugging Face Spaces**

1. Create a new Space at https://huggingface.co/spaces
2. Choose **Streamlit** as the SDK
3. Push this repository (or clone + `git push`)
4. The Space auto-detects `app.py` as the entry point

---

**Project structure**

```
PneumonieDetector/
├── app.py                  # Streamlit application (entry point)
├── requirements.txt        # Python dependencies
├── utils/
│   ├── __init__.py
│   └── inference.py        # Model loading, overlay, PDF generation
├── static/
│   ├── css/style.css       # Dark-theme CSS (reused from Flask version)
│   └── images/confusion_matrix.png
├── model/                  # Trained model files (gitignored)
│   └── model_unet_full.keras
├── RAPPORT.md              # Project report (French)
├── PIPELINE.md             # Pipeline documentation
└── README.md
```

---

**Inference internals**

- `utils/inference.py` handles image preprocessing, model loading (with `@st.cache_resource`), prediction, mask overlay, and PDF generation.
- The model is a U-Net with an auxiliary classifier branch. Inference strips the conditional variable input so the caller only supplies the image.
- The overlay pipeline: resize mask → jet heatmap → blend with blue-background X-ray → bounding boxes around connected components with confidence labels.

---

**Performance notes**

- The model is loaded once via Streamlit's `@st.cache_resource` and reused across all user sessions.
- Inference runs on CPU (no GPU required). Typical inference time is 1–3 seconds.
- Input validation enforces a 10 MB upload limit and JPG/JPEG/PNG only.

---

**Security & Privacy**

- This is a demonstration tool. Do not use for clinical decisions.
- Follow privacy/regulatory requirements when processing medical images.
- All processing is done in-browser session; images are not stored server-side.
