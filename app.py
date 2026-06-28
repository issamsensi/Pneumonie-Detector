"""Pneumonia Detection — Streamlit Application."""

from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

from utils.inference import (
    load_model as _load_model,
    analyze_uploaded_image as _analyze,
    create_report_pdf,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model_unet_full.keras"
STATIC_DIR = BASE_DIR / "static"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class _UploadWrapper:
    __slots__ = ("filename", "stream")
    def __init__(self, filename: str, stream: io.BytesIO) -> None:
        self.filename = filename
        self.stream = stream


st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="\U0001fa81",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS — dark theme, lightweight
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* Dark background */
.stApp { background: #0a0e1a !important; }
.stApp > [data-testid="stAppViewContainer"] { background: transparent !important; }
#MainMenu, footer, .stApp header, header.stAppHeader { display: none !important; }

/* Text colors */
h1, h2, h3, h4, h5, h6 { color: #e5e7eb !important; }
p, li, .stMarkdown, .stWrite { color: #d1d5db; }
.stCaption, .st-emotion-caption { color: #94a3b8; }

/* File uploader dropzone */
[data-testid="stFileUploaderDropzone"] {
    min-height: 320px !important;
    border-radius: 22px !important;
    border: 1.5px dashed rgba(148,163,184,0.28) !important;
    background: linear-gradient(180deg, rgba(15,23,42,0.9), rgba(17,24,39,0.95)) !important;
}

/* Buttons */
.stButton > button {
    border-radius: 14px !important;
    font-weight: 700 !important;
    padding: 14px 18px !important;
    background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
    color: white !important;
    border: 0 !important;
}
.stDownloadButton > button {
    border-radius: 14px !important;
    font-weight: 700 !important;
    padding: 8px 16px !important;
    background: rgba(59,130,246,0.12) !important;
    color: #dbeafe !important;
    border: 1px solid rgba(59,130,246,0.24) !important;
}

/* Metrics */
[data-testid="stMetricValue"] { color: #e5e7eb !important; }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }

/* Text input / textarea */
.stTextInput > div > input, .stTextArea > div > textarea {
    border-radius: 14px !important;
    border: 1px solid rgba(148,163,184,0.2) !important;
    background: rgba(2,6,23,0.35) !important;
    color: #e5e7eb !important;
}

/* Image borders */
[data-testid="stImage"] {
    background: rgba(2,6,23,0.35);
    border: 1px solid rgba(148,163,184,0.14);
    border-radius: 18px;
    overflow: hidden;
    padding: 12px;
}

/* Divider */
hr { border-color: rgba(148,163,184,0.18); }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model check — show helpful message if missing
# ---------------------------------------------------------------------------

if not MODEL_PATH.exists():
    st.error(
        "**Model file not found.** The trained U-Net model is required.\n\n"
        f"Expected at: `{MODEL_PATH}`\n\n"
        "If running on Hugging Face Spaces, upload the model file "
        "via the **Files** tab of your Space.\n\n"
        "*The app cannot function until the model is uploaded.*"
    )
    st.stop()


# ---------------------------------------------------------------------------
# Model (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_cached_model() -> None:
    _load_model(MODEL_PATH)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

for key, default in (
    ("result", None),
    ("analyzed", False),
    ("prev_file", ""),
):
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b64encode(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ===================================================================
# UI
# ===================================================================

st.title("Pneumonia Detection")
st.write("Upload a chest X-ray image for analysis with segmentation heatmap overlay.")
st.caption("Research purpose only \u2014 does not replace professional medical diagnosis.")

uploaded_file = st.file_uploader(
    "Choose a chest X-ray (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:

    if st.session_state.prev_file != uploaded_file.name:
        st.session_state.prev_file = uploaded_file.name
        st.session_state.analyzed = False
        st.session_state.result = None

    file_bytes: bytes = uploaded_file.read()

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        st.error("File exceeds the 10 MB limit.")
        st.stop()

    size_mb = len(file_bytes) / (1024 * 1024)
    st.write(f"**{uploaded_file.name}** \u2014 {size_mb:.2f} MB")

    preview_img = Image.open(io.BytesIO(file_bytes))
    st.image(preview_img, use_container_width=True)

    if st.button("Analyze", type="primary", use_container_width=True):
        with st.spinner("Analyzing X-ray and generating segmentation heatmap\u2026"):
            try:
                get_cached_model()
                wrapper = _UploadWrapper(
                    uploaded_file.name,
                    io.BytesIO(file_bytes),
                )
                result = _analyze(wrapper, MODEL_PATH)
                st.session_state.result = result
                st.session_state.analyzed = True
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                st.session_state.analyzed = False

if st.session_state.analyzed and st.session_state.result is not None:
    r: dict = st.session_state.result
    is_pneu = r["verdict"] == "pneumonia"
    verdict_text = "PNEUMONIA DETECTED" if is_pneu else "NORMAL"
    verdict_detail = (
        "The model identified suspicious regions that warrant clinical confirmation."
        if is_pneu else
        "The model did not identify suspicious regions in this study."
    )

    st.divider()

    st.subheader("Verdict")
    st.markdown(f"### {verdict_text}")
    st.write(verdict_detail)

    st.subheader("Results")
    col_orig, col_over = st.columns(2)
    with col_orig:
        st.image(r["original_b64"], caption="Original X-ray", use_container_width=True)
    with col_over:
        st.image(r["overlay_b64"], caption="Segmentation heatmap", use_container_width=True)

    col_conf, col_reg, col_fn = st.columns(3)
    with col_conf:
        st.metric("Confidence", f"{round(r['confidence'] * 100)}%")
    with col_reg:
        st.metric("Affected regions", str(r["regions"]))
    with col_fn:
        st.metric("File", st.session_state.prev_file)

    cm_path = STATIC_DIR / "images" / "confusion_matrix.png"
    cm_b64 = _b64encode(cm_path) if cm_path.exists() else ""
    cm_src = f"data:image/png;base64,{cm_b64}" if cm_b64 else ""
    if cm_src:
        st.subheader("Model Performance")
        st.image(cm_src, caption="Confusion Matrix (validation set)", use_container_width=False)

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1: st.metric("Accuracy", "0.93")
        with col2: st.metric("Precision", "0.93")
        with col3: st.metric("Recall", "0.89")
        with col4: st.metric("F1-score", "0.91")
        with col5: st.metric("IoU", "0.71")
        with col6: st.metric("AUROC", "0.98")

    st.subheader("Report")

    patient_id = st.text_input(
        "Patient ID (optional)",
        placeholder="e.g. PAT-12345",
        max_chars=80,
    )

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        ov_bytes = base64.b64decode(r["overlay_b64"].split(",", 1)[1])
        st.download_button(
            "Save result as PNG",
            data=ov_bytes,
            file_name=f"pneumonia_overlay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
            use_container_width=True,
        )
    with col_dl2:
        pid = (patient_id or "anonymous").strip() or "anonymous"
        pdf_buf = create_report_pdf(
            patient_id=pid,
            verdict=r["verdict"],
            confidence=str(r["confidence"]),
            regions=str(r["regions"]),
            analysis_date=r["analysis_date"],
        )
        st.download_button(
            "Download report as PDF",
            data=pdf_buf,
            file_name=f"pneumonia_report_{pid}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.text_area(
        "Analysis report",
        value=r["report_text"],
        height=220,
        disabled=True,
    )
