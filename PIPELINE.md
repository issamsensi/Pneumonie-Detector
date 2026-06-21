# Pipeline — Pneumonia Detection U-Net

## Overview

```
RSNA Dataset → Preprocessing → U-Net Training → Inference Model → Flask Web App
```

The project trains a residual U-Net on chest X-rays to simultaneously produce a **segmentation mask** (highlighting affected areas) and a **classification verdict** (normal vs pneumonia). The trained model is served through a Flask web app with Grad-CAM-style overlays and PDF report generation.

---

## 1. Dataset

### Source
**RSNA Pneumonia Detection Challenge** — chest X-rays with bounding-box annotations for lung opacities. The **RSNA Pneumonia Processed Dataset** (on Kaggle) converts those bounding boxes into pixel-level segmentation masks.

### Structure
```
/kaggle/input/datasets/<user>/rsna-pneumonia-processed-dataset/
├── stage2_train_metadata.csv    # Patient metadata + labels
├── stage2_test_metadata.csv
├── Training/
│   ├── Images/                  # 26,684 chest X-rays (PNG)
│   └── Masks/                   # 26,684 corresponding masks (PNG)
└── Test/
    ├── Images/
    └── Masks/
```

### Metadata columns
| Column | Description |
|--------|-------------|
| `patientId` | Unique ID, links to `{id}.png` in Images/ and Masks/ |
| `age` | Patient age (capped at 90, scaled to [0,1]) |
| `sex` | Male (0) / Female (1) |
| `position` | AP (0) / PA (1) |
| `class` | `Lung Opacity` or `Normal` |
| `Target` | 1 = pneumonia, 0 = normal |

### Class distribution
The dataset is imbalanced: more normal cases than pneumonia. Class weights are computed to compensate.

---

## 2. Preprocessing

### Tabular data
- Encode `sex`: M→0, F→1
- Encode `position`: AP→0, PA→1
- Cap `age` at 90, then MinMaxScaler to [0,1]
- Split: 72% train / 8% val / 10% test (stratified by Target)

### Image data (tf.data pipeline)
```
Read PNG → decode → resize to 224×224 → normalize to [0,1]
Read mask PNG → decode → resize → binarize (>0 → 1.0)
```
- Training: brightness/contrast augmentation, shuffle, repeat, batch of 64
- Validation/Test: no augmentation, no shuffle, batch of 64
- `prefetch(AUTOTUNE)` for GPU throughput

### Challenge: path resolution
Kaggle mounts datasets at unpredictable paths (e.g., `/kaggle/input/datasets/<user>/rsna-pneumonia-processed-dataset/`). The notebook auto-detects the correct root by walking the tree to find `stage2_train_metadata.csv`.

---

## 3. Model Architecture

### Residual U-Net
```
Input: 224×224×3 image  +  3 tabular features (age, sex, position)
```

```
Encoder (depth 4):
  Conv2D 32 → Conv2D 32 → MaxPool → ResidualBlock(32)
  Conv2D 64 → Conv2D 64 → MaxPool → ResidualBlock(64)
  Conv2D 128 → Conv2D 128 → MaxPool → ResidualBlock(128)
  Conv2D 256 → Conv2D 256 → MaxPool → ResidualBlock(256)

Bottleneck:
  Conv2D 512 → Conv2D 512 → Dropout(0.3)

Decoder (depth 4):
  Conv2DTranspose 256 → Concat[skip] → ResidualBlock(512)
  Conv2DTranspose 128 → Concat[skip] → ResidualBlock(256)
  Conv2DTranspose 64  → Concat[skip] → ResidualBlock(128)
  Conv2DTranspose 32  → Concat[skip] → ResidualBlock(64)

Tabular fusion:
  Dense(64) → BN → ReLU → Reshape(1×1×64) → UpSampling 224×224
  → Concat with decoder output

Outputs:
  • mask_output:      Conv2D(1, sigmoid) → 224×224 segmentation mask
  • target_output:    GAP → Dense(64) → Dense(1, sigmoid) → binary classification
```

### Key design choices
- **Residual blocks** in encoder/decoder (skip connections + BN + LeakyReLU) for gradient flow
- **LeakyReLU(0.1)** instead of ReLU to avoid dead neurons
- **IoU loss** for mask (better for segmentation than BCE)
- **Weighted BCE** for classification (handles class imbalance)
- **Tabular fusion** via upsampled dense features concatenated at the decoder end

### Parameters
- ~2-5M trainable params (compact vs YOLOv8's ~11M)
- Input size 224×224 (vs YOLO's 640×640) → 8× fewer pixels → faster epochs

---

## 4. Training

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=1e-3) |
| Batch size | 64 |
| Epochs | 20 (with early stopping patience 5) |
| Loss weights | mask=0.7, target=0.3 |
| LR schedule | ReduceLROnPlateau (factor 0.2, patience 2) |
| Class weights | Balanced (computed from training set) |

### Hardware
- **Kaggle T4 GPU** (15 GB VRAM, single GPU)
- Epoch time: ~X minutes (batch 64, 27K images)

### Loss functions
```
IoU Loss (mask):
  L_mask = 1 - (intersection + ε) / (union + ε)

Weighted BCE (classification):
  L_cls = w * BCE(y_true, y_pred)
  where w = class_weight[1] for pneumonia, class_weight[0] for normal
```

### Results
```
Mask IoU:          ~0.71
Classification:
  Accuracy:        ~0.93
  AUROC:           ~0.98
  Precision:       ~0.93 (pneumonia)
  Recall:          ~0.89 (pneumonia)
  F1 Score:        ~0.91 (pneumonia)
```

### Learning curves
- Loss converges within 8-12 epochs
- Early stopping activates around epoch 10-15
- No significant overfitting (val/train curves track closely)

---

## 5. Model Export

### Two versions saved

**1. Full model** (`model_unet_full.keras`)
- Two inputs: image + tabular
- Two outputs: mask + classification
- Used as source for inference wrapper

**2. Inference model** (`model_unet_inference.keras`)
- Single input: image only
- Tabular inputs baked-in with training-set mean values
- Created by wrapping the full model with a `Lambda` layer that tiles the mean tabular values
- Ready for deployment (but requires `safe_mode=False` to deserialize the Lambda)

### Kaggle → Local transfer
- Files are in `/kaggle/working/` on Kaggle
- Download via Data tab → `/kaggle/working/` → click on file
- Place in `model/` directory of the web app

---

## 6. Web Application

### Stack
- **Backend:** Flask (Python)
- **Model:** TensorFlow/Keras (U-Net)
- **Frontend:** HTML + CSS + vanilla JS (dark theme, responsive)
- **PDF:** ReportLab

### Inference flow
```
User uploads X-ray → Flask receives file → 
  utils/inference.analyze_uploaded_image():
    1. Validate file type (JPG/JPEG/PNG, <10MB)
    2. Open with PIL, strip EXIF orientation, convert to RGB
    3. Resize to 224×224, normalize to [0,1]
    4. Load U-Net model (cached after first load)
    5. Run predict → get mask (224×224) + class score
    6. Determine verdict: class_score > 0.5 → pneumonia
    7. Compute confidence (inverted for "normal" verdict)
    8. Count connected components in mask → "regions"
    9. Generate overlay: blue-tinted background + JET heatmap + bounding boxes
    10. Build report text + optional PDF
    11. Return JSON with base64-encoded images + metadata
```

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/analyze` | POST | Upload image, get analysis JSON |
| `/download/report/<patient_id>` | GET | Download PDF report |

### Response format
```json
{
  "verdict": "pneumonia" | "normal",
  "confidence": 0.8743,
  "regions": 3,
  "original_b64": "data:image/png;base64,...",
  "overlay_b64": "data:image/png;base64,...",
  "analysis_date": "2026-06-21 20:32",
  "report_text": "Analysis Result: PNEUMONIA DETECTED\n..."
}
```

### Visualisation
The segmentation mask from U-Net serves directly as the heatmap (no Grad-CAM needed):
1. Original image → grayscale → blue-tinted background
2. Mask → JET colormap → overlaid on background
3. Connected components → bounding boxes with confidence %

### PDF Report
Generated server-side with ReportLab, includes:
- Patient ID
- Verdict + confidence
- Regions detected
- Disclaimer table

---

## 7. From YOLO to U-Net — Why the Switch

| Aspect | YOLOv8 (old) | U-Net (new) |
|--------|-------------|-------------|
| Task | Object detection | Segmentation + classification |
| Output | Bounding boxes | Pixel mask + class |
| Parameters | ~11M | ~2-5M |
| Input size | 640×640 | 224×224 |
| Epoch time | Slow (large model, large input) | Fast (compact, 8× fewer pixels) |
| "Normal" confidence | Heuristic (1.0 - max box score) | Learned (dedicated classification head) |
| Heatmap | Grad-CAM (hack, fragile) | Native mask output |
| Framework | PyTorch (ultralytics) | TensorFlow/Keras |

### Problems with YOLO
- High loss, slow convergence
- Long epoch times on modest hardware
- Grad-CAM required separate backward pass
- "Normal" confidence was a heuristic, not learned

### Benefits of U-Net
- Pixel-level precision (see exactly where the model is looking)
- Mask IS the heatmap — no Grad-CAM needed
- Dual output (mask + class) is natural for medical imaging
- Faster training, smaller model, better accuracy

---

## 8. Project Structure

```
PneumonieDetector/
├── app.py                           # Flask application
├── requirements.txt                 # Dependencies
├── PIPELINE.md                      # This file
├── model/
│   ├── model_unet_full.keras        # Full U-Net (image + tabular)
│   ├── model_unet_inference.keras   # Image-only U-Net
│   └── pneumonia_best.pt            # (deprecated) old YOLO model
├── utils/
│   ├── __init__.py
│   ├── inference.py                 # Model loading, inference, overlay, PDF
│   └── gradcam.py                   # (deprecated) replaced by mask overlay
├── templates/
│   └── index.html                   # Web UI
├── static/
│   ├── css/style.css                # Dark theme, responsive layout
│   └── js/app.js                    # Upload, analysis, download handlers
├── pneumonie.ipynb                  # Kaggle training notebook
└── rsna.ipynb                       # (deprecated) old YOLO notebook
```

---

## 9. Running the App

```bash
pip install -r requirements.txt
python app.py
# → http://0.0.0.0:5000
```

### Dependencies
```
Flask>=3.0.0            # Web framework
tensorflow>=2.15.0      # Model inference
opencv-python>=4.9.0    # Image processing
numpy>=1.26.0           # Array operations
Pillow>=10.0.0          # Image I/O
reportlab>=4.2.0        # PDF generation
```

### Training on Kaggle
1. Go to [Kaggle](https://kaggle.com)
2. Upload `pneumonie.ipynb`
3. Add **RSNA Pneumonia Processed Dataset** (via Add Data → search)
4. Set accelerator to **T4 GPU**
5. Run all cells
6. Download `model_unet_inference.keras` from `/kaggle/working/`
7. Place in `model/` directory
8. Run the web app

---

## 10. Limitations & Future Work

### Current limitations
- **Tabular data at inference:** The model expects age/sex/position; the web app uses default values since it only receives an image
- **No tests:** The inference pipeline has no automated tests
- **CPU inference:** Runs on CPU by default (~1-2s per image); GPU would be faster
- **Production readiness:** Debug mode enabled, no WSGI server, no containerization
- **Single dataset:** Trained only on RSNA; generalization to other X-ray sources is unvalidated

### Possible improvements
- Retrain without tabular input (simpler deployment)
- Add confidence calibration (temperature scaling)
- Containerize with Docker
- Add CI/CD pipeline
- Experiment with deeper encoders (EfficientNet backbone)
- Add test-time augmentation for more robust predictions
