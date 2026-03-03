# Installation and Setup Guide

> Complete instructions for setting up, running, and troubleshooting the  
> Real-Time Few-Shot Face Recognition System on Windows 11 / macOS / Linux.

---

## 1) System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.10 or 3.11 |
| RAM | 4 GB | 8 GB |
| Webcam | USB / built-in | 720p or higher |
| OS | Windows 10 / Ubuntu 20.04 / macOS 12 | Windows 11 |
| GPU | Not required | Optional — CPU inference is fast |

---

## 2) Clone or Download the Project

```bash
git clone <your-repo-url>
cd FewShotFace
```

Or, if you have a ZIP archive, extract it and open the folder.

---

## 3) Create a Virtual Environment

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` in your terminal prompt.

---

## 4) Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### What gets installed

| Package | Purpose |
|---|---|
| fastapi + uvicorn | Web API server |
| flask | Template routing compatibility |
| opencv-python | Webcam capture and image processing |
| torch | PyTorch for FaceNet backend |
| facenet-pytorch | MTCNN detector + FaceNet embedder |
| onnxruntime | ArcFace ONNX inference |
| insightface | Additional face analysis utilities |
| numpy | Numerical operations |
| scikit-learn | Evaluation helpers |
| PyQt5 | Desktop GUI |
| pillow + jinja2 | Image utilities and template rendering |

> **Note:** The `insightface` wheel in `requirements.txt` is a pre-compiled  
> Windows binary (`insightface-0.7.3-cp312-cp312-win_amd64.whl`).  
> On Linux/macOS install via: `pip install insightface`

---

## 5) Download the ArcFace ONNX Model (optional but recommended)

The project auto-falls back to FaceNet if the ONNX model is missing.  
For best accuracy, download the ArcFace model and place it at:

```
FewShotFace/models/arcface.onnx
```

Common source: [InsightFace model zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo)  
(file: `buffalo_l` pack → extract `w600k_r50.onnx` → rename to `arcface.onnx`)

If the file is absent, the system automatically uses FaceNet (InceptionResnetV1 / vggface2).

---

## 6) Run the Web Dashboard

```bash
python app.py
```

Then open your browser at: **http://localhost:8000**

The dashboard has three cards (left panel):
- **Step 1 — Enrollment**: enter name, set image count, click Capture Faces
- **Step 2 — Training**: select backend, click Generate Embeddings
- **Step 3 — Monitoring**: set similarity metric + threshold, click Start Live Recognition

---

## 7) Run the Desktop GUI

```bash
python gui.py
```

The PyQt5 window opens with the same 3-step workflow in a native desktop layout.

---

## 8) First-Time Full Workflow

```bash
# 1. Enroll at least one person (opens webcam, press nothing — auto-captures)
python register.py --name YourName --num-images 8

# 2. (If you have mobile photos) — normalize them first
python fix_mobile_photos.py --name YourName

# 3. Generate embeddings (builds prototypes from dataset/)
python generate_embeddings.py --backend auto

# 4. Test accuracy before going live
python evaluate_accuracy.py --threshold 0.80 -q

# 5. Start live recognition
python recognize.py --threshold 0.80
```

---

## 9) Fixing the Mobile Photo Domain Gap

If your dataset contains photos taken on a phone (not the webcam), run this before Step 3:

```bash
# Process all identities in-place
python fix_mobile_photos.py

# Preview only (no files written)
python fix_mobile_photos.py --dry-run

# Adjust for very dark or blurry mobile photos
python fix_mobile_photos.py --clip-limit 3.5 --sharpen-strength 1.3

# Save corrected images to a separate folder
python fix_mobile_photos.py --output-dir dataset_normalized/
```

After running, re-generate embeddings:

```bash
python generate_embeddings.py
```

---

## 10) Measuring and Reporting Accuracy

```bash
# Single threshold evaluation with full per-identity table
python evaluate_accuracy.py --threshold 0.70

# Find best threshold automatically
python evaluate_accuracy.py --sweep

# Test specific thresholds
python evaluate_accuracy.py --sweep --sweep-values 0.70 0.75 0.80 0.85

# Force a specific backend
python evaluate_accuracy.py --backend onnx --threshold 0.80

# Quiet mode (summary tables only)
python evaluate_accuracy.py --threshold 0.80 -q
```

The output includes a comparison table vs. VGG-16, ResNet-50, AlexNet, GoogleNet, and MobileNet.

---

## 11) Solving the Family Member False Positive Problem

**Symptom:** Mother (not enrolled) recognized as you with ~80% confidence.

**Root cause:** Shared facial features → similar cosine embedding → crosses threshold.

**Fix — 3 steps:**

```bash
# Step A: Enroll the family member as a separate identity
# (via GUI Step 1 or web dashboard, name = "Mother", capture 8–10 webcam images)

# Step B: Re-generate embeddings (now two distinct prototypes exist)
python generate_embeddings.py

# Step C: Raise the recognition threshold
# Web dashboard: move Strictness slider to 0.80
# GUI: move Strictness slider to 0.80
# CLI:
python recognize.py --threshold 0.80

# Step D: Verify the fix
python evaluate_accuracy.py --threshold 0.80 -q
```

**Why it works:** Once the family member has her own prototype, the model separates the two identities by comparing relative distances to each prototype, rather than the absolute threshold alone.

---

## 12) Recommended Threshold Values

| Threshold | Behaviour | Recommended For |
|---|---|---|
| 0.65–0.70 | Lenient — maximises recall | Solo demo, single known user |
| **0.78–0.82** | **Balanced — best F1 score** | **Most real deployments (recommended)** |
| 0.85–0.90 | Strict — minimises false positives | Security-critical, family members enrolled |

Run `python evaluate_accuracy.py --sweep` to find your personal optimum on your own dataset.

---

## 13) Troubleshooting

| Problem | Fix |
|---|---|
| `Camera not found` or `cv2 error` | Close any other app using the webcam; try `--camera-id 1` |
| `PyQt5 GUI does not open` | Confirm PyQt5 is installed: `pip show PyQt5` |
| `ONNX model not found` | Place `arcface.onnx` in `models/` or it will auto-fallback to FaceNet |
| `No prototypes found` | Run Step 2 (generate embeddings) before starting recognition |
| `Import errors` | Activate virtual environment: `.venv\Scripts\activate` |
| `KMP_DUPLICATE_LIB_OK error` | Already handled in `gui.py`; for other scripts set env var manually |
| `insightface install fails on Linux` | Use `pip install insightface` (not the Windows .whl) |
| `Low recognition accuracy` | Run `fix_mobile_photos.py`, re-train, then use `evaluate_accuracy.py --sweep` |
| `Family member confusion` | See Section 11 above |

---

## 14) Directory Layout After First Run

```text
FewShotFace/
├── dataset/
│   └── Dhaval/            ← enrolled person folder
│       ├── Dhaval_xxx_1.jpg
│       └── ...
├── embeddings/
│   ├── embeddings.npy     ← raw embedding matrix
│   ├── labels.npy         ← per-sample identity labels
│   ├── prototypes.npy     ← mean class vectors (used for matching)
│   └── class_names.npy   ← ordered class label list
└── models/
    └── arcface.onnx       ← optional ONNX model
```

---

## 15) File Reference Quick-Map

| File | Run When |
|---|---|
| `app.py` | Want web interface |
| `gui.py` | Want desktop interface |
| `register.py` | Enroll from CLI |
| `generate_embeddings.py` | Re-train after new enrollments |
| `recognize.py` | Run recognition from CLI |
| `fix_mobile_photos.py` | Dataset has mobile camera photos |
| `evaluate_accuracy.py` | Measure accuracy for project report |

For full function documentation see [COMPONENTS.md](COMPONENTS.md).
