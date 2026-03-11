# Installation and Setup Guide

FewShotFace supports three operation modes: Web App (`app.py`), Desktop GUI (`gui.py`), and CLI scripts.

---

## 1. System Requirements

| Requirement | Minimum |
|---|---|
| Python | 3.10 – 3.12 |
| RAM | 8 GB |
| Webcam | 720p or higher |
| OS | Windows 10/11 (primary); Linux/macOS possible |

---

## 2. Create Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Main packages installed:

| Package | Purpose |
|---|---|
| `opencv-python` | Image I/O and preprocessing |
| `numpy` | Numerical operations |
| `torch`, `torchvision` | PyTorch backend for FaceNet |
| `facenet-pytorch`, `mtcnn` | Face detection (MTCNN) and FaceNet embedding |
| `onnxruntime` | ONNX model inference (`models/w600k_r50.onnx`) |
| `insightface` | InsightFace buffalo_l / buffalo_sc backends |
| `deepface` | DeepFace multi-model backend |
| `PyQt5` | Desktop GUI |
| `Flask` | Web server |
| `scikit-learn`, `pillow` | Utilities |

---

## 4. Run Web Application (Recommended)

```powershell
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## 5. Run Desktop GUI

```powershell
python gui.py
```

---

## 6. Full Pipeline via CLI

### 6.1 Register identity

```powershell
python register.py --name "Dhaval" --num-images 12 --camera-id 0
```

### 6.2 Optional photo normalisation

```powershell
python fix_mobile_photos.py
```

### 6.3 Generate embeddings

```powershell
python generate_embeddings.py --backend auto
```

### 6.4 Start recognition

```powershell
python recognize.py --backend auto --threshold 0.80 --camera-id 0
```

### 6.5 Evaluate accuracy

```powershell
# Single threshold
python evaluate_accuracy.py --backend auto --threshold 0.80

# Threshold sweep (recommended)
python evaluate_accuracy.py --backend auto --sweep

# Skip multi-model comparison (faster single-backend run)
python evaluate_accuracy.py --backend auto --no-compare
```

---

## 7. Recommended Defaults

| Setting | Value |
|---|---|
| Images per person | 10–15 |
| Threshold | 0.80 |
| Vote frames | 5–7 |
| Camera ID | 0 |

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| Import errors in VS Code | Select interpreter: `.venv\Scripts\python.exe` |
| Webcam not opening | Close other apps using camera; retry with `--camera-id 1` |
| Frequent *Unknown* detections | Add more images per user; re-run `generate_embeddings.py`; try threshold `0.78` |
| False positives | Raise threshold to `0.82`–`0.86`; enroll lookalike users as separate identities |
| ONNX model missing or wrong size | Run `python download_models.py --list` to verify `models/w600k_r50.onnx` (~174 MB) |
| Evaluation too slow | Already optimised: MTCNN runs once (cached crops), backends run in parallel |

---

## 9. Notes

- `embeddings/prototypes.npy` must exist before recognition or evaluation.
- Rebuild embeddings after adding new users.
- The default ONNX model is `models/w600k_r50.onnx` (iResNet-50 backbone, WebFace600K trained, 512-d output).


## 2. Create Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Main packages installed:
- opencv-python
- numpy
- torch, torchvision
- facenet-pytorch, mtcnn
- onnxruntime, insightface
- PyQt5
- scikit-learn, prettytable, pillow, deepface

## 4. Run Desktop GUI

```powershell
python gui.py
```

## 5. Run Full Pipeline via CLI

### 5.1 Register identity

```powershell
python register.py --name "Dhaval" --num-images 12 --camera-id 0
```

### 5.2 Optional photo normalization

```powershell
python fix_mobile_photos.py
```

### 5.3 Generate embeddings

```powershell
python generate_embeddings.py --backend auto
```

### 5.4 Start recognition

```powershell
python recognize.py --backend auto --threshold 0.80 --camera-id 0
```

### 5.5 Evaluate

```powershell
python evaluate_accuracy.py --backend auto --sweep
python evaluate_models.py
```

## 6. Recommended Defaults

- Images per person: 10 to 15
- Threshold: 0.80
- Vote frames: 5 to 7
- Camera id: 0

## 7. Troubleshooting

- Import errors in VS Code:
  - Select interpreter: `.venv\Scripts\python.exe`
- Webcam not opening:
  - Close other apps using camera
  - Retry with `--camera-id 1`
- Many Unknown detections:
  - Add more images per user
  - Re-run `generate_embeddings.py`
  - Try threshold 0.78
- False positives:
  - Raise threshold to 0.82 to 0.86
  - Enroll lookalike users as separate identities

## 8. Notes

- `embeddings/prototypes.npy` must exist before recognition.
- Rebuild embeddings after adding new users.
- No web dashboard is included in this version.
