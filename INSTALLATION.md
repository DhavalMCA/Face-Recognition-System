# Installation and Setup Guide

This project now runs in two modes only:
- Desktop GUI (`gui.py`)
- CLI scripts (`register.py`, `generate_embeddings.py`, `recognize.py`)

## 1. System Requirements

- Python: 3.10 to 3.12
- RAM: 8 GB recommended
- Webcam: 720p or higher
- OS: Windows 10/11 (primary), Linux/macOS possible

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
