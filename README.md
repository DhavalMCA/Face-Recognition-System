# FewShotFace — Real-Time Few-Shot Face Recognition

A production-ready face recognition system that works with as few as 8–15 images per person — no model retraining required.

Three ways to operate:
- **Web Application** (`app.py`) — modern browser UI (recommended)
- **Desktop App** (`gui.py`) — PyQt5 standalone app
- **Command-Line** — full pipeline via individual scripts

---

## 1. What This Project Delivers

| Feature | Detail |
|---|---|
| Few-shot onboarding | Register a new person with 8–15 images |
| Real-time webcam recognition | Stable labels with frame voting |
| Multi-backend comparison | InsightFace, FaceNet, ONNX, DeepFace evaluated live |
| **Single-pass face detection** | MTCNN runs once; all backends share the cached crops |
| **Parallel backend evaluation** | `ThreadPoolExecutor` runs all comparison models concurrently |
| Accuracy reporting | Per-identity table + unified model comparison table |
| Threshold sweep | Sweep 0.60–0.90 and pick the optimal operating point |
| Mobile photo normalization | White-balance / CLAHE correction for mixed datasets |

---

## 2. Recommended Workflow

```
1. Enroll users           register.py  or  GUI Step 1
2. (Optional) normalise   fix_mobile_photos.py
3. Build embeddings        generate_embeddings.py  or  GUI Step 2
4. Live recognition        recognize.py  or  GUI Step 3
5. Evaluate accuracy       evaluate_accuracy.py
```

See `END_TO_END_WORKING.md` for full technical detail at each stage.

---

## 3. Quick Start (Windows)

### 3.1 Create and activate environment

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3.2 Launch Web Application (recommended)

```powershell
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

### 3.3 Launch Desktop GUI

```powershell
python gui.py
```

---

## 4. Operator Guide

### Step 1 — Enroll Person

- Enter name → capture 8–15 images
- Cover front, left tilt, right tilt, near/medium/far distances

### Step 2 — Build Embeddings

```powershell
python generate_embeddings.py --backend auto
```

### Step 3 — Start Recognition

```powershell
python recognize.py --backend auto --threshold 0.80 --camera-id 0
```

Recommended threshold range: `0.78`–`0.82` for balanced behaviour.

---

## 5. Accuracy Evaluation

### Single threshold

```powershell
python evaluate_accuracy.py --backend auto --threshold 0.80
```

### Sweep multiple thresholds

```powershell
python evaluate_accuracy.py --backend auto --sweep
```

### What the output looks like

```
  ╔══════════════════════════════════════════════════════════════════════════╗
  ║  #    Model                  Accuracy  Precision    Recall  Specificity       F1
  ╠══════════════════════════════════════════════════════════════════════════╣
  ║     1  insightface(buffalo_l)   97.50%     97.50%   95.00%       98.75%   96.20%  ◄
  ║     2  FaceNet                  94.17%     95.83%   88.89%       97.22%   92.22%
  ║     3  ONNX (w600k_r50)         92.50%     91.67%   85.00%       96.67%   88.18%
  ╚══════════════════════════════════════════════════════════════════════════╝

  Primary model : insightface(buffalo_l)  │  Accuracy: 97.50%  │  Rank: #1 of 8
```

**Speed optimisations applied:**
- MTCNN face detection runs **once** via `precompute_face_crops()` — result shared across all backends
- All comparison backends run **in parallel** via `ThreadPoolExecutor(max_workers=4)`
- Net result: evaluation of 8 models is roughly as fast as evaluating 2 sequentially

---

## 6. Suggested Default Settings

| Setting | Value |
|---|---|
| Enrollment images / person | 10–15 |
| Recognition threshold | 0.80 |
| Vote frames | 5–7 |
| Camera resolution | 1280×720 |

### Threshold tuning

| Range | Effect |
|---|---|
| 0.82–0.86 | High-security — fewer false positives, may miss some genuine users |
| 0.78–0.82 | Balanced — good accuracy, low false alarms |
| 0.74–0.78 | Lenient — maximises recall, more false positives |

---

## 7. Common Commands

```powershell
# Enroll
python register.py --name "Dhaval" --num-images 12 --camera-id 0

# Normalize mobile photos (optional)
python fix_mobile_photos.py

# Build embeddings
python generate_embeddings.py --backend auto

# Recognize
python recognize.py --backend auto --threshold 0.80 --camera-id 0

# Evaluate (all backends + threshold sweep)
python evaluate_accuracy.py --backend auto --sweep

# List ONNX models in models/ folder
python download_models.py --list
```

---

## 8. Project Structure

```
FewShotFace/
├── app.py                   Flask web server + REST API
├── gui.py                   PyQt5 desktop application
├── register.py              Face enrollment (webcam capture)
├── generate_embeddings.py   Build embeddings and prototypes
├── recognize.py             Live webcam recognition
├── evaluate_accuracy.py     Accuracy evaluation + multi-model comparison
├── fix_mobile_photos.py     Mobile photo normalisation
├── utils.py                 FaceEmbedder, detection helpers, prototypes
├── similarity.py            Cosine/Euclidean similarity + prediction logic
├── templates/index.html     Web UI layout
├── static/                  CSS and JS for web UI
├── dataset/                 Per-identity image folders
├── embeddings/              Generated .npy files and prototypes
├── download_models.py        ONNX model management utility (list, verify, download)
├── models/                  ONNX model files (w600k_r50.onnx used by default)
├── README.md
├── COMPONENTS.md
├── INSTALLATION.md
└── END_TO_END_WORKING.md
```

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| Import errors in VS Code | Set interpreter to `.venv\Scripts\python.exe` |
| Webcam not opening | Close other camera apps; try `--camera-id 1` |
| Frequent *Unknown* labels | Add more images; lower threshold to `0.78`; rebuild embeddings |
| False positives | Raise threshold to `0.82`–`0.86`; enroll lookalike family members |
| Missing prototype error | Run Step 2 (`generate_embeddings.py`) again |
| ONNX model not found | Run `python download_models.py --list` to verify `models/w600k_r50.onnx` exists |
| Evaluation takes too long | Already optimised: detection runs once, backends run in parallel |

---

## 10. Delivery Checklist (Before Demo)

1. Verify all required identities are enrolled
2. Regenerate embeddings fresh
3. Confirm recognition at near and medium distance
4. Keep threshold at `0.80` as baseline; have `0.78` and `0.82` ready
5. Run `evaluate_accuracy.py --sweep` and save report screenshot

---

## 11. License

See `LICENSE`.


## 2. Recommended Client Workflow

Follow this exact sequence for reliable results:

1. Enroll users (`register.py` or GUI Step 1)
2. (Optional) Normalize mobile photos (`fix_mobile_photos.py`)
3. Build embeddings/prototypes (`generate_embeddings.py` or GUI Step 2)
4. Start live recognition (`recognize.py` or GUI Step 3)
5. Run evaluation report (`evaluate_accuracy.py`)

For a detailed technical flow with inputs/outputs at each stage, see `END_TO_END_WORKING.md`.

## 3. Quick Start (Windows)

### 3.1 Create and activate environment

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3.2 Launch Web Application (Recommended)

```powershell
python app.py
```
Then open your browser and navigate to `http://127.0.0.1:5000`

### 3.3 Launch Desktop GUI

```powershell
python gui.py
```

## 4. Operator Guide (Simple)

### Step 1 - Enroll Person

- Enter name
- Capture 8-15 face images
- Include front, left tilt, right tilt, near, medium, far

### Step 2 - Build Embeddings

- Click Build Embeddings (or run CLI command below)
- Wait for completion message

```powershell
python generate_embeddings.py --backend auto
```

### Step 3 - Start Recognition

- Start recognition
- Use threshold around `0.78`-`0.82` for balanced behavior
- If false positives occur, increase threshold

```powershell
python recognize.py --backend auto --threshold 0.80
```

## 5. Optimization Built Into the System

### Accuracy Optimization

- ArcFace/FaceNet embedding backends
- L2-normalized embeddings
- Prototype-based class representation
- Optional outlier-resistant prototype generation

### Stability Optimization

- Multi-frame voting (`FrameVoter`) to avoid unstable labels
- Temporal smoothing (`FaceTracker`) for live stream consistency

### Distance Optimization

- Small-face mode for distant faces
- Enhanced crop + contrast normalization for tiny detections

### Lighting Optimization

- CLAHE enhancement in the embedding pipeline
- Optional photo correction utility for mixed camera datasets

## 6. Suggested Default Settings

- Enrollment images per person: `10-15`
- Recognition threshold: `0.80`
- Vote frames: `5-7`
- Camera resolution for recognition: `1280x720` when possible

### Why Threshold and Vote Frames Matter

- Threshold controls acceptance strictness:
  - Higher threshold (`0.82` to `0.86`) reduces false positives but may label known faces as `Unknown` more often.
  - Lower threshold (`0.76` to `0.80`) recognizes known users more easily but can increase mistaken matches.
- Frame voting (`FrameVoter`) stabilizes labels across consecutive frames:
  - A single frame can be noisy (motion blur, partial face, lighting flicker).
  - Voting over the last `5-7` frames prevents rapid label flipping and gives smoother live output.
- Practical tuning rule:
  - If you see wrong identity matches, increase threshold first.
  - If labels flicker between a name and `Unknown`, increase vote frames slightly.
  - If recognition feels slow to update after a person enters/leaves, reduce vote frames slightly.

## 7. Common Commands

### Register

```powershell
python register.py --name "Dhaval" --num-images 12 --camera-id 0
```

### Optional mobile-photo normalization

```powershell
python fix_mobile_photos.py
```

### Generate embeddings

```powershell
python generate_embeddings.py --backend auto
```

### Start recognition

```powershell
python recognize.py --backend auto --threshold 0.80 --camera-id 0
```

### Evaluate

```powershell
python evaluate_accuracy.py --backend auto --sweep
python evaluate_models.py
```

## 8. Project Structure (Important Files)

```text
FewShotFace/
|- app.py
|- templates/
|- static/
|- gui.py
|- register.py
|- generate_embeddings.py
|- recognize.py
|- evaluate_accuracy.py
|- evaluate_models.py
|- fix_mobile_photos.py
|- download_models.py
|- utils.py
|- similarity.py
|- dataset/
|- embeddings/
|- models/
|- README.md
|- END_TO_END_WORKING.md
```

## 9. Troubleshooting (Client Friendly)

- Import errors in VS Code:
  - Ensure interpreter is `.venv\Scripts\python.exe`
- Webcam not opening:
  - Close other camera apps, then retry with `--camera-id 1`
- Recognition says Unknown often:
  - Add more enrollment images and rebuild embeddings
  - Lower threshold slightly (for example `0.78`)
- False positives:
  - Raise threshold to `0.82`-`0.86`
  - Enroll lookalike family members as separate identities
- Missing prototype files:
  - Run Step 2 again (`generate_embeddings.py`)
- ONNX model not found or inference fails:
  - The active ONNX model is `models/w600k_r50.onnx` (iResNet-50, WebFace600K, 512-d).
  - Run `python download_models.py --list` to confirm it is present and check its size (~174 MB).
  - The system auto-falls back to FaceNet if ONNX is unavailable.

## 10. Delivery Notes for Client Demo

Before demo day:

1. Verify all required identities are enrolled
2. Regenerate embeddings fresh
3. Keep threshold at `0.80` as baseline
4. Keep one backup threshold profile (`0.78`, `0.82`)
5. Run `evaluate_accuracy.py --sweep` and save report screenshots

## 11. License

See `LICENSE`.
