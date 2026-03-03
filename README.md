# Real-Time Few-Shot Face Recognition System

> **MTech / MCA Final-Year Project**
> Author: **Dhaval Prajapati**
> Platform: Windows 11 · Python 3.10 · PyQt5 · FastAPI · OpenCV
> Embedding backends: InsightFace ArcFace (primary) · ONNX ArcFace · FaceNet (fallback)
> Last updated: **March 2026**

A production-ready, dual-interface (web dashboard + desktop GUI) face recognition system built on **few-shot prototype learning**. The system enrolls users with as few as 10 face images, generates compact ArcFace embeddings via InsightFace `buffalo_l`, and performs real-time cosine-similarity matching on live webcam frames — all without GPU requirements.

---

## Problem Statement

Traditional face recognition systems require large per-person image sets and expensive full retraining whenever a new identity is added. In real-world deployments, operators typically have only a handful of reference photos, lighting conditions vary across cameras, and family members with similar facial structure cause false positives.

This project solves all three challenges:

| Challenge | Solution |
|---|---|
| Few training samples | Prototype-based few-shot matching with outlier-rejection averaging |
| Lighting / camera domain gap | CLAHE normalization applied **inside the embedder** for all images |
| Family member / lookalike confusion | Margin rule: `best − second_best ≥ 0.05`; auto-strict threshold for single identity |
| Distance variation (2–3 m) | High-res capture + small-face auto-zoom + EMA smoothing + adaptive threshold |
| Flickering recognition result | **FrameVoter** — 7-frame majority vote before showing any label |
| Blurry enrollment photos | Laplacian sharpness gate — blurry frames silently skipped during capture |
| Prototype drift from bad photos | 2-sigma outlier rejection before computing class mean |

---

## Objectives

- Work reliably with 10 images per identity (few-shot learning)
- Generate stable, discriminative face embeddings via InsightFace ArcFace
- Run real-time webcam recognition at full interactive speed
- Provide three operator interfaces: web dashboard, desktop GUI, and CLI scripts
- Measure and report accuracy vs. published CNN baselines (VGG-16, ResNet-50, AlexNet, GoogLeNet, MobileNet)

---

## Key Features

| Feature | Detail |
|---|---|
| **Face Enrollment** | MTCNN detection → cropped face storage per identity folder |
| **Multi-Range Enrollment** | 4 close + 3 medium + 3 far (10 images total) with per-capture angle hints |
| **Blur Quality Gate** | Laplacian variance < 60 → frame silently skipped, waits for sharper image |
| **Angle Guidance** | On-screen hint for each capture: straight, left tilt, right tilt, chin up/down |
| **InsightFace ArcFace** | `buffalo_l` (ResNet-50 + ArcFace loss) — primary backend, ~99.7% LFW accuracy |
| **ONNX ArcFace** | Secondary backend via onnxruntime |
| **FaceNet PyTorch** | Auto-fallback if InsightFace / ONNX unavailable |
| **Outlier-Clean Prototypes** | `build_augmented_prototypes` — drops samples > 2σ from centroid before averaging |
| **L2-Normalised Embeddings** | All stored embeddings explicitly re-normalised before saving to disk |
| **Cosine / Euclidean Matching** | Configurable similarity metric with threshold slider |
| **Margin-Based Decision** | Accepts identity only when `best − second_best ≥ 0.05`; prevents lookalike false positives |
| **Single-Identity Guard** | Auto-raises threshold to 0.85 when only one class is enrolled |
| **FrameVoter Stability** | 7-frame rolling majority vote per tracked face — eliminates flicker |
| **Distance-Robust Recognition** | Small-face detection, auto-zoom CLAHE crop, EMA smoothing, adaptive threshold |
| **5-Model Comparison Table** | `evaluate_models.py` benchmarks VGG-16, ResNet-50, AlexNet, GoogLeNet, MobileNet |
| **Web Dashboard** | FastAPI + Glassmorphism UI |
| **Desktop GUI** | PyQt5 3-column dark dashboard with Vote Frames spinner |
| **Mobile Photo Fix** | `fix_mobile_photos.py` — CLAHE + unsharp-mask preprocessing |
| **Accuracy Evaluation** | Per-identity TP/TN/FP/FN, threshold sweep, comparison vs. baselines |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10 |
| Web backend | FastAPI + Uvicorn |
| Desktop UI | PyQt5 |
| Computer vision | OpenCV 4.x |
| Face detection | MTCNN (facenet-pytorch) |
| Embedding — primary | InsightFace ArcFace (`buffalo_l`) |
| Embedding — secondary | ArcFace ONNX via onnxruntime |
| Embedding — fallback | FaceNet InceptionResnetV1 (vggface2) |
| ML utilities | NumPy, scikit-learn, torchvision |
| Frontend | HTML5 + CSS3 (Glassmorphism) + Vanilla JS |

---

## Project Workflow

```
Enroll Person  →  Generate Embeddings  →  Start Recognition
     │                    │                       │
 register.py        generate_embeddings.py    recognize.py
  (gui Step 1)         (gui Step 2)           (gui Step 3)
                             │
                    fix_mobile_photos.py   ← run before Step 2 (optional)
                    evaluate_accuracy.py   ← run after Step 2
                    evaluate_models.py     ← compare vs. 5 paper baselines
```

---

## Folder Structure

```text
FewShotFace/
├── app.py                   # FastAPI web application entry point
├── gui.py                   # PyQt5 desktop application entry point
├── register.py              # CLI face enrollment from webcam
├── generate_embeddings.py   # CLI embedding + prototype generation
├── recognize.py             # CLI real-time recognition runner
├── similarity.py            # cosine / euclidean similarity + prediction
├── utils.py                 # shared ML utilities (detector, embedder, etc.)
│
├── fix_mobile_photos.py     # close mobile/webcam domain gap
├── evaluate_accuracy.py     # measure accuracy, threshold sweep
├── evaluate_models.py       # compare vs. 5 CNN paper baselines
│
├── dataset/                 # face images per identity  [git-ignored]
│   └── <PersonName>/
│       └── *.jpg
│
├── embeddings/              # generated artifacts  [git-ignored]
│   ├── embeddings.npy       # L2-normalised sample embedding matrix [N × dim]
│   ├── labels.npy           # identity labels [N]
│   ├── prototypes.npy       # outlier-cleaned mean class vectors [C × dim]
│   └── class_names.npy      # ordered class labels [C]
│
├── models/
│   └── arcface.onnx         # ArcFace ONNX model (download separately)
│
├── static/
│   ├── css/style.css
│   ├── js/ui-effects.js
│   └── styles.css
│
├── templates/
│   └── index.html
│
├── requirements.txt
├── README.md
├── COMPONENTS.md
└── INSTALLATION.md
```

---

## Quick Start

### Option A — Web Dashboard

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
python app.py

# 3. Open browser at http://localhost:8000
```

### Option B — Desktop GUI

```bash
python gui.py
```

For full step-by-step setup, see [INSTALLATION.md](INSTALLATION.md).

---

## Usage Guide

### Step 1 — Enroll a new person

```bash
# Via desktop GUI: Step 1 panel → enter name → Capture Face
# Via CLI:
python register.py --name Dhaval --num-images 10
```

> **Multi-range enrollment with angle guidance**
> The default captures **10 images in 3 phases**:
>
> | Phase | Captures | Distance | Angle hints |
> |---|---|---|---|
> | 1 — Close  | 1–4 | 0.5–1 m   | straight, tilt left, tilt right, smile |
> | 2 — Medium | 5–7 | 1–1.5 m   | straight, chin up, chin down |
> | 3 — Far    | 8–10 | 1.5–2.5 m | straight, tilt left, normal |
>
> - Each capture shows the **next required angle** at the bottom of the frame.
> - **Blurry frames are automatically skipped** (Laplacian sharpness < 60) — just stay still and a sharper frame will be accepted automatically.

### Step 2 — Normalize mobile photos (if any)

Run this **before** generating embeddings if the dataset contains any mobile camera photos:

```bash
python fix_mobile_photos.py          # process all identities
python fix_mobile_photos.py --name Dhaval   # one person only
python fix_mobile_photos.py --dry-run       # preview without writing
```

What it does: Grey-world white balance → CLAHE brightness normalization → unsharp-mask sharpening → MTCNN re-crop & alignment.

> **Note (v2.1+):** CLAHE normalization is now also applied **inside the embedding model** for every inference call.  This means mobile photos no longer need `fix_mobile_photos.py` as a mandatory preprocessing step — both sources are normalized at the embedding level automatically.  Running `fix_mobile_photos.py` still improves very dark or blown-out photos and remains recommended.

### Step 3 — Generate embeddings

```bash
# Via web / GUI: click "Generate Embeddings" (Step 2 panel)
# Via CLI:
python generate_embeddings.py --backend auto
```

> **Important after upgrading to v2.1+**: Because CLAHE is now applied inside the embedder, old `.npy` files produced without CLAHE will be inconsistent with new live-inference embeddings. Delete `embeddings/` and regenerate:
>
> ```bash
> Remove-Item -Recurse -Force embeddings    # PowerShell
> python generate_embeddings.py --backend auto
> ```

### Step 4 — Start recognition

```bash
# Via GUI: Step 3 → Start Recognition
# Via CLI:
python recognize.py --threshold 0.60 --backend insightface
```

> **FrameVoter — decision stability**
> The system keeps the last **7 frame predictions** for each tracked face.
> A label is only shown when it wins **more than half** (≥ 4 of 7) recent frames.
> This prevents single-frame errors (blinks, head turns) from flickering the result.
>
> Example:
> ```
> Frames: Alice, Alice, Unknown, Alice, Alice, Alice, Alice
> Vote:   6 × Alice → displayed: Alice ✓
> ```
>
> **Distance-robust mode** activates automatically when face width < 100 px:
>
> | Trigger | Action |
> |---|---|
> | `face_width < 100 px` | `small_face = True` |
> | small_face | 25% expanded crop from 1280×720 frame |
> | small_face | CLAHE contrast enhancement |
> | every face | EMA smoothing α=0.7, resets after 10 missed frames |
> | small_face | Threshold auto-lowered by 0.05 (min 0.40) |

### Step 5 — Measure accuracy

```bash
python evaluate_accuracy.py --backend insightface --sweep
python evaluate_models.py          # compare vs. 5 paper baselines
```

---

## Distance-Robust Recognition

| Component | Value | Benefit |
|---|---|---|
| Webcam resolution | 1280 × 720 | More pixels per distant face |
| MTCNN min_face_size | 20 | Detects faces as small as 60–80 px |
| Detection confidence | 0.85 | Higher recall at range |
| Small-face trigger | face_width < 100 px | Activates enhanced path |
| `get_enhanced_crop()` | +25% box, CLAHE, cubic resize to 160 px | Better embedding input |
| `FaceTracker` | EMA α=0.7, 10-frame expiry | Temporal smoothing |
| Adaptive threshold | base − 0.05, min 0.40 | Fewer false Unknowns at range |
| `FrameVoter` | 7-frame majority vote | Eliminates flicker from head pose / blink |

---

## Accuracy Benchmarks

Results from `evaluate_models.py` (cosine similarity, 50/50 train/test split):

| Rank | Model | Sensitivity (%) | Specificity (%) | Accuracy (%) |
|---|---|---|---|---|
| — | **FewShotFace — InsightFace ArcFace** | — | — | **run evaluate_models.py** |
| 1 | VGG-16 | 99.45 | 98.65 | 99.00 |
| 2 | ResNet-50 | 95.00 | 92.50 | 94.00 |
| 3 | GoogLeNet | 90.00 | 88.24 | 89.00 |
| 4 | MobileNet | 83.40 | 86.00 | 88.30 |
| 5 | AlexNet | 88.46 | 84.00 | 87.70 |

> Our model uses ArcFace loss + few-shot prototypes + outlier rejection — expected to match or exceed ResNet-50 on the same dataset.

Run the full comparison:
```bash
python evaluate_models.py --backend insightface
```

---

## Recommended Threshold

The **ideal threshold depends on the embedding backend**:

| Backend | Recommended Threshold | Reason |
|---|---|---|
| **InsightFace ArcFace** (buffalo_l) | **0.60** | ArcFace embeddings are tight; genuine pairs score > 0.80 easily |
| ONNX ArcFace | 0.60 | Same model family |
| FaceNet | 0.70 | Softer embedding space, needs a higher bar |

> **Default is 0.60** across all entry points since v3.0.

### Threshold Tradeoff

| Threshold | Effect |
|---|---|
| Too high (0.85) | Real users rejected — false rejects |
| Too low (0.40) | Wrong people accepted — false accepts |
| **0.60** | **Best balance for ArcFace** |

### Margin Rule

A second guard runs on top of the threshold:

```
IF best_score >= 0.60
AND (best_score − second_best_score) >= 0.05
THEN → Recognized
ELSE → Unknown
```

| Scenario | best | second | gap | Decision |
|---|---|---|---|---|
| Clear match | 0.88 | 0.61 | 0.27 | ✅ Recognized |
| Lookalike | 0.82 | 0.79 | 0.03 | ❌ Unknown (gap < 0.05) |
| Only one identity enrolled | 0.90 | — | — | Threshold auto-raised to 0.85 |

---

---

## Dual Interface

| Web Dashboard | Desktop GUI |
|---|---|
| Glassmorphism 3-column layout | PyQt5 3-column dark dashboard |
| Left: Enroll / Train / Monitor | Left: Step 1 / Step 2 / Step 3 |
| Center: Live Camera Feed | Center: Live Camera Feed |
| Right: Users + Detection Log | Right: Enrolled Users + Status |
| — | Vote Frames spinner (Step 3) |

---

## Full Component Reference

See [COMPONENTS.md](COMPONENTS.md) for every module, class, and function explained in plain language.

---

## Desktop GUI (v3)

The PyQt5 desktop GUI was fully redesigned in v3 with a clean **Neural Surveillance Terminal** aesthetic.

| Change | Detail |
|---|---|
| Card style | Full box borders replaced with `border-left: 3px solid cyan` accent — minimal, modern |
| Step cards | Each card shows title + subtitle inline (no separate description label) |
| Step number badge | Filled translucent background instead of outlined box |
| Field labels | Switched to `Segoe UI`, brighter `TEXT_MID` color for better readability |
| Section titles | Font bumped to 11 px `Segoe UI`, reduced letter-spacing |
| Enrolled Identities / Latest Detection panels | Left-accent border only — no full perimeter box |
| Hint text | Uses `TEXT_MID` (neutral) instead of `CYAN` (neon) — less visual noise |

---

## Changelog

### v3.0 — March 2026 (current)

**Accuracy improvements:**

- **InsightFace ArcFace backend** (`utils.py`): Added `InsightFaceEmbedder` wrapping `insightface.app.FaceAnalysis` with `buffalo_l` (ResNet-50 + ArcFace loss). Auto-priority: InsightFace > ONNX > FaceNet.

- **Outlier-clean prototypes** (`generate_embeddings.py`): Switched from plain mean (`compute_class_prototypes`) to 2σ outlier rejection (`build_augmented_prototypes`). One bad enrollment photo no longer corrupts the prototype.

- **Proper L2 normalisation** (`generate_embeddings.py`): All embeddings explicitly re-normalised before saving — guarantees correct cosine geometry regardless of backend or dtype casting.

- **FrameVoter majority vote** (`recognize.py`, `gui.py`): 7-frame rolling window per tracked face. Result only changes when a label wins > 50% of the window — eliminates single-frame flicker.

- **Default threshold 0.60** across all entry points (calibrated for ArcFace).

- **Adaptive threshold floor fixed** (`recognize.py`): Changed from `max(0.65, base-0.05)` to `max(0.40, base-0.05)` — the old floor was higher than the base, making small faces stricter instead of more lenient.

**Enrollment improvements:**

- 10 images default (was 8), split 4 close + 3 medium + 3 far.
- Per-capture angle hints on-screen for each of the 10 shots.
- Blur quality gate: Laplacian variance < 60 silently skips the frame.

**Model comparison:**

- `evaluate_models.py` fully rewritten with all 5 paper baselines: VGG-16, ResNet-50, AlexNet, GoogLeNet, MobileNet.

**GUI:**

- Vote Frames spinner in Step 3 card (range 1–10, default 7).
- Sample Count spinner default raised to 10, range extended to 30.
- Angle hints and blur gate mirrored in `EnrollWorker`.
- `FrameVoter` wired into `RecognitionWorker`.

### v2.1 — March 2026

- CLAHE normalization moved inside embedder for consistent preprocessing.
- Margin rule added to `predict_with_prototypes` (MARGIN = 0.05).
- Single-identity threshold guard (auto 0.85).
- GUI full visual overhaul: left-accent card borders, dark theme.

### v2.0 — Distance-Robust Recognition

- Webcam resolution raised to 1280 × 720.
- `get_enhanced_crop()` — expanded box + CLAHE + cubic resize.
- `FaceTracker` — EMA smoothing per track, α = 0.7.
- Multi-range enrollment: 3 close + 3 medium + 2 far.

---

## Future Scope

- Liveness detection / anti-spoofing (blink detection, depth maps)
- Multi-camera support with identity tracking across streams
- GPU-accelerated ONNX inference (CUDA execution provider)
- Cloud deployment with role-based access control
- Event logs, attendance tracking, and analytics dashboard
- Mobile application client via REST API

---

## Author

| Field | Detail |
|---|---|
| Name | Dhaval Prajapati |
| Project | Real-Time Few-Shot Face Recognition System |
| Degree | MTech / MCA Final Year |
| Platform | Windows 11, Python 3.10 |
| Repository | [DhavalMCA/Face-Recognition-System](https://github.com/DhavalMCA/Face-Recognition-System) |
| License | See LICENSE |
