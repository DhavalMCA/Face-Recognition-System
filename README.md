# Real-Time Few-Shot Face Recognition System

> **MTech / MCA Final-Year Project**  
> Author: **Dhaval Prajapati**  
> Platform: Windows 11 · Python 3.10 · PyQt5 · FastAPI · OpenCV  
> Embedding backends: ArcFace (ONNX) · FaceNet (PyTorch fallback)
> Last updated: **March 2026**

A production-ready, dual-interface (web dashboard + desktop GUI) face recognition system built on **few-shot prototype learning**. The system enrolls users with as few as 5–10 face images, generates compact ArcFace / FaceNet embeddings, and performs real-time cosine-similarity matching on live webcam frames — all without GPU requirements.

---

## Problem Statement

Traditional face recognition systems require large per-person image sets and expensive full retraining whenever a new identity is added. In real-world deployments, operators typically have only a handful of reference photos, lighting conditions vary across cameras, and family members with similar facial structure cause false positives.

This project solves all three challenges:

| Challenge | Solution |
|---|---|
| Few training samples | Prototype-based few-shot matching (mean class vector) |
| Lighting / camera domain gap | CLAHE normalization applied **inside the embedder** for all images (mobile + webcam) |
| Family member / lookalike confusion | Margin rule: `best − second_best ≥ 0.05`; auto-strict threshold when only one identity enrolled |
| Distance variation (2–3 m) | High-res capture + small-face auto-zoom + EMA smoothing + adaptive threshold |

---

## Objectives

- Work reliably with 5–10 images per identity (few-shot learning)
- Generate stable, discriminative face embeddings via ArcFace / FaceNet
- Run real-time webcam recognition at full interactive speed
- Provide three operator interfaces: web dashboard, desktop GUI, and CLI scripts
- Measure and report accuracy vs. published CNN baselines (VGG-16, ResNet-50, etc.)

---

## Key Features

| Feature | Detail |
|---|---|
| **Face Enrollment** | MTCNN detection → cropped face storage per identity folder |
| **Multi-Range Enrollment** | 3 close + 3 medium + 2 far samples collected per identity |
| **Dual Embedding Backend** | ArcFace ONNX (primary) · FaceNet PyTorch (auto-fallback) |
| **Unified CLAHE Preprocessing** | CLAHE applied inside `embed()` for **every** image — mobile and webcam normalized identically |
| **Few-Shot Prototypes** | L2-normalized mean class embedding computed once; updated on retrain |
| **Cosine / Euclidean Matching** | Configurable similarity metric with threshold slider |
| **Margin-Based Decision** | Accepts identity only when `best − second_best ≥ 0.05`; prevents lookalike false positives |
| **Single-Identity Guard** | Auto-raises threshold to `0.85` when only one class is enrolled (no contrastive signal) |
| **Distance-Robust Recognition** | Small-face detection, auto-zoom CLAHE crop, EMA smoothing, adaptive threshold |
| **Web Dashboard** | FastAPI + Glassmorphism UI (index.html + style.css) |
| **Desktop GUI** | PyQt5 3-column dark dashboard — clean left-accent card style, no decorative box borders |
| **Mobile Photo Fix** | `fix_mobile_photos.py` — CLAHE + unsharp-mask preprocessing to close camera domain gap |
| **Accuracy Evaluation** | Per-identity TP/TN/FP/FN, comparison table vs. 5 CNN baselines |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10 |
| Web backend | FastAPI + Uvicorn |
| Desktop UI | PyQt5 |
| Computer vision | OpenCV 4.x |
| Face detection | MTCNN (facenet-pytorch) |
| Embedding — primary | ArcFace ONNX via onnxruntime |
| Embedding — fallback | FaceNet InceptionResnetV1 (vggface2) |
| ML utilities | NumPy, scikit-learn |
| Frontend | HTML5 + CSS3 (Glassmorphism) + Vanilla JS |

---

## Project Workflow

```
Enroll Person  →  Generate Embeddings  →  Start Recognition
     │                    │                       │
 register.py        generate_embeddings.py    recognize.py
  (gui Step 1)         (gui Step 2)           (gui Step 3)
                             │
                    fix_mobile_photos.py   ← run before Step 2
                    evaluate_accuracy.py   ← run after Step 2
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
├── fix_mobile_photos.py     # Problem 1 — close mobile/webcam domain gap
├── evaluate_accuracy.py     # Problem 3 — measure accuracy vs. CNN baselines
├── evaluate_models.py       # additional model comparison helper
│
├── dataset/                 # face images per identity  [git-ignored]
│   └── <PersonName>/        # one folder per enrolled user
│       └── *.jpg
│
├── embeddings/              # generated artifacts  [git-ignored]
│   ├── embeddings.npy       # sample embedding matrix [N × dim]
│   ├── labels.npy           # identity labels [N]
│   ├── prototypes.npy       # mean class vectors [C × dim]
│   └── class_names.npy      # ordered class labels [C]
│
├── models/
│   └── arcface.onnx         # ArcFace ONNX model (download separately)
│
├── static/
│   ├── css/style.css        # Glassmorphism dashboard stylesheet
│   ├── js/ui-effects.js     # UI animation helpers
│   └── styles.css           # legacy stylesheet (kept for compatibility)
│
├── templates/
│   └── index.html           # web dashboard template
│
├── requirements.txt         # pip dependencies
├── README.md                # this file
├── COMPONENTS.md            # module-by-module function reference
└── INSTALLATION.md          # full setup and troubleshooting guide
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
# Via web dashboard: click "Capture Faces" in the Enrollment card
# Via desktop GUI:   Step 1 panel → enter name → Capture Face
# Via CLI:
python register.py --name Dhaval --num-images 8
```

> **Multi-range enrollment (recommended)**  
> The default captures **8 images in 3 phases**:
>
> | Phase | Images | Distance | On-screen cue |
> |---|---|---|---|
> | 1 — Close  | 1–3 | 0.5–1 m   | cyan label   |
> | 2 — Medium | 4–6 | 1–1.5 m   | green label  |
> | 3 — Far    | 7–8 | 1.5–2.5 m | orange label |
>
> The script prints *"Step back slightly for next samples"* at each phase
> transition.  Enrolling across distances makes the class prototype robust
> to range variation during live recognition.

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
# Via web / GUI: click "Start Live Recognition" (Step 3 panel)
# Via CLI:
python recognize.py --threshold 0.80
```

> **Distance-robust mode** activates automatically when a face bounding box
> width drops below 100 px (subject ~2–3 m away).  The overlay shows
> `dist-mode` next to the bounding box.
>
> | Trigger | Action |
> |---|---|
> | `face_width < 100 px` | `small_face = True` |
> | small_face | 25 % expanded crop from 1280 × 720 frame |
> | small_face | CLAHE contrast enhancement |
> | every face | EMA smoothing α = 0.7, resets after 10 missed frames |
> | small_face | Threshold auto-lowered by 0.05 (min 0.65) |

### Step 5 — Measure accuracy

```bash
python evaluate_accuracy.py --threshold 0.70       # single threshold
python evaluate_accuracy.py --sweep                # find best threshold
python evaluate_accuracy.py --threshold 0.80 -q    # quiet / summary only
```

---

## Distance-Robust Recognition (v2)

Version 2 of FewShotFace extends the recognition pipeline so the system
remains reliable when the subject is standing **2–3 metres from the webcam**.

### What was added

| Component | Change | Benefit |
|---|---|---|
| Webcam capture | Resolution raised to **1280 × 720** | More pixels per distant face |
| MTCNN detector | `min_face_size=20`, thresholds `[0.5, 0.6, 0.6]` | Detects faces as small as 60–80 px |
| Detection confidence | `min_confidence=0.85` (from 0.90) | Higher recall at range |
| Small-face flag | `face_width < 100 px → small_face=True` | Triggers enhanced processing path |
| `get_enhanced_crop()` | Expand box 25 %, CLAHE, `INTER_CUBIC` resize to 160 px | Better embedding input quality |
| `FaceTracker` class | EMA per track, α = 0.7, 10-frame expiry | Temporal smoothing removes flicker |
| Adaptive threshold | `max(0.65, base − 0.05)` when `small_face` | Fewer false Unknowns at range |
| Enrollment phases | 3 close + 3 medium + 2 far (8 images) | Prototype covers all distances |

### New symbols in `utils.py`

| Symbol | Type | Description |
|---|---|---|
| `apply_clahe_enhancement(face_rgb)` | function | CLAHE on luminance channel |
| `get_enhanced_crop(frame, box, margin, size)` | function | Expand → CLAHE → cubic resize |
| `FaceTracker` | class | EMA smoother with proximity-based track matching |

### Debug output

Each frame prints one line per detected face:
```
[BBox] x1=420 y1=180 x2=480 y2=260  w=60 h=80  small=True
```

---

## Accuracy Benchmarks

Example output from `evaluate_accuracy.py --sweep`:

| Model | Specificity (%) | Sensitivity (%) | Accuracy (%) |
|---|---|---|---|
| VGG-16 | 98.65 | 99.45 | 99.00 |
| ResNet-50 | 92.50 | 95.00 | 94.00 |
| GoogleNet | 88.24 | 90.00 | 89.00 |
| MobileNet | 86.00 | 83.40 | 88.30 |
| AlexNet | 84.00 | 88.46 | 87.70 |
| **FewShotFace (ONNX, t=0.80)** | **—** | **—** | **Your result** |

Run `python evaluate_accuracy.py --sweep` to fill in your model's column.

---

## Recommended Threshold

### Threshold Choice Reasoning

| Threshold | Sensitivity | Specificity | Use Case |
|---|---|---|---|
| 0.60 | High | Low | Maximum recall, many false positives |
| 0.70 | Balanced | Balanced | Default — good for solo demo |
| **0.80** | **Balanced** | **High** | **Recommended — general use** |
| 0.85 | Lower | Very High | Auto-applied when only one identity enrolled |

### Margin Rule (v2.1+)

The threshold alone is insufficient when a family member's face scores close to the enrolled person's score. The v2.1 decision logic adds a **margin guard**:

```
IF best_score >= threshold
AND (best_score − second_best_score) >= 0.05
THEN → Recognized
ELSE → Unknown
```

| Scenario | best | second | gap | Decision |
|---|---|---|---|---|
| Clear match | 0.88 | 0.61 | 0.27 | ✅ Recognized |
| Lookalike | 0.82 | 0.79 | 0.03 | ❌ Unknown (gap < 0.05) |
| Only one identity enrolled | 0.90 | — | — | Threshold auto-raised to 0.85 |

**For the family confusion problem:** the margin rule resolves it automatically without requiring the family member to be enrolled. Enrolling the family member as a separate identity further sharpens separation and is still recommended for the best accuracy.

---

## Dual Interface Screenshots

| Web Dashboard | Desktop GUI |
|---|---|
| Glassmorphism 3-column layout | PyQt5 3-column dark dashboard |
| Left: Enroll / Train / Monitor | Left: Step 1 / Step 2 / Step 3 |
| Center: Live Camera Feed | Center: Live Camera Feed |
| Right: Users + Detection Log | Right: Enrolled Users + Status |

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

### v2.1 — March 2026

**Bug fixes:**

- **Mobile / webcam domain gap** (`utils.py`):  
  CLAHE contrast normalization is now applied inside `FacenetEmbedder.embed()` and `ONNXEmbedder.embed()` for every image before ArcFace inference. Previously CLAHE was only used in the live `get_enhanced_crop()` path, so enrollment embeddings of mobile photos were not normalized the same way as webcam frames. Both sources now share an identical preprocessing chain.

- **Family member false positives** (`similarity.py`):  
  `predict_with_prototypes()` now applies a **margin rule** on top of the threshold check. A face is accepted only when `best_score − second_best_score ≥ 0.05`. This prevents a lookalike from being accepted simply because their score marginally exceeds the threshold.  
  Additionally, when only one identity is enrolled the effective threshold is automatically raised to `0.85` (no contrastive signal exists in single-class mode).

**GUI (v3) — `gui.py`:**

- Complete visual overhaul: left-accent card borders, subtitle rows in step headers, removed all decorative box borders from text labels and panel sections.
- `_step_card()` accepts an optional `subtitle` parameter; all three steps use it.
- `_step_number()` changed from outlined box to filled translucent badge.
- `_field_label()` and `_section_title()` migrated to `Segoe UI` for readability.

### v2.0 — Distance-Robust Recognition

- Webcam resolution raised to 1280 × 720.
- MTCNN `min_face_size=20`, thresholds `[0.5, 0.6, 0.6]`.
- `get_enhanced_crop()` — expanded box + CLAHE + cubic resize.
- `FaceTracker` — EMA smoothing per track, α = 0.7.
- Adaptive threshold: `max(0.65, base − 0.05)` when `small_face = True`.
- `register.py` multi-range enrollment: 3 close + 3 medium + 2 far.

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
| License | See LICENSE |
