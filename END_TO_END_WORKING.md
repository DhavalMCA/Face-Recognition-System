# End-to-End Working Guide

This document explains exactly how the FewShotFace system works from enrollment to evaluation, with inputs, outputs, and optimisation notes at every stage.

---

## 1. Goal

Build a practical face recognition pipeline that:
- Works with small data per user (few-shot — 8–15 images)
- Supports quick onboarding of new users without model retraining
- Runs in real time on webcam
- Remains stable under lighting and distance variation
- Reports measurable accuracy with multi-model comparison

---

## 2. High-Level Pipeline

```
Step 1: Enroll Faces
  ↓
Step 2: (Optional) Normalize Mobile Photos
  ↓
Step 3: Generate Embeddings + Prototypes
  ↓
Step 4: Run Live Recognition
  ↓
Step 5: Evaluate Accuracy
```

---

## 3. Step-by-Step Operational Flow

### Step 1 — Enrollment (`register.py`)

**Purpose:** Capture face samples per identity and store them in `dataset/<name>/`.

**Inputs:** Person name, number of images, camera ID.

```powershell
python register.py --name "Dhaval" --num-images 12 --camera-id 0
```

**Output:** Cropped face images in `dataset/Dhaval/`

**Optimisations:**
- Multi-pose capture guidance (front, left/right tilt, distance variation)
- Blur filtering (only sharp frames are saved)

---

### Step 2 — Photo Normalisation (`fix_mobile_photos.py`) \[Optional\]

**Purpose:** Reduce quality gap between mobile photos and webcam frames.

**When to use:** Dataset includes phone photos or strong lighting variation.

```powershell
python fix_mobile_photos.py
```

**Output:** Improved face images with corrected lighting and sharpness.

**Optimisations applied:** White-balance correction, CLAHE contrast enhancement, unsharp mask.

---

### Step 3 — Embedding and Prototype Build (`generate_embeddings.py`)

**Purpose:** Convert each face image into a numeric embedding and build class prototypes.

```powershell
python generate_embeddings.py --backend auto
```

**Output files** in `embeddings/`:

| File | Contents |
|---|---|
| `embeddings.npy` | All individual face embeddings |
| `labels.npy` | Identity label per embedding |
| `prototypes.npy` | Mean embedding (prototype) per class |
| `class_names.npy` | Ordered list of enrolled identities |
| `backend.json` | Backend used to generate this set |

**Optimisations:**
- L2 normalisation for stable cosine similarity
- Prototype-based class representation (mean of L2-normalised embeddings)
- Backend fallback strategy: InsightFace → ONNX → FaceNet

---

### Step 4 — Live Recognition (`recognize.py`)

**Purpose:** Detect and identify faces from live camera stream.

```powershell
python recognize.py --backend auto --threshold 0.80 --camera-id 0
```

**Live frame flow:**

1. Capture frame from webcam
2. Detect face(s) with MTCNN
3. Apply enhanced crop for small/distant faces
4. Generate embedding for each face
5. Compare with stored prototypes (cosine similarity)
6. Apply threshold → assign identity or *Unknown*
7. Vote across last N frames (`FrameVoter`)
8. Display result overlay (name + confidence)

**Optimisations:**
- `FrameVoter` — prevents label flicker across frames
- `FaceTracker` — temporal smoothing for smooth live output
- Adaptive small-face enhancement for subjects at distance

---

### Step 5 — Accuracy Evaluation (`evaluate_accuracy.py`)

**Purpose:** Quantify recognition quality, compare all backends, and find optimal threshold.

```powershell
# Single threshold with all-backend comparison
python evaluate_accuracy.py --backend auto --threshold 0.80

# Sweep 0.60–0.90 and compare all backends
python evaluate_accuracy.py --backend auto --sweep

# Skip comparison (faster; primary backend only)
python evaluate_accuracy.py --backend auto --no-compare
```

**Speed architecture:**

```
┌─ precompute_face_crops() ─────────────────────────────────┐
│  MTCNN detection runs ONCE across all dataset images.     │
│  Result stored as FaceCropCache = Dict[path, face_rgb]    │
└──────────────────────────────────────────────────────────┘
         │ shared face cache (no re-detection)
         ▼
  ┌──────────────────────────────────┐
  │  Primary backend                 │  build_in_memory_prototypes()
  │  build protos + evaluate         │  evaluate(..., face_cache=cache)
  └──────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────────────────────┐
  │  _run_backends_parallel()                                │
  │  ThreadPoolExecutor(max_workers=4)                       │
  │    ├── InsightFace (buffalo_l)  ← embed only, no detect  │
  │    ├── InsightFace (buffalo_sc) ← embed only             │
  │    ├── FaceNet                  ← embed only             │
  │    ├── ONNX (w600k_r50)         ← embed only             │
  │    ├── DeepFace ArcFace         ← embed only             │
  │    ├── DeepFace VGG-Face        ← embed only             │
  │    ├── DeepFace Facenet512      ← embed only             │
  │    └── DeepFace SFace           ← embed only             │
  └──────────────────────────────────────────────────────────┘
         │
         ▼
  print_comparison_table()  — ranked box-bordered table
  print_threshold_recommendation()  — advisory box
```

**Output — Unified Model Comparison table columns:**
- Rank, Model, Accuracy, Precision, Recall (Sensitivity), Specificity, F1
- Primary model marked `◄`; sorted by Accuracy descending
- Footer: `Primary model : <name>  │  Accuracy: xx.xx%  │  Rank: #N of 8`

**Evaluation methodology (one-vs-rest per identity):**

| Metric | Formula |
|---|---|
| Sensitivity / Recall | TP / (TP + FN) |
| Specificity | TN / (TN + FP) |
| Accuracy | (TP + TN) / (TP + TN + FP + FN) |
| Precision | TP / (TP + FP) |
| F1 | 2 × Precision × Recall / (Precision + Recall) |

Final model metric = macro-average across all enrolled identities.

---

## 4. Recommended Runtime Settings

| Setting | Value |
|---|---|
| Images per identity | 10–15 |
| Recognition threshold | 0.80 |
| Vote frames | 5–7 |
| Camera resolution | 1280×720 |

---

## 5. Decision Tuning Guide

**Too many *Unknown* labels:**
- Add more enrollment images and rebuild embeddings
- Lower threshold slightly (e.g. `0.78`)

**False positives (wrong person recognised):**
- Raise threshold (`0.82`–`0.86`)
- Enroll lookalike family members as separate identities
- Rebuild embeddings after enrollment

**Family member confusion specifically:**
- The threshold recommendation section in evaluation output explains this in detail
- Short answer: enroll the family member separately + raise threshold to `0.80`

---

## 6. Client Demo Checklist

1. Confirm camera access and lighting
2. Confirm all required users are enrolled
3. Regenerate embeddings once fresh
4. Verify recognition at near and medium distance
5. Have fallback thresholds ready (`0.78`, `0.80`, `0.82`)
6. Run `evaluate_accuracy.py --sweep` and save the comparison table screenshot

---

## 7. Failure Recovery

| Symptom | Recovery |
|---|---|
| Recognition fails to start | Check `embeddings/prototypes.npy` exists; re-run Step 3 |
| Wrong camera | Try `--camera-id 1` |
| ONNX model not found | Run `python download_models.py --list`; confirm `models/w600k_r50.onnx` is ~174 MB |
| Evaluation crashes on a backend | Backend is skipped with a `⚠` warning; others continue |

---

## 8. Why This Is Client-Friendly

- No model retraining when adding new users — just re-run `generate_embeddings.py`
- Fast onboarding: 8–15 images per person is sufficient
- Works on standard CPU — no GPU required
- Usable from Web App, Desktop GUI, or CLI
- Measurable evaluation for reporting: accuracy, precision, recall, specificity, F1


## 2. High-Level Pipeline

```text
Step 1: Enroll Faces
      ->
Step 2: (Optional) Normalize Mobile Photos
      ->
Step 3: Generate Embeddings + Prototypes
      ->
Step 4: Run Live Recognition
      ->
Step 5: Evaluate Accuracy
```

## 3. Step-by-Step Operational Flow

### Step 1 - Enrollment (`register.py`)

Purpose:
- Capture face samples per identity and store them in `dataset/<name>/`.

Input:
- Person name
- Number of images
- Camera id

Command example:
```powershell
python register.py --name "Dhaval" --num-images 12 --camera-id 0
```

Output:
- Cropped face images in `dataset/Dhaval/`

Optimization used:
- Multi-pose capture guidance (front, left/right tilt, distance variation)
- Blur filtering during capture (better training quality)

---

### Step 2 - Photo Normalization (`fix_mobile_photos.py`) [Optional]

Purpose:
- Reduce quality gap between mobile photos and webcam frames.

When to use:
- If dataset includes phone photos or strong lighting variation.

Command example:
```powershell
python fix_mobile_photos.py
```

Output:
- Improved face images (lighting and sharpness corrected)

Optimization used:
- White-balance correction
- CLAHE contrast enhancement
- Unsharp mask

---

### Step 3 - Embedding and Prototype Build (`generate_embeddings.py`)

Purpose:
- Convert each face image into a numeric embedding and build class prototypes.

Command example:
```powershell
python generate_embeddings.py --backend auto
```

Output files (in `embeddings/`):
- `embeddings.npy`
- `labels.npy`
- `prototypes.npy`
- `class_names.npy`
- `backend.json`

Optimization used:
- L2 normalization for stable similarity comparison
- Prototype-based few-shot representation
- Backend fallback strategy (`insightface -> onnx -> facenet`)

---

### Step 4 - Live Recognition (`recognize.py`)

Purpose:
- Detect and identify faces from live camera stream.

Command example:
```powershell
python recognize.py --backend auto --threshold 0.80 --camera-id 0
```

Live frame flow:
1. Capture frame from webcam
2. Detect face(s)
3. Apply enhanced crop for small/distant faces
4. Generate embedding for each face
5. Compare with stored prototypes
6. Apply threshold and decision logic
7. Show result overlay (name/confidence)

Optimization used:
- Frame voting for result stability (`FrameVoter`)
- Temporal smoothing (`FaceTracker`)
- Adaptive handling for small faces at distance

---

### Step 5 - Accuracy Evaluation (`evaluate_accuracy.py`)

Purpose:
- Quantify recognition quality and find optimal threshold.

Command examples:
```powershell
python evaluate_accuracy.py --backend auto --threshold 0.80
python evaluate_accuracy.py --backend auto --sweep
```

Output:
- Accuracy metrics and threshold-performance comparison

Optimization used:
- Threshold sweep to select best operating point per dataset

## 4. Recommended Runtime Settings

- Images per identity: `10-15`
- Recognition threshold start point: `0.80`
- Vote frames: `5-7`
- Camera resolution: `1280x720` (recommended)

## 5. Decision Tuning Guide

If system labels known users as Unknown too often:
- Add more enrollment images
- Rebuild embeddings
- Reduce threshold slightly (for example `0.78`)

If system gives false positives:
- Increase threshold (`0.82` to `0.86`)
- Enroll lookalike users as separate identities
- Rebuild embeddings after enrollment

## 6. Client Demo Checklist

Before demo:
1. Confirm camera access and lighting
2. Confirm all required users are enrolled
3. Regenerate embeddings once
4. Verify recognition at near and medium distance
5. Keep fallback thresholds ready (`0.78`, `0.80`, `0.82`)
6. Run one quick evaluation (`--sweep`) for evidence

## 7. Failure Recovery Checklist

If recognition fails unexpectedly:
1. Check camera id (`--camera-id 0/1`)
2. Verify `embeddings/prototypes.npy` exists
3. Re-run Step 3 (`generate_embeddings.py`)
4. Re-open app and retest

## 8. Why This Is Client-Friendly

- No model retraining required when adding users
- Fast onboarding workflow
- Works on standard CPU machines
- Usable from Web App, Desktop GUI, or CLI depending on operator preference
- Includes measurable evaluation for reporting
