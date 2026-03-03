# End-to-End Working of the Few-Shot Face Recognition System

## 1. Introduction
This project is a real-time face recognition system designed to operate on a **few-shot learning** basis. While traditional CNN-based systems (like VGG or ResNet) require hundreds of images per person and complete model retraining when a new user is added, this system achieves high accuracy with only **5 to 10 images per person** and incorporates new users in seconds by computing "Prototypes" (mean embedding vectors), completely eliminating the need for GPUs or retraining.

---

## 2. Core Methodologies to Solve Real-World Problems
- **Problem 1: Few Samples per User**  
  *Solution*: Uses prototype-based matching. It translates face images into high-dimensional vectors and creates an "average" vector (prototype) for each person.
- **Problem 2: Domain Gap (Mobile vs. Webcam)**  
  *Solution*: Normalizes mobile-captured photos to match the webcam distribution using a custom pipeline (White-balance correction + CLAHE + Unsharp masking).
- **Problem 3: Family Member Confusion (Similar Faces)**  
  *Solution*: High precision cosine-similarity thresholds (recommended `0.80`) combined with separate distinct prototype registration for similar-looking people.
- **Problem 4: Distance Variation (2–3 m from webcam)**  
  *Solution*: High-resolution capture (1280 × 720), MTCNN tuned for small faces, auto-zoom CLAHE crop, per-face EMA temporal smoothing, and adaptive thresholding reduce false "Unknown" at medium and far range.

---

## 3. End-to-End Workflow Pipelines

The system executes a strictly sequential pipeline to recognize faces. 

### Step 1: Face Enrollment (`register.py`)
- **Goal**: Register a person's facial features into the database at **multiple distances**.
- **Process**: A user stands in front of the webcam. The script captures frames across three guided phases. **MTCNN** precisely localizes the face, crops it, and saves it.

  | Phase | Count | Distance | On-screen colour |
  |---|---|---|---|
  | 1 — Close  | 3 images | 0.5–1 m   | Cyan  |
  | 2 — Medium | 3 images | 1–1.5 m   | Green |
  | 3 — Far    | 2 images | 1.5–2.5 m | Orange |

  The script prints *"Step back slightly for next samples"* at each phase change.
- **Output**: 8 saved cropped images in `dataset/<PersonName>/`, named with phase tag (`_p1`, `_p2`, `_p3`).

### Step 2: Lighting & Domain Normalization (`fix_mobile_photos.py`) — *Optional*
- **Goal**: Fix lighting imbalances, especially if photos were provided via a smartphone instead of the active webcam.
- **Process**: 
  1. **Grey-world white balance**: Removes warm/cool camera tints.
  2. **CLAHE**: Equalizes brightness without shifting colors.
  3. **Unsharp mask**: Recovers fine facial details compressed by phone cameras.
  4. **MTCNN**: Re-detects and rigidly aligns the face at consistent scaling.
- **Output**: Clean, normalized dataset images.

### Step 3: Embeddings & Prototype Generation (`generate_embeddings.py`)
- **Goal**: Convert pixel data into mathematical feature representation.
- **Process**: 
  - Every cropped face image in the dataset is passed through a pre-trained deep learning model (Primary: **ArcFace ONNX**, Fallback: **FaceNet PyTorch**). 
  - The model outputs a dense numerical array (an embedding vector) representing facial geometry.
  - The system then calculates a **Mean Class Prototype**: the mathematical average of all vectors for a single person.
- **Output**: The extracted mean representations are saved to disk (`embeddings/prototypes.npy` and `embeddings/class_names.npy`).

### Step 4: Live Recognition (`recognize.py`)
- **Goal**: Identify faces streaming from the active camera in real time, including subjects at 2–3 m distance.
- **Process**:
  1. **Capture**: A live frame is grabbed from the webcam at **1280 × 720** resolution.
  2. **Detect**: MTCNN (tuned for small faces: `min_face_size=20`, thresholds `[0.5, 0.6, 0.6]`) finds the live face and crops it.
  3. **Small-face check**: If bounding box width < 100 px the system enters *distance mode*:
     - Expands the crop by 25 % on all sides in the full-resolution frame.
     - Applies **CLAHE** contrast normalisation on the luminance channel.
     - Resizes to 160 × 160 with bicubic interpolation.
  4. **Embed**: The crop is passed through ArcFace/FaceNet to generate a live embedding vector.
  5. **Temporal smoothing**: **EMA** (`α = 0.7`) is applied per tracked face. Tracks reset after 10 consecutive missed frames.
  6. **Adaptive threshold**: In distance mode the decision threshold is lowered by 0.05 (clamped at 0.65) to reduce false "Unknown" at range.
  7. **Compare**: Cosine Similarity (or Euclidean Distance) between the smoothed embedding and all saved `prototypes.npy`.
  8. **Decide**: If the highest similarity score exceeds the effective threshold the identity is accepted; otherwise labelled "Unknown".
- **Output**: Visual bounding box with Name, Confidence, Score, effective Threshold, and `dist-mode` indicator drawn on the feed.

### Step 5: Accuracy Verification (`evaluate_accuracy.py`)
- **Goal**: Scientifically prove the system's accuracy against recognized metrics.
- **Process**: Runs a One-vs-Rest calculation for all images to measure True Positives, True Negatives, False Positives, and False Negatives. 
- **Output**: Prints statistical Sensitivity, Specificity, Accuracy, and an F1 score, automatically comparing the results against baselines like VGG-16, ResNet-50, and MobileNet.

---

---

## 4. Distance-Robust Recognition Pipeline (v2)

This section describes the internal flow activated when a face is detected at
medium or far range (bounding box width < 100 px).

```
Full 1280×720 frame
        │
        ▼
MTCNN detection  (min_face_size=20, thresholds=[0.5,0.6,0.6])
        │
        ├─ face_width ≥ 100 px  ──► standard face_rgb crop
        │
        └─ face_width < 100 px  ──► SMALL FACE MODE
                │
                ├─ Expand box 25% in original full frame
                ├─ Crop expanded region
                ├─ CLAHE contrast enhancement (LAB luminance)
                └─ Bicubic resize to 160×160
                │
                ▼
        ArcFace / FaceNet embedding
                │
                ▼
        FaceTracker EMA smoothing
        smoothed = 0.7 * current + 0.3 * previous_smoothed
        (reset after 10 missed frames)
                │
                ▼
        Adaptive threshold
        effective_thr = max(0.65, base_thr - 0.05)  ← small face only
                │
                ▼
        Cosine similarity vs. prototypes
                │
        ┌───────┴────────┐
     Known              Unknown
```

### New utility components

| File | Symbol | Role |
|---|---|---|
| `utils.py` | `apply_clahe_enhancement()` | CLAHE on luminance channel of RGB face crop |
| `utils.py` | `get_enhanced_crop()` | Expand + CLAHE + bicubic resize for small faces |
| `utils.py` | `FaceTracker` | Per-face EMA smoother with proximity-based track matching |
| `utils.py` | `load_face_detector()` | Updated: `min_face_size=20`, thresholds `[0.5,0.6,0.6]` |
| `recognize.py` | `recognize_realtime()` | 1280×720 stream, all distance-robust stages wired together |
| `register.py` | `register_person()` | 3-phase multi-range enrollment |

---

## 5. Architectural Interfaces

The actual ML pipeline is completely detached and can be triggered flawlessly through three available interfaces:
1. **Desktop GUI (`gui.py`)**: A professional 3-column dark-themed PyQt5 desktop application built for exhibitions using strictly threaded workers to prevent freezing.
2. **Web Dashboard (`app.py`)**: A FastAPI-powered local web server with a stunning Glassmorphism HTML/CSS/JS frontend interface.
3. **CLI Scripts**: Standard python terminal execution for silent or programmatic background training and evaluating.

## 6. Technology Stack Summary
- **Backend/Logic**: Python 3.10
- **Computer Vision**: OpenCV 4.x
- **Face Detection Layer**: MTCNN (`facenet-pytorch`) — tuned to `min_face_size=20`
- **Feature Extraction Layer**: ArcFace via `onnxruntime`
- **Distance-Robust helpers**: `FaceTracker`, `get_enhanced_crop`, `apply_clahe_enhancement` (in `utils.py`)
- **Mathematics / ML**: `NumPy`, `scikit-learn`
- **Web App**: `FastAPI` + `Uvicorn`
- **Desktop App**: `PyQt5`

---

## 7. Accuracy Benchmarks

Example output from `evaluate_accuracy.py --sweep`:

| Model | Specificity (%) | Sensitivity (%) | Accuracy (%) |
|---|---|---|---|
| VGG-16 | 98.65 | 99.45 | 99.00 |
| ResNet-50 | 92.50 | 95.00 | 94.00 |
| GoogleNet | 88.24 | 90.00 | 89.00 |
| MobileNet | 86.00 | 83.40 | 88.30 |
| AlexNet | 84.00 | 88.46 | 87.70 |
| **FewShotFace (ONNX, t=0.80)** | **—** | **—** | **Your result** |

## 8. Recommended Threshold

### Threshold Choice Reasoning

| Threshold | Sensitivity | Specificity | Use Case |
|---|---|---|---|
| 0.60 | High | Low | Maximum recall, many false positives |
| 0.70 | Balanced | Balanced | Default — good for solo demo |
| **0.80** | **Balanced** | **High** | **Recommended — prevents family confusion** |
| 0.85 | Lower | Very High | Security-critical, may miss valid users |

**For the family confusion problem:** raise threshold to `0.80` AND enroll the family member as a separate identity. Two distinct prototypes allow the model to separate similar faces by comparing relative distances rather than relying solely on the absolute threshold.
