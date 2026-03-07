# FewShotFace Client Code Explanation (Detailed)

This document is written for client discussions and handover meetings.
It explains the full codebase in plain language, then gives technical depth where needed.

## 1. Executive summary

FewShotFace is a face recognition system designed for practical onboarding with small data.
You can register a person using only a small set of images, generate feature data, and run real-time recognition without retraining a deep model from scratch each time.

Two ways to use it:
- Desktop operator app: `gui.py`
- Command-line scripts: `register.py`, `generate_embeddings.py`, `recognize.py`, `evaluate_accuracy.py`

Business value:
- Fast onboarding for new users
- Low operational complexity
- Real-time detection and recognition
- Measurable quality through evaluation scripts

## 2. End-to-end workflow (client view)

### Step 1: Enroll people
- File: `register.py` (or GUI Step 1)
- What happens: webcam captures face samples for each person with guidance.
- Output: images saved in `dataset/<person_name>/`

### Step 2: Optional photo normalization
- File: `fix_mobile_photos.py`
- What happens: if data comes from mixed cameras (mobile + webcam), lighting and sharpness are normalized.
- Output: improved image consistency (in-place or separate output folder)

### Step 3: Build recognition data
- File: `generate_embeddings.py` (or GUI Step 2)
- What happens: images are converted into numeric face vectors (embeddings), then class prototypes are built.
- Output: `.npy` artifacts in `embeddings/` + calibrated threshold JSON

### Step 4: Run live recognition
- File: `recognize.py` (or GUI Step 3)
- What happens: camera frame -> detect face -> extract embedding -> compare with prototypes -> show person name or `Unknown`.
- Output: live recognition UI overlay

### Step 5: Measure performance
- Files: `evaluate_accuracy.py`, `evaluate_models.py`
- What happens: checks quality metrics and helps select better threshold/model settings.
- Output: printed metrics and threshold comparison

## 3. Data flow (technical but simple)

```text
Webcam/Image Input
   -> Face detection
   -> Face crop + alignment + quality checks
   -> Embedding extraction (FaceNet/ONNX/InsightFace/DeepFace)
   -> Prototype/kNN similarity decision
   -> Name + confidence on screen
```

Training/build flow:

```text
dataset/<person>/images
   -> generate_embeddings.py
   -> embeddings.npy + labels.npy
   -> prototypes.npy + class_names.npy
   -> auto_threshold.json
```

## 4. Complete file-by-file explanation

### 4.1 Project docs and setup files

#### `README.md`
Purpose:
- Main overview, quick start, recommended defaults, troubleshooting.

When client asks:
- "How do we run this quickly?"
- "What are safe default threshold values?"

#### `INSTALLATION.md`
Purpose:
- Environment setup and install commands.

When client asks:
- "How do we install on another machine?"
- "What are the exact setup steps for Windows?"

#### `COMPONENTS.md`
Purpose:
- Architecture map and component responsibility.

When client asks:
- "Which module does what in production flow?"

#### `END_TO_END_WORKING.md`
Purpose:
- Stage-by-stage process with inputs/outputs and tuning guidance.

When client asks:
- "Explain full flow from enrollment to recognition report."

#### `requirements.txt`
Purpose:
- Python dependency list.

When client asks:
- "Which libraries are required for GUI and recognition?"

#### `LICENSE`
Purpose:
- Legal usage/distribution terms.

When client asks:
- "Can this be shipped to end users?"

### 4.2 Core operational scripts

#### `register.py`
What it does:
- Captures enrollment images from webcam.
- Uses face detection and quality checks to reject poor captures.
- Guides user through multiple poses/distances.

Input:
- `--name`, `--num-images`, `--camera-id`

Output:
- Face images saved to `dataset/<name>/`

Why it matters:
- Enrollment quality directly affects recognition accuracy.

Client-ready examples:
- "Increase capture count from 10 to 15 for better stability."
- "Make enrollment stricter to avoid blurry samples."

#### `fix_mobile_photos.py`
What it does:
- Reduces camera domain gap using white balance correction, CLAHE, unsharp mask.
- Optional face recrop for consistency.

Input:
- dataset path, identity filter, dry-run, output-dir, clip/sharpen settings

Output:
- Corrected images (in-place or separate folder)

Why it matters:
- Mixed source images can lower accuracy; this makes data more uniform.

Client-ready examples:
- "Normalize only one person folder before retraining."
- "Do dry-run and show what would be changed."

#### `generate_embeddings.py`
What it does:
- Converts images into embeddings.
- Applies quality filtering + optional TTA.
- Builds single and multi-prototypes.
- Calibrates threshold and saves it.

Input:
- dataset folder, backend selection, TTA options, quality threshold

Output files in `embeddings/`:
- `embeddings.npy`
- `labels.npy`
- `prototypes.npy`
- `class_names.npy`
- `multi_prototypes.npy`
- `multi_labels.npy`
- `auto_threshold.json`

Why it matters:
- This is the build step that powers live recognition.

Client-ready examples:
- "Use insightface backend for best quality."
- "Disable TTA for speed and compare results."

#### `recognize.py`
What it does:
- Runs real-time recognition from webcam.
- Uses adaptive thresholding, frame voting, tracking-based smoothing.
- Supports prototype and kNN fusion.

Input:
- threshold, backend, camera id, vote frames, minimum quality

Output:
- Real-time screen labels with confidence and performance info.

Why it matters:
- This is the production runtime behavior clients interact with.

Client-ready examples:
- "Reduce false positives; tune threshold and margin logic."
- "Stabilize flickering names in low light."

#### `evaluate_accuracy.py`
What it does:
- Computes per-identity and macro metrics.
- Supports threshold sweep.
- Reports sensitivity, specificity, precision, F1, accuracy.

Why it matters:
- Gives measurable evidence of system quality.

Client-ready examples:
- "Find best threshold for current dataset."
- "Generate periodic quality report after new enrollments."

#### `evaluate_models.py`
What it does:
- Compares proposed model path with classic baseline backbones.
- Uses train/query split and macro metrics.

Why it matters:
- Useful for technical reporting and benchmark comparison.

Client-ready examples:
- "Show where our chosen backend ranks versus baselines."

### 4.3 Core logic modules

#### `utils.py`
What it contains:
- Face detection helpers and data structures (`Detection`)
- Face quality scoring
- Crop enhancement and landmark alignment
- Temporal tracker (`FaceTracker`)
- Embedding backend wrappers (`FaceEmbedder`, FaceNet, ONNX, InsightFace, DeepFace, ViT)
- TTA helper and prototype builders
- Saved artifact loading helpers

Why it matters:
- It is the shared engine room used by all major scripts.

Client-ready asks:
- "Improve distant face handling."
- "Change backend fallback priority."

#### `similarity.py`
What it contains:
- Similarity functions (cosine/euclidean/ensemble)
- Prediction logic (`predict_with_prototypes`, `predict_with_knn`, `combined_predict`)
- Confidence calibration
- Auto-threshold calibration

Why it matters:
- Final identity decision quality depends heavily on this module.

Client-ready asks:
- "Make decision stricter for lookalike faces."
- "Tune ensemble balance for lower false positives."

### 4.4 Desktop application

#### `gui.py`
What it provides:
- End-user/operator desktop app with polished interface.
- 3 guided steps for enrollment, build, and recognition.
- Worker threads for non-blocking camera/build/runtime processing:
  - `EnrollWorker`
  - `TrainWorker`
  - `RecognitionWorker`

Why it matters:
- Primary operational interface for non-technical users.

Client-ready asks:
- "Add single-click full pipeline button."
- "Add exportable operator activity logs."

## 5. Folder-level explanation

### `dataset/`
- Raw enrollment images grouped by person.
- Example: `dataset/Dhaval/`, `dataset/Bhavanaben/`

Client note:
- This is the primary source data folder.

### `embeddings/`
- Generated model-ready data used at runtime.
- Contains threshold metadata and prototype vectors.

Client note:
- Must be regenerated after adding/removing users.

### `models/`
- ONNX model files for detection/embedding.
- Includes fallback options like `w600k_r50.onnx`.

Client note:
- Model file integrity and compatibility are critical for startup.

### `__pycache__/`
- Python runtime cache; not business data.

## 6. What to tell client in a handover meeting

Use this simple explanation:
- "We first register faces into `dataset/`."
- "Then we build recognition vectors into `embeddings/`."
- "Live recognition only reads those generated embeddings and compares faces in real time."
- "If new people are added, we rebuild embeddings once and continue."

## 7. What client can request from this codebase

### Operational requests
- Add a new person enrollment policy (minimum quality, image count).
- Add multi-camera selection in UI.
- Add unknown-face capture folder for audit.

### Accuracy requests
- Lower false positives in similar-looking users.
- Improve recognition at distance and low light.
- Tune threshold profile per deployment site.

### Reporting requests
- Export evaluation summary as CSV/JSON.
- Add periodic quality trend reports.
- Compare model behavior before and after new enrollments.

### Product requests
- Add role-based UI modes (admin/operator).
- Add attendance/event logging integration.
- Add deployment profile presets.

## 8. Run order (must follow)

1. Enroll: `register.py` or GUI Step 1
2. Optional normalize: `fix_mobile_photos.py`
3. Build embeddings: `generate_embeddings.py` or GUI Step 2
4. Recognize live: `recognize.py` or GUI Step 3
5. Evaluate: `evaluate_accuracy.py` and `evaluate_models.py`

Important rule:
- After any dataset change, run Step 3 again.

## 9. Command reference for technical team

```powershell
python gui.py
python register.py --name "Dhaval" --num-images 12 --camera-id 0
python fix_mobile_photos.py
python generate_embeddings.py --backend auto
python recognize.py --backend auto --threshold 0.80 --camera-id 0 --vote-frames 7
python evaluate_accuracy.py --sweep
python evaluate_models.py --dataset dataset --train-ratio 0.5
```

## 10. Client FAQ (ready answers)

Q: Why does the system show `Unknown` for a known person sometimes?
A: Usually due to low image quality, distance, lighting, or strict threshold. Add more enrollment images and rebuild embeddings.

Q: Why do we need to rebuild embeddings after adding a person?
A: Because the recognition database (prototypes and labels) is generated from dataset images and must include the new user.

Q: Can this run without internet?
A: Yes, after environment setup and model availability, runtime is local.

Q: Is model retraining required when adding people?
A: No deep retraining; only regenerate embeddings/prototypes.

Q: What is the safest default threshold?
A: Start near `0.80`, then tune based on false positives vs unknown rate using `evaluate_accuracy.py --sweep`.
