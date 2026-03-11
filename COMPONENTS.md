# Components Guide

FewShotFace supports three operation modes: Web App (`app.py`), Desktop GUI (`gui.py`), and CLI scripts.

---

## 1. Architecture Overview

```
Web App (app.py)  ─┐
Desktop GUI (gui.py) ─┤─▶  Core ML pipeline
CLI scripts        ─┘        │
                             ▼
              register → embeddings → recognize → evaluate
```

---

## 2. Application Entry Points

### `app.py` — Flask Web Server

Purpose: Modern browser-based operator interface.

Key responsibilities:
- Step 1 enrollment (webcam capture via browser)
- Step 2 embedding generation
- Step 3 live recognition stream
- `/api/evaluate` — runs full multi-backend evaluation

Evaluation route behaviour:
- Calls `precompute_face_crops()` once → shared crop cache
- Runs primary backend evaluation with cached crops
- Evaluates all `COMPARISON_BACKENDS` sequentially, each using the same cache
- Returns unified comparison table + threshold recommendation

### `gui.py` — PyQt5 Desktop Application

Purpose: Standalone desktop operator interface.

Key responsibilities:
- Step 1 enrollment, Step 2 training, Step 3 live recognition
- `AccuracyWorker(QThread)` — background thread for evaluation

`AccuracyWorker.run()` evaluation flow:
1. `load_face_detector()` — MTCNN loaded once
2. `precompute_face_crops()` — all face crops cached in memory
3. `build_in_memory_prototypes()` — primary model protos built from cache
4. `run_accuracy_evaluation()` — primary evaluation using cache
5. Sequential loop over `COMPARISON_BACKENDS`, each `evaluate_backend()` call uses cache
6. `print_comparison_table()` — unified ranked table printed to report

Worker threads:
- `EnrollWorker`
- `TrainWorker`
- `RecognitionWorker`
- `AccuracyWorker`

---

## 3. CLI Scripts

### `register.py`
Captures face samples per identity and stores them under `dataset/<name>/`.

### `generate_embeddings.py`
Converts all dataset images to embeddings and builds class prototypes.

Output files in `embeddings/`:
- `embeddings.npy`, `labels.npy`
- `prototypes.npy`, `class_names.npy`
- `backend.json`

### `recognize.py`
Live webcam recognition using prototype matching.

Key runtime steps: detect → embed → match prototype → vote across frames → display.

### `evaluate_accuracy.py`

Accuracy evaluation and multi-backend comparison engine.

Key public API:

| Symbol | Purpose |
|---|---|
| `precompute_face_crops(dataset_dir, detector)` | Run MTCNN once; cache all face crops |
| `embed_image(..., face_cache=None)` | Embed one image; use cache to skip detection |
| `build_in_memory_prototypes(..., face_cache=None)` | Build prototypes without writing to disk |
| `evaluate(..., face_cache=None)` | Core one-vs-rest evaluation loop |
| `evaluate_backend(label, backend, ..., face_cache=None)` | Build protos + evaluate one backend |
| `_run_backends_parallel(backends, ..., face_cache)` | ThreadPoolExecutor over all comparison backends |
| `_metrics_from_results(results)` | Macro-average metrics dict from EvalResult list |
| `print_comparison_table(all_models, primary_label)` | Ranked table with box borders and Specificity column |
| `print_threshold_recommendation(results, threshold)` | Dynamically padded advisory box |
| `COMPARISON_BACKENDS` | List of 8 real backends evaluated live on local dataset |
| `FaceCropCache` | `Dict[str, Optional[np.ndarray]]` — type alias for crop cache |

Comparison backends (evaluated at runtime):
- InsightFace buffalo_l
- InsightFace buffalo_sc
- FaceNet
- ONNX w600k_r50
- DeepFace ArcFace
- DeepFace VGG-Face
- DeepFace Facenet512
- DeepFace SFace

### `fix_mobile_photos.py`
Optional preprocessing: white-balance correction, CLAHE, unsharp mask for mixed-camera datasets.

### `download_models.py`
ONNX model management utility.

Usage:
```powershell
python download_models.py --list      # show all models/ files with size
python download_models.py --sources   # show InsightFace download instructions
python download_models.py --url <URL> --name mymodel.onnx --verify
```

### `evaluate_models.py`
Additional model-level comparison output (kept for legacy compatibility).

---

## 4. Core Utility Modules

### `utils.py`

- `FaceEmbedder` — unified multi-backend wrapper (auto / facenet / onnx / insightface / deepface)
  - Default ONNX model: `models/w600k_r50.onnx` (iResNet-50, WebFace600K, 512-d embeddings)
  - Fallback priority: InsightFace → ONNX → FaceNet if a backend is unavailable
- `detect_faces()` — MTCNN detection with padding and confidence filtering
- `load_face_detector()` — lazy-loaded MTCNN singleton
- `get_identity_folders()` — list enrolled identity directories
- `build_augmented_prototypes()` — mean L2-normalised prototype per class

### `similarity.py`

- `predict_with_prototypes()` — cosine or Euclidean nearest-prototype prediction
- `cosine_similarity()`, `euclidean_distance()` helpers

---

## 5. Data and Artifacts

| Path | Contents |
|---|---|
| `dataset/<name>/` | Raw face images per enrolled identity |
| `embeddings/prototypes.npy` | Mean per-class embedding (required for recognition) |
| `embeddings/class_names.npy` | Ordered list of enrolled identities |
| `embeddings/embeddings.npy` | All individual embeddings (used for rebuild) |
| `embeddings/backend.json` | Backend used when embeddings were generated |
| `models/w600k_r50.onnx` | Primary ONNX model — iResNet-50, WebFace600K, 512-d (~174 MB) |
| `models/det_10g.onnx` | InsightFace detection model |

---

## 6. Run Order

1. `register.py`
2. `generate_embeddings.py`
3. `recognize.py`
4. `evaluate_accuracy.py`

After adding new identities, re-run `generate_embeddings.py` before recognition.


## 2. Main Components

### `app.py`

Purpose:
- Operator-facing modern Web application (Flask).

Key responsibilities:
- Step 1 enrollment
- Step 2 embedding generation
- Step 3 live recognition
- Runtime controls (threshold, vote frames, backend)

Worker threads:
- Handled asynchronously via JavaScript polling and background threads.

### `gui.py`

Purpose:
- Operator-facing desktop interface (PyQt5).

Key responsibilities:
- Step 1 enrollment
- Step 2 embedding generation
- Step 3 live recognition
- Runtime controls (threshold, vote frames, backend)

Worker threads:
- `EnrollWorker`
- `TrainWorker`
- `RecognitionWorker`

### `register.py`

Purpose:
- Capture and store face samples for each identity.

Output:
- Face images under `dataset/<name>/`

### `generate_embeddings.py`

Purpose:
- Generate embeddings and class prototypes.

Output files:
- `embeddings/embeddings.npy`
- `embeddings/labels.npy`
- `embeddings/prototypes.npy`
- `embeddings/class_names.npy`
- `embeddings/backend.json`

### `recognize.py`

Purpose:
- Real-time recognition using live camera frames.

Key runtime logic:
- Face detection
- Embedding extraction
- Similarity matching to prototypes
- Frame voting and smoothing for stable labels

### `evaluate_accuracy.py`

Purpose:
- Evaluate threshold behavior and report metrics.

### `evaluate_models.py`

Purpose:
- Compare model-level performance outputs.

### `fix_mobile_photos.py`

Purpose:
- Optional preprocessing for mixed-camera datasets.

## 3. Core Utility Modules

### `utils.py`

Contains:
- Face detection helpers
- Embedding model loading
- Crop enhancement and alignment
- Prototype helpers and data loading

### `similarity.py`

Contains:
- Similarity metrics (cosine/euclidean)
- Prototype prediction logic

## 4. Data and Artifacts

### `dataset/`
- Per-identity image folders used for training/build step.

### `embeddings/`
- Generated vectors and prototypes used at recognition time.

### `models/`
- ONNX and related model files.

## 5. Execution Contract

Run order should be:

1. `register.py`
2. `generate_embeddings.py`
3. `recognize.py`
4. `evaluate_accuracy.py`

If new users are added, run `generate_embeddings.py` again before recognition.

## 6. Added Component

- **Web dashboard/API**: A modern responsive web interface consisting of `app.py` (Flask Server), `templates/index.html` (layout), `static/css/style.css`, and `static/js/main.js`.
