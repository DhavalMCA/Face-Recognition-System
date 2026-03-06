# Components Guide

This version of FewShotFace includes desktop and CLI components only.

## 1. Architecture Overview

```text
Desktop GUI (gui.py) and CLI scripts
            |
            v
      Core ML pipeline
            |
            v
   register -> embeddings -> recognize -> evaluate
```

## 2. Main Components

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

## 6. Removed Component

- Web dashboard/API (`app.py`, `templates/`, `static/`) has been removed from this version.
