# End-to-End Working Guide (Client Version)

This document explains exactly how the FewShotFace system works from enrollment to live recognition, with clear inputs, outputs, and optimization points.

## 1. Goal

Build a practical face recognition pipeline that:
- Works with small data per user (few-shot)
- Supports quick onboarding of new users
- Runs in real time on webcam
- Remains stable under lighting and distance variation

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
