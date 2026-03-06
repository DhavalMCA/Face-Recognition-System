# FewShotFace - Real-Time Few-Shot Face Recognition

Client-ready face recognition system with two ways to operate:
- Desktop app (`gui.py`)
- Command line scripts

The system is optimized for low-data onboarding (few-shot): you can register a person with 8-15 images and start recognition without retraining a deep model.

## 1. What This Project Delivers

- Fast onboarding of new users
- Real-time webcam recognition
- Better stability with frame voting (reduced label flicker)
- Better distance handling (small-face enhancement)
- Better robustness in uneven lighting (CLAHE normalization)
- Evaluation scripts for measurable accuracy reporting

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

### 3.2 Launch Desktop GUI (recommended for operators)

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
|- gui.py
|- register.py
|- generate_embeddings.py
|- recognize.py
|- evaluate_accuracy.py
|- evaluate_models.py
|- fix_mobile_photos.py
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
- ONNX model error (`INVALID_PROTOBUF` for `models/arcface.onnx`):
  - The app now auto-falls back to `models/w600k_r50.onnx` (or FaceNet if ONNX is unavailable), so GUI/recognition should still start.
  - If you want ArcFace specifically, replace `models/arcface.onnx` with a valid file from a trusted source.

## 10. Delivery Notes for Client Demo

Before demo day:

1. Verify all required identities are enrolled
2. Regenerate embeddings fresh
3. Keep threshold at `0.80` as baseline
4. Keep one backup threshold profile (`0.78`, `0.82`)
5. Run `evaluate_accuracy.py --sweep` and save report screenshots

## 11. License

See `LICENSE`.
