# Components and Functions Reference

> Complete module-by-module breakdown of every file, class, and function in the project.  
> Audience: developer reference, project report appendix, viva preparation.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     OPERATOR INTERFACES                             │
│  Web Dashboard (app.py + index.html)  │  Desktop GUI (gui.py)       │
└──────────────────────┬───────────────────────────┬─────────────────┘
                       │                           │
              REST API (FastAPI)           Qt Signals/Slots
                       │                           │
┌──────────────────────▼───────────────────────────▼─────────────────┐
│                     PIPELINE SCRIPTS                                │
│   register.py  │  generate_embeddings.py  │  recognize.py           │
└──────────────────────┬─────────────────────────────────────────────┘
                       │
┌──────────────────────▼─────────────────────────────────────────────┐
│                     CORE ML LAYER                                   │
│       utils.py (detection + embedding)  │  similarity.py            │
└─────────────────────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼─────────────────────────────────────────────┐
│                 TOOLING / EVALUATION SCRIPTS                        │
│   fix_mobile_photos.py  │  evaluate_accuracy.py  │ evaluate_models.py│
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1) Web API Component (app.py)

Purpose:
- Serves dashboard UI
- Provides API endpoints
- Triggers pipeline scripts

Classes:
- RegisterRequest: validates payload for user registration
- EmbeddingRequest: validates payload for embedding generation
- RecognitionRequest: validates payload for recognition start

Functions:
- _run_script(command): runs child script and captures output
- dashboard(request): returns main web page
- health(): API health check
- list_users(): returns identity folder names
- list_users_details(): returns user names with image counts
- register_user(payload): starts registration script
- update_embeddings(payload): starts embedding generation script
- start_recognition(payload): starts recognition process in detached mode

---

## 2) Enrollment Component (register.py)

Purpose:
- Captures user face images from webcam and stores them in dataset

Functions:
- register_person(person_name, num_images, dataset_dir, camera_id): runs capture workflow and saves cropped face images
- parse_args(): reads command line arguments for enrollment

---

## 3) Embedding Generation Component (generate_embeddings.py)

Purpose:
- Converts stored face images into embedding vectors
- Saves labels and class prototypes

Functions:
- generate_embeddings(dataset_dir, embeddings_dir, backend, onnx_model_path): creates embeddings, labels, prototypes, class names
- parse_args(): reads command line arguments for embedding generation

---

## 4) Real-Time Recognition Component (recognize.py)

Purpose:
- Runs webcam recognition in real time using saved prototypes

Functions:
- _load_or_build_prototypes(embeddings_dir): loads precomputed prototypes or builds from embeddings
- recognize_realtime(embeddings_dir, metric, threshold, backend, onnx_model_path, camera_id): full real-time recognition loop
- parse_args(): reads command line arguments for recognition

---

## 5) Similarity and Decision Component (similarity.py)

Purpose:
- Compares query embedding with known class prototypes
- Decides known vs unknown

Functions:
- cosine_similarity(query_embedding, gallery_embeddings): similarity scores where higher is better
- euclidean_distance(query_embedding, gallery_embeddings): distances where lower is better
- predict_with_prototypes(query_embedding, prototypes, class_names, metric, threshold): final identity prediction and confidence

---

## 6) Shared Utilities Component (utils.py)

Purpose:
- Common building blocks used by all pipeline scripts

Classes:
- Detection: face detection container (box, confidence, cropped face)
- FacenetEmbedder: FaceNet embedding backend
- ONNXEmbedder: ONNX embedding backend
- FaceEmbedder: backend selector (auto/facenet/onnx)

Functions:
- ensure_dir(path): creates directory if missing
- get_identity_folders(dataset_dir): returns identity subfolders
- load_face_detector(image_size, margin): initializes MTCNN detector
- _normalize_face(face_rgb, target_size): resizes/normalizes face for model input
- detect_faces(frame_bgr, detector, min_confidence, padding): detects and crops faces
- load_saved_embeddings(embeddings_path, labels_path): loads embeddings and labels with validation
- compute_class_prototypes(embeddings, labels): computes normalized mean vectors per class

---

## 7) Desktop GUI Component (gui.py)

Purpose:
- Provides a Windows-style 3-column desktop dashboard for full project workflow
- Runs enrollment, training, and live recognition in separate QThread workers

Color Palette:
- Background: #0F172A
- Accent: #6366F1 (Indigo)
- Success: #10B981 (Emerald)
- Danger: #EF4444 (Red)
- Text: #F1F5F9 / Muted: #94A3B8

Layout (3-column Glassmorphism Dashboard):
- Left panel: Step 1 Enrollment card, Step 2 Training card, Step 3 Monitoring card
- Center panel: Live camera feed with neon-glow border
- Right panel: Enrolled users list + Latest detection status

Helper Functions:
- _card_style(extra): returns glassmorphism card stylesheet (rgba bg, neon border, 16px radius)
- _btn_style(bg, hover, text): returns gradient button stylesheet (indigo/emerald/red gradients)
- _badge(text, color, bg): builds compact status badge label for user cards
- _shadow(widget, blur, color, alpha): applies QGraphicsDropShadowEffect (blur=25, offset=(0,4))
- main(): application entry point — sets Fusion style, creates MainWindow, runs event loop

Worker Classes (ML logic — DO NOT MODIFY):
- EnrollWorker(QThread): captures webcam frames, detects faces, saves crops to dataset/
  - Signals: frame_ready, face_captured, finished_signal, error
- TrainWorker(QThread): loads images, extracts embeddings, saves prototypes
  - Signals: progress, finished_signal, error
- RecognitionWorker(QThread): runs live detection+embedding+similarity loop
  - Signals: frame_ready, person_detected, error

Main UI Class (MainWindow):
- _build_ui(): constructs QGridLayout 3-column dashboard
- _build_header(): sticky header with title and 4 status indicators
- _build_step1(): enrollment card (name input, image count, thumbnail strip, capture button)
- _build_step2(): training card (progress bar, status label, train button)
- _build_step3(): monitoring card (strictness slider, start/stop buttons)
- _build_video_panel(): center camera frame with neon glow border
- _build_users_panel(): scrollable enrolled users list with avatar thumbnails
- _build_status_panel(): latest detection badge (green=Authorized, red=Unknown)
- _on_enroll(): validates input and starts EnrollWorker
- _on_train(): starts TrainWorker, monitors progress bar
- _on_monitor(): validates prototypes exist, starts RecognitionWorker
- _on_stop_monitor(): safely terminates RecognitionWorker thread
- _on_person_detected(name, conf, ts): updates status panel with green/red badge
- _display_frame(frame): converts OpenCV BGR frame to QPixmap and paints video label
- _refresh_status(): updates 4 header indicators (model/camera/embeddings/system)
- _refresh_users(): rebuilds user card list from dataset/ directory
- _user_card(name, count, imgs, trained): builds one user row with thumbnail, name, badge
- closeEvent(): ensures all worker threads stop cleanly before window closes

---

## 8) Mobile Photo Preprocessing (fix_mobile_photos.py)

Purpose:
- Closes the domain gap between mobile camera photos and webcam photos
- Mobile photos have warm white balance, higher resolution, and wider pose variety
- Without normalization, mobile photo embeddings fall outside the webcam training distribution

Preprocessing Pipeline (applied in order):
1. Grey-world white balance — cancels warm/cool camera color casts
2. CLAHE on L channel (LAB color space) — equalises brightness without shifting colors
3. Unsharp mask sharpening — restores fine facial detail compressed by JPEG
4. MTCNN re-detect and re-crop — aligns face at consistent scale (224×224)

Functions:
- apply_clahe(bgr, clip_limit): CLAHE on LAB luminance channel
- apply_unsharp_mask(bgr, strength): high-frequency detail boost via Gaussian subtraction
- normalize_white_balance(bgr): per-channel mean scaling (Grey World assumption)
- preprocess_image(bgr, clip_limit, sharpen_strength): full 4-stage pipeline → 224×224 output
- detect_and_crop_face(bgr, detector, min_confidence): MTCNN crop with 0.85 confidence
- process_identity_folder(...): processes all images in one identity folder with stats
- main(): CLI entry point — parses args, loads detector, processes identities

CLI Arguments:
- --name: process only one identity folder
- --output-dir: save to separate directory instead of overwriting
- --dry-run: simulate without writing any files
- --clip-limit: CLAHE aggressiveness (default 2.5, range 1.5–4.0)
- --sharpen-strength: unsharp mask weight (default 0.8, increase for blurry images)
- --no-detect: skip MTCNN crop — just normalize lighting

Usage:
```bash
python fix_mobile_photos.py                          # process all in-place
python fix_mobile_photos.py --name Dhaval            # one person
python fix_mobile_photos.py --dry-run                # preview
python fix_mobile_photos.py --clip-limit 3.0 --sharpen-strength 1.2  # aggressive
```

---

## 9) Accuracy Evaluation (evaluate_accuracy.py)

Purpose:
- Measures per-identity recognition performance using one-vs-rest binary classification
- Computes TP, TN, FP, FN, Sensitivity, Specificity, Accuracy, Precision, F1
- Prints comparison table against 5 published CNN baseline models
- Accepts threshold argument to test different operating points

Evaluation Methodology:
- For each image of identity X: predict identity using predict_with_prototypes
- For the true class X:
  - TP if predicted = X; FN if predicted ≠ X
- For every other class Y:
  - FP if predicted = Y; TN if predicted ≠ Y
- Macro-average metrics across all enrolled identities

Metrics:
- Sensitivity (Recall / TPR) = TP / (TP + FN)  — how well the model finds each person
- Specificity (TNR)          = TN / (TN + FP)  — how well the model rejects non-matches
- Accuracy                   = (TP + TN) / (TP + TN + FP + FN)
- Precision                  = TP / (TP + FP)
- F1                         = 2 × (Sensitivity × Precision) / (Sensitivity + Precision)

Classes:
- EvalResult: counter container per identity (tp, tn, fp, fn + derived metrics as properties)

Functions:
- load_prototypes(embeddings_dir): loads prototypes.npy and class_names.npy
- embed_image(img_path, detector, embedder): detect + embed one image file
- evaluate(...): full pass over all dataset images, returns list of EvalResult
- print_per_identity_table(results): tabular per-identity metric output
- print_comparison_table(results, threshold, backend): your model vs VGG-16 / ResNet-50 / AlexNet / GoogleNet / MobileNet
- print_sweep_table(sweep_data): multi-threshold sweep output with best threshold highlighted
- print_threshold_recommendation(results, threshold): threshold guidance + family problem explanation
- main(): CLI orchestration — loads resources, runs evaluation, prints all reports

Published Baselines Included:
- VGG-16: Specificity 98.65% · Sensitivity 99.45% · Accuracy 99.00%
- ResNet-50: Specificity 92.50% · Sensitivity 95.00% · Accuracy 94.00%
- AlexNet: Specificity 84.00% · Sensitivity 88.46% · Accuracy 87.70%
- GoogleNet: Specificity 88.24% · Sensitivity 90.00% · Accuracy 89.00%
- MobileNet: Specificity 86.00% · Sensitivity 83.40% · Accuracy 88.30%

CLI Arguments:
- --threshold (-t): recognition threshold (default 0.70)
- --metric: cosine or euclidean (default cosine)
- --sweep: run at all threshold values and print sweep table
- --sweep-values: custom threshold list for sweep
- --backend: auto / facenet / onnx
- --quiet (-q): suppress per-image output

Usage:
```bash
python evaluate_accuracy.py --threshold 0.70
python evaluate_accuracy.py --sweep
python evaluate_accuracy.py --threshold 0.80 --backend onnx -q
```

---

## 10) Frontend Dashboard Component

### templates/index.html

Purpose:
- 3-column Glassmorphism web dashboard with full API integration

Layout:
- Left panel: Enrollment (Step 1), Training with progress bar (Step 2), Monitoring with slider (Step 3)
- Center panel: Live camera feed placeholder with animated neon border + waiting badge
- Right panel: Stats row (Users / Images / Backend), Enrolled Users, Latest Detection, Activity Log

Key Frontend Functions:
- log(message, type): writes timestamped color-coded activity log entries
- clearLog(): clears activity log console
- setLoading(btnId, loading): toggles spinner / disabled state on buttons
- setTrainingProgress(value): animates progress bar (0–100) with shimmer effect
- updateLatestDetection(name, detail): refreshes latest detection card on right panel
- checkHealth(): GET /health — updates Online/Offline badge in header
- loadUsers(): GET /api/users/details — renders user avatar cards and updates stats
- registerUser(event): POST /register — enrollment flow with camera status update
- generateEmbeddings(): POST /embeddings/update — animated training progress
- startRecognition(): POST /recognition/start — starts live recognition session

Flask / FastAPI Route IDs (must not be changed):
- /health  · /api/users/details  · /register  · /embeddings/update  · /recognition/start

### static/css/style.css

Purpose:
- Full Glassmorphism dashboard stylesheet for web UI (replaces legacy styles.css)

Design System:
- Background: #0F172A (primary) / #1E293B (secondary)
- Accent: #6366F1 · Success: #10B981 · Danger: #EF4444
- Card: rgba(255,255,255,0.05) · border-radius: 18px · backdrop-filter: blur(15px)

Main Style Groups:
- CSS custom properties (--accent, --success, --danger, --card-bg, etc.)
- Ambient glow orbs (animated background gradients)
- Sticky glass header with status badge and pulse animation
- 3-column CSS Grid dashboard (collapses to single column on tablet/mobile)
- Glassmorphism cards with neon hover glow border
- Gradient buttons (indigo/emerald) with scale hover animation
- Custom range slider with accent track and glowing handle
- Animated training progress bar with shimmer
- Camera container with animated rotating neon border
- Camera waiting status badge (amber ↔ green state)
- Scrollable user list and activity log console
- Reveal fade-in animation for all cards on page load
- Responsive breakpoints: 1320px (narrower columns) · 1060px (stack vertical) · 680px (mobile)

### static/js/ui-effects.js

Purpose:
- Lightweight UI-only animation enhancements (no ML or API logic)

Functions:
- updateCameraBadgeState(): switches camera badge between amber (waiting) and green (active)
- enhanceCardHover(): adds subtle translateY(-2px) lift on glass card mouse enter/leave
- bindProgressAnimation(): MutationObserver adds glow filter to progress bar when non-zero
- DOMContentLoaded handler: wires MutationObserver to camera status text node

---

## End-to-End Flow Summary

```
1. Operator enrolls person
   register.py (or gui Step 1 / web Step 1)
   → MTCNN detects face → crops saved to dataset/<Name>/

2. (Optional) Normalize mobile photos
   fix_mobile_photos.py
   → CLAHE + sharpening + WB correction applied in-place

3. Generate embeddings and prototypes
   generate_embeddings.py (or gui Step 2 / web Step 2)
   → FaceEmbedder extracts vectors → compute_class_prototypes builds mean class vectors
   → saved to embeddings/prototypes.npy + class_names.npy

4. Start real-time recognition
   recognize.py (or gui Step 3 / web Step 3)
   → Webcam frame → detect_faces → embed_face → predict_with_prototypes
   → cosine similarity vs prototypes → name + confidence displayed

5. Evaluate accuracy
   evaluate_accuracy.py
   → Every dataset image tested against prototypes
   → Per-identity TP/TN/FP/FN computed
   → Comparison table vs VGG-16 / ResNet-50 / AlexNet / GoogleNet / MobileNet printed
```
