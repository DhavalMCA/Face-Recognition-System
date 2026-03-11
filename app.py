from __future__ import annotations

import io
import sys
import time
import os
import ssl
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional
import threading

# Fix SSL CERTIFICATE_VERIFY_FAILED for model downloading
ssl._create_default_https_context = ssl._create_unverified_context
if sys.platform.startswith("win"):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import torch
from flask import Flask, render_template, Response, request, jsonify

# Include utils
from utils import (
    FaceEmbedder, FaceTracker, align_face_from_landmarks,
    build_augmented_prototypes, compute_class_prototypes, detect_faces,
    ensure_dir, get_enhanced_crop, get_identity_folders,
    load_face_detector, load_saved_embeddings
)
from similarity import predict_with_prototypes
from recognize import FrameVoter
from evaluate_accuracy import (
    evaluate as run_accuracy_evaluation,
    load_prototypes as load_accuracy_prototypes,
    print_comparison_table, print_per_identity_table, print_threshold_recommendation,
    _metrics_from_results, evaluate_backend, build_in_memory_prototypes,
    precompute_face_crops, COMPARISON_BACKENDS, _comparison_resolved_name,
)

app = Flask(__name__)

# Config
SIMILARITY_METRIC = "cosine"
THRESHOLD = 0.65
VOTE_FRAMES = 7
CAMERA_INDEX = 0
EMBEDDING_BACKEND = "auto"
ONNX_MODEL_PATH = "models/w600k_r50.onnx"
DATASET_DIR = "dataset"
EMBEDDINGS_DIR = "embeddings"

class EngineConfig:
    def __init__(self):
        self.camera_index = CAMERA_INDEX
        self.backend = EMBEDDING_BACKEND
        self.deepface_model = "ArcFace"
        self.threshold = THRESHOLD
        self.vote_frames = VOTE_FRAMES
        
cfg = EngineConfig()

class AppState:
    def __init__(self):
        self.mode = "idle" # idle, enroll, train, monitor
        self.current_frame = None
        self.lock = threading.Lock()
        
        # Enroll State
        self.enroll_name = ""
        self.enroll_target = 0
        self.enroll_captured = 0
        self.enroll_message = ""
        
        # Train State
        self.train_progress = 0
        self.train_status = "IDLE"
        self.train_message = ""
        
        # Monitor State
        self.last_detection = "— AWAITING STREAM"
        self.monitor_error = ""

state = AppState()

# Background thread objects
camera_thread = None
train_thread_obj = None

def get_camera():
    cap = (cv2.VideoCapture(cfg.camera_index, cv2.CAP_DSHOW) 
           if sys.platform.startswith("win") else cv2.VideoCapture(cfg.camera_index))
    return cap

def train_worker(backend, df_model):
    with state.lock:
        state.train_status = "EXTRACTING FEATURES..."
        state.train_progress = 0
        state.train_message = ""

    try:
        ensure_dir(EMBEDDINGS_DIR)
        folders = get_identity_folders(DATASET_DIR)
        if not folders:
            with state.lock:
                state.train_status = "ERROR"
                state.train_message = "No identity folders found. Enroll users first."
                state.mode = "idle"
            return
            
        with state.lock: state.train_progress = 10
        detector = load_face_detector()
        embedder = FaceEmbedder(backend=backend, onnx_model_path=ONNX_MODEL_PATH, deepface_model=df_model)
        
        with state.lock: state.train_progress = 25
        all_embeddings, all_labels = [], []
        
        for i, folder in enumerate(folders):
            imgs = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
            for img_path in imgs:
                img = cv2.imread(str(img_path))
                if img is None: continue
                dets = detect_faces(img, detector, min_confidence=0.80, padding=0.10)
                if not dets: continue
                largest = max(dets, key=lambda d: (d.box[2]-d.box[0])*(d.box[3]-d.box[1]))
                if largest.landmarks_5pt is not None:
                    aligned = align_face_from_landmarks(img, largest.landmarks_5pt)
                    face_input = aligned if aligned is not None else largest.face_rgb
                else:
                    face_input = largest.face_rgb
                emb = embedder.embed_face(face_input)
                all_embeddings.append(emb)
                all_labels.append(folder.name)
            with state.lock:
                state.train_progress = 25 + int(65 * (i + 1) / len(folders))

        if not all_embeddings:
            with state.lock:
                state.train_status = "ERROR"
                state.train_message = "No usable face images found."
                state.mode = "idle"
            return
            
        emb_arr = np.vstack(all_embeddings).astype(np.float32)
        lbl_arr = np.array(all_labels, dtype=str)
        norms = np.linalg.norm(emb_arr, axis=1, keepdims=True)
        emb_arr = emb_arr / (norms + 1e-8)
        
        import json as _json
        metadata = {"backend": backend, "deepface_model": df_model if backend == "deepface" else None}
        with open(Path(EMBEDDINGS_DIR) / "backend.json", "w") as _fh:
            _json.dump(metadata, _fh)
            
        np.save(Path(EMBEDDINGS_DIR) / "embeddings.npy", emb_arr)
        np.save(Path(EMBEDDINGS_DIR) / "labels.npy", lbl_arr)
        
        protos, names = build_augmented_prototypes(emb_arr, lbl_arr)
        np.save(Path(EMBEDDINGS_DIR) / "prototypes.npy", protos)
        np.save(Path(EMBEDDINGS_DIR) / "class_names.npy", names)
        
        with state.lock:
            state.train_progress = 100
            state.train_status = "DONE"
            state.train_message = f"Trained on {len(emb_arr)} samples | {len(names)} identities"
            state.mode = "idle"
            
    except Exception as e:
        with state.lock:
            state.train_status = "ERROR"
            state.train_message = str(e)
            state.mode = "idle"

def camera_loop():
    cap = None
    try:
        import json as _json
        detector = load_face_detector()
        HINTS = {
            1: "Look straight at camera", 2: "Tilt head slightly LEFT", 3: "Tilt head slightly RIGHT",
            4: "Look straight, smile slightly", 5: "Look straight (medium distance ~1.5 m)",
            6: "Turn chin slightly UP", 7: "Turn chin slightly DOWN", 8: "Look straight (step back ~2 m)",
            9: "Tilt head slightly LEFT (far)", 10: "Look straight, normal expression",
            11: "Tilt head slightly RIGHT (far)", 12: "Step back ~3 m, look straight",
            13: "Near distance, look straight", 14: "Side lighting, look straight",
            15: "Look straight, neutral expression (final)"
        }
        BLUR_THRESHOLD = 60.0
        
        # For monitor
        embedder = None
        prototypes, class_names = None, None
        voter = None
        tracker = None
        base_threshold = cfg.threshold
        vf = cfg.vote_frames
        
        enroll_frame_count = 0
        min_gap = 6
        
        while True:
            with state.lock:
                current_mode = state.mode
                
            if current_mode == 'train' or current_mode == 'idle':
                if cap is not None:
                    cap.release()
                    cap = None
                time.sleep(0.1)
                # Still output a blank frame or placeholder if we want
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "SYSTEM IDLE / OFFLINE", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                with state.lock:
                    state.current_frame = blank
                continue

            if cap is None:
                cap = get_camera()
                # reset monitor state if needed
                if current_mode == 'monitor':
                    # load model
                    try:
                        embedder = FaceEmbedder(backend=cfg.backend, onnx_model_path=ONNX_MODEL_PATH, deepface_model=cfg.deepface_model)
                        proto_path = Path(EMBEDDINGS_DIR) / "prototypes.npy"
                        names_path = Path(EMBEDDINGS_DIR) / "class_names.npy"
                        if proto_path.exists() and names_path.exists():
                            prototypes = np.load(proto_path).astype(np.float32)
                            class_names = np.load(names_path).astype(str)
                        else:
                            emb, lbl = load_saved_embeddings(Path(EMBEDDINGS_DIR) / "embeddings.npy", Path(EMBEDDINGS_DIR) / "labels.npy")
                            prototypes, class_names = compute_class_prototypes(emb, lbl)
                            
                        base_threshold = cfg.threshold
                        vf = cfg.vote_frames
                        voter = FrameVoter(window=vf, max_missed_frames=12)
                        tracker = FaceTracker(alpha=0.7, max_missed_frames=10)
                        
                        meta_path = Path(EMBEDDINGS_DIR) / "backend.json"
                        if meta_path.exists():
                            with open(meta_path) as _fh:
                                meta = _json.load(_fh)
                            trained_backend = meta.get("backend")
                            trained_df = meta.get("deepface_model")
                            if (str(trained_backend) != str(cfg.backend) or (cfg.backend == "deepface" and str(trained_df) != str(cfg.deepface_model))):
                                with state.lock:
                                    state.monitor_error = "Model mismatch! Re-train required."
                                    state.mode = 'idle'
                                continue
                        
                        with state.lock:
                            state.monitor_error = ""

                    except Exception as e:
                        with state.lock:
                            state.monitor_error = str(e)
                            state.mode = 'idle'
                        continue

            if not cap.isOpened():
                time.sleep(0.5)
                continue
                
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
                
            if current_mode == 'enroll':
                enroll_frame_count += 1
                detections = detect_faces(frame, detector, min_confidence=0.90, padding=0.12)
                for det in detections:
                    x1, y1, x2, y2 = det.box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 212, 255), 2)
                    
                with state.lock:
                    target = state.enroll_target
                    captured = state.enroll_captured
                    name = state.enroll_name
                    
                if detections and enroll_frame_count % min_gap == 0:
                    largest = max(detections, key=lambda d: (d.box[2]-d.box[0])*(d.box[3]-d.box[1]))
                    gray = cv2.cvtColor(largest.face_rgb, cv2.COLOR_RGB2GRAY)
                    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    if sharpness < BLUR_THRESHOLD:
                        cv2.putText(frame, f"Too blurry  (sharp={sharpness:.0f})", (10, frame.shape[0] - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 80, 255), 2)
                    else:
                        person_dir = Path(DATASET_DIR) / name
                        ensure_dir(person_dir)
                        face_bgr = cv2.cvtColor(largest.face_rgb, cv2.COLOR_RGB2BGR)
                        fname = f"{name}_{int(time.time()*1000)}_{captured+1}.jpg"
                        cv2.imwrite(str(person_dir / fname), face_bgr)
                        captured += 1
                        with state.lock:
                            state.enroll_captured = captured
                            if captured >= target:
                                state.enroll_message = f"Successfully enrolled {captured} face samples."
                                state.mode = 'idle'
                                
                with state.lock:
                    captured = state.enroll_captured
                cv2.putText(frame, f"CAPTURED: {captured}/{target}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 255), 2)
                next_hint = HINTS.get(captured + 1, "Look naturally")
                cv2.putText(frame, f"Next: {next_hint}", (10, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 80), 1)
                
            elif current_mode == 'monitor':
                # Check for parameter updates safely
                if voter is not None and getattr(voter, "_window", 0) != cfg.vote_frames:
                     vf = cfg.vote_frames
                     voter.window = vf
                base_threshold = cfg.threshold
                     
                detections = detect_faces(frame, detector, min_confidence=0.88, padding=0.12)
                active_boxes = [d.box for d in detections]
                
                if voter and tracker:
                    voter.expire(active_boxes)
                    tracker.expire_tracks(active_boxes)
                
                frame_width = frame.shape[1]
                latest_dets = []
                
                for det in detections:
                    x1, y1, x2, y2 = det.box
                    face_w = x2 - x1
                    if face_w < 120:
                        face_crop = get_enhanced_crop(frame, det.box, margin=0.25, target_size=160)
                    else:
                        if det.landmarks_5pt is not None:
                            aligned = align_face_from_landmarks(frame, det.landmarks_5pt)
                            face_crop = aligned if aligned is not None else det.face_rgb
                        else:
                            face_crop = det.face_rgb

                    if embedder and tracker and prototypes is not None and class_names is not None:
                        raw_embedding = embedder.embed_face(face_crop)
                        embedding = tracker.update(det.box, raw_embedding)
                        quality = face_w / max(1, frame_width)
                        effective_thr = min(0.90, max(0.40, base_threshold * (0.8 + 0.4 * quality)))
                        
                        raw_result = predict_with_prototypes(embedding, prototypes, class_names, metric=SIMILARITY_METRIC, threshold=effective_thr)
                        raw_conf = raw_result.get("confidence", 0.0)
                        
                        voted_name, voted_conf = voter.vote(det.box, str(raw_result["name"]), float(raw_conf))
                        name = voted_name
                        conf = voted_conf
                        color = (0, 255, 136) if name != "Unknown" else (51, 51, 255)
                        status = "AUTHORIZED" if name != "Unknown" else "UNKNOWN"
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (x1, max(0, y1 - 50)), (x2, y1), (8, 12, 20), -1)
                        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
                        cv2.putText(frame, name, (x1 + 6, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 235, 250), 2)
                        cv2.putText(frame, f"{conf*100:.0f}% | {status}", (x1 + 6, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)
                        
                        latest_dets.append(f"{name} ({conf*100:.0f}%)")
                
                if latest_dets:
                    with state.lock:
                        state.last_detection = " | ".join(latest_dets)
                
            with state.lock:
                state.current_frame = frame.copy()
                
    except Exception as e:
        print(f"Camera Loop Error: {e}")
        pass

# Start BG Thread
th = threading.Thread(target=camera_loop, daemon=True)
th.start()

def gen_frames():
    while True:
        with state.lock:
            frame = state.current_frame
        
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/state', methods=['GET'])
def get_state():
    with state.lock:
        s = {
            "mode": state.mode,
            "enroll_captured": state.enroll_captured,
            "enroll_target": state.enroll_target,
            "enroll_msg": state.enroll_message,
            "train_progress": state.train_progress,
            "train_status": state.train_status,
            "train_message": state.train_message,
            "monitor_last_detection": state.last_detection,
            "monitor_error": state.monitor_error
        }
    cfg_data = {
        "params": {
            "threshold": cfg.threshold,
            "vote_frames": cfg.vote_frames,
            "camera_index": cfg.camera_index,
            "backend": cfg.backend,
            "deepface_model": cfg.deepface_model
        }
    }
    s.update(cfg_data)
    return jsonify(s)

@app.route('/api/enroll', methods=['POST'])
def enroll():
    data = request.json
    name = data.get('name', '').strip()
    target = int(data.get('count', 12))
    if not name:
        return jsonify({"error": "Name required"}), 400
        
    with state.lock:
        state.mode = 'enroll'
        state.enroll_name = name
        state.enroll_target = target
        state.enroll_captured = 0
        state.enroll_message = ""
        
    return jsonify({"status": "ok"})

@app.route('/api/train', methods=['POST'])
def train():
    data = request.json
    backend = data.get('backend', cfg.backend)
    df_model = data.get('deepface_model', cfg.deepface_model)
    
    with state.lock:
        state.mode = 'idle' # Cannot monitor while training
        
    th = threading.Thread(target=train_worker, args=(backend, df_model), daemon=True)
    th.start()
    
    return jsonify({"status": "ok"})

@app.route('/api/monitor/toggle', methods=['POST'])
def toggle_monitor():
    data = request.json
    action = data.get('action') # 'start' or 'stop'
    
    with state.lock:
        if action == 'start':
            state.mode = 'monitor'
            state.monitor_error = ""
        else:
            state.mode = 'idle'
            state.last_detection = "— AWAITING STREAM"
            
    return jsonify({"status": "ok"})

@app.route('/api/config', methods=['POST'])
def update_config():
    data = request.json
    if "threshold" in data: cfg.threshold = float(data["threshold"])
    if "vote_frames" in data: cfg.vote_frames = int(data["vote_frames"])
    if "camera_index" in data: cfg.camera_index = int(data["camera_index"])
    if "backend" in data: cfg.backend = data["backend"]
    if "deepface_model" in data: cfg.deepface_model = data["deepface_model"]
    return jsonify({"status": "ok"})

@app.route('/api/users', methods=['GET'])
def get_users():
    ensure_dir(DATASET_DIR)
    folders = get_identity_folders(DATASET_DIR)
    users = []
    for f in folders:
        imgs = len(list(f.glob("*.jpg")) + list(f.glob("*.png")) + list(f.glob("*.jpeg")))
        users.append({"name": f.name, "count": imgs})
    return jsonify({"users": users})

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    from evaluate_accuracy import evaluate as run_accuracy_evaluation, load_prototypes as load_accuracy_prototypes
    from evaluate_accuracy import print_comparison_table, print_per_identity_table, print_threshold_recommendation
    import json as _json
    
    try:
        backend = cfg.backend
        df_model = cfg.deepface_model
        
        meta_path = Path(EMBEDDINGS_DIR) / "backend.json"
        if meta_path.exists():
            with open(meta_path) as _fh:
                meta = _json.load(_fh)
            trained_backend = str(meta.get("backend", backend))
            trained_df = str(meta.get("deepface_model") or df_model)
            if (trained_backend != str(backend) or (trained_backend == "deepface" and trained_df != str(df_model))):
                return jsonify({"error": f"Model mismatch! Trained with {trained_backend.upper()}, but active model is {str(backend).upper()}."}), 400

        prototypes, class_names = load_accuracy_prototypes(Path(EMBEDDINGS_DIR))
        detector = load_face_detector()
        embedder = FaceEmbedder(backend=backend, onnx_model_path=ONNX_MODEL_PATH, deepface_model=df_model)

        # Pre-detect all face crops once — shared across every backend.
        face_cache = precompute_face_crops(Path(DATASET_DIR), detector)

        t0 = time.perf_counter()
        results, total, no_face = run_accuracy_evaluation(
            dataset_dir=Path(DATASET_DIR), prototypes=prototypes, class_names=class_names,
            detector=detector, embedder=embedder, threshold=cfg.threshold,
            metric=SIMILARITY_METRIC, quiet=True, face_cache=face_cache,
        )
        elapsed = time.perf_counter() - t0

        report = io.StringIO()
        with redirect_stdout(report):
            print("FewShotFace — Accuracy Evaluation")
            print("=" * 72)
            print(f"Threshold        : {cfg.threshold:.2f}")
            print(f"Metric           : {SIMILARITY_METRIC}")
            print(f"Backend          : {embedder.backend_name}")
            print(f"Total images     : {total}")
            print(f"No face detected : {no_face}")
            print(f"Tested images    : {total - no_face}")
            print(f"Evaluation time  : {elapsed:.2f}s\n")
            print("Per-Identity Metrics")
            print_per_identity_table(results)
            print("\nEvaluating all comparison backends (face crops pre-cached)...")
            all_model_metrics = {embedder.backend_name: _metrics_from_results(results)}
            for lbl, bk, df_m, if_m in COMPARISON_BACKENDS:
                if _comparison_resolved_name(bk, df_m, if_m) == embedder.backend_name:
                    continue
                m = evaluate_backend(
                    label=lbl, backend=bk,
                    deepface_model=df_m, insightface_model=if_m,
                    dataset_dir=Path(DATASET_DIR), detector=detector,
                    threshold=cfg.threshold, metric=SIMILARITY_METRIC,
                    quiet=True, face_cache=face_cache,
                )
                if m is not None:
                    all_model_metrics[lbl] = m
            print("\nUnified Model Comparison")
            print_comparison_table(all_model_metrics, embedder.backend_name)
            print_threshold_recommendation(results, cfg.threshold)

        return jsonify({"report": report.getvalue().strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
