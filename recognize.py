"""Real-time inference module for few-shot face recognition.

Purpose:
    Performs webcam-based online recognition using previously generated
    prototypes and embedding vectors.

Role in pipeline:
    This module is the final runtime stage. It consumes stored features and
    applies similarity-based identity prediction on live frames.

Few-shot contribution:
    Uses prototype matching instead of full classifier retraining, allowing
    recognition with very limited samples per identity.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from similarity import predict_with_prototypes
from utils import (
    FaceEmbedder,
    FaceTracker,
    compute_class_prototypes,
    detect_faces,
    get_enhanced_crop,
    load_face_detector,
    load_saved_embeddings,
)


def _load_or_build_prototypes(embeddings_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load class prototypes from disk or build them from saved embeddings.

    Function name:
        _load_or_build_prototypes

    Purpose:
        Reuses precomputed prototype artifacts when available; otherwise
        computes prototypes from raw embedding/label files.

    Parameters:
        embeddings_dir (str): Directory containing embedding artifacts.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - prototypes: Prototype matrix [num_classes, embedding_dim]
            - class_names: Identity labels corresponding to prototypes

    Role in face recognition process:
        Provides class reference vectors used in similarity comparison during
        real-time recognition.
    """
    proto_path = Path(embeddings_dir) / "prototypes.npy"
    names_path = Path(embeddings_dir) / "class_names.npy"

    # Fast path: load precomputed prototypes for low-latency startup.
    if proto_path.exists() and names_path.exists():
        prototypes = np.load(proto_path).astype(np.float32)
        class_names = np.load(names_path).astype(str)
        return prototypes, class_names

    # Fallback path: derive prototypes from saved sample-level embeddings.
    embeddings, labels = load_saved_embeddings(
        Path(embeddings_dir) / "embeddings.npy",
        Path(embeddings_dir) / "labels.npy",
    )
    return compute_class_prototypes(embeddings, labels)


def recognize_realtime(
    embeddings_dir: str = "embeddings",
    metric: str = "cosine",
    threshold: float = 0.60,
    backend: str = "auto",
    onnx_model_path: str = "models/arcface.onnx",
    camera_id: int = 0,
) -> None:
    """Run real-time webcam recognition using prototype similarity matching.

    Function name:
        recognize_realtime

    Purpose:
        Captures live frames, detects faces, extracts embeddings, compares each
        embedding with class prototypes, and overlays recognition result.

    Parameters:
        embeddings_dir (str): Directory containing saved embeddings/prototypes.
        metric (str): Similarity metric ('cosine' or 'euclidean').
        threshold (float): Decision threshold for known vs unknown matching.
        backend (str): Embedding backend ('auto'/'facenet'/'onnx').
        onnx_model_path (str): Path to ONNX embedding model.
        camera_id (int): OpenCV camera index.

    Returns:
        None

    Role in face recognition process:
        Implements full real-time inference flow and final recognition decision
        (Authorized/Known identity or Unknown) for each detected face.
    """
    # Initialize detection and feature extraction models for streaming mode.
    detector = load_face_detector()
    embedder = FaceEmbedder(backend=backend, onnx_model_path=onnx_model_path)

    # Load class prototypes that represent enrolled identities.
    prototypes, class_names = _load_or_build_prototypes(embeddings_dir)

    if len(prototypes) == 0:
        raise RuntimeError("No prototypes found. Run generate_embeddings.py first.")

    import sys as _sys
    # Open webcam stream for real-time inference.
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW) if _sys.platform.startswith("win") else cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions/device id.")

    # ----------------------------------------------------------------
    # High-resolution capture (1280x720) so distant faces contain
    # enough pixels for reliable embedding extraction.
    # ----------------------------------------------------------------
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ----------------------------------------------------------------
    # Temporal embedding smoother: one EMA track per visible face.
    # alpha=0.7 weights the current frame heavily while smoothing
    # frame-to-frame noise. Tracks reset after 10 missed frames.
    # ----------------------------------------------------------------
    tracker = FaceTracker(alpha=0.7, max_missed_frames=10)

    # Minimum face width (pixels) below which small-face mode activates.
    SMALL_FACE_PX = 100

    print("=" * 80)
    print("Few-Shot Face Recognition Started  [Distance-Robust Mode]")
    print(f"Backend       : {embedder.backend_name}")
    print(f"Metric        : {metric}")
    print(f"Threshold     : {threshold}  (auto-reduced by 0.05 for small faces, min 0.65)")
    print(f"Known classes : {len(class_names)}")
    print(f"Resolution    : {actual_w}x{actual_h}")
    print("Press 'q' to stop")
    print("=" * 80)

    prev_time = time.time()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                continue

            # Face detection stage on current frame.
            # Lower min_confidence (0.85) improves recall for small distant faces.
            detections = detect_faces(frame, detector, min_confidence=0.85, padding=0.12)

            # ----------------------------------------------------------------
            # Expire stale tracks for faces no longer in frame, then process
            # each active detection.
            # ----------------------------------------------------------------
            active_boxes = [det.box for det in detections]
            tracker.expire_tracks(active_boxes)

            # Evaluate each detected face independently.
            for det in detections:
                x1, y1, x2, y2 = det.box
                face_w = x2 - x1
                face_h = y2 - y1

                # ----------------------------------------------------------
                # Bounding box size monitoring.
                # Faces narrower than SMALL_FACE_PX pixels indicate a subject
                # standing 2-3 m from the camera (distance-mode activation).
                # ----------------------------------------------------------
                small_face = face_w < SMALL_FACE_PX
                print(
                    f"[BBox] x1={x1} y1={y1} x2={x2} y2={y2}  "
                    f"w={face_w} h={face_h}  small={small_face}",
                    flush=True,
                )

                if small_face:
                    # Auto-zoom: expand box by 25%, crop, resize to 160x160,
                    # and apply CLAHE before passing to the embedding model.
                    face_crop = get_enhanced_crop(
                        frame, det.box, margin=0.25, target_size=160
                    )
                else:
                    # Standard crop — use the already-extracted face region.
                    face_crop = det.face_rgb

                # Feature extraction: face crop -> raw embedding vector.
                raw_embedding = embedder.embed_face(face_crop)

                # ----------------------------------------------------------
                # Temporal embedding smoothing (EMA per tracked face).
                # Reduces recognition flicker caused by head pose / lighting
                # variation between consecutive frames.
                # smoothed = 0.7 * current + 0.3 * previous
                # ----------------------------------------------------------
                embedding = tracker.update(det.box, raw_embedding)

                # ----------------------------------------------------------
                # Adaptive thresholding for distance.
                # Small faces yield noisier embeddings so we slightly relax
                # the threshold to reduce false "Unknown" rejections, clamped
                # at 0.65 to preserve identity discrimination.
                # ----------------------------------------------------------
                if small_face:
                    effective_threshold = max(0.65, threshold - 0.05)
                else:
                    effective_threshold = threshold

                # Similarity comparison + threshold-based identity decision.
                result = predict_with_prototypes(
                    query_embedding=embedding,
                    prototypes=prototypes,
                    class_names=class_names,
                    metric=metric,
                    threshold=effective_threshold,
                )

                name = str(result["name"])
                confidence = float(result["confidence"])

                # Recognition decision visualization:
                # green = known/authorized, red = unknown.
                color = (0, 200, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Show small-face indicator so operator knows distance mode.
                if small_face:
                    cv2.putText(
                        frame,
                        "dist-mode",
                        (x1, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        (0, 200, 255),
                        1,
                    )

                # Display metric-specific score for academic interpretability.
                if metric == "cosine":
                    score_text = f"cos={result['score']:.3f}"
                else:
                    score_text = f"dist={result['score']:.3f}"

                thr_text = f"thr={effective_threshold:.2f}"
                label = f"{name} | conf={confidence * 100:.1f}% | {score_text} | {thr_text}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(18, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    2,
                )

            # Lightweight FPS calculation for runtime performance reporting.
            current_time = time.time()
            fps = 1.0 / max(1e-6, current_time - prev_time)
            prev_time = current_time

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Classes: {len(class_names)}",
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.imshow("FewShotFace - Real-Time Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """Parse command-line configuration for real-time recognition.

    Function name:
        parse_args

    Purpose:
        Defines CLI interface for metric selection, threshold tuning, backend
        selection, and camera source configuration.

    Parameters:
        None

    Returns:
        argparse.Namespace: Parsed runtime recognition settings.

    Role in face recognition process:
        Allows reproducible evaluation of threshold and metric effects in
        few-shot real-time recognition experiments.
    """
    parser = argparse.ArgumentParser(description="Run real-time few-shot face recognition")
    parser.add_argument("--embeddings-dir", default="embeddings", help="Embeddings directory")
    parser.add_argument(
        "--metric",
        default="cosine",
        choices=["cosine", "euclidean"],
        help="Similarity metric",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Threshold for known/unknown decision",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "facenet", "onnx"],
        help="Embedding backend",
    )
    parser.add_argument(
        "--onnx-model-path",
        default="models/arcface.onnx",
        help="Path to ONNX face embedding model",
    )
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam device id")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    recognize_realtime(
        embeddings_dir=args.embeddings_dir,
        metric=args.metric,
        threshold=args.threshold,
        backend=args.backend,
        onnx_model_path=args.onnx_model_path,
        camera_id=args.camera_id,
    )
