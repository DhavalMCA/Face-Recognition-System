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
from collections import Counter, deque
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

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


# ───────────────────────────────────────────────────────────────────────
# FrameVoter — Majority-vote decision stability
# ───────────────────────────────────────────────────────────────────────

class FrameVoter:
    """Per-face rolling majority-vote decision stabiliser.

    Accumulates the last `window` per-frame identity predictions for each
    tracked face and returns the label that appeared most often.  This
    eliminates single-frame noise (blink, sudden head turn) that would
    otherwise flicker the recognition result on screen.

    Example (window=5)::

        frames  [Alice, Alice, Unknown, Alice, Alice]
        vote    → Alice  (4 of 5 votes)   ✔

    Only a result that holds the majority is returned; if no label wins
    more than half the votes the output is "Unknown".
    """

    def __init__(
        self,
        window: int = 5,
        max_match_dist: int = 120,
        max_missed_frames: int = 12,
    ) -> None:
        """Initialise voter.

        Parameters:
            window (int): Number of recent frames to consider (3 – 7 works well).
            max_match_dist (int): Pixel radius for re-associating tracks.
            max_missed_frames (int): Frames a track can be absent before reset.
        """
        if window < 1:
            raise ValueError("window must be ≥ 1")
        self.window = window
        self.max_match_dist = max_match_dist
        self.max_missed_frames = max_missed_frames
        # {track_id: deque of (name, confidence) tuples}
        self._history: Dict[int, Deque[Tuple[str, float]]] = {}
        self._centers: Dict[int, Tuple[int, int]] = {}
        self._missed:  Dict[int, int] = {}
        self._next_id: int = 0

    # ── internal helpers ────────────────────────────────────

    @staticmethod
    def _center(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = box
        return (x1 + x2) // 2, (y1 + y2) // 2

    def _match(self, center: Tuple[int, int]) -> Optional[int]:
        best_id, best_dist = None, float(self.max_match_dist)
        for tid, c in self._centers.items():
            d = ((c[0] - center[0]) ** 2 + (c[1] - center[1]) ** 2) ** 0.5
            if d < best_dist:
                best_dist, best_id = d, tid
        return best_id

    # ── public API ────────────────────────────────────────

    def vote(
        self,
        box: Tuple[int, int, int, int],
        name: str,
        confidence: float,
    ) -> Tuple[str, float]:
        """Record a single-frame prediction and return the majority-vote result.

        Parameters:
            box (Tuple): Bounding box (x1, y1, x2, y2) of the detected face.
            name (str): Single-frame identity prediction (may be 'Unknown').
            confidence (float): Single-frame confidence [0, 1].

        Returns:
            Tuple[str, float]: (voted_name, mean_confidence_of_voted_class)
        """
        center = self._center(box)
        tid = self._match(center)

        if tid is None:
            tid = self._next_id
            self._next_id += 1
            self._history[tid] = deque(maxlen=self.window)
            self._missed[tid]  = 0

        self._centers[tid] = center
        self._missed[tid]  = 0
        self._history[tid].append((name, confidence))

        # Majority vote over window
        labels = [n for n, _ in self._history[tid]]
        counts = Counter(labels)
        top_name, top_count = counts.most_common(1)[0]

        # Only accept the vote if it holds > half the window.
        if top_count > len(labels) / 2:
            matched_confs = [c for n, c in self._history[tid] if n == top_name]
            return top_name, float(np.mean(matched_confs))
        return "Unknown", 0.0

    def expire(self, active_boxes: list) -> None:
        """Age absent tracks and remove stale ones.

        Call once per frame after processing all active detections.

        Parameters:
            active_boxes: List of bounding boxes present in the current frame.
        """
        active_centers = [self._center(b) for b in active_boxes]
        covered: set = set()
        for c in active_centers:
            tid = self._match(c)
            if tid is not None:
                covered.add(tid)

        for tid in list(self._history.keys()):
            if tid not in covered:
                self._missed[tid] = self._missed.get(tid, 0) + 1

        stale = [t for t, m in self._missed.items() if m >= self.max_missed_frames]
        for tid in stale:
            self._history.pop(tid, None)
            self._centers.pop(tid, None)
            self._missed.pop(tid, None)


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
    vote_frames: int = 7,
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

    # ── Frame-vote stabiliser: one rolling majority-vote history per face. ─
    # vote_frames=5 means a label must win >2 of the last 5 frame predictions
    # before it is displayed, eliminating single-frame flicker.
    voter = FrameVoter(window=vote_frames, max_missed_frames=12)
    SMALL_FACE_PX = 100

    print("=" * 80)
    print("Few-Shot Face Recognition Started  [Distance-Robust Mode]")
    print(f"Backend       : {embedder.backend_name}")
    print(f"Metric        : {metric}")
    print(f"Threshold     : {threshold}  (auto-reduced by 0.05 for small faces, min 0.40)")
    print(f"Vote window   : {vote_frames} frames  (majority vote for stability)")
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

            # Expire stale tracks (embedding smoother AND vote history).
            active_boxes = [det.box for det in detections]
            tracker.expire_tracks(active_boxes)
            voter.expire(active_boxes)

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
                    effective_threshold = max(0.40, threshold - 0.05)
                else:
                    effective_threshold = threshold

                # Similarity comparison + threshold-based identity decision.
                raw_result = predict_with_prototypes(
                    query_embedding=embedding,
                    prototypes=prototypes,
                    class_names=class_names,
                    metric=metric,
                    threshold=effective_threshold,
                )

                # ── Majority-vote stabilisation ───────────────────────
                # Replace the raw single-frame prediction with the label that
                # won the majority vote over the last `vote_frames` frames.
                voted_name, voted_conf = voter.vote(
                    box=det.box,
                    name=str(raw_result["name"]),
                    confidence=float(raw_result["confidence"]),
                )
                result = dict(raw_result)   # copy to avoid mutating original
                result["name"]       = voted_name
                result["confidence"] = voted_conf

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
        default=0.60,
        help="Threshold for known/unknown decision",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "facenet", "onnx", "insightface"],
        help="Embedding backend (insightface = ArcFace, highest accuracy)",
    )
    parser.add_argument(
        "--onnx-model-path",
        default="models/arcface.onnx",
        help="Path to ONNX face embedding model",
    )
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam device id")
    parser.add_argument(
        "--vote-frames",
        type=int,
        default=7,
        help="Majority-vote window: number of consecutive frames used to stabilise the decision (default: 7)",
    )
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
        vote_frames=args.vote_frames,
    )
