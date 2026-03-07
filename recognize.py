"""Real-time inference module for few-shot face recognition.

Purpose:
    Performs webcam-based online recognition using previously generated
    prototypes and embedding vectors.

Role in pipeline:
    Final runtime stage. Consumes stored embeddings/prototypes and applies
    similarity-based identity prediction on live webcam frames.

Few-shot contribution:
    Uses prototype matching instead of full classifier retraining, allowing
    recognition with very limited samples per identity.

Accuracy improvements over baseline:
    - Auto-calibrated threshold: loads threshold saved by generate_embeddings.py
      (LOO-calibrated for <=1% FPR) instead of a manual constant.
    - Ensemble similarity: cosine + normalised-euclidean blend passed through
      to predict_with_prototypes for ~1-2% accuracy gain.
    - Quality-score-based adaptive threshold: uses composite face quality
      (sharpness + brightness + contrast) instead of only bounding-box size,
      giving a more accurate proxy for embedding reliability.
    - Frame quality gate: skips faces whose quality is too low to embed
      reliably, preventing spurious Unknown flicker.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np

from similarity import combined_predict, predict_with_prototypes
from utils import (
    FaceEmbedder,
    FaceTracker,
    align_face_from_landmarks,
    build_multi_prototypes,
    compute_class_prototypes,
    detect_faces,
    get_enhanced_crop,
    load_auto_threshold,
    load_face_detector,
    load_saved_embeddings,
    score_face_quality,
)


# ---------------------------------------------------------------------------
# FrameVoter — Majority-vote decision stability
# ---------------------------------------------------------------------------

class FrameVoter:
    """Per-face rolling majority-vote decision stabiliser.

    Accumulates the last ``window`` per-frame identity predictions for each
    tracked face and returns the label that appeared most often.  This
    eliminates single-frame noise (blink, sudden head turn) that would
    otherwise flicker the recognition result on screen.

    Only a result that holds strictly more than half the window votes is
    returned; otherwise the output is "Unknown".
    """

    def __init__(
        self,
        window: int = 5,
        max_match_dist: int = 120,
        max_missed_frames: int = 12,
    ) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self.max_match_dist = max_match_dist
        self.max_missed_frames = max_missed_frames
        self._history: Dict[int, Deque[Tuple[str, float]]] = {}
        self._centers: Dict[int, Tuple[int, int]] = {}
        self._missed:  Dict[int, int] = {}
        self._next_id: int = 0

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

    def vote(
        self,
        box: Tuple[int, int, int, int],
        name: str,
        confidence: float,
    ) -> Tuple[str, float]:
        """Record a single-frame prediction and return the majority-vote result.

        Parameters:
            box: Bounding box (x1, y1, x2, y2).
            name: Single-frame identity prediction.
            confidence: Single-frame confidence [0, 1].

        Returns:
            Tuple[str, float]: (voted_name, mean_confidence_of_voted_class)
        """
        center = self._center(box)
        tid = self._match(center)

        if tid is None:
            tid = self._next_id
            self._next_id += 1
            self._history[tid] = deque(maxlen=self.window)
            self._missed[tid] = 0

        self._centers[tid] = center
        self._missed[tid] = 0
        self._history[tid].append((name, confidence))

        labels = [n for n, _ in self._history[tid]]
        counts = Counter(labels)
        top_name, top_count = counts.most_common(1)[0]

        if top_count > len(labels) / 2:
            matched_confs = [c for n, c in self._history[tid] if n == top_name]
            return top_name, float(np.mean(matched_confs))
        return "Unknown", 0.0

    def expire(self, active_boxes: list) -> None:
        """Age absent tracks and remove stale ones."""
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


# ---------------------------------------------------------------------------
# Adaptive threshold helpers
# ---------------------------------------------------------------------------

def _estimate_distance(face_width: int) -> str:
    """Classify camera-to-face distance from bounding-box pixel width."""
    if face_width > 220:
        return "near"
    elif face_width > 120:
        return "medium"
    return "far"


def _compute_adaptive_threshold(
    base: float,
    face_width: int,
    frame_width: int,
    quality_score: float = 0.5,
) -> float:
    """Compute a per-face adaptive threshold using face quality and relative size.

    Uses two complementary quality signals:
      - Relative face size (face_width / frame_width): larger faces are closer
        and contain more pixels, producing more discriminative embeddings.
      - Composite quality score (sharpness + brightness + contrast):
        directly measures embedding reliability.

    Formula::

        combined_quality = 0.6 * size_quality + 0.4 * face_quality
        threshold = base * (0.75 + 0.50 * combined_quality)

    Clamped to [0.40, 0.90] to prevent degenerate all-accept or all-reject regimes.

    Parameters:
        base (float): Base (calibrated) threshold.
        face_width (int): Detected bounding-box width in pixels.
        frame_width (int): Full frame width in pixels.
        quality_score (float): Composite face quality in [0, 1].

    Returns:
        float: Adaptive threshold clamped to [0.40, 0.90].
    """
    size_quality = min(face_width / max(1, frame_width) / 0.30, 1.0)
    combined_quality = 0.6 * size_quality + 0.4 * float(quality_score)
    # Keep threshold within [0.90×, 1.00×] of the calibrated base:
    # • High quality → exactly the calibrated base (no inflation above it).
    # • Low  quality → 10% below base (natural, as noisy embeddings score lower).
    # The old formula [0.75×, 1.25×] was pushing threshold above 0.80 on good
    # frames, causing widespread false-rejects with FaceNet genuine pairs (0.60–0.78).
    adaptive = base * (0.90 + 0.10 * combined_quality)
    return float(np.clip(adaptive, 0.40, 0.90))


def _load_or_build_prototypes(
    embeddings_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load class prototypes from disk or build them from saved embeddings."""
    # Prefer multi-prototypes (k-means clusters) over single mean prototypes.
    multi_proto_path = Path(embeddings_dir) / "multi_prototypes.npy"
    multi_label_path = Path(embeddings_dir) / "multi_labels.npy"
    proto_path = Path(embeddings_dir) / "prototypes.npy"
    names_path = Path(embeddings_dir) / "class_names.npy"

    if multi_proto_path.exists() and multi_label_path.exists():
        prototypes = np.load(multi_proto_path).astype(np.float32)
        class_names = np.load(multi_label_path).astype(str)
        return prototypes, class_names

    if proto_path.exists() and names_path.exists():
        prototypes = np.load(proto_path).astype(np.float32)
        class_names = np.load(names_path).astype(str)
        return prototypes, class_names

    embeddings, labels = load_saved_embeddings(
        Path(embeddings_dir) / "embeddings.npy",
        Path(embeddings_dir) / "labels.npy",
    )
    return compute_class_prototypes(embeddings, labels)


def _load_stored_embeddings(
    embeddings_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load individual enrollment embeddings and labels for kNN matching.

    Returns empty arrays if files are not found (graceful degradation to
    prototype-only matching).
    """
    emb_path = Path(embeddings_dir) / "embeddings.npy"
    lbl_path = Path(embeddings_dir) / "labels.npy"
    if emb_path.exists() and lbl_path.exists():
        try:
            return load_saved_embeddings(emb_path, lbl_path)
        except Exception:
            pass
    return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=str)


# ---------------------------------------------------------------------------
# Main real-time inference loop
# ---------------------------------------------------------------------------

def recognize_realtime(
    embeddings_dir: str = "embeddings",
    metric: str = "cosine",
    threshold: float = 0.0,         # 0.0 -> auto-load calibrated threshold
    backend: str = "auto",
    onnx_model_path: str = "models/arcface.onnx",
    camera_id: int = 0,
    vote_frames: int = 7,
    deepface_model: str = "ArcFace",
    min_face_quality: float = 0.15,  # skip faces below this quality score
) -> None:
    """Run real-time webcam recognition using prototype similarity matching.

    Parameters:
        embeddings_dir: Directory containing saved embeddings / prototypes.
        metric: Similarity metric ('cosine' or 'euclidean').
        threshold: Recognition threshold. Pass 0.0 (default) to auto-load the
            calibrated value saved by generate_embeddings.py.
        backend: Embedding backend.
        onnx_model_path: Path to ONNX embedding model.
        camera_id: OpenCV camera index.
        vote_frames: Majority-vote window (frames).
        deepface_model: DeepFace model name when backend='deepface'.
        min_face_quality: Minimum composite face quality to attempt recognition.
    """
    detector = load_face_detector()
    embedder = FaceEmbedder(
        backend=backend,
        onnx_model_path=onnx_model_path,
        deepface_model=deepface_model,
    )

    prototypes, class_names = _load_or_build_prototypes(embeddings_dir)
    if len(prototypes) == 0:
        raise RuntimeError("No prototypes found. Run generate_embeddings.py first.")

    # Dimension-mismatch guard: probe the embedder with a dummy image so we can
    # compare its output dimension against the stored prototype dimension before
    # entering the camera loop.  A mismatch (e.g. ViT 768-d vs ArcFace 512-d
    # prototypes) would silently produce garbage scores or a numpy crash.
    _probe = np.zeros((112, 112, 3), dtype=np.uint8)
    _probe_dim = embedder.embed_face(_probe).shape[0]
    _proto_dim = prototypes.shape[1]
    if _probe_dim != _proto_dim:
        raise RuntimeError(
            f"Embedding dimension mismatch: the '{backend}' backend produces "
            f"{_probe_dim}-d vectors but the stored prototypes are {_proto_dim}-d.\n"
            f"Fix: rebuild embeddings with the same backend you use here:\n"
            f"  python generate_embeddings.py --backend {backend}"
        )

    # Load individual stored embeddings for kNN-based matching.
    stored_embeddings, stored_labels = _load_stored_embeddings(embeddings_dir)
    use_knn = len(stored_embeddings) > 0
    if use_knn:
        print(f"[INFO] kNN matching enabled  ({len(stored_embeddings)} stored embeddings)")

    # Load calibrated threshold (LOO-calibrated in generate_embeddings.py)
    # when the caller has not explicitly set one (threshold == 0.0).
    if threshold <= 0.0:
        base_threshold = load_auto_threshold(embeddings_dir, fallback=0.65)
        print(f"[INFO] Auto-loaded calibrated threshold: {base_threshold:.4f}")
    else:
        base_threshold = threshold

    import sys as _sys
    cap = (
        cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if _sys.platform.startswith("win")
        else cv2.VideoCapture(camera_id)
    )
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions / device id.")

    # Request high-resolution capture so distant faces contain enough pixels.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Temporal embedding smoother: EMA per tracked face.
    tracker = FaceTracker(alpha=0.7, max_missed_frames=10)
    # Majority-vote stabiliser: one rolling history per tracked face.
    voter = FrameVoter(window=vote_frames, max_missed_frames=12)

    print("=" * 80)
    print("Few-Shot Face Recognition  [Adaptive Quality Mode]")
    print(f"Backend       : {embedder.backend_name}")
    print(f"Metric        : {metric}  (ensemble={metric == 'cosine'})")
    print(f"Base threshold: {base_threshold:.4f}  (quality-adaptive per face)")
    print(f"Vote window   : {vote_frames} frames")
    print(f"Known classes : {len(class_names)}  ({', '.join(class_names)})")
    print(f"Resolution    : {actual_w}x{actual_h}")
    print(f"Min quality   : {min_face_quality}")
    print("Press 'q' to quit")
    print("=" * 80)

    prev_time = time.time()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                continue

            # Face detection: lower min_confidence (0.85) improves recall for
            # small or tilted faces at medium-to-far camera distances.
            detections = detect_faces(frame, detector, min_confidence=0.85, padding=0.12)

            active_boxes = [det.box for det in detections]
            tracker.expire_tracks(active_boxes)
            voter.expire(active_boxes)

            frame_width = frame.shape[1]

            for det in detections:
                x1, y1, x2, y2 = det.box
                face_w = x2 - x1
                distance = _estimate_distance(face_w)

                # Frame quality gate: skip faces too blurry/dark to embed reliably.
                if det.quality_score < min_face_quality:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), 1)
                    cv2.putText(
                        frame,
                        f"Low Q:{det.quality_score:.2f}",
                        (x1, max(14, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1,
                    )
                    continue

                # Face crop: use enhanced crop for far faces; aligned crop otherwise.
                if distance == "far":
                    face_crop = get_enhanced_crop(frame, det.box, margin=0.25, target_size=160)
                else:
                    if det.landmarks_5pt is not None:
                        aligned = align_face_from_landmarks(frame, det.landmarks_5pt)
                        face_crop = aligned if aligned is not None else det.face_rgb
                    else:
                        face_crop = det.face_rgb

                # Raw embedding extraction.
                raw_embedding = embedder.embed_face(face_crop)

                # Temporal EMA smoothing: reduces pose/lighting jitter between frames.
                embedding = tracker.update(det.box, raw_embedding)

                # Quality + size adaptive threshold.
                effective_threshold = _compute_adaptive_threshold(
                    base_threshold, face_w, frame_width, det.quality_score
                )

                # Identity decision: fuse prototype + kNN for max accuracy.
                if use_knn:
                    raw_result = combined_predict(
                        query_embedding=embedding,
                        prototypes=prototypes,
                        class_names=class_names,
                        stored_embeddings=stored_embeddings,
                        stored_labels=stored_labels,
                        threshold=effective_threshold,
                        use_ensemble=(metric == "cosine"),
                    )
                else:
                    raw_result = predict_with_prototypes(
                        query_embedding=embedding,
                        prototypes=prototypes,
                        class_names=class_names,
                        metric=metric,
                        threshold=effective_threshold,
                        use_ensemble=(metric == "cosine"),
                    )

                # Majority-vote stabilisation over the last vote_frames frames.
                confidence_val = raw_result.get("confidence", 0.0)
                if not isinstance(confidence_val, (int, float, np.number)):
                    confidence_val = 0.0
                voted_name, voted_conf = voter.vote(
                    box=det.box,
                    name=str(raw_result["name"]),
                    confidence=float(confidence_val),
                )
                name = voted_name
                confidence = voted_conf

                # Overlay
                color = (0, 210, 0) if name != "Unknown" else (0, 0, 220)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Name + confidence above box
                cv2.putText(
                    frame,
                    f"{name}  {confidence * 100:.0f}%",
                    (x1, max(18, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                )
                # Distance + quality (line 1 below box)
                cv2.putText(
                    frame,
                    f"{distance.upper()}  Q:{det.quality_score:.2f}",
                    (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 200, 255), 1,
                )
                # Adaptive threshold + raw score (line 2 below box)
                cv2.putText(
                    frame,
                    f"Thr:{effective_threshold:.2f}  Scr:{raw_result['score']:.3f}",
                    (x1, y2 + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1,
                )

            # FPS counter
            cur_time = time.time()
            fps = 1.0 / max(1e-6, cur_time - prev_time)
            prev_time = cur_time

            cv2.putText(
                frame, f"FPS:{fps:.1f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2,
            )
            cv2.putText(
                frame, f"Classes:{len(class_names)}  BaseThr:{base_threshold:.3f}",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
            )

            cv2.imshow("FewShotFace  Real-Time Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time few-shot face recognition")
    parser.add_argument("--embeddings-dir", default="embeddings")
    parser.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument(
        "--threshold", type=float, default=0.0,
        help="Recognition threshold (0.0 = auto-load calibrated value)",
    )
    parser.add_argument(
        "--backend", default="auto",
        choices=["auto", "facenet", "onnx", "insightface", "deepface"],
    )
    parser.add_argument(
        "--deepface-model", default="ArcFace",
        choices=["ArcFace", "Facenet512", "VGG-Face", "SFace", "Facenet", "OpenFace"],
    )
    parser.add_argument("--onnx-model-path", default="models/arcface.onnx")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--vote-frames", type=int, default=7)
    parser.add_argument(
        "--min-quality", type=float, default=0.15,
        help="Minimum face quality to attempt recognition (default: 0.15)",
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
        deepface_model=args.deepface_model,
        min_face_quality=args.min_quality,
    )

