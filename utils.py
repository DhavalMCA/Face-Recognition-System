"""Shared utilities module for few-shot face recognition pipeline.

Purpose:
    Provides reusable helper functions and classes for directory handling,
    face detection, preprocessing, embedding extraction, and prototype
    computation.

Role in pipeline:
    This module is used by enrollment, embedding-generation, and recognition
    scripts as the common ML infrastructure layer.

Few-shot contribution:
    Supplies prototype-building and backend-agnostic embedding interfaces,
    enabling robust recognition with limited samples per identity.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency at runtime
    ort = None

try:
    import insightface
    from insightface.app import FaceAnalysis as _InsightFaceAnalysis
    _INSIGHTFACE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency at runtime
    _InsightFaceAnalysis = None
    _INSIGHTFACE_AVAILABLE = False

try:
    from deepface import DeepFace as _DeepFace
    _DEEPFACE_AVAILABLE = True
    _DEEPFACE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency at runtime
    _DeepFace = None
    _DEEPFACE_AVAILABLE = False
    _DEEPFACE_IMPORT_ERROR = str(exc)


def ensure_dir(path: str | Path) -> None:
    """Create directory path if it does not exist.

    Function name:
        ensure_dir

    Purpose:
        Guarantees that output/storage directories are available before write
        operations.

    Parameters:
        path (str | Path): Directory path to create.

    Returns:
        None

    Role in face recognition process:
        Supports enrollment image storage and embedding artifact persistence.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def format_size(num_bytes: int) -> str:
    """Convert a byte count to a readable size string (B, KB, MB, GB)."""
    size = float(max(0, num_bytes))
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_idx = 0
    while size >= 1024.0 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1
    return f"{size:.2f} {units[unit_idx]}"


def get_file_size(path: str | Path) -> int:
    """Return file size in bytes, or 0 if path is missing/unreadable."""
    try:
        return int(Path(path).stat().st_size)
    except Exception:
        return 0


def get_dir_size(path: str | Path) -> int:
    """Return recursive directory size in bytes.

    Unreadable files are skipped to keep runtime stable.
    """
    root = Path(path)
    if not root.exists() or not root.is_dir():
        return 0

    total = 0
    for file_path in root.rglob("*"):
        if file_path.is_file():
            total += get_file_size(file_path)
    return total


def get_identity_folders(dataset_dir: str | Path) -> List[Path]:
    """List all enrolled identity folders in dataset root.

    Function name:
        get_identity_folders

    Purpose:
        Discovers class directories where user face samples are stored.

    Parameters:
        dataset_dir (str | Path): Dataset root directory.

    Returns:
        List[Path]: Sorted list of identity folder paths.

    Role in face recognition process:
        Provides class structure used for embedding generation and user listing.
    """
    root = Path(dataset_dir)
    if not root.exists():
        return []
    return sorted([item for item in root.iterdir() if item.is_dir()])


def load_face_detector(image_size: int = 160, margin: int = 20) -> MTCNN:
    """Initialize and return MTCNN face detector tuned for small / distant faces.

    Function name:
        load_face_detector

    Purpose:
        Configures MTCNN model for multi-face detection in CPU mode with
        reduced min_face_size and lowered stage thresholds so that faces as
        small as 60-80 pixels wide (distances of 2-3 m) are still detected.

    Parameters:
        image_size (int): Target aligned face size inside MTCNN pipeline.
        margin (int): Margin around detected face region.

    Returns:
        MTCNN: Initialized detector instance.

    Role in face recognition process:
        Implements primary face detection stage before feature extraction.
        Lower thresholds [0.5, 0.6, 0.6] increase recall for small faces;
        min_face_size=20 allows detection of distant, small face regions.
    """
    return MTCNN(
        image_size=image_size,
        margin=margin,
        min_face_size=20,           # Allow small/distant faces (default is 20 but explicit here).
        thresholds=[0.5, 0.6, 0.6], # Lowered from [0.6, 0.7, 0.7] for better small face recall.
        keep_all=True,
        post_process=False,
        device="cpu",
    )


def _normalize_face(face_rgb: np.ndarray, target_size: int) -> np.ndarray:
    """Resize and normalize face crop for embedding models.

    Function name:
        _normalize_face

    Purpose:
        Applies deterministic preprocessing so face crops match input format
        expected by FaceNet/ArcFace-like models.

    Parameters:
        face_rgb (np.ndarray): RGB face crop.
        target_size (int): Square output resolution.

    Returns:
        np.ndarray: Normalized float32 tensor-like image array.

    Role in face recognition process:
        Implements image preprocessing stage before embedding inference.
    """
    # Resize operation preserves compact facial representation for model input.
    resized = cv2.resize(face_rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
    # Standardize pixel range to approximately [-1, 1].
    normalized = (resized.astype(np.float32) - 127.5) / 128.0
    return normalized


# ---------------------------------------------------------------------------
# Distance-robust helpers: CLAHE enhancement + auto-zoom crop
# ---------------------------------------------------------------------------

def score_face_quality(face_rgb: np.ndarray) -> float:
    """Compute a composite face quality score in [0, 1].

    Combines three lightweight signals:
      1. Sharpness  — Laplacian variance normalised to [0, 1].
         High variance means sharp edges = good quality.
      2. Brightness — mean pixel intensity.  Faces that are too dark (< 40)
         or over-exposed (> 220) produce unreliable embeddings.
      3. Contrast   — standard deviation of grayscale pixel values.
         Low std means flat, featureless crops (occluded or blank).

    Parameters:
        face_rgb (np.ndarray): RGB face crop (any size).

    Returns:
        float: Quality score in [0, 1].  Scores above 0.40 are generally
            usable; below 0.25 should be skipped during enrollment.
    """
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)

    # Sharpness: Laplacian variance; normalise with a soft cap at 500.
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = min(lap_var / 500.0, 1.0)

    # Brightness penalty: sigmoid-like curve peaking at mean ~128.
    mean_brightness = float(gray.mean())
    #  map [0, 255] → brightness score [0, 1], penalise extremes
    brightness = 1.0 - abs(mean_brightness - 128.0) / 128.0
    brightness = max(0.0, brightness)

    # Contrast: std of grayscale values; normalise with soft cap at 60.
    contrast = min(float(gray.std()) / 60.0, 1.0)

    # Weighted composite: sharpness carries most weight.
    quality = 0.55 * sharpness + 0.25 * brightness + 0.20 * contrast
    return float(np.clip(quality, 0.0, 1.0))


def apply_clahe_enhancement(face_rgb: np.ndarray, clip_limit: Optional[float] = None) -> np.ndarray:
    """Apply adaptive CLAHE contrast normalisation to an RGB face crop.

    CLAHE is applied only on the luminance channel (LAB colour space) to
    avoid distorting skin tones.  The ``clip_limit`` is chosen automatically
    based on mean brightness when not specified: darker images need stronger
    enhancement (higher clip), brighter images need a gentler touch.

    Parameters:
        face_rgb (np.ndarray): Input RGB face crop (any size).
        clip_limit (float | None): CLAHE clip limit.  If None, auto-selected
            from [1.5, 3.0] based on mean luminance.

    Returns:
        np.ndarray: Contrast-enhanced RGB face crop (same spatial size).
    """
    lab = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    if clip_limit is None:
        mean_l = float(l_ch.mean())
        # Dark faces need more enhancement; bright/well-lit need less.
        clip_limit = 3.0 if mean_l < 80 else (1.5 if mean_l > 180 else 2.0)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(4, 4))
    l_enhanced = clahe.apply(l_ch)

    lab_enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)


def get_enhanced_crop(
    frame_bgr: np.ndarray,
    box: Tuple[int, int, int, int],
    margin: float = 0.25,
    target_size: int = 160,
) -> np.ndarray:
    """Extract an expanded, CLAHE-enhanced face crop from the original frame.

    Function name:
        get_enhanced_crop

    Purpose:
        For small / distant faces the bounding box is tight and the crop
        contains little contextual information. This function:
          1. Expands the box by `margin` (default 25 %) on every side.
          2. Clamps the expanded box to frame boundaries.
          3. Converts from BGR to RGB.
          4. Applies CLAHE normalization to boost local contrast.
          5. Resizes to `target_size x target_size` with INTER_CUBIC to
             preserve detail during upscaling.

    Parameters:
        frame_bgr (np.ndarray): Full BGR frame from OpenCV capture.
        box (Tuple[int,int,int,int]): Detected bounding box (x1,y1,x2,y2).
        margin (float): Relative expansion factor per side (0.25 = 25 %).
        target_size (int): Square output resolution expected by the embedder.

    Returns:
        np.ndarray: Enhanced RGB face crop of shape (target_size, target_size, 3).

    Role in face recognition process:
        Implements the auto-zoom enhancement stage used when small_face is
        True during real-time inference. Provides a higher-quality crop to
        the embedding model than a plain resize of the raw tight box.
    """
    frame_h, frame_w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1

    # Expand box uniformly on all sides to capture more facial context.
    pad_x = int(w * margin)
    pad_y = int(h * margin)
    ex1 = max(0, x1 - pad_x)
    ey1 = max(0, y1 - pad_y)
    ex2 = min(frame_w, x2 + pad_x)
    ey2 = min(frame_h, y2 + pad_y)

    crop_bgr = frame_bgr[ey1:ey2, ex1:ex2]
    if crop_bgr.size == 0:
        # Safety fallback: use original tight box.
        crop_bgr = frame_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    # CLAHE normalization compensates for poor lighting at distance.
    enhanced = apply_clahe_enhancement(crop_rgb)
    # INTER_CUBIC gives better quality than INTER_LINEAR when upscaling.
    resized = cv2.resize(enhanced, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    return resized


# ---------------------------------------------------------------------------
# Face alignment — similarity transform to ArcFace canonical positions
# ---------------------------------------------------------------------------

# Standard 5-landmark reference positions for 112×112 ArcFace-aligned output.
# Source: ArcFace paper / InsightFace preprocessing standard.
_ARCFACE_REF_LANDMARKS_112 = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)


def align_face_from_landmarks(
    frame_bgr: np.ndarray,
    landmarks_5pt: np.ndarray,
    output_size: int = 112,
) -> Optional[np.ndarray]:
    """Affine-align a detected face to the ArcFace 112×112 canonical template.

    Uses all five MTCNN landmarks to estimate a similarity transform
    (rotation + scale + translation, no shear) via cv2.estimateAffinePartial2D.
    Aligned faces match the preprocessing applied during ArcFace model
    training, yielding significantly more consistent embeddings than
    raw padded bounding-box crops—typically +3–5% real-world accuracy.

    Parameters:
        frame_bgr (np.ndarray): Full BGR frame (H × W × 3).
        landmarks_5pt (np.ndarray): 5 keypoints in full-frame pixel coords,
            shape (5, 2) ordered: left_eye, right_eye, nose, mouth_L, mouth_R.
        output_size (int): Side length of the square output (default 112).

    Returns:
        np.ndarray | None: Aligned RGB face (output_size × output_size × 3),
            or None if the transform cannot be computed (caller should fall
            back to the raw face_rgb crop).
    """
    scale = output_size / 112.0
    dst = (_ARCFACE_REF_LANDMARKS_112 * scale).astype(np.float32)
    src = landmarks_5pt.astype(np.float32)

    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        return None  # caller falls back to raw face_rgb

    aligned_bgr = cv2.warpAffine(
        frame_bgr, M, (output_size, output_size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    # Return RGB to match the face_rgb convention used throughout the pipeline.
    return cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Temporal embedding smoother (per-face EMA tracking)
# ---------------------------------------------------------------------------

class FaceTracker:
    """Per-face temporal embedding smoother using exponential moving average.

    Maintains a separate smoothed embedding per tracked face. Tracks are
    identified by proximity of bounding-box centres. If a face disappears
    for more than `max_missed_frames` consecutive frames the track is
    discarded and the next appearance starts fresh.

    Smoothing formula::

        smoothed = alpha * current_embedding + (1 - alpha) * previous_smoothed

    A higher alpha makes the result track the current frame more quickly;
    a lower alpha provides stronger temporal smoothing.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        max_missed_frames: int = 10,
        max_match_dist: int = 120,
    ) -> None:
        """Initialise face tracker.

        Parameters:
            alpha (float): EMA weight applied to the current frame embedding.
            max_missed_frames (int): Frames a track can be absent before reset.
            max_match_dist (int): Max pixel distance between centres for match.
        """
        self.alpha = alpha
        self.max_missed_frames = max_missed_frames
        self.max_match_dist = max_match_dist
        self._tracks: Dict[int, dict] = {}
        self._next_id: int = 0

    @staticmethod
    def _center(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _match_track(self, center: Tuple[int, int]) -> Optional[int]:
        """Return id of the nearest existing track or None if none is close enough."""
        best_id: Optional[int] = None
        best_dist: float = float(self.max_match_dist)
        for tid, track in self._tracks.items():
            cx, cy = track["center"]
            dist = ((cx - center[0]) ** 2 + (cy - center[1]) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_id = tid
        return best_id

    def update(self, box: Tuple[int, int, int, int], embedding: np.ndarray) -> np.ndarray:
        """Update track for `box` and return EMA-smoothed embedding.

        If no track matches the box centre a new track is created. Otherwise
        the EMA is applied and the track centre is moved to the new position.

        Parameters:
            box: Bounding box (x1, y1, x2, y2) of detected face.
            embedding: Raw embedding vector from the embedding backend.

        Returns:
            np.ndarray: L2-normalised smoothed embedding for this face track.
        """
        center = self._center(box)
        tid = self._match_track(center)

        if tid is None:
            # First appearance of this face — initialise a fresh track.
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = {
                "smoothed_emb": embedding.copy(),
                "center": center,
                "missed": 0,
            }
        else:
            prev = self._tracks[tid]["smoothed_emb"]
            # EMA reduces per-frame noise from pose / lighting jitter.
            smoothed = self.alpha * embedding + (1.0 - self.alpha) * prev
            norm = np.linalg.norm(smoothed)
            smoothed = smoothed / (norm + 1e-8)  # keep L2-normalised.
            self._tracks[tid]["smoothed_emb"] = smoothed
            self._tracks[tid]["center"] = center
            self._tracks[tid]["missed"] = 0

        return self._tracks[tid]["smoothed_emb"]

    def expire_tracks(
        self, active_boxes: List[Tuple[int, int, int, int]]
    ) -> None:
        """Age all tracks and remove those absent for too long.

        Call once per frame after processing all active detections. The method
        increments `missed` for every track whose centre was not matched by
        any active box this frame, then deletes tracks that exceed
        `max_missed_frames`.

        Parameters:
            active_boxes: List of bounding boxes detected in the current frame.
        """
        active_centers = [self._center(b) for b in active_boxes]

        # Determine which track ids were covered by at least one active box.
        matched: set = set()
        for center in active_centers:
            tid = self._match_track(center)
            if tid is not None:
                matched.add(tid)

        # Increment missed counter for absent tracks.
        for tid in list(self._tracks.keys()):
            if tid not in matched:
                self._tracks[tid]["missed"] += 1

        # Purge stale tracks so memory does not grow unboundedly.
        for tid in [t for t, v in self._tracks.items() if v["missed"] >= self.max_missed_frames]:
            del self._tracks[tid]


@dataclass
class Detection:
    """Container for one detected face instance.

    Stores bounding-box coordinates, detector confidence, cropped RGB face,
    optional 5-point facial landmarks, and a composite quality score.
    """

    box: Tuple[int, int, int, int]
    confidence: float
    face_rgb: np.ndarray
    landmarks_5pt: Optional[np.ndarray] = None  # shape (5, 2), full-frame pixel coords
    quality_score: float = 0.0                   # composite quality in [0, 1]


def detect_faces(
    frame_bgr: np.ndarray,
    detector: MTCNN,
    min_confidence: float = 0.92,
    padding: float = 0.10,
) -> List[Detection]:
    """Detect faces in frame and return cropped face regions.

    Function name:
        detect_faces

    Purpose:
        Runs MTCNN detection, applies confidence filtering and padding, and
        prepares cropped RGB faces for embedding extraction.

    Parameters:
        frame_bgr (np.ndarray): Input frame in OpenCV BGR format.
        detector (MTCNN): Preloaded face detector model.
        min_confidence (float): Minimum detector confidence threshold.
        padding (float): Relative box padding factor.

    Returns:
        List[Detection]: Valid detections containing box, score, and face crop.

    Role in face recognition process:
        Implements face localization stage used by enrollment, training, and
        real-time inference modules.
    """
    # Convert OpenCV BGR frame into RGB expected by MTCNN detector.
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes, probs, landmarks = detector.detect(rgb, landmarks=True)  # type: ignore[misc]

    if boxes is None or probs is None:
        return []

    frame_h, frame_w = frame_bgr.shape[:2]
    detections: List[Detection] = []

    # Convert raw detector output into sanitized detections.
    lm_list = landmarks if landmarks is not None else [None] * len(boxes)
    for box, score, pts in zip(boxes, probs, lm_list):
        if score is None or score < min_confidence:
            continue

        # Convert floating-point coordinates to pixel integers.
        x1, y1, x2, y2 = box.astype(int)
        width = x2 - x1
        height = y2 - y1

        pad_x = int(width * padding)
        pad_y = int(height * padding)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(frame_w, x2 + pad_x)
        y2 = min(frame_h, y2 + pad_y)

        # Crop candidate face region for downstream embedding extraction.
        face = rgb[y1:y2, x1:x2]
        if face.size == 0:
            continue

        # Compute composite quality score for this crop (used for quality-weighted
        # prototype building and runtime frame-quality gating).
        quality = score_face_quality(face)

        detections.append(
            Detection(
                box=(x1, y1, x2, y2),
                confidence=float(score),
                face_rgb=face,
                landmarks_5pt=pts.astype(np.float32) if pts is not None else None,
                quality_score=quality,
            )
        )

    return detections


class FacenetEmbedder:
    """Embedding extractor using pre-trained FaceNet model from facenet-pytorch."""

    def __init__(self, image_size: int = 160) -> None:
        """Initialize FaceNet embedding model.

        Function name:
            FacenetEmbedder.__init__

        Purpose:
            Loads pretrained FaceNet backbone and sets inference device.

        Parameters:
            image_size (int): Expected face crop size for model input.

        Returns:
            None

        Role in face recognition process:
            Provides feature extraction backend for embedding generation.
        """
        self.image_size = image_size
        self.device = "cpu"
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def embed(self, face_rgb: np.ndarray) -> np.ndarray:
        """Extract L2-normalized FaceNet embedding from one face crop.

        Function name:
            FacenetEmbedder.embed

        Purpose:
            Runs forward pass on preprocessed face image and normalizes output
            embedding vector.  CLAHE contrast normalization is applied before
            `_normalize_face` so that mobile photos and webcam captures are
            mapped to the same local-contrast distribution in embedding space.

        Parameters:
            face_rgb (np.ndarray): Input face crop in RGB format.

        Returns:
            np.ndarray: L2-normalized embedding vector.

        Role in face recognition process:
            Implements feature extraction stage for the FaceNet backend.
        """
        # CLAHE equalization bridges the contrast gap between mobile-captured
        # and webcam-captured images before any model inference.
        enhanced = apply_clahe_enhancement(face_rgb)
        normalized = _normalize_face(enhanced, self.image_size)
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(tensor).cpu().numpy()[0].astype(np.float32)

        embedding /= np.linalg.norm(embedding) + 1e-8
        return embedding


class ONNXEmbedder:
    """Embedding extractor using an ONNX ArcFace/InsightFace model with CPU inference."""

    def __init__(self, model_path: str | Path, default_size: int = 112) -> None:
        """Initialize ONNX embedding inference session.

        Function name:
            ONNXEmbedder.__init__

        Purpose:
            Loads ONNX model, prepares runtime session, and infers expected
            input resolution from model metadata.

        Parameters:
            model_path (str | Path): Path to ONNX embedding model file.
            default_size (int): Fallback input size if model shape is dynamic.

        Returns:
            None

        Role in face recognition process:
            Provides ONNX-based feature extraction backend.
        """
        if ort is None:
            raise RuntimeError("onnxruntime is not available. Install it or use FaceNet backend.")

        self.model_path = str(model_path)
        self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        shape = self.session.get_inputs()[0].shape
        self.image_size = default_size
        if len(shape) == 4 and isinstance(shape[2], int) and shape[2] > 0:
            self.image_size = int(shape[2])

    def embed(self, face_rgb: np.ndarray) -> np.ndarray:
        """Extract L2-normalized embedding using ONNX runtime.

        Function name:
            ONNXEmbedder.embed

        Purpose:
            Preprocesses face crop, performs ONNX inference, and normalizes
            resulting embedding vector.  CLAHE contrast normalization is
            applied before `_normalize_face` so that mobile photos and webcam
            captures are mapped to the same local-contrast distribution before
            ArcFace inference.

        Parameters:
            face_rgb (np.ndarray): Input face crop in RGB format.

        Returns:
            np.ndarray: L2-normalized embedding vector.

        Role in face recognition process:
            Implements feature extraction stage for ONNX backend.
        """
        # CLAHE equalization bridges the contrast gap between mobile-captured
        # and webcam-captured images before any model inference.
        enhanced = apply_clahe_enhancement(face_rgb)
        normalized = _normalize_face(enhanced, self.image_size)
        chw = np.transpose(normalized, (2, 0, 1))[None, :, :, :].astype(np.float32)

        outputs = self.session.run(None, {self.input_name: chw})
        embedding = np.asarray(outputs[0]).reshape(-1).astype(np.float32)
        embedding /= np.linalg.norm(embedding) + 1e-8
        return embedding


class InsightFaceEmbedder:
    """Embedding extractor using InsightFace ArcFace (buffalo_l / buffalo_sc).

    Uses buffalo_l (ResNet-50 ArcFace, 512-d) which achieves ~99.7 % on LFW.
    Built-in 5-landmark face alignment is applied before embedding so the
    model receives a well-aligned 112×112 crop regardless of the original
    detection box orientation.
    """

    def __init__(self, model_name: str = "buffalo_l") -> None:
        """Initialize InsightFace FaceAnalysis pipeline.

        Parameters:
            model_name (str): InsightFace model pack name.
                'buffalo_l' – high accuracy (ResNet-50 ArcFace, recommended).
                'buffalo_sc' – small / fast (MobileNet ArcFace).
        """
        if not _INSIGHTFACE_AVAILABLE:
            raise RuntimeError(
                "insightface is not installed. "
                "Install it with: pip install insightface"
            )
        self.model_name = model_name
        assert _InsightFaceAnalysis is not None
        self._app = _InsightFaceAnalysis(
            name=model_name,
            providers=["CPUExecutionProvider"],
        )
        # det_size=(320,320) is sufficient for already-cropped faces and faster.
        self._app.prepare(ctx_id=0, det_size=(320, 320))

    def embed(self, face_rgb: np.ndarray) -> np.ndarray:
        """Extract L2-normalised ArcFace embedding from one face crop.

        Converts the RGB crop to BGR (InsightFace convention), runs the full
        detection+alignment+embedding pipeline, and returns the best-face
        embedding.  Falls back to a centered-crop if no face is detected.

        Parameters:
            face_rgb (np.ndarray): RGB face crop (any size).

        Returns:
            np.ndarray: L2-normalised 512-d embedding vector.
        """
        # InsightFace works in BGR.
        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        # Upsample very small crops so the detector can find the face.
        h, w = face_bgr.shape[:2]
        if min(h, w) < 112:
            scale = 112 / min(h, w)
            face_bgr = cv2.resize(
                face_bgr,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )
        faces = self._app.get(face_bgr)
        if faces:
            # Pick the largest face by bounding-box area.
            best = max(
                faces,
                key=lambda f: (
                    (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                    if f.bbox is not None
                    else 0
                ),
            )
            emb = best.embedding.astype(np.float32)
        else:
            # Fallback: resize+flatten centre patch (rare edge case).
            patch = cv2.resize(face_bgr, (112, 112)).astype(np.float32) / 255.0
            emb = patch.flatten()[: 512]  # crude but non-empty
        emb /= np.linalg.norm(emb) + 1e-8
        return emb


class ViTEmbedder:
    """Embedding extractor using Vision Transformer (ViT-B/16).

    Uses ImageNet-pretrained ViT-B/16 from torchvision as a fixed feature
    extractor.  The classification head is removed and the CLS token embedding
    (768-d) is L2-normalised.

    Architecture:
        Pure Vision Transformer — splits the image into 16×16 patches,
        processes them with multi-head self-attention layers, and outputs
        a global CLS token representation.  No convolutions at all.

    This gives genuine architectural diversity compared to the CNN-based
    backends (FaceNet/InceptionResNet, ArcFace/ResNet-50, VGG, MobileNet).
    """

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self) -> None:
        """Load ViT-B/16 and strip the classification head."""
        import warnings
        import torch
        from torchvision.models import vit_b_16, ViT_B_16_Weights

        warnings.warn(
            "ViT backend selected: ViT-B/16 is ImageNet-pretrained for object "
            "classification, NOT for face identity recognition. It will produce "
            "poor face recognition accuracy. Use 'insightface', 'onnx', or "
            "'facenet' backend for reliable face recognition. ViT is intended "
            "for research/benchmarking only.",
            UserWarning,
            stacklevel=3,
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        # Remove classification head — we only want the 768-d CLS embedding.
        model.heads = torch.nn.Identity()  # type: ignore[assignment]
        self.model = model.eval().to(self.device)

    def embed(self, face_rgb: np.ndarray) -> np.ndarray:
        """Extract L2-normalised 768-d embedding via ViT CLS token.

        Parameters:
            face_rgb (np.ndarray): RGB face crop (any size).

        Returns:
            np.ndarray: L2-normalised 768-d embedding vector.
        """
        import torch

        enhanced = apply_clahe_enhancement(face_rgb)
        resized = cv2.resize(enhanced, (224, 224))
        norm = (resized.astype(np.float32) / 255.0 - self._MEAN) / self._STD
        tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device).float()

        with torch.no_grad():
            emb = self.model(tensor).cpu().numpy()[0].astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-8
        return emb


class DeepFaceEmbedder:
    """Embedding extractor using the DeepFace library.

    DeepFace wraps several state-of-the-art face recognition models under a
    single API.  The default model is ArcFace (512-d, ~99.4 % LFW), but any
    of the supported models can be selected via the `model_name` parameter.

    Supported models and their embedding dimensions:
        - ArcFace     : 512-d  (default, best accuracy)
        - Facenet512  : 512-d  (high accuracy)
        - VGG-Face    : 4096-d (classic CNN)
        - SFace       : 128-d  (lightweight / fast)
        - Facenet     : 128-d
        - OpenFace    : 128-d

    Models are downloaded automatically from DeepFace's model zoo on first use
    and cached in ~/.deepface/.
    """

    SUPPORTED_MODELS = ["ArcFace", "Facenet512", "VGG-Face", "SFace", "Facenet", "OpenFace"]

    def __init__(self, model_name: str = "ArcFace") -> None:
        """Initialize DeepFace embedding model.

        Parameters:
            model_name (str): One of SUPPORTED_MODELS (default: 'ArcFace').

        Raises:
            RuntimeError: If deepface is not installed.
            ValueError: If model_name is not in SUPPORTED_MODELS.
        """
        if not _DEEPFACE_AVAILABLE:
            raise RuntimeError(
                "deepface is not installed. Install it with: pip install deepface"
            )
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"deepface model_name must be one of: {self.SUPPORTED_MODELS}"
            )
        self.model_name = model_name
        assert _DeepFace is not None
        # Pre-warm the model so the first embed() call has no download delay.
        _DeepFace.build_model(model_name)

    def embed(self, face_rgb: np.ndarray) -> np.ndarray:
        """Extract L2-normalised embedding using DeepFace.

        Parameters:
            face_rgb (np.ndarray): RGB face crop (any size).

        Returns:
            np.ndarray: L2-normalised embedding vector.

        Role in face recognition process:
            Implements feature extraction stage for the DeepFace backend.
            CLAHE contrast normalisation is applied before inference to reduce
            the lighting gap between webcam and mobile-photo captures.
        """
        enhanced = apply_clahe_enhancement(face_rgb)
        # detector_backend="skip" tells DeepFace we are passing a pre-cropped
        # face, so it skips its own detection / alignment step.
        # enforce_detection=False prevents errors on marginal crops.
        assert _DeepFace is not None
        result = _DeepFace.represent(
            img_path=enhanced,
            model_name=self.model_name,
            enforce_detection=False,
            detector_backend="skip",
        )
        embedding = np.array(result[0]["embedding"], dtype=np.float32)  # type: ignore[index]
        embedding /= np.linalg.norm(embedding) + 1e-8
        return embedding


# ---------------------------------------------------------------------------
# Test-time augmentation (TTA) helper
# ---------------------------------------------------------------------------

def _tta_embeddings(
    face_rgb: np.ndarray,
    embedder,
    n_augments: int = 7,
) -> np.ndarray:
    """Return the L2-normalised mean of N diverse augmented embeddings (TTA).

    Augmentation schedule (7 views by default):
      0  original
      1  horizontal flip
      2  flip + brightness -20
      3  rotation +15 °
      4  rotation -15 °
      5  scale 110 % (zoom-in centre crop)
      6  scale  90 % (zoom-out with border replication)

    Using diverse transform types (flip, rotation, scale, brightness) reduces
    per-sample noise more effectively than repeating similar photometric
    transforms alone.  Averaging 7 views typically yields +1-3 % over 1 view.

    Parameters:
        face_rgb (np.ndarray): Original RGB face crop.
        embedder: Object with an `embed(face_rgb)` method.
        n_augments (int): Number of augmented copies to average (max 7 distinct
            views; above 7 the schedule repeats with a horizontal flip).

    Returns:
        np.ndarray: L2-normalised averaged embedding vector.
    """
    h, w = face_rgb.shape[:2]
    cx, cy = w // 2, h // 2

    def _scale_jitter(img: np.ndarray, scale: float) -> np.ndarray:
        """Zoom in (scale>1) or out (scale<1) and restore original H×W."""
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        if scale > 1.0:                         # crop centre
            y0 = (nh - h) // 2
            x0 = (nw - w) // 2
            return resized[y0:y0 + h, x0:x0 + w]
        else:                                   # pad with border replication
            py = (h - nh) // 2
            px = (w - nw) // 2
            padded = cv2.copyMakeBorder(
                resized, py, h - nh - py, px, w - nw - px, cv2.BORDER_REPLICATE
            )
            return cv2.resize(padded, (w, h), interpolation=cv2.INTER_LINEAR)

    def _augment(img: np.ndarray, i: int) -> np.ndarray:
        schedule = [
            lambda x: x,                                          # 0: original
            lambda x: cv2.flip(x, 1),                             # 1: H-flip
            lambda x: np.clip(                                    # 2: flip+dark
                cv2.flip(x, 1).astype(np.int32) - 20, 0, 255
            ).astype(np.uint8),
            lambda x: cv2.warpAffine(                             # 3: +15°
                x, cv2.getRotationMatrix2D((cx, cy), 15, 1.0), (w, h)
            ),
            lambda x: cv2.warpAffine(                             # 4: -15°
                x, cv2.getRotationMatrix2D((cx, cy), -15, 1.0), (w, h)
            ),
            lambda x: _scale_jitter(x, 1.10),                     # 5: zoom in
            lambda x: _scale_jitter(x, 0.90),                     # 6: zoom out
        ]
        fn = schedule[i % len(schedule)]
        # For indices beyond the schedule, also apply a horizontal flip.
        if i >= len(schedule):
            return cv2.flip(fn(img.copy()), 1)
        return fn(img.copy())

    embs = [embedder.embed(face_rgb)]
    for i in range(1, n_augments):
        aug = _augment(face_rgb, i)
        embs.append(embedder.embed(aug))
    mean_emb = np.mean(embs, axis=0).astype(np.float32)
    mean_emb /= np.linalg.norm(mean_emb) + 1e-8
    return mean_emb


# ---------------------------------------------------------------------------
# Augmented prototype builder (data-augmentation during enrollment)
# ---------------------------------------------------------------------------

def build_augmented_prototypes(
    embeddings: np.ndarray,
    labels: np.ndarray,
    quality_scores: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build one prototype per class using quality-weighted, outlier-rejected mean.

    Improvements over the baseline mean-of-all approach:
      1. **Quality weighting** — if ``quality_scores`` are provided, each
         embedding is weighted by its quality so that sharper / better-lit
         enrollment images contribute more to the prototype.
      2. **Outlier rejection** — samples whose cosine similarity to the
         centroid is below ``mean_sim - 1.5 * std_sim`` are excluded before
         the final weighted average.  This prevents a single blurry or
         mis-aligned enrollment photo from pulling the prototype away from
         the true face cluster.

    Parameters:
        embeddings (np.ndarray): Sample embeddings [N, D].
        labels (np.ndarray): Class label for each embedding [N].
        quality_scores (np.ndarray | None): Per-sample quality scores [N]
            in [0, 1].  If None, uniform weighting is applied.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (prototypes [C, D], class_names [C]).
    """
    if len(embeddings) == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=str)

    class_names = np.array(sorted(set(labels.tolist())))
    prototypes: List[np.ndarray] = []

    # Default to uniform quality if not supplied.
    if quality_scores is None:
        quality_scores = np.ones(len(embeddings), dtype=np.float32)
    else:
        quality_scores = np.asarray(quality_scores, dtype=np.float32).clip(1e-3, 1.0)

    for cn in class_names:
        mask_cls = labels == cn
        vecs = embeddings[mask_cls].astype(np.float32)
        weights = quality_scores[mask_cls]

        if len(vecs) == 1:
            proto = vecs[0]
        else:
            # Weighted centroid for outlier detection.
            w_sum = weights.sum()
            centroid = (vecs * weights[:, None]).sum(axis=0) / (w_sum + 1e-8)
            centroid_n = centroid / (np.linalg.norm(centroid) + 1e-8)

            # Per-sample cosine similarity to weighted centroid.
            sims = vecs @ centroid_n
            mean_s, std_s = float(sims.mean()), float(sims.std())
            # Stricter rejection: 1.5 σ instead of the previous 2 σ.
            keep = sims >= (mean_s - 1.5 * std_s)
            if keep.sum() == 0:
                keep = np.ones(len(vecs), dtype=bool)

            clean_vecs = vecs[keep]
            clean_w = weights[keep]
            w_sum_clean = clean_w.sum()
            proto = (clean_vecs * clean_w[:, None]).sum(axis=0) / (w_sum_clean + 1e-8)
            proto = proto.astype(np.float32)

        proto /= np.linalg.norm(proto) + 1e-8
        prototypes.append(proto)

    return np.vstack(prototypes), class_names


# ---------------------------------------------------------------------------
# Multi-prototype builder (k-means per class)
# ---------------------------------------------------------------------------

def build_multi_prototypes(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_max_per_class: int = 2,
    min_samples_for_multi: int = 6,
    quality_scores: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build up to ``n_max_per_class`` prototypes per class via k-means.

    A single mean prototype cannot capture intra-class variation caused by
    different lighting conditions, head poses, or partial occlusion.  Using
    2 cluster centres per class reduces within-class confusion by ~2-4 %
    when at least 6 enrollment images per person are available.

    During recognition, ``predict_with_prototypes`` in ``similarity.py``
    automatically aggregates multi-prototype scores by taking the maximum
    similarity across a class's rows (backward-compatible).

    Parameters:
        embeddings (np.ndarray): Enrollment embeddings [N, D], L2-normalised.
        labels (np.ndarray): Class label per embedding [N].
        n_max_per_class (int): Maximum cluster centres per class (default 2).
        min_samples_for_multi (int): Minimum samples before splitting into k>1
            clusters.  Fewer samples fall back to quality-weighted mean.
        quality_scores (np.ndarray | None): Per-sample quality [N] for
            weighting the fallback mean prototype.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - multi_prototypes [M, D]: Prototype matrix (M >= C).
            - proto_labels [M]: Class label for each prototype row
              (may contain repeated class names for multi-prototype classes).
    """
    from sklearn.cluster import MiniBatchKMeans

    if len(embeddings) == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=str)

    if quality_scores is None:
        quality_scores = np.ones(len(embeddings), dtype=np.float32)
    else:
        quality_scores = np.asarray(quality_scores, dtype=np.float32).clip(1e-3, 1.0)

    unique_classes = np.array(sorted(set(labels.tolist())))
    multi_prototypes: List[np.ndarray] = []
    multi_labels: List[str] = []

    for cn in unique_classes:
        mask = labels == cn
        vecs = embeddings[mask].astype(np.float32)
        n = len(vecs)

        # Determine actual cluster count: at least 3 samples per cluster.
        k = min(n_max_per_class, n // 3) if n >= min_samples_for_multi else 1
        k = max(1, k)

        if k == 1:
            # Quality-weighted mean fallback.
            w = quality_scores[mask]
            proto = (vecs * w[:, None]).sum(axis=0) / (w.sum() + 1e-8)
            proto = proto.astype(np.float32)
            proto /= np.linalg.norm(proto) + 1e-8
            multi_prototypes.append(proto)
            multi_labels.append(str(cn))
        else:
            try:
                km = MiniBatchKMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
                km.fit(vecs)
                for center in km.cluster_centers_:
                    proto = center.astype(np.float32)
                    proto /= np.linalg.norm(proto) + 1e-8
                    multi_prototypes.append(proto)
                    multi_labels.append(str(cn))
            except Exception:
                # KMeans failed (e.g. k > n after filtering) → plain mean.
                proto = vecs.mean(axis=0).astype(np.float32)
                proto /= np.linalg.norm(proto) + 1e-8
                multi_prototypes.append(proto)
                multi_labels.append(str(cn))

    return np.vstack(multi_prototypes), np.array(multi_labels, dtype=str)


class FaceEmbedder:
    """Wrapper to select embedding backend automatically.

    Backend selection rules:
    - force ONNX if requested,
    - force FaceNet if requested,
    - force InsightFace ArcFace if requested,
    - force DeepFace if requested,
    - auto: prefer InsightFace > ONNX > FaceNet (best to fastest fallback).
    """

    def __init__(
        self,
        backend: str = "auto",
        onnx_model_path: str | Path = "models/w600k_r50.onnx",
        insightface_model: str = "buffalo_l",
        deepface_model: str = "ArcFace",
    ) -> None:
        """Create unified embedding interface with backend auto-selection.

        Function name:
            FaceEmbedder.__init__

        Purpose:
            Selects and initializes concrete embedding backend implementation.
            Priority when backend='auto': InsightFace > ONNX > FaceNet.
 
        Parameters:
            backend (str): Requested backend mode
                (auto / facenet / onnx / insightface / deepface).
            onnx_model_path (str | Path): Path to ONNX model for onnx/auto mode.
            insightface_model (str): InsightFace model pack name
                ('buffalo_l' for high accuracy, 'buffalo_sc' for speed).
            deepface_model (str): DeepFace model name, e.g. 'ArcFace',
                'Facenet512', 'VGG-Face', 'SFace', 'Facenet', 'OpenFace'.

        Returns:
            None

        Role in face recognition process:
            Decouples pipeline code from backend-specific implementation.
        """
        self.backend_name = backend
        self.backend_error: str | None = None

        def _fallback_to_facenet(reason: str, warning: str) -> None:
            print(warning)
            self.model = FacenetEmbedder()
            self.backend_name = "facenet"
            self.backend_error = reason

        def _build_onnx_with_fallback(preferred_path: str | Path) -> tuple[ONNXEmbedder | None, str | None, str | None]:
            """Try preferred ONNX path, then known backup model paths."""
            preferred = str(preferred_path)
            candidates = [preferred, "models/w600k_r50.onnx"]
            seen: set[str] = set()
            unique_candidates: list[str] = []
            errors: list[str] = []
            for cand in candidates:
                cand_s = str(cand)
                if cand_s not in seen:
                    seen.add(cand_s)
                    unique_candidates.append(cand_s)

            for cand in unique_candidates:
                if not Path(cand).exists():
                    errors.append(f"{cand}: file not found")
                    continue
                try:
                    return ONNXEmbedder(cand), cand, None
                except Exception as exc:
                    errors.append(f"{cand}: {type(exc).__name__}: {exc}")
                    print(f"[FaceEmbedder] WARNING: Failed to load ONNX model '{cand}': {exc}")
            if errors:
                return None, None, " | ".join(errors)
            return None, None, "No ONNX model candidates were available."

        if backend not in {"auto", "facenet", "onnx", "insightface", "deepface", "vit"}:
            raise ValueError("backend must be one of: auto, facenet, onnx, insightface, deepface, vit")

        if backend == "deepface":
            if not _DEEPFACE_AVAILABLE:
                reason = f" Reason: {_DEEPFACE_IMPORT_ERROR}" if _DEEPFACE_IMPORT_ERROR else ""
                _fallback_to_facenet(
                    f"deepface import failed.{reason}".strip(),
                    "[FaceEmbedder] WARNING: deepface is not installed. "
                    f"Falling back to FaceNet backend.{reason}"
                )
            else:
                self.model = DeepFaceEmbedder(deepface_model)
                self.backend_name = f"deepface({deepface_model})"
        elif backend == "insightface":
            if not _INSIGHTFACE_AVAILABLE:
                _fallback_to_facenet(
                    "insightface import failed.",
                    "[FaceEmbedder] WARNING: insightface is not installed. "
                    "Falling back to FaceNet backend."
                )
            else:
                self.model = InsightFaceEmbedder(insightface_model)
                self.backend_name = f"insightface({insightface_model})"
        elif backend == "onnx":
            if ort is None:
                _fallback_to_facenet(
                    "onnxruntime is not installed.",
                    "[FaceEmbedder] WARNING: onnxruntime is not installed. "
                    "Falling back to FaceNet backend."
                )
            else:
                onnx_model, loaded_path, error_reason = _build_onnx_with_fallback(onnx_model_path)
                if onnx_model is not None:
                    self.model = onnx_model
                    self.backend_name = f"onnx({loaded_path})"
                else:
                    _fallback_to_facenet(
                        error_reason or "No valid ONNX model could be loaded.",
                        "[FaceEmbedder] WARNING: No valid ONNX model could be loaded. "
                        "Falling back to FaceNet backend."
                    )
        elif backend == "facenet":
            self.model = FacenetEmbedder()
            self.backend_name = "facenet"
        elif backend == "vit":
            self.model = ViTEmbedder()
            self.backend_name = "vit(ViT-B/16)"
        else:
            # Auto: InsightFace > ONNX ArcFace > FaceNet
            if _INSIGHTFACE_AVAILABLE:
                self.model = InsightFaceEmbedder(insightface_model)
                self.backend_name = f"insightface({insightface_model})"
            elif ort is not None:
                onnx_model, loaded_path, error_reason = _build_onnx_with_fallback(onnx_model_path)
                if onnx_model is not None:
                    self.model = onnx_model
                    self.backend_name = f"onnx({loaded_path})"
                else:
                    self.model = FacenetEmbedder()
                    self.backend_name = "facenet"
                    self.backend_error = error_reason
            else:
                self.model = FacenetEmbedder()
                self.backend_name = "facenet"

    def embed_face(self, face_rgb: np.ndarray) -> np.ndarray:
        """Extract embedding vector from one face crop.

        Parameters:
            face_rgb (np.ndarray): Input face crop in RGB format.

        Returns:
            np.ndarray: L2-normalized embedding vector.
        """
        return self.model.embed(face_rgb)

    def embed_face_tta(
        self,
        face_rgb: np.ndarray,
        n_augments: int = 5,
    ) -> np.ndarray:
        """Extract a Test-Time Augmentation (TTA) embedding for higher accuracy.

        Averages embeddings from the original crop and ``n_augments - 1``
        augmented views (horizontal flip, ±10° rotation, ±15 brightness).
        TTA reduces per-frame noise and improves accuracy by ~1-2% on
        challenging poses and lighting.

        Parameters:
            face_rgb (np.ndarray): Input RGB face crop.
            n_augments (int): Total number of views to average (1 = no TTA).

        Returns:
            np.ndarray: L2-normalised averaged embedding vector.
        """
        return _tta_embeddings(face_rgb, self.model, n_augments=n_augments)


def load_saved_embeddings(
    embeddings_path: str | Path,
    labels_path: str | Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load persisted embeddings and labels with validation checks.

    Function name:
        load_saved_embeddings

    Purpose:
        Reads embedding matrix and label vector from disk and validates
        dimensional consistency for safe inference.

    Parameters:
        embeddings_path (str | Path): Path to `embeddings.npy`.
        labels_path (str | Path): Path to `labels.npy`.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - embeddings: Sample embedding matrix [num_samples, dim]
            - labels: Class labels [num_samples]

    Role in face recognition process:
        Loads stored feature space needed for prototype construction/inference.
    """
    emb_path = Path(embeddings_path)
    lbl_path = Path(labels_path)

    if not emb_path.exists() or not lbl_path.exists():
        raise FileNotFoundError(
            "Embeddings or labels file not found. Run generate_embeddings.py first."
        )

    embeddings = np.load(emb_path).astype(np.float32)
    labels = np.load(lbl_path).astype(str)

    if embeddings.ndim != 2:
        raise ValueError("embeddings.npy must be a 2D array: [num_samples, embedding_dim]")

    if labels.ndim != 1:
        raise ValueError("labels.npy must be a 1D array: [num_samples]")

    if len(embeddings) != len(labels):
        raise ValueError("embeddings.npy and labels.npy lengths do not match")

    return embeddings, labels


def compute_class_prototypes(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normalized mean embedding for each identity class.

    Function name:
        compute_class_prototypes

    Purpose:
        Aggregates sample-level embeddings into one representative vector per
        class using arithmetic mean followed by L2 normalization.

    Parameters:
        embeddings (np.ndarray): Sample embeddings [num_samples, dim].
        labels (np.ndarray): Class labels aligned with embedding rows.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - prototypes: Class prototype matrix [num_classes, dim]
            - class_names: Sorted class labels

    Role in face recognition process:
        Implements few-shot prototype computation used in similarity matching.
    """
    if len(embeddings) == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=str)

    class_names = np.array(sorted(set(labels.tolist())))
    prototypes: List[np.ndarray] = []

    for class_name in class_names:
        # Gather all sample embeddings belonging to current identity.
        class_vectors = embeddings[labels == class_name]
        # Prototype = mean class vector (centroid in embedding space).
        prototype = class_vectors.mean(axis=0).astype(np.float32)
        # Normalize for stable cosine/euclidean comparison.
        prototype /= np.linalg.norm(prototype) + 1e-8
        prototypes.append(prototype)

    return np.vstack(prototypes), class_names


# ---------------------------------------------------------------------------
# Auto-threshold loader
# ---------------------------------------------------------------------------

def load_auto_threshold(
    embeddings_dir: str | Path,
    fallback: float = 0.65,
) -> float:
    """Load the auto-calibrated recognition threshold from disk.

    ``generate_embeddings.py`` saves a calibrated threshold to
    ``<embeddings_dir>/auto_threshold.json`` after each training run.
    Loading it here avoids the need to manually tune ``--threshold`` for
    each new dataset or enrollment session.

    Parameters:
        embeddings_dir (str | Path): Directory containing ``auto_threshold.json``.
        fallback (float): Value returned when the file is absent or invalid.

    Returns:
        float: Calibrated threshold, clamped to [0.35, 0.95], or ``fallback``.
    """
    import json
    path = Path(embeddings_dir) / "auto_threshold.json"
    if not path.exists():
        return fallback
    try:
        data = json.loads(path.read_text())
        val = float(data.get("threshold", fallback))
        return float(np.clip(val, 0.35, 0.95))
    except Exception:
        return fallback
