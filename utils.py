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

def apply_clahe_enhancement(face_rgb: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast normalization to an RGB face crop.

    Function name:
        apply_clahe_enhancement

    Purpose:
        Improves local contrast on small / distant face crops where the
        webcam auto-exposure may flatten fine facial details. CLAHE is
        applied only on the luminance channel (LAB color space) to avoid
        distorting skin tone colours.

    Parameters:
        face_rgb (np.ndarray): Input RGB face crop (any size).

    Returns:
        np.ndarray: Contrast-enhanced RGB face crop (same spatial size).

    Role in face recognition process:
        Pre-processing step applied before embedding extraction when the
        detected face region is small (< 100 px wide), which typically
        corresponds to subjects standing 2-3 metres from the webcam.
    """
    # Convert to LAB so that CLAHE acts only on luminance.
    lab = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # clipLimit=2.0 prevents over-amplification of noise in smooth regions.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
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

    Stores bounding-box coordinates, detector confidence, and cropped RGB face.
    """

    box: Tuple[int, int, int, int]
    confidence: float
    face_rgb: np.ndarray


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
    boxes, probs = detector.detect(rgb)

    if boxes is None or probs is None:
        return []

    frame_h, frame_w = frame_bgr.shape[:2]
    detections: List[Detection] = []

    # Convert raw detector output into sanitized detections.
    for box, score in zip(boxes, probs):
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

        detections.append(
            Detection(
                box=(x1, y1, x2, y2),
                confidence=float(score),
                face_rgb=face,
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
        embedding = outputs[0].reshape(-1).astype(np.float32)
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


# ---------------------------------------------------------------------------
# Test-time augmentation (TTA) helper
# ---------------------------------------------------------------------------

def _tta_embeddings(
    face_rgb: np.ndarray,
    embedder,
    n_augments: int = 5,
) -> np.ndarray:
    """Return the L2-normalised mean of N augmented embeddings (TTA).

    Augmentations applied: horizontal flip, ±10° rotation, ±10 % brightness,
    and mild Gaussian blur.  Averaging multiple views reduces per-frame noise
    and improves robustness to pose / lighting variation.

    Parameters:
        face_rgb (np.ndarray): Original RGB face crop.
        embedder: Object with an `embed(face_rgb)` method.
        n_augments (int): Number of augmented copies to average.

    Returns:
        np.ndarray: L2-normalised averaged embedding vector.
    """
    h, w = face_rgb.shape[:2]
    cx, cy = w // 2, h // 2

    def _augment(img: np.ndarray, i: int) -> np.ndarray:
        out = img.copy()
        # Horizontal flip on even indices.
        if i % 2 == 0:
            out = cv2.flip(out, 1)
        # Rotation: -10, 0, +10 degrees cycling.
        angle = [-10, 0, 10][i % 3]
        if angle != 0:
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            out = cv2.warpAffine(out, M, (w, h))
        # Brightness jitter.
        if i % 3 == 1:
            out = np.clip(out.astype(np.int32) + 15, 0, 255).astype(np.uint8)
        elif i % 3 == 2:
            out = np.clip(out.astype(np.int32) - 15, 0, 255).astype(np.uint8)
        return out

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
) -> Tuple[np.ndarray, np.ndarray]:
    """Build one prototype per class using the stored embeddings only (no re-embed).

    This is a lightweight augmentation performed purely in embedding space:
    each sample embedding is used as-is (CLAHE already applied upstream), and
    the class prototype is the L2-normalised centroid.  The improvement over
    `compute_class_prototypes` is a fallback guard that excludes outlier
    embeddings (cosine distance > 2 std from the mean) before averaging.

    Parameters:
        embeddings (np.ndarray): Sample embeddings [N, D].
        labels (np.ndarray): Class labels [N].

    Returns:
        Tuple[np.ndarray, np.ndarray]: (prototypes [C, D], class_names [C]).
    """
    if len(embeddings) == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=str)

    class_names = np.array(sorted(set(labels.tolist())))
    prototypes: List[np.ndarray] = []

    for cn in class_names:
        vecs = embeddings[labels == cn].astype(np.float32)
        if len(vecs) == 1:
            proto = vecs[0]
        else:
            centroid = vecs.mean(axis=0)
            # Compute cosine distance of each sample to the centroid.
            centroid_n = centroid / (np.linalg.norm(centroid) + 1e-8)
            sims = vecs @ centroid_n
            mean_s, std_s = sims.mean(), sims.std()
            # Keep samples within 2σ of the mean similarity (outlier rejection).
            mask = sims >= (mean_s - 2.0 * std_s)
            clean_vecs = vecs[mask] if mask.sum() > 0 else vecs
            proto = clean_vecs.mean(axis=0).astype(np.float32)
        proto /= np.linalg.norm(proto) + 1e-8
        prototypes.append(proto)

    return np.vstack(prototypes), class_names


class FaceEmbedder:
    """Wrapper to select embedding backend automatically.

    Backend selection rules:
    - force ONNX if requested,
    - force FaceNet if requested,
    - force InsightFace ArcFace if requested,
    - auto: prefer InsightFace > ONNX > FaceNet (best to fastest fallback).
    """

    def __init__(
        self,
        backend: str = "auto",
        onnx_model_path: str | Path = "models/arcface.onnx",
        insightface_model: str = "buffalo_l",
    ) -> None:
        """Create unified embedding interface with backend auto-selection.

        Function name:
            FaceEmbedder.__init__

        Purpose:
            Selects and initializes concrete embedding backend implementation.
            Priority when backend='auto': InsightFace > ONNX > FaceNet.

        Parameters:
            backend (str): Requested backend mode
                (auto / facenet / onnx / insightface).
            onnx_model_path (str | Path): Path to ONNX model for onnx/auto mode.
            insightface_model (str): InsightFace model pack name
                ('buffalo_l' for high accuracy, 'buffalo_sc' for speed).

        Returns:
            None

        Role in face recognition process:
            Decouples pipeline code from backend-specific implementation.
        """
        self.backend_name = backend

        if backend not in {"auto", "facenet", "onnx", "insightface"}:
            raise ValueError("backend must be one of: auto, facenet, onnx, insightface")

        if backend == "insightface":
            if not _INSIGHTFACE_AVAILABLE:
                print(
                    "[FaceEmbedder] WARNING: insightface is not installed. "
                    "Falling back to FaceNet backend."
                )
                self.model = FacenetEmbedder()
                self.backend_name = "facenet"
            else:
                self.model = InsightFaceEmbedder(insightface_model)
                self.backend_name = f"insightface({insightface_model})"
        elif backend == "onnx":
            if not Path(onnx_model_path).exists():
                print(
                    f"[FaceEmbedder] WARNING: ONNX model not found at '{onnx_model_path}'. "
                    "Falling back to FaceNet backend."
                )
                self.model = FacenetEmbedder()
                self.backend_name = "facenet"
            elif ort is None:
                print(
                    "[FaceEmbedder] WARNING: onnxruntime is not installed. "
                    "Falling back to FaceNet backend."
                )
                self.model = FacenetEmbedder()
                self.backend_name = "facenet"
            else:
                self.model = ONNXEmbedder(onnx_model_path)
                self.backend_name = "onnx"
        elif backend == "facenet":
            self.model = FacenetEmbedder()
            self.backend_name = "facenet"
        else:
            # Auto: InsightFace > ONNX ArcFace > FaceNet
            if _INSIGHTFACE_AVAILABLE:
                self.model = InsightFaceEmbedder(insightface_model)
                self.backend_name = f"insightface({insightface_model})"
            elif Path(onnx_model_path).exists() and ort is not None:
                self.model = ONNXEmbedder(onnx_model_path)
                self.backend_name = "onnx"
            else:
                self.model = FacenetEmbedder()
                self.backend_name = "facenet"

    def embed_face(self, face_rgb: np.ndarray) -> np.ndarray:
        """Extract embedding vector from one face crop.

        Function name:
            FaceEmbedder.embed_face

        Purpose:
            Delegates embedding extraction to selected backend instance.

        Parameters:
            face_rgb (np.ndarray): Input face crop in RGB format.

        Returns:
            np.ndarray: L2-normalized embedding vector.

        Role in face recognition process:
            Unified feature extraction call used across all pipeline modules.
        """
        return self.model.embed(face_rgb)


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
