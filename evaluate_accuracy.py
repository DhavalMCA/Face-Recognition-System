"""evaluate_accuracy.py — Model Accuracy Measurement and Comparison.

Purpose:
    Measures the recognition accuracy of the deployed few-shot face recognition
    model across every enrolled image in the dataset and prints a comparison
    table with published results from common baselines.

Evaluation methodology (one-vs-rest binary classification per identity):
    For every test image from identity X and using threshold T:
        - Positive class  = identity X
        - Negative class  = every other enrolled identity

    Decision matrix:
        TP  True positive   Image is X, model predicts X
        FN  False negative  Image is X, model predicts Unknown or Y≠X
        TN  True negative   Image is not X, model does not predict X
        FP  False positive  Image is not X, model predicts X

    Per-identity metrics:
        Sensitivity (Recall / TPR) = TP / (TP + FN)
        Specificity (TNR)          = TN / (TN + FP)
        Accuracy                   = (TP + TN) / (TP + TN + FP + FN)

    Final model metrics = macro-average across all enrolled identities.

Usage:
    # Test with default threshold (0.70)
    python evaluate_accuracy.py

    # Test with alternative threshold
    python evaluate_accuracy.py --threshold 0.80

    # Sweep multiple thresholds and pick the best
    python evaluate_accuracy.py --sweep

    # Suppress per-image details
    python evaluate_accuracy.py --quiet

    # Change metric used for similarity matching
    python evaluate_accuracy.py --metric euclidean

    # Specify custom embeddings directory
    python evaluate_accuracy.py --embeddings-dir embeddings/
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Project root on path ──────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils import (
    FaceEmbedder,
    detect_faces,
    get_identity_folders,
    load_face_detector,
)
from similarity import predict_with_prototypes

# ── Defaults ──────────────────────────────────────────────────────────
DATASET_DIR = "dataset"
EMBEDDINGS_DIR = "embeddings"
ONNX_MODEL_PATH = "models/w600k_r50.onnx"
DEFAULT_THRESHOLD = 0.70
DEFAULT_METRIC = "cosine"
SWEEP_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

# ── Comparison backends (run live on the local dataset) ──────────────
# Each entry: (display_label, backend, deepface_model, insightface_model)
# Set deepface_model / insightface_model to None when not needed.
COMPARISON_BACKENDS: List[Tuple[str, str, str | None, str | None]] = [
    ("InsightFace (buffalo_l)",  "insightface", None,          "buffalo_l"),
    ("InsightFace (buffalo_sc)", "insightface", None,          "buffalo_sc"),
    ("FaceNet",                  "facenet",      None,          None),
    ("ONNX (w600k_r50)",         "onnx",         None,          None),
    ("DeepFace ArcFace",         "deepface",     "ArcFace",     None),
    ("DeepFace VGG-Face",        "deepface",     "VGG-Face",    None),
    ("DeepFace Facenet512",      "deepface",     "Facenet512",  None),
    ("DeepFace SFace",           "deepface",     "SFace",       None),
]


# ── Type alias for the pre-computed face-crop cache ─────────────────
# Maps str(image_path) -> RGB face crop ndarray, or None if no face found.
FaceCropCache = Dict[str, Optional[np.ndarray]]


def precompute_face_crops(
    dataset_dir: Path,
    detector,
    min_confidence: float = 0.82,
) -> FaceCropCache:
    """Run MTCNN detection once across every dataset image and cache the results.

    This eliminates redundant detection when evaluating multiple backends:
    detection runs exactly once regardless of how many models are compared.

    Args:
        dataset_dir:    Root dataset directory.
        detector:       Loaded MTCNN detector.
        min_confidence: Minimum detection confidence (default 0.82).

    Returns:
        Dict mapping ``str(image_path)`` to the cropped face RGB array, or
        ``None`` when no face was found in that image.
    """
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    cache: FaceCropCache = {}

    for folder in get_identity_folders(dataset_dir):
        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() not in exts:
                continue
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                cache[str(img_path)] = None
                continue
            detections = detect_faces(bgr, detector, min_confidence=min_confidence, padding=0.12)
            if not detections:
                cache[str(img_path)] = None
            else:
                largest = max(
                    detections,
                    key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]),
                )
                cache[str(img_path)] = largest.face_rgb

    return cache


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════

def load_prototypes(embeddings_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load prototype matrix and class names from disk.

    Args:
        embeddings_dir: Path to embeddings folder.

    Returns:
        Tuple of (prototypes [num_classes, dim], class_names [num_classes]).

    Raises:
        FileNotFoundError: If prototypes.npy or class_names.npy is missing.
    """
    proto_path = embeddings_dir / "prototypes.npy"
    names_path = embeddings_dir / "class_names.npy"

    if not proto_path.exists():
        raise FileNotFoundError(
            f"Prototypes not found: {proto_path}\n"
            "Run Step 2 (Train Recognition Engine) first."
        )
    if not names_path.exists():
        raise FileNotFoundError(
            f"Class names not found: {names_path}\n"
            "Run Step 2 (Train Recognition Engine) first."
        )

    prototypes = np.load(proto_path).astype(np.float32)
    class_names = np.load(names_path).astype(str)
    return prototypes, class_names


def build_in_memory_prototypes(
    dataset_dir: Path,
    detector,
    embedder: FaceEmbedder,
    face_cache: FaceCropCache | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build prototype matrix from the dataset without writing to disk.

    Embeds every image in ``dataset_dir`` using ``embedder``, then computes
    the mean (L2-normalised) embedding per identity as its prototype.

    Args:
        dataset_dir: Root dataset directory containing one sub-folder per identity.
        detector:    Loaded MTCNN detector (used only when face_cache is None).
        embedder:    FaceEmbedder initialised with the desired backend.
        face_cache:  Pre-computed crop cache; when provided detection is skipped.

    Returns:
        Tuple of (prototypes [num_classes, dim], class_names [num_classes]).
    """
    from utils import build_augmented_prototypes as _build_protos

    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    folders = get_identity_folders(dataset_dir)

    all_embeddings: List[np.ndarray] = []
    all_labels: List[str] = []

    for folder in folders:
        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() not in exts:
                continue
            ok, emb = embed_image(img_path, detector, embedder, face_cache=face_cache)
            if ok and emb is not None:
                all_embeddings.append(emb)
                all_labels.append(folder.name)

    if not all_embeddings:
        raise RuntimeError('No usable face images found for prototype building.')

    emb_array = np.vstack(all_embeddings).astype(np.float32)
    norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
    emb_array /= norms + 1e-8
    labels_array = np.array(all_labels, dtype=str)

    prototypes, class_names = _build_protos(emb_array, labels_array)
    return prototypes, class_names


def evaluate_backend(
    label: str,
    backend: str,
    deepface_model: str | None,
    insightface_model: str | None,
    dataset_dir: Path,
    detector,
    threshold: float,
    metric: str,
    quiet: bool = True,
    face_cache: FaceCropCache | None = None,
) -> Dict[str, float] | None:
    """Build prototypes and evaluate one backend; return macro-averaged metrics.

    Returns None if the backend is unavailable or encounters an error.

    Args:
        label:              Human-readable name for the comparison table.
        backend:            FaceEmbedder backend string.
        deepface_model:     DeepFace model name (or None).
        insightface_model:  InsightFace model name (or None).
        dataset_dir:        Root dataset directory.
        detector:           Loaded MTCNN detector.
        threshold:          Recognition similarity threshold.
        metric:             'cosine' or 'euclidean'.
        quiet:              Suppress per-image output.
        face_cache:         Pre-computed crop cache; skips detection entirely.

    Returns:
        Dict with 'accuracy', 'precision', 'sensitivity', 'specificity', 'f1',
        or None on error.
    """
    try:
        kwargs: dict = {"backend": backend, "onnx_model_path": ONNX_MODEL_PATH}
        if deepface_model:
            kwargs["deepface_model"] = deepface_model
        if insightface_model:
            kwargs["insightface_model"] = insightface_model

        embedder = FaceEmbedder(**kwargs)
        prototypes, class_names = build_in_memory_prototypes(
            dataset_dir, detector, embedder, face_cache=face_cache
        )
        results, _, _ = evaluate(
            dataset_dir=dataset_dir,
            prototypes=prototypes,
            class_names=class_names,
            detector=detector,
            embedder=embedder,
            threshold=threshold,
            metric=metric,
            quiet=quiet,
            face_cache=face_cache,
        )
        return {
            "accuracy":    float(np.mean([r.accuracy    for r in results])),
            "precision":   float(np.mean([r.precision   for r in results])),
            "sensitivity": float(np.mean([r.sensitivity for r in results])),
            "specificity": float(np.mean([r.specificity for r in results])),
            "f1":          float(np.mean([r.f1          for r in results])),
        }
    except Exception as exc:
        print(f"  ⚠  Backend '{label}' skipped: {exc}")
        return None


def embed_image(
    img_path: Path,
    detector,
    embedder: FaceEmbedder,
    min_confidence: float = 0.82,
    face_cache: FaceCropCache | None = None,
) -> Tuple[bool, np.ndarray | None]:
    """Read one image, detect a face, extract its embedding.

    A slightly lower min_confidence (0.82 vs runtime 0.92) accepts edge-case
    angles common in mobile photos.

    When ``face_cache`` is provided the detection step is skipped entirely —
    the pre-cropped face RGB is used directly, which is much faster for
    multi-backend comparison runs.

    Args:
        img_path:       Path to image file.
        detector:       MTCNN detector (used only when face_cache is None).
        embedder:       FaceEmbedder instance.
        min_confidence: Minimum MTCNN face detection confidence.
        face_cache:     Optional pre-computed crop cache from
                        ``precompute_face_crops()``.

    Returns:
        (success, embedding_or_None)
    """
    if face_cache is not None:
        face_rgb = face_cache.get(str(img_path))
        if face_rgb is None:
            return False, None
        embedding = embedder.embed_face(face_rgb)
        return True, embedding

    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return False, None

    detections = detect_faces(bgr, detector, min_confidence=min_confidence, padding=0.12)
    if not detections:
        return False, None

    # Use the largest detected face as the primary face.
    largest = max(detections, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
    embedding = embedder.embed_face(largest.face_rgb)
    return True, embedding


# ══════════════════════════════════════════════════════════════════════
#  Core evaluation logic
# ══════════════════════════════════════════════════════════════════════

class EvalResult:
    """Stores one-vs-rest evaluation counters for a single identity class."""

    def __init__(self, name: str):
        """Initialise counters for identity `name`."""
        self.name = name
        self.tp = 0   # model predicted this identity, true label is this identity
        self.tn = 0   # model did not predict this identity, true label is someone else
        self.fp = 0   # model predicted this identity, true label is someone else
        self.fn = 0   # model did not predict this identity, true label is this identity

    @property
    def sensitivity(self) -> float:
        """TP / (TP + FN) — how well the model detects this person."""
        denom = self.tp + self.fn
        return (self.tp / denom * 100) if denom > 0 else 0.0

    @property
    def specificity(self) -> float:
        """TN / (TN + FP) — how well the model rejects others."""
        denom = self.tn + self.fp
        return (self.tn / denom * 100) if denom > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """(TP + TN) / (TP + TN + FP + FN)."""
        denom = self.tp + self.tn + self.fp + self.fn
        return ((self.tp + self.tn) / denom * 100) if denom > 0 else 0.0

    @property
    def precision(self) -> float:
        """TP / (TP + FP)."""
        denom = self.tp + self.fp
        return (self.tp / denom * 100) if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of sensitivity and precision."""
        s = self.sensitivity
        p = self.precision
        denom = s + p
        return (2 * s * p / denom) if denom > 0 else 0.0


def evaluate(
    dataset_dir: Path,
    prototypes: np.ndarray,
    class_names: np.ndarray,
    detector,
    embedder: FaceEmbedder,
    threshold: float,
    metric: str,
    quiet: bool,
    face_cache: FaceCropCache | None = None,
) -> Tuple[List[EvalResult], int, int]:
    """Run full evaluation across all enrolled identities.

    For each image in each identity folder:
      1. Detect and embed the face (or use ``face_cache`` to skip detection).
      2. Predict identity using predict_with_prototypes.
      3. Update one-vs-rest counters for EVERY enrolled class.

    Args:
        dataset_dir:  Root dataset directory.
        prototypes:   Prototype matrix [num_classes, dim].
        class_names:  Class name array [num_classes].
        detector:     MTCNN detector (used only when face_cache is None).
        embedder:     FaceEmbedder.
        threshold:    Recognition threshold.
        metric:       'cosine' or 'euclidean'.
        quiet:        Suppress per-image output.
        face_cache:   Pre-computed crop cache; when provided detection is skipped.

    Returns:
        (list_of_eval_results, total_images_tested, total_no_face)
    """
    folders = get_identity_folders(dataset_dir)
    if not folders:
        print("✗  No identity folders found in dataset/.")
        sys.exit(1)

    # Initialise one EvalResult per enrolled identity.
    results: Dict[str, EvalResult] = {cn: EvalResult(cn) for cn in class_names.tolist()}

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    total_images = 0
    total_no_face = 0

    for folder in folders:
        true_label = folder.name
        if true_label not in results:
            # Identity in dataset but not in prototypes (untrained) → skip.
            print(f"  ⚠  '{true_label}' is in dataset/ but has no prototype — skipping.")
            continue

        image_paths = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]
        if not image_paths:
            continue

        if not quiet:
            print(f"\n  ── {true_label} ({len(image_paths)} images)")

        for img_path in image_paths:
            total_images += 1
            ok, embedding = embed_image(img_path, detector, embedder, face_cache=face_cache)

            if not ok or embedding is None:
                # No face detected — count as FN for the true identity.
                results[true_label].fn += 1
                # For all other classes, count as TN (image was not misidentified).
                for cn in results:
                    if cn != true_label:
                        results[cn].tn += 1
                total_no_face += 1
                if not quiet:
                    print(f"      ⚠  {img_path.name} — no face detected (FN for {true_label})")
                continue

            prediction = predict_with_prototypes(
                query_embedding=embedding,
                prototypes=prototypes,
                class_names=class_names,
                metric=metric,
                threshold=threshold,
            )
            predicted_label = prediction["name"]
            score = prediction["score"]

            # Update one-vs-rest counters for every class.
            for cn, result in results.items():
                if cn == true_label:
                    # The true class.
                    if predicted_label == true_label:
                        result.tp += 1   # correctly recognised
                    else:
                        result.fn += 1   # missed the correct person
                else:
                    # Every other class.
                    if predicted_label == cn:
                        result.fp += 1   # wrongly attributed to this class
                    else:
                        result.tn += 1   # correctly not attributed to this class

            if not quiet:
                verdict = "✓ " if predicted_label == true_label else "✗ "
                tag = f"predicted: {predicted_label}  score: {score:.4f}"
                print(f"      {verdict} {img_path.name}  →  {tag}")

    return list(results.values()), total_images, total_no_face


# ══════════════════════════════════════════════════════════════════════
#  Reporting
# ══════════════════════════════════════════════════════════════════════

def _bar(value: float, width: int = 10) -> str:
    """ASCII progress bar for a 0–100 percentage value."""
    filled = round(value / 100 * width)
    return "█" * filled + "░" * (width - filled)


def print_per_identity_table(results: List[EvalResult]) -> None:
    """Print per-identity metrics table."""
    header = (
        f"  {'Identity':<18}  "
        f"{'Sensitivity':>13}  "
        f"{'Specificity':>13}  "
        f"{'Accuracy':>10}  "
        f"{'Precision':>11}  "
        f"{'F1-Score':>9}  "
        f"TP  FN  FP  TN"
    )
    sep = "  " + "─" * (len(header) - 2)
    print(header)
    print(sep)
    for r in results:
        print(
            f"  {r.name:<18}  "
            f"{r.sensitivity:>12.2f}%  "
            f"{r.specificity:>12.2f}%  "
            f"{r.accuracy:>9.2f}%  "
            f"{r.precision:>10.2f}%  "
            f"{r.f1:>8.2f}%  "
            f"{r.tp:>3d} {r.fn:>3d} {r.fp:>3d} {r.tn:>3d}"
        )


def _comparison_resolved_name(bk: str, df_model: str | None, if_model: str | None) -> str:
    """Return the backend_name string that FaceEmbedder would produce for this entry."""
    if bk == "deepface":
        return f"deepface({df_model})" if df_model else "deepface(ArcFace)"
    if bk == "insightface":
        return f"insightface({if_model})" if if_model else "insightface(buffalo_l)"
    if bk == "facenet":
        return "facenet"
    if bk == "onnx":
        return f"onnx({ONNX_MODEL_PATH})"
    return bk


def print_comparison_table(
    all_models: Dict[str, Dict[str, float]],
    primary_label: str,
) -> None:
    """Print a formatted comparison table across all evaluated backends.

    Columns: Rank | Model | Accuracy | Precision | Recall | Specificity | F1
    Rows are sorted by Accuracy descending.  The primary model is marked ◄.

    Args:
        all_models:    Dict mapping display label → metric dict.
        primary_label: Label of the primary / reference backend (gets ◄ marker).
    """
    if not all_models:
        return

    # Sort rows by accuracy descending.
    sorted_items = sorted(
        all_models.items(), key=lambda kv: kv[1].get("accuracy", 0), reverse=True
    )

    col_rank  = 4
    col_model = max(len(k) for k in all_models) + 2
    # fixed widths: Accuracy, Precision, Recall, Specificity, F1
    header_fmt = (
        f"  {'#':>{col_rank}}  "
        f"{'Model':<{col_model}}  "
        f"{'Accuracy':>10}  "
        f"{'Precision':>10}  "
        f"{'Recall':>8}  "
        f"{'Specificity':>12}  "
        f"{'F1':>8}"
    )
    inner_w = len(header_fmt)
    top    = "  ╔" + "═" * (inner_w - 3) + "╗"
    mid    = "  ╠" + "═" * (inner_w - 3) + "╣"
    bottom = "  ╚" + "═" * (inner_w - 3) + "╝"

    print(f"\n{top}")
    print(f"  ║{header_fmt[3:]}")
    print(mid)

    primary_acc: float = 0.0
    primary_rank: int = 0
    total = len(sorted_items)

    for rank_idx, (model_name, m) in enumerate(sorted_items, start=1):
        is_primary = model_name == primary_label
        if is_primary:
            primary_acc = m.get("accuracy", 0.0)
            primary_rank = rank_idx
        marker = " ◄" if is_primary else "  "
        row = (
            f"  ║  {rank_idx:>{col_rank}}  "
            f"{model_name:<{col_model}}  "
            f"{m.get('accuracy',    0.0):>9.2f}%  "
            f"{m.get('precision',   0.0):>9.2f}%  "
            f"{m.get('sensitivity', 0.0):>7.2f}%  "
            f"{m.get('specificity', 0.0):>11.2f}%  "
            f"{m.get('f1',          0.0):>7.2f}%"
            f"{marker}"
        )
        print(row)

    print(bottom)

    if primary_rank:
        print(
            f"\n  Primary model : {primary_label}"
            f"  │  Accuracy: {primary_acc:.2f}%"
            f"  │  Rank: #{primary_rank} of {total}"
        )


def print_sweep_table(sweep_data: List[dict]) -> None:
    """Print threshold sweep comparison table."""
    print(
        f"\n  {'Threshold':>12}  "
        f"{'Sensitivity (%)':>17}  "
        f"{'Specificity (%)':>17}  "
        f"{'Accuracy (%)':>14}  "
        f"{'F1 (%)':>8}"
    )
    print("  " + "─" * 77)
    for row in sweep_data:
        print(
            f"  {row['threshold']:>12.2f}  "
            f"{row['sensitivity']:>16.2f}%  "
            f"{row['specificity']:>16.2f}%  "
            f"{row['accuracy']:>13.2f}%  "
            f"{row['f1']:>7.2f}%"
        )
    # Highlight threshold with best accuracy.
    best = max(sweep_data, key=lambda r: r["accuracy"])
    print(
        f"\n  ★  Best accuracy {best['accuracy']:.2f}%  "
        f"at threshold = {best['threshold']:.2f}\n"
        f"\n  ★  Best F1 score at threshold = "
        f"{max(sweep_data, key=lambda r: r['f1'])['threshold']:.2f}"
    )


def print_threshold_recommendation(results: List[EvalResult], threshold: float) -> None:
    """Explain the threshold choice and intra-family robustness."""
    sens = float(np.mean([r.sensitivity for r in results]))
    spec = float(np.mean([r.specificity for r in results]))

    W = 65  # inner width (between the two │ borders)

    def row(text: str = "") -> str:
        """Return a padded content row: '  │  <text padded to W-2>│'."""
        return f"  │  {text:<{W - 2}}│"

    def divider(left: str = "├", right: str = "┤") -> str:
        return f"  {left}{'─' * W}{right}"

    lines = [
        f"  ┌{'─' * W}┐",
        f"  │{'THRESHOLD RECOMMENDATION':^{W}}│",
        divider(),
        row(f"Current threshold : {threshold:.2f}"),
        row(f"Sensitivity (avg) : {sens:.2f}%"),
        row(f"Specificity (avg) : {spec:.2f}%"),
        divider(),
        row("Problem: Family member confusion (mother recognised as you)"),
        row("Root cause: Shared facial features → similar embeddings"),
        row(),
        row("Recommended thresholds:"),
        row("  0.80 – 0.85  High-security (few false positives)"),
        row("  0.70 – 0.75  Balanced (good accuracy, low false alarms)"),
        row("  0.60 – 0.65  Lenient  (maximises recall, more false pos.)"),
        row(),
        row("For the family confusion problem:"),
        row("  → Raise threshold to 0.80"),
        row("  → Enroll your mother as a separate identity"),
        row('     (run:  python gui.py  → Step 1 → "Mother")'),
        row("  → Re-train embeddings (Step 2)"),
        row(),
        row("Why few-shot prototypes handle intra-family similarity:"),
        row("  • Each enrolled person has their own class prototype"),
        row("    (mean embedding of all their samples)."),
        row("  • When two people share facial structure, their prototypes"),
        row("    are close together in embedding space."),
        row("  • A higher threshold forces the model to only accept a"),
        row("    face when it is unambiguously close to ONE prototype"),
        row("    and clearly farther from all other prototypes."),
        row("  • Adding the mother's samples creates a distinct prototype"),
        row("    that the model can use to SEPARATE the two identities,"),
        row("    rather than collapsing both onto the son's prototype."),
        f"  └{'─' * W}┘",
    ]
    print("\n" + "\n".join(lines) + "\n")


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate FewShotFace model accuracy and compare with baselines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Recognition threshold (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--metric",
        choices=["cosine", "euclidean"],
        default=DEFAULT_METRIC,
        help=f"Similarity metric (default: {DEFAULT_METRIC}).",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run evaluation at multiple thresholds and print sweep table.",
    )
    parser.add_argument(
        "--sweep-values",
        nargs="+",
        type=float,
        default=SWEEP_THRESHOLDS,
        help="Threshold values for --sweep (default: 0.60 0.65 0.70 0.75 0.80 0.85 0.90).",
    )
    parser.add_argument(
        "--dataset-dir",
        default=DATASET_DIR,
        help=f"Dataset directory (default: {DATASET_DIR}).",
    )
    parser.add_argument(
        "--embeddings-dir",
        default=EMBEDDINGS_DIR,
        help=f"Embeddings directory (default: {EMBEDDINGS_DIR}).",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "facenet", "onnx", "insightface", "deepface"],
        default="auto",
        help="Embedding backend (auto selects InsightFace > ONNX > FaceNet; deepface = multi-model via DeepFace library).",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-image output.",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        dest="no_compare",
        help=(
            "Skip multi-backend comparison — evaluate only the primary "
            "--backend and do not run other models.  Faster, but produces "
            "a single-row table."
        ),
    )
    return parser.parse_args()


def _metrics_from_results(results: List[EvalResult]) -> Dict[str, float]:
    """Compute macro-averaged metrics dict from a list of EvalResult objects."""
    return {
        "accuracy":    float(np.mean([r.accuracy    for r in results])),
        "precision":   float(np.mean([r.precision   for r in results])),
        "sensitivity": float(np.mean([r.sensitivity for r in results])),
        "specificity": float(np.mean([r.specificity for r in results])),
        "f1":          float(np.mean([r.f1          for r in results])),
    }


def _run_backends_parallel(
    backends: List[Tuple[str, str, str | None, str | None]],
    primary_backend_name: str,
    dataset_dir: Path,
    detector,
    threshold: float,
    metric: str,
    face_cache: FaceCropCache,
    max_workers: int = 4,
) -> Dict[str, Dict[str, float]]:
    """Evaluate all comparison backends in parallel, using a shared face cache.

    Face detection is already done (stored in ``face_cache``), so each worker
    only loads its model and runs the (fast) embedding + prediction steps.

    Args:
        backends:             List from COMPARISON_BACKENDS to evaluate.
        primary_backend_name: backend_name of the already-evaluated primary
                              model; matching backends are skipped.
        dataset_dir:          Root dataset directory.
        detector:             MTCNN detector (passed through; not re-used for
                              detection since face_cache is provided).
        threshold:            Recognition threshold.
        metric:               'cosine' or 'euclidean'.
        face_cache:           Pre-computed face crops (from precompute_face_crops).
        max_workers:          Thread-pool size (default 4).

    Returns:
        Dict of label -> metrics_dict for every backend that succeeded.
    """
    collected: Dict[str, Dict[str, float]] = {}

    def _task(lbl: str, bk: str, df_m: str | None, if_m: str | None):
        if _comparison_resolved_name(bk, df_m, if_m) == primary_backend_name:
            return lbl, None
        m = evaluate_backend(
            label=lbl, backend=bk,
            deepface_model=df_m, insightface_model=if_m,
            dataset_dir=dataset_dir, detector=detector,
            threshold=threshold, metric=metric,
            quiet=True, face_cache=face_cache,
        )
        return lbl, m

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_task, lbl, bk, df_m, if_m): lbl
            for lbl, bk, df_m, if_m in backends
        }
        for future in as_completed(futures):
            lbl, m = future.result()
            if m is not None:
                collected[lbl] = m
                print(
                    f"  ✓  {lbl:<30}  "
                    f"acc={m['accuracy']:.2f}%  "
                    f"prec={m['precision']:.2f}%  "
                    f"rec={m['sensitivity']:.2f}%  "
                    f"f1={m['f1']:.2f}%"
                )
    return collected


def main() -> None:
    """Evaluation entry point."""
    import os
    os.chdir(str(PROJECT_DIR))

    args = parse_args()
    dataset_dir = Path(args.dataset_dir)

    print(f"\n{'═' * 72}")
    print( "   FewShotFace — Unified Model Accuracy Comparison")
    print(f"{'═' * 72}\n")

    # ── Step 1: Load detector once ────────────────────────────────
    print("⟳  Loading face detector...", end="  ", flush=True)
    detector = load_face_detector()
    print("✓  MTCNN ready.")

    # ── Step 2: Pre-detect all faces ONCE (shared across backends) ─
    print("⟳  Pre-detecting face crops across all dataset images...", end="  ", flush=True)
    t_det = time.perf_counter()
    face_cache = precompute_face_crops(dataset_dir, detector)
    n_found = sum(1 for v in face_cache.values() if v is not None)
    print(f"✓  {n_found}/{len(face_cache)} faces found  ({time.perf_counter() - t_det:.1f}s)\n")

    # ── Step 3: Initialise primary backend ────────────────────────
    print(f"⟳  Initialising primary backend ({args.backend})...", end="  ", flush=True)
    primary_embedder = FaceEmbedder(backend=args.backend, onnx_model_path=ONNX_MODEL_PATH)
    print(f"✓  {primary_embedder.backend_name} ready.")

    print("⟳  Building prototypes for primary backend...", end="  ", flush=True)
    primary_protos, primary_names = build_in_memory_prototypes(
        dataset_dir, detector, primary_embedder, face_cache=face_cache
    )
    print(f"✓  {len(primary_names)} identities: {', '.join(primary_names.tolist())}\n")

    primary_label = primary_embedder.backend_name

    # ── Sweep mode ─────────────────────────────────────────────────
    if args.sweep:
        print(f"  Running threshold sweep: {args.sweep_values}\n")
        sweep_data = []
        for t in sorted(set(args.sweep_values)):
            r, _, _ = evaluate(
                dataset_dir=dataset_dir,
                prototypes=primary_protos,
                class_names=primary_names,
                detector=detector,
                embedder=primary_embedder,
                threshold=t,
                metric=args.metric,
                quiet=True,
                face_cache=face_cache,
            )
            sweep_data.append({
                "threshold": t,
                "sensitivity": float(np.mean([x.sensitivity for x in r])),
                "specificity": float(np.mean([x.specificity for x in r])),
                "accuracy":    float(np.mean([x.accuracy    for x in r])),
                "f1":          float(np.mean([x.f1          for x in r])),
            })
        print("  ── Threshold Sweep Results ────────────────────────────────────────")
        print_sweep_table(sweep_data)

        best_t = max(sweep_data, key=lambda x: x["accuracy"])["threshold"]
        best_results, _, _ = evaluate(
            dataset_dir=dataset_dir,
            prototypes=primary_protos,
            class_names=primary_names,
            detector=detector,
            embedder=primary_embedder,
            threshold=best_t,
            metric=args.metric,
            quiet=True,
            face_cache=face_cache,
        )
        all_models: Dict[str, Dict[str, float]] = {primary_label: _metrics_from_results(best_results)}
        if not args.no_compare:
            all_models.update(
                _run_backends_parallel(
                    COMPARISON_BACKENDS, primary_embedder.backend_name,
                    dataset_dir, detector, best_t, args.metric, face_cache,
                )
            )
        print(f"\n  ══ Unified Model Comparison (threshold = {best_t:.2f}) {'═' * 22}")
        print_comparison_table(all_models, primary_label)
        print_threshold_recommendation(best_results, best_t)
        return

    # ── Single threshold mode ──────────────────────────────────────
    t0 = time.perf_counter()
    print(f"  Evaluating primary model at threshold = {args.threshold:.2f} | metric = {args.metric}\n")

    results, total, no_face = evaluate(
        dataset_dir=dataset_dir,
        prototypes=primary_protos,
        class_names=primary_names,
        detector=detector,
        embedder=primary_embedder,
        threshold=args.threshold,
        metric=args.metric,
        quiet=args.quiet,
        face_cache=face_cache,
    )

    elapsed = time.perf_counter() - t0
    print(f"\n  Total images       : {total}")
    print(f"  No face detected   : {no_face}")
    print(f"  Successfully tested: {total - no_face}")
    print(f"  Evaluation time    : {elapsed:.2f}s")

    print(f"\n  ── Per-Identity Metrics — {primary_label} (threshold = {args.threshold:.2f}) ────")
    print_per_identity_table(results)

    # ── Collect all model metrics for the unified comparison table ─
    all_model_metrics: Dict[str, Dict[str, float]] = {primary_label: _metrics_from_results(results)}

    if not args.no_compare:
        print(f"\n  ── Evaluating all comparison backends (parallel) ───────────────────────")
        print(f"  (threshold = {args.threshold:.2f} | metric = {args.metric})\n")
        all_model_metrics.update(
            _run_backends_parallel(
                COMPARISON_BACKENDS, primary_embedder.backend_name,
                dataset_dir, detector, args.threshold, args.metric, face_cache,
            )
        )

    print(f"\n  {'═' * 72}")
    print(f"  Unified Model Comparison Report  (threshold = {args.threshold:.2f})")
    print(f"  {'═' * 72}")
    print_comparison_table(all_model_metrics, primary_label)

    print_threshold_recommendation(results, args.threshold)
    print("═" * 72 + "\n")


if __name__ == "__main__":
    main()
