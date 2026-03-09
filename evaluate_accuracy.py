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
from pathlib import Path
from typing import Dict, List, Tuple

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
ONNX_MODEL_PATH = "models/arcface.onnx"
DEFAULT_THRESHOLD = 0.70
DEFAULT_METRIC = "cosine"
SWEEP_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

# ── Published comparison baselines ───────────────────────────────────
# Source: Comparative study metrics for face recognition benchmarks.
BENCHMARK_MODELS: Dict[str, Dict[str, float]] = {
    "VGG-16":     {"specificity": 98.65, "sensitivity": 99.45, "accuracy": 99.00},
    "ResNet-50":  {"specificity": 92.50, "sensitivity": 95.00, "accuracy": 94.00},
    "AlexNet":    {"specificity": 84.00, "sensitivity": 88.46, "accuracy": 87.70},
    "GoogleNet":  {"specificity": 88.24, "sensitivity": 90.00, "accuracy": 89.00},
    "MobileNet":  {"specificity": 86.00, "sensitivity": 83.40, "accuracy": 88.30},
}


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


def embed_image(
    img_path: Path,
    detector,
    embedder: FaceEmbedder,
    min_confidence: float = 0.82,
) -> Tuple[bool, np.ndarray | None]:
    """Read one image, detect a face, extract its embedding.

    A slightly lower min_confidence (0.82 vs runtime 0.92) accepts edge-case
    angles common in mobile photos.

    Args:
        img_path:       Path to image file.
        detector:       MTCNN detector.
        embedder:       FaceEmbedder instance.
        min_confidence: Minimum MTCNN face detection confidence.

    Returns:
        (success, embedding_or_None)
    """
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
) -> Tuple[List[EvalResult], int, int]:
    """Run full evaluation across all enrolled identities.

    For each image in each identity folder:
      1. Detect and embed the face.
      2. Predict identity using predict_with_prototypes.
      3. Update one-vs-rest counters for EVERY enrolled class.

    Args:
        dataset_dir:  Root dataset directory.
        prototypes:   Prototype matrix [num_classes, dim].
        class_names:  Class name array [num_classes].
        detector:     MTCNN detector.
        embedder:     FaceEmbedder.
        threshold:    Recognition threshold.
        metric:       'cosine' or 'euclidean'.
        quiet:        Suppress per-image output.

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
            ok, embedding = embed_image(img_path, detector, embedder)

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


def print_comparison_table(
    results: List[EvalResult],
    threshold: float,
    backend_name: str,
) -> None:
    """Print comparison table: your model vs published baselines."""
    # Compute macro-averaged metrics for your model.
    my_sensitivity = float(np.mean([r.sensitivity for r in results]))
    my_specificity = float(np.mean([r.specificity for r in results]))
    my_accuracy = float(np.mean([r.accuracy for r in results]))

    # Collect all models (baselines + yours) into one list.
    all_models: Dict[str, Dict[str, float]] = {**BENCHMARK_MODELS}
    my_label = f"FewShotFace ({backend_name}, t={threshold:.2f})"
    all_models[my_label] = {
        "specificity": my_specificity,
        "sensitivity": my_sensitivity,
        "accuracy": my_accuracy,
    }

    col_model = max(len(k) for k in all_models) + 2
    line_width = col_model + 17 + 14 + 11 + 4
    sep = "─" * line_width

    print(f"\n  {'Model':<{col_model}}  {'Specificity (%)':>15}  {'Sensitivity (%)':>15}  {'Accuracy (%)':>14}")
    print("  " + sep)

    for model_name, m in all_models.items():
        tag = " ◄ YOUR MODEL" if model_name == my_label else ""
        print(
            f"  {model_name:<{col_model}}  "
            f"{m['specificity']:>14.2f}  "
            f"{m['sensitivity']:>15.2f}  "
            f"{m['accuracy']:>13.2f}  "
            f"{tag}"
        )

    print("  " + sep)

    # Show rank among baselines.
    acc_values = sorted([v["accuracy"] for v in BENCHMARK_MODELS.values()], reverse=True)
    rank = next((i + 1 for i, v in enumerate(acc_values) if my_accuracy > v), len(acc_values) + 1)
    print(
        f"\n  Macro-averaged accuracy of your model: {my_accuracy:.2f}%\n"
        f"  Achieves rank #{rank} out of {len(BENCHMARK_MODELS)} baseline models."
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
    sens = np.mean([r.sensitivity for r in results])
    spec = np.mean([r.specificity for r in results])

    print(
        """
  ┌─────────────────────────────────────────────────────────────────┐
  │                 THRESHOLD RECOMMENDATION                        │
  ├─────────────────────────────────────────────────────────────────┤
  │  Current threshold : {thr:.2f}                                    │
  │  Sensitivity (avg) : {sens:.2f}%                                 │
  │  Specificity (avg) : {spec:.2f}%                                 │
  ├─────────────────────────────────────────────────────────────────┤
  │  Problem: Family member confusion (mother recognised as you)    │
  │  Root cause: Shared facial features → similar embeddings        │
  │                                                                 │
  │  Recommended thresholds:                                        │
  │    0.80 – 0.85  High-security (few false positives)             │
  │    0.70 – 0.75  Balanced (good accuracy, low false alarms)      │
  │    0.60 – 0.65  Lenient  (maximises recall, more false pos.)    │
  │                                                                 │
  │  For the family confusion problem:                              │
  │    → Raise threshold to 0.80                                    │
  │    → Enroll your mother as a separate identity                  │
  │      (run:  python gui.py  → Step 1 → "Mother")                 │
  │    → Re-train embeddings (Step 2)                               │
  │                                                                 │
  │  Why few-shot prototypes handle intra-family similarity:        │
  │    • Each enrolled person has their own class prototype         │
  │      (mean embedding of all their samples).                     │
  │    • When two people share facial structure, their prototypes   │
  │      are close together in embedding space.                     │
  │    • A higher threshold forces the model to only accept a       │
  │      face when it is unambiguously close to ONE prototype       │
  │      and clearly farther from all other prototypes.             │
  │    • Adding the mother's samples creates a distinct prototype   │
  │      that the model can use to SEPARATE the two identities,     │
  │      rather than collapsing both onto the son's prototype.      │
  └─────────────────────────────────────────────────────────────────┘
""".format(
            thr=threshold, sens=sens, spec=spec
        )
    )


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
    return parser.parse_args()


def run_single_eval(
    dataset_dir: Path,
    embeddings_dir: Path,
    threshold: float,
    metric: str,
    backend: str,
    quiet: bool,
    detector,
    embedder: FaceEmbedder,
    prototypes: np.ndarray,
    class_names: np.ndarray,
) -> List[EvalResult]:
    """Wrapper that runs one full evaluation pass at `threshold`."""
    results, total, no_face = evaluate(
        dataset_dir=dataset_dir,
        prototypes=prototypes,
        class_names=class_names,
        detector=detector,
        embedder=embedder,
        threshold=threshold,
        metric=metric,
        quiet=quiet,
    )
    if not quiet:
        print(
            f"\n  Total images      : {total}\n"
            f"  No face detected  : {no_face}\n"
            f"  Successfully tested: {total - no_face}"
        )
    return results


def main() -> None:
    """Evaluation entry point."""
    import os
    os.chdir(str(PROJECT_DIR))

    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    embeddings_dir = Path(args.embeddings_dir)

    print(f"\n{'═' * 72}")
    print( "   FewShotFace — Accuracy Evaluation")
    print(f"{'═' * 72}\n")

    # ── Load resources ─────────────────────────────────────────────
    print("⟳  Loading prototypes...", end="  ", flush=True)
    prototypes, class_names = load_prototypes(embeddings_dir)
    print(f"✓  {len(class_names)} enrolled identities: {', '.join(class_names.tolist())}")

    print("⟳  Loading face detector...", end="  ", flush=True)
    detector = load_face_detector()
    print("✓  MTCNN ready.")

    print(f"⟳  Loading {args.backend} embedder...", end="  ", flush=True)
    embedder = FaceEmbedder(backend=args.backend, onnx_model_path=ONNX_MODEL_PATH)
    print(f"✓  {embedder.backend_name} backend ready.\n")

    # ── Sweep mode ─────────────────────────────────────────────────
    if args.sweep:
        print(f"  Running threshold sweep: {args.sweep_values}\n")
        sweep_data = []
        for t in sorted(set(args.sweep_values)):
            r = run_single_eval(
                dataset_dir, embeddings_dir, t, args.metric, args.backend,
                True, detector, embedder, prototypes, class_names,
            )
            sweep_data.append({
                "threshold": t,
                "sensitivity": float(np.mean([x.sensitivity for x in r])),
                "specificity": float(np.mean([x.specificity for x in r])),
                "accuracy":    float(np.mean([x.accuracy    for x in r])),
                "f1":          float(np.mean([x.f1          for x in r])),
            })
        print( "  ── Threshold Sweep Results ────────────────────────────────────────")
        print_sweep_table(sweep_data)

        # Also show comparison table at the most common threshold.
        target_t = 0.70 if 0.70 in args.sweep_values else args.sweep_values[0]
        best_results = run_single_eval(
            dataset_dir, embeddings_dir, target_t, args.metric, args.backend,
            True, detector, embedder, prototypes, class_names,
        )
        print(f"\n  ── Comparison Table (threshold = {target_t:.2f}) ───────────────────────")
        print_comparison_table(best_results, target_t, embedder.backend_name)
        print_threshold_recommendation(best_results, target_t)
        return

    # ── Single threshold mode ──────────────────────────────────────
    t0 = time.perf_counter()
    print(f"  Evaluating at threshold = {args.threshold:.2f} | metric = {args.metric}\n")

    results, total, no_face = evaluate(
        dataset_dir=dataset_dir,
        prototypes=prototypes,
        class_names=class_names,
        detector=detector,
        embedder=embedder,
        threshold=args.threshold,
        metric=args.metric,
        quiet=args.quiet,
    )

    elapsed = time.perf_counter() - t0

    print(f"\n  Total images      : {total}")
    print(f"  No face detected  : {no_face}")
    print(f"  Successfully tested: {total - no_face}")
    print(f"  Evaluation time   : {elapsed:.2f}s")

    print(f"\n  ── Per-Identity Metrics (threshold = {args.threshold:.2f}) ──────────────────")
    print_per_identity_table(results)

    print(f"\n  ── Comparison Table ───────────────────────────────────────────────────")
    print_comparison_table(results, args.threshold, embedder.backend_name)

    print_threshold_recommendation(results, args.threshold)

    print("═" * 72 + "\n")


if __name__ == "__main__":
    main()
