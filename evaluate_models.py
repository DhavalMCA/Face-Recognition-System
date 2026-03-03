"""evaluate_models.py — Multi-Model Comparison for Few-Shot Face Recognition.

Compares our proposed FewShotFace (ArcFace/InsightFace backbone) against five
traditional deep-learning architectures used in the reference paper:
    VGG-16, ResNet-50, AlexNet, GoogLeNet (Inception v1), MobileNet V2.

All models use the same evaluation protocol:
    • 50 % of each identity's face crops → prototype (support set).
    • Remaining 50 % → query set.
    • 1-NN cosine similarity against prototypes → predicted label.
    • Macro-averaged Sensitivity, Specificity, and Accuracy reported.

Our model additionally applies:
    • InsightFace ArcFace buffalo_l (512-d, ResNet-50, ~99.7 % LFW accuracy).
    • Outlier-cleaned prototype building (rejects embeddings >2σ from centroid).
    • Test-time augmentation (TTA): 5 augmented embeddings averaged per query.

Usage:
    python evaluate_models.py
    python evaluate_models.py --dataset dataset --train-ratio 0.5
    python evaluate_models.py --no-tta   # disable TTA for speed
"""

from __future__ import annotations

import argparse
import os
import random
import ssl
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Fix SSL certificate issues on some Windows environments.
ssl._create_default_https_context = ssl._create_unverified_context

# Fix Intel MKL duplicate symbol warning on Windows.
if os.name == "nt":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, confusion_matrix

import sys
_PROJECT_DIR = str(Path(__file__).resolve().parent)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from utils import (
    FaceEmbedder,
    _INSIGHTFACE_AVAILABLE,
    _tta_embeddings,
    build_augmented_prototypes,
    detect_faces,
    get_identity_folders,
    load_face_detector,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Baseline feature extractors (ImageNet-pretrained, no face-specific tuning)
# ══════════════════════════════════════════════════════════════════════════════

class BaselineEmbedder:
    """ImageNet-pretrained CNN used as a fixed feature extractor.

    The classification head is removed and global-average-pooling is applied
    to the final feature map to obtain an L2-normalised embedding vector.

    Supported architectures: VGG-16, ResNet-50, AlexNet, GoogLeNet, MobileNet.
    """

    _ARCH_MAP = {
        "VGG-16":    ("vgg16",        models.VGG16_Weights.IMAGENET1K_V1),
        "ResNet-50": ("resnet50",     models.ResNet50_Weights.IMAGENET1K_V1),
        "AlexNet":   ("alexnet",      models.AlexNet_Weights.IMAGENET1K_V1),
        "GoogLeNet": ("googlenet",    models.GoogLeNet_Weights.IMAGENET1K_V1),
        "MobileNet": ("mobilenet_v2", models.MobileNet_V2_Weights.IMAGENET1K_V1),
    }

    def __init__(self, model_name: str, target_size: int = 224) -> None:
        """Load pretrained backbone and strip the classification head.

        Parameters:
            model_name (str): One of the keys in _ARCH_MAP.
            target_size (int): Spatial resize before forward pass.
        """
        if model_name not in self._ARCH_MAP:
            raise ValueError(
                f"Unknown baseline '{model_name}'. "
                f"Choose from: {list(self._ARCH_MAP)}"
            )
        self.model_name = model_name
        self.target_size = target_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        arch_fn, weights = self._ARCH_MAP[model_name]
        base = getattr(models, arch_fn)(weights=weights)

        if model_name == "VGG-16":
            # Use only the convolutional feature extractor.
            self.backbone = base.features
        elif model_name in ("ResNet-50", "GoogLeNet"):
            # Remove the final fully-connected layer.
            self.backbone = nn.Sequential(*list(base.children())[:-1])
        elif model_name == "AlexNet":
            # Features + adaptive pooling (skip the classifier).
            self.backbone = nn.Sequential(base.features, base.avgpool)
        elif model_name == "MobileNet":
            self.backbone = base.features
        else:
            self.backbone = base

        self.backbone = self.backbone.eval().to(self.device)

    # ImageNet normalization parameters.
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, face_rgb: np.ndarray) -> torch.Tensor:
        """Resize → ImageNet normalize → CHW tensor."""
        resized = cv2.resize(face_rgb, (self.target_size, self.target_size))
        norm = (resized.astype(np.float32) / 255.0 - self._MEAN) / self._STD
        tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device).float()

    def embed(self, face_rgb: np.ndarray) -> np.ndarray:
        """Extract L2-normalised embedding via global average pooling.

        Parameters:
            face_rgb (np.ndarray): RGB face crop (any size).

        Returns:
            np.ndarray: L2-normalised 1-D feature vector.
        """
        tensor = self._preprocess(face_rgb)
        with torch.no_grad():
            feats = self.backbone(tensor)
            # Apply global average pooling if spatial dims remain.
            if feats.dim() > 2:
                feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1))
            vec = feats.view(feats.size(0), -1).cpu().numpy()[0].astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-8
        return vec


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset preparation
# ══════════════════════════════════════════════════════════════════════════════

def prepare_dataset_split(
    dataset_dir: str,
    train_ratio: float = 0.5,
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]:
    """Split enrolled face crops into support (train) and query (test) sets.

    Parameters:
        dataset_dir (str): Root directory with one sub-folder per identity.
        train_ratio (float): Fraction of images used to build prototypes.

    Returns:
        Tuple of (train_data, test_data), where each is a dict mapping
        identity name → list of RGB face crops.
    """
    folders = get_identity_folders(dataset_dir)
    detector = load_face_detector()

    train_data: Dict[str, List[np.ndarray]] = {}
    test_data:  Dict[str, List[np.ndarray]] = {}

    print(f"\n  Loading dataset from '{dataset_dir}' "
          f"(train ratio = {train_ratio * 100:.0f} %)...")

    total_train = total_test = 0

    for folder in folders:
        image_files = (
            list(folder.glob("*.jpg"))
            + list(folder.glob("*.jpeg"))
            + list(folder.glob("*.png"))
            + list(folder.glob("*.bmp"))
        )

        valid_faces: List[np.ndarray] = []
        for img_path in sorted(image_files):
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            dets = detect_faces(bgr, detector, min_confidence=0.75, padding=0.12)
            if dets:
                largest = max(
                    dets,
                    key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]),
                )
                valid_faces.append(largest.face_rgb)

        if len(valid_faces) < 2:
            print(f"  ⚠  Skipping '{folder.name}' — need ≥2 valid face crops "
                  f"(found {len(valid_faces)}).")
            continue

        random.seed(42)
        random.shuffle(valid_faces)

        split = max(1, int(len(valid_faces) * train_ratio))
        train_data[folder.name] = valid_faces[:split]
        test_data[folder.name]  = valid_faces[split:]

        total_train += len(train_data[folder.name])
        total_test  += len(test_data[folder.name])
        print(f"    {folder.name:<20}  support={len(train_data[folder.name])}  "
              f"query={len(test_data[folder.name])}")

    print(f"\n  Identities : {len(train_data)}")
    print(f"  Support set: {total_train} images")
    print(f"  Query set  : {total_test} images")
    return train_data, test_data


# ══════════════════════════════════════════════════════════════════════════════
#  Core evaluation engine
# ══════════════════════════════════════════════════════════════════════════════

def _build_prototype(
    faces: List[np.ndarray],
    embedder,
    use_tta: bool,
    is_our_model: bool,
) -> np.ndarray:
    """Build one class prototype from a list of face crops.

    For our model, outlier-cleaned averaging is applied.
    For baselines, plain mean is used (matches the paper methodology).
    """
    embs: List[np.ndarray] = []
    for face in faces:
        if use_tta and is_our_model:
            embs.append(_tta_embeddings(face, embedder, n_augments=5))
        else:
            embs.append(embedder.embed(face))

    arr = np.array(embs, dtype=np.float32)   # [N, D]

    if is_our_model and len(arr) > 1:
        # Outlier rejection: drop embeddings >2σ from the centroid.
        centroid = arr.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-8
        sims = arr @ centroid
        m, s = sims.mean(), sims.std()
        mask = sims >= (m - 2.0 * s)
        arr = arr[mask] if mask.sum() > 0 else arr

    proto = arr.mean(axis=0).astype(np.float32)
    proto /= np.linalg.norm(proto) + 1e-8
    return proto


def evaluate_model(
    model_label: str,
    embedder,
    train_data: Dict[str, List[np.ndarray]],
    test_data:  Dict[str, List[np.ndarray]],
    use_tta: bool = False,
    is_our_model: bool = False,
) -> Dict:
    """Evaluate a single embedder on the prepared dataset split.

    Protocol:
        1. Build one prototype per class from support images.
        2. Embed each query image.
        3. Assign query to the class with highest cosine similarity.
        4. Compute macro Sensitivity, Specificity, Accuracy.

    Parameters:
        model_label (str): Display name for the table.
        embedder: Object with an ``embed(face_rgb)`` method.
        train_data: Support-set dict {identity: [face_rgb, ...]}.
        test_data:  Query-set dict  {identity: [face_rgb, ...]}.
        use_tta (bool): Apply test-time augmentation on queries.
        is_our_model (bool): Apply outlier-cleaned prototype building.

    Returns:
        Dict with keys Model, Specificity, Sensitivity, Accuracy (all floats).
    """
    print(f"  Evaluating  {model_label} ...", end="  ", flush=True)
    t0 = time.perf_counter()

    # ── Build prototypes ──────────────────────────────────────────────
    prototypes: Dict[str, np.ndarray] = {}
    for name, faces in train_data.items():
        prototypes[name] = _build_prototype(faces, embedder, use_tta, is_our_model)

    class_names  = list(prototypes.keys())
    proto_matrix = np.array([prototypes[n] for n in class_names], dtype=np.float32)

    # ── Predict on query set ──────────────────────────────────────────
    y_true: List[str] = []
    y_pred: List[str] = []

    for true_name, faces in test_data.items():
        if true_name not in prototypes:
            continue
        for face in faces:
            if use_tta and is_our_model:
                emb = _tta_embeddings(face, embedder, n_augments=5)
            else:
                emb = embedder.embed(face)

            sims     = proto_matrix @ emb          # cosine similarity
            pred_idx = int(np.argmax(sims))
            y_true.append(true_name)
            y_pred.append(class_names[pred_idx])

    # ── Metrics ───────────────────────────────────────────────────────
    if not y_true:
        elapsed = time.perf_counter() - t0
        print(f"no queries found.  ({elapsed:.1f}s)")
        return {"Model": model_label, "Specificity": 0.0, "Sensitivity": 0.0, "Accuracy": 0.0}

    acc = accuracy_score(y_true, y_pred) * 100.0
    cm  = confusion_matrix(y_true, y_pred, labels=class_names)

    sensitivities: List[float] = []
    specificities: List[float] = []

    for i in range(len(class_names)):
        tp = int(cm[i, i])
        fn = int(np.sum(cm[i, :]) - tp)
        fp = int(np.sum(cm[:, i]) - tp)
        tn = int(np.sum(cm) - (tp + fp + fn))

        sensitivities.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    elapsed = time.perf_counter() - t0
    print(
        f"accuracy={acc:.2f}%  "
        f"({len(y_true)} queries, {elapsed:.1f}s)"
    )

    return {
        "Model":       model_label,
        "Specificity": float(np.mean(specificities)) * 100.0,
        "Sensitivity": float(np.mean(sensitivities)) * 100.0,
        "Accuracy":    acc,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Reporting
# ══════════════════════════════════════════════════════════════════════════════

def print_results_table(results: List[Dict], our_label: str) -> None:
    """Print a formatted comparison table to stdout."""
    # Sort by accuracy descending so the best model is at the top.
    results_sorted = sorted(results, key=lambda r: r["Accuracy"], reverse=True)

    table = PrettyTable()
    table.field_names = [
        "Model",
        "Specificity (%)",
        "Sensitivity (%)",
        "Accuracy (%)",
    ]
    table.align["Model"] = "l"
    table.float_format = ".2"

    for r in results_sorted:
        marker = "  ◄ OUR MODEL" if r["Model"] == our_label else ""
        table.add_row([
            r["Model"] + marker,
            f"{r['Specificity']:.2f}",
            f"{r['Sensitivity']:.2f}",
            f"{r['Accuracy']:.2f}",
        ])

    # Rank our model.
    our_acc    = next(r["Accuracy"] for r in results if r["Model"] == our_label)
    baseline_accs = sorted(
        [r["Accuracy"] for r in results if r["Model"] != our_label],
        reverse=True,
    )
    rank = next(
        (i + 1 for i, v in enumerate(baseline_accs) if our_acc > v),
        len(baseline_accs) + 1,
    )

    sep = "═" * 72
    print(f"\n{sep}")
    print("  Table: Comparison of Deep Learning Models for Face Recognition")
    print(f"{sep}\n")
    print(table)
    print(
        f"\n  ★  Our model accuracy : {our_acc:.2f}%\n"
        f"  ★  Rank among baselines: #{rank} of {len(baseline_accs)}\n"
    )
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare FewShotFace (ArcFace) vs traditional CNN baselines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        default="dataset",
        help="Path to dataset root (default: dataset).",
    )
    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.5,
        help="Fraction of images per identity used as prototypes (default: 0.5).",
    )
    p.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable test-time augmentation for the proposed model (faster).",
    )
    p.add_argument(
        "--insightface-model",
        default="buffalo_l",
        choices=["buffalo_l", "buffalo_sc"],
        help=(
            "InsightFace model pack: "
            "'buffalo_l' = ResNet-50 ArcFace (high accuracy, default); "
            "'buffalo_sc' = MobileNet ArcFace (fast)."
        ),
    )
    p.add_argument(
        "--baselines-only",
        action="store_true",
        help="Skip the proposed model and run only the five baselines.",
    )
    return p.parse_args()


def main() -> None:
    """Entry point: evaluate all models and print comparison table."""
    import os as _os
    _os.chdir(_PROJECT_DIR)

    args = parse_args()
    use_tta = not args.no_tta

    # ── Prepare data ──────────────────────────────────────────────────
    train_data, test_data = prepare_dataset_split(
        args.dataset, train_ratio=args.train_ratio
    )
    if not train_data or not test_data:
        print("\n  ✗  Not enough data to evaluate. "
              "Ensure each identity folder has ≥2 face images.")
        return

    results: List[Dict] = []
    our_label = ""

    # ── Proposed model: FewShotFace (InsightFace ArcFace) ─────────────
    if not args.baselines_only:
        print(f"\n{'─' * 60}")
        tta_tag    = "+TTA" if use_tta else ""
        pack       = args.insightface_model
        our_label  = f"FewShotFace-ArcFace ({pack}{tta_tag})"

        if _INSIGHTFACE_AVAILABLE:
            our_embedder = FaceEmbedder(
                backend="insightface",
                insightface_model=pack,
            )
            print(f"  Backbone: InsightFace {pack} (ArcFace, 512-d)")
        else:
            print(
                "  ⚠  insightface not found — falling back to FaceNet.\n"
                "     Install it:  pip install insightface"
            )
            our_embedder = FaceEmbedder(backend="facenet")
            our_label    = f"FewShotFace-FaceNet{tta_tag}"

        res = evaluate_model(
            model_label=our_label,
            embedder=our_embedder,
            train_data=train_data,
            test_data=test_data,
            use_tta=use_tta,
            is_our_model=True,
        )
        results.append(res)

    # ── Five paper baseline models ────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  Baselines (ImageNet pretrained, global-avg-pool features):\n")

    baseline_names = [
        "VGG-16",
        "ResNet-50",
        "AlexNet",
        "GoogLeNet",
        "MobileNet",
    ]

    for name in baseline_names:
        try:
            embedder = BaselineEmbedder(name)
            res = evaluate_model(
                model_label=name,
                embedder=embedder,
                train_data=train_data,
                test_data=test_data,
                use_tta=False,       # Baselines don't get TTA (fair comparison).
                is_our_model=False,
            )
            results.append(res)
        except Exception as exc:
            print(f"  ✗  {name} failed: {exc}")

    # ── Print table ────────────────────────────────────────────────────
    if results:
        if not our_label and results:
            our_label = results[0]["Model"]
        print_results_table(results, our_label)
    else:
        print("\n  No results to display.")


if __name__ == "__main__":
    main()
