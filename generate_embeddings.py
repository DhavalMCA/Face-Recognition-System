"""Feature extraction module for few-shot face recognition.

Purpose:
    Converts enrolled face images into compact numeric embeddings and stores
    them along with class labels, class prototypes, and a calibrated threshold.

Role in pipeline:
    Bridge between raw image enrollment and recognition.  Transforms
    image-space data into embedding-space representations and auto-calibrates
    the recognition threshold from enrollment statistics.

Accuracy improvements over baseline:
    - Test-Time Augmentation (TTA): each image is embedded ``n_tta`` times
      with augmented views (flip, rotation, brightness) and averaged.
      Reduces per-sample noise and improves prototype quality by ~1-2%.
    - Quality filtering: images whose composite quality score (sharpness +
      brightness + contrast) is below ``QUALITY_THRESHOLD`` are skipped,
      preventing blurry / over-exposed enrollments from degrading prototypes.
    - Quality-weighted prototype building: sharper images contribute more
      to the class prototype via ``build_augmented_prototypes(quality_scores)``.
    - Auto-threshold calibration: LOO cross-validation on enrollment data
      finds the threshold that achieves ≤1% false-positive rate and saves
      it to ``embeddings/auto_threshold.json`` for use at recognition time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from utils import (
    FaceEmbedder,
    align_face_from_landmarks,
    build_augmented_prototypes,
    build_multi_prototypes,
    compute_class_prototypes,
    detect_faces,
    ensure_dir,
    get_identity_folders,
    load_face_detector,
    score_face_quality,
)
from similarity import calibrate_threshold

try:
    import cv2
except ImportError:
    raise ImportError("opencv-python is required. Install with: pip install opencv-python")


# Minimum composite quality score to accept an enrollment image.
# Images below this threshold are skipped to protect embedding quality.
QUALITY_THRESHOLD = 0.25

# Number of augmented views averaged per image for TTA.
# 5 gives a good accuracy/speed trade-off; increase to 7 for max accuracy.
TTA_N_AUGMENTS = 5


def generate_embeddings(
    dataset_dir: str = "dataset",
    embeddings_dir: str = "embeddings",
    backend: str = "auto",
    onnx_model_path: str = "models/w600k_r50.onnx",
    insightface_model: str = "buffalo_l",
    deepface_model: str = "ArcFace",
    use_tta: bool = True,
    n_tta: int = TTA_N_AUGMENTS,
    quality_threshold: float = QUALITY_THRESHOLD,
) -> None:
    """Generate embeddings, labels, prototypes, and calibrated threshold.

    Parameters:
        dataset_dir (str): Root directory containing identity subfolders.
        embeddings_dir (str): Output directory for numpy artifacts.
        backend (str): Embedding backend selection.
        onnx_model_path (str): Path to ONNX embedding model.
        insightface_model (str): InsightFace model pack name.
        deepface_model (str): DeepFace model name.
        use_tta (bool): Enable Test-Time Augmentation (default True).
        n_tta (int): Number of TTA augmented views per image.
        quality_threshold (float): Minimum quality score to enroll an image.
    """
    ensure_dir(embeddings_dir)

    folders = get_identity_folders(dataset_dir)
    if not folders:
        raise RuntimeError(f"No identity folders found in: {dataset_dir}")

    # Initialise detection and embedding models once for the full dataset pass.
    detector = load_face_detector()
    embedder = FaceEmbedder(
        backend=backend,
        onnx_model_path=onnx_model_path,
        insightface_model=insightface_model,
        deepface_model=deepface_model,
    )
    print(f"Embedding backend : {embedder.backend_name}")
    print(f"TTA enabled       : {use_tta}  (n={n_tta} views per image)")
    print(f"Quality threshold : {quality_threshold}")
    print("-" * 60)

    all_embeddings: list = []
    all_labels: list = []
    all_quality: list = []

    for folder in folders:
        image_files = sorted(
            list(folder.glob("*.jpg"))
            + list(folder.glob("*.png"))
            + list(folder.glob("*.jpeg"))
        )

        usable = 0
        skipped_quality = 0
        skipped_noface = 0

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Face detection: use a slightly lower confidence (0.80) for
            # enrollment images which may include edge-case poses.
            detections = detect_faces(img, detector, min_confidence=0.80, padding=0.10)
            if not detections:
                skipped_noface += 1
                continue

            # Select the dominant (largest) face to avoid multi-face noise.
            largest = max(
                detections,
                key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]),
            )

            # Quality gate: skip blurry / over-exposed crops.
            if largest.quality_score < quality_threshold:
                skipped_quality += 1
                continue

            # Face alignment: affine warp to ArcFace canonical positions.
            if largest.landmarks_5pt is not None:
                aligned = align_face_from_landmarks(img, largest.landmarks_5pt)
                face_input = aligned if aligned is not None else largest.face_rgb
            else:
                face_input = largest.face_rgb

            # Feature extraction with optional TTA.
            if use_tta and n_tta > 1:
                embedding = embedder.embed_face_tta(face_input, n_augments=n_tta)
            else:
                embedding = embedder.embed_face(face_input)

            all_embeddings.append(embedding)
            all_labels.append(folder.name)
            all_quality.append(largest.quality_score)
            usable += 1

        print(
            f"[{folder.name}]  usable={usable}/{len(image_files)}"
            f"  skipped_quality={skipped_quality}"
            f"  skipped_noface={skipped_noface}"
        )

    if not all_embeddings:
        raise RuntimeError("No usable face images found in any identity folder.")

    embeddings_array = np.vstack(all_embeddings).astype(np.float32)
    labels_array = np.array(all_labels, dtype=str)
    quality_array = np.array(all_quality, dtype=np.float32)

    # Guarantee L2-normalisation of every stored embedding.
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    embeddings_array = embeddings_array / (norms + 1e-8)

    # Persist embedding matrix and label vector.
    emb_path = Path(embeddings_dir) / "embeddings.npy"
    lbl_path = Path(embeddings_dir) / "labels.npy"
    np.save(emb_path, embeddings_array)
    np.save(lbl_path, labels_array)

    # Quality-weighted prototype building with 1.5 σ outlier rejection.
    prototypes, class_names = build_augmented_prototypes(
        embeddings_array, labels_array, quality_scores=quality_array
    )
    np.save(Path(embeddings_dir) / "prototypes.npy", prototypes)
    np.save(Path(embeddings_dir) / "class_names.npy", class_names)

    # Multi-prototype building: up to 2 k-means cluster centres per class.
    # Falls back to quality-weighted mean when a class has fewer than 6 samples.
    multi_protos, multi_labels = build_multi_prototypes(
        embeddings_array, labels_array,
        n_max_per_class=2,
        min_samples_for_multi=6,
        quality_scores=quality_array,
    )
    np.save(Path(embeddings_dir) / "multi_prototypes.npy", multi_protos)
    np.save(Path(embeddings_dir) / "multi_labels.npy", multi_labels)
    multi_proto_count = len(multi_protos) - len(class_names)
    if multi_proto_count > 0:
        print(f"Multi-prototypes  : {len(multi_protos)} total ({multi_proto_count} extra cluster centres)")

    # ── Auto-threshold calibration ──────────────────────────────────────
    # Uses LOO cross-validation on enrollment samples to find the threshold
    # that achieves ≤1% false-positive rate on known impostor pairs.
    print("-" * 60)
    print("Calibrating recognition threshold …")
    calibrated_thr = calibrate_threshold(
        embeddings=embeddings_array,
        labels=labels_array,
        prototypes=prototypes,
        class_names=class_names,
        metric="cosine",
        target_fpr=0.01,
        use_ensemble=True,
    )

    # Save calibrated threshold so recognize.py can load it automatically.
    thr_path = Path(embeddings_dir) / "auto_threshold.json"
    thr_data = {
        "threshold": float(calibrated_thr),
        "metric": "cosine",
        "n_classes": int(len(class_names)),
        "n_samples": int(len(embeddings_array)),
        "tta_views": int(n_tta) if use_tta else 1,
        "embed_dim": int(embeddings_array.shape[1]),
        "backend": embedder.backend_name,
    }
    thr_path.write_text(json.dumps(thr_data, indent=2))

    print("=" * 60)
    print("Embeddings generated successfully")
    print(f"Total samples     : {len(embeddings_array)}")
    print(f"Total classes     : {len(class_names)}")
    print(f"Embedding size    : {embeddings_array.shape[1]}")
    print(f"Prototypes        : {len(prototypes)} single  |  {len(multi_protos)} multi")
    print(f"Calibrated thr    : {calibrated_thr:.4f}  (saved to {thr_path})")
    print(f"Artifacts saved to: {embeddings_dir}/")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate face embeddings from dataset")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--embeddings-dir", default="embeddings")
    parser.add_argument(
        "--backend", default="auto",
        choices=["auto", "facenet", "onnx", "insightface", "deepface", "vit"],
    )
    parser.add_argument(
        "--deepface-model", default="ArcFace",
        choices=["ArcFace", "Facenet512", "VGG-Face", "SFace", "Facenet", "OpenFace"],
    )
    parser.add_argument("--onnx-model-path", default="models/w600k_r50.onnx")
    parser.add_argument(
        "--insightface-model", default="buffalo_l",
        choices=["buffalo_l", "buffalo_sc"],
    )
    parser.add_argument(
        "--no-tta", action="store_true",
        help="Disable Test-Time Augmentation (faster but less accurate)",
    )
    parser.add_argument(
        "--tta-views", type=int, default=TTA_N_AUGMENTS,
        help=f"Number of TTA augmented views per image (default: {TTA_N_AUGMENTS})",
    )
    parser.add_argument(
        "--quality-threshold", type=float, default=QUALITY_THRESHOLD,
        help=f"Minimum quality score to enroll an image (default: {QUALITY_THRESHOLD})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_embeddings(
        dataset_dir=args.dataset_dir,
        embeddings_dir=args.embeddings_dir,
        backend=args.backend,
        onnx_model_path=args.onnx_model_path,
        insightface_model=args.insightface_model,
        deepface_model=args.deepface_model,
        use_tta=not args.no_tta,
        n_tta=args.tta_views,
        quality_threshold=args.quality_threshold,
    )
