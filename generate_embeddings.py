"""Feature extraction module for few-shot face recognition.

Purpose:
    Converts enrolled face images into compact numeric embeddings and stores
    them along with class labels and class prototypes.

Role in pipeline:
    This module is the bridge between raw image enrollment and recognition.
    It transforms image-space data into embedding-space representations.

Few-shot contribution:
    By computing class prototypes (mean embedding per identity), this module
    enables prototype-based matching that works effectively with a small
    number of samples per class.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils import (
    FaceEmbedder,
    build_augmented_prototypes,
    compute_class_prototypes,
    detect_faces,
    ensure_dir,
    get_identity_folders,
    load_face_detector,
)

try:
    import cv2
except ImportError:
    raise ImportError("opencv-python is required. Install with: pip install opencv-python")


def generate_embeddings(
    dataset_dir: str = "dataset",
    embeddings_dir: str = "embeddings",
    backend: str = "auto",
    onnx_model_path: str = "models/arcface.onnx",
    insightface_model: str = "buffalo_l",
) -> None:
    """Generate embeddings, labels, and prototypes from dataset images.

    Function name:
        generate_embeddings

    Purpose:
        Iterates through each enrolled identity, detects valid faces, extracts
        embeddings using selected backend, and saves all artifacts needed for
        fast prototype-based recognition.

    Parameters:
        dataset_dir (str): Root directory containing identity subfolders.
        embeddings_dir (str): Output directory for numpy artifacts.
        backend (str): Embedding backend selection (auto/facenet/onnx).
        onnx_model_path (str): Path to ONNX embedding model.

    Returns:
        None

    Role in face recognition process:
        Implements feature extraction + embedding storage stages. These saved
        vectors are later used for similarity computation at inference time.
    """
    # Ensure output directory exists before writing numpy artifacts.
    ensure_dir(embeddings_dir)

    folders = get_identity_folders(dataset_dir)
    if not folders:
        raise RuntimeError(f"No identity folders found in: {dataset_dir}")

    # Initialize detection and embedding models once for full dataset pass.
    detector = load_face_detector()
    embedder = FaceEmbedder(
        backend=backend,
        onnx_model_path=onnx_model_path,
        insightface_model=insightface_model,
    )
    print(f"Embedding backend selected: {embedder.backend_name}")

    all_embeddings = []
    all_labels = []

    # Process each identity folder independently (class-wise traversal).
    for folder in folders:
        image_files = (
            list(folder.glob("*.jpg"))
            + list(folder.glob("*.png"))
            + list(folder.glob("*.jpeg"))
        )

        # Count usable samples that pass face-detection and quality checks.
        usable = 0
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Face detection logic for each enrolled image.
            detections = detect_faces(img, detector, min_confidence=0.80, padding=0.10)
            if not detections:
                continue

            # Select dominant face region to avoid multi-face noise.
            largest = max(
                detections,
                key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]),
            )

            # Feature extraction stage: convert cropped face to embedding vector.
            embedding = embedder.embed_face(largest.face_rgb)
            all_embeddings.append(embedding)
            all_labels.append(folder.name)
            usable += 1

        print(f"[INFO] {folder.name}: {usable}/{len(image_files)} usable face images")

    if not all_embeddings:
        raise RuntimeError("No usable face images found in any identity folder.")

    embeddings_array = np.vstack(all_embeddings).astype(np.float32)
    labels_array = np.array(all_labels, dtype=str)

    # ── Guarantee L2-normalisation of every stored embedding ──────────
    # Each embedder should already produce unit vectors, but subtle
    # numeric drift (EMA, dtype casts) can break that.  Re-normalising
    # here ensures cosine comparisons are geometrically meaningful.
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    embeddings_array = embeddings_array / (norms + 1e-8)

    # Embedding storage stage: persist matrix [N x D] and label vector [N].
    emb_path = Path(embeddings_dir) / "embeddings.npy"
    lbl_path = Path(embeddings_dir) / "labels.npy"
    np.save(emb_path, embeddings_array)
    np.save(lbl_path, labels_array)

    # ── Better prototype: outlier-rejection mean ───────────────────────
    # build_augmented_prototypes drops samples whose cosine distance from
    # the centroid exceeds 2σ before averaging.  This prevents a single
    # blurry / mis-aligned enrollment photo from shifting the prototype
    # away from the true face cluster, which is the #1 cause of false
    # rejects in few-shot settings.
    prototypes, class_names = build_augmented_prototypes(embeddings_array, labels_array)
    np.save(Path(embeddings_dir) / "prototypes.npy", prototypes)
    np.save(Path(embeddings_dir) / "class_names.npy", class_names)

    print("=" * 70)
    print("Embeddings generated successfully")
    print(f"Total samples  : {len(embeddings_array)}")
    print(f"Total classes  : {len(class_names)}")
    print(f"Embedding size : {embeddings_array.shape[1]}")
    print(f"Saved file     : {emb_path}")
    print(f"Saved file     : {lbl_path}")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    """Parse command-line configuration for embedding generation.

    Function name:
        parse_args

    Purpose:
        Defines CLI interface for selecting dataset location, output location,
        and embedding backend.

    Parameters:
        None

    Returns:
        argparse.Namespace: Parsed runtime configuration values.

    Role in face recognition process:
        Enables controlled experimentation of feature extraction settings for
        academic evaluation and reproducibility.
    """
    parser = argparse.ArgumentParser(description="Generate face embeddings from dataset")
    parser.add_argument("--dataset-dir", default="dataset", help="Dataset root directory")
    parser.add_argument("--embeddings-dir", default="embeddings", help="Output directory for embeddings")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "facenet", "onnx", "insightface"],
        help="Embedding backend (insightface = ArcFace buffalo_l, best accuracy)",
    )
    parser.add_argument(
        "--onnx-model-path",
        default="models/arcface.onnx",
        help="Path to ONNX face embedding model",
    )
    parser.add_argument(
        "--insightface-model",
        default="buffalo_l",
        choices=["buffalo_l", "buffalo_sc"],
        help="InsightFace model pack (buffalo_l = ResNet-50 ArcFace, best accuracy)",
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
    )
