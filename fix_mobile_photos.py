"""fix_mobile_photos.py — Domain-Gap Correction for Mobile Camera Photos.

Problem being solved:
    Mobile camera photos (high-res, warm lighting, varied angles) and webcam
    photos (low-res, indoor lighting, frontal) produce embeddings in different
    regions of feature space.  This causes the recognition engine to fail for
    mobile photos even when the correct person is enrolled.

Fix strategy:
    1. Resize every image to a consistent resolution (224 × 224).
    2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the
       luminance channel to normalize brightness and contrast across all
       lighting conditions — compensates for warm vs. cool white-balance
       differences.
    3. Apply an unsharp-mask filter to sharpen soft/blurry mobile crops and
       bring high-frequency facial details (edges, pores, wrinkles) to the
       same level as webcam images.
    4. Run MTCNN face detection to confirm a face is found and to re-crop and
       align the face region before saving.
    5. Save the preprocessed image *in-place* (overwriting the original) so
       that the next `generate_embeddings.py` run picks up normalized images.

Usage:
    # Process every identity in dataset/ (default)
    python fix_mobile_photos.py

    # Process only one person
    python fix_mobile_photos.py --name Dhaval

    # Preview without saving (dry run)
    python fix_mobile_photos.py --dry-run

    # Save preprocessed copies to a separate output folder
    python fix_mobile_photos.py --output-dir dataset_normalized/

    # Control CLAHE clip limit (higher → more aggressive contrast correction)
    python fix_mobile_photos.py --clip-limit 3.0

    # Skip face detection step (just normalize lighting / sharpness)
    python fix_mobile_photos.py --no-detect
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# ── Project root on path ──────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils import detect_faces, ensure_dir, get_identity_folders, load_face_detector

# ── Constants ────────────────────────────────────────────────────────
TARGET_SIZE = 224          # pixels — consistent input size for both backends
DEFAULT_CLIP_LIMIT = 2.5   # CLAHE clip limit (1-4 is reasonable)
CLAHE_TILE = (8, 8)        # CLAHE tile grid size
SHARPEN_STRENGTH = 0.8     # 0=no sharpening, 1=moderate, >1=aggressive
DATASET_DIR = "dataset"


# ══════════════════════════════════════════════════════════════════════
#  Preprocessing helpers
# ══════════════════════════════════════════════════════════════════════

def apply_clahe(bgr: np.ndarray, clip_limit: float = DEFAULT_CLIP_LIMIT) -> np.ndarray:
    """Apply CLAHE to the luminance channel of a BGR image.

    Converting to LAB and equalising only the L channel avoids color shifts
    while improving contrast and brightening dark mobile photos.

    Args:
        bgr:        Input image in BGR format (uint8).
        clip_limit: CLAHE contrast limit (2-4 eliminates over-amplification).

    Returns:
        uint8 BGR image with enhanced luminance.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=CLAHE_TILE)
    l_eq = clahe.apply(l_channel)

    lab_eq = cv2.merge([l_eq, a_channel, b_channel])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def apply_unsharp_mask(bgr: np.ndarray, strength: float = SHARPEN_STRENGTH) -> np.ndarray:
    """Sharpen an image using an unsharp mask.

    Unsharp masking amplifies the difference between the original and a
    Gaussian-blurred version.  It brings out edge detail that webcam images
    naturally capture but mobile camera JPEG compression can soften.

    Args:
        bgr:      Input image in BGR format (uint8).
        strength: Weight added to the high-frequency component.

    Returns:
        Sharpened uint8 BGR image.
    """
    # Gaussian blur acts as a low-pass filter.
    blurred = cv2.GaussianBlur(bgr, (5, 5), sigmaX=1.0, sigmaY=1.0)
    # high-freq = original − blurred
    high_freq = cv2.subtract(bgr.astype(np.float32), blurred.astype(np.float32))
    sharpened = bgr.astype(np.float32) + strength * high_freq
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def normalize_white_balance(bgr: np.ndarray) -> np.ndarray:
    """Normalize white balance using the Gray World assumption.

    Mobile photos often have a warm (yellowish/reddish) white balance while
    webcam images tend toward neutral.  This method scales each channel so
    its mean equals the global mean brightness, approximately neutralising
    the color cast.

    Args:
        bgr: Input image in BGR format (uint8).

    Returns:
        Color-corrected uint8 BGR image.
    """
    b, g, r = cv2.split(bgr.astype(np.float32))
    mean_r, mean_g, mean_b = r.mean(), g.mean(), b.mean()
    global_mean = (mean_r + mean_g + mean_b) / 3.0 + 1e-6

    r_balanced = np.clip(r * (global_mean / mean_r), 0, 255)
    g_balanced = np.clip(g * (global_mean / mean_g), 0, 255)
    b_balanced = np.clip(b * (global_mean / mean_b), 0, 255)

    return cv2.merge([b_balanced, g_balanced, r_balanced]).astype(np.uint8)


def preprocess_image(
    bgr: np.ndarray,
    clip_limit: float = DEFAULT_CLIP_LIMIT,
    sharpen_strength: float = SHARPEN_STRENGTH,
) -> np.ndarray:
    """Full preprocessing pipeline: resize → WB normalize → CLAHE → sharpen.

    Args:
        bgr:             Input image (any resolution, BGR uint8).
        clip_limit:      CLAHE clip limit for contrast normalization.
        sharpen_strength: Unsharp mask weight.

    Returns:
        Preprocessed BGR image at TARGET_SIZE × TARGET_SIZE.
    """
    # 1. Resize to standard resolution.
    resized = cv2.resize(bgr, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LANCZOS4)
    # 2. Grey-world white balance to cancel mobile color casts.
    wb = normalize_white_balance(resized)
    # 3. CLAHE on luminance to normalise brightness across lighting conditions.
    clahe_img = apply_clahe(wb, clip_limit=clip_limit)
    # 4. Unsharp mask to sharpen soft JPEG faces.
    sharpened = apply_unsharp_mask(clahe_img, strength=sharpen_strength)
    return sharpened


def detect_and_crop_face(
    bgr: np.ndarray,
    detector,
    min_confidence: float = 0.85,
) -> Optional[np.ndarray]:
    """Detect the largest face in the image and return its aligned crop.

    Using a slightly lower confidence threshold (0.85 vs runtime 0.92) to
    handle the wider pose/angle variance in mobile photos.

    Args:
        bgr:            Input image in BGR format.
        detector:       Pre-loaded MTCNN face detector.
        min_confidence: Minimum detector confidence.

    Returns:
        Cropped face as BGR uint8 array, or None if no face is found.
    """
    detections = detect_faces(bgr, detector, min_confidence=min_confidence, padding=0.15)
    if not detections:
        return None

    # Use largest detected bounding box as the primary face.
    largest = max(detections, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
    # detect_faces returns RGB; convert back to BGR for cv2's pipeline.
    face_bgr = cv2.cvtColor(largest.face_rgb, cv2.COLOR_RGB2BGR)
    return face_bgr


# ══════════════════════════════════════════════════════════════════════
#  Core processing logic
# ══════════════════════════════════════════════════════════════════════

def process_identity_folder(
    folder: Path,
    detector,
    output_root: Optional[Path],
    dry_run: bool,
    clip_limit: float,
    sharpen_strength: float,
    no_detect: bool,
) -> dict:
    """Process all images in one identity folder.

    Args:
        folder:           Path to identity sub-folder inside dataset/.
        detector:         MTCNN face detector instance.
        output_root:      Optional alternative output root.  If None, edit in-place.
        dry_run:          If True, only simulate — write nothing.
        clip_limit:       CLAHE clip limit.
        sharpen_strength: Unsharp mask weight.
        no_detect:        If True, skip face detection and just normalize lighting.

    Returns:
        Dictionary with statistics for this identity.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths: List[Path] = [
        p for p in sorted(folder.iterdir())
        if p.suffix.lower() in exts and p.is_file()
    ]

    stats = {
        "name": folder.name,
        "total": len(image_paths),
        "processed": 0,
        "no_face": 0,
        "errors": 0,
    }

    if not image_paths:
        print(f"  ⚠  No images found in '{folder.name}' — skipping.")
        return stats

    # Determine output folder.
    if output_root is not None:
        out_folder = output_root / folder.name
        if not dry_run:
            ensure_dir(out_folder)
    else:
        out_folder = folder  # in-place

    for img_path in image_paths:
        try:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                print(f"    ✗  Cannot read: {img_path.name}")
                stats["errors"] += 1
                continue

            original_bgr = bgr.copy()

            if not no_detect:
                # Attempt face detection + crop first.
                face_bgr = detect_and_crop_face(bgr, detector)
                if face_bgr is None:
                    # No face found — still normalize the raw image but warn.
                    print(f"    ⚠  No face detected in '{img_path.name}' — normalizing full frame.")
                    face_bgr = bgr
                    stats["no_face"] += 1

                processed = preprocess_image(face_bgr, clip_limit, sharpen_strength)
            else:
                # Skip detection — just preprocess the full image.
                processed = preprocess_image(bgr, clip_limit, sharpen_strength)

            out_path = out_folder / img_path.name
            if not dry_run:
                cv2.imwrite(str(out_path), processed, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(
                f"    {'[DRY RUN] ' if dry_run else ''}✓  {img_path.name} → "
                f"{out_path.relative_to(PROJECT_DIR)}"
            )
            stats["processed"] += 1

        except Exception as exc:
            print(f"    ✗  Error processing '{img_path.name}': {exc}")
            stats["errors"] += 1

    return stats


# ══════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Normalize mobile camera photos to reduce embedding domain gap.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        default=DATASET_DIR,
        help="Root dataset directory (default: dataset/)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Process only one identity (folder name).  Omit to process all.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Save preprocessed images to a separate folder.  "
             "Omit to overwrite in-place.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate without writing any files.",
    )
    parser.add_argument(
        "--clip-limit",
        type=float,
        default=DEFAULT_CLIP_LIMIT,
        help=f"CLAHE clip limit (default: {DEFAULT_CLIP_LIMIT}).  "
             f"Range 1.5–4.0 recommended.",
    )
    parser.add_argument(
        "--sharpen-strength",
        type=float,
        default=SHARPEN_STRENGTH,
        help=f"Unsharp mask weight (default: {SHARPEN_STRENGTH}).  "
             f"Increase for blur, decrease for sharp images.",
    )
    parser.add_argument(
        "--no-detect",
        action="store_true",
        help="Skip MTCNN face detection — just normalize lighting/sharpness.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for mobile photo preprocessing pipeline."""
    # Change CWD to project root so relative paths resolve correctly.
    import os
    os.chdir(str(PROJECT_DIR))

    args = parse_args()

    dataset_root = Path(args.dataset_dir)
    if not dataset_root.exists():
        print(f"✗  Dataset directory not found: {dataset_root}")
        sys.exit(1)

    output_root = Path(args.output_dir) if args.output_dir else None

    # Determine which identity folders to process.
    if args.name:
        target = dataset_root / args.name
        if not target.is_dir():
            print(f"✗  Identity folder not found: {target}")
            sys.exit(1)
        folders = [target]
    else:
        folders = get_identity_folders(dataset_root)
        if not folders:
            print("✗  No identity folders found in dataset/.")
            sys.exit(1)

    mode = "[DRY RUN] " if args.dry_run else ""
    dest = f"→ {args.output_dir}" if output_root else "(in-place)"
    print(
        f"\n{'=' * 64}\n"
        f"  {mode}Mobile Photo Normalizer\n"
        f"  Dataset : {dataset_root}\n"
        f"  Output  : {dest}\n"
        f"  Identities: {len(folders)}\n"
        f"  CLAHE clip limit : {args.clip_limit}\n"
        f"  Sharpen strength : {args.sharpen_strength}\n"
        f"  Face detection   : {'disabled' if args.no_detect else 'enabled'}\n"
        f"{'=' * 64}\n"
    )

    # Load face detector once and reuse across all identities.
    if not args.no_detect:
        print("⟳  Loading MTCNN face detector...", flush=True)
        detector = load_face_detector()
        print("✓  Detector ready.\n")
    else:
        detector = None

    total_processed = 0
    total_no_face = 0
    total_errors = 0

    for folder in folders:
        print(f"── Identity: {folder.name}")
        stats = process_identity_folder(
            folder=folder,
            detector=detector,
            output_root=output_root,
            dry_run=args.dry_run,
            clip_limit=args.clip_limit,
            sharpen_strength=args.sharpen_strength,
            no_detect=args.no_detect,
        )
        print(
            f"   Processed: {stats['processed']}/{stats['total']}  |  "
            f"No face: {stats['no_face']}  |  Errors: {stats['errors']}\n"
        )
        total_processed += stats["processed"]
        total_no_face += stats["no_face"]
        total_errors += stats["errors"]

    print("═" * 64)
    print(f"  Total images processed : {total_processed}")
    print(f"  Images without face    : {total_no_face}")
    print(f"  Errors                 : {total_errors}")
    if not args.dry_run:
        print(
            "\n  ✅ Done.  Now re-run generate_embeddings.py (or use\n"
            "     the 'Train Engine' step in the GUI) to rebuild\n"
            "     prototypes from the normalized images."
        )
    else:
        print("\n  ℹ  Dry run complete — no files were written.")
    print("═" * 64)


if __name__ == "__main__":
    main()
