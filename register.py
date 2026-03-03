"""Enrollment module for few-shot face recognition dataset creation.

Purpose:
    This module captures face samples from a live webcam stream and stores
    cropped face images under identity-specific folders.

Role in pipeline:
    It is the first stage of the end-to-end recognition workflow. The quality
    and diversity of captured images directly influence embedding quality and
    downstream recognition accuracy.

Few-shot contribution:
    Instead of requiring large datasets, this module collects a small number
    of high-quality samples (typically 2-5) per user, enabling practical
    few-shot enrollment for academic and real-world scenarios.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from utils import detect_faces, ensure_dir, load_face_detector


def register_person(
    person_name: str,
    num_images: int,
    dataset_dir: str = "dataset",
    camera_id: int = 0,
) -> None:
    """Capture and store face samples for one identity across multiple distances.

    Function name:
        register_person

    Purpose:
        Opens webcam stream, detects faces frame-by-frame, and saves cropped
        face images across three distance phases:

          Phase 1 — close range    : first 3 captures (strongly lit, large face)
          Phase 2 — medium range   : next  3 captures (normal sitting distance)
          Phase 3 — slightly far   : last  2 captures (2 m+ / standing back)

        This diversity of distances produces a more robust class prototype that
        generalises to subjects viewed at varying distances during recognition.

    Parameters:
        person_name (str): Identity label used as folder name in dataset.
        num_images (int): Total number of face samples to capture (default 8).
        dataset_dir (str): Root directory where identity folders are stored.
        camera_id (int): Index of webcam device for OpenCV capture.

    Returns:
        None

    Role in face recognition process:
        Implements the upgraded enrollment pipeline that captures samples at
        multiple distances so the resulting class prototype is robust to
        real-world distance variation during live recognition.
    """
    # Enrollment phase boundaries.
    # Phase 1 ends at image CLOSE_END (exclusive); Phase 2 ends at MED_END.
    # Any captures from MED_END onwards belong to Phase 3 (far range).
    CLOSE_END = 3          # images 1-3  → close range
    MED_END   = 6          # images 4-6  → medium range
    # images 7+ → slightly far range

    PHASE_INSTRUCTIONS = {
        1: "Stay close to camera (0.5-1 m)   [Phase 1/3 - Close]",
        2: "Step back slightly - medium distance (1-1.5 m)   [Phase 2/3 - Medium]",
        3: "Step back further for next samples (1.5-2.5 m)   [Phase 3/3 - Far]",
    }
    PHASE_COLORS = {
        1: (0, 200, 255),   # cyan   - close
        2: (0, 255, 100),   # green  - medium
        3: (255, 140, 0),   # orange - far
    }

    # Load face detector once to reuse it for all incoming video frames.
    detector = load_face_detector()

    # Create (or reuse) the identity folder where captured samples are stored.
    person_dir = Path(dataset_dir) / person_name
    ensure_dir(person_dir)

    import sys
    # Webcam initialization (CAP_DSHOW on Windows improves device handling).
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions/device id.")

    print("=" * 70)
    print(f"Registering identity: {person_name}")
    print(f"Target images       : {num_images}  (3 close + 3 medium + 2 far)")
    print("Controls            : Press 'q' to quit early")
    print("Info                : Keep only one face in front of webcam")
    print("=" * 70)
    print(PHASE_INSTRUCTIONS[1])
    print("=" * 70)

    # Runtime counters used to control sampling quality and spacing.
    captured = 0
    frame_count = 0
    min_frame_gap = 8

    try:
        while captured < num_images:
            success, frame = cap.read()
            if not success:
                continue

            frame_count += 1

            # Determine current enrollment phase based on images captured so far.
            if captured < CLOSE_END:
                phase = 1
            elif captured < MED_END:
                phase = 2
            else:
                phase = 3

            pcolor = PHASE_COLORS[phase]

            # Announce phase transition on first frame of each new phase
            # so the operator hears the prompt change.
            if frame_count == 1 or (
                captured == CLOSE_END and frame_count % min_frame_gap == 1
            ) or (
                captured == MED_END and frame_count % min_frame_gap == 1
            ):
                pass  # instructions are printed on capture below

            # Face detection stage: locate candidate faces in current frame.
            detections = detect_faces(frame, detector, min_confidence=0.85, padding=0.15)

            # Draw all detected face boxes so user sees tracking feedback.
            for det in detections:
                x1, y1, x2, y2 = det.box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)

            # Enrollment sampling strategy:
            # use the largest detected face for consistency across captures.
            if detections and frame_count % min_frame_gap == 0:
                largest = max(
                    detections,
                    key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]),
                )

                # Image preprocessing for storage: convert RGB crop to BGR
                # because OpenCV writes images in BGR format.
                face_bgr = cv2.cvtColor(largest.face_rgb, cv2.COLOR_RGB2BGR)
                save_name = f"{person_name}_{int(time.time() * 1000)}_{captured + 1}_p{phase}.jpg"
                save_path = person_dir / save_name
                cv2.imwrite(str(save_path), face_bgr)
                captured += 1

                print(
                    f"  [{captured}/{num_images}] Captured — {PHASE_INSTRUCTIONS[phase]}"
                )

                # Announce next phase when boundary is crossed.
                if captured == CLOSE_END and captured < num_images:
                    print("\n" + "=" * 70)
                    print(PHASE_INSTRUCTIONS[2])
                    print("=" * 70 + "\n")
                elif captured == MED_END and captured < num_images:
                    print("\n" + "=" * 70)
                    print(PHASE_INSTRUCTIONS[3])
                    print("=" * 70 + "\n")

                x1, y1, x2, y2 = largest.box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # --- Overlay: person name and capture counter ---
            cv2.putText(
                frame,
                f"Person: {person_name}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Captured: {captured}/{num_images}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # --- Overlay: phase instruction bar ---
            instruction = PHASE_INSTRUCTIONS[phase]
            cv2.putText(
                frame,
                instruction,
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                pcolor,
                2,
            )

            cv2.imshow("Registration - FewShotFace", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Registration stopped by user.")
                break

    finally:
        # Always release hardware resources, even on manual exit/error.
        cap.release()
        cv2.destroyAllWindows()

    print(f"Registration completed. Saved {captured} images in: {person_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for enrollment execution.

    Function name:
        parse_args

    Purpose:
        Defines and validates command-line inputs required to run the
        enrollment pipeline from terminal.

    Parameters:
        None

    Returns:
        argparse.Namespace: Parsed enrollment configuration values.

    Role in face recognition process:
        Provides reproducible and configurable control over data collection,
        which is critical for academic experimentation and evaluation.
    """
    parser = argparse.ArgumentParser(description="Register a new user with webcam face captures")
    parser.add_argument("--name", required=True, help="Person label/name (e.g. dhaval)")
    parser.add_argument(
        "--num-images",
        type=int,
        default=8,
        help="Number of face images to capture (recommended: 8 — 3 close + 3 medium + 2 far)",
    )
    parser.add_argument("--dataset-dir", default="dataset", help="Dataset root directory")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam device id")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    num_images = max(2, int(args.num_images))
    register_person(
        person_name=args.name.strip(),
        num_images=num_images,
        dataset_dir=args.dataset_dir,
        camera_id=args.camera_id,
    )
