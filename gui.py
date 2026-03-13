"""AI Face Recognition Monitoring System — PyQt5 Desktop GUI.

A professional 3-step workflow interface for the FewShotFace recognition
system. Neural Surveillance Terminal aesthetic — fully responsive layout.

Usage:
    python gui.py
"""

from __future__ import annotations

import io
import sys
import time
import os
import ssl
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

# ── Fix SSL CERTIFICATE_VERIFY_FAILED for model downloading ──────────
ssl._create_default_https_context = ssl._create_unverified_context

# ── Fix PyQt5/PyTorch DLL conflict on Windows ────────────────────────
if sys.platform.startswith("win"):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── project root on path ──────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import cv2
import numpy as np
import torch  # noqa: F401

from utils import (
    FaceEmbedder,
    FaceTracker,
    align_face_from_landmarks,
    build_augmented_prototypes,
    compute_class_prototypes,
    detect_faces,
    ensure_dir,
    format_size,
    get_dir_size,
    get_enhanced_crop,
    get_file_size,
    get_identity_folders,
    load_face_detector,
    load_saved_embeddings,
)
from similarity import predict_with_prototypes
from recognize import FrameVoter
from evaluate_accuracy import (
    evaluate as run_accuracy_evaluation,
    load_prototypes as load_accuracy_prototypes,
    print_comparison_table,
    print_per_identity_table,
    print_threshold_recommendation,
    _metrics_from_results,
    evaluate_backend,
    build_in_memory_prototypes,
    precompute_face_crops,
    COMPARISON_BACKENDS,
    _comparison_resolved_name,
)

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot, QSize
from PyQt5.QtGui import (
    QColor, QImage, QPixmap, QPainter, QPalette, QBrush, QFont
)
from PyQt5.QtWidgets import (
    QApplication, QFrame, QHBoxLayout,
    QComboBox, QDialog, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QSizePolicy, QSpinBox, QStackedWidget,
    QVBoxLayout, QWidget, QGraphicsDropShadowEffect, QSlider, QSplitter,
    QSpacerItem,
)

# ── Internal ML config ──────────────────────────────────────────────
SIMILARITY_METRIC = "cosine"
THRESHOLD         = 0.65   # 0.60-0.70 recommended for ArcFace; 0.65 is mid-range
VOTE_FRAMES       = 7
CAMERA_INDEX      = 0
EMBEDDING_BACKEND = "auto"
ONNX_MODEL_PATH   = "models/w600k_r50.onnx"
DATASET_DIR       = "dataset"
EMBEDDINGS_DIR    = "embeddings"


def _parse_embedder_backend(backend_name: str) -> dict:
    """Convert FaceEmbedder backend_name into stable metadata fields."""
    name = str(backend_name)
    meta = {
        "backend": "facenet",
        "deepface_model": None,
        "insightface_model": None,
        "onnx_model_path": None,
        "resolved_backend_name": name,
    }
    if name.startswith("deepface(") and name.endswith(")"):
        meta["backend"] = "deepface"
        meta["deepface_model"] = name[len("deepface("):-1]
    elif name.startswith("insightface(") and name.endswith(")"):
        meta["backend"] = "insightface"
        meta["insightface_model"] = name[len("insightface("):-1]
    elif name.startswith("onnx(") and name.endswith(")"):
        meta["backend"] = "onnx"
        meta["onnx_model_path"] = name[len("onnx("):-1]
    elif name.startswith("vit("):
        meta["backend"] = "vit"
    elif name == "facenet":
        meta["backend"] = "facenet"
    return meta

# ══════════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM — Neural Surveillance Terminal
# ══════════════════════════════════════════════════════════════════════

BG        = "#080C14"
BG2       = "#0C1220"
SURFACE   = "#0F1A2E"
SURFACE2  = "#142035"
PANEL     = "#0A1628"
BORDER    = "#1E3A5F"
BORDER2   = "#0D2545"

CYAN      = "#00D4FF"
CYAN_DIM  = "rgba(0,212,255,0.12)"

GREEN     = "#00FF88"
GREEN_DIM = "rgba(0,255,136,0.10)"

RED       = "#FF3366"
RED_DIM   = "rgba(255,51,102,0.12)"

AMBER     = "#FFAA00"
AMBER_DIM = "rgba(255,170,0,0.12)"

TEXT      = "#D0E4F7"
TEXT_DIM  = "#4A7FA5"
TEXT_MID  = "#7BA8CC"

MONO_FONT = "Consolas"
SANS_FONT = "Segoe UI"

# ══════════════════════════════════════════════════════════════════════
#  Stylesheet
# ══════════════════════════════════════════════════════════════════════
STYLESHEET = f"""
QMainWindow, QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: '{SANS_FONT}';
}}
QLabel {{
    background: transparent;
    color: {TEXT};
    border: none;
}}
QLineEdit {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 7px 12px;
    color: {TEXT};
    font-size: 12px;
    font-family: '{SANS_FONT}';
    selection-background-color: {CYAN};
    min-height: 18px;
}}
QLineEdit:focus {{
    border: 1px solid {CYAN};
    background: {SURFACE2};
}}
QLineEdit::placeholder {{
    color: {TEXT_MID};
    font-style: italic;
}}
QSpinBox {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    color: {TEXT};
    font-size: 12px;
    min-height: 18px;
}}
QSpinBox:focus {{
    border: 1px solid {CYAN};
}}
QSpinBox::up-button, QSpinBox::down-button {{
    background: {SURFACE2};
    border: none;
    width: 14px;
}}
QProgressBar {{
    background: {SURFACE};
    border: none;
    border-radius: 3px;
    text-align: center;
    color: {TEXT};
    font-size: 10px;
    font-family: '{MONO_FONT}';
    height: 5px;
    max-height: 5px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {CYAN}, stop:1 #0099CC);
    border-radius: 3px;
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollBar:vertical {{
    background: {BG2};
    width: 4px;
    border-radius: 2px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 2px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0;
    border: none;
}}
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {{
    background: none;
}}
QScrollBar:horizontal {{
    height: 0;
    background: transparent;
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {SURFACE2};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {CYAN};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {TEXT};
    width: 12px;
    height: 12px;
    margin: -4px 0;
    border-radius: 6px;
    border: 2px solid {CYAN};
}}
QSlider::handle:horizontal:hover {{
    background: {CYAN};
}}
QComboBox {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    color: {TEXT};
    font-size: 11px;
    font-family: '{MONO_FONT}';
    min-height: 18px;
}}
QComboBox:focus {{
    border: 1px solid {CYAN};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_MID};
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background: {SURFACE2};
    color: {TEXT};
    border: 1px solid {BORDER};
    selection-background-color: {CYAN};
    selection-color: {BG};
    font-family: '{MONO_FONT}';
    font-size: 11px;
    padding: 4px;
    outline: none;
}}
QSplitter::handle {{
    background: {BORDER2};
    width: 3px;
    border-radius: 2px;
}}
QSplitter::handle:hover {{
    background: {CYAN};
}}
"""


# ══════════════════════════════════════════════════════════════════════
#  Helper Widgets / Factories
# ══════════════════════════════════════════════════════════════════════

def _shadow(widget: QWidget, blur: int = 16):
    e = QGraphicsDropShadowEffect()
    e.setBlurRadius(blur)
    c = QColor("#000000")
    c.setAlpha(120)
    e.setColor(c)
    e.setOffset(0, 2)
    widget.setGraphicsEffect(e)


def _btn(label: str, color: str = CYAN, parent=None) -> QPushButton:
    btn = QPushButton(label, parent)
    btn.setCursor(getattr(Qt, "PointingHandCursor"))
    btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    btn.setMinimumHeight(36)
    if color == CYAN:
        bg   = "qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #006680,stop:1 #004D66)"
        bg_h = "qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0099CC,stop:1 #007AB8)"
        text_col = border_col = CYAN
    elif color == GREEN:
        bg   = "rgba(0,80,50,0.6)"
        bg_h = "rgba(0,120,70,0.8)"
        text_col = border_col = GREEN
    else:
        bg   = "rgba(100,20,40,0.6)"
        bg_h = "rgba(150,30,60,0.8)"
        text_col = border_col = RED

    btn.setStyleSheet(f"""
        QPushButton {{
            background: {bg};
            color: {text_col};
            border: 1px solid {border_col};
            border-radius: 5px;
            padding: 8px 16px;
            font-size: 12px;
            font-weight: 700;
            font-family: '{SANS_FONT}';
            letter-spacing: 0.5px;
        }}
        QPushButton:hover {{
            background: {bg_h};
            border-color: {text_col};
        }}
        QPushButton:disabled {{
            background: rgba(20,30,50,0.5);
            color: {TEXT_DIM};
            border-color: {BORDER2};
        }}
    """)
    return btn


def _section_title(text: str, color: str = CYAN) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(f"""
        color: {color};
        font-size: 9px;
        font-family: '{SANS_FONT}';
        font-weight: 700;
        letter-spacing: 2px;
        background: transparent;
        border: none;
    """)
    return lbl


def _mono_label(text: str, size: int = 11, color: str = TEXT) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"""
        color: {color};
        font-family: '{MONO_FONT}';
        font-size: {size}px;
        background: transparent;
        border: none;
    """)
    return lbl


def _field_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"""
        color: {TEXT_MID};
        font-size: 10px;
        font-family: '{SANS_FONT}';
        font-weight: 600;
        letter-spacing: 0.5px;
        background: transparent;
        border: none;
    """)
    return lbl


def _divider() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setFrameShadow(QFrame.Plain)
    f.setFixedHeight(1)
    f.setStyleSheet(f"background: {BORDER2}; border: none; margin: 1px 0;")
    return f


def _badge(text: str, color: str, dim: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"""
        color: {color};
        background: {dim};
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 3px;
        font-size: 8px;
        font-weight: 700;
        font-family: '{MONO_FONT}';
        padding: 2px 6px;
        letter-spacing: 1px;
    """)
    return lbl


# ══════════════════════════════════════════════════════════════════════
#  Scan Line Overlay
# ══════════════════════════════════════════════════════════════════════

class ScanLineWidget(QWidget):
    """Animated horizontal scan line for cyberpunk visual effect."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(getattr(Qt, "WA_TransparentForMouseEvents"))
        self.setAttribute(getattr(Qt, "WA_TranslucentBackground"))
        self._y = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)

    def _tick(self):
        self._y = (self._y + 2) % max(1, self.height())
        self.update()

    def paintEvent(self, a0):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(getattr(Qt, "NoPen"))
        p.setBrush(QBrush(QColor(0, 212, 255, 16)))
        p.drawRect(0, self._y, self.width(), 3)


# ══════════════════════════════════════════════════════════════════════
#  Indicator Pill Widget
# ══════════════════════════════════════════════════════════════════════

class IndicatorPill(QFrame):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setFixedHeight(24)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 0, 12, 0)
        lay.setSpacing(5)

        self.dot = QLabel("◆")
        self.txt = QLabel(label)
        self.dot.setFixedWidth(8)
        lay.addWidget(self.dot)
        lay.addWidget(self.txt)
        self.setActive(False)

    def setActive(self, active: bool):
        if active:
            self.dot.setStyleSheet(
                f"color: {GREEN}; font-size: 6px; background: transparent; border: none;")
            self.txt.setStyleSheet(
                f"color: {GREEN}; font-size: 9px; font-family: '{MONO_FONT}';"
                f" font-weight: 700; background: transparent; border: none;")
            self.setStyleSheet(
                f"background: {GREEN_DIM}; border: 1px solid rgba(0,255,136,0.18);"
                f" border-radius: 12px;")
        else:
            self.dot.setStyleSheet(
                f"color: {TEXT_DIM}; font-size: 6px; background: transparent; border: none;")
            self.txt.setStyleSheet(
                f"color: {TEXT_DIM}; font-size: 9px; font-family: '{MONO_FONT}';"
                f" background: transparent; border: none;")
            self.setStyleSheet(
                f"background: {SURFACE}; border: 1px solid {BORDER2}; border-radius: 12px;")


# ══════════════════════════════════════════════════════════════════════
#  Worker Threads  (logic completely unchanged)
# ══════════════════════════════════════════════════════════════════════

class EnrollWorker(QThread):
    frame_ready     = pyqtSignal(np.ndarray)
    face_captured   = pyqtSignal(np.ndarray, int)
    finished_signal = pyqtSignal(str)
    error           = pyqtSignal(str)

    def __init__(self, name: str, num_images: int):
        super().__init__()
        self.name       = name
        self.num_images = num_images
        self._running   = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            started_at = time.perf_counter()
            detector   = load_face_detector()
            person_dir = Path(DATASET_DIR) / self.name
            ensure_dir(person_dir)
            cap = (cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
                   if sys.platform.startswith("win")
                   else cv2.VideoCapture(CAMERA_INDEX))
            if not cap.isOpened():
                self.error.emit("Could not open webcam. Check camera permissions.")
                return

            captured, frame_count, min_gap = 0, 0, 6
            last_img_size = format_size(0)
            HINTS = {
                1:  "Look straight at camera",
                2:  "Tilt head slightly LEFT",
                3:  "Tilt head slightly RIGHT",
                4:  "Look straight, smile slightly",
                5:  "Look straight (medium distance ~1.5 m)",
                6:  "Turn chin slightly UP",
                7:  "Turn chin slightly DOWN",
                8:  "Look straight (step back ~2 m)",
                9:  "Tilt head slightly LEFT (far)",
                10: "Look straight, normal expression",
                11: "Tilt head slightly RIGHT (far)",
                12: "Step back ~3 m, look straight",
                13: "Near distance, look straight",
                14: "Side lighting, look straight",
                15: "Look straight, neutral expression (final)",
            }
            BLUR_THRESHOLD = 60.0

            while self._running and captured < self.num_images:
                ok, frame = cap.read()
                if not ok:
                    continue
                frame_count += 1
                detections = detect_faces(frame, detector, min_confidence=0.90, padding=0.12)
                for det in detections:
                    x1, y1, x2, y2 = det.box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 212, 255), 2)
                if detections and frame_count % min_gap == 0:
                    largest = max(detections,
                                  key=lambda d: (d.box[2]-d.box[0])*(d.box[3]-d.box[1]))
                    gray      = cv2.cvtColor(largest.face_rgb, cv2.COLOR_RGB2GRAY)
                    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    if sharpness < BLUR_THRESHOLD:
                        cv2.putText(frame,
                                    f"Too blurry — stay still  (sharp={sharpness:.0f})",
                                    (10, frame.shape[0] - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 80, 255), 2)
                        self.frame_ready.emit(frame)
                        cv2.waitKey(1)
                        continue
                    face_bgr = cv2.cvtColor(largest.face_rgb, cv2.COLOR_RGB2BGR)
                    fname    = f"{self.name}_{int(time.time()*1000)}_{captured+1}.jpg"
                    saved_path = person_dir / fname
                    cv2.imwrite(str(saved_path), face_bgr)
                    last_img_size = format_size(get_file_size(saved_path))
                    captured += 1
                    self.face_captured.emit(largest.face_rgb, captured)

                cv2.putText(frame, f"CAPTURED: {captured}/{self.num_images}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 255), 2)
                next_hint = HINTS.get(captured + 1, "Look naturally")
                cv2.putText(frame, f"Next: {next_hint}",
                            (10, frame.shape[0] - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 80), 1)
                self.frame_ready.emit(frame)
                cv2.waitKey(1)

            cap.release()
            elapsed = max(0.0, time.perf_counter() - started_at)
            folder_size = format_size(get_dir_size(person_dir))
            self.finished_signal.emit(
                f"Successfully enrolled {captured} face samples in {elapsed:.2f}s "
                f"(last: {last_img_size}, total: {folder_size})."
            )
        except Exception as e:
            self.error.emit(str(e))


class TrainWorker(QThread):
    progress        = pyqtSignal(int)
    finished_signal = pyqtSignal(str)
    error           = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.backend: str = EMBEDDING_BACKEND
        self.deepface_model: str = "ArcFace"

    def run(self):
        try:
            started_at = time.perf_counter()
            backend  = getattr(self, "backend", EMBEDDING_BACKEND)
            df_model = getattr(self, "deepface_model", "ArcFace")
            ensure_dir(EMBEDDINGS_DIR)
            folders = get_identity_folders(DATASET_DIR)
            if not folders:
                self.error.emit("No identity folders found. Enroll users first.")
                return
            self.progress.emit(10)
            detector = load_face_detector()
            embedder = FaceEmbedder(backend=backend,
                                    onnx_model_path=ONNX_MODEL_PATH,
                                    deepface_model=df_model)
            resolved = _parse_embedder_backend(embedder.backend_name)
            if backend != "auto" and resolved["backend"] != backend:
                self.error.emit(
                    f"Requested backend '{backend}' but runtime resolved to "
                    f"'{embedder.backend_name}'. Install dependencies/model files for {backend} "
                    "or switch to an available backend."
                )
                return
            self.progress.emit(25)
            all_embeddings, all_labels = [], []
            for i, folder in enumerate(folders):
                imgs = (list(folder.glob("*.jpg")) +
                        list(folder.glob("*.png")) +
                        list(folder.glob("*.jpeg")))
                for img_path in imgs:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    dets = detect_faces(img, detector, min_confidence=0.80, padding=0.10)
                    if not dets:
                        continue
                    largest = max(dets,
                                  key=lambda d: (d.box[2]-d.box[0])*(d.box[3]-d.box[1]))
                    # Apply face alignment for +3-5% accuracy on FaceNet/ONNX backends.
                    if largest.landmarks_5pt is not None:
                        aligned = align_face_from_landmarks(img, largest.landmarks_5pt)
                        face_input = aligned if aligned is not None else largest.face_rgb
                    else:
                        face_input = largest.face_rgb
                    emb = embedder.embed_face(face_input)
                    all_embeddings.append(emb)
                    all_labels.append(folder.name)
                self.progress.emit(25 + int(65 * (i + 1) / len(folders)))

            if not all_embeddings:
                self.error.emit("No usable face images found.")
                return

            emb_arr = np.vstack(all_embeddings).astype(np.float32)
            lbl_arr = np.array(all_labels, dtype=str)

            norms   = np.linalg.norm(emb_arr, axis=1, keepdims=True)
            emb_arr = emb_arr / (norms + 1e-8)

            import json as _json
            metadata = {
                "backend": resolved["backend"],
                "deepface_model": resolved["deepface_model"],
                "insightface_model": resolved["insightface_model"],
                "onnx_model_path": resolved["onnx_model_path"],
                "resolved_backend_name": resolved["resolved_backend_name"],
                "requested_backend": backend,
                "requested_deepface_model": df_model if backend == "deepface" else None,
            }
            with open(Path(EMBEDDINGS_DIR) / "backend.json", "w") as _fh:
                _json.dump(metadata, _fh)

            np.save(Path(EMBEDDINGS_DIR) / "embeddings.npy", emb_arr)
            np.save(Path(EMBEDDINGS_DIR) / "labels.npy",     lbl_arr)

            protos, names = build_augmented_prototypes(emb_arr, lbl_arr)
            np.save(Path(EMBEDDINGS_DIR) / "prototypes.npy",  protos)
            np.save(Path(EMBEDDINGS_DIR) / "class_names.npy", names)

            elapsed = time.perf_counter() - started_at
            artifacts_size = (
                get_file_size(Path(EMBEDDINGS_DIR) / "embeddings.npy")
                + get_file_size(Path(EMBEDDINGS_DIR) / "labels.npy")
                + get_file_size(Path(EMBEDDINGS_DIR) / "prototypes.npy")
                + get_file_size(Path(EMBEDDINGS_DIR) / "class_names.npy")
            )

            self.progress.emit(100)
            self.finished_signal.emit(
                f"Trained on {len(emb_arr)} samples | {len(names)} identities | "
                f"Time: {elapsed:.2f}s | Size: {format_size(artifacts_size)}")
        except Exception as e:
            self.error.emit(str(e))


class AccuracyWorker(QThread):
    finished_signal = pyqtSignal(str)
    error           = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.threshold: float = THRESHOLD
        self.metric: str = SIMILARITY_METRIC
        self.backend: str = EMBEDDING_BACKEND
        self.deepface_model: str = "ArcFace"

    def run(self):
        try:
            import json as _json

            backend = getattr(self, "backend", EMBEDDING_BACKEND)
            df_model = getattr(self, "deepface_model", "ArcFace")
            if_model = None
            meta_path = Path(EMBEDDINGS_DIR) / "backend.json"
            if meta_path.exists():
                try:
                    with open(meta_path) as _fh:
                        meta = _json.load(_fh)
                    trained_backend = str(meta.get("backend", backend))
                    trained_df = str(meta.get("deepface_model") or df_model)
                    trained_if = meta.get("insightface_model")
                    if str(backend) == "auto":
                        backend = trained_backend
                        if trained_backend == "deepface":
                            df_model = trained_df
                        elif trained_backend == "insightface":
                            if_model = trained_if
                    elif (trained_backend != str(backend) or
                            (trained_backend == "deepface" and trained_df != str(df_model))):
                        self.error.emit(
                            f"Model mismatch! Trained with {trained_backend.upper()}, "
                            f"but active model is {str(backend).upper()}.\n"
                            "Please run Step 2 (BUILD EMBEDDINGS) to re-train."
                        )
                        return
                except Exception:
                    pass

            prototypes, class_names = load_accuracy_prototypes(Path(EMBEDDINGS_DIR))
            detector = load_face_detector()
            embedder = FaceEmbedder(
                backend=backend,
                onnx_model_path=ONNX_MODEL_PATH,
                deepface_model=df_model,
                insightface_model=if_model or "buffalo_l",
            )
            resolved = _parse_embedder_backend(embedder.backend_name)
            if backend != "auto" and resolved["backend"] != backend:
                self.error.emit(
                    f"Requested backend '{backend}' but runtime resolved to '{embedder.backend_name}'. "
                    "This usually means missing dependencies or model files on this machine."
                )
                return

            # Pre-detect all face crops once — shared across every backend.
            face_cache = precompute_face_crops(Path(DATASET_DIR), detector)

            t0 = time.perf_counter()
            results, total, no_face = run_accuracy_evaluation(
                dataset_dir=Path(DATASET_DIR),
                prototypes=prototypes,
                class_names=class_names,
                detector=detector,
                embedder=embedder,
                threshold=getattr(self, "threshold", THRESHOLD),
                metric=getattr(self, "metric", SIMILARITY_METRIC),
                quiet=True,
                face_cache=face_cache,
            )
            elapsed = time.perf_counter() - t0

            report = io.StringIO()
            with redirect_stdout(report):
                print("FewShotFace — Accuracy Evaluation")
                print("=" * 72)
                print(f"Threshold        : {self.threshold:.2f}")
                print(f"Metric           : {self.metric}")
                print(f"Backend          : {embedder.backend_name}")
                print(f"Total images     : {total}")
                print(f"No face detected : {no_face}")
                print(f"Tested images    : {total - no_face}")
                print(f"Evaluation time  : {elapsed:.2f}s")
                print()
                print("Per-Identity Metrics")
                print_per_identity_table(results)
                print("\nEvaluating all comparison backends (face crops pre-cached)...")
                all_model_metrics = {embedder.backend_name: _metrics_from_results(results)}
                for lbl, bk, df_m, if_m in COMPARISON_BACKENDS:
                    if _comparison_resolved_name(bk, df_m, if_m) == embedder.backend_name:
                        continue
                    m = evaluate_backend(
                        label=lbl, backend=bk,
                        deepface_model=df_m, insightface_model=if_m,
                        dataset_dir=Path(DATASET_DIR), detector=detector,
                        threshold=self.threshold, metric=self.metric,
                        quiet=True, face_cache=face_cache,
                    )
                    if m is not None:
                        all_model_metrics[lbl] = m
                print("\nUnified Model Comparison")
                print_comparison_table(all_model_metrics, embedder.backend_name)
                print_threshold_recommendation(results, self.threshold)

            self.finished_signal.emit(report.getvalue().strip())
        except Exception as e:
            self.error.emit(str(e))


class RecognitionWorker(QThread):
    frame_ready     = pyqtSignal(np.ndarray)
    person_detected = pyqtSignal(str, float, str)
    error           = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._running = True
        self.backend: str = EMBEDDING_BACKEND
        self.deepface_model: str = "ArcFace"
        self.live_threshold: float = THRESHOLD
        self.live_vote_frames: int = VOTE_FRAMES

    def stop(self):
        self._running = False

    def run(self):
        try:
            import json as _json

            detector   = load_face_detector()
            backend    = getattr(self, "backend", EMBEDDING_BACKEND)
            df_model   = getattr(self, "deepface_model", "ArcFace")
            if_model   = None

            meta_path = Path(EMBEDDINGS_DIR) / "backend.json"
            if meta_path.exists():
                try:
                    with open(meta_path) as _fh:
                        meta = _json.load(_fh)
                    trained_backend = str(meta.get("backend") or "").strip().lower()
                    trained_df = str(meta.get("deepface_model") or "").strip()
                    trained_if = str(meta.get("insightface_model") or "").strip()

                    if backend == "auto" and trained_backend:
                        backend = trained_backend
                        if trained_backend == "deepface" and trained_df:
                            df_model = trained_df
                        if trained_backend == "insightface" and trained_if:
                            if_model = trained_if
                    elif (
                        str(trained_backend) != str(backend) or
                        (backend == "deepface" and str(trained_df) != str(df_model))
                    ):
                        self.error.emit(
                            f"Model mismatch! Trained with {str(trained_backend).upper()}, "
                            f"but active model is {str(backend).upper()}.\n"
                            "Please run Step 2 (BUILD EMBEDDINGS) to re-train."
                        )
                        return
                except Exception:
                    pass

            embedder   = FaceEmbedder(backend=backend,
                                      onnx_model_path=ONNX_MODEL_PATH,
                                      deepface_model=df_model,
                                      insightface_model=if_model or "buffalo_l")
            resolved = _parse_embedder_backend(embedder.backend_name)
            if backend != "auto" and resolved["backend"] != backend:
                self.error.emit(
                    f"Requested backend '{backend}' but runtime resolved to '{embedder.backend_name}'. "
                    "Please install missing dependencies/model files and retry."
                )
                return
            proto_path = Path(EMBEDDINGS_DIR) / "prototypes.npy"
            names_path = Path(EMBEDDINGS_DIR) / "class_names.npy"
            if proto_path.exists() and names_path.exists():
                prototypes  = np.load(proto_path).astype(np.float32)
                class_names = np.load(names_path).astype(str)
            else:
                emb, lbl = load_saved_embeddings(
                    Path(EMBEDDINGS_DIR) / "embeddings.npy",
                    Path(EMBEDDINGS_DIR) / "labels.npy")
                prototypes, class_names = compute_class_prototypes(emb, lbl)
            if len(prototypes) == 0:
                self.error.emit("No trained data. Run Step 2 first.")
                return

            base_threshold = getattr(self, "live_threshold", THRESHOLD)

            vf      = getattr(self, "live_vote_frames", VOTE_FRAMES)
            voter   = FrameVoter(window=vf, max_missed_frames=12)
            tracker = FaceTracker(alpha=0.7, max_missed_frames=10)
            cap     = (cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
                       if sys.platform.startswith("win")
                       else cv2.VideoCapture(CAMERA_INDEX))
            if not cap.isOpened():
                self.error.emit("Could not open webcam.")
                return

            while self._running:
                ok, frame = cap.read()
                if not ok:
                    continue
                detections  = detect_faces(frame, detector,
                                           min_confidence=0.88, padding=0.12)
                active_boxes = [d.box for d in detections]
                voter.expire(active_boxes)
                tracker.expire_tracks(active_boxes)

                frame_width = frame.shape[1]

                for det in detections:
                    x1, y1, x2, y2 = det.box
                    face_w = x2 - x1

                    if face_w < 120:
                        face_crop = get_enhanced_crop(
                            frame, det.box, margin=0.25, target_size=160)
                    else:
                        # Apply affine alignment to canonical ArcFace eye positions.
                        if det.landmarks_5pt is not None:
                            aligned = align_face_from_landmarks(frame, det.landmarks_5pt)
                            face_crop = aligned if aligned is not None else det.face_rgb
                        else:
                            face_crop = det.face_rgb

                    raw_embedding = embedder.embed_face(face_crop)
                    embedding     = tracker.update(det.box, raw_embedding)

                    quality       = face_w / max(1, frame_width)
                    effective_thr = min(0.90, max(0.40,
                        base_threshold * (0.8 + 0.4 * quality)))

                    raw_result = predict_with_prototypes(
                        query_embedding=embedding,
                        prototypes=prototypes,
                        class_names=class_names,
                        metric=SIMILARITY_METRIC,
                        threshold=effective_thr)

                    raw_conf = raw_result.get("confidence", 0.0)
                    conf_value: Any = raw_conf if isinstance(raw_conf, (int, float, str)) else 0.0
                    voted_name, voted_conf = voter.vote(
                        box=det.box,
                        name=str(raw_result["name"]),
                        confidence=float(conf_value))

                    name   = voted_name
                    conf   = voted_conf
                    color  = (0, 255, 136) if name != "Unknown" else (51, 51, 255)
                    status = "AUTHORIZED"   if name != "Unknown" else "UNKNOWN"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, max(0, y1 - 50)), (x2, y1),
                                  (8, 12, 20), -1)
                    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
                    cv2.putText(frame, name, (x1 + 6, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 235, 250), 2)
                    cv2.putText(frame, f"{conf*100:.2f}% | {status}",
                                (x1 + 6, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)
                    self.person_detected.emit(
                        name, conf, datetime.now().strftime("%H:%M:%S"))

                self.frame_ready.emit(frame)
                cv2.waitKey(1)
            cap.release()
        except Exception as e:
            self.error.emit(str(e))


# ══════════════════════════════════════════════════════════════════════
#  Main Window
# ══════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Face Recognition Monitoring System")
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(STYLESHEET)

        self.live_threshold        = THRESHOLD
        self.live_vote_frames      = VOTE_FRAMES
        self.active_backend        = EMBEDDING_BACKEND
        self.active_deepface_model = "ArcFace"

        self.enroll_worker: Optional[EnrollWorker]      = None
        self.train_worker:  Optional[TrainWorker]       = None
        self.accuracy_worker: Optional[AccuracyWorker]  = None
        self.recog_worker:  Optional[RecognitionWorker] = None
        self._last_auth_state: Optional[str] = None

        self._build_ui()
        self._refresh_status()
        self._refresh_users()

    # ── UI Construction ───────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())

        # ── Body uses a QSplitter for drag-to-resize panes ────────────
        splitter = QSplitter(getattr(Qt, "Horizontal"))
        splitter.setHandleWidth(3)
        splitter.setChildrenCollapsible(False)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background: {BORDER2};
                border-radius: 1px;
            }}
            QSplitter::handle:hover {{
                background: {CYAN};
            }}
        """)

        # ── LEFT PANE — workflow steps ─────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(getattr(Qt, "ScrollBarAlwaysOff"))
        left_scroll.setMinimumWidth(300)

        left_w = QWidget()
        left_w.setStyleSheet("background: transparent;")
        left_lay = QVBoxLayout(left_w)
        left_lay.setContentsMargins(12, 12, 8, 12)
        left_lay.setSpacing(10)
        left_lay.addWidget(self._build_step1())
        left_lay.addWidget(self._build_step2())
        left_lay.addWidget(self._build_step3())
        left_lay.addStretch()
        left_scroll.setWidget(left_w)

        # ── CENTER PANE — video feed ───────────────────────────────────
        center_w = self._build_video_panel()
        center_w.setMinimumWidth(380)

        # ── RIGHT PANE — identities + detection ───────────────────────
        right_w = QWidget()
        right_w.setStyleSheet("background: transparent;")
        right_w.setMinimumWidth(240)

        right_lay = QVBoxLayout(right_w)
        right_lay.setContentsMargins(8, 12, 12, 12)
        right_lay.setSpacing(10)
        right_lay.addWidget(self._build_users_panel(), stretch=1)
        right_lay.addWidget(self._build_status_panel())

        splitter.addWidget(left_scroll)
        splitter.addWidget(center_w)
        splitter.addWidget(right_w)

        # Default proportions: 28% | 44% | 28%
        splitter.setStretchFactor(0, 28)
        splitter.setStretchFactor(1, 44)
        splitter.setStretchFactor(2, 28)

        root.addWidget(splitter, stretch=1)

    # ── Header ────────────────────────────────────────────────────────

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setFrameShape(QFrame.NoFrame)
        header.setFixedHeight(56)
        header.setStyleSheet(f"""
            background: {BG2};
            border-bottom: 1px solid {BORDER};
        """)

        lay = QHBoxLayout(header)
        lay.setContentsMargins(16, 0, 16, 0)
        lay.setSpacing(0)

        # ── Logo block ─────────────────────────────────────────────────
        logo_w = QWidget()
        logo_w.setStyleSheet("background: transparent; border: none;")
        logo_lay = QHBoxLayout(logo_w)
        logo_lay.setContentsMargins(0, 0, 0, 0)
        logo_lay.setSpacing(8)

        for sym, sz in [("[", 20), ("◈", 16), ("]", 20)]:
            lbl = QLabel(sym)
            lbl.setStyleSheet(f"""
                color: {CYAN}; font-size: {sz}px; font-weight: 900;
                font-family: '{MONO_FONT}'; background: transparent; border: none;
            """)
            logo_lay.addWidget(lbl)

        logo_lay.addSpacing(6)

        title_w = QWidget()
        title_w.setStyleSheet("background: transparent; border: none;")
        tv = QVBoxLayout(title_w)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.setSpacing(1)

        t1 = QLabel("AI FACE RECOGNITION")
        t1.setStyleSheet(f"""
            color: {TEXT}; font-size: 13px; font-weight: 700;
            font-family: '{MONO_FONT}'; letter-spacing: 2px;
            background: transparent; border: none;
        """)
        t2 = QLabel("MONITORING SYSTEM  v2.0")
        t2.setStyleSheet(f"""
            color: {TEXT_DIM}; font-size: 9px; font-family: '{MONO_FONT}';
            letter-spacing: 1px; background: transparent; border: none;
        """)
        tv.addWidget(t1)
        tv.addWidget(t2)
        logo_lay.addWidget(title_w)
        lay.addWidget(logo_w)
        lay.addStretch()

        # ── Clock ──────────────────────────────────────────────────────
        self.clock_lbl = QLabel()
        self.clock_lbl.setStyleSheet(f"""
            color: {CYAN}; font-size: 11px; font-family: '{MONO_FONT}';
            background: transparent; border: none;
        """)
        self._update_clock()
        ct = QTimer(self)
        ct.timeout.connect(self._update_clock)
        ct.start(1000)
        lay.addWidget(self.clock_lbl)
        lay.addSpacing(20)

        # ── Active model label ────────────────────────────────────────
        self.active_model_lbl = QLabel("AUTO")
        self.active_model_lbl.setStyleSheet(f"""
            color: {AMBER}; font-size: 9px; font-family: '{MONO_FONT}';
            font-weight: 700; letter-spacing: 1px;
            background: transparent; border: none;
        """)
        lay.addWidget(self.active_model_lbl)
        lay.addSpacing(10)

        # ── Indicator pills ───────────────────────────────────────────
        self.ind_model      = IndicatorPill("MODEL")
        self.ind_camera     = IndicatorPill("CAMERA")
        self.ind_embeddings = IndicatorPill("EMBEDDINGS")
        self.ind_system     = IndicatorPill("SYSTEM")
        for pill in [self.ind_model, self.ind_camera,
                     self.ind_embeddings, self.ind_system]:
            lay.addWidget(pill)
            lay.addSpacing(5)

        return header

    def _update_clock(self):
        self.clock_lbl.setText(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))

    def _set_indicator(self, widget: IndicatorPill, active: bool):
        widget.setActive(active)

    # ── Step Card Factory ─────────────────────────────────────────────

    def _step_card(self, num: str, title: str,
                   subtitle: str = "") -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setFrameShape(QFrame.NoFrame)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        card.setStyleSheet(f"""
            QFrame {{
                background: {PANEL};
                border: none;
                border-radius: 6px;
            }}
        """)
        _shadow(card, 12)

        v = QVBoxLayout(card)
        v.setContentsMargins(14, 12, 14, 14)
        v.setSpacing(10)

        # Header row
        hdr = QHBoxLayout()
        hdr.setSpacing(8)
        hdr.setContentsMargins(0, 0, 0, 0)

        step_tag = QLabel(f"STEP {num}")
        step_tag.setStyleSheet(f"""
            color: {CYAN}; background: {CYAN_DIM}; border: none;
            border-radius: 3px; font-size: 8px; font-weight: 700;
            font-family: '{MONO_FONT}'; letter-spacing: 1px; padding: 2px 6px;
        """)
        step_tag.setFixedHeight(18)
        step_tag.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        title_w = QWidget()
        title_w.setStyleSheet("background: transparent; border: none;")
        tb = QVBoxLayout(title_w)
        tb.setSpacing(1)
        tb.setContentsMargins(0, 0, 0, 0)

        t_lbl = QLabel(title)
        t_lbl.setStyleSheet(f"""
            color: {TEXT}; font-size: 13px; font-weight: 700;
            font-family: '{SANS_FONT}'; background: transparent; border: none;
        """)
        tb.addWidget(t_lbl)

        if subtitle:
            s_lbl = QLabel(subtitle)
            s_lbl.setStyleSheet(f"""
                color: {TEXT_DIM}; font-size: 9px;
                font-family: '{SANS_FONT}'; background: transparent; border: none;
            """)
            tb.addWidget(s_lbl)

        hdr.addWidget(step_tag)
        hdr.addWidget(title_w, stretch=1)
        v.addLayout(hdr)
        v.addWidget(_divider())

        return card, v

    # ── Step 1 — Enroll ───────────────────────────────────────────────

    def _build_step1(self) -> QFrame:
        card, v = self._step_card(
            "01", "Enroll New Identity",
            "Capture face samples for a person")

        # Name field
        name_row = QHBoxLayout()
        name_row.addWidget(_field_label("SUBJECT NAME"))
        name_row.addStretch()
        req = QLabel("● REQUIRED")
        req.setStyleSheet(f"""
            color: {CYAN}; font-size: 8px; font-family: '{MONO_FONT}';
            font-weight: 700; letter-spacing: 1px;
            background: transparent; border: none;
        """)
        name_row.addWidget(req)
        v.addLayout(name_row)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g.  Dhaval")
        self.name_input.setToolTip("Enter the full name of the person to enroll")
        self.name_input.setStyleSheet(f"""
            QLineEdit {{
                background: {SURFACE2};
                border: 1.5px solid {CYAN};
                border-radius: 6px;
                padding: 7px 12px;
                color: {TEXT};
                font-size: 12px;
                font-family: '{SANS_FONT}';
                min-height: 18px;
            }}
            QLineEdit:focus {{
                border: 1.5px solid {CYAN};
                background: {SURFACE2};
            }}
            QLineEdit::placeholder {{
                color: {TEXT_DIM};
                font-style: italic;
            }}
        """)
        v.addWidget(self.name_input)

        hint = QLabel("Enter name, then click  CAPTURE FACE  to begin")
        hint.setStyleSheet(f"""
            color: {TEXT_DIM}; font-size: 9px; font-family: '{SANS_FONT}';
            background: transparent; border: none;
        """)
        hint.setWordWrap(True)
        v.addWidget(hint)

        # Sample count row
        cnt_row = QHBoxLayout()
        cnt_row.addWidget(_field_label("SAMPLE COUNT"))
        cnt_row.addStretch()
        self.img_spin = QSpinBox()
        self.img_spin.setRange(2, 30)
        self.img_spin.setValue(12)
        self.img_spin.setToolTip(
            "10–15 images per person recommended · quality > quantity\n"
            "Include: front, left, right, near, medium (~1.5 m), far (~2-3 m)")
        self.img_spin.setFixedWidth(65)
        cnt_row.addWidget(self.img_spin)
        v.addLayout(cnt_row)

        tip = QLabel("Tip: 10–15 images · front, left, right · near, medium (~1.5 m), far (~2-3 m)")
        tip.setStyleSheet(f"""
            color: {AMBER}; font-size: 8px; font-family: '{SANS_FONT}';
            background: transparent; border: none;
        """)
        tip.setWordWrap(True)
        v.addWidget(tip)

        # Thumbnail strip
        thumb_frame = QFrame()
        thumb_frame.setFrameShape(QFrame.NoFrame)
        thumb_frame.setFixedHeight(42)
        thumb_frame.setStyleSheet(f"""
            background: {SURFACE};
            border: none;
            border-radius: 4px;
        """)
        self.thumb_layout = QHBoxLayout(thumb_frame)
        self.thumb_layout.setContentsMargins(6, 4, 6, 4)
        self.thumb_layout.setSpacing(4)
        self.thumb_layout.addStretch()
        v.addWidget(thumb_frame)

        self.btn_enroll = _btn("◉  CAPTURE FACE", CYAN)
        self.btn_enroll.clicked.connect(self._on_enroll)
        v.addWidget(self.btn_enroll)

        return card

    # ── Step 2 — Train ────────────────────────────────────────────────

    def _build_step2(self) -> QFrame:
        card, v = self._step_card(
            "02", "Build Recognition Engine",
            "Generate face embeddings from enrolled data")

        # Progress row
        prog_row = QHBoxLayout()
        prog_row.addWidget(_field_label("TRAINING PROGRESS"))
        prog_row.addStretch()
        self.train_pct_lbl = _mono_label("0%", 10, CYAN)
        prog_row.addWidget(self.train_pct_lbl)
        v.addLayout(prog_row)

        self.train_progress = QProgressBar()
        self.train_progress.setValue(0)
        self.train_progress.setTextVisible(False)
        self.train_progress.setFixedHeight(5)
        v.addWidget(self.train_progress)

        self.train_status_lbl = _mono_label("IDLE  —  awaiting command", 10, TEXT_DIM)
        self.train_status_lbl.setWordWrap(True)
        v.addWidget(self.train_status_lbl)

        v.addWidget(_divider())

        # Backend selector
        backend_hdr = QHBoxLayout()
        backend_hdr.addWidget(_field_label("RECOGNITION MODEL"))
        backend_hdr.addStretch()
        v.addLayout(backend_hdr)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems([
            "Auto  (best available)",
            "InsightFace  ArcFace",
            "FaceNet  InceptionResNet",
            "ONNX  Custom Model",
            "DeepFace  Multi-model",
            "Vision Transformer  (ViT-B/16)"
        ])
        self.backend_combo.setCurrentIndex(0)
        self.backend_combo.setToolTip(
            "Select the face embedding backend:\n\n"
            "  Auto — picks InsightFace > ONNX > FaceNet\n"
            "  InsightFace — ArcFace ResNet-50 (best accuracy)\n"
            "  FaceNet — InceptionResNet (reliable fallback)\n"
            "  ONNX — custom .onnx model file\n"
            "  DeepFace — multiple sub-models available\n"
            "  Vision Transformer — pure attention ViT-B/16")
        self.backend_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)

        self.deepface_combo = QComboBox()
        self.deepface_combo.addItems([
            "ArcFace", "Facenet512", "VGG-Face", "SFace", "Facenet", "OpenFace"
        ])
        self.deepface_combo.setToolTip("Select the DeepFace sub-model architecture")
        self.deepface_combo.setFixedWidth(90)
        self.deepface_combo.currentIndexChanged.connect(self._on_deepface_changed)
        self.deepface_combo.hide()
        _sp = self.deepface_combo.sizePolicy()
        _sp.setRetainSizeWhenHidden(True)
        self.deepface_combo.setSizePolicy(_sp)

        combo_row = QHBoxLayout()
        combo_row.setSpacing(6)
        combo_row.addWidget(self.backend_combo, stretch=1)
        combo_row.addWidget(self.deepface_combo)
        v.addLayout(combo_row)

        self.backend_info_lbl = _mono_label(
            "Auto-detect: InsightFace → ONNX → FaceNet", 9, TEXT_DIM)
        self.backend_info_lbl.setWordWrap(True)
        v.addWidget(self.backend_info_lbl)

        self.btn_train = _btn("⬡  BUILD EMBEDDINGS", CYAN)
        self.btn_train.clicked.connect(self._on_train)
        v.addWidget(self.btn_train)

        eval_hint = QLabel("Run an offline accuracy report using the current threshold setting.")
        eval_hint.setWordWrap(True)
        eval_hint.setStyleSheet(f"""
            color: {TEXT_DIM}; font-size: 8px;
            font-family: '{SANS_FONT}'; background: transparent; border: none;
        """)
        v.addWidget(eval_hint)

        self.btn_evaluate = _btn("◫  EVALUATE ACCURACY", CYAN)
        self.btn_evaluate.clicked.connect(self._on_evaluate_accuracy)
        v.addWidget(self.btn_evaluate)

        return card

    # ── Step 3 — Monitor ─────────────────────────────────────────────

    def _build_step3(self) -> QFrame:
        card, v = self._step_card(
            "03", "Launch Live Monitoring",
            "Real-time face recognition on webcam stream")

        # Threshold slider
        thr_row = QHBoxLayout()
        thr_row.addWidget(_field_label("CONFIDENCE THRESHOLD"))
        thr_row.addStretch()
        self.slider_val_lbl = _mono_label(f"{THRESHOLD:.2f}", 10, CYAN)
        thr_row.addWidget(self.slider_val_lbl)
        v.addLayout(thr_row)

        self.strict_slider = QSlider(getattr(Qt, "Horizontal"))
        self.strict_slider.setMinimum(40)
        self.strict_slider.setMaximum(90)
        self.strict_slider.setValue(int(THRESHOLD * 100))
        self.strict_slider.setTickPosition(QSlider.NoTicks)
        self.strict_slider.valueChanged.connect(self._on_slider_changed)
        v.addWidget(self.strict_slider)

        range_row = QHBoxLayout()
        range_row.addWidget(_mono_label("0.40 · LOW", 8, TEXT_DIM))
        range_row.addStretch()
        range_row.addWidget(_mono_label("0.90 · HIGH", 8, TEXT_DIM))
        v.addLayout(range_row)

        # Vote frames
        vote_row = QHBoxLayout()
        vote_row.addWidget(_field_label("VOTE FRAMES"))
        vote_row.addStretch()
        self.vote_spin = QSpinBox()
        self.vote_spin.setRange(1, 10)
        self.vote_spin.setValue(VOTE_FRAMES)
        self.vote_spin.setFixedWidth(65)
        self.vote_spin.setToolTip(
            "Majority-vote window: how many consecutive frames\n"
            "must agree before showing the recognition result.\n"
            "Increase to 5–7 for stability; reduce to 1–2 for instant response.")
        self.vote_spin.valueChanged.connect(self._on_vote_spin_changed)
        vote_row.addWidget(self.vote_spin)
        v.addLayout(vote_row)

        vote_hint = QLabel("Higher = more stable · Lower = faster  (3–5 recommended)")
        vote_hint.setStyleSheet(f"""
            color: {TEXT_DIM}; font-size: 8px;
            font-family: '{SANS_FONT}'; background: transparent; border: none;
        """)
        vote_hint.setWordWrap(True)
        v.addWidget(vote_hint)

        # Start / Stop buttons in a stack so card height never changes
        self.btn_stack = QStackedWidget()
        self.btn_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.btn_monitor = _btn("▶  START RECOGNITION", GREEN)
        self.btn_monitor.clicked.connect(self._on_monitor)
        self.btn_stack.addWidget(self.btn_monitor)   # index 0

        self.btn_stop = _btn("■  STOP MONITORING", RED)
        self.btn_stop.clicked.connect(self._on_stop_monitor)
        self.btn_stack.addWidget(self.btn_stop)      # index 1

        self.btn_stack.setCurrentIndex(0)
        v.addWidget(self.btn_stack)

        return card

    # ── Video Panel ───────────────────────────────────────────────────

    def _build_video_panel(self) -> QFrame:
        card = QFrame()
        card.setFrameShape(QFrame.NoFrame)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        card.setStyleSheet(f"""
            QFrame {{
                background: {BG2};
                border: 1px solid {BORDER};
                border-radius: 6px;
            }}
        """)
        _shadow(card, 24)

        v = QVBoxLayout(card)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)

        # Title bar
        bar = QFrame()
        bar.setFrameShape(QFrame.NoFrame)
        bar.setFixedHeight(26)
        bar.setStyleSheet(f"background: {SURFACE}; border: none; border-radius: 3px;")
        bar_lay = QHBoxLayout(bar)
        bar_lay.setContentsMargins(10, 0, 10, 0)
        bar_lay.setSpacing(7)
        for col in [GREEN, AMBER, RED]:
            dot = QLabel("●")
            dot.setStyleSheet(
                f"color: {col}; font-size: 8px; background: transparent; border: none;")
            bar_lay.addWidget(dot)
        bar_lay.addSpacing(10)
        bar_lay.addWidget(_mono_label("FEED  /  CAM_0", 9, TEXT_DIM))
        bar_lay.addStretch()
        self.feed_status_lbl = _mono_label("● OFFLINE", 9, TEXT_DIM)
        bar_lay.addWidget(self.feed_status_lbl)
        v.addWidget(bar)

        # Video label — Ignored policy so it never forces layout shifts
        self.video_label = QLabel()
        self.video_label.setAlignment(getattr(Qt, "AlignCenter"))
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video_label.setMinimumSize(300, 240)
        self.video_label.setStyleSheet(f"""
            background: {BG};
            border: 1px solid {BORDER2};
            border-radius: 3px;
            color: {TEXT_DIM};
            font-family: '{MONO_FONT}';
            font-size: 12px;
        """)
        self.video_label.setText(
            "▷  CAMERA FEED  ◁\n\nInitialize a workflow step to begin stream")

        # Scanline overlay
        self._scanline = ScanLineWidget(self.video_label)
        self._scanline.setGeometry(self.video_label.rect())

        v.addWidget(self.video_label, stretch=1)
        return card

    def resizeEvent(self, a0):
        super().resizeEvent(a0)
        if hasattr(self, "_scanline") and hasattr(self, "video_label"):
            self._scanline.setGeometry(self.video_label.rect())

    # ── Enrolled Identities Panel ─────────────────────────────────────

    def _build_users_panel(self) -> QFrame:
        card = QFrame()
        card.setFrameShape(QFrame.NoFrame)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        card.setStyleSheet(f"""
            QFrame {{
                background: {PANEL};
                border: none;
                border-radius: 6px;
            }}
        """)
        _shadow(card, 12)

        v = QVBoxLayout(card)
        v.setContentsMargins(14, 12, 12, 12)
        v.setSpacing(8)

        hdr = QHBoxLayout()
        hdr.addWidget(_section_title("ENROLLED IDENTITIES"))
        hdr.addStretch()
        refresh_btn = QPushButton("⟳")
        refresh_btn.setStyleSheet(f"""
            background: transparent; color: {TEXT_DIM};
            font-size: 15px; border: none; padding: 0;
        """)
        refresh_btn.setFixedSize(20, 20)
        refresh_btn.setCursor(getattr(Qt, "PointingHandCursor"))
        refresh_btn.clicked.connect(self._refresh_users)
        hdr.addWidget(refresh_btn)
        v.addLayout(hdr)
        v.addWidget(_divider())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        content.setStyleSheet("background: transparent; border: none;")
        self.users_layout = QVBoxLayout(content)
        self.users_layout.setContentsMargins(0, 0, 0, 0)
        self.users_layout.setSpacing(5)
        scroll.setWidget(content)
        v.addWidget(scroll, stretch=1)
        return card

    # ── Latest Detection Panel ────────────────────────────────────────

    def _build_status_panel(self) -> QFrame:
        card = QFrame()
        card.setFrameShape(QFrame.NoFrame)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        card.setStyleSheet(f"""
            QFrame {{
                background: {PANEL};
                border: none;
                border-radius: 6px;
            }}
        """)
        card.setFixedHeight(96)
        _shadow(card)

        v = QVBoxLayout(card)
        v.setContentsMargins(14, 10, 12, 10)
        v.setSpacing(6)
        v.addWidget(_section_title("LATEST DETECTION"))
        v.addWidget(_divider())

        self.auth_label = QLabel("—  AWAITING STREAM")
        self.auth_label.setAlignment(getattr(Qt, "AlignCenter"))
        self.auth_label.setStyleSheet(f"""
            color: {TEXT_DIM}; font-size: 11px; font-family: '{MONO_FONT}';
            font-weight: 700; background: transparent; border: none;
            padding: 4px 0; letter-spacing: 1px;
        """)
        self.auth_label.setWordWrap(True)
        v.addWidget(self.auth_label)
        return card

    # ══════════════════════════════════════════════════════════════════
    #  Slots  (logic completely unchanged)
    # ══════════════════════════════════════════════════════════════════

    @pyqtSlot()
    def _on_enroll(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation Error",
                                "Please enter a subject name.")
            return
        cnt = self.img_spin.value()
        while self.thumb_layout.count() > 1:
            item = self.thumb_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        self.btn_enroll.setEnabled(False)
        self.btn_enroll.setText("● CAPTURING...")
        self.backend_combo.setEnabled(False)
        self.deepface_combo.setEnabled(False)
        self.feed_status_lbl.setText("● ENROLL")
        self.feed_status_lbl.setStyleSheet(
            f"color: {CYAN}; font-size: 9px; font-family: '{MONO_FONT}';"
            f" background: transparent; border: none;")

        self.enroll_worker = EnrollWorker(name, cnt)
        self.enroll_worker.frame_ready.connect(self._display_frame)
        self.enroll_worker.face_captured.connect(self._on_face_captured)
        self.enroll_worker.finished_signal.connect(self._on_enroll_finished)
        self.enroll_worker.error.connect(self._on_worker_error)
        self.enroll_worker.start()

    @pyqtSlot(np.ndarray, int)
    def _on_face_captured(self, face_rgb: np.ndarray, count: int):
        face_resized = cv2.resize(face_rgb, (30, 30))
        qi  = QImage(face_resized.data, 30, 30, 3 * 30, QImage.Format_RGB888)
        lbl = QLabel()
        lbl.setFixedSize(30, 30)
        lbl.setStyleSheet(f"border: 1px solid {CYAN}; border-radius: 2px;")
        lbl.setPixmap(QPixmap.fromImage(qi))
        self.thumb_layout.insertWidget(self.thumb_layout.count() - 1, lbl)

    @pyqtSlot(str)
    def _on_enroll_finished(self, message: str):
        self.btn_enroll.setEnabled(True)
        self.btn_enroll.setText("◉  CAPTURE FACE")
        self.backend_combo.setEnabled(True)
        self.deepface_combo.setEnabled(True)
        self.name_input.clear()
        self.feed_status_lbl.setText("● OFFLINE")
        self.feed_status_lbl.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 9px; font-family: '{MONO_FONT}';"
            f" background: transparent; border: none;")
        QMessageBox.information(self, "Enrollment Complete", message)
        self._refresh_users()
        self.video_label.setText(
            "▷  CAMERA FEED  ◁\n\nInitialize a workflow step to begin stream")

    @pyqtSlot()
    def _on_train(self):
        self.btn_train.setEnabled(False)
        self.btn_train.setText("⬡  PROCESSING...")
        self.backend_combo.setEnabled(False)
        self.deepface_combo.setEnabled(False)
        self.train_progress.setValue(0)
        self.train_pct_lbl.setText("0%")
        self.train_status_lbl.setText("EXTRACTING FEATURES...")

        self.train_worker = TrainWorker()
        self.train_worker.backend       = self.active_backend
        self.train_worker.deepface_model = self.active_deepface_model
        self.train_worker.progress.connect(self._on_train_progress)
        self.train_worker.finished_signal.connect(self._on_train_finished)
        self.train_worker.error.connect(self._on_worker_error)
        self.train_worker.start()

    @pyqtSlot(int)
    def _on_train_progress(self, val: int):
        self.train_progress.setValue(val)
        self.train_pct_lbl.setText(f"{val}%")

    @pyqtSlot(str)
    def _on_train_finished(self, msg: str):
        self.btn_train.setEnabled(True)
        self.btn_train.setText("⬡  BUILD EMBEDDINGS")
        self.train_status_lbl.setText(f"✓  {msg.upper()}")
        self.backend_combo.setEnabled(True)
        self.deepface_combo.setEnabled(True)
        self.train_status_lbl.setStyleSheet(
            f"color: {GREEN}; font-size: 10px; font-family: '{MONO_FONT}';"
            f" background: transparent; border: none;")
        self._set_indicator(self.ind_embeddings, True)
        self._refresh_users()
        QMessageBox.information(self, "Training Complete", msg)

    def _set_accuracy_running(self, running: bool):
        self.btn_evaluate.setEnabled(not running)
        self.btn_evaluate.setText("◫  ANALYZING..." if running else "◫  EVALUATE ACCURACY")
        self.btn_enroll.setEnabled(not running)
        self.btn_train.setEnabled(not running)
        self.btn_monitor.setEnabled(not running)
        self.backend_combo.setEnabled(not running)
        self.deepface_combo.setEnabled(not running and self.active_backend == "deepface")

    @pyqtSlot()
    def _on_evaluate_accuracy(self):
        if self.enroll_worker and self.enroll_worker.isRunning():
            QMessageBox.warning(self, "Capture In Progress",
                                "Finish enrollment before running the accuracy report.")
            return
        if self.train_worker and self.train_worker.isRunning():
            QMessageBox.warning(self, "Training In Progress",
                                "Wait for Step 2 to finish before running the accuracy report.")
            return
        if self.recog_worker:
            QMessageBox.warning(self, "Monitoring Active",
                                "Stop live monitoring before running the accuracy report.")
            return
        if not Path(EMBEDDINGS_DIR, "prototypes.npy").exists():
            QMessageBox.warning(self, "Not Trained",
                                "Run Step 2 to build the recognition engine first.")
            return

        self._set_accuracy_running(True)
        self.train_status_lbl.setText(
            f"RUNNING ACCURACY EVALUATION  ·  threshold {self.live_threshold:.2f}")
        self.train_status_lbl.setStyleSheet(
            f"color: {CYAN}; font-size: 10px; font-family: '{MONO_FONT}';"
            f" background: transparent; border: none;")

        self.accuracy_worker = AccuracyWorker()
        self.accuracy_worker.threshold = self.live_threshold
        self.accuracy_worker.metric = SIMILARITY_METRIC
        self.accuracy_worker.backend = self.active_backend
        self.accuracy_worker.deepface_model = self.active_deepface_model
        self.accuracy_worker.finished_signal.connect(self._on_accuracy_finished)
        self.accuracy_worker.error.connect(self._on_accuracy_error)
        self.accuracy_worker.start()

    @pyqtSlot(str)
    def _on_accuracy_finished(self, report: str):
        self.accuracy_worker = None
        self._set_accuracy_running(False)
        self.train_status_lbl.setText(
            f"✓  ACCURACY REPORT READY  ·  threshold {self.live_threshold:.2f}")
        self.train_status_lbl.setStyleSheet(
            f"color: {GREEN}; font-size: 10px; font-family: '{MONO_FONT}';"
            f" background: transparent; border: none;")
        self._show_accuracy_report(report)

    @pyqtSlot(str)
    def _on_accuracy_error(self, msg: str):
        self.accuracy_worker = None
        self._set_accuracy_running(False)
        self.train_status_lbl.setText("✗  ACCURACY EVALUATION FAILED")
        self.train_status_lbl.setStyleSheet(
            f"color: {RED}; font-size: 10px; font-family: '{MONO_FONT}';"
            f" background: transparent; border: none;")
        QMessageBox.critical(self, "Accuracy Evaluation Failed", msg)

    def _show_accuracy_report(self, report: str):
        dialog = QDialog(self)
        dialog.setWindowTitle("Accuracy Evaluation")
        dialog.setModal(True)
        dialog.resize(960, 720)
        dialog.setStyleSheet(STYLESHEET)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title = QLabel("OFFLINE ACCURACY REPORT")
        title.setStyleSheet(f"""
            color: {CYAN}; font-size: 12px; font-weight: 700;
            font-family: '{MONO_FONT}'; background: transparent; border: none;
        """)
        layout.addWidget(title)

        viewer = QPlainTextEdit()
        viewer.setReadOnly(True)
        viewer.setPlainText(report)
        viewer.setLineWrapMode(QPlainTextEdit.NoWrap)
        viewer.setStyleSheet(f"""
            QPlainTextEdit {{
                background: {BG2};
                color: {TEXT};
                border: 1px solid {BORDER};
                border-radius: 6px;
                padding: 8px;
                font-family: '{MONO_FONT}';
                font-size: 11px;
            }}
        """)
        layout.addWidget(viewer, stretch=1)

        close_btn = _btn("CLOSE REPORT", CYAN, dialog)
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.exec_()

    @pyqtSlot()
    def _on_monitor(self):
        if not Path(EMBEDDINGS_DIR, "prototypes.npy").exists():
            QMessageBox.warning(self, "Not Trained",
                                "Run Step 2 to build the recognition engine first.")
            return
        self.btn_stack.setCurrentIndex(1)
        self.btn_enroll.setEnabled(False)
        self.btn_train.setEnabled(False)
        self.backend_combo.setEnabled(False)
        self.deepface_combo.setEnabled(False)
        self.feed_status_lbl.setText("● LIVE")
        self.feed_status_lbl.setStyleSheet(
            f"color: {GREEN}; font-size: 9px; font-family: '{MONO_FONT}';"
            f" background: transparent; border: none;")

        self.recog_worker = RecognitionWorker()
        self.recog_worker.backend          = self.active_backend
        self.recog_worker.deepface_model   = self.active_deepface_model
        self.recog_worker.live_threshold   = self.live_threshold
        self.recog_worker.live_vote_frames = self.live_vote_frames
        self.recog_worker.frame_ready.connect(self._display_frame)
        self.recog_worker.person_detected.connect(self._on_person_detected)
        self.recog_worker.error.connect(self._on_worker_error)
        self.recog_worker.start()

    @pyqtSlot()
    def _on_stop_monitor(self):
        if self.recog_worker:
            self.recog_worker.stop()
            self.recog_worker.wait()
            self.recog_worker = None
        self.btn_stack.setCurrentIndex(0)
        self.btn_enroll.setEnabled(True)
        self.btn_train.setEnabled(True)
        self.backend_combo.setEnabled(True)
        self.deepface_combo.setEnabled(True)
        self.video_label.clear()
        self.video_label.setText("▷  CAMERA FEED  ◁\n\nMonitoring stopped")
        self.feed_status_lbl.setText("● OFFLINE")
        self.feed_status_lbl.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 9px; font-family: '{MONO_FONT}';"
            f" background: transparent; border: none;")
        self.auth_label.setText("—  STREAM TERMINATED")
        self.auth_label.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 11px; font-family: '{MONO_FONT}';"
            f" font-weight: 700; background: transparent; border: none;"
            f" padding: 4px 0; letter-spacing: 1px;")

    @pyqtSlot(str, float, str)
    def _on_person_detected(self, name: str, conf: float, ts: str):
        if name != "Unknown":
            state = "authorized"
            self.auth_label.setText(
                f"✓  AUTHORIZED  —  {name}  [{conf*100:.2f}%]")
            if state != self._last_auth_state:
                self._last_auth_state = state
                self.auth_label.setStyleSheet(
                    f"color: {GREEN}; font-size: 11px; font-family: '{MONO_FONT}';"
                    f" font-weight: 700; background: transparent; border: none;"
                    f" padding: 4px 0; letter-spacing: 1px;")
        else:
            state = "unknown"
            self.auth_label.setText(
                f"✗  UNKNOWN SUBJECT  —  {conf*100:.2f}%  MATCH")
            if state != self._last_auth_state:
                self._last_auth_state = state
                self.auth_label.setStyleSheet(
                    f"color: {RED}; font-size: 11px; font-family: '{MONO_FONT}';"
                    f" font-weight: 700; background: transparent; border: none;"
                    f" padding: 4px 0; letter-spacing: 1px;")

    @pyqtSlot(int)
    def _on_slider_changed(self, value: int):
        self.live_threshold = value / 100.0
        self.slider_val_lbl.setText(f"{self.live_threshold:.2f}")
        if self.recog_worker:
            self.recog_worker.live_threshold = self.live_threshold

    @pyqtSlot(int)
    def _on_vote_spin_changed(self, value: int):
        self.live_vote_frames = value
        if self.recog_worker:
            self.recog_worker.live_vote_frames = value

    @pyqtSlot(int)
    def _on_backend_changed(self, index: int):
        _BACKEND_MAP = {
            0: ("auto",        "Auto-detect: InsightFace → ONNX → FaceNet"),
            1: ("insightface", "InsightFace ArcFace ResNet-50  ·  512-d  ·  ~99.7% LFW"),
            2: ("facenet",     "FaceNet InceptionResNetV1  ·  512-d  ·  ~99.6% LFW"),
            3: ("onnx",        "Custom ONNX model  ·  uses models/w600k_r50.onnx"),
            4: ("deepface",    "DeepFace library  ·  select sub-model"),
            5: ("vit",         "Vision Transformer (ViT-B/16)  ·  768-d  ·  Pure Attention"),
        }
        backend, desc = _BACKEND_MAP.get(index, ("auto", ""))
        self.active_backend = backend
        self.backend_info_lbl.setText(desc)

        if backend == "deepface":
            self.deepface_combo.show()
            self.active_model_lbl.setText(
                f"{backend.upper()}({self.active_deepface_model})")
        else:
            self.deepface_combo.hide()
            self.active_model_lbl.setText(backend.upper())

        self.train_status_lbl.setText(
            f"MODEL CHANGED → {self.active_model_lbl.text()}  ·  re-train required")
        self.train_status_lbl.setStyleSheet(
            f"color: {AMBER}; font-size: 10px; font-family: '{MONO_FONT}';"
            f" background: transparent; border: none;")
        self.train_progress.setValue(0)
        self.train_pct_lbl.setText("0%")

    @pyqtSlot(int)
    def _on_deepface_changed(self, index: int):
        self.active_deepface_model = self.deepface_combo.currentText()
        self.active_model_lbl.setText(f"DEEPFACE({self.active_deepface_model})")
        self.train_status_lbl.setText(
            f"MODEL CHANGED → DEEPFACE({self.active_deepface_model})  ·  re-train required")
        self.train_status_lbl.setStyleSheet(
            f"color: {AMBER}; font-size: 10px; font-family: '{MONO_FONT}';"
            f" background: transparent; border: none;")
        self.train_progress.setValue(0)
        self.train_pct_lbl.setText("0%")

    def _on_worker_error(self, msg: str):
        QMessageBox.critical(self, "System Error", msg)
        self.btn_enroll.setEnabled(True)
        self.btn_enroll.setText("◉  CAPTURE FACE")
        self.btn_train.setEnabled(True)
        self.btn_train.setText("⬡  BUILD EMBEDDINGS")
        if self.btn_stop.isVisible():
            self._on_stop_monitor()

    # ── Display ───────────────────────────────────────────────────────

    @pyqtSlot(np.ndarray)
    def _display_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qi  = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pm  = QPixmap.fromImage(qi)
        scaled = pm.scaled(
            self.video_label.size(),
            getattr(Qt, "KeepAspectRatio"),
            getattr(Qt, "SmoothTransformation"))
        self.video_label.setPixmap(scaled)

    # ── Refresh ───────────────────────────────────────────────────────

    def _refresh_status(self):
        self._set_indicator(self.ind_model, True)
        cap = (cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
               if sys.platform.startswith("win")
               else cv2.VideoCapture(CAMERA_INDEX))
        cam_ok = cap.isOpened()
        cap.release()
        self._set_indicator(self.ind_camera, cam_ok)
        self._set_indicator(
            self.ind_embeddings,
            Path(EMBEDDINGS_DIR, "prototypes.npy").exists())
        self._set_indicator(self.ind_system, True)

    def _refresh_users(self):
        while self.users_layout.count():
            item = self.users_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        folders = get_identity_folders(DATASET_DIR)
        emb_ok  = Path(EMBEDDINGS_DIR, "prototypes.npy").exists()

        if not folders:
            ph = QLabel("NO IDENTITIES ENROLLED")
            ph.setStyleSheet(f"""
                color: {TEXT_DIM}; font-size: 10px; font-family: '{MONO_FONT}';
                padding: 14px; border: none;
            """)
            ph.setAlignment(getattr(Qt, "AlignCenter"))
            self.users_layout.addWidget(ph)
        else:
            for folder in folders:
                imgs = (list(folder.glob("*.jpg")) +
                        list(folder.glob("*.png")) +
                        list(folder.glob("*.jpeg")))
                size_human = format_size(get_dir_size(folder))
                self.users_layout.addWidget(
                    self._user_card(folder.name, len(imgs), size_human, imgs, emb_ok))

        self.users_layout.addStretch()

    def _user_card(self, name: str, count: int,
                   size_human: str, imgs: list, trained: bool) -> QFrame:
        card = QFrame()
        card.setFrameShape(QFrame.NoFrame)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        card.setStyleSheet(f"""
            QFrame {{
                background: {SURFACE};
                border: none;
                border-radius: 4px;
            }}
            QFrame:hover {{
                background: {SURFACE2};
            }}
        """)
        h = QHBoxLayout(card)
        h.setContentsMargins(8, 7, 8, 7)
        h.setSpacing(8)

        # Avatar
        av = QLabel()
        av.setFixedSize(34, 34)
        av.setAlignment(getattr(Qt, "AlignCenter"))
        if imgs:
            img = cv2.imread(str(imgs[0]))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_rgb = cv2.resize(img_rgb, (34, 34))
                qi = QImage(img_rgb.data, 34, 34, 3 * 34, QImage.Format_RGB888)
                av.setPixmap(QPixmap.fromImage(qi).scaled(
                    34, 34, getattr(Qt, "KeepAspectRatio"), getattr(Qt, "SmoothTransformation")))
                av.setStyleSheet(f"border: 1px solid {CYAN}; border-radius: 2px;")
            else:
                av.setText(name[0].upper())
                av.setStyleSheet(f"""
                    background: {CYAN_DIM}; color: {CYAN}; font-size: 14px;
                    font-weight: 700; font-family: '{MONO_FONT}';
                    border: 1px solid {CYAN}; border-radius: 2px;
                """)
        else:
            av.setText(name[0].upper())
            av.setStyleSheet(f"""
                background: {CYAN_DIM}; color: {CYAN}; font-size: 14px;
                font-weight: 700; font-family: '{MONO_FONT}';
                border: 1px solid {CYAN}; border-radius: 2px;
            """)
        h.addWidget(av)

        # Info
        info = QVBoxLayout()
        info.setSpacing(2)
        info.setContentsMargins(0, 0, 0, 0)
        n_lbl = QLabel(name)
        n_lbl.setStyleSheet(f"""
            font-size: 11px; font-weight: 700; color: {TEXT};
            font-family: '{SANS_FONT}'; background: transparent; border: none;
        """)
        c_lbl = QLabel(f"{count} samples | {size_human}")
        c_lbl.setStyleSheet(f"""
            font-size: 9px; color: {TEXT_DIM}; font-family: '{MONO_FONT}';
            background: transparent; border: none;
        """)
        info.addWidget(n_lbl)
        info.addWidget(c_lbl)
        h.addLayout(info, stretch=1)

        # Badge
        badge_text = "READY" if trained else "UNTRAINED"
        badge_col  = GREEN  if trained else AMBER
        badge_dim  = GREEN_DIM if trained else AMBER_DIM
        badge = QLabel(badge_text)
        badge.setStyleSheet(f"""
            color: {badge_col}; background: {badge_dim};
            border: 1px solid rgba(255,255,255,0.08); border-radius: 3px;
            font-size: 7px; font-weight: 700; font-family: '{MONO_FONT}';
            padding: 2px 5px; letter-spacing: 1px;
        """)
        badge.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        h.addWidget(badge)
        return card

    # ── Cleanup ───────────────────────────────────────────────────────

    def closeEvent(self, a0):
        if self.enroll_worker:
            self.enroll_worker.stop()
            self.enroll_worker.wait(2000)
        if self.accuracy_worker:
            self.accuracy_worker.wait(2000)
        if self.recog_worker:
            self.recog_worker.stop()
            self.recog_worker.wait(2000)
        if a0 is not None:
            a0.accept()


# ══════════════════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════════════════

def main():
    os.chdir(str(PROJECT_DIR))
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(8,  12,  20))
    palette.setColor(QPalette.WindowText,      QColor(208, 228, 247))
    palette.setColor(QPalette.Base,            QColor(12,  18,  32))
    palette.setColor(QPalette.AlternateBase,   QColor(15,  26,  46))
    palette.setColor(QPalette.Text,            QColor(208, 228, 247))
    palette.setColor(QPalette.Button,          QColor(12,  18,  32))
    palette.setColor(QPalette.ButtonText,      QColor(208, 228, 247))
    palette.setColor(QPalette.Highlight,       QColor(0,  212, 255))
    palette.setColor(QPalette.HighlightedText, QColor(8,   12,  20))
    app.setPalette(palette)

    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    # ── Auto-use .venv if available ──
    venv_python = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
    if (
        venv_python.exists()
        and Path(sys.executable).resolve() != venv_python.resolve()
    ):
        import subprocess
        sys.exit(subprocess.call([str(venv_python)] + sys.argv))
    main()