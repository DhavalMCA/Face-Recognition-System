"""AI Face Recognition Monitoring System — PyQt5 Desktop GUI.

A professional 3-step workflow interface for the FewShotFace recognition
system. Redesigned with Neural Surveillance Terminal aesthetic.

Usage:
    python gui.py
"""

from __future__ import annotations

import sys
import time
import os
import ssl
from datetime import datetime
from pathlib import Path
from typing import List, Optional

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
    compute_class_prototypes,
    detect_faces,
    ensure_dir,
    get_identity_folders,
    load_face_detector,
    load_saved_embeddings,
)
from similarity import predict_with_prototypes
from recognize import FrameVoter

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QImage, QPixmap, QPainter, QPalette, QBrush
from PyQt5.QtWidgets import (
    QApplication, QFrame, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QScrollArea, QSizePolicy, QSpinBox, QVBoxLayout,
    QWidget, QGraphicsDropShadowEffect, QSlider,
)

# ── Internal ML config ──────────────────────────────────────────────
SIMILARITY_METRIC = "cosine"
THRESHOLD         = 0.60
VOTE_FRAMES       = 7
CAMERA_INDEX      = 0
EMBEDDING_BACKEND = "auto"
ONNX_MODEL_PATH   = "models/arcface.onnx"
DATASET_DIR       = "dataset"
EMBEDDINGS_DIR    = "embeddings"

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
QLabel {{ background: transparent; color: {TEXT}; border: none; }}
QLineEdit {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 14px;
    color: {TEXT};
    font-size: 13px;
    font-family: '{SANS_FONT}';
    selection-background-color: {CYAN};
}}
QLineEdit:focus {{
    border: 1px solid {CYAN};
    background: {SURFACE2};
}}
QLineEdit::placeholder {{ color: {TEXT_MID}; font-style: italic; }}
QSpinBox {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 14px;
    color: {TEXT};
    font-size: 13px;
}}
QSpinBox:focus {{ border: 1px solid {CYAN}; }}
QProgressBar {{
    background: {SURFACE};
    border: none;
    border-radius: 3px;
    text-align: center;
    color: {TEXT};
    font-size: 11px;
    font-family: '{MONO_FONT}';
    height: 6px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {CYAN}, stop:1 #0099CC);
    border-radius: 3px;
}}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: {BG2};
    width: 4px;
    border-radius: 2px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 2px;
    min-height: 24px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
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
    width: 12px; height: 12px;
    margin: -4px 0;
    border-radius: 6px;
    border: 2px solid {CYAN};
}}
QSlider::handle:horizontal:hover {{ background: {CYAN}; }}
"""


# ══════════════════════════════════════════════════════════════════════
#  Helper Widgets / Factories
# ══════════════════════════════════════════════════════════════════════

def _shadow(widget: QWidget, blur: int = 18):
    e = QGraphicsDropShadowEffect()
    e.setBlurRadius(blur)
    c = QColor("#000000")
    c.setAlpha(140)
    e.setColor(c)
    e.setOffset(0, 3)
    widget.setGraphicsEffect(e)


def _btn(label: str, color: str = CYAN, parent=None) -> QPushButton:
    btn = QPushButton(label, parent)
    btn.setCursor(Qt.PointingHandCursor)
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
            border-radius: 4px;
            padding: 10px 22px;
            font-size: 13px;
            font-weight: 600;
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


def _step_number(n: str) -> QLabel:
    lbl = QLabel(n)
    lbl.setFixedSize(28, 28)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet(f"""
        color: {CYAN}; background: transparent; border: none;
        font-size: 11px; font-weight: 700; font-family: '{MONO_FONT}';
    """)
    return lbl


def _section_title(text: str, color: str = CYAN) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(f"""
        color: {color}; font-size: 10px; font-family: '{SANS_FONT}';
        font-weight: 700; letter-spacing: 2px;
        background: transparent; border: none;
    """)
    return lbl


def _mono_label(text: str, size: int = 12, color: str = TEXT) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"""
        color: {color}; font-family: '{MONO_FONT}';
        font-size: {size}px; background: transparent; border: none;
    """)
    return lbl


def _field_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color: {TEXT_MID}; font-size: 11px; font-family: '{SANS_FONT}';"
        f" font-weight: 600; letter-spacing: 0.5px; background: transparent; border: none;"
    )
    return lbl


def _divider() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setFrameShadow(QFrame.Plain)
    f.setFixedHeight(1)
    f.setStyleSheet(f"background: {BORDER2}; border: none; margin: 2px 0;")
    return f


# ══════════════════════════════════════════════════════════════════════
#  Scan Line Overlay
# ══════════════════════════════════════════════════════════════════════

class ScanLineWidget(QWidget):
    """Animated horizontal scan line for cyberpunk visual effect."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._y = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)

    def _tick(self):
        self._y = (self._y + 2) % max(1, self.height())
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(0, 212, 255, 18)))
        p.drawRect(0, self._y, self.width(), 3)


# ══════════════════════════════════════════════════════════════════════
#  Worker Threads  (logic completely unchanged)
# ══════════════════════════════════════════════════════════════════════

class EnrollWorker(QThread):
    frame_ready     = pyqtSignal(np.ndarray)
    face_captured   = pyqtSignal(np.ndarray, int)
    finished_signal = pyqtSignal(int)
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
            HINTS = {
                1:  "Look straight at camera",
                2:  "Tilt head slightly LEFT",
                3:  "Tilt head slightly RIGHT",
                4:  "Look straight, smile slightly",
                5:  "Look straight (medium distance)",
                6:  "Turn chin slightly UP",
                7:  "Turn chin slightly DOWN",
                8:  "Look straight (step back 2 m)",
                9:  "Tilt head slightly LEFT (far)",
                10: "Look straight, normal expression",
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
                    cv2.imwrite(str(person_dir / fname), face_bgr)
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
            self.finished_signal.emit(captured)
        except Exception as e:
            self.error.emit(str(e))


class TrainWorker(QThread):
    progress        = pyqtSignal(int)
    finished_signal = pyqtSignal(str)
    error           = pyqtSignal(str)

    def run(self):
        try:
            ensure_dir(EMBEDDINGS_DIR)
            folders = get_identity_folders(DATASET_DIR)
            if not folders:
                self.error.emit("No identity folders found. Enroll users first.")
                return
            self.progress.emit(10)
            detector = load_face_detector()
            embedder = FaceEmbedder(backend=EMBEDDING_BACKEND,
                                    onnx_model_path=ONNX_MODEL_PATH)
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
                    emb = embedder.embed_face(largest.face_rgb)
                    all_embeddings.append(emb)
                    all_labels.append(folder.name)
                self.progress.emit(25 + int(65 * (i + 1) / len(folders)))

            if not all_embeddings:
                self.error.emit("No usable face images found.")
                return

            emb_arr = np.vstack(all_embeddings).astype(np.float32)
            lbl_arr = np.array(all_labels, dtype=str)
            np.save(Path(EMBEDDINGS_DIR) / "embeddings.npy", emb_arr)
            np.save(Path(EMBEDDINGS_DIR) / "labels.npy",     lbl_arr)
            protos, names = compute_class_prototypes(emb_arr, lbl_arr)
            np.save(Path(EMBEDDINGS_DIR) / "prototypes.npy",  protos)
            np.save(Path(EMBEDDINGS_DIR) / "class_names.npy", names)
            self.progress.emit(100)
            self.finished_signal.emit(
                f"Trained on {len(emb_arr)} samples · {len(names)} identities")
        except Exception as e:
            self.error.emit(str(e))


class RecognitionWorker(QThread):
    frame_ready     = pyqtSignal(np.ndarray)
    person_detected = pyqtSignal(str, float, str)
    error           = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            detector   = load_face_detector()
            embedder   = FaceEmbedder(backend=EMBEDDING_BACKEND,
                                      onnx_model_path=ONNX_MODEL_PATH)
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

            vf    = getattr(self, "live_vote_frames", VOTE_FRAMES)
            voter = FrameVoter(window=vf, max_missed_frames=12)
            cap   = (cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
                     if sys.platform.startswith("win")
                     else cv2.VideoCapture(CAMERA_INDEX))
            if not cap.isOpened():
                self.error.emit("Could not open webcam.")
                return

            while self._running:
                ok, frame = cap.read()
                if not ok:
                    continue
                detections = detect_faces(frame, detector,
                                          min_confidence=0.88, padding=0.12)
                voter.expire([d.box for d in detections])
                for det in detections:
                    embedding  = embedder.embed_face(det.face_rgb)
                    live_thr   = getattr(self, "live_threshold", THRESHOLD)
                    raw_result = predict_with_prototypes(
                        query_embedding=embedding,
                        prototypes=prototypes,
                        class_names=class_names,
                        metric=SIMILARITY_METRIC,
                        threshold=live_thr)
                    voted_name, voted_conf = voter.vote(
                        box=det.box,
                        name=str(raw_result["name"]),
                        confidence=float(raw_result["confidence"]))

                    x1, y1, x2, y2 = det.box
                    name   = voted_name
                    conf   = voted_conf
                    color  = (0, 255, 136) if name != "Unknown" else (51, 51, 255)
                    status = "AUTHORIZED"  if name != "Unknown" else "UNKNOWN"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, max(0, y1 - 50)), (x2, y1),
                                  (8, 12, 20), -1)
                    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
                    cv2.putText(frame, name, (x1 + 6, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 235, 250), 2)
                    cv2.putText(frame, f"{conf*100:.0f}% | {status}",
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
#  Indicator Pill Widget
# ══════════════════════════════════════════════════════════════════════

class IndicatorPill(QFrame):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setFixedHeight(28)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 14, 0)
        lay.setSpacing(6)

        self.dot = QLabel("◆")
        self.txt = QLabel(label)
        lay.addWidget(self.dot)
        lay.addWidget(self.txt)
        self.setActive(False)

    def setActive(self, active: bool):
        if active:
            self.dot.setStyleSheet(
                f"color: {GREEN}; font-size: 7px; background: transparent; border: none;")
            self.txt.setStyleSheet(
                f"color: {GREEN}; font-size: 11px; font-family: '{MONO_FONT}';"
                f" font-weight: 600; background: transparent; border: none;")
            self.setStyleSheet(
                f"background: {GREEN_DIM}; border: 1px solid rgba(0,255,136,0.20);"
                f" border-radius: 14px;")
        else:
            self.dot.setStyleSheet(
                f"color: {TEXT_DIM}; font-size: 7px; background: transparent; border: none;")
            self.txt.setStyleSheet(
                f"color: {TEXT_DIM}; font-size: 11px; font-family: '{MONO_FONT}';"
                f" background: transparent; border: none;")
            self.setStyleSheet(
                f"background: {SURFACE}; border: 1px solid {BORDER2}; border-radius: 14px;")


# ══════════════════════════════════════════════════════════════════════
#  Main Window
# ══════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Face Recognition Monitoring System")
        self.setMinimumSize(1300, 820)
        self.setStyleSheet(STYLESHEET)
        self.live_threshold   = THRESHOLD
        self.live_vote_frames = VOTE_FRAMES

        self.enroll_worker: Optional[EnrollWorker]      = None
        self.train_worker:  Optional[TrainWorker]       = None
        self.recog_worker:  Optional[RecognitionWorker] = None

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

        # ── Body wrapper ──────────────────────────────────────────────
        body = QFrame()
        body.setFrameShape(QFrame.NoFrame)
        body.setStyleSheet("background: transparent; border: none;")

        # QHBoxLayout replaces QGridLayout.
        # Explicit fixed-width widgets on the sides prevent Qt from
        # redistributing column space when video frames arrive.
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(16, 16, 16, 16)
        body_lay.setSpacing(14)

        # ── LEFT — fixed 380 px ───────────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Hard-pin the width so no content change can push the video panel.
        left_scroll.setFixedWidth(380)
        left_scroll.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        left_w = QWidget()
        left_w.setStyleSheet("background: transparent;")
        left_lay = QVBoxLayout(left_w)
        left_lay.setContentsMargins(0, 0, 4, 0)
        left_lay.setSpacing(10)
        left_lay.addWidget(self._build_step1())
        left_lay.addWidget(self._build_step2())
        left_lay.addWidget(self._build_step3())
        # addStretch inside scroll content only — keeps cards at top,
        # has zero effect on the scroll area's own fixed width.
        left_lay.addStretch()
        left_scroll.setWidget(left_w)

        # ── CENTER — flexible, absorbs all leftover space ─────────────
        center_w = self._build_video_panel()
        center_w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # ── RIGHT — fixed 320 px ──────────────────────────────────────
        right_w = QWidget()
        right_w.setStyleSheet("background: transparent;")
        right_w.setFixedWidth(320)
        right_w.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        right_lay = QVBoxLayout(right_w)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(10)
        right_lay.addWidget(self._build_users_panel(), 1)  # grows vertically
        right_lay.addWidget(self._build_status_panel())    # fixed height

        # stretch=0 (default) for left/right → they never grow.
        # stretch=1 for center              → takes all remaining space.
        body_lay.addWidget(left_scroll)          # stretch=0
        body_lay.addWidget(center_w, stretch=1)  # stretch=1
        body_lay.addWidget(right_w)              # stretch=0

        root.addWidget(body, 1)

    # ── Header ────────────────────────────────────────────────────────

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setFrameShape(QFrame.NoFrame)
        header.setFixedHeight(62)
        header.setStyleSheet(f"background: {BG2}; border-bottom: 1px solid {BORDER};")

        lay = QHBoxLayout(header)
        lay.setContentsMargins(20, 0, 20, 0)
        lay.setSpacing(0)

        logo_frame = QFrame()
        logo_frame.setFrameShape(QFrame.NoFrame)
        logo_frame.setStyleSheet("background: transparent; border: none;")
        logo_lay = QHBoxLayout(logo_frame)
        logo_lay.setContentsMargins(0, 0, 0, 0)
        logo_lay.setSpacing(10)

        bracket_l = QLabel("[")
        bracket_l.setStyleSheet(
            f"color: {CYAN}; font-size: 22px; font-weight: 900;"
            f" font-family: '{MONO_FONT}'; background: transparent; border: none;")
        icon_lbl = QLabel("◈")
        icon_lbl.setStyleSheet(
            f"color: {CYAN}; font-size: 18px; background: transparent; border: none;")
        bracket_r = QLabel("]")
        bracket_r.setStyleSheet(
            f"color: {CYAN}; font-size: 22px; font-weight: 900;"
            f" font-family: '{MONO_FONT}'; background: transparent; border: none;")

        title_frame = QFrame()
        title_frame.setFrameShape(QFrame.NoFrame)
        title_frame.setStyleSheet("background: transparent; border: none;")
        title_vlay = QVBoxLayout(title_frame)
        title_vlay.setContentsMargins(0, 0, 0, 0)
        title_vlay.setSpacing(1)

        t1 = QLabel("AI FACE RECOGNITION")
        t1.setStyleSheet(
            f"color: {TEXT}; font-size: 14px; font-weight: 700;"
            f" font-family: '{MONO_FONT}'; letter-spacing: 2px;"
            f" background: transparent; border: none;")
        t2 = QLabel("MONITORING SYSTEM  v2.0")
        t2.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 10px; font-family: '{MONO_FONT}';"
            f" letter-spacing: 1px; background: transparent; border: none;")
        title_vlay.addWidget(t1)
        title_vlay.addWidget(t2)

        logo_lay.addWidget(bracket_l)
        logo_lay.addWidget(icon_lbl)
        logo_lay.addWidget(bracket_r)
        logo_lay.addSpacing(4)
        logo_lay.addWidget(title_frame)

        lay.addWidget(logo_frame)
        lay.addStretch()

        self.clock_lbl = QLabel()
        self.clock_lbl.setStyleSheet(
            f"color: {CYAN}; font-size: 12px; font-family: '{MONO_FONT}';"
            f" background: transparent; border: none;")
        self._update_clock()
        clock_timer = QTimer(self)
        clock_timer.timeout.connect(self._update_clock)
        clock_timer.start(1000)
        lay.addWidget(self.clock_lbl)
        lay.addSpacing(24)

        self.ind_model      = IndicatorPill("MODEL")
        self.ind_camera     = IndicatorPill("CAMERA")
        self.ind_embeddings = IndicatorPill("EMBEDDINGS")
        self.ind_system     = IndicatorPill("SYSTEM")
        for pill in [self.ind_model, self.ind_camera,
                     self.ind_embeddings, self.ind_system]:
            lay.addWidget(pill)
            lay.addSpacing(6)

        return header

    def _update_clock(self):
        self.clock_lbl.setText(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))

    def _set_indicator(self, widget: IndicatorPill, active: bool):
        widget.setActive(active)

    # ── Step Card ─────────────────────────────────────────────────────

    def _step_card(self, num: str, title: str,
                   subtitle: str = "") -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setFrameShape(QFrame.NoFrame)
        card.setStyleSheet(f"""
            QFrame {{
                background: {PANEL};
                border: none;
                border-radius: 6px;
            }}
        """)
        _shadow(card, 14)

        v = QVBoxLayout(card)
        v.setContentsMargins(18, 16, 18, 18)
        v.setSpacing(12)

        hdr = QHBoxLayout()
        hdr.setSpacing(10)
        hdr.setContentsMargins(0, 0, 0, 0)

        step_tag = QLabel(f"STEP {num}")
        step_tag.setStyleSheet(f"""
            color: {CYAN}; background: {CYAN_DIM}; border: none;
            border-radius: 3px; font-size: 9px; font-weight: 700;
            font-family: '{MONO_FONT}'; letter-spacing: 1px; padding: 3px 7px;
        """)
        step_tag.setFixedHeight(20)

        title_block = QVBoxLayout()
        title_block.setSpacing(2)
        title_block.setContentsMargins(0, 0, 0, 0)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            f"color: {TEXT}; font-size: 14px; font-weight: 700;"
            f" font-family: '{SANS_FONT}'; background: transparent; border: none;")
        title_block.addWidget(title_lbl)

        if subtitle:
            sub_lbl = QLabel(subtitle)
            sub_lbl.setStyleSheet(
                f"color: {TEXT_DIM}; font-size: 10px;"
                f" font-family: '{SANS_FONT}'; background: transparent; border: none;")
            title_block.addWidget(sub_lbl)

        hdr.addWidget(step_tag)
        hdr.addLayout(title_block)
        hdr.addStretch()
        v.addLayout(hdr)
        v.addWidget(_divider())

        return card, v

    # ── Step 1 — Enroll ───────────────────────────────────────────────

    def _build_step1(self) -> QFrame:
        card, v = self._step_card(
            "01", "Enroll New Identity",
            "Capture face samples for a person")

        name_hdr = QHBoxLayout()
        name_hdr.setSpacing(6)
        req_badge = QLabel("● REQUIRED")
        req_badge.setStyleSheet(
            f"color: {CYAN}; font-size: 9px; font-family: '{MONO_FONT}';"
            f" font-weight: 700; letter-spacing: 1px; background: transparent; border: none;")
        name_hdr.addWidget(_field_label("SUBJECT NAME"))
        name_hdr.addStretch()
        name_hdr.addWidget(req_badge)
        v.addLayout(name_hdr)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g.  Dhaval")
        self.name_input.setToolTip("Enter the full name of the person to enroll")
        self.name_input.setStyleSheet(f"""
            QLineEdit {{
                background: {SURFACE2}; border: 1.5px solid {CYAN};
                border-radius: 6px; padding: 9px 14px;
                color: {TEXT}; font-size: 13px; font-family: '{SANS_FONT}';
            }}
            QLineEdit:focus {{ border: 1.5px solid {CYAN}; background: {SURFACE2}; }}
            QLineEdit::placeholder {{ color: {TEXT_DIM}; font-style: italic; }}
        """)
        v.addWidget(self.name_input)

        name_hint = QLabel(
            "Enter name above, then click  CAPTURE FACE  to begin enrollment")
        name_hint.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 10px; font-family: '{SANS_FONT}';"
            f" background: transparent; border: none;")
        name_hint.setWordWrap(True)
        v.addWidget(name_hint)

        row = QHBoxLayout()
        row.addWidget(_field_label("SAMPLE COUNT"))
        row.addStretch()
        self.img_spin = QSpinBox()
        self.img_spin.setRange(2, 30)
        self.img_spin.setValue(10)
        self.img_spin.setToolTip(
            "8–12 images per person recommended\n"
            "(angles: straight, left, right + near, medium, far)")
        self.img_spin.setFixedWidth(70)
        row.addWidget(self.img_spin)
        v.addLayout(row)

        hint_lbl = QLabel(
            "Tip: 8–12 images  ·  vary angles (left, straight, right)"
            "  ·  step back for far shots")
        hint_lbl.setStyleSheet(
            f"color: {AMBER}; font-size: 9px; font-family: '{SANS_FONT}';"
            f" background: transparent; border: none;")
        hint_lbl.setWordWrap(True)
        v.addWidget(hint_lbl)

        thumb_container = QFrame()
        thumb_container.setFrameShape(QFrame.NoFrame)
        thumb_container.setFixedHeight(44)
        thumb_container.setStyleSheet(
            f"background: {SURFACE}; border: none; border-radius: 4px;")
        self.thumb_layout = QHBoxLayout(thumb_container)
        self.thumb_layout.setContentsMargins(6, 4, 6, 4)
        self.thumb_layout.setSpacing(4)
        self.thumb_layout.addStretch()
        v.addWidget(thumb_container)

        self.btn_enroll = _btn("◉  CAPTURE FACE", CYAN)
        self.btn_enroll.clicked.connect(self._on_enroll)
        v.addWidget(self.btn_enroll)

        return card

    # ── Step 2 — Train ────────────────────────────────────────────────

    def _build_step2(self) -> QFrame:
        card, v = self._step_card(
            "02", "Build Recognition Engine",
            "Generate face embeddings from enrolled data")

        prog_row = QHBoxLayout()
        self.train_pct_lbl = _mono_label("0%", 11, CYAN)
        prog_row.addWidget(_field_label("TRAINING PROGRESS"))
        prog_row.addStretch()
        prog_row.addWidget(self.train_pct_lbl)
        v.addLayout(prog_row)

        self.train_progress = QProgressBar()
        self.train_progress.setValue(0)
        self.train_progress.setTextVisible(False)
        self.train_progress.setFixedHeight(6)
        v.addWidget(self.train_progress)

        self.train_status_lbl = _mono_label("IDLE  —  awaiting command", 11, TEXT_DIM)
        v.addWidget(self.train_status_lbl)

        self.btn_train = _btn("⬡  BUILD EMBEDDINGS", CYAN)
        self.btn_train.clicked.connect(self._on_train)
        v.addWidget(self.btn_train)

        return card

    # ── Step 3 — Monitor ─────────────────────────────────────────────

    def _build_step3(self) -> QFrame:
        card, v = self._step_card(
            "03", "Launch Live Monitoring",
            "Real-time face recognition on webcam stream")

        thr_row = QHBoxLayout()
        thr_row.addWidget(_field_label("CONFIDENCE THRESHOLD"))
        thr_row.addStretch()
        self.slider_val_lbl = _mono_label(f"{THRESHOLD:.2f}", 11, CYAN)
        thr_row.addWidget(self.slider_val_lbl)
        v.addLayout(thr_row)

        self.strict_slider = QSlider(Qt.Horizontal)
        self.strict_slider.setMinimum(40)
        self.strict_slider.setMaximum(90)
        self.strict_slider.setValue(int(THRESHOLD * 100))
        self.strict_slider.setTickPosition(QSlider.NoTicks)
        self.strict_slider.valueChanged.connect(self._on_slider_changed)
        v.addWidget(self.strict_slider)

        range_row = QHBoxLayout()
        range_row.addWidget(_mono_label("0.40 · LOW", 9, TEXT_DIM))
        range_row.addStretch()
        range_row.addWidget(_mono_label("0.90 · HIGH", 9, TEXT_DIM))
        v.addLayout(range_row)

        vote_row = QHBoxLayout()
        vote_row.addWidget(_field_label("VOTE FRAMES"))
        vote_row.addStretch()
        self.vote_spin = QSpinBox()
        self.vote_spin.setRange(1, 10)
        self.vote_spin.setValue(VOTE_FRAMES)
        self.vote_spin.setFixedWidth(70)
        self.vote_spin.setToolTip(
            "Majority-vote window: how many consecutive frames\n"
            "must agree before showing the recognition result.\n"
            "Increase to 5–7 for stability; reduce to 1–2 for instant response.")
        self.vote_spin.valueChanged.connect(self._on_vote_spin_changed)
        vote_row.addWidget(self.vote_spin)
        v.addLayout(vote_row)

        vote_hint = QLabel(
            "Higher = more stable · Lower = faster response (3–5 recommended)")
        vote_hint.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 9px; font-family: '{SANS_FONT}';"
            f" background: transparent; border: none;")
        vote_hint.setWordWrap(True)
        v.addWidget(vote_hint)

        self.btn_monitor = _btn("▶  START RECOGNITION", GREEN)
        self.btn_monitor.clicked.connect(self._on_monitor)
        v.addWidget(self.btn_monitor)

        self.btn_stop = _btn("■  STOP MONITORING", RED)
        self.btn_stop.clicked.connect(self._on_stop_monitor)
        self.btn_stop.hide()
        v.addWidget(self.btn_stop)

        return card

    # ── Video Panel ───────────────────────────────────────────────────

    def _build_video_panel(self) -> QFrame:
        card = QFrame()
        card.setFrameShape(QFrame.NoFrame)
        card.setStyleSheet(f"""
            QFrame {{
                background: {BG2};
                border: 1px solid {BORDER};
                border-radius: 6px;
            }}
        """)
        _shadow(card, 28)

        v = QVBoxLayout(card)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)

        # Title bar
        bar = QFrame()
        bar.setFrameShape(QFrame.NoFrame)
        bar.setFixedHeight(28)
        bar.setStyleSheet(
            f"background: {SURFACE}; border: none; border-radius: 3px;")
        bar_lay = QHBoxLayout(bar)
        bar_lay.setContentsMargins(10, 0, 10, 0)
        bar_lay.setSpacing(8)
        for col in [GREEN, AMBER, RED]:
            dot = QLabel("●")
            dot.setStyleSheet(
                f"color: {col}; font-size: 9px; background: transparent; border: none;")
            bar_lay.addWidget(dot)
        bar_lay.addSpacing(12)
        bar_lay.addWidget(_mono_label("FEED  /  CAM_0", 10, TEXT_DIM))
        bar_lay.addStretch()
        self.feed_status_lbl = _mono_label("● OFFLINE", 10, TEXT_DIM)
        bar_lay.addWidget(self.feed_status_lbl)
        v.addWidget(bar)

        # ── Video label ───────────────────────────────────────────────
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)

        # QSizePolicy.Ignored → sizeHint reports (0,0) to the layout engine.
        # Video frames paint inside the already-allocated space; they can
        # NEVER force the card or the flanking panels to shift position.
        # The label still visually fills all available space via stretch=1.
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        # Minimum floor prevents collapse when no frame is displayed.
        self.video_label.setMinimumSize(320, 260)

        self.video_label.setStyleSheet(f"""
            background: {BG};
            border: 1px solid {BORDER2};
            border-radius: 3px;
            color: {TEXT_DIM};
            font-family: '{MONO_FONT}';
            font-size: 13px;
        """)
        self.video_label.setText(
            "▷  CAMERA FEED  ◁\n\nInitialize a workflow step to begin stream")

        # Scanline overlay — geometry synced in resizeEvent
        self._scanline = ScanLineWidget(self.video_label)
        self._scanline.setGeometry(self.video_label.rect())

        # stretch=1 gives the label all remaining vertical space without
        # propagating its own preferred size back to the layout.
        v.addWidget(self.video_label, stretch=1)

        return card

    # ── resizeEvent — keep scanline overlay in sync ───────────────────

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, "_scanline") and hasattr(self, "video_label"):
            self._scanline.setGeometry(self.video_label.rect())

    # ── Enrolled Identities Panel ─────────────────────────────────────

    def _build_users_panel(self) -> QFrame:
        card = QFrame()
        card.setFrameShape(QFrame.NoFrame)
        card.setStyleSheet(f"""
            QFrame {{
                background: {PANEL};
                border: none;
                border-radius: 6px;
            }}
        """)
        _shadow(card, 14)

        v = QVBoxLayout(card)
        v.setContentsMargins(16, 14, 14, 14)
        v.setSpacing(10)

        hdr = QHBoxLayout()
        hdr.addWidget(_section_title("ENROLLED IDENTITIES"))
        hdr.addStretch()
        refresh_btn = QPushButton("⟳")
        refresh_btn.setStyleSheet(
            f"background: transparent; color: {TEXT_DIM}; font-size: 16px;"
            f" border: none; padding: 0;")
        refresh_btn.setFixedSize(22, 22)
        refresh_btn.setCursor(Qt.PointingHandCursor)
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
        self.users_layout.setSpacing(6)
        scroll.setWidget(content)
        v.addWidget(scroll)
        return card

    # ── Latest Detection Panel ────────────────────────────────────────

    def _build_status_panel(self) -> QFrame:
        card = QFrame()
        card.setFrameShape(QFrame.NoFrame)
        card.setStyleSheet(f"""
            QFrame {{
                background: {PANEL};
                border: none;
                border-radius: 6px;
            }}
        """)
        card.setFixedHeight(110)
        _shadow(card)

        v = QVBoxLayout(card)
        v.setContentsMargins(16, 12, 14, 12)
        v.setSpacing(8)
        v.addWidget(_section_title("LATEST DETECTION"))
        v.addWidget(_divider())

        self.auth_label = QLabel("—  AWAITING STREAM")
        self.auth_label.setAlignment(Qt.AlignCenter)
        self.auth_label.setStyleSheet(f"""
            color: {TEXT_DIM}; font-size: 12px; font-family: '{MONO_FONT}';
            font-weight: 700; background: transparent; border: none;
            padding: 6px 0; letter-spacing: 1px;
        """)
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
            if item.widget():
                item.widget().deleteLater()

        self.btn_enroll.setEnabled(False)
        self.btn_enroll.setText("● CAPTURING...")
        self.feed_status_lbl.setText("● ENROLL")
        self.feed_status_lbl.setStyleSheet(
            f"color: {CYAN}; font-size: 10px; font-family: '{MONO_FONT}';"
            f" background: transparent; border: none;")

        self.enroll_worker = EnrollWorker(name, cnt)
        self.enroll_worker.frame_ready.connect(self._display_frame)
        self.enroll_worker.face_captured.connect(self._on_face_captured)
        self.enroll_worker.finished_signal.connect(self._on_enroll_finished)
        self.enroll_worker.error.connect(self._on_worker_error)
        self.enroll_worker.start()

    @pyqtSlot(np.ndarray, int)
    def _on_face_captured(self, face_rgb: np.ndarray, count: int):
        face_resized = cv2.resize(face_rgb, (32, 32))
        qi  = QImage(face_resized.data, 32, 32, 3 * 32, QImage.Format_RGB888)
        lbl = QLabel()
        lbl.setFixedSize(32, 32)
        lbl.setStyleSheet(f"border: 1px solid {CYAN}; border-radius: 2px;")
        lbl.setPixmap(QPixmap.fromImage(qi))
        self.thumb_layout.insertWidget(self.thumb_layout.count() - 1, lbl)

    @pyqtSlot(int)
    def _on_enroll_finished(self, captured: int):
        self.btn_enroll.setEnabled(True)
        self.btn_enroll.setText("◉  CAPTURE FACE")
        self.name_input.clear()
        self.feed_status_lbl.setText("● OFFLINE")
        self.feed_status_lbl.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 10px; font-family: '{MONO_FONT}';"
            f" background: transparent; border: none;")
        QMessageBox.information(self, "Enrollment Complete",
                                f"Successfully enrolled {captured} face samples.")
        self._refresh_users()
        self.video_label.setText(
            "▷  CAMERA FEED  ◁\n\nInitialize a workflow step to begin stream")

    @pyqtSlot()
    def _on_train(self):
        self.btn_train.setEnabled(False)
        self.btn_train.setText("⬡  PROCESSING...")
        self.train_progress.setValue(0)
        self.train_pct_lbl.setText("0%")
        self.train_status_lbl.setText("EXTRACTING FEATURES...")

        self.train_worker = TrainWorker()
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
        self.train_status_lbl.setStyleSheet(
            f"color: {GREEN}; font-size: 11px; font-family: '{MONO_FONT}';"
            f" background: transparent; border: none;")
        self._set_indicator(self.ind_embeddings, True)
        self._refresh_users()
        QMessageBox.information(self, "Training Complete", msg)

    @pyqtSlot()
    def _on_monitor(self):
        if not Path(EMBEDDINGS_DIR, "prototypes.npy").exists():
            QMessageBox.warning(self, "Not Trained",
                                "Run Step 2 to build the recognition engine first.")
            return
        self.btn_monitor.hide()
        self.btn_stop.show()
        self.btn_enroll.setEnabled(False)
        self.btn_train.setEnabled(False)
        self.feed_status_lbl.setText("● LIVE")
        self.feed_status_lbl.setStyleSheet(
            f"color: {GREEN}; font-size: 10px; font-family: '{MONO_FONT}';"
            f" background: transparent; border: none;")

        self.recog_worker = RecognitionWorker()
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
        self.btn_stop.hide()
        self.btn_monitor.show()
        self.btn_enroll.setEnabled(True)
        self.btn_train.setEnabled(True)
        self.video_label.clear()
        self.video_label.setText("▷  CAMERA FEED  ◁\n\nMonitoring stopped")
        self.feed_status_lbl.setText("● OFFLINE")
        self.feed_status_lbl.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 10px; font-family: '{MONO_FONT}';"
            f" background: transparent; border: none;")
        self.auth_label.setText("—  STREAM TERMINATED")
        self.auth_label.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 12px; font-family: '{MONO_FONT}';"
            f" font-weight: 700; background: transparent; border: none;"
            f" padding: 6px 0; letter-spacing: 1px;")

    @pyqtSlot(str, float, str)
    def _on_person_detected(self, name: str, conf: float, ts: str):
        if name != "Unknown":
            self.auth_label.setText(
                f"✓  AUTHORIZED  —  {name}  [{conf*100:.0f}%]")
            self.auth_label.setStyleSheet(
                f"color: {GREEN}; font-size: 12px; font-family: '{MONO_FONT}';"
                f" font-weight: 700; background: transparent; border: none;"
                f" padding: 6px 0; letter-spacing: 1px;")
        else:
            self.auth_label.setText(
                f"✗  UNKNOWN SUBJECT  —  {conf*100:.0f}%  MATCH")
            self.auth_label.setStyleSheet(
                f"color: {RED}; font-size: 12px; font-family: '{MONO_FONT}';"
                f" font-weight: 700; background: transparent; border: none;"
                f" padding: 6px 0; letter-spacing: 1px;")

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
        qi = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pm = QPixmap.fromImage(qi)
        # Scale to current label size. Because the label uses
        # QSizePolicy.Ignored, this scaled size is never fed back into
        # the layout as a new size hint — no shifting, ever.
        scaled = pm.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
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
            if item.widget():
                item.widget().deleteLater()

        folders = get_identity_folders(DATASET_DIR)
        emb_ok  = Path(EMBEDDINGS_DIR, "prototypes.npy").exists()

        if not folders:
            ph = QLabel("NO IDENTITIES ENROLLED")
            ph.setStyleSheet(
                f"color: {TEXT_DIM}; font-size: 11px; font-family: '{MONO_FONT}';"
                f" padding: 16px; border: none;")
            ph.setAlignment(Qt.AlignCenter)
            self.users_layout.addWidget(ph)
        else:
            for folder in folders:
                imgs = (list(folder.glob("*.jpg")) +
                        list(folder.glob("*.png")) +
                        list(folder.glob("*.jpeg")))
                self.users_layout.addWidget(
                    self._user_card(folder.name, len(imgs), imgs, emb_ok))

        self.users_layout.addStretch()

    def _user_card(self, name: str, count: int,
                   imgs: list, trained: bool) -> QFrame:
        card = QFrame()
        card.setFrameShape(QFrame.NoFrame)
        card.setStyleSheet(f"""
            QFrame {{
                background: {SURFACE};
                border: none;
                border-radius: 4px;
            }}
            QFrame:hover {{ background: {SURFACE2}; }}
        """)
        h = QHBoxLayout(card)
        h.setContentsMargins(10, 8, 10, 8)
        h.setSpacing(10)

        av = QLabel()
        av.setFixedSize(36, 36)
        av.setAlignment(Qt.AlignCenter)
        if imgs:
            img = cv2.imread(str(imgs[0]))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_rgb = cv2.resize(img_rgb, (36, 36))
                qi = QImage(img_rgb.data, 36, 36, 3 * 36, QImage.Format_RGB888)
                av.setPixmap(QPixmap.fromImage(qi).scaled(
                    36, 36, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                av.setStyleSheet(
                    f"border: 1px solid {CYAN}; border-radius: 2px;")
            else:
                av.setText(name[0].upper())
                av.setStyleSheet(
                    f"background: {CYAN_DIM}; color: {CYAN}; font-size: 15px;"
                    f" font-weight: 700; font-family: '{MONO_FONT}';"
                    f" border: 1px solid {CYAN}; border-radius: 2px;")
        else:
            av.setText(name[0].upper())
            av.setStyleSheet(
                f"background: {CYAN_DIM}; color: {CYAN}; font-size: 15px;"
                f" font-weight: 700; font-family: '{MONO_FONT}';"
                f" border: 1px solid {CYAN}; border-radius: 2px;")
        h.addWidget(av)

        info = QVBoxLayout()
        info.setSpacing(2)
        n_lbl = QLabel(name)
        n_lbl.setStyleSheet(
            f"font-size: 12px; font-weight: 700; color: {TEXT};"
            f" font-family: '{SANS_FONT}'; background: transparent; border: none;")
        c_lbl = QLabel(f"{count} samples")
        c_lbl.setStyleSheet(
            f"font-size: 10px; color: {TEXT_DIM}; font-family: '{MONO_FONT}';"
            f" background: transparent; border: none;")
        info.addWidget(n_lbl)
        info.addWidget(c_lbl)
        h.addLayout(info)
        h.addStretch()

        badge = QLabel("READY" if trained else "UNTRAINED")
        badge.setStyleSheet(
            (f"color: {GREEN}; background: {GREEN_DIM};"
             f" border: 1px solid rgba(0,255,136,0.2); border-radius: 3px;"
             f" font-size: 9px; font-weight: 700; font-family: '{MONO_FONT}';"
             f" padding: 3px 8px; letter-spacing: 1px;")
            if trained else
            (f"color: {AMBER}; background: {AMBER_DIM};"
             f" border: 1px solid rgba(255,170,0,0.2); border-radius: 3px;"
             f" font-size: 9px; font-weight: 700; font-family: '{MONO_FONT}';"
             f" padding: 3px 8px; letter-spacing: 1px;"))
        h.addWidget(badge)
        return card

    # ── Cleanup ───────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self.enroll_worker:
            self.enroll_worker.stop()
            self.enroll_worker.wait(2000)
        if self.recog_worker:
            self.recog_worker.stop()
            self.recog_worker.wait(2000)
        event.accept()


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
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()