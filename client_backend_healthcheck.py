from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path


def check_import(module_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        return True, str(version)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def print_header() -> None:
    print("=" * 72)
    print("FewShotFace Client Health Check")
    print("=" * 72)
    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {platform.python_version()}")
    print(f"Platform          : {platform.platform()}")
    print()


def print_import_checks() -> bool:
    print("[1/3] Dependency import checks")
    modules = [
        "numpy",
        "cv2",
        "torch",
        "tensorflow",
        "deepface",
        "onnxruntime",
        "insightface",
        "facenet_pytorch",
        "sklearn",
    ]
    all_ok = True
    for name in modules:
        ok, msg = check_import(name)
        tag = "OK" if ok else "FAIL"
        print(f"  - {name:<15} {tag:<4} {msg}")
        all_ok = all_ok and ok
    print()
    return all_ok


def print_embedder_resolution(backend: str, deepface_model: str, insightface_model: str, onnx_model_path: str) -> None:
    print("[2/3] FaceEmbedder backend resolution")
    try:
        from utils import FaceEmbedder

        embedder = FaceEmbedder(
            backend=backend,
            deepface_model=deepface_model,
            insightface_model=insightface_model,
            onnx_model_path=onnx_model_path,
        )
        print(f"  Requested backend : {backend}")
        print(f"  Resolved backend  : {embedder.backend_name}")
        if backend != "auto" and not str(embedder.backend_name).startswith(backend):
            print("  WARNING: Requested backend resolved to a different backend.")
            print("           Client should fix dependencies/model files before training/evaluation.")
    except Exception as exc:
        print(f"  FAIL: Could not initialize FaceEmbedder: {type(exc).__name__}: {exc}")
    print()


def print_backend_metadata(embeddings_dir: str) -> None:
    print("[3/3] Trained backend metadata check")
    path = Path(embeddings_dir) / "backend.json"
    if not path.exists():
        print(f"  INFO: {path} not found. Train once to generate backend metadata.")
        print()
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"  backend.json path      : {path}")
        print(f"  requested_backend      : {data.get('requested_backend')}")
        print(f"  requested_deepface     : {data.get('requested_deepface_model')}")
        print(f"  resolved_backend       : {data.get('backend')}")
        print(f"  resolved_backend_name  : {data.get('resolved_backend_name')}")
        print(f"  insightface_model      : {data.get('insightface_model')}")
        print(f"  onnx_model_path        : {data.get('onnx_model_path')}")
    except Exception as exc:
        print(f"  FAIL: Could not read backend.json: {type(exc).__name__}: {exc}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FewShotFace client-side environment and backend health check")
    parser.add_argument("--backend", default="deepface", choices=["auto", "facenet", "onnx", "insightface", "deepface", "vit"])
    parser.add_argument("--deepface-model", default="ArcFace")
    parser.add_argument("--insightface-model", default="buffalo_l")
    parser.add_argument("--onnx-model-path", default="models/w600k_r50.onnx")
    parser.add_argument("--embeddings-dir", default="embeddings")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print_header()
    imports_ok = print_import_checks()
    print_embedder_resolution(
        backend=args.backend,
        deepface_model=args.deepface_model,
        insightface_model=args.insightface_model,
        onnx_model_path=args.onnx_model_path,
    )
    print_backend_metadata(args.embeddings_dir)

    if imports_ok:
        print("Result: Imports look healthy.")
    else:
        print("Result: One or more imports failed. Fix failing modules before use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
