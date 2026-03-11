"""download_models.py — Download and register custom ONNX embedding models.

Usage examples:
    # List all models currently in models/
    python download_models.py --list

    # Download a model from any URL
    python download_models.py --url https://example.com/mymodel.onnx --name mymodel.onnx

    # Download a model and verify it loads correctly with ONNXEmbedder
    python download_models.py --url https://example.com/mymodel.onnx --name mymodel.onnx --verify

    # Show well-known InsightFace model download sources
    python download_models.py --sources

Known working ONNX models (already included in this project):
    models/w600k_r50.onnx   WebFace600K + ResNet-50 ArcFace  (512-d, 166 MB) — default

Less common / alternative models (must be downloaded fresh — NOT the repo copy):
    models/arcface.onnx     Glint360K + ResNet-100 ArcFace   (512-d, 299 MB) — high accuracy
                            Note: the arcface.onnx committed in this repo is corrupt; download
                            a fresh copy if you need this model (see --sources).

To get additional InsightFace ONNX models:
    pip install insightface
    python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l').prepare(ctx_id=0)"
    # The .onnx files will be cached in ~/.insightface/models/buffalo_l/
    # Copy them to this project's models/ directory.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

# ── Project root on path ─────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"

# ── Well-known model registry ────────────────────────────────────────
# Friendly-name → (relative path inside models/, description, min size bytes)
KNOWN_MODELS: dict[str, tuple[str, str, int]] = {
    "w600k_r50":  ("w600k_r50.onnx",  "WebFace600K ResNet-50 ArcFace  (512-d, ~166 MB)",  150_000_000),
    "arcface":    ("arcface.onnx",     "Glint360K ResNet-100 ArcFace   (512-d, ~299 MB)",  280_000_000),
    "det_10g":    ("det_10g.onnx",     "InsightFace SCRFD face detector (~16 MB)",           10_000_000),
    "genderage":  ("genderage.onnx",   "InsightFace age/gender model   (~1.2 MB)",              500_000),
    "1k3d68":     ("1k3d68.onnx",      "InsightFace 3D landmark model  (~137 MB)",          100_000_000),
    "2d106det":   ("2d106det.onnx",    "InsightFace 2D landmark model  (~4.8 MB)",            3_000_000),
}


# ── Helpers ──────────────────────────────────────────────────────────

def _format_bytes(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.1f} KB"
    return f"{n} B"


def list_models() -> None:
    """Print all .onnx files currently present in models/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    onnx_files = sorted(MODELS_DIR.glob("*.onnx"))
    if not onnx_files:
        print("  No .onnx files found in models/")
        return

    print(f"\n  {'File':<25}  {'Size':>10}  Status")
    print("  " + "─" * 60)
    for f in onnx_files:
        size = f.stat().st_size
        # Match against known registry
        status = "✓ ready"
        for name, (fname, desc, min_size) in KNOWN_MODELS.items():
            if f.name == fname:
                if size < min_size:
                    status = "⚠ incomplete / corrupted"
                else:
                    status = f"✓ {name}"
                break
        print(f"  {f.name:<25}  {_format_bytes(size):>10}  {status}")
    print()


def show_sources() -> None:
    """Print known sources for downloading InsightFace ONNX models."""
    print("""
  ── Known ONNX Model Sources ─────────────────────────────────────────────
  
  InsightFace buffalo_l pack (includes w600k_r50.onnx + detector models):
    Install via Python:
      pip install insightface
      python -c "from insightface.app import FaceAnalysis; \\
                 FaceAnalysis(name='buffalo_l').prepare(ctx_id=0)"
    Models cached at:  ~/.insightface/models/buffalo_l/

  InsightFace buffalo_sc (MobileNet, smaller/faster):
      python -c "from insightface.app import FaceAnalysis; \\
                 FaceAnalysis(name='buffalo_sc').prepare(ctx_id=0)"
    Models cached at:  ~/.insightface/models/buffalo_sc/

  Hugging Face (deepinsight/insightface):
    Browse at: https://huggingface.co/deepinsight/insightface
    Models are under the 'models/' directory of that repository.

  After downloading, copy the .onnx file to this project's models/ directory
  and run:
    python download_models.py --list
  to confirm it is registered, then use:
    python evaluate_accuracy.py --backend onnx --onnx-model models/yourmodel.onnx
    """)


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar_w = 30
        filled = int(pct / 100 * bar_w)
        bar = "█" * filled + "░" * (bar_w - filled)
        print(
            f"\r  [{bar}] {pct:5.1f}%  "
            f"{_format_bytes(downloaded)} / {_format_bytes(total_size)}",
            end="",
            flush=True,
        )
    else:
        print(f"\r  Downloaded {_format_bytes(downloaded)}", end="", flush=True)


def download_model(url: str, name: str, verify: bool = False) -> Path:
    """Download a model from ``url`` and save it as ``models/<name>``.

    Args:
        url:    HTTP/HTTPS URL to the .onnx file.
        name:   Output filename (saved under models/).
        verify: If True, attempt to load the model with onnxruntime after download.

    Returns:
        Path to the saved model file.

    Raises:
        SystemExit: On download failure or verification failure.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / name

    if dest.exists():
        print(f"  File already exists: {dest}  ({_format_bytes(dest.stat().st_size)})")
        ans = input("  Overwrite? [y/N] ").strip().lower()
        if ans != "y":
            print("  Aborted.")
            return dest

    print(f"\n  Downloading: {url}")
    print(f"  Destination: {dest}\n")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
    except Exception as exc:
        print(f"\n  ✗ Download failed: {exc}")
        if dest.exists():
            dest.unlink()
        sys.exit(1)
    print(f"\n  ✓ Saved  {dest}  ({_format_bytes(dest.stat().st_size)})")

    # Basic sanity check: ONNX files start with the protobuf magic bytes.
    try:
        header = dest.read_bytes()[:8]
        if not header:
            raise ValueError("File is empty.")
    except Exception as exc:
        print(f"  ✗ File read error: {exc}")
        sys.exit(1)

    if verify:
        _verify_onnx(dest)

    return dest


def _verify_onnx(model_path: Path) -> None:
    """Try loading the model with onnxruntime and printing its input shape."""
    print(f"\n  Verifying with onnxruntime: {model_path.name} ...", end="  ")
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inp = session.get_inputs()[0]
        print(f"✓  input='{inp.name}'  shape={inp.shape}  dtype={inp.type}")
    except ImportError:
        print("⚠ onnxruntime not installed — skipping runtime verification.")
    except Exception as exc:
        print(f"\n  ✗ Model failed to load: {exc}")
        sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and manage ONNX models for FewShotFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all .onnx files currently in models/.",
    )
    parser.add_argument(
        "--sources",
        action="store_true",
        help="Show known sources for downloading InsightFace ONNX models.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL to download a model from.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            "Output filename for the downloaded model (saved in models/)."
            " Defaults to the last path segment of --url."
        ),
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After downloading, verify the model loads correctly with onnxruntime.",
    )
    return parser.parse_args()


def main() -> None:
    import os
    os.chdir(str(PROJECT_DIR))

    args = parse_args()

    if args.list:
        list_models()
        return

    if args.sources:
        show_sources()
        return

    if args.url:
        name = args.name or Path(args.url.split("?")[0]).name
        if not name.endswith(".onnx"):
            print(f"  ⚠ Warning: '{name}' does not end in .onnx — proceed anyway? [y/N] ", end="")
            if input().strip().lower() != "y":
                sys.exit(0)
        download_model(args.url, name, verify=args.verify)
        print()
        list_models()
        return

    # No action specified — show help.
    print("\n  No action specified. Use --help for usage, --list to see current models,")
    print("  or --sources for download links.\n")
    list_models()


if __name__ == "__main__":
    main()
