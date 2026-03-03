"""Web API orchestration module for few-shot face recognition system.

Purpose:
    Exposes HTTP endpoints for enrollment, embedding generation, and real-time
    recognition control, while also serving the dashboard interface.

Role in pipeline:
    Acts as integration layer between user interface and backend ML modules
    (`register.py`, `generate_embeddings.py`, `recognize.py`).

Few-shot contribution:
    Provides a controlled workflow API so users can execute few-shot stages
    sequentially: face capture -> feature extraction -> recognition.

Note:
    The current implementation uses FastAPI routes (conceptually equivalent to
    Flask routes for request/response orchestration in this project context).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict

# Ensure project root is on sys.path for local imports
_PROJECT_DIR = str(Path(__file__).resolve().parent)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from utils import get_identity_folders


# Request schema for "Register New User" action from dashboard.
class RegisterRequest(BaseModel):
    """Schema for registration request payload.

    Purpose:
        Validates enrollment input before invoking `register.py`.
    """

    name: str = Field(..., min_length=1, description="Identity name")
    num_images: int = Field(5, ge=2, le=30, description="Number of images to capture")
    camera_id: int = Field(0, ge=0, description="Webcam device id")


# Request schema for "Generate Embeddings" action.
class EmbeddingRequest(BaseModel):
    """Schema for embedding update request payload.

    Purpose:
        Validates backend/model options before invoking `generate_embeddings.py`.
    """

    backend: str = Field("auto", pattern="^(auto|facenet|onnx|insightface)$")
    onnx_model_path: str = Field("models/arcface.onnx")


# Request schema for "Start Recognition" action.
class RecognitionRequest(BaseModel):
    """Schema for recognition start request payload.

    Purpose:
        Validates real-time inference configuration before launching
        `recognize.py`.
    """

    metric: str = Field("cosine", pattern="^(cosine|euclidean)$")
    threshold: float = Field(0.60, description="Similarity threshold")
    backend: str = Field("auto", pattern="^(auto|facenet|onnx|insightface)$")
    onnx_model_path: str = Field("models/arcface.onnx")
    camera_id: int = Field(0, ge=0)


# Main API application object.
app = FastAPI(
    title="FewShotFace API",
    description="API to register users, update embeddings, and start real-time recognition",
    version="1.0.0",
)

# Serve static files and templates
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _run_script(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Execute a backend script and capture terminal output.

    Function name:
        _run_script

    Purpose:
        Runs a child Python process synchronously and collects stdout/stderr
        for API-level error handling.

    Parameters:
        command (list[str]): Command tokens to execute.

    Returns:
        subprocess.CompletedProcess[str]: Process result object with exit code
        and captured output.

    Role in face recognition process:
        Provides process-level integration between API routes and ML modules.
    """
    return subprocess.run(command, check=False, capture_output=True, text=True)


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    """Serve web dashboard template.

    Function name:
        dashboard

    Purpose:
        Handles GET request for root URL and renders operator UI.

    Parameters:
        request (Request): Incoming HTTP request object.

    Returns:
        HTMLResponse: Rendered dashboard page.

    Role in face recognition process:
        Entry point for users to trigger enrollment/training/recognition stages.
    """
    # Route: GET /  -> dashboard interface
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health() -> Dict[str, str]:
    """Provide lightweight server health status.

    Function name:
        health

    Purpose:
        Confirms API service availability for frontend status checks.

    Parameters:
        None

    Returns:
        Dict[str, str]: Service status information.

    Role in face recognition process:
        Ensures orchestration layer is reachable before pipeline operations.
    """
    # Route: GET /health  -> used by UI heartbeat check
    return {"status": "ok", "message": "FewShotFace backend is running"}


@app.get("/users")
def list_users() -> Dict[str, list[str]]:
    """Return list of enrolled identity names.

    Function name:
        list_users

    Purpose:
        Reads identity folder names from dataset storage.

    Parameters:
        None

    Returns:
        Dict[str, list[str]]: Names of enrolled users.

    Role in face recognition process:
        Exposes enrollment state to UI and API clients.
    """
    # Route: GET /users  -> compact identity list
    users = [folder.name for folder in get_identity_folders("dataset")]
    return {"users": users}


@app.get("/api/users/details")
def list_users_details() -> Dict[str, list]:
    """Return enrolled users with sample-image counts.

    Function name:
        list_users_details

    Purpose:
        Aggregates dataset statistics by identity for dashboard display.

    Parameters:
        None

    Returns:
        Dict[str, list]: Structured list containing user names and image counts.

    Role in face recognition process:
        Provides visibility into enrollment completeness before training.
    """
    # Route: GET /api/users/details  -> detailed enrollment summary
    folders = get_identity_folders("dataset")
    users = []
    for folder in folders:
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
        users.append({"name": folder.name, "images": len(images)})
    return {"users": users}


@app.post("/register")
def register_user(payload: RegisterRequest) -> Dict[str, str]:
    """Handle registration request and launch enrollment module.

    Function name:
        register_user

    Purpose:
        Parses request payload, builds command, and invokes `register.py` to
        collect face samples for a new identity.

    Parameters:
        payload (RegisterRequest): Validated enrollment request body.

    Returns:
        Dict[str, str]: Success status and user-facing message.

    Role in face recognition process:
        API gateway for face capture/enrollment stage.
    """
    # Route: POST /register  -> starts enrollment pipeline
    command = [
        sys.executable,
        "register.py",
        "--name",
        payload.name,
        "--num-images",
        str(payload.num_images),
        "--camera-id",
        str(payload.camera_id),
    ]

    # Request handling note:
    # create a new console on Windows so webcam/OpenCV UI works reliably.
    creation_flags = 0
    if sys.platform.startswith("win"):
        creation_flags = subprocess.CREATE_NEW_CONSOLE

    try:
        # Backend integration: execute enrollment script as child process.
        proc = subprocess.Popen(command, creationflags=creation_flags)
        proc.wait()  # Wait for registration to complete
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Registration failed: {exc}",
        ) from exc

    return {"status": "success", "message": f"User '{payload.name}' registered"}


@app.post("/embeddings/update")
def update_embeddings(payload: EmbeddingRequest) -> Dict[str, str]:
    """Handle embedding generation request.

    Function name:
        update_embeddings

    Purpose:
        Invokes `generate_embeddings.py` to extract feature vectors and store
        prototype artifacts from enrolled images.

    Parameters:
        payload (EmbeddingRequest): Validated backend/model configuration.

    Returns:
        Dict[str, str]: Operation status and result message.

    Role in face recognition process:
        API gateway for feature extraction + embedding storage stage.
    """
    # Route: POST /embeddings/update  -> triggers feature extraction workflow
    command = [
        sys.executable,
        "generate_embeddings.py",
        "--backend",
        payload.backend,
        "--onnx-model-path",
        payload.onnx_model_path,
    ]

    # Request handling: run script synchronously and inspect exit status.
    result = _run_script(command)
    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding generation failed: {result.stderr.strip() or result.stdout.strip()}",
        )

    return {"status": "success", "message": "Embeddings updated successfully"}


@app.post("/recognition/start")
def start_recognition(payload: RecognitionRequest) -> Dict[str, str]:
    """Handle recognition start request and launch runtime inference module.

    Function name:
        start_recognition

    Purpose:
        Validates request settings and starts `recognize.py` as detached process
        so API remains responsive.

    Parameters:
        payload (RecognitionRequest): Validated runtime recognition settings.

    Returns:
        Dict[str, str]: Start status and informational message.

    Role in face recognition process:
        API gateway for real-time similarity matching and decision stage.
    """
    # Route: POST /recognition/start  -> launches real-time recognition process
    command = [
        sys.executable,
        "recognize.py",
        "--metric",
        payload.metric,
        "--threshold",
        str(payload.threshold),
        "--backend",
        payload.backend,
        "--onnx-model-path",
        payload.onnx_model_path,
        "--camera-id",
        str(payload.camera_id),
    ]

    # Start detached so request returns without blocking web server thread.
    creation_flags = 0
    if sys.platform.startswith("win"):
        creation_flags = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]

    try:
        # Backend integration: spawn independent recognition runtime.
        subprocess.Popen(command, creationflags=creation_flags)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Could not start recognition: {exc}") from exc

    return {"status": "success", "message": "Recognition process started"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
