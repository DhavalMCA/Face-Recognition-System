param(
    [string]$Backend = "deepface",
    [string]$DeepFaceModel = "ArcFace",
    [string]$InsightFaceModel = "buffalo_l",
    [string]$OnnxModelPath = "models/w600k_r50.onnx"
)

Write-Host "=== FewShotFace Client Check ===" -ForegroundColor Cyan
Write-Host "Working dir: $PWD"
Write-Host ""

Write-Host "[Step 1] Python and pip info" -ForegroundColor Yellow
python --version
python -m pip --version
Write-Host ""

Write-Host "[Step 2] Upgrade packaging tools" -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel
Write-Host ""

Write-Host "[Step 3] Install project requirements" -ForegroundColor Yellow
python -m pip install -r requirements.txt
Write-Host ""

Write-Host "[Step 4] Run backend healthcheck" -ForegroundColor Yellow
python .\client_backend_healthcheck.py --backend $Backend --deepface-model $DeepFaceModel --insightface-model $InsightFaceModel --onnx-model-path $OnnxModelPath
Write-Host ""

Write-Host "Done. If any FAIL lines appear, send that full output." -ForegroundColor Green
