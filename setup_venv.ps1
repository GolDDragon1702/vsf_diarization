# Setup virtual environment for Speaker Diarization pipeline
# Run: .\setup_venv.ps1
# Optional flags:
#   .\setup_venv.ps1 -CUDA      # install GPU (CUDA 12.8) version of PyTorch
#   .\setup_venv.ps1 -Force     # delete existing venv and recreate

param(
    [switch]$CUDA,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$VenvDir = "venv"

# ── 1. Remove existing venv if -Force ─────────────────────────────────────────
if ($Force -and (Test-Path $VenvDir)) {
    Write-Host "Removing existing venv ..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $VenvDir
}

# ── 2. Create venv ────────────────────────────────────────────────────────────
if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating virtual environment ..." -ForegroundColor Cyan
    python -m venv $VenvDir
} else {
    Write-Host "Virtual environment already exists. Skipping creation." -ForegroundColor Gray
}

# ── 3. Activate ───────────────────────────────────────────────────────────────
$Activate = ".\$VenvDir\Scripts\Activate.ps1"
if (-not (Test-Path $Activate)) {
    Write-Error "venv creation failed — $Activate not found."
    exit 1
}
& $Activate

# ── 4. Upgrade pip ────────────────────────────────────────────────────────────
Write-Host "`nUpgrading pip ..." -ForegroundColor Cyan
python -m pip install --upgrade pip --quiet

# ── 5. Install PyTorch ────────────────────────────────────────────────────────
if ($CUDA) {
    Write-Host "`nInstalling PyTorch with CUDA 12.8 support ..." -ForegroundColor Cyan
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 --quiet
} else {
    Write-Host "`nInstalling PyTorch (CPU) ..." -ForegroundColor Cyan
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
}

# ── 6. Install package (editable) + entry points vsf-* ───────────────────────
Write-Host "`nInstalling vsf-diarization package (editable) ..." -ForegroundColor Cyan
pip install -e . --quiet          # thêm '.[qwen]' nếu cần Qwen3-ASR

# ── 7. Done ───────────────────────────────────────────────────────────────────
Write-Host "`n✓ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Activate venv    : .\venv\Scripts\Activate.ps1"
Write-Host "  2. Set HF token     : copy .env.example .env  (then edit .env)"
Write-Host "  3. Offline (ASR)    : vsf-diarize video.mp4 --asr whisper --language vi"
Write-Host "  4. Offline (diar)   : vsf-diarize video.mp4 --asr none --num-speakers 2"
Write-Host "  5. Streaming        : vsf-stream --source video.mp4 --language vi"
Write-Host ""
Write-Host "Verify GPU availability:" -ForegroundColor Gray
Write-Host "  python -c ""import torch; print(torch.cuda.is_available())"""
