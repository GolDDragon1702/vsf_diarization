#!/usr/bin/env bash
# ============================================================================
# Chạy benchmark Nemotron-3.5-ASR-Streaming-0.6b trên WSL2 / Linux.
#
# VÌ SAO KHÔNG CHẠY ĐƯỢC TRÊN WINDOWS NATIVE:
#   NeMo transcribe() cho model RNNT-prompt này lỗi trên Windows (numpy decode
#   rỗng; file input dính khóa manifest tạm WinError 32). Trên Linux/WSL thì OK.
#
# CÁCH DÙNG (bên trong WSL Ubuntu, đứng ở thư mục project):
#   cd /mnt/c/Users/longph31/Documents/diarization
#   bash run_nemotron.sh                # chạy toàn bộ 11 file reviewed
#   bash run_nemotron.sh test/test01.wav   # chạy 1 file
#
# Script tự cài 1 lần (torch cu128 + NeMo git@main + deps) vào venv ở $HOME,
# lần chạy sau sẽ bỏ qua bước cài. Kết quả lưu outputs/asr_nemotron.json
# ============================================================================
set -euo pipefail

PROJ="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ"
VENV="$HOME/.venvs/nemo_diar"          # đặt ở Linux fs cho nhanh (không để trên /mnt/c)
PY="$VENV/bin/python"
ARGS=("$@"); [ ${#ARGS[@]} -eq 0 ] && ARGS=("test/")   # mặc định: cả thư mục test/

# ── 1. Kiểm tra Python 3.10–3.12 ────────────────────────────────────────────
PYBIN=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    v=$("$c" -c 'import sys;print("%d.%d"%sys.version_info[:2])')
    case "$v" in 3.10|3.11|3.12) PYBIN="$c"; break;; esac
  fi
done
if [ -z "$PYBIN" ]; then
  echo "‼  Cần Python 3.10–3.12. Cài: sudo apt update && sudo apt install -y python3.12-venv"
  exit 1
fi
echo "== Dùng $PYBIN ($("$PYBIN" --version)) =="

# ── 2. Tạo venv + cài (chỉ lần đầu) ─────────────────────────────────────────
NEED_INSTALL=1
if [ -x "$PY" ] && "$PY" -c "import nemo.collections.asr.models.rnnt_bpe_models_prompt" 2>/dev/null; then
  NEED_INSTALL=0
  echo "== venv đã sẵn sàng, bỏ qua cài đặt =="
fi

if [ "$NEED_INSTALL" -eq 1 ]; then
  echo "== [1/4] system deps (ffmpeg, libsndfile) =="
  sudo apt-get update -y && sudo apt-get install -y libsndfile1 ffmpeg build-essential

  echo "== [2/4] tạo venv =="
  mkdir -p "$(dirname "$VENV")"
  "$PYBIN" -m venv "$VENV"
  "$PY" -m pip install -U pip wheel Cython packaging

  echo "== [3/4] cài NeMo (git@main) + deps  (~10-20 phút lần đầu) =="
  "$PY" -m pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main"
  # NeMo hay kéo torch CPU về — ép lại bản CUDA 12.8 cho khớp GPU:
  "$PY" -m pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128
  "$PY" -m pip install soundfile jiwer librosa python-dotenv

  echo "== [4/4] kiểm tra import =="
  "$PY" -c "import torch, nemo.collections.asr as a; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('NeMo OK')"
fi

# ── 3. Chạy benchmark ───────────────────────────────────────────────────────
echo "== Chạy Nemotron trên: ${ARGS[*]} =="
"$PY" eval/eval_asr_models.py "${ARGS[@]}" --model nemotron --language vi \
  --output outputs/asr_nemotron.json

echo ""
echo "✓ Xong → outputs/asr_nemotron.json"
echo "  So sánh với asr_whisper.json / asr_qwen.json (xem REPORT.md mục 5.2)."
