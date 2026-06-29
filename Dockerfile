# Vietnamese Speech Diarization & ASR — GPU image (CUDA 12.8)
# Build : docker build -t vsf-diarization .
# CLI   : docker run --gpus all -e HF_TOKEN=hf_xxx \
#           -v "$PWD/test:/app/test" -v "$PWD/outputs:/app/outputs" \
#           vsf-diarization  vsf-diarize test/test01.wav --asr whisper --language vi
# Server: docker run --gpus all -e HF_TOKEN=hf_xxx -p 8000:8000 \
#           vsf-diarization  vsf-serve --host 0.0.0.0   # → /transcribe /ws/stream /demo
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg libsndfile1 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch + torchaudio (CUDA 12.8) cài TRƯỚC để pyannote không kéo bản CPU
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Cài package (copy tối thiểu để cache layer khi chỉ đổi code)
COPY pyproject.toml README.md ./
COPY vsf_diarization ./vsf_diarization
RUN pip install --no-cache-dir ".[serve]"   # +REST/WS API & demo (thêm ",qwen" nếu cần Qwen3-ASR)

# Console scripts (vsf-diarize / vsf-stream / vsf-serve / vsf-evaluate /
# vsf-eval-streaming / vsf-create-gt) nằm sẵn trên PATH. HF_TOKEN truyền lúc run qua -e.
# Mount test/, ground_truth/, outputs/ làm volume khi chạy.
EXPOSE 8000
CMD ["bash"]
