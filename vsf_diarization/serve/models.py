"""
Cached model loader + lightweight metrics (MLOps).

Pyannote + Whisper mất ~10–20s để load và chiếm VRAM. Server load **một lần** rồi
tái sử dụng cho mọi request (REST, WebSocket, demo) thay vì load lại mỗi lần.
Thread-safe qua khoá; an toàn cho FastAPI chạy nhiều worker-thread.

Cấu hình qua biến môi trường:
    HF_TOKEN           token Hugging Face (bắt buộc)
    VSF_WHISPER_MODEL  model Whisper (mặc định 'turbo')
    VSF_COMPUTE_TYPE   'float16' để ưu tiên độ chính xác; mặc định int8_float16
    VSF_DISABLE_ASR=1  chỉ load diarization (bỏ Whisper)
"""

import os
import threading
import time
from dataclasses import dataclass, field

_LOCK = threading.Lock()
_MODELS: dict | None = None


def get_models() -> dict:
    """Trả về {'pipeline', 'whisper'} đã cache; load một lần ở lần gọi đầu."""
    global _MODELS
    if _MODELS is not None:
        return _MODELS
    with _LOCK:
        if _MODELS is not None:
            return _MODELS

        import torch
        from vsf_diarization.core.utils import (
            get_hf_token, load_diarization_pipeline, load_whisper,
        )
        torch.set_grad_enabled(False)                    # inference-only → tiết kiệm RAM

        t0 = time.perf_counter()
        pipeline = load_diarization_pipeline(get_hf_token())
        whisper = None
        if os.environ.get("VSF_DISABLE_ASR") != "1":
            whisper = load_whisper(
                os.environ.get("VSF_WHISPER_MODEL", "turbo"),
                os.environ.get("VSF_COMPUTE_TYPE"),
            )
        METRICS.model_load_s = round(time.perf_counter() - t0, 2)
        METRICS.device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODELS = {"pipeline": pipeline, "whisper": whisper}

        from vsf_diarization.serve import observability as obs
        obs.set_model_load(METRICS.model_load_s)
        obs.logger.info("Models loaded in %.1fs on %s", METRICS.model_load_s, METRICS.device)
    return _MODELS


# ── Monitoring ─────────────────────────────────────────────────────────────────

@dataclass
class Metrics:
    started_at: float = field(default_factory=time.time)
    device: str = "?"
    model_load_s: float | None = None
    rest_requests: int = 0
    ws_sessions: int = 0
    turns_emitted: int = 0
    audio_seconds: float = 0.0      # tổng audio đã xử lý
    proc_seconds: float = 0.0       # tổng thời gian xử lý

    def record(self, audio_s: float, proc_s: float, turns: int) -> None:
        self.audio_seconds += audio_s
        self.proc_seconds += proc_s
        self.turns_emitted += turns

    def snapshot(self) -> dict:
        rtf = round(self.proc_seconds / self.audio_seconds, 3) if self.audio_seconds else None
        return {
            "uptime_s": round(time.time() - self.started_at, 1),
            "device": self.device,
            "model_load_s": self.model_load_s,
            "models_loaded": _MODELS is not None,
            "rest_requests": self.rest_requests,
            "ws_sessions": self.ws_sessions,
            "turns_emitted": self.turns_emitted,
            "audio_seconds": round(self.audio_seconds, 1),
            "avg_rtf": rtf,
        }


METRICS = Metrics()
