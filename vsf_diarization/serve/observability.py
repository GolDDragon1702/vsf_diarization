"""
Observability: structured logging + Prometheus metrics + request middleware.

- Logging: định dạng gọn `time level logger | message`, mọi request HTTP/WS được log
  (method, path, status, thời gian xử lý).
- Metrics (Prometheus, scrape tại /metrics/prometheus):
    vsf_requests_total{endpoint,status}      Counter   số request
    vsf_request_seconds{endpoint}            Histogram độ trễ xử lý
    vsf_rtf{endpoint}                        Histogram real-time factor (proc/audio)
    vsf_audio_seconds_total                  Counter   tổng audio đã xử lý
    vsf_turns_total                          Counter   tổng turn phát ra
    vsf_active_ws_sessions                   Gauge     phiên WebSocket đang mở
    vsf_model_load_seconds                   Gauge     thời gian load model
    vsf_gpu_memory_bytes{type}               Gauge     VRAM allocated/reserved

prometheus_client là tuỳ chọn — thiếu nó thì API vẫn chạy, chỉ tắt /metrics/prometheus.
"""

import logging
import time

logger = logging.getLogger("vsf")

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST,
    )
    _PROM = True
except ImportError:                                  # pragma: no cover
    _PROM = False


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("uvicorn.access", "faster_whisper", "httpx"):      # giảm log nhiễu
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Prometheus metrics ──────────────────────────────────────────────────────────

if _PROM:
    REQUESTS = Counter("vsf_requests_total", "Số request", ["endpoint", "status"])
    LATENCY = Histogram("vsf_request_seconds", "Độ trễ xử lý (s)", ["endpoint"],
                        buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60))
    RTF = Histogram("vsf_rtf", "Real-time factor (proc/audio)", ["endpoint"],
                    buckets=(0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3))
    AUDIO_SECONDS = Counter("vsf_audio_seconds_total", "Tổng audio đã xử lý (s)")
    TURNS = Counter("vsf_turns_total", "Tổng turn phát ra")
    ACTIVE_WS = Gauge("vsf_active_ws_sessions", "Phiên WebSocket đang mở")
    MODEL_LOAD = Gauge("vsf_model_load_seconds", "Thời gian load model (s)")
    GPU_MEM = Gauge("vsf_gpu_memory_bytes", "VRAM (bytes)", ["type"])


def record_job(endpoint: str, latency_s: float, audio_s: float, turns: int) -> None:
    """Ghi nhận một job xong (REST hoặc WS): độ trễ, RTF, audio, turn."""
    update_gpu_memory()
    if not _PROM:
        return
    LATENCY.labels(endpoint).observe(latency_s)
    AUDIO_SECONDS.inc(audio_s)
    TURNS.inc(turns)
    if audio_s > 0:
        RTF.labels(endpoint).observe(latency_s / audio_s)


def inc_request(endpoint: str, status: str) -> None:
    if _PROM:
        REQUESTS.labels(endpoint, status).inc()


def ws_open() -> None:
    if _PROM:
        ACTIVE_WS.inc()


def ws_close() -> None:
    if _PROM:
        ACTIVE_WS.dec()


def set_model_load(seconds: float) -> None:
    if _PROM:
        MODEL_LOAD.set(seconds)


def update_gpu_memory() -> None:
    if not _PROM:
        return
    try:
        import torch
        if torch.cuda.is_available():
            GPU_MEM.labels("allocated").set(torch.cuda.memory_allocated())
            GPU_MEM.labels("reserved").set(torch.cuda.memory_reserved())
    except Exception:
        pass


# ── FastAPI wiring ──────────────────────────────────────────────────────────────

def instrument(app) -> None:
    """Thêm middleware log + đếm request và route /metrics/prometheus."""
    from starlette.requests import Request
    from starlette.responses import Response, PlainTextResponse

    @app.middleware("http")
    async def _log_and_count(request: Request, call_next):
        t0 = time.perf_counter()
        try:
            response: Response = await call_next(request)
        except Exception:
            logger.exception("%s %s -> 500", request.method, request.url.path)
            inc_request(request.url.path, "500")
            raise
        dt = (time.perf_counter() - t0) * 1000
        inc_request(request.url.path, str(response.status_code))
        logger.info("%s %s -> %s (%.0fms)", request.method, request.url.path,
                    response.status_code, dt)
        return response

    @app.get("/metrics/prometheus", include_in_schema=False)
    def prometheus_metrics():
        if not _PROM:
            return PlainTextResponse("prometheus_client chưa cài\n", status_code=501)
        update_gpu_memory()
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
