"""
REST + WebSocket API cho diarization + ASR (FastAPI).

Endpoints:
    GET  /health            kiểm tra sống + models đã load chưa
    GET  /metrics           thống kê (uptime, device, avg RTF, #request …)
    POST /transcribe        upload file audio → diarization + ASR (JSON)
    WS   /ws/stream         gửi PCM float32 16kHz mono → nhận turn real-time
    /demo                   giao diện Gradio (mount khi chạy server)

Chạy:
    vsf-serve                       # hoặc: uvicorn vsf_diarization.serve.api:app
    vsf-serve --host 0.0.0.0 --port 8000 --no-demo

Models load lazy ở request đầu (xem serve/models.py) và được cache dùng lại.
"""

import io
import json
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Query
from starlette.responses import RedirectResponse
from starlette.staticfiles import StaticFiles

from vsf_diarization.core.utils import load_audio, SAMPLE_RATE
from vsf_diarization.core.streaming_session import StreamingSession
from vsf_diarization.serve.models import get_models, METRICS
from vsf_diarization.serve import observability as obs

STATIC_DIR = Path(__file__).parent / "static"


def create_app(mount_demo: bool = True) -> FastAPI:
    obs.setup_logging()
    app = FastAPI(title="VSF Diarization", version="0.1.0",
                  description="Vietnamese speaker diarization + ASR (pyannote + Whisper)")
    obs.instrument(app)                              # logging middleware + /metrics/prometheus

    @app.get("/health")
    def health():
        return {"status": "ok", "models_loaded": METRICS.snapshot()["models_loaded"]}

    @app.get("/metrics")
    def metrics():
        return METRICS.snapshot()

    @app.post("/transcribe")
    async def transcribe(
        file: UploadFile = File(...),
        language: str = Query("vi"),
        num_speakers: int | None = Query(None),
        chunk: float = Query(6.0),
        step: float = Query(1.0),
        threshold: float = Query(0.70),
        min_asr: float = Query(1.0),
    ):
        """Upload audio (wav/mp3/…) → list turn {speaker,start,end,text} + RTF."""
        raw = await file.read()
        data = load_audio(io.BytesIO(raw))
        dur = len(data) / SAMPLE_RATE

        m = get_models()
        session = StreamingSession(
            m["pipeline"], m["whisper"], chunk_s=chunk, step_s=step,
            threshold=threshold, num_speakers=num_speakers,
            language=language, min_asr=min_asr,
        )
        t0 = time.perf_counter()
        turns = session.feed(data) + session.finalize()
        proc = time.perf_counter() - t0

        METRICS.rest_requests += 1
        METRICS.record(dur, proc, len(turns))
        obs.record_job("/transcribe", proc, dur, len(turns))
        return {
            "file": file.filename,
            "duration_s": round(dur, 2),
            "rtf": round(proc / dur, 3) if dur else None,
            "num_turns": len(turns),
            "turns": turns,
        }

    @app.websocket("/ws/stream")
    async def ws_stream(ws: WebSocket):
        """Real-time: client gửi binary PCM float32 mono 16kHz; nhận lại JSON turn.
        Gửi text 'EOF' (hoặc đóng kết nối) để chốt turn cuối.
        Tuỳ chọn: gửi 1 text JSON đầu tiên để cấu hình (language, num_speakers …)."""
        await ws.accept()
        METRICS.ws_sessions += 1
        obs.ws_open()
        m = get_models()
        cfg = {"language": "vi", "num_speakers": None, "chunk_s": 6.0,
               "step_s": 1.0, "threshold": 0.70, "min_asr": 1.0}
        session = StreamingSession(m["pipeline"], m["whisper"], **cfg)
        configured = False
        fed_s = 0.0
        n_turns = 0
        t0 = time.perf_counter()
        try:
            try:
                while True:
                    msg = await ws.receive()
                    if msg.get("text") is not None:
                        txt = msg["text"]
                        if txt.strip().upper() == "EOF":
                            break
                        if not configured:               # config trước khi gửi audio
                            try:
                                cfg.update({k: v for k, v in json.loads(txt).items() if k in cfg})
                                session = StreamingSession(m["pipeline"], m["whisper"], **cfg)
                                configured = True
                            except (ValueError, TypeError):
                                pass
                        continue
                    if msg.get("bytes") is not None:
                        configured = True
                        audio = np.frombuffer(msg["bytes"], dtype=np.float32)
                        fed_s += len(audio) / SAMPLE_RATE
                        for turn in session.feed(audio):
                            n_turns += 1
                            await ws.send_json({"event": "turn", **turn})
            except WebSocketDisconnect:
                pass

            for turn in session.finalize():
                try:
                    n_turns += 1
                    await ws.send_json({"event": "turn", **turn})
                except RuntimeError:
                    break
            proc = time.perf_counter() - t0
            METRICS.record(fed_s, proc, n_turns)
            obs.record_job("/ws/stream", proc, fed_s, n_turns)
            try:
                await ws.send_json({"event": "done"})
                await ws.close()
            except RuntimeError:
                pass
        finally:
            obs.ws_close()

    if mount_demo:
        try:
            import gradio as gr
            from vsf_diarization.serve.demo import build_demo
            app = gr.mount_gradio_app(app, build_demo(), path="/demo")
        except Exception as e:                           # gradio chưa cài → API vẫn chạy
            print(f"[serve] Gradio demo tắt ({e}). Cài: pip install '.[serve]'")

    # Frontend tĩnh tại /app; '/' redirect vào đó
    if STATIC_DIR.is_dir():
        @app.get("/", include_in_schema=False)
        def _root():
            return RedirectResponse("/app/")
        app.mount("/app", StaticFiles(directory=str(STATIC_DIR), html=True), name="app")

    return app


app = create_app(mount_demo=False)   # mặc định cho `uvicorn ...:app` (không kéo gradio)


def main():
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser(description="VSF diarization API server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-demo", action="store_true", help="Tắt giao diện Gradio /demo")
    ap.add_argument("--warmup", action="store_true", help="Load models ngay khi khởi động")
    args = ap.parse_args()

    application = create_app(mount_demo=not args.no_demo)
    if args.warmup:
        print("Warmup: loading models …")
        get_models()
    base = f"http://{args.host}:{args.port}"
    print(f"→ App:        {base}/            (giao diện web)")
    print(f"→ Metrics:    {base}/metrics  ·  {base}/metrics/prometheus")
    if not args.no_demo:
        print(f"→ Gradio:     {base}/demo")
    uvicorn.run(application, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
