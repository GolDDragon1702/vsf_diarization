"""
Gradio demo: upload / ghi mic → transcript tô màu theo người nói.

Chạy độc lập:   python -m vsf_diarization.serve.demo
Hoặc mount vào API tại /demo (xem serve/api.py).
"""

import time

from vsf_diarization.core.utils import load_audio, fmt_time, SAMPLE_RATE
from vsf_diarization.core.streaming_session import StreamingSession
from vsf_diarization.serve.models import get_models, METRICS

_PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]


def _render(turns: list[dict]) -> str:
    colors, html = {}, []
    for t in turns:
        spk = t["speaker"]
        colors.setdefault(spk, _PALETTE[len(colors) % len(_PALETTE)])
        ts = f"{fmt_time(t['start'])[3:]} → {fmt_time(t['end'])[3:]}"
        html.append(
            f"<div style='margin:6px 0;padding:8px 12px;border-left:4px solid {colors[spk]};"
            f"background:#f8fafc;border-radius:4px'>"
            f"<b style='color:{colors[spk]}'>{spk}</b> "
            f"<span style='color:#94a3b8;font-size:.8em'>[{ts}]</span><br>"
            f"<span style='color:#0f172a'>{t['text']}</span></div>"
        )
    return "<div style='font-family:sans-serif'>" + "".join(html) + "</div>"


def transcribe(audio_path, language, num_speakers):
    if not audio_path:
        return "<i>Hãy upload hoặc ghi âm trước.</i>", {}
    data = load_audio(audio_path)
    dur = len(data) / SAMPLE_RATE
    m = get_models()
    session = StreamingSession(
        m["pipeline"], m["whisper"], language=language or None,
        num_speakers=int(num_speakers) if num_speakers else None,
    )
    t0 = time.perf_counter()
    turns = session.feed(data) + session.finalize()
    proc = time.perf_counter() - t0
    METRICS.record(dur, proc, len(turns))

    stats = {"duration_s": round(dur, 2), "rtf": round(proc / dur, 3) if dur else None,
             "num_turns": len(turns), "speakers": sorted({t["speaker"] for t in turns})}
    return _render(turns), stats


def build_demo():
    import gradio as gr

    with gr.Blocks(title="VSF Diarization") as demo:
        gr.Markdown("## 🎙️ Vietnamese Speaker Diarization + ASR\n"
                    "Upload / ghi âm hội thoại → ai nói câu gì (pyannote + Whisper).")
        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.Audio(sources=["upload", "microphone"], type="filepath",
                                 label="Audio")
                language = gr.Textbox(value="vi", label="Ngôn ngữ (mã ISO, vd 'vi')")
                num_speakers = gr.Number(label="Số người nói (để trống = tự động)",
                                         precision=0)
                btn = gr.Button("Phân tích", variant="primary")
            with gr.Column(scale=2):
                out_html = gr.HTML(label="Transcript")
                out_json = gr.JSON(label="Thống kê")
        btn.click(transcribe, [audio, language, num_speakers], [out_html, out_json])
    return demo


def main():
    build_demo().launch()


if __name__ == "__main__":
    main()
