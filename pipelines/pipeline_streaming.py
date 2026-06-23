#!/usr/bin/env python3
"""
Streaming end-to-end — ASR + Speaker Diarization (dùng chung stream_online()).
[--no-asr = chỉ diarization streaming]

Diarization streaming dùng cùng thuật toán đã benchmark trong core/diarize_online.py
(sliding window + majority-vote 50ms/frame). Mỗi khi một speaker-turn trở nên ổn
định, Whisper nhận dạng đoạn đó và in ra: [t_start -> t_end] SPEAKER_XX: text.

Usage (chạy từ thư mục gốc dự án):
    # Giả lập streaming từ file (đo & xem output)
    python pipelines/pipeline_streaming.py --source test/test01.wav --language vi --num-speakers 2

    # Chỉ diarization (không ASR) — thay stream_diarization.py
    python pipelines/pipeline_streaming.py --source test/test01.wav --no-asr

    # Từ microphone (ghi tới Ctrl-C rồi xử lý)
    python pipelines/pipeline_streaming.py --source mic --language vi

Tham số streaming (giống stream_online):
    --chunk 6   cửa sổ (s); 6 = latency thấp ~0.9s, 9 = chính xác hơn nhưng ~1.35s
    --step 1    bước trượt (s)
    --threshold 0.70   ngưỡng cosine speaker registry
"""

import argparse
import pathlib
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # cho phép chạy trực tiếp
from core.utils import (
    get_hf_token, get_audio_input, load_audio, load_diarization_pipeline,
    load_whisper, fmt_time, SAMPLE_RATE,
)
from core.diarize_online import stream_online

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")


# ── ASR cho một speaker-turn ───────────────────────────────────────────────────

def transcribe_turn(whisper, data: np.ndarray, start: float, end: float, language: str | None) -> str:
    """Nhận dạng đoạn audio [start, end] bằng Whisper, trả về text."""
    sl = data[int(start * SAMPLE_RATE): int(end * SAMPLE_RATE)]
    if len(sl) < int(0.2 * SAMPLE_RATE):
        return ""
    # vad_filter=True giảm hallucination của Whisper trên đoạn ngắn / gần im lặng.
    segs, _ = whisper.transcribe(sl, language=language, vad_filter=True)
    return " ".join(s.text.strip() for s in segs).strip()


# ── Vòng lặp streaming ─────────────────────────────────────────────────────────

def run_stream(data, pipeline, whisper, language, chunk_s, step_s, threshold,
               num_speakers, realtime=False, min_asr=1.0, verbose=True):
    """
    Chạy stream_online() để diarize tăng dần; gộp các segment liền kề cùng speaker
    thành 1 turn, đến khi đổi speaker thì chốt turn:
      - turn dài  ≥ min_asr  → Whisper nhận dạng
      - turn ngắn < min_asr  → in "..." (bỏ qua ASR: tránh hallucinate + nhanh hơn)
    Trả về list segment {speaker, start, end, text}.
    """
    cur = None       # turn đang gom: {speaker, start, end}
    results = []

    def flush(turn):
        if not turn:
            return
        dur = turn["end"] - turn["start"]
        if dur < 0.3:                    # bỏ fragment quá ngắn (nhiễu biên), khớp merge_segments
            return
        if dur < min_asr:
            text = "..."                 # turn ngắn → không ASR (tránh hallucinate)
        else:
            text = transcribe_turn(whisper, data, turn["start"], turn["end"], language) or "..."
        turn["text"] = text
        results.append(turn)
        if verbose:
            print(f"[{fmt_time(turn['start'])} -> {fmt_time(turn['end'])}]  "
                  f"{turn['speaker']}: {text}")

    for segs in stream_online(data, pipeline, chunk_s=chunk_s, step_s=step_s,
                              registry_threshold=threshold, num_speakers=num_speakers):
        for seg in segs:
            if cur and seg["speaker"] == cur["speaker"] and seg["start"] <= cur["end"] + 0.6:
                cur["end"] = seg["end"]                      # cùng người → nối dài turn
            else:
                flush(cur)                                   # đổi người → chốt turn cũ
                cur = {"speaker": seg["speaker"], "start": seg["start"], "end": seg["end"]}
        if realtime:
            time.sleep(step_s)                               # giả lập nhịp real-time

    flush(cur)                                               # chốt turn cuối
    return results


def run_stream_diar_only(data, pipeline, chunk_s, step_s, threshold,
                         num_speakers, realtime=False):
    """Streaming chỉ diarization (không ASR) — gộp segment liền kề cùng speaker.
    Thay cho stream_diarization.py cũ; trả về list {speaker, start, end}."""
    cur, results = None, []
    for segs in stream_online(data, pipeline, chunk_s=chunk_s, step_s=step_s,
                              registry_threshold=threshold, num_speakers=num_speakers):
        for seg in segs:
            if cur and seg["speaker"] == cur["speaker"] and seg["start"] <= cur["end"] + 0.6:
                cur["end"] = seg["end"]
            else:
                if cur:
                    results.append(cur)
                    print(f"[{fmt_time(cur['start'])} -> {fmt_time(cur['end'])}]  {cur['speaker']}")
                cur = {"speaker": seg["speaker"], "start": seg["start"], "end": seg["end"]}
        if realtime:
            time.sleep(step_s)
    if cur:
        results.append(cur)
        print(f"[{fmt_time(cur['start'])} -> {fmt_time(cur['end'])}]  {cur['speaker']}")
    return results


# ── Nguồn audio ────────────────────────────────────────────────────────────────

def record_mic() -> np.ndarray:
    """Ghi từ microphone tới khi Ctrl-C, trả về toàn bộ audio (16kHz mono)."""
    import queue
    import sounddevice as sd

    q: queue.Queue = queue.Queue()
    block = int(0.5 * SAMPLE_RATE)
    sd_stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=block,
                               callback=lambda indata, *_: q.put(indata[:, 0].copy()))
    chunks = []
    print("Đang ghi từ microphone … (Ctrl-C để dừng và xử lý)")
    with sd_stream:
        try:
            while True:
                chunks.append(q.get())
        except KeyboardInterrupt:
            print("\nĐã dừng ghi, đang xử lý …")
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Streaming end-to-end: diarization + ASR (stream_online)")
    ap.add_argument("--source", default="mic", help="'mic' hoặc đường dẫn file audio/video")
    ap.add_argument("--hf-token")
    ap.add_argument("--whisper-model", default="turbo")
    ap.add_argument("--language", help="Mã ngôn ngữ, vd 'vi'")
    ap.add_argument("--chunk", type=float, default=6.0, help="Cửa sổ (s) — latency ≈ chunk × RTF")
    ap.add_argument("--step", type=float, default=1.0, help="Bước trượt (s)")
    ap.add_argument("--threshold", type=float, default=0.70, help="Ngưỡng speaker registry")
    ap.add_argument("--num-speakers", type=int, help="Gợi ý số người nói")
    ap.add_argument("--min-asr", type=float, default=1.0,
                    help="Turn ngắn hơn (s) sẽ in '...' thay vì nhận dạng (tránh hallucinate)")
    ap.add_argument("--realtime", action="store_true", help="Sleep theo step để giả lập nhịp real-time")
    ap.add_argument("--no-asr", action="store_true",
                    help="Chỉ diarization streaming (không Whisper) — thay stream_diarization.py")
    ap.add_argument("--output", help="Lưu segment ra JSON")
    args = ap.parse_args()

    hf_token = get_hf_token(args.hf_token)
    print("Loading models …")
    pipeline = load_diarization_pipeline(hf_token)
    whisper = None if args.no_asr else load_whisper(args.whisper_model)

    if args.source == "mic":
        data = record_mic()
    else:
        data = load_audio(get_audio_input(args.source))

    if len(data) < SAMPLE_RATE:
        sys.exit("Audio quá ngắn (< 1s).")

    dur = len(data) / SAMPLE_RATE
    print(f"\n{'-'*70}")
    print(f"  Streaming{' (diar-only)' if args.no_asr else ''}  |  {dur:.1f}s  "
          f"|  chunk={args.chunk}s step={args.step}s thr={args.threshold}")
    print(f"{'-'*70}\n")

    t0 = time.perf_counter()
    if args.no_asr:
        segs = run_stream_diar_only(data, pipeline, args.chunk, args.step,
                                    args.threshold, args.num_speakers, realtime=args.realtime)
    else:
        segs = run_stream(data, pipeline, whisper, args.language, args.chunk, args.step,
                          args.threshold, args.num_speakers, realtime=args.realtime,
                          min_asr=args.min_asr)
    elapsed = time.perf_counter() - t0

    if args.no_asr:
        print(f"\n{'-'*70}")
        print(f"  Xong: {len(segs)} segment  |  {elapsed:.1f}s xử lý  |  RTF {elapsed/dur:.2f}x")
        print(f"{'-'*70}")
    else:
        asr_turns = sum(1 for s in segs if s["text"] != "...")
        print(f"\n{'-'*70}")
        print(f"  Xong: {len(segs)} turn ({asr_turns} có ASR, {len(segs)-asr_turns} '...')  "
              f"|  {elapsed:.1f}s xử lý  |  RTF {elapsed/dur:.2f}x"
              f"{'  (đã sleep real-time)' if args.realtime else ''}")
        print(f"{'-'*70}")

    if args.output:
        import json
        from pathlib import Path
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(segs, f, indent=2, ensure_ascii=False)
        print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
