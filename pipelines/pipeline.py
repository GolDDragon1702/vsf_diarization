#!/usr/bin/env python3
"""
Pipeline offline: speaker diarization (+ ASR tuỳ chọn).
Gộp phase1_diarization (--asr none) + phase2_asr_diarization (--asr whisper)
+ phase2_qwen_asr (--asr qwen).

Usage (chạy từ thư mục gốc dự án):
    # Chỉ diarization
    python pipelines/pipeline.py test/test01.wav --asr none --num-speakers 2
    # Diarization + Whisper (mặc định)
    python pipelines/pipeline.py test/test01.wav --asr whisper --language vi --output out.json
    # Diarization + Qwen3-ASR
    python pipelines/pipeline.py test/test01.wav --asr qwen --language vi
"""

import argparse
import json
import pathlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # cho phép chạy trực tiếp
sys.stdout.reconfigure(encoding="utf-8")

from core.utils import (
    get_hf_token, get_audio_input, load_audio, load_diarization_pipeline,
    load_whisper, load_qwen_asr, whisper_transcribe, qwen_transcribe,
    fmt_time, SAMPLE_RATE,
)
from core.diarize_offline import run_offline


def main():
    ap = argparse.ArgumentParser(description="Offline diarization (+ASR tuỳ chọn)")
    ap.add_argument("input", help="Audio hoặc video")
    ap.add_argument("--asr", choices=["none", "whisper", "qwen"], default="whisper",
                    help="Bộ ASR: none = chỉ diarization")
    ap.add_argument("--hf-token")
    ap.add_argument("--language")
    ap.add_argument("--num-speakers", type=int)
    ap.add_argument("--whisper-model", default="turbo")
    ap.add_argument("--qwen-model", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--output")
    args = ap.parse_args()

    hf_token = get_hf_token(args.hf_token)
    audio_path = get_audio_input(args.input)

    print("Loading models …")
    pipeline = load_diarization_pipeline(hf_token)
    whisper = load_whisper(args.whisper_model) if args.asr == "whisper" else None
    qwen = load_qwen_asr(args.qwen_model) if args.asr == "qwen" else None

    data = load_audio(audio_path)
    duration = len(data) / SAMPLE_RATE

    print("Running diarization …")
    turns, t_diar = run_offline(data, pipeline, args.num_speakers)
    print(f"Diarization: {len(turns)} segments ({t_diar:.1f}s)\n")

    detected_lang = args.language
    t0 = time.perf_counter()
    if args.asr == "whisper":
        segments, detected_lang = whisper_transcribe(data, turns, whisper, args.language)
    elif args.asr == "qwen":
        segments = qwen_transcribe(data, turns, qwen, args.language)
    else:                                            # none → chỉ diarization
        segments = turns
    t_asr = time.perf_counter() - t0

    speakers = sorted({s["speaker"] for s in segments})
    result = {
        "input": args.input,
        "asr": args.asr,
        "detected_language": detected_lang,
        "speakers": speakers,
        "segments": segments,
        "timing": {"diarization_s": round(t_diar, 3), "asr_s": round(t_asr, 3),
                   "rtf": round((t_diar + t_asr) / duration, 2) if duration else 0},
    }

    print(f"\n{'-'*70}")
    print(f"  ASR={args.asr}  Lang={detected_lang}  Speakers: {', '.join(speakers)}  "
          f"RTF: {result['timing']['rtf']}x")
    print(f"{'-'*70}")
    for seg in segments:
        head = f"[{fmt_time(seg['start'])} -> {fmt_time(seg['end'])}]  {seg['speaker']}"
        print(f"{head}: {seg['text']}" if "text" in seg else head)
    print(f"{'-'*70}\n")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
