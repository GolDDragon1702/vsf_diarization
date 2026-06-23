#!/usr/bin/env python3
"""
Sweep min_asr cho Phase 3 streaming (tìm trade-off WER vs coverage).

Tối ưu: với mỗi file chỉ diarize + transcribe MỖI turn 1 lần (transcript của turn
dài là superset). Sau đó mỗi min_asr chỉ cần mask turn dur < min_asr -> "...".
=> 1× chi phí thay vì N× (DER không đổi theo min_asr; chỉ WER/CER/coverage đổi).

Usage:
  python sweep_phase3.py test/ --language vi --min-asr 1.0 1.5 2.0 \
      --output outputs/phase3_sweep.json
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # cho phép chạy trực tiếp
from core.utils import (
    get_hf_token, load_audio, load_diarization_pipeline, load_whisper,
    compute_der, compute_asr_metrics, load_gt, iter_wavs, SAMPLE_RATE,
)
from core.diarize_online import stream_online
from pipelines.pipeline_streaming import transcribe_turn


def diarize_and_transcribe(data, pipeline, whisper, language, chunk_s, step_s,
                           threshold, num_speakers):
    """Trả về list turn {speaker,start,end,dur,text} — transcribe MỌI turn >= 0.3s."""
    turns, cur = [], None

    def flush(t):
        if not t:
            return
        dur = t["end"] - t["start"]
        if dur < 0.3:                                   # nhiễu biên, khớp run_stream
            return
        t["dur"] = dur
        t["text"] = transcribe_turn(whisper, data, t["start"], t["end"], language) or "..."
        turns.append(t)

    for segs in stream_online(data, pipeline, chunk_s=chunk_s, step_s=step_s,
                              registry_threshold=threshold, num_speakers=num_speakers):
        for seg in segs:
            if cur and seg["speaker"] == cur["speaker"] and seg["start"] <= cur["end"] + 0.6:
                cur["end"] = seg["end"]
            else:
                flush(cur)
                cur = {"speaker": seg["speaker"], "start": seg["start"], "end": seg["end"]}
    flush(cur)
    return turns


def mask(turns, min_asr):
    """Áp ngưỡng min_asr: turn dur < min_asr -> '...' (deletion khi tính WER)."""
    return [{"speaker": t["speaker"], "start": t["start"], "end": t["end"],
             "text": "..." if t["dur"] < min_asr else t["text"]} for t in turns]


def main():
    ap = argparse.ArgumentParser(description="Sweep min_asr cho Phase 3 streaming")
    ap.add_argument("input", nargs="+", help="WAV file(s) hoặc folder")
    ap.add_argument("--hf-token")
    ap.add_argument("--whisper-model", default="turbo")
    ap.add_argument("--language", default="vi")
    ap.add_argument("--chunk", type=float, default=6.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--min-asr", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    ap.add_argument("--output")
    args = ap.parse_args()

    wav_files = iter_wavs(args.input)
    if not wav_files:
        sys.exit("No WAV files found")

    print("Loading models …", flush=True)
    pipeline = load_diarization_pipeline(get_hf_token(args.hf_token))
    whisper = load_whisper(args.whisper_model)
    thresholds = sorted(args.min_asr)
    print(f"chunk={args.chunk}s step={args.step}s thr={args.threshold} "
          f"min_asr sweep={thresholds}\n", flush=True)

    # min_asr -> {der, wer, cer, cov} list các file
    agg = {m: {"der": [], "wer": [], "cer": [], "cov": []} for m in thresholds}
    per_file = []

    files = [w for w in wav_files if load_gt(str(w))[0] is not None]
    for i, wav in enumerate(files, 1):
        gt_segs, n_spk = load_gt(str(wav))
        data = load_audio(str(wav))
        dur = len(data) / SAMPLE_RATE

        t0 = time.perf_counter()
        turns = diarize_and_transcribe(data, pipeline, whisper, args.language,
                                       args.chunk, args.step, args.threshold, n_spk)
        rtf = round((time.perf_counter() - t0) / dur, 3)

        der = compute_der(gt_segs, mask(turns, 0))["DER_%"]   # DER độc lập min_asr
        row = {"file": Path(wav).name, "DER_%": der, "rtf": rtf, "by_min_asr": {}}
        print(f"[{i}/{len(files)}] {Path(wav).name:<11} DER {der:>6.2f}%  "
              f"RTF {rtf:.2f}x  | {len(turns)} turn", flush=True)

        for m in thresholds:
            segs = mask(turns, m)
            asr = compute_asr_metrics(gt_segs, segs)
            spoken = sum(s["end"] - s["start"] for s in segs) or 1e-9
            cov = round(sum(s["end"] - s["start"] for s in segs
                            if s["text"] != "...") / spoken * 100, 1)
            row["by_min_asr"][m] = {"WER_%": asr["WER_%"], "CER_%": asr["CER_%"], "cov_%": cov}
            agg[m]["der"].append(der)
            if asr["WER_%"] is not None:
                agg[m]["wer"].append(asr["WER_%"]); agg[m]["cer"].append(asr["CER_%"])
            agg[m]["cov"].append(cov)
            print(f"        min_asr={m}: WER {str(asr['WER_%']):>6}%  "
                  f"CER {str(asr['CER_%']):>6}%  cov {cov:>5}%", flush=True)
        per_file.append(row)

    def mean(x):
        return round(sum(x) / len(x), 2) if x else None

    print(f"\n{'='*64}")
    print(f"  SWEEP min_asr — TRUNG BÌNH ({len(files)} file)  [DER cố định, ASR đổi]")
    print(f"  {'min_asr':>8} | {'WER':>7} | {'CER':>7} | {'coverage':>9}")
    print(f"  {'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*9}")
    summary = {}
    for m in thresholds:
        wer, cer, cov = mean(agg[m]["wer"]), mean(agg[m]["cer"]), mean(agg[m]["cov"])
        summary[m] = {"WER_%": wer, "CER_%": cer, "cov_%": cov}
        print(f"  {m:>8} | {str(wer):>6}% | {str(cer):>6}% | {str(cov):>8}%")
    print(f"  DER (mọi min_asr): {mean(agg[thresholds[0]]['der'])}%   "
          f"RTF: {mean([r['rtf'] for r in per_file])}x")
    print(f"{'='*64}", flush=True)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out = {"config": {"chunk_s": args.chunk, "step_s": args.step,
                          "threshold": args.threshold, "min_asr_sweep": thresholds},
               "summary_mean": summary,
               "DER_%": mean(agg[thresholds[0]]["der"]),
               "files": per_file}
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Saved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
