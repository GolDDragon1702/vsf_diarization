#!/usr/bin/env python3
"""
Đánh giá streaming end-to-end (diarization + ASR) vs ground truth.

Chạy đúng pipeline streaming của pipeline_streaming.run_stream() (stream_online +
Whisper per-turn, turn ngắn < min_asr → "..."), rồi đo:
  - DER  : chất lượng diarization streaming (so với online batch ở REPORT mục 6.2)
  - WER/CER end-to-end : transcript streaming vs GT — turn "..." tính là deletion
  - ASR coverage : % thời lượng speech được nhận dạng (turn ≥ min_asr)

Usage (chạy từ thư mục gốc dự án):
  python eval/eval_streaming.py test/ --language vi --min-asr 1.0 --output outputs/streaming_eval.json
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

from vsf_diarization.core.utils import (
    get_hf_token, load_audio, load_diarization_pipeline, load_whisper,
    compute_der, compute_asr_metrics, load_gt, iter_wavs, SAMPLE_RATE,
)
from vsf_diarization.pipelines.pipeline_streaming import run_stream


def main():
    ap = argparse.ArgumentParser(description="Đánh giá streaming end-to-end vs GT")
    ap.add_argument("input", nargs="+", help="WAV file(s) hoặc folder")
    ap.add_argument("--hf-token")
    ap.add_argument("--whisper-model", default="turbo")
    ap.add_argument("--language", default="vi")
    ap.add_argument("--chunk", type=float, default=6.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--min-asr", type=float, default=1.0)
    ap.add_argument("--output")
    args = ap.parse_args()

    wav_files = iter_wavs(args.input)
    if not wav_files:
        sys.exit("No WAV files found")

    print("Loading models …")
    pipeline = load_diarization_pipeline(get_hf_token(args.hf_token))
    whisper = load_whisper(args.whisper_model)
    print(f"chunk={args.chunk}s step={args.step}s thr={args.threshold} min_asr={args.min_asr}s\n")

    results = []
    agg = {"der": [], "wer": [], "cer": [], "cov": [], "rtf": []}

    for wav in wav_files:
        gt_segs, n_spk = load_gt(str(wav))
        if gt_segs is None:
            continue
        data = load_audio(str(wav))
        dur = len(data) / SAMPLE_RATE

        t0 = time.perf_counter()
        segs = run_stream(data, pipeline, whisper, args.language, args.chunk, args.step,
                          args.threshold, n_spk, realtime=False, min_asr=args.min_asr,
                          verbose=False)
        rtf = round((time.perf_counter() - t0) / dur, 3)

        der = compute_der(gt_segs, segs)
        asr = compute_asr_metrics(gt_segs, segs)         # "..." → text rỗng → deletion
        spoken = sum(s["end"] - s["start"] for s in segs) or 1e-9
        covered = sum(s["end"] - s["start"] for s in segs if s["text"] != "...")
        cov = round(covered / spoken * 100, 1)

        r = {"file": Path(wav).name, "duration_s": round(dur, 2),
             "DER_%": der["DER_%"], "miss_%": der["miss_%"], "fa_%": der["fa_%"],
             "conf_%": der["conf_%"], "WER_%": asr["WER_%"], "CER_%": asr["CER_%"],
             "asr_coverage_%": cov, "n_turns": len(segs), "rtf": rtf}
        results.append(r)
        print(f"  {r['file']:<11} DER {der['DER_%']:>6.2f}%  WER {str(asr['WER_%']):>6}%  "
              f"CER {str(asr['CER_%']):>6}%  cov {cov:>5}%  RTF {rtf:.2f}x")
        agg["der"].append(der["DER_%"])
        if asr["WER_%"] is not None:
            agg["wer"].append(asr["WER_%"]); agg["cer"].append(asr["CER_%"])
        agg["cov"].append(cov); agg["rtf"].append(rtf)

    if results:
        def m(x): return round(sum(x) / len(x), 2) if x else None
        print(f"\n{'='*60}")
        print(f"  STREAMING END-TO-END — TRUNG BÌNH ({len(results)} file)")
        print(f"    DER          : {m(agg['der'])}%   (online batch ref: 13.81%)")
        print(f"    WER (e2e)    : {m(agg['wer'])}%   (offline pipeline ref: 11.84%)")
        print(f"    CER (e2e)    : {m(agg['cer'])}%")
        print(f"    ASR coverage : {m(agg['cov'])}%")
        print(f"    RTF          : {m(agg['rtf'])}x")
        print(f"{'='*60}")

        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            out = {"config": {"chunk_s": args.chunk, "step_s": args.step,
                              "threshold": args.threshold, "min_asr_s": args.min_asr},
                   "mean": {"DER_%": m(agg["der"]), "WER_%": m(agg["wer"]),
                            "CER_%": m(agg["cer"]), "asr_coverage_%": m(agg["cov"]),
                            "rtf": m(agg["rtf"])},
                   "files": results}
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
