#!/usr/bin/env python3
"""
Compare offline (pyannote) vs online (diart-style) diarization strategies.

If a reviewed ground truth exists in ground_truth/<name>.json, DER is
computed against it for both strategies. Otherwise only mutual DER is shown.

Usage:
  python compare_diarization.py test/test01.wav
  python compare_diarization.py test/test01.wav test/test02.wav --output outputs/diar_compare.json
  python compare_diarization.py test/ --window 12 --step 1 --threshold 0.70
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # cho phép chạy trực tiếp
from core.utils import (
    get_hf_token, load_audio, load_diarization_pipeline,
    compute_der, fmt_time, load_gt, SAMPLE_RATE,
)
from core.diarize_offline import run_offline
from core.diarize_online import run_online


# ── Stats ─────────────────────────────────────────────────────────────────────

def _frag_stats(segs: list[dict]) -> dict:
    if not segs:
        return {"num_segments": 0, "num_speakers": 0, "avg_dur_s": 0.0}
    durs = [s["end"] - s["start"] for s in segs]
    return {
        "num_segments": len(segs),
        "num_speakers": len({s["speaker"] for s in segs}),
        "avg_dur_s": round(float(np.mean(durs)), 2),
    }


# ── Per-file comparison ───────────────────────────────────────────────────────

def compare_file(
    wav_path: str, pipeline,
    window_s: float, step_s: float, threshold: float,
    num_speakers: int | None,
) -> dict:
    gt_segs, gt_n_spk = load_gt(wav_path, require_reviewed=False)
    effective_n_spk = num_speakers if num_speakers is not None else gt_n_spk

    data = load_audio(wav_path)
    duration = len(data) / SAMPLE_RATE

    offline_segs, t_off = run_offline(data, pipeline, effective_n_spk)
    online_segs,  t_on  = run_online(data, pipeline, window_s, step_s, threshold, effective_n_spk)

    has_gt = gt_segs is not None
    return {
        "file": wav_path,
        "duration_s": round(duration, 2),
        "has_ground_truth": has_gt,
        "offline": {
            "rtf": round(t_off / duration, 3),
            **_frag_stats(offline_segs),
            "der_vs_gt": compute_der(gt_segs, offline_segs) if has_gt else None,
            "segments": offline_segs,
        },
        "online": {
            "rtf": round(t_on / duration, 3),
            **_frag_stats(online_segs),
            "der_vs_gt": compute_der(gt_segs, online_segs) if has_gt else None,
            "segments": online_segs,
        },
        "online_vs_offline": compute_der(offline_segs, online_segs),
        "config": {"window_s": window_s, "step_s": step_s, "threshold": threshold},
    }


# ── Printer ───────────────────────────────────────────────────────────────────

def print_result(r: dict) -> None:
    sep = "=" * 72
    thin = "-" * 72
    print(f"\n{sep}")
    print(f"  File    : {r['file']}  ({r['duration_s']}s)")
    print(thin)
    print(f"  {'Metric':<28} {'Offline (pyannote)':>18} {'Online (diart-style)':>20}")
    print(thin)
    for label, key in [("RTF", "rtf"), ("Segments", "num_segments"),
                       ("Speakers detected", "num_speakers"), ("Avg seg dur (s)", "avg_dur_s")]:
        ov, nv = r["offline"][key], r["online"][key]
        fmt = ".3f" if key == "rtf" else (".2f" if key == "avg_dur_s" else "d")
        print(f"  {label:<28} {ov:>18{fmt}} {nv:>20{fmt}}")

    if r["has_ground_truth"]:
        print(thin)
        od, nd = r["offline"]["der_vs_gt"], r["online"]["der_vs_gt"]
        print(f"  {'DER vs Ground Truth':<28} {od['DER_%']:>17.2f}% {nd['DER_%']:>19.2f}%")
        for sub, key in [("Miss", "miss_%"), ("False Alarm", "fa_%"), ("Speaker Conf", "conf_%")]:
            print(f"    {sub:<26} {od[key]:>17.2f}% {nd[key]:>19.2f}%")

    print(thin)
    m = r["online_vs_offline"]
    print(f"  Online vs Offline (mutual DER) : {m['DER_%']:>6.2f}%")
    print(sep)

    # Segment preview
    off_segs = r["offline"]["segments"]
    on_segs  = r["online"]["segments"]
    n = min(8, max(len(off_segs), len(on_segs)))
    print(f"\n  {'[Offline]':<38} {'[Online]'}")
    print(f"  {'-'*36} {'-'*36}")
    for i in range(n):
        left  = f"[{fmt_time(off_segs[i]['start'])}] {off_segs[i]['speaker']}" if i < len(off_segs) else ""
        right = f"[{fmt_time(on_segs[i]['start'])}]  {on_segs[i]['speaker']}" if i < len(on_segs) else ""
        print(f"  {left:<38} {right}")
    if len(off_segs) > n or len(on_segs) > n:
        print(f"  ... ({len(off_segs)} offline, {len(on_segs)} online)")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare offline vs online diarization")
    parser.add_argument("input", nargs="+", help="WAV file(s) or folder")
    parser.add_argument("--hf-token")
    parser.add_argument("--window",    type=float, default=6.0, help="Online window size (s)")
    parser.add_argument("--step",      type=float, default=1.0,  help="Online step size (s)")
    parser.add_argument("--threshold", type=float, default=0.70, help="Speaker registry threshold")
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--output", help="Save JSON results")
    args = parser.parse_args()

    wav_files = []
    for inp in args.input:
        p = Path(inp)
        wav_files.extend(sorted(p.glob("*.wav")) if p.is_dir() else [p])
    if not wav_files:
        sys.exit("No WAV files found")

    hf_token = get_hf_token(args.hf_token)
    print("\nLoading pyannote pipeline...")
    pipeline = load_diarization_pipeline(hf_token)
    print(f"Device   : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Window   : {args.window}s  |  Step: {args.step}s  |  Threshold: {args.threshold}")
    print(f"Files    : {len(wav_files)}\n")

    all_results = []
    agg_off_der, agg_on_der, agg_off_rtf, agg_on_rtf = [], [], [], []

    for wav in wav_files:
        print(f"Processing {wav.name} ...")
        try:
            r = compare_file(str(wav), pipeline, args.window, args.step,
                             args.threshold, args.num_speakers)
            print_result(r)
            all_results.append(r)
            agg_off_rtf.append(r["offline"]["rtf"])
            agg_on_rtf.append(r["online"]["rtf"])
            if r["has_ground_truth"]:
                agg_off_der.append(r["offline"]["der_vs_gt"]["DER_%"])
                agg_on_der.append(r["online"]["der_vs_gt"]["DER_%"])
        except Exception as e:
            print(f"  ERROR on {wav.name}: {e}")
            import traceback; traceback.print_exc()

    if len(all_results) > 1:
        def avg(lst): return f"{sum(lst)/len(lst):.2f}%" if lst else "N/A"
        sep = "=" * 72
        print(f"\n{sep}")
        print(f"  AGGREGATE ({len(all_results)} files)")
        print(f"  {'Metric':<35} {'Offline':>18} {'Online':>20}")
        print(f"  {'-'*73}")
        print(f"  {'Mean RTF':35} {sum(agg_off_rtf)/len(agg_off_rtf):>17.3f}x "
              f"{sum(agg_on_rtf)/len(agg_on_rtf):>19.3f}x")
        if agg_off_der:
            print(f"  {'Mean DER vs Ground Truth':35} {avg(agg_off_der):>18} {avg(agg_on_der):>20}")
        print(f"{sep}\n")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
