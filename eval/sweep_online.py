#!/usr/bin/env python3
"""
Sweep online (diart-style) diarization hyper-parameters to find the config
that minimises mean DER across the reviewed ground-truth set.

Loads the pyannote pipeline once and reuses it across every (window, threshold)
configuration, so a full sweep costs roughly one GPU pass per config rather than
re-loading the model each time (as repeated compare_diarization.py calls would).

Usage:
  python sweep_online.py test/test01.wav test/test02.wav \
      --windows 4 6 9 --thresholds 0.70 0.80 --step 1 \
      --output outputs/sweep_online.json
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # cho phép chạy trực tiếp
from core.utils import (get_hf_token, load_audio, load_diarization_pipeline,
                        compute_der, load_gt, iter_wavs, SAMPLE_RATE)
from core.diarize_online import run_online


def main():
    ap = argparse.ArgumentParser(description="Sweep online diarization hyper-parameters")
    ap.add_argument("input", nargs="+", help="WAV file(s) or folder")
    ap.add_argument("--hf-token")
    ap.add_argument("--windows", type=float, nargs="+", default=[4, 6, 9])
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.70, 0.80])
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--output")
    args = ap.parse_args()

    wav_files = iter_wavs(args.input)
    if not wav_files:
        sys.exit("No WAV files found")

    pipeline = load_diarization_pipeline(get_hf_token(args.hf_token))
    print(f"\nFiles: {len(wav_files)}  |  windows={args.windows}  "
          f"thresholds={args.thresholds}  step={args.step}\n")

    # Pre-load audio + GT once per file.
    files = []
    for wav in wav_files:
        gt_segs, n_spk = load_gt(str(wav))
        if gt_segs is None:
            print(f"  [skip] {wav.name}: no ground truth")
            continue
        data = load_audio(str(wav))
        files.append((wav.name, data, gt_segs, n_spk, len(data) / SAMPLE_RATE))

    configs = [(w, t) for w in args.windows for t in args.thresholds]
    # per_config[(w,t)] = {file_name: DER%}
    per_config: dict[tuple, dict] = {c: {} for c in configs}

    for name, data, gt_segs, n_spk, dur in files:
        print(f"{'='*60}\n  {name}  ({dur:.1f}s, {n_spk} spk GT)\n{'='*60}")
        for (w, t) in configs:
            segs, _ = run_online(data, pipeline, window_s=w, step_s=args.step,
                                 registry_threshold=t, num_speakers=n_spk)
            der = compute_der(gt_segs, segs)
            per_config[(w, t)][name] = der
            print(f"  win={w:>4}  thr={t:.2f}   DER={der['DER_%']:>6.2f}%   "
                  f"(miss={der['miss_%']:.1f} fa={der['fa_%']:.1f} conf={der['conf_%']:.1f})")
        print()

    # Aggregate.
    print(f"\n{'='*60}\n  SWEEP SUMMARY (mean over {len(files)} files)\n{'='*60}")
    print(f"  {'window':>8} {'threshold':>10} {'mean DER':>10} {'mean conf':>10}")
    print(f"  {'-'*42}")
    rows = []
    for (w, t) in configs:
        ders = [per_config[(w, t)][f[0]]["DER_%"] for f in files]
        confs = [per_config[(w, t)][f[0]]["conf_%"] for f in files]
        mean_der = sum(ders) / len(ders)
        mean_conf = sum(confs) / len(confs)
        rows.append({"window_s": w, "threshold": t,
                     "mean_der": round(mean_der, 2), "mean_conf": round(mean_conf, 2),
                     "per_file": {f[0]: per_config[(w, t)][f[0]] for f in files}})
        print(f"  {w:>8} {t:>10.2f} {mean_der:>9.2f}% {mean_conf:>9.2f}%")
    best = min(rows, key=lambda r: r["mean_der"])
    print(f"\n  BEST: window={best['window_s']}  threshold={best['threshold']}  "
          f"mean DER={best['mean_der']}%\n{'='*60}\n")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"configs": rows, "best": {k: best[k] for k in ("window_s", "threshold", "mean_der")}},
                      f, indent=2, ensure_ascii=False)
        print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
