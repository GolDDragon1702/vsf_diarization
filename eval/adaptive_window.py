#!/usr/bin/env python3
"""
Adaptive per-file window — chọn window online theo từng file mà KHÔNG cần GT.

Heuristic "offline-agreement":
  1. Chạy offline pyannote 1 lần  -> reference R (mạnh nhất, GT-free).
  2. Với mỗi window w, chạy online -> O_w.
  3. Chọn w* = argmin DER(R, O_w)   (online nào khớp offline nhất thì chọn).
  4. Dùng O_{w*}.

Lý do: offline 3.1 là bản diarization tốt nhất ta có sẵn mà không cần nhãn tay;
window online "đúng" sẽ tái hiện kết quả offline gần nhất. Đây là cách chọn window
self-supervised, dùng được lúc deploy (không có GT).

Báo cáo: so DER thật (vs GT) của adaptive với oracle (min theo GT) và window cố định.

Usage:
  python adaptive_window.py test/ --windows 4 6 9 --output outputs/adaptive_window.json
"""

import argparse
import json
import sys
import pathlib
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # cho phép chạy trực tiếp
from core.utils import (get_hf_token, load_audio, load_diarization_pipeline,
                        compute_der, load_gt, iter_wavs, SAMPLE_RATE)
from core.diarize_online import run_online
from core.diarize_offline import run_offline


def main():
    ap = argparse.ArgumentParser(description="Adaptive per-file window (offline-agreement)")
    ap.add_argument("input", nargs="+")
    ap.add_argument("--hf-token")
    ap.add_argument("--windows", type=float, nargs="+", default=[4, 6, 9])
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--output")
    args = ap.parse_args()

    wav_files = iter_wavs(args.input)
    if not wav_files:
        sys.exit("No WAV files found")

    print("Loading pipeline …", flush=True)
    pipeline = load_diarization_pipeline(get_hf_token(args.hf_token))
    windows = sorted(args.windows)
    print(f"windows={windows} step={args.step} thr={args.threshold}\n", flush=True)

    files = [(w, *load_gt(str(w))) for w in wav_files]
    files = [(w, g, n) for (w, g, n) in files if g is not None]

    rows = []
    fixed = {w: [] for w in windows}   # DER vs GT cho từng window cố định
    adaptive_ders, oracle_ders = [], []
    hits = 0

    for wav, gt_segs, n_spk in files:
        data = load_audio(str(wav))
        dur = len(data) / SAMPLE_RATE
        ref, _ = run_offline(data, pipeline, num_speakers=n_spk)   # reference GT-free

        per_w = {}
        for w in windows:
            o, _ = run_online(data, pipeline, window_s=w, step_s=args.step,
                              registry_threshold=args.threshold, num_speakers=n_spk)
            agree = compute_der(ref, o)["DER_%"]      # bất đồng với offline (thấp = khớp)
            der_gt = compute_der(gt_segs, o)["DER_%"]  # chất lượng thật (chỉ để báo cáo)
            per_w[w] = {"agree_vs_offline_%": agree, "DER_vs_GT_%": der_gt}
            fixed[w].append(der_gt)

        chosen = min(windows, key=lambda w: per_w[w]["agree_vs_offline_%"])   # GT-free pick
        oracle = min(windows, key=lambda w: per_w[w]["DER_vs_GT_%"])          # cheat (GT)
        chosen_der = per_w[chosen]["DER_vs_GT_%"]
        oracle_der = per_w[oracle]["DER_vs_GT_%"]
        adaptive_ders.append(chosen_der)
        oracle_ders.append(oracle_der)
        if chosen == oracle:
            hits += 1

        rows.append({"file": wav.name, "duration_s": round(dur, 1),
                     "chosen_w": chosen, "oracle_w": oracle,
                     "chosen_DER_%": chosen_der, "oracle_DER_%": oracle_der,
                     "per_window": per_w,
                     "offline_ref": ref})   # lưu để chẩn đoán test03 không cần chạy lại

        mark = "✓" if chosen == oracle else f"✗ (oracle w{oracle:g}={oracle_der:.1f})"
        print(f"  {wav.name:<11} chọn w{chosen:g} -> DER {chosen_der:>6.2f}%   {mark}", flush=True)

    def m(x):
        return round(sum(x) / len(x), 2) if x else None

    print(f"\n{'='*60}")
    print(f"  ADAPTIVE WINDOW ({len(files)} file)")
    print(f"    Adaptive (offline-agreement) : {m(adaptive_ders)}%   "
          f"(chọn đúng oracle {hits}/{len(files)} file)")
    print(f"    Oracle (min theo GT)         : {m(oracle_ders)}%")
    for w in windows:
        print(f"    Fixed window={w:g}              : {m(fixed[w])}%")
    print(f"{'='*60}", flush=True)

    if args.output:
        # offline_ref ra file riêng (nặng), summary gọn
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out = {"config": {"windows": windows, "step_s": args.step, "threshold": args.threshold},
               "summary": {"adaptive_%": m(adaptive_ders), "oracle_%": m(oracle_ders),
                           "oracle_hits": f"{hits}/{len(files)}",
                           "fixed_%": {f"w{w:g}": m(fixed[w]) for w in windows}},
               "files": rows}
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Saved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
