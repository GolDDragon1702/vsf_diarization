#!/usr/bin/env python3
"""
Evaluate pipeline(s) against manually-reviewed ground truth.

Metrics:
  Diarization — DER, Miss, FA, Speaker Confusion
  ASR         — WER, CER

Usage:
  python evaluate.py test/test01.wav --language vi
  python evaluate.py test/test01.wav test/test02.wav --language vi
  python evaluate.py test/ --language vi --output outputs/eval_results.json
  python evaluate.py test/ --compare-qwen --language vi
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

from vsf_diarization.core.utils import (
    get_hf_token, load_audio, load_diarization_pipeline, load_whisper,
    whisper_transcribe, qwen_transcribe, compute_der, compute_asr_metrics, fmt_time,
    read_gt as load_ground_truth, iter_wavs, SAMPLE_RATE,
)
from vsf_diarization.core.diarize_offline import run_offline


# ── Per-file evaluation ───────────────────────────────────────────────────────

def evaluate_file(
    wav_path: str,
    diar_pipeline,
    whisper,
    qwen,
    language: str | None,
    num_speakers: int | None,
) -> dict:
    gt = load_ground_truth(wav_path)
    gt_segs = gt["segments"] if gt else None
    effective_n_spk = num_speakers or (len(gt["speaker_labels"]) if gt else None)

    data = load_audio(wav_path)
    duration = len(data) / SAMPLE_RATE

    result = {
        "file": wav_path,
        "duration_s": round(duration, 2),
        "gt_status": gt.get("annotation_status") if gt else None,
    }

    # ── Whisper pipeline ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    turns, _ = run_offline(data, diar_pipeline, effective_n_spk)
    w_segs, detected_lang = whisper_transcribe(data, turns, whisper, language)
    t_w = time.perf_counter() - t0

    w_metrics = {}
    if gt_segs:
        w_metrics = {**compute_der(gt_segs, w_segs), **compute_asr_metrics(gt_segs, w_segs)}
    w_metrics["rtf"] = round(t_w / duration, 3)
    w_metrics["segments"] = w_segs
    result["whisper"] = w_metrics

    # ── Qwen3-ASR pipeline (optional) ────────────────────────────────────────
    if qwen is not None:
        t0 = time.perf_counter()
        turns_q, _ = run_offline(data, diar_pipeline, effective_n_spk)
        q_segs = qwen_transcribe(data, turns_q, qwen, language)
        t_q = time.perf_counter() - t0

        q_metrics = {}
        if gt_segs:
            q_metrics = {**compute_der(gt_segs, q_segs), **compute_asr_metrics(gt_segs, q_segs)}
        q_metrics["rtf"] = round(t_q / duration, 3)
        q_metrics["segments"] = q_segs
        result["qwen"] = q_metrics

    return result, detected_lang


def print_result(r: dict, whisper_model: str, has_qwen: bool) -> None:
    gt_status = r.get("gt_status") or "no GT"
    print(f"  GT: {gt_status}   duration: {r['duration_s']}s")

    def print_metrics(label, m):
        print(f"  [{label}]")
        if "DER_%" in m:
            print(f"    DER  : {m['DER_%']:>6.2f}%   "
                f"(miss={m['miss_%']}%  fa={m['fa_%']}%  conf={m['conf_%']}%)")
        if m.get("WER_%") is not None:
            print(f"    WER  : {m['WER_%']:>6.2f}%")
            print(f"    CER  : {m['CER_%']:>6.2f}%")
        print(f"    RTF  : {m['rtf']:.3f}x")

    print_metrics(f"Whisper {whisper_model}", r["whisper"])
    if has_qwen and "qwen" in r:
        print()
        print_metrics("Qwen3-ASR-1.7B", r["qwen"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline(s) against ground truth")
    parser.add_argument("input", nargs="+", help="WAV file(s) or folder")
    parser.add_argument("--hf-token")
    parser.add_argument("--whisper-model", default="turbo")
    parser.add_argument("--compute-type", help="Whisper: 'float16' (chính xác hơn) | mặc định int8_float16 (nhanh ~2.6×)")
    parser.add_argument("--language")
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--compare-qwen", action="store_true",
                        help="Also evaluate Qwen3-ASR-1.7B")
    parser.add_argument("--output", help="Save results JSON (e.g. outputs/eval_results.json)")
    args = parser.parse_args()

    wav_files = iter_wavs(args.input)
    if not wav_files:
        sys.exit("No WAV files found")

    hf_token = get_hf_token(args.hf_token)
    print("\nLoading models...")
    diar_pipeline = load_diarization_pipeline(hf_token)
    whisper = load_whisper(args.whisper_model, args.compute_type)
    qwen = None
    if args.compare_qwen:
        from utils import load_qwen_asr
        qwen = load_qwen_asr()
    print("Models loaded.\n")

    all_results = []
    agg: dict[str, list] = {
        "w_der": [], "w_wer": [], "w_cer": [],
        "q_der": [], "q_wer": [], "q_cer": [],
    }

    for wav in wav_files:
        print(f"{'='*60}\n  {wav.name}\n{'='*60}")
        r, lang = evaluate_file(
            str(wav), diar_pipeline, whisper, qwen,
            args.language, args.num_speakers,
        )
        print_result(r, args.whisper_model, args.compare_qwen)
        print()
        all_results.append(r)

        for key, m, prefix in [("w", r["whisper"], "w"), ("q", r.get("qwen", {}), "q")]:
            for metric, agg_key in [("DER_%", f"{prefix}_der"),
                                     ("WER_%", f"{prefix}_wer"),
                                     ("CER_%", f"{prefix}_cer")]:
                v = m.get(metric)
                if v is not None:
                    agg[agg_key].append(v)

    # ── Aggregate summary ─────────────────────────────────────────────────────
    if len(all_results) > 1:
        def avg(lst): return f"{sum(lst)/len(lst):.2f}" if lst else "N/A"
        print(f"\n{'='*60}")
        print(f"  AGGREGATE ({len(all_results)} files)")
        print(f"  {'Metric':<20} {'Whisper':>10} {'Qwen3':>10}")
        print(f"  {'-'*40}")
        print(f"  {'Mean DER (%)':20} {avg(agg['w_der']):>10} {avg(agg['q_der']):>10}")
        print(f"  {'Mean WER (%)':20} {avg(agg['w_wer']):>10} {avg(agg['q_wer']):>10}")
        print(f"  {'Mean CER (%)':20} {avg(agg['w_cer']):>10} {avg(agg['q_cer']):>10}")
        print(f"{'='*60}\n")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
