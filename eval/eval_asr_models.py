#!/usr/bin/env python3
"""
Controlled ASR-only benchmark across models (Whisper / Qwen3 / Nemotron).

Each reviewed ground-truth *segment span* is sliced from the audio and handed to
the chosen ASR model — every model therefore sees identical audio with ideal
(GT) boundaries, so WER/CER measure pure recognition quality, decoupled from
diarization. RTF = total transcription wall-time / audio duration.

One model per run (--model), so it runs in either virtualenv:
  venv (main)  : --model whisper | qwen
  venv_nemo    : --model nemotron   (needs nemo_toolkit[asr])

Usage:
  python eval_asr_models.py test/ --model whisper --language vi --output outputs/asr_whisper.json
  venv_nemo/Scripts/python eval_asr_models.py test/ --model nemotron --language vi \
      --output outputs/asr_nemotron.json
"""

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000
GT_DIR = Path("ground_truth")


# ── Audio / text helpers (self-contained so no heavy shared imports) ───────────

def load_audio(path: str) -> np.ndarray:
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    data = data[:, 0]
    if sr != SAMPLE_RATE:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
    return np.ascontiguousarray(data)


def _normalize(segs: list[dict]) -> str:
    parts = []
    for s in sorted(segs, key=lambda x: x["start"]):
        t = re.sub(r"[^\w\s]", " ", s.get("text", "").lower().strip(), flags=re.UNICODE)
        t = re.sub(r"\s+", " ", t).strip()
        if t:
            parts.append(t)
    return " ".join(parts)


def compute_wer_cer(gt_segs: list[dict], hyp_segs: list[dict]) -> dict:
    from jiwer import wer, cer
    ref, hyp = _normalize(gt_segs), _normalize(hyp_segs)
    if not ref or not hyp:
        return {"WER_%": None, "CER_%": None}
    return {"WER_%": round(wer(ref, hyp) * 100, 2),
            "CER_%": round(cer(ref, hyp) * 100, 2)}


# ── Model adapters: each returns transcribe(audio_1d_float32) -> str ───────────

def make_whisper(model_name: str, language: str | None):
    from faster_whisper import WhisperModel
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ct = "float16" if dev == "cuda" else "int8"
    m = WhisperModel(model_name, device=dev, compute_type=ct)

    def tr(audio: np.ndarray) -> str:
        # vad_filter=True is Whisper's standard config: it suppresses hallucination
        # on segments that contain trailing silence (matches the offline pipeline).
        segs, _ = m.transcribe(audio, language=language, vad_filter=True)
        return " ".join(s.text.strip() for s in segs).strip()
    return tr


_QWEN_LANG = {"vi": "Vietnamese", "en": "English", "zh": "Chinese"}


def make_qwen(language: str | None):
    from qwen_asr import Qwen3ASRModel
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    m = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B", dtype=dtype, device_map=dev, max_new_tokens=512)
    qlang = _QWEN_LANG.get(language, language) if language else None

    def tr(audio: np.ndarray) -> str:
        try:
            out = m.transcribe(audio=(audio, SAMPLE_RATE), language=qlang)
            return (out[0].text or "").strip()
        except Exception:
            return ""
    return tr


_NEMO_LANG = {"vi": "vi-VN", "en": "en-US", "zh": "zh-CN"}


def make_nemotron(language: str | None):
    """
    nvidia/nemotron-3.5-asr-streaming-0.6b — FastConformer cache-aware RNNT with
    language-ID prompt conditioning. Loaded as EncDecRNNTBPEModelWithPrompt; the
    target language is passed as the ``target_lang`` keyword (captured by the
    model's ``**prompt``), e.g. target_lang="vi-VN".

    NOTE: verified to install/load on Windows, but NeMo's transcribe() inference
    path for this model is broken on native Windows (numpy input decodes empty;
    file input hits a temp-manifest file-lock). Run this adapter under Linux/WSL.
    """
    import nemo.collections.asr as nemo_asr
    import torch
    m = nemo_asr.models.ASRModel.from_pretrained("nvidia/nemotron-3.5-asr-streaming-0.6b")
    if torch.cuda.is_available():
        m = m.cuda()
    m.eval()
    # Full attention context = offline/non-streaming decoding for batch transcription.
    if hasattr(m.encoder, "set_default_att_context_size"):
        m.encoder.set_default_att_context_size([-1, -1])
    tlang = _NEMO_LANG.get(language, language) if language else "auto"

    def tr(audio: np.ndarray) -> str:
        out = m.transcribe([audio.astype("float32")], batch_size=1, verbose=False, target_lang=tlang)
        item = out[0] if isinstance(out, (list, tuple)) else out
        text = getattr(item, "text", item)
        return (text or "").strip() if isinstance(text, str) else str(text).strip()
    return tr


# ── Ground truth ──────────────────────────────────────────────────────────────

def load_gt(wav_path: str) -> dict | None:
    gt_path = GT_DIR / (Path(wav_path).stem + ".json")
    if not gt_path.exists():
        return None
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)
    if gt.get("annotation_status") != "reviewed":
        print(f"  [skip] {gt_path.name}: not reviewed")
        return None
    return gt


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Controlled ASR-only benchmark per GT segment")
    ap.add_argument("input", nargs="+", help="WAV file(s) or folder")
    ap.add_argument("--model", required=True, choices=["whisper", "qwen", "nemotron"])
    ap.add_argument("--whisper-model", default="turbo")
    ap.add_argument("--language", default="vi")
    ap.add_argument("--min-seg", type=float, default=0.2, help="Skip GT segments shorter than this (s)")
    ap.add_argument("--output")
    args = ap.parse_args()

    wav_files = []
    for inp in args.input:
        p = Path(inp)
        wav_files.extend(sorted(p.glob("*.wav")) if p.is_dir() else [p])
    if not wav_files:
        sys.exit("No WAV files found")

    print(f"\nLoading model: {args.model} ...")
    if args.model == "whisper":
        transcribe = make_whisper(args.whisper_model, args.language)
        model_label = f"whisper-{args.whisper_model}"
    elif args.model == "qwen":
        transcribe = make_qwen(args.language)
        model_label = "qwen3-asr-1.7b"
    else:
        transcribe = make_nemotron(args.language)
        model_label = "nemotron-3.5-asr-streaming-0.6b"
    print("Model loaded.\n")

    all_results = []
    agg = {"wer": [], "cer": [], "rtf": []}

    for wav in wav_files:
        gt = load_gt(str(wav))
        if gt is None:
            continue
        data = load_audio(str(wav))
        duration = len(data) / SAMPLE_RATE
        gt_segs = gt["segments"]

        hyp_segs = []
        t0 = time.perf_counter()
        for s in gt_segs:
            if s["end"] - s["start"] < args.min_seg:
                hyp_segs.append({"start": s["start"], "end": s["end"], "text": ""})
                continue
            sl = data[int(s["start"] * SAMPLE_RATE): int(s["end"] * SAMPLE_RATE)]
            text = transcribe(sl)
            hyp_segs.append({"start": s["start"], "end": s["end"], "text": text})
        elapsed = time.perf_counter() - t0

        metrics = compute_wer_cer(gt_segs, hyp_segs)
        rtf = round(elapsed / duration, 3)
        r = {"file": Path(wav).name, "duration_s": round(duration, 2),
             **metrics, "rtf": rtf, "n_segments": len(gt_segs),
            "hyp_segments": hyp_segs}
        all_results.append(r)
        print(f"{'='*56}\n  {Path(wav).name}  ({duration:.1f}s, {len(gt_segs)} segs)")
        print(f"    WER : {metrics['WER_%']}%   CER : {metrics['CER_%']}%   RTF : {rtf}x")
        if metrics["WER_%"] is not None:
            agg["wer"].append(metrics["WER_%"])
            agg["cer"].append(metrics["CER_%"])
        agg["rtf"].append(rtf)

    if all_results:
        def mean(x): return round(sum(x) / len(x), 2) if x else None
        print(f"\n{'='*56}\n  AGGREGATE — {model_label} ({len(all_results)} files)")
        print(f"    Mean WER : {mean(agg['wer'])}%")
        print(f"    Mean CER : {mean(agg['cer'])}%")
        print(f"    Mean RTF : {mean(agg['rtf'])}x")
        print(f"{'='*56}\n")

        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            out = {"model": model_label, "language": args.language,
                   "mean_wer": mean(agg["wer"]), "mean_cer": mean(agg["cer"]),
                   "mean_rtf": mean(agg["rtf"]), "files": all_results}
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
