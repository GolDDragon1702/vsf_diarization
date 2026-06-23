"""Shared utilities for all pipeline scripts."""

import json
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

SAMPLE_RATE = 16000
GT_DIR = Path("ground_truth")


# ── Ground truth ──────────────────────────────────────────────────────────────

def read_gt(wav_path: str, require_reviewed: bool = False) -> dict | None:
    """GT dict đầy đủ cho file wav; None nếu thiếu (hoặc chưa reviewed khi yêu cầu)."""
    gt_path = GT_DIR / (Path(wav_path).stem + ".json")
    if not gt_path.exists():
        return None
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)
    if require_reviewed and gt.get("annotation_status") != "reviewed":
        return None
    return gt


def load_gt(wav_path: str, require_reviewed: bool = True) -> tuple[list[dict] | None, int | None]:
    """(segments, num_speakers) cho eval/diarization; (None, None) nếu thiếu/chưa reviewed."""
    gt = read_gt(wav_path, require_reviewed=require_reviewed)
    if gt is None:
        return None, None
    n_spk = len(gt.get("speaker_labels", {})) or None
    return gt["segments"], n_spk


def iter_wavs(inputs: list[str]) -> list[Path]:
    """Mở rộng list file/folder thành danh sách .wav đã sort."""
    out: list[Path] = []
    for inp in inputs:
        p = Path(inp)
        out.extend(sorted(p.glob("*.wav")) if p.is_dir() else [p])
    return out


# ── Auth ──────────────────────────────────────────────────────────────────────

def get_hf_token(cli_token: str | None = None) -> str:
    token = cli_token or os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN missing. Set it in .env or pass --hf-token.")
    return token


# ── Audio I/O ─────────────────────────────────────────────────────────────────

def load_audio(path: str) -> np.ndarray:
    """Load audio file to 16kHz mono float32 numpy array."""
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    data = data[:, 0]
    if sr != SAMPLE_RATE:
        ratio = SAMPLE_RATE / sr
        n = int(len(data) * ratio)
        idx = np.clip((np.arange(n) / ratio).astype(int), 0, len(data) - 1)
        data = data[idx]
    return data


def is_video(path: str) -> bool:
    return path.lower().endswith((".mp4", ".avi", ".mkv", ".mov", ".webm"))


def extract_audio(input_path: str, output_path: str | None = None) -> str:
    """Extract/convert to 16kHz mono WAV using ffmpeg."""
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + "_16k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1",
         "-c:a", "pcm_s16le", output_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return output_path


def get_audio_input(input_path: str) -> str:
    """Return path to a 16kHz WAV, extracting from video if needed."""
    if is_video(input_path):
        print(f"Extracting audio from {input_path} ...")
        return extract_audio(input_path)
    return input_path


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_diarization_pipeline(hf_token: str):
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", token=hf_token,
    )
    if torch.cuda.is_available():
        pipeline = pipeline.to(torch.device("cuda"))
    return pipeline


def load_whisper(model_name: str = "turbo"):
    from faster_whisper import WhisperModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def load_qwen_asr(model_name: str = "Qwen/Qwen3-ASR-1.7B"):
    from qwen_asr import Qwen3ASRModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    return Qwen3ASRModel.from_pretrained(
        model_name, dtype=dtype, device_map=device, max_new_tokens=512,
    )


# ── Diarization helpers ───────────────────────────────────────────────────────

def merge_segments(segments: list[dict], gap: float = 0.5, min_dur: float = 0.3) -> list[dict]:
    """Drop segments shorter than min_dur; merge same-speaker segments with gap ≤ gap."""
    segments = sorted(
        [s for s in segments if s["end"] - s["start"] >= min_dur],
        key=lambda s: s["start"],
    )
    if not segments:
        return []
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg["speaker"] == prev["speaker"] and seg["start"] - prev["end"] <= gap:
            prev["end"] = seg["end"]
            prev["duration"] = round(prev["end"] - prev["start"], 3)
        else:
            merged.append(seg.copy())
    return merged


def speaker_at(turns: list[dict], t: float) -> str:
    """Return speaker active at time t; falls back to nearest speaker if t is in a gap."""
    matches = [s for s in turns if s["start"] <= t <= s["end"]]
    if matches:
        return min(matches, key=lambda s: s["end"] - s["start"])["speaker"]
    if not turns:
        return "UNKNOWN"
    return min(turns, key=lambda s: min(abs(s["start"] - t), abs(s["end"] - t)))["speaker"]


def merge_text_segments(segments: list[dict], gap: float = 1.0) -> list[dict]:
    """Merge adjacent same-speaker text segments with gap ≤ gap seconds."""
    if not segments:
        return []
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg["speaker"] == prev["speaker"] and seg["start"] - prev["end"] <= gap:
            prev["end"] = seg["end"]
            prev["text"] = prev["text"] + " " + seg["text"]
        else:
            merged.append(seg.copy())
    return merged


# ── ASR functions ─────────────────────────────────────────────────────────────

def whisper_transcribe(
    data: np.ndarray, turns: list[dict], whisper, language: str | None,
) -> tuple[list[dict], str]:
    """Transcribe with Whisper; align each word to its speaker via midpoint lookup."""
    segs_iter, info = whisper.transcribe(
        data, word_timestamps=True, language=language, vad_filter=True,
    )
    results: list[dict] = []
    cur_speaker: str | None = None
    cur_words: list[str] = []
    cur_start = cur_end = 0.0
    for seg in segs_iter:
        for word in (seg.words or []):
            mid = (word.start + word.end) / 2
            spk = speaker_at(turns, mid)
            if spk != cur_speaker:
                if cur_speaker and cur_words:
                    results.append({"speaker": cur_speaker,
                                    "start": round(cur_start, 3), "end": round(cur_end, 3),
                                    "text": " ".join(cur_words)})
                cur_speaker, cur_words = spk, [word.word.strip()]
                cur_start, cur_end = word.start, word.end
            else:
                cur_words.append(word.word.strip())
                cur_end = word.end
    if cur_speaker and cur_words:
        results.append({"speaker": cur_speaker,
                        "start": round(cur_start, 3), "end": round(cur_end, 3),
                        "text": " ".join(cur_words)})
    return merge_text_segments(results), info.language


_QWEN_LANG_MAP = {
    "vi": "Vietnamese", "en": "English", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "fr": "French", "de": "German",
}


def qwen_transcribe(
    data: np.ndarray, turns: list[dict], qwen, language: str | None,
) -> list[dict]:
    """Transcribe with Qwen3-ASR; process each speaker turn independently."""
    qwen_lang = _QWEN_LANG_MAP.get(language, language) if language else None
    min_samples = int(0.3 * SAMPLE_RATE)
    results: list[dict] = []
    for turn in turns:
        seg = data[int(turn["start"] * SAMPLE_RATE): int(turn["end"] * SAMPLE_RATE)]
        if len(seg) < min_samples:
            continue
        try:
            out = qwen.transcribe(audio=(seg, SAMPLE_RATE), language=qwen_lang)
            text = (out[0].text or "").strip()
        except Exception:
            text = ""
        if text:
            results.append({"speaker": turn["speaker"],
                            "start": round(turn["start"], 3), "end": round(turn["end"], 3),
                            "text": text})
    return merge_text_segments(results)


# ── Metrics ───────────────────────────────────────────────────────────────────

def segs_to_annotation(segments: list[dict]):
    """Convert segment list to pyannote Annotation."""
    from pyannote.core import Annotation, Segment
    ann = Annotation()
    for s in segments:
        ann[Segment(s["start"], s["end"])] = s["speaker"]
    return ann


def compute_der(gt_segs: list[dict], hyp_segs: list[dict]) -> dict:
    """Diarization Error Rate with collar=0.25s."""
    from pyannote.metrics.diarization import DiarizationErrorRate
    metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        details = metric(segs_to_annotation(gt_segs), segs_to_annotation(hyp_segs), detailed=True)
    total = details["total"] if details["total"] > 0 else 1e-9
    return {
        "DER_%":  round(details["diarization error rate"] * 100, 2),
        "miss_%": round(details["missed detection"] / total * 100, 2),
        "fa_%":   round(details["false alarm"] / total * 100, 2),
        "conf_%": round(details["confusion"] / total * 100, 2),
    }


def compute_asr_metrics(gt_segs: list[dict], hyp_segs: list[dict]) -> dict:
    """Word Error Rate and Character Error Rate after text normalization."""
    from jiwer import wer, cer

    def normalize(segs):
        parts = []
        for s in sorted(segs, key=lambda x: x["start"]):
            t = re.sub(r"[^\w\s]", " ", s.get("text", "").lower().strip(), flags=re.UNICODE)
            t = re.sub(r"\s+", " ", t).strip()
            if t:
                parts.append(t)
        return " ".join(parts)

    ref, hyp = normalize(gt_segs), normalize(hyp_segs)
    if not ref or not hyp:
        return {"WER_%": None, "CER_%": None}
    return {
        "WER_%": round(wer(ref, hyp) * 100, 2),
        "CER_%": round(cer(ref, hyp) * 100, 2),
    }


# ── Formatting ────────────────────────────────────────────────────────────────

def fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"
